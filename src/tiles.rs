//! Generic tile quilting: build a `PMTiles` archive from any [`TileSource`].
//!
//! For every `(z, col, row)` in the data's zoom range, candidate source items
//! are selected by coverage overlap, sorted by zoom fit (ties broken by
//! [`TileSource::tiebreak`] — newer edition wins), then added greedily until
//! the tile is covered by their combined [`TileSource::coverage`].  A sparse
//! [`Quilt`] additionally leaves out tiles a client can rebuild from an
//! ancestor.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use geo::{Intersects, MultiPolygon, Rect};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use martin_tile_utils::{bbox_to_xyz, xyz_to_bbox};
use once_cell::sync::OnceCell;
use pmtiles::{PmTilesWriter, TileCoord, TileId};
use rayon::prelude::*;
use tracing::info;

use crate::bbox::Bbox;
use crate::tile_geom::TileGeom;
use crate::tile_grid::TileGrid;
use crate::tile_source::{TileAccumulator, TileSource};
use crate::zoom::{scale_from_zoom, zoom_from_scale};

/// A single encoded tile ready for insertion into the output archive.
type EncodedTile = (u64, TileCoord, Vec<u8>);

/// The tiles an archive stores, keyed (and so ordered) by `PMTiles` tile ID.
type ArchiveTiles = BTreeMap<u64, (TileCoord, Vec<u8>)>;

// ── Entry point ───────────────────────────────────────────────────────────────

/// Quilt `quilt`'s items into a `PMTiles` v3 archive at `output`.
///
/// Stores the tiles [`archive_tiles`] keeps. Returns `(min_zoom, max_zoom)`.
///
/// # Errors
/// Returns an error if `max_zoom` is below the data's minimum zoom, a tile
/// fails to encode, or writing/finalizing the `PMTiles` archive fails.
pub fn write_pmtiles<S: TileSource>(
    quilt: &Quilt<S>,
    output: &Path,
    max_zoom: Option<u8>,
) -> Result<(u8, u8)> {
    let (zoom_floor, zoom_ceil, overall) = quilt.zoom_range_and_bounds(max_zoom)?;
    let tile_bytes = archive_tiles(quilt, zoom_floor..=zoom_ceil, overall)?;

    let metadata = build_metadata(quilt.is_sparse());
    let file = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = PmTilesWriter::new(S::tile_type())
        .min_zoom(zoom_floor)
        .max_zoom(zoom_ceil)
        .bounds(overall.west, overall.south, overall.east, overall.north)
        .metadata(&metadata)
        .create(file)
        .context("creating PMTiles writer")?;

    let pb_write = ProgressBar::new(tile_bytes.len() as u64).with_style(bar_style());
    {
        profiling::scope!("writing pmtiles");
        for (_, (coord, bytes)) in tile_bytes {
            writer.add_tile(coord, &bytes).context("writing tile")?;
            pb_write.inc(1);
        }
    }
    pb_write.finish_and_clear();
    {
        profiling::scope!("finalizing pmtiles");
        writer.finalize().context("finalizing PMTiles")?;
    }

    info!(output = %output.display(), "PMTiles written");
    Ok((zoom_floor, zoom_ceil))
}

/// Render every tile position of `overall` at `zooms` and keep the ones an
/// archive stores: [`TileOutcome::Rendered`] bytes, and for a sparse quilt
/// [`TileAccumulator::blank`] in place of [`TileOutcome::Blank`] (a missing
/// tile would make the client fill it from an ancestor). Uncovered and
/// omitted tiles are left out.
///
/// # Errors
/// Returns an error if a tile fails to render or encode.
fn archive_tiles<S: TileSource>(
    quilt: &Quilt<S>,
    zooms: std::ops::RangeInclusive<u8>,
    overall: Bbox,
) -> Result<ArchiveTiles> {
    let mut tile_bytes = ArchiveTiles::new();
    let omitted = AtomicU64::new(0);
    let blank = if quilt.is_sparse() {
        Some(S::Accumulator::blank()?)
    } else {
        None
    };
    let (zoom_floor, zoom_ceil) = (*zooms.start(), *zooms.end());

    let total_tiles: u64 = (zoom_floor..=zoom_ceil)
        .map(|z| {
            let (c0, r0, c1, r1) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            u64::from(c1 - c0 + 1) * u64::from(r1 - r0 + 1)
        })
        .sum();

    info!(
        count = quilt.items().len(),
        zoom_floor,
        zoom_ceil,
        total_tiles,
        sparse = quilt.is_sparse(),
        "encoding tiles",
    );
    let pb = ProgressBar::new(total_tiles).with_style(bar_style());

    for z in zoom_floor..=zoom_ceil {
        let (col_lo, row_lo, col_hi, row_hi) =
            bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
        let width = col_hi - col_lo + 1;
        let height = row_hi - row_lo + 1;
        let count = u64::from(width) * u64::from(height);

        (0u64..count)
            .into_par_iter()
            .progress_with(pb.clone())
            .map(|idx| -> Result<Option<EncodedTile>> {
                // Tiles are encoded across rayon worker threads concurrently, so a
                // single global frame mark doesn't fit; Tracy's non-continuous
                // (secondary) frame set supports overlapping FrameMarkStart/End
                // pairs and renders each tile as its own row in the Frames panel.
                #[cfg(feature = "profiling")]
                let _frame = tracy_client::non_continuous_frame!("tile");
                #[allow(clippy::cast_possible_truncation)] // idx % width < width ≤ u32::MAX
                let col = col_lo + (idx % u64::from(width)) as u32;
                #[allow(clippy::cast_possible_truncation)] // idx / width < height ≤ u32::MAX
                let row = row_lo + (idx / u64::from(width)) as u32;

                let tile = TileXyz::new(z, col, row).with_context(|| {
                    format!("tile {z}/{col}/{row} is off the Web Mercator grid")
                })?;
                let bytes = match quilt.render_tile(tile)? {
                    TileOutcome::Rendered(bytes) => bytes,
                    TileOutcome::Blank => match &blank {
                        Some(blank) => blank.clone(),
                        None => return Ok(None),
                    },
                    TileOutcome::Omitted => {
                        omitted.fetch_add(1, Ordering::Relaxed);
                        return Ok(None);
                    }
                    TileOutcome::Uncovered => return Ok(None),
                };
                let coord = TileCoord::new(z, col, row).context("invalid tile coord")?;
                Ok(Some((TileId::from(coord).value(), coord, bytes)))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .for_each(|(id, coord, bytes)| {
                tile_bytes.insert(id, (coord, bytes));
            });
    }
    pb.finish_and_clear();
    info!(
        encoded = tile_bytes.len(),
        omitted = omitted.into_inner(),
        "tiles encoded"
    );
    Ok(tile_bytes)
}

/// A Web Mercator tile address [`Quilt::render_tile`] can render.
///
/// Zoom is at most [`martin_tile_utils::MAX_ZOOM`] (the deepest zoom
/// `xyz_to_bbox` projects without panicking) and column/row lie inside that
/// zoom's `2^z × 2^z` grid. Parse untrusted `(z, x, y)` into this once, at
/// the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TileXyz {
    z: u8,
    x: u32,
    y: u32,
}

impl TileXyz {
    /// `None` unless `z ≤ MAX_ZOOM` and `x, y < 2^z`.
    #[must_use]
    pub fn new(z: u8, x: u32, y: u32) -> Option<Self> {
        martin_tile_utils::TileCoord::is_possible_on_zoom_level(z, x, y).then_some(Self { z, x, y })
    }

    #[must_use]
    pub const fn z(self) -> u8 {
        self.z
    }

    /// Column, counted east from the antimeridian.
    #[must_use]
    pub const fn x(self) -> u32 {
        self.x
    }

    /// Row, counted south from the top of the Web Mercator square.
    #[must_use]
    pub const fn y(self) -> u32 {
        self.y
    }
}

/// What [`Quilt::render_tile`] produced for one tile.
#[derive(Debug, PartialEq, Eq)]
pub enum TileOutcome {
    /// No item's coverage reaches the tile, at this zoom or any other.
    Uncovered,
    /// Items cover the tile but their content draws nothing there.
    Blank,
    /// Left out of a sparse quilt: the client rebuilds it from its nearest
    /// stored ancestor (a `MapLibre` `sparse` vector source slices that
    /// ancestor, a raster source upscales it) — see [`Quilt`]. Also reported
    /// for a tile only bboxes (not coverage polygons) reach, when all those
    /// items are too coarse: the ancestor is empty there too, so the client
    /// still draws nothing.
    Omitted,
    /// Encoded tile bytes.
    Rendered(Vec<u8>),
}

/// Items to quilt, plus the per-item data every tile needs, computed once.
///
/// # Sparse quilts
///
/// A sparse quilt omits a tile at zoom `z` when every item contributing to
/// it has a [`TileSource::detail_zoom`] coarser than `z` — decided from the
/// bbox candidates alone when they all are, skipping the polygon algebra.
/// Such a tile shows what its nearest stored ancestor shows there: let `D`
/// be the finest detail zoom among its contributors. Any item overlapping
/// the tile with a native zoom finer than `D` would sort ahead of them and
/// contribute, so none exists — the candidate order inside the tile is
/// therefore the same at every zoom from `D` to `z`, and each contributor
/// renders the same features at or above its detail zoom. The ancestor at
/// `D` is stored (its contributor with detail zoom `D` keeps it), so the
/// nearest stored ancestor lies in `D..z`.
///
/// Same content, not same positions: a vector tile rounds vertices to a
/// 4096-step grid across itself, so a tile sliced from an ancestor `g`
/// levels up keeps a grid `2^g` times coarser (`2^g / 8` screen px for
/// 512 px tiles; ~21 m at 57°N for a z8 ancestor) and its lines can kink and
/// jump where they meet a stored neighbour. Accepted: the charts that end up
/// sliced that deep are far less accurate than that grid. Raster ancestors
/// are likewise upscaled rather than resampled from the source.
///
/// `MapLibre` searches at most [`CLIENT_ANCESTOR_SEARCH`] levels up, so
/// gaps deeper than that draw nothing there: where only charts more than
/// that many zooms coarser than the max zoom cover, the finest zooms stay
/// blank. Cap `--max-zoom` accordingly, or don't quilt sparsely.
pub struct Quilt<S> {
    items: Vec<S>,
    /// Per-item data, indexed like `items`.
    meta: Vec<ItemMeta>,
    zoom_offset: f64,
    sparse: bool,
}

/// How many zoom levels up `MapLibre` GL JS looks for a stored ancestor of a
/// missing tile (`TileManager.maxUnderzooming`).
pub const CLIENT_ANCESTOR_SEARCH: u8 = 10;

/// What [`Quilt::render_tile`] needs from an item on every tile. The bbox and
/// native zoom come from cheap [`TileSource`] methods and are computed up
/// front; the detail zoom may need the item's content (see [`TileSource`]'s
/// lazy items), so a sparse quilt computes it on first use.
struct ItemMeta {
    bbox: Bbox,
    native_zoom: u8,
    /// Single-flight: concurrent tiles needing it wait for one computation
    /// (one load of a lazy item); a failure isn't stored.
    detail_zoom: OnceCell<u8>,
}

impl<S: TileSource> Quilt<S> {
    /// `zoom_offset` shifts every item's native scale to a zoom level
    /// (positive = finer, negative = coarser). `sparse` enables
    /// [`TileOutcome::Omitted`].
    #[must_use]
    pub fn new(items: Vec<S>, zoom_offset: f64, sparse: bool) -> Self {
        let meta = items
            .iter()
            .map(|item| ItemMeta {
                bbox: item.bbox(),
                native_zoom: zoom_from_scale(item.native_scale(), zoom_offset),
                detail_zoom: OnceCell::new(),
            })
            .collect();
        Self {
            items,
            meta,
            zoom_offset,
            sparse,
        }
    }

    /// Item `i`'s [`TileSource::detail_zoom`]: asked once, and not
    /// concurrently; asked again only after a failure.
    fn detail_zoom(&self, i: usize) -> Result<u8> {
        self.meta[i]
            .detail_zoom
            .get_or_try_init(|| self.items[i].detail_zoom(self.zoom_offset))
            .copied()
    }

    /// Whether every item in `items` has a detail zoom coarser than `z`.
    /// Stops at the first that doesn't, so later items aren't asked; an
    /// item natively at `z` or finer answers without being asked (its detail
    /// zoom is never coarser than its native zoom).
    fn all_coarser_than(&self, items: &[usize], z: u8) -> Result<bool> {
        if items.iter().any(|&i| self.meta[i].native_zoom >= z) {
            return Ok(false);
        }
        for &i in items {
            if self.detail_zoom(i)? >= z {
                return Ok(false);
            }
        }
        Ok(true)
    }

    #[must_use]
    pub fn items(&self) -> &[S] {
        &self.items
    }

    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// Compute `(zoom_floor, zoom_ceil, overall_bbox)`: the tile zoom range
    /// and combined `WGS84` extent that a [`write_pmtiles`] batch or a live
    /// server's `/style.json` should advertise.
    ///
    /// `zoom_floor` is the coarsest native zoom among the items, minus 2 (so
    /// the finest chart's detail still fills down a couple of levels below
    /// its own native zoom). `zoom_ceil` is the finest native zoom, capped by
    /// `max_zoom` when given.
    ///
    /// # Errors
    /// Returns an error if there are no items or `max_zoom` is below the
    /// data's minimum zoom.
    pub fn zoom_range_and_bounds(&self, max_zoom: Option<u8>) -> Result<(u8, u8, Bbox)> {
        anyhow::ensure!(!self.items.is_empty(), "no items to tile");
        let native_zooms = || self.meta.iter().map(|m| m.native_zoom);
        let zoom_floor = native_zooms().min().unwrap_or(0).saturating_sub(2);
        let zoom_ceil_native = native_zooms().max().unwrap_or(0);
        let zoom_ceil = match max_zoom {
            Some(cap) if cap < zoom_floor => {
                anyhow::bail!("max_zoom {cap} is below the data's minimum zoom {zoom_floor}");
            }
            Some(cap) => cap.min(zoom_ceil_native),
            None => zoom_ceil_native,
        };

        let overall = self
            .meta
            .iter()
            .fold(Bbox::bottom(), |acc, m| acc.join(&m.bbox));

        Ok((zoom_floor, zoom_ceil, overall))
    }

    /// Render one tile from the items' combined coverage.
    ///
    /// Safe to call concurrently and on demand: the only state shared across
    /// calls is each item's memoised detail zoom. Backs both
    /// [`write_pmtiles`]'s batch loop and a live tile server.
    ///
    /// # Errors
    /// Returns an error if an item's content can't be loaded (see
    /// [`TileSource`]), or merging or encoding this tile's contents fails
    /// ([`TileAccumulator::push`] / [`TileAccumulator::encode`]).
    pub fn render_tile(&self, tile: TileXyz) -> Result<TileOutcome> {
        profiling::scope!("tile");
        let (z, col, row) = (tile.z(), tile.x(), tile.y());
        let tile_wgs84 = Bbox::from(xyz_to_bbox(z, col, row, col, row));

        // Candidates: items whose bbox overlaps this tile.
        let mut candidates: Vec<usize> = {
            profiling::scope!("Collecting candidates");
            (0..self.items.len())
                .filter(|&i| self.meta[i].bbox.overlaps(&tile_wgs84))
                .collect()
        };

        if candidates.is_empty() {
            return Ok(TileOutcome::Uncovered);
        }
        // Contributors are a subset of candidates, so this decides the
        // common open-water case without any polygon algebra.
        if self.sparse && self.all_coarser_than(&candidates, z)? {
            return Ok(TileOutcome::Omitted);
        }

        // Sort by zoom fit alone (catalog data): coarsest-appropriate zoom
        // first, finer after, too coarse last. Equally fitting candidates
        // end up adjacent; the greedy loop orders each such group by
        // tie-break key only when it reaches it.
        candidates.sort_unstable_by_key(|&i| self.zoom_fit(i, z));
        let contributions = {
            profiling::scope!("Assigning coverage");
            self.assign_coverage(&candidates, z, tile_wgs84)?
        };

        if contributions.is_empty() {
            return Ok(TileOutcome::Uncovered);
        }
        let contributors: Vec<usize> = contributions.iter().map(|&(i, _)| i).collect();
        if self.sparse && self.all_coarser_than(&contributors, z)? {
            return Ok(TileOutcome::Omitted);
        }

        let items = &self.items;
        let tile_merc = tile_wgs84.to_web_mercator();
        let tile_scale = scale_from_zoom(z, self.zoom_offset);
        let mut acc = S::Accumulator::empty();
        {
            profiling::scope!("Collecting features");
            for (i, contrib) in contributions {
                acc.push(items[i].render(&TileGeom {
                    geom: contrib,
                    merc: tile_merc,
                    scale: tile_scale,
                })?)?;
            }
        }

        let bytes = acc.encode()?;
        Ok(if bytes.is_empty() {
            TileOutcome::Blank
        } else {
            TileOutcome::Rendered(bytes)
        })
    }

    /// Greedy coverage over `candidates`, sorted by [`Self::zoom_fit`]:
    /// include an item only if its contribution adds area not yet covered.
    /// Real polygon algebra correctly handles disjoint coverage areas (e.g.
    /// NE + SW ≠ full tile), unlike a bounding-box hull. All of it runs on
    /// the tile's own grid (see [`TileGrid`]), so the tile counts as covered
    /// exactly when it is, and every real contribution counts, however thin.
    ///
    /// A group of equally fitting candidates is ordered by tie-break key
    /// only when the loop reaches it, and only among the members whose bbox
    /// still meets the uncovered area; groups past the point where the tile
    /// is covered are asked nothing.
    fn assign_coverage(
        &self,
        candidates: &[usize],
        z: u8,
        tile_wgs84: Bbox,
    ) -> Result<Vec<(usize, MultiPolygon)>> {
        let grid = TileGrid::new(tile_wgs84)?;
        let mut uncovered = grid.whole();
        let mut contributions = Vec::new();
        for group in candidates.chunk_by(|&a, &b| self.zoom_fit(a, z) == self.zoom_fit(b, z)) {
            // From the catalog: a member whose bbox misses what's left of
            // the tile can't contribute. What's left, not its bbox: once the
            // fitting charts cover most of a tile, the gaps they leave still
            // span it, and the finer charts inside their coverage would all
            // pass a bbox test.
            let remaining = Bbox::from(&uncovered);
            let mut members: Vec<usize> = group
                .iter()
                .copied()
                .filter(|&i| {
                    let bbox = self.meta[i].bbox;
                    bbox.overlaps(&remaining)
                        && uncovered.intersects(&Rect::new(
                            (bbox.west, bbox.south),
                            (bbox.east, bbox.north),
                        ))
                })
                .collect();
            if members.len() > 1 {
                self.order_by_tiebreak(&mut members)?;
            }
            for i in members {
                let coverage = grid.cut(&self.items[i].coverage()?);
                let contrib = grid.intersection(&coverage, &uncovered);
                if contrib.0.is_empty() {
                    continue;
                }
                uncovered = grid.difference(&uncovered, &contrib);
                contributions.push((i, contrib));
                if uncovered.0.is_empty() {
                    return Ok(contributions);
                }
            }
        }
        Ok(contributions)
    }

    /// Order `members` (equally fitting candidates) newest edition first —
    /// see [`TileSource::tiebreak`] — and equal editions by source, so that
    /// the input order never decides which chart a tile shows.
    fn order_by_tiebreak(&self, members: &mut [usize]) -> Result<()> {
        let mut keyed = members
            .iter()
            .map(|&i| {
                Ok((
                    Reverse(self.items[i].tiebreak()?),
                    self.items[i].source(),
                    i,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        keyed.sort_unstable_by(|a, b| (&a.0, &a.1).cmp(&(&b.0, &b.1)));
        for (slot, (_, _, i)) in members.iter_mut().zip(keyed) {
            *slot = i;
        }
        Ok(())
    }

    /// Sort key fitting item `i` to tile zoom `z`: coarsest-appropriate zoom
    /// first, finer after, too coarse last.
    fn zoom_fit(&self, i: usize, z: u8) -> (bool, i32) {
        let (native, zi) = (i32::from(self.meta[i].native_zoom), i32::from(z));
        (native < zi, if native >= zi { native } else { -native })
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

#[allow(clippy::literal_string_with_formatting_args)]
fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {spinner:.green} {msg:20}  {elapsed_precise}  [{wide_bar:.cyan/blue}]  {human_pos}/{human_len}  ({per_sec}, {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

/// Archive metadata. `sparse: true` follows tileserver-gl, which then
/// answers a missing tile with 404 (fall back to an ancestor) instead of 204
/// (draw nothing); omitted when false so a raster archive keeps
/// tileserver-gl's raster default.
fn build_metadata(sparse: bool) -> String {
    let mut metadata = serde_json::json!({
        "name": "chart",
        "description": "Nautical chart — converted from OESU",
        "vector_layers": []
    });
    if sparse {
        metadata["sparse"] = serde_json::Value::Bool(true);
    }
    metadata.to_string()
}

#[cfg(test)]
mod tests {
    use image::Rgba;
    use martin_tile_utils::wgs84_to_webmercator;

    use super::*;
    use crate::rnc::RncCell;
    use crate::rnc_source::RasterAccumulator;

    fn xyz(z: u8, col: u32, row: u32) -> TileXyz {
        TileXyz::new(z, col, row).expect("tile on the Web Mercator grid")
    }

    #[test]
    fn tile_xyz_accepts_exactly_the_renderable_grid() {
        let max = martin_tile_utils::MAX_ZOOM;
        let last = (1u32 << max) - 1;
        let tile = TileXyz::new(max, last, last).expect("deepest zoom's last tile");
        assert_eq!((tile.z(), tile.x(), tile.y()), (max, last, last));
        // `xyz_to_bbox` panics past MAX_ZOOM, so no tile may exist there.
        assert_eq!(TileXyz::new(max + 1, 0, 0), None);
        assert_eq!(TileXyz::new(u8::MAX, 0, 0), None);
        // Column/row one past the 2^z grid edge.
        assert_eq!(TileXyz::new(3, 8, 0), None);
        assert_eq!(TileXyz::new(3, 0, 8), None);
        assert!(TileXyz::new(3, 7, 7).is_some());
        assert_eq!(TileXyz::new(0, 1, 0), None);
    }

    /// Build a minimal single-tile `.rnc` byte buffer: one `8×8` solid-color
    /// PNG, rectangular `cover` (the bbox itself), given `edate`.
    fn build_rnc(bbox: Bbox, color: [u8; 4], scale: f64, edate: &str) -> Vec<u8> {
        build_rnc_covered(bbox, &[], color, scale, edate)
    }

    /// [`build_rnc`] with an explicit `cover`: rings of `(lon, lat)` vertices
    /// (empty: the bbox itself).
    #[allow(clippy::cast_possible_truncation)] // test PNG size is a tiny constant
    fn build_rnc_covered(
        bbox: Bbox,
        cover: &[&[(f64, f64)]],
        color: [u8; 4],
        scale: f64,
        edate: &str,
    ) -> Vec<u8> {
        let img = image::RgbaImage::from_pixel(8, 8, Rgba(color));
        let mut png = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
            .expect("encoding test PNG cannot fail");

        let offsets = [
            16 + 3 * 4,
            16 + 3 * 4 + png.len() as u32,
            16 + 3 * 4 + png.len() as u32,
        ];
        let mut buf = rnc_format::SIGNATURE.to_vec();
        buf.push(4); // format version
        buf.extend_from_slice(&1u32.to_le_bytes()); // cols
        buf.extend_from_slice(&1u32.to_le_bytes()); // rows
        for o in offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        buf.extend_from_slice(&png);
        let footer = serde_json::json!({
            "cover": cover
                .iter()
                .map(|ring| {
                    ring.iter()
                        .flat_map(|&(lon, lat)| {
                            <[f64; 2]>::from(rnc_format::wgs84_to_rnc_merc(lon, lat))
                        })
                        .collect::<Vec<f64>>()
                })
                .collect::<Vec<_>>(),
            "lat0": bbox.south, "lat1": bbox.north,
            "lon0": bbox.west, "lon1": bbox.east,
            "edate": edate, "name": "TEST", "scale": scale,
        });
        buf.extend_from_slice(serde_json::to_string(&footer).unwrap().as_bytes());
        buf
    }

    /// Two fully-overlapping cells with identical `native_scale` (so they
    /// land in the same floored tile zoom and tie the zoom-fit sort key)
    /// but different edition dates. The newer one must win the whole tile
    /// — its color, not the older's, comes out — regardless of input order.
    #[test]
    fn newer_edition_wins_tie_at_same_zoom() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        // Sources sort the older cell first, so only the edition (not the
        // tie fallback by source, nor the input order) can pick the newer.
        let older = RncCell::parse(
            "A-OLDER".to_owned(),
            build_rnc(bbox, [200, 30, 30, 255], 3_000_000.0, "01/01/2020"),
        )
        .expect("older cell parses");
        let newer = RncCell::parse(
            "B-NEWER".to_owned(),
            build_rnc(bbox, [30, 200, 30, 255], 3_000_000.0, "01/01/2026"),
        )
        .expect("newer cell parses");
        assert!(
            older.edition_date() < newer.edition_date(),
            "fixture sanity: OLDER must parse strictly before NEWER"
        );

        // Render at the cell's own natural zoom so its bbox fills (most
        // of) the tile — center is reliably inside its contribution.
        let z = zoom_from_scale(3_000_000, 0.0);
        let (col, row, _, _) = bbox_to_xyz(bbox.west, bbox.south, bbox.east, bbox.north, z);

        // Older listed first: only the tiebreak (not array order) should
        // determine which one wins.
        let quilt = Quilt::new(vec![older, newer], 0.0, false);
        let TileOutcome::Rendered(bytes) = quilt.render_tile(xyz(z, col, row)).expect("renders")
        else {
            panic!("tile has coverage");
        };
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .expect("re-decodes")
            .to_rgba8();

        // Sample the bbox's own centroid, projected into this tile's pixel
        // space — robust regardless of where the bbox sits within the tile.
        let tile_merc = Bbox::from(xyz_to_bbox(z, col, row, col, row)).to_web_mercator();
        let (centroid_x, centroid_y) = wgs84_to_webmercator(
            f64::midpoint(bbox.west, bbox.east),
            f64::midpoint(bbox.south, bbox.north),
        );
        let extent = f64::from(crate::rnc_source::TILE_PX);
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let px =
            (((centroid_x - tile_merc.west) / (tile_merc.east - tile_merc.west)) * extent) as u32;
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let py = (((tile_merc.north - centroid_y) / (tile_merc.north - tile_merc.south)) * extent)
            as u32;

        assert_eq!(
            img.get_pixel(px, py).0,
            [30, 200, 30, 255],
            "newer edition's color should win the fully-overlapping tile"
        );
    }

    /// Two raster cells tied in everything but their source draw the same
    /// tile whichever comes first in the input.
    #[test]
    fn tied_raster_cells_draw_the_same_in_either_input_order() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let cell = |source: &str, color| {
            RncCell::parse(
                source.to_owned(),
                build_rnc(bbox, color, 3_000_000.0, "01/01/2026"),
            )
            .expect("cell parses")
        };
        let z = zoom_from_scale(3_000_000, 0.0);
        let (col, row, _, _) = bbox_to_xyz(bbox.west, bbox.south, bbox.east, bbox.north, z);
        let render = |names: [(&str, [u8; 4]); 2]| {
            let quilt = Quilt::new(names.map(|(s, c)| cell(s, c)).into(), 0.0, false);
            quilt.render_tile(xyz(z, col, row)).expect("renders")
        };
        let (red, green) = (("a.rnc", [200, 30, 30, 255]), ("b.rnc", [30, 200, 30, 255]));
        assert_eq!(render([red, green]), render([green, red]));
    }

    /// A sparse quilt must store every tile some contributor still adds
    /// detail to — including mixed tiles — omit the rest, and keep reporting
    /// uncovered tiles as such (a live server answers those 204, not 404).
    #[test]
    fn sparse_quilt_omits_exactly_the_tiles_past_every_contributors_detail_zoom() {
        let coarse = RncCell::parse(
            "COARSE".to_owned(),
            build_rnc(
                Bbox {
                    west: 10.0,
                    south: 56.0,
                    east: 11.0,
                    north: 57.0,
                },
                [200, 30, 30, 255],
                3_000_000.0,
                "01/01/2026",
            ),
        )
        .expect("coarse cell parses");
        let fine = RncCell::parse(
            "FINE".to_owned(),
            build_rnc(
                Bbox {
                    west: 10.4,
                    south: 56.4,
                    east: 10.6,
                    north: 56.6,
                },
                [30, 200, 30, 255],
                50_000.0,
                "01/01/2026",
            ),
        )
        .expect("fine cell parses");
        assert_eq!(
            (
                coarse.detail_zoom(0.0).unwrap(),
                fine.detail_zoom(0.0).unwrap()
            ),
            (7, 13),
            "fixture sanity: native zooms decide both detail zooms"
        );

        let sparse = Quilt::new(vec![coarse, fine], 0.0, true);
        let outcome = |lon: f64, lat: f64, z: u8| {
            let (col, row, _, _) = bbox_to_xyz(lon, lat, lon, lat, z);
            match sparse.render_tile(xyz(z, col, row)).expect("renders") {
                TileOutcome::Uncovered => "uncovered",
                TileOutcome::Blank => "blank",
                TileOutcome::Omitted => "omitted",
                TileOutcome::Rendered(_) => "rendered",
            }
        };

        // Coarse-only area: stored up to its detail zoom, omitted past it.
        assert_eq!(outcome(10.05, 56.95, 7), "rendered");
        assert_eq!(outcome(10.05, 56.95, 10), "omitted");
        // Mixed tile: the fine cell still adds detail, so the coarse part
        // must be stored alongside it.
        assert_eq!(outcome(10.5, 56.5, 10), "rendered");
        // Fine-only tiles: stored through its detail zoom, omitted past it.
        assert_eq!(outcome(10.5, 56.5, 13), "rendered");
        assert_eq!(outcome(10.5, 56.5, 14), "omitted");
        // Nothing there at all is never "omitted".
        assert_eq!(outcome(50.0, 0.0, 10), "uncovered");

        let dense = Quilt::new(sparse.items, 0.0, false);
        let (col, row, _, _) = bbox_to_xyz(10.05, 56.95, 10.05, 56.95, 10);
        assert!(
            matches!(
                dense.render_tile(xyz(10, col, row)).expect("renders"),
                TileOutcome::Rendered(_)
            ),
            "a dense quilt renders every covered tile"
        );
    }

    /// Color of the pixel at `(lon, lat)` in raster tile `(z, col, row)`.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    fn pixel_at(png: &[u8], (z, col, row): (u8, u32, u32), lon: f64, lat: f64) -> [u8; 4] {
        let img = image::load_from_memory_with_format(png, image::ImageFormat::Png)
            .expect("tile decodes")
            .to_rgba8();
        let merc = Bbox::from(xyz_to_bbox(z, col, row, col, row)).to_web_mercator();
        let (x, y) = wgs84_to_webmercator(lon, lat);
        let px = f64::from(crate::rnc_source::TILE_PX);
        let ix = ((x - merc.west) / (merc.east - merc.west) * px) as u32;
        let iy = ((merc.north - y) / (merc.north - merc.south) * px) as u32;
        img.get_pixel(ix.min(255), iy.min(255)).0
    }

    /// What a sparse archive leaves out must look, at each spot, like the
    /// nearest stored ancestor the client falls back to — and that must be
    /// what the dense quilt renders there. Covers the case where ancestor
    /// and child pick different charts: CLEAR (native z13, transparent)
    /// wins its z13 tiles and draws nothing, while at z12 MID (native z12,
    /// blue) wins the same spot. Leaving that z13 tile out would let MID's
    /// blue show through, so the archive must store a blank instead.
    #[test]
    #[allow(clippy::too_many_lines)] // one fixture, one invariant sweep
    fn sparse_archive_omissions_match_their_nearest_stored_ancestor() {
        let boxes = [
            (
                "COARSE",
                (10.4, 56.4, 10.6, 56.6),
                [200, 30, 30, 255],
                3_000_000.0,
            ),
            (
                "FINE",
                (10.43, 56.43, 10.5, 56.5),
                [30, 200, 30, 255],
                50_000.0,
            ),
            (
                "MID",
                (10.52, 56.52, 10.6, 56.6),
                [30, 30, 200, 255],
                100_000.0,
            ),
            ("CLEAR", (10.54, 56.54, 10.6, 56.6), [0, 0, 0, 0], 50_000.0),
        ];
        let items = || {
            boxes
                .iter()
                .map(|&(name, (west, south, east, north), color, scale)| {
                    let bbox = Bbox {
                        west,
                        south,
                        east,
                        north,
                    };
                    RncCell::parse(name.to_owned(), build_rnc(bbox, color, scale, "01/01/2026"))
                        .expect("cell parses")
                })
                .collect::<Vec<_>>()
        };
        let sparse = Quilt::new(items(), 0.0, true);
        let dense = Quilt::new(items(), 0.0, false);
        let (floor, ceil, overall) = sparse.zoom_range_and_bounds(None).expect("zoom range");
        let stored = archive_tiles(&sparse, floor..=ceil, overall).expect("archive renders");
        let blank = RasterAccumulator::blank().expect("blank encodes");
        assert!(
            stored.values().any(|(_, bytes)| *bytes == blank),
            "fixture sanity: CLEAR's z13 tiles must be stored as blanks"
        );
        let dense_stored = archive_tiles(&dense, floor..=ceil, overall).expect("dense renders");
        assert!(
            dense_stored.values().all(|(_, bytes)| *bytes != blank),
            "a dense archive leaves blank tiles out (it is never sliced from an ancestor)"
        );

        // Sample each tile at its center, unless that is within two pixels
        // of the coarsest stored tile of a chart edge (raster upscaling
        // smears edges; the claim is about which chart shows, not about
        // resampling).
        let near_edge = |lon: f64, lat: f64, tol: f64| {
            boxes.iter().any(|(_, (w, s, e, n), _, _)| {
                [lon - w, e - lon, lat - s, n - lat]
                    .iter()
                    .any(|d| d.abs() < tol)
            })
        };
        let mut omitted = 0;
        for z in floor..=ceil {
            let (c0, r0, c1, r1) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            for col in c0..=c1 {
                for row in r0..=r1 {
                    let id = |z, col, row| {
                        TileId::from(TileCoord::new(z, col, row).expect("valid coord")).value()
                    };
                    if stored.contains_key(&id(z, col, row)) {
                        continue;
                    }
                    let expected = match dense.render_tile(xyz(z, col, row)).expect("dense renders")
                    {
                        TileOutcome::Uncovered => continue,
                        TileOutcome::Blank => blank.clone(),
                        TileOutcome::Rendered(bytes) => bytes,
                        TileOutcome::Omitted => unreachable!("dense quilts never omit"),
                    };
                    omitted += 1;
                    let tile = Bbox::from(xyz_to_bbox(z, col, row, col, row));
                    let (lon, lat) = (
                        f64::midpoint(tile.west, tile.east),
                        f64::midpoint(tile.south, tile.north),
                    );
                    let (az, (acol, arow), ancestor) = (1..=CLIENT_ANCESTOR_SEARCH)
                        .filter_map(|up| z.checked_sub(up))
                        .find_map(|az| {
                            let shift = z - az;
                            let at = (col >> shift, row >> shift);
                            stored.get(&id(az, at.0, at.1)).map(|(_, b)| (az, at, b))
                        })
                        .unwrap_or_else(|| {
                            panic!("z{z}/{col}/{row}: no stored ancestor within reach")
                        });
                    let ancestor_px_deg = 360.0 / f64::from(1u32 << az) / 256.0;
                    if near_edge(lon, lat, 2.0 * ancestor_px_deg) {
                        continue;
                    }
                    assert_eq!(
                        pixel_at(ancestor, (az, acol, arow), lon, lat),
                        pixel_at(&expected, (z, col, row), lon, lat),
                        "z{z}/{col}/{row}: fallback to z{az} must show what a dense render shows"
                    );
                }
            }
        }
        assert!(omitted > 0, "fixture sanity: something must be omitted");

        let metadata = |sparse| -> serde_json::Value {
            serde_json::from_str(&build_metadata(sparse)).expect("metadata is JSON")
        };
        assert_eq!(metadata(true)["sparse"], true, "tileserver-gl reads this");
        assert!(
            metadata(false).get("sparse").is_none(),
            "dense raster archives keep tileserver-gl's raster default"
        );
    }

    #[test]
    fn zoom_range_and_bounds_empty_items_must_bail() {
        let result = Quilt::<RncCell>::new(Vec::new(), 0.0, false).zoom_range_and_bounds(None);
        assert!(
            result.is_err(),
            "empty items must bail, but got Ok with ±Infinity bbox"
        );
    }

    /// One 1:3 000 000 cell (native and detail zoom 7, zoom floor 5).
    fn overview_quilt(sparse: bool) -> Quilt<RncCell> {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let cell = RncCell::parse(
            "OVERVIEW".to_owned(),
            build_rnc(bbox, [200, 30, 30, 255], 3_000_000.0, "01/01/2026"),
        )
        .expect("cell parses");
        Quilt::new(vec![cell], 0.0, sparse)
    }

    #[test]
    fn max_zoom_may_not_go_below_the_zoom_floor() {
        let quilt = overview_quilt(false);
        assert!(quilt.zoom_range_and_bounds(Some(4)).is_err());
        let (floor, ceil, _) = quilt.zoom_range_and_bounds(Some(5)).expect("at the floor");
        assert_eq!((floor, ceil), (5, 5));
        let (_, ceil, _) = quilt
            .zoom_range_and_bounds(Some(6))
            .expect("above the floor");
        assert_eq!(ceil, 6);
    }

    /// A finer cell whose bbox reaches a tile but whose cover polygon does
    /// not is a candidate, not a contributor: the tile is decided by the
    /// coarse cell alone and omitted, even though the finer cell's detail
    /// zoom would keep the tile if it contributed.
    #[test]
    fn bbox_only_candidates_do_not_keep_a_tile() {
        let coarse = RncCell::parse(
            "COARSE".to_owned(),
            build_rnc(
                Bbox {
                    west: 10.0,
                    south: 56.0,
                    east: 11.0,
                    north: 57.0,
                },
                [200, 30, 30, 255],
                3_000_000.0,
                "01/01/2026",
            ),
        )
        .expect("coarse cell parses");
        // Cover is the triangle below the SW–NE diagonal of its bbox.
        let triangle: &[(f64, f64)] = &[(10.4, 56.4), (10.6, 56.4), (10.6, 56.6)];
        let fine = RncCell::parse(
            "FINE".to_owned(),
            build_rnc_covered(
                Bbox {
                    west: 10.4,
                    south: 56.4,
                    east: 10.6,
                    north: 56.6,
                },
                &[triangle],
                [30, 200, 30, 255],
                50_000.0,
                "01/01/2026",
            ),
        )
        .expect("fine cell parses");
        let quilt = Quilt::new(vec![coarse, fine], 0.0, true);

        // z12 tile in the bbox's NW corner, wholly above the diagonal.
        let (col, row, _, _) = bbox_to_xyz(10.42, 56.58, 10.42, 56.58, 12);
        let tile = Bbox::from(xyz_to_bbox(12, col, row, col, row));
        assert!(
            tile.east > 10.4 && tile.north > 56.4 && tile.east - 10.4 < tile.south - 56.4,
            "fixture sanity: tile overlaps the fine bbox but not its triangle: {tile:?}"
        );
        assert_eq!(
            quilt.render_tile(xyz(12, col, row)).expect("renders"),
            TileOutcome::Omitted
        );
    }

    // ── Lazy items: what a quilt asks for ────────────────────────────────

    /// Counts the content requests a lazy item would have to load for.
    #[derive(Default)]
    struct Asked {
        coverage: std::sync::atomic::AtomicUsize,
        detail_zoom: std::sync::atomic::AtomicUsize,
        tiebreak: std::sync::atomic::AtomicUsize,
        render: std::sync::atomic::AtomicUsize,
    }

    impl Asked {
        fn count(counter: &std::sync::atomic::AtomicUsize) -> usize {
            counter.load(Ordering::Relaxed)
        }

        /// `(coverage, detail_zoom, tiebreak, render)` requests so far.
        fn all(&self) -> (usize, usize, usize, usize) {
            (
                Self::count(&self.coverage),
                Self::count(&self.detail_zoom),
                Self::count(&self.tiebreak),
                Self::count(&self.render),
            )
        }
    }

    /// A stand-in lazy item: `bbox` and `scale` are free, every other
    /// method counts as a load and fails when `broken`.
    struct Probe {
        name: &'static str,
        bbox: Bbox,
        scale: u32,
        edition: u32,
        broken: bool,
        /// How long each load takes, so concurrent tiles can overlap on it.
        load_time: std::time::Duration,
        asked: Asked,
    }

    impl Probe {
        fn new(name: &'static str, bbox: Bbox, zoom: u8, edition: u32) -> Self {
            // 10% finer than the band's rounded edge, well before the next.
            let scale = scale_from_zoom(zoom, 0.0) / 10 * 9;
            assert_eq!(zoom_from_scale(scale, 0.0), zoom, "fixture sanity");
            Self {
                name,
                bbox,
                scale,
                edition,
                broken: false,
                load_time: std::time::Duration::ZERO,
                asked: Asked::default(),
            }
        }

        fn load(&self, counter: &std::sync::atomic::AtomicUsize) -> Result<()> {
            counter.fetch_add(1, Ordering::Relaxed);
            std::thread::sleep(self.load_time);
            anyhow::ensure!(!self.broken, "{} failed to load", self.name);
            Ok(())
        }
    }

    /// Joins the names of a tile's contributors, in quilt order.
    struct Names(Vec<&'static str>);

    impl TileAccumulator for Names {
        type Content = &'static str;
        fn empty() -> Self {
            Self(Vec::new())
        }
        fn push(&mut self, name: &'static str) -> Result<()> {
            self.0.push(name);
            Ok(())
        }
        fn encode(self) -> Result<Vec<u8>> {
            Ok(self.0.join(",").into_bytes())
        }
        fn blank() -> Result<Vec<u8>> {
            Ok(b"-".to_vec())
        }
    }

    impl TileSource for Probe {
        type Content = &'static str;
        type Accumulator = Names;
        type Tiebreaker = u32;

        fn source(&self) -> String {
            self.name.to_owned()
        }
        fn bbox(&self) -> Bbox {
            self.bbox
        }
        fn coverage(&self) -> Result<MultiPolygon> {
            self.load(&self.asked.coverage)?;
            Ok(MultiPolygon::from(geo::Polygon::from(self.bbox)))
        }
        fn native_scale(&self) -> u32 {
            self.scale
        }
        fn detail_zoom(&self, zoom_offset: f64) -> Result<u8> {
            self.load(&self.asked.detail_zoom)?;
            Ok(zoom_from_scale(self.scale, zoom_offset))
        }
        fn tiebreak(&self) -> Result<u32> {
            self.load(&self.asked.tiebreak)?;
            Ok(self.edition)
        }
        fn render(&self, _tile: &TileGeom) -> Result<&'static str> {
            self.load(&self.asked.render)?;
            Ok(self.name)
        }
        fn tile_type() -> pmtiles::TileType {
            pmtiles::TileType::Mvt
        }
    }

    /// The z10 tile under (10.5°E, 56.5°N), and a bbox 0.1° around it.
    fn probe_tile() -> (TileXyz, Bbox) {
        probe_tile_at(10, 10.5, 56.5, 0.1)
    }

    /// The zoom-`z` tile under `(lon, lat)`, and a bbox `margin` degrees
    /// around it.
    fn probe_tile_at(z: u8, lon: f64, lat: f64, margin: f64) -> (TileXyz, Bbox) {
        let (col, row, _, _) = bbox_to_xyz(lon, lat, lon, lat, z);
        let tile = Bbox::from(xyz_to_bbox(z, col, row, col, row));
        let around = Bbox {
            west: tile.west - margin,
            south: tile.south - margin,
            east: tile.east + margin,
            north: tile.north + margin,
        };
        (xyz(z, col, row), around)
    }

    fn rendered(outcome: TileOutcome) -> String {
        match outcome {
            TileOutcome::Rendered(bytes) => String::from_utf8(bytes).expect("names"),
            other => panic!("expected a rendered tile, got {other:?}"),
        }
    }

    /// Building a quilt reads only the free metadata, items outside a tile
    /// are never asked for content, and a dense quilt never asks for detail
    /// zooms.
    #[test]
    fn a_quilt_asks_no_content_up_front_nor_of_items_off_the_tile() {
        let (tile, around) = probe_tile();
        let far = Bbox {
            west: -60.0,
            south: -10.0,
            east: -59.0,
            north: -9.0,
        };
        let quilt = Quilt::new(
            vec![
                Probe::new("FIT", around, 10, 1),
                Probe::new("FAR", far, 10, 1),
            ],
            0.0,
            false,
        );
        for item in quilt.items() {
            assert_eq!(item.asked.all(), (0, 0, 0, 0), "{} at build", item.name);
        }

        assert_eq!(rendered(quilt.render_tile(tile).expect("renders")), "FIT");
        let asked: Vec<_> = quilt.items().iter().map(|i| i.asked.all()).collect();
        assert_eq!(asked, [(1, 0, 0, 1), (0, 0, 0, 0)]);
    }

    /// Items tied on zoom fit are ordered by their tie-break key — the only
    /// ones asked for it — and the newest comes first.
    #[test]
    fn only_tied_candidates_are_asked_for_their_tiebreak() {
        let (tile, around) = probe_tile();
        let quilt = Quilt::new(
            vec![
                Probe::new("OLD", around, 10, 1),
                Probe::new("NEW", around, 10, 2),
                Probe::new("FINER", around, 12, 3),
            ],
            0.0,
            false,
        );
        let names = rendered(quilt.render_tile(tile).expect("renders"));
        assert_eq!(names.split(',').next(), Some("NEW"), "{names}");
        let tiebreaks: Vec<_> = quilt.items().iter().map(|i| i.asked.all().2).collect();
        assert_eq!(tiebreaks, [1, 1, 0]);
    }

    /// A tied group the greedy loop never reaches, because a better-fitting
    /// item already covers the tile, is asked nothing.
    #[test]
    fn a_tied_group_behind_a_covering_item_is_not_asked_for_dates() {
        let (tile, exact) = probe_tile_at(10, 10.5, 56.5, 0.0);
        let quilt = Quilt::new(
            vec![
                Probe::new("FIT", exact, 10, 1),
                Probe::new("LATE_A", exact, 12, 1),
                Probe::new("LATE_B", exact, 12, 2),
            ],
            0.0,
            false,
        );
        assert_eq!(rendered(quilt.render_tile(tile).expect("renders")), "FIT");
        let tiebreaks: Vec<_> = quilt.items().iter().map(|i| i.asked.all().2).collect();
        assert_eq!(tiebreaks, [0, 0, 0]);
    }

    /// In a reached group, a member whose bbox misses the still uncovered
    /// part of the tile can't contribute: it isn't asked for its date, and
    /// a single remaining member needs no ordering at all.
    #[test]
    fn a_tied_member_off_the_remaining_gap_is_not_asked_for_its_date() {
        let (tile, exact) = probe_tile_at(10, 10.5, 56.5, 0.0);
        let mid = f64::midpoint(exact.west, exact.east);
        let west_half = Bbox { east: mid, ..exact };
        let east_half = Bbox { west: mid, ..exact };
        let west_quarter = Bbox {
            east: f64::midpoint(exact.west, mid),
            ..exact
        };
        let quilt = Quilt::new(
            vec![
                Probe::new("WEST", west_half, 10, 1),
                Probe::new("EAST", east_half, 12, 1),
                Probe::new("OFF", west_quarter, 12, 2),
            ],
            0.0,
            false,
        );
        assert_eq!(
            rendered(quilt.render_tile(tile).expect("renders")),
            "WEST,EAST"
        );
        let asked: Vec<_> = quilt.items().iter().map(|i| i.asked.all()).collect();
        assert_eq!(asked, [(1, 0, 0, 1), (1, 0, 0, 1), (0, 0, 0, 0)]);
    }

    /// What decides whether a candidate is asked is the uncovered part of
    /// the tile itself, not its bbox: two coarse items covering opposite
    /// quarters leave a gap whose bbox is the whole tile, yet a finer item
    /// inside a covered quarter can't contribute and isn't loaded at all.
    #[test]
    fn an_item_inside_the_covered_part_is_not_asked_whatever_the_gap_spans() {
        let (tile, exact) = probe_tile_at(10, 10.5, 56.5, 0.0);
        let (mid_x, mid_y) = (
            f64::midpoint(exact.west, exact.east),
            f64::midpoint(exact.south, exact.north),
        );
        let inset = |b: Bbox| {
            let (dx, dy) = ((b.east - b.west) / 4.0, (b.north - b.south) / 4.0);
            Bbox {
                west: b.west + dx,
                south: b.south + dy,
                east: b.east - dx,
                north: b.north - dy,
            }
        };
        let south_west = Bbox {
            east: mid_x,
            north: mid_y,
            ..exact
        };
        let north_east = Bbox {
            west: mid_x,
            south: mid_y,
            ..exact
        };
        let north_west = Bbox {
            east: mid_x,
            south: mid_y,
            ..exact
        };
        let quilt = Quilt::new(
            vec![
                Probe::new("SW", south_west, 10, 1),
                Probe::new("NE", north_east, 10, 1),
                Probe::new("COVERED", inset(south_west), 12, 1),
                Probe::new("GAP", inset(north_west), 12, 1),
            ],
            0.0,
            false,
        );
        assert_eq!(
            rendered(quilt.render_tile(tile).expect("renders")),
            "NE,SW,GAP"
        );
        let coverage: Vec<_> = quilt.items().iter().map(|i| i.asked.all().0).collect();
        assert_eq!(coverage, [1, 1, 0, 1]);
    }

    /// Items equal on zoom fit and tie-break key (e.g. two cells without an
    /// edition date) are ordered by source, so the order the items were
    /// given in never decides which one a tile shows first.
    #[test]
    fn full_ties_follow_the_source_not_the_input_order() {
        let (tile, around) = probe_tile();
        let first = |names: [&'static str; 2]| {
            let quilt = Quilt::new(
                names.map(|n| Probe::new(n, around, 10, 1)).into(),
                0.0,
                false,
            );
            let names = rendered(quilt.render_tile(tile).expect("renders"));
            names.split(',').next().map(str::to_owned)
        };
        assert_eq!(first(["A", "B"]).as_deref(), Some("A"));
        assert_eq!(first(["B", "A"]).as_deref(), Some("A"));
    }

    /// A tile inside its best fit's coverage is covered by it alone: the
    /// boolean ops' rounding leaves a remainder that must not send the
    /// quilt on to load and render further candidates — also where that
    /// remainder is largest, a fine tile under a large coverage (geo snaps
    /// to a grid set by the coverage's extent, not the tile's).
    #[test]
    fn a_tile_inside_one_coverage_asks_no_other_candidate() {
        for (z, lat, margin) in [(10, 56.5, 0.1), (18, 56.5, 3.0), (18, 70.5, 3.0)] {
            let (tile, around) = probe_tile_at(z, 10.5, lat, margin);
            let quilt = Quilt::new(
                vec![
                    Probe::new("FIT", around, z, 1),
                    Probe::new("FINER", around, z + 2, 1),
                ],
                0.0,
                false,
            );
            let case = format!("z{z} at {lat}°N, {margin}° margin");
            assert_eq!(
                rendered(quilt.render_tile(tile).expect("renders")),
                "FIT",
                "{case}"
            );
            assert_eq!(quilt.items()[1].asked.all(), (0, 0, 0, 0), "{case}");
        }
    }

    /// Every real contribution counts, however thin: a cell covering a
    /// strip along the tile's west edge contributes there, also when the
    /// strip is half a unit of a 4096-unit vector tile wide — its points
    /// still draw at icon size, and at finer zooms the strip widens.
    #[test]
    fn a_thin_strip_still_contributes() {
        let (tile, around) = probe_tile();
        let bounds = Bbox::from(xyz_to_bbox(
            tile.z(),
            tile.x(),
            tile.y(),
            tile.x(),
            tile.y(),
        ));
        let unit = (bounds.east - bounds.west) / 4096.0;
        let west_strip = |width_units: f64| Bbox {
            west: bounds.west - unit,
            south: bounds.south - unit,
            east: width_units.mul_add(unit, bounds.west),
            north: bounds.north + unit,
        };
        for width_units in [10.0, 0.5] {
            let quilt = Quilt::new(
                vec![
                    Probe::new("NARROW", west_strip(width_units), 10, 1),
                    Probe::new("WIDE", around, 12, 1),
                ],
                0.0,
                false,
            );
            let names = rendered(quilt.render_tile(tile).expect("renders"));
            assert_eq!(names, "NARROW,WIDE", "{width_units} units");
        }
    }

    /// A sparse quilt asks each item for its detail zoom once, however many
    /// tiles need it — even when they render at the same time.
    #[test]
    fn a_sparse_quilt_asks_each_detail_zoom_once() {
        // Natively coarser than the tile, so the detail zoom must be asked.
        let (tile, around) = probe_tile_at(12, 10.5, 56.5, 0.1);
        let mut item = Probe::new("COARSE", around, 10, 1);
        item.load_time = std::time::Duration::from_millis(50);
        let quilt = Quilt::new(vec![item], 0.0, true);
        let start = std::sync::Barrier::new(4);
        std::thread::scope(|threads| {
            for _ in 0..4 {
                threads.spawn(|| {
                    start.wait();
                    let outcome = quilt.render_tile(tile).expect("renders");
                    assert_eq!(outcome, TileOutcome::Omitted);
                });
            }
        });
        assert_eq!(quilt.items()[0].asked.all().1, 1);
    }

    /// A candidate natively at the tile's zoom or finer decides "not all
    /// coarser" from the catalog alone: nobody is asked for a detail zoom.
    #[test]
    fn a_natively_fine_candidate_spares_the_detail_zooms() {
        let (tile, around) = probe_tile();
        let quilt = Quilt::new(
            vec![
                Probe::new("FIT", around, 10, 1),
                Probe::new("COARSE", around, 7, 1),
            ],
            0.0,
            true,
        );
        assert_eq!(rendered(quilt.render_tile(tile).expect("renders")), "FIT");
        let detail_zooms: Vec<_> = quilt.items().iter().map(|i| i.asked.all().1).collect();
        assert_eq!(detail_zooms, [0, 0]);
    }

    /// A load failure fails the tile rather than leaving a hole in it, and
    /// a later request asks again.
    #[test]
    fn a_failed_load_fails_the_tile() {
        let (tile, around) = probe_tile_at(13, 10.5, 56.5, 0.1);
        let mut broken = Probe::new("BROKEN", around, 10, 1);
        broken.broken = true;
        let quilt = Quilt::new(vec![broken], 0.0, true);
        for attempt in 1..=2 {
            let err = quilt.render_tile(tile).expect_err("the tile fails");
            assert!(err.to_string().contains("BROKEN failed to load"), "{err}");
            assert_eq!(quilt.items()[0].asked.all().1, attempt, "not memoised");
        }
    }

    /// A sparse quilt stores the tile at an item's own detail zoom and omits
    /// only strictly finer ones — also through the shortcut that decides
    /// from bbox candidates alone.
    #[test]
    fn a_lone_item_keeps_its_detail_zoom_tile() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let cell = RncCell::parse(
            "OVERVIEW".to_owned(),
            build_rnc(bbox, [200, 30, 30, 255], 3_000_000.0, "01/01/2026"),
        )
        .expect("cell parses");
        assert_eq!(cell.detail_zoom(0.0).unwrap(), 7, "fixture sanity");
        let quilt = Quilt::new(vec![cell], 0.0, true);
        let outcome = |z| {
            let (col, row, _, _) = bbox_to_xyz(10.5, 56.5, 10.5, 56.5, z);
            quilt.render_tile(xyz(z, col, row)).expect("renders")
        };
        assert!(matches!(outcome(7), TileOutcome::Rendered(_)));
        assert_eq!(outcome(8), TileOutcome::Omitted);
    }

    /// Sparse omission keys on [`TileSource::detail_zoom`], not the native
    /// zoom: a cell whose source pixels are finer than its 1:3M scale keeps
    /// every tile up to its pixel zoom and omits only beyond it.
    #[test]
    fn sparse_quilt_keeps_tiles_up_to_a_pixel_detail_zoom_above_native() {
        // 8 px across 0.001° → 2.88 M px around the world: z13 tiles
        // (2.1 M px) still skip source pixels, z14 (4.19 M px) do not.
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 10.001,
            north: 56.001,
        };
        let cell = RncCell::parse(
            "DENSE".to_owned(),
            build_rnc(bbox, [200, 30, 30, 255], 3_000_000.0, "01/01/2026"),
        )
        .expect("cell parses");
        assert_eq!(zoom_from_scale(3_000_000, 0.0), 7, "fixture sanity");
        assert_eq!(cell.detail_zoom(0.0).unwrap(), 14, "fixture sanity");
        let quilt = Quilt::new(vec![cell], 0.0, true);
        let outcome = |z| {
            let (col, row, _, _) = bbox_to_xyz(10.0005, 56.0005, 10.0005, 56.0005, z);
            quilt.render_tile(xyz(z, col, row)).expect("renders")
        };
        // Coarse tiles may render blank (the cell is sub-pixel there); what
        // matters is that none is omitted below the pixel detail zoom.
        for z in 8..=13 {
            assert_ne!(outcome(z), TileOutcome::Omitted, "z{z} must be stored");
        }
        assert!(matches!(outcome(14), TileOutcome::Rendered(_)));
        assert_eq!(outcome(15), TileOutcome::Omitted);
    }

    /// `write_pmtiles` must hand tiles to the writer in ascending `TileId`
    /// order — that is what lets the archive be flagged `clustered`, which
    /// readers rely on — and the archive header must advertise the same zoom
    /// range the function returns.
    #[test]
    fn write_pmtiles_writes_clustered_archive_with_returned_zoom_range() {
        let bbox = Bbox {
            west: 8.0,
            south: 54.0,
            east: 12.0,
            north: 58.0,
        };
        let quilt = Quilt::new(
            vec![
                RncCell::parse(
                    "CELL".to_owned(),
                    build_rnc(bbox, [30, 200, 30, 255], 3_000_000.0, "01/01/2026"),
                )
                .expect("cell parses"),
            ],
            0.0,
            false,
        );
        let path = std::env::temp_dir().join(format!(
            "quilt-tiler-{}-write_pmtiles_clustered.pmtiles",
            std::process::id()
        ));

        let written = write_pmtiles(&quilt, &path, None);
        let archive = std::fs::read(&path);
        let _ = std::fs::remove_file(&path);
        let zooms = written.expect("write_pmtiles succeeds");
        let header = pmtiles::Header::try_from_bytes(bytes::Bytes::from(
            archive.expect("archive was written"),
        ))
        .expect("archive has a valid PMTiles v3 header");

        let (zoom_floor, zoom_ceil, _) = quilt
            .zoom_range_and_bounds(None)
            .expect("zoom range resolves");
        assert_eq!(zooms, (zoom_floor, zoom_ceil));
        assert_eq!((header.min_zoom, header.max_zoom), zooms);
        // Fixture sanity: a 2×2 block or larger at some zoom, so a write
        // order that ignores the Hilbert curve would be detectable.
        assert!(
            header.n_addressed_tiles().map_or(0, u64::from) > 4,
            "fixture must span several tiles per zoom"
        );
        assert!(
            header.clustered(),
            "tiles must be written in ascending TileId order"
        );
    }

    /// Per-process, per-test archive path under the system temp dir.
    fn temp_archive_path(tag: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("quilt-tiler-{}-{tag}.pmtiles", std::process::id()))
    }

    fn parse_cell(name: &str, bbox: Bbox, color: [u8; 4]) -> RncCell {
        RncCell::parse(
            name.to_owned(),
            build_rnc(bbox, color, 3_000_000.0, "01/01/2026"),
        )
        .expect("cell parses")
    }

    /// Re-open a written archive with the `PMTiles` reader: the header must
    /// advertise the source's tile type, the returned zoom range and the
    /// items' combined bounds, and every `(z, x, y)` in that range must hold
    /// exactly the bytes [`Quilt::render_tile`] renders there — present where
    /// covered, absent in the gap between two disjoint cells, nothing extra.
    #[tokio::test]
    async fn write_pmtiles_archive_round_trips_through_reader() {
        // Disjoint cells with different footprints: the gap leaves uncovered
        // tiles inside the overall bbox, and a tile range wider than tall
        // makes a swapped col/row derivation miss or invent tiles.
        let west = Bbox {
            west: 0.0,
            south: 50.0,
            east: 1.0,
            north: 54.0,
        };
        let east = Bbox {
            west: 10.0,
            south: 50.0,
            east: 11.0,
            north: 51.0,
        };
        let quilt = Quilt::new(
            vec![
                parse_cell("WEST", west, [200, 30, 30, 255]),
                parse_cell("EAST", east, [30, 30, 200, 255]),
            ],
            0.0,
            false,
        );
        let path = temp_archive_path("write_pmtiles_round_trip");

        let written = write_pmtiles(&quilt, &path, None);
        let reader = pmtiles::AsyncPmTilesReader::new_with_path(&path).await;
        let (zoom_floor, zoom_ceil) = written.expect("write_pmtiles succeeds");
        let reader = reader.expect("archive re-opens with the PMTiles reader");
        let header = reader.get_header();

        let (expected_floor, expected_ceil, overall) = quilt
            .zoom_range_and_bounds(None)
            .expect("zoom range resolves");
        assert_eq!((zoom_floor, zoom_ceil), (expected_floor, expected_ceil));
        assert!(zoom_floor < zoom_ceil, "fixture must span several zooms");
        assert_eq!((header.min_zoom, header.max_zoom), (zoom_floor, zoom_ceil));
        assert_eq!(header.tile_type, pmtiles::TileType::Png);
        assert_eq!(header.tile_type, RncCell::tile_type());
        let bounds = [
            header.min_longitude,
            header.min_latitude,
            header.max_longitude,
            header.max_latitude,
        ];
        let expected_bounds = [overall.west, overall.south, overall.east, overall.north];
        assert!(
            bounds
                .iter()
                .zip(expected_bounds)
                .all(|(got, want)| (got - want).abs() < 1e-6),
            "header bounds {bounds:?} must be (west, south, east, north) {expected_bounds:?}"
        );

        let mut present = 0u64;
        let mut absent = 0u64;
        let mut non_square = false;
        for z in zoom_floor..=zoom_ceil {
            let (col_lo, row_lo, col_hi, row_hi) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            non_square |= col_hi - col_lo != row_hi - row_lo;
            for row in row_lo..=row_hi {
                for col in col_lo..=col_hi {
                    let expected = match quilt.render_tile(xyz(z, col, row)).expect("render_tile") {
                        TileOutcome::Rendered(bytes) => Some(bytes),
                        TileOutcome::Uncovered | TileOutcome::Blank | TileOutcome::Omitted => None,
                    };
                    let coord = TileCoord::new(z, col, row).expect("valid tile coord");
                    let stored = reader
                        .get_tile_decompressed(coord)
                        .await
                        .expect("reading tile from archive");
                    assert_eq!(
                        stored.as_deref(),
                        expected.as_deref(),
                        "archive tile ({z}, {col}, {row}) must match render_tile"
                    );
                    if expected.is_some() {
                        present += 1;
                    } else {
                        absent += 1;
                    }
                }
            }
        }
        let _ = std::fs::remove_file(&path);

        assert!(absent > 0, "fixture must leave uncovered tiles in the bbox");
        assert!(
            non_square,
            "fixture needs a non-square tile range at some zoom"
        );
        assert_eq!(
            header.n_addressed_tiles().map_or(0, u64::from),
            present,
            "archive must hold exactly the rendered tiles, nothing outside the zoom/bbox range"
        );
    }

    /// A `max_zoom` cap below the data's zoom floor must fail before any
    /// output file is created.
    #[test]
    fn write_pmtiles_max_zoom_below_floor_errors_without_creating_output() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let quilt = Quilt::new(
            vec![parse_cell("CELL", bbox, [30, 200, 30, 255])],
            0.0,
            false,
        );
        let (zoom_floor, _, _) = quilt
            .zoom_range_and_bounds(None)
            .expect("zoom range resolves");
        assert!(zoom_floor > 0, "fixture needs a non-zero zoom floor");
        let path = temp_archive_path("write_pmtiles_max_zoom_below_floor");
        let _ = std::fs::remove_file(&path);

        let result = write_pmtiles(&quilt, &path, Some(zoom_floor - 1));
        let created = path.exists();
        let _ = std::fs::remove_file(&path);

        let err = result.expect_err("max_zoom below the zoom floor must fail");
        assert!(
            err.to_string().contains("below the data's minimum zoom"),
            "unexpected error: {err:#}"
        );
        assert!(
            !created,
            "no output file may be created on a zoom-range error"
        );
    }

    /// An output path that cannot be created must surface as an error naming
    /// that path, not a panic or a silently missing archive.
    #[test]
    fn write_pmtiles_unwritable_output_names_path_in_error() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let quilt = Quilt::new(
            vec![parse_cell("CELL", bbox, [30, 200, 30, 255])],
            0.0,
            false,
        );
        let path = std::env::temp_dir()
            .join(format!("quilt-tiler-{}-missing-dir", std::process::id()))
            .join("out.pmtiles");

        let err = write_pmtiles(&quilt, &path, None)
            .expect_err("writing into a missing directory must fail");

        assert_eq!(err.to_string(), format!("creating {}", path.display()));
    }

    /// One `.rnc` with a zero footer scale in a batch must be skipped by the
    /// loader rather than accepted as 1:1 — otherwise its zoom 22 becomes the
    /// whole archive's `zoom_ceil`.
    #[test]
    fn scale_zero_rnc_does_not_raise_zoom_ceiling() {
        let bbox = Bbox {
            west: 10.0,
            south: 56.0,
            east: 11.0,
            north: 57.0,
        };
        let dir = std::env::temp_dir().join(format!(
            "quilt-tiler-scale-zero-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        std::fs::create_dir_all(&dir).expect("creating temp dir");
        let good = dir.join("GOOD.rnc");
        let bad = dir.join("BAD.rnc");
        std::fs::write(
            &good,
            build_rnc(bbox, [0, 0, 255, 255], 3_000_000.0, "01/01/2026"),
        )
        .expect("writing GOOD.rnc");
        std::fs::write(&bad, build_rnc(bbox, [255, 0, 0, 255], 0.0, "01/01/2026"))
            .expect("writing BAD.rnc");

        let loaded = crate::loader::load_cells(&[&good, &bad], 0.0);
        std::fs::remove_dir_all(&dir).expect("removing temp dir");
        let Ok(crate::loader::Cells::Raster(cells)) = loaded else {
            panic!("expected raster cells");
        };

        let sources: Vec<&str> = cells.iter().map(RncCell::source).collect();
        assert_eq!(
            sources,
            [good.to_string_lossy()],
            "scale-0 cell must be skipped"
        );
        let (_, zoom_ceil, _) = Quilt::new(cells, 0.0, false)
            .zoom_range_and_bounds(None)
            .expect("non-empty items");
        assert_eq!(zoom_ceil, zoom_from_scale(3_000_000, 0.0));
    }
}
