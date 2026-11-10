//! Generic tile quilting: build a `PMTiles` archive from any [`TileSource`].
//!
//! For every `(z, col, row)` in the data's zoom range, candidate source items
//! are selected by coverage overlap, sorted by zoom fit (ties broken by
//! [`TileSource::tiebreak`] — newer edition wins), then added greedily until
//! the tile is covered by their combined [`TileSource::Coverage`].  A sparse
//! [`Quilt`] additionally leaves out tiles a client can rebuild from an
//! ancestor.

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use martin_tile_utils::{bbox_to_xyz, wgs84_to_webmercator, xyz_to_bbox};
use pmtiles::{PmTilesWriter, TileCoord, TileId};
use rayon::prelude::*;
use tracing::{info, warn};

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::tile_geom::TileGeom;
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
    quilt.warn_unreachable_ancestors(zoom_ceil);
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

                let bytes = match quilt.render_tile(z, col, row)? {
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
/// Such a tile equals its nearest stored ancestor clipped to it: let `D` be
/// the finest detail zoom among its contributors. Any item overlapping the
/// tile with a native zoom finer than `D` would sort ahead of them and
/// contribute, so none exists — the candidate order inside the tile is
/// therefore the same at every zoom from `D` to `z`, and each contributor
/// renders identically at or above its detail zoom. The ancestor at `D` is
/// stored (its contributor with detail zoom `D` keeps it), so the nearest
/// stored ancestor lies in `D..z`.
///
/// `MapLibre` searches at most [`CLIENT_ANCESTOR_SEARCH`] levels up, so
/// gaps deeper than that draw nothing there — see
/// [`Quilt::warn_unreachable_ancestors`].
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

/// What [`Quilt::render_tile`] needs from an item on every tile, computed
/// once: [`TileSource::coverage`] clones a whole polygon and the zooms take
/// a `log2` or a feature scan.
struct ItemMeta {
    bbox: Bbox,
    native_zoom: u8,
    detail_zoom: u8,
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
                bbox: item.coverage().into(),
                native_zoom: zoom_from_scale(item.native_scale(), zoom_offset),
                detail_zoom: item.detail_zoom(zoom_offset),
            })
            .collect();
        Self {
            items,
            meta,
            zoom_offset,
            sparse,
        }
    }

    #[must_use]
    pub fn items(&self) -> &[S] {
        &self.items
    }

    #[must_use]
    pub const fn is_sparse(&self) -> bool {
        self.sparse
    }

    /// Warn when a sparse quilt leaves tiles whose nearest stored ancestor
    /// lies beyond [`CLIENT_ANCESTOR_SEARCH`]: below `zoom_ceil`, an area
    /// only covered by an item with a detail zoom that much coarser renders
    /// empty in `MapLibre`.
    pub fn warn_unreachable_ancestors(&self, zoom_ceil: u8) {
        if !self.sparse {
            return;
        }
        let reach = zoom_ceil.saturating_sub(CLIENT_ANCESTOR_SEARCH);
        let affected = self.meta.iter().filter(|m| m.detail_zoom < reach).count();
        if let Some(coarsest) = self.meta.iter().map(|m| m.detail_zoom).min()
            && affected > 0
        {
            warn!(
                affected,
                coarsest_detail_zoom = coarsest,
                zoom_ceil,
                "items whose detail zoom is more than {CLIENT_ANCESTOR_SEARCH} levels below the \
                 max zoom: where only they cover, MapLibre finds no stored ancestor from zoom {} \
                 up and draws nothing",
                coarsest + CLIENT_ANCESTOR_SEARCH + 1
            );
        }
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

    /// Render one `(z, col, row)` tile from the items' combined coverage.
    ///
    /// Pure given `(z, col, row)` — no shared mutable state across calls —
    /// so it's safe to call concurrently and on demand. Backs both
    /// [`write_pmtiles`]'s batch loop and a live tile server.
    ///
    /// # Errors
    /// Returns an error if merging or encoding this tile's contents fails
    /// ([`TileAccumulator::push`] / [`TileAccumulator::encode`]).
    pub fn render_tile(&self, z: u8, col: u32, row: u32) -> Result<TileOutcome> {
        profiling::scope!("tile");
        let items = &self.items;
        let meta = &self.meta;
        let zi = i32::from(z);
        let tile_wgs84 = Bbox::from(xyz_to_bbox(z, col, row, col, row));

        // Candidates: items whose coverage bbox overlaps this tile.
        let mut candidates: Vec<usize> = {
            profiling::scope!("Collecting candidates");
            (0..items.len())
                .filter(|&i| meta[i].bbox.overlaps(&tile_wgs84))
                .collect()
        };

        if candidates.is_empty() {
            return Ok(TileOutcome::Uncovered);
        }
        // Contributors are a subset of candidates, so this decides the
        // common open-water case without any polygon algebra.
        if self.sparse && candidates.iter().all(|&i| meta[i].detail_zoom < z) {
            return Ok(TileOutcome::Omitted);
        }

        {
            profiling::scope!("Sorting candidates");
            // Sort: coarsest-appropriate zoom first, finer after, too-coarse
            // last; ties at the same floored zoom broken by recency (newer
            // edition wins — see `TileSource::tiebreak`).
            candidates.sort_unstable_by_key(|&i| {
                let nz = i32::from(meta[i].native_zoom);
                (
                    nz < zi,
                    if nz >= zi { nz } else { -nz },
                    Reverse(items[i].tiebreak()),
                )
            });
        }

        // Greedy coverage: include an item only if its contribution adds
        // area not yet covered.  `Coverage`'s real polygon algebra correctly
        // handles disjoint coverage areas (e.g. NE + SW ≠ full tile), unlike
        // a bounding-box hull.
        let mut uncovered = S::Coverage::from(tile_wgs84);
        let mut contributions: Vec<(usize, S::Coverage)> = Vec::new();
        {
            profiling::scope!("Assigning coverage");
            for &i in &candidates {
                let contrib = items[i].coverage().meet(&uncovered);
                if contrib.area() == 0.0 {
                    continue;
                }
                uncovered = uncovered.minus(&contrib);
                contributions.push((i, contrib));
                if uncovered.area() == 0.0 {
                    break;
                }
            }
        }

        if contributions.is_empty() {
            return Ok(TileOutcome::Uncovered);
        }
        if self.sparse && contributions.iter().all(|&(i, _)| meta[i].detail_zoom < z) {
            return Ok(TileOutcome::Omitted);
        }

        let tile_merc = tile_mercator_bbox(tile_wgs84);
        let tile_scale = scale_from_zoom(z, self.zoom_offset);
        let mut acc = S::Accumulator::empty();
        {
            profiling::scope!("Collecting features");
            for (i, contrib) in contributions {
                acc.push(items[i].render(&TileGeom {
                    geom: contrib.into(),
                    merc: tile_merc,
                    scale: tile_scale,
                }))?;
            }
        }

        let bytes = acc.encode()?;
        Ok(if bytes.is_empty() {
            TileOutcome::Blank
        } else {
            TileOutcome::Rendered(bytes)
        })
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

/// Convert a WGS84 bbox to Web Mercator metres.
fn tile_mercator_bbox(wgs84: Bbox) -> Bbox {
    let (w_m, s_m) = wgs84_to_webmercator(wgs84.west, wgs84.south);
    let (e_m, n_m) = wgs84_to_webmercator(wgs84.east, wgs84.north);
    Bbox {
        west: w_m,
        south: s_m,
        east: e_m,
        north: n_m,
    }
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

    use super::*;
    use crate::rnc::RncCell;
    use crate::rnc_source::RasterAccumulator;

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
        let mut buf = vec![0u8; 8];
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
        let older = RncCell::parse(
            "OLDER".to_owned(),
            build_rnc(bbox, [200, 30, 30, 255], 3_000_000.0, "01/01/2020"),
        )
        .expect("older cell parses");
        let newer = RncCell::parse(
            "NEWER".to_owned(),
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
        let TileOutcome::Rendered(bytes) = quilt.render_tile(z, col, row).expect("renders") else {
            panic!("tile has coverage");
        };
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .expect("re-decodes")
            .to_rgba8();

        // Sample the bbox's own centroid, projected into this tile's pixel
        // space — robust regardless of where the bbox sits within the tile.
        let tile_merc = tile_mercator_bbox(Bbox::from(xyz_to_bbox(z, col, row, col, row)));
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
            (coarse.detail_zoom(0.0), fine.detail_zoom(0.0)),
            (7, 13),
            "fixture sanity: native zooms decide both detail zooms"
        );

        let sparse = Quilt::new(vec![coarse, fine], 0.0, true);
        let outcome = |lon: f64, lat: f64, z: u8| {
            let (col, row, _, _) = bbox_to_xyz(lon, lat, lon, lat, z);
            match sparse.render_tile(z, col, row).expect("renders") {
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
                dense.render_tile(10, col, row).expect("renders"),
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
        let merc = tile_mercator_bbox(Bbox::from(xyz_to_bbox(z, col, row, col, row)));
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
                    let expected = match dense.render_tile(z, col, row).expect("dense renders") {
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

    /// Log lines `warn_unreachable_ancestors` emits for `zoom_ceil`.
    fn unreachable_warnings(quilt: &Quilt<RncCell>, zoom_ceil: u8) -> String {
        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().expect("capture lock").extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        tracing::subscriber::with_default(subscriber, || {
            quilt.warn_unreachable_ancestors(zoom_ceil);
        });
        let bytes = capture.0.lock().expect("capture lock").clone();
        String::from_utf8(bytes).expect("log is UTF-8")
    }

    #[test]
    fn warns_exactly_when_an_ancestor_is_out_of_client_reach() {
        // Detail zoom 7: from z17 the walk still reaches z7, from z18 not.
        let sparse = overview_quilt(true);
        assert_eq!(unreachable_warnings(&sparse, 17), "");
        let warning = unreachable_warnings(&sparse, 18);
        assert!(
            warning.contains("WARN") && warning.contains("affected=1"),
            "got: {warning}"
        );
        assert_eq!(
            unreachable_warnings(&overview_quilt(false), 18),
            "",
            "a dense quilt leaves no gaps"
        );
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
            quilt.render_tile(12, col, row).expect("renders"),
            TileOutcome::Omitted
        );
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
        assert_eq!(cell.detail_zoom(0.0), 7, "fixture sanity");
        let quilt = Quilt::new(vec![cell], 0.0, true);
        let outcome = |z| {
            let (col, row, _, _) = bbox_to_xyz(10.5, 56.5, 10.5, 56.5, z);
            quilt.render_tile(z, col, row).expect("renders")
        };
        assert!(matches!(outcome(7), TileOutcome::Rendered(_)));
        assert_eq!(outcome(8), TileOutcome::Omitted);
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

        let cells = crate::loader::load_rnc_cells(&[&good, &bad], 0.0);
        std::fs::remove_dir_all(&dir).expect("removing temp dir");

        let names: Vec<&str> = cells.iter().map(RncCell::name).collect();
        assert_eq!(names, ["GOOD"], "scale-0 cell must be skipped");
        let (_, zoom_ceil, _) = Quilt::new(cells, 0.0, false)
            .zoom_range_and_bounds(None)
            .expect("non-empty items");
        assert_eq!(zoom_ceil, zoom_from_scale(3_000_000, 0.0));
    }
}
