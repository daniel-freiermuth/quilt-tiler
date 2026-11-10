//! Generic tile quilting: build a `PMTiles` archive from any [`TileSource`].
//!
//! For every `(z, col, row)` in the data's zoom range, candidate source items
//! are selected by coverage overlap, sorted by zoom fit (ties broken by
//! [`TileSource::tiebreak`] — newer edition wins), then added greedily until
//! the tile is covered by their combined [`TileSource::Coverage`].

use std::cmp::Reverse;
use std::collections::BTreeMap;
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use martin_tile_utils::{bbox_to_xyz, wgs84_to_webmercator, xyz_to_bbox};
use pmtiles::{PmTilesWriter, TileCoord, TileId};
use rayon::prelude::*;
use tracing::info;

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::tile_geom::TileGeom;
use crate::tile_source::{TileAccumulator, TileSource};
use crate::zoom::{scale_from_zoom, zoom_from_scale};

/// A single encoded tile ready for insertion into the output archive.
type EncodedTile = (u64, TileCoord, Vec<u8>);

// ── Entry point ───────────────────────────────────────────────────────────────

/// Quilt `quilt`'s items into a `PMTiles` v3 archive at `output`. Returns
/// `(min_zoom, max_zoom)`.
///
/// # Errors
/// Returns an error if `max_zoom` is below the data's minimum zoom, a tile
/// fails to encode, or writing/finalizing the `PMTiles` archive fails.
pub fn write_pmtiles<S: TileSource>(
    quilt: &Quilt<S>,
    output: &Path,
    max_zoom: Option<u8>,
) -> Result<(u8, u8)> {
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    let (zoom_floor, zoom_ceil, overall) = quilt.zoom_range_and_bounds(max_zoom)?;

    let total_tiles: u64 = (zoom_floor..=zoom_ceil)
        .map(|z| {
            let (c0, r0, c1, r1) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            u64::from(c1 - c0 + 1) * u64::from(r1 - r0 + 1)
        })
        .sum();

    info!(
        count = quilt.items().len(),
        zoom_floor, zoom_ceil, total_tiles, "encoding tiles",
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

                let Some(bytes) = quilt.render_tile(z, col, row)? else {
                    return Ok(None);
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
    info!(encoded = tile_bytes.len(), "tiles encoded");

    let metadata = build_metadata();
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

/// Items to quilt, plus the per-item data every tile needs, computed once.
pub struct Quilt<S> {
    items: Vec<S>,
    /// Per-item data, indexed like `items`.
    meta: Vec<ItemMeta>,
    zoom_offset: f64,
}

/// What [`Quilt::render_tile`] needs from an item on every tile, computed
/// once: [`TileSource::coverage`] clones a whole polygon and the native zoom
/// takes a `log2`.
struct ItemMeta {
    bbox: Bbox,
    native_zoom: u8,
}

impl<S: TileSource> Quilt<S> {
    /// `zoom_offset` shifts every item's native scale to a zoom level
    /// (positive = finer, negative = coarser).
    #[must_use]
    pub fn new(items: Vec<S>, zoom_offset: f64) -> Self {
        let meta = items
            .iter()
            .map(|item| ItemMeta {
                bbox: item.coverage().into(),
                native_zoom: zoom_from_scale(item.native_scale(), zoom_offset),
            })
            .collect();
        Self {
            items,
            meta,
            zoom_offset,
        }
    }

    #[must_use]
    pub fn items(&self) -> &[S] {
        &self.items
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
    /// Returns `None` if nothing overlaps it, or [`TileAccumulator::encode`]
    /// decides to omit it. Pure given `(z, col, row)` — no shared mutable
    /// state across calls — so it's safe to call concurrently and on demand.
    /// Backs both [`write_pmtiles`]'s batch loop and a live tile server.
    ///
    /// # Errors
    /// Returns an error if merging or encoding this tile's contents fails
    /// ([`TileAccumulator::push`] / [`TileAccumulator::encode`]).
    pub fn render_tile(&self, z: u8, col: u32, row: u32) -> Result<Option<Vec<u8>>> {
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
            return Ok(None);
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
            return Ok(None);
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
        Ok(if bytes.is_empty() { None } else { Some(bytes) })
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

fn build_metadata() -> String {
    serde_json::json!({
        "name": "chart",
        "description": "Nautical chart — converted from OESU",
        "vector_layers": []
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use image::Rgba;

    use super::*;
    use crate::rnc::RncCell;

    /// Build a minimal single-tile `.rnc` byte buffer: one `8×8` solid-color
    /// PNG, rectangular `cover` (the bbox itself), given `edate`.
    #[allow(clippy::cast_possible_truncation)] // test PNG size is a tiny constant
    fn build_rnc(bbox: Bbox, color: [u8; 4], scale: f64, edate: &str) -> Vec<u8> {
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
            "cover": [],
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
        let bytes = Quilt::new(vec![older, newer], 0.0)
            .render_tile(z, col, row)
            .expect("render_tile succeeds")
            .expect("tile has coverage");
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

    #[test]
    fn zoom_range_and_bounds_empty_items_must_bail() {
        let result = Quilt::<RncCell>::new(Vec::new(), 0.0).zoom_range_and_bounds(None);
        assert!(
            result.is_err(),
            "empty items must bail, but got Ok with ±Infinity bbox"
        );
    }

    /// One 1:3 000 000 cell (native zoom 7, zoom floor 5).
    fn overview_quilt() -> Quilt<RncCell> {
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
        Quilt::new(vec![cell], 0.0)
    }

    #[test]
    fn max_zoom_may_not_go_below_the_zoom_floor() {
        let quilt = overview_quilt();
        assert!(quilt.zoom_range_and_bounds(Some(4)).is_err());
        let (floor, ceil, _) = quilt.zoom_range_and_bounds(Some(5)).expect("at the floor");
        assert_eq!((floor, ceil), (5, 5));
        let (_, ceil, _) = quilt
            .zoom_range_and_bounds(Some(6))
            .expect("above the floor");
        assert_eq!(ceil, 6);
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
        let (_, zoom_ceil, _) = Quilt::new(cells, 0.0)
            .zoom_range_and_bounds(None)
            .expect("non-empty items");
        assert_eq!(zoom_ceil, zoom_from_scale(3_000_000, 0.0));
    }
}
