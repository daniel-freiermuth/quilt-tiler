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
use pmtiles::{PmTilesWriter, TileCoord};
use rayon::prelude::*;
use tracing::info;

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::tile_geom::TileGeom;
use crate::tile_source::TileSource;
use crate::zoom::{scale_from_zoom, zoom_from_scale};

/// A single encoded tile ready for insertion into the output archive.
type EncodedTile = (u64, TileCoord, Vec<u8>);

// ── Entry point ───────────────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
/// Quilt `items` using source type `S` into a `PMTiles` v3 archive at `output`.
///
/// `zoom_offset` shifts every item's native scale to a zoom level (positive =
/// finer, negative = coarser).  Returns `(min_zoom, max_zoom)`.
///
/// # Errors
/// Returns an error if `max_zoom` is below the data's minimum zoom, a tile
/// fails to encode, or writing/finalizing the `PMTiles` archive fails.
pub fn write_pmtiles<S: TileSource>(
    items: &[S],
    output: &Path,
    max_zoom: Option<u8>,
    zoom_offset: f64,
) -> Result<(u8, u8)> {
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    let (zoom_floor, zoom_ceil, overall) = zoom_range_and_bounds(items, max_zoom, zoom_offset)?;

    let total_tiles: u64 = (zoom_floor..=zoom_ceil)
        .map(|z| {
            let (c0, r0, c1, r1) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            u64::from(c1 - c0 + 1) * u64::from(r1 - r0 + 1)
        })
        .sum();

    info!(
        count = items.len(),
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

                let Some(bytes) = render_tile(items, z, col, row, zoom_offset)? else {
                    return Ok(None);
                };
                let coord = TileCoord::new(z, col, row).context("invalid tile coord")?;
                Ok(Some((tile_id(z, col, row), coord, bytes)))
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

/// Compute `(zoom_floor, zoom_ceil, overall_bbox)` for `items`: the tile zoom
/// range and combined `WGS84` extent that a [`write_pmtiles`] batch or a live
/// server's `/style.json` should advertise.
///
/// `zoom_floor` is the coarsest native zoom among `items`, minus 2 (so the
/// finest chart's detail still fills down a couple of levels below its own
/// native zoom). `zoom_ceil` is the finest native zoom, capped by `max_zoom`
/// when given.
///
/// # Errors
/// Returns an error if `max_zoom` is below the data's minimum zoom.
pub fn zoom_range_and_bounds<S: TileSource>(
    items: &[S],
    max_zoom: Option<u8>,
    zoom_offset: f64,
) -> Result<(u8, u8, Bbox)> {
    let zoom_floor = items
        .iter()
        .map(|item| zoom_from_scale(item.native_scale(), zoom_offset))
        .min()
        .unwrap_or(0)
        .saturating_sub(2);
    let zoom_ceil_native = items
        .iter()
        .map(|item| zoom_from_scale(item.native_scale(), zoom_offset))
        .max()
        .unwrap_or(0);
    let zoom_ceil = match max_zoom {
        Some(cap) if cap < zoom_floor => {
            anyhow::bail!("max_zoom {cap} is below the data's minimum zoom {zoom_floor}");
        }
        Some(cap) => cap.min(zoom_ceil_native),
        None => zoom_ceil_native,
    };

    let overall = items.iter().fold(Bbox::bottom(), |acc, item| {
        acc.join(&item.coverage().into())
    });

    Ok((zoom_floor, zoom_ceil, overall))
}

/// Render one `(z, col, row)` tile from `items`'s combined coverage.
///
/// Returns `None` if nothing overlaps it, or [`TileSource::encode`] decides
/// to omit it. Pure given `(z, col, row)` — no shared mutable state across
/// calls — so it's safe to call concurrently and on demand. Backs both
/// [`write_pmtiles`]'s batch loop and a live tile server.
///
/// # Errors
/// Returns an error if [`TileSource::encode`] fails for this tile's contents.
pub fn render_tile<S: TileSource>(
    items: &[S],
    z: u8,
    col: u32,
    row: u32,
    zoom_offset: f64,
) -> Result<Option<Vec<u8>>> {
    profiling::scope!("tile");
    let zi = i32::from(z);
    let tile_wgs84 = Bbox::from(xyz_to_bbox(z, col, row, col, row));
    let tile_merc = tile_mercator_bbox(tile_wgs84);
    let tile_scale = scale_from_zoom(z, zoom_offset);

    // Candidates: items whose coverage bbox overlaps this tile.
    let mut candidates: Vec<usize> = {
        profiling::scope!("Collecting candidates");
        (0..items.len())
            .filter(|&i| {
                let bbox: Bbox = items[i].coverage().into();
                bbox.overlaps(&tile_wgs84)
            })
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
            let nz = i32::from(zoom_from_scale(items[i].native_scale(), zoom_offset));
            (
                nz < zi,
                if nz >= zi { nz } else { -nz },
                Reverse(items[i].tiebreak()),
            )
        });
    }

    // Greedy coverage: include an item only if its contribution adds area
    // not yet covered.  `Coverage`'s real polygon algebra correctly handles
    // disjoint coverage areas (e.g. NE + SW ≠ full tile), unlike a
    // bounding-box hull.
    let mut uncovered = S::Coverage::from(tile_wgs84);
    let mut contents: Vec<S::Content> = Vec::new();

    {
        profiling::scope!("Collecting features");
        for &i in &candidates {
            let contrib = items[i].coverage().meet(&uncovered);
            if contrib.area() == 0.0 {
                continue;
            }
            let item_tile = TileGeom {
                geom: contrib.clone().into(),
                merc: tile_merc,
                scale: tile_scale,
            };
            contents.push(items[i].render(&item_tile));
            uncovered = uncovered.minus(&contrib);
            if uncovered.area() == 0.0 {
                break;
            }
        }
    }

    let bytes = S::encode(contents)?;
    Ok(if bytes.is_empty() { None } else { Some(bytes) })
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

// ── PMTiles tile ID (Hilbert curve) ───────────────────────────────────────────

/// Compute the `PMTiles` v3 `TileID` for `(z, x, y)`.
///
/// `TileID = (4^z − 1) / 3 + hilbert_xy_to_d(2^z, x, y)`
fn tile_id(z: u8, x: u32, y: u32) -> u64 {
    if z == 0 {
        return 0;
    }
    let base = (4u64.pow(u32::from(z)) - 1) / 3;
    base + hilbert_xy_to_d(1u64 << z, u64::from(x), u64::from(y))
}

#[allow(clippy::many_single_char_names)] // n, x, y, d, s are standard Hilbert variables
fn hilbert_xy_to_d(n: u64, mut x: u64, mut y: u64) -> u64 {
    let mut d = 0u64;
    let mut s = n / 2;
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = (n - 1) - x;
                y = (n - 1) - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
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
        let bytes = render_tile(&[older, newer], z, col, row, 0.0)
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
}
