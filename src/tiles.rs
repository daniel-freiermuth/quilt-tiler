//! Direct tile writing: OESU cells → MVT tiles → `PMTiles` archive.
//! Three-pass pipeline:
//!   1. Build a `source_map` of which cells are responsible for each tile.
//!   2. Bottom-up sweep (fine → coarse): propagate fill annotations without
//!      re-examining individual cells — each tile checks only its 4 children.
//!   3. Parallel encode: for each annotated tile, collect features from all
//!      responsible cells into a shared layer map, then encode one MVT tile.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use fast_mvt::{
    MvtFeature, MvtGeometry, MvtLayer, MvtLineString, MvtMultiLineString, MvtPoint, MvtPolygon,
    MvtTile, MvtValue, DEFAULT_EXTENT,
};
use martin_tile_utils::{bbox_to_xyz, wgs84_to_webmercator, xyz_to_bbox};
use pmtiles::{PmTilesWriter, TileCoord, TileType};
use tracing::{debug, info};
use rayon::prelude::*;
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};

use s57::{attribute_acronym, object_acronym};
use crate::zoom::zoom_from_scale;

const EXTENT: f64 = 4096.0;
/// A single encoded tile ready for `BTreeMap` insertion: `(tile_id, zoom, col, row, mvt_bytes)`.
type EncodedTile = (u64, u8, u32, u32, Vec<u8>);
/// Key identifying a tile in the source map: `(zoom, col, row)`.
type TileKey = (u8, u32, u32);
/// Per-tile fill annotation: `(source_zoom, cell_indices, is_partial)`.
/// `is_partial` is `true` for pass-1 native tiles where the union of contributing
/// cell bboxes does not cover the full tile; `false` for all fill-propagated tiles.
type TileAnnotation = (u8, Vec<usize>, bool);

// ── Public entry point ───────────────────────────────────────────────────────

/// Encode all parsed `cells` as MVT tiles and write a `PMTiles` v3 archive to
/// `output`. Returns `(min_zoom, max_zoom)`.
///
/// **Three-pass pipeline:**
///
/// 1. Build `source_map[(zoom, col, row)] = (source_zoom, cell_indices)`.
///    Only native tiles are inserted here; `source_zoom == zoom` for all of them.
///    Multiple cells at the same native zoom covering the same tile are merged.
///
/// 2. Bottom-up sweep, `zoom_ceil-1` → `zoom_floor`.  For each unannotated tile,
///    inspect the 4 children at `zoom+1` (already annotated):
///    - All 4 present, all same `source_zoom` S → annotate with `(S, union_cells)`.
///      When `S == zoom+1` this is one-level fill-down.  When `S > zoom+1` the
///      fill-down cascades transitively.  When `S < zoom+1` fill-up propagates
///      upward through the tree.
///    - Otherwise → `find_native_ancestor`: walk toward `zoom_floor` and use the
///      first native tile that covers this location (fill-up fallback).
///
/// 3. Parallel encode.  For each annotated tile, collect features from all
///    responsible cells into a shared per-object-type layer map, then encode
///    a single MVT tile with one layer per S-57 object acronym.
// The three passes share source_map, native_count, and bounds; splitting them
// into sub-functions would just scatter those locals. Accept the length.
#[allow(clippy::too_many_lines)]
pub fn write_pmtiles(cells: &[s57::S57Cell], output: &Path, max_zoom: Option<u8>, zoom_offset: f64) -> Result<(u8, u8)> {
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    // Scale-to-zoom with user offset applied and result clamped to [0, 22].
    let zoom = |scale: u32| zoom_from_scale(scale, zoom_offset);

    let zoom_floor       = cells.iter().map(|c| zoom(c.native_scale)).min().unwrap_or(0).saturating_sub(2);
    let zoom_ceil_native = cells.iter().map(|c| zoom(c.native_scale)).max().unwrap_or(0);
    let zoom_ceil = match max_zoom {
        Some(cap) if cap < zoom_floor => {
            anyhow::bail!("--max-zoom {cap} is below the data's minimum zoom {zoom_floor}");
        }
        Some(cap) => cap.min(zoom_ceil_native),
        None => zoom_ceil_native,
    };
    let bounds = cells.iter().fold(
        [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY],
        |mut acc, c| {
            let [w, s, e, n] = c.bounds;
            acc[0] = acc[0].min(w);
            acc[1] = acc[1].min(s);
            acc[2] = acc[2].max(e);
            acc[3] = acc[3].max(n);
            acc
        },
    );
    let [bw, bs, be, bn] = if bounds[0].is_finite() { bounds } else { [-180.0, -85.0, 180.0, 85.0] };

    // ── Pass 1: native responsibility map ────────────────────────────────────
    info!(cells = cells.len(), zoom_floor, zoom_ceil_native, zoom_ceil, "pass 1: building native tile map");
    let pb1 = ProgressBar::new(cells.len() as u64).with_style(spinner_style());
    // Phase 1a: collect cell indices and per-cell coverage rectangles (clipped to
    // tile bounds) into a raw map.
    let mut raw_map: HashMap<TileKey, (Vec<usize>, Vec<[f64; 4]>)> = HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        let z = zoom(cell.native_scale);
        let [west, south, east, north] = cell.bounds;
        let (col_lo, row_lo, col_hi, row_hi) = bbox_to_xyz(west, south, east, north, z);
        for col in col_lo..=col_hi {
            for row in row_lo..=row_hi {
                let tile_bbox = xyz_to_bbox(z, col, row, col, row);
                let cov = [
                    west.max(tile_bbox[0]), south.max(tile_bbox[1]),
                    east.min(tile_bbox[2]), north.min(tile_bbox[3]),
                ];
                let e = raw_map.entry((z, col, row)).or_default();
                e.0.push(i);
                e.1.push(cov);
            }
        }
        pb1.inc(1);
    }
    pb1.finish_and_clear();
    // Phase 1b: compute partiality — a tile is partial when the union of its cells'
    // clipped bboxes does not cover the full tile.  Two adjacent cells that share
    // an exact boundary produce a union equal to the tile area and are not partial.
    let mut source_map: HashMap<TileKey, TileAnnotation> = HashMap::with_capacity(raw_map.len());
    for ((z, col, row), (cell_idxs, cov_rects)) in raw_map {
        let tile_bbox = xyz_to_bbox(z, col, row, col, row);
        let tile_area = (tile_bbox[2] - tile_bbox[0]) * (tile_bbox[3] - tile_bbox[1]);
        let covered   = rect_union_area(&cov_rects, tile_bbox);
        let is_partial = tile_area > 0.0 && covered < tile_area * (1.0 - 1e-6);
        source_map.insert((z, col, row), (z, cell_idxs, is_partial));
    }
    let native_count = source_map.len();
    info!(native_tiles = native_count, "pass 1 done");

    // ── Pass 2: bottom-up fill propagation ───────────────────────────────────
    // Sequential sweep; process one zoom level at a time so children at z+1
    // are always fully annotated before we inspect them from z.
    info!(zoom_floor, zoom_ceil_native, zoom_ceil, "pass 2: fill propagation");
    let pb2 = ProgressBar::new(u64::from(zoom_ceil_native.saturating_sub(zoom_floor)))
        .with_style(spinner_style())
        .with_message("pass 2");
    let mut new_entries: Vec<(TileKey, TileAnnotation)> = Vec::new();
    for z in (zoom_floor..=zoom_ceil_native).rev() {
        pb2.set_message(format!("pass 2  z={z}"));
        let (col_lo, row_lo, col_hi, row_hi) = bbox_to_xyz(bw, bs, be, bn, z);
        for col in col_lo..=col_hi {
            for row in row_lo..=row_hi {
                let key = (z, col, row);
                if source_map.contains_key(&key) {
                    continue;
                }
                let children: [Option<&TileAnnotation>; 4] = [
                    source_map.get(&(z + 1, 2 * col,     2 * row    )),
                    source_map.get(&(z + 1, 2 * col + 1, 2 * row    )),
                    source_map.get(&(z + 1, 2 * col,     2 * row + 1)),
                    source_map.get(&(z + 1, 2 * col + 1, 2 * row + 1)),
                ];
                let present: Vec<&TileAnnotation> = children.iter().filter_map(|x| *x).collect();
                let ann = if !present.is_empty() && present.iter().all(|a| a.0 > z) {
                    // Partial children: union whatever is available.
                    // This fills overview tiles (z below the coarsest native zoom)
                    // and handles sparse intra-range coverage gaps.
                    let source_z = present.iter().map(|a| a.0).min().unwrap();
                    let mut idxs: Vec<usize> = present.iter()
                        .flat_map(|a| a.1.iter().copied())
                        .collect();
                    idxs.sort_unstable();
                    idxs.dedup();
                    Some((source_z, idxs, false))
                } else {
                    // No children at all or some down-filled children
                    find_native_ancestor(z, col, row, &source_map, zoom_floor)
                };
                if let Some(a) = ann {
                    new_entries.push((key, a));
                }
            }
        }
        let added = new_entries.len();
        source_map.extend(std::mem::take(&mut new_entries));
        debug!(z, added, total = source_map.len(), "pass 2: zoom done");
        pb2.inc(1);
    }
    pb2.finish_and_clear();
    info!(
        annotated_tiles = source_map.len(),
        filled = source_map.len() - native_count,
        "pass 2 done",
    );

    // Drop tiles above the zoom cap before encoding; they have already served
    // as fill-down sources for every tile at ≤ zoom_ceil in pass 2.
    if zoom_ceil < zoom_ceil_native {
        source_map.retain(|&(z, _, _), _| z <= zoom_ceil);
        info!(retained = source_map.len(), zoom_ceil, "zoom cap applied");
    }

    // ── Pass 3: parallel encode ───────────────────────────────────────────────
    info!(annotated_tiles = source_map.len(), "pass 3: encoding tiles");
    let pb3 = ProgressBar::new(source_map.len() as u64).with_style(bar_style());
    let tiles: Vec<EncodedTile> = source_map
        .par_iter()
        .progress_with(pb3.clone())
        .map(|(&(z, col, row), (_, idxs, partial))| -> Result<Option<EncodedTile>> {
            profiling::scope!("tile");
            let tile_wgs84 = xyz_to_bbox(z, col, row, col, row);
            let tile_merc  = tile_mercator_bbox(tile_wgs84);
            let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();
            for &i in idxs {
                collect_cell_features(&cells[i], tile_wgs84, tile_merc, z, zoom_offset, &mut layers)?;
            }
            // For partial native tiles: augment with adjacent data.
            //
            // Two sources, tried in order:
            //
            // 1. z+1 children with source_z > z  →  finer cells that clip into
            //    this tile but were not in the pass-1 annotation (BL7II5/AL7ID6 case).
            //
            // 2. If no finer data found: walk up the ancestor chain looking for the
            //    nearest coarser tile whose cells differ from the ones we already have.
            //    This covers tiles where the native cell only clips a small fringe and
            //    the rest of the area is owned by a coarser-scale chart (58.41N/11.26E case).
            //
            // Guarded by `is_partial` so fully-covered tiles pay neither cost.
            if *partial {
                let mut seen: Vec<usize> = idxs.clone();

                // — finer pass —
                let child_keys = [
                    (z + 1, 2 * col,     2 * row    ),
                    (z + 1, 2 * col + 1, 2 * row    ),
                    (z + 1, 2 * col,     2 * row + 1),
                    (z + 1, 2 * col + 1, 2 * row + 1),
                ];
                let mut added_finer = false;
                for ck in &child_keys {
                    if let Some((child_src_z, child_idxs, _)) = source_map.get(ck) {
                        if *child_src_z > z {
                            for &i in child_idxs {
                                if !seen.contains(&i) {
                                    seen.push(i);
                                    collect_cell_features(&cells[i], tile_wgs84, tile_merc, z, zoom_offset, &mut layers)?;
                                    added_finer = true;
                                }
                            }
                        }
                    }
                }

                // — coarser pass (only when finer pass found nothing) —
                if !added_finer {
                    let (mut az, mut ac, mut ar) = (z, col, row);
                    while az > zoom_floor {
                        az -= 1;
                        ac >>= 1;
                        ar >>= 1;
                        if let Some((_, anc_idxs, _)) = source_map.get(&(az, ac, ar)) {
                            let mut added_anc = false;
                            for &i in anc_idxs {
                                if !seen.contains(&i) {
                                    seen.push(i);
                                    collect_cell_features(&cells[i], tile_wgs84, tile_merc, z, zoom_offset, &mut layers)?;
                                    added_anc = true;
                                }
                            }
                            if added_anc { break; }
                            // ancestor's cells are a strict subset of seen — try coarser
                        }
                    }
                }
            }
            let bytes = encode_tile(layers)?;
            if bytes.is_empty() { return Ok(None); }
            Ok(Some((tile_id(z, col, row), z, col, row, bytes)))
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect();
    pb3.finish_and_clear();
    info!(tiles = tiles.len(), "pass 3 done");

    for (id, z, col, row, bytes) in tiles {
        merge_tile(&mut tile_bytes, id, z, col, row, bytes)?;
    }

    let metadata = build_metadata();
    let file =
        File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = PmTilesWriter::new(TileType::Mvt)
        .min_zoom(zoom_floor)
        .max_zoom(zoom_ceil)
        .bounds(bw, bs, be, bn)
        .metadata(&metadata)
        .create(file)
        .context("creating PMTiles writer")?;

    let pb4 = ProgressBar::new(tile_bytes.len() as u64).with_style(bar_style());
    for (_, (coord, bytes)) in tile_bytes {
        writer.add_tile(coord, &bytes).context("writing tile")?;
        pb4.inc(1);
    }
    pb4.finish_and_clear();
    writer.finalize().context("finalizing PMTiles")?;

    info!(output = %output.display(), "PMTiles written");
    Ok((zoom_floor, zoom_ceil))
}

// ── Fill helpers ─────────────────────────────────────────────────────────────

/// Walk toward `zoom_floor`, returning a clone of the first `source_map` entry
/// found at an ancestor tile of `(z, col, row)`.
///
/// During pass 2 any entry at a coarser zoom was placed there by pass 1 (native
/// claim), because the bottom-up sweep hasn't reached those levels yet when this
/// is called.  The first hit is therefore always a native claim.
fn find_native_ancestor(
    z: u8,
    col: u32,
    row: u32,
    source_map: &HashMap<(u8, u32, u32), (u8, Vec<usize>, bool)>,
    zoom_floor: u8,
) -> Option<(u8, Vec<usize>, bool)> {
    for z_prime in (zoom_floor..z).rev() {
        let shift = z - z_prime;
        if let Some(ann) = source_map.get(&(z_prime, col >> shift, row >> shift)) {
            // Tiles filled from an ancestor are never considered partial: the
            // ancestor was the best available data and no finer augmentation applies.
            return Some((ann.0, ann.1.clone(), false));
        }
    }
    None
}

/// Area of the union of `rects` clipped to `tile [W, S, E, N]`.
///
/// Uses coordinate-compressed sweep-line: O(N²) for N rectangles, which is
/// acceptable because N is typically 1–4 native cells per tile.
fn rect_union_area(rects: &[[f64; 4]], tile: [f64; 4]) -> f64 {
    let mut xs: Vec<f64> = Vec::with_capacity(rects.len() * 2 + 2);
    xs.push(tile[0]);
    xs.push(tile[2]);
    for r in rects {
        let x0 = r[0].max(tile[0]).min(tile[2]);
        let x1 = r[2].max(tile[0]).min(tile[2]);
        if x0 > tile[0] { xs.push(x0); }
        if x1 < tile[2] { xs.push(x1); }
    }
    xs.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
    xs.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    let mut area = 0.0_f64;
    for win in xs.windows(2) {
        let (x0, x1) = (win[0], win[1]);
        let xmid = (x0 + x1) * 0.5;
        // Collect y-intervals from rectangles whose x-range spans xmid.
        let mut segs: Vec<[f64; 2]> = rects
            .iter()
            .filter(|r| r[0] <= xmid && r[2] >= xmid)
            .map(|r| [r[1].max(tile[1]), r[3].min(tile[3])])
            .filter(|[y0, y1]| y1 > y0)
            .collect();
        segs.sort_unstable_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
        // Sweep to compute union length of y-intervals.
        let mut y_cover = 0.0_f64;
        let mut hi = tile[1];
        for [y0, y1] in segs {
            let lo = y0.max(hi);
            if lo < y1 { y_cover += y1 - lo; hi = hi.max(y1); }
        }
        area += (x1 - x0) * y_cover;
    }
    area
}

// ── Progress bar styles ──────────────────────────────────────────────────────

/// Counter spinner for fast passes (pass 1, pass 2): shows elapsed + pos/len.
#[allow(clippy::literal_string_with_formatting_args)] // indicatif template syntax, not format args
fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {spinner:.green} {msg:20}  {elapsed_precise}  {pos}/{len}",
    )
    .unwrap_or_else(|_| ProgressStyle::default_spinner())
}

/// Wide progress bar for slow passes (pass 3, `PMTiles` write): adds rate + ETA.
#[allow(clippy::literal_string_with_formatting_args)] // indicatif template syntax, not format args
fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "  {spinner:.green} {msg:20}  {elapsed_precise}  [{wide_bar:.cyan/blue}]  {human_pos}/{human_len}  ({per_sec}, {eta})",
    )
    .unwrap_or_else(|_| ProgressStyle::default_bar())
    .progress_chars("=>-")
}

/// Collect all features from `cell` that intersect `tile_wgs84` into `layers`.
/// Features are keyed by S-57 object acronym; entries are accumulated so that
/// callers can call this for multiple cells and get a single merged layer map.
#[profiling::function]
fn collect_cell_features(
    cell: &s57::S57Cell,
    tile_wgs84: [f64; 4],
    tile_merc: [f64; 4],
    tile_zoom: u8,
    zoom_offset: f64,
    layers: &mut HashMap<&'static str, Vec<MvtFeature>>,
) -> Result<()> {
    for feat in &cell.features {
        if !feat_intersects(feat, tile_wgs84) {
            continue;
        }
        let Some(layer_name) = object_acronym(feat.type_code) else {
            continue;
        };
        let feats = to_mvt_features(feat, tile_wgs84, tile_merc, tile_zoom, zoom_offset);
        if !feats.is_empty() {
            layers.entry(layer_name).or_default().extend(feats);
        }
    }

    // Light sector arcs — separate pass that uses the arc bounding box for intersection
    // rather than the point position, because arcs extend up to 1200 m beyond the light.
    for feat in &cell.features {
        if object_acronym(feat.type_code) != Some("LIGHTS") {
            continue;
        }
        let s57::Geometry::Point { lon, lat } = &feat.geometry else { continue };
        let (lon, lat) = (*lon, *lat);

        // Honour SCAMIN: same check as to_mvt_features (attribute code 133).
        if let Some(attr) = feat.attributes.iter().find(|a| a.code == 133)
            && let s57::AttrValue::Int(scamin) = attr.value
            && zoom_from_scale(scamin, zoom_offset) > tile_zoom
        {
            continue;
        }

        // Arc bounding box: centre ± 2×r_m (radials are 2× the arc radius).
        let valnmr = feat.attributes.iter()
            .find(|a| a.code == 178)
            .and_then(|a| if let s57::AttrValue::Double(v) = a.value { Some(v) } else { None })
            .unwrap_or(3.0);
        let r_m   = (200.0_f64 + valnmr * 50.0).min(600.0);
        let d_lat = r_m * 2.0 / 111_320.0;
        let d_lon = r_m * 2.0 / (111_320.0 * lat.to_radians().cos());

        if lon + d_lon < tile_wgs84[0] || lon - d_lon > tile_wgs84[2]
            || lat + d_lat < tile_wgs84[1] || lat - d_lat > tile_wgs84[3]
        {
            continue;
        }

        light_sectors_to_mvt(lon, lat, &feat.attributes, tile_wgs84, tile_merc, layers);
    }

    Ok(())
}

/// Insert `bytes` for tile `(zoom, col, row)` into `tile_bytes`.
fn merge_tile(
    tile_bytes: &mut BTreeMap<u64, (TileCoord, Vec<u8>)>,
    id: u64,
    zoom: u8,
    col: u32,
    row: u32,
    bytes: Vec<u8>,
) -> Result<()> {
    match tile_bytes.entry(id) {
        std::collections::btree_map::Entry::Occupied(mut e) => {
            e.get_mut().1.extend(bytes);
        }
        std::collections::btree_map::Entry::Vacant(e) => {
            let coord = TileCoord::new(zoom, col, row).context("invalid tile coord")?;
            e.insert((coord, bytes));
        }
    }
    Ok(())
}

// ── Coordinate transform ─────────────────────────────────────────────────────

/// Convert a WGS84 tile bbox `[west, south, east, north]` to Web Mercator
/// metres `[west_m, south_m, east_m, north_m]`.
fn tile_mercator_bbox(wgs84: [f64; 4]) -> [f64; 4] {
    let (w_m, s_m) = wgs84_to_webmercator(wgs84[0], wgs84[1]);
    let (e_m, n_m) = wgs84_to_webmercator(wgs84[2], wgs84[3]);
    [w_m, s_m, e_m, n_m]
}

/// Project `(lon, lat)` WGS84 to tile pixel coordinates `(x, y)` in
/// `[0, EXTENT]` space.  Geometry is clipped to the tile bbox before reaching
/// this function, so all projected coordinates stay within the valid range.
#[allow(clippy::cast_possible_truncation)] // deliberate floor-truncation
fn to_px(lon: f64, lat: f64, merc: [f64; 4]) -> fast_mvt::MvtCoord {
    let (x_m, y_m) = wgs84_to_webmercator(lon, lat);
    let px = ((x_m - merc[0]) / (merc[2] - merc[0]) * EXTENT) as i32;
    let py = ((merc[3] - y_m) / (merc[3] - merc[1]) * EXTENT) as i32; // y=0 at north
    (px, py).into()
}

// ── Feature intersection test ────────────────────────────────────────────────

fn feat_intersects(feat: &s57::Feature, tile: [f64; 4]) -> bool {
    let Some((fw, fs, fe, fn_)) = feat_bbox(feat) else {
        return false;
    };
    // Overlap when neither axis is disjoint.
    fw <= tile[2] && fe >= tile[0] && fs <= tile[3] && fn_ >= tile[1]
}

fn feat_bbox(feat: &s57::Feature) -> Option<(f64, f64, f64, f64)> {
    match &feat.geometry {
        s57::Geometry::None => None,
        s57::Geometry::Point { lon, lat } => Some((*lon, *lat, *lon, *lat)),
        s57::Geometry::MultiPoint(pts) => {
            bbox_of(pts.iter().map(|p| (p[0], p[1])))
        }
        s57::Geometry::Line(strokes) => {
            bbox_of(strokes.iter().flat_map(|s| s.iter()).map(|p| (p[0], p[1])))
        }
        s57::Geometry::Area(ag) => {
            bbox_of(ag.rings.iter().flat_map(|r| r.iter()).map(|p| (p[0], p[1])))
        }
    }
}

fn bbox_of(mut pts: impl Iterator<Item = (f64, f64)>) -> Option<(f64, f64, f64, f64)> {
    let first = pts.next()?;
    let (mut w, mut s, mut e, mut n) = (first.0, first.1, first.0, first.1);
    for (lon, lat) in pts {
        if lon < w {
            w = lon;
        }
        if lat < s {
            s = lat;
        }
        if lon > e {
            e = lon;
        }
        if lat > n {
            n = lat;
        }
    }
    Some((w, s, e, n))
}

// ── Geometry clipping ─────────────────────────────────────────────────────────

/// Clip a polyline stroke to the rectangle `bbox = [west, south, east, north]`.
///
/// Uses Liang-Barsky per-segment clipping.  A stroke that enters and exits the
/// bbox multiple times is split into separate sub-strokes; sub-strokes with
/// fewer than 2 vertices are discarded.
#[profiling::function]
fn clip_stroke(stroke: &[[f64; 2]], bbox: [f64; 4]) -> Vec<Vec<[f64; 2]>> {
    let [west, south, east, north] = bbox;
    let mut result: Vec<Vec<[f64; 2]>> = Vec::new();
    let mut current: Vec<[f64; 2]> = Vec::new();

    for seg in stroke.windows(2) {
        let p0 = seg[0];
        let p1 = seg[1];
        match clip_segment_lb(p0, p1, west, south, east, north) {
            None => {
                // Segment fully outside — flush current sub-stroke if valid.
                if current.len() >= 2 {
                    result.push(std::mem::take(&mut current));
                } else {
                    current.clear();
                }
            }
            Some((q0, q1)) => {
                if current.is_empty() {
                    current.push(q0);
                } else {
                    let last = *current.last().expect("non-empty");
                    // Clipped start differs from last accumulated point: stroke left
                    // the bbox and re-entered — start a new sub-stroke.
                    if (q0[0] - last[0]).abs() > f64::EPSILON
                        || (q0[1] - last[1]).abs() > f64::EPSILON
                    {
                        if current.len() >= 2 {
                            result.push(std::mem::take(&mut current));
                        } else {
                            current.clear();
                        }
                        current.push(q0);
                    }
                }
                current.push(q1);
            }
        }
    }
    if current.len() >= 2 {
        result.push(current);
    }
    result
}

/// Liang-Barsky segment clipping against an axis-aligned rectangle.
/// Returns the clipped endpoints `(q0, q1)`, or `None` when fully outside.
#[allow(clippy::many_single_char_names)]
fn clip_segment_lb(
    p0: [f64; 2],
    p1: [f64; 2],
    west: f64,
    south: f64,
    east: f64,
    north: f64,
) -> Option<([f64; 2], [f64; 2])> {
    let dx = p1[0] - p0[0];
    let dy = p1[1] - p0[1];
    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;
    // Each half-plane: p*t ≤ q — entering when p < 0, exiting when p > 0.
    for (p, q) in [
        (-dx, p0[0] - west),  // x ≥ west
        (dx, east - p0[0]),   // x ≤ east
        (-dy, p0[1] - south), // y ≥ south
        (dy, north - p0[1]),  // y ≤ north
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None; // parallel and outside
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t); // entering half-plane
            } else {
                t1 = t1.min(t); // exiting half-plane
            }
            if t0 > t1 {
                return None; // entry past exit — fully outside
            }
        }
    }
    Some((
        [t0.mul_add(dx, p0[0]), t0.mul_add(dy, p0[1])],
        [t1.mul_add(dx, p0[0]), t1.mul_add(dy, p0[1])],
    ))
}

/// Clip a polygon ring to the rectangle `bbox = [west, south, east, north]`
/// using Sutherland-Hodgman.  Returns the clipped ring; empty when the ring
/// is entirely outside.  The ring need not be explicitly closed.
#[profiling::function]
fn clip_ring(ring: &[[f64; 2]], bbox: [f64; 4]) -> Vec<[f64; 2]> {
    let [west, south, east, north] = bbox;
    let r = clip_ring_half_plane(ring, |p| p[0] >= west, |a, b| {
        let t = (west - a[0]) / (b[0] - a[0]);
        [west, t.mul_add(b[1] - a[1], a[1])]
    });
    let r = clip_ring_half_plane(&r, |p| p[0] <= east, |a, b| {
        let t = (east - a[0]) / (b[0] - a[0]);
        [east, t.mul_add(b[1] - a[1], a[1])]
    });
    let r = clip_ring_half_plane(&r, |p| p[1] >= south, |a, b| {
        let t = (south - a[1]) / (b[1] - a[1]);
        [t.mul_add(b[0] - a[0], a[0]), south]
    });
    clip_ring_half_plane(&r, |p| p[1] <= north, |a, b| {
        let t = (north - a[1]) / (b[1] - a[1]);
        [t.mul_add(b[0] - a[0], a[0]), north]
    })
}

/// Sutherland-Hodgman single half-plane clipping pass.
///
/// `inside(p)` — true when `p` is on the visible side of the clip edge.
/// `intersect(a, b)` — intersection of segment `a→b` with the clip edge;
/// called only when exactly one of `a`, `b` is inside.
fn clip_ring_half_plane(
    ring: &[[f64; 2]],
    inside: impl Fn([f64; 2]) -> bool,
    intersect: impl Fn([f64; 2], [f64; 2]) -> [f64; 2],
) -> Vec<[f64; 2]> {
    if ring.is_empty() {
        return Vec::new();
    }
    let n = ring.len();
    let mut out = Vec::with_capacity(n + 2);
    for i in 0..n {
        let s = ring[i];
        let e = ring[(i + 1) % n];
        match (inside(s), inside(e)) {
            (true, true) => out.push(e),
            (true, false) => out.push(intersect(s, e)),
            (false, true) => {
                out.push(intersect(s, e));
                out.push(e);
            }
            (false, false) => {}
        }
    }
    out
}

// ── Feature conversion ───────────────────────────────────────────────────────

/// Convert one OESU feature to zero or more MVT features in tile pixel space.
///
/// All geometry is clipped to `tile_wgs84`: line strokes are split at tile
/// boundaries, polygon rings are clipped via Sutherland-Hodgman.  `MultiPoint`
/// soundings are additionally filtered to their exact containing tile.
#[profiling::function]
fn to_mvt_features(feat: &s57::Feature, tile_wgs84: [f64; 4], merc: [f64; 4], tile_zoom: u8, zoom_offset: f64) -> Vec<MvtFeature> {
    // SCAMIN: skip features whose minimum display scale is finer than this tile's zoom.
    // Code 133 = SCAMIN in the S-57 attribute table.
    const SCAMIN_CODE: u16 = 133;
    if let Some(attr) = feat.attributes.iter().find(|a| a.code == SCAMIN_CODE)
        && let s57::AttrValue::Int(scamin) = attr.value
        && zoom_from_scale(scamin, zoom_offset) > tile_zoom
    {
        return vec![];
    }

    let props = build_props(&feat.attributes);

    match &feat.geometry {
        s57::Geometry::None => vec![],

        s57::Geometry::Point { lon, lat } => {
            let c = to_px(*lon, *lat, merc);
            let mut f = MvtFeature::new(MvtGeometry::Point(MvtPoint::new(c.x, c.y)));
            f.properties = props;
            vec![f]
        }

        s57::Geometry::MultiPoint(pts) => pts
            .iter()
            .filter(|[lon, lat, _]| {
                // Each sounding belongs to exactly one tile.
                *lon >= tile_wgs84[0] && *lon <= tile_wgs84[2]
                    && *lat >= tile_wgs84[1] && *lat <= tile_wgs84[3]
            })
            .map(|[lon, lat, depth]| {
                let c = to_px(*lon, *lat, merc);
                let mut f = MvtFeature::new(MvtGeometry::Point(MvtPoint::new(c.x, c.y)));
                f.properties.clone_from(&props);
                f.add_tag_double("VALDCO", *depth);
                f
            })
            .collect(),

        s57::Geometry::Line(strokes) => {
            if strokes.is_empty() {
                return vec![];
            }
            // Clip each stroke to the tile bbox; a stroke that exits and re-enters
            // is split into multiple sub-strokes.
            let clipped: Vec<Vec<[f64; 2]>> = strokes
                .iter()
                .flat_map(|s| clip_stroke(s, tile_wgs84))
                .collect();
            if clipped.is_empty() {
                return vec![];
            }
            let geom = if clipped.len() == 1 {
                let ls: MvtLineString =
                    clipped[0].iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect();
                MvtGeometry::LineString(ls)
            } else {
                let lines: Vec<MvtLineString> = clipped
                    .iter()
                    .map(|s| s.iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect())
                    .collect();
                MvtGeometry::MultiLineString(MvtMultiLineString::new(lines))
            };
            let mut f = MvtFeature::new(geom);
            f.properties = props;
            vec![f]
        }

        s57::Geometry::Area(ag) => {
            if ag.rings.is_empty() {
                return vec![];
            }
            // Clip the exterior ring; drop the feature if nothing survives.
            let exterior_pts = clip_ring(&ag.rings[0], tile_wgs84);
            if exterior_pts.len() < 3 {
                return vec![];
            }
            let exterior: MvtLineString =
                exterior_pts.iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect();
            // Clip holes; discard any that vanish entirely.
            let holes: Vec<MvtLineString> = ag.rings[1..]
                .iter()
                .filter_map(|r| {
                    let clipped = clip_ring(r, tile_wgs84);
                    if clipped.len() < 3 {
                        return None;
                    }
                    Some(clipped.iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect())
                })
                .collect();
            let mut f =
                MvtFeature::new(MvtGeometry::Polygon(MvtPolygon::new(exterior, holes)));
            f.properties = props;
            vec![f]
        }
    }
}

#[profiling::function]
fn build_props(attrs: &[s57::Attribute]) -> Vec<(String, MvtValue)> {
    attrs
        .iter()
        .filter_map(|attr| {
            let key = attribute_acronym(attr.code)?;
            let val = match &attr.value {
                s57::AttrValue::Int(i) => MvtValue::UInt(u64::from(*i)),
                s57::AttrValue::Double(f) => MvtValue::Double(*f),
                s57::AttrValue::Str(s) => MvtValue::String(s.clone()),
            };
            Some((key.to_string(), val))
        })
        .collect()
}
// ── Light sector geometry ─────────────────────────────────────────────────────

/// Map S-57 COLOUR first-value to a CSS hex string suitable for tile properties.
/// White lights use off-white so they remain legible against light backgrounds.
fn light_colour_hex(colour: &str) -> &'static str {
    match colour.split(',').next().unwrap_or("").trim() {
        "3"  => "#ee2222",  // Red
        "4"  => "#22aa22",  // Green
        "5"  => "#2255ee",  // Blue
        "6"  => "#ccaa00",  // Yellow
        "9"  => "#cc8800",  // Amber
        "11" => "#ee7700",  // Orange
        "12" => "#cc22cc",  // Magenta
        _    => "#f8fafc",  // White (code 1 or unknown)
    }
}

/// Compute the destination point at `bearing_deg` (degrees clockwise from N)
/// and `dist_m` metres from `(lon, lat)`, using a flat-Earth approximation
/// valid for the short distances used here (≤ 1200 m).
fn bearing_offset(lon: f64, lat: f64, bearing_deg: f64, dist_m: f64) -> [f64; 2] {
    let d_lat = dist_m / 111_320.0;
    let d_lon = dist_m / (111_320.0 * lat.to_radians().cos());
    let math_rad = (90.0 - bearing_deg).to_radians();
    [lon + d_lon * math_rad.cos(), lat + d_lat * math_rad.sin()]
}

/// Generate arc and radial sector features for one LIGHTS point and append them
/// to `layers["LIGHTS_SECTOR"]`.
///
/// Radius formula mirrors the original client-side pixel heuristic:
/// `clamp(200 + VALNMR × 50, 200, 600)` metres, which corresponds to
/// roughly 33–100 px at zoom 13 / latitude 58 °N.
///
/// Attribute codes:  CATLIT=37  COLOUR=75  SECTR1=136  SECTR2=137  VALNMR=178
fn light_sectors_to_mvt(
    lon: f64,
    lat: f64,
    attrs: &[s57::Attribute],
    tile_wgs84: [f64; 4],
    tile_merc: [f64; 4],
    layers: &mut HashMap<&'static str, Vec<MvtFeature>>,
) {
    let mut catlit: Option<MvtValue> = None;
    let mut colour = "";
    let mut sectr1: Option<f64> = None;
    let mut sectr2: Option<f64> = None;
    let mut valnmr: f64 = 3.0;

    for attr in attrs {
        match attr.code {
            37  => { catlit = Some(match &attr.value {
                         s57::AttrValue::Int(i)    => MvtValue::UInt(u64::from(*i)),
                         s57::AttrValue::Str(s)    => MvtValue::String(s.clone()),
                         s57::AttrValue::Double(f) => MvtValue::Double(*f),
                     }); }
            75  => { if let s57::AttrValue::Str(s) = &attr.value { colour = s.as_str(); } }
            136 => { if let s57::AttrValue::Double(v) = attr.value { sectr1 = Some(v); } }
            137 => { if let s57::AttrValue::Double(v) = attr.value { sectr2 = Some(v); } }
            178 => { if let s57::AttrValue::Double(v) = attr.value { valnmr = v; } }
            _   => {}
        }
    }

    let color   = light_colour_hex(colour);
    let r_m     = (200.0_f64 + valnmr * 50.0).min(600.0_f64);

    let has_sectors = matches!((sectr1, sectr2), (Some(s1), Some(s2)) if s1 != s2);
    let (from_brg, to_brg_raw) = if has_sectors {
        (sectr1.unwrap(), sectr2.unwrap())
    } else {
        (0.0, 360.0)
    };
    let to_brg = if to_brg_raw <= from_brg { to_brg_raw + 360.0 } else { to_brg_raw };

    // Arc: one point every 3° for a smooth curve.
    let span  = to_brg - from_brg;
    let steps = ((span / 3.0).ceil() as usize).max(4);
    let arc: Vec<[f64; 2]> = (0..=steps)
        .map(|i| {
            let brg = from_brg + span * (i as f64 / steps as f64);
            bearing_offset(lon, lat, brg, r_m)
        })
        .collect();

    let mut push_line = |pts: Vec<[f64; 2]>, kind: &'static str| {
        for stroke in clip_stroke(&pts, tile_wgs84) {
            if stroke.len() < 2 { continue; }
            let ls: MvtLineString = stroke.iter()
                .map(|[x, y]| to_px(*x, *y, tile_merc))
                .collect();
            let mut f = MvtFeature::new(MvtGeometry::LineString(ls));
            f.properties.push(("kind".into(),  MvtValue::String(kind.into())));
            f.properties.push(("color".into(), MvtValue::String(color.into())));
            if let Some(ref cv) = catlit {
                f.properties.push(("CATLIT".into(), cv.clone()));
            }
            layers.entry("LIGHTS_SECTOR").or_default().push(f);
        }
    };

    push_line(arc, "arc");

    // Radial boundary lines at 2× arc radius, only for sector lights.
    if has_sectors {
        for brg in [sectr1.unwrap(), sectr2.unwrap()] {
            push_line(vec![[lon, lat], bearing_offset(lon, lat, brg, r_m * 2.0)], "radial");
        }
    }
}


// ── MVT tile encoding ────────────────────────────────────────────────────────

#[profiling::function]
fn encode_tile(layers: HashMap<&'static str, Vec<MvtFeature>>) -> Result<Vec<u8>> {
    let mut tile = MvtTile::new();
    for (name, features) in layers {
        if features.is_empty() {
            continue;
        }
        let mut layer = MvtLayer::new(name, DEFAULT_EXTENT);
        for feat in features {
            layer.add_feature(feat);
        }
        tile.add_layer(layer);
    }
    if tile.layers.is_empty() {
        return Ok(vec![]);
    }
    tile.encode().context("encoding MVT tile")
}

// ── PMTiles metadata ─────────────────────────────────────────────────────────

fn build_metadata() -> String {
    // Minimal TileJSON-compatible metadata. `vector_layers` is intentionally
    // empty for now; field schemas can be derived from S-57 and added later.
    serde_json::json!({
        "name": "chart",
        "description": "Nautical chart — converted from OESU",
        "vector_layers": []
    })
    .to_string()
}

// ── PMTiles TileID (Hilbert curve) ───────────────────────────────────────────

/// Compute the `PMTiles` v3 `TileID` for tile `(z, x, y)`.
///
/// `TileID = (4^z − 1) / 3 + hilbert_xy_to_d(2^z, x, y)`
fn tile_id(z: u8, x: u32, y: u32) -> u64 {
    if z == 0 {
        return 0;
    }
    let base = (4u64.pow(u32::from(z)) - 1) / 3;
    base + hilbert_xy_to_d(1u64 << z, u64::from(x), u64::from(y))
}

#[allow(clippy::many_single_char_names)] // n, x, y, d, s are standard Hilbert curve variables
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
    use super::*;

    // ── clip_stroke ────────────────────────────────────────────────────────────

    #[test]
    fn stroke_fully_inside_is_unchanged() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let stroke = vec![[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]];
        assert_eq!(clip_stroke(&stroke, bbox), vec![stroke]);
    }

    #[test]
    fn stroke_fully_outside_is_empty() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let stroke = vec![[11.0, 0.0], [15.0, 0.0]];
        assert!(clip_stroke(&stroke, bbox).is_empty());
    }

    #[test]
    fn stroke_clips_to_east_edge() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let stroke = vec![[2.0, 5.0], [15.0, 5.0]];
        let result = clip_stroke(&stroke, bbox);
        assert_eq!(result.len(), 1);
        let [q0x, q0y] = result[0][0];
        let [q1x, q1y] = result[0][1];
        assert!((q0x - 2.0).abs() < 1e-10 && (q0y - 5.0).abs() < 1e-10);
        assert!((q1x - 10.0).abs() < 1e-10 && (q1y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn stroke_exits_and_re_enters_splits_into_two() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        // [2,5]→[8,5] inside; [8,5]→[12,5] exits east; [12,5]→[8,2] re-enters
        let stroke = vec![[2.0, 5.0], [8.0, 5.0], [12.0, 5.0], [8.0, 2.0]];
        let result = clip_stroke(&stroke, bbox);
        assert_eq!(result.len(), 2, "expected two sub-strokes, got {result:?}");
    }

    // ── clip_ring ──────────────────────────────────────────────────────────────

    #[allow(clippy::float_cmp)] // ring vertices pass through unmodified — exact equality is correct
    #[test]
    fn ring_fully_inside_is_unchanged() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let ring = vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]];
        assert_eq!(clip_ring(&ring, bbox), ring);
    }

    #[test]
    fn ring_fully_outside_is_empty() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let ring = vec![[11.0, 11.0], [19.0, 11.0], [19.0, 19.0], [11.0, 19.0]];
        assert!(clip_ring(&ring, bbox).is_empty());
    }

    #[test]
    fn ring_clipped_to_east_edge() {
        let bbox = [0.0_f64, 0.0, 10.0, 10.0];
        let ring = vec![[5.0, 1.0], [15.0, 1.0], [15.0, 9.0], [5.0, 9.0]];
        let result = clip_ring(&ring, bbox);
        assert!(!result.is_empty());
        assert!(
            result.iter().all(|[lon, _]| *lon <= 10.0 + 1e-10),
            "all x should be ≤ east=10, got {result:?}"
        );
    }

    #[test]
    fn ring_enclosing_bbox_clips_to_bbox_corners() {
        // A large polygon that completely contains the tile bbox should clip to
        // exactly the four corners of the bbox.
        let bbox = [2.0_f64, 2.0, 8.0, 8.0];
        let ring = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let result = clip_ring(&ring, bbox);
        assert_eq!(result.len(), 4, "should produce exactly 4 corners");
        assert!(
            result.iter().all(|[lon, lat]| {
                *lon >= 2.0 - 1e-10
                    && *lon <= 8.0 + 1e-10
                    && *lat >= 2.0 - 1e-10
                    && *lat <= 8.0 + 1e-10
            }),
            "corners should be within bbox, got {result:?}"
        );
    }
}
