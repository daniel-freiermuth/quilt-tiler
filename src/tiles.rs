//! Direct tile writing: OESU cells → MVT tiles → `PMTiles` archive.
//! Three-pass pipeline:
//!   1. Build a `source_map` of which cells are responsible for each tile.
//!   2. Bottom-up sweep (fine → coarse): propagate fill annotations without
//!      re-examining individual cells — each tile checks only its 4 children.
//!   3. Parallel encode: for each annotated tile, clip+project each responsible
//!      cell's geometry and merge via protobuf byte-append.

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
/// Per-tile fill annotation: `(source_zoom, cell_indices)`.
type TileAnnotation = (u8, Vec<usize>);

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
/// 3. Parallel encode.  For each annotated tile, clip each responsible cell's
///    geometry to the tile's WGS84 bbox, project to MVT pixel space, and merge
///    the resulting byte blobs via protobuf repeated-field append.
// The three passes share source_map, native_count, and bounds; splitting them
// into sub-functions would just scatter those locals. Accept the length.
#[allow(clippy::too_many_lines)]
pub fn write_pmtiles(cells: &[s57::S57Cell], output: &Path, max_zoom: Option<u8>) -> Result<(u8, u8)> {
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    let zoom_floor       = cells.iter().map(|c| zoom_from_scale(c.native_scale)).min().unwrap_or(0);
    let zoom_ceil_native = cells.iter().map(|c| zoom_from_scale(c.native_scale)).max().unwrap_or(0);
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
    let mut source_map: HashMap<TileKey, TileAnnotation> = HashMap::new();
    for (i, cell) in cells.iter().enumerate() {
        let z = zoom_from_scale(cell.native_scale);
        let [west, south, east, north] = cell.bounds;
        let (col_lo, row_lo, col_hi, row_hi) = bbox_to_xyz(west, south, east, north, z);
        for col in col_lo..=col_hi {
            for row in row_lo..=row_hi {
                source_map.entry((z, col, row))
                    .or_insert_with(|| (z, Vec::new()))
                    .1.push(i);
            }
        }
        pb1.inc(1);
    }
    pb1.finish_and_clear();
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
    for z in (zoom_floor..zoom_ceil_native).rev() {
        pb2.set_message(format!("pass 2  z={z}"));
        let (col_lo, row_lo, col_hi, row_hi) = bbox_to_xyz(bw, bs, be, bn, z);
        for col in col_lo..=col_hi {
            for row in row_lo..=row_hi {
                let key = (z, col, row);
                if source_map.contains_key(&key) {
                    continue;
                }
                let c00 = source_map.get(&(z + 1, 2 * col,     2 * row    ));
                let c10 = source_map.get(&(z + 1, 2 * col + 1, 2 * row    ));
                let c01 = source_map.get(&(z + 1, 2 * col,     2 * row + 1));
                let c11 = source_map.get(&(z + 1, 2 * col + 1, 2 * row + 1));
                let ann = if let (Some(a0), Some(a1), Some(a2), Some(a3)) = (c00, c10, c01, c11) {
                    if a0.0 == a1.0 && a0.0 == a2.0 && a0.0 == a3.0 {
                        let mut idxs: Vec<usize> = a0.1.iter()
                            .chain(&a1.1).chain(&a2.1).chain(&a3.1)
                            .copied().collect();
                        idxs.sort_unstable();
                        idxs.dedup();
                        Some((a0.0, idxs))
                    } else {
                        find_native_ancestor(z, col, row, &source_map, zoom_floor)
                    }
                } else {
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
        .map(|(&(z, col, row), (_, idxs))| -> Result<Option<EncodedTile>> {
            profiling::scope!("tile");
            let tile_wgs84 = xyz_to_bbox(z, col, row, col, row);
            let tile_merc  = tile_mercator_bbox(tile_wgs84);
            let mut bytes  = Vec::<u8>::new();
            for &i in idxs {
                bytes.extend(encode_cell_features(&cells[i], tile_wgs84, tile_merc)?);
            }
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
    source_map: &HashMap<(u8, u32, u32), (u8, Vec<usize>)>,
    zoom_floor: u8,
) -> Option<(u8, Vec<usize>)> {
    for z_prime in (zoom_floor..z).rev() {
        let shift = z - z_prime;
        if let Some(ann) = source_map.get(&(z_prime, col >> shift, row >> shift)) {
            return Some(ann.clone());
        }
    }
    None
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

/// Encode all features from `cell` that intersect `tile_wgs84` into a raw MVT
/// byte blob.  Returns an empty `Vec` when no features land in the tile.
#[profiling::function]
fn encode_cell_features(
    cell: &s57::S57Cell,
    tile_wgs84: [f64; 4],
    tile_merc: [f64; 4],
) -> Result<Vec<u8>> {
    let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();
    for feat in &cell.features {
        if !feat_intersects(feat, tile_wgs84) {
            continue;
        }
        let Some(layer_name) = object_acronym(feat.type_code) else {
            continue;
        };
        let feats = to_mvt_features(feat, tile_wgs84, tile_merc);
        if !feats.is_empty() {
            layers.entry(layer_name).or_default().extend(feats);
        }
    }
    if layers.is_empty() {
        return Ok(Vec::new());
    }
    encode_tile(layers)
}

/// Insert or append `bytes` for tile `(zoom, col, row)` into `tile_bytes`.
/// Appending is valid because `Tile { repeated Layer layers }` is a protobuf
/// repeated field; concatenating two encoded `Tile` messages unions their layers.
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
fn to_mvt_features(feat: &s57::Feature, tile_wgs84: [f64; 4], merc: [f64; 4]) -> Vec<MvtFeature> {
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
