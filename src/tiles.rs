//! Direct tile writing: OESU cells → MVT tiles → `PMTiles` archive.
//!
//! For every tile `(z, col, row)` in the output zoom range, all cells whose
//! bounding box intersects the tile are candidates.  Candidates are sorted so
//! that the cell whose native zoom is closest to `z` from above (coarsest
//! appropriate) goes first, then finer cells in ascending order, and finally
//! coarser-than-`z` cells as a last resort.  Cells are added greedily until
//! the tile bbox is covered; the rest are skipped.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use fast_mvt::{
    DEFAULT_EXTENT, MvtFeature, MvtGeometry, MvtLayer, MvtLineString, MvtMultiLineString, MvtPoint,
    MvtPolygon, MvtTile, MvtValue,
};
use indicatif::{ParallelProgressIterator, ProgressBar, ProgressStyle};
use martin_tile_utils::{bbox_to_xyz, wgs84_to_webmercator, xyz_to_bbox};
use pmtiles::{PmTilesWriter, TileCoord, TileType};
use rayon::prelude::*;
use tracing::info;

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::zoom::zoom_from_scale;
use s57::{attribute_acronym, object_acronym};

const EXTENT: f64 = 4096.0;
/// A single encoded tile ready for `BTreeMap` insertion: `(tile_id, zoom, col, row, mvt_bytes)`.
type EncodedTile = (u64, u8, u32, u32, Vec<u8>);

// ── Public entry point ───────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)]
/// Encode all parsed `cells` as MVT tiles and write a `PMTiles` v3 archive to
/// `output`. Returns `(min_zoom, max_zoom)`.
pub fn write_pmtiles(
    cells: &[s57::S57Cell],
    output: &Path,
    max_zoom: Option<u8>,
    zoom_offset: f64,
) -> Result<(u8, u8)> {
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    let zoom = |scale: u32| zoom_from_scale(scale, zoom_offset);

    let zoom_floor = cells
        .iter()
        .map(|c| zoom(c.native_scale))
        .min()
        .unwrap_or(0)
        .saturating_sub(2);
    let zoom_ceil_native = cells
        .iter()
        .map(|c| zoom(c.native_scale))
        .max()
        .unwrap_or(0);
    let zoom_ceil = match max_zoom {
        Some(cap) if cap < zoom_floor => {
            anyhow::bail!("--max-zoom {cap} is below the data's minimum zoom {zoom_floor}");
        }
        Some(cap) => cap.min(zoom_ceil_native),
        None => zoom_ceil_native,
    };
    let overall = {
        let b = cells.iter().fold(Bbox::bottom(), |acc, c| {
            let [w, s, e, n] = c.bounds;
            acc.join(&Bbox {
                west: w,
                south: s,
                east: e,
                north: n,
            })
        });
        if b.is_bottom() {
            Bbox {
                west: -180.0,
                south: -85.0,
                east: 180.0,
                north: 85.0,
            }
        } else {
            b
        }
    };

    // Pre-compute native zoom per cell to avoid re-calling zoom_from_scale in
    // the hot loop (called once per candidate per tile).
    let cell_zoom: Vec<u8> = cells.iter().map(|c| zoom(c.native_scale)).collect();

    // Total tile coordinates across all zoom levels (used as progress denominator).
    let total_tiles: u64 = (zoom_floor..=zoom_ceil)
        .map(|z| {
            let (c0, r0, c1, r1) =
                bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
            u64::from(c1 - c0 + 1) * u64::from(r1 - r0 + 1)
        })
        .sum();

    info!(
        cells = cells.len(),
        zoom_floor, zoom_ceil_native, zoom_ceil, total_tiles, "encoding tiles",
    );
    let pb = ProgressBar::new(total_tiles).with_style(bar_style());

    for z in zoom_floor..=zoom_ceil {
        let (col_lo, row_lo, col_hi, row_hi) =
            bbox_to_xyz(overall.west, overall.south, overall.east, overall.north, z);
        let width = col_hi - col_lo + 1; // u32
        let height = row_hi - row_lo + 1;
        let count = u64::from(width) * u64::from(height);
        let zi = i32::from(z);

        let tiles: Vec<EncodedTile> = (0u64..count)
            .into_par_iter()
            .progress_with(pb.clone())
            .map(|idx| -> Result<Option<EncodedTile>> {
                profiling::scope!("tile");
                #[allow(clippy::cast_possible_truncation)] // idx % width < width ≤ u32::MAX
                let col = col_lo + (idx % u64::from(width)) as u32;
                #[allow(clippy::cast_possible_truncation)] // idx / width < height ≤ u32::MAX
                let row = row_lo + (idx / u64::from(width)) as u32;
                let tile_wgs84 = Bbox::from(xyz_to_bbox(z, col, row, col, row));
                let tile_merc = tile_mercator_bbox(tile_wgs84);

                // Candidates: cells whose bounding box overlaps this tile.
                let mut candidates: Vec<usize> = (0..cells.len())
                    .filter(|&i| Bbox::from(cells[i].bounds).overlaps(&tile_wgs84))
                    .collect();

                if candidates.is_empty() {
                    return Ok(None);
                }

                // Sort candidates so the coarsest cell that is still
                // fine-enough for zoom z comes first (key = 0), then ascending
                // through finer cells, then coarser-than-z cells last.
                candidates.sort_unstable_by_key(|&i| {
                    let nz = i32::from(cell_zoom[i]);
                    (nz < zi, if nz >= zi { nz } else { -nz })
                });

                // Add cells greedily: include a cell only if its contribution
                // (bbox clipped to tile) adds area not yet in `covered`.
                // Stop early when the full tile is covered.
                let mut covered = Bbox::bottom();
                let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();

                for &i in &candidates {
                    let contrib = Bbox::from(cells[i].bounds).meet(&tile_wgs84);
                    if covered.subsumes(&contrib) {
                        continue;
                    }
                    collect_cell_features(
                        &cells[i],
                        tile_wgs84,
                        tile_merc,
                        z,
                        zoom_offset,
                        &mut layers,
                    );
                    covered = covered.join(&contrib);
                    if covered.subsumes(&tile_wgs84) {
                        break;
                    }
                }

                let bytes = encode_tile(layers)?;
                if bytes.is_empty() {
                    return Ok(None);
                }
                Ok(Some((tile_id(z, col, row), z, col, row, bytes)))
            })
            .collect::<Result<Vec<_>>>()?
            .into_iter()
            .flatten()
            .collect();

        for (id, tz, tc, tr, bytes) in tiles {
            merge_tile(&mut tile_bytes, id, tz, tc, tr, bytes)?;
        }
    }
    pb.finish_and_clear();
    info!(encoded = tile_bytes.len(), "tiles encoded");

    let metadata = build_metadata();
    let file = File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = PmTilesWriter::new(TileType::Mvt)
        .min_zoom(zoom_floor)
        .max_zoom(zoom_ceil)
        .bounds(overall.west, overall.south, overall.east, overall.north)
        .metadata(&metadata)
        .create(file)
        .context("creating PMTiles writer")?;

    let pb_write = ProgressBar::new(tile_bytes.len() as u64).with_style(bar_style());
    for (_, (coord, bytes)) in tile_bytes {
        writer.add_tile(coord, &bytes).context("writing tile")?;
        pb_write.inc(1);
    }
    pb_write.finish_and_clear();
    writer.finalize().context("finalizing PMTiles")?;

    info!(output = %output.display(), "PMTiles written");
    Ok((zoom_floor, zoom_ceil))
}

/// Progress bar style for tile encoding and `PMTiles` write.
#[allow(clippy::literal_string_with_formatting_args)]
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
    tile_wgs84: Bbox,
    tile_merc: Bbox,
    tile_zoom: u8,
    zoom_offset: f64,
    layers: &mut HashMap<&'static str, Vec<MvtFeature>>,
) {
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
        let s57::Geometry::Point { lon, lat } = &feat.geometry else {
            continue;
        };
        let (lon, lat) = (*lon, *lat);

        // Honour SCAMIN: same check as to_mvt_features (attribute code 133).
        if let Some(attr) = feat.attributes.iter().find(|a| a.code == 133)
            && let s57::AttrValue::Int(scamin) = attr.value
            && zoom_from_scale(scamin, zoom_offset) > tile_zoom
        {
            continue;
        }

        // Arc bounding box: centre ± 2×r_m (radials are 2× the arc radius).
        let valnmr = feat
            .attributes
            .iter()
            .find(|a| a.code == 178)
            .and_then(|a| {
                if let s57::AttrValue::Double(v) = a.value {
                    Some(v)
                } else {
                    None
                }
            })
            .unwrap_or(3.0);
        let r_m = valnmr.mul_add(50.0, 200.0_f64).min(600.0);
        let d_lat = r_m * 2.0 / 111_320.0;
        let d_lon = r_m * 2.0 / (111_320.0 * lat.to_radians().cos());

        let arc_bbox = Bbox {
            west: lon - d_lon,
            south: lat - d_lat,
            east: lon + d_lon,
            north: lat + d_lat,
        };
        if !arc_bbox.overlaps(&tile_wgs84) {
            continue;
        }

        light_sectors_to_mvt(lon, lat, &feat.attributes, tile_wgs84, tile_merc, layers);
    }
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

/// Project `(lon, lat)` WGS84 to tile pixel coordinates `(x, y)` in
/// `[0, EXTENT]` space.  Geometry is clipped to the tile bbox before reaching
/// this function, so all projected coordinates stay within the valid range.
#[allow(clippy::cast_possible_truncation)] // deliberate floor-truncation
fn to_px(lon: f64, lat: f64, merc: Bbox) -> fast_mvt::MvtCoord {
    let (x_m, y_m) = wgs84_to_webmercator(lon, lat);
    let px = ((x_m - merc.west) / (merc.east - merc.west) * EXTENT) as i32;
    let py = ((merc.north - y_m) / (merc.north - merc.south) * EXTENT) as i32; // y=0 at north
    (px, py).into()
}

// ── Feature intersection test ────────────────────────────────────────────────

fn feat_intersects(feat: &s57::Feature, tile: Bbox) -> bool {
    feat_bbox(feat).is_some_and(|b| b.overlaps(&tile))
}

fn feat_bbox(feat: &s57::Feature) -> Option<Bbox> {
    match &feat.geometry {
        s57::Geometry::None => None,
        s57::Geometry::Point { lon, lat } => Some(Bbox::point(*lon, *lat)),
        s57::Geometry::MultiPoint(pts) => Bbox::of(pts.iter().map(|p| (p[0], p[1]))),
        s57::Geometry::Line(strokes) => {
            Bbox::of(strokes.iter().flat_map(|s| s.iter()).map(|p| (p[0], p[1])))
        }
        s57::Geometry::Area(ag) => {
            Bbox::of(ag.rings.iter().flat_map(|r| r.iter()).map(|p| (p[0], p[1])))
        }
    }
}

// ── Geometry clipping ─────────────────────────────────────────────────────────

/// Clip a polyline stroke to the rectangle `bbox = [west, south, east, north]`.
///
/// Uses Liang-Barsky per-segment clipping.  A stroke that enters and exits the
/// bbox multiple times is split into separate sub-strokes; sub-strokes with
/// fewer than 2 vertices are discarded.
#[profiling::function]
fn clip_stroke(stroke: &[[f64; 2]], bbox: Bbox) -> Vec<Vec<[f64; 2]>> {
    let Bbox {
        west,
        south,
        east,
        north,
    } = bbox;
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
fn clip_ring(ring: &[[f64; 2]], bbox: Bbox) -> Vec<[f64; 2]> {
    let Bbox {
        west,
        south,
        east,
        north,
    } = bbox;
    let r = clip_ring_half_plane(
        ring,
        |p| p[0] >= west,
        |a, b| {
            let t = (west - a[0]) / (b[0] - a[0]);
            [west, t.mul_add(b[1] - a[1], a[1])]
        },
    );
    let r = clip_ring_half_plane(
        &r,
        |p| p[0] <= east,
        |a, b| {
            let t = (east - a[0]) / (b[0] - a[0]);
            [east, t.mul_add(b[1] - a[1], a[1])]
        },
    );
    let r = clip_ring_half_plane(
        &r,
        |p| p[1] >= south,
        |a, b| {
            let t = (south - a[1]) / (b[1] - a[1]);
            [t.mul_add(b[0] - a[0], a[0]), south]
        },
    );
    clip_ring_half_plane(
        &r,
        |p| p[1] <= north,
        |a, b| {
            let t = (north - a[1]) / (b[1] - a[1]);
            [t.mul_add(b[0] - a[0], a[0]), north]
        },
    )
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
fn to_mvt_features(
    feat: &s57::Feature,
    tile_wgs84: Bbox,
    merc: Bbox,
    tile_zoom: u8,
    zoom_offset: f64,
) -> Vec<MvtFeature> {
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
                *lon >= tile_wgs84.west
                    && *lon <= tile_wgs84.east
                    && *lat >= tile_wgs84.south
                    && *lat <= tile_wgs84.north
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
                let ls: MvtLineString = clipped[0]
                    .iter()
                    .map(|[lon, lat]| to_px(*lon, *lat, merc))
                    .collect();
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
            let exterior: MvtLineString = exterior_pts
                .iter()
                .map(|[lon, lat]| to_px(*lon, *lat, merc))
                .collect();
            // Clip holes; discard any that vanish entirely.
            let holes: Vec<MvtLineString> = ag.rings[1..]
                .iter()
                .filter_map(|r| {
                    let clipped = clip_ring(r, tile_wgs84);
                    if clipped.len() < 3 {
                        return None;
                    }
                    Some(
                        clipped
                            .iter()
                            .map(|[lon, lat]| to_px(*lon, *lat, merc))
                            .collect(),
                    )
                })
                .collect();
            let mut f = MvtFeature::new(MvtGeometry::Polygon(MvtPolygon::new(exterior, holes)));
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
        "3" => "#ee2222",  // Red
        "4" => "#22aa22",  // Green
        "5" => "#2255ee",  // Blue
        "6" => "#ccaa00",  // Yellow
        "9" => "#cc8800",  // Amber
        "11" => "#ee7700", // Orange
        "12" => "#cc22cc", // Magenta
        _ => "#f8fafc",    // White (code 1 or unknown)
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
    tile_wgs84: Bbox,
    tile_merc: Bbox,
    layers: &mut HashMap<&'static str, Vec<MvtFeature>>,
) {
    let mut catlit: Option<MvtValue> = None;
    let mut colour = "";
    let mut sectr1: Option<f64> = None;
    let mut sectr2: Option<f64> = None;
    let mut valnmr: f64 = 3.0;

    for attr in attrs {
        match attr.code {
            37 => {
                catlit = Some(match &attr.value {
                    s57::AttrValue::Int(i) => MvtValue::UInt(u64::from(*i)),
                    s57::AttrValue::Str(s) => MvtValue::String(s.clone()),
                    s57::AttrValue::Double(f) => MvtValue::Double(*f),
                });
            }
            75 => {
                if let s57::AttrValue::Str(s) = &attr.value {
                    colour = s.as_str();
                }
            }
            136 => {
                if let s57::AttrValue::Double(v) = attr.value {
                    sectr1 = Some(v);
                }
            }
            137 => {
                if let s57::AttrValue::Double(v) = attr.value {
                    sectr2 = Some(v);
                }
            }
            178 => {
                if let s57::AttrValue::Double(v) = attr.value {
                    valnmr = v;
                }
            }
            _ => {}
        }
    }

    let hex = light_colour_hex(colour);
    let r_m = valnmr.mul_add(50.0, 200.0_f64).min(600.0_f64);

    #[allow(clippy::float_cmp)] // exact equality: same bearing = no sector
    let has_sectors = matches!((sectr1, sectr2), (Some(s1), Some(s2)) if s1 != s2);
    let (from_brg, to_brg_raw) = if has_sectors {
        (sectr1.unwrap(), sectr2.unwrap())
    } else {
        (0.0, 360.0)
    };
    let to_brg = if to_brg_raw <= from_brg {
        to_brg_raw + 360.0
    } else {
        to_brg_raw
    };

    // Arc: one point every 3° for a smooth curve.
    let span = to_brg - from_brg;
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // span ≥ 0, ≤ 360
    let steps = ((span / 3.0).ceil() as usize).max(4);
    let arc: Vec<[f64; 2]> = (0..=steps)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)] // steps ≤ ~120; no precision concern
            let brg = from_brg + span * (i as f64 / steps as f64);
            bearing_offset(lon, lat, brg, r_m)
        })
        .collect();

    let mut push_line = |pts: Vec<[f64; 2]>, kind: &'static str| {
        for stroke in clip_stroke(&pts, tile_wgs84) {
            if stroke.len() < 2 {
                continue;
            }
            let ls: MvtLineString = stroke
                .iter()
                .map(|[x, y]| to_px(*x, *y, tile_merc))
                .collect();
            let mut f = MvtFeature::new(MvtGeometry::LineString(ls));
            f.properties
                .push(("kind".into(), MvtValue::String(kind.into())));
            f.properties
                .push(("color".into(), MvtValue::String(hex.into())));
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
            push_line(
                vec![[lon, lat], bearing_offset(lon, lat, brg, r_m * 2.0)],
                "radial",
            );
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
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
        let stroke = vec![[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]];
        assert_eq!(clip_stroke(&stroke, bbox), vec![stroke]);
    }

    #[test]
    fn stroke_fully_outside_is_empty() {
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
        let stroke = vec![[11.0, 0.0], [15.0, 0.0]];
        assert!(clip_stroke(&stroke, bbox).is_empty());
    }

    #[test]
    fn stroke_clips_to_east_edge() {
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
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
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
        // [2,5]→[8,5] inside; [8,5]→[12,5] exits east; [12,5]→[8,2] re-enters
        let stroke = vec![[2.0, 5.0], [8.0, 5.0], [12.0, 5.0], [8.0, 2.0]];
        let result = clip_stroke(&stroke, bbox);
        assert_eq!(result.len(), 2, "expected two sub-strokes, got {result:?}");
    }

    // ── clip_ring ──────────────────────────────────────────────────────────────

    #[allow(clippy::float_cmp)] // ring vertices pass through unmodified — exact equality is correct
    #[test]
    fn ring_fully_inside_is_unchanged() {
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
        let ring = vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]];
        assert_eq!(clip_ring(&ring, bbox), ring);
    }

    #[test]
    fn ring_fully_outside_is_empty() {
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
        let ring = vec![[11.0, 11.0], [19.0, 11.0], [19.0, 19.0], [11.0, 19.0]];
        assert!(clip_ring(&ring, bbox).is_empty());
    }

    #[test]
    fn ring_clipped_to_east_edge() {
        let bbox = Bbox::from([0.0_f64, 0.0, 10.0, 10.0]);
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
        let bbox = Bbox::from([2.0_f64, 2.0, 8.0, 8.0]);
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
