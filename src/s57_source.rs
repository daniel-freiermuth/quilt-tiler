//! [`TileSource`] implementation for OESU/S-57 vector cells → MVT tiles.

use std::collections::HashMap;

use anyhow::{Context, Result};
use fast_mvt::{
    DEFAULT_EXTENT, MvtFeature, MvtGeometry, MvtLayer, MvtLineString, MvtMultiLineString, MvtPoint,
    MvtPolygon, MvtTile, MvtValue,
};
use geo::{MultiPolygon, Polygon};
use martin_tile_utils::wgs84_to_webmercator;
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::tile_geom::TileGeom;
use crate::tile_source::TileSource;

const EXTENT: f64 = 4096.0;

// ── TileSource impl ──────────────────────────────────────────────────────────

impl TileSource for s57::S57Cell {
    type Content = HashMap<&'static str, Vec<MvtFeature>>;
    type Coverage = MultiPolygon;

    fn coverage(&self) -> Self::Coverage {
        MultiPolygon::new(
            self.coverage
                .iter()
                .map(|ring| Polygon::new(ring.clone().into(), vec![]))
                .collect(),
        )
    }

    fn native_scale(&self) -> u32 {
        self.native_scale
    }

    fn source(&self) -> String {
        self.source.clone()
    }

    #[profiling::function]
    fn render(&self, tile: &TileGeom) -> Self::Content {
        let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();

        for feat in &self.features {
            if !feat_intersects(feat, tile.wgs84) {
                continue;
            }
            let Some(layer_name) = s57::object_acronym(feat.type_code) else {
                continue;
            };
            let feats = to_mvt_features(feat, tile);
            if !feats.is_empty() {
                layers.entry(layer_name).or_default().extend(feats);
            }
        }

        // Light sector arcs — separate pass: arcs extend beyond the light position,
        // so the arc bounding box is used for intersection rather than the point.
        for feat in &self.features {
            if s57::object_acronym(feat.type_code) != Some("LIGHTS") {
                continue;
            }
            let s57::Geometry::Point { lon, lat } = &feat.geometry else {
                continue;
            };
            let (lon, lat) = (*lon, *lat);

            // SCAMIN in scale space: tile.scale > scamin → tile is too coarse.
            if let Some(attr) = feat.attributes.iter().find(|a| a.code == 133)
                && let s57::AttrValue::Int(scamin) = attr.value
                && scamin < tile.scale
            {
                continue;
            }

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
            if !arc_bbox.overlaps(&tile.wgs84) {
                continue;
            }
            light_sectors_to_mvt(lon, lat, &feat.attributes, tile, &mut layers);
        }

        layers
    }

    fn encode(contents: Vec<Self::Content>) -> Result<Vec<u8>> {
        let mut merged: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();
        for content in contents {
            for (layer, feats) in content {
                merged.entry(layer).or_default().extend(feats);
            }
        }
        encode_tile(merged)
    }

    fn tile_type() -> TileType {
        TileType::Mvt
    }
}

// ── MVT encoding ─────────────────────────────────────────────────────────────

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

// ── Coordinate projection ────────────────────────────────────────────────────

/// Project `(lon, lat)` WGS84 to tile pixel coordinates in `[0, EXTENT]` space.
///
/// Geometry is clipped to the tile bbox before this is called, so all
/// projected coordinates stay within the valid range.
#[allow(clippy::cast_possible_truncation)] // deliberate floor-truncation to pixel
fn to_px(lon: f64, lat: f64, merc: Bbox) -> fast_mvt::MvtCoord {
    let (x_m, y_m) = wgs84_to_webmercator(lon, lat);
    let px = ((x_m - merc.west) / (merc.east - merc.west) * EXTENT) as i32;
    let py = ((merc.north - y_m) / (merc.north - merc.south) * EXTENT) as i32; // y=0 at north
    (px, py).into()
}

// ── Feature filtering ────────────────────────────────────────────────────────

fn feat_intersects(feat: &s57::Feature, tile: Bbox) -> bool {
    feat_bbox(feat).is_some_and(|b| b.overlaps(&tile))
}

fn feat_bbox(feat: &s57::Feature) -> Option<Bbox> {
    match &feat.geometry {
        s57::Geometry::None => None,
        s57::Geometry::Point { lon, lat } => Some(Bbox::point(*lon, *lat)),
        s57::Geometry::Soundings(pts) => Bbox::of(pts.iter().map(|p| (p[0], p[1]))),
        s57::Geometry::Line(strokes) => {
            Bbox::of(strokes.iter().flat_map(|s| s.iter()).map(|p| (p[0], p[1])))
        }
        s57::Geometry::Area(ag) => {
            Bbox::of(ag.rings.iter().flat_map(|r| r.iter()).map(|p| (p[0], p[1])))
        }
    }
}

// ── Geometry clipping ────────────────────────────────────────────────────────

/// Clip a polyline stroke to `bbox` using Liang-Barsky per-segment clipping.
///
/// A stroke that exits and re-enters the bbox is split into separate
/// sub-strokes; sub-strokes with fewer than 2 vertices are discarded.
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

/// Liang-Barsky segment clipping. Returns clipped `(q0, q1)` or `None` if
/// the segment is fully outside.
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
    for (p, q) in [
        (-dx, p0[0] - west),
        (dx, east - p0[0]),
        (-dy, p0[1] - south),
        (dy, north - p0[1]),
    ] {
        if p == 0.0 {
            if q < 0.0 {
                return None;
            }
        } else {
            let t = q / p;
            if p < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
            if t0 > t1 {
                return None;
            }
        }
    }
    Some((
        [t0.mul_add(dx, p0[0]), t0.mul_add(dy, p0[1])],
        [t1.mul_add(dx, p0[0]), t1.mul_add(dy, p0[1])],
    ))
}

/// Clip a polygon ring to `bbox` using Sutherland-Hodgman.
///
/// Returns the clipped ring; empty when the ring is entirely outside.
/// The ring need not be explicitly closed.
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

// ── Feature → MVT conversion ─────────────────────────────────────────────────

/// Convert one S-57 feature to zero or more MVT features in tile pixel space.
///
/// All geometry is clipped to `tile.wgs84`.  `MultiPoint` soundings are
/// additionally filtered to their exact containing tile.
#[profiling::function]
fn to_mvt_features(feat: &s57::Feature, tile: &TileGeom) -> Vec<MvtFeature> {
    // SCAMIN: skip features whose minimum display scale is coarser than this tile.
    // tile.scale > scamin → tile is too coarse to show this feature.
    const SCAMIN_CODE: u16 = 133;
    if let Some(attr) = feat.attributes.iter().find(|a| a.code == SCAMIN_CODE)
        && let s57::AttrValue::Int(scamin) = attr.value
        && scamin < tile.scale
    {
        return vec![];
    }

    let props = build_props(&feat.attributes);

    match &feat.geometry {
        s57::Geometry::None => vec![],

        s57::Geometry::Point { lon, lat } => {
            let c = to_px(*lon, *lat, tile.merc);
            let mut f = MvtFeature::new(MvtGeometry::Point(MvtPoint::new(c.x, c.y)));
            f.properties = props;
            vec![f]
        }

        s57::Geometry::Soundings(pts) => pts
            .iter()
            .filter(|[lon, lat, _]| {
                // Each sounding belongs to exactly one tile.
                *lon >= tile.wgs84.west
                    && *lon <= tile.wgs84.east
                    && *lat >= tile.wgs84.south
                    && *lat <= tile.wgs84.north
            })
            .map(|[lon, lat, depth]| {
                let c = to_px(*lon, *lat, tile.merc);
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
            let clipped: Vec<Vec<[f64; 2]>> = strokes
                .iter()
                .flat_map(|s| clip_stroke(s, tile.wgs84))
                .collect();
            if clipped.is_empty() {
                return vec![];
            }
            let geom = if clipped.len() == 1 {
                let ls: MvtLineString = clipped[0]
                    .iter()
                    .map(|[lon, lat]| to_px(*lon, *lat, tile.merc))
                    .collect();
                MvtGeometry::LineString(ls)
            } else {
                let lines: Vec<MvtLineString> = clipped
                    .iter()
                    .map(|s| {
                        s.iter()
                            .map(|[lon, lat]| to_px(*lon, *lat, tile.merc))
                            .collect()
                    })
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
            let exterior_pts = clip_ring(&ag.rings[0], tile.wgs84);
            if exterior_pts.len() < 3 {
                return vec![];
            }
            let exterior: MvtLineString = exterior_pts
                .iter()
                .map(|[lon, lat]| to_px(*lon, *lat, tile.merc))
                .collect();
            let holes: Vec<MvtLineString> = ag.rings[1..]
                .iter()
                .filter_map(|r| {
                    let clipped = clip_ring(r, tile.wgs84);
                    if clipped.len() < 3 {
                        return None;
                    }
                    Some(
                        clipped
                            .iter()
                            .map(|[lon, lat]| to_px(*lon, *lat, tile.merc))
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

fn build_props(attrs: &[s57::Attribute]) -> Vec<(String, MvtValue)> {
    attrs
        .iter()
        .filter_map(|attr| {
            let key = s57::attribute_acronym(attr.code)?;
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

/// Flat-Earth bearing + distance → destination point.  Valid for ≤ 1200 m.
fn bearing_offset(lon: f64, lat: f64, bearing_deg: f64, dist_m: f64) -> [f64; 2] {
    let d_lat = dist_m / 111_320.0;
    let d_lon = dist_m / (111_320.0 * lat.to_radians().cos());
    let math_rad = (90.0 - bearing_deg).to_radians();
    [lon + d_lon * math_rad.cos(), lat + d_lat * math_rad.sin()]
}

/// Generate arc and radial sector features for one `LIGHTS` point.
///
/// Appends to `layers["LIGHTS_SECTOR"]`.
/// Attribute codes: `CATLIT=37  COLOUR=75  SECTR1=136  SECTR2=137  VALNMR=178`
fn light_sectors_to_mvt(
    lon: f64,
    lat: f64,
    attrs: &[s57::Attribute],
    tile: &TileGeom,
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
    let has_sectors = matches!((&sectr1, &sectr2), (Some(s1), Some(s2)) if s1 != s2);
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
    let span = to_brg - from_brg;

    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)] // span ∈ [0, 360]
    let steps = ((span / 3.0).ceil() as usize).max(4);
    let arc: Vec<[f64; 2]> = (0..=steps)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)] // steps ≤ ~120
            let brg = from_brg + span * (i as f64 / steps as f64);
            bearing_offset(lon, lat, brg, r_m)
        })
        .collect();

    let mut push_line = |pts: Vec<[f64; 2]>, kind: &'static str| {
        for stroke in clip_stroke(&pts, tile.wgs84) {
            if stroke.len() < 2 {
                continue;
            }
            let ls: MvtLineString = stroke
                .iter()
                .map(|[x, y]| to_px(*x, *y, tile.merc))
                .collect();
            let mut f = MvtFeature::new(MvtGeometry::LineString(ls));
            f.properties
                .push(("kind".into(), MvtValue::String(kind.into())));
            f.properties
                .push(("color".into(), MvtValue::String(hex.into())));
            if let Some(cv) = &catlit {
                f.properties.push(("CATLIT".into(), cv.clone()));
            }
            layers.entry("LIGHTS_SECTOR").or_default().push(f);
        }
    };

    push_line(arc, "arc");
    if has_sectors {
        for brg in [sectr1.unwrap(), sectr2.unwrap()] {
            push_line(
                vec![[lon, lat], bearing_offset(lon, lat, brg, r_m * 2.0)],
                "radial",
            );
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── clip_stroke ────────────────────────────────────────────────────────

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
        let stroke = vec![[2.0, 5.0], [8.0, 5.0], [12.0, 5.0], [8.0, 2.0]];
        let result = clip_stroke(&stroke, bbox);
        assert_eq!(result.len(), 2, "expected two sub-strokes, got {result:?}");
    }

    // ── clip_ring ──────────────────────────────────────────────────────────

    #[allow(clippy::float_cmp)] // ring vertices pass through unmodified
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
