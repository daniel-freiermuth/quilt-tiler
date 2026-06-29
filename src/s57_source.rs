//! [`TileSource`] implementation for OESU/S-57 vector cells → MVT tiles.

use std::collections::HashMap;

use anyhow::{Context, Result};
use fast_mvt::{
    DEFAULT_EXTENT, MvtFeature, MvtGeometry, MvtLayer, MvtLineString, MvtTile, MvtValue,
};
use geo::{
    BooleanOps, Coord, HasDimensions, Intersects, LineString, MapCoords, MultiLineString,
    MultiPolygon, Point, Polygon, coord,
};

use martin_tile_utils::wgs84_to_webmercator;
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::tile_geom::TileGeom;
use crate::tile_source::TileSource;

/// Pixel-space scale `to_px` projects into — must match the MVT layer's
/// declared extent ([`DEFAULT_EXTENT`]), or geometry and the tile's own
/// coordinate-space header disagree.  Derived from it, not duplicated.
#[allow(clippy::cast_precision_loss)] // exact: any u32 fits a f64 mantissa
const EXTENT: f64 = DEFAULT_EXTENT.get() as f64;

// ── TileSource impl ──────────────────────────────────────────────────────────

impl TileSource for s57::S57Cell {
    type Content = HashMap<&'static str, Vec<MvtFeature>>;
    type Coverage = MultiPolygon;
    type Tiebreaker = s57::EditionDate;

    #[profiling::function]
    fn coverage(&self) -> Self::Coverage {
        self.coverage.clone()
    }

    fn native_scale(&self) -> u32 {
        self.native_scale
    }

    fn tiebreak(&self) -> Self::Tiebreaker {
        self.edition_date
    }

    fn source(&self) -> String {
        self.source.clone()
    }

    #[profiling::function]
    fn render(&self, tile: &TileGeom) -> Self::Content {
        let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();

        for feat in &self.features {
            let Some(layer_name) = s57::object_acronym(feat.type_code) else {
                continue;
            };
            {
                profiling::scope!("Test feature intersection");
                if !feat_intersects(feat, &tile.geom) {
                    continue;
                }
            }
            let feats = to_mvt_features(feat, tile);
            if !feats.is_empty() {
                layers.entry(layer_name).or_default().extend(feats);
            }
        }

        let lateral_cardinal_buoy_positions: std::collections::HashSet<(i64, i64)> = self
            .features
            .iter()
            .filter_map(|f| {
                let acronym = s57::object_acronym(f.type_code)?;
                if !is_lateral_or_cardinal_buoy(acronym) {
                    return None;
                }
                let s57::Geometry::Point(p) = &f.geometry else {
                    return None;
                };
                Some(quantize_point(*p))
            })
            .collect();

        {
            profiling::scope!("Collecting lighthouses");
            // Light sector arcs — separate pass: arcs extend beyond the light position,
            // so the arc bounding box is used for intersection rather than the point.
            for feat in &self.features {
                if s57::object_acronym(feat.type_code) != Some("LIGHTS") {
                    continue;
                }
                let s57::Geometry::Point(center) = &feat.geometry else {
                    continue;
                };

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
                let d_lon = r_m * 2.0 / (111_320.0 * center.y().to_radians().cos());
                let arc_bbox = Bbox {
                    west: center.x() - d_lon,
                    south: center.y() - d_lat,
                    east: center.x() + d_lon,
                    north: center.y() + d_lat,
                };
                if !tile.geom.intersects(&Polygon::from(arc_bbox)) {
                    continue;
                }
                let on_lateral_cardinal_buoy =
                    lateral_cardinal_buoy_positions.contains(&quantize_point(*center));
                light_sectors_to_mvt(
                    *center,
                    &feat.attributes,
                    tile,
                    on_lateral_cardinal_buoy,
                    &mut layers,
                );
            }
        }

        layers
    }

    #[profiling::function]
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
fn to_px(wgs84_coord: Coord, merc: Bbox) -> fast_mvt::MvtCoord {
    let (x_m, y_m) = wgs84_to_webmercator(wgs84_coord.x, wgs84_coord.y);
    let px = ((x_m - merc.west) / (merc.east - merc.west) * EXTENT) as i32;
    let py = ((merc.north - y_m) / (merc.north - merc.south) * EXTENT) as i32; // y=0 at north
    (px, py).into()
}

// ── Feature filtering ────────────────────────────────────────────────────────

fn feat_intersects(feat: &s57::Feature, tile_geom: &MultiPolygon) -> bool {
    match &feat.geometry {
        s57::Geometry::None => false,
        s57::Geometry::Point(p) => tile_geom.intersects(p),
        s57::Geometry::Soundings(pts) => pts.iter().any(|(p, _)| tile_geom.intersects(p)),
        s57::Geometry::Line(ls) => tile_geom.intersects(ls),
        s57::Geometry::Area(poly) => tile_geom.intersects(poly),
    }
}

// ── Geometry clipping ────────────────────────────────────────────────────────

/// Clip a polyline stroke to `clip` — an arbitrary [`MultiPolygon`] region,
/// not necessarily a single rectangle.
///
/// A stroke that exits and re-enters `clip` is split into separate
/// sub-strokes; sub-strokes with fewer than 2 vertices are discarded.
#[profiling::function]
fn clip_stroke(line: &LineString, clip: &MultiPolygon) -> MultiLineString {
    clip.clip(&MultiLineString::new(vec![line.clone()]), false)
}

/// Clip a polygon ring to `clip` — an arbitrary [`MultiPolygon`] region, not
/// necessarily a single rectangle.
///
/// Returns the clipped polygon(s); empty when entirely outside `clip`.  The
/// ring need not be explicitly closed.
#[profiling::function]
fn clip_ring(subject: &Polygon, clip: &MultiPolygon) -> MultiPolygon {
    subject.intersection(clip)
}

// ── Feature → MVT conversion ─────────────────────────────────────────────────

/// Convert one S-57 feature to zero or more MVT features in tile pixel space.
///
/// All geometry is clipped to `tile.geom`.  `Soundings` points are
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

        s57::Geometry::Point(p) => {
            let c = to_px((*p).into(), tile.merc);
            let mut f = MvtFeature::new(MvtGeometry::Point(c.into()));
            f.properties = props;
            vec![f]
        }

        s57::Geometry::Soundings(pts) => pts
            .iter()
            .filter(|(wgs_coord, _)| {
                // Each sounding belongs to exactly one tile.
                tile.geom.intersects(wgs_coord)
            })
            .map(|(wgs_coord, depth)| {
                let c = to_px((*wgs_coord).into(), tile.merc);
                let mut f = MvtFeature::new(MvtGeometry::Point(c.into()));
                f.properties.clone_from(&props);
                f.add_tag_double("VALDCO", *depth);
                f
            })
            .collect(),

        s57::Geometry::Line(stroke) => {
            if stroke.is_empty() {
                return vec![];
            }
            let clipped: MultiLineString = clip_stroke(stroke, &tile.geom);
            if clipped.is_empty() {
                return vec![];
            }
            let mvt_linestring = clipped.map_coords(|coord| to_px(coord, tile.merc));
            let geom = MvtGeometry::MultiLineString(mvt_linestring);
            let mut f = MvtFeature::new(geom);
            f.properties = props;
            vec![f]
        }

        s57::Geometry::Area(ag) => {
            if ag.is_empty() {
                return vec![];
            }
            let clipped_wgs84 = clip_ring(ag, &tile.geom);
            if clipped_wgs84.is_empty() {
                return vec![];
            }
            let clipped_px = clipped_wgs84.map_coords(|coord| to_px(coord, tile.merc));

            let mut f = MvtFeature::new(MvtGeometry::MultiPolygon(clipped_px));
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
        "9" => "#cc8800",  // Amber
        "11" => "#ee7700", // Orange
        "12" => "#cc22cc", // Magenta
        // Yellow (code 6) and white (code 1 or unknown) — white rendered as
        // yellow too: a near-white ring is invisible against a pale chart
        // background.
        _ => "#ccaa00",
    }
}

/// Quantizes a point to ~1cm precision for exact-coincidence lookups (e.g.
/// matching a `LIGHTS` point against the buoy/beacon it is mounted on).
fn quantize_point(p: Point) -> (i64, i64) {
    #[allow(clippy::cast_possible_truncation)] // bounded by ±180/90 deg * 1e7
    (
        (p.x() * 1.0e7).round() as i64,
        (p.y() * 1.0e7).round() as i64,
    )
}

/// Flat-Earth bearing + distance → destination point.  Valid for ≤ 1200 m.
fn bearing_offset(coord: Coord, bearing_deg: f64, dist_m: f64) -> Coord {
    let d_lat = dist_m / 111_320.0;
    let d_lon = dist_m / (111_320.0 * coord.y.to_radians().cos());
    let math_rad = (90.0 - bearing_deg).to_radians();
    coord![x: coord.x + d_lon * math_rad.cos(), y: coord.y + d_lat * math_rad.sin()]
}

/// `true` for lateral and cardinal buoys: their lights are plain all-round
/// lights with no real sector data, so the synthetic "no sector" full circle
/// (see `light_sectors_to_mvt`) is just clutter and is suppressed for them.
fn is_lateral_or_cardinal_buoy(acronym: &str) -> bool {
    matches!(acronym, "BOYLAT" | "BOYCAR")
}

/// Emits a small flare-icon marker for a buoy-mounted all-round light that
/// has no real sector data, and therefore no range-circle drawn for it
/// (the synthetic "no sector" circle is suppressed as clutter — see
/// `light_sectors_to_mvt`).  CATLIT 6/8 (flood / subsidiary light) aren't
/// standalone aids to navigation, so those are skipped too.
fn emit_buoy_light_flare(
    center: Point,
    colour: &str,
    catlit: Option<&MvtValue>,
    tile: &TileGeom,
    layers: &mut HashMap<&'static str, Vec<MvtFeature>>,
) {
    let is_flood_or_subsidiary = matches!(catlit, Some(MvtValue::UInt(6 | 8)));
    if is_flood_or_subsidiary || !tile.geom.intersects(&center) {
        return;
    }
    let mut f = MvtFeature::new(MvtGeometry::Point(to_px(center.into(), tile.merc).into()));
    f.properties
        .push(("COLOUR".into(), MvtValue::String(colour.into())));
    if let Some(cv) = catlit {
        f.properties.push(("CATLIT".into(), cv.clone()));
    }
    layers.entry("LIGHTS_FLARE").or_default().push(f);
}

/// Generate arc and radial sector features for one `LIGHTS` point.
///
/// Appends to `layers["LIGHTS_SECTOR"]`.
/// Attribute codes: `CATLIT=37  COLOUR=75  SECTR1=136  SECTR2=137  VALNMR=178`
fn light_sectors_to_mvt(
    center: Point,
    attrs: &[s57::Attribute],
    tile: &TileGeom,
    on_lateral_cardinal_buoy: bool,
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

    // SECTR1/SECTR2 are encoded as bearings observed *from seaward towards
    // the light* (IHO S-57 light-sector convention) — i.e. the bearing a
    // vessel on the sector boundary would read pointing at the light. The
    // boundary ray drawn outward *from* the light therefore needs the
    // reciprocal bearing: flip by 180°.
    let sectr1 = sectr1.map(|b| (b + 180.0) % 360.0);
    let sectr2 = sectr2.map(|b| (b + 180.0) % 360.0);

    let hex = light_colour_hex(colour);
    let r_m = valnmr.mul_add(50.0, 200.0_f64).min(600.0_f64);

    #[allow(clippy::float_cmp)] // exact equality: same bearing = no sector
    let has_sectors = matches!((&sectr1, &sectr2), (Some(s1), Some(s2)) if s1 != s2);
    if !has_sectors && on_lateral_cardinal_buoy {
        // Plain all-round buoy light: skip the synthetic "no sector" full
        // circle (clutter — it conveys no real sector information here),
        // and draw a small tilted flare icon instead so the buoy's light
        // still shows up on the chart.
        emit_buoy_light_flare(center, colour, catlit.as_ref(), tile, layers);
        return;
    }
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
    let arc = LineString::new(
        (0..=steps)
            .map(|i| {
                #[allow(clippy::cast_precision_loss)] // steps ≤ ~120
                let brg = from_brg + span * (i as f64 / steps as f64);
                bearing_offset(center.into(), brg, r_m)
            })
            .collect(),
    );

    let mut push_line = |pts: LineString, kind: &'static str| {
        for stroke in clip_stroke(&pts, &tile.geom) {
            let ls: MvtLineString = stroke.map_coords(|c| to_px(c, tile.merc));
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
                LineString(vec![
                    center.into(),
                    bearing_offset(center.into(), brg, r_m * 2.0),
                ]),
                "radial",
            );
        }
    }
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    /// Exterior ring coords with the closing duplicate (boolean-ops output
    /// always closes rings) stripped, so vertex counts match the input.
    fn distinct_corners(poly: &Polygon) -> Vec<Coord> {
        let mut coords: Vec<Coord> = poly.exterior().0.clone();
        if coords.first() == coords.last() {
            coords.pop();
        }
        coords
    }

    /// A single-rectangle clip region.
    fn rect(west: f64, south: f64, east: f64, north: f64) -> MultiPolygon {
        MultiPolygon::new(vec![Polygon::from(Bbox {
            west,
            south,
            east,
            north,
        })])
    }

    // ── clip_stroke ────────────────────────────────────────────────────────

    #[test]
    fn stroke_fully_inside_is_unchanged() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let stroke = LineString::from(vec![[2.0, 2.0], [5.0, 5.0], [8.0, 8.0]]);
        assert_eq!(
            clip_stroke(&stroke, &clip),
            MultiLineString::new(vec![stroke.clone()])
        );
    }

    #[test]
    fn stroke_fully_outside_is_empty() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let stroke = LineString::from(vec![[11.0, 0.0], [15.0, 0.0]]);
        assert!(clip_stroke(&stroke, &clip).is_empty());
    }

    #[test]
    fn stroke_clips_to_east_edge() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let stroke = LineString::from(vec![[2.0, 5.0], [15.0, 5.0]]);
        let result = clip_stroke(&stroke, &clip);
        assert_eq!(result.0.len(), 1);
        let q0 = result.0[0].0[0];
        let q1 = result.0[0].0[1];
        assert!((q0.x - 2.0).abs() < 1e-10 && (q0.y - 5.0).abs() < 1e-10);
        assert!((q1.x - 10.0).abs() < 1e-10 && (q1.y - 5.0).abs() < 1e-10);
    }

    #[test]
    fn stroke_exits_and_re_enters_splits_into_two() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let stroke = LineString::from(vec![[2.0, 5.0], [8.0, 5.0], [12.0, 5.0], [8.0, 2.0]]);
        let result = clip_stroke(&stroke, &clip);
        assert_eq!(
            result.0.len(),
            2,
            "expected two sub-strokes, got {result:?}"
        );
    }

    #[test]
    fn stroke_clips_to_two_disjoint_rects() {
        // Non-rectangular clip region: two separate rects, not their hull.
        // A stroke crossing the gap between them must split into two pieces
        // and never touch the uncovered middle strip [4, 6].
        let clip = MultiPolygon::new(vec![
            Polygon::from(Bbox {
                west: 0.0,
                south: 0.0,
                east: 4.0,
                north: 10.0,
            }),
            Polygon::from(Bbox {
                west: 6.0,
                south: 0.0,
                east: 10.0,
                north: 10.0,
            }),
        ]);
        let stroke = LineString::from(vec![[1.0, 5.0], [9.0, 5.0]]);
        let result = clip_stroke(&stroke, &clip);
        assert_eq!(
            result.0.len(),
            2,
            "expected two sub-strokes, one per rect, got {result:?}"
        );
        for ls in &result.0 {
            for c in ls.coords() {
                assert!(
                    c.x <= 4.0 + 1e-10 || c.x >= 6.0 - 1e-10,
                    "coordinate {c:?} falls in the uncovered gap"
                );
            }
        }
    }

    // ── clip_ring ──────────────────────────────────────────────────────────

    #[test]
    fn ring_fully_inside_is_unchanged() {
        // Same vertex set as the input, no clipping needed — winding/start
        // point may differ from the boolean-op engine's normalisation.
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let points = vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]];
        let ring = Polygon::new(LineString::from(points.clone()), vec![]);
        let result = clip_ring(&ring, &clip);
        assert_eq!(result.0.len(), 1);
        let corners = distinct_corners(&result.0[0]);
        assert_eq!(corners.len(), points.len());
        for p in &points {
            assert!(
                corners
                    .iter()
                    .any(|q| (q.x - p[0]).abs() < 1e-10 && (q.y - p[1]).abs() < 1e-10),
                "missing vertex {p:?} in {corners:?}"
            );
        }
    }

    #[test]
    fn ring_fully_outside_is_empty() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let ring = Polygon::new(
            LineString::from(vec![[11.0, 11.0], [19.0, 11.0], [19.0, 19.0], [11.0, 19.0]]),
            vec![],
        );
        assert!(clip_ring(&ring, &clip).is_empty());
    }

    #[test]
    fn ring_clipped_to_east_edge() {
        let clip = rect(0.0, 0.0, 10.0, 10.0);
        let ring = Polygon::new(
            LineString::from(vec![[5.0, 1.0], [15.0, 1.0], [15.0, 9.0], [5.0, 9.0]]),
            vec![],
        );
        let result = clip_ring(&ring, &clip);
        assert!(!result.is_empty());
        assert!(
            result
                .0
                .iter()
                .all(|p| p.exterior().0.iter().all(|c| c.x <= 10.0 + 1e-10)),
            "all x should be ≤ east=10, got {result:?}"
        );
    }

    #[test]
    fn ring_enclosing_bbox_clips_to_bbox_corners() {
        let clip = rect(2.0, 2.0, 8.0, 8.0);
        let ring = Polygon::new(
            LineString::from(vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]),
            vec![],
        );
        let result = clip_ring(&ring, &clip);
        assert_eq!(result.0.len(), 1, "should produce a single clipped polygon");
        let corners = distinct_corners(&result.0[0]);
        assert_eq!(corners.len(), 4, "should produce exactly 4 corners");
        assert!(
            corners.iter().all(|c| {
                c.x >= 2.0 - 1e-10 && c.x <= 8.0 + 1e-10 && c.y >= 2.0 - 1e-10 && c.y <= 8.0 + 1e-10
            }),
            "corners should be within bbox, got {corners:?}"
        );
    }

    #[test]
    fn ring_clipped_to_two_disjoint_rects_yields_two_polygons() {
        // Non-rectangular clip region: a ring spanning both rects (and the
        // uncovered gap between them) must split into two separate polygons.
        let clip = MultiPolygon::new(vec![
            Polygon::from(Bbox {
                west: 0.0,
                south: 0.0,
                east: 4.0,
                north: 10.0,
            }),
            Polygon::from(Bbox {
                west: 6.0,
                south: 0.0,
                east: 10.0,
                north: 10.0,
            }),
        ]);
        let ring = Polygon::new(
            LineString::from(vec![[1.0, 1.0], [9.0, 1.0], [9.0, 9.0], [1.0, 9.0]]),
            vec![],
        );
        let result = clip_ring(&ring, &clip);
        assert_eq!(
            result.0.len(),
            2,
            "expected one polygon per rect, got {result:?}"
        );
        for p in &result.0 {
            for c in p.exterior().coords() {
                assert!(
                    c.x <= 4.0 + 1e-10 || c.x >= 6.0 - 1e-10,
                    "coordinate {c:?} falls in the uncovered gap"
                );
            }
        }
    }

    // ── light_sectors_to_mvt: buoy circle suppression ────────────────────────

    /// A small square tile region centered on `center`, wide enough to
    /// contain any light-sector arc (max radius 600 m ≪ `margin_deg`).
    fn test_tile_geom(center: Point, margin_deg: f64) -> TileGeom {
        let (west, south) = (center.x() - margin_deg, center.y() - margin_deg);
        let (east, north) = (center.x() + margin_deg, center.y() + margin_deg);
        let (west_m, south_m) = wgs84_to_webmercator(west, south);
        let (east_m, north_m) = wgs84_to_webmercator(east, north);
        TileGeom {
            geom: rect(west, south, east, north),
            merc: Bbox {
                west: west_m,
                south: south_m,
                east: east_m,
                north: north_m,
            },
            scale: 0,
        }
    }

    fn kind_of(f: &MvtFeature) -> Option<&str> {
        f.properties.iter().find_map(|(k, v)| {
            if k != "kind" {
                return None;
            }
            match v {
                MvtValue::String(s) => Some(s.as_str()),
                _ => Some(""),
            }
        })
    }

    #[test]
    fn lateral_or_cardinal_buoy_predicate_matches_only_boylat_boycar() {
        assert!(is_lateral_or_cardinal_buoy("BOYLAT"));
        assert!(is_lateral_or_cardinal_buoy("BOYCAR"));
        assert!(!is_lateral_or_cardinal_buoy("BCNLAT"));
        assert!(!is_lateral_or_cardinal_buoy("BCNCAR"));
        assert!(!is_lateral_or_cardinal_buoy("LIGHTS"));
    }

    #[test]
    fn buoy_light_without_sector_emits_no_circle_but_emits_flare() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let mut layers = HashMap::new();
        let attrs = vec![s57::Attribute {
            code: 75,
            value: s57::AttrValue::Str("1".into()),
        }];
        light_sectors_to_mvt(center, &attrs, &tile, true, &mut layers);
        assert!(
            layers.get("LIGHTS_SECTOR").is_none_or(Vec::is_empty),
            "buoy-mounted all-round light must not draw a synthetic range circle"
        );
        let flare = layers
            .get("LIGHTS_FLARE")
            .expect("buoy-mounted light without a circle must still show a flare icon");
        assert_eq!(flare.len(), 1);
        assert!(
            flare[0]
                .properties
                .iter()
                .any(|(k, v)| k == "COLOUR" && matches!(v, MvtValue::String(s) if s == "1"))
        );
    }

    #[test]
    fn flood_or_subsidiary_buoy_light_emits_neither_circle_nor_flare() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        for catlit in [6_u32, 8_u32] {
            let mut layers = HashMap::new();
            let attrs = vec![s57::Attribute {
                code: 37,
                value: s57::AttrValue::Int(catlit),
            }];
            light_sectors_to_mvt(center, &attrs, &tile, true, &mut layers);
            assert!(
                layers.get("LIGHTS_SECTOR").is_none_or(Vec::is_empty),
                "CATLIT {catlit} must not draw a circle"
            );
            assert!(
                layers.get("LIGHTS_FLARE").is_none_or(Vec::is_empty),
                "CATLIT {catlit} (flood/subsidiary) must not draw a flare icon either"
            );
        }
    }

    #[test]
    fn buoy_light_outside_tile_emits_no_flare() {
        let center = Point::new(10.0, 55.0);
        let far_away_tile = test_tile_geom(Point::new(20.0, 55.0), 0.1);
        let mut layers = HashMap::new();
        light_sectors_to_mvt(center, &[], &far_away_tile, true, &mut layers);
        assert!(layers.get("LIGHTS_FLARE").is_none_or(Vec::is_empty));
    }

    #[test]
    fn non_buoy_light_without_sector_still_emits_circle() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let mut layers = HashMap::new();
        light_sectors_to_mvt(center, &[], &tile, false, &mut layers);
        let feats = layers
            .get("LIGHTS_SECTOR")
            .expect("standalone light should draw its nominal-range circle");
        assert!(feats.iter().any(|f| kind_of(f) == Some("arc")));
    }

    #[test]
    fn sector_bearings_are_drawn_reciprocal_to_seaward_convention() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let attrs = vec![
            s57::Attribute {
                code: 136,
                value: s57::AttrValue::Double(0.0),
            },
            s57::Attribute {
                code: 137,
                value: s57::AttrValue::Double(1.0),
            },
        ];
        let mut layers = HashMap::new();
        light_sectors_to_mvt(center, &attrs, &tile, false, &mut layers);
        let feats = layers
            .get("LIGHTS_SECTOR")
            .expect("sector features expected");
        let radial = feats
            .iter()
            .find(|f| kind_of(f) == Some("radial"))
            .expect("expected a radial boundary line");
        let MvtGeometry::LineString(ls) = &radial.geometry else {
            panic!("radial feature must be a LineString");
        };
        let center_px = to_px(center.into(), tile.merc);
        let tip = ls.0[1];
        // SECTR1 = 0° means a vessel at sea sees the light bearing due
        // north of itself — so the vessel (and the boundary ray drawn
        // outward from the light) lies due *south* of the light, not
        // due north. Regression for the reciprocal-bearing fix.
        assert!(
            tip.y > center_px.y,
            "boundary ray for SECTR1=0° must point south (larger pixel y) \
             of the light, got tip={tip:?} center={center_px:?}"
        );
        assert!(
            (tip.x - center_px.x).abs() <= 2,
            "due-south ray should have ~zero east/west pixel offset, \
             got tip={tip:?} center={center_px:?}"
        );
    }

    #[test]
    fn buoy_light_with_real_sector_still_emits_arc_and_radials() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let attrs = vec![
            s57::Attribute {
                code: 136,
                value: s57::AttrValue::Double(10.0),
            },
            s57::Attribute {
                code: 137,
                value: s57::AttrValue::Double(90.0),
            },
        ];
        let mut layers = HashMap::new();
        light_sectors_to_mvt(center, &attrs, &tile, true, &mut layers);
        let feats = layers
            .get("LIGHTS_SECTOR")
            .expect("real sector data must still be drawn even on a buoy");
        assert!(feats.iter().any(|f| kind_of(f) == Some("arc")));
        assert_eq!(
            feats
                .iter()
                .filter(|f| kind_of(f) == Some("radial"))
                .count(),
            2
        );
        assert!(
            layers.get("LIGHTS_FLARE").is_none_or(Vec::is_empty),
            "a light with real sector data already shown via arcs needs no flare icon"
        );
    }

    #[test]
    fn light_colour_hex_white_renders_as_yellow() {
        assert_eq!(light_colour_hex("1"), "#ccaa00");
        assert_eq!(light_colour_hex(""), "#ccaa00");
    }
}
