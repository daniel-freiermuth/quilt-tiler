//! [`TileSource`] implementation for OESU/S-57 vector cells → MLT tiles.
//!
//! `render` returns plain [`RawFeature`] tuples (geometry + property list) per
//! layer — no builder state, no schema decisions.  [`S57Accumulator`] collects
//! those tuples across cells and lazily feeds them into [`LayerBuf`]s via
//! [`TileAccumulator::push`].  [`LayerBuf`] wraps [`TileLayerBuilder`] and
//! registers property columns on the fly (back-filling earlier features with
//! typed nulls).  [`S57Accumulator::encode`] finishes the builders and
//! serialises each layer as MLT.

use std::collections::HashMap;
use std::collections::hash_map::Entry;

use anyhow::{Context, Result};
use geo::{
    BooleanOps, Coord, HasDimensions, Intersects, LineString, MapCoords, MultiLineString,
    MultiPolygon, Point, Polygon, coord,
};
use martin_tile_utils::wgs84_to_webmercator;
use mlt_core::{
    PropKind, PropValue, PropertyKey, TileLayer, TileLayerBuilder, encoder::EncoderConfig,
};
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::tile_geom::TileGeom;
use crate::tile_source::{TileAccumulator, TileSource};

/// Standard tile grid extent: 4096 units per axis.
const TILE_EXTENT: u32 = 4096;
#[allow(clippy::cast_precision_loss)] // exact: 4096 fits a f64 mantissa
const EXTENT: f64 = TILE_EXTENT as f64;

/// A feature in tile pixel-space: geometry + untyped property list.
///
/// This is the pure data that [`TileSource::render`] returns.  Schema
/// inference and columnar encoding happen later inside [`S57Accumulator`].
type RawFeature = (geo::Geometry<i32>, Vec<(String, PropValue)>);

// ── LayerBuf ──────────────────────────────────────────────────────────────────

/// Accumulates MLT features for one layer during rendering.
///
/// Properties are registered lazily: the first time a name is seen,
/// [`TileLayerBuilder::add_property`] is called (which back-fills all already-
/// pushed features with a typed null).  Relies on S-57's stable per-layer
/// attribute types, so the first-seen type for a column is authoritative.
pub struct LayerBuf {
    builder: TileLayerBuilder,
    /// `name` → `(PropertyKey, registered kind)`
    keys: HashMap<String, (PropertyKey, PropKind)>,
    count: usize,
}

impl LayerBuf {
    fn new(name: &str) -> Result<Self> {
        Ok(Self {
            builder: TileLayer::builder(name, TILE_EXTENT)
                .with_context(|| format!("creating MLT layer builder for {name}"))?,
            keys: HashMap::new(),
            count: 0,
        })
    }

    /// Push one pixel-space feature.  New property names are registered on the
    /// fly; existing features get a typed null via `add_property`'s back-fill.
    ///
    /// # Errors
    /// Returns an error if the MLT builder rejects a property or feature —
    /// i.e. the layer schema has become inconsistent.
    fn push(&mut self, geom: geo::Geometry<i32>, props: Vec<(String, PropValue)>) -> Result<()> {
        // One pass: get-or-register each column, collecting (key, kind, value).
        let mut keyed_props: Vec<(PropertyKey, PropKind, PropValue)> =
            Vec::with_capacity(props.len());
        for (name, val) in props {
            let (key, kind) = match self.keys.entry(name) {
                Entry::Occupied(e) => *e.get(),
                Entry::Vacant(e) => {
                    let kind = PropKind::from(&val);
                    let key = self
                        .builder
                        .add_property(e.key(), kind)
                        .with_context(|| format!("registering MLT property {}", e.key()))?;
                    *e.insert((key, kind))
                }
            };
            keyed_props.push((key, kind, val));
        }

        let mut feat = self.builder.feature(geom);
        for (key, kind, val) in keyed_props {
            feat.property(key, coerce(val, kind))
                .context("setting MLT feature property")?;
        }
        feat.finish().context("finishing MLT feature")?;
        self.count += 1;
        Ok(())
    }

    #[must_use]
    pub const fn feature_count(&self) -> usize {
        self.count
    }

    fn finish(self) -> TileLayer {
        self.builder.finish()
    }
}

// ── TileSource impl ───────────────────────────────────────────────────────────

impl TileSource for s57::S57Cell {
    type Content = HashMap<&'static str, Vec<RawFeature>>;
    type Accumulator = S57Accumulator;
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
        let mut layers: HashMap<&'static str, Vec<RawFeature>> = HashMap::new();

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
            push_features(feat, tile, layers.entry(layer_name).or_default());
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
                light_sectors_to_features(
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

    fn tile_type() -> TileType {
        TileType::Mlt
    }
}

// ── Accumulator ───────────────────────────────────────────────────────────────

/// Accumulates raw per-cell features into [`LayerBuf`]s, then encodes as MLT.
pub struct S57Accumulator(HashMap<&'static str, LayerBuf>);

impl TileAccumulator for S57Accumulator {
    type Content = HashMap<&'static str, Vec<RawFeature>>;

    fn empty() -> Self {
        Self(HashMap::new())
    }

    fn push(&mut self, content: Self::Content) -> Result<()> {
        for (name, feats) in content {
            let buf = match self.0.entry(name) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => e.insert(LayerBuf::new(name)?),
            };
            for (geom, props) in feats {
                buf.push(geom, props)?;
            }
        }
        Ok(())
    }

    fn encode(self) -> Result<Vec<u8>> {
        let cfg = EncoderConfig::default();
        let mut layers: Vec<_> = self.0.into_iter().collect();
        layers.sort_unstable_by_key(|(name, _)| *name);
        let mut out = Vec::new();
        for (_, buf) in layers {
            if buf.feature_count() == 0 {
                continue;
            }
            let encoded = buf.finish().encode(cfg).context("encoding MLT layer")?;
            out.extend_from_slice(&encoded);
        }
        Ok(out)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

/// Coerce `val` to `kind`, converting to a string representation as a last resort.
///
/// S-57 attribute types are stable per attribute code, so this is normally a
/// no-op.  It defends against the rare case where the same attribute appears
/// with different types across features (e.g. CATLIT as both U64 and Str).
fn coerce(val: PropValue, kind: PropKind) -> PropValue {
    if PropKind::from(&val) == kind {
        return val;
    }
    match (val, kind) {
        (PropValue::U64(Some(v)), PropKind::I64) => {
            PropValue::I64(Some(i64::try_from(v).unwrap_or(i64::MAX)))
        }
        (PropValue::F32(Some(v)), PropKind::F64) => PropValue::F64(Some(f64::from(v))),
        (v, PropKind::Str) => PropValue::Str(Some(match v {
            PropValue::Bool(Some(b)) => b.to_string(),
            PropValue::I64(Some(i)) => i.to_string(),
            PropValue::U64(Some(u)) => u.to_string(),
            PropValue::F32(Some(f)) => f.to_string(),
            PropValue::F64(Some(f)) => f.to_string(),
            _ => return PropValue::Str(None),
        })),
        (_, k) => PropValue::null(k),
    }
}

// ── Coordinate projection ─────────────────────────────────────────────────────

/// Project `(lon, lat)` WGS84 to tile pixel coordinates in `[0, EXTENT]` space.
///
/// Geometry is clipped to the tile bbox before this is called, so all
/// projected coordinates stay within the valid range.
#[allow(clippy::cast_possible_truncation)] // deliberate floor-truncation to pixel
fn to_px(wgs84_coord: Coord, merc: Bbox) -> Coord<i32> {
    let (x_m, y_m) = wgs84_to_webmercator(wgs84_coord.x, wgs84_coord.y);
    let px = ((x_m - merc.west) / (merc.east - merc.west) * EXTENT) as i32;
    let py = ((merc.north - y_m) / (merc.north - merc.south) * EXTENT) as i32; // y=0 at north
    coord! { x: px, y: py }
}

// ── Feature filtering ─────────────────────────────────────────────────────────

fn feat_intersects(feat: &s57::Feature, tile_geom: &MultiPolygon) -> bool {
    match &feat.geometry {
        s57::Geometry::None => false,
        s57::Geometry::Point(p) => tile_geom.intersects(p),
        s57::Geometry::Soundings(pts) => pts.iter().any(|(p, _)| tile_geom.intersects(p)),
        s57::Geometry::Line(ls) => tile_geom.intersects(ls),
        s57::Geometry::Area(poly) => tile_geom.intersects(poly),
    }
}

// ── Geometry clipping ─────────────────────────────────────────────────────────

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

// ── Feature → tile pixel push ─────────────────────────────────────────────────

/// Push pixel-space features for one S-57 feature directly into `out`.
/// Geometry is clipped to `tile.geom`; soundings are additionally filtered to
/// their exact containing tile.
#[profiling::function]
fn push_features(feat: &s57::Feature, tile: &TileGeom, out: &mut Vec<RawFeature>) {
    // SCAMIN: skip features whose minimum display scale is coarser than this tile.
    const SCAMIN_CODE: u16 = 133;
    if let Some(attr) = feat.attributes.iter().find(|a| a.code == SCAMIN_CODE)
        && let s57::AttrValue::Int(scamin) = attr.value
        && scamin < tile.scale
    {
        return;
    }

    let props = build_props(&feat.attributes);

    match &feat.geometry {
        s57::Geometry::None => {}

        s57::Geometry::Point(p) => {
            let c = to_px((*p).into(), tile.merc);
            out.push((geo::Geometry::Point(geo::Point::new(c.x, c.y)), props));
        }

        s57::Geometry::Soundings(pts) => {
            for (wgs_coord, depth) in pts.iter().filter(|(p, _)| tile.geom.intersects(p)) {
                let c = to_px((*wgs_coord).into(), tile.merc);
                let mut feat_props = props.clone();
                feat_props.push(("VALDCO".to_string(), PropValue::F64(Some(*depth))));
                out.push((geo::Geometry::Point(geo::Point::new(c.x, c.y)), feat_props));
            }
        }

        s57::Geometry::Line(stroke) => {
            if stroke.is_empty() {
                return;
            }
            let clipped: MultiLineString = clip_stroke(stroke, &tile.geom);
            if clipped.is_empty() {
                return;
            }
            let px: geo::MultiLineString<i32> = clipped.map_coords(|coord| to_px(coord, tile.merc));
            out.push((geo::Geometry::MultiLineString(px), props));
        }

        s57::Geometry::Area(ag) => {
            if ag.is_empty() {
                return;
            }
            let clipped = clip_ring(ag, &tile.geom);
            if clipped.is_empty() {
                return;
            }
            let px: geo::MultiPolygon<i32> = clipped.map_coords(|coord| to_px(coord, tile.merc));
            out.push((geo::Geometry::MultiPolygon(px), props));
        }
    }
}

fn build_props(attrs: &[s57::Attribute]) -> Vec<(String, PropValue)> {
    attrs
        .iter()
        .filter_map(|attr| {
            let key = s57::attribute_acronym(attr.code)?;
            let val = match &attr.value {
                s57::AttrValue::Int(i) => PropValue::U64(Some(u64::from(*i))),
                s57::AttrValue::Double(f) => PropValue::F64(Some(*f)),
                s57::AttrValue::Str(s) => PropValue::Str(Some(s.clone())),
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
    coord![x: d_lon.mul_add(math_rad.cos(), coord.x), y: d_lat.mul_add(math_rad.sin(), coord.y)]
}

/// `true` for lateral and cardinal buoys: their lights are plain all-round
/// lights with no real sector data, so the synthetic "no sector" full circle
/// (see `light_sectors_to_features`) is just clutter and is suppressed for them.
fn is_lateral_or_cardinal_buoy(acronym: &str) -> bool {
    matches!(acronym, "BOYLAT" | "BOYCAR")
}

/// Emits a small flare-icon marker for a buoy-mounted all-round light that
/// has no real sector data, and therefore no range-circle drawn for it
/// (the synthetic "no sector" circle is suppressed as clutter — see
/// `light_sectors_to_features`).  CATLIT 6/8 (flood / subsidiary light) aren't
/// standalone aids to navigation, so those are skipped too.
fn emit_buoy_light_flare(
    center: Point,
    colour: &str,
    catlit: Option<&PropValue>,
    tile: &TileGeom,
    out: &mut Vec<RawFeature>,
) {
    let is_flood_or_subsidiary = match catlit {
        Some(PropValue::U64(Some(6 | 8))) => true,
        Some(PropValue::Str(Some(s))) if s == "6" || s == "8" => true,
        _ => false,
    };
    if is_flood_or_subsidiary || !tile.geom.intersects(&center) {
        return;
    }
    let c = to_px(center.into(), tile.merc);
    let geom = geo::Geometry::Point(geo::Point::new(c.x, c.y));
    let mut props = vec![(
        "COLOUR".to_string(),
        PropValue::Str(Some(colour.to_string())),
    )];
    if let Some(cv) = catlit {
        props.push(("CATLIT".to_string(), cv.clone()));
    }
    out.push((geom, props));
}

/// Generate arc and radial sector features for one `LIGHTS` point.
///
/// Appends to `layers["LIGHTS_SECTOR"]`.
/// Attribute codes: `CATLIT=37  COLOUR=75  SECTR1=136  SECTR2=137  VALNMR=178`
#[allow(clippy::too_many_lines)]
fn light_sectors_to_features(
    center: Point,
    attrs: &[s57::Attribute],
    tile: &TileGeom,
    on_lateral_cardinal_buoy: bool,
    layers: &mut HashMap<&'static str, Vec<RawFeature>>,
) {
    let mut catlit: Option<PropValue> = None;
    let mut colour = "";
    let mut sectr1: Option<f64> = None;
    let mut sectr2: Option<f64> = None;
    let mut valnmr: f64 = 3.0;

    for attr in attrs {
        match attr.code {
            37 => {
                catlit = Some(match &attr.value {
                    s57::AttrValue::Int(i) => PropValue::U64(Some(u64::from(*i))),
                    s57::AttrValue::Str(s) => PropValue::Str(Some(s.clone())),
                    s57::AttrValue::Double(f) => PropValue::F64(Some(*f)),
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
    let sectors = match (sectr1, sectr2) {
        (Some(s1), Some(s2)) if s1 != s2 => Some([s1, s2]),
        _ => None,
    };
    if sectors.is_none() && on_lateral_cardinal_buoy {
        // Plain all-round buoy light: skip the synthetic "no sector" full
        // circle (clutter — it conveys no real sector information here),
        // and draw a small tilted flare icon instead so the buoy's light
        // still shows up on the chart.
        emit_buoy_light_flare(
            center,
            colour,
            catlit.as_ref(),
            tile,
            layers.entry("LIGHTS_FLARE").or_default(),
        );
        return;
    }
    let [from_brg, to_brg_raw] = sectors.unwrap_or([0.0, 360.0]);
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
                let brg = f64::mul_add(span, i as f64 / steps as f64, from_brg);
                bearing_offset(center.into(), brg, r_m)
            })
            .collect(),
    );

    let sector_feats = layers.entry("LIGHTS_SECTOR").or_default();
    let mut push_line = |pts: LineString, kind: &'static str| {
        for stroke in clip_stroke(&pts, &tile.geom) {
            let ls: geo::LineString<i32> = stroke.map_coords(|c| to_px(c, tile.merc));
            let mut props = vec![
                ("kind".to_string(), PropValue::Str(Some(kind.to_string()))),
                ("color".to_string(), PropValue::Str(Some(hex.to_string()))),
            ];
            if let Some(cv) = &catlit {
                props.push(("CATLIT".to_string(), cv.clone()));
            }
            sector_feats.push((geo::Geometry::LineString(ls), props));
        }
    };

    push_line(arc, "arc");
    if let Some(boundaries) = sectors {
        for brg in boundaries {
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

// ── Tests ─────────────────────────────────────────────────────────────────────

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

    // ── push_features ─────────────────────────────────────────────────────

    /// Build a minimal `s57::Feature` for `push_features` tests.
    fn feat(geometry: s57::Geometry, attributes: Vec<s57::Attribute>) -> s57::Feature {
        s57::Feature {
            type_code: 1,
            id: 1,
            primitive: 0,
            attributes,
            geometry,
        }
    }

    #[test]
    fn push_features_none_geometry_produces_nothing() {
        let tile = test_tile_geom(Point::new(10.0, 55.0), 0.1);
        let f = feat(s57::Geometry::None, vec![]);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_point_projects_to_tile_center() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let f = feat(s57::Geometry::Point(center), vec![]);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1, "exactly one feature expected");
        let geo::Geometry::Point(p) = &out[0].0 else {
            panic!("expected Point geometry");
        };
        // Center of tile → pixel ~(EXTENT/2, EXTENT/2).  Mercator projection
        // is non-linear in latitude, so the y-pixel may be off by a few.
        assert!(
            (p.x() - 2048).abs() <= 2 && (p.y() - 2048).abs() <= 4,
            "center point should map to ~(2048,2048), got ({}, {})",
            p.x(),
            p.y()
        );
    }

    #[test]
    fn push_features_point_carries_properties() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        // code 75 = COLOUR
        let attrs = vec![s57::Attribute {
            code: 75,
            value: s57::AttrValue::Str("3".into()),
        }];
        let f = feat(s57::Geometry::Point(center), attrs);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1);
        assert!(
            out[0]
                .1
                .iter()
                .any(|(k, v)| k == "COLOUR" && matches!(v, PropValue::Str(Some(s)) if s == "3")),
            "expected COLOUR=3 in properties, got {:?}",
            out[0].1
        );
    }

    #[test]
    fn push_features_soundings_filters_outside_tile_and_adds_valdco() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let inside = Point::new(10.0, 55.0);
        let outside = Point::new(20.0, 55.0); // far outside
        let soundings = vec![(inside, 12.5), (outside, 99.0)];
        let f = feat(s57::Geometry::Soundings(soundings), vec![]);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1, "only the inside sounding should survive");
        // Check VALDCO property
        let valdco = out[0]
            .1
            .iter()
            .find(|(k, _)| k == "VALDCO")
            .expect("sounding must carry VALDCO property");
        assert_eq!(valdco.1, PropValue::F64(Some(12.5)));
    }

    #[test]
    fn push_features_soundings_all_outside_produces_nothing() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let outside = Point::new(20.0, 60.0);
        let f = feat(s57::Geometry::Soundings(vec![(outside, 5.0)]), vec![]);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_line_empty_produces_nothing() {
        let tile = test_tile_geom(Point::new(10.0, 55.0), 0.1);
        let f = feat(s57::Geometry::Line(LineString::new(vec![])), vec![]);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_line_fully_outside_produces_nothing() {
        let tile = test_tile_geom(Point::new(10.0, 55.0), 0.1);
        // Line entirely outside the tile bbox
        let f = feat(
            s57::Geometry::Line(LineString::from(vec![[20.0, 60.0], [21.0, 60.0]])),
            vec![],
        );
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_line_clips_to_tile() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        // Line starts inside, extends far outside to the east
        let f = feat(
            s57::Geometry::Line(LineString::from(vec![[10.0, 55.0], [20.0, 55.0]])),
            vec![],
        );
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1, "clipped line should produce one feature");
        let geo::Geometry::MultiLineString(mls) = &out[0].0 else {
            panic!("expected MultiLineString geometry");
        };
        assert!(!mls.0.is_empty(), "clipped result must not be empty");
        // All pixel x-coords should be in [0, EXTENT]
        for ls in &mls.0 {
            for c in ls.coords() {
                assert!(
                    c.x >= 0 && c.x <= TILE_EXTENT.cast_signed(),
                    "pixel x out of range: {c:?}"
                );
            }
        }
    }

    #[test]
    fn push_features_area_empty_produces_nothing() {
        let tile = test_tile_geom(Point::new(10.0, 55.0), 0.1);
        let f = feat(
            s57::Geometry::Area(Polygon::new(LineString::new(vec![]), vec![])),
            vec![],
        );
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_area_fully_outside_produces_nothing() {
        let tile = test_tile_geom(Point::new(10.0, 55.0), 0.1);
        let f = feat(
            s57::Geometry::Area(Polygon::new(
                LineString::from(vec![[20.0, 60.0], [21.0, 60.0], [21.0, 61.0], [20.0, 61.0]]),
                vec![],
            )),
            vec![],
        );
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn push_features_area_clips_to_tile() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        // Polygon extends beyond the tile to the east
        let f = feat(
            s57::Geometry::Area(Polygon::new(
                LineString::from(vec![
                    [9.95, 54.95],
                    [10.2, 54.95],
                    [10.2, 55.05],
                    [9.95, 55.05],
                ]),
                vec![],
            )),
            vec![],
        );
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1, "clipped polygon should produce one feature");
        let geo::Geometry::MultiPolygon(mp) = &out[0].0 else {
            panic!("expected MultiPolygon geometry");
        };
        assert!(!mp.0.is_empty(), "clipped result must not be empty");
        // All pixel coords should be in [0, EXTENT]
        for poly in &mp.0 {
            for c in poly.exterior().coords() {
                assert!(
                    c.x >= 0
                        && c.x <= TILE_EXTENT.cast_signed()
                        && c.y >= 0
                        && c.y <= TILE_EXTENT.cast_signed(),
                    "pixel coord out of range: {c:?}"
                );
            }
        }
    }

    #[test]
    fn push_features_scamin_below_tile_scale_is_skipped() {
        let center = Point::new(10.0, 55.0);
        let mut tile = test_tile_geom(center, 0.1);
        tile.scale = 100;
        // SCAMIN = 50 < tile.scale = 100 → feature should be skipped
        let attrs = vec![s57::Attribute {
            code: 133, // SCAMIN
            value: s57::AttrValue::Int(50),
        }];
        let f = feat(s57::Geometry::Point(center), attrs);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert!(
            out.is_empty(),
            "feature with SCAMIN < tile.scale must be skipped"
        );
    }

    #[test]
    fn push_features_scamin_above_tile_scale_passes() {
        let center = Point::new(10.0, 55.0);
        let mut tile = test_tile_geom(center, 0.1);
        tile.scale = 100;
        // SCAMIN = 200 >= tile.scale = 100 → feature should NOT be skipped
        let attrs = vec![s57::Attribute {
            code: 133,
            value: s57::AttrValue::Int(200),
        }];
        let f = feat(s57::Geometry::Point(center), attrs);
        let mut out = Vec::new();
        push_features(&f, &tile, &mut out);
        assert_eq!(out.len(), 1, "feature with SCAMIN >= tile.scale must pass");
    }

    // ── build_props ───────────────────────────────────────────────────────

    #[test]
    fn build_props_converts_all_attr_value_variants() {
        let attrs = vec![
            s57::Attribute {
                code: 75, // COLOUR
                value: s57::AttrValue::Str("1".into()),
            },
            s57::Attribute {
                code: 178, // VALNMR
                value: s57::AttrValue::Double(3.5),
            },
            s57::Attribute {
                code: 133, // SCAMIN
                value: s57::AttrValue::Int(50_000),
            },
        ];
        let props = build_props(&attrs);
        assert_eq!(props.len(), 3);
        assert_eq!(props[0].0, "COLOUR");
        assert_eq!(props[0].1, PropValue::Str(Some("1".into())));
        assert_eq!(props[1].0, "VALNMR");
        assert_eq!(props[1].1, PropValue::F64(Some(3.5)));
        assert_eq!(props[2].0, "SCAMIN");
        assert_eq!(props[2].1, PropValue::U64(Some(50_000)));
    }

    #[test]
    fn build_props_skips_unknown_attribute_codes() {
        let attrs = vec![
            s57::Attribute {
                code: 75, // COLOUR — known
                value: s57::AttrValue::Str("1".into()),
            },
            s57::Attribute {
                code: 9999, // unknown code
                value: s57::AttrValue::Int(42),
            },
        ];
        let props = build_props(&attrs);
        assert_eq!(props.len(), 1, "unknown codes must be filtered out");
        assert_eq!(props[0].0, "COLOUR");
    }

    // ── light_sectors_to_features: buoy circle suppression ───────────────────

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

    fn kind_of(feat: &RawFeature) -> Option<&str> {
        feat.1.iter().find_map(|(k, v)| {
            if k != "kind" {
                return None;
            }
            if let PropValue::Str(Some(s)) = v {
                Some(s.as_str())
            } else {
                None
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
        light_sectors_to_features(center, &attrs, &tile, true, &mut layers);
        assert!(
            layers.get("LIGHTS_SECTOR").is_none_or(Vec::is_empty),
            "buoy-mounted all-round light must not draw a synthetic range circle"
        );
        let flare = layers
            .remove("LIGHTS_FLARE")
            .expect("buoy-mounted light without a circle must still show a flare icon");
        assert_eq!(flare.len(), 1);
        assert!(
            flare[0]
                .1
                .iter()
                .any(|(k, v)| k == "COLOUR" && matches!(v, PropValue::Str(Some(s)) if s == "1"))
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
            light_sectors_to_features(center, &attrs, &tile, true, &mut layers);
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
    fn flood_or_subsidiary_buoy_light_string_encoded_also_suppressed() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        for catlit_str in ["6", "8"] {
            let mut layers = HashMap::new();
            let attrs = vec![s57::Attribute {
                code: 37,
                value: s57::AttrValue::Str(catlit_str.into()),
            }];
            light_sectors_to_features(center, &attrs, &tile, true, &mut layers);
            assert!(
                layers.get("LIGHTS_SECTOR").is_none_or(Vec::is_empty),
                "string-encoded CATLIT {catlit_str} must not draw a circle"
            );
            assert!(
                layers.get("LIGHTS_FLARE").is_none_or(Vec::is_empty),
                "string-encoded CATLIT {catlit_str} (flood/subsidiary) must not draw a flare"
            );
        }
    }

    #[test]
    fn buoy_light_outside_tile_emits_no_flare() {
        let center = Point::new(10.0, 55.0);
        let far_away_tile = test_tile_geom(Point::new(20.0, 55.0), 0.1);
        let mut layers = HashMap::new();
        light_sectors_to_features(center, &[], &far_away_tile, true, &mut layers);
        assert!(layers.get("LIGHTS_FLARE").is_none_or(Vec::is_empty));
    }

    #[test]
    fn non_buoy_light_without_sector_still_emits_circle() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, 0.1);
        let mut layers = HashMap::new();
        light_sectors_to_features(center, &[], &tile, false, &mut layers);
        let sector = layers
            .remove("LIGHTS_SECTOR")
            .expect("standalone light should draw its nominal-range circle");
        assert!(sector.iter().any(|f| kind_of(f) == Some("arc")));
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
        light_sectors_to_features(center, &attrs, &tile, false, &mut layers);
        let sector = layers
            .remove("LIGHTS_SECTOR")
            .expect("sector features expected");
        let radial = sector
            .iter()
            .find(|f| kind_of(f) == Some("radial"))
            .expect("expected a radial boundary line");
        let geo::Geometry::LineString(ls) = &radial.0 else {
            panic!("radial feature must be a LineString");
        };
        let center_px = to_px(center.into(), tile.merc);
        let tip = ls.0[1];
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
    fn bearing_offset_is_clockwise_from_north_in_metres() {
        let origin = coord! { x: 10.0, y: 55.0 };
        let metres_per_deg_lat = 111_320.0;
        let metres_per_deg_lon = metres_per_deg_lat * 55.0_f64.to_radians().cos();
        let close = |a: f64, b: f64| (a - b).abs() < 1e-9;

        let north = bearing_offset(origin, 0.0, 1000.0);
        assert!(
            close(north.x, 10.0),
            "0° must not move east/west: {north:?}"
        );
        assert!(
            close(north.y, 55.0 + 1000.0 / metres_per_deg_lat),
            "{north:?}"
        );

        // Clockwise: 90° is east, not west.
        let east = bearing_offset(origin, 90.0, 1000.0);
        assert!(
            close(east.x, 10.0 + 1000.0 / metres_per_deg_lon),
            "{east:?}"
        );
        assert!(
            close(east.y, 55.0),
            "90° must not move north/south: {east:?}"
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
        light_sectors_to_features(center, &attrs, &tile, true, &mut layers);
        let sector = layers
            .remove("LIGHTS_SECTOR")
            .expect("real sector data must still be drawn even on a buoy");
        assert!(sector.iter().any(|f| kind_of(f) == Some("arc")));
        assert_eq!(
            sector
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

    // ── light_sectors_to_features: arc geometry (span, wraparound, radius) ──

    /// ±0.025° at 55°N still contains the longest radial (2 × 600 m ≈ 1200 m
    /// vs ≈ 1600 m east/west half-width) while keeping pixels ≲ 1.4 m on the
    /// ground, so bearings/radii can be measured to ~1°/~4 m.
    const GEOM_MARGIN_DEG: f64 = 0.025;
    const BEARING_TOL_DEG: f64 = 1.0;
    const RADIUS_TOL_M: f64 = 4.0;

    /// Compass bearing (0° = north, clockwise, in `[0, 360)`) and ground
    /// distance in metres of pixel `px` from `center`.  Undoes both the
    /// tile's anisotropic pixel scale (equal degree spans in lon/lat) and the
    /// Mercator 1/cos(lat) stretch, independently of `bearing_offset`.
    fn polar_from_center(tile: &TileGeom, center: Point, px: Coord<i32>) -> (f64, f64) {
        let c = to_px(center.into(), tile.merc);
        let ground_per_merc = center.y().to_radians().cos();
        let m_per_px_x = (tile.merc.east - tile.merc.west) / EXTENT * ground_per_merc;
        let m_per_px_y = (tile.merc.north - tile.merc.south) / EXTENT * ground_per_merc;
        let east = f64::from(px.x - c.x) * m_per_px_x;
        let north = f64::from(c.y - px.y) * m_per_px_y; // pixel y grows southward
        (
            east.atan2(north).to_degrees().rem_euclid(360.0),
            east.hypot(north),
        )
    }

    /// Signed clockwise turn from bearing `a` to bearing `b`, in `(-180, 180]`.
    fn clockwise_delta(a: f64, b: f64) -> f64 {
        180.0 - (a - b + 180.0).rem_euclid(360.0)
    }

    fn assert_bearing_near(actual: f64, expected: f64, what: &str) {
        assert!(
            clockwise_delta(expected, actual).abs() <= BEARING_TOL_DEG,
            "{what}: bearing {actual:.2}° not within {BEARING_TOL_DEG}° of {expected}°"
        );
    }

    fn light_attrs(
        sectr1: Option<f64>,
        sectr2: Option<f64>,
        valnmr: Option<f64>,
    ) -> Vec<s57::Attribute> {
        [(136, sectr1), (137, sectr2), (178, valnmr)]
            .into_iter()
            .filter_map(|(code, v)| {
                v.map(|v| s57::Attribute {
                    code,
                    value: s57::AttrValue::Double(v),
                })
            })
            .collect()
    }

    /// Polar coordinates of rendered light-sector geometry: every arc vertex
    /// in drawing order, and each radial's outer tip in emission order.
    struct SectorPolar {
        arc: Vec<(f64, f64)>,
        radial_tips: Vec<(f64, f64)>,
    }

    impl SectorPolar {
        /// Total clockwise sweep of the arc, summed vertex to vertex.
        fn arc_span(&self) -> f64 {
            self.arc
                .windows(2)
                .map(|w| clockwise_delta(w[0].0, w[1].0))
                .sum()
        }
    }

    fn render_sector_polar(
        center: Point,
        tile: &TileGeom,
        attrs: &[s57::Attribute],
    ) -> SectorPolar {
        let mut layers = HashMap::new();
        light_sectors_to_features(center, attrs, tile, false, &mut layers);
        let feats = layers
            .remove("LIGHTS_SECTOR")
            .expect("light must emit sector features");
        let line_of = |f: &RawFeature| -> Vec<Coord<i32>> {
            let geo::Geometry::LineString(ls) = &f.0 else {
                panic!("sector feature must be a LineString, got {:?}", f.0);
            };
            ls.0.clone()
        };
        let arcs: Vec<_> = feats.iter().filter(|f| kind_of(f) == Some("arc")).collect();
        assert_eq!(arcs.len(), 1, "arc must be one unclipped stroke");
        SectorPolar {
            arc: line_of(arcs[0])
                .into_iter()
                .map(|c| polar_from_center(tile, center, c))
                .collect(),
            radial_tips: feats
                .iter()
                .filter(|f| kind_of(f) == Some("radial"))
                .map(|f| {
                    let pts = line_of(f);
                    assert_eq!(pts.len(), 2, "radial is a single center→tip segment");
                    polar_from_center(tile, center, pts[1])
                })
                .collect(),
        }
    }

    #[test]
    fn light_sector_arc_sweeps_clockwise_from_sectr1_to_sectr2_across_north() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, GEOM_MARGIN_DEG);
        // (SECTR1, SECTR2, drawn start, drawn end, clockwise span)
        let cases = [
            // flips to 350→10: must sweep 20° through north, not the 340° complement
            (170.0, 190.0, 350.0, 10.0, 20.0),
            // raw from > to: span is still clockwise SECTR1→SECTR2
            (300.0, 60.0, 120.0, 240.0, 120.0),
            // raw from > to *and* flipped from > to: wide sector across north
            (100.0, 20.0, 280.0, 200.0, 280.0),
            // no wraparound anywhere
            (10.0, 90.0, 190.0, 270.0, 80.0),
        ];
        for (s1, s2, start, end, span) in cases {
            let what = format!("SECTR1={s1} SECTR2={s2}");
            let polar = render_sector_polar(center, &tile, &light_attrs(Some(s1), Some(s2), None));
            let (first, last) = (polar.arc[0], polar.arc[polar.arc.len() - 1]);
            assert_bearing_near(first.0, start, &format!("{what}: arc start"));
            assert_bearing_near(last.0, end, &format!("{what}: arc end"));
            assert!(
                (polar.arc_span() - span).abs() <= BEARING_TOL_DEG,
                "{what}: arc sweeps {:.2}°, expected {span}°",
                polar.arc_span()
            );
            assert_eq!(polar.radial_tips.len(), 2, "{what}: two boundary radials");
            // Arc endpoints coincide with the boundary radials (SECTR1 first).
            assert_bearing_near(
                polar.radial_tips[0].0,
                first.0,
                &format!("{what}: SECTR1 radial"),
            );
            assert_bearing_near(
                polar.radial_tips[1].0,
                last.0,
                &format!("{what}: SECTR2 radial"),
            );
        }
    }

    #[test]
    fn light_without_two_distinct_sector_limits_draws_full_circle_without_radials() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, GEOM_MARGIN_DEG);
        for (s1, s2) in [
            (Some(45.0), Some(45.0)),
            (Some(45.0), None),
            (None, Some(45.0)),
            (None, None),
        ] {
            let polar = render_sector_polar(center, &tile, &light_attrs(s1, s2, None));
            assert!(
                (polar.arc_span() - 360.0).abs() <= BEARING_TOL_DEG,
                "SECTR1={s1:?} SECTR2={s2:?}: expected full 360° circle, swept {:.2}°",
                polar.arc_span()
            );
            assert!(
                polar.radial_tips.is_empty(),
                "SECTR1={s1:?} SECTR2={s2:?}: all-round light must not draw radials"
            );
        }
    }

    #[test]
    fn light_sector_radius_is_200m_plus_50m_per_valnmr_capped_at_600m() {
        let center = Point::new(10.0, 55.0);
        let tile = test_tile_geom(center, GEOM_MARGIN_DEG);
        // (VALNMR, arc radius in metres); radials extend to twice the radius.
        let cases = [
            (None, 350.0), // default nominal range 3 NM
            (Some(5.0), 450.0),
            (Some(8.0), 600.0),  // exactly at the cap
            (Some(20.0), 600.0), // 1200 m uncapped
        ];
        for (valnmr, r_m) in cases {
            let polar =
                render_sector_polar(center, &tile, &light_attrs(Some(10.0), Some(90.0), valnmr));
            for (brg, dist) in &polar.arc {
                assert!(
                    (dist - r_m).abs() <= RADIUS_TOL_M,
                    "VALNMR={valnmr:?}: arc vertex at {brg:.1}° is {dist:.1} m out, expected {r_m} m"
                );
            }
            assert_eq!(polar.radial_tips.len(), 2);
            for (brg, dist) in &polar.radial_tips {
                assert!(
                    (dist - 2.0 * r_m).abs() <= RADIUS_TOL_M,
                    "VALNMR={valnmr:?}: radial at {brg:.1}° is {dist:.1} m long, expected {} m",
                    2.0 * r_m
                );
            }
        }
    }

    #[test]
    fn light_colour_hex_white_renders_as_yellow() {
        assert_eq!(light_colour_hex("1"), "#ccaa00");
        assert_eq!(light_colour_hex(""), "#ccaa00");
    }

    // ── LayerBuf::push — schema inference & property handling ─────────────

    /// Trivial point geometry for property-focused tests.
    fn pt() -> geo::Geometry<i32> {
        geo::Geometry::Point(geo::Point::new(100, 200))
    }

    /// Shorthand: build a `LayerBuf`, push the given feature property lists,
    /// finish, and return the resulting `TileLayer` for assertions.
    fn push_all(rows: Vec<Vec<(&str, PropValue)>>) -> TileLayer {
        let mut buf = LayerBuf::new("test").expect("valid layer name");
        for row in rows {
            buf.push(
                pt(),
                row.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
            )
            .expect("push succeeds");
        }
        buf.finish()
    }

    #[test]
    fn push_first_feature_registers_columns_with_correct_kind() {
        let layer = push_all(vec![vec![
            ("name", PropValue::Str(Some("A".into()))),
            ("depth", PropValue::F64(Some(3.5))),
            ("count", PropValue::U64(Some(7))),
        ]]);

        assert_eq!(layer.property_names(), &["name", "depth", "count"]);
        assert_eq!(layer.feature_count(), 1);

        let props = layer.features()[0].properties();
        assert_eq!(props[0], PropValue::Str(Some("A".into())));
        assert_eq!(props[1], PropValue::F64(Some(3.5)));
        assert_eq!(props[2], PropValue::U64(Some(7)));
    }

    #[test]
    fn push_backfills_earlier_features_with_typed_nulls() {
        // Feature 0 has only "a"; feature 1 introduces "b".
        let layer = push_all(vec![
            vec![("a", PropValue::I64(Some(1)))],
            vec![
                ("a", PropValue::I64(Some(2))),
                ("b", PropValue::Str(Some("hello".into()))),
            ],
        ]);

        assert_eq!(layer.property_names(), &["a", "b"]);

        // Feature 0 was pushed before "b" existed → back-filled with Str(None).
        let f0 = layer.features()[0].properties();
        assert_eq!(f0[0], PropValue::I64(Some(1)));
        assert_eq!(
            f0[1],
            PropValue::Str(None),
            "back-fill must be a typed null"
        );

        // Feature 1 has both columns populated.
        let f1 = layer.features()[1].properties();
        assert_eq!(f1[0], PropValue::I64(Some(2)));
        assert_eq!(f1[1], PropValue::Str(Some("hello".into())));
    }

    #[test]
    fn push_coerces_mismatched_type_to_first_seen_kind() {
        // First feature establishes "val" as I64.
        // Second feature provides "val" as U64 → coerced to I64.
        let layer = push_all(vec![
            vec![("val", PropValue::I64(Some(10)))],
            vec![("val", PropValue::U64(Some(20)))],
        ]);

        let f1 = layer.features()[1].properties();
        assert_eq!(
            f1[0],
            PropValue::I64(Some(20)),
            "U64(20) must coerce to I64(20)"
        );
    }

    #[test]
    fn push_coerces_f32_to_f64_when_column_is_f64() {
        let layer = push_all(vec![
            vec![("depth", PropValue::F64(Some(1.0)))],
            vec![("depth", PropValue::F32(Some(2.5)))],
        ]);

        let f1 = layer.features()[1].properties();
        assert_eq!(
            f1[0],
            PropValue::F64(Some(2.5_f64)),
            "F32(2.5) must widen to F64(2.5)"
        );
    }

    #[test]
    fn push_coerces_incompatible_type_to_string_fallback() {
        // Column is Str; second feature provides a Bool → coerced to Str.
        let layer = push_all(vec![
            vec![("flag", PropValue::Str(Some("yes".into())))],
            vec![("flag", PropValue::Bool(Some(true)))],
        ]);

        let f1 = layer.features()[1].properties();
        assert_eq!(
            f1[0],
            PropValue::Str(Some("true".into())),
            "Bool(true) coerced to Str column must become Str(\"true\")"
        );
    }

    #[test]
    fn push_coerces_to_null_when_no_string_representation() {
        // Column is I64; second feature provides a Bool → no direct coerce
        // path, falls through to null(I64).
        let layer = push_all(vec![
            vec![("x", PropValue::I64(Some(1)))],
            vec![("x", PropValue::Bool(Some(true)))],
        ]);

        let f1 = layer.features()[1].properties();
        assert_eq!(
            f1[0],
            PropValue::I64(None),
            "incompatible coercion with no conversion path must produce typed null"
        );
    }

    #[test]
    fn push_empty_props_creates_feature_with_no_columns() {
        let layer = push_all(vec![vec![]]);

        assert_eq!(layer.feature_count(), 1);
        assert_eq!(layer.property_names(), &[] as &[&str]);
        assert_eq!(layer.features()[0].properties(), []);
    }

    #[test]
    fn push_feature_with_no_props_after_one_with_props_still_backfills() {
        // Feature 0 has "a"; feature 1 has nothing → "a" column exists,
        // feature 1 gets a null for "a" via the builder.
        let layer = push_all(vec![vec![("a", PropValue::U64(Some(42)))], vec![]]);

        assert_eq!(layer.feature_count(), 2);
        let f1 = layer.features()[1].properties();
        assert_eq!(
            f1[0],
            PropValue::U64(None),
            "feature with no props should have null in already-registered column"
        );
    }

    #[test]
    fn push_property_key_order_is_stable_across_features() {
        // Feature 0: a, b, c.  Feature 1: c, a, b (different insertion order
        // but all keys already registered → order stays a, b, c).
        let layer = push_all(vec![
            vec![
                ("a", PropValue::I64(Some(1))),
                ("b", PropValue::I64(Some(2))),
                ("c", PropValue::I64(Some(3))),
            ],
            vec![
                ("c", PropValue::I64(Some(30))),
                ("a", PropValue::I64(Some(10))),
                ("b", PropValue::I64(Some(20))),
            ],
        ]);

        assert_eq!(
            layer.property_names(),
            &["a", "b", "c"],
            "column order follows first-seen registration"
        );

        // Values in feature 1 must land in the correct columns despite
        // different iteration order in the input.
        let f1 = layer.features()[1].properties();
        assert_eq!(f1[0], PropValue::I64(Some(10)), "column a");
        assert_eq!(f1[1], PropValue::I64(Some(20)), "column b");
        assert_eq!(f1[2], PropValue::I64(Some(30)), "column c");
    }

    #[test]
    fn push_null_valued_first_feature_infers_kind_from_variant() {
        // A null Str is still PropKind::Str — the column type is determined
        // by the enum variant, not by the inner Option value.
        let layer = push_all(vec![
            vec![("tag", PropValue::Str(None))],
            vec![("tag", PropValue::Str(Some("hi".into())))],
        ]);

        // Both features should store Str values.
        assert_eq!(layer.features()[0].properties()[0], PropValue::Str(None));
        assert_eq!(
            layer.features()[1].properties()[0],
            PropValue::Str(Some("hi".into()))
        );
    }

    #[test]
    fn push_feature_count_tracks_correctly() {
        let mut buf = LayerBuf::new("fc").expect("valid layer name");
        assert_eq!(buf.feature_count(), 0);
        buf.push(pt(), vec![]).expect("push succeeds");
        assert_eq!(buf.feature_count(), 1);
        buf.push(pt(), vec![]).expect("push succeeds");
        assert_eq!(buf.feature_count(), 2);
    }

    // ── coerce unit tests ─────────────────────────────────────────────────

    #[test]
    fn coerce_same_kind_is_identity() {
        let val = PropValue::I64(Some(42));
        assert_eq!(coerce(val.clone(), PropKind::I64), val);
    }

    #[test]
    fn coerce_u64_to_i64() {
        assert_eq!(
            coerce(PropValue::U64(Some(100)), PropKind::I64),
            PropValue::I64(Some(100))
        );
    }

    #[test]
    fn coerce_u64_overflow_to_i64_clamps() {
        assert_eq!(
            coerce(PropValue::U64(Some(u64::MAX)), PropKind::I64),
            PropValue::I64(Some(i64::MAX))
        );
    }

    #[test]
    fn coerce_f32_to_f64() {
        assert_eq!(
            coerce(PropValue::F32(Some(1.5)), PropKind::F64),
            PropValue::F64(Some(1.5_f64))
        );
    }

    #[test]
    fn coerce_numeric_to_str() {
        assert_eq!(
            coerce(PropValue::I64(Some(99)), PropKind::Str),
            PropValue::Str(Some("99".into()))
        );
        assert_eq!(
            coerce(PropValue::U64(Some(7)), PropKind::Str),
            PropValue::Str(Some("7".into()))
        );
        assert_eq!(
            coerce(PropValue::Bool(Some(false)), PropKind::Str),
            PropValue::Str(Some("false".into()))
        );
    }

    #[test]
    fn coerce_null_to_str_yields_null_str() {
        assert_eq!(
            coerce(PropValue::I64(None), PropKind::Str),
            PropValue::Str(None)
        );
    }

    #[test]
    fn coerce_incompatible_yields_typed_null() {
        // Bool → I64 has no conversion path → typed null.
        assert_eq!(
            coerce(PropValue::Bool(Some(true)), PropKind::I64),
            PropValue::I64(None)
        );
    }

    // ── S57Accumulator::encode — layer sorting, empty-layer skip ──────────

    /// Build a single-layer `HashMap` content entry for `S57Accumulator::push`.
    fn layer_content(
        name: &'static str,
        features: Vec<Vec<(&str, PropValue)>>,
    ) -> HashMap<&'static str, Vec<RawFeature>> {
        let raw: Vec<RawFeature> = features
            .into_iter()
            .map(|props| {
                (
                    pt(),
                    props.into_iter().map(|(k, v)| (k.to_owned(), v)).collect(),
                )
            })
            .collect();
        HashMap::from([(name, raw)])
    }

    #[test]
    fn encode_single_layer_produces_non_empty_bytes() {
        let mut acc = S57Accumulator::empty();
        acc.push(layer_content(
            "DEPARE",
            vec![vec![("DRVAL1", PropValue::F64(Some(1.0)))]],
        ))
        .expect("push succeeds");
        let bytes = acc.encode().expect("encode should succeed");
        assert!(
            !bytes.is_empty(),
            "single layer with one feature should produce bytes"
        );
    }

    #[test]
    fn encode_all_empty_layers_returns_empty_vec() {
        let mut acc = S57Accumulator::empty();
        // Push content maps whose feature vecs are empty — LayerBuf entries
        // are created but their feature_count stays 0.
        acc.push(HashMap::from([("DEPARE", Vec::<RawFeature>::new())]))
            .expect("push succeeds");
        acc.push(HashMap::from([("LNDARE", Vec::<RawFeature>::new())]))
            .expect("push succeeds");
        let bytes = acc.encode().expect("encode should succeed");
        assert!(bytes.is_empty(), "all-empty layers should produce no bytes");
    }

    #[test]
    fn encode_skips_empty_layers_among_populated_ones() {
        // Two layers: DEPARE has a feature, LNDARE is empty.
        let mut acc = S57Accumulator::empty();
        acc.push(layer_content(
            "DEPARE",
            vec![vec![("DRVAL1", PropValue::F64(Some(2.0)))]],
        ))
        .expect("push succeeds");
        acc.push(HashMap::from([("LNDARE", Vec::<RawFeature>::new())]))
            .expect("push succeeds");

        // Encode with both layers present (one empty).
        let bytes_mixed = acc.encode().expect("encode should succeed");

        // Encode with only the populated layer.
        let mut acc_single = S57Accumulator::empty();
        acc_single
            .push(layer_content(
                "DEPARE",
                vec![vec![("DRVAL1", PropValue::F64(Some(2.0)))]],
            ))
            .expect("push succeeds");
        let bytes_single = acc_single.encode().expect("encode should succeed");

        assert_eq!(
            bytes_mixed, bytes_single,
            "empty layers should not contribute bytes"
        );
    }

    #[test]
    fn encode_multi_layer_output_is_deterministic_regardless_of_insertion_order() {
        // Push layers in alphabetical order: BUOYAG, DEPARE, LNDARE.
        let mut acc_alpha = S57Accumulator::empty();
        acc_alpha
            .push(layer_content(
                "BUOYAG",
                vec![vec![("COLOUR", PropValue::Str(Some("1".into())))]],
            ))
            .expect("push succeeds");
        acc_alpha
            .push(layer_content(
                "DEPARE",
                vec![vec![("DRVAL1", PropValue::F64(Some(3.0)))]],
            ))
            .expect("push succeeds");
        acc_alpha
            .push(layer_content(
                "LNDARE",
                vec![vec![("NATION", PropValue::Str(Some("US".into())))]],
            ))
            .expect("push succeeds");
        let bytes_alpha = acc_alpha.encode().expect("encode should succeed");

        // Push layers in reverse order: LNDARE, DEPARE, BUOYAG.
        let mut acc_rev = S57Accumulator::empty();
        acc_rev
            .push(layer_content(
                "LNDARE",
                vec![vec![("NATION", PropValue::Str(Some("US".into())))]],
            ))
            .expect("push succeeds");
        acc_rev
            .push(layer_content(
                "DEPARE",
                vec![vec![("DRVAL1", PropValue::F64(Some(3.0)))]],
            ))
            .expect("push succeeds");
        acc_rev
            .push(layer_content(
                "BUOYAG",
                vec![vec![("COLOUR", PropValue::Str(Some("1".into())))]],
            ))
            .expect("push succeeds");
        let bytes_rev = acc_rev.encode().expect("encode should succeed");

        assert_eq!(
            bytes_alpha, bytes_rev,
            "layer insertion order must not affect output (sort by name)"
        );
    }

    #[test]
    fn encode_multi_layer_concatenates_in_sorted_name_order() {
        // Encode each layer individually to get its bytes.
        let mut acc_b = S57Accumulator::empty();
        acc_b
            .push(layer_content(
                "BUOYAG",
                vec![vec![("COLOUR", PropValue::Str(Some("3".into())))]],
            ))
            .expect("push succeeds");
        let bytes_b = acc_b.encode().expect("encode should succeed");

        let mut acc_d = S57Accumulator::empty();
        acc_d
            .push(layer_content(
                "DEPARE",
                vec![vec![("DRVAL1", PropValue::F64(Some(5.0)))]],
            ))
            .expect("push succeeds");
        let bytes_d = acc_d.encode().expect("encode should succeed");

        // Encode both together (pushed in reverse order to exercise sort).
        let mut acc_both = S57Accumulator::empty();
        acc_both
            .push(layer_content(
                "DEPARE",
                vec![vec![("DRVAL1", PropValue::F64(Some(5.0)))]],
            ))
            .expect("push succeeds");
        acc_both
            .push(layer_content(
                "BUOYAG",
                vec![vec![("COLOUR", PropValue::Str(Some("3".into())))]],
            ))
            .expect("push succeeds");
        let bytes_both = acc_both.encode().expect("encode should succeed");

        // The multi-layer output should be BUOYAG bytes ++ DEPARE bytes
        // (sorted alphabetically: B before D).
        let mut expected = bytes_b;
        expected.extend_from_slice(&bytes_d);
        assert_eq!(
            bytes_both, expected,
            "multi-layer encode should concatenate layers in sorted name order"
        );
    }
}
