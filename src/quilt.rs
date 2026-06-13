//! Spatial/zoom attribution for overlapping OESU chart coverage.
//!
//! The quilting algorithm assigns each GeoJSON feature a `minzoom` (stamped by
//! `convert::cell_to_geojson` from the chart's native scale) and, for features
//! that fall inside a finer chart's coverage area, a `maxzoom` so that the
//! coarser chart gracefully hands off to the finer one at the right zoom level.
//!
//! # Processing order
//!
//! Charts are sorted finest → coarsest (smallest `native_scale` number first).
//! The `covered_zones` list accumulates each chart's feature bounding box as we go.
//! When processing a coarser chart its features are tested against all already-
//! seen finer zones, and the **coarsest covering zone** (lowest minzoom > feature's
//! own minzoom) determines `maxzoom`. This ensures a minzoom=12 feature in an area
//! covered by both a minzoom=14 and a minzoom=15 chart hands off at zoom 13 (when
//! the minzoom=14 chart kicks in) — not at zoom 14, which would leave both the
//! minzoom=12 and minzoom=14 charts visible simultaneously at zoom 14.
//!
//! Zone geometry is the chart's feature **bounding box** (not the COVR polygon)
//! because COVR only covers navigable water; coastline features (COALNE) run
//! along the shore and land — outside the water-only COVR but inside the chart's
//! spatial extent. Using the bbox ensures those features are correctly attributed.
//!
//! - **Polygon features**: split across zone boundaries with `BooleanOps` (zones
//!   processed coarsest-first so each piece gets the right `maxzoom`).
//! - **Line features**: intersection test — if any segment of the line crosses
//!   into a finer zone the whole feature gets the coarsest covering zone's maxzoom.
//! - **Point and other types**: containment test on the representative point.
//! - **Open territory** (no finer zone covers the geometry): no `maxzoom`.
use geo::{BooleanOps, BoundingRect, Contains, Coord, Intersects, LineString, MultiPolygon, Point, Polygon, Rect};
use geojson::{Feature, Geometry, GeometryValue, Position};
use serde_json::json;

// ── Public types ─────────────────────────────────────────────────────────────

/// A chart's effective coverage polygon tagged with the minimum zoom at which
/// that chart's features appear in tiles.
pub struct CoveredZone {
    pub poly:    MultiPolygon<f64>,
    pub minzoom: u8,
    /// Pre-computed bbox for cheap spatial pretest before full containment checks.
    pub bbox:    Rect<f64>,
}

impl CoveredZone {
    /// Returns `None` if `poly` is empty (no bbox computable).
    pub fn new(poly: MultiPolygon<f64>, minzoom: u8) -> Option<Self> {
        let bbox = poly.bounding_rect()?;
        Some(Self { poly, minzoom, bbox })
    }
}

// ── Public functions ─────────────────────────────────────────────────────────

/// Build the effective zone polygon for one chart.
///
/// The zone represents "everywhere this chart has features" — used to detect
/// that a coarser chart's feature falls in an area already covered by this
/// finer one. The COVR polygon covers only navigable water, which misses
/// coastline (COALNE) features running along the shore or on land. Using the
/// chart's overall feature bounding box as the zone correctly captures those.
///
/// NOCOVR areas are subtracted so classified/restricted sub-areas don't
/// suppress coarser features unnecessarily.
///
/// `bounds` is `[west, south, east, north]` in WGS84 degrees.
pub fn build_effective_zone(
    bounds:      [f64; 4],
    no_coverage: &[Vec<[f64; 2]>],
) -> MultiPolygon<f64> {
    let [west, south, east, north] = bounds;
    // Guard against degenerate charts with no spatial extent.
    if west >= east || south >= north {
        return MultiPolygon::new(vec![]);
    }
    let bbox_poly = Polygon::new(
        LineString::from(vec![
            Coord { x: west,  y: south },
            Coord { x: east,  y: south },
            Coord { x: east,  y: north },
            Coord { x: west,  y: north },
            Coord { x: west,  y: south },
        ]),
        vec![],
    );
    let zone = MultiPolygon::new(vec![bbox_poly]);
    if no_coverage.is_empty() {
        return zone;
    }
    zone.difference(&rings_to_multipoly(no_coverage))
}

/// Convert COVR/NOCOVR ring lists (WGS84 `[lon, lat]` pairs) to a
/// `geo::MultiPolygon`. Each ring becomes an independent outer polygon;
/// OESU COVR records are always simple, non-holed shapes.
pub fn rings_to_multipoly(rings: &[Vec<[f64; 2]>]) -> MultiPolygon<f64> {
    MultiPolygon::new(
        rings
            .iter()
            .filter(|r| !r.is_empty())
            .map(|ring| {
                Polygon::new(
                    LineString::from_iter(
                        ring.iter().map(|&[lon, lat]| Coord { x: lon, y: lat }),
                    ),
                    vec![],
                )
            })
            .collect(),
    )
}

/// Quilt a single GeoJSON feature against the already-accumulated finer-chart
/// zones. Returns 1–N features:
///
/// - Polygon features straddling a zone boundary are split into pieces, each
///   with its own `maxzoom`.
/// - Line features use a full intersection test so that long coastlines
///   spanning many small harbour COVRs are correctly handed off.
/// - Point and other types use a representative-point containment test.
/// - Features in open territory (no finer zone covers the geometry) are
///   returned unchanged.
pub fn quilt_feature(feat: Feature, covered_zones: &[CoveredZone]) -> Vec<Feature> {
    if covered_zones.is_empty() {
        return vec![feat];
    }
    if let Some(geom) = &feat.geometry {
        match &geom.value {
            // Polygon: split across zone boundaries so each piece gets its own maxzoom.
            GeometryValue::Polygon { coordinates: rings } => {
                if let Some(poly) = geojson_rings_to_geo_poly(rings) {
                    return split_and_annotate(feat, poly, covered_zones);
                }
            }
            // Line: intersection test — a long coastline can cross many harbour COVRs;
            // testing only the midpoint would miss most of them.
            GeometryValue::LineString { coordinates: pts } => {
                let line = pts_to_linestring(pts);
                if let Some(mz) = maxzoom_for_line(covered_zones, &line) {
                    let mut f = feat;
                    add_maxzoom(&mut f, mz);
                    return vec![f];
                }
                return vec![feat];
            }
            GeometryValue::MultiLineString { coordinates: lines } => {
                // Finest zone that intersects any component line wins.
                let mz = lines
                    .iter()
                    .filter_map(|pts| {
                        let line = pts_to_linestring(pts);
                        maxzoom_for_line(covered_zones, &line)
                    })
                    .max();
                if let Some(mz) = mz {
                    let mut f = feat;
                    add_maxzoom(&mut f, mz);
                    return vec![f];
                }
                return vec![feat];
            }
            _ => {}
        }
    }
    // Point, MultiPoint, and unrecognised types: representative-point test.
    if let Some(pt) = representative_point(&feat) {
        if let Some(mz) = maxzoom_for_point(covered_zones, &pt) {
            let mut f = feat;
            add_maxzoom(&mut f, mz);
            return vec![f];
        }
    }
    vec![feat]
}

/// Add a `maxzoom` entry to a feature's top-level `tippecanoe` object.
/// The object lives in `foreign_members` (not `properties`) so tippecanoe
/// picks it up for zoom gating.
pub fn add_maxzoom(feature: &mut Feature, maxzoom: u8) {
    let fm = feature.foreign_members.get_or_insert_with(serde_json::Map::new);
    let tc = fm.entry("tippecanoe").or_insert_with(|| json!({}));
    if let Some(obj) = tc.as_object_mut() {
        obj.insert("maxzoom".into(), json!(maxzoom));
    }
}

// ── Representative point ─────────────────────────────────────────────────────

/// Extract a representative `geo::Point` from a GeoJSON feature for containment
/// testing. For polygons the exterior-ring centroid is used; for lines the
/// midpoint; for multi-variants the first element.
fn representative_point(feature: &Feature) -> Option<Point<f64>> {
    let geom = feature.geometry.as_ref()?;
    Some(match &geom.value {
        GeometryValue::Point { coordinates: c } if c.len() >= 2 => {
            Point::new(c[0], c[1])
        }

        GeometryValue::MultiPoint { coordinates: pts }
            if !pts.is_empty() && pts[0].len() >= 2 =>
        {
            Point::new(pts[0][0], pts[0][1])
        }

        GeometryValue::LineString { coordinates: pts } if !pts.is_empty() => {
            let m = &pts[pts.len() / 2];
            if m.len() >= 2 { Point::new(m[0], m[1]) } else { return None; }
        }

        GeometryValue::MultiLineString { coordinates: lines }
            if !lines.is_empty() && !lines[0].is_empty() =>
        {
            let m = &lines[0][lines[0].len() / 2];
            if m.len() >= 2 { Point::new(m[0], m[1]) } else { return None; }
        }

        GeometryValue::Polygon { coordinates: rings }
            if !rings.is_empty() && !rings[0].is_empty() =>
        {
            ring_centroid(&rings[0])?
        }

        GeometryValue::MultiPolygon { coordinates: polys }
            if !polys.is_empty() && !polys[0].is_empty() && !polys[0][0].is_empty() =>
        {
            ring_centroid(&polys[0][0])?
        }

        _ => return None,
    })
}

/// Average of the `Position`s in a GeoJSON ring (exterior centroid approximation).
#[allow(clippy::cast_precision_loss)] // ring len is at most millions of vertices, fits f64
fn ring_centroid(ring: &[Position]) -> Option<Point<f64>> {
    if ring.is_empty() {
        return None;
    }
    let n = ring.len() as f64;
    let (sx, sy) = ring
        .iter()
        .fold((0.0_f64, 0.0_f64), |(sx, sy), p| (sx + p[0], sy + p[1]));
    Some(Point::new(sx / n, sy / n))
}

// ── Zone containment ─────────────────────────────────────────────────────────

/// Return the `maxzoom` value for the finest covered zone that contains `pt`,
/// or `None` if the point is in open territory (no finer chart covers it).
///
/// Return the `maxzoom` value for the coarsest covered zone that contains `pt`,
/// or `None` if the point is in open territory.
///
/// The coarsest covering zone (lowest `minzoom` > feature's own `minzoom`) wins
/// because that is the "next finer chart" this feature hands off to. Using the
/// finest zone would over-cap intermediate-scale features (e.g. a minzoom=12
/// feature would be suppressed until zoom 14 when a minzoom=15 chart covers it,
/// but the minzoom=14 chart already handles zoom 14 — so the cap should be 13).
fn maxzoom_for_point(zones: &[CoveredZone], pt: &Point<f64>) -> Option<u8> {
    let (px, py) = (pt.x(), pt.y());
    zones
        .iter()
        // Cheap bbox pretest before the full polygon containment check.
        .filter(|z| {
            z.bbox.min().x <= px && px <= z.bbox.max().x
                && z.bbox.min().y <= py && py <= z.bbox.max().y
                && z.poly.contains(pt)
        })
        .map(|z| z.minzoom)
        .min()                   // coarsest covering zone = first chart to take over
        .map(|mz| mz.saturating_sub(1))
}

/// Return the `maxzoom` for the coarsest covered zone that intersects `line`.
/// See `maxzoom_for_point` for why `min` (not `max`) is the right aggregation.
fn maxzoom_for_line(zones: &[CoveredZone], line: &LineString<f64>) -> Option<u8> {
    let line_bbox = line.bounding_rect()?;
    zones
        .iter()
        .filter(|z| rects_overlap(&line_bbox, &z.bbox) && z.poly.intersects(line))
        .map(|z| z.minzoom)
        .min()                   // coarsest covering zone = first chart to take over
        .map(|mz| mz.saturating_sub(1))
}

/// Convert GeoJSON `Position` list to a `geo::LineString`.
fn pts_to_linestring(pts: &[Position]) -> LineString<f64> {
    LineString::from_iter(
        pts.iter()
            .filter(|p| p.len() >= 2)
            .map(|p| Coord { x: p[0], y: p[1] }),
    )
}

// ── Polygon split ─────────────────────────────────────────────────────────────

/// Entry point for the polygon split path. Splits `poly` across zone boundaries
/// and reconstructs GeoJSON features from each piece.
fn split_and_annotate(
    feat:          Feature,
    poly:          Polygon<f64>,
    covered_zones: &[CoveredZone],
) -> Vec<Feature> {
    let pieces = split_polygon_for_zones(poly, covered_zones);

    // Fast path: single open-territory piece — polygon doesn't touch any zone.
    if pieces.len() == 1 && pieces[0].1.is_none() {
        return vec![feat];
    }

    pieces
        .into_iter()
        .filter_map(|(mp, maxzoom)| {
            if mp.0.is_empty() {
                return None;
            }
            let mut f = feat.clone();
            f.geometry = Some(Geometry::new(multipoly_to_geojson_value(&mp)));
            if let Some(mz) = maxzoom {
                add_maxzoom(&mut f, mz);
            }
            Some(f)
        })
        .collect()
}

/// Split `poly` across zone boundaries. Returns `(piece, maxzoom)` pairs.
///
/// Zones are processed **coarsest-first** (ascending `minzoom`). This ensures
/// that a polygon area covered by both a coarser fine chart (mz=14) and a finer
/// fine chart (mz=15) is claimed by the coarser one first (maxzoom=13), rather
/// than by the finer one (maxzoom=14). Without this, intermediate-scale polygons
/// would remain visible at the zoom where the coarser fine chart already handles
/// the area, producing two overlapping layers.
///
/// The last entry (if non-empty) has `maxzoom = None` — open territory visible
/// at all zoom levels above the chart's `minzoom`.
fn split_polygon_for_zones(
    poly:  Polygon<f64>,
    zones: &[CoveredZone],
) -> Vec<(MultiPolygon<f64>, Option<u8>)> {
    let mut remaining = MultiPolygon::new(vec![poly]);
    let mut result: Vec<(MultiPolygon<f64>, Option<u8>)> = Vec::new();

    // `zones` is in finest-first order (descending minzoom); reverse to get
    // coarsest-first (ascending minzoom) so each piece gets the right maxzoom.
    for zone in zones.iter().rev() {
        if remaining.0.is_empty() {
            break;
        }
        // Bbox pretest: skip zones with no geographic overlap.
        if let Some(rb) = remaining.bounding_rect() {
            if !rects_overlap(&rb, &zone.bbox) {
                continue;
            }
        }
        let overlap = remaining.intersection(&zone.poly);
        remaining   = remaining.difference(&zone.poly);
        if !overlap.0.is_empty() {
            result.push((overlap, Some(zone.minzoom.saturating_sub(1))));
        }
    }
    if !remaining.0.is_empty() {
        result.push((remaining, None));
    }
    result
}

/// Returns `true` when two rectangles have any overlapping area.
fn rects_overlap(a: &Rect<f64>, b: &Rect<f64>) -> bool {
    a.min().x <= b.max().x
        && a.max().x >= b.min().x
        && a.min().y <= b.max().y
        && a.max().y >= b.min().y
}

// ── GeoJSON ↔ geo conversions ─────────────────────────────────────────────────

/// Convert a GeoJSON `Polygon` ring list to a `geo::Polygon`.
/// Returns `None` if the exterior ring is absent or degenerate.
fn geojson_rings_to_geo_poly(rings: &[Vec<Position>]) -> Option<Polygon<f64>> {
    if rings.is_empty() || rings[0].is_empty() {
        return None;
    }
    let to_ls = |ring: &[Position]| -> LineString<f64> {
        LineString::from_iter(
            ring.iter()
                .filter(|p| p.len() >= 2)
                .map(|p| Coord { x: p[0], y: p[1] }),
        )
    };
    Some(Polygon::new(
        to_ls(&rings[0]),
        rings[1..].iter().map(|r| to_ls(r)).collect(),
    ))
}

/// Convert a `geo::MultiPolygon` back to a GeoJSON geometry value.
/// A single-polygon result is emitted as `Polygon` (matching the input type);
/// a multi-polygon result is emitted as `MultiPolygon`.
fn multipoly_to_geojson_value(mp: &MultiPolygon<f64>) -> GeometryValue {
    if mp.0.len() == 1 {
        GeometryValue::Polygon { coordinates: poly_to_rings(&mp.0[0]) }
    } else {
        GeometryValue::MultiPolygon {
            coordinates: mp.0.iter().map(poly_to_rings).collect(),
        }
    }
}

/// Convert one `geo::Polygon` to the GeoJSON nested-ring representation:
/// `[exterior_ring, interior_ring, …]` where each ring is `[Position, …]`.
fn poly_to_rings(poly: &Polygon<f64>) -> Vec<Vec<Position>> {
    let coords_to_ring = |ls: &geo::LineString<f64>| -> Vec<Position> {
        ls.coords().map(|c| Position::from([c.x, c.y])).collect()
    };
    let mut rings = vec![coords_to_ring(poly.exterior())];
    for interior in poly.interiors() {
        rings.push(coords_to_ring(interior));
    }
    rings
}
