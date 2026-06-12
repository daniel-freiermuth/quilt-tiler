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
//! The `covered_zones` list accumulates each chart's effective COVR as we go.
//! When processing a coarser chart its features are tested against all already-
//! seen finer zones:
//!
//! - **Polygon features**: split across zone boundaries with `BooleanOps` so
//!   each piece gets the exact `maxzoom` for its geographic sub-area.
//! - **All other types**: representative-point containment test (centroid/
//!   midpoint); the whole feature gets a single `maxzoom`.
//! - **Open territory** (no finer chart covers the point): no `maxzoom` — the
//!   feature is visible at all zoom levels above its `minzoom`.

use geo::{BooleanOps, BoundingRect, Contains, Coord, LineString, MultiPolygon, Point, Polygon, Rect};
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

/// Build the effective coverage polygon for one chart:
///   `effective = union(COVR polygons) − union(NOCOVR polygons)`
pub fn build_effective_covr(
    coverage:    &[Vec<[f64; 2]>],
    no_coverage: &[Vec<[f64; 2]>],
) -> MultiPolygon<f64> {
    if coverage.is_empty() {
        return MultiPolygon::new(vec![]);
    }
    let covr = rings_to_multipoly(coverage);
    if no_coverage.is_empty() {
        return covr;
    }
    covr.difference(&rings_to_multipoly(no_coverage))
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
/// - All other geometry types use a representative-point containment test.
/// - Features whose representative point is in open territory are returned
///   unchanged (their `minzoom` was already stamped by `cell_to_geojson`).
pub fn quilt_feature(feat: Feature, covered_zones: &[CoveredZone]) -> Vec<Feature> {
    // Polygon: full BooleanOps split across zone boundaries.
    if !covered_zones.is_empty() {
        if let Some(geom) = &feat.geometry {
            if let GeometryValue::Polygon { coordinates: rings } = &geom.value {
                if let Some(poly) = geojson_rings_to_geo_poly(rings) {
                    return split_and_annotate(feat, poly, covered_zones);
                }
            }
        }
    }

    // Non-polygon (or no zones yet): centroid/midpoint containment test.
    if let Some(pt) = representative_point(&feat) {
        if let Some(mz) = maxzoom_for_point(covered_zones, &pt) {
            let mut f = feat;
            add_maxzoom(&mut f, mz);
            return vec![f];
        }
    }
    vec![feat]
}

/// Add a `maxzoom` entry to a feature's existing `tippecanoe` property.
/// `minzoom` is already present from `cell_to_geojson`.
pub fn add_maxzoom(feature: &mut Feature, maxzoom: u8) {
    let props = feature.properties.get_or_insert_with(serde_json::Map::new);
    let tc = props.entry("tippecanoe").or_insert_with(|| json!({}));
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
/// When multiple zones overlap the point, the finest one (highest `minzoom`)
/// wins — it is the last chart to hand off ownership at that location.
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
        .max()
        .map(|mz| mz.saturating_sub(1))
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

/// Split `poly` across all zone boundaries. Returns `(clipped_piece, maxzoom)`
/// pairs. Zones are processed finest-first (highest `minzoom` first) so the
/// finest chart's boundary takes precedence at intersections.
///
/// The last entry (if non-empty) always has `maxzoom = None` — it is the
/// open-territory remainder visible at all zoom levels above the chart's minzoom.
fn split_polygon_for_zones(
    poly:  Polygon<f64>,
    zones: &[CoveredZone],
) -> Vec<(MultiPolygon<f64>, Option<u8>)> {
    let mut remaining = MultiPolygon::new(vec![poly]);
    let mut result: Vec<(MultiPolygon<f64>, Option<u8>)> = Vec::new();

    // `zones` is already in finest-first order (descending minzoom) because
    // `main` inserts charts in finest→coarsest order. No re-sort needed.
    for zone in zones.iter() {
        if remaining.0.is_empty() {
            break;
        }
        // Bbox pretest: skip zones with no geographic overlap — avoids almost
        // all BooleanOps calls for charts whose COVR doesn't touch this polygon.
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
