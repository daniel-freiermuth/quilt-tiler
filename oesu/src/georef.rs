use geo::{Coord, coord};

/// Simple Mercator → WGS84 conversion, matching `OpenCPN`'s `fromSM()`.
/// Coordinates in VET/VCT tables are stored as f32 easting/northing in metres,
/// projected relative to the cell centroid (`ref_lat`, `ref_lon`).
///
/// Returns the WGS84 coordinate as `Coord { x: longitude, y: latitude }`.
pub fn from_sm(east: f64, north: f64, ref_lat: f64, ref_lon: f64) -> Coord {
    const WGS84_A: f64 = 6_378_137.0;
    use std::f64::consts::PI;

    let lon = east / WGS84_A.to_radians() + ref_lon;

    let lat_r = ref_lat.to_radians();
    // Inverse Mercator: undo the log(tan()) forward projection
    let lat = 2.0f64
        .mul_add(
            ((north / WGS84_A) + (PI / 4.0 + lat_r / 2.0).tan().ln())
                .exp()
                .atan(),
            -(PI / 2.0),
        )
        .to_degrees();

    coord![x:lon, y:lat]
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const WGS84_A: f64 = 6_378_137.0;

    /// Forward Simple Mercator projection (WGS-84 → easting/northing in metres),
    /// inverse of `from_sm`.  Used only as a test helper for round-trip checks.
    fn to_sm(lon: f64, lat: f64, ref_lat: f64, ref_lon: f64) -> (f64, f64) {
        let east = (lon - ref_lon) * WGS84_A.to_radians();

        let lat_r = lat.to_radians();
        let ref_lat_r = ref_lat.to_radians();
        let north =
            WGS84_A * ((PI / 4.0 + lat_r / 2.0).tan().ln() - (PI / 4.0 + ref_lat_r / 2.0).tan().ln());

        (east, north)
    }

    // ── boundary: identity at centroid ─────────────────────────────────

    #[test]
    fn origin_at_equator_returns_origin() {
        let c = from_sm(0.0, 0.0, 0.0, 0.0);
        assert!((c.x).abs() < 1e-12, "lon: {}", c.x);
        assert!((c.y).abs() < 1e-12, "lat: {}", c.y);
    }

    #[test]
    fn zero_offset_returns_centroid() {
        for (ref_lat, ref_lon) in [(57.7, 11.8), (-33.9, 151.2), (0.0, -90.0)] {
            let c = from_sm(0.0, 0.0, ref_lat, ref_lon);
            assert!(
                (c.x - ref_lon).abs() < 1e-9,
                "ref=({ref_lat},{ref_lon}) lon: expected {ref_lon}, got {}",
                c.x
            );
            assert!(
                (c.y - ref_lat).abs() < 1e-9,
                "ref=({ref_lat},{ref_lon}) lat: expected {ref_lat}, got {}",
                c.y
            );
        }
    }

    // ── boundary: high latitude ────────────────────────────────────────

    #[test]
    fn high_latitude_round_trip() {
        // At 70° the cos(lat) distortion factor is ~0.34; numerical precision matters.
        let ref_lat = 70.0;
        let ref_lon = 25.0;
        for (target_lon, target_lat) in [(25.5, 70.3), (24.0, 69.5), (26.0, 71.0)] {
            let (east, north) = to_sm(target_lon, target_lat, ref_lat, ref_lon);
            let c = from_sm(east, north, ref_lat, ref_lon);
            assert!(
                (c.x - target_lon).abs() < 1e-9,
                "lon: expected {target_lon}, got {}",
                c.x
            );
            assert!(
                (c.y - target_lat).abs() < 1e-9,
                "lat: expected {target_lat}, got {}",
                c.y
            );
        }
    }

    // ── boundary: large easting/northing at cell edge ──────────────────

    #[test]
    fn large_offsets_round_trip() {
        // Typical chart cells can span several km; test with 50 km offsets.
        let ref_lat = 45.0;
        let ref_lon = -123.0;
        let offsets_m: &[f64] = &[50_000.0, -50_000.0, 100_000.0];
        for &east in offsets_m {
            for &north in offsets_m {
                let c = from_sm(east, north, ref_lat, ref_lon);
                let (e2, n2) = to_sm(c.x, c.y, ref_lat, ref_lon);
                assert!(
                    (e2 - east).abs() < 1e-4,
                    "east round-trip: expected {east}, got {e2}"
                );
                assert!(
                    (n2 - north).abs() < 1e-4,
                    "north round-trip: expected {north}, got {n2}"
                );
            }
        }
    }

    // ── error path: NaN / infinite inputs ──────────────────────────────

    #[test]
    fn nan_easting_propagates() {
        let c = from_sm(f64::NAN, 0.0, 0.0, 0.0);
        assert!(c.x.is_nan(), "expected NaN lon, got {}", c.x);
    }

    #[test]
    fn nan_northing_propagates() {
        let c = from_sm(0.0, f64::NAN, 0.0, 0.0);
        assert!(c.y.is_nan(), "expected NaN lat, got {}", c.y);
    }

    #[test]
    fn infinite_easting_propagates() {
        let c = from_sm(f64::INFINITY, 0.0, 0.0, 0.0);
        assert!(c.x.is_infinite(), "expected infinite lon, got {}", c.x);
    }

    #[test]
    fn infinite_northing_propagates() {
        let c = from_sm(0.0, f64::INFINITY, 0.0, 0.0);
        // exp(inf) = inf, atan(inf) = PI/2, so lat = 2*(PI/2) - PI/2 = PI/2 → 90°
        assert!(c.y.is_finite(), "lat should saturate, got {}", c.y);
        assert!(
            (c.y - 90.0).abs() < 1e-9,
            "expected ~90° for +inf northing, got {}",
            c.y
        );
    }

    // ── round-trip: SM forward → from_sm inverse ───────────────────────

    #[test]
    fn round_trip_various_locations() {
        let cases: &[(f64, f64, f64, f64)] = &[
            // (target_lon, target_lat, ref_lon, ref_lat)
            (11.8, 57.7, 11.0, 57.0),   // Gothenburg
            (-179.0, -60.0, -178.0, -59.0), // southern hemisphere near date line
            (0.0, 0.0, 1.0, 1.0),        // near equator
            (179.0, 80.0, 178.0, 79.0),  // Arctic
            (-73.9, 40.7, -74.0, 40.5),  // New York
        ];
        for &(target_lon, target_lat, ref_lon, ref_lat) in cases {
            let (east, north) = to_sm(target_lon, target_lat, ref_lat, ref_lon);
            let c = from_sm(east, north, ref_lat, ref_lon);
            assert!(
                (c.x - target_lon).abs() < 1e-9,
                "({target_lon},{target_lat}) ref ({ref_lon},{ref_lat}): lon {}, expected {target_lon}",
                c.x
            );
            assert!(
                (c.y - target_lat).abs() < 1e-9,
                "({target_lon},{target_lat}) ref ({ref_lon},{ref_lat}): lat {}, expected {target_lat}",
                c.y
            );
        }
    }

    // ── southern hemisphere ────────────────────────────────────────────

    #[test]
    fn negative_ref_lat_identity() {
        let c = from_sm(0.0, 0.0, -45.0, 170.0);
        assert!(
            (c.x - 170.0).abs() < 1e-9,
            "lon: expected 170.0, got {}",
            c.x
        );
        assert!(
            (c.y - (-45.0)).abs() < 1e-9,
            "lat: expected -45.0, got {}",
            c.y
        );
    }
}
