use geo_types::{Coord, coord};

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
