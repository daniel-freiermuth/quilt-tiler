//! Fixtures shared by the end-to-end tests that run the built binaries.
//!
//! Each integration-test crate compiles this module separately and uses only
//! part of it, hence the `dead_code` allowance.
#![allow(dead_code)]

use std::path::{Path, PathBuf};

/// A per-test scratch directory under the system temp dir, removed on drop
/// so a failing assertion does not leak files.
pub struct Scratch(PathBuf);

impl Scratch {
    pub fn new(test: &str) -> Self {
        let dir =
            std::env::temp_dir().join(format!("quilt-tiler-e2e-{}-{test}", std::process::id()));
        // A leftover from an earlier crashed run with the same pid.
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).expect("creating scratch dir");
        Self(dir)
    }

    pub fn path(&self) -> &Path {
        &self.0
    }

    pub fn join(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }

    pub fn write(&self, name: &str, data: &[u8]) -> PathBuf {
        let path = self.0.join(name);
        std::fs::write(&path, data).expect("writing scratch file");
        path
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Geographic box `[west, south, east, north]` in WGS84 degrees.
pub type Box4 = [f64; 4];

/// Coarse raster cell: 1:3 000 000, native (and detail) zoom 7.
pub const COARSE_SCALE: f64 = 3_000_000.0;
pub const COARSE_ZOOM: u8 = 7;
pub const COARSE_BOX: Box4 = [10.0, 56.0, 11.0, 56.5];

/// Fine raster cell inside the coarse one: 1:50 000, native zoom 13.
pub const FINE_SCALE: f64 = 50_000.0;
pub const FINE_ZOOM: u8 = 13;
pub const FINE_BOX: Box4 = [10.25, 56.25, 10.5, 56.375];

/// Minimal single-tile `.rnc`: one solid-colour `8×8` PNG over `bbox`
/// (a 1×1 subtile grid), followed by the JSON footer.
pub fn build_rnc(bbox: Box4, color: [u8; 4], scale: f64, name: &str) -> Vec<u8> {
    let img = image::RgbaImage::from_pixel(8, 8, image::Rgba(color));
    let mut png = Vec::new();
    img.write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png)
        .expect("encoding test PNG");

    // 8 bytes of header, cols, rows, then 1 tile + 2 trailing offsets.
    let table_end: u32 = 16 + 3 * 4;
    let png_end = table_end + u32::try_from(png.len()).expect("tiny test PNG");
    let mut buf = vec![0u8; 8];
    buf.extend_from_slice(&1u32.to_le_bytes()); // cols
    buf.extend_from_slice(&1u32.to_le_bytes()); // rows
    for offset in [table_end, png_end, png_end] {
        buf.extend_from_slice(&offset.to_le_bytes());
    }
    buf.extend_from_slice(&png);
    let [west, south, east, north] = bbox;
    let footer = serde_json::json!({
        "cover": [],
        "lat0": south, "lat1": north,
        "lon0": west, "lon1": east,
        "edate": "01/01/2026", "name": name, "scale": scale,
    });
    buf.extend_from_slice(footer.to_string().as_bytes());
    buf
}

/// Write the coarse and fine raster fixtures into `dir`; returns their paths.
pub fn write_raster_fixtures(dir: &Scratch) -> [PathBuf; 2] {
    [
        dir.write(
            "coarse.rnc",
            &build_rnc(COARSE_BOX, [200, 30, 30, 255], COARSE_SCALE, "COARSE"),
        ),
        dir.write(
            "fine.rnc",
            &build_rnc(FINE_BOX, [30, 200, 30, 255], FINE_SCALE, "FINE"),
        ),
    ]
}

/// Slippy-map `(x, y)` of the tile at zoom `z` containing `(lon, lat)`.
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss
)] // in-range test coordinates
pub fn tile_containing(lon: f64, lat: f64, z: u8) -> (u32, u32) {
    let n = f64::from(1u32 << z);
    let x = ((lon + 180.0) / 360.0 * n).floor();
    let lat_rad = lat.to_radians();
    let y = ((1.0 - lat_rad.tan().asinh() / std::f64::consts::PI) / 2.0 * n).floor();
    (x as u32, y as u32)
}
