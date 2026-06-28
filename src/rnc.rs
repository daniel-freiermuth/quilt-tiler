//! Parser for `.rnc` files.
//!
//! Each `.rnc` file is fully self-describing — no sidecar `index.json` is needed:
//!
//! ```text
//! offset                size            content
//! ------                ----            -------
//! 0                     8 B             header (ignored)
//! 8                     4 B             u32 LE: k = grid columns
//! 12                    4 B             u32 LE: l = grid rows
//! 16                    (k*l + 2) * 4   offset table: u32 LE byte offsets;
//!                                       tile n occupies bytes [h[n], h[n+1])
//! 16 + (k*l+2)*4        variable        k * l raw PNG image blobs, packed
//! ...                                   trailing JSON footer (marker `{"cover"`)
//! ```
//!
//! The footer carries the cell's `WGS-84` bounds and native scale — exactly what
//! [`crate::tile_source::TileSource`] needs, with no separate catalog lookup.
//!
//! Tiles are laid out on a grid uniform in its own spherical Mercator
//! projection (not `WGS-84`), so resampling a tile (see `crate::rnc_source`)
//! has to reproject through that same projection — see [`wgs84_to_nv_merc`].

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result, bail};
use image::RgbaImage;

use crate::bbox::Bbox;

/// Internal Mercator sphere circumference, in metres.
///
/// A full-circumference spherical Mercator distinct from standard `WebMercator`
/// (`EPSG:3857`, which uses the WGS-84 equatorial radius). [`wgs84_to_nv_merc`]
/// is the closed-form inverse of that crate's forward formula, using the same
/// base so the two stay self-consistent.
const MERC_SPHERE_BASE: f64 = 4.003_017_861_858_939_4e7;

/// `WGS-84` (lon, lat) in degrees → Special Mercator (x, y) in metres.
///
/// Inverse of `merc_to_wgs84`:
/// `lon = x/BASE*360 - 180`, `lat = atan(sinh(π - 2π·y/BASE))·180/π`.
pub fn wgs84_to_nv_merc(lon: f64, lat: f64) -> (f64, f64) {
    let x = (lon + 180.0) / 360.0 * MERC_SPHERE_BASE;
    let y = lat
        .to_radians()
        .tan()
        .asinh()
        .mul_add(-1.0, std::f64::consts::PI)
        * MERC_SPHERE_BASE
        / (2.0 * std::f64::consts::PI);
    (x, y)
}

/// Trailing JSON footer embedded in every `.rnc` file (marker `{"cover"`).
#[derive(serde::Deserialize, Debug)]
struct RncFooter {
    lat0: f64,
    lat1: f64,
    lon0: f64,
    lon1: f64,
    /// Scale denominator, e.g. `3_000_000.0`.
    scale: f64,
}

/// Locate and parse the trailing JSON footer.
fn parse_footer(data: &[u8]) -> Result<RncFooter> {
    const MARKER: &[u8] = b"{\"cover\"";
    let pos = data
        .windows(MARKER.len())
        .enumerate()
        .rev()
        .find(|(_, w)| *w == MARKER)
        .map(|(i, _)| i)
        .context("RNC footer marker `{\"cover\"` not found")?;
    serde_json::from_slice(&data[pos..]).context("failed to parse RNC footer JSON")
}

/// Read a little-endian `u32` at byte offset `at`, bounds-checked.
fn read_u32_le(data: &[u8], at: usize) -> Result<u32> {
    let b = data
        .get(at..at + 4)
        .with_context(|| format!("offset {at} out of bounds (file is {} bytes)", data.len()))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// One parsed raster cell: a `cols`×`rows` grid of independently
/// decodable PNG tiles, uniform in its own Mercator projection.
#[derive(Debug)]
pub struct RncCell {
    /// Cell name (from the input filename) — used as [`TileSource::source`].
    name: String,
    /// Raw `.rnc` file bytes; PNG blobs are sliced out of this on demand.
    data: Vec<u8>,
    /// Byte offsets into `data` for each grid tile; tile `n` occupies
    /// `data[offsets[n]..offsets[n + 1]]`.
    offsets: Vec<u32>,
    cols: u32,
    rows: u32,
    /// `WGS-84` bounding box, parsed from the footer.
    bbox: Bbox,
    /// Cell extent in special Mercator metres: `(xmin, ymin, xmax, ymax)`.
    /// `ymin` is the NORTH edge (smaller `y` = higher latitude).
    merc: (f64, f64, f64, f64),
    native_scale: u32,
    /// Lazily-decoded subtile cache, keyed by grid index `row * cols + col`.
    cache: Mutex<HashMap<u32, Arc<RgbaImage>>>,
}

impl RncCell {
    /// Parse a `.rnc` file already read into memory.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Offset-table sizes and the native-scale cast are both bounds-checked
    // or clamped immediately before the cast (see comments below).
    pub fn parse(name: String, data: Vec<u8>) -> Result<Self> {
        let cols = read_u32_le(&data, 8).context("reading grid column count")?;
        let rows = read_u32_le(&data, 12).context("reading grid row count")?;
        let n_tiles = u64::from(cols) * u64::from(rows);
        if n_tiles == 0 {
            bail!("empty tile grid ({cols}x{rows})");
        }
        let n_offsets = n_tiles.checked_add(2).context("tile grid too large")?;
        let table_bytes = n_offsets.checked_mul(4).context("offset table too large")?;
        let table_end = 16u64
            .checked_add(table_bytes)
            .context("offset table overflows file size")?;
        if (data.len() as u64) < table_end {
            bail!(
                "truncated offset table: file is {} bytes, need at least {table_end}",
                data.len()
            );
        }

        let mut offsets = Vec::with_capacity(n_offsets as usize);
        for i in 0..n_offsets {
            #[allow(clippy::cast_possible_truncation)] // i*4 + 16 << table_end <= data.len()
            let at = (16 + i * 4) as usize;
            offsets.push(read_u32_le(&data, at)?);
        }
        for w in offsets.windows(2) {
            if w[1] < w[0] {
                bail!("offset table is not monotonically non-decreasing");
            }
        }
        #[allow(clippy::cast_possible_truncation)] // n_tiles < offsets.len(), checked above
        let last_tile_end = offsets[n_tiles as usize] as usize;
        if last_tile_end > data.len() {
            bail!("offset table points past end of file");
        }

        let footer = parse_footer(&data).context("parsing .rnc footer")?;

        let (west, east) = if footer.lon0 <= footer.lon1 {
            (footer.lon0, footer.lon1)
        } else {
            (footer.lon1, footer.lon0)
        };
        let (south, north) = if footer.lat0 <= footer.lat1 {
            (footer.lat0, footer.lat1)
        } else {
            (footer.lat1, footer.lat0)
        };
        let bbox = Bbox {
            west,
            south,
            east,
            north,
        };

        let (xmin, y_north) = wgs84_to_nv_merc(west, north);
        let (xmax, y_south) = wgs84_to_nv_merc(east, south);
        let (ymin, ymax) = (y_north.min(y_south), y_north.max(y_south));
        if !(xmax > xmin && ymax > ymin) {
            bail!("degenerate cell extent: bbox {bbox:?}");
        }

        let native_scale = footer.scale.round().clamp(1.0, f64::from(u32::MAX)) as u32;

        Ok(Self {
            name,
            data,
            offsets,
            cols,
            rows,
            bbox,
            merc: (xmin, ymin, xmax, ymax),
            native_scale,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn bbox(&self) -> Bbox {
        self.bbox
    }

    pub const fn native_scale(&self) -> u32 {
        self.native_scale
    }

    pub const fn cols(&self) -> u32 {
        self.cols
    }

    pub const fn rows(&self) -> u32 {
        self.rows
    }

    /// Cell extent in special Mercator metres: `(xmin, ymin, xmax, ymax)`.
    pub const fn merc_extent(&self) -> (f64, f64, f64, f64) {
        self.merc
    }

    /// Decode (and cache) grid tile `n = row * cols + col`.
    pub fn subtile_image(&self, n: u32) -> Result<Arc<RgbaImage>> {
        {
            let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(img) = cache.get(&n) {
                return Ok(img.clone());
            }
        }
        let lo = *self
            .offsets
            .get(n as usize)
            .context("tile index out of bounds")? as usize;
        let hi = *self
            .offsets
            .get(n as usize + 1)
            .context("tile index out of bounds")? as usize;
        let bytes = self
            .data
            .get(lo..hi)
            .context("tile blob out of file bounds")?;
        let img = image::load_from_memory_with_format(bytes, image::ImageFormat::Png)
            .with_context(|| format!("{}: decoding subtile {n}", self.name))?
            .to_rgba8();
        let img = Arc::new(img);
        let img = {
            let mut cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            cache.entry(n).or_insert_with(|| img).clone()
        };
        Ok(img)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn solid_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img = RgbaImage::from_pixel(w, h, image::Rgba(rgba));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encoding test PNG cannot fail");
        out
    }

    /// Build a minimal valid `.rnc` buffer with a `cols`×`rows` grid of
    /// identical solid-colour PNGs and the given footer fields.
    #[allow(clippy::cast_possible_truncation)] // test PNG sizes are tiny constants
    fn build_rnc(
        cols: u32,
        rows: u32,
        lon0: f64,
        lat0: f64,
        lon1: f64,
        lat1: f64,
        scale: f64,
    ) -> Vec<u8> {
        let png = solid_png(4, 4, [255, 0, 0, 255]);
        let n_tiles = cols * rows;
        let n_offsets = n_tiles + 2;
        let table_start = 16u32;
        let table_bytes = n_offsets * 4;
        let blobs_start = table_start + table_bytes;

        let mut offsets = Vec::with_capacity(n_offsets as usize);
        for i in 0..=n_tiles {
            offsets.push(blobs_start + i * png.len() as u32);
        }
        // Trailing sentinel slot (semantics unused by the reader).
        offsets.push(*offsets.last().expect("at least one offset"));

        let mut buf = vec![0u8; 8]; // ignored header
        buf.extend_from_slice(&cols.to_le_bytes());
        buf.extend_from_slice(&rows.to_le_bytes());
        for o in &offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        for _ in 0..n_tiles {
            buf.extend_from_slice(&png);
        }
        let footer = serde_json::json!({
            "cover": [],
            "lat0": lat0, "lat1": lat1, "lon0": lon0, "lon1": lon1,
            "edate": "01/01/2026", "name": "TEST", "scale": scale,
        });
        buf.extend_from_slice(serde_json::to_string(&footer).unwrap().as_bytes());
        buf
    }

    #[test]
    fn parses_header_bbox_and_scale() {
        let data = build_rnc(2, 3, 11.0, 57.0, 12.0, 58.0, 3_000_000.0);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");
        assert_eq!(cell.cols(), 2);
        assert_eq!(cell.rows(), 3);
        assert_eq!(cell.native_scale(), 3_000_000);
        let b = cell.bbox();
        assert!((b.west - 11.0).abs() < 1e-9);
        assert!((b.east - 12.0).abs() < 1e-9);
        assert!((b.south - 57.0).abs() < 1e-9);
        assert!((b.north - 58.0).abs() < 1e-9);
    }

    #[test]
    fn normalizes_swapped_footer_corners() {
        // lat0/lon0 given as the "max" corner — parser must not assume order.
        let data = build_rnc(1, 1, 12.0, 58.0, 11.0, 57.0, 1_000_000.0);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");
        let b = cell.bbox();
        assert!((b.west - 11.0).abs() < 1e-9);
        assert!((b.east - 12.0).abs() < 1e-9);
    }

    #[test]
    fn decodes_subtile_pixels() {
        let data = build_rnc(2, 1, 11.0, 57.0, 12.0, 58.0, 3_000_000.0);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");
        let img = cell.subtile_image(0).expect("subtile decodes");
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
        // Cached path returns the same data.
        let img2 = cell.subtile_image(0).expect("cached subtile");
        assert_eq!(img2.get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    #[test]
    fn rejects_truncated_header() {
        let err = RncCell::parse("TEST".to_owned(), vec![0u8; 10]).unwrap_err();
        assert!(format!("{err:#}").contains("offset"));
    }

    #[test]
    fn rejects_zero_size_grid() {
        let mut data = vec![0u8; 16];
        data[8..12].copy_from_slice(&0u32.to_le_bytes());
        data[12..16].copy_from_slice(&5u32.to_le_bytes());
        let err = RncCell::parse("TEST".to_owned(), data).unwrap_err();
        assert!(err.to_string().contains("empty tile grid"));
    }

    #[test]
    fn rejects_missing_footer() {
        let mut data = build_rnc(1, 1, 11.0, 57.0, 12.0, 58.0, 3_000_000.0);
        // Truncate right after the PNG blob, dropping the JSON footer.
        let png_len = solid_png(4, 4, [255, 0, 0, 255]).len();
        let blobs_start = 16 + (1u32 + 2) * 4;
        data.truncate((blobs_start as usize) + png_len);
        let err = RncCell::parse("TEST".to_owned(), data).unwrap_err();
        assert!(err.to_string().contains("footer"));
    }

    #[test]
    fn mercator_round_trip_is_monotonic_in_latitude() {
        // Higher latitude must map to a smaller y.
        let (_, y_low) = wgs84_to_nv_merc(11.0, 50.0);
        let (_, y_high) = wgs84_to_nv_merc(11.0, 60.0);
        assert!(y_high < y_low);
        // Equator/prime-meridian sits at the centre of the sphere base.
        let (x0, y0) = wgs84_to_nv_merc(0.0, 0.0);
        assert!((x0 - MERC_SPHERE_BASE / 2.0).abs() < 1e-6);
        assert!((y0 - MERC_SPHERE_BASE / 2.0).abs() < 1e-6);
    }
}
