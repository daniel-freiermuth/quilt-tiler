//! Shared parser for raster-cell `.rnc` files.
//!
//! Canonical home for the reverse-engineered `.rnc` wire format.
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
//! Image decoding is deliberately **not** included here: `quilt-tiler`
//! needs bulk decode + disjoint-coverage compositing for batch tiling. Both
//! are project-specific; only the format itself is shared.
//!
//! # Coordinate systems
//!
//! `.rnc` tiles are laid out on a grid uniform in this format's own
//! spherical Mercator projection — a different sphere radius than standard
//! `WebMercator`/`EPSG:3857` (mean-radius sphere vs. WGS-84 equatorial-radius
//! sphere, ~0.1% circumference difference). [`wgs84_to_rnc_merc`] /
//! [`rnc_merc_to_wgs84`] convert to/from it; never mix its metre values with
//! standard `WebMercator` metres directly — always round-trip through
//! `WGS-84` as the common intermediate.
//!
//! Per the format's own documented tile-extent formula, grid cells are
//! uniform **in this projection's metres, not in `WGS-84` degrees** — see
//! [`grid_cell_wgs84_bounds`] and [`locate_grid_cell`].

use anyhow::{Context, Result, bail};
use geo::{Area, Coord, LineString, MultiPolygon, Polygon, coord};
use serde::Deserialize;

/// This format's internal Mercator sphere circumference, in metres.
pub const MERC_SPHERE_BASE: f64 = 4.003_017_861_858_939_4e7;

/// `WGS-84` (lon, lat) in degrees → this format's Mercator (x, y) in metres.
#[must_use]
pub fn wgs84_to_rnc_merc(lon: f64, lat: f64) -> (f64, f64) {
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

/// This format's Mercator (x, y) in metres → `WGS-84` (lon, lat) in degrees.
///
/// Inverse of [`wgs84_to_rnc_merc`].
#[must_use]
pub fn rnc_merc_to_wgs84(x: f64, y: f64) -> (f64, f64) {
    let lon = (x / MERC_SPHERE_BASE).mul_add(360.0, -180.0);
    let inner = (2.0 * std::f64::consts::PI).mul_add(-(y / MERC_SPHERE_BASE), std::f64::consts::PI);
    let lat = inner.sinh().atan().to_degrees();
    (lon, lat)
}

/// `WGS-84` bounding box of grid cell `(col, row)`.
///
/// Cell `(col, row)` lives in a `cols`×`rows` grid spanning `merc_bbox =
/// (xmin, ymin, xmax, ymax)` — uniform in this format's Mercator metres, per
/// the format's documented tile-extent formula. `ymin` is the NORTH edge
/// (smaller `y` = higher latitude).
///
/// Returns `(west, south, east, north)`.
#[must_use]
#[allow(clippy::many_single_char_names)]
pub fn grid_cell_wgs84_bounds(
    merc_bbox: (f64, f64, f64, f64),
    cols: u32,
    rows: u32,
    col: u32,
    row: u32,
) -> (f64, f64, f64, f64) {
    let (xmin, ymin, xmax, ymax) = merc_bbox;
    let cell_w = (xmax - xmin) / f64::from(cols);
    let cell_h = (ymax - ymin) / f64::from(rows);
    let x0 = f64::from(col).mul_add(cell_w, xmin);
    let y0 = f64::from(row).mul_add(cell_h, ymin);
    let (west, north) = rnc_merc_to_wgs84(x0, y0);
    let (east, south) = rnc_merc_to_wgs84(x0 + cell_w, y0 + cell_h);
    (west, south, east, north)
}

/// Map a `WGS-84` point to its `(col, row)` grid index and fractional
/// position within that grid cell — uniform in this format's Mercator metres,
/// matching [`grid_cell_wgs84_bounds`]'s convention.
///
/// `col`/`row` are clamped into `[0, cols)`/`[0, rows)`; fractions are
/// clamped into `[0, 1)`. Always returns a result — callers that need to
/// know whether the point was actually inside `merc_bbox` should bbox-test
/// before calling.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names
)]
pub fn locate_grid_cell(
    lon: f64,
    lat: f64,
    merc_bbox: (f64, f64, f64, f64),
    cols: u32,
    rows: u32,
) -> (u32, u32, f64, f64) {
    let (xmin, ymin, xmax, ymax) = merc_bbox;
    let (x, y) = wgs84_to_rnc_merc(lon, lat);
    let u = ((x - xmin) / (xmax - xmin)).clamp(0.0, 0.999_999_9);
    let v = ((y - ymin) / (ymax - ymin)).clamp(0.0, 0.999_999_9);
    let col = (u * f64::from(cols)).floor() as u32;
    let row = (v * f64::from(rows)).floor() as u32;
    let fx = u * f64::from(cols) - f64::from(col);
    let fy = v * f64::from(rows) - f64::from(row);
    (col, row, fx, fy)
}

/// Read a little-endian `u32` at byte offset `at`, bounds-checked.
fn read_u32_le(data: &[u8], at: usize) -> Result<u32> {
    let b = data
        .get(at..at + 4)
        .with_context(|| format!("offset {at} out of bounds (file is {} bytes)", data.len()))?;
    Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// Parsed `.rnc` binary header: grid dimensions and the tile offset table.
#[derive(Debug, Clone)]
pub struct RncHeader {
    pub cols: u32,
    pub rows: u32,
    /// Byte offsets into the file for each grid tile, length `cols*rows + 2`
    /// (the trailing sentinel beyond `cols*rows + 1` has no defined meaning —
    /// only `offsets[n]..offsets[n + 1]` for `n < cols*rows` is used).
    pub offsets: Vec<u32>,
}

impl RncHeader {
    /// Parse the 16-byte fixed header plus the variable-length offset table.
    ///
    /// Does not touch the footer or any PNG blob — only enough to know where
    /// they are.
    ///
    /// # Errors
    /// Returns an error when the header or offset table is truncated,
    /// malformed, or describes an empty/oversized grid.
    #[allow(clippy::cast_possible_truncation)] // table_end is bounds-checked below
    pub fn parse(data: &[u8]) -> Result<Self> {
        let cols = read_u32_le(data, 8).context("reading grid column count")?;
        let rows = read_u32_le(data, 12).context("reading grid row count")?;
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
            let at = (16 + i * 4) as usize;
            offsets.push(read_u32_le(data, at)?);
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

        Ok(Self {
            cols,
            rows,
            offsets,
        })
    }

    /// Byte range `[start, end)` of grid tile `(col, row)` within the
    /// `.rnc` file's bytes. `n = row * cols + col`.
    #[must_use]
    pub fn tile_range(&self, col: u32, row: u32) -> Option<(usize, usize)> {
        if col >= self.cols || row >= self.rows {
            return None;
        }
        let n = (row * self.cols + col) as usize;
        let lo = *self.offsets.get(n)?;
        let hi = *self.offsets.get(n + 1)?;
        Some((lo as usize, hi as usize))
    }

    /// Slice out grid tile `(col, row)`'s raw PNG bytes from `data` (the
    /// full `.rnc` file this header was parsed from).
    #[must_use]
    pub fn tile_bytes<'a>(&self, data: &'a [u8], col: u32, row: u32) -> Option<&'a [u8]> {
        let (lo, hi) = self.tile_range(col, row)?;
        data.get(lo..hi)
    }
}

/// Trailing JSON footer embedded in every `.rnc` file (marker `{"cover"`).
#[derive(Deserialize, Debug, Clone)]
pub struct RncFooter {
    pub lat0: f64,
    pub lat1: f64,
    pub lon0: f64,
    pub lon1: f64,
    /// Scale denominator, e.g. `3_000_000.0`.
    pub scale: f64,
    /// Edition date, e.g. `"01/06/2026"` (DD/MM/YYYY). Cosmetic — not used
    /// for any computation here.
    #[serde(default)]
    pub edate: String,
    /// Human-readable cell name (distinct from the `.rnc` filename, which
    /// is usually a code like `H5207_6`). Cosmetic — not used for any
    /// computation here.
    #[serde(default)]
    pub name: String,
    /// Coverage polygon, this format's Mercator metres.
    ///
    /// Real files store this as a **flat** single ring
    /// `[x0, y0, x1, y1, …]`. No official schema, so [`deserialize_cover`]
    /// also accepts a nested `[[x0,y0,…], …]` shape defensively, normalizing
    /// either way to "list of rings". Empty when the field is absent.
    #[serde(default, deserialize_with = "deserialize_cover")]
    pub cover: Vec<Vec<f64>>,
}

/// Normalize `cover` into "list of rings" regardless of whether the source
/// JSON nests rings (`[[x0,y0,…], [x1,y1,…]]`) or gives one flat ring
/// (`[x0,y0,x1,y1,…]`) directly — see [`RncFooter::cover`].
fn deserialize_cover<'de, D>(deserializer: D) -> std::result::Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let value = serde_json::Value::deserialize(deserializer)?;
    let Some(items) = value.as_array() else {
        return Ok(Vec::new());
    };
    let is_nested = items.first().is_some_and(serde_json::Value::is_array);
    if is_nested {
        Ok(items
            .iter()
            .map(|ring| {
                ring.as_array()
                    .map(|r| r.iter().filter_map(serde_json::Value::as_f64).collect())
                    .unwrap_or_default()
            })
            .collect())
    } else {
        let ring: Vec<f64> = items.iter().filter_map(serde_json::Value::as_f64).collect();
        Ok(if ring.is_empty() {
            Vec::new()
        } else {
            vec![ring]
        })
    }
}

impl RncFooter {
    /// Locate (marker `{"cover"`, last occurrence) and parse the trailing
    /// JSON footer out of a full `.rnc` file's bytes.
    ///
    /// # Errors
    /// Returns an error when the footer marker is missing or the JSON after
    /// it doesn't match [`RncFooter`]'s shape.
    pub fn parse(data: &[u8]) -> Result<Self> {
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

    /// `WGS-84` bounding box, corners normalized so `west <= east` and
    /// `south <= north` (the footer doesn't guarantee corner order).
    ///
    /// Returns `(west, south, east, north)`.
    #[must_use]
    pub fn wgs84_bbox(&self) -> (f64, f64, f64, f64) {
        let (west, east) = if self.lon0 <= self.lon1 {
            (self.lon0, self.lon1)
        } else {
            (self.lon1, self.lon0)
        };
        let (south, north) = if self.lat0 <= self.lat1 {
            (self.lat0, self.lat1)
        } else {
            (self.lat1, self.lat0)
        };
        (west, south, east, north)
    }

    /// This format's Mercator bounding box derived from [`Self::wgs84_bbox`].
    ///
    /// Returns `(xmin, ymin, xmax, ymax)`; `ymin` is the NORTH edge.
    #[must_use]
    pub fn merc_bbox(&self) -> (f64, f64, f64, f64) {
        let (west, south, east, north) = self.wgs84_bbox();
        let (xmin, y_north) = wgs84_to_rnc_merc(west, north);
        let (xmax, y_south) = wgs84_to_rnc_merc(east, south);
        (xmin, y_north.min(y_south), xmax, y_north.max(y_south))
    }

    /// Exact `WGS-84` coverage polygon: [`Self::cover`]'s rings when present
    /// and valid, else [`Self::wgs84_bbox`] as a rectangle.
    ///
    /// Each `cover` ring becomes an independent exterior (no holes — `cover`
    /// has no NOCOVR-equivalent concept); degenerate rings (fewer than 3
    /// points, zero area) are dropped.
    #[must_use]
    pub fn coverage(&self) -> MultiPolygon {
        cover_to_multipolygon(&self.cover).unwrap_or_else(|| {
            let (west, south, east, north) = self.wgs84_bbox();
            MultiPolygon::new(vec![Polygon::new(
                LineString::new(vec![
                    coord! { x: west, y: south },
                    coord! { x: east, y: south },
                    coord! { x: east, y: north },
                    coord! { x: west, y: north },
                    coord! { x: west, y: south },
                ]),
                vec![],
            )])
        })
    }
}

/// Convert footer `cover` rings (flat format-native Mercator `x, y` pairs)
/// into a `WGS-84` [`MultiPolygon`]. Returns `None` when nothing usable
/// remains, so the caller can fall back to the bbox rectangle.
fn cover_to_multipolygon(rings: &[Vec<f64>]) -> Option<MultiPolygon> {
    let polygons: Vec<Polygon> = rings
        .iter()
        .filter_map(|ring| {
            if ring.len() < 6 || ring.len() % 2 != 0 {
                return None; // need at least 3 points (6 numbers)
            }
            let mut coords: Vec<Coord> = ring
                .chunks_exact(2)
                .map(|p| {
                    let (lon, lat) = rnc_merc_to_wgs84(p[0], p[1]);
                    coord! { x: lon, y: lat }
                })
                .collect();
            if coords.first() != coords.last() {
                coords.push(coords[0]);
            }
            let poly = Polygon::new(LineString::new(coords), vec![]);
            (poly.unsigned_area() > 0.0).then_some(poly)
        })
        .collect();
    (!polygons.is_empty()).then(|| MultiPolygon::new(polygons))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::tuple_array_conversions)] // [x, y] is the natural flat_map yield shape
    fn flat_ring(vertices: &[(f64, f64)]) -> Vec<f64> {
        vertices
            .iter()
            .flat_map(|&(lon, lat)| {
                let (x, y) = wgs84_to_rnc_merc(lon, lat);
                [x, y]
            })
            .collect()
    }

    fn footer_json(
        lon0: f64,
        lat0: f64,
        lon1: f64,
        lat1: f64,
        scale: f64,
        cover: &serde_json::Value,
    ) -> Vec<u8> {
        let v = serde_json::json!({
            "cover": cover,
            "lat0": lat0, "lat1": lat1, "lon0": lon0, "lon1": lon1,
            "edate": "01/01/2026", "name": "TEST", "scale": scale,
        });
        serde_json::to_vec(&v).unwrap()
    }

    #[test]
    fn mercator_projection_round_trips() {
        for (lon, lat) in [(11.8, 57.7), (-179.0, -60.0), (0.0, 0.0), (179.0, 80.0)] {
            let (x, y) = wgs84_to_rnc_merc(lon, lat);
            let (lon2, lat2) = rnc_merc_to_wgs84(x, y);
            assert!((lon - lon2).abs() < 1e-9);
            assert!((lat - lat2).abs() < 1e-9);
        }
    }

    #[test]
    fn matches_real_rnc_footer_values() {
        let (x0, y0) = wgs84_to_rnc_merc(11.456_114_757_055_763, 58.252_470_735_147_824);
        let (x1, y1) = wgs84_to_rnc_merc(11.468_593_385_364_44, 58.239_311_601_695_76);
        assert!((x0 - 21_288_951.0).abs() < 1.0);
        assert!((y0 - 12_003_521.0).abs() < 1.0);
        assert!((x1 - 21_290_338.0).abs() < 1.0);
        assert!((y1 - 12_006_301.0).abs() < 1.0);
    }

    #[test]
    fn header_parses_dims_and_offsets() {
        // 2x1 grid, two 10-byte "tiles".
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&2u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        let offsets = [36u32, 46, 56, 56];
        for o in offsets {
            data.extend_from_slice(&o.to_le_bytes());
        }
        data.extend_from_slice(&[0u8; 24]);
        let header = RncHeader::parse(&data).expect("valid header parses");
        assert_eq!(header.cols, 2);
        assert_eq!(header.rows, 1);
        assert_eq!(header.tile_range(0, 0), Some((36, 46)));
        assert_eq!(header.tile_range(1, 0), Some((46, 56)));
        assert_eq!(header.tile_range(2, 0), None);
    }

    #[test]
    fn header_rejects_truncated_table() {
        let data = vec![0u8; 10];
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(format!("{err:#}").contains("offset"));
    }

    #[test]
    fn footer_parses_flat_cover_ring() {
        // The real on-disk shape (not the docs' claimed nested shape).
        let data = footer_json(
            11.0,
            57.0,
            12.0,
            58.0,
            3_000_000.0,
            &serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
        );
        let footer = RncFooter::parse(&data).expect("flat cover parses");
        assert_eq!(footer.cover, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
    }

    #[test]
    fn footer_parses_nested_cover_rings() {
        let data = footer_json(
            11.0,
            57.0,
            12.0,
            58.0,
            3_000_000.0,
            &serde_json::json!([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
        );
        let footer = RncFooter::parse(&data).expect("nested cover parses");
        assert_eq!(footer.cover, vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]);
    }

    #[test]
    fn footer_normalizes_swapped_corners() {
        let data = footer_json(12.0, 58.0, 11.0, 57.0, 1_000_000.0, &serde_json::json!([]));
        let footer = RncFooter::parse(&data).expect("valid footer parses");
        let (west, south, east, north) = footer.wgs84_bbox();
        assert!((west - 11.0).abs() < 1e-9);
        assert!((east - 12.0).abs() < 1e-9);
        assert!((south - 57.0).abs() < 1e-9);
        assert!((north - 58.0).abs() < 1e-9);
    }

    #[test]
    fn coverage_uses_cover_polygon_when_present() {
        let bbox_corners = (11.0, 57.0, 12.0, 58.0);
        let triangle = flat_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.0)]);
        let data = footer_json(
            bbox_corners.0,
            bbox_corners.1,
            bbox_corners.2,
            bbox_corners.3,
            3_000_000.0,
            &serde_json::json!(triangle),
        );
        let footer = RncFooter::parse(&data).expect("valid footer parses");

        let bbox_rect = Polygon::new(
            LineString::new(vec![
                coord! { x: 11.0, y: 57.0 },
                coord! { x: 12.0, y: 57.0 },
                coord! { x: 12.0, y: 58.0 },
                coord! { x: 11.0, y: 58.0 },
                coord! { x: 11.0, y: 57.0 },
            ]),
            vec![],
        );
        let bbox_area = bbox_rect.unsigned_area();
        let cover_area = footer.coverage().unsigned_area();
        assert!((cover_area - bbox_area / 2.0).abs() < bbox_area * 1e-6);
    }

    #[test]
    fn coverage_falls_back_to_bbox_when_cover_absent() {
        let data = footer_json(11.0, 57.0, 12.0, 58.0, 3_000_000.0, &serde_json::json!([]));
        let footer = RncFooter::parse(&data).expect("valid footer parses");
        let (west, south, east, north) = footer.wgs84_bbox();
        let bbox_area = (east - west) * (north - south);
        assert!((footer.coverage().unsigned_area() - bbox_area).abs() < 1e-9);
    }

    #[test]
    fn grid_cell_bounds_and_locate_are_consistent() {
        let (xmin, ymin) = wgs84_to_rnc_merc(11.0, 58.0);
        let (xmax, ymax) = wgs84_to_rnc_merc(12.0, 57.0);
        let merc_bbox = (xmin, ymin, xmax, ymax);
        let (cols, rows) = (3, 2);

        // Build the midpoint of grid cell (1, 1) in Mercator space (not by
        // degree-averaging its WGS-84 corners — that's a different point,
        // since the projection is nonlinear in latitude).
        let cell_w = (xmax - xmin) / f64::from(cols);
        let cell_h = (ymax - ymin) / f64::from(rows);
        let x_mid = 1.5_f64.mul_add(cell_w, xmin);
        let y_mid = 1.5_f64.mul_add(cell_h, ymin);
        let (lon, lat) = rnc_merc_to_wgs84(x_mid, y_mid);

        let (col, row, fx, fy) = locate_grid_cell(lon, lat, merc_bbox, cols, rows);
        assert_eq!((col, row), (1, 1));
        assert!((fx - 0.5).abs() < 1e-6);
        assert!((fy - 0.5).abs() < 1e-6);
    }
}
