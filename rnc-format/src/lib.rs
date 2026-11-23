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
#![deny(clippy::indexing_slicing)]
#![cfg_attr(
    test,
    allow(
        clippy::unwrap_used,
        clippy::expect_used,
        clippy::panic,
        clippy::indexing_slicing
    )
)]

use geo::{Area, Coord, LineString, MultiPolygon, Polygon, coord};
use serde::Deserialize;

/// Why a `.rnc` file could not be parsed.
///
/// Every variant means the file itself is unusable (truncated, corrupt, or
/// not an `.rnc` file); none is transient. Recovery: skip the file — retrying
/// the same bytes always fails the same way.
#[derive(Debug, thiserror::Error)]
pub enum RncError {
    /// A fixed-position `u32` lies beyond the end of the file (typically a
    /// file shorter than the 16-byte header).
    #[error("offset {offset} out of bounds (file is {len} bytes)")]
    OutOfBounds { offset: usize, len: usize },
    /// The header declares a grid with zero columns or rows.
    #[error("empty tile grid ({cols}x{rows})")]
    EmptyGrid { cols: u32, rows: u32 },
    /// The declared grid's offset table size overflows `u64`.
    #[error("tile grid too large ({cols}x{rows})")]
    GridTooLarge { cols: u32, rows: u32 },
    /// The file ends before the offset table the header declares.
    #[error("truncated offset table: file is {len} bytes, need at least {need}")]
    TruncatedOffsetTable { len: usize, need: u64 },
    /// Tile offsets decrease somewhere, so tile byte ranges would be inverted.
    #[error("offset table is not monotonically non-decreasing")]
    OffsetsNotMonotonic,
    /// The last tile ends past the end of the file.
    #[error("offset table points past end of file (offset {offset}, file is {len} bytes)")]
    OffsetPastEof { offset: usize, len: usize },
    /// No `{"cover"` footer marker anywhere in the file.
    #[error("RNC footer marker `{{\"cover\"` not found")]
    FooterMissing,
    /// The footer JSON is malformed or doesn't match [`RncFooter`]'s shape
    /// (including a malformed `cover` array).
    #[error("failed to parse RNC footer JSON: {0}")]
    FooterJson(#[from] serde_json::Error),
    /// A `cover` ring isn't an even count of at least 6 numbers.
    #[error("`cover` ring {ring} has {numbers} numbers; need an even count of at least 6")]
    InvalidCoverRing { ring: usize, numbers: usize },
    /// A `cover` ring encloses no area.
    #[error("`cover` ring {ring} has zero area")]
    ZeroAreaCoverRing { ring: usize },
    /// A `cover` vertex lies outside the footer's own Mercator bbox
    /// `(xmin, ymin, xmax, ymax)`, beyond [`COVER_BBOX_TOLERANCE`].
    #[error(
        "`cover` ring {ring} vertex ({x}, {y}) lies outside the footer bbox \
         {merc_bbox:?} (format Mercator metres)"
    )]
    CoverVertexOutsideBbox {
        ring: usize,
        x: f64,
        y: f64,
        merc_bbox: (f64, f64, f64, f64),
    },
}

/// Result of `.rnc` parsing; the error defaults to [`RncError`].
pub type Result<T, E = RncError> = std::result::Result<T, E>;

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
    // Derive cell index and in-cell fraction from the same product, so
    // `fx`/`fy` stay in [0, 1). A fused `mul_add` would round differently
    // from `floor` and can go slightly negative at cell edges.
    let (sx, sy) = (u * f64::from(cols), v * f64::from(rows));
    let col = sx.floor() as u32;
    let row = sy.floor() as u32;
    let fx = sx - sx.floor();
    let fy = sy - sy.floor();
    (col, row, fx, fy)
}

/// Read a little-endian `u32` at byte offset `at`, bounds-checked.
fn read_u32_le(data: &[u8], at: usize) -> Result<u32> {
    let b = data
        .get(at..)
        .and_then(<[u8]>::first_chunk::<4>)
        .ok_or(RncError::OutOfBounds {
            offset: at,
            len: data.len(),
        })?;
    Ok(u32::from_le_bytes(*b))
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
        let cols = read_u32_le(data, 8)?;
        let rows = read_u32_le(data, 12)?;
        let n_tiles = u64::from(cols) * u64::from(rows);
        if n_tiles == 0 {
            return Err(RncError::EmptyGrid { cols, rows });
        }
        let table_end = n_tiles
            .checked_add(2)
            .and_then(|n| n.checked_mul(4))
            .and_then(|bytes| bytes.checked_add(16))
            .ok_or(RncError::GridTooLarge { cols, rows })?;
        let n_offsets = n_tiles + 2; // cannot overflow: checked above
        if (data.len() as u64) < table_end {
            return Err(RncError::TruncatedOffsetTable {
                len: data.len(),
                need: table_end,
            });
        }

        let mut offsets = Vec::with_capacity(n_offsets as usize);
        for i in 0..n_offsets {
            let at = (16 + i * 4) as usize;
            offsets.push(read_u32_le(data, at)?);
        }
        if !offsets.is_sorted() {
            return Err(RncError::OffsetsNotMonotonic);
        }
        // n_tiles < offsets.len() == n_tiles + 2, so this index always exists.
        #[allow(clippy::cast_possible_truncation)]
        let last_tile_end = offsets
            .get(n_tiles as usize)
            .map_or(usize::MAX, |&o| o as usize);
        if last_tile_end > data.len() {
            return Err(RncError::OffsetPastEof {
                offset: last_tile_end,
                len: data.len(),
            });
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
    /// Edition date, e.g. `"01/06/2026"` (DD/MM/YYYY). Parsed downstream
    /// (`quilt_tiler::rnc::RncCell`) into a comparable date used to break
    /// tile-selection ties — see `quilt_tiler::tiles::Quilt::render_tile`.
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
    /// either way to "list of rings". Empty when the field is absent or `[]`.
    /// Every ring holds an even number (≥ 6) of numbers; anything else is a
    /// parse error.
    #[serde(default, deserialize_with = "deserialize_cover")]
    pub cover: Vec<Vec<f64>>,
}

/// On-disk shapes accepted for [`RncFooter::cover`].
#[derive(Deserialize)]
#[serde(untagged)]
enum CoverShape {
    /// `[x0, y0, x1, y1, …]` — one ring (the real on-disk shape).
    Flat(Vec<f64>),
    /// `[[x0, y0, …], [x0, y0, …]]` — list of rings.
    Nested(Vec<Vec<f64>>),
}

/// Normalize `cover` into "list of rings" regardless of whether the source
/// JSON nests rings (`[[x0,y0,…], [x1,y1,…]]`) or gives one flat ring
/// (`[x0,y0,x1,y1,…]`) directly — see [`RncFooter::cover`].
///
/// Rejects any other shape, non-numeric elements, and rings that aren't
/// an even count of at least 6 numbers (3 `x, y` points).
fn deserialize_cover<'de, D>(deserializer: D) -> std::result::Result<Vec<Vec<f64>>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let rings = match CoverShape::deserialize(deserializer).map_err(|_| {
        serde::de::Error::custom(
            "`cover` must be a flat `[x0, y0, …]` or nested `[[x0, y0, …], …]` array of numbers",
        )
    })? {
        CoverShape::Flat(ring) if ring.is_empty() => Vec::new(),
        CoverShape::Flat(ring) => vec![ring],
        CoverShape::Nested(rings) => rings,
    };
    if let Some((i, ring)) = rings
        .iter()
        .enumerate()
        .find(|(_, ring)| ring.len() < 6 || ring.len() % 2 != 0)
    {
        return Err(serde::de::Error::custom(format!(
            "`cover` ring {i} has {} numbers; need an even count of at least 6 (3 x, y points)",
            ring.len()
        )));
    }
    Ok(rings)
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
        let footer = data
            .windows(MARKER.len())
            .rposition(|w| w == MARKER)
            .and_then(|pos| data.get(pos..))
            .ok_or(RncError::FooterMissing)?;
        Ok(serde_json::from_slice(footer)?)
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

    /// Exact `WGS-84` coverage polygon: [`Self::cover`]'s rings when present,
    /// else [`Self::wgs84_bbox`] as a rectangle.
    ///
    /// Each `cover` ring becomes an independent exterior (no holes — `cover`
    /// has no NOCOVR-equivalent concept).
    ///
    /// # Errors
    /// Returns an error when any `cover` ring is unusable (odd length, fewer
    /// than 3 points, zero area, or a vertex outside [`Self::merc_bbox`]
    /// beyond [`COVER_BBOX_TOLERANCE`]) instead of silently substituting the
    /// bbox.
    pub fn coverage(&self) -> Result<MultiPolygon> {
        if self.cover.is_empty() {
            let (west, south, east, north) = self.wgs84_bbox();
            return Ok(MultiPolygon::new(vec![Polygon::new(
                LineString::new(vec![
                    coord! { x: west, y: south },
                    coord! { x: east, y: south },
                    coord! { x: east, y: north },
                    coord! { x: west, y: north },
                    coord! { x: west, y: south },
                ]),
                vec![],
            )]));
        }
        cover_to_multipolygon(&self.cover, self.merc_bbox())
    }
}

/// Allowed `cover` overshoot past [`RncFooter::merc_bbox`], per axis, as a
/// fraction of the footer extent's span.
///
/// Real footers store `cover` in whole metres with vertices on the bbox
/// edges, so they overshoot it by rounding; anything beyond this is not
/// this cell's cover.
pub const COVER_BBOX_TOLERANCE: f64 = 0.01;

/// Convert footer `cover` rings (flat format-native Mercator `x, y` pairs)
/// into a `WGS-84` [`MultiPolygon`], failing on the first unusable ring.
///
/// `merc_bbox` is the footer's own extent `(xmin, ymin, xmax, ymax)`; every
/// vertex must lie within it, widened by [`COVER_BBOX_TOLERANCE`].
fn cover_to_multipolygon(
    rings: &[Vec<f64>],
    merc_bbox: (f64, f64, f64, f64),
) -> Result<MultiPolygon> {
    let (xmin, ymin, xmax, ymax) = merc_bbox;
    let x_slack = (xmax - xmin) * COVER_BBOX_TOLERANCE;
    let y_slack = (ymax - ymin) * COVER_BBOX_TOLERANCE;
    let x_range = (xmin - x_slack)..=(xmax + x_slack);
    let y_range = (ymin - y_slack)..=(ymax + y_slack);
    let polygons = rings
        .iter()
        .enumerate()
        .map(|(i, ring)| {
            let (points, rest) = ring.as_chunks::<2>();
            if points.len() < 3 || !rest.is_empty() {
                return Err(RncError::InvalidCoverRing {
                    ring: i,
                    numbers: ring.len(),
                });
            }
            if let Some(&[x, y]) = points
                .iter()
                .find(|&&[x, y]| !(x_range.contains(&x) && y_range.contains(&y)))
            {
                return Err(RncError::CoverVertexOutsideBbox {
                    ring: i,
                    x,
                    y,
                    merc_bbox,
                });
            }
            let mut coords: Vec<Coord> = points
                .iter()
                .map(|&[x, y]| {
                    let (lon, lat) = rnc_merc_to_wgs84(x, y);
                    coord! { x: lon, y: lat }
                })
                .collect();
            if let Some(&first) = coords.first()
                && coords.last() != Some(&first)
            {
                coords.push(first);
            }
            let poly = Polygon::new(LineString::new(coords), vec![]);
            // The `else` branch also rejects a NaN area.
            if poly.unsigned_area() > 0.0 {
                Ok(poly)
            } else {
                Err(RncError::ZeroAreaCoverRing { ring: i })
            }
        })
        .collect::<Result<Vec<Polygon>>>()?;
    Ok(MultiPolygon::new(polygons))
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
    fn header_tile_range_indexes_row_major_across_rows() {
        // 3 cols x 2 rows; tile n = row * cols + col is n + 1 bytes long, so
        // every tile has a distinct range and a row/col mix-up is visible.
        let (cols, rows) = (3u32, 2u32);
        let blobs_start = 16 + (cols * rows + 2) * 4;
        let mut offsets = vec![blobs_start];
        for len in 1..=cols * rows {
            offsets.push(offsets.last().unwrap() + len);
        }
        offsets.push(*offsets.last().unwrap()); // trailing sentinel
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&cols.to_le_bytes());
        data.extend_from_slice(&rows.to_le_bytes());
        for o in &offsets {
            data.extend_from_slice(&o.to_le_bytes());
        }
        data.resize(*offsets.last().unwrap() as usize, 0);
        let header = RncHeader::parse(&data).expect("valid header parses");

        // Row 0.
        assert_eq!(header.tile_range(0, 0), Some((48, 49)));
        assert_eq!(header.tile_range(2, 0), Some((51, 54)));
        // Row 1 starts at n = cols, not n = rows or n = 1.
        assert_eq!(header.tile_range(0, 1), Some((54, 58)));
        assert_eq!(header.tile_range(1, 1), Some((58, 63)));
        assert_eq!(header.tile_range(2, 1), Some((63, 69)));
        // Out of range on either axis.
        assert_eq!(header.tile_range(3, 0), None);
        assert_eq!(header.tile_range(0, 2), None);
    }

    #[test]
    fn header_rejects_truncated_table() {
        let data = vec![0u8; 10];
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(
            matches!(err, RncError::OutOfBounds { offset: 8, len: 10 }),
            "{err:?}"
        );
    }

    /// Build a minimal `.rnc` byte buffer for a `cols×rows` grid with the
    /// given offset table entries (must be `cols*rows + 2` values).
    fn make_rnc_bytes(cols: u32, rows: u32, offsets: &[u32], payload_len: usize) -> Vec<u8> {
        let mut data = vec![0u8; 8]; // 8-byte ignored header
        data.extend_from_slice(&cols.to_le_bytes());
        data.extend_from_slice(&rows.to_le_bytes());
        for &o in offsets {
            data.extend_from_slice(&o.to_le_bytes());
        }
        data.resize(data.len() + payload_len, 0xAB);
        data
    }

    #[test]
    fn header_rejects_zero_cols() {
        // cols=0, rows=5 → n_tiles=0 → "empty tile grid"
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&5u32.to_le_bytes());
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(
            matches!(err, RncError::EmptyGrid { cols: 0, rows: 5 }),
            "{err:?}"
        );
    }

    #[test]
    fn header_rejects_zero_rows() {
        // cols=3, rows=0 → n_tiles=0 → "empty tile grid"
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&3u32.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(
            matches!(err, RncError::EmptyGrid { cols: 3, rows: 0 }),
            "{err:?}"
        );
    }

    #[test]
    fn header_rejects_overflow_grid() {
        // cols=u32::MAX, rows=u32::MAX → n_tiles overflows checked_mul(4)
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        data.extend_from_slice(&u32::MAX.to_le_bytes());
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(matches!(err, RncError::GridTooLarge { .. }), "{err:?}");
    }

    #[test]
    fn header_rejects_non_monotonic_offsets() {
        // 1x1 grid: offsets [100, 90, 100] — second < first → non-monotonic
        let offsets = [100u32, 90, 100];
        let data = make_rnc_bytes(1, 1, &offsets, 80);
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(matches!(err, RncError::OffsetsNotMonotonic), "{err:?}");
    }

    #[test]
    fn header_rejects_offsets_past_eof() {
        // 1x1 grid: valid offsets but last tile end points beyond data length.
        // Header = 8 + 4 + 4 = 16, offset table = 3 * 4 = 12 → table ends at 28.
        // Set offsets to [28, 128, 128] but provide only 20 bytes of payload
        // (total 48 bytes), so offset 128 > 48.
        let offsets = [28u32, 128, 128];
        let data = make_rnc_bytes(1, 1, &offsets, 20);
        let err = RncHeader::parse(&data).unwrap_err();
        assert!(
            matches!(
                err,
                RncError::OffsetPastEof {
                    offset: 128,
                    len: 48
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn header_parses_minimal_1x1() {
        // 1x1 grid: 3 offsets, one tile of 10 bytes.
        // Header = 16 bytes, offset table = 3 * 4 = 12 bytes → table ends at 28.
        let offsets = [28u32, 38, 38];
        let data = make_rnc_bytes(1, 1, &offsets, 10);
        let header = RncHeader::parse(&data).expect("minimal 1x1 should parse");
        assert_eq!(header.cols, 1);
        assert_eq!(header.rows, 1);
        assert_eq!(header.tile_range(0, 0), Some((28, 38)));
        assert_eq!(header.tile_range(1, 0), None);
        assert_eq!(header.tile_range(0, 1), None);
    }

    #[test]
    fn header_offset_table_length_check_is_exact() {
        // 1x1 grid: the 3-entry table ends at byte 28, as does the file.
        let exact = make_rnc_bytes(1, 1, &[28, 28, 28], 0);
        let header = RncHeader::parse(&exact).expect("table ending the file parses");
        assert_eq!(header.tile_range(0, 0), Some((28, 28)));

        let err = RncHeader::parse(&exact[..27]).unwrap_err();
        assert!(
            matches!(err, RncError::TruncatedOffsetTable { len: 27, need: 28 }),
            "{err:?}"
        );
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
        let cover_area = footer.coverage().expect("valid cover").unsigned_area();
        assert!((cover_area - bbox_area / 2.0).abs() < bbox_area * 1e-6);
    }

    #[test]
    fn coverage_falls_back_to_bbox_when_cover_absent() {
        let data = footer_json(11.0, 57.0, 12.0, 58.0, 3_000_000.0, &serde_json::json!([]));
        let footer = RncFooter::parse(&data).expect("valid footer parses");
        let (west, south, east, north) = footer.wgs84_bbox();
        let bbox_area = (east - west) * (north - south);
        assert!(
            (footer.coverage().expect("bbox fallback").unsigned_area() - bbox_area).abs() < 1e-9
        );
    }

    #[test]
    fn footer_rejects_malformed_cover() {
        for cover in [
            serde_json::json!("oops"),
            serde_json::json!(null),
            serde_json::json!([1.0, null, 2.0, 3.0, 4.0, 5.0, 6.0]),
            serde_json::json!([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], "ring"]),
            serde_json::json!([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]),
            serde_json::json!([1.0, 2.0, 3.0, 4.0]),
            serde_json::json!([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0], []]),
        ] {
            let data = footer_json(11.0, 57.0, 12.0, 58.0, 3_000_000.0, &cover);
            let err = RncFooter::parse(&data).unwrap_err();
            assert!(
                matches!(err, RncError::FooterJson(_)),
                "cover {cover} must be rejected as malformed JSON: {err:?}"
            );
        }
    }

    #[test]
    fn coverage_rejects_zero_area_ring() {
        let collinear = flat_ring(&[(11.0, 57.0), (11.5, 57.0), (12.0, 57.0)]);
        let data = footer_json(
            11.0,
            57.0,
            12.0,
            58.0,
            3_000_000.0,
            &serde_json::json!(collinear),
        );
        let footer = RncFooter::parse(&data).expect("well-shaped cover parses");
        assert!(matches!(
            footer.coverage(),
            Err(RncError::ZeroAreaCoverRing { ring: 0 })
        ));
    }

    /// `cover` is a public field, so `coverage` re-checks ring shape for
    /// footers not built by `parse`.
    #[test]
    fn coverage_rejects_misshapen_ring() {
        let data = footer_json(11.0, 57.0, 12.0, 58.0, 3_000_000.0, &serde_json::json!([]));
        let mut footer = RncFooter::parse(&data).expect("valid footer parses");
        let triangle = flat_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.0)]);
        for (ring, numbers) in [(vec![1.0, 2.0, 3.0, 4.0], 4), (vec![1.0; 7], 7)] {
            footer.cover = vec![triangle.clone(), ring];
            assert!(
                matches!(
                    footer.coverage(),
                    Err(RncError::InvalidCoverRing { ring: 1, numbers: n }) if n == numbers
                ),
                "{numbers} numbers: {:?}",
                footer.coverage()
            );
        }
    }

    #[test]
    fn coverage_rejects_cover_outside_footer_bbox() {
        let (x0, y0) = wgs84_to_rnc_merc(11.0, 57.0);
        for cover in [
            // Finite but absurd Mercator metres.
            vec![0.0, 0.0, 1e300, 0.0, 0.0, 1e300],
            // Degrees stored where metres belong (wrong units).
            vec![11.0, 57.0, 12.0, 57.0, 12.0, 58.0],
            // Plausible metres, but one vertex 0.5° past each side in turn.
            flat_ring(&[(11.0, 57.0), (12.5, 57.0), (12.0, 58.0)]),
            flat_ring(&[(10.5, 57.0), (12.0, 57.0), (12.0, 58.0)]),
            flat_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.5)]),
            flat_ring(&[(11.0, 56.5), (12.0, 57.0), (12.0, 58.0)]),
            // One vertex far outside, the rest in the bbox.
            vec![x0, y0, x0 + 1e7, y0, x0, y0 - 1e5],
        ] {
            let data = footer_json(
                11.0,
                57.0,
                12.0,
                58.0,
                3_000_000.0,
                &serde_json::json!(cover),
            );
            let footer = RncFooter::parse(&data).expect("well-shaped cover parses");
            let err = footer
                .coverage()
                .expect_err(&format!("cover {cover:?} must be rejected"));
            assert!(
                matches!(err, RncError::CoverVertexOutsideBbox { ring: 0, .. }),
                "unexpected: {err:?}"
            );
        }
    }

    #[test]
    fn coverage_accepts_cover_rounded_to_whole_metres_on_bbox_edges() {
        // Real footers store `cover` as whole metres, so vertices on the
        // bbox edges land up to 0.5 m outside it.
        let ring: Vec<f64> = flat_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.0), (11.0, 58.0)])
            .into_iter()
            .map(f64::round)
            .collect();
        let data = footer_json(
            11.0,
            57.0,
            12.0,
            58.0,
            3_000_000.0,
            &serde_json::json!(ring),
        );
        let footer = RncFooter::parse(&data).expect("valid footer parses");
        footer
            .coverage()
            .expect("cover on the bbox edges is accepted");
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
