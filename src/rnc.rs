//! [`RncCell`] — bulk PNG decode + caching on top of the shared
//! [`rnc_format`] parser.
//!
//! The wire format itself (binary layout, footer schema, `cover` polygon,
//! Mercator projection) lives in the `rnc-format` crate
//! — see that crate's doc comment for the full format
//! description and the coordinate-system caveats. This module only
//! adds what's specific to batch tiling: owning the file bytes, decoding
//! grid subtiles to [`image::RgbaImage`] on demand, and caching them across
//! the many [`crate::rnc_source`] render calls one cell receives.

use std::collections::HashMap;
use std::sync::{Arc, Mutex, PoisonError};

use anyhow::{Context, Result, bail};
use geo::MultiPolygon;
use image::RgbaImage;
use rnc_format::{RncFooter, RncHeader};
use s57::EditionDate;

use crate::bbox::Bbox;

/// One parsed raster cell: a `cols`×`rows` grid of independently decodable
/// PNG tiles, uniform in this format's own Mercator projection.
#[derive(Debug)]
pub struct RncCell {
    /// Cell name (from the input filename) — used as [`crate::tile_source::TileSource::source`].
    name: String,
    /// Raw `.rnc` file bytes; PNG blobs are sliced out of this on demand.
    data: Vec<u8>,
    header: RncHeader,
    /// `WGS-84` bounding box, parsed from the footer.
    bbox: Bbox,
    /// Exact `WGS-84` coverage polygon — the footer's `cover` rings when
    /// present and valid, else `bbox` as a rectangle (see
    /// [`rnc_format::RncFooter::coverage`]).
    coverage: MultiPolygon,
    /// Cell extent in this format's Mercator metres: `(xmin, ymin, xmax, ymax)`.
    /// `ymin` is the NORTH edge (smaller `y` = higher latitude).
    merc: (f64, f64, f64, f64),
    native_scale: u32,
    /// Footer `edate`, parsed; [`EditionDate::unknown`] if absent or
    /// unparseable. Consumed by `TileSource::tiebreak` to rank otherwise
    /// equally-zoom-fit tile candidates — not a general identity/ordering
    /// for this type, so `RncCell` deliberately has no `Eq`/`Ord` impl.
    edition_date: EditionDate,
    /// Lazily-decoded subtile cache, keyed by grid index `row * cols + col`.
    cache: Mutex<HashMap<u32, Arc<RgbaImage>>>,
}

impl RncCell {
    /// Parse a `.rnc` file already read into memory.
    ///
    /// # Errors
    /// Returns an error if the header or footer can't be parsed (truncated
    /// or corrupt `.rnc` file).
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // native_scale's cast is clamped immediately beforehand.
    pub fn parse(name: String, data: Vec<u8>) -> Result<Self> {
        let header = RncHeader::parse(&data).context("parsing .rnc header")?;
        let footer = RncFooter::parse(&data).context("parsing .rnc footer")?;

        let (west, south, east, north) = footer.wgs84_bbox();
        let bbox = Bbox {
            west,
            south,
            east,
            north,
        };
        let merc @ (xmin, ymin, xmax, ymax) = footer.merc_bbox();
        if !(xmax > xmin && ymax > ymin) {
            bail!("degenerate cell extent: bbox {bbox:?}");
        }

        let coverage = footer.coverage();
        let native_scale = footer.scale.round().clamp(1.0, f64::from(u32::MAX)) as u32;
        let edition_date = EditionDate::from_ddmmyyyy(&footer.edate);

        Ok(Self {
            name,
            data,
            header,
            bbox,
            coverage,
            merc,
            native_scale,
            edition_date,
            cache: Mutex::new(HashMap::new()),
        })
    }

    pub const fn edition_date(&self) -> EditionDate {
        self.edition_date
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    pub const fn bbox(&self) -> Bbox {
        self.bbox
    }

    pub fn coverage(&self) -> MultiPolygon {
        self.coverage.clone()
    }

    pub const fn native_scale(&self) -> u32 {
        self.native_scale
    }

    pub const fn cols(&self) -> u32 {
        self.header.cols
    }

    pub const fn rows(&self) -> u32 {
        self.header.rows
    }

    /// Cell extent in this format's Mercator metres: `(xmin, ymin, xmax, ymax)`.
    pub const fn merc_extent(&self) -> (f64, f64, f64, f64) {
        self.merc
    }

    /// Decode (and cache) grid tile `n = row * cols + col`.
    ///
    /// # Errors
    /// Returns an error if `n` is out of range or the PNG blob it points to
    /// fails to decode.
    pub fn subtile_image(&self, n: u32) -> Result<Arc<RgbaImage>> {
        {
            let cache = self.cache.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(img) = cache.get(&n) {
                return Ok(img.clone());
            }
        }
        let col = n % self.header.cols;
        let row = n / self.header.cols;
        let bytes = self
            .header
            .tile_bytes(&self.data, col, row)
            .context("tile index out of bounds")?;
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
    use geo::Area;
    use rnc_format::wgs84_to_rnc_merc;
    use std::io::Cursor;

    fn solid_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img = RgbaImage::from_pixel(w, h, image::Rgba(rgba));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encoding test PNG cannot fail");
        out
    }

    /// Flatten WGS-84 ring vertices into the footer's Mercator
    /// `[x0, y0, x1, y1, …]` shape.
    #[allow(clippy::tuple_array_conversions)] // [x, y] is the natural flat_map yield shape
    fn cover_ring(vertices: &[(f64, f64)]) -> Vec<f64> {
        vertices
            .iter()
            .flat_map(|&(lon, lat)| {
                let (x, y) = wgs84_to_rnc_merc(lon, lat);
                [x, y]
            })
            .collect()
    }

    /// Build a minimal valid `.rnc` buffer with a `cols`×`rows` grid of
    /// identical solid-colour PNGs and the given footer fields.
    fn build_rnc(
        cols: u32,
        rows: u32,
        bbox: (f64, f64, f64, f64),
        scale: f64,
        cover: &[Vec<f64>],
    ) -> Vec<u8> {
        let png = solid_png(4, 4, [255, 0, 0, 255]);
        let pngs = vec![png; (cols * rows) as usize];
        build_rnc_tiles(cols, rows, &pngs, bbox, scale, cover)
    }

    /// Build a minimal valid `.rnc` buffer whose grid tile `n = row * cols +
    /// col` is `pngs[n]`, with the given footer fields.
    #[allow(clippy::cast_possible_truncation)] // test PNG sizes are tiny constants
    fn build_rnc_tiles(
        cols: u32,
        rows: u32,
        pngs: &[Vec<u8>],
        (lon0, lat0, lon1, lat1): (f64, f64, f64, f64),
        scale: f64,
        cover: &[Vec<f64>],
    ) -> Vec<u8> {
        let n_tiles = cols * rows;
        assert_eq!(pngs.len(), n_tiles as usize, "one PNG per grid tile");
        let n_offsets = n_tiles + 2;
        let table_start = 16u32;
        let table_bytes = n_offsets * 4;
        let blobs_start = table_start + table_bytes;

        let mut offsets = Vec::with_capacity(n_offsets as usize);
        offsets.push(blobs_start);
        for png in pngs {
            offsets.push(offsets.last().expect("at least one offset") + png.len() as u32);
        }
        // Trailing sentinel slot (semantics unused by the reader).
        offsets.push(*offsets.last().expect("at least one offset"));

        let mut buf = vec![0u8; 8]; // ignored header
        buf.extend_from_slice(&cols.to_le_bytes());
        buf.extend_from_slice(&rows.to_le_bytes());
        for o in &offsets {
            buf.extend_from_slice(&o.to_le_bytes());
        }
        for png in pngs {
            buf.extend_from_slice(png);
        }
        let footer = serde_json::json!({
            "cover": cover,
            "lat0": lat0, "lat1": lat1, "lon0": lon0, "lon1": lon1,
            "edate": "01/01/2026", "name": "TEST", "scale": scale,
        });
        buf.extend_from_slice(serde_json::to_string(&footer).unwrap().as_bytes());
        buf
    }

    #[test]
    fn parses_header_bbox_and_scale() {
        let data = build_rnc(2, 3, (11.0, 57.0, 12.0, 58.0), 3_000_000.0, &[]);
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
    fn decodes_subtile_pixels() {
        let data = build_rnc(2, 1, (11.0, 57.0, 12.0, 58.0), 3_000_000.0, &[]);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");
        let img = cell.subtile_image(0).expect("subtile decodes");
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
        // Cached path returns the same data.
        let img2 = cell.subtile_image(0).expect("cached subtile");
        assert_eq!(img2.get_pixel(0, 0).0, [255, 0, 0, 255]);
    }

    /// Distinct colour for grid tile `n` (red channel encodes `n`).
    #[allow(clippy::cast_possible_truncation)] // test grids have < 256 tiles
    fn tile_color(n: u32) -> [u8; 4] {
        [n as u8 * 10, 100, 200, 255]
    }

    /// A 3-col x 2-row cell whose tile `n` is solid [`tile_color`]`(n)`.
    fn distinct_3x2_cell() -> RncCell {
        let (cols, rows) = (3, 2);
        let pngs: Vec<Vec<u8>> = (0..cols * rows)
            .map(|n| solid_png(4, 4, tile_color(n)))
            .collect();
        let data = build_rnc_tiles(
            cols,
            rows,
            &pngs,
            (11.0, 57.0, 12.0, 58.0),
            3_000_000.0,
            &[],
        );
        RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses")
    }

    #[test]
    fn subtile_image_indexes_row_major_across_rows() {
        let cell = distinct_3x2_cell();
        // n = row * cols + col: every index, including row 1 (n >= cols),
        // must return its own blob — not a transposed or neighbouring one.
        for n in 0..6 {
            let img = cell.subtile_image(n).expect("subtile decodes");
            assert_eq!(img.get_pixel(0, 0).0, tile_color(n), "subtile {n}");
        }
    }

    #[test]
    fn subtile_image_rejects_index_past_grid() {
        let cell = distinct_3x2_cell();
        for n in [6, 7, u32::MAX] {
            let err = cell.subtile_image(n).expect_err("index past grid must error");
            assert!(
                format!("{err:#}").contains("tile index out of bounds"),
                "n = {n}: {err:#}"
            );
        }
    }

    #[test]
    fn rejects_truncated_header() {
        let err = RncCell::parse("TEST".to_owned(), vec![0u8; 10]).unwrap_err();
        assert!(format!("{err:#}").contains("offset"));
    }

    #[test]
    fn cover_polygon_is_smaller_than_bbox_when_present() {
        // Triangle over the SW/SE/NE corners of the 11-12E, 57-58N bbox —
        // omits the NW corner, so its area is exactly half the rectangle's.
        let bbox = Bbox {
            west: 11.0,
            south: 57.0,
            east: 12.0,
            north: 58.0,
        };
        let triangle = cover_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.0)]);
        let data = build_rnc(1, 1, (11.0, 57.0, 12.0, 58.0), 3_000_000.0, &[triangle]);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");

        let bbox_area = geo::Polygon::from(bbox).unsigned_area();
        let cover_area = cell.coverage().unsigned_area();
        assert!(
            (cover_area - bbox_area / 2.0).abs() < bbox_area * 1e-6,
            "triangle cover ({cover_area}) should be half the bbox rectangle ({bbox_area})"
        );
    }

    #[test]
    fn falls_back_to_bbox_rectangle_when_cover_is_degenerate() {
        // A two-point "ring" (4 numbers) has no area — must be dropped.
        let degenerate = vec![1.0, 2.0, 3.0, 4.0];
        let data = build_rnc(1, 1, (11.0, 57.0, 12.0, 58.0), 3_000_000.0, &[degenerate]);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");
        let bbox_area = geo::Polygon::from(cell.bbox()).unsigned_area();
        assert!((cell.coverage().unsigned_area() - bbox_area).abs() < 1e-9);
    }
}
