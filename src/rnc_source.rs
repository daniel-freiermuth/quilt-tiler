//! [`TileSource`] implementation for [`crate::rnc::RncCell`] raster cells → PNG tiles.
//!
//! Unlike the vector path ([`crate::s57_source`]), a raster tile is built by
//! resampling: each destination pixel is projected back to `WGS-84`, tested
//! against this item's exact contribution polygon for the tile (the real
//! `cover` shape from the footer, not just its bounding box — see
//! `crate::rnc::cover_to_multipolygon`), and — if inside — reprojected into
//! the source cell's own (non-`WGS-84`) grid to pick a source pixel.
//! Nearest-neighbour sampling; see [`render`](TileSource::render) for why
//! that's sufficient for a first pass.

use anyhow::{Context, Result};
use geo::{Contains, MultiPolygon, Point};
use image::RgbaImage;
use martin_tile_utils::webmercator_to_wgs84;
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::rnc::{RncCell, wgs84_to_nv_merc};
use crate::tile_geom::TileGeom;
use crate::tile_source::TileSource;

/// Output raster tile size in pixels. `256` is the universal default for XYZ
/// raster tiles (`MapLibre` raster sources default `tileSize` to `256`).
pub const TILE_PX: u32 = 256;

impl TileSource for RncCell {
    /// A flat RGBA8 buffer, `TILE_PX * TILE_PX * 4` bytes, row-major from the
    /// top-left (north-west) corner. Transparent (`alpha = 0`) outside this
    /// item's contribution area for the tile.
    type Content = Vec<u8>;
    type Coverage = MultiPolygon;

    fn source(&self) -> String {
        self.name().to_owned()
    }

    fn coverage(&self) -> Self::Coverage {
        Self::coverage(self)
    }

    fn native_scale(&self) -> u32 {
        Self::native_scale(self)
    }

    /// Resample this cell into `tile`'s pixel grid.
    ///
    /// Nearest-neighbour, not area-averaged or bilinear: at zooms coarser
    /// than the cell's native zoom this aliases rather than blurs, same as
    /// a raw paper-chart scan viewed zoomed-out. Acceptable for a first
    /// pass — revisit with profiling data if visual quality demands it
    /// (see `.github/copilot-instructions.md`: optimisations need evidence).
    #[allow(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::min_ident_chars, // x/y/u/v/n match the projection math in src/rnc.rs's docs
        clippy::many_single_char_names
    )]
    fn render(&self, tile: &TileGeom) -> Self::Content {
        let mut buf = vec![0u8; (TILE_PX * TILE_PX * 4) as usize];

        // This item's exact contribution polygon for this tile (already
        // clipped by the quilting loop in `tiles::write_pmtiles` so disjoint
        // contributors never double-paint). `contrib_bbox` is a cheap
        // broad-phase rejection before the exact-but-pricier polygon test
        // below — most pixels in a partially-covered tile are unambiguously
        // outside it.
        let contrib_bbox = Bbox::from(&tile.geom);
        if contrib_bbox.is_bottom() {
            return buf;
        }

        let (xmin, ymin, xmax, ymax) = self.merc_extent();
        let cols = self.cols();
        let rows = self.rows();
        let merc_w = tile.merc.east - tile.merc.west;
        let merc_h = tile.merc.north - tile.merc.south;

        // Caches the most recently decoded subtile: adjacent destination
        // pixels overwhelmingly land in the same source subtile, so this
        // avoids a cache lookup per pixel in the common case.
        let mut current: Option<(u32, std::sync::Arc<RgbaImage>)> = None;
        let mut decode_failed = false;

        for py in 0..TILE_PX {
            let y_m =
                ((f64::from(py) + 0.5) / f64::from(TILE_PX)).mul_add(-merc_h, tile.merc.north);
            for px in 0..TILE_PX {
                let x_m =
                    ((f64::from(px) + 0.5) / f64::from(TILE_PX)).mul_add(merc_w, tile.merc.west);
                let (lon, lat) = webmercator_to_wgs84(x_m, y_m);

                if lon < contrib_bbox.west
                    || lon > contrib_bbox.east
                    || lat < contrib_bbox.south
                    || lat > contrib_bbox.north
                    || !tile.geom.contains(&Point::new(lon, lat))
                {
                    continue;
                }

                let (x, y) = wgs84_to_nv_merc(lon, lat);
                let u = ((x - xmin) / (xmax - xmin)).clamp(0.0, 0.999_999_9);
                let v = ((y - ymin) / (ymax - ymin)).clamp(0.0, 0.999_999_9);
                let col = (u * f64::from(cols)).floor() as u32;
                let row = (v * f64::from(rows)).floor() as u32;
                let n = row * cols + col;

                let img = match &current {
                    Some((cur_n, img)) if *cur_n == n => img.clone(),
                    _ => match self.subtile_image(n) {
                        Ok(img) => {
                            current = Some((n, img.clone()));
                            img
                        }
                        Err(e) => {
                            if !decode_failed {
                                tracing::warn!(
                                    cell = self.name(),
                                    n,
                                    error = %e,
                                    "failed to decode raster subtile; leaving transparent"
                                );
                                decode_failed = true;
                            }
                            continue;
                        }
                    },
                };

                let (img_w, img_h) = (img.width(), img.height());
                if img_w == 0 || img_h == 0 {
                    continue;
                }
                let lu = u * f64::from(cols) - f64::from(col);
                let lv = v * f64::from(rows) - f64::from(row);
                let sx = ((lu * f64::from(img_w)) as u32).min(img_w - 1);
                let sy = ((lv * f64::from(img_h)) as u32).min(img_h - 1);
                let p = img.get_pixel(sx, sy);

                let idx = ((py * TILE_PX + px) * 4) as usize;
                buf[idx..idx + 4].copy_from_slice(&p.0);
            }
        }

        buf
    }

    /// Composite contributions (already spatially disjoint) onto one canvas
    /// and PNG-encode it. Returns an empty `Vec` — omitting the tile — when
    /// every pixel stayed transparent.
    fn encode(contents: Vec<Self::Content>) -> Result<Vec<u8>> {
        let mut canvas = vec![0u8; (TILE_PX * TILE_PX * 4) as usize];
        let mut any_opaque = false;
        for content in &contents {
            for (dst, src) in canvas.chunks_exact_mut(4).zip(content.chunks_exact(4)) {
                if src[3] > 0 {
                    dst.copy_from_slice(src);
                    any_opaque = true;
                }
            }
        }
        if !any_opaque {
            return Ok(Vec::new());
        }

        let img = RgbaImage::from_raw(TILE_PX, TILE_PX, canvas)
            .context("building output raster tile buffer")?;
        let mut out = Vec::new();
        img.write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)
            .context("encoding PNG tile")?;
        Ok(out)
    }

    fn tile_type() -> TileType {
        TileType::Png
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rnc::RncCell;
    use geo::{MultiPolygon, Polygon};
    use image::Rgba;
    use std::io::Cursor;

    fn solid_png(w: u32, h: u32, rgba: [u8; 4]) -> Vec<u8> {
        let img = RgbaImage::from_pixel(w, h, Rgba(rgba));
        let mut out = Vec::new();
        img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
            .expect("encoding test PNG cannot fail");
        out
    }

    #[allow(clippy::cast_possible_truncation)] // test PNG sizes are tiny constants
    fn build_rnc(cols: u32, rows: u32, bbox: Bbox, color: [u8; 4], scale: f64) -> Vec<u8> {
        let png = solid_png(8, 8, color);
        let n_tiles = cols * rows;
        let n_offsets = n_tiles + 2;
        let table_start = 16u32;
        let blobs_start = table_start + n_offsets * 4;

        let mut offsets = Vec::with_capacity(n_offsets as usize);
        for i in 0..=n_tiles {
            offsets.push(blobs_start + i * png.len() as u32);
        }
        offsets.push(*offsets.last().expect("at least one offset"));

        let mut buf = vec![0u8; 8];
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
            "lat0": bbox.south, "lat1": bbox.north,
            "lon0": bbox.west, "lon1": bbox.east,
            "edate": "01/01/2026", "name": "TEST", "scale": scale,
        });
        buf.extend_from_slice(serde_json::to_string(&footer).unwrap().as_bytes());
        buf
    }

    /// A `TileGeom` covering the whole of `merc` in Web-Mercator metres, with
    /// `geom` clipped to `contrib` (also in `WGS-84` degrees).
    fn make_tile(merc: Bbox, contrib: Bbox, scale: u32) -> TileGeom {
        TileGeom {
            geom: MultiPolygon::new(vec![Polygon::from(contrib)]),
            merc,
            scale,
        }
    }

    #[test]
    fn render_paints_inside_contribution_and_leaves_outside_transparent() {
        let bbox = Bbox {
            west: 11.0,
            south: 57.0,
            east: 12.0,
            north: 58.0,
        };
        let data = build_rnc(1, 1, bbox, [200, 30, 30, 255], 3_000_000.0);
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");

        let (w_m, s_m) = martin_tile_utils::wgs84_to_webmercator(bbox.west, bbox.south);
        let (e_m, n_m) = martin_tile_utils::wgs84_to_webmercator(bbox.east, bbox.north);
        let merc = Bbox {
            west: w_m,
            south: s_m,
            east: e_m,
            north: n_m,
        };

        // Contribution is only the western half of the cell.
        let contrib = Bbox {
            west: bbox.west,
            south: bbox.south,
            east: f64::midpoint(bbox.west, bbox.east),
            north: bbox.north,
        };
        let tile = make_tile(merc, contrib, 3_000_000);

        let out = TileSource::render(&cell, &tile);
        assert_eq!(out.len(), (TILE_PX * TILE_PX * 4) as usize);

        // West edge (inside contribution) must be painted.
        let west_idx = (TILE_PX / 8 * 4) as usize; // a column well inside the west half
        assert_eq!(
            out[west_idx + 3],
            255,
            "pixel inside contribution should be opaque"
        );
        assert_eq!(&out[west_idx..west_idx + 3], &[200, 30, 30]);

        // East edge (outside contribution) must stay transparent.
        let east_idx = ((TILE_PX - TILE_PX / 8) * 4) as usize;
        assert_eq!(
            out[east_idx + 3],
            0,
            "pixel outside contribution should be transparent"
        );
    }

    #[test]
    fn encode_composites_disjoint_contributions() {
        let mut left = vec![0u8; (TILE_PX * TILE_PX * 4) as usize];
        let mut right = left.clone();
        for py in 0..TILE_PX {
            for px in 0..TILE_PX / 2 {
                let idx = ((py * TILE_PX + px) * 4) as usize;
                left[idx..idx + 4].copy_from_slice(&[255, 0, 0, 255]);
            }
            for px in TILE_PX / 2..TILE_PX {
                let idx = ((py * TILE_PX + px) * 4) as usize;
                right[idx..idx + 4].copy_from_slice(&[0, 255, 0, 255]);
            }
        }
        let bytes = RncCell::encode(vec![left, right]).expect("encode succeeds");
        assert!(!bytes.is_empty());
        let img = image::load_from_memory_with_format(&bytes, image::ImageFormat::Png)
            .expect("re-decodes")
            .to_rgba8();
        assert_eq!(img.get_pixel(0, 0).0, [255, 0, 0, 255]);
        assert_eq!(img.get_pixel(TILE_PX - 1, 0).0, [0, 255, 0, 255]);
    }

    /// Flatten WGS-84 ring vertices into the footer's Mercator
    /// `[x0, y0, x1, y1, …]` shape (mirrors `crate::rnc::tests::cover_ring`).
    #[allow(clippy::tuple_array_conversions)] // [x, y] is the natural flat_map yield shape
    fn cover_ring(vertices: &[(f64, f64)]) -> Vec<f64> {
        vertices
            .iter()
            .flat_map(|&(lon, lat)| {
                let (x, y) = crate::rnc::wgs84_to_nv_merc(lon, lat);
                [x, y]
            })
            .collect()
    }

    #[test]
    fn render_honors_non_rectangular_cover_polygon() {
        let bbox = Bbox {
            west: 11.0,
            south: 57.0,
            east: 12.0,
            north: 58.0,
        };
        // Triangle over SW/SE/NE — excludes the NW corner of the bbox.
        let triangle = cover_ring(&[(11.0, 57.0), (12.0, 57.0), (12.0, 58.0)]);
        let png = {
            let img = RgbaImage::from_pixel(8, 8, Rgba([200, 30, 30, 255]));
            let mut out = Vec::new();
            img.write_to(&mut Cursor::new(&mut out), image::ImageFormat::Png)
                .unwrap();
            out
        };
        let n_offsets = 3u32;
        let blobs_start = 16 + n_offsets * 4;
        let offsets = [
            blobs_start,
            blobs_start + u32::try_from(png.len()).unwrap(),
            blobs_start + u32::try_from(png.len()).unwrap(),
        ];
        let mut data = vec![0u8; 8];
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        for o in &offsets {
            data.extend_from_slice(&o.to_le_bytes());
        }
        data.extend_from_slice(&png);
        let footer = serde_json::json!({
            "cover": [triangle],
            "lat0": bbox.south, "lat1": bbox.north,
            "lon0": bbox.west, "lon1": bbox.east,
            "edate": "01/01/2026", "name": "TEST", "scale": 3_000_000.0,
        });
        data.extend_from_slice(serde_json::to_string(&footer).unwrap().as_bytes());
        let cell = RncCell::parse("TEST".to_owned(), data).expect("valid .rnc parses");

        let (w_m, s_m) = martin_tile_utils::wgs84_to_webmercator(bbox.west, bbox.south);
        let (e_m, n_m) = martin_tile_utils::wgs84_to_webmercator(bbox.east, bbox.north);
        let merc = Bbox {
            west: w_m,
            south: s_m,
            east: e_m,
            north: n_m,
        };
        let tile = TileGeom {
            geom: cell.coverage(),
            merc,
            scale: 3_000_000,
        };

        let out = TileSource::render(&cell, &tile);

        // Pixel index for a given (lon, lat), top-left origin, north-up.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let pixel_alpha = |lon: f64, lat: f64| -> u8 {
            let (x_m, y_m) = martin_tile_utils::wgs84_to_webmercator(lon, lat);
            let px = (((x_m - merc.west) / (merc.east - merc.west)) * f64::from(TILE_PX)) as u32;
            let py = (((merc.north - y_m) / (merc.north - merc.south)) * f64::from(TILE_PX)) as u32;
            let idx = ((py.min(TILE_PX - 1) * TILE_PX + px.min(TILE_PX - 1)) * 4) as usize;
            out[idx + 3]
        };

        // Inside the triangle (near its centroid).
        assert_eq!(
            pixel_alpha(11.7, 57.3),
            255,
            "centroid of cover triangle should be painted"
        );
        // Inside the bbox rectangle but outside the triangle (NW corner).
        assert_eq!(
            pixel_alpha(11.05, 57.95),
            0,
            "NW corner excluded by cover polygon should stay transparent"
        );
    }

    #[test]
    fn encode_returns_empty_for_fully_transparent_tile() {
        let blank = vec![0u8; (TILE_PX * TILE_PX * 4) as usize];
        let bytes = RncCell::encode(vec![blank]).expect("encode succeeds");
        assert!(bytes.is_empty());
    }
}
