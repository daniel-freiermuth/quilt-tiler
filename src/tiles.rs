//! Direct tile writing: OESU cells → MVT tiles → `PMTiles` archive.
//!
//! Strategy (first pass): each chart is tiled at exactly
//! `zoom_from_scale(native_scale)`. Charts at the same zoom level may cover
//! overlapping or adjacent areas; their features are merged per tile.
//! Zoom levels between scale bands are intentionally empty — `MapLibre` GL
//! overzooms the nearest available tile automatically.

use std::collections::{BTreeMap, HashMap};
use std::fs::File;
use std::path::Path;

use anyhow::{Context, Result};
use fast_mvt::{
    MvtFeature, MvtGeometry, MvtLayer, MvtLineString, MvtMultiLineString, MvtPoint, MvtPolygon,
    MvtTile, MvtValue, DEFAULT_EXTENT,
};
use martin_tile_utils::{bbox_to_xyz, wgs84_to_webmercator, xyz_to_bbox};
use pmtiles::{PmTilesWriter, TileCoord, TileType};
use tracing::info;

use crate::s57::{attribute_acronym, object_acronym};
use crate::zoom::zoom_from_scale;

const EXTENT: f64 = 4096.0;

// ── Public entry point ───────────────────────────────────────────────────────

/// Encode all parsed `cells` as MVT tiles and write a `PMTiles` v3 archive to
/// `output`. Tiles are written in Hilbert-curve (`TileID`) order as required by
/// the `PMTiles` spec.
///
/// Memory model: each chart/tile pair is encoded to raw MVT bytes immediately.
/// Multiple charts covering the same tile are merged by concatenating their
/// MVT byte blobs — valid because `Tile { repeated Layer layers = 3 }` is a
/// protobuf repeated field; concatenating two encoded Tile messages unions
/// their layers.
pub fn write_pmtiles(cells: &[oesu::OesuCell], output: &Path) -> Result<(u8, u8)> {
    // Accumulate raw MVT bytes per TileID; BTreeMap keeps entries in sorted order,
    // matching the PMTiles Hilbert-curve requirement without a separate sort pass.
    let mut tile_bytes: BTreeMap<u64, (TileCoord, Vec<u8>)> = BTreeMap::new();

    let mut min_zoom = u8::MAX;
    let mut max_zoom = u8::MIN;
    let mut bounds = [f64::INFINITY, f64::INFINITY, f64::NEG_INFINITY, f64::NEG_INFINITY];

    for cell in cells {
        let zoom = zoom_from_scale(cell.native_scale);
        min_zoom = min_zoom.min(zoom);
        max_zoom = max_zoom.max(zoom);

        let [west, south, east, north] = cell.bounds;
        bounds[0] = bounds[0].min(west);
        bounds[1] = bounds[1].min(south);
        bounds[2] = bounds[2].max(east);
        bounds[3] = bounds[3].max(north);

        let (col_lo, row_lo, col_hi, row_hi) = bbox_to_xyz(west, south, east, north, zoom);

        for col in col_lo..=col_hi {
            for row in row_lo..=row_hi {
                let tile_wgs84 = xyz_to_bbox(zoom, col, row, col, row);
                let tile_merc = tile_mercator_bbox(tile_wgs84);

                // Collect this chart's features for this tile.
                let mut layers: HashMap<&'static str, Vec<MvtFeature>> = HashMap::new();
                for feat in &cell.features {
                    if !feat_intersects(feat, tile_wgs84) {
                        continue;
                    }
                    let Some(layer_name) = object_acronym(feat.type_code) else {
                        continue;
                    };
                    let feats = to_mvt_features(feat, tile_wgs84, tile_merc);
                    if !feats.is_empty() {
                        layers.entry(layer_name).or_default().extend(feats);
                    }
                }

                if layers.is_empty() {
                    continue;
                }

                // Encode immediately; drop all MvtFeature allocations.
                let bytes = encode_tile(layers)?;
                if bytes.is_empty() {
                    continue;
                }

                // Merge into BTreeMap entry (appending bytes for multi-chart tiles).
                let id = tile_id(zoom, col, row);
                match tile_bytes.entry(id) {
                    std::collections::btree_map::Entry::Occupied(mut e) => {
                        e.get_mut().1.extend(bytes);
                    }
                    std::collections::btree_map::Entry::Vacant(e) => {
                        let coord = TileCoord::new(zoom, col, row).context("invalid tile coord")?;
                        e.insert((coord, bytes));
                    }
                }
            }
        }
    }

    info!(tiles = tile_bytes.len(), min_zoom, max_zoom, "writing tiles");

    // BTreeMap is already sorted by TileID — no separate sort needed.

    // Write the PMTiles archive.
    let [bw, bs, be, bn] = if bounds[0].is_finite() {
        bounds
    } else {
        [-180.0, -85.0, 180.0, 85.0]
    };
    let min_zoom = if min_zoom == u8::MAX { 0 } else { min_zoom };
    let max_zoom = if max_zoom == u8::MIN { 0 } else { max_zoom };

    let metadata = build_metadata();
    let file =
        File::create(output).with_context(|| format!("creating {}", output.display()))?;
    let mut writer = PmTilesWriter::new(TileType::Mvt)
        .min_zoom(min_zoom)
        .max_zoom(max_zoom)
        .bounds(bw, bs, be, bn)
        .metadata(&metadata)
        .create(file)
        .context("creating PMTiles writer")?;

    for (_, (coord, bytes)) in tile_bytes {
        writer.add_tile(coord, &bytes).context("writing tile")?;
    }
    writer.finalize().context("finalizing PMTiles")?;

    info!(output = %output.display(), "PMTiles written");
    Ok((min_zoom, max_zoom))
}

// ── Coordinate transform ─────────────────────────────────────────────────────

/// Convert a WGS84 tile bbox `[west, south, east, north]` to Web Mercator
/// metres `[west_m, south_m, east_m, north_m]`.
fn tile_mercator_bbox(wgs84: [f64; 4]) -> [f64; 4] {
    let (w_m, s_m) = wgs84_to_webmercator(wgs84[0], wgs84[1]);
    let (e_m, n_m) = wgs84_to_webmercator(wgs84[2], wgs84[3]);
    [w_m, s_m, e_m, n_m]
}

/// Project `(lon, lat)` WGS84 to tile pixel coordinates `(x, y)` in
/// `[0, 4096)` space. Coordinates outside the tile extent are valid — the
/// MVT spec allows it, and `MapLibre` GL clips client-side.
#[allow(clippy::cast_possible_truncation)] // deliberate floor-truncation
fn to_px(lon: f64, lat: f64, merc: [f64; 4]) -> fast_mvt::MvtCoord {
    let (x_m, y_m) = wgs84_to_webmercator(lon, lat);
    let px = ((x_m - merc[0]) / (merc[2] - merc[0]) * EXTENT) as i32;
    let py = ((merc[3] - y_m) / (merc[3] - merc[1]) * EXTENT) as i32; // y=0 at north
    (px, py).into()
}

// ── Feature intersection test ────────────────────────────────────────────────

fn feat_intersects(feat: &oesu::Feature, tile: [f64; 4]) -> bool {
    let Some((fw, fs, fe, fn_)) = feat_bbox(feat) else {
        return false;
    };
    // Overlap when neither axis is disjoint.
    fw <= tile[2] && fe >= tile[0] && fs <= tile[3] && fn_ >= tile[1]
}

fn feat_bbox(feat: &oesu::Feature) -> Option<(f64, f64, f64, f64)> {
    match &feat.geometry {
        oesu::Geometry::None => None,
        oesu::Geometry::Point { lon, lat } => Some((*lon, *lat, *lon, *lat)),
        oesu::Geometry::MultiPoint(pts) => {
            bbox_of(pts.iter().map(|p| (p[0], p[1])))
        }
        oesu::Geometry::Line(strokes) => {
            bbox_of(strokes.iter().flat_map(|s| s.iter()).map(|p| (p[0], p[1])))
        }
        oesu::Geometry::Area(ag) => {
            bbox_of(ag.rings.iter().flat_map(|r| r.iter()).map(|p| (p[0], p[1])))
        }
    }
}

fn bbox_of(mut pts: impl Iterator<Item = (f64, f64)>) -> Option<(f64, f64, f64, f64)> {
    let first = pts.next()?;
    let (mut w, mut s, mut e, mut n) = (first.0, first.1, first.0, first.1);
    for (lon, lat) in pts {
        if lon < w {
            w = lon;
        }
        if lat < s {
            s = lat;
        }
        if lon > e {
            e = lon;
        }
        if lat > n {
            n = lat;
        }
    }
    Some((w, s, e, n))
}

// ── Feature conversion ───────────────────────────────────────────────────────

/// Convert one OESU feature to zero or more MVT features in tile pixel space.
///
/// `tile_wgs84` is used to filter `MultiPoint` (soundings): only the points
/// that fall within the tile are emitted, avoiding 20× duplication across tiles.
fn to_mvt_features(feat: &oesu::Feature, tile_wgs84: [f64; 4], merc: [f64; 4]) -> Vec<MvtFeature> {
    let props = build_props(&feat.attributes);

    match &feat.geometry {
        oesu::Geometry::None => vec![],

        oesu::Geometry::Point { lon, lat } => {
            let c = to_px(*lon, *lat, merc);
            let mut f = MvtFeature::new(MvtGeometry::Point(MvtPoint::new(c.x, c.y)));
            f.properties = props;
            vec![f]
        }

        oesu::Geometry::MultiPoint(pts) => pts
            .iter()
            .filter(|[lon, lat, _]| {
                // Each sounding belongs to exactly one tile.
                *lon >= tile_wgs84[0] && *lon <= tile_wgs84[2]
                    && *lat >= tile_wgs84[1] && *lat <= tile_wgs84[3]
            })
            .map(|[lon, lat, depth]| {
                let c = to_px(*lon, *lat, merc);
                let mut f = MvtFeature::new(MvtGeometry::Point(MvtPoint::new(c.x, c.y)));
                f.properties.clone_from(&props);
                f.add_tag_double("VALDCO", *depth);
                f
            })
            .collect(),

        oesu::Geometry::Line(strokes) => {
            if strokes.is_empty() {
                return vec![];
            }
            let geom = if strokes.len() == 1 {
                let ls: MvtLineString = strokes[0]
                    .iter()
                    .map(|[lon, lat]| to_px(*lon, *lat, merc))
                    .collect();
                MvtGeometry::LineString(ls)
            } else {
                let lines: Vec<MvtLineString> = strokes
                    .iter()
                    .map(|s| s.iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect())
                    .collect();
                MvtGeometry::MultiLineString(MvtMultiLineString::new(lines))
            };
            let mut f = MvtFeature::new(geom);
            f.properties = props;
            vec![f]
        }

        oesu::Geometry::Area(ag) => {
            if ag.rings.is_empty() {
                return vec![];
            }
            let exterior: MvtLineString = ag.rings[0]
                .iter()
                .map(|[lon, lat]| to_px(*lon, *lat, merc))
                .collect();
            let holes: Vec<MvtLineString> = ag.rings[1..]
                .iter()
                .map(|r| r.iter().map(|[lon, lat]| to_px(*lon, *lat, merc)).collect())
                .collect();
            let mut f = MvtFeature::new(MvtGeometry::Polygon(MvtPolygon::new(exterior, holes)));
            f.properties = props;
            vec![f]
        }
    }
}

fn build_props(attrs: &[oesu::Attribute]) -> Vec<(String, MvtValue)> {
    attrs
        .iter()
        .filter_map(|attr| {
            let key = attribute_acronym(attr.code)?;
            let val = match &attr.value {
                oesu::AttrValue::Int(i) => MvtValue::UInt(u64::from(*i)),
                oesu::AttrValue::Double(f) => MvtValue::Double(*f),
                oesu::AttrValue::Str(s) => MvtValue::String(s.clone()),
            };
            Some((key.to_string(), val))
        })
        .collect()
}

// ── MVT tile encoding ────────────────────────────────────────────────────────

fn encode_tile(layers: HashMap<&'static str, Vec<MvtFeature>>) -> Result<Vec<u8>> {
    let mut tile = MvtTile::new();
    for (name, features) in layers {
        if features.is_empty() {
            continue;
        }
        let mut layer = MvtLayer::new(name, DEFAULT_EXTENT);
        for feat in features {
            layer.add_feature(feat);
        }
        tile.add_layer(layer);
    }
    if tile.layers.is_empty() {
        return Ok(vec![]);
    }
    tile.encode().context("encoding MVT tile")
}

// ── PMTiles metadata ─────────────────────────────────────────────────────────

fn build_metadata() -> String {
    // Minimal TileJSON-compatible metadata. `vector_layers` is intentionally
    // empty for now; field schemas can be derived from S-57 and added later.
    serde_json::json!({
        "name": "chart",
        "description": "Nautical chart — converted from OESU",
        "vector_layers": []
    })
    .to_string()
}

// ── PMTiles TileID (Hilbert curve) ───────────────────────────────────────────

/// Compute the `PMTiles` v3 `TileID` for tile `(z, x, y)`.
///
/// `TileID = (4^z − 1) / 3 + hilbert_xy_to_d(2^z, x, y)`
fn tile_id(z: u8, x: u32, y: u32) -> u64 {
    if z == 0 {
        return 0;
    }
    let base = (4u64.pow(u32::from(z)) - 1) / 3;
    base + hilbert_xy_to_d(1u64 << z, u64::from(x), u64::from(y))
}

#[allow(clippy::many_single_char_names)] // n, x, y, d, s are standard Hilbert curve variables
fn hilbert_xy_to_d(n: u64, mut x: u64, mut y: u64) -> u64 {
    let mut d = 0u64;
    let mut s = n / 2;
    while s > 0 {
        let rx = u64::from((x & s) > 0);
        let ry = u64::from((y & s) > 0);
        d += s * s * ((3 * rx) ^ ry);
        if ry == 0 {
            if rx == 1 {
                x = (n - 1) - x;
                y = (n - 1) - y;
            }
            std::mem::swap(&mut x, &mut y);
        }
        s /= 2;
    }
    d
}
