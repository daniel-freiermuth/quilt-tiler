//! OSENC binary format parser.
//!
//! Reads the little-endian packed-struct format used by `OpenCPN` for its
//! internal Simplified ENC files.  Record layout:
//!   u16 `record_type`  +  u32 `record_length` (total, incl. 6-byte header)
//!
//! A decrypted `.oesu` stream always starts with:
//!   1. `SERVER_STATUS_RECORD` (200): decrypt/expiry validation
//!   2. `HEADER_SENC_VERSION` (1): version must be in range 200–299
//!      (version 1024 is a known sentinel for signature failure)
//!
//! Raster charts use the `.oernc` extension and BSB binary format; passing
//! one to this parser is detected and rejected with a clear error.

mod georef;

use std::collections::{HashMap, HashSet};
use std::io::{Cursor, Read};

use anyhow::{Context, Result, bail};
use geo::{Area, Coord, Intersects, LineString, MultiPolygon, Point, Polygon, point};
use s57::{AttrValue, Attribute, Feature, Geometry};

// ── Record type constants ────────────────────────────────────────────────────

const HEADER_SENC_VERSION: u16 = 1;
const HEADER_CELL_NAME: u16 = 2;
const HEADER_CELL_PUBLISHDATE: u16 = 3;
const HEADER_CELL_EDITION: u16 = 4;
const HEADER_CELL_UPDATEDATE: u16 = 5;
const HEADER_CELL_UPDATE: u16 = 6;
const HEADER_CELL_NATIVESCALE: u16 = 7;
const HEADER_CELL_SENCCREATEDATE: u16 = 8;
const HEADER_CELL_SOUNDINGDATUM: u16 = 9;

const FEATURE_ID_RECORD: u16 = 64;
const FEATURE_ATTRIBUTE_RECORD: u16 = 65;

const FEATURE_GEOMETRY_RECORD_POINT: u16 = 80;
const FEATURE_GEOMETRY_RECORD_LINE: u16 = 81;
const FEATURE_GEOMETRY_RECORD_AREA: u16 = 82;
const FEATURE_GEOMETRY_RECORD_MULTIPOINT: u16 = 83;
/// Extended area geometry: i16 SM coords scaled by a `f64 scale_factor`.
const FEATURE_GEOMETRY_RECORD_AREA_EXT: u16 = 84;

/// Extended VET: i16 SM coords scaled by a leading `f64 scale_factor`.
const VECTOR_EDGE_NODE_TABLE_EXT_RECORD: u16 = 85;
/// Extended VCT: i16 SM coords scaled by a leading `f64 scale_factor`.
const VECTOR_CONNECTED_NODE_TABLE_EXT_RECORD: u16 = 86;

const VECTOR_EDGE_NODE_TABLE_RECORD: u16 = 96;
const VECTOR_CONNECTED_NODE_TABLE_RECORD: u16 = 97;

const CELL_COVR_RECORD: u16 = 98;
const CELL_NOCOVR_RECORD: u16 = 99;
const CELL_EXTENT_RECORD: u16 = 100;
const CELL_TXTDSC_INFO_FILE_RECORD: u16 = 101;

const SERVER_STATUS_RECORD: u16 = 200;

/// An edge entry from the Vector Edge Node Table (VET, record 96 or 85).
struct EdgeEntry {
    /// Intermediate (non-endpoint) points as [lon, lat] pairs — still in SM at
    /// parse time, resolved to WGS84 after the full record scan.
    points: Vec<[f64; 2]>,
}

/// A connected node from the Vector Connected Node Table (VCT, record 97 or 86).
struct NodeEntry {
    lon: f64,
    lat: f64,
}

#[allow(dead_code)] // fields parsed for cursor advancement; only name/scale/bounds/features are forwarded to S57Cell
struct OesuCell {
    name: String,
    native_scale: u32,
    senc_version: u16,
    publish_date: String,
    edition: u16,
    update_date: String,
    update_number: u16,
    senc_create_date: String,
    sounding_datum: String,
    expire_days_remaining: u16,
    grace_days_remaining: u16,
    ref_lat: f64,
    ref_lon: f64,
    bounds: [f64; 4],
    features: Vec<Feature>,
    coverage: MultiPolygon,
    text_descriptions: HashMap<String, String>,
    source: String,
}

impl From<OesuCell> for s57::S57Cell {
    fn from(c: OesuCell) -> Self {
        Self {
            name: c.name,
            native_scale: c.native_scale,
            bounds: c.bounds,
            features: c.features,
            coverage: c.coverage,
            source: c.source,
            text_descriptions: c.text_descriptions,
        }
    }
}

// ── Internal parse structures ────────────────────────────────────────────────

#[derive(Debug)]
struct RawFeature {
    type_code: u16,
    id: u16,
    primitive: u8,
    attributes: Vec<Attribute>,
    raw_geometry: RawGeometry,
}

/// OpenGL primitive type stored in the OSENC `TriPrim` chain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TriPrimType {
    Triangles = 4,     // GL_TRIANGLES
    TriangleStrip = 5, // GL_TRIANGLE_STRIP
    TriangleFan = 6,   // GL_TRIANGLE_FAN
}

impl TriPrimType {
    /// Returns `None` for unrecognised values instead of panicking.
    const fn from_u8(v: u8) -> Option<Self> {
        match v {
            4 => Some(Self::Triangles),
            5 => Some(Self::TriangleStrip),
            6 => Some(Self::TriangleFan),
            _ => None,
        }
    }
}

/// Internal `TriPrim` before SM → WGS84 coordinate conversion.
#[derive(Debug)]
struct RawTriPrim {
    _prim_type: u8,
    /// [W, S, E, N] — WGS84 degrees.
    /// For EXT records (84) the bbox is converted from SM to WGS84 at parse time,
    /// since `CELL_EXTENT_RECORD` always precedes feature geometry in the stream.
    _bbox: [f64; 4],
    /// SM (east, north) coordinate pairs, one per vertex.
    _vertices: Vec<[f32; 2]>,
}

#[derive(Debug)]
enum RawGeometry {
    None,
    Point(Point),
    Sounding(Vec<[f32; 3]>), // (east, north, depth) in SM
    Line(Vec<[i32; 4]>),     // [start_node, edge_id, end_node, dir]
    Area {
        contour_count: u32,
        vertex_counts: Vec<u32>,
        tri_prims: Vec<RawTriPrim>,
        edge_refs: Vec<[i32; 4]>,
    },
}

// ── Reader helpers ───────────────────────────────────────────────────────────

fn read_u8(c: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut b = [0u8; 1];
    c.read_exact(&mut b)?;
    Ok(b[0])
}
fn read_i16(c: &mut Cursor<&[u8]>) -> Result<i16> {
    let mut b = [0u8; 2];
    c.read_exact(&mut b)?;
    Ok(i16::from_le_bytes(b))
}
fn read_u16(c: &mut Cursor<&[u8]>) -> Result<u16> {
    let mut b = [0u8; 2];
    c.read_exact(&mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(c: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_i32(c: &mut Cursor<&[u8]>) -> Result<i32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)?;
    Ok(i32::from_le_bytes(b))
}
fn read_f32(c: &mut Cursor<&[u8]>) -> Result<f32> {
    let mut b = [0u8; 4];
    c.read_exact(&mut b)?;
    Ok(f32::from_le_bytes(b))
}
fn read_f64(c: &mut Cursor<&[u8]>) -> Result<f64> {
    let mut b = [0u8; 8];
    c.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}
fn read_cstring(c: &mut Cursor<&[u8]>, max: usize) -> Result<String> {
    let mut buf = Vec::with_capacity(32);
    for _ in 0..max {
        let b = read_u8(c)?;
        if b == 0 {
            break;
        }
        buf.push(b);
    }
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

// ── Main entry point ─────────────────────────────────────────────────────────

#[allow(
    clippy::too_many_lines,          // binary format parser is inherently long
    clippy::cast_possible_truncation, // Cursor positions bounded by payload_len (≤ u32)
    clippy::similar_names,           // sw/nw/ne/se and resolved_vct/resolved_vet are domain vocab
    clippy::missing_errors_doc,      // private binary parser; error cases in record comments
)]
pub fn parse_file(source: String, data: &[u8]) -> Result<s57::S57Cell> {
    // ── Prologue: validate SERVER_STATUS + version ───────────────────────────
    let (data, expire_days_remaining, grace_days_remaining) = strip_server_status(data)?;
    let senc_version = read_senc_version(data)?;

    // ── Accumulator state ────────────────────────────────────────────────────
    let mut name = String::new();
    let mut native_scale: u32 = 0;
    let mut publish_date = String::new();
    let mut edition: u16 = 0;
    let mut update_date = String::new();
    let mut update_number: u16 = 0;
    let mut senc_create_date = String::new();
    let mut sounding_datum = String::new();
    let mut ref_lat = 0.0f64;
    let mut ref_lon = 0.0f64;
    let mut bounds = [0.0f64; 4];
    let mut raw_features: Vec<RawFeature> = Vec::new();
    let mut raw_covr: Vec<LineString> = Vec::new();
    let mut raw_nocovr: Vec<LineString> = Vec::new();
    let mut text_descriptions: HashMap<String, String> = HashMap::new();
    let mut vet: HashMap<u32, EdgeEntry> = HashMap::new();
    let mut vct: HashMap<u32, NodeEntry> = HashMap::new();

    let mut c = Cursor::new(data);
    let mut current: Option<RawFeature> = None;

    loop {
        let mut hdr = [0u8; 6];
        match c.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let rec_type = u16::from_le_bytes([hdr[0], hdr[1]]);
        let rec_len = u32::from_le_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]);
        if rec_len < 6 {
            if rec_type != 0 || rec_len != 0 {
                tracing::warn!(rec_type, rec_len, "record with length < 6, stopping parse");
            }
            break;
        }
        let payload_len = (rec_len - 6) as usize;

        let mut payload_bytes = vec![0u8; payload_len];
        c.read_exact(&mut payload_bytes)
            .with_context(|| format!("reading payload of record type {rec_type}"))?;
        let mut p = Cursor::new(payload_bytes.as_slice());

        match rec_type {
            // ── Header metadata ──────────────────────────────────────────────
            HEADER_SENC_VERSION => { /* already consumed in prologue */ }

            HEADER_CELL_NAME => {
                name = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_PUBLISHDATE => {
                publish_date = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_EDITION => {
                edition = read_u16(&mut p).unwrap_or(0);
            }
            HEADER_CELL_UPDATEDATE => {
                update_date = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_UPDATE => {
                update_number = read_u16(&mut p).unwrap_or(0);
            }
            HEADER_CELL_NATIVESCALE => {
                native_scale = read_u32(&mut p).unwrap_or(0);
            }
            HEADER_CELL_SENCCREATEDATE => {
                senc_create_date = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_SOUNDINGDATUM => {
                sounding_datum = read_cstring(&mut p, payload_len).unwrap_or_default();
            }

            // ── Cell spatial metadata ────────────────────────────────────────
            CELL_EXTENT_RECORD => {
                // 8 × f64: sw_lat, sw_lon, nw_lat, nw_lon, ne_lat, ne_lon, se_lat, se_lon
                if payload_len >= 64 {
                    let sw_lat = read_f64(&mut p)?;
                    let sw_lon = read_f64(&mut p)?;
                    let nw_lat = read_f64(&mut p)?;
                    let nw_lon = read_f64(&mut p)?;
                    let ne_lat = read_f64(&mut p)?;
                    let ne_lon = read_f64(&mut p)?;
                    let se_lat = read_f64(&mut p)?;
                    let se_lon = read_f64(&mut p)?;

                    let s_lat = sw_lat.min(se_lat);
                    let n_lat = nw_lat.max(ne_lat);
                    let w_lon = sw_lon.min(nw_lon);
                    let e_lon = ne_lon.max(se_lon);

                    ref_lat = f64::midpoint(n_lat, s_lat);
                    ref_lon = f64::midpoint(e_lon, w_lon);
                    bounds = [w_lon, s_lat, e_lon, n_lat];
                } else {
                    tracing::warn!(payload_len, "CELL_EXTENT_RECORD too short, skipping");
                }
            }

            CELL_COVR_RECORD => {
                if let Some(covr) = parse_covr_payload(&mut p, payload_len) {
                    raw_covr.push(covr);
                } else {
                    tracing::warn!("CELL_COVR_RECORD: could not parse coverage polygon");
                }
            }
            CELL_NOCOVR_RECORD => {
                if let Some(covr) = parse_covr_payload(&mut p, payload_len) {
                    raw_nocovr.push(covr);
                } else {
                    tracing::warn!("CELL_NOCOVR_RECORD: could not parse no-coverage polygon");
                }
            }

            CELL_TXTDSC_INFO_FILE_RECORD => {
                // Payload: u32 name_len, u32 content_len, <name_len bytes filename>, <content>
                if payload_len >= 8 {
                    if let (Ok(name_len), Ok(_content_len)) = (read_u32(&mut p), read_u32(&mut p)) {
                        let name_len = name_len as usize;
                        let fname = read_cstring(&mut p, name_len).unwrap_or_default();
                        // Content starts after the name field (possibly past its null terminator)
                        let consumed = 8 + name_len;
                        if payload_len > consumed && !fname.is_empty() {
                            let content_raw = &payload_bytes[consumed..];
                            let content = String::from_utf8_lossy(content_raw)
                                .trim_end_matches('\0')
                                .to_owned();
                            if !content.is_empty() {
                                text_descriptions.insert(fname, content);
                            }
                        }
                    }
                } else {
                    tracing::warn!(payload_len, "CELL_TXTDSC_INFO_FILE_RECORD too short");
                }
            }

            // ── Features ─────────────────────────────────────────────────────
            FEATURE_ID_RECORD => {
                if let Some(f) = current.take() {
                    raw_features.push(f);
                }
                if payload_len >= 5 {
                    let type_code = read_u16(&mut p)?;
                    let id = read_u16(&mut p)?;
                    let primitive = read_u8(&mut p)?;
                    current = Some(RawFeature {
                        type_code,
                        id,
                        primitive,
                        attributes: Vec::new(),
                        raw_geometry: RawGeometry::None,
                    });
                }
            }

            FEATURE_ATTRIBUTE_RECORD => {
                if payload_len < 3 {
                    tracing::warn!(payload_len, "FEATURE_ATTRIBUTE_RECORD too short");
                    continue;
                }
                let attr_code = read_u16(&mut p)?;
                let value_type = read_u8(&mut p)?;
                let value = match value_type {
                    0 => {
                        if payload_len >= 7 {
                            AttrValue::Int(read_u32(&mut p)?)
                        } else {
                            tracing::warn!(attr_code, "int attribute payload too short");
                            continue;
                        }
                    }
                    2 => {
                        if payload_len >= 11 {
                            AttrValue::Double(read_f64(&mut p)?)
                        } else {
                            tracing::warn!(attr_code, "double attribute payload too short");
                            continue;
                        }
                    }
                    4 => {
                        let remaining = payload_len - 3;
                        AttrValue::Str(read_cstring(&mut p, remaining)?)
                    }
                    other => {
                        // Types 1 (int list) and 3 (double list) are also unimplemented in OpenCPN.
                        tracing::warn!(
                            attr_code,
                            value_type = other,
                            "unhandled attribute value type"
                        );
                        continue;
                    }
                };
                if let Some(f) = &mut current {
                    f.attributes.push(Attribute {
                        code: attr_code,
                        value,
                    });
                }
            }

            // ── Feature geometry (standard float coords) ─────────────────────
            FEATURE_GEOMETRY_RECORD_POINT => {
                if payload_len >= 16 {
                    let lat = read_f64(&mut p)?;
                    let lon = read_f64(&mut p)?;
                    if let Some(f) = &mut current {
                        f.raw_geometry = RawGeometry::Point(point![x: lon, y: lat]);
                    }
                } else {
                    tracing::warn!(payload_len, "GEOM_POINT too short");
                }
            }

            FEATURE_GEOMETRY_RECORD_LINE => {
                if payload_len >= 36 {
                    // 4×f64 extent (unused), u32 edge_count, then count×4×i32 edge refs
                    let _s = read_f64(&mut p)?;
                    let _n = read_f64(&mut p)?;
                    let _w = read_f64(&mut p)?;
                    let _e = read_f64(&mut p)?;
                    let count = read_u32(&mut p)? as usize;
                    let mut edge_refs = Vec::with_capacity(count);
                    for _ in 0..count {
                        if p.position() as usize + 16 > payload_len {
                            break;
                        }
                        let start_node = read_i32(&mut p)?;
                        let edge_id = read_i32(&mut p)?;
                        let end_node = read_i32(&mut p)?;
                        let dir = read_i32(&mut p)?;
                        edge_refs.push([start_node, edge_id, end_node, dir]);
                    }
                    if let Some(f) = &mut current {
                        f.raw_geometry = RawGeometry::Line(edge_refs);
                    }
                } else {
                    tracing::warn!(payload_len, "GEOM_LINE too short");
                }
            }

            FEATURE_GEOMETRY_RECORD_AREA => {
                match parse_area_payload(&mut p, payload_len, false, 0.0, ref_lat, ref_lon) {
                    Ok(raw) => {
                        if let Some(f) = &mut current {
                            f.raw_geometry = raw;
                        }
                    }
                    Err(e) => tracing::warn!("GEOM_AREA parse error: {e:#}"),
                }
            }

            FEATURE_GEOMETRY_RECORD_AREA_EXT => {
                // Header has an extra f64 scale_factor after the standard 44-byte header.
                // TriPrim bbox and vertices use i16 SM coords divided by scale_factor.
                // CELL_EXTENT_RECORD always precedes feature geometry in the stream,
                // so ref_lat/ref_lon are valid here for bbox SM→WGS84 conversion.
                if payload_len < 52 {
                    tracing::warn!(payload_len, "GEOM_AREA_EXT too short");
                    continue;
                }
                // Read the scale_factor from the EXT header (at offset 44 in payload)
                let mut hdr_p = Cursor::new(payload_bytes.as_slice());
                for _ in 0..5 {
                    let _ = read_f64(&mut hdr_p);
                } // skip 4×f64 extent
                let _ = read_u32(&mut hdr_p); // contour_count
                let _ = read_u32(&mut hdr_p); // triprim_count
                let _ = read_u32(&mut hdr_p); // edge_count
                let scale_factor = match read_f64(&mut hdr_p) {
                    Ok(sf) => sf,
                    Err(e) => {
                        tracing::warn!("GEOM_AREA_EXT: failed to read scale_factor: {e:#}");
                        continue;
                    }
                };
                match parse_area_payload(&mut p, payload_len, true, scale_factor, ref_lat, ref_lon)
                {
                    Ok(raw) => {
                        if let Some(f) = &mut current {
                            f.raw_geometry = raw;
                        }
                    }
                    Err(e) => tracing::warn!("GEOM_AREA_EXT parse error: {e:#}"),
                }
            }

            FEATURE_GEOMETRY_RECORD_MULTIPOINT => {
                if payload_len >= 36 {
                    // 4×f64 extent (unused), u32 count, then count×3×f32 (east, north, depth)
                    let _s = read_f64(&mut p)?;
                    let _n = read_f64(&mut p)?;
                    let _w = read_f64(&mut p)?;
                    let _e = read_f64(&mut p)?;
                    let count = read_u32(&mut p)? as usize;
                    let mut pts = Vec::with_capacity(count);
                    for _ in 0..count {
                        if p.position() as usize + 12 > payload_len {
                            break;
                        }
                        let east = read_f32(&mut p)?;
                        let north = read_f32(&mut p)?;
                        let depth = read_f32(&mut p)?;
                        pts.push([east, north, depth]);
                    }
                    if let Some(f) = &mut current {
                        f.raw_geometry = RawGeometry::Sounding(pts);
                    }
                } else {
                    tracing::warn!(payload_len, "GEOM_MULTIPOINT too short");
                }
            }

            // ── Vector tables (standard f32 coords) ──────────────────────────
            VECTOR_EDGE_NODE_TABLE_RECORD => {
                // u32 n_edges; for each: u32 edge_index, u32 point_count, count×2×f32
                if payload_len < 4 {
                    tracing::warn!(payload_len, "VET record too short");
                    continue;
                }
                let n_edges = read_u32(&mut p)? as usize;
                for _ in 0..n_edges {
                    if p.position() as usize + 8 > payload_len {
                        break;
                    }
                    let edge_index = read_u32(&mut p)?;
                    let point_count = read_u32(&mut p)? as usize;
                    let mut points = Vec::with_capacity(point_count);
                    for _ in 0..point_count {
                        if p.position() as usize + 8 > payload_len {
                            break;
                        }
                        let east = f64::from(read_f32(&mut p)?);
                        let north = f64::from(read_f32(&mut p)?);
                        points.push([east, north]);
                    }
                    vet.insert(edge_index, EdgeEntry { points });
                }
            }

            VECTOR_CONNECTED_NODE_TABLE_RECORD => {
                // u32 n_nodes; for each: u32 node_index, 2×f32 (east, north)
                if payload_len < 4 {
                    tracing::warn!(payload_len, "VCT record too short");
                    continue;
                }
                let n_nodes = read_u32(&mut p)? as usize;
                for _ in 0..n_nodes {
                    if p.position() as usize + 12 > payload_len {
                        break;
                    }
                    let node_index = read_u32(&mut p)?;
                    let east = f64::from(read_f32(&mut p)?);
                    let north = f64::from(read_f32(&mut p)?);
                    vct.insert(
                        node_index,
                        NodeEntry {
                            lon: east,
                            lat: north,
                        },
                    );
                }
            }

            // ── Vector tables (extended i16 coords) ──────────────────────────
            VECTOR_EDGE_NODE_TABLE_EXT_RECORD => {
                // Payload: f64 scale_factor, u32 n_edges;
                // for each: u32 edge_index, u32 point_count, count×2×i16
                if payload_len < 12 {
                    tracing::warn!(payload_len, "VET_EXT record too short");
                    continue;
                }
                let scale_factor = read_f64(&mut p)?;
                let n_edges = read_u32(&mut p)? as usize;
                for _ in 0..n_edges {
                    if p.position() as usize + 8 > payload_len {
                        break;
                    }
                    let edge_index = read_u32(&mut p)?;
                    let point_count = read_u32(&mut p)? as usize;
                    let mut points = Vec::with_capacity(point_count);
                    for _ in 0..point_count {
                        if p.position() as usize + 4 > payload_len {
                            break;
                        }
                        let east = f64::from(read_i16(&mut p)?) / scale_factor;
                        let north = f64::from(read_i16(&mut p)?) / scale_factor;
                        points.push([east, north]);
                    }
                    vet.insert(edge_index, EdgeEntry { points });
                }
            }

            VECTOR_CONNECTED_NODE_TABLE_EXT_RECORD => {
                // Payload: f64 scale_factor, u32 n_nodes;
                // for each: u32 node_index, 2×i16
                if payload_len < 12 {
                    tracing::warn!(payload_len, "VCT_EXT record too short");
                    continue;
                }
                let scale_factor = read_f64(&mut p)?;
                let n_nodes = read_u32(&mut p)? as usize;
                for _ in 0..n_nodes {
                    if p.position() as usize + 8 > payload_len {
                        break;
                    }
                    let node_index = read_u32(&mut p)?;
                    let east = f64::from(read_i16(&mut p)?) / scale_factor;
                    let north = f64::from(read_i16(&mut p)?) / scale_factor;
                    vct.insert(
                        node_index,
                        NodeEntry {
                            lon: east,
                            lat: north,
                        },
                    );
                }
            }

            // ── Status / unknown ─────────────────────────────────────────────
            SERVER_STATUS_RECORD => {
                // Consumed in the prologue; unexpected if it appears mid-stream.
                tracing::warn!("SERVER_STATUS_RECORD (200) appeared mid-stream, skipping");
            }

            unknown => {
                tracing::warn!(rec_type = unknown, rec_len, "unknown record type, skipping");
            }
        }
    }

    // Push the last in-flight feature.
    if let Some(f) = current.take() {
        raw_features.push(f);
    }

    // ── Resolve SM coords → WGS84 ────────────────────────────────────────────
    let mut resolved_vct: HashMap<u32, Point> = HashMap::with_capacity(vct.len());
    for (idx, node) in &vct {
        resolved_vct.insert(
            *idx,
            crate::georef::from_sm(node.lon, node.lat, ref_lat, ref_lon).into(),
        );
    }

    let mut resolved_vet: HashMap<u32, LineString> = HashMap::with_capacity(vet.len());
    for (idx, edge) in &vet {
        let pts = edge
            .points
            .iter()
            .map(|&[e, n]| crate::georef::from_sm(e, n, ref_lat, ref_lon))
            .collect();
        resolved_vet.insert(*idx, pts);
    }

    // ── Resolve feature geometry ─────────────────────────────────────────────
    let features: Vec<Feature> = raw_features
        .into_iter()
        .map(|raw| {
            let geometry = resolve_geometry(
                raw.raw_geometry,
                ref_lat,
                ref_lon,
                &resolved_vet,
                &resolved_vct,
            );
            Feature {
                type_code: raw.type_code,
                id: raw.id,
                primitive: raw.primitive,
                attributes: raw.attributes,
                geometry,
            }
        })
        .collect();

    // ── Resolve coverage polygons ────────────────────────────────────────────
    // See `parse_covr_payload`'s doc comment: these points are already
    // WGS84 degrees, not SM metres — no from_sm() conversion here.
    let coverage = decode_covr(&raw_covr, &raw_nocovr);
    if coverage.unsigned_area() == 0.0 {
        tracing::warn!(
            raw_covr = raw_covr.len(),
            raw_nocovr = raw_nocovr.len(),
            name,
            "cell has zero coverage area (no COVR record?)"
        );
    }

    Ok(OesuCell {
        name,
        native_scale,
        senc_version,
        publish_date,
        edition,
        update_date,
        update_number,
        senc_create_date,
        sounding_datum,
        expire_days_remaining,
        grace_days_remaining,
        ref_lat,
        ref_lon,
        bounds,
        features,
        coverage,
        text_descriptions,
        source,
    }
    .into())
}

/// Assign each NOCOVR ring as a hole of whichever COVR exterior it falls
/// inside, building the cell's coverage [`MultiPolygon`].
///
/// Overlapping COVR exteriors and NOCOVR rings that match no (or several)
/// COVR exteriors are data-quality issues, not parse failures: they are
/// logged and handled best-effort rather than panicking.
fn decode_covr(covr: &[LineString], no_covr: &[LineString]) -> MultiPolygon {
    let exteriors: Vec<Polygon> = covr
        .iter()
        .map(|ring| Polygon::new(ring.clone(), vec![]))
        .collect();
    let interiors: Vec<Polygon> = no_covr
        .iter()
        .map(|ring| Polygon::new(ring.clone(), vec![]))
        .collect();

    for i in 0..exteriors.len() {
        for j in (i + 1)..exteriors.len() {
            if exteriors[i].intersects(&exteriors[j]) {
                tracing::warn!(i, j, "overlapping COVR exteriors");
            }
        }
    }

    let mut taken_interiors: HashSet<usize> = HashSet::new();
    let polys: Vec<Polygon> = exteriors
        .iter()
        .map(|ext| {
            let holes: Vec<LineString> = interiors
                .iter()
                .enumerate()
                .filter_map(|(index, int)| {
                    if !int.intersects(ext) {
                        return None;
                    }
                    if !taken_interiors.insert(index) {
                        tracing::debug!(index, "NOCOVR ring claimed by multiple COVR exteriors");
                    }
                    Some(int.exterior().clone())
                })
                .collect();
            Polygon::new(ext.exterior().clone(), holes)
        })
        .collect();
    let unclaimed = interiors.len() - taken_interiors.len();
    if unclaimed > 0 {
        tracing::warn!(unclaimed, "NOCOVR rings outside any COVR exterior, ignored");
    }

    MultiPolygon::new(polys)
}

// ── Prologue helpers ─────────────────────────────────────────────────────────

/// Strip and validate the leading `SERVER_STATUS_RECORD` (200).
/// Returns a slice starting at the next record, plus the expiry counters.
#[allow(clippy::items_after_statements)] // PAYLOAD const follows early-exit guards; clearer here
fn strip_server_status(data: &[u8]) -> Result<(&[u8], u16, u16)> {
    if data.len() < 6 {
        bail!(
            "file too short to be a valid SENC stream ({} bytes)",
            data.len()
        );
    }

    let rec_type = u16::from_le_bytes([data[0], data[1]]);
    let rec_len = u32::from_le_bytes([data[2], data[3], data[4], data[5]]) as usize;

    if rec_type != SERVER_STATUS_RECORD {
        // Not a SENC file: check for BSB raster chart signature.
        if data.starts_with(b"!BSB") || data.starts_with(b"BSB/") {
            bail!(
                "file is a BSB raster chart (.oernc), not a vector SENC (.oesu); \
                 raster charts are not supported by this tool"
            );
        }
        bail!(
            "expected SERVER_STATUS_RECORD (200) as first record, found type {rec_type}; \
             the file may be corrupt, still encrypted, or a raster chart (.oernc)"
        );
    }

    // Payload layout (12 bytes):
    //   u16 serverStatus, u16 decryptStatus, u16 expireStatus,
    //   u16 expireDaysRemaining, u16 graceDaysAllowed, u16 graceDaysRemaining
    const PAYLOAD: usize = 12;
    if rec_len < 6 + PAYLOAD {
        bail!(
            "SERVER_STATUS_RECORD too short ({rec_len} bytes, need at least {})",
            6 + PAYLOAD
        );
    }
    if data.len() < rec_len {
        bail!("file truncated inside SERVER_STATUS_RECORD");
    }

    let pl = &data[6..6 + PAYLOAD];
    let server_status = u16::from_le_bytes([pl[0], pl[1]]);
    let decrypt_status = u16::from_le_bytes([pl[2], pl[3]]);
    let expire_status = u16::from_le_bytes([pl[4], pl[5]]);
    let expire_days = u16::from_le_bytes([pl[6], pl[7]]);
    let grace_allowed = u16::from_le_bytes([pl[8], pl[9]]);
    let grace_days = u16::from_le_bytes([pl[10], pl[11]]);

    tracing::debug!(
        server_status,
        decrypt_status,
        expire_status,
        expire_days,
        grace_allowed,
        grace_days,
        "SERVER_STATUS"
    );

    if decrypt_status == 0 {
        bail!(
            "chart decryption/signature failure (decryptStatus=0, serverStatus={server_status}); \
             the chart key may be missing or invalid"
        );
    }
    if expire_status == 0 {
        bail!(
            "chart license has expired \
             (expireStatus=0, expireDays={expire_days}, graceDays={grace_days}); \
             renew your chart subscription to continue using this chart"
        );
    }

    Ok((&data[rec_len..], expire_days, grace_days))
}

/// Read and range-check the SENC version from `HEADER_SENC_VERSION` (record 1).
/// Called immediately after `strip_server_status` so the first bytes of `data`
/// are the version record.
fn read_senc_version(data: &[u8]) -> Result<u16> {
    if data.len() < 8 {
        bail!("file too short to contain a SENC version record");
    }

    let rec_type = u16::from_le_bytes([data[0], data[1]]);
    let rec_len = u32::from_le_bytes([data[2], data[3], data[4], data[5]]);

    if rec_type != HEADER_SENC_VERSION {
        if data.starts_with(b"!BSB") || data.starts_with(b"BSB/") {
            bail!(
                "expected SENC version record after SERVER_STATUS, found BSB raster data; \
                 this appears to be an .oernc raster chart, which is not supported"
            );
        }
        bail!(
            "expected HEADER_SENC_VERSION (type 1) after server status, found type {rec_type}; \
             the file may be corrupt or in an unknown format"
        );
    }
    if rec_len < 8 {
        bail!("HEADER_SENC_VERSION record too short ({rec_len} bytes, need 8)");
    }

    let version = u16::from_le_bytes([data[6], data[7]]);

    if version == 1024 {
        bail!(
            "SENC version 1024 is a signature-failure sentinel; \
             the chart key is invalid or the file is corrupt"
        );
    }
    if !(200..=299).contains(&version) {
        bail!(
            "unsupported SENC version {version} (valid range is 200–299); \
             this file requires a newer parser or is in an incompatible format"
        );
    }

    tracing::debug!(version, "SENC version OK");
    Ok(version)
}

// ── Record sub-parsers ───────────────────────────────────────────────────────

/// Parse a COVR or NOCOVR payload: `u32 point_count`, then `count × 2 × f32`
/// `(lat_deg, lon_deg)` pairs in WGS84 degrees — already final units, no
/// SM→WGS84 resolution needed (unlike VET/VCT), just a `[lon, lat]` reorder
/// and widen to `f64`.
fn parse_covr_payload(p: &mut Cursor<&[u8]>, payload_len: usize) -> Option<LineString> {
    if payload_len < 4 {
        return None;
    }
    let count = read_u32(p).ok()? as usize;
    let mut coords = Vec::with_capacity(count);
    for _ in 0..count {
        let lat = read_f32(p).ok()?;
        let lon = read_f32(p).ok()?;
        coords.push(Coord {
            x: f64::from(lon),
            y: f64::from(lat),
        });
    }
    Some(LineString::new(coords))
}

/// Parse an area geometry payload (records 82 and 84).
///
/// If `ext` is true, the payload layout differs:
///   - the fixed header is 52 bytes instead of 44 (extra `f64 scale_factor`)
///   - `TriPrim` bbox uses 4×i16 SM coords (scaled by `scale_factor`, then `fromSM`)
///   - `TriPrim` vertices use `nvert × 2 × i16` (divided by `scale_factor`)
///
/// For non-EXT records `scale_factor` and `ref_lat`/`ref_lon` are ignored.
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
fn parse_area_payload(
    p: &mut Cursor<&[u8]>,
    payload_len: usize,
    ext: bool,
    scale_factor: f64,
    ref_lat: f64,
    ref_lon: f64,
) -> Result<RawGeometry> {
    let min_len = if ext { 52 } else { 44 };
    anyhow::ensure!(
        payload_len >= min_len,
        "area payload too short ({payload_len} bytes)"
    );

    // 4×f64 extent (informational, not used for geometry)
    let _s = read_f64(p)?;
    let _n = read_f64(p)?;
    let _w = read_f64(p)?;
    let _e = read_f64(p)?;

    let contour_count = read_u32(p)?;
    let triprim_count = read_u32(p)? as usize;
    let edge_count = read_u32(p)? as usize;

    if ext {
        // Consume the scale_factor field; caller already read it to pass in.
        let _ = read_f64(p)?;
    }

    // Per-contour vertex counts (contour_count × u32).
    let n_contours = contour_count as usize;
    let mut vertex_counts = Vec::with_capacity(n_contours);
    for k in 0..n_contours {
        anyhow::ensure!(
            p.position() as usize + 4 <= payload_len,
            "truncated vertex count array at entry {k}/{n_contours}"
        );
        vertex_counts.push(read_u32(p)?);
    }

    // TriPrim chain.
    let mut tri_prims = Vec::with_capacity(triprim_count);
    for k in 0..triprim_count {
        anyhow::ensure!(
            p.position() as usize + 5 <= payload_len,
            "truncated TriPrim header at entry {k}/{triprim_count}"
        );
        let prim_type = read_u8(p)?;
        if TriPrimType::from_u8(prim_type).is_none() {
            tracing::warn!(prim_type, entry = k, "unrecognized TriPrim primitive type");
        }
        let nvert = read_u32(p)? as usize;

        let (bbox, vertices) = if ext {
            // bbox: 4×i16 SM coords [min_east, max_east, min_north, max_north]
            anyhow::ensure!(
                p.position() as usize + 4 * 2 + nvert * 2 * 2 <= payload_len,
                "truncated EXT TriPrim body at entry {k}/{triprim_count} (nvert={nvert})"
            );
            let min_east = f64::from(read_i16(p)?) / scale_factor;
            let max_east = f64::from(read_i16(p)?) / scale_factor;
            let min_north = f64::from(read_i16(p)?) / scale_factor;
            let max_north = f64::from(read_i16(p)?) / scale_factor;
            let min_coord = crate::georef::from_sm(min_east, min_north, ref_lat, ref_lon);
            let max_coord = crate::georef::from_sm(max_east, max_north, ref_lat, ref_lon);
            let bbox = [min_coord.x, min_coord.y, max_coord.x, max_coord.y];

            let mut verts = Vec::with_capacity(nvert);
            for _ in 0..nvert {
                let east = f32::from(read_i16(p)?) / scale_factor as f32;
                let north = f32::from(read_i16(p)?) / scale_factor as f32;
                verts.push([east, north]);
            }
            (bbox, verts)
        } else {
            // bbox: 4×f64 already in WGS84 [min_lon, max_lon, min_lat, max_lat]
            anyhow::ensure!(
                p.position() as usize + 32 + nvert * 8 <= payload_len,
                "truncated TriPrim body at entry {k}/{triprim_count} (nvert={nvert})"
            );
            let minlon = read_f64(p)?;
            let maxlon = read_f64(p)?;
            let minlat = read_f64(p)?;
            let maxlat = read_f64(p)?;
            let bbox = [minlon, minlat, maxlon, maxlat]; // → [W, S, E, N]

            let mut verts = Vec::with_capacity(nvert);
            for _ in 0..nvert {
                let east = read_f32(p)?;
                let north = read_f32(p)?;
                verts.push([east, north]);
            }
            (bbox, verts)
        };

        tri_prims.push(RawTriPrim {
            _prim_type: prim_type,
            _bbox: bbox,
            _vertices: vertices,
        });
    }

    // Edge reference table: edge_count × 4 × i32.
    // The o-charts server always emits 4-int entries (stride 4); the OSENC spec
    // says 3-int entries for version ≤ 200, but all known files are version 201+.
    let cursor = p.position() as usize;
    let remaining = payload_len.saturating_sub(cursor);
    anyhow::ensure!(
        remaining == edge_count * 16,
        "unexpected {remaining} bytes for edge section (expected {})",
        edge_count * 16,
    );

    let mut edge_refs = Vec::with_capacity(edge_count);
    for _ in 0..edge_count {
        let start_node = read_i32(p)?;
        let edge_id = read_i32(p)?;
        let end_node = read_i32(p)?;
        let dir = read_i32(p)?;
        edge_refs.push([start_node, edge_id, end_node, dir]);
    }

    Ok(RawGeometry::Area {
        contour_count,
        vertex_counts,
        tri_prims,
        edge_refs,
    })
}

// ── Geometry resolution ──────────────────────────────────────────────────────

#[allow(clippy::too_many_lines)] // geometry variant dispatch is inherently long
fn resolve_geometry(
    raw: RawGeometry,
    ref_lat: f64,
    ref_lon: f64,
    vet: &HashMap<u32, LineString>,
    vct: &HashMap<u32, Point>,
) -> Geometry {
    match raw {
        RawGeometry::None => Geometry::None,

        RawGeometry::Point(p) => Geometry::Point(p),

        RawGeometry::Sounding(pts) => {
            let resolved = pts
                .iter()
                .map(|[e, n, d]| {
                    let coord =
                        crate::georef::from_sm(f64::from(*e), f64::from(*n), ref_lat, ref_lon);
                    (coord.into(), f64::from(*d))
                })
                .collect();
            Geometry::Soundings(resolved)
        }

        RawGeometry::Line(edge_refs) => {
            let line = build_ring(&edge_refs, vet, vct, false);
            Geometry::Line(line)
        }

        RawGeometry::Area {
            contour_count,
            vertex_counts: _vertex_counts,
            tri_prims: _tri_prims,
            edge_refs,
        } => {
            if edge_refs.is_empty() {
                tracing::warn!(expected = contour_count, "no contours in area geometry");
                return Geometry::None;
            }

            let expected_rings = contour_count as usize;
            let total_edges = edge_refs.len();
            let mut rings: Vec<LineString> = Vec::with_capacity(expected_rings);
            let mut ring_start = 0usize;
            let mut prev_end = edge_refs[0][0];

            for i in 0..total_edges {
                let [start_node, _edge, end_node, _dir] = edge_refs[i];

                if prev_end != start_node {
                    tracing::error!(
                        edge_index = i,
                        end_node,
                        next_start_node = edge_refs.get(i + 1).map(|r| r[0]),
                        "topology break: non-contiguous edge at index {i}"
                    );
                    return Geometry::None;
                }
                prev_end = end_node;

                let is_last = i + 1 == total_edges;
                let ring_closed = end_node == edge_refs[ring_start][0];

                if is_last && !ring_closed {
                    tracing::warn!(
                        edge_index = i,
                        end_node,
                        first_start = edge_refs[ring_start][0],
                        "force-closing unclosed ring at end of edge list"
                    );
                }

                if ring_closed || is_last {
                    let ring_coords = build_ring(&edge_refs[ring_start..=i], vet, vct, true);
                    rings.push(ring_coords);
                    if !is_last {
                        ring_start = i + 1;
                        prev_end = edge_refs[ring_start][0];
                    }
                }
            }

            if rings.len() != expected_rings {
                if expected_rings <= 5 || rings.len().abs_diff(expected_rings) <= 2 {
                    tracing::warn!(
                        expected = expected_rings,
                        got = rings.len(),
                        edge_refs = ?edge_refs,
                        "area ring count mismatch"
                    );
                } else {
                    tracing::warn!(
                        expected = expected_rings,
                        got = rings.len(),
                        "area ring count mismatch"
                    );
                }
            }

            Geometry::Area(Polygon::new(rings[0].clone(), rings[1..].to_vec()))
        }
    }
}

/// Build a coordinate ring from a sequence of edge reference entries.
/// Each entry: `[start_node_rcid, edge_rcid, end_node_rcid, dir]`
/// `close = true` appends the first point at the end (polygon rings).
#[allow(clippy::cast_sign_loss)] // RCID values in start/end fields are always non-negative
fn build_ring(
    edge_refs: &[[i32; 4]],
    vet: &HashMap<u32, LineString>,
    vct: &HashMap<u32, Point>,
    close: bool,
) -> LineString {
    let mut coords: Vec<Coord> = Vec::new();

    for [start_rcid, edge_rcid, _end_rcid, dir] in edge_refs {
        if let Some(&point) = vct.get(&(*start_rcid as u32))
            && (coords.is_empty() || coords.last() != Some(&point.into()))
        {
            coords.push(point.into());
        }

        if *edge_rcid == 0 {
            continue;
        }

        let reverse = *dir == 1;

        if let Some(pts) = vet.get(&(*edge_rcid as u32)) {
            if reverse {
                coords.extend(pts.coords().rev());
            } else {
                coords.extend(pts.coords());
            }
        }
    }

    if let Some([_, _, end_rcid, _]) = edge_refs.last()
        && let Some(&point) = vct.get(&(*end_rcid as u32))
        && coords.last() != Some(&point.into())
    {
        coords.push(point.into());
    }

    if close && coords.len() >= 2 {
        let first = coords[0];
        if coords.last() != Some(&first) {
            coords.push(first);
        }
    }

    LineString::new(coords)
}
