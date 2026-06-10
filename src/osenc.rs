//! OSENC binary format parser.
//!
//! Reads the little-endian packed-struct format used by `OpenCPN` for its
//! internal Simplified ENC files.  Record layout:
//!   u16 `record_type`  +  u32 `record_length` (total, incl. 6-byte header)
//!
//! Two-pass approach:
//!   Pass 1 – collect features (records 64/65/80-83) and cell metadata (100).
//!   Pass 2 – collect VET (96) and VCT (97) tables, then resolve geometry.

use std::collections::HashMap;
use std::io::{Cursor, Read, Seek, SeekFrom};

use anyhow::{Context, Result};

// ── Record type constants ────────────────────────────────────────────────────
const HEADER_SENC_VERSION: u16 = 1;
const HEADER_CELL_NAME: u16 = 2;
const HEADER_CELL_NATIVESCALE: u16 = 7;

const FEATURE_ID_RECORD: u16 = 64;
const FEATURE_ATTRIBUTE_RECORD: u16 = 65;

const FEATURE_GEOMETRY_RECORD_POINT: u16 = 80;
const FEATURE_GEOMETRY_RECORD_LINE: u16 = 81;
const FEATURE_GEOMETRY_RECORD_AREA: u16 = 82;
const FEATURE_GEOMETRY_RECORD_MULTIPOINT: u16 = 83;

const VECTOR_EDGE_NODE_TABLE_RECORD: u16 = 96;
const VECTOR_CONNECTED_NODE_TABLE_RECORD: u16 = 97;

const CELL_EXTENT_RECORD: u16 = 100;

// ── Public data model ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub enum AttrValue {
    Int(u32),
    Double(f64),
    Str(String),
}

#[derive(Debug, Clone)]
pub struct Attribute {
    pub code: u16,
    pub value: AttrValue,
}

#[derive(Debug, Clone)]
pub enum Geometry {
    None,
    Point { lon: f64, lat: f64 },
    MultiPoint(Vec<[f64; 3]>), // [lon, lat, depth]
    Line(Vec<Vec<[f64; 2]>>),  // list of rings/strokes, each a list of [lon, lat]
    Area(Vec<Vec<[f64; 2]>>),  // outer ring + inner rings, each [lon, lat]
}

#[derive(Debug, Clone)]
pub struct Feature {
    pub type_code: u16,
    pub id: u16,
    pub primitive: u8, // GEO_POINT=0, GEO_LINE=1, GEO_AREA=2, GEO_META=3 (matches OpenCPN GeoPrim_t enum)
    pub attributes: Vec<Attribute>,
    pub geometry: Geometry,
}

/// An edge entry from the Vector Edge Node Table (VET, record 96).
#[derive(Debug)]
pub struct EdgeEntry {
    pub points: Vec<[f64; 2]>, // intermediate points only [lon, lat]
}

/// A connected node from the Vector Connected Node Table (VCT, record 97).
#[derive(Debug)]
pub struct NodeEntry {
    pub lon: f64,
    pub lat: f64,
}

#[derive(Debug)]
pub struct OsencCell {
    #[allow(dead_code)]
    pub name: String,
    pub native_scale: u32,
    #[allow(dead_code)]
    pub senc_version: u16,
    /// Reference lat/lon computed from `CELL_EXTENT_RECORD` centroid
    pub ref_lat: f64,
    pub ref_lon: f64,
    /// Geographic bounds [W, S, E, N]
    pub bounds: [f64; 4],
    pub features: Vec<Feature>,
}

// ── Internal parse structures ────────────────────────────────────────────────

/// Unresolved feature – geometry is raw refs, resolved later.
#[derive(Debug)]
struct RawFeature {
    type_code: u16,
    id: u16,
    primitive: u8,
    attributes: Vec<Attribute>,
    raw_geometry: RawGeometry,
}

#[derive(Debug)]
enum RawGeometry {
    None,
    Point { lon: f64, lat: f64 },
    MultiPoint(Vec<[f32; 3]>),           // (east, north, depth) in SM
    Line(Vec<[i32; 3]>),                  // edge ref triples
    Area { contour_count: u32, edge_refs: Vec<[i32; 3]> },
}

// ── Reader helpers ───────────────────────────────────────────────────────────

fn read_u8(c: &mut Cursor<&[u8]>) -> Result<u8> {
    let mut b = [0u8; 1];
    c.read_exact(&mut b)?;
    Ok(b[0])
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
    clippy::cast_possible_truncation, // Cursor positions are bounded by payload_len (≤ u32)
    clippy::similar_names,           // sw/nw/ne/se and resolved_vct/resolved_vet are domain vocab
)]
pub fn parse_file(data: &[u8]) -> Result<OsencCell> {
    let mut name = String::new();
    let mut native_scale: u32 = 0;
    let mut senc_version: u16 = 0;
    let mut ref_lat = 0.0f64;
    let mut ref_lon = 0.0f64;
    let mut bounds = [0.0f64; 4];
    let mut raw_features: Vec<RawFeature> = Vec::new();
    let mut vet: HashMap<u32, EdgeEntry> = HashMap::new();
    let mut vct: HashMap<u32, NodeEntry> = HashMap::new();

    // ── Single pass: collect everything ─────────────────────────────────────
    let mut c = Cursor::new(data);
    let mut current: Option<RawFeature> = None;

    loop {
        // Read 6-byte record header
        let mut hdr = [0u8; 6];
        match c.read_exact(&mut hdr) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(e) => return Err(e.into()),
        }
        let rec_type = u16::from_le_bytes([hdr[0], hdr[1]]);
        let rec_len = u32::from_le_bytes([hdr[2], hdr[3], hdr[4], hdr[5]]);
        if rec_len < 6 {
            break;
        }
        let payload_len = (rec_len - 6) as usize;

        // Read the full payload into a sub-cursor
        let mut payload_bytes = vec![0u8; payload_len];
        c.read_exact(&mut payload_bytes)
            .with_context(|| format!("reading payload of record type {rec_type}"))?;
        let mut p = Cursor::new(payload_bytes.as_slice());

        match rec_type {
            HEADER_SENC_VERSION => {
                senc_version = read_u16(&mut p).unwrap_or(0);
            }
            HEADER_CELL_NAME => {
                name = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_NATIVESCALE => {
                native_scale = read_u32(&mut p).unwrap_or(0);
            }

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
                }
            }

            FEATURE_ID_RECORD => {
                // Push previous feature if any
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
                    continue;
                }
                let attr_code = read_u16(&mut p)?;
                let value_type = read_u8(&mut p)?;
                let value = match value_type {
                    0 => {
                        // u32 integer
                        if payload_len >= 7 {
                            AttrValue::Int(read_u32(&mut p)?)
                        } else {
                            continue;
                        }
                    }
                    2 => {
                        // f64 double
                        if payload_len >= 11 {
                            AttrValue::Double(read_f64(&mut p)?)
                        } else {
                            continue;
                        }
                    }
                    4 => {
                        // null-terminated string; payload bytes 3..end
                        let remaining = payload_len - 3;
                        AttrValue::Str(read_cstring(&mut p, remaining)?)
                    }
                    _ => continue, // types 1 (int list) and 3 (double list) unimplemented in OpenCPN too
                };
                if let Some(ref mut f) = current {
                    f.attributes.push(Attribute { code: attr_code, value });
                }
            }

            FEATURE_GEOMETRY_RECORD_POINT => {
                if payload_len >= 16 {
                    let lat = read_f64(&mut p)?;
                    let lon = read_f64(&mut p)?;
                    if let Some(ref mut f) = current {
                        f.raw_geometry = RawGeometry::Point { lon, lat };
                    }
                }
            }

            FEATURE_GEOMETRY_RECORD_LINE => {
                if payload_len >= 36 {
                    // 4×f64 extent (skip), u32 edge count, then count×3×i32 edge refs
                    let _s = read_f64(&mut p)?;
                    let _n = read_f64(&mut p)?;
                    let _w = read_f64(&mut p)?;
                    let _e = read_f64(&mut p)?;
                    let count = read_u32(&mut p)? as usize;
                    let mut edge_refs = Vec::with_capacity(count);
                    for _ in 0..count {
                        if p.position() as usize + 12 > payload_len {
                            break;
                        }
                        let start_node = read_i32(&mut p)?;
                        let edge_id = read_i32(&mut p)?;
                        let end_node = read_i32(&mut p)?;
                        edge_refs.push([start_node, edge_id, end_node]);
                    }
                    if let Some(ref mut f) = current {
                        f.raw_geometry = RawGeometry::Line(edge_refs);
                    }
                }
            }

            FEATURE_GEOMETRY_RECORD_AREA => {
                if payload_len >= 44 {
                    // 4×f64 extent, u32 contour_count, u32 triprim_count, u32 edge_count
                    let _s = read_f64(&mut p)?;
                    let _n = read_f64(&mut p)?;
                    let _w = read_f64(&mut p)?;
                    let _e = read_f64(&mut p)?;
                    let contour_count = read_u32(&mut p)?;
                    let _triprim_count = read_u32(&mut p)?;
                    let edge_count = read_u32(&mut p)? as usize;

                    // Skip the TriPrim chain that follows the header.
                    // Each TriPrim: u8 type, u32 nVert, 4×f64 bbox, nVert×2×f32 verts
                    // Edge refs are at the very END of the payload.
                    // Seek there directly.
                    let edge_table_bytes = edge_count * 12; // 3×i32 per edge ref
                    let edge_table_start = payload_len.saturating_sub(edge_table_bytes);
                    p.seek(SeekFrom::Start(edge_table_start as u64))?;

                    let mut edge_refs = Vec::with_capacity(edge_count);
                    for _ in 0..edge_count {
                        let start_node = read_i32(&mut p)?;
                        let edge_id = read_i32(&mut p)?;
                        let end_node = read_i32(&mut p)?;
                        edge_refs.push([start_node, edge_id, end_node]);
                    }
                    if let Some(ref mut f) = current {
                        f.raw_geometry = RawGeometry::Area { contour_count, edge_refs };
                    }
                }
            }

            FEATURE_GEOMETRY_RECORD_MULTIPOINT => {
                // 4×f64 extent, u32 count, then count×3×f32 (east, north, depth)
                if payload_len >= 36 {
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
                    if let Some(ref mut f) = current {
                        f.raw_geometry = RawGeometry::MultiPoint(pts);
                    }
                }
            }

            VECTOR_EDGE_NODE_TABLE_RECORD => {
                // u32 nCount; for each: u32 edge_index, u32 point_count, point_count×2×f32
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
                        points.push([east, north]); // SM coords, resolved later
                    }
                    vet.insert(edge_index, EdgeEntry { points });
                }
            }

            VECTOR_CONNECTED_NODE_TABLE_RECORD => {
                // u32 nCount; for each: u32 node_index, 2×f32 (east, north)
                let n_nodes = read_u32(&mut p)? as usize;
                for _ in 0..n_nodes {
                    if p.position() as usize + 12 > payload_len {
                        break;
                    }
                    let node_index = read_u32(&mut p)?;
                    let east = f64::from(read_f32(&mut p)?);
                    let north = f64::from(read_f32(&mut p)?);
                    vct.insert(node_index, NodeEntry { lon: east, lat: north }); // SM for now
                }
            }

            _ => {} // ignore unknown / coverage records
        }
    }

    // Push last feature
    if let Some(f) = current.take() {
        raw_features.push(f);
    }

    // ── Resolve coordinates ──────────────────────────────────────────────────
    // Convert VCT node SM coords → WGS84
    let mut resolved_vct: HashMap<u32, [f64; 2]> = HashMap::with_capacity(vct.len());
    for (idx, node) in &vct {
        resolved_vct.insert(*idx, crate::georef::from_sm(node.lon, node.lat, ref_lat, ref_lon));
    }

    // Convert VET edge SM coords → WGS84
    let mut resolved_vet: HashMap<u32, Vec<[f64; 2]>> = HashMap::with_capacity(vet.len());
    for (idx, edge) in &vet {
        let pts: Vec<[f64; 2]> = edge
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

    Ok(OsencCell {
        name,
        native_scale,
        senc_version,
        ref_lat,
        ref_lon,
        bounds,
        features,
    })
}

// ── Geometry resolution ──────────────────────────────────────────────────────

fn resolve_geometry(
    raw: RawGeometry,
    ref_lat: f64,
    ref_lon: f64,
    vet: &HashMap<u32, Vec<[f64; 2]>>,
    vct: &HashMap<u32, [f64; 2]>,
) -> Geometry {
    match raw {
        RawGeometry::None => Geometry::None,

        RawGeometry::Point { lon, lat } => Geometry::Point { lon, lat },

        RawGeometry::MultiPoint(pts) => {
            let resolved = pts
                .iter()
                .map(|[e, n, d]| {
                    let [lon, lat] =
                        crate::georef::from_sm(f64::from(*e), f64::from(*n), ref_lat, ref_lon);
                    [lon, lat, f64::from(*d)]
                })
                .collect();
            Geometry::MultiPoint(resolved)
        }

        RawGeometry::Line(edge_refs) => {
            // Build a single polyline by chaining edge segments
            let coords = build_ring(&edge_refs, vet, vct, false);
            if coords.is_empty() {
                Geometry::None
            } else {
                Geometry::Line(vec![coords])
            }
        }

        RawGeometry::Area { contour_count, edge_refs } => {
            // Area geometry: edge_refs describe one or more rings.
            // contour_count tells us how many rings; we split at natural
            // start/end node boundaries.
            // Simplification: treat all edge_refs as one outer ring.
            // Most chart cells have simple polygons; inner rings (holes) are rare.
            let _ = contour_count;
            let coords = build_ring(&edge_refs, vet, vct, true);
            if coords.is_empty() {
                Geometry::None
            } else {
                Geometry::Area(vec![coords])
            }
        }
    }
}

/// Build a coordinate ring from a sequence of edge reference triples.
/// Each triple: [`start_node_rcid`, `edge_rcid`, `end_node_rcid`]
/// Negative `edge_rcid` means traverse in reverse.
/// `close` = true appends the first point at the end (for polygons).
#[allow(clippy::cast_sign_loss)] // RCID values in start/end fields are always non-negative
fn build_ring(
    edge_refs: &[[i32; 3]],
    vet: &HashMap<u32, Vec<[f64; 2]>>,
    vct: &HashMap<u32, [f64; 2]>,
    close: bool,
) -> Vec<[f64; 2]> {
    let mut coords: Vec<[f64; 2]> = Vec::new();

    for [start_rcid, edge_rcid, _end_rcid] in edge_refs {
        // Prepend start connected node
        if let Some(&[lon, lat]) = vct.get(&(*start_rcid as u32))
            && (coords.is_empty() || coords.last() != Some(&[lon, lat])) {
                coords.push([lon, lat]);
            }

        if *edge_rcid == 0 {
            continue;
        }

        let reverse = *edge_rcid < 0;
        let eid = edge_rcid.unsigned_abs();

        if let Some(pts) = vet.get(&eid) {
            if reverse {
                coords.extend(pts.iter().rev().copied());
            } else {
                coords.extend(pts.iter().copied());
            }
        }
    }

    // Append final end node from the last edge ref
    if let Some([_, _, end_rcid]) = edge_refs.last()
        && let Some(&[lon, lat]) = vct.get(&(*end_rcid as u32))
            && coords.last() != Some(&[lon, lat]) {
                coords.push([lon, lat]);
            }

    // Close polygon ring
    if close && coords.len() >= 2 {
        let first = coords[0];
        if coords.last() != Some(&first) {
            coords.push(first);
        }
    }

    coords
}
