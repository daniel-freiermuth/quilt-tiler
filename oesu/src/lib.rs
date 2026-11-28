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

mod georef;

use std::collections::HashMap;
use std::io::{Cursor, Read};

use geo::{Area, Coord, LineString, MultiPolygon, Point, Polygon, point};
use s57::{AttrValue, Attribute, Feature, Geometry};

// ── Errors ───────────────────────────────────────────────────────────────────

/// Why a SENC stream could not be parsed.
///
/// Variants split into two groups:
/// - **Domain refusals** ([`Self::is_refusal`]): the file is intact, but this
///   tool must not or cannot use it (expired licence, decryption/signature
///   failure, raster chart, unsupported SENC version). Retrying the same file
///   never helps; the user has to act (renew, re-download, re-key).
/// - **Corrupt or truncated data**: the stream is malformed. Recovery: skip
///   the file; other cells are unaffected.
#[derive(Debug, thiserror::Error)]
pub enum OesuError {
    /// `expireStatus == 0`. Recovery: renew the chart subscription.
    #[error(
        "chart license has expired (expireDays={expire_days}, graceDays={grace_days}); \
         renew your chart subscription to continue using this chart"
    )]
    LicenseExpired { expire_days: u16, grace_days: u16 },
    /// `decryptStatus == 0`. Recovery: install a valid chart key.
    #[error(
        "chart decryption/signature failure (decryptStatus=0, serverStatus={server_status}); \
         the chart key may be missing or invalid"
    )]
    DecryptionFailed { server_status: u16 },
    /// SENC version 1024: the decrypter's signature-failure sentinel.
    /// Recovery: install a valid chart key or re-download the chart.
    #[error(
        "SENC version 1024 is a signature-failure sentinel; \
         the chart key is invalid or the file is corrupt"
    )]
    SignatureFailure,
    /// A BSB raster chart (`.oernc`) was passed instead of a vector SENC.
    /// Recovery: none in this tool; raster charts are not supported.
    #[error(
        "file is a BSB raster chart (.oernc), not a vector SENC (.oesu); \
         raster charts are not supported by this tool"
    )]
    UnsupportedRaster,
    /// SENC version outside 200–299. Recovery: a newer parser is required.
    #[error(
        "unsupported SENC version {0} (valid range is 200–299); \
         this file requires a newer parser or is in an incompatible format"
    )]
    UnsupportedVersion(u16),
    /// The prologue's first or second record is not the required type.
    #[error(
        "expected record type {expected}, found type {found}; \
         the file may be corrupt, still encrypted, or in an unknown format"
    )]
    UnexpectedRecord { expected: u16, found: u16 },
    /// A record declares more payload than the stream still holds.
    #[error(
        "record type {record_type} declares {declared} payload bytes but only {remaining} remain"
    )]
    Truncated {
        record_type: u16,
        declared: usize,
        remaining: usize,
    },
    /// A record whose fields must all be present is shorter than its layout.
    #[error("{record} too short ({len} bytes, need {need})")]
    RecordTooShort {
        record: &'static str,
        len: usize,
        need: usize,
    },
    /// An area payload's count-driven `section` runs past the payload end.
    #[error("{record}: truncated {section} at entry {entry}/{count}")]
    TruncatedSection {
        record: &'static str,
        section: &'static str,
        entry: usize,
        count: usize,
    },
    /// An area payload's edge section doesn't hold `edge_count` entries.
    #[error("unexpected {remaining} bytes for edge section (expected {expected})")]
    EdgeSectionMismatch { remaining: usize, expected: usize },
    /// `CELL_EXTENT_RECORD` has a non-finite or out-of-range latitude.
    #[error("CELL_EXTENT_RECORD latitude {0} is not a valid latitude")]
    InvalidExtentLatitude(f64),
    /// `CELL_EXTENT_RECORD` has a non-finite longitude.
    #[error("CELL_EXTENT_RECORD longitude {0} is not finite")]
    InvalidExtentLongitude(f64),
    /// `record` needs the cell's reference point, but arrived before
    /// `CELL_EXTENT_RECORD`.
    #[error("{record} precedes CELL_EXTENT_RECORD")]
    RecordBeforeExtent { record: &'static str },
    /// The stream ended without a non-zero `HEADER_CELL_NATIVESCALE`.
    #[error("cell has no HEADER_CELL_NATIVESCALE record (or scale is zero)")]
    MissingNativeScale,
    /// The stream ended without a `CELL_EXTENT_RECORD`.
    #[error("cell has no CELL_EXTENT_RECORD")]
    MissingCellExtent,
    /// A field read ran past the end of its record or stream.
    #[error("reading record data: {0}")]
    Io(#[from] std::io::Error),
}

impl OesuError {
    /// `true` for domain refusals — an intact file this tool must not or
    /// cannot use — as opposed to corrupt or truncated data.
    #[must_use]
    pub const fn is_refusal(&self) -> bool {
        matches!(
            self,
            Self::LicenseExpired { .. }
                | Self::DecryptionFailed { .. }
                | Self::SignatureFailure
                | Self::UnsupportedRaster
                | Self::UnsupportedVersion(_)
        )
    }
}

/// Result of SENC parsing; the error defaults to [`OesuError`].
pub type Result<T, E = OesuError> = std::result::Result<T, E>;

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

#[allow(dead_code)] // fields parsed for cursor advancement; only name/scale/bounds/features/dates are forwarded to S57Cell
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
        // `update_date` mirrors `publish_date` until a real update is
        // applied (then it's strictly newer) — see oesu/src/lib.rs's own
        // header parsing. Prefer it; fall back when a malformed cell omits
        // the UPDATEDATE record entirely.
        let raw = if c.update_date.is_empty() {
            &c.publish_date
        } else {
            &c.update_date
        };
        let edition_date = if raw.is_empty() {
            s57::EditionDate::unknown()
        } else {
            s57::EditionDate::parse_ccyymmdd(raw).unwrap_or_else(|| {
                tracing::warn!(
                    cell = %c.source,
                    raw = %raw,
                    "unparseable edition date; ranking as oldest"
                );
                s57::EditionDate::unknown()
            })
        };
        Self {
            name: c.name,
            native_scale: c.native_scale,
            bounds: c.bounds,
            features: c.features,
            coverage: c.coverage,
            source: c.source,
            text_descriptions: c.text_descriptions,
            edition_date,
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
    /// so an EXT area arriving before `CELL_EXTENT_RECORD` is rejected.
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
        _vertex_counts: Vec<u32>,
        _tri_prims: Vec<RawTriPrim>,
        edge_refs: Vec<[i32; 4]>,
    },
}

/// SM reference point: the centre of the cell's `CELL_EXTENT_RECORD`.
///
/// Every SM-relative coordinate in the stream (VET/VCT nodes, soundings,
/// EXT area `TriPrim` bboxes) is georeferenced against it.
#[derive(Clone, Copy, Debug, PartialEq)]
struct GeoRef {
    lat: f64,
    lon: f64,
}

/// Decoded `CELL_EXTENT_RECORD`: SM reference point plus WGS84 bounds.
#[derive(Clone, Copy, Debug, PartialEq)]
struct CellExtent {
    georef: GeoRef,
    /// `[west, south, east, north]` in WGS84 degrees.
    bounds: [f64; 4],
}

// ── Record accumulator ──────────────────────────────────────────────────────

/// Accumulator for record-by-record parsing of an OESU binary stream.
///
/// Each `parse_*` method handles one record type from the main dispatch loop,
/// keeping the loop body a clean one-line-per-arm dispatcher.
#[derive(Default)]
struct CellBuilder {
    name: String,
    native_scale: u32,
    publish_date: String,
    edition: u16,
    update_date: String,
    update_number: u16,
    senc_create_date: String,
    sounding_datum: String,
    /// `None` until `CELL_EXTENT_RECORD` is seen; deliberately not defaulted
    /// to 0/0, which would silently georeference the cell at Null Island.
    extent: Option<CellExtent>,
    raw_features: Vec<RawFeature>,
    raw_covr: Vec<LineString>,
    raw_nocovr: Vec<LineString>,
    text_descriptions: HashMap<String, String>,
    vet: HashMap<u32, EdgeEntry>,
    vct: HashMap<u32, NodeEntry>,
    current: Option<RawFeature>,
}

#[allow(clippy::cast_possible_truncation)] // Cursor positions and counts bounded by payload_len (≤ u32)
impl CellBuilder {
    /// `CELL_EXTENT_RECORD` (100): 8 × f64 corner coordinates.
    #[allow(clippy::similar_names)] // sw/nw/ne/se are domain vocabulary
    fn parse_cell_extent(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        if payload_len < 64 {
            return Err(OesuError::RecordTooShort {
                record: "CELL_EXTENT_RECORD",
                len: payload_len,
                need: 64,
            });
        }
        let sw_lat = read_f64(p)?;
        let sw_lon = read_f64(p)?;
        let nw_lat = read_f64(p)?;
        let nw_lon = read_f64(p)?;
        let ne_lat = read_f64(p)?;
        let ne_lon = read_f64(p)?;
        let se_lat = read_f64(p)?;
        let se_lon = read_f64(p)?;
        for lat in [sw_lat, nw_lat, ne_lat, se_lat] {
            if !(lat.is_finite() && (-90.0..=90.0).contains(&lat)) {
                return Err(OesuError::InvalidExtentLatitude(lat));
            }
        }
        for lon in [sw_lon, nw_lon, ne_lon, se_lon] {
            if !lon.is_finite() {
                return Err(OesuError::InvalidExtentLongitude(lon));
            }
        }

        let s_lat = sw_lat.min(se_lat);
        let n_lat = nw_lat.max(ne_lat);
        let w_lon = sw_lon.min(nw_lon);
        let e_lon = ne_lon.max(se_lon);

        self.extent = Some(CellExtent {
            georef: GeoRef {
                lat: f64::midpoint(n_lat, s_lat),
                lon: f64::midpoint(e_lon, w_lon),
            },
            bounds: [w_lon, s_lat, e_lon, n_lat],
        });
        Ok(())
    }

    /// The SM reference point, or an error naming `record` if the stream has
    /// not delivered `CELL_EXTENT_RECORD` yet.
    fn georef(&self, record: &'static str) -> Result<GeoRef> {
        self.extent
            .map(|e| e.georef)
            .ok_or(OesuError::RecordBeforeExtent { record })
    }

    /// `CELL_TXTDSC_INFO_FILE_RECORD` (101): embedded text-description file.
    fn parse_txtdsc(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) {
        let (Ok(name_len), Ok(_content_len)) = (read_u32(p), read_u32(p)) else {
            tracing::warn!(payload_len, "CELL_TXTDSC_INFO_FILE_RECORD too short");
            return;
        };
        let name_len = name_len as usize;
        let fname = read_cstring(p, name_len).unwrap_or_default();
        let consumed = 8 + name_len;
        if !fname.is_empty()
            && let Some(content_raw) = p.get_ref().get(consumed..)
        {
            let content = String::from_utf8_lossy(content_raw)
                .trim_end_matches('\0')
                .to_owned();
            if !content.is_empty() {
                self.text_descriptions.insert(fname, content);
            }
        }
    }

    /// `FEATURE_ID_RECORD` (64): start of a new feature.
    fn parse_feature_id(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        if let Some(f) = self.current.take() {
            self.raw_features.push(f);
        }
        if payload_len >= 5 {
            let type_code = read_u16(p)?;
            let id = read_u16(p)?;
            let primitive = read_u8(p)?;
            self.current = Some(RawFeature {
                type_code,
                id,
                primitive,
                attributes: Vec::new(),
                raw_geometry: RawGeometry::None,
            });
        }
        Ok(())
    }

    /// `FEATURE_ATTRIBUTE_RECORD` (65): attribute key-value for the current feature.
    fn parse_feature_attribute(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        if payload_len < 3 {
            tracing::warn!(payload_len, "FEATURE_ATTRIBUTE_RECORD too short");
            return Ok(());
        }
        let attr_code = read_u16(p)?;
        let value_type = read_u8(p)?;
        let value = match value_type {
            0 => {
                if payload_len >= 7 {
                    AttrValue::Int(read_u32(p)?)
                } else {
                    tracing::warn!(attr_code, "int attribute payload too short");
                    return Ok(());
                }
            }
            2 => {
                if payload_len >= 11 {
                    AttrValue::Double(read_f64(p)?)
                } else {
                    tracing::warn!(attr_code, "double attribute payload too short");
                    return Ok(());
                }
            }
            4 => {
                let remaining = payload_len - 3;
                AttrValue::Str(read_cstring(p, remaining)?)
            }
            other => {
                // Types 1 (int list) and 3 (double list) are also unimplemented in OpenCPN.
                tracing::warn!(
                    attr_code,
                    value_type = other,
                    "unhandled attribute value type"
                );
                return Ok(());
            }
        };
        if let Some(f) = &mut self.current {
            f.attributes.push(Attribute {
                code: attr_code,
                value,
            });
        }
        Ok(())
    }

    /// `FEATURE_GEOMETRY_RECORD_POINT` (80).
    fn parse_geometry_point(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        if payload_len < 16 {
            tracing::warn!(payload_len, "GEOM_POINT too short");
            return Ok(());
        }
        let lat = read_f64(p)?;
        let lon = read_f64(p)?;
        if !(lat.is_finite() && lon.is_finite()) {
            tracing::warn!(lat, lon, "GEOM_POINT has non-finite coordinates, skipping");
            return Ok(());
        }
        if let Some(f) = &mut self.current {
            f.raw_geometry = RawGeometry::Point(point![x: lon, y: lat]);
        }
        Ok(())
    }

    /// `FEATURE_GEOMETRY_RECORD_LINE` (81): edge-reference line geometry.
    fn parse_geometry_line(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        if payload_len < 36 {
            tracing::warn!(payload_len, "GEOM_LINE too short");
            return Ok(());
        }
        // 4×f64 extent (unused), u32 edge_count, then count×4×i32 edge refs
        let _s = read_f64(p)?;
        let _n = read_f64(p)?;
        let _w = read_f64(p)?;
        let _e = read_f64(p)?;
        let count = read_u32(p)? as usize;
        let mut edge_refs = Vec::new();
        for _ in 0..count {
            if p.position() as usize + 16 > payload_len {
                break;
            }
            let start_node = read_i32(p)?;
            let edge_id = read_i32(p)?;
            let end_node = read_i32(p)?;
            let dir = read_i32(p)?;
            edge_refs.push([start_node, edge_id, end_node, dir]);
        }
        if let Some(f) = &mut self.current {
            f.raw_geometry = RawGeometry::Line(edge_refs);
        }
        Ok(())
    }

    /// `FEATURE_GEOMETRY_RECORD_AREA_EXT` (84): extended area with i16 SM coords.
    fn parse_geometry_area_ext(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        // Read scale_factor from offset 44: skip 4×f64 extent + 3×u32 counts.
        // It is the last header field, so failing to read it means the
        // payload is shorter than the 52-byte EXT header.
        let mut hdr_p = Cursor::new(*p.get_ref());
        for _ in 0..4 {
            let _ = read_f64(&mut hdr_p);
        }
        let _ = read_u32(&mut hdr_p); // contour_count
        let _ = read_u32(&mut hdr_p); // triprim_count
        let _ = read_u32(&mut hdr_p); // edge_count
        let Ok(scale_factor) = read_f64(&mut hdr_p) else {
            tracing::warn!(payload_len, "GEOM_AREA_EXT too short");
            return Ok(());
        };
        let georef = self.georef("FEATURE_GEOMETRY_RECORD_AREA_EXT")?;
        match parse_area_payload(p, payload_len, Some((scale_factor, georef))) {
            Ok(raw) => {
                if let Some(f) = &mut self.current {
                    f.raw_geometry = raw;
                }
            }
            Err(e) => tracing::warn!("GEOM_AREA_EXT parse error: {e:#}"),
        }
        Ok(())
    }

    /// `FEATURE_GEOMETRY_RECORD_MULTIPOINT` (83): sounding point cloud.
    fn parse_geometry_multipoint(
        &mut self,
        p: &mut Cursor<&[u8]>,
        payload_len: usize,
    ) -> Result<()> {
        if payload_len < 36 {
            tracing::warn!(payload_len, "GEOM_MULTIPOINT too short");
            return Ok(());
        }
        // 4×f64 extent (unused), u32 count, then count×3×f32 (east, north, depth)
        let _s = read_f64(p)?;
        let _n = read_f64(p)?;
        let _w = read_f64(p)?;
        let _e = read_f64(p)?;
        let count = read_u32(p)? as usize;
        let mut pts = Vec::new();
        for _ in 0..count {
            if p.position() as usize + 12 > payload_len {
                break;
            }
            let east = read_f32(p)?;
            let north = read_f32(p)?;
            let depth = read_f32(p)?;
            pts.push([east, north, depth]);
        }
        if let Some(f) = &mut self.current {
            f.raw_geometry = RawGeometry::Sounding(pts);
        }
        Ok(())
    }

    /// `VECTOR_EDGE_NODE_TABLE_RECORD` (96): VET with f32 SM coordinates.
    fn parse_vet(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        let Ok(n_edges) = read_u32(p) else {
            tracing::warn!(payload_len, "VET record too short");
            return Ok(());
        };
        let n_edges = n_edges as usize;
        for _ in 0..n_edges {
            if p.position() as usize + 8 > payload_len {
                break;
            }
            let edge_index = read_u32(p)?;
            let point_count = read_u32(p)? as usize;
            let mut points = Vec::new();
            for _ in 0..point_count {
                if p.position() as usize + 8 > payload_len {
                    break;
                }
                let east = f64::from(read_f32(p)?);
                let north = f64::from(read_f32(p)?);
                points.push([east, north]);
            }
            if points.iter().flatten().all(|v| v.is_finite()) {
                self.vet.insert(edge_index, EdgeEntry { points });
            } else {
                tracing::warn!(edge_index, "VET edge has non-finite coordinates, skipping");
            }
        }
        Ok(())
    }

    /// `VECTOR_CONNECTED_NODE_TABLE_RECORD` (97): VCT with f32 SM coordinates.
    fn parse_vct(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        let Ok(n_nodes) = read_u32(p) else {
            tracing::warn!(payload_len, "VCT record too short");
            return Ok(());
        };
        let n_nodes = n_nodes as usize;
        for _ in 0..n_nodes {
            if p.position() as usize + 12 > payload_len {
                break;
            }
            let node_index = read_u32(p)?;
            let east = f64::from(read_f32(p)?);
            let north = f64::from(read_f32(p)?);
            if !(east.is_finite() && north.is_finite()) {
                tracing::warn!(node_index, "VCT node has non-finite coordinates, skipping");
                continue;
            }
            self.vct.insert(
                node_index,
                NodeEntry {
                    lon: east,
                    lat: north,
                },
            );
        }
        Ok(())
    }

    /// `VECTOR_EDGE_NODE_TABLE_EXT_RECORD` (85): VET with i16 scaled SM coordinates.
    fn parse_vet_ext(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        let (Ok(scale_factor), Ok(n_edges)) = (read_f64(p), read_u32(p)) else {
            tracing::warn!(payload_len, "VET_EXT record too short");
            return Ok(());
        };
        if !is_valid_scale_factor(scale_factor) {
            tracing::warn!(scale_factor, "VET_EXT invalid scale_factor, skipping");
            return Ok(());
        }
        let n_edges = n_edges as usize;
        for _ in 0..n_edges {
            if p.position() as usize + 8 > payload_len {
                break;
            }
            let edge_index = read_u32(p)?;
            let point_count = read_u32(p)? as usize;
            let mut points = Vec::new();
            for _ in 0..point_count {
                if p.position() as usize + 4 > payload_len {
                    break;
                }
                let east = f64::from(read_i16(p)?) / scale_factor;
                let north = f64::from(read_i16(p)?) / scale_factor;
                points.push([east, north]);
            }
            self.vet.insert(edge_index, EdgeEntry { points });
        }
        Ok(())
    }

    /// `VECTOR_CONNECTED_NODE_TABLE_EXT_RECORD` (86): VCT with i16 scaled SM coords.
    fn parse_vct_ext(&mut self, p: &mut Cursor<&[u8]>, payload_len: usize) -> Result<()> {
        let (Ok(scale_factor), Ok(n_nodes)) = (read_f64(p), read_u32(p)) else {
            tracing::warn!(payload_len, "VCT_EXT record too short");
            return Ok(());
        };
        if !is_valid_scale_factor(scale_factor) {
            tracing::warn!(scale_factor, "VCT_EXT invalid scale_factor, skipping");
            return Ok(());
        }
        let n_nodes = n_nodes as usize;
        for _ in 0..n_nodes {
            if p.position() as usize + 8 > payload_len {
                break;
            }
            let node_index = read_u32(p)?;
            let east = f64::from(read_i16(p)?) / scale_factor;
            let north = f64::from(read_i16(p)?) / scale_factor;
            self.vct.insert(
                node_index,
                NodeEntry {
                    lon: east,
                    lat: north,
                },
            );
        }
        Ok(())
    }

    /// Push the last in-flight feature (if any) into the accumulator.
    fn flush_current_feature(&mut self) {
        if let Some(f) = self.current.take() {
            self.raw_features.push(f);
        }
    }
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

/// An EXT record's i16 → SM divisor must be a positive finite number;
/// anything else produces infinite/NaN (or mirrored) coordinates.
fn is_valid_scale_factor(scale_factor: f64) -> bool {
    scale_factor.is_finite() && scale_factor > 0.0
}

// ── Main entry point ─────────────────────────────────────────────────────────

/// `true` if `data` is for this parser.
///
/// That is a SENC stream (it starts with the mandatory `SERVER_STATUS_RECORD`)
/// or a BSB raster chart, which [`parse_file`] refuses with
/// [`OesuError::UnsupportedRaster`]. Cheap; [`parse_file`] still validates the rest.
#[must_use]
pub fn detect(data: &[u8]) -> bool {
    data.first_chunk::<2>()
        .is_some_and(|&t| u16::from_le_bytes(t) == SERVER_STATUS_RECORD)
        || is_bsb_raster(data)
}

/// Parse a decrypted SENC stream into an [`s57::S57Cell`].
///
/// # Errors
/// Returns an [`OesuError`]: a domain refusal ([`OesuError::is_refusal`])
/// when the prologue rejects the chart, otherwise a corrupt/truncated-data
/// variant.
#[allow(
    clippy::too_many_lines,           // 196 lines (down from 520); post-processing prevents going below 100
    clippy::cast_possible_truncation, // Cursor positions bounded by payload_len (≤ u32)
    clippy::similar_names,            // resolved_vct/resolved_vet are domain vocab
)]
pub fn parse_file(source: String, data: &[u8]) -> Result<s57::S57Cell> {
    // ── Prologue: validate SERVER_STATUS + version ───────────────────────────
    let (data, expire_days_remaining, grace_days_remaining) = strip_server_status(data)?;
    let senc_version = read_senc_version(data)?;

    let mut b = CellBuilder::default();
    let mut c = Cursor::new(data);

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
        let remaining = data.len().saturating_sub(c.position() as usize);
        if payload_len > remaining {
            return Err(OesuError::Truncated {
                record_type: rec_type,
                declared: payload_len,
                remaining,
            });
        }

        let mut payload_bytes = vec![0u8; payload_len];
        c.read_exact(&mut payload_bytes)?;
        let mut p = Cursor::new(payload_bytes.as_slice());

        match rec_type {
            // ── Header metadata (simple field assignments) ───────────────
            HEADER_SENC_VERSION => { /* already consumed in prologue */ }
            HEADER_CELL_NAME => {
                b.name = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_PUBLISHDATE => {
                b.publish_date = read_cstring(&mut p, payload_len)?;
            }
            // Edition/update numbers have no consumer (only the dates rank
            // cells), so a short record is not worth dropping the cell over.
            HEADER_CELL_EDITION => match read_u16(&mut p) {
                Ok(edition) => b.edition = edition,
                Err(e) => {
                    tracing::warn!(payload_len, "HEADER_CELL_EDITION too short, ignoring: {e}");
                }
            },
            HEADER_CELL_UPDATEDATE => {
                b.update_date = read_cstring(&mut p, payload_len)?;
            }
            HEADER_CELL_UPDATE => match read_u16(&mut p) {
                Ok(update_number) => b.update_number = update_number,
                Err(e) => {
                    tracing::warn!(payload_len, "HEADER_CELL_UPDATE too short, ignoring: {e}");
                }
            },
            HEADER_CELL_NATIVESCALE => {
                b.native_scale = read_u32(&mut p).map_err(|_| OesuError::RecordTooShort {
                    record: "HEADER_CELL_NATIVESCALE",
                    len: payload_len,
                    need: 4,
                })?;
            }
            HEADER_CELL_SENCCREATEDATE => {
                b.senc_create_date = read_cstring(&mut p, payload_len).unwrap_or_default();
            }
            HEADER_CELL_SOUNDINGDATUM => {
                b.sounding_datum = read_cstring(&mut p, payload_len).unwrap_or_default();
            }

            // ── Cell spatial / coverage ──────────────────────────────────
            CELL_EXTENT_RECORD => b.parse_cell_extent(&mut p, payload_len)?,
            CELL_COVR_RECORD => {
                if let Some(covr) = parse_covr_payload(&mut p, payload_len) {
                    b.raw_covr.push(covr);
                } else {
                    tracing::warn!("CELL_COVR_RECORD: could not parse coverage polygon");
                }
            }
            CELL_NOCOVR_RECORD => {
                if let Some(covr) = parse_covr_payload(&mut p, payload_len) {
                    b.raw_nocovr.push(covr);
                } else {
                    tracing::warn!("CELL_NOCOVR_RECORD: could not parse no-coverage polygon");
                }
            }
            CELL_TXTDSC_INFO_FILE_RECORD => b.parse_txtdsc(&mut p, payload_len),

            // ── Feature records ──────────────────────────────────────────
            FEATURE_ID_RECORD => b.parse_feature_id(&mut p, payload_len)?,
            FEATURE_ATTRIBUTE_RECORD => b.parse_feature_attribute(&mut p, payload_len)?,

            // ── Feature geometry ─────────────────────────────────────────
            FEATURE_GEOMETRY_RECORD_POINT => b.parse_geometry_point(&mut p, payload_len)?,
            FEATURE_GEOMETRY_RECORD_LINE => b.parse_geometry_line(&mut p, payload_len)?,
            FEATURE_GEOMETRY_RECORD_AREA => match parse_area_payload(&mut p, payload_len, None) {
                Ok(raw) => {
                    if let Some(f) = &mut b.current {
                        f.raw_geometry = raw;
                    }
                }
                Err(e) => tracing::warn!("GEOM_AREA parse error: {e:#}"),
            },
            FEATURE_GEOMETRY_RECORD_AREA_EXT => {
                b.parse_geometry_area_ext(&mut p, payload_len)?;
            }
            FEATURE_GEOMETRY_RECORD_MULTIPOINT => {
                b.parse_geometry_multipoint(&mut p, payload_len)?;
            }

            // ── Vector tables ────────────────────────────────────────────
            VECTOR_EDGE_NODE_TABLE_RECORD => b.parse_vet(&mut p, payload_len)?,
            VECTOR_CONNECTED_NODE_TABLE_RECORD => b.parse_vct(&mut p, payload_len)?,
            VECTOR_EDGE_NODE_TABLE_EXT_RECORD => b.parse_vet_ext(&mut p, payload_len)?,
            VECTOR_CONNECTED_NODE_TABLE_EXT_RECORD => b.parse_vct_ext(&mut p, payload_len)?,

            // ── Status / unknown ─────────────────────────────────────────
            SERVER_STATUS_RECORD => {
                tracing::warn!("SERVER_STATUS_RECORD (200) appeared mid-stream, skipping");
            }
            unknown => {
                tracing::warn!(rec_type = unknown, rec_len, "unknown record type, skipping");
            }
        }
    }

    if b.native_scale == 0 {
        return Err(OesuError::MissingNativeScale);
    }
    let Some(CellExtent {
        georef: GeoRef {
            lat: ref_lat,
            lon: ref_lon,
        },
        bounds,
    }) = b.extent
    else {
        return Err(OesuError::MissingCellExtent);
    };

    // Push the last in-flight feature.
    b.flush_current_feature();

    // Destructure the builder so post-processing uses plain locals.
    // `current` was consumed by flush; the explicit `_` documents intent.
    #[allow(clippy::unneeded_field_pattern)]
    let CellBuilder {
        name,
        native_scale,
        publish_date,
        edition,
        update_date,
        update_number,
        senc_create_date,
        sounding_datum,
        extent: _,
        raw_features,
        raw_covr,
        raw_nocovr,
        text_descriptions,
        vet,
        vct,
        current: _,
    } = b;

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
    let coverage = decode_covr(&source, &raw_covr, &raw_nocovr);
    if coverage.unsigned_area() == 0.0 {
        tracing::warn!(
            raw_covr = raw_covr.len(),
            raw_nocovr = raw_nocovr.len(),
            source,
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

/// Build the cell's coverage from its COVR (covered) and NOCOVR (not
/// covered) rings: clean up each ring, then resolve them with
/// [`s57::resolve_coverage`], which documents the COVR/NOCOVR taxonomy and the
/// full-erasure guard.
fn decode_covr(source: &str, covr: &[LineString], no_covr: &[LineString]) -> MultiPolygon {
    // OESU COVR/NOCOVR rings arrive with occasional duplicate consecutive
    // coords (zero-length edges) and may not be closed. Deduplicate and
    // close every ring before use, warning when a ring actually needed it
    // (so genuinely clean input stays quiet).
    let normalise = |kind: &str, index: usize, ring: &LineString| -> LineString {
        let mut coords: Vec<Coord> = Vec::new();
        for c in ring.coords() {
            if coords.last() != Some(c) {
                coords.push(*c);
            }
        }
        let removed = ring.0.len() - coords.len();
        if removed > 0 {
            tracing::info!(
                source,
                kind,
                index,
                removed,
                "ring had duplicate consecutive coords, deduplicated"
            );
        }
        if coords.first() != coords.last()
            && let Some(&first) = coords.first()
        {
            coords.push(first);
            tracing::warn!(source, kind, index, "ring was not closed, closed it");
        }
        LineString::new(coords)
    };

    let exteriors: Vec<Polygon> = covr
        .iter()
        .enumerate()
        .map(|(i, ring)| Polygon::new(normalise("COVR", i, ring), vec![]))
        .collect();
    let interiors: Vec<Polygon> = no_covr
        .iter()
        .enumerate()
        .map(|(i, ring)| Polygon::new(normalise("NOCOVR", i, ring), vec![]))
        .collect();

    s57::resolve_coverage(source, &exteriors, &interiors)
}

// ── Prologue helpers ─────────────────────────────────────────────────────────

/// Strip and validate the leading `SERVER_STATUS_RECORD` (200).
/// Returns a slice starting at the next record, plus the expiry counters.
#[allow(clippy::items_after_statements)] // PAYLOAD const follows early-exit guards; clearer here
fn strip_server_status(data: &[u8]) -> Result<(&[u8], u16, u16)> {
    if data.len() < 6 {
        return Err(OesuError::RecordTooShort {
            record: "SERVER_STATUS_RECORD",
            len: data.len(),
            need: 6,
        });
    }

    let mut header = Cursor::new(data);
    let rec_type = read_u16(&mut header)?;
    let rec_len = read_u32(&mut header)? as usize;

    if rec_type != SERVER_STATUS_RECORD {
        // Not a SENC file: check for BSB raster chart signature.
        if is_bsb_raster(data) {
            return Err(OesuError::UnsupportedRaster);
        }
        return Err(OesuError::UnexpectedRecord {
            expected: SERVER_STATUS_RECORD,
            found: rec_type,
        });
    }

    // Payload layout (12 bytes):
    //   u16 serverStatus, u16 decryptStatus, u16 expireStatus,
    //   u16 expireDaysRemaining, u16 graceDaysAllowed, u16 graceDaysRemaining
    const PAYLOAD: usize = 12;
    let too_short = OesuError::RecordTooShort {
        record: "SERVER_STATUS_RECORD",
        len: rec_len,
        need: 6 + PAYLOAD,
    };
    if rec_len < 6 + PAYLOAD {
        return Err(too_short);
    }
    let Some(rest) = data.get(rec_len..) else {
        return Err(OesuError::Truncated {
            record_type: SERVER_STATUS_RECORD,
            declared: rec_len - 6,
            remaining: data.len().saturating_sub(6),
        });
    };

    let mut pl = Cursor::new(data.get(6..6 + PAYLOAD).ok_or(too_short)?);
    let server_status = read_u16(&mut pl)?;
    let decrypt_status = read_u16(&mut pl)?;
    let expire_status = read_u16(&mut pl)?;
    let expire_days = read_u16(&mut pl)?;
    let grace_allowed = read_u16(&mut pl)?;
    let grace_days = read_u16(&mut pl)?;

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
        return Err(OesuError::DecryptionFailed { server_status });
    }
    if expire_status == 0 {
        return Err(OesuError::LicenseExpired {
            expire_days,
            grace_days,
        });
    }

    Ok((rest, expire_days, grace_days))
}

/// `true` if `data` starts with a BSB raster chart (`.oernc`) signature.
fn is_bsb_raster(data: &[u8]) -> bool {
    data.starts_with(b"!BSB") || data.starts_with(b"BSB/")
}

/// Read and range-check the SENC version from `HEADER_SENC_VERSION` (record 1).
/// Called immediately after `strip_server_status` so the first bytes of `data`
/// are the version record.
fn read_senc_version(data: &[u8]) -> Result<u16> {
    if data.len() < 8 {
        return Err(OesuError::RecordTooShort {
            record: "HEADER_SENC_VERSION",
            len: data.len(),
            need: 8,
        });
    }

    let mut c = Cursor::new(data);
    let rec_type = read_u16(&mut c)?;
    let rec_len = read_u32(&mut c)?;

    if rec_type != HEADER_SENC_VERSION {
        if is_bsb_raster(data) {
            return Err(OesuError::UnsupportedRaster);
        }
        return Err(OesuError::UnexpectedRecord {
            expected: HEADER_SENC_VERSION,
            found: rec_type,
        });
    }
    if rec_len < 8 {
        return Err(OesuError::RecordTooShort {
            record: "HEADER_SENC_VERSION",
            len: rec_len as usize,
            need: 8,
        });
    }

    let version = read_u16(&mut c)?;

    if version == 1024 {
        return Err(OesuError::SignatureFailure);
    }
    if !(200..=299).contains(&version) {
        return Err(OesuError::UnsupportedVersion(version));
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
    let mut coords = Vec::new();
    for _ in 0..count {
        let lat = read_f32(p).ok()?;
        let lon = read_f32(p).ok()?;
        if !(lat.is_finite() && lon.is_finite()) {
            return None;
        }
        coords.push(Coord {
            x: f64::from(lon),
            y: f64::from(lat),
        });
    }
    Some(LineString::new(coords))
}

/// Parse an area geometry payload (records 82 and 84).
///
/// `ext` is `Some((scale_factor, georef))` for EXT records, whose layout differs:
///   - the fixed header is 52 bytes instead of 44 (extra `f64 scale_factor`)
///   - `TriPrim` bbox uses 4×i16 SM coords (scaled by `scale_factor`, then `fromSM`
///     against `georef`)
///   - `TriPrim` vertices use `nvert × 2 × i16` (divided by `scale_factor`)
#[allow(
    clippy::too_many_lines,
    clippy::cast_possible_truncation,
    clippy::similar_names
)]
fn parse_area_payload(
    p: &mut Cursor<&[u8]>,
    payload_len: usize,
    ext: Option<(f64, GeoRef)>,
) -> Result<RawGeometry> {
    let (record, min_len) = if ext.is_some() {
        ("FEATURE_GEOMETRY_RECORD_AREA_EXT", 52)
    } else {
        ("FEATURE_GEOMETRY_RECORD_AREA", 44)
    };
    if payload_len < min_len {
        return Err(OesuError::RecordTooShort {
            record,
            len: payload_len,
            need: min_len,
        });
    }
    let truncated = |section, entry, count| OesuError::TruncatedSection {
        record,
        section,
        entry,
        count,
    };

    // 4×f64 extent (informational, not used for geometry)
    let _s = read_f64(p)?;
    let _n = read_f64(p)?;
    let _w = read_f64(p)?;
    let _e = read_f64(p)?;

    let contour_count = read_u32(p)?;
    let triprim_count = read_u32(p)? as usize;
    let edge_count = read_u32(p)? as usize;

    if ext.is_some() {
        // Consume the scale_factor field; caller already read it to pass in.
        let _ = read_f64(p)?;
    }

    // Per-contour vertex counts (contour_count × u32).
    let n_contours = contour_count as usize;
    let mut vertex_counts = Vec::new();
    for k in 0..n_contours {
        if p.position() as usize + 4 > payload_len {
            return Err(truncated("vertex count array", k, n_contours));
        }
        vertex_counts.push(read_u32(p)?);
    }

    // TriPrim chain.
    let mut tri_prims = Vec::new();
    for k in 0..triprim_count {
        if p.position() as usize + 5 > payload_len {
            return Err(truncated("TriPrim header", k, triprim_count));
        }
        let prim_type = read_u8(p)?;
        if TriPrimType::from_u8(prim_type).is_none() {
            tracing::warn!(prim_type, entry = k, "unrecognized TriPrim primitive type");
        }
        let nvert = read_u32(p)? as usize;

        let (bbox, vertices) = if let Some((scale_factor, georef)) = ext {
            // bbox: 4×i16 SM coords [min_east, max_east, min_north, max_north]
            if p.position() as usize + 4 * 2 + nvert * 2 * 2 > payload_len {
                return Err(truncated("TriPrim body", k, triprim_count));
            }
            let min_east = f64::from(read_i16(p)?) / scale_factor;
            let max_east = f64::from(read_i16(p)?) / scale_factor;
            let min_north = f64::from(read_i16(p)?) / scale_factor;
            let max_north = f64::from(read_i16(p)?) / scale_factor;
            let min_coord = crate::georef::from_sm(min_east, min_north, georef.lat, georef.lon);
            let max_coord = crate::georef::from_sm(max_east, max_north, georef.lat, georef.lon);
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
            if p.position() as usize + 32 + nvert * 8 > payload_len {
                return Err(truncated("TriPrim body", k, triprim_count));
            }
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
    if remaining != edge_count * 16 {
        return Err(OesuError::EdgeSectionMismatch {
            remaining,
            expected: edge_count * 16,
        });
    }

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
        _vertex_counts: vertex_counts,
        _tri_prims: tri_prims,
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
            _vertex_counts: _,
            _tri_prims: _,
            edge_refs,
        } => {
            if edge_refs.is_empty() {
                tracing::warn!(expected = contour_count, "no contours in area geometry");
                return Geometry::None;
            }

            // A new ring starts wherever an edge does not begin at the node
            // where the previous edge ended.
            let mut rings = edge_refs
                .chunk_by(|&[_, _, prev_end, _], &[start_node, _, _, _]| prev_end == start_node)
                .map(|ring| build_ring(ring, vet, vct, true));
            let Some(exterior) = rings.next() else {
                return Geometry::None;
            };
            Geometry::Area(Polygon::new(exterior, rings.collect()))
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

    if close
        && coords.len() >= 2
        && let Some(&first) = coords.first()
        && coords.last() != Some(&first)
    {
        coords.push(first);
    }

    LineString::new(coords)
}

#[cfg(test)]
mod tests {
    use super::*;

    trait PolygonExt {
        fn total_rings(&self) -> usize;
    }
    impl PolygonExt for Polygon {
        fn total_rings(&self) -> usize {
            1 + self.interiors().len()
        }
    }

    /// Builds a `vct`/`vet` pair where each node id `n` resolves to the point
    /// `(n as f64, n as f64)` and every edge id is a direct (no-curve) link —
    /// sufficient for ring-topology tests, which only care about node/edge
    /// connectivity, not real coordinates.
    fn node_points(node_ids: &[i32]) -> HashMap<u32, Point> {
        node_ids
            .iter()
            .map(|&n| (n.cast_unsigned(), point!(x: f64::from(n), y: f64::from(n))))
            .collect()
    }

    /// Regression test for a real `.oesu` AREA record whose boundary forms a
    /// pinch: a 7-edge main ring (1374→…→1374) plus a 2-edge spike ring
    /// (27→28→27) that touches the main ring at node 28. Both are valid
    /// boundary rings; both are separated by the chain-break at node 27.
    #[test]
    fn resolve_geometry_pinched_contour_assembles_two_rings() {
        let edge_refs = vec![
            [1374, 1522, 1407, 0],
            [1407, 1722, 1751, 1],
            [1751, 1579, 1750, 1],
            [1750, 1721, 1732, 0],
            [1732, 1746, 28, 0],
            [28, 0, 1770, 0],
            [1770, 40, 1374, 0],
            [27, 33, 28, 1],
            [28, 32, 27, 1],
        ];
        let nodes: Vec<i32> = vec![1374, 1407, 1751, 1750, 1732, 28, 1770, 27];
        let vct = node_points(&nodes);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Area {
            contour_count: 1,
            _vertex_counts: vec![59],
            _tri_prims: Vec::new(),
            edge_refs,
        };

        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Area(polygon) = geometry else {
            panic!("expected Geometry::Area, got {geometry:?}");
        };
        assert_eq!(polygon.total_rings(), 2, "main ring + spike ring");
    }

    /// A continuous chain that revisits its own start vertex stays as one ring
    /// with the chain-break algorithm (no node-revisit splitting).
    #[test]
    fn resolve_geometry_splits_self_touching_chain_into_two_rings() {
        let edge_refs = vec![
            [1, 101, 2, 0],
            [2, 102, 3, 0],
            [3, 103, 1, 0], // closes piece A here
            [1, 104, 4, 0],
            [4, 105, 5, 0],
            [5, 106, 1, 0], // closes piece B here
        ];
        let vct = node_points(&[1, 2, 3, 4, 5]);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Area {
            contour_count: 2,
            _vertex_counts: vec![3, 3],
            _tri_prims: Vec::new(),
            edge_refs,
        };

        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Area(polygon) = geometry else {
            panic!("expected Geometry::Area, got {geometry:?}");
        };
        assert_eq!(
            polygon.total_rings(),
            1,
            "continuous chain-break algorithm produces one ring"
        );
    }

    /// Two adjacent self-loop edges anchored at the same node: chain-break sees
    /// them as one continuous ring (both loops end/start at 662).
    #[test]
    fn resolve_geometry_splits_adjacent_self_loops_sharing_an_anchor() {
        let edge_refs = vec![[662, 810, 662, 0], [662, 811, 662, 0]];
        let vct = node_points(&[662]);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Area {
            contour_count: 2,
            _vertex_counts: vec![1, 1],
            _tri_prims: Vec::new(),
            edge_refs,
        };

        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Area(polygon) = geometry else {
            panic!("expected Geometry::Area, got {geometry:?}");
        };
        assert_eq!(
            polygon.total_rings(),
            1,
            "chain-break merges self-loops that share a start/end node into one ring"
        );
    }

    /// Regression test for a real `.oesu` AREA record where two substantial
    /// rings merely touch at a shared node (906): they must assemble as 5
    /// separate rings, not get merged.
    #[test]
    fn resolve_geometry_keeps_substantial_touching_rings_separate() {
        let edge_refs = vec![
            [906, 0, 466, 1],
            [466, 0, 220, 0],
            [220, 320, 221, 0],
            [221, 321, 222, 0],
            [222, 748, 491, 0],
            [491, 747, 490, 0],
            [490, 746, 489, 0],
            [489, 171, 488, 1],
            [488, 745, 487, 0],
            [487, 744, 486, 1],
            [486, 743, 804, 0],
            [804, 742, 485, 0],
            [485, 0, 905, 1],
            [905, 740, 1002, 0],
            [1002, 680, 964, 0],
            [964, 1183, 845, 1],
            [845, 1124, 906, 1], // closes piece A here (start=906)
            [907, 0, 217, 0],
            [217, 317, 218, 1],
            [218, 318, 219, 1],
            [219, 0, 906, 0],
            [906, 1186, 907, 0], // closes piece B here (start=907)
            [1006, 774, 990, 1],
            [990, 773, 1006, 1],
            [994, 874, 994, 1],
            [745, 1058, 223, 1],
            [223, 315, 216, 1],
            [216, 0, 744, 0],
            [744, 0, 745, 1],
        ];
        let nodes: Vec<i32> = edge_refs.iter().map(|e| e[0]).collect();
        let vct = node_points(&nodes);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Area {
            contour_count: 5,
            _vertex_counts: vec![17, 5, 2, 1, 4],
            _tri_prims: Vec::new(),
            edge_refs,
        };

        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Area(polygon) = geometry else {
            panic!("expected Geometry::Area, got {geometry:?}");
        };
        assert_eq!(
            polygon.total_rings(),
            5,
            "two substantial rings merely touching must stay separate, not merge into one"
        );
    }

    // ── resolve_geometry Line-variant tests ─────────────────────────────────

    /// A single edge with `edge_rcid = 0` (no intermediate VET points) and
    /// `dir = 0` (forward) must produce `[start_node, end_node]` — and
    /// crucially, no closing coordinate (close = false for lines).
    #[test]
    fn resolve_geometry_line_single_edge_no_vet() {
        let edge_refs = vec![[10, 0, 20, 0]];
        let vct = node_points(&[10, 20]);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Line(edge_refs);
        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Line(line) = geometry else {
            panic!("expected Geometry::Line, got {geometry:?}");
        };

        let coords: Vec<Coord> = line.coords().copied().collect();
        assert_eq!(
            coords,
            vec![Coord { x: 10.0, y: 10.0 }, Coord { x: 20.0, y: 20.0 }],
            "single edge → start + end, no closing coord"
        );
    }

    /// A single edge whose `edge_rcid` resolves in the VET must produce
    /// `start_node + edge_coords + end_node` in forward order.
    #[test]
    fn resolve_geometry_line_single_edge_with_vet_forward() {
        let edge_refs = vec![[10, 100, 20, 0]];
        let vct = node_points(&[10, 20]);
        let vet: HashMap<u32, LineString> = [(
            100u32,
            LineString::new(vec![
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 18.0, y: 18.0 },
            ]),
        )]
        .into();

        let raw = RawGeometry::Line(edge_refs);
        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Line(line) = geometry else {
            panic!("expected Geometry::Line, got {geometry:?}");
        };

        let coords: Vec<Coord> = line.coords().copied().collect();
        assert_eq!(
            coords,
            vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 18.0, y: 18.0 },
                Coord { x: 20.0, y: 20.0 },
            ],
            "forward edge: start + VET coords + end"
        );
    }

    /// A reversed edge (`dir = 1`) must emit intermediate VET coords in
    /// reverse order while still using start/end nodes from the edge-ref.
    #[test]
    fn resolve_geometry_line_reversed_edge() {
        let edge_refs = vec![[10, 100, 20, 1]];
        let vct = node_points(&[10, 20]);
        let vet: HashMap<u32, LineString> = [(
            100u32,
            LineString::new(vec![
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 18.0, y: 18.0 },
            ]),
        )]
        .into();

        let raw = RawGeometry::Line(edge_refs);
        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Line(line) = geometry else {
            panic!("expected Geometry::Line, got {geometry:?}");
        };

        let coords: Vec<Coord> = line.coords().copied().collect();
        assert_eq!(
            coords,
            vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 18.0, y: 18.0 },
                Coord { x: 15.0, y: 15.0 },
                Coord { x: 12.0, y: 12.0 },
                Coord { x: 20.0, y: 20.0 },
            ],
            "reversed edge: VET coords appear in reverse order"
        );
    }

    /// A multi-edge line must chain start + end nodes with deduplication
    /// at shared vertices, and must NOT close (no appended first coord).
    #[test]
    fn resolve_geometry_line_multi_edge_no_close() {
        let edge_refs = vec![[10, 0, 20, 0], [20, 0, 30, 0], [30, 0, 40, 0]];
        let vct = node_points(&[10, 20, 30, 40]);
        let vet: HashMap<u32, LineString> = HashMap::new();

        let raw = RawGeometry::Line(edge_refs);
        let geometry = resolve_geometry(raw, 0.0, 0.0, &vet, &vct);
        let Geometry::Line(line) = geometry else {
            panic!("expected Geometry::Line, got {geometry:?}");
        };

        let coords: Vec<Coord> = line.coords().copied().collect();
        assert_eq!(
            coords,
            vec![
                Coord { x: 10.0, y: 10.0 },
                Coord { x: 20.0, y: 20.0 },
                Coord { x: 30.0, y: 30.0 },
                Coord { x: 40.0, y: 40.0 },
            ],
            "multi-edge line: deduped chain, first coord NOT appended at end"
        );
        // The critical close=false contract: first ≠ last for a non-closed polyline.
        assert_ne!(
            coords.first(),
            coords.last(),
            "line must NOT be closed (close=false)"
        );
    }

    // ── decode_covr ring cleanup (coverage semantics: s57::resolve_coverage) ──

    #[test]
    fn decode_covr_deduplicates_consecutive_coords() {
        // Ring with duplicate consecutive coordinates — should be normalised
        // and produce the same polygon as a clean ring.
        let dirty_ring = LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 0.0, y: 0.0 }, // duplicate
            Coord { x: 5.0, y: 0.0 },
            Coord { x: 5.0, y: 0.0 }, // duplicate
            Coord { x: 5.0, y: 5.0 },
            Coord { x: 0.0, y: 5.0 },
            Coord { x: 0.0, y: 0.0 },
        ]);
        let result = decode_covr("test", &[dirty_ring], &[]);
        assert_eq!(result.0.len(), 1);
        let area = result.unsigned_area();
        assert!(
            (area - 25.0).abs() < 1e-6,
            "deduped ring area = 25, got {area}"
        );
    }

    #[test]
    fn decode_covr_auto_closes_unclosed_ring() {
        // Ring missing the closing coordinate — should be auto-closed.
        let unclosed = LineString::new(vec![
            Coord { x: 0.0, y: 0.0 },
            Coord { x: 3.0, y: 0.0 },
            Coord { x: 3.0, y: 4.0 },
            Coord { x: 0.0, y: 4.0 },
            // no closing coord back to (0,0)
        ]);
        let result = decode_covr("test", &[unclosed], &[]);
        assert_eq!(result.0.len(), 1, "auto-closed ring → one polygon");
        let area = result.unsigned_area();
        assert!(
            (area - 12.0).abs() < 1e-6,
            "3×4 auto-closed ring area = 12, got {area}"
        );
    }

    // ── CellBuilder record parsers ───────────────────────────────────────

    /// Little-endian byte builder for synthetic OSENC payloads.
    #[derive(Default)]
    struct Bytes(Vec<u8>);

    impl Bytes {
        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }
        fn u16(mut self, v: u16) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn i16(mut self, v: i16) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn u32(mut self, v: u32) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn i32(mut self, v: i32) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn f32(mut self, v: f32) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn f64(mut self, v: f64) -> Self {
            self.0.extend(v.to_le_bytes());
            self
        }
        fn raw(mut self, v: &[u8]) -> Self {
            self.0.extend_from_slice(v);
            self
        }
        /// The unused 4×f64 extent that line, multipoint and area headers open with.
        fn extent(self) -> Self {
            self.f64(0.0).f64(0.0).f64(0.0).f64(0.0)
        }
    }

    /// A decrypted SENC stream: the `SERVER_STATUS` + version prologue
    /// `parse_file` requires, followed by `records`.
    fn senc_stream(records: Vec<(u16, Bytes)>) -> Vec<u8> {
        let status = Bytes::default().u16(1).u16(1).u16(1).u16(30).u16(0).u16(0);
        let version = Bytes::default().u16(201);
        [
            (SERVER_STATUS_RECORD, status),
            (HEADER_SENC_VERSION, version),
        ]
        .into_iter()
        .chain(records)
        .fold(Bytes::default(), |out, (rec_type, payload)| {
            let rec_len = u32::try_from(payload.0.len() + 6).unwrap();
            out.u16(rec_type).u32(rec_len).raw(&payload.0)
        })
        .0
    }

    /// A `CELL_EXTENT_RECORD` whose (lat, lon) corners, in SW, NW, NE, SE
    /// order, span the given bounds.
    fn cell_extent(south: f64, west: f64, north: f64, east: f64) -> (u16, Bytes) {
        let payload = [south, west, north, west, north, east, south, east]
            .iter()
            .fold(Bytes::default(), |b, &v| b.f64(v));
        (CELL_EXTENT_RECORD, payload)
    }

    /// Runs one record parser the way `parse_file` does: a fresh cursor over
    /// the payload, plus the payload's length.
    fn feed<R>(
        b: &mut CellBuilder,
        payload: &[u8],
        parse: impl FnOnce(&mut CellBuilder, &mut Cursor<&[u8]>, usize) -> R,
    ) -> R {
        parse(b, &mut Cursor::new(payload), payload.len())
    }

    /// A builder past its `CELL_EXTENT_RECORD`, with an open feature for
    /// attribute/geometry records to attach to.
    fn builder_with_feature() -> CellBuilder {
        CellBuilder {
            extent: Some(CellExtent {
                georef: GeoRef {
                    lat: 54.75,
                    lon: 10.75,
                },
                bounds: [9.5, 54.0, 12.0, 55.5],
            }),
            current: Some(RawFeature {
                type_code: 1,
                id: 1,
                primitive: 0,
                attributes: Vec::new(),
                raw_geometry: RawGeometry::None,
            }),
            ..CellBuilder::default()
        }
    }

    fn current_feature(b: &CellBuilder) -> &RawFeature {
        b.current.as_ref().expect("builder has an open feature")
    }

    fn vet_points(b: &CellBuilder) -> Vec<(u32, Vec<[f64; 2]>)> {
        let mut edges: Vec<_> = b.vet.iter().map(|(&k, e)| (k, e.points.clone())).collect();
        edges.sort_by_key(|&(k, _)| k);
        edges
    }

    fn vct_coords(b: &CellBuilder) -> Vec<(u32, [f64; 2])> {
        let mut nodes: Vec<_> = b.vct.iter().map(|(&k, n)| (k, [n.lon, n.lat])).collect();
        nodes.sort_by_key(|&(k, _)| k);
        nodes
    }

    #[test]
    fn parse_file_decodes_header_and_keeps_last_feature() {
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            cell_extent(54.0, 10.0, 55.0, 11.0),
            (HEADER_CELL_NAME, Bytes::default().raw(b"TEST01\0")),
            (FEATURE_ID_RECORD, Bytes::default().u16(42).u16(7).u8(0)),
            (
                FEATURE_GEOMETRY_RECORD_POINT,
                Bytes::default().f64(54.5).f64(10.25),
            ),
        ]);
        let cell = parse_file("test.oesu".to_owned(), &data).unwrap();
        assert_eq!(cell.name, "TEST01");
        assert_eq!(cell.native_scale, 50_000);
        let [feature] = cell.features.as_slice() else {
            panic!("expected exactly one feature, got {:?}", cell.features);
        };
        assert_eq!((feature.type_code, feature.id), (42, 7));
        let Geometry::Point(p) = &feature.geometry else {
            panic!("expected a point, got {:?}", feature.geometry);
        };
        assert_eq!((p.x(), p.y()), (10.25, 54.5), "record stores lat, then lon");
    }

    /// A COVR record's `(lat, lon)` f32 pairs become the cell's coverage
    /// ring in `[lon, lat]` WGS84 degrees, with no SM conversion.
    #[test]
    fn parse_file_builds_coverage_from_covr_lat_lon_pairs() {
        let covr = Bytes::default()
            .u32(4)
            .f32(54.0)
            .f32(10.0)
            .f32(54.0)
            .f32(11.5)
            .f32(55.25)
            .f32(10.0)
            .f32(54.0)
            .f32(10.0);
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            cell_extent(54.0, 10.0, 55.25, 11.5),
            (CELL_COVR_RECORD, covr),
        ]);
        let cell = parse_file("test.oesu".to_owned(), &data).unwrap();
        let [polygon] = cell.coverage.0.as_slice() else {
            panic!("expected one coverage polygon, got {:?}", cell.coverage);
        };
        let ring: Vec<_> = polygon.exterior().coords().map(|c| (c.x, c.y)).collect();
        assert_eq!(
            ring,
            [(10.0, 54.0), (11.5, 54.0), (10.0, 55.25), (10.0, 54.0)]
        );
    }

    /// A record shorter than its fixed header is logged and skipped: the cell
    /// still parses, and the record attaches nothing to the open feature.
    #[test]
    fn short_records_are_skipped_without_error() {
        type Parser = fn(&mut CellBuilder, &mut Cursor<&[u8]>, usize) -> Result<()>;
        let cases: [(&str, usize, Parser); 10] = [
            ("TXTDSC", 8, |b, p, n| {
                b.parse_txtdsc(p, n);
                Ok(())
            }),
            ("FEATURE_ATTRIBUTE", 3, CellBuilder::parse_feature_attribute),
            ("GEOM_POINT", 16, CellBuilder::parse_geometry_point),
            ("GEOM_LINE", 36, CellBuilder::parse_geometry_line),
            ("GEOM_AREA_EXT", 52, |b, p, n| {
                b.parse_geometry_area_ext(p, n)
            }),
            (
                "GEOM_MULTIPOINT",
                36,
                CellBuilder::parse_geometry_multipoint,
            ),
            ("VET", 4, CellBuilder::parse_vet),
            ("VCT", 4, CellBuilder::parse_vct),
            ("VET_EXT", 12, CellBuilder::parse_vet_ext),
            ("VCT_EXT", 12, CellBuilder::parse_vct_ext),
        ];
        for (name, header_len, parse) in cases {
            let mut b = builder_with_feature();
            let result = feed(&mut b, &vec![0; header_len - 1], parse);
            assert!(result.is_ok(), "{name}: {result:?}");
            let feature = current_feature(&b);
            assert!(matches!(feature.raw_geometry, RawGeometry::None), "{name}");
            assert!(feature.attributes.is_empty(), "{name}");
            assert!(b.text_descriptions.is_empty(), "{name}");
            assert!(b.vet.is_empty() && b.vct.is_empty(), "{name}");
        }
    }

    #[test]
    fn cell_extent_takes_outer_bounds_of_corners_and_centres_reference() {
        let mut b = CellBuilder::default();
        // (lat, lon) corners in SW, NW, NE, SE order, skewed so that each
        // bound comes from a different corner.
        let payload = Bytes::default()
            .f64(54.0)
            .f64(10.0)
            .f64(55.0)
            .f64(9.5)
            .f64(55.5)
            .f64(11.0)
            .f64(54.5)
            .f64(12.0);
        feed(&mut b, &payload.0, CellBuilder::parse_cell_extent).unwrap();
        let expected = CellExtent {
            georef: GeoRef {
                lat: 54.75,
                lon: 10.75,
            },
            bounds: [9.5, 54.0, 12.0, 55.5],
        };
        assert_eq!(b.extent, Some(expected));
    }

    /// A truncated extent must not leave the builder unreferenced and let the
    /// cell parse anyway.
    #[test]
    fn short_cell_extent_is_rejected() {
        let mut b = CellBuilder::default();
        let err = feed(&mut b, &[0; 63], CellBuilder::parse_cell_extent).unwrap_err();
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "CELL_EXTENT_RECORD",
                    len: 63,
                    need: 64
                }
            ),
            "{err:?}"
        );
        assert_eq!(b.extent, None);
    }

    /// `ref_lat`/`ref_lon` feed every SM→WGS84 conversion; a non-finite or
    /// out-of-range corner would turn every finite VET/VCT/area coordinate
    /// into NaN, so the cell is rejected instead of rendering NaN geometry.
    #[test]
    fn cell_extent_rejects_non_finite_or_out_of_range_corners() {
        for (corner, bad) in [
            (2, f64::NAN),
            (2, 200.0),
            (0, -90.5),
            (5, f64::INFINITY),
            (7, f64::NAN),
        ] {
            let mut values = [54.0, 10.0, 55.0, 9.5, 55.5, 11.0, 54.5, 12.0];
            values[corner] = bad;
            let payload = values.iter().fold(Bytes::default(), |b, &v| b.f64(v));
            let mut b = CellBuilder::default();
            let err = feed(&mut b, &payload.0, CellBuilder::parse_cell_extent).unwrap_err();
            // Even payload slots are latitudes, odd ones longitudes.
            let expected_variant = if corner % 2 == 0 {
                matches!(err, OesuError::InvalidExtentLatitude(_))
            } else {
                matches!(err, OesuError::InvalidExtentLongitude(_))
            };
            assert!(expected_variant, "corner value {corner} = {bad}: {err:?}");
        }
    }

    #[test]
    fn txtdsc_stores_content_under_its_file_name() {
        let mut b = CellBuilder::default();
        let payload = Bytes::default()
            .u32(6)
            .u32(6)
            .raw(b"a.txt\0")
            .raw(b"Hello\0");
        feed(&mut b, &payload.0, CellBuilder::parse_txtdsc);
        assert_eq!(
            b.text_descriptions,
            HashMap::from([("a.txt".to_owned(), "Hello".to_owned())])
        );
    }

    #[test]
    fn feature_id_closes_previous_feature_and_opens_next() {
        let mut b = CellBuilder::default();
        for payload in [
            Bytes::default().u16(42).u16(7).u8(1),
            Bytes::default().u16(43).u16(8).u8(2),
        ] {
            feed(&mut b, &payload.0, CellBuilder::parse_feature_id).unwrap();
        }
        let closed: Vec<_> = b
            .raw_features
            .iter()
            .map(|f| (f.type_code, f.id, f.primitive))
            .collect();
        assert_eq!(closed, [(42, 7, 1)]);
        let open = current_feature(&b);
        assert_eq!((open.type_code, open.id, open.primitive), (43, 8, 2));
    }

    #[test]
    fn short_feature_id_closes_previous_feature_without_opening_one() {
        let mut b = builder_with_feature();
        feed(&mut b, &[0; 4], CellBuilder::parse_feature_id).unwrap();
        assert_eq!(b.raw_features.len(), 1);
        assert!(b.current.is_none());
    }

    #[test]
    #[allow(clippy::float_cmp)] // 0.5 round-trips through f64 bytes exactly
    fn feature_attribute_decodes_each_value_type_at_its_minimum_length() {
        let mut b = builder_with_feature();
        for payload in [
            Bytes::default().u16(1).u8(0).u32(12),
            Bytes::default().u16(2).u8(2).f64(0.5),
            // Unterminated: the string runs to the end of the payload.
            Bytes::default().u16(3).u8(4).raw(b"BUOY"),
            Bytes::default().u16(4).u8(4),
        ] {
            feed(&mut b, &payload.0, CellBuilder::parse_feature_attribute).unwrap();
        }
        let attrs = &current_feature(&b).attributes;
        let [int, double, string, empty] = attrs.as_slice() else {
            panic!("expected four attributes, got {attrs:?}");
        };
        assert!(
            matches!(
                int,
                Attribute {
                    code: 1,
                    value: AttrValue::Int(12)
                }
            ),
            "{int:?}"
        );
        assert!(
            matches!(double, Attribute { code: 2, value: AttrValue::Double(v) } if *v == 0.5),
            "{double:?}"
        );
        assert!(
            matches!(string, Attribute { code: 3, value: AttrValue::Str(s) } if s == "BUOY"),
            "{string:?}"
        );
        assert!(
            matches!(empty, Attribute { code: 4, value: AttrValue::Str(s) } if s.is_empty()),
            "{empty:?}"
        );
    }

    /// An entry count larger than the payload holds keeps the entries that
    /// fit instead of failing the cell; the same holds for the line,
    /// multipoint and vector-table parsers below.
    #[test]
    fn geometry_line_keeps_the_edge_refs_the_payload_holds() {
        for (payload, expected) in [
            (
                Bytes::default()
                    .extent()
                    .u32(2)
                    .i32(10)
                    .i32(20)
                    .i32(11)
                    .i32(1),
                vec![[10, 20, 11, 1]],
            ),
            (Bytes::default().extent().u32(1), vec![]),
        ] {
            let mut b = builder_with_feature();
            feed(&mut b, &payload.0, CellBuilder::parse_geometry_line).unwrap();
            let geometry = &current_feature(&b).raw_geometry;
            let RawGeometry::Line(refs) = geometry else {
                panic!("expected a line, got {geometry:?}");
            };
            assert_eq!(refs, &expected);
        }
    }

    #[test]
    fn geometry_multipoint_keeps_the_soundings_the_payload_holds() {
        for (payload, expected) in [
            (
                Bytes::default()
                    .extent()
                    .u32(2)
                    .f32(1.5)
                    .f32(-2.5)
                    .f32(3.25),
                vec![[1.5, -2.5, 3.25]],
            ),
            (Bytes::default().extent().u32(1), vec![]),
        ] {
            let mut b = builder_with_feature();
            feed(&mut b, &payload.0, CellBuilder::parse_geometry_multipoint).unwrap();
            let geometry = &current_feature(&b).raw_geometry;
            let RawGeometry::Sounding(points) = geometry else {
                panic!("expected soundings, got {geometry:?}");
            };
            assert_eq!(points, &expected);
        }
    }

    #[test]
    fn geometry_area_ext_reads_edge_refs_after_the_52_byte_header() {
        let mut b = builder_with_feature();
        // extent, contour/triprim/edge counts, scale_factor, one edge ref.
        let payload = Bytes::default()
            .extent()
            .u32(0)
            .u32(0)
            .u32(1)
            .f64(10.0)
            .i32(1)
            .i32(2)
            .i32(3)
            .i32(0);
        feed(&mut b, &payload.0, CellBuilder::parse_geometry_area_ext).unwrap();
        let geometry = &current_feature(&b).raw_geometry;
        let RawGeometry::Area { edge_refs, .. } = geometry else {
            panic!("expected an area, got {geometry:?}");
        };
        assert_eq!(edge_refs, &[[1, 2, 3, 0]]);
    }

    /// Regression test for the PR #1 off-by-one: skipping five f64s instead
    /// of four read `scale_factor` from offset 52 rather than 44. `TriPrim`
    /// vertices are the only output that depends on it.
    #[test]
    fn geometry_area_ext_divides_triprim_vertices_by_the_scale_factor_at_offset_44() {
        let mut b = builder_with_feature();
        // extent, contour/triprim/edge counts, scale_factor, vertex_counts[0],
        // one GL_TRIANGLES TriPrim (type, nvert, 4×i16 bbox, 3×2 i16 vertices),
        // one edge ref.
        let payload = Bytes::default()
            .extent()
            .u32(1)
            .u32(1)
            .u32(1)
            .f64(100.0)
            .u32(3)
            .u8(4)
            .u32(3)
            .i16(-300)
            .i16(100)
            .i16(-25)
            .i16(400)
            .i16(100)
            .i16(200)
            .i16(-300)
            .i16(400)
            .i16(50)
            .i16(-25)
            .i32(1)
            .i32(2)
            .i32(3)
            .i32(0);
        feed(&mut b, &payload.0, CellBuilder::parse_geometry_area_ext).unwrap();
        let geometry = &current_feature(&b).raw_geometry;
        let RawGeometry::Area {
            _vertex_counts: vertex_counts,
            _tri_prims: tri_prims,
            edge_refs,
            ..
        } = geometry
        else {
            panic!("expected an area, got {geometry:?}");
        };
        assert_eq!(vertex_counts, &[3]);
        assert_eq!(edge_refs, &[[1, 2, 3, 0]]);
        let [
            RawTriPrim {
                _prim_type: prim_type,
                _vertices: vertices,
                ..
            },
        ] = tri_prims.as_slice()
        else {
            panic!("expected one TriPrim, got {tri_prims:?}");
        };
        assert_eq!(*prim_type, 4);
        assert_eq!(
            vertices,
            &[[1.0, 2.0], [-3.0, 4.0], [0.5, -0.25]],
            "i16 vertices must be divided by the header scale_factor (100)"
        );
    }

    /// An EXT `TriPrim` bbox is i16 SM scaled by `scale_factor`, converted to
    /// WGS84 at parse time against the cell extent's reference point.
    #[test]
    fn geometry_area_ext_georeferences_triprim_bbox_against_the_cell_extent() {
        let mut b = builder_with_feature();
        // extent, no contours, one TriPrim, no edges, scale_factor; the
        // TriPrim has no vertices, only its [min_e, max_e, min_n, max_n] bbox.
        let payload = Bytes::default()
            .extent()
            .u32(0)
            .u32(1)
            .u32(0)
            .f64(100.0)
            .u8(4)
            .u32(0)
            .i16(-300)
            .i16(100)
            .i16(-25)
            .i16(400);
        feed(&mut b, &payload.0, CellBuilder::parse_geometry_area_ext).unwrap();
        let geometry = &current_feature(&b).raw_geometry;
        let RawGeometry::Area {
            _tri_prims: tri_prims,
            ..
        } = geometry
        else {
            panic!("expected an area, got {geometry:?}");
        };
        let [RawTriPrim { _bbox: bbox, .. }] = tri_prims.as_slice() else {
            panic!("expected one TriPrim, got {tri_prims:?}");
        };
        let min = crate::georef::from_sm(-3.0, -0.25, 54.75, 10.75);
        let max = crate::georef::from_sm(1.0, 4.0, 54.75, 10.75);
        assert_eq!(*bbox, [min.x, min.y, max.x, max.y]);
    }

    #[test]
    fn vet_keeps_the_edges_and_points_the_payload_holds() {
        // One edge whose single point ends exactly at the payload end.
        let mut b = CellBuilder::default();
        let exact = Bytes::default().u32(1).u32(7).u32(1).f32(1.5).f32(-2.5);
        feed(&mut b, &exact.0, CellBuilder::parse_vet).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]])]);

        // Three edges claimed; the second claims two points but the payload
        // ends after its header.
        let mut b = CellBuilder::default();
        let truncated = Bytes::default()
            .u32(3)
            .u32(7)
            .u32(1)
            .f32(1.5)
            .f32(-2.5)
            .u32(9)
            .u32(2);
        feed(&mut b, &truncated.0, CellBuilder::parse_vet).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]]), (9, vec![])]);
    }

    #[test]
    fn vct_keeps_the_nodes_the_payload_holds() {
        let mut b = CellBuilder::default();
        let payload = Bytes::default()
            .u32(3)
            .u32(7)
            .f32(1.5)
            .f32(-2.5)
            .u32(9)
            .f32(3.0)
            .f32(4.0);
        feed(&mut b, &payload.0, CellBuilder::parse_vct).unwrap();
        assert_eq!(vct_coords(&b), [(7, [1.5, -2.5]), (9, [3.0, 4.0])]);
    }

    #[test]
    fn vet_ext_divides_by_scale_factor_and_keeps_what_the_payload_holds() {
        // One edge whose single point ends exactly at the payload end.
        let mut b = CellBuilder::default();
        let exact = Bytes::default()
            .f64(10.0)
            .u32(1)
            .u32(7)
            .u32(1)
            .i16(15)
            .i16(-25);
        feed(&mut b, &exact.0, CellBuilder::parse_vet_ext).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]])]);

        // Three edges claimed; the second claims two points but the payload
        // ends after its header.
        let mut b = CellBuilder::default();
        let truncated = Bytes::default()
            .f64(10.0)
            .u32(3)
            .u32(7)
            .u32(1)
            .i16(15)
            .i16(-25)
            .u32(9)
            .u32(2);
        feed(&mut b, &truncated.0, CellBuilder::parse_vet_ext).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]]), (9, vec![])]);
    }

    #[test]
    fn vct_ext_divides_by_scale_factor_and_keeps_what_the_payload_holds() {
        let mut b = CellBuilder::default();
        let payload = Bytes::default()
            .f64(10.0)
            .u32(3)
            .u32(7)
            .i16(15)
            .i16(-25)
            .u32(9)
            .i16(30)
            .i16(40);
        feed(&mut b, &payload.0, CellBuilder::parse_vct_ext).unwrap();
        assert_eq!(vct_coords(&b), [(7, [1.5, -2.5]), (9, [3.0, 4.0])]);
    }

    // ── parse_file record loop ───────────────────────────────────────────

    /// A record header whose declared `rec_len` may disagree with what follows.
    fn record_header(rec_type: u16, rec_len: u32) -> Bytes {
        Bytes::default().u16(rec_type).u32(rec_len)
    }

    /// A complete native-scale record, for appending after a terminator.
    fn scale_record(scale: u32) -> Bytes {
        record_header(HEADER_CELL_NATIVESCALE, 6 + 4).u32(scale)
    }

    fn parse_err(data: &[u8]) -> OesuError {
        match parse_file("test.oesu".to_owned(), data) {
            Ok(cell) => panic!("expected parse error, got cell {:?}", cell.name),
            Err(e) => e,
        }
    }

    /// A `SERVER_STATUS` + version prologue with the given status fields
    /// (`serverStatus, decryptStatus, expireStatus, expireDays, graceAllowed,
    /// graceDays`) and SENC version.
    fn prologue(status: [u16; 6], version: u16) -> Vec<u8> {
        let status = status.iter().fold(Bytes::default(), |b, &v| b.u16(v));
        record_header(SERVER_STATUS_RECORD, 6 + 12)
            .raw(&status.0)
            .raw(&record_header(HEADER_SENC_VERSION, 6 + 2).u16(version).0)
            .0
    }

    /// Domain refusals — an intact file this tool must not use — are typed
    /// distinctly from corrupt or truncated streams.
    #[test]
    fn prologue_refusals_are_typed_and_distinct_from_corruption() {
        let err = parse_err(&prologue([1, 1, 0, 0, 7, 3], 201));
        assert!(
            matches!(
                err,
                OesuError::LicenseExpired {
                    expire_days: 0,
                    grace_days: 3
                }
            ),
            "{err:?}"
        );
        assert!(err.is_refusal());

        let err = parse_err(&prologue([5, 0, 1, 30, 0, 0], 201));
        assert!(
            matches!(err, OesuError::DecryptionFailed { server_status: 5 }),
            "{err:?}"
        );
        assert!(err.is_refusal());

        let err = parse_err(&prologue([1, 1, 1, 30, 0, 0], 1024));
        assert!(matches!(err, OesuError::SignatureFailure), "{err:?}");
        assert!(err.is_refusal());

        let err = parse_err(&prologue([1, 1, 1, 30, 0, 0], 300));
        assert!(matches!(err, OesuError::UnsupportedVersion(300)), "{err:?}");
        assert!(err.is_refusal());

        let err = parse_err(b"!BSB/VER=3.0,CED/SE=1");
        assert!(matches!(err, OesuError::UnsupportedRaster), "{err:?}");
        assert!(err.is_refusal());

        let err = parse_err(b"not a senc stream");
        assert!(
            matches!(
                err,
                OesuError::UnexpectedRecord {
                    expected: SERVER_STATUS_RECORD,
                    ..
                }
            ),
            "{err:?}"
        );
        assert!(!err.is_refusal());

        // SERVER_STATUS declares 12 payload bytes; only 4 follow.
        let mut truncated = prologue([1, 1, 1, 30, 0, 0], 201);
        truncated.truncate(10);
        let err = parse_err(&truncated);
        assert!(
            matches!(
                err,
                OesuError::Truncated {
                    record_type: SERVER_STATUS_RECORD,
                    declared: 12,
                    remaining: 4
                }
            ),
            "{err:?}"
        );
        assert!(!err.is_refusal());
    }

    #[test]
    fn detect_claims_senc_streams_and_bsb_rasters_only() {
        assert!(detect(&prologue([1, 1, 1, 30, 0, 0], 201)));
        // Refused by parse_file with a specific error, so still ours to report.
        assert!(detect(b"!BSB/VER=3.0,CED/SE=1"));
        for other in [&b""[..], b"\xc8", b"{\"A_DEPARE\":[]}", b"RXTILES\x04"] {
            assert!(!detect(other), "{other:?}");
        }
    }

    /// Each prologue length check sits exactly at its boundary and reports
    /// the record and sizes involved.
    #[test]
    fn prologue_length_checks_are_exact() {
        let err = parse_err(&record_header(SERVER_STATUS_RECORD, 6 + 12).0[..5]);
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "SERVER_STATUS_RECORD",
                    len: 5,
                    need: 6
                }
            ),
            "{err:?}"
        );

        // A complete 6-byte header is enough to read; the payload is missing.
        let err = parse_err(&record_header(SERVER_STATUS_RECORD, 6 + 12).0);
        assert!(
            matches!(
                err,
                OesuError::Truncated {
                    record_type: SERVER_STATUS_RECORD,
                    declared: 12,
                    remaining: 0
                }
            ),
            "{err:?}"
        );

        // One byte short of the 12-byte payload, though the stream goes on.
        let mut short_status = prologue([1, 1, 1, 30, 0, 0], 201);
        short_status[2..6].copy_from_slice(&17u32.to_le_bytes());
        let err = parse_err(&short_status);
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "SERVER_STATUS_RECORD",
                    len: 17,
                    need: 18
                }
            ),
            "{err:?}"
        );

        // HEADER_SENC_VERSION declaring one byte less than its 8-byte minimum.
        let mut short_version = prologue([1, 1, 1, 30, 0, 0], 201);
        short_version[20..24].copy_from_slice(&7u32.to_le_bytes());
        let err = parse_err(&short_version);
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "HEADER_SENC_VERSION",
                    len: 7,
                    need: 8
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn parse_file_missing_native_scale_is_rejected() {
        let data = senc_stream(vec![(HEADER_CELL_NAME, Bytes::default().raw(b"TEST01\0"))]);
        let err = parse_err(&data);
        assert!(matches!(err, OesuError::MissingNativeScale), "{err:?}");
    }

    #[test]
    fn parse_file_zero_native_scale_is_rejected() {
        let data = senc_stream(vec![(HEADER_CELL_NATIVESCALE, Bytes::default().u32(0))]);
        let err = parse_err(&data);
        assert!(matches!(err, OesuError::MissingNativeScale), "{err:?}");
    }

    /// A self-consistent record too short for the u32 scale must error, not
    /// silently default to 0.
    #[test]
    fn parse_file_short_native_scale_payload_is_rejected() {
        let data = senc_stream(vec![(
            HEADER_CELL_NATIVESCALE,
            Bytes::default().u16(10_000),
        )]);
        let err = parse_err(&data);
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "HEADER_CELL_NATIVESCALE",
                    len: 2,
                    need: 4
                }
            ),
            "{err:?}"
        );
    }

    /// Run `f` under a thread-local subscriber; return its result and the
    /// formatted (ANSI-free) log lines it emitted.
    fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let bytes = capture.0.lock().unwrap().clone();
        (out, String::from_utf8(bytes).unwrap())
    }

    /// Edition/update-number records too short for their u16 are not worth
    /// dropping the whole cell over (nothing consumes either number): the
    /// cell still loads, with a warning naming the record.
    #[test]
    fn parse_file_short_edition_payloads_warn_and_keep_the_cell() {
        for (rec_type, name) in [
            (HEADER_CELL_EDITION, "HEADER_CELL_EDITION"),
            (HEADER_CELL_UPDATE, "HEADER_CELL_UPDATE"),
        ] {
            let data = senc_stream(vec![
                (HEADER_CELL_NATIVESCALE, Bytes::default().u32(22_000)),
                cell_extent(54.0, 10.0, 55.0, 11.0),
                (rec_type, Bytes::default().u8(1)),
            ]);
            let (cell, logs) = capture_logs(|| parse_file("test.oesu".to_owned(), &data));
            let cell = cell.unwrap_or_else(|e| panic!("{name}: cell dropped: {e:#}"));
            assert_eq!(cell.native_scale, 22_000);
            assert!(
                logs.contains("WARN") && logs.contains(name) && logs.contains("payload_len=1"),
                "got: {logs}"
            );
        }
    }

    /// Parse a minimal cell carrying the given optional PUBLISHDATE /
    /// UPDATEDATE cstrings; return its edition date and the captured log.
    fn edition_date_of(publish: Option<&str>, update: Option<&str>) -> (s57::EditionDate, String) {
        let mut records = vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(22_000)),
            cell_extent(54.0, 10.0, 55.0, 11.0),
        ];
        for (rec_type, date) in [
            (HEADER_CELL_PUBLISHDATE, publish),
            (HEADER_CELL_UPDATEDATE, update),
        ] {
            if let Some(date) = date {
                let cstring = [date.as_bytes(), b"\0"].concat();
                records.push((rec_type, Bytes::default().raw(&cstring)));
            }
        }
        let data = senc_stream(records);
        let (cell, logs) = capture_logs(|| parse_file("test.oesu".to_owned(), &data));
        (cell.unwrap().edition_date, logs)
    }

    #[test]
    fn edition_date_prefers_update_date_over_publish_date() {
        let (date, logs) = edition_date_of(Some("20200101"), Some("20211209"));
        assert_eq!(date, s57::EditionDate::new(2021, 12, 9));
        assert!(!logs.contains("edition date"), "got: {logs}");
    }

    #[test]
    fn edition_date_falls_back_to_publish_date_without_update_date() {
        let (date, logs) = edition_date_of(Some("20200101"), None);
        assert_eq!(date, s57::EditionDate::new(2020, 1, 1));
        assert!(!logs.contains("edition date"), "got: {logs}");
    }

    #[test]
    fn absent_edition_date_is_unknown_without_warning() {
        let (date, logs) = edition_date_of(None, None);
        assert_eq!(date, s57::EditionDate::unknown());
        assert!(!logs.contains("edition date"), "got: {logs}");
    }

    /// A present-but-malformed date still loads the cell, ranked oldest —
    /// but loudly, naming the cell source and the raw value.
    #[test]
    fn malformed_update_date_ranks_oldest_and_warns() {
        let (date, logs) = edition_date_of(Some("20200101"), Some("2021-12-9"));
        assert_eq!(date, s57::EditionDate::unknown());
        assert!(
            logs.contains("WARN")
                && logs.contains("unparseable edition date")
                && logs.contains("cell=test.oesu")
                && logs.contains("raw=2021-12-9"),
            "got: {logs}"
        );
    }

    #[test]
    fn parse_file_record_longer_than_stream_is_rejected() {
        let mut data = senc_stream(vec![(
            HEADER_CELL_NATIVESCALE,
            Bytes::default().u32(22_000),
        )]);
        // Declares a 4-byte payload but only 2 bytes follow.
        data.extend(record_header(HEADER_CELL_EDITION, 6 + 4).u16(1).0);
        let err = parse_err(&data);
        assert!(
            matches!(
                err,
                OesuError::Truncated {
                    record_type: HEADER_CELL_EDITION,
                    declared: 4,
                    remaining: 2
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn parse_file_huge_rec_len_is_rejected_before_allocating() {
        let mut data = senc_stream(vec![(
            HEADER_CELL_NATIVESCALE,
            Bytes::default().u32(22_000),
        )]);
        // rec_len = u32::MAX must be rejected by a remaining-bytes check,
        // not by a ~4 GiB zeroed allocation followed by a failed read.
        data.extend(record_header(HEADER_CELL_EDITION, u32::MAX).u16(1).0);
        let err = parse_err(&data);
        assert!(
            matches!(
                err,
                OesuError::Truncated {
                    record_type: HEADER_CELL_EDITION,
                    remaining: 2,
                    ..
                }
            ),
            "{err:?}"
        );
    }

    #[test]
    fn parse_file_zero_terminator_stops_parsing() {
        let mut data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(22_000)),
            cell_extent(54.0, 10.0, 55.0, 11.0),
        ]);
        data.extend(record_header(0, 0).0);
        data.extend(scale_record(99_999).0);
        let cell = parse_file("test.oesu".to_owned(), &data).unwrap();
        assert_eq!(cell.native_scale, 22_000, "records after 0/0 are ignored");
    }

    #[test]
    fn parse_file_short_nonzero_record_stops_parsing() {
        let mut data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(22_000)),
            cell_extent(54.0, 10.0, 55.0, 11.0),
        ]);
        data.extend(record_header(HEADER_CELL_NATIVESCALE, 3).0);
        data.extend(scale_record(99_999).0);
        let cell = parse_file("test.oesu".to_owned(), &data).unwrap();
        assert_eq!(
            cell.native_scale, 22_000,
            "records after rec_len < 6 are ignored"
        );
    }

    /// Stopping early must not bypass the post-loop native-scale guard.
    #[test]
    fn parse_file_terminator_before_native_scale_is_rejected() {
        let mut data = senc_stream(vec![]);
        data.extend(record_header(0, 0).0);
        data.extend(scale_record(22_000).0);
        let err = parse_err(&data);
        assert!(matches!(err, OesuError::MissingNativeScale), "{err:?}");
    }

    /// Every SM-relative coordinate is georeferenced against the extent's
    /// centre, so a cell without one must not load with geometry around 0°N 0°E.
    #[test]
    fn parse_file_missing_cell_extent_is_rejected() {
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            (
                VECTOR_CONNECTED_NODE_TABLE_RECORD,
                Bytes::default().u32(1).u32(1).f32(100.0).f32(-200.0),
            ),
        ]);
        let err = parse_err(&data);
        assert!(matches!(err, OesuError::MissingCellExtent), "{err:?}");
    }

    #[test]
    fn parse_file_short_cell_extent_is_rejected() {
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            (CELL_EXTENT_RECORD, Bytes::default().f64(54.0).f64(10.0)),
        ]);
        let err = parse_err(&data);
        assert!(
            matches!(
                err,
                OesuError::RecordTooShort {
                    record: "CELL_EXTENT_RECORD",
                    len: 16,
                    need: 64
                }
            ),
            "{err:?}"
        );
    }

    /// An EXT area converts its `TriPrim` bboxes at parse time, so it needs
    /// the reference point already; arriving first is a malformed stream.
    #[test]
    fn parse_file_area_ext_before_cell_extent_is_rejected() {
        let area = Bytes::default().extent().u32(0).u32(0).u32(0).f64(10.0);
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            (FEATURE_ID_RECORD, Bytes::default().u16(42).u16(1).u8(3)),
            (FEATURE_GEOMETRY_RECORD_AREA_EXT, area),
            cell_extent(54.0, 10.0, 55.0, 11.0),
        ]);
        let err = parse_err(&data);
        assert!(
            matches!(
                err,
                OesuError::RecordBeforeExtent {
                    record: "FEATURE_GEOMETRY_RECORD_AREA_EXT"
                }
            ),
            "{err:?}"
        );
    }

    /// End to end: EXT tables divide i16 coords by `scale_factor`, and a LINE
    /// walking node 1 → edge 10 → node 2 resolves into nodes framing the
    /// edge points.
    #[test]
    fn parse_file_resolves_ext_tables_into_line_coords() {
        let nodes = Bytes::default()
            .f64(4.0)
            .u32(2)
            .u32(1)
            .i16(400)
            .i16(-800)
            .u32(2)
            .i16(1200)
            .i16(40);
        let edges = Bytes::default()
            .f64(4.0)
            .u32(1)
            .u32(10)
            .u32(2)
            .i16(600)
            .i16(-400)
            .i16(1000)
            .i16(0);
        let line = Bytes::default()
            .extent()
            .u32(1)
            .i32(1)
            .i32(10)
            .i32(2)
            .i32(0);
        let data = senc_stream(vec![
            (HEADER_CELL_NATIVESCALE, Bytes::default().u32(50_000)),
            cell_extent(54.0, 9.5, 55.5, 12.0),
            (VECTOR_CONNECTED_NODE_TABLE_EXT_RECORD, nodes),
            (VECTOR_EDGE_NODE_TABLE_EXT_RECORD, edges),
            (FEATURE_ID_RECORD, Bytes::default().u16(42).u16(1).u8(1)),
            (FEATURE_GEOMETRY_RECORD_LINE, line),
        ]);
        let cell = parse_file("test.oesu".to_owned(), &data).unwrap();
        let [feature] = cell.features.as_slice() else {
            panic!("expected one feature, got {:?}", cell.features);
        };
        let Geometry::Line(line) = &feature.geometry else {
            panic!("expected a line, got {:?}", feature.geometry);
        };
        // The extent's centre (54.75°N, 10.75°E) is the SM reference point.
        let sm = |east, north| crate::georef::from_sm(east, north, 54.75, 10.75);
        assert_eq!(
            line.0,
            [
                sm(100.0, -200.0),
                sm(150.0, -100.0),
                sm(250.0, 0.0),
                sm(300.0, 10.0),
            ]
        );
    }

    /// A zero, negative or non-finite divisor would turn every coordinate
    /// into NaN/inf (or mirror it), so the whole EXT table is skipped.
    #[test]
    fn vet_ext_and_vct_ext_skip_invalid_scale_factor() {
        for scale in [0.0, -2.0, f64::NAN, f64::INFINITY] {
            let mut b = CellBuilder::default();
            let edges = Bytes::default()
                .f64(scale)
                .u32(1)
                .u32(7)
                .u32(1)
                .i16(15)
                .i16(-25);
            let nodes = Bytes::default().f64(scale).u32(1).u32(7).i16(15).i16(-25);
            feed(&mut b, &edges.0, CellBuilder::parse_vet_ext).unwrap();
            feed(&mut b, &nodes.0, CellBuilder::parse_vct_ext).unwrap();
            assert_eq!(vet_points(&b), [], "scale_factor {scale}");
            assert_eq!(vct_coords(&b), [], "scale_factor {scale}");
        }
    }

    /// A POINT record holds raw f64 WGS84 lat/lon; a NaN/inf value would
    /// reach geo through the light-sector bbox and panic the render, so the
    /// geometry is left unset.
    #[test]
    fn point_skips_non_finite_coordinates() {
        for (lat, lon) in [(f64::NAN, 10.1), (55.1, f64::INFINITY)] {
            let mut b = builder_with_feature();
            let payload = Bytes::default().f64(lat).f64(lon);
            feed(&mut b, &payload.0, CellBuilder::parse_geometry_point).unwrap();
            let geometry = &current_feature(&b).raw_geometry;
            assert!(matches!(geometry, RawGeometry::None), "{geometry:?}");
        }
    }

    /// Plain VET/VCT/COVR coordinates are raw f32s; a NaN/inf vertex would
    /// reach geo's boolean ops and panic the render, so the edge, node or
    /// ring is skipped like other malformed records.
    #[test]
    fn vet_vct_and_covr_skip_non_finite_coordinates() {
        for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let mut b = CellBuilder::default();
            let edges = Bytes::default()
                .u32(2)
                .u32(7)
                .u32(2)
                .f32(0.0)
                .f32(0.0)
                .f32(bad)
                .f32(1.0)
                .u32(8)
                .u32(1)
                .f32(1.5)
                .f32(-2.5);
            let nodes = Bytes::default()
                .u32(2)
                .u32(7)
                .f32(1.0)
                .f32(bad)
                .u32(8)
                .f32(1.5)
                .f32(-2.5);
            feed(&mut b, &edges.0, CellBuilder::parse_vet).unwrap();
            feed(&mut b, &nodes.0, CellBuilder::parse_vct).unwrap();
            assert_eq!(vet_points(&b), [(8, vec![[1.5, -2.5]])], "{bad}");
            assert_eq!(vct_coords(&b), [(8, [1.5, -2.5])], "{bad}");

            let covr = Bytes::default()
                .u32(2)
                .f32(55.0)
                .f32(10.0)
                .f32(bad)
                .f32(10.1);
            let covr_len = covr.0.len();
            assert_eq!(
                parse_covr_payload(&mut Cursor::new(covr.0.as_slice()), covr_len),
                None,
                "{bad}"
            );
        }
    }

    /// A `u32::MAX` entry count must not be trusted for pre-allocation: the
    /// parsers keep what the payload holds instead of aborting the process
    /// on a tens-of-GiB allocation.
    #[test]
    fn huge_entry_counts_keep_what_the_payload_holds() {
        let mut b = builder_with_feature();
        let line = Bytes::default()
            .extent()
            .u32(u32::MAX)
            .i32(1)
            .i32(2)
            .i32(3)
            .i32(0);
        feed(&mut b, &line.0, CellBuilder::parse_geometry_line).unwrap();
        let geometry = &current_feature(&b).raw_geometry;
        assert!(
            matches!(geometry, RawGeometry::Line(refs) if refs == &[[1, 2, 3, 0]]),
            "{geometry:?}"
        );

        let multipoint = Bytes::default()
            .extent()
            .u32(u32::MAX)
            .f32(1.5)
            .f32(-2.5)
            .f32(3.25);
        feed(
            &mut b,
            &multipoint.0,
            CellBuilder::parse_geometry_multipoint,
        )
        .unwrap();
        let geometry = &current_feature(&b).raw_geometry;
        assert!(
            matches!(geometry, RawGeometry::Sounding(pts) if pts == &[[1.5, -2.5, 3.25]]),
            "{geometry:?}"
        );

        let mut b = CellBuilder::default();
        let vet = Bytes::default()
            .u32(1)
            .u32(7)
            .u32(u32::MAX)
            .f32(1.5)
            .f32(-2.5);
        feed(&mut b, &vet.0, CellBuilder::parse_vet).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]])]);

        let mut b = CellBuilder::default();
        let vet_ext = Bytes::default()
            .f64(10.0)
            .u32(1)
            .u32(7)
            .u32(u32::MAX)
            .i16(15)
            .i16(-25);
        feed(&mut b, &vet_ext.0, CellBuilder::parse_vet_ext).unwrap();
        assert_eq!(vet_points(&b), [(7, vec![[1.5, -2.5]])]);
    }

    /// COVR/NOCOVR point counts are untrusted too: a `u32::MAX` count must be
    /// rejected as truncated, not pre-allocated (64 GiB of `Coord`s).
    #[test]
    fn huge_covr_count_is_rejected_without_preallocation() {
        let covr = Bytes::default().u32(u32::MAX).f32(1.5).f32(-2.5);
        let covr_len = covr.0.len();
        assert_eq!(
            parse_covr_payload(&mut Cursor::new(covr.0.as_slice()), covr_len),
            None
        );
    }

    /// `AREA`/`AREA_EXT` headers are untrusted too: a `u32::MAX` contour or
    /// `TriPrim` count over an empty body is a truncation error, not a
    /// 16/256 GiB pre-allocation that aborts the process.
    #[test]
    fn area_payload_huge_counts_are_truncation_errors() {
        let georef = GeoRef {
            lat: 54.75,
            lon: 10.75,
        };
        for ext in [None, Some((10.0, georef))] {
            for (contours, triprims, expected) in [
                (u32::MAX, 0, "vertex count array"),
                (0, u32::MAX, "TriPrim header"),
            ] {
                let mut payload = Bytes::default().extent().u32(contours).u32(triprims).u32(0);
                if ext.is_some() {
                    payload = payload.f64(10.0);
                }
                let err = parse_area_payload(
                    &mut Cursor::new(payload.0.as_slice()),
                    payload.0.len(),
                    ext,
                )
                .unwrap_err();
                assert!(
                    matches!(
                        err,
                        OesuError::TruncatedSection { section, entry: 0, .. } if section == expected
                    ),
                    "ext={ext:?}: {err:?}"
                );
            }
        }
    }

    /// Every `AREA`/`AREA_EXT` section check accepts a payload that ends
    /// exactly where the section does and names the section one byte short.
    #[test]
    fn area_payload_section_checks_are_exact() {
        fn parse(payload: &[u8], ext: Option<(f64, GeoRef)>) -> Result<RawGeometry> {
            parse_area_payload(&mut Cursor::new(payload), payload.len(), ext)
        }
        fn assert_truncated(result: &Result<RawGeometry>, expected: &str) {
            assert!(
                matches!(
                    result,
                    Err(OesuError::TruncatedSection { section, entry: 0, count: 1, .. })
                        if *section == expected
                ),
                "expected truncated {expected}: {result:?}"
            );
        }
        let georef = GeoRef {
            lat: 54.75,
            lon: 10.75,
        };
        let header = |contours, triprims, edges, ext: bool| {
            let b = Bytes::default()
                .extent()
                .u32(contours)
                .u32(triprims)
                .u32(edges);
            if ext { b.f64(10.0) } else { b }
        };

        // The vertex count array ends the payload.
        parse(&header(1, 0, 0, false).u32(4).0, None).unwrap();

        // A TriPrim header that ends the payload leaves no room for its body.
        let header_only = header(0, 1, 0, false).u8(1).u32(0);
        assert_truncated(&parse(&header_only.0, None), "TriPrim body");

        // Plain body: 4×f64 bbox + nvert × 2×f32.
        let plain = header(0, 1, 0, false)
            .u8(1)
            .u32(1)
            .extent()
            .f32(1.5)
            .f32(-2.5);
        parse(&plain.0, None).unwrap();
        assert_truncated(&parse(&plain.0[..plain.0.len() - 1], None), "TriPrim body");

        // EXT body: 4×i16 bbox + nvert × 2×i16.
        let ext = Some((10.0, georef));
        let packed = header(0, 1, 0, true)
            .u8(1)
            .u32(1)
            .i16(0)
            .i16(1)
            .i16(0)
            .i16(1)
            .i16(15)
            .i16(-25);
        parse(&packed.0, ext).unwrap();
        assert_truncated(&parse(&packed.0[..packed.0.len() - 1], ext), "TriPrim body");

        let err = parse(&header(0, 0, 2, false).0, None).unwrap_err();
        assert!(
            matches!(
                err,
                OesuError::EdgeSectionMismatch {
                    remaining: 0,
                    expected: 32
                }
            ),
            "{err:?}"
        );
    }

    /// The fixed `AREA` header is 44 bytes; `AREA_EXT` adds an 8-byte
    /// `scale_factor` for 52. One byte short is `RecordTooShort`, exact is
    /// an empty area.
    #[test]
    fn area_payload_rejects_records_shorter_than_the_fixed_header() {
        let georef = GeoRef {
            lat: 54.75,
            lon: 10.75,
        };
        for (ext, record, need) in [
            (None, "FEATURE_GEOMETRY_RECORD_AREA", 44),
            (Some((10.0, georef)), "FEATURE_GEOMETRY_RECORD_AREA_EXT", 52),
        ] {
            let mut payload = Bytes::default().extent().u32(0).u32(0).u32(0);
            if ext.is_some() {
                payload = payload.f64(10.0);
            }
            assert_eq!(payload.0.len(), need);

            let exact = parse_area_payload(&mut Cursor::new(payload.0.as_slice()), need, ext);
            assert!(
                matches!(&exact, Ok(RawGeometry::Area { edge_refs, .. }) if edge_refs.is_empty()),
                "{record}: {exact:?}"
            );

            let short = &payload.0[..need - 1];
            let err = parse_area_payload(&mut Cursor::new(short), short.len(), ext).unwrap_err();
            assert!(
                matches!(
                    err,
                    OesuError::RecordTooShort { record: r, len, need: n }
                        if r == record && len == need - 1 && n == need
                ),
                "{record}: {err:?}"
            );
        }
    }

    #[test]
    fn duplicate_vet_and_vct_indices_keep_the_last_record() {
        let mut b = CellBuilder::default();
        for v in [1.0, 9.0] {
            let edges = Bytes::default().u32(1).u32(7).u32(1).f32(v).f32(v);
            let nodes = Bytes::default().u32(1).u32(7).f32(v).f32(v);
            feed(&mut b, &edges.0, CellBuilder::parse_vet).unwrap();
            feed(&mut b, &nodes.0, CellBuilder::parse_vct).unwrap();
        }
        assert_eq!(vet_points(&b), [(7, vec![[9.0, 9.0]])]);
        assert_eq!(vct_coords(&b), [(7, [9.0, 9.0])]);
    }
}
