//! Parser for `.djson` vector cells.
//!
//! A `.djson` document is one JSON object: a `__system__` entry with cell
//! metadata, and one array per S-57 object class keyed `{P,L,A}_{CLASS}`
//! (e.g. `A_DEPARE`; the prefix gives the geometry kind). Each record holds
//! its S-57 attributes plus internal keys:
//!
//! - `_XY`: flat `[x0, y0, x1, y1, …]` in the format's spherical Mercator metres
//!   (see [`rnc_format::rnc_merc_to_wgs84`] — the same sphere `.rnc` uses).
//! - `_SL` / `_SO` / `_SP`: cumulative end vertex indices splitting `_XY` into
//!   line parts, polygons and rings (see [`part_ranges`], [`polygon_ranges`]).
//! - `_SD`: sounding depths, one per `_XY` pair (`P_SOUNDG` only).
//!
//! [`detect`] recognises a document by its content; [`parse`] turns it into
//! [`Layer`]s of [`Record`]s with `WGS-84` geometry, and
//! `s57::S57Cell::try_from` converts that into an [`s57::S57Cell`].
//! Fetching and decompression are left to the caller.
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

mod s57_cell;

use std::ops::Range;

use geo::{Coord, LineString, MultiLineString, MultiPolygon, Point, Polygon};
use serde_json::{Map, Value};

/// Why a `.djson` document could not be parsed or converted.
#[derive(Debug, thiserror::Error)]
pub enum DjsonError {
    /// The bytes are not valid JSON.
    #[error("invalid djson JSON: {0}")]
    Json(#[from] serde_json::Error),
    /// The JSON root is not an object of layers.
    #[error("djson root is not an object")]
    RootNotObject,
    /// No usable `__system__.mima`, so the cell has no bounds or scale.
    #[error("no usable `__system__.mima` (bounds and scale)")]
    MissingSystem,
    /// The `__system__.mima` scale is not a `1:N` denominator (`N ≥ 1`).
    #[error("invalid scale {0}: must be ≥ 1")]
    InvalidScale(f64),
}

/// `true` if `data` looks like a `.djson` document: a JSON object, i.e. its
/// first non-whitespace byte is `{`. Cheap; [`parse`] still validates it.
#[must_use]
pub fn detect(data: &[u8]) -> bool {
    data.iter()
        .find(|b| !b.is_ascii_whitespace())
        .is_some_and(|&b| b == b'{')
}

/// Result of `.djson` parsing; the error defaults to [`DjsonError`].
pub type Result<T, E = DjsonError> = std::result::Result<T, E>;

/// Layer key of sounding records; each record packs many depths.
pub const SOUNDG_LAYER: &str = "P_SOUNDG";

/// Record keys that are geometry or rendering hints, not S-57 attributes.
pub const INTERNAL_KEYS: &[&str] = &[
    "_XY", "_SD", "_SL", "_SO", "_SP", "_RR", "_Z", "_LY", "_BL", "_PN", "_MOVEX", "_MOVEY",
    "_LIDXY",
];

/// One parsed `.djson` cell.
#[derive(Debug, Clone)]
pub struct DjsonCell {
    /// Where the document came from (e.g. a file path); identifies the cell
    /// in logs and names it when `__system__` has no `name`.
    pub source: String,
    /// Cell metadata; `None` when `__system__` is absent or has no usable `mima`.
    pub system: Option<System>,
    /// Layers sorted by key, `__system__` excluded.
    pub layers: Vec<Layer>,
}

/// Cell metadata from `__system__`.
#[derive(Debug, Clone, PartialEq)]
pub struct System {
    /// `(xmin, ymin, xmax, ymax)` in the format's Mercator metres; `ymin` is the north edge.
    pub merc_bbox: (f64, f64, f64, f64),
    /// Scale denominator (`mima[4]`).
    pub scale: f64,
    pub name: Option<String>,
    pub pid: Option<u64>,
}

/// Geometry kind of a layer, from its key prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GeomKind {
    Point,
    Line,
    /// `A_` and any unknown prefix.
    Area,
}

impl GeomKind {
    #[must_use]
    pub fn from_layer_key(layer_key: &str) -> Self {
        match layer_key.chars().next() {
            Some('P') => Self::Point,
            Some('L') => Self::Line,
            _ => Self::Area,
        }
    }
}

/// All records of one S-57 object class.
#[derive(Debug, Clone)]
pub struct Layer {
    /// Raw layer key, e.g. `"A_DEPARE"`.
    pub key: String,
    /// Records with usable geometry; others are dropped (see [`parse`]).
    pub records: Vec<Record>,
}

impl Layer {
    /// S-57 object class: `"A_DEPARE"` → `"DEPARE"`.
    #[must_use]
    pub fn class(&self) -> &str {
        s57_class(&self.key)
    }

    #[must_use]
    pub fn kind(&self) -> GeomKind {
        GeomKind::from_layer_key(&self.key)
    }
}

/// One feature record.
#[derive(Debug, Clone)]
pub struct Record {
    pub geometry: Geometry,
    /// `SCAMIN`: hide when the map scale denominator exceeds it.
    pub scamin: Option<f64>,
    /// Every key except [`INTERNAL_KEYS`], values as in the document
    /// (`SCAMIN` included).
    pub attributes: Map<String, Value>,
}

/// Record geometry in `WGS-84` (x = longitude, y = latitude, degrees).
#[derive(Debug, Clone, PartialEq)]
pub enum Geometry {
    /// First `_XY` pair of a point record.
    Point(Point),
    /// `P_SOUNDG`: one `(position, depth)` per `_XY` pair.
    Soundings(Vec<(Point, f64)>),
    /// Line parts split by `_SL` (falling back to `_SO`, then `_SP`).
    Lines(MultiLineString),
    /// Polygons split by `_SO`, rings (outer first, then holes) by `_SP`.
    /// Rings are closed.
    Areas(MultiPolygon),
}

/// `"A_DEPARE"` → `"DEPARE"`, `"P_LIT"` → `"LIT"`; keys without `_` unchanged.
#[must_use]
pub fn s57_class(layer_key: &str) -> &str {
    layer_key
        .split_once('_')
        .map_or(layer_key, |(_, class)| class)
}

/// Parse a decompressed `.djson` document read from `source` (e.g. a file path).
///
/// Lenient below the root: non-array layers, non-object records and records
/// without usable geometry are skipped. Geometry is unusable when `_XY` is
/// missing or holds a non-number, a part array (`_SL` / `_SO` / `_SP`)
/// holds anything but whole non-negative numbers, a `P_SOUNDG` record's
/// `_SD` holds a non-number or does not have one depth per `_XY` pair, or
/// the part arrays leave no line or polygon. A trailing odd `_XY` value is
/// ignored.
///
/// # Errors
/// Returns an error when `bytes` is not JSON or its root is not an object.
pub fn parse(source: impl Into<String>, bytes: &[u8]) -> Result<DjsonCell> {
    let doc: Value = serde_json::from_slice(bytes)?;
    from_value(source, &doc)
}

/// [`parse`] for an already-decoded JSON document.
///
/// # Errors
/// Returns an error when the root is not an object.
pub fn from_value(source: impl Into<String>, doc: &Value) -> Result<DjsonCell> {
    let root = doc.as_object().ok_or(DjsonError::RootNotObject)?;
    let system = root.get("__system__").and_then(parse_system);
    let layers = root
        .iter()
        .filter(|(key, _)| *key != "__system__")
        .filter_map(|(key, records)| {
            let records = records
                .as_array()?
                .iter()
                .filter_map(Value::as_object)
                .filter_map(|rec| parse_record(key, rec))
                .collect();
            Some(Layer {
                key: key.clone(),
                records,
            })
        })
        .collect();
    Ok(DjsonCell {
        source: source.into(),
        system,
        layers,
    })
}

fn parse_system(sys: &Value) -> Option<System> {
    let mima = float_array(sys.get("mima")?)?;
    let [xmin, ymin, xmax, ymax, scale, ..] = mima.as_slice() else {
        return None;
    };
    Some(System {
        merc_bbox: (*xmin, *ymin, *xmax, *ymax),
        scale: *scale,
        name: sys.get("name").and_then(Value::as_str).map(str::to_owned),
        pid: sys.get("pid").and_then(Value::as_u64),
    })
}

fn parse_record(layer_key: &str, rec: &Map<String, Value>) -> Option<Record> {
    let xy = rec.get("_XY").and_then(float_array)?;
    let coords: Vec<Coord> = xy
        .as_chunks::<2>()
        .0
        .iter()
        .map(|&[x, y]| {
            let (lon, lat) = rnc_format::rnc_merc_to_wgs84(x, y);
            Coord { x: lon, y: lat }
        })
        .collect();
    let first = *coords.first()?;
    let n = coords.len();

    let geometry = if layer_key == SOUNDG_LAYER {
        let depths = rec.get("_SD").and_then(float_array)?;
        if depths.len() != n {
            return None;
        }
        Geometry::Soundings(
            coords
                .iter()
                .zip(depths)
                .map(|(&c, depth)| (Point::from(c), depth))
                .collect(),
        )
    } else {
        match GeomKind::from_layer_key(layer_key) {
            GeomKind::Point => Geometry::Point(Point::from(first)),
            GeomKind::Line => {
                let mut parts = Vec::new();
                for key in ["_SL", "_SO", "_SP"] {
                    parts = index_array(rec, key)?;
                    if !parts.is_empty() {
                        break;
                    }
                }
                Geometry::Lines(MultiLineString::new(
                    part_ranges(n, non_empty(&parts))
                        .into_iter()
                        .filter_map(|r| coords.get(r).map(|c| LineString::new(c.to_vec())))
                        .collect(),
                ))
            }
            GeomKind::Area => {
                let so = index_array(rec, "_SO")?;
                let sp = index_array(rec, "_SP")?;
                Geometry::Areas(MultiPolygon::new(
                    polygon_ranges(n, non_empty(&so), non_empty(&sp))
                        .into_iter()
                        .filter_map(|rings| polygon(&coords, rings))
                        .collect(),
                ))
            }
        }
    };
    let empty = match &geometry {
        Geometry::Point(_) => false,
        Geometry::Soundings(s) => s.is_empty(),
        Geometry::Lines(lines) => lines.0.is_empty(),
        Geometry::Areas(polys) => polys.0.is_empty(),
    };
    if empty {
        return None;
    }

    Some(Record {
        geometry,
        scamin: rec.get("SCAMIN").and_then(Value::as_f64),
        attributes: rec
            .iter()
            .filter(|(k, _)| !INTERNAL_KEYS.contains(&k.as_str()))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect(),
    })
}

/// Build a polygon from vertex ranges: the first is the exterior, the rest holes.
fn polygon(coords: &[Coord], rings: Vec<Range<usize>>) -> Option<Polygon> {
    let mut rings = rings
        .into_iter()
        .filter_map(|r| coords.get(r).map(|c| LineString::new(c.to_vec())));
    let exterior = rings.next()?;
    Some(Polygon::new(exterior, rings.collect()))
}

/// Vertex index ranges described by a part array (`_SL` / `_SO` / `_SP`).
///
/// Part arrays hold cumulative end vertex indices, not per-part lengths
/// (reading continues from where the previous part ended, up to `ends[i]`),
/// and `_SO` additionally starts with `0`.  Empty ranges (such as that
/// leading `0`) are skipped; ends are clamped to `n`.  `None` → one part.
#[must_use]
pub fn part_ranges(n: usize, parts: Option<&[u32]>) -> Vec<Range<usize>> {
    match parts {
        None | Some(&[]) => std::iter::once(0..n).collect(),
        Some(ends) => {
            let mut out = Vec::with_capacity(ends.len());
            let mut start = 0usize;
            for &end in ends {
                let end = (end as usize).min(n);
                if end > start {
                    out.push(start..end);
                    start = end;
                }
            }
            out
        }
    }
}

/// Vertex index ranges of each polygon's rings (outer ring first, then holes).
///
/// `_SO` boundaries split the
/// vertices into polygons; inside each polygon, the `_SP` entries strictly
/// between its bounds start its holes.  Without `_SO` there is one polygon
/// whose rings are given by `_SP` alone (earcut hole indices).
#[must_use]
pub fn polygon_ranges(n: usize, so: Option<&[u32]>, sp: Option<&[u32]>) -> Vec<Vec<Range<usize>>> {
    let Some(so) = so.filter(|s| !s.is_empty()) else {
        return vec![part_ranges(n, sp)];
    };
    part_ranges(n, Some(so))
        .into_iter()
        .map(|poly| {
            let mut rings = Vec::new();
            let mut start = poly.start;
            for &hole in sp.unwrap_or_default() {
                let hole = hole as usize;
                if hole > start && hole < poly.end {
                    rings.push(start..hole);
                    start = hole;
                }
            }
            rings.push(start..poly.end);
            rings
        })
        .collect()
}

/// All elements as numbers; `None` if any is not one. Dropping just that
/// element would shift every later x/y pair or depth into the wrong slot.
fn float_array(v: &Value) -> Option<Vec<f64>> {
    v.as_array()?.iter().map(Value::as_f64).collect()
}

/// Vertex indices of a part array; empty when the key is missing or not an
/// array. `None` (unusable record) when any entry is not a whole
/// non-negative number: dropping just that entry would merge its
/// neighbouring parts into one line or polygon.
fn index_array(rec: &Map<String, Value>, key: &str) -> Option<Vec<u32>> {
    let Some(arr) = rec.get(key).and_then(Value::as_array) else {
        return Some(Vec::new());
    };
    arr.iter()
        .map(|v| v.as_f64().and_then(whole_number))
        .collect()
}

/// An empty part array means "no parts given".
fn non_empty(indices: &[u32]) -> Option<&[u32]> {
    (!indices.is_empty()).then_some(indices)
}

/// `2.0` → `2`; `None` for fractions, negatives, non-finite values and
/// values beyond `u32`.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)] // range-checked
pub(crate) fn whole_number(f: f64) -> Option<u32> {
    (f.is_finite() && f >= 0.0 && f.fract() == 0.0 && f <= f64::from(u32::MAX)).then_some(f as u32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// `n` vertices along the equator, 1 km apart, as a flat `_XY` array.
    fn xy(n: u32) -> Vec<f64> {
        let (_, y0) = rnc_format::wgs84_to_rnc_merc(0.0, 0.0);
        (0..n).flat_map(|i| [f64::from(i) * 1000.0, y0]).collect()
    }

    #[allow(clippy::needless_pass_by_value)] // call sites build the record inline
    fn one_record(key: &str, rec: Value) -> Record {
        let cell = from_value("t", &json!({ key: [rec] })).unwrap();
        cell.layers[0].records[0].clone()
    }

    fn line_lens(g: &Geometry) -> Vec<usize> {
        let Geometry::Lines(lines) = g else {
            panic!("{g:?}")
        };
        lines.0.iter().map(|l| l.0.len()).collect()
    }

    /// Ring vertex counts per polygon, closing vertex included.
    fn ring_lens(g: &Geometry) -> Vec<Vec<usize>> {
        let Geometry::Areas(polys) = g else {
            panic!("{g:?}")
        };
        polys
            .0
            .iter()
            .map(|p| {
                std::iter::once(p.exterior())
                    .chain(p.interiors())
                    .map(|r| r.0.len())
                    .collect()
            })
            .collect()
    }

    #[test]
    fn line_parts_are_cumulative_end_indices() {
        let rec = one_record("L_COALNE", json!({ "_XY": xy(12), "_SL": [4, 8, 12] }));
        assert_eq!(line_lens(&rec.geometry), [4, 4, 4]);
        // Part 3 starts at vertex 8, not joined onto part 2.
        let Geometry::Lines(lines) = &rec.geometry else {
            unreachable!()
        };
        let (lon8, _) = rnc_format::rnc_merc_to_wgs84(8000.0, 0.0);
        assert!((lines.0[2].0[0].x - lon8).abs() < 1e-12);
    }

    #[test]
    fn line_parts_fall_back_from_sl_to_so_to_sp() {
        let sl = one_record("L_X", json!({ "_XY": xy(4), "_SL": [4], "_SO": [0, 2, 4] }));
        assert_eq!(line_lens(&sl.geometry), [4]);
        let so = one_record("L_X", json!({ "_XY": xy(4), "_SO": [0, 2, 4], "_SP": [4] }));
        assert_eq!(line_lens(&so.geometry), [2, 2]);
        let sp = one_record("L_X", json!({ "_XY": xy(4), "_SP": [1, 4] }));
        assert_eq!(line_lens(&sp.geometry), [1, 3]);
    }

    #[test]
    fn part_indices_may_be_whole_floats() {
        let rec = one_record("L_X", json!({ "_XY": xy(4), "_SL": [2.0, 4.0] }));
        assert_eq!(line_lens(&rec.geometry), [2, 2]);
    }

    #[test]
    fn part_ranges_clamp_to_vertex_count() {
        assert_eq!(part_ranges(10, Some(&[4, 12])), [0..4, 4..10]);
    }

    #[test]
    fn polygon_sp_rings_are_outer_then_holes() {
        // _SP = [5, 10, 15]: outer ring + two holes, each 5 vertices
        // (earcut.js: holeIndices[0] = end of outer ring, last = total).
        let rec = one_record("A_DEPARE", json!({ "_XY": xy(15), "_SP": [5, 10, 15] }));
        // Each 5-vertex ring gets closed → 6 coordinates.
        assert_eq!(ring_lens(&rec.geometry), [vec![6, 6, 6]]);
    }

    #[test]
    fn polygon_offsets_with_leading_zero() {
        // _SO = [0, 4, 8, 12]: polygon boundaries starting at 0.
        let rec = one_record("A_DEPARE", json!({ "_XY": xy(12), "_SO": [0, 4, 8, 12] }));
        assert_eq!(ring_lens(&rec.geometry), [vec![5], vec![5], vec![5]]);
    }

    #[test]
    fn polygon_so_splits_polygons_and_sp_splits_their_rings() {
        // Slice _XY by _SO, then the _SP
        // entries inside each slice are that polygon's hole starts.
        let rec = one_record(
            "A_DEPARE",
            json!({ "_XY": xy(20), "_SO": [0, 10, 20], "_SP": [5, 8, 10, 15, 20] }),
        );
        assert_eq!(ring_lens(&rec.geometry), [vec![6, 4, 3], vec![6, 6]]);
    }

    #[test]
    fn soundings_pair_every_position_with_its_depth() {
        let rec = one_record(
            SOUNDG_LAYER,
            json!({ "_XY": xy(3), "_SD": [1.5, 2.5, 3.5] }),
        );
        let Geometry::Soundings(s) = &rec.geometry else {
            panic!()
        };
        assert_eq!(
            s.iter().map(|&(_, d)| d).collect::<Vec<_>>(),
            [1.5, 2.5, 3.5]
        );
        assert_ne!(s[0].0, s[1].0, "each sounding at its own _XY pair");
    }

    #[test]
    fn point_takes_the_first_pair_and_odd_xy_is_tolerated() {
        let rec = one_record("P_LIGHTS", json!({ "_XY": [0.0, 0.0, 1000.0] }));
        let (lon, lat) = rnc_format::rnc_merc_to_wgs84(0.0, 0.0);
        assert_eq!(rec.geometry, Geometry::Point(Point::new(lon, lat)));
    }

    #[test]
    fn records_without_usable_geometry_are_skipped() {
        let cell = from_value(
            "t",
            &json!({
                "P_LIGHTS": [{ "OBJNAM": "no xy" }, { "_XY": [0.0] }, 42, { "_XY": [0.0, 0.0] }],
                SOUNDG_LAYER: [
                    { "_XY": [0.0, 0.0] },
                    // A gap or a short _SD would pair depths with the wrong positions.
                    { "_XY": xy(3), "_SD": [1.0, null, 3.0] },
                    { "_XY": xy(3), "_SD": [1.0, 2.0] },
                    { "_XY": xy(3), "_SD": [] },
                ],
                // A gap in _XY would swap every later x and y.
                "L_GAP": [{ "_XY": [0.0, null, 1000.0, 0.0, 2000.0, 0.0] }],
                // Part arrays that leave no part.
                "L_EMPTY": [{ "_XY": xy(2), "_SL": [0] }],
                "A_EMPTY": [{ "_XY": xy(3), "_SO": [0] }],
                // A dropped part index would merge neighbouring parts.
                "L_BAD_INDEX": [
                    { "_XY": xy(4), "_SL": [null, 4] },
                    { "_XY": xy(4), "_SL": [-2, 4] },
                    { "_XY": xy(4), "_SL": [2.5, 4] },
                ],
                "A_BAD_INDEX": [
                    { "_XY": xy(8), "_SO": [0, "4", 8] },
                    { "_XY": xy(8), "_SP": [4, null] },
                ],
                "A_BROKEN": "not an array",
            }),
        )
        .unwrap();
        let counts: Vec<_> = cell
            .layers
            .iter()
            .map(|l| (l.key.as_str(), l.records.len()))
            .collect();
        assert_eq!(
            counts,
            [
                ("A_BAD_INDEX", 0),
                ("A_EMPTY", 0),
                ("L_BAD_INDEX", 0),
                ("L_EMPTY", 0),
                ("L_GAP", 0),
                ("P_LIGHTS", 1),
                ("P_SOUNDG", 0)
            ]
        );
    }

    #[test]
    fn layers_are_sorted_by_key() {
        let cell = from_value("t", &json!({ "Z_B": [], "A_A": [], "M_C": [] })).unwrap();
        let keys: Vec<_> = cell.layers.iter().map(|l| l.key.as_str()).collect();
        assert_eq!(keys, ["A_A", "M_C", "Z_B"]);
    }

    #[test]
    fn attributes_keep_s57_keys_and_drop_internal_ones() {
        let rec = one_record(
            "P_LIGHTS",
            json!({
                "_XY": [0.0, 0.0], "_SD": [1.0], "_SL": [1], "_SO": [0, 1], "_SP": [1], "_RR": 1,
                "_Z": 1, "_LY": 1, "_BL": 1, "_PN": 1, "_MOVEX": 1, "_MOVEY": 1, "_LIDXY": [1],
                "SCAMIN": 50000, "OBJNAM": "Light",
            }),
        );
        let keys: Vec<_> = rec.attributes.keys().map(String::as_str).collect();
        assert_eq!(keys, ["OBJNAM", "SCAMIN"]);
        assert_eq!(
            rec.attributes["SCAMIN"],
            json!(50000),
            "value kept as in the document"
        );
        assert_eq!(rec.scamin, Some(50000.0));
    }

    #[test]
    fn system_block_gives_bbox_and_scale() {
        let cell = from_value("t", &json!({
            "__system__": { "mima": [1.0, 2.0, 3.0, 4.0, 50000, 50000], "name": "CELL0001", "pid": 7 },
        }))
        .unwrap();
        assert!(cell.layers.is_empty(), "__system__ is not a layer");
        assert_eq!(
            cell.system,
            Some(System {
                merc_bbox: (1.0, 2.0, 3.0, 4.0),
                scale: 50000.0,
                name: Some("CELL0001".into()),
                pid: Some(7),
            })
        );
        let short = from_value("t", &json!({ "__system__": { "mima": [1.0, 2.0] } })).unwrap();
        assert_eq!(short.system, None);
    }

    #[test]
    fn layer_class_and_kind_follow_the_key() {
        let cell = from_value(
            "t",
            &json!({ "A_DEPARE": [], "L_DEPCNT": [], "P_LIT": [], "X": [] }),
        )
        .unwrap();
        let got: Vec<_> = cell.layers.iter().map(|l| (l.class(), l.kind())).collect();
        assert_eq!(
            got,
            [
                ("DEPARE", GeomKind::Area),
                ("DEPCNT", GeomKind::Line),
                ("LIT", GeomKind::Point),
                ("X", GeomKind::Area),
            ]
        );
    }

    #[test]
    fn non_object_root_and_bad_json_are_errors() {
        assert!(matches!(parse("t", b"[]"), Err(DjsonError::RootNotObject)));
        assert!(matches!(parse("t", b"{"), Err(DjsonError::Json(_))));
    }

    #[test]
    fn detect_recognises_a_json_object_by_its_first_byte() {
        for djson in [&b"{\"A_DEPARE\":[]}"[..], b"  \n\t{}"] {
            assert!(detect(djson), "{djson:?}");
        }
        for other in [
            &b""[..],
            b"   ",
            b"[]",
            b"RXTILES\x04",
            b"\xc8\x00\x12\x00\x00\x00",
        ] {
            assert!(!detect(other), "{other:?}");
        }
    }
}
