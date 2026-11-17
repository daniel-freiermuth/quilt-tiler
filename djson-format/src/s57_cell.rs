//! [`DjsonCell`] → [`s57::S57Cell`], the model chart tilers consume.
//!
//! - Native scale and bounds come from `__system__.mima`.
//! - Coverage is resolved from the `M_COVR` areas — `CATCOV = 1` minus
//!   `CATCOV = 2` — by [`s57::resolve_coverage`], like SENC cells' coverage;
//!   without any covered area, the `mima` bounding box.
//! - Layer classes and attributes outside the S-57 catalogue are skipped.
//! - Attribute values take their type from the catalogue ([`s57::AttrType`]),
//!   not from JSON, which cannot tell `5` from `5.0`.
//! - Multi-part records become one feature per line or polygon, as
//!   [`s57::Geometry`] holds a single one.
//! - `.djson` carries no edition date, so cells rank as
//!   [`s57::EditionDate::unknown`] in tie-breaks.

use geo::{LineString, MultiPolygon, Polygon, coord};
use s57::{AttrType, AttrValue, Attribute, Feature, Geometry as S57Geometry};
use serde_json::Value;

use crate::{DjsonCell, DjsonError, Geometry, Record, Result, whole_number};

/// S-57 object class of coverage areas.
const COVERAGE_CLASS: &str = "M_COVR";

impl TryFrom<DjsonCell> for s57::S57Cell {
    type Error = DjsonError;

    /// The cell's `source` identifies it in logs and names it when
    /// `__system__` has no `name`.
    ///
    /// # Errors
    /// [`DjsonError::MissingSystem`] without a usable `__system__.mima`,
    /// [`DjsonError::InvalidScale`] when its scale is not a `1:N` denominator.
    fn try_from(cell: DjsonCell) -> Result<Self> {
        let DjsonCell {
            source,
            system,
            layers,
        } = cell;
        let system = system.ok_or(DjsonError::MissingSystem)?;
        if !(system.scale.is_finite() && system.scale >= 1.0) {
            return Err(DjsonError::InvalidScale(system.scale));
        }
        // Range-checked above; clamped to u32::MAX.
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let native_scale = system.scale.round().min(f64::from(u32::MAX)) as u32;

        let (xmin, y_north, xmax, y_south) = system.merc_bbox;
        let (west, north) = rnc_format::rnc_merc_to_wgs84(xmin, y_north);
        let (east, south) = rnc_format::rnc_merc_to_wgs84(xmax, y_south);
        let bounds = [
            west.min(east),
            south.min(north),
            west.max(east),
            south.max(north),
        ];

        let mut features = Vec::new();
        let (mut covered, mut uncovered) = (Vec::new(), Vec::new());
        for layer in layers {
            let class = layer.class();
            let Some(type_code) = s57::object_code(class) else {
                tracing::debug!(
                    cell = %source,
                    class,
                    records = layer.records.len(),
                    "skipped records of a non-S-57 class"
                );
                continue;
            };
            let is_coverage = class == COVERAGE_CLASS;
            for rec in layer.records {
                if is_coverage && let Geometry::Areas(polys) = &rec.geometry {
                    // Typed like every other attribute, so `2`, `"2"` and `2.0`
                    // all mean "not covered".
                    let catcov = rec
                        .attributes
                        .get("CATCOV")
                        .and_then(|v| attr_value(AttrType::Enumerated, v));
                    match catcov {
                        Some(AttrValue::Int(2)) => uncovered.extend(polys.iter().cloned()),
                        _ => covered.extend(polys.iter().cloned()),
                    }
                }
                push_features(type_code, rec, &mut features);
            }
        }

        let coverage = if covered.is_empty() {
            MultiPolygon::new(vec![bbox_polygon(bounds)])
        } else {
            s57::resolve_coverage(&source, &covered, &uncovered)
        };

        Ok(Self {
            name: system.name.unwrap_or_else(|| source.clone()),
            native_scale,
            bounds,
            features,
            coverage,
            source,
            edition_date: s57::EditionDate::unknown(),
        })
    }
}

/// One [`Feature`] per point/sounding record, line part or polygon.
fn push_features(type_code: u16, rec: Record, out: &mut Vec<Feature>) {
    let attributes = attributes(&rec);
    let mut push = |geometry: S57Geometry| {
        out.push(Feature {
            type_code,
            attributes: attributes.clone(),
            geometry,
        });
    };
    match rec.geometry {
        Geometry::Point(p) => push(S57Geometry::Point(p)),
        Geometry::Soundings(s) => push(S57Geometry::Soundings(s)),
        Geometry::Lines(lines) => {
            for line in lines {
                push(S57Geometry::Line(line));
            }
        }
        Geometry::Areas(polys) => {
            for poly in polys {
                push(S57Geometry::Area(poly));
            }
        }
    }
}

/// S-57 attributes of a record, typed by the attribute catalogue.
fn attributes(rec: &Record) -> Vec<Attribute> {
    rec.attributes
        .iter()
        .filter_map(|(acronym, value)| {
            let (code, kind) = s57::attribute(acronym)?;
            Some(Attribute {
                code,
                value: attr_value(kind, value)?,
            })
        })
        .collect()
}

/// Convert a JSON value to the catalogue's type; `None` when it does not fit.
fn attr_value(kind: AttrType, value: &Value) -> Option<AttrValue> {
    match kind {
        AttrType::Enumerated | AttrType::Integer => {
            let n = match value {
                Value::Number(n) => n.as_f64()?,
                Value::String(s) => s.trim().parse().ok()?,
                _ => return None,
            };
            whole_number(n).map(AttrValue::Int)
        }
        AttrType::Float => match value {
            Value::Number(n) => n.as_f64().map(AttrValue::Double),
            Value::String(s) => s.trim().parse().ok().map(AttrValue::Double),
            _ => None,
        },
        AttrType::List | AttrType::Coded | AttrType::FreeText => match value {
            Value::String(s) => Some(AttrValue::Str(s.clone())),
            Value::Number(n) => Some(AttrValue::Str(n.to_string())),
            _ => None,
        },
    }
}

fn bbox_polygon([west, south, east, north]: [f64; 4]) -> Polygon {
    Polygon::new(
        LineString::new(vec![
            coord! { x: west, y: south },
            coord! { x: east, y: south },
            coord! { x: east, y: north },
            coord! { x: west, y: north },
            coord! { x: west, y: south },
        ]),
        vec![],
    )
}

#[cfg(test)]
#[allow(clippy::float_cmp)] // attribute values are converted exactly
mod tests {
    use super::*;
    use geo::Area;
    use serde_json::json;

    /// Format Mercator `(x, y)` of a WGS-84 position.
    fn m(lon: f64, lat: f64) -> [f64; 2] {
        rnc_format::wgs84_to_rnc_merc(lon, lat).into()
    }

    /// Flat `_XY` for a list of WGS-84 positions.
    fn xy(pts: &[(f64, f64)]) -> Vec<f64> {
        pts.iter().flat_map(|&(lon, lat)| m(lon, lat)).collect()
    }

    fn square(lon: f64, lat: f64, d: f64) -> Vec<(f64, f64)> {
        vec![
            (lon, lat),
            (lon + d, lat),
            (lon + d, lat + d),
            (lon, lat + d),
        ]
    }

    fn system() -> Value {
        let [xmin, ymin] = m(10.0, 55.0);
        let [xmax, ymax] = m(11.0, 54.0);
        json!({ "mima": [xmin, ymin, xmax, ymax, 50_000, 50_000], "name": "CELL1" })
    }

    fn try_convert(doc: &Value) -> Result<s57::S57Cell> {
        s57::S57Cell::try_from(crate::from_value("test", doc).unwrap())
    }

    fn convert(doc: &Value) -> s57::S57Cell {
        try_convert(doc).unwrap()
    }

    #[test]
    fn system_gives_name_scale_and_bounds() {
        let cell = convert(&json!({ "__system__": system() }));
        assert_eq!(cell.name, "CELL1");
        assert_eq!(cell.native_scale, 50_000);
        let [w, s, e, n] = cell.bounds;
        for (got, want) in [(w, 10.0), (s, 54.0), (e, 11.0), (n, 55.0)] {
            assert!((got - want).abs() < 1e-9, "{:?}", cell.bounds);
        }
        // No M_COVR: coverage falls back to the bbox.
        assert!((cell.coverage.unsigned_area() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn name_falls_back_to_the_source() {
        let mut sys = system();
        sys.as_object_mut().unwrap().remove("name");
        let cell = s57::S57Cell::try_from(
            crate::from_value("cells/X.djson", &json!({ "__system__": sys })).unwrap(),
        )
        .unwrap();
        assert_eq!(cell.name, "cells/X.djson");
        assert_eq!(cell.source, "cells/X.djson");
    }

    #[test]
    fn missing_system_or_bad_scale_is_an_error() {
        assert!(matches!(
            try_convert(&json!({ "P_LIGHTS": [] })),
            Err(DjsonError::MissingSystem)
        ));
        let mut sys = system();
        sys["mima"][4] = json!(0);
        assert!(matches!(
            try_convert(&json!({ "__system__": sys })),
            Err(DjsonError::InvalidScale(s)) if s == 0.0
        ));
    }

    #[test]
    fn coverage_is_covered_minus_uncovered_m_covr() {
        for not_covered in [json!(2), json!("2"), json!(2.0)] {
            let cell = convert(&json!({
                "__system__": system(),
                "A_M_COVR": [
                    { "_XY": xy(&square(10.0, 54.0, 0.25)), "CATCOV": 1 },
                    { "_XY": xy(&square(10.5, 54.5, 0.25)), "CATCOV": 1 },
                    { "_XY": xy(&square(10.0, 54.0, 0.1)), "CATCOV": not_covered },
                ],
            }));
            // Two 0.25° squares (0.0625 each) minus a 0.1° notch (0.01) out of the first.
            let want = 0.115;
            let got = cell.coverage.unsigned_area();
            assert!((got - want).abs() < 1e-9, "CATCOV {not_covered}: {got}");
        }
    }

    /// A not-covered area containing every covered one is a nominal-extent
    /// placeholder, not a gap: the cell keeps its covered areas.
    #[test]
    fn not_covered_area_containing_everything_keeps_coverage() {
        let cell = convert(&json!({
            "__system__": system(),
            "A_M_COVR": [
                { "_XY": xy(&square(10.0, 54.0, 0.25)), "CATCOV": 1 },
                { "_XY": xy(&square(10.5, 54.5, 0.25)), "CATCOV": 1 },
                { "_XY": xy(&square(10.0, 54.0, 1.0)), "CATCOV": 2 },
            ],
        }));
        let got = cell.coverage.unsigned_area();
        assert!((got - 0.125).abs() < 1e-9, "both 0.25° squares kept: {got}");
    }

    #[test]
    fn attributes_follow_the_catalogue_type_and_unknown_ones_are_dropped() {
        let cell = convert(&json!({
            "__system__": system(),
            "P_LIGHTS": [{
                "_XY": xy(&[(10.5, 54.5)]),
                "SECTR1": 180, "VALNMR": "12.5", "COLOUR": 1, "CATLIT": "1,3",
                "SCAMIN": 50000, "OBJNAM": "Light", "FIDN": "x", "XYZABC": 1,
            }],
        }));
        let f = &cell.features[0];
        assert_eq!(s57::object_acronym(f.type_code), Some("LIGHTS"));
        let by_name = |n: &str| {
            let (code, _) = s57::attribute(n).unwrap();
            f.attributes
                .iter()
                .find(|a| a.code == code)
                .map(|a| a.value.clone())
        };
        assert!(matches!(by_name("SECTR1"), Some(AttrValue::Double(v)) if v == 180.0));
        assert!(matches!(by_name("VALNMR"), Some(AttrValue::Double(v)) if v == 12.5));
        assert!(matches!(by_name("COLOUR"), Some(AttrValue::Str(s)) if s == "1"));
        assert!(matches!(by_name("CATLIT"), Some(AttrValue::Str(s)) if s == "1,3"));
        assert!(matches!(by_name("SCAMIN"), Some(AttrValue::Int(50_000))));
        assert!(matches!(by_name("OBJNAM"), Some(AttrValue::Str(s)) if s == "Light"));
        assert_eq!(
            f.attributes.len(),
            6,
            "FIDN and unknown keys are not S-57 attributes"
        );
    }

    #[test]
    fn integer_attributes_accept_whole_numbers_in_any_json_form() {
        for ok in [json!(2), json!("2"), json!(2.0), json!(" 2.0 ")] {
            assert!(
                matches!(
                    attr_value(AttrType::Enumerated, &ok),
                    Some(AttrValue::Int(2))
                ),
                "{ok}"
            );
        }
        for bad in [json!(2.5), json!(-1), json!("two"), json!(true)] {
            assert!(attr_value(AttrType::Enumerated, &bad).is_none(), "{bad}");
        }
    }

    #[test]
    fn multi_part_records_split_and_unknown_classes_skip() {
        let mut area = square(10.1, 54.1, 0.1);
        area.extend(square(10.5, 54.5, 0.1));
        let cell = convert(&json!({
            "__system__": system(),
            "L_COALNE": [{ "_XY": xy(&[(10.1, 54.1), (10.2, 54.1), (10.3, 54.2), (10.4, 54.2)]), "_SL": [2, 4] }],
            "A_DEPARE": [{ "_XY": xy(&area), "_SO": [0, 4, 8], "DRVAL1": 0, "DRVAL2": 5 }],
            "P_SOUNDG": [{ "_XY": xy(&[(10.2, 54.2), (10.3, 54.3)]), "_SD": [1.5, 2.5] }],
            "A_XXXXXX": [{ "_XY": xy(&square(10.0, 54.0, 0.1)) }],
        }));
        let kinds: Vec<_> = cell
            .features
            .iter()
            .map(|f| {
                let kind = match f.geometry {
                    S57Geometry::None => "none",
                    S57Geometry::Point(_) => "point",
                    S57Geometry::Soundings(_) => "soundings",
                    S57Geometry::Line(_) => "line",
                    S57Geometry::Area(_) => "area",
                };
                (s57::object_acronym(f.type_code).unwrap(), kind)
            })
            .collect();
        assert_eq!(
            kinds,
            [
                ("DEPARE", "area"),
                ("DEPARE", "area"),
                ("COALNE", "line"),
                ("COALNE", "line"),
                ("SOUNDG", "soundings")
            ]
        );
        let depare = &cell.features[0];
        let (drval2, _) = s57::attribute("DRVAL2").unwrap();
        assert!(
            depare
                .attributes
                .iter()
                .any(|a| a.code == drval2 && matches!(a.value, AttrValue::Double(v) if v == 5.0))
        );
        assert!(matches!(&cell.features[4].geometry, S57Geometry::Soundings(s) if s.len() == 2));
    }
}
