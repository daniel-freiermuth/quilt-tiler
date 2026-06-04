//! Convert a parsed `OsencCell` into a map of `GeoJSON` `FeatureCollections`,
//! one per S-57 object class acronym (e.g. "DEPARE", "LNDARE").
//!
//! Each `GeoJSON` feature includes:
//!   - geometry: Point / `MultiPoint` / `LineString` / Polygon
//!   - properties: decoded S-57 attributes by acronym name
//!   - properties\["layer"\]: the S-57 acronym (for tippecanoe layer assignment)
//!   - `"tippecanoe"`: `{"layer": "<ACRONYM>"}`

use std::collections::HashMap;

use geojson::{Feature, FeatureCollection, Geometry, Value};
use serde_json::{json, Map, Value as JsonValue};

use crate::osenc::{AttrValue, OsencCell};
use crate::s57::{attribute_acronym, object_acronym};

pub fn cell_to_geojson(
    cell: &OsencCell,
) -> HashMap<String, FeatureCollection> {
    let mut by_layer: HashMap<String, Vec<Feature>> = HashMap::new();

    for feat in &cell.features {
        let acronym = object_acronym(feat.type_code)
            .unwrap_or("UNKNOWN")
            .to_string();

        // Build properties map
        let mut props = Map::new();
        props.insert("layer".into(), json!(acronym));
        props.insert("_id".into(), json!(feat.id));
        props.insert("_primitive".into(), json!(feat.primitive));

        for attr in &feat.attributes {
            let key = attribute_acronym(attr.code).map_or_else(|| format!("attr_{}", attr.code), std::string::ToString::to_string);
            let val: JsonValue = match &attr.value {
                AttrValue::Int(v) => json!(v),
                AttrValue::Double(v) => json!(v),
                AttrValue::Str(s) => json!(s),
            };
            props.insert(key, val);
        }

        // tippecanoe layer hint
        let tippecanoe = json!({ "layer": acronym });
        props.insert("tippecanoe".into(), tippecanoe);

        // Build GeoJSON geometry
        let geom = match &feat.geometry {
            crate::osenc::Geometry::None => None,

            crate::osenc::Geometry::Point { lon, lat } => Some(Geometry::new(
                Value::Point(vec![*lon, *lat]),
            )),

            crate::osenc::Geometry::MultiPoint(pts) => {
                let coords: Vec<Vec<f64>> = pts
                    .iter()
                    .map(|[lon, lat, depth]| {
                        if *depth == 0.0 {
                            vec![*lon, *lat]
                        } else {
                            vec![*lon, *lat, *depth]
                        }
                    })
                    .collect();
                Some(Geometry::new(Value::MultiPoint(coords)))
            }

            crate::osenc::Geometry::Line(rings) => {
                if rings.len() == 1 {
                    let coords: Vec<Vec<f64>> =
                        rings[0].iter().map(|[lon, lat]| vec![*lon, *lat]).collect();
                    Some(Geometry::new(Value::LineString(coords)))
                } else {
                    let coords: Vec<Vec<Vec<f64>>> = rings
                        .iter()
                        .map(|r| r.iter().map(|[lon, lat]| vec![*lon, *lat]).collect())
                        .collect();
                    Some(Geometry::new(Value::MultiLineString(coords)))
                }
            }

            crate::osenc::Geometry::Area(rings) => {
                // Outer ring + optional inner rings
                let coords: Vec<Vec<Vec<f64>>> = rings
                    .iter()
                    .map(|r| r.iter().map(|[lon, lat]| vec![*lon, *lat]).collect())
                    .collect();
                if coords.is_empty() {
                    None
                } else {
                    Some(Geometry::new(Value::Polygon(coords)))
                }
            }
        };

        let feature = Feature {
            bbox: None,
            geometry: geom,
            id: None,
            properties: Some(props),
            foreign_members: None,
        };

        by_layer.entry(acronym).or_default().push(feature);
    }

    // Wrap each layer's features in a FeatureCollection
    by_layer
        .into_iter()
        .map(|(acronym, features)| {
            (
                acronym,
                FeatureCollection {
                    bbox: None,
                    features,
                    foreign_members: None,
                },
            )
        })
        .collect()
}
