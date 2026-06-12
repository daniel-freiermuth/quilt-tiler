//! Convert a parsed `OesuCell` into a map of `GeoJSON` `FeatureCollections`,
//! one per S-57 object class acronym (e.g. "DEPARE", "LNDARE").
//!
//! Each `GeoJSON` feature includes:
//!   - geometry: Point / `MultiPoint` / `LineString` / Polygon
//!   - properties: decoded S-57 attributes by acronym name
//!   - properties\["layer"\]: the S-57 acronym (for tippecanoe layer assignment)
//!   - `"tippecanoe"`: `{"layer": "<ACRONYM>"}`

use std::collections::HashMap;

use geojson::{Feature, FeatureCollection, Geometry, GeometryValue};
use serde_json::{json, Map, Value as JsonValue};

use oesu::{AttrValue, OesuCell};
use crate::s57::{attribute_acronym, object_acronym};

pub fn cell_to_geojson(
    cell: &OesuCell,
    minzoom: u8,
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

        // tippecanoe layer + minzoom hint (maxzoom added later by quilting).
        let tippecanoe = json!({ "layer": acronym, "minzoom": minzoom });
        props.insert("tippecanoe".into(), tippecanoe);

        // Build GeoJSON geometry.
        // SOUNDG MultiPoint is special: tippecanoe drops Z coordinates, losing
        // all depth values. Split each sounding point into a separate Point
        // feature with depth stored as the VALDCO property instead.
        if let oesu::Geometry::MultiPoint(pts) = &feat.geometry {
            for [lon, lat, depth] in pts {
                let mut snd_props = props.clone();
                // Overwrite VALDCO with the actual sounding depth.
                // Format to at most 1 decimal place, dropping trailing zero.
                let depth_str = if (depth - depth.round()).abs() < 0.05 {
                    #[allow(clippy::cast_possible_truncation)] // clamped by the abs() check above
                    let d = *depth as i32;
                    format!("{d}")
                } else {
                    format!("{depth:.1}")
                };
                snd_props.insert("VALDCO".into(), json!(depth_str));
                let feature = Feature {
                    bbox: None,
                    geometry: Some(Geometry::new(GeometryValue::Point {
                        coordinates: vec![*lon, *lat].into(),
                    })),
                    id: None,
                    properties: Some(snd_props),
                    foreign_members: None,
                };
                by_layer.entry(acronym.clone()).or_default().push(feature);
            }
            continue;
        }

        // Build GeoJSON geometry for all other types
        let geom = match &feat.geometry {
            oesu::Geometry::None => None,

            oesu::Geometry::Point { lon, lat } => Some(Geometry::new(
                GeometryValue::Point { coordinates: vec![*lon, *lat].into() },
            )),

            oesu::Geometry::MultiPoint(_) => unreachable!("handled above"),

            oesu::Geometry::Line(rings) => {
                if rings.len() == 1 {
                    let coords =
                        rings[0].iter().map(|[lon, lat]| vec![*lon, *lat].into()).collect();
                    Some(Geometry::new(GeometryValue::LineString { coordinates: coords }))
                } else {
                    let coords = rings
                        .iter()
                        .map(|r| r.iter().map(|[lon, lat]| vec![*lon, *lat].into()).collect())
                        .collect();
                    Some(Geometry::new(GeometryValue::MultiLineString { coordinates: coords }))
                }
            }

            oesu::Geometry::Area(area) => {
                // Outer ring + optional inner rings
                let coords: Vec<_> = area.rings
                    .iter()
                    .map(|r| r.iter().map(|[lon, lat]| vec![*lon, *lat].into()).collect())
                    .collect();
                if coords.is_empty() {
                    None
                } else {
                    Some(Geometry::new(GeometryValue::Polygon { coordinates: coords }))
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
