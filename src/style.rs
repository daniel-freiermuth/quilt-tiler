//! `MapLibre` GL style generation for OSENC vector tiles.
//!
//! Fixed layers (land, buoys, lights, …) are embedded from `src/style.json`.
//! Depth-related layers (DEPARE fill gradient and DEPCNT safety contour) are
//! generated at runtime from the caller's depth configuration so the chart
//! can be tuned to a specific vessel's draft without editing JSON by hand.
//!
//! The style does **not** include a source definition — the caller registers
//! the tile source under the name `"enc"` before applying the style.

const STYLE_JSON: &str = include_str!("style.json");

/// Build a `MapLibre` GL style JSON string with configurable depth styling.
///
/// # Parameters
/// * `safety_depth` — depth in metres at or below which water is dangerous.
///   The DEPCNT contour at exactly this depth is drawn as a prominent red line.
///   DEPARE areas shallower than this get the darkest blue fill.
/// * `shoal_depth` — upper boundary of the "shallow but navigable" zone.
///   DEPARE areas between `safety_depth` and `shoal_depth` get a medium blue;
///   areas deeper than `shoal_depth` get a very light blue (open water).
/// * `tile_url` — full MVT tile URL template, e.g.
///   `http://localhost:3000/chart/{z}/{x}/{y}`.  Embedded in `sources.enc`.
/// * `min_zoom` / `max_zoom` — zoom range for the tile source.
///
/// # Panics
/// Panics if the embedded `style.json` is malformed (compile-time guarantee).
#[must_use]
pub fn build_style(
    safety_depth: f64,
    shoal_depth: f64,
    tile_url: &str,
    min_zoom: u8,
    max_zoom: u8,
) -> String {
    use serde_json::{Value, json};

    let mut style: Value =
        serde_json::from_str(STYLE_JSON).expect("embedded style.json is valid JSON");

    // --- generated depth layers -------------------------------------------

    // Single DEPARE fill layer using a MapLibre `step` expression so the
    // colour boundaries track the configured depths automatically.
    let depare = json!({
        "id": "DEPARE",
        "type": "fill",
        "source": "enc",
        "source-layer": "DEPARE",
        "paint": {
            // step: output0, stop1, output1, stop2, output2
            //   input < stop1          → output0 (dangerous, darkest blue)
            //   stop1 ≤ input < stop2  → output1 (shallow, medium blue)
            //   input ≥ stop2          → output2 (deep, lightest blue)
            "fill-color": [
                "step", ["to-number", ["get", "DRVAL1"], 9999],
                "#5b9bd5",
                safety_depth, "#aed6f1",
                shoal_depth,  "#d6eaf8"
            ],
            "fill-outline-color": "#5dade2"
        }
    });

    // All depth contours at normal weight
    let depcnt = json!({
        "id": "DEPCNT",
        "type": "line",
        "source": "enc",
        "source-layer": "DEPCNT",
        "paint": {
            "line-color": "#5dade2",
            "line-width": 0.5,
            "line-opacity": 0.7
        }
    });

    // Safety-depth contour rendered prominently in red on top of normal ones
    let depcnt_safety = json!({
        "id": "DEPCNT-safety",
        "type": "line",
        "source": "enc",
        "source-layer": "DEPCNT",
        "filter": ["==", ["to-number", ["get", "VALDCO"], -1.0], safety_depth],
        "paint": {
            "line-color": "#e74c3c",
            "line-width": 2.0,
            "line-opacity": 1.0
        }
    });

    // --- rebuild layers array in place ------------------------------------
    let old_layers = style["layers"]
        .as_array()
        .expect("style.json layers is an array")
        .clone();

    let mut new_layers: Vec<Value> = Vec::with_capacity(old_layers.len() + 1);
    let mut depare_inserted = false;

    for layer in old_layers {
        let id = layer["id"].as_str().unwrap_or("").to_string();
        if id.starts_with("DEPARE") {
            // Collapse all old DEPARE-* variants into a single generated layer
            if !depare_inserted {
                new_layers.push(depare.clone());
                depare_inserted = true;
            }
        } else if id == "DEPCNT" {
            // Replace with base contour + safety-depth highlight
            new_layers.push(depcnt.clone());
            new_layers.push(depcnt_safety.clone());
        } else {
            new_layers.push(layer);
        }
    }

    style["layers"] = Value::Array(new_layers);

    // --- inject sources block so the style is self-contained ---------------
    // MapLibre requires every source referenced in layers to be defined here.
    // The caller supplies the actual tile URL (e.g. from martin or tileserver-gl).
    style["sources"] = json!({
        "enc": {
            "type": "vector",
            "encoding": "mlt",
            "tiles": [tile_url],
            "minzoom": min_zoom,
            "maxzoom": max_zoom
        }
    });

    serde_json::to_string_pretty(&style).expect("style serialisation cannot fail")
}

/// Build a minimal `MapLibre` GL raster style for a PNG tile source.
///
/// Unlike [`build_style`], there is no vector layer styling to generate —
/// a raster source only needs a `tileSize` and one `raster` layer.
///
/// # Parameters
/// * `tile_url` — full PNG tile URL template, e.g.
///   `http://localhost:3000/chart/{z}/{x}/{y}`.
/// * `min_zoom` / `max_zoom` — zoom range for the tile source.
/// # Panics
/// Panics if `serde_json` fails to serialise the style (cannot happen — the
/// value is built entirely from this function's own literals and inputs).
#[must_use]
pub fn build_raster_style(tile_url: &str, min_zoom: u8, max_zoom: u8) -> String {
    use serde_json::json;

    let style = json!({
        "version": 8,
        "sources": {
            "raster": {
                "type": "raster",
                "tiles": [tile_url],
                "tileSize": crate::rnc_source::TILE_PX,
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
            }
        },
        "layers": [
            {
                "id": "raster",
                "type": "raster",
                "source": "raster",
            }
        ]
    });

    serde_json::to_string_pretty(&style).expect("style serialisation cannot fail")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    fn layer<'a>(style: &'a Value, id: &str) -> &'a Value {
        style["layers"]
            .as_array()
            .expect("layers is an array")
            .iter()
            .find(|l| l["id"] == id)
            .unwrap_or_else(|| panic!("layer {id} not found in built style"))
    }

    /// Build the style as a [`Value`] with the given depth configuration.
    fn build(safety_depth: f64, shoal_depth: f64) -> Value {
        serde_json::from_str(&build_style(
            safety_depth,
            shoal_depth,
            "http://localhost/{z}/{x}/{y}",
            6,
            18,
        ))
        .expect("build_style output is valid JSON")
    }

    /// Return every layer whose `"id"` starts with `prefix`.
    fn layers_starting_with<'a>(style: &'a Value, prefix: &str) -> Vec<&'a Value> {
        style["layers"]
            .as_array()
            .expect("layers is an array")
            .iter()
            .filter(|l| {
                l["id"]
                    .as_str()
                    .is_some_and(|id| id.starts_with(prefix))
            })
            .collect()
    }

    // --- DEPARE layer contracts ------------------------------------------

    #[test]
    fn depare_layer_uses_safety_and_shoal_depth_in_step_expression() {
        let style = build(4.5, 12.0);
        let depare = layer(&style, "DEPARE");

        assert_eq!(depare["type"], "fill");
        assert_eq!(depare["source-layer"], "DEPARE");

        let fill_color = depare["paint"]["fill-color"]
            .as_array()
            .expect("fill-color is an array (step expression)");

        // step expression: ["step", input, output0, stop1, output1, stop2, output2]
        assert_eq!(fill_color[0], "step", "fill-color must be a step expression");

        // The stops must match the exact depth values passed in.
        assert_eq!(
            fill_color[3], 4.5,
            "first stop must be safety_depth"
        );
        assert_eq!(
            fill_color[5], 12.0,
            "second stop must be shoal_depth"
        );

        // Input expression: read DRVAL1 as a number.
        let input = fill_color[1]
            .as_array()
            .expect("step input is an expression array");
        assert_eq!(input[0], "to-number");
    }

    #[test]
    fn old_depare_variants_collapsed_to_single_generated_layer() {
        let style = build(5.0, 10.0);
        let depare_layers = layers_starting_with(&style, "DEPARE");

        // The template has DEPARE-shoal, DEPARE-shallow, DEPARE-deep — all
        // three must be collapsed into a single "DEPARE" layer.
        assert_eq!(
            depare_layers.len(),
            1,
            "expected exactly one DEPARE layer, got {}: {:?}",
            depare_layers.len(),
            depare_layers
                .iter()
                .map(|l| l["id"].as_str().unwrap_or("?"))
                .collect::<Vec<_>>()
        );
        assert_eq!(depare_layers[0]["id"], "DEPARE");
    }

    // --- DEPCNT layer contracts ------------------------------------------

    #[test]
    fn depcnt_replaced_with_base_contour_and_safety_highlight() {
        let style = build(7.0, 15.0);
        let layers = style["layers"].as_array().expect("layers is an array");

        let base_idx = layers
            .iter()
            .position(|l| l["id"] == "DEPCNT")
            .expect("base DEPCNT layer missing");
        let safety_idx = layers
            .iter()
            .position(|l| l["id"] == "DEPCNT-safety")
            .expect("DEPCNT-safety layer missing");

        // Safety highlight must be drawn directly after the base contour.
        assert_eq!(
            safety_idx,
            base_idx + 1,
            "DEPCNT-safety must immediately follow DEPCNT"
        );

        // Both must be line layers on the DEPCNT source-layer.
        let base = &layers[base_idx];
        let safety = &layers[safety_idx];
        assert_eq!(base["type"], "line");
        assert_eq!(safety["type"], "line");
        assert_eq!(base["source-layer"], "DEPCNT");
        assert_eq!(safety["source-layer"], "DEPCNT");
    }

    #[test]
    fn safety_depth_contour_filter_matches_exact_value() {
        let style = build(3.25, 10.0);
        let safety = layer(&style, "DEPCNT-safety");

        let filter = safety["filter"]
            .as_array()
            .expect("DEPCNT-safety must have a filter");

        // ["==", ["to-number", ["get", "VALDCO"], -1.0], safety_depth]
        assert_eq!(filter[0], "==");
        assert_eq!(
            filter[2], 3.25,
            "filter comparison value must equal safety_depth"
        );

        // The filtered attribute must be VALDCO.
        let expr = filter[1]
            .as_array()
            .expect("filter left-hand side is an expression");
        assert_eq!(expr[0], "to-number");
        let get = expr[1]
            .as_array()
            .expect("to-number input is a get expression");
        assert_eq!(get[0], "get");
        assert_eq!(get[1], "VALDCO");
    }

    // --- Boundary: degenerate case when safety == shoal ------------------

    #[test]
    fn degenerate_safety_equals_shoal_produces_valid_step() {
        // When safety_depth == shoal_depth the step expression has two
        // adjacent stops at the same value.  MapLibre evaluates the first
        // match, so the middle colour band simply has zero width — the
        // chart goes straight from "dangerous" to "open water".
        let style = build(6.0, 6.0);
        let depare = layer(&style, "DEPARE");

        let fill_color = depare["paint"]["fill-color"]
            .as_array()
            .expect("fill-color is a step expression");

        // Both stops present and equal.
        assert_eq!(fill_color[3], 6.0);
        assert_eq!(fill_color[5], 6.0);

        // Still produces three colour outputs (dangerous, shallow, deep)
        // even when the shallow band is degenerate.
        assert_eq!(
            fill_color.len(),
            7,
            "step expression must have 7 elements: \
             [step, input, output0, stop1, output1, stop2, output2]"
        );
    }

    #[test]
    fn cardinal_buoy_body_and_topmark_are_separate_layers_with_topmark_on_top() {
        let style: Value = serde_json::from_str(&build_style(
            5.0,
            10.0,
            "http://localhost/{z}/{x}/{y}",
            6,
            18,
        ))
        .expect("build_style output is valid JSON");

        let layers = style["layers"].as_array().expect("layers is an array");
        let body_idx = layers
            .iter()
            .position(|l| l["id"] == "BOYCAR-body")
            .expect("BOYCAR-body layer missing");
        let topmark_idx = layers
            .iter()
            .position(|l| l["id"] == "TOPMAR")
            .expect("TOPMAR layer missing");
        assert!(
            body_idx < topmark_idx,
            "buoy body must be drawn before (i.e. beneath) the topmark"
        );
        assert!(
            !layers.iter().any(|l| l["id"] == "BOYCAR-topmark"),
            "BOYCAR-topmark (CATCAM-guessed cone icon) must be gone — topmarks \
             are real S-57 TOPMAR objects with their own TOPSHP shape attribute"
        );

        let body = layer(&style, "BOYCAR-body");
        assert_eq!(body["source-layer"], "BOYCAR");

        // Topmarks are shared across every buoy/beacon class, not just
        // cardinal buoys: one generic layer sourced from the real TOPMAR
        // object class, keyed on its own TOPSHP attribute.
        let topmark = layer(&style, "TOPMAR");
        assert_eq!(topmark["source-layer"], "TOPMAR");
        let icon_image = topmark["layout"]["icon-image"]
            .as_array()
            .expect("icon-image is a match expression array");
        assert_eq!(icon_image[1], serde_json::json!(["get", "TOPSHP"]));
    }
}
