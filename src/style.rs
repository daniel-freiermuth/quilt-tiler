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
pub fn build_style(safety_depth: f64, shoal_depth: f64, tile_url: &str, min_zoom: u8, max_zoom: u8) -> String {
    use serde_json::{json, Value};

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
            "tiles": [tile_url],
            "minzoom": min_zoom,
            "maxzoom": max_zoom
        }
    });

    serde_json::to_string_pretty(&style).expect("style serialisation cannot fail")
}
