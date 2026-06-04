//! Generate a `MapLibre` GL style JSON for the produced vector tiles.
//!
//! The style references a `vector` source with the tile URL template,
//! then defines one layer per major S-57 object class with nautical paint rules.

use serde_json::{json, Value};

/// Generate a complete `MapLibre` style JSON.
///
/// `tile_url` should be an XYZ template, e.g.
///   "<http://localhost:3000/charts/OC-46/{z}/{x}/{y}.pbf>"
/// `bounds` is [W, S, E, N] in decimal degrees.
/// `min_zoom` / `max_zoom` are the tile zoom levels.
pub fn generate_style(
    tile_url: &str,
    bounds: [f64; 4],
    min_zoom: u8,
    max_zoom: u8,
) -> Value {
    let center = [
        f64::midpoint(bounds[0], bounds[2]),
        f64::midpoint(bounds[1], bounds[3]),
    ];

    json!({
        "version": 8,
        "name": "OSENC Chart",
        "center": center,
        "zoom": 10,
        "sources": {
            "enc": {
                "type": "vector",
                "tiles": [tile_url],
                "minzoom": min_zoom,
                "maxzoom": max_zoom,
                "bounds": bounds
            }
        },
        "layers": layers()
    })
}

#[allow(clippy::too_many_lines)]
fn layers() -> Value {
    json!([
        // ── Land ────────────────────────────────────────────────────────────
        {
            "id": "LNDARE",
            "type": "fill",
            "source": "enc",
            "source-layer": "LNDARE",
            "paint": {
                "fill-color": "#f5e9c8",
                "fill-outline-color": "#c8b880"
            }
        },
        {
            "id": "BUAARE",
            "type": "fill",
            "source": "enc",
            "source-layer": "BUAARE",
            "paint": {
                "fill-color": "#e8d8a0",
                "fill-outline-color": "#c8b880"
            }
        },
        // ── Depth areas ─────────────────────────────────────────────────────
        // Very shallow (0–2 m)
        {
            "id": "DEPARE-shoal",
            "type": "fill",
            "source": "enc",
            "source-layer": "DEPARE",
            "filter": ["<=", ["get", "DRVAL1"], 2],
            "paint": {
                "fill-color": "#aed6f1",
                "fill-outline-color": "#5dade2"
            }
        },
        // Shallow (2–10 m)
        {
            "id": "DEPARE-shallow",
            "type": "fill",
            "source": "enc",
            "source-layer": "DEPARE",
            "filter": ["all", [">", ["get", "DRVAL1"], 2], ["<=", ["get", "DRVAL1"], 10]],
            "paint": {
                "fill-color": "#d6eaf8",
                "fill-outline-color": "#5dade2"
            }
        },
        // Deep (>10 m)
        {
            "id": "DEPARE-deep",
            "type": "fill",
            "source": "enc",
            "source-layer": "DEPARE",
            "filter": [">", ["get", "DRVAL1"], 10],
            "paint": {
                "fill-color": "#eaf4fc",
                "fill-outline-color": "#85c1e9"
            }
        },
        // ── Dredged areas ───────────────────────────────────────────────────
        {
            "id": "DRGARE",
            "type": "fill",
            "source": "enc",
            "source-layer": "DRGARE",
            "paint": {
                "fill-color": "#a9cce3",
                "fill-pattern": null
            }
        },
        // ── Restricted / special areas ───────────────────────────────────
        {
            "id": "RESARE",
            "type": "fill",
            "source": "enc",
            "source-layer": "RESARE",
            "paint": {
                "fill-color": "rgba(200, 0, 0, 0.08)",
                "fill-outline-color": "rgba(200, 0, 0, 0.6)"
            }
        },
        {
            "id": "SEAARE",
            "type": "fill",
            "source": "enc",
            "source-layer": "SEAARE",
            "paint": {
                "fill-color": "rgba(100, 149, 237, 0.05)",
                "fill-outline-color": "rgba(100, 149, 237, 0.4)"
            }
        },
        // ── Coastline ───────────────────────────────────────────────────────
        {
            "id": "COALNE",
            "type": "line",
            "source": "enc",
            "source-layer": "COALNE",
            "paint": {
                "line-color": "#5d6d7e",
                "line-width": 1.5
            }
        },
        {
            "id": "SLCONS",
            "type": "line",
            "source": "enc",
            "source-layer": "SLCONS",
            "paint": {
                "line-color": "#808b96",
                "line-width": 1
            }
        },
        // ── Depth contours ──────────────────────────────────────────────────
        {
            "id": "DEPCNT",
            "type": "line",
            "source": "enc",
            "source-layer": "DEPCNT",
            "paint": {
                "line-color": "#5dade2",
                "line-width": 0.5,
                "line-opacity": 0.7
            }
        },
        // ── Navigation lines ─────────────────────────────────────────────
        {
            "id": "NAVLNE",
            "type": "line",
            "source": "enc",
            "source-layer": "NAVLNE",
            "paint": {
                "line-color": "#884ea0",
                "line-width": 1,
                "line-dasharray": [4, 2]
            }
        },
        {
            "id": "RECTRC",
            "type": "line",
            "source": "enc",
            "source-layer": "RECTRC",
            "paint": {
                "line-color": "#884ea0",
                "line-width": 1
            }
        },
        {
            "id": "TRAFIC",
            "type": "line",
            "source": "enc",
            "source-layer": "TRAFIC",
            "paint": {
                "line-color": "#884ea0",
                "line-width": 1,
                "line-dasharray": [6, 3]
            }
        },
        // ── Soundings ───────────────────────────────────────────────────────
        {
            "id": "SOUNDG",
            "type": "symbol",
            "source": "enc",
            "source-layer": "SOUNDG",
            "minzoom": 13,
            "layout": {
                "text-field": ["get", "VALDCO"],
                "text-size": 10,
                "text-font": ["Roboto Regular"],
                "text-allow-overlap": false,
                "symbol-placement": "point"
            },
            "paint": {
                "text-color": "#1a5276",
                "text-halo-color": "rgba(255,255,255,0.8)",
                "text-halo-width": 1
            }
        },
        // ── Lateral buoys ────────────────────────────────────────────────
        {
            "id": "BOYLAT",
            "type": "circle",
            "source": "enc",
            "source-layer": "BOYLAT",
            "minzoom": 10,
            "paint": {
                "circle-color": [
                    "match", ["get", "COLOUR"],
                    "3", "#e74c3c",   // red = port
                    "4", "#2ecc71",   // green = starboard
                    "#888888"         // default
                ],
                "circle-radius": 5,
                "circle-stroke-color": "#ffffff",
                "circle-stroke-width": 1
            }
        },
        {
            "id": "BOYCAR",
            "type": "circle",
            "source": "enc",
            "source-layer": "BOYCAR",
            "minzoom": 10,
            "paint": {
                "circle-color": "#f39c12",
                "circle-radius": 5,
                "circle-stroke-color": "#000000",
                "circle-stroke-width": 1
            }
        },
        {
            "id": "BOYSPP",
            "type": "circle",
            "source": "enc",
            "source-layer": "BOYSPP",
            "minzoom": 10,
            "paint": {
                "circle-color": "#f1c40f",
                "circle-radius": 5,
                "circle-stroke-color": "#000000",
                "circle-stroke-width": 1
            }
        },
        // ── Lateral beacons ──────────────────────────────────────────────
        {
            "id": "BCNLAT",
            "type": "circle",
            "source": "enc",
            "source-layer": "BCNLAT",
            "minzoom": 10,
            "paint": {
                "circle-color": [
                    "match", ["get", "COLOUR"],
                    "3", "#e74c3c",
                    "4", "#2ecc71",
                    "#888888"
                ],
                "circle-radius": 4,
                "circle-stroke-color": "#ffffff",
                "circle-stroke-width": 1
            }
        },
        {
            "id": "BCNSPP",
            "type": "circle",
            "source": "enc",
            "source-layer": "BCNSPP",
            "minzoom": 10,
            "paint": {
                "circle-color": "#f1c40f",
                "circle-radius": 4,
                "circle-stroke-color": "#000000",
                "circle-stroke-width": 1
            }
        },
        // ── Lights ──────────────────────────────────────────────────────────
        {
            "id": "LIGHTS",
            "type": "circle",
            "source": "enc",
            "source-layer": "LIGHTS",
            "minzoom": 10,
            "paint": {
                "circle-color": "#f7dc6f",
                "circle-radius": 4,
                "circle-stroke-color": "#d4ac0d",
                "circle-stroke-width": 1.5
            }
        },
        // ── Wrecks & obstructions ────────────────────────────────────────
        {
            "id": "WRECKS",
            "type": "circle",
            "source": "enc",
            "source-layer": "WRECKS",
            "minzoom": 11,
            "paint": {
                "circle-color": "#e74c3c",
                "circle-radius": 4,
                "circle-stroke-color": "#922b21",
                "circle-stroke-width": 1
            }
        },
        {
            "id": "OBSTRN",
            "type": "circle",
            "source": "enc",
            "source-layer": "OBSTRN",
            "minzoom": 11,
            "paint": {
                "circle-color": "#e67e22",
                "circle-radius": 3,
                "circle-stroke-color": "#784212",
                "circle-stroke-width": 1
            }
        },
        // ── Rocks ────────────────────────────────────────────────────────
        {
            "id": "UWTROC",
            "type": "circle",
            "source": "enc",
            "source-layer": "UWTROC",
            "minzoom": 11,
            "paint": {
                "circle-color": "#c0392b",
                "circle-radius": 3,
                "circle-stroke-color": "#7b241c",
                "circle-stroke-width": 1
            }
        },
        // ── Buoy labels ──────────────────────────────────────────────────
        {
            "id": "BOYLAT-label",
            "type": "symbol",
            "source": "enc",
            "source-layer": "BOYLAT",
            "minzoom": 12,
            "layout": {
                "text-field": ["get", "OBJNAM"],
                "text-size": 10,
                "text-font": ["Roboto Regular"],
                "text-offset": [0, 1.2],
                "text-anchor": "top"
            },
            "paint": {
                "text-color": "#1a5276",
                "text-halo-color": "rgba(255,255,255,0.8)",
                "text-halo-width": 1
            }
        }
    ])
}
