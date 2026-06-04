//! Static `MapLibre` GL style JSON for OSENC vector tiles.
//!
//! The style defines paint rules for each S-57 object class layer.
//! It does not include a source definition — the caller adds the tile source
//! under the name `"enc"` before applying the style.

/// The style JSON, embedded at compile time from `src/style.json`.
pub const STYLE_JSON: &str = include_str!("style.json");
