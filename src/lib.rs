//! Shared core: parsing chart cells, quilting them into tiles, and
//! generating `MapLibre` GL style JSON.
//!
//! Consumed by the batch `PMTiles` writer (`src/main.rs`) and the live tile
//! server (`src/bin/tileserver.rs`) — both build on the same
//! [`tile_source::TileSource`] implementations and [`tiles::render_tile`].

pub mod bbox;
pub mod lattice;
pub mod loader;
pub mod rnc;
pub mod rnc_source;
pub mod s57_source;
pub mod style;
pub mod tile_geom;
pub mod tile_source;
pub mod tiles;
pub mod zoom;
