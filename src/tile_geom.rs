//! Precomputed spatial context for a single `(z, col, row)` tile.

use crate::bbox::Bbox;

/// Spatial context for one tile: the same rectangle in two coordinate systems,
/// plus the nominal scale — everything a [`crate::tile_source::TileSource`]
/// render call needs.
#[derive(Copy, Clone, Debug)]
pub struct TileGeom {
    /// Tile extent in WGS84 geographic coordinates.
    pub wgs84: Bbox,
    /// Tile extent in Web Mercator metres.
    pub merc: Bbox,
    /// Nominal scale denominator at this zoom level (`zoom_offset` already
    /// applied).  SCAMIN checks compare directly: skip if `scamin < tile.scale`.
    pub scale: u32,
}
