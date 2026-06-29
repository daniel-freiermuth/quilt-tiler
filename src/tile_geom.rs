//! Precomputed spatial context for a single `(z, col, row)` tile.

use crate::bbox::Bbox;
use geo::MultiPolygon;

/// Spatial context for one tile.
///
/// Holds its extent (possibly non-rectangular — `geom` need not be a single
/// rectangle) in WGS84, the same extent's Mercator-projected bounding
/// rectangle (pixel projection is inherently rectangular), plus the nominal
/// scale — everything a [`crate::tile_source::TileSource`] render call needs.
#[derive(Clone, Debug)]
pub struct TileGeom {
    /// Coverage contribution for this source in WGS84.  Area features are
    /// clipped to this region so coverage-owning cells don't overlap.
    pub geom: MultiPolygon,
    /// Bounding box of the full tile in Web Mercator metres, used to
    /// project clipped WGS84 coordinates to tile-pixel space.
    pub merc: Bbox,
    /// Nominal scale denominator at this zoom level (`zoom_offset` already
    /// applied).  SCAMIN checks compare directly: skip if `scamin < tile.scale`.
    pub scale: u32,
}
