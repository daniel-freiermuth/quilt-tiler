//! [`TileSource`] — the trait that [`crate::tiles::write_pmtiles`] is generic over.
//!
//! Implement this to quilt a new kind of source data into a `PMTiles` archive.
//! Current implementation: [`crate::s57_source`] (`S-57` cells → MVT).
//!
//! All methods are associated functions (no `&self`): the implementing type
//! carries no runtime state.  If per-run configuration is ever needed, wrap
//! the item type in a newtype and implement the trait on the wrapper.
//!
//! Note: the trait is intentionally *not* object-safe (static functions cannot
//! be called through `dyn TileSource`).  It is designed for monomorphised
//! static dispatch only.

use anyhow::Result;
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;
use crate::tile_geom::TileGeom;

/// A source of data that can be quilted into a `PMTiles` archive.
pub trait TileSource: Sync {
    /// Accumulated tile content produced by [`Self::render`] and consumed by
    /// [`Self::encode`] (e.g. a `HashMap` of MVT layers, or a pixel buffer).
    type Content: Send;

    /// Lattice element used to track coverage within a tile.
    ///
    /// Must convert to and from [`Bbox`]: `write_pmtiles` needs `From<Bbox>` to
    /// construct the tile-shaped coverage sentinel, and `Into<Bbox>` to
    /// aggregate item extents into the overall bounding box for tile iteration.
    type Coverage: BoundedLattice + From<Bbox> + Into<Bbox>;

    /// Geographic coverage of this item.
    fn coverage(&self) -> Self::Coverage;

    /// Native display scale denominator (e.g. `50_000` for 1:50 000).
    fn native_scale(&self) -> u32;

    /// Render this item into tile-space content.
    fn render(&self, tile: &TileGeom) -> Self::Content;

    /// Encode one tile's accumulated `contents` into raw bytes.
    ///
    /// Return an empty `Vec` to omit the tile from the archive.
    fn encode(contents: Vec<Self::Content>) -> Result<Vec<u8>>;

    /// `PMTiles` tile type emitted by this source (e.g. `TileType::Mvt`).
    fn tile_type() -> TileType;
}
