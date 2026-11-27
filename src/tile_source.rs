//! [`TileSource`] and [`TileAccumulator`] — the traits that
//! [`crate::tiles::write_pmtiles`] is generic over.
//!
//! Implement both to quilt a new kind of source data into a `PMTiles` archive.
//! Current implementations: [`crate::s57_source`] (S-57 cells → MVT) and
//! [`crate::rnc_source`] (raster cells → PNG).
//!
//! Both traits are intentionally *not* object-safe — they are designed for
//! monomorphised static dispatch only.

use anyhow::Result;
use geo::MultiPolygon;
use pmtiles::TileType;

use crate::bbox::Bbox;
use crate::tile_geom::TileGeom;

/// Accumulates per-cell [`TileSource::Content`]s into a single tile, then
/// encodes them to raw bytes.
///
/// Separating accumulation from rendering keeps [`TileSource::render`] pure
/// (no shared mutable state) and makes the cross-cell merge step explicit.
pub trait TileAccumulator: Send {
    /// The per-cell content type produced by [`TileSource::render`].
    type Content: Send;

    /// Create an empty accumulator for a fresh tile.
    fn empty() -> Self;

    /// Merge one cell's rendered content into this accumulator.
    ///
    /// # Errors
    /// Returns an error if the content cannot be merged (e.g. an encoder
    /// rejects it).
    fn push(&mut self, content: Self::Content) -> Result<()>;

    /// Finish accumulation and encode to raw tile bytes.
    ///
    /// Return an empty `Vec` to omit the tile from the archive.
    ///
    /// # Errors
    /// Returns an error if serialisation fails.
    fn encode(self) -> Result<Vec<u8>>;

    /// Non-empty tile bytes that draw nothing. A sparse archive stores this
    /// for a covered tile whose content encodes to nothing: `PMTiles` cannot
    /// hold an empty tile, and leaving it out would make the client fill it
    /// from an ancestor instead.
    ///
    /// # Errors
    /// Returns an error if serialisation fails.
    fn blank() -> Result<Vec<u8>>;
}

/// A source of data that can be quilted into a `PMTiles` archive.
///
/// # Lazy items
///
/// An item may load its content on demand — from disk or the network — and
/// keep only what [`Self::bbox`], [`Self::native_scale`] and
/// [`Self::tiebreak`] need up front. [`crate::tiles::Quilt`] asks for
/// content only when a tile needs it:
///
/// - [`Self::bbox`] and [`Self::native_scale`]: once per item, when the
///   quilt is built.
/// - [`Self::tiebreak`]: only for candidates whose zoom fit ties with
///   another's.
/// - [`Self::coverage`]: per candidate, until the tile is covered.
/// - [`Self::render`]: only for the items contributing to a tile.
/// - [`Self::detail_zoom`]: only by a sparse quilt, once per item, and only
///   of a tile's candidates when none of them is natively at the tile's
///   zoom or finer.
///
/// The methods that may need content return [`Result`], so a failed load
/// fails the tile instead of silently leaving a hole in it.
pub trait TileSource: Sync {
    /// Per-cell content produced by [`Self::render`] and consumed by
    /// [`Self::Accumulator::push`].
    type Content: Send;

    /// Accumulator that collects one tile's worth of per-cell content and
    /// encodes it to bytes.
    type Accumulator: TileAccumulator<Content = Self::Content>;

    /// Tie-break key for `tiles::Quilt::render_tile`'s candidate sort, when
    /// multiple candidates are equally well-fitted to a tile's zoom level
    /// (same floored `zoom_from_scale`) — the greater value wins the tie.
    type Tiebreaker: Ord;

    /// Source identifier for this item (debug/diagnostic use by callers).
    #[allow(dead_code)] // consumed by debug logging in a follow-up commit
    fn source(&self) -> String;

    /// WGS84 bounds containing all of [`Self::coverage`]. Picks a tile's
    /// candidates, so a looser bbox costs a [`Self::coverage`] call per extra
    /// candidate, and a tighter one loses coverage.
    fn bbox(&self) -> Bbox;

    /// Geographic coverage of this item, in WGS84.
    ///
    /// `write_pmtiles` clips each candidate's coverage to only the area it
    /// actually contributes to a tile, not the whole tile, and passes that
    /// region to [`Self::render`] as [`TileGeom::geom`] — so overlapping
    /// sources never double-paint.
    ///
    /// # Errors
    /// When the item's content can't be loaded.
    fn coverage(&self) -> Result<MultiPolygon>;

    /// Native display scale denominator (e.g. `50_000` for 1:50 000).
    fn native_scale(&self) -> u32;

    /// Finest zoom at which rendering this item still gains content.  At any
    /// finer zoom it shows the same features from the same chart — so a
    /// sparse quilt may leave such tiles out and let the client slice
    /// (vector) or upscale (raster) the ancestor instead. Positions are not
    /// kept: the client's copy carries the ancestor's coordinate grid (see
    /// [`crate::tiles::Quilt`]).
    ///
    /// Never coarser than the item's native zoom
    /// (`zoom::zoom_from_scale(native_scale, zoom_offset)`).  Called at most
    /// once per item per quilt, never concurrently — again only after it
    /// failed — so a scan over (or a load of) the item's content is fine.
    ///
    /// # Errors
    /// When the item's content can't be loaded.
    fn detail_zoom(&self, zoom_offset: f64) -> Result<u8>;

    /// This item's recency/priority ranking — see [`Self::Tiebreaker`].
    ///
    /// # Errors
    /// When the item's content can't be loaded.
    fn tiebreak(&self) -> Result<Self::Tiebreaker>;

    /// Render this item into tile-space content.  `tile.geom` is this item's
    /// exact contribution region for this tile — not necessarily the whole
    /// tile — so all geometry should be clipped against it.
    ///
    /// # Errors
    /// When the item's content can't be loaded.
    fn render(&self, tile: &TileGeom) -> Result<Self::Content>;

    /// `PMTiles` tile type emitted by this source (e.g. `TileType::Mvt`).
    fn tile_type() -> TileType;
}
