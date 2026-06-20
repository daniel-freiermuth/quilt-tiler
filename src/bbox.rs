//! Axis-aligned bounding box — the [`BoundedLattice`] instance for bbox algebra.

use crate::lattice::BoundedLattice;

/// An axis-aligned bounding box in WGS84 or projected coordinates.
///
/// Lattice order: `a ≥ b` iff `a` fully contains `b`.
/// Meet = intersection, join = bounding hull.
#[derive(Copy, Clone, Debug)]
pub struct Bbox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Bbox {
    /// A degenerate point-extent bbox.
    #[inline]
    pub const fn point(lon: f64, lat: f64) -> Self {
        Self {
            west: lon,
            south: lat,
            east: lon,
            north: lat,
        }
    }

    /// Smallest bbox enclosing all `pts`; `None` when the iterator is empty.
    pub fn of(mut pts: impl Iterator<Item = (f64, f64)>) -> Option<Self> {
        let (lon, lat) = pts.next()?;
        let mut b = Self::point(lon, lat);
        for (lon, lat) in pts {
            b.west = b.west.min(lon);
            b.south = b.south.min(lat);
            b.east = b.east.max(lon);
            b.north = b.north.max(lat);
        }
        Some(b)
    }

    #[inline]
    pub fn is_bottom(&self) -> bool {
        self.west > self.east || self.south > self.north
    }
}

impl BoundedLattice for Bbox {
    #[inline]
    fn bottom() -> Self {
        Self {
            west: f64::INFINITY,
            south: f64::INFINITY,
            east: f64::NEG_INFINITY,
            north: f64::NEG_INFINITY,
        }
    }

    #[inline]
    fn join(&self, other: &Self) -> Self {
        Self {
            west: self.west.min(other.west),
            south: self.south.min(other.south),
            east: self.east.max(other.east),
            north: self.north.max(other.north),
        }
    }

    #[inline]
    fn meet(&self, other: &Self) -> Self {
        Self {
            west: self.west.max(other.west),
            south: self.south.max(other.south),
            east: self.east.min(other.east),
            north: self.north.min(other.north),
        }
    }

    #[inline]
    fn subsumes(&self, other: &Self) -> bool {
        self.west <= other.west
            && self.south <= other.south
            && self.east >= other.east
            && self.north >= other.north
    }

    /// Avoids constructing the meet.
    #[inline]
    #[allow(clippy::suspicious_operation_groupings)] // cross-axis comparisons are intentional
    fn overlaps(&self, other: &Self) -> bool {
        !self.is_bottom()
            && !other.is_bottom()
            && self.west <= other.east
            && self.east >= other.west
            && self.south <= other.north
            && self.north >= other.south
    }
}

/// Converts a `[west, south, east, north]` array (e.g. from `xyz_to_bbox`).
impl From<[f64; 4]> for Bbox {
    #[inline]
    fn from([west, south, east, north]: [f64; 4]) -> Self {
        Self {
            west,
            south,
            east,
            north,
        }
    }
}
