//! Axis-aligned bounding box — the [`BoundedLattice`] instance for bbox algebra.

use geo::{BoundingRect, MultiPolygon, Polygon};

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
    #[must_use]
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
    #[must_use]
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

    /// Plain rectangle area in degrees² (or projected units² for [`Self`]
    /// values in metres).  `0.0` for [`Self::is_bottom`].
    fn area(&self) -> f64 {
        if self.is_bottom() {
            0.0
        } else {
            (self.east - self.west) * (self.north - self.south)
        }
    }

    /// Conservative rectangle difference: a [`Self`] cannot represent the
    /// exact (possibly L-shaped) remainder of a partial overlap, so this
    /// returns ⊥ only when `other` fully covers `self`, and `self`
    /// unchanged otherwise — erring toward "still uncovered" rather than
    /// risking a false "fully covered".
    fn minus(&self, other: &Self) -> Self {
        if other.subsumes(self) {
            Self::bottom()
        } else {
            *self
        }
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

impl From<&MultiPolygon> for Bbox {
    fn from(value: &MultiPolygon) -> Self {
        value.bounding_rect().map_or_else(Self::bottom, |b_rect| {
            let sw_coord = b_rect.min();
            let ne_coord = b_rect.max();
            Self {
                north: ne_coord.y,
                south: sw_coord.y,
                west: sw_coord.x,
                east: ne_coord.x,
            }
        })
    }
}

impl From<MultiPolygon> for Bbox {
    fn from(value: MultiPolygon) -> Self {
        Self::from(&value)
    }
}

impl From<Bbox> for Polygon {
    fn from(value: Bbox) -> Self {
        Self::new(
            vec![
                [value.east, value.north],
                [value.west, value.north],
                [value.west, value.south],
                [value.east, value.south],
                [value.east, value.north],
            ]
            .into(),
            vec![],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a "normal" bbox around a region.
    fn bbox_a() -> Bbox {
        Bbox { west: 1.0, south: 2.0, east: 5.0, north: 6.0 }
    }

    /// A second bbox that partially overlaps `bbox_a`.
    fn bbox_b() -> Bbox {
        Bbox { west: 3.0, south: 4.0, east: 7.0, north: 8.0 }
    }

    /// A bbox fully inside `bbox_a`.
    fn bbox_inner() -> Bbox {
        Bbox { west: 2.0, south: 3.0, east: 4.0, north: 5.0 }
    }

    /// A bbox completely disjoint from `bbox_a`.
    fn bbox_disjoint() -> Bbox {
        Bbox { west: 10.0, south: 10.0, east: 12.0, north: 12.0 }
    }

    // ── bottom ──────────────────────────────────────────────────────

    #[test]
    fn bottom_is_bottom() {
        assert!(Bbox::bottom().is_bottom());
    }

    #[test]
    fn normal_bbox_is_not_bottom() {
        assert!(!bbox_a().is_bottom());
    }

    // ── join ────────────────────────────────────────────────────────

    #[test]
    fn join_with_bottom_returns_other() {
        let a = bbox_a();
        let joined = a.join(&Bbox::bottom());
        // ⊥ is (+∞,+∞,−∞,−∞); min/max with a normal bbox yields the normal bbox.
        assert_eq!(joined.west, a.west);
        assert_eq!(joined.south, a.south);
        assert_eq!(joined.east, a.east);
        assert_eq!(joined.north, a.north);
    }

    #[test]
    fn join_bottom_with_other_returns_other() {
        let a = bbox_a();
        let joined = Bbox::bottom().join(&a);
        assert_eq!(joined.west, a.west);
        assert_eq!(joined.south, a.south);
        assert_eq!(joined.east, a.east);
        assert_eq!(joined.north, a.north);
    }

    #[test]
    fn join_two_normal_is_bounding_hull() {
        let j = bbox_a().join(&bbox_b());
        assert_eq!(j.west, 1.0);
        assert_eq!(j.south, 2.0);
        assert_eq!(j.east, 7.0);
        assert_eq!(j.north, 8.0);
    }

    // ── meet ────────────────────────────────────────────────────────

    #[test]
    fn meet_non_overlapping_returns_bottom() {
        let m = bbox_a().meet(&bbox_disjoint());
        assert!(m.is_bottom(), "meet of disjoint boxes must be bottom");
    }

    #[test]
    fn meet_overlapping_returns_intersection() {
        let m = bbox_a().meet(&bbox_b());
        assert_eq!(m.west, 3.0);
        assert_eq!(m.south, 4.0);
        assert_eq!(m.east, 5.0);
        assert_eq!(m.north, 6.0);
    }

    // ── overlaps ────────────────────────────────────────────────────

    #[test]
    fn overlaps_returns_false_for_bottom() {
        assert!(!Bbox::bottom().overlaps(&bbox_a()));
        assert!(!bbox_a().overlaps(&Bbox::bottom()));
    }

    #[test]
    fn overlaps_returns_false_for_disjoint() {
        assert!(!bbox_a().overlaps(&bbox_disjoint()));
    }

    #[test]
    fn overlaps_returns_true_for_partial() {
        assert!(bbox_a().overlaps(&bbox_b()));
    }

    #[test]
    fn overlaps_returns_true_for_contained() {
        assert!(bbox_a().overlaps(&bbox_inner()));
    }

    #[test]
    fn overlaps_with_degenerate_inverted_bbox() {
        // An "inverted" bbox where west > east acts like bottom.
        let inverted = Bbox { west: 5.0, south: 2.0, east: 1.0, north: 6.0 };
        assert!(inverted.is_bottom());
        assert!(!inverted.overlaps(&bbox_a()));
        assert!(!bbox_a().overlaps(&inverted));
    }

    // ── area ────────────────────────────────────────────────────────

    #[test]
    fn area_of_bottom_is_zero() {
        assert_eq!(Bbox::bottom().area(), 0.0);
    }

    #[test]
    fn area_of_normal_bbox() {
        // bbox_a: 4 wide × 4 tall = 16
        assert!((bbox_a().area() - 16.0).abs() < f64::EPSILON);
    }

    // ── subsumes ────────────────────────────────────────────────────

    #[test]
    fn subsumes_inner() {
        assert!(bbox_a().subsumes(&bbox_inner()));
    }

    #[test]
    fn subsumes_self() {
        assert!(bbox_a().subsumes(&bbox_a()));
    }

    #[test]
    fn does_not_subsume_partial_overlap() {
        assert!(!bbox_a().subsumes(&bbox_b()));
    }

    // ── minus ───────────────────────────────────────────────────────

    #[test]
    fn minus_full_subsumption_returns_bottom() {
        // bbox_a fully contains bbox_inner, so bbox_inner − bbox_a = ⊥
        let result = bbox_inner().minus(&bbox_a());
        assert!(result.is_bottom(), "fully subsumed minus must return bottom");
    }

    #[test]
    fn minus_partial_overlap_returns_self_conservative() {
        // Conservative contract: partial overlap → returns self unchanged.
        let a = bbox_a();
        let result = a.minus(&bbox_b());
        assert_eq!(result.west, a.west);
        assert_eq!(result.south, a.south);
        assert_eq!(result.east, a.east);
        assert_eq!(result.north, a.north);
    }

    #[test]
    fn minus_disjoint_returns_self() {
        let a = bbox_a();
        let result = a.minus(&bbox_disjoint());
        assert_eq!(result.west, a.west);
        assert_eq!(result.east, a.east);
    }

    #[test]
    fn minus_self_returns_bottom() {
        let result = bbox_a().minus(&bbox_a());
        assert!(result.is_bottom(), "a − a must be bottom");
    }
}
