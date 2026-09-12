//! [`BoundedLattice`] — a set with meet (∧), join (∨), a least element (⊥),
//! and a greatest element (⊤).
//!
//! Current implementation: [`crate::bbox::Bbox`].
//! Intended next implementation: exact polygon boolean regions.

use geo::{Area, BooleanOps, Contains, Intersects, MultiPolygon};

/// A bounded lattice.
///
/// The partial order is: `a ≥ b` iff `a.subsumes(b)` ("a covers b entirely").
/// Meet is intersection, join is union/hull.
pub trait BoundedLattice: Sized {
    /// The least element ⊥ — empty / identity for join.
    fn bottom() -> Self;

    /// Least upper bound ∨ (union / hull).
    #[must_use]
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound ∧ (intersection / clip).
    #[must_use]
    fn meet(&self, other: &Self) -> Self;

    /// `true` when `self ≥ other` in the lattice order (self covers other).
    fn subsumes(&self, other: &Self) -> bool;

    fn overlaps(&self, other: &Self) -> bool;

    fn area(&self) -> f64;

    #[must_use]
    fn minus(&self, other: &Self) -> Self;
}

impl BoundedLattice for MultiPolygon {
    fn bottom() -> Self {
        Self::empty()
    }

    #[profiling::function]
    fn join(&self, other: &Self) -> Self {
        self.union(other)
    }

    #[profiling::function]
    fn meet(&self, other: &Self) -> Self {
        self.intersection(other)
    }

    #[profiling::function]
    fn subsumes(&self, other: &Self) -> bool {
        other.0.is_empty() || self.contains(other)
    }

    #[profiling::function]
    fn overlaps(&self, other: &Self) -> bool {
        self.intersects(other)
    }

    fn area(&self) -> f64 {
        self.unsigned_area()
    }

    fn minus(&self, other: &Self) -> Self {
        self.difference(other)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bbox::Bbox;
    use geo::Polygon;

    /// Helper: axis-aligned rectangle as a single-polygon `MultiPolygon`.
    fn rect(west: f64, south: f64, east: f64, north: f64) -> MultiPolygon {
        MultiPolygon::new(vec![Polygon::from(Bbox {
            west,
            south,
            east,
            north,
        })])
    }

    // ── bottom ──────────────────────────────────────────────────────

    #[test]
    fn bottom_returns_empty_multipolygon() {
        let b = MultiPolygon::bottom();
        assert!(b.0.is_empty());
        assert!(b.area().abs() < f64::EPSILON);
    }

    // ── join ────────────────────────────────────────────────────────

    #[test]
    fn join_disjoint_is_union() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        let b = rect(5.0, 5.0, 6.0, 6.0);
        let joined = a.join(&b);
        // Two disjoint unit squares → area = 2.
        let expected_area = a.area() + b.area();
        assert!((joined.area() - expected_area).abs() < 1e-10);
        // The union must subsume both inputs.
        assert!(joined.subsumes(&a));
        assert!(joined.subsumes(&b));
    }

    #[test]
    fn join_with_bottom_returns_self() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let joined = a.join(&MultiPolygon::bottom());
        assert!((joined.area() - a.area()).abs() < 1e-10);
        assert!(joined.subsumes(&a));
    }

    #[test]
    fn join_overlapping_is_union_area() {
        let a = rect(0.0, 0.0, 2.0, 2.0); // area 4
        let b = rect(1.0, 1.0, 3.0, 3.0); // area 4, overlap area 1
        let joined = a.join(&b);
        // Union area = 4 + 4 − 1 = 7
        assert!((joined.area() - 7.0).abs() < 1e-10);
    }

    // ── meet ────────────────────────────────────────────────────────

    #[test]
    fn meet_overlapping_is_intersection() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let b = rect(1.0, 1.0, 3.0, 3.0);
        let met = a.meet(&b);
        // Intersection is the 1×1 square [1,1]→[2,2].
        assert!((met.area() - 1.0).abs() < 1e-10);
        // Both inputs must subsume the intersection.
        assert!(a.subsumes(&met));
        assert!(b.subsumes(&met));
    }

    #[test]
    fn meet_disjoint_is_empty() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        let b = rect(5.0, 5.0, 6.0, 6.0);
        let met = a.meet(&b);
        assert!(met.0.is_empty() || met.area() < 1e-10);
    }

    #[test]
    fn meet_with_bottom_is_empty() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let met = a.meet(&MultiPolygon::bottom());
        assert!(met.0.is_empty() || met.area() < 1e-10);
    }

    // ── minus ───────────────────────────────────────────────────────

    #[test]
    fn minus_removes_subtracted_region() {
        let a = rect(0.0, 0.0, 4.0, 4.0); // area 16
        let b = rect(0.0, 0.0, 2.0, 2.0); // area 4, inside a
        let diff = a.minus(&b);
        // 16 − 4 = 12
        assert!((diff.area() - 12.0).abs() < 1e-10);
        // The difference must not overlap the subtracted region.
        assert!(!diff.overlaps(&b) || diff.meet(&b).area() < 1e-10);
    }

    #[test]
    fn minus_disjoint_returns_self() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        let b = rect(5.0, 5.0, 6.0, 6.0);
        let diff = a.minus(&b);
        assert!((diff.area() - a.area()).abs() < 1e-10);
    }

    #[test]
    fn minus_bottom_returns_self() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let diff = a.minus(&MultiPolygon::bottom());
        assert!((diff.area() - a.area()).abs() < 1e-10);
    }

    // ── subsumes ────────────────────────────────────────────────────

    #[test]
    fn subsumes_inner_rect() {
        let outer = rect(0.0, 0.0, 10.0, 10.0);
        let inner = rect(2.0, 2.0, 8.0, 8.0);
        assert!(outer.subsumes(&inner));
        assert!(!inner.subsumes(&outer));
    }

    #[test]
    fn subsumes_self() {
        let a = rect(0.0, 0.0, 3.0, 3.0);
        assert!(a.subsumes(&a));
    }

    /// ⊥ is subsumed by everything — lattice-correct: we short-circuit
    /// on empty geometry since `geo::Contains` returns `false` for it.
    #[test]
    fn subsumes_bottom() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        assert!(a.subsumes(&MultiPolygon::bottom()));
    }

    #[test]
    fn subsumes_partial_overlap_is_false() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let b = rect(1.0, 1.0, 3.0, 3.0);
        assert!(!a.subsumes(&b));
        assert!(!b.subsumes(&a));
    }

    // ── overlaps ────────────────────────────────────────────────────

    #[test]
    fn overlaps_with_empty_is_false() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        assert!(!a.overlaps(&MultiPolygon::bottom()));
        assert!(!MultiPolygon::bottom().overlaps(&a));
    }

    #[test]
    fn overlaps_partial_is_true() {
        let a = rect(0.0, 0.0, 2.0, 2.0);
        let b = rect(1.0, 1.0, 3.0, 3.0);
        assert!(a.overlaps(&b));
        assert!(b.overlaps(&a));
    }

    #[test]
    fn overlaps_disjoint_is_false() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        let b = rect(5.0, 5.0, 6.0, 6.0);
        assert!(!a.overlaps(&b));
    }

    // ── area ────────────────────────────────────────────────────────

    #[test]
    fn area_of_unit_square() {
        let a = rect(0.0, 0.0, 1.0, 1.0);
        assert!((a.area() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn area_of_bottom_is_zero() {
        assert!(MultiPolygon::bottom().area().abs() < f64::EPSILON);
    }
}
