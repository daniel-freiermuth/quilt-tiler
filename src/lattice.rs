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
        self.contains(other)
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
