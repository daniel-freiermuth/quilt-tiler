//! [`BoundedLattice`] — a set with meet (∧), join (∨), a least element (⊥),
//! and a greatest element (⊤).
//!
//! Current implementation: [`crate::bbox::Bbox`].
//! Intended next implementation: exact polygon boolean regions.

/// A bounded lattice.
///
/// The partial order is: `a ≥ b` iff `a.subsumes(b)` ("a covers b entirely").
/// Meet is intersection, join is union/hull.
pub trait BoundedLattice: Sized {
    /// The least element ⊥ — empty / identity for join.
    fn bottom() -> Self;

    /// Least upper bound ∨ (union / hull).
    fn join(&self, other: &Self) -> Self;

    /// Greatest lower bound ∧ (intersection / clip).
    fn meet(&self, other: &Self) -> Self;

    /// `true` when `self ≥ other` in the lattice order (self covers other).
    fn subsumes(&self, other: &Self) -> bool;

    fn overlaps(&self, other: &Self) -> bool;
}
