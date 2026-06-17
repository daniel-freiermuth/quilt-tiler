//! [`RectUnion`] — a [`BoundedLattice`] instance that tracks coverage as a
//! union of axis-aligned rectangles.
//!
//! Compared to a plain [`Bbox`], this correctly handles the case where two
//! disjoint rectangles (e.g. NE and SW corners) are joined: the hull bbox
//! would report full coverage, but `RectUnion` does not.
//!
//! The rects may overlap; correctness does not depend on disjointness.

use crate::bbox::Bbox;
use crate::lattice::BoundedLattice;

/// A union of axis-aligned bounding boxes.
///
/// Lattice order: `a ≥ b` iff every rect in `b` is fully covered by the
/// geometric union of rects in `a`.
#[derive(Clone, Debug, Default)]
pub struct RectUnion {
    rects: Vec<Bbox>,
}

impl From<Bbox> for RectUnion {
    #[inline]
    fn from(b: Bbox) -> Self {
        if b.is_bottom() {
            Self::bottom()
        } else {
            Self { rects: vec![b] }
        }
    }
}

impl BoundedLattice for RectUnion {
    #[inline]
    fn bottom() -> Self {
        Self { rects: Vec::new() }
    }

    /// Bounding hull: union of both rect lists.
    fn join(&self, other: &Self) -> Self {
        let mut rects = Vec::with_capacity(self.rects.len() + other.rects.len());
        rects.extend_from_slice(&self.rects);
        rects.extend_from_slice(&other.rects);
        Self { rects }
    }

    /// Pairwise bbox intersections of the two sets.
    fn meet(&self, other: &Self) -> Self {
        let mut rects = Vec::new();
        for a in &self.rects {
            for b in &other.rects {
                let m = a.meet(b);
                if !m.is_bottom() {
                    rects.push(m);
                }
            }
        }
        Self { rects }
    }

    #[inline]
    fn is_bottom(&self) -> bool {
        self.rects.is_empty()
    }

    /// `true` when every rect in `other` is fully covered by the union of
    /// rects in `self`.
    fn subsumes(&self, other: &Self) -> bool {
        other.rects.iter().all(|r| self.covers_rect(r))
    }

    /// Avoids allocating the full `meet`; short-circuits on first overlap.
    fn overlaps(&self, other: &Self) -> bool {
        self.rects.iter().any(|a| other.rects.iter().any(|b| a.overlaps(b)))
    }
}

impl RectUnion {
    /// `true` when the geometric union of `self.rects` fully covers `target`.
    ///
    /// Uses a sweep-line over x-coordinates: for each vertical strip between
    /// consecutive x-boundaries, checks that the contributing rects' y-ranges
    /// span `[target.south, target.north]` completely.
    fn covers_rect(&self, target: &Bbox) -> bool {
        if target.is_bottom() {
            return true;
        }

        // Clip all rects to target; keep non-empty ones.
        let clipped: Vec<Bbox> = self.rects.iter()
            .map(|r| r.meet(target))
            .filter(|r| !r.is_bottom())
            .collect();

        if clipped.is_empty() {
            return false;
        }

        // Collect x-breakpoints from target edges and interior rect edges.
        let mut xs: Vec<f64> = Vec::with_capacity(2 + clipped.len() * 2);
        xs.push(target.west);
        xs.push(target.east);
        for r in &clipped {
            xs.push(r.west);
            xs.push(r.east);
        }
        xs.sort_unstable_by(f64::total_cmp);
        xs.dedup();

        // For each x-strip [x0, x1], check that the rects spanning the strip
        // (i.e. west ≤ x0 && east ≥ x1) cover [target.south, target.north].
        for w in xs.windows(2) {
            let (x0, x1) = (w[0], w[1]);
            if x0 >= x1 {
                continue;
            }
            let y_ranges: Vec<(f64, f64)> = clipped.iter()
                .filter(|r| r.west <= x0 && r.east >= x1)
                .map(|r| (r.south, r.north))
                .collect();
            if !covers_interval(&y_ranges, target.south, target.north) {
                return false;
            }
        }

        true
    }
}

/// `true` when the union of `ranges` covers the closed interval `[lo, hi]`.
///
/// `ranges` need not be sorted or disjoint.
fn covers_interval(ranges: &[(f64, f64)], lo: f64, hi: f64) -> bool {
    if lo >= hi {
        return true;
    }
    let mut ranges = ranges.to_vec();
    ranges.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
    let mut covered = lo;
    for (s, n) in ranges {
        if s > covered {
            return false; // gap before this interval
        }
        if n > covered {
            covered = n;
        }
        if covered >= hi {
            return true;
        }
    }
    covered >= hi
}

#[cfg(test)]
mod tests {
    use super::*;

    fn b(w: f64, s: f64, e: f64, n: f64) -> RectUnion {
        RectUnion::from(Bbox { west: w, south: s, east: e, north: n })
    }

    #[test]
    fn bottom_is_empty() {
        assert!(RectUnion::bottom().is_bottom());
        assert!(!b(0.0, 0.0, 1.0, 1.0).is_bottom());
    }

    #[test]
    fn single_rect_subsumes_itself() {
        let r = b(0.0, 0.0, 10.0, 10.0);
        assert!(r.subsumes(&r));
    }

    #[test]
    fn ne_sw_do_not_cover_tile() {
        // The original Bbox bug: hull of NE+SW fills the tile, but the area
        // does not.
        let tile = b(0.0, 0.0, 10.0, 10.0);
        let ne   = b(5.0, 5.0, 10.0, 10.0);
        let sw   = b(0.0, 0.0,  5.0,  5.0);
        let covered = ne.join(&sw);
        assert!(!covered.subsumes(&tile));
    }

    #[test]
    fn north_south_halves_cover_tile() {
        let tile  = b(0.0, 0.0, 10.0, 10.0);
        let north = b(0.0, 5.0, 10.0, 10.0);
        let south = b(0.0, 0.0, 10.0,  5.0);
        assert!(north.join(&south).subsumes(&tile));
    }

    #[test]
    fn four_quadrants_cover_tile() {
        let tile = b(0.0, 0.0, 10.0, 10.0);
        let covered = b(0.0, 0.0,  5.0,  5.0)
            .join(&b(5.0, 0.0, 10.0,  5.0))
            .join(&b(0.0, 5.0,  5.0, 10.0))
            .join(&b(5.0, 5.0, 10.0, 10.0));
        assert!(covered.subsumes(&tile));
    }

    #[test]
    fn three_quadrants_do_not_cover_tile() {
        let tile = b(0.0, 0.0, 10.0, 10.0);
        let covered = b(0.0, 0.0,  5.0,  5.0)
            .join(&b(5.0, 0.0, 10.0,  5.0))
            .join(&b(0.0, 5.0,  5.0, 10.0));
        // NW+SW+SE; NE missing.
        assert!(!covered.subsumes(&tile));
    }

    #[test]
    fn partial_strip_does_not_cover() {
        let tile    = b(0.0, 0.0, 10.0, 10.0);
        let partial = b(0.0, 0.0,  5.0, 10.0); // west half only
        assert!(!partial.subsumes(&tile));
    }

    #[test]
    fn overlapping_rects_cover_tile() {
        // Two rects that overlap in the middle and together cover the tile.
        let tile = b(0.0, 0.0, 10.0, 10.0);
        let left  = b(0.0, 0.0, 7.0, 10.0);
        let right = b(3.0, 0.0, 10.0, 10.0);
        assert!(left.join(&right).subsumes(&tile));
    }
}
