//! Axis-aligned bounding box.

use geo::{BoundingRect, MultiPolygon, Polygon};
use martin_tile_utils::wgs84_to_webmercator;

/// An axis-aligned bounding box in WGS84 or projected coordinates.
///
/// [`Self::bottom`] is the empty box; [`Self::join`] is the bounding hull.
#[derive(Copy, Clone, Debug)]
pub struct Bbox {
    pub west: f64,
    pub south: f64,
    pub east: f64,
    pub north: f64,
}

impl Bbox {
    /// The empty box: identity for [`Self::join`], overlaps nothing.
    #[inline]
    #[must_use]
    pub const fn bottom() -> Self {
        Self {
            west: f64::INFINITY,
            south: f64::INFINITY,
            east: f64::NEG_INFINITY,
            north: f64::NEG_INFINITY,
        }
    }

    #[inline]
    #[must_use]
    pub fn is_bottom(&self) -> bool {
        self.west > self.east || self.south > self.north
    }

    /// Bounding hull of `self` and `other`.
    #[inline]
    #[must_use]
    pub const fn join(&self, other: &Self) -> Self {
        Self {
            west: self.west.min(other.west),
            south: self.south.min(other.south),
            east: self.east.max(other.east),
            north: self.north.max(other.north),
        }
    }

    /// `true` when the boxes share at least a point; never for an empty box.
    #[inline]
    #[must_use]
    #[allow(clippy::suspicious_operation_groupings)] // cross-axis comparisons are intentional
    pub fn overlaps(&self, other: &Self) -> bool {
        !self.is_bottom()
            && !other.is_bottom()
            && self.west <= other.east
            && self.east >= other.west
            && self.south <= other.north
            && self.north >= other.south
    }

    /// Project a WGS84 box (degrees) to Web Mercator metres.
    #[inline]
    #[must_use]
    pub(crate) fn to_web_mercator(self) -> Self {
        let (west, south) = wgs84_to_webmercator(self.west, self.south);
        let (east, north) = wgs84_to_webmercator(self.east, self.north);
        Self {
            west,
            south,
            east,
            north,
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
// Exact equality is intended: join only selects literal coordinates
// (min/max), no arithmetic that could introduce rounding.
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;

    /// Helper: a "normal" bbox around a region.
    fn bbox_a() -> Bbox {
        Bbox {
            west: 1.0,
            south: 2.0,
            east: 5.0,
            north: 6.0,
        }
    }

    /// A second bbox that partially overlaps `bbox_a`.
    fn bbox_b() -> Bbox {
        Bbox {
            west: 3.0,
            south: 4.0,
            east: 7.0,
            north: 8.0,
        }
    }

    /// A bbox fully inside `bbox_a`.
    fn bbox_inner() -> Bbox {
        Bbox {
            west: 2.0,
            south: 3.0,
            east: 4.0,
            north: 5.0,
        }
    }

    /// A bbox completely disjoint from `bbox_a`.
    fn bbox_disjoint() -> Bbox {
        Bbox {
            west: 10.0,
            south: 10.0,
            east: 12.0,
            north: 12.0,
        }
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

    #[test]
    fn point_bbox_is_not_bottom() {
        // Zero width and zero height: a single sounding or light is still a
        // real extent, only an inverted one is ⊥.
        let point = Bbox {
            west: 11.5,
            south: 57.5,
            east: 11.5,
            north: 57.5,
        };
        assert!(!point.is_bottom());
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
        let inverted = Bbox {
            west: 5.0,
            south: 2.0,
            east: 1.0,
            north: 6.0,
        };
        assert!(inverted.is_bottom());
        assert!(!inverted.overlaps(&bbox_a()));
        assert!(!bbox_a().overlaps(&inverted));
    }
}
