//! Polygon algebra on one fixed grid per tile.
//!
//! geo's boolean ops (via `i_overlay`) compute exactly on an integer grid,
//! but pick that grid per call from the bounding box of the call's inputs.
//! Chaining calls whose inputs differ in extent — a whole chart cell, then
//! tile-sized pieces — snaps the same edge to different grids, and the
//! difference leaves strips along it that aren't in the data.
//!
//! [`TileGrid`] fixes the grid to the tile instead: every input is first cut
//! to the tile exactly ([`TileGrid::cut`], plain float arithmetic that puts
//! crossing vertices exactly on the tile's edges), so all of them fit one
//! integer grid spanning the tile, and every operation on it agrees with
//! every other. A coverage containing the tile yields exactly the tile, and
//! what's left after it is exactly empty.
//!
//! Chart data has gaps of its own: neighbouring cells whose borders nearly
//! coincide leave strips between them down to one unit of the data's
//! resolution. Results drop every part that is nowhere wider than
//! [`NOISE_WIDTH_DEG`]: a part stays, unchanged, when shrinking it by half
//! that width leaves anything. The test is local — a noise strip attached
//! to a real gap stays with the gap — and in degrees, so the same at every
//! zoom.

use anyhow::{Context, Result};
use geo::{Coord, LineString, MultiPolygon, Polygon};

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay::ShapeType;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::core::solver::Solver;
use i_overlay::float::overlay::{FloatOverlay, OverlayOptions};
use i_overlay::i_float::adapter::FloatPointAdapter;
use i_overlay::i_float::float::rect::FloatRect;
use i_overlay::i_float::int::point::IntPoint;
use i_overlay::mesh::int::outline::offset::IntOutlineOffset;
use i_overlay::mesh::int::style::IntOutlineStyle;

use crate::bbox::Bbox;

/// Narrowest strip chart data can express: S-57 stores coordinates as
/// integers scaled by `COMF`, normally 10^7, so 1e-7° (~1 cm). Borders one
/// such unit apart are noise from cells that don't quite meet; an
/// assumption for the other formats, which state no resolution.
pub const DATA_RESOLUTION_DEG: f64 = 1e-7;

/// The noise cutoff: midway between one unit of [`DATA_RESOLUTION_DEG`]
/// (noise) and two (kept as real), so that the tile grid's rounding of
/// either can't move it across.
pub const NOISE_WIDTH_DEG: f64 = 1.5 * DATA_RESOLUTION_DEG;

/// A tile's extent, and the integer grid all of its polygon algebra uses.
pub struct TileGrid {
    rect: Bbox,
    adapter: FloatPointAdapter<[f64; 2], i32>,
    /// Half of [`NOISE_WIDTH_DEG`] in grid units, at least one.
    noise_radius: i32,
}

impl TileGrid {
    /// # Errors
    /// When `rect` is not a valid rectangle (non-finite or inverted).
    pub fn new(rect: Bbox) -> Result<Self> {
        let float_rect = FloatRect::new(rect.west, rect.east, rect.south, rect.north)
            .map_err(|e| anyhow::anyhow!("{e:?}"))
            .with_context(|| format!("tile rectangle {rect:?}"))?;
        let adapter = FloatPointAdapter::<[f64; 2], i32>::new(float_rect);
        // Smaller than the tile's extent in grid units, which fits `i32`, for
        // any tile wider than the data's resolution.
        #[allow(clippy::cast_possible_truncation)]
        let noise_radius = ((NOISE_WIDTH_DEG / 2.0) * adapter.dir_scale())
            .round()
            .max(1.0) as i32;
        Ok(Self {
            rect,
            adapter,
            noise_radius,
        })
    }

    /// The tile as a polygon.
    #[must_use]
    pub fn whole(&self) -> MultiPolygon {
        MultiPolygon::from(Polygon::from(self.rect))
    }

    /// `geom` cut to the tile: each ring clipped against the tile's four
    /// edges (Sutherland–Hodgman), crossings placed exactly on them. A ring
    /// along a concave outline may come out with zero-width spans along a
    /// tile edge; [`Self::intersection`] and [`Self::difference`] resolve
    /// them, so pass the result through one of those before using it.
    #[must_use]
    pub fn cut(&self, geom: &MultiPolygon) -> MultiPolygon {
        let polygons = geom.0.iter().filter_map(|poly| {
            let exterior = self.cut_ring(poly.exterior())?;
            let holes = poly
                .interiors()
                .iter()
                .filter_map(|ring| self.cut_ring(ring))
                .collect();
            Some(Polygon::new(exterior, holes))
        });
        MultiPolygon::new(polygons.collect())
    }

    /// The parts of `a` inside `b`, without parts nowhere wider than
    /// [`NOISE_WIDTH_DEG`]. Both must lie within the tile (see
    /// [`Self::cut`]).
    #[must_use]
    pub fn intersection(&self, a: &MultiPolygon, b: &MultiPolygon) -> MultiPolygon {
        self.overlay(a, b, OverlayRule::Intersect)
    }

    /// The parts of `a` outside `b`, without parts nowhere wider than
    /// [`NOISE_WIDTH_DEG`]. Both must lie within the tile (see
    /// [`Self::cut`]).
    #[must_use]
    pub fn difference(&self, a: &MultiPolygon, b: &MultiPolygon) -> MultiPolygon {
        self.overlay(a, b, OverlayRule::Difference)
    }

    fn overlay(&self, a: &MultiPolygon, b: &MultiPolygon, rule: OverlayRule) -> MultiPolygon {
        let mut overlay = FloatOverlay::new_custom(
            self.adapter.clone(),
            OverlayOptions::ogc(),
            Solver::default(),
            0,
        );
        // `unsafe_` only in name: every point must lie inside the adapter's
        // rect, which inputs cut to the tile do.
        for (geom, shape_type) in [(a, ShapeType::Subject), (b, ShapeType::Clip)] {
            for ring in geom
                .0
                .iter()
                .flat_map(|p| std::iter::once(p.exterior()).chain(p.interiors()))
            {
                overlay = overlay.unsafe_add_contour(&open_path(ring), shape_type);
            }
        }
        let shapes = overlay.overlay(rule, FillRule::EvenOdd);
        MultiPolygon::new(
            shapes
                .into_iter()
                .filter(|shape| !self.is_noise(shape))
                .filter_map(|shape| {
                    let mut rings = shape.into_iter().map(closed_ring);
                    let exterior = rings.next()?;
                    Some(Polygon::new(exterior, rings.collect()))
                })
                .collect(),
        )
    }

    /// Whether shrinking `shape` (an overlay result: outer ring
    /// counter-clockwise, holes clockwise) by half of [`NOISE_WIDTH_DEG`]
    /// leaves nothing, i.e. it is nowhere wider.
    fn is_noise(&self, shape: &[Vec<[f64; 2]>]) -> bool {
        // Back to the grid the overlay computed on; exact, its points are
        // grid points.
        let int_shape: Vec<Vec<IntPoint<i32>>> = shape
            .iter()
            .map(|ring| ring.iter().map(|p| self.adapter.float_to_int(p)).collect())
            .collect();
        // Construction can't fail: the `Result` reports optional range
        // validation, and points of a tile with a small offset stay in range.
        int_shape
            .outline(&IntOutlineStyle::new(-self.noise_radius))
            .is_ok_and(|shrunk| shrunk.is_empty())
    }

    /// `ring` clipped to the tile; `None` when nothing of it remains.
    fn cut_ring(&self, ring: &LineString) -> Option<LineString> {
        let r = self.rect;
        // Open path: geo rings repeat their first point at the end.
        let mut points: Vec<Coord> = ring
            .0
            .split_last()
            .map_or(&[][..], |(_, rest)| rest)
            .to_vec();
        for edge in [
            Edge::West(r.west),
            Edge::East(r.east),
            Edge::South(r.south),
            Edge::North(r.north),
        ] {
            if points.is_empty() {
                return None;
            }
            points = clip_against(&points, edge);
        }
        (points.len() >= 3).then(|| {
            let mut ring = LineString::new(points);
            ring.close();
            ring
        })
    }
}

/// One edge of the tile, with the side of it that is kept.
#[derive(Clone, Copy)]
enum Edge {
    West(f64),
    East(f64),
    South(f64),
    North(f64),
}

impl Edge {
    fn keeps(self, c: Coord) -> bool {
        match self {
            Self::West(x) => c.x >= x,
            Self::East(x) => c.x <= x,
            Self::South(y) => c.y >= y,
            Self::North(y) => c.y <= y,
        }
    }

    /// Where segment `a`–`b` crosses this edge; the edge's own coordinate is
    /// set exactly, the other one interpolated.
    fn crossing(self, a: Coord, b: Coord) -> Coord {
        match self {
            Self::West(x) | Self::East(x) => Coord {
                x,
                y: a.y + (b.y - a.y) * (x - a.x) / (b.x - a.x),
            },
            Self::South(y) | Self::North(y) => Coord {
                x: a.x + (b.x - a.x) * (y - a.y) / (b.y - a.y),
                y,
            },
        }
    }
}

/// One Sutherland–Hodgman pass: the closed polygon `points` clipped to the
/// kept side of `edge`.
fn clip_against(points: &[Coord], edge: Edge) -> Vec<Coord> {
    let mut out = Vec::with_capacity(points.len() + 2);
    let Some(&last) = points.last() else {
        return out;
    };
    let mut prev = last;
    for &cur in points {
        match (edge.keeps(prev), edge.keeps(cur)) {
            (true, true) => out.push(cur),
            (true, false) => out.push(edge.crossing(prev, cur)),
            (false, true) => {
                out.push(edge.crossing(prev, cur));
                out.push(cur);
            }
            (false, false) => {}
        }
        prev = cur;
    }
    out
}

fn open_path(ring: &LineString) -> Vec<[f64; 2]> {
    let coords = ring.0.split_last().map_or(&[][..], |(_, rest)| rest);
    coords.iter().map(|c| [c.x, c.y]).collect()
}

fn closed_ring(path: Vec<[f64; 2]>) -> LineString {
    let mut ring = LineString::new(path.into_iter().map(|[x, y]| Coord { x, y }).collect());
    ring.close();
    ring
}

#[cfg(test)]
mod tests {
    use geo::Area;

    use super::*;

    const TILE: Bbox = Bbox {
        west: 10.0,
        south: 56.0,
        east: 10.25,
        north: 56.25,
    };

    fn rect(west: f64, south: f64, east: f64, north: f64) -> MultiPolygon {
        MultiPolygon::from(Polygon::from(Bbox {
            west,
            south,
            east,
            north,
        }))
    }

    #[test]
    fn a_coverage_containing_the_tile_leaves_nothing() {
        let grid = TileGrid::new(TILE).expect("valid tile");
        let huge = rect(4.0, 50.0, 16.0, 62.0);
        let contrib = grid.intersection(&grid.cut(&huge), &grid.whole());
        assert_eq!(grid.difference(&grid.whole(), &contrib).0, vec![]);
    }

    #[test]
    fn cut_keeps_crossings_exactly_on_the_tile_edges() {
        let grid = TileGrid::new(TILE).expect("valid tile");
        // A diamond poking out of every side of the tile.
        let diamond = MultiPolygon::from(Polygon::new(
            LineString::from(vec![
                (10.125, 55.9),
                (10.4, 56.125),
                (10.125, 56.4),
                (9.8, 56.125),
                (10.125, 55.9),
            ]),
            vec![],
        ));
        let cut = grid.cut(&diamond);
        for c in cut.0[0].exterior().coords() {
            assert!(
                (TILE.west..=TILE.east).contains(&c.x) && (TILE.south..=TILE.north).contains(&c.y),
                "{c:?}"
            );
        }
        assert!(cut.0[0].exterior().coords().any(|c| c.x == TILE.west));
        assert!(cut.0[0].exterior().coords().any(|c| c.y == TILE.north));
        // And the crossings are where the diamond's edges meet the tile's:
        // the cut is the diamond's part inside the tile.
        let inside = geo::BooleanOps::intersection(&diamond, &grid.whole());
        assert!(
            (cut.unsigned_area() - inside.unsigned_area()).abs() < 1e-12,
            "{} vs {}",
            cut.unsigned_area(),
            inside.unsigned_area()
        );
    }

    #[test]
    fn cut_keeps_holes_and_drops_what_lies_outside() {
        let grid = TileGrid::new(TILE).expect("valid tile");
        let hole = LineString::from(vec![
            (10.1, 56.1),
            (10.15, 56.1),
            (10.15, 56.15),
            (10.1, 56.15),
            (10.1, 56.1),
        ]);
        let outer = LineString::from(vec![
            (9.0, 55.0),
            (11.0, 55.0),
            (11.0, 57.0),
            (9.0, 57.0),
            (9.0, 55.0),
        ]);
        let far = Polygon::from(Bbox {
            west: 20.0,
            south: 20.0,
            east: 21.0,
            north: 21.0,
        });
        let cut = grid.cut(&MultiPolygon::new(vec![
            Polygon::new(outer, vec![hole]),
            far,
        ]));
        assert_eq!(cut.0.len(), 1, "the far polygon is gone");
        let resolved = grid.intersection(&cut, &grid.whole());
        // The tile minus a 0.05° square hole. The hole's corners snap to the
        // grid (step 0.25° / 2^30 here), so allow a few steps of area.
        let expected = 0.0625 - 0.0025;
        assert!(
            (resolved.unsigned_area() - expected).abs() < 1e-9,
            "{}",
            resolved.unsigned_area()
        );
    }

    #[test]
    fn a_partial_coverage_leaves_exactly_the_rest() {
        let grid = TileGrid::new(TILE).expect("valid tile");
        let west_half = rect(4.0, 50.0, 10.125, 62.0);
        let contrib = grid.intersection(&grid.cut(&west_half), &grid.whole());
        let rest = grid.difference(&grid.whole(), &contrib);
        assert_eq!(rest.0.len(), 1);
        let east_half = grid.intersection(&grid.cut(&rect(10.125, 50.0, 16.0, 62.0)), &rest);
        assert_eq!(grid.difference(&rest, &east_half).0, vec![]);
    }

    /// Neighbouring cells whose borders, on the data's integer grid, are
    /// one unit apart (or closer) leave no strip behind; two units apart is
    /// a real gap. Whatever the zoom and wherever the border falls on the
    /// tile's grid.
    #[test]
    fn gaps_of_one_data_unit_are_noise_two_are_real() {
        for z in 4..=22 {
            let side = 360.0 / f64::from(1 << z);
            for lon in [10.5, 3.3, -71.1, 151.2] {
                let west_edge = (lon / side).floor() * side;
                let tile = Bbox {
                    west: west_edge,
                    south: 56.5,
                    east: west_edge + side,
                    north: 56.5 + side,
                };
                let grid = TileGrid::new(tile).expect("valid tile");
                let border = (lon / DATA_RESOLUTION_DEG).round() * DATA_RESOLUTION_DEG;
                for (gap, real) in [
                    (1e-9, false),
                    (DATA_RESOLUTION_DEG, false),
                    (2.0 * DATA_RESOLUTION_DEG, true),
                ] {
                    let west = grid
                        .intersection(&grid.cut(&rect(-180.0, 0.0, border, 80.0)), &grid.whole());
                    let rest = grid.difference(&grid.whole(), &west);
                    let east =
                        grid.intersection(&grid.cut(&rect(border + gap, 0.0, 180.0, 80.0)), &rest);
                    let left = grid.difference(&rest, &east);
                    assert_eq!(!left.0.is_empty(), real, "z{z} lon {lon}: gap {gap:e}°");
                }
            }
        }
    }

    /// A real gap stays a gap where it touches a noise strip: next to a
    /// border off by half the data's resolution, a 1e-6° (~10 cm) notch
    /// is left for the next chart, on tiles from about z10 to z22 alike.
    #[test]
    fn a_real_gap_attached_to_a_noise_strip_is_kept() {
        for side in [
            0.25,
            360.0 / 4096.0,
            360.0 / f64::from(1 << 18),
            360.0 / f64::from(1 << 22),
        ] {
            let tile = Bbox {
                west: 10.0,
                south: 56.0,
                east: 10.0 + side,
                north: 56.0 + side,
            };
            let grid = TileGrid::new(tile).expect("valid tile");
            let mid = 10.0 + side / 2.0;
            let (border, notch) = (mid + DATA_RESOLUTION_DEG / 2.0, 1e-6);
            // East of the border, but for a notch at the tile's north edge,
            // next to the border.
            let east_with_notch = MultiPolygon::from(Polygon::new(
                LineString::from(vec![
                    (border, 50.0),
                    (16.0, 50.0),
                    (16.0, tile.north),
                    (border + notch, tile.north),
                    (border + notch, tile.north - notch),
                    (border, tile.north - notch),
                    (border, 50.0),
                ]),
                vec![],
            ));
            let west = grid.intersection(&grid.cut(&rect(4.0, 50.0, mid, 62.0)), &grid.whole());
            let rest = grid.difference(&grid.whole(), &west);
            let east = grid.intersection(&grid.cut(&east_with_notch), &rest);
            let left = grid.difference(&rest, &east);
            let area = left.unsigned_area();
            assert!(
                area >= notch * notch,
                "tile side {side}°: {area:e} deg² left, the notch alone is {:e}",
                notch * notch
            );
        }
    }
}
