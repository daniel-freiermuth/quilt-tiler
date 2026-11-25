//! A cell's coverage from its S-57 `M_COVR` areas: covered (`CATCOV = 1`)
//! minus not covered (`CATCOV = 2`), shared by every format producing
//! [`crate::S57Cell`]s.

use geo::{Area, BooleanOps, Intersects, MultiPolygon, Polygon};

/// Resolve a cell's coverage from its covered and not-covered areas.
///
/// # Covered/not-covered taxonomy (measured across 582 SENC cells / 614
/// intersecting pairs via `oesu`'s `examples/probe_covr_taxonomy.rs` —
/// re-run it against fresh chart exports if this ever needs re-validating)
///
/// The relationship between a covered area and an intersecting not-covered
/// area falls into exactly one of two buckets, measured by
/// `intersection_area(covered, not_covered) / covered_area`. There is no
/// middle ground: 0 of 614 pairs landed strictly between 0% and 100%.
///
/// 1. **Adjacent/complement (444 pairs, ~72%).** Overlap is exactly 0%.
///    The not-covered area shares ~85% of its vertices with the covered one
///    on average (it retraces most of its boundary, e.g. a coastline, then
///    closes around the cell's nominal grid extent) — the fingerprint of a
///    shape that was *cut from* the covered area rather than independently
///    surveyed. Subtracting it is a true geometric no-op: confirmed
///    empirically, the difference changes 0/582 cells' total coverage area
///    relative to ignoring not-covered areas outright.
/// 2. **Full erasure (170 pairs, ~28%).** Overlap is exactly 100%: the
///    not-covered area completely contains the covered one. In 144/170 cases
///    it is a perfect axis-aligned rectangle matching the cell's nominal grid
///    quadrant (e.g. exactly 0.25°×0.25°), emitted regardless of how many or
///    what shape covered pieces it actually contains — most visibly when the
///    covered areas are a handful of small disjoint islands inside one
///    nominal cell. It shares *zero* vertices with the covered area in every
///    one of these 170 pairs (vs. ~85% in bucket 1) — confirms this is
///    unrelated geometry, not a coastline trace. Subtracting it would erase
///    real, explicitly-charted data; it is detected and skipped instead.
///
/// The textbook case — a small *genuine* gap fully nested inside an
/// otherwise-charted area, producing a partial (neither 0% nor 100%)
/// overlap — does not occur anywhere in that corpus. The difference plus
/// erasure guard handles it correctly *if* it ever shows up, at zero
/// behavioural cost today, since it's proven equivalent to ignoring
/// not-covered areas entirely for every cell measured. That's why the logic
/// stays even though a simpler "just ignore them" implementation currently
/// produces bit-identical output.
///
/// Not-covered areas are not guaranteed to nest inside a covered one, so a
/// structural `Polygon::new(ext, vec![hole])` (which requires strict
/// containment) would silently break geo's `BooleanOps` when violated —
/// intersection degenerates to returning the other operand unchanged. A
/// boolean [`BooleanOps::difference`] is correct regardless of nesting.
///
/// Overlapping covered areas, not-covered areas that match no covered one,
/// and full-erasure not-covered areas are data-quality issues, not failures:
/// they are logged (with `source`) and handled best-effort.
///
/// The result has no overlapping polygons: geo's boolean operations use the
/// even-odd fill rule, so a region shared by two members of one
/// `MultiPolygon` would count as *outside* it when tiles are clipped to it.
#[must_use]
pub fn resolve_coverage(source: &str, covered: &[Polygon], uncovered: &[Polygon]) -> MultiPolygon {
    let mut overlapping = false;
    for (i, a) in covered.iter().enumerate() {
        for (j, b) in covered.iter().enumerate().skip(i + 1) {
            if a.intersects(b) {
                tracing::warn!(source, i, j, "overlapping covered areas");
                overlapping = true;
            }
        }
    }

    let mut claimed = vec![false; uncovered.len()];
    let mut polys: Vec<Polygon> = Vec::new();
    for area in covered {
        let mut remaining = MultiPolygon::new(vec![area.clone()]);
        for (index, gap) in uncovered.iter().enumerate() {
            if !gap.intersects(area) {
                continue;
            }
            if let Some(c) = claimed.get_mut(index) {
                *c = true;
            }
            let after = remaining.difference(gap);
            // An already-empty remainder stays empty either way.
            if after.unsigned_area() == 0.0 {
                // A not-covered area that fully overlaps a covered area's
                // entire remaining area isn't a genuine data gap — real gaps
                // carve a partial notch out of charted data. Full erasure is
                // the other common shape: a clean rectangle equal to the
                // cell's nominal grid extent. Trusting it here would silently
                // render real charted islands as blank. Keep the area instead.
                tracing::info!(
                    source,
                    index,
                    "not-covered area fully overlaps a covered area's remaining area \
                     (likely a nominal-extent placeholder, not a real data gap); \
                     ignoring it for this area and keeping the covered area"
                );
                continue;
            }
            remaining = after;
        }
        polys.extend(remaining);
    }
    let unclaimed = claimed.iter().filter(|c| !**c).count();
    if unclaimed > 0 {
        tracing::warn!(
            source,
            unclaimed,
            "not-covered areas outside any covered area, ignored"
        );
    }

    if !overlapping {
        return MultiPolygon::new(polys);
    }
    // Union one by one: overlapping covered areas must merge, not cancel.
    polys.into_iter().fold(MultiPolygon::new(vec![]), |acc, p| {
        acc.union(&MultiPolygon::new(vec![p]))
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use geo::{LineString, coord};

    fn rect(x0: f64, y0: f64, x1: f64, y1: f64) -> Polygon {
        Polygon::new(
            LineString::new(vec![
                coord! { x: x0, y: y0 },
                coord! { x: x1, y: y0 },
                coord! { x: x1, y: y1 },
                coord! { x: x0, y: y1 },
                coord! { x: x0, y: y0 },
            ]),
            vec![],
        )
    }

    fn area(m: &MultiPolygon) -> f64 {
        m.unsigned_area()
    }

    /// Run `f` under a thread-local subscriber; return its result and the
    /// formatted (ANSI-free) log lines it emitted.
    fn capture_logs<T>(f: impl FnOnce() -> T) -> (T, String) {
        #[derive(Clone, Default)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let capture = Capture::default();
        let writer = capture.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .finish();
        let out = tracing::subscriber::with_default(subscriber, f);
        let bytes = capture.0.lock().unwrap().clone();
        (out, String::from_utf8(bytes).unwrap())
    }

    const UNCLAIMED: &str = "not-covered areas outside any covered area";

    #[test]
    fn no_covered_area_gives_empty_coverage() {
        assert_eq!(resolve_coverage("t", &[], &[]).0, vec![]);
        let only_gap = resolve_coverage("t", &[], &[rect(0.0, 0.0, 1.0, 1.0)]);
        assert_eq!(
            only_gap.0,
            vec![],
            "a not-covered area alone covers nothing"
        );
    }

    #[test]
    fn covered_area_without_gaps_passes_through() {
        let result = resolve_coverage("t", &[rect(0.0, 0.0, 10.0, 10.0)], &[]);
        assert_eq!(result.0.len(), 1);
        assert!((area(&result) - 100.0).abs() < 1e-6, "{}", area(&result));
    }

    #[test]
    fn full_erasure_guard_keeps_the_covered_area() {
        // A not-covered area containing the covered one is a nominal-extent
        // placeholder, not a gap.
        let result = resolve_coverage(
            "t",
            &[rect(1.0, 1.0, 5.0, 5.0)],
            &[rect(0.0, 0.0, 10.0, 10.0)],
        );
        assert_eq!(result.0.len(), 1);
        assert!((area(&result) - 16.0).abs() < 1e-6, "{}", area(&result));
    }

    #[test]
    fn partial_gap_is_subtracted() {
        let (result, logs) = capture_logs(|| {
            resolve_coverage(
                "t",
                &[rect(0.0, 0.0, 10.0, 10.0)],
                &[rect(2.0, 2.0, 4.0, 4.0)],
            )
        });
        assert!((area(&result) - 96.0).abs() < 1e-6, "{}", area(&result));
        assert!(!logs.contains(UNCLAIMED), "the gap is inside: {logs}");
    }

    /// A not-covered area outside every covered one changes nothing but is
    /// reported, as a data-quality issue.
    #[test]
    fn gap_outside_every_covered_area_is_ignored_and_reported() {
        let (result, logs) = capture_logs(|| {
            resolve_coverage(
                "t",
                &[rect(0.0, 0.0, 5.0, 5.0)],
                &[rect(100.0, 100.0, 110.0, 110.0)],
            )
        });
        assert_eq!(result.0.len(), 1);
        assert!((area(&result) - 25.0).abs() < 1e-6, "{}", area(&result));
        assert!(logs.contains(UNCLAIMED), "{logs}");
        assert!(logs.contains("unclaimed=1"), "{logs}");
    }

    /// Overlapping not-covered areas each remove their own part; the region
    /// they share must not cancel out (even-odd fill) and stay covered.
    #[test]
    fn overlapping_gaps_are_all_subtracted() {
        let result = resolve_coverage(
            "t",
            &[rect(0.0, 0.0, 1.0, 1.0)],
            &[rect(0.0, 0.0, 0.5, 0.5), rect(0.25, 0.0, 0.75, 0.5)],
        );
        // 1 minus the gaps' union (0.75 × 0.5).
        assert!((area(&result) - 0.625).abs() < 1e-9, "{}", area(&result));
    }

    /// Tiling intersects the coverage with each tile: overlapping covered
    /// areas must still cover their shared region there.
    #[test]
    fn overlapping_covered_areas_cover_their_shared_region_in_a_tile() {
        let coverage = resolve_coverage(
            "t",
            &[rect(0.0, 0.0, 2.0, 1.0), rect(1.0, 0.0, 3.0, 1.0)],
            &[],
        );
        let tile = MultiPolygon::new(vec![rect(0.0, 0.0, 3.0, 1.0)]);
        let in_tile = coverage.intersection(&tile);
        assert!((area(&in_tile) - 3.0).abs() < 1e-9, "{}", area(&in_tile));
    }
}
