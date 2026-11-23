//! A chart cell's edition/publish date, kept only as a comparable ranking
//! key — never used for calendar arithmetic.
//!
//! Consumed by [`crate::S57Cell`]'s and `RncCell`'s [`Ord`] impls to break
//! ties between cells the tile quilter ranks as otherwise equally suited to
//! a tile's zoom level (see `quilt_tiler::tiles::Quilt::render_tile`): the more
//! recently published/updated cell wins.

/// Comparable calendar date, canonicalised to `CCYY * 10_000 + MM * 100 + DD`
/// so ordering is a plain integer comparison.
///
/// [`Self::unknown`] is the smallest possible value, so a missing date never
/// outranks a real one — it only ever loses ties, it can't win them.
///
/// No calendar validation (e.g. day 31 in February is accepted): only
/// relative ordering matters here, not calendar correctness.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default, Hash)]
pub struct EditionDate(u32);

impl EditionDate {
    /// The oldest possible value — sorts before every real date.
    #[must_use]
    pub const fn unknown() -> Self {
        Self(0)
    }

    /// Construct from calendar components, with no validation.
    #[must_use]
    pub fn new(year: u16, month: u8, day: u8) -> Self {
        Self(u32::from(year) * 10_000 + u32::from(month) * 100 + u32::from(day))
    }

    /// Parse an 8-digit `CCYYMMDD` string (the OSENC `HEADER_CELL_PUBLISHDATE`
    /// / `HEADER_CELL_UPDATEDATE` wire format). `None` for anything that
    /// isn't exactly 8 ASCII digits — never panics. Callers decide how to
    /// surface a present-but-malformed date (it must not vanish silently).
    #[must_use]
    pub fn parse_ccyymmdd(s: &str) -> Option<Self> {
        let b = s.as_bytes();
        if b.len() != 8 || !b.iter().all(u8::is_ascii_digit) {
            return None;
        }
        let year = s[0..4].parse().ok()?;
        let month = s[4..6].parse().ok()?;
        let day = s[6..8].parse().ok()?;
        Some(Self::new(year, month, day))
    }

    /// Parse a `DD/MM/YYYY` string (the `.rnc` footer `edate` wire format,
    /// e.g. `"01/06/2026"`). `None` for anything that doesn't split into
    /// three numeric `/`-separated fields — never panics.
    #[must_use]
    pub fn parse_ddmmyyyy(s: &str) -> Option<Self> {
        let mut parts = s.split('/');
        let (Some(d), Some(m), Some(y), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return None;
        };
        Some(Self::new(y.parse().ok()?, m.parse().ok()?, d.parse().ok()?))
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccyymmdd_parses_real_osenc_dates() {
        assert_eq!(
            EditionDate::parse_ccyymmdd("20211209"),
            Some(EditionDate::new(2021, 12, 9))
        );
        assert_eq!(
            EditionDate::parse_ccyymmdd("20260520"),
            Some(EditionDate::new(2026, 5, 20))
        );
    }

    #[test]
    fn ddmmyyyy_parses_real_rnc_dates() {
        assert_eq!(
            EditionDate::parse_ddmmyyyy("01/06/2026"),
            Some(EditionDate::new(2026, 6, 1))
        );
        assert_eq!(
            EditionDate::parse_ddmmyyyy("05/01/2026"),
            Some(EditionDate::new(2026, 1, 5))
        );
    }

    /// Malformed input is reported as `None`, distinguishable from a real
    /// date — never silently mapped to [`EditionDate::unknown`].
    #[test]
    fn malformed_input_is_none_not_a_panic() {
        for s in ["", "2021120", "2021-12-09", "202112091"] {
            assert_eq!(EditionDate::parse_ccyymmdd(s), None, "{s:?}");
        }
        for s in [
            "",
            "01-06-2026",
            "2026-06-01",
            "01/06",
            "01/06/2026/1",
            "1/6/99999",
        ] {
            assert_eq!(EditionDate::parse_ddmmyyyy(s), None, "{s:?}");
        }
    }

    #[test]
    fn unknown_sorts_oldest() {
        assert!(EditionDate::unknown() < EditionDate::new(1, 1, 1));
    }

    #[test]
    fn newer_year_outranks_older_regardless_of_month_day() {
        assert!(EditionDate::new(2026, 1, 1) > EditionDate::new(2025, 12, 31));
    }
}
