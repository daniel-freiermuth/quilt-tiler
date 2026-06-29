//! A chart cell's edition/publish date, kept only as a comparable ranking
//! key — never used for calendar arithmetic.
//!
//! Consumed by [`crate::S57Cell`]'s and `RncCell`'s [`Ord`] impls to break
//! ties between cells the tile quilter ranks as otherwise equally suited to
//! a tile's zoom level (see `quilt_tiler::tiles::render_tile`): the more
//! recently published/updated cell wins.

/// Comparable calendar date, canonicalised to `CCYY * 10_000 + MM * 100 + DD`
/// so ordering is a plain integer comparison.
///
/// [`Self::unknown`] is the smallest possible value, so a missing or
/// unparseable date never outranks a real one — it only ever loses ties,
/// it can't win them.
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
    /// / `HEADER_CELL_UPDATEDATE` wire format). Returns [`Self::unknown`] for
    /// anything that isn't exactly 8 ASCII digits — never panics.
    #[must_use]
    pub fn from_ccyymmdd(s: &str) -> Self {
        let b = s.as_bytes();
        if b.len() != 8 || !b.iter().all(u8::is_ascii_digit) {
            return Self::unknown();
        }
        let Ok(year) = s[0..4].parse() else {
            return Self::unknown();
        };
        let Ok(month) = s[4..6].parse() else {
            return Self::unknown();
        };
        let Ok(day) = s[6..8].parse() else {
            return Self::unknown();
        };
        Self::new(year, month, day)
    }

    /// Parse a `DD/MM/YYYY` string (the `.rnc` footer `edate` wire format,
    /// e.g. `"01/06/2026"`). Returns [`Self::unknown`] for anything that
    /// doesn't split into three numeric `/`-separated fields — never panics.
    #[must_use]
    pub fn from_ddmmyyyy(s: &str) -> Self {
        let mut parts = s.split('/');
        let (Some(d), Some(m), Some(y), None) =
            (parts.next(), parts.next(), parts.next(), parts.next())
        else {
            return Self::unknown();
        };
        let (Ok(day), Ok(month), Ok(year)) = (d.parse(), m.parse(), y.parse()) else {
            return Self::unknown();
        };
        Self::new(year, month, day)
    }
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ccyymmdd_parses_real_osenc_dates() {
        assert_eq!(
            EditionDate::from_ccyymmdd("20211209"),
            EditionDate::new(2021, 12, 9)
        );
        assert_eq!(
            EditionDate::from_ccyymmdd("20260520"),
            EditionDate::new(2026, 5, 20)
        );
    }

    #[test]
    fn ddmmyyyy_parses_real_rnc_dates() {
        assert_eq!(
            EditionDate::from_ddmmyyyy("01/06/2026"),
            EditionDate::new(2026, 6, 1)
        );
        assert_eq!(
            EditionDate::from_ddmmyyyy("05/01/2026"),
            EditionDate::new(2026, 1, 5)
        );
    }

    #[test]
    fn malformed_input_is_unknown_not_a_panic() {
        assert_eq!(EditionDate::from_ccyymmdd(""), EditionDate::unknown());
        assert_eq!(
            EditionDate::from_ccyymmdd("2021120"),
            EditionDate::unknown()
        );
        assert_eq!(
            EditionDate::from_ccyymmdd("2021-12-09"),
            EditionDate::unknown()
        );
        assert_eq!(EditionDate::from_ddmmyyyy(""), EditionDate::unknown());
        assert_eq!(
            EditionDate::from_ddmmyyyy("01-06-2026"),
            EditionDate::unknown()
        );
        assert_eq!(EditionDate::from_ddmmyyyy("01/06"), EditionDate::unknown());
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
