//! Native-scale → tile zoom level mapping.

/// Combined constant: `40_075_016 / (256 × 0.00028) ≈ 559_082_264`.
pub const ZOOM_K: f64 = 559_082_264.0;

/// Compute the tile zoom level that best represents a chart at `1:native_scale`.
///
/// Formula: `z = floor(log2(ZOOM_K / native_scale) + offset)`, clamped to `[0, 22]`.
/// Pass `offset = 0.0` for the unshifted result.  Fractional offsets are applied
/// before flooring, so they shift the scale breakpoints between zoom levels rather
/// than nudging an already-rounded integer.
///
/// `native_scale` must be ≥ 1; passing 0 produces division by zero in the
/// formula (returns zoom 22 via `f64::INFINITY` clamping, but the result is
/// meaningless).  Callers are expected to reject cells with unknown scale
/// at parse time.
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn zoom_from_scale(native_scale: u32, offset: f64) -> u8 {
    let log2 = (ZOOM_K / f64::from(native_scale)).log2();
    // Safety: value is clamped to [0.0, 22.0] before cast.
    (log2 + offset).floor().clamp(0.0, 22.0) as u8
}

/// Compute the nominal scale denominator for a tile at `zoom` with `offset`
/// applied.  Inverse of [`zoom_from_scale`].
#[must_use]
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn scale_from_zoom(zoom: u8, offset: f64) -> u32 {
    // Safety: value is clamped to a positive finite range before cast.
    (ZOOM_K / (f64::from(zoom) - offset).exp2())
        .round()
        .clamp(1.0, f64::from(u32::MAX)) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── zoom_from_scale boundaries ──────────────────────────────────────


    #[test]
    fn very_large_native_scale_clamps_to_zoom_0() {
        assert_eq!(zoom_from_scale(u32::MAX, 0.0), 0);
    }

    #[test]
    fn very_small_native_scale_clamps_to_zoom_22() {
        assert_eq!(zoom_from_scale(1, 0.0), 22);
    }

    #[test]
    fn negative_offset_clamps_zoom_to_zero() {
        // log2(ZOOM_K / 1_000_000) ≈ 9.13; offset −15 → floor(−5.87) → clamp 0.
        assert_eq!(zoom_from_scale(1_000_000, -15.0), 0);
    }

    #[test]
    fn large_positive_offset_clamps_zoom_to_22() {
        // log2(ZOOM_K / 1_000_000) ≈ 9.13; offset +20 → floor(29.13) → clamp 22.
        assert_eq!(zoom_from_scale(1_000_000, 20.0), 22);
    }

    // ── scale_from_zoom contract ────────────────────────────────────────

    #[test]
    fn scale_from_zoom_is_always_at_least_one() {
        // Extreme zoom or offset must never produce 0.
        assert!(scale_from_zoom(255, 0.0) >= 1);
        assert!(scale_from_zoom(22, 0.0) >= 1);
        assert!(scale_from_zoom(0, -100.0) >= 1);
    }

    #[test]
    fn scale_from_zoom_large_positive_offset_saturates_to_u32_max() {
        // zoom 0 with offset +100 → ZOOM_K / 2^(−100) = ZOOM_K × 2^100 → clamp to u32::MAX.
        assert_eq!(scale_from_zoom(0, 100.0), u32::MAX);
    }

    // ── round-trip stability ────────────────────────────────────────────

    #[test]
    fn zoom_round_trip_within_one_level() {
        // zoom → representative scale → zoom may lose at most one level
        // because zoom_from_scale floors while scale_from_zoom rounds.
        for z in 0..=22_u8 {
            let s = scale_from_zoom(z, 0.0);
            let z2 = zoom_from_scale(s, 0.0);
            assert!(
                z2 == z || z.checked_sub(1) == Some(z2),
                "round-trip failed: z={z} → scale={s} → z2={z2}",
            );
        }
    }

    #[test]
    fn scale_round_trip_within_factor_of_two() {
        // scale → zoom → scale stays within a factor of 2 (one zoom step).
        for native in [500_000_u32, 1_000_000, 3_000_000, 10_000_000] {
            let z = zoom_from_scale(native, 0.0);
            let s_back = scale_from_zoom(z, 0.0);
            let ratio = f64::from(s_back) / f64::from(native);
            assert!(
                (0.5..=2.0).contains(&ratio),
                "scale round-trip out of range: native={native} → z={z} → s_back={s_back} (ratio={ratio:.3})",
            );
        }
    }
}
