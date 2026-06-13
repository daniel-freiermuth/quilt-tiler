//! Native-scale → tile zoom level mapping.

/// Combined constant: `40_075_016 / (256 × 0.00028) ≈ 559_082_264`.
pub const ZOOM_K: f64 = 559_082_264.0;

/// Compute the tile zoom level that best represents a chart at `1:native_scale`.
///
/// Formula: `z = floor(log2(ZOOM_K / native_scale))`, clamped to `[0, 22]`.
pub fn zoom_from_scale(native_scale: u32) -> u8 {
    if native_scale == 0 {
        return 14;
    }
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    // Safety: value is clamped to [0.0, 22.0] before cast.
    let z = (ZOOM_K / f64::from(native_scale))
        .log2()
        .floor()
        .clamp(0.0, 22.0) as u8;
    z
}
