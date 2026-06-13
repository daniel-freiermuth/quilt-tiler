//! Native-scale → tile zoom level mapping.

/// Combined constant: `40_075_016 / (256 × 0.00028) ≈ 559_082_264`.
pub const ZOOM_K: f64 = 559_082_264.0;

/// Compute the tile zoom level that best represents a chart at `1:native_scale`.
///
/// Formula: `z = floor(log2(ZOOM_K / native_scale) + offset)`, clamped to `[0, 22]`.
/// Pass `offset = 0.0` for the unshifted result.  Fractional offsets are applied
/// before flooring, so they shift the scale breakpoints between zoom levels rather
/// than nudging an already-rounded integer.
#[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
pub fn zoom_from_scale(native_scale: u32, offset: f64) -> u8 {
    let log2 = if native_scale == 0 {
        14.0
    } else {
        (ZOOM_K / f64::from(native_scale)).log2()
    };
    // Safety: value is clamped to [0.0, 22.0] before cast.
    (log2 + offset).floor().clamp(0.0, 22.0) as u8
}
