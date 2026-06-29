//! Native-scale → tile zoom level mapping.

/// Combined constant: `40_075_016 / (256 × 0.00028) ≈ 559_082_264`.
pub const ZOOM_K: f64 = 559_082_264.0;

/// Compute the tile zoom level that best represents a chart at `1:native_scale`.
///
/// Formula: `z = floor(log2(ZOOM_K / native_scale) + offset)`, clamped to `[0, 22]`.
/// Pass `offset = 0.0` for the unshifted result.  Fractional offsets are applied
/// before flooring, so they shift the scale breakpoints between zoom levels rather
/// than nudging an already-rounded integer.
#[must_use]
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
