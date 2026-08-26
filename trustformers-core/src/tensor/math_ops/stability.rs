//! Numerical stability utilities for tensor mathematical operations
//!
//! This module provides constants and functions for detecting and containing
//! *genuine* numerical hazards: non-finite values (`NaN`, `±inf`) and magnitudes
//! large enough that a subsequent multiply/accumulate would overflow `f32`/`f64`.
//!
//! # What is intentionally **not** treated as a hazard
//!
//! Very small magnitudes (gradual underflow towards zero) are **not** a stability
//! problem for the operations in this crate. Historically [`is_stable_f32`]
//! classified any `0 < |x| <= 1e-7` as "unstable", which made ordinary trained
//! weights (a `4096x4096` matrix drawn from `N(0, 0.02)` contains ~67 such
//! elements in expectation) trip the compensated scalar fallback in `matmul`,
//! and made [`stabilize_f32`] silently rewrite tiny activations to `±1e-7`.
//! Both behaviours were removed: underflow towards zero is well-defined IEEE-754
//! behaviour and is the mathematically correct answer.

/// Smallest magnitude historically considered "significant" for `f32`.
///
/// Retained for backwards compatibility with downstream code that uses it as an
/// epsilon for division guards. It is **not** used by [`is_stable_f32`] or
/// [`stabilize_f32`] any more.
pub const STABILITY_EPSILON_F32: f32 = 1e-7;
/// Smallest magnitude historically considered "significant" for `f64`.
///
/// See [`STABILITY_EPSILON_F32`] for why this no longer participates in the
/// stability predicates.
pub const STABILITY_EPSILON_F64: f64 = 1e-15;
/// Largest `f32` magnitude that can be squared/accumulated without overflowing.
pub const MAX_SAFE_VALUE_F32: f32 = 1e30;
/// Largest `f64` magnitude that can be squared/accumulated without overflowing.
pub const MAX_SAFE_VALUE_F64: f64 = 1e300;

/// Check if a float value is numerically stable.
///
/// A value is stable when it is finite and its magnitude is below
/// [`MAX_SAFE_VALUE_F32`]. Underflow towards zero is *not* an instability:
/// `is_stable_f32(1e-30)` and `is_stable_f32(0.0)` are both `true`, while
/// `NaN`, `±inf` and `1e31` are `false`.
#[inline]
pub fn is_stable_f32(x: f32) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F32
}

/// Check if a float value is numerically stable (64-bit).
///
/// See [`is_stable_f32`]; underflow towards zero is not treated as unstable.
#[inline]
pub fn is_stable_f64(x: f64) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F64
}

/// Stabilize a float value by clamping to safe ranges.
///
/// * `NaN`/`±inf` become `0.0`
/// * magnitudes above [`MAX_SAFE_VALUE_F32`] are clamped to `±MAX_SAFE_VALUE_F32`
/// * every other value (including denormals and exact zero) is returned unchanged
#[inline]
pub fn stabilize_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    if x.abs() > MAX_SAFE_VALUE_F32 {
        x.signum() * MAX_SAFE_VALUE_F32
    } else {
        x
    }
}

/// Stabilize a float value by clamping to safe ranges (64-bit).
///
/// See [`stabilize_f32`] for the exact semantics.
#[inline]
pub fn stabilize_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x.abs() > MAX_SAFE_VALUE_F64 {
        x.signum() * MAX_SAFE_VALUE_F64
    } else {
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the "tiny weights are unstable" defect: values in
    /// `(0, 1e-7]` used to be reported as unstable, which forced `matmul` onto a
    /// scalar Kahan fallback for ordinary trained weight matrices.
    #[test]
    fn small_magnitudes_are_stable() {
        for &x in &[1e-8f32, 1e-20, 1e-30, f32::MIN_POSITIVE, -1e-8, 0.0] {
            assert!(is_stable_f32(x), "{x} should be considered stable");
        }
        for &x in &[1e-16f64, 1e-200, -1e-16, 0.0] {
            assert!(is_stable_f64(x), "{x} should be considered stable");
        }
    }

    #[test]
    fn genuine_hazards_are_unstable() {
        assert!(!is_stable_f32(f32::NAN));
        assert!(!is_stable_f32(f32::INFINITY));
        assert!(!is_stable_f32(-f32::INFINITY));
        assert!(!is_stable_f32(1e31));
        assert!(!is_stable_f64(f64::NAN));
        assert!(!is_stable_f64(1e301));
    }

    /// Regression test: `stabilize_*` must not rewrite tiny values any more.
    #[test]
    fn stabilize_preserves_small_values() {
        assert_eq!(stabilize_f32(1e-30), 1e-30);
        assert_eq!(stabilize_f32(-1e-30), -1e-30);
        assert_eq!(stabilize_f64(1e-300), 1e-300);
        assert_eq!(stabilize_f32(0.0), 0.0);
    }

    #[test]
    fn stabilize_contains_real_hazards() {
        assert_eq!(stabilize_f32(f32::NAN), 0.0);
        assert_eq!(stabilize_f32(f32::INFINITY), 0.0);
        assert_eq!(stabilize_f32(1e31), MAX_SAFE_VALUE_F32);
        assert_eq!(stabilize_f64(-1e301), -MAX_SAFE_VALUE_F64);
    }
}
