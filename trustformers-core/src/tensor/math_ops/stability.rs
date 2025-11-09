//! Numerical stability utilities for tensor mathematical operations
//!
//! This module provides constants and functions for maintaining numerical
//! stability during tensor computations, including overflow/underflow protection
//! and safe value clamping.

/// Numerical stability constants for mathematical operations
pub const STABILITY_EPSILON_F32: f32 = 1e-7;
pub const STABILITY_EPSILON_F64: f64 = 1e-15;
pub const MAX_SAFE_VALUE_F32: f32 = 1e30;
pub const MAX_SAFE_VALUE_F64: f64 = 1e300;

/// Check if a float value is numerically stable
pub fn is_stable_f32(x: f32) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F32 && (x.abs() > STABILITY_EPSILON_F32 || x == 0.0)
}

/// Check if a float value is numerically stable (64-bit)
pub fn is_stable_f64(x: f64) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F64 && (x.abs() > STABILITY_EPSILON_F64 || x == 0.0)
}

/// Stabilize a float value by clamping to safe ranges
pub fn stabilize_f32(x: f32) -> f32 {
    if !x.is_finite() {
        return 0.0;
    }
    if x.abs() > MAX_SAFE_VALUE_F32 {
        x.signum() * MAX_SAFE_VALUE_F32
    } else if x.abs() < STABILITY_EPSILON_F32 && x != 0.0 {
        x.signum() * STABILITY_EPSILON_F32
    } else {
        x
    }
}

/// Stabilize a float value by clamping to safe ranges (64-bit)
pub fn stabilize_f64(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    if x.abs() > MAX_SAFE_VALUE_F64 {
        x.signum() * MAX_SAFE_VALUE_F64
    } else if x.abs() < STABILITY_EPSILON_F64 && x != 0.0 {
        x.signum() * STABILITY_EPSILON_F64
    } else {
        x
    }
}
