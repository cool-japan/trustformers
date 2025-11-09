//! Common utilities and constants for tensor mathematical operations.

use super::super::Tensor;

/// Numerical stability constants for mathematical operations
#[allow(dead_code)] // Public API utility constant
pub const STABILITY_EPSILON_F32: f32 = 1e-7;
#[allow(dead_code)] // Public API utility constant
pub const STABILITY_EPSILON_F64: f64 = 1e-15;
#[allow(dead_code)] // Public API utility constant
pub const MAX_SAFE_VALUE_F32: f32 = 1e30;
#[allow(dead_code)] // Public API utility constant
pub const MAX_SAFE_VALUE_F64: f64 = 1e300;

/// Check if a float value is numerically stable
#[allow(dead_code)] // Public API utility function
pub fn is_stable_f32(x: f32) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F32 && (x.abs() > STABILITY_EPSILON_F32 || x == 0.0)
}

/// Check if a float value is numerically stable (64-bit)
#[allow(dead_code)] // Public API utility function
pub fn is_stable_f64(x: f64) -> bool {
    x.is_finite() && x.abs() < MAX_SAFE_VALUE_F64 && (x.abs() > STABILITY_EPSILON_F64 || x == 0.0)
}

/// Stabilize a float value by clamping to safe ranges
#[allow(dead_code)] // Public API utility function
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
#[allow(dead_code)] // Public API utility function
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

impl Tensor {
    /// Check if two shapes are broadcastable according to numpy-style broadcasting rules
    pub fn shapes_are_broadcastable(shape1: &[usize], shape2: &[usize]) -> bool {
        let max_len = shape1.len().max(shape2.len());
        let mut s1 = vec![1; max_len];
        let mut s2 = vec![1; max_len];

        // Right-align shapes for broadcasting comparison
        for (i, &dim) in shape1.iter().rev().enumerate() {
            if i < max_len {
                s1[max_len - 1 - i] = dim;
            }
        }
        for (i, &dim) in shape2.iter().rev().enumerate() {
            if i < max_len {
                s2[max_len - 1 - i] = dim;
            }
        }

        // Check if broadcasting is possible
        for (d1, d2) in s1.iter().zip(s2.iter()) {
            if *d1 != *d2 && *d1 != 1 && *d2 != 1 {
                return false; // Incompatible
            }
        }
        true // Compatible through broadcasting
    }
}
