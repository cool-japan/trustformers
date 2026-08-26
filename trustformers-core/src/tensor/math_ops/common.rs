//! Common utilities and constants for tensor mathematical operations.

use super::super::Tensor;

// Numerical stability constants and predicates live in `super::stability`.
// They used to be duplicated verbatim here, which allowed the two copies to
// drift apart; this module now re-exports the single source of truth so the
// legacy `math_ops::common::*` paths keep working.
#[allow(unused_imports)]
pub use super::stability::{
    is_stable_f32, is_stable_f64, stabilize_f32, stabilize_f64, MAX_SAFE_VALUE_F32,
    MAX_SAFE_VALUE_F64, STABILITY_EPSILON_F32, STABILITY_EPSILON_F64,
};

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
