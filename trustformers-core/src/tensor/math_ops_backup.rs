//! Tensor mathematical operations with numerical stability enhancements.
//!
//! This module contains mathematical operations for tensors including
//! basic arithmetic, matrix operations, and advanced mathematical functions.
//! All operations include numerical stability features such as overflow/underflow
//! protection, NaN/infinity detection, and optimized algorithms.
//!
//! ## Refactoring Summary
//!
//! Previously this was a single 2,662-line file containing all tensor math operations.
//! It has been split into focused modules:
//!
//! - `math_ops/stability.rs` - Numerical stability utilities (44 lines)
//! - `math_ops/broadcasting.rs` - Broadcasting compatibility functions (26 lines)
//! - `math_ops/arithmetic.rs` - Basic arithmetic operations (~350 lines)
//! - `math_ops/linear_algebra.rs` - Matrix operations and norms (~200 lines)
//! - `math_ops/mathematical.rs` - Math functions (sqrt, log, exp, trig) (~400 lines)
//! - `math_ops/statistical.rs` - Statistical operations (~250 lines)
//! - `math_ops/utilities.rs` - Utility operations and tensor manipulation (~300 lines)
//!
//! This refactoring improves:
//! - Code maintainability and readability
//! - Module compilation times
//! - Test isolation
//! - Code reuse through focused modules
//! - Developer experience when working on specific mathematical operations

use super::Tensor;
use crate::errors::{TrustformersError, Result};

// Re-export all the mathematical operations modules
pub use crate::tensor::math_ops::*;

// Import the math_ops modules
use crate::tensor::math_ops;

impl Tensor {
    // Re-export all functions with full backward compatibility

    // === ARITHMETIC OPERATIONS ===

    /// Element-wise addition with numerical stability enhancements
    pub fn add(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::arithmetic::add(self, other)
    }

    /// Element-wise subtraction with broadcasting support
    pub fn sub(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::arithmetic::sub(self, other)
    }

    /// Element-wise multiplication with broadcasting support
    pub fn mul(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::arithmetic::mul(self, other)
    }

    /// Element-wise division with division-by-zero protection
    pub fn div(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::arithmetic::div(self, other)
    }

    /// Scalar multiplication
    pub fn scalar_mul(&self, scalar: f32) -> Result<Tensor> {
        math_ops::arithmetic::scalar_mul(self, scalar)
    }

    /// Scalar division
    pub fn scalar_div(&self, scalar: f32) -> Result<Tensor> {
        math_ops::arithmetic::scalar_div(self, scalar)
    }

    /// Add scalar value
    pub fn add_scalar(&self, scalar: f32) -> Result<Tensor> {
        math_ops::arithmetic::add_scalar(self, scalar)
    }

    /// Subtract scalar value
    pub fn sub_scalar(&self, scalar: f32) -> Result<Tensor> {
        math_ops::arithmetic::sub_scalar(self, scalar)
    }

    /// Divide by scalar (alias for scalar_div)
    pub fn div_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_div(scalar)
    }

    /// Multiply by scalar (alias for scalar_mul)
    pub fn mul_scalar(&self, scalar: f32) -> Result<Tensor> {
        self.scalar_mul(scalar)
    }

    // === LINEAR ALGEBRA OPERATIONS ===

    /// Matrix multiplication with numerical stability enhancements
    pub fn matmul(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::linear_algebra::matmul(self, other)
    }

    /// L2 norm calculation
    pub fn norm(&self) -> Result<f32> {
        math_ops::linear_algebra::norm(self)
    }

    /// Squared L2 norm calculation
    pub fn norm_squared(&self) -> Result<Tensor> {
        math_ops::linear_algebra::norm_squared(self)
    }

    /// Gradient clipping based on norm
    pub fn clip_grad_norm(&self, max_norm: f32) -> Result<Tensor> {
        math_ops::linear_algebra::clip_grad_norm(self, max_norm)
    }

    // === MATHEMATICAL FUNCTIONS ===

    /// Element-wise power operation
    pub fn pow(&self, exponent: f32) -> Result<Tensor> {
        math_ops::mathematical::pow(self, exponent)
    }

    /// Element-wise square root
    pub fn sqrt(&self) -> Result<Tensor> {
        math_ops::mathematical::sqrt(self)
    }

    /// Element-wise reciprocal square root
    pub fn rsqrt(&self) -> Result<Tensor> {
        math_ops::mathematical::rsqrt(self)
    }

    /// Element-wise square
    pub fn square(&self) -> Result<Tensor> {
        math_ops::mathematical::square(self)
    }

    /// Element-wise reciprocal
    pub fn reciprocal(&self) -> Result<Tensor> {
        math_ops::mathematical::reciprocal(self)
    }

    /// Element-wise exponential
    pub fn exp(&self) -> Result<Tensor> {
        math_ops::mathematical::exp(self)
    }

    /// Element-wise natural logarithm
    pub fn log(&self) -> Result<Tensor> {
        math_ops::mathematical::log(self)
    }

    /// Element-wise sine
    pub fn sin(&self) -> Result<Tensor> {
        math_ops::mathematical::sin(self)
    }

    /// Element-wise cosine
    pub fn cos(&self) -> Result<Tensor> {
        math_ops::mathematical::cos(self)
    }

    /// Element-wise tangent
    pub fn tan(&self) -> Result<Tensor> {
        math_ops::mathematical::tan(self)
    }

    /// Element-wise arcsine
    pub fn asin(&self) -> Result<Tensor> {
        math_ops::mathematical::asin(self)
    }

    /// Element-wise arccosine
    pub fn acos(&self) -> Result<Tensor> {
        math_ops::mathematical::acos(self)
    }

    /// Element-wise arctangent
    pub fn atan(&self) -> Result<Tensor> {
        math_ops::mathematical::atan(self)
    }

    /// Element-wise absolute value
    pub fn abs(&self) -> Result<Tensor> {
        math_ops::mathematical::abs(self)
    }

    /// Element-wise negation
    pub fn neg(&self) -> Result<Tensor> {
        math_ops::mathematical::neg(self)
    }

    /// Element-wise sign function
    pub fn sign(&self) -> Result<Tensor> {
        math_ops::mathematical::sign(self)
    }

    /// Element-wise rounding
    pub fn round(&self) -> Result<Tensor> {
        math_ops::mathematical::round(self)
    }

    /// Element-wise floor
    pub fn floor(&self) -> Result<Tensor> {
        math_ops::mathematical::floor(self)
    }

    /// Element-wise ceiling
    pub fn ceil(&self) -> Result<Tensor> {
        math_ops::mathematical::ceil(self)
    }

    /// Element-wise truncation
    pub fn trunc(&self) -> Result<Tensor> {
        math_ops::mathematical::trunc(self)
    }

    /// Check for NaN values
    pub fn isnan(&self) -> Result<Tensor> {
        math_ops::mathematical::isnan(self)
    }

    /// Check for infinite values
    pub fn isinf(&self) -> Result<Tensor> {
        math_ops::mathematical::isinf(self)
    }

    /// Check for finite values
    pub fn isfinite(&self) -> Result<Tensor> {
        math_ops::mathematical::isfinite(self)
    }

    // === STATISTICAL OPERATIONS ===

    /// Sum reduction
    pub fn sum(&self, axes: Option<Vec<usize>>, keepdims: bool) -> Result<Tensor> {
        math_ops::statistical::sum(self, axes, keepdims)
    }

    /// Sum along specified axes
    pub fn sum_axes(&self, axes: &[usize]) -> Result<Tensor> {
        math_ops::statistical::sum_axes(self, axes)
    }

    /// Sum along single axis
    pub fn sum_axis(&self, axis: usize) -> Result<Tensor> {
        math_ops::statistical::sum_axis(self, axis)
    }

    /// Mean calculation
    pub fn mean(&self) -> Result<Tensor> {
        math_ops::statistical::mean(self)
    }

    /// Mean along specified axes
    pub fn mean_axes(&self, axes: &[usize]) -> Result<Tensor> {
        math_ops::statistical::mean_axes(self, axes)
    }

    /// Mean along single axis
    pub fn mean_axis(&self, axis: usize) -> Result<Tensor> {
        math_ops::statistical::mean_axis(self, axis)
    }

    /// Variance calculation
    pub fn variance(&self, axes: Option<&[usize]>, keepdims: bool) -> Result<Tensor> {
        math_ops::statistical::variance(self, axes, keepdims)
    }

    /// Standard deviation calculation
    pub fn std_dev(&self, axes: Option<&[usize]>, keepdims: bool) -> Result<Tensor> {
        math_ops::statistical::std_dev(self, axes, keepdims)
    }

    /// Standard deviation (alias)
    pub fn std(&self) -> Result<Tensor> {
        math_ops::statistical::std(self)
    }

    /// Find minimum and maximum values
    pub fn min_max(&self) -> Result<(f32, f32)> {
        math_ops::statistical::min_max(self)
    }

    /// Maximum value
    pub fn max_value(&self) -> Result<Tensor> {
        math_ops::statistical::max_value(self)
    }

    /// Element-wise maximum
    pub fn max(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::statistical::max(self, other)
    }

    /// Argument of maximum values
    pub fn argmax(&self, axis: i32) -> Result<Tensor> {
        math_ops::statistical::argmax(self, axis)
    }

    // === UTILITY OPERATIONS ===

    /// Scale tensor by factor
    pub fn scale(&self, factor: f32) -> Result<Tensor> {
        math_ops::utilities::scale(self, factor)
    }

    /// Clamp values to range
    pub fn clamp(&self, min_val: f32, max_val: f32) -> Result<Tensor> {
        math_ops::utilities::clamp(self, min_val, max_val)
    }

    /// Broadcast tensor to target shape
    pub fn broadcast_to(&self, shape: &[usize]) -> Result<Tensor> {
        math_ops::utilities::broadcast_to(self, shape)
    }

    /// Get scalar value at index
    pub fn get_scalar(&self, indices: &[usize]) -> Result<f32> {
        math_ops::utilities::get_scalar(self, indices)
    }

    /// Set scalar value at index
    pub fn set_scalar(&self, indices: &[usize], value: f32) -> Result<Tensor> {
        math_ops::utilities::set_scalar(self, indices, value)
    }

    /// Element-wise greater than comparison
    pub fn greater(&self, other: &Tensor) -> Result<Tensor> {
        math_ops::utilities::greater(self, other)
    }

    /// Linear interpolation
    pub fn lerp(&self, other: &Tensor, weight: f32) -> Result<Tensor> {
        math_ops::utilities::lerp(self, other, weight)
    }

    /// Subtract scaled tensor
    pub fn sub_scaled(&self, other: &Tensor, factor: f32) -> Result<Tensor> {
        math_ops::utilities::sub_scaled(self, other, factor)
    }

    /// Add scaled tensor
    pub fn add_scaled(&self, other: &Tensor, factor: f32) -> Result<Tensor> {
        math_ops::utilities::add_scaled(self, other, factor)
    }

    // === SPECIALIZED OPERATIONS ===

    /// Power with 64-bit exponent
    pub fn pow_scalar(&self, exponent: f64) -> Result<Tensor> {
        math_ops::mathematical::pow_scalar(self, exponent)
    }

    /// Layer normalization.
    ///
    /// LN(x) = (x − mean(x)) / sqrt(var(x) + ε) computed over the specified
    /// `axis` (typically the last dimension). The canonical implementation
    /// lives in `math_ops/advanced.rs` which is the module compiled into the
    /// crate; this backup shim records the same signature for reference.
    ///
    /// Note: this file (`math_ops_backup.rs`) is not included in the module
    /// tree — the live implementation is in `math_ops::advanced`.
    pub fn layer_norm(&self, axis: i32, epsilon: f32) -> Result<Tensor> {
        let ndim = match self {
            Tensor::F32(a) => a.ndim(),
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Layer norm only supported for F32 tensors in backup shim",
                    "layer_norm",
                ))
            },
        };
        let canonical_axis =
            if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };
        if canonical_axis >= ndim {
            return Err(TrustformersError::tensor_op_error(
                &format!("Axis {} out of bounds for {}-dim tensor", axis, ndim),
                "layer_norm",
            ));
        }
        // The real implementation is in math_ops/advanced.rs (compiled).
        // This backup re-implements the same logic inline for reference.
        use scirs2_core::ndarray::Axis;
        match self {
            Tensor::F32(a) => {
                let last_dim = a.ndim() - 1;
                if canonical_axis != last_dim {
                    return Err(TrustformersError::tensor_op_error(
                        "Backup layer_norm only supports last-dimension normalization",
                        "layer_norm",
                    ));
                }
                let mean = a.mean_axis(Axis(canonical_axis)).ok_or_else(|| {
                    TrustformersError::tensor_op_error("mean computation failed", "layer_norm")
                })?;
                let var = a.map_axis(Axis(canonical_axis), |lane| {
                    let lane_mean =
                        lane.mean().unwrap_or(0.0_f32);
                    lane.mapv(|x| (x - lane_mean).powi(2)).mean().unwrap_or(0.0_f32)
                });
                let mut result = a.clone();
                for (i, mut lane) in result.axis_iter_mut(Axis(canonical_axis)).enumerate() {
                    let m = mean[i];
                    let v = var[i];
                    lane.mapv_inplace(|x| (x - m) / (v + epsilon).sqrt());
                }
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Layer norm only supported for F32 tensors",
                "layer_norm",
            )),
        }
    }

    /// Cross entropy loss.
    ///
    /// For each sample: `−Σ y_true · log(y_pred + ε)`.
    /// Uses numerically stable log to avoid `log(0)`.
    ///
    /// Note: the canonical implementation is in `math_ops/advanced.rs`.
    pub fn cross_entropy(&self, targets: &Tensor, reduction: &str) -> Result<Tensor> {
        use scirs2_core::ndarray::{ArrayD, IxDyn, Zip};
        match (self, targets) {
            (Tensor::F32(predictions), Tensor::F32(targets)) => {
                const EPS: f32 = 1e-8;
                let log_preds = predictions.mapv(|x| (x + EPS).ln());
                let losses = Zip::from(&log_preds)
                    .and(targets)
                    .map_collect(|&lp, &t| -t * lp);
                match reduction {
                    "mean" => {
                        let mean_loss = losses.mean().unwrap_or(0.0_f32);
                        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), mean_loss)))
                    },
                    "sum" => {
                        let sum_loss = losses.sum();
                        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_loss)))
                    },
                    "none" => Ok(Tensor::F32(losses)),
                    _ => Err(TrustformersError::tensor_op_error(
                        "Invalid reduction; use 'mean', 'sum', or 'none'",
                        "cross_entropy",
                    )),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cross entropy only supported for F32 tensors",
                "cross_entropy",
            )),
        }
    }

    /// Cosine similarity.
    ///
    /// `cos_sim(a, b) = (a · b) / (‖a‖ · ‖b‖ + ε)` computed along `dim`.
    ///
    /// Note: the canonical implementation is in `math_ops/advanced.rs`.
    pub fn cosine_similarity(&self, other: &Tensor, dim: i32, eps: f32) -> Result<Tensor> {
        use scirs2_core::ndarray::{Axis, Zip};
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                let ndim = a.ndim();
                let axis =
                    if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };
                let dot = Zip::from(a).and(b).map_collect(|&x, &y| x * y).sum_axis(Axis(axis));
                let norm_a =
                    a.mapv(|x| x * x).sum_axis(Axis(axis)).mapv(|x| (x + eps).sqrt());
                let norm_b =
                    b.mapv(|x| x * x).sum_axis(Axis(axis)).mapv(|x| (x + eps).sqrt());
                let similarity = Zip::from(&dot)
                    .and(&norm_a)
                    .and(&norm_b)
                    .map_collect(|&d, &na, &nb| d / (na * nb));
                Ok(Tensor::F32(similarity))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cosine similarity only supported for F32 tensors",
                "cosine_similarity",
            )),
        }
    }

    /// Log softmax.
    ///
    /// `log_softmax(x)_i = x_i − max(x) − log(Σ exp(x_j − max(x)))`.
    /// The max subtraction is the standard log-sum-exp trick for numerical
    /// stability, preventing overflow in `exp`.
    ///
    /// Note: the canonical implementation is in `math_ops/advanced.rs`.
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        use scirs2_core::ndarray::Axis;
        match self {
            Tensor::F32(a) => {
                let ndim = a.ndim();
                let axis =
                    if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };
                if axis >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!("Axis {} out of bounds for {}-dim tensor", dim, ndim),
                        "log_softmax",
                    ));
                }
                // max over axis for numerical stability
                let max_vals = a.map_axis(Axis(axis), |lane| {
                    lane.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
                });
                let mut max_shape = a.shape().to_vec();
                max_shape[axis] = 1;
                let max_exp = max_vals
                    .into_shape_with_order(max_shape.clone())
                    .map_err(|_| {
                        TrustformersError::tensor_op_error(
                            "reshape failed for max_vals",
                            "log_softmax",
                        )
                    })?;
                let broadcast_max =
                    max_exp.broadcast(a.raw_dim()).ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "broadcast failed for max_vals",
                            "log_softmax",
                        )
                    })?;
                let shifted = a - &broadcast_max;
                let log_sum_exp = shifted
                    .mapv(|x| x.exp())
                    .sum_axis(Axis(axis))
                    .mapv(|x| x.ln())
                    .into_shape_with_order(max_shape)
                    .map_err(|_| {
                        TrustformersError::tensor_op_error(
                            "reshape failed for log_sum_exp",
                            "log_softmax",
                        )
                    })?;
                let broadcast_lse =
                    log_sum_exp.broadcast(a.raw_dim()).ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "broadcast failed for log_sum_exp",
                            "log_softmax",
                        )
                    })?;
                Ok(Tensor::F32(shifted - &broadcast_lse))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Log softmax only supported for F32 tensors",
                "log_softmax",
            )),
        }
    }
}

// Re-export compatibility functions if they existed in the original
// These might be needed for backward compatibility

// Convenience functions for shapes_are_broadcastable if it was public
pub use math_ops::broadcasting::shapes_are_broadcastable;

// Stability functions if they were public
pub use math_ops::stability::{
    is_stable_f32, is_stable_f64, stabilize_f32, stabilize_f64,
    STABILITY_EPSILON_F32, STABILITY_EPSILON_F64, MAX_SAFE_VALUE_F32, MAX_SAFE_VALUE_F64
};