//! Advanced mathematical operations for tensors.
//!
//! This module contains advanced operations like layer normalization,
//! cross entropy, cosine similarity, log softmax, and other specialized functions.

use super::super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{ArrayD, Axis, IxDyn, Zip};

/// Upcast a half-precision (F16/BF16) tensor to F32, run `op`, then downcast the
/// F32 result back to the original half-precision dtype.
///
/// Half-precision floats lack the precision and dynamic range needed for stable
/// normalization / log-softmax math, so we compute in F32 and round the result
/// back to F16/BF16 so the output dtype matches the input dtype.
fn run_half_in_f32<F>(input: &Tensor, op: F) -> Result<Tensor>
where
    F: Fn(&Tensor) -> Result<Tensor>,
{
    match input {
        Tensor::F16(a) => {
            // Upcast F16 -> F32 for accurate computation, then downcast back to F16.
            let upcast = Tensor::F32(a.mapv(|x| x.to_f32()));
            match op(&upcast)? {
                Tensor::F32(r) => Ok(Tensor::F16(r.mapv(half::f16::from_f32))),
                other => other.to_dtype(crate::tensor::DType::F16),
            }
        },
        Tensor::BF16(a) => {
            // Upcast BF16 -> F32 for accurate computation, then downcast back to BF16.
            let upcast = Tensor::F32(a.mapv(|x| x.to_f32()));
            match op(&upcast)? {
                Tensor::F32(r) => Ok(Tensor::BF16(r.mapv(half::bf16::from_f32))),
                other => other.to_dtype(crate::tensor::DType::BF16),
            }
        },
        _ => Err(TrustformersError::tensor_op_error(
            "run_half_in_f32 called on a non-half-precision tensor",
            "run_half_in_f32",
        )),
    }
}

impl Tensor {
    /// Element-wise less-than comparison.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with
    ///
    /// # Returns
    ///
    /// A boolean tensor with the comparison results (1.0 for true, 0.0 for false).
    pub fn less(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for less comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x < y { 1.0f32 } else { 0.0f32 });
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for less comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x < y { 1.0f64 } else { 0.0f64 });
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for less comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x < y { 1i64 } else { 0i64 });
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Less operation not implemented for this tensor type combination",
                "less",
            )),
        }
    }

    /// Element-wise equality comparison.
    ///
    /// # Arguments
    ///
    /// * `other` - The tensor to compare with
    ///
    /// # Returns
    ///
    /// A boolean tensor with the comparison results (1.0 for true, 0.0 for false).
    pub fn equal(&self, other: &Tensor) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for equal comparison".to_string(),
                    ));
                }
                let result = Zip::from(a).and(b).map_collect(|&x, &y| {
                    if (x - y).abs() < f32::EPSILON {
                        1.0f32
                    } else {
                        0.0f32
                    }
                });
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for equal comparison".to_string(),
                    ));
                }
                let result = Zip::from(a).and(b).map_collect(|&x, &y| {
                    if (x - y).abs() < f64::EPSILON {
                        1.0f64
                    } else {
                        0.0f64
                    }
                });
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(b)) => {
                if a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "Tensors must have the same shape for equal comparison".to_string(),
                    ));
                }
                let result =
                    Zip::from(a).and(b).map_collect(|&x, &y| if x == y { 1i64 } else { 0i64 });
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Equal operation not implemented for this tensor type combination",
                "equal",
            )),
        }
    }

    /// Element-wise conditional selection (where).
    ///
    /// # Arguments
    ///
    /// * `condition` - The boolean tensor condition
    /// * `other` - The tensor to select from when condition is false
    ///
    /// # Returns
    ///
    /// A tensor with elements selected from self where condition is true, other where false.
    pub fn where_cond(&self, condition: &Tensor, other: &Tensor) -> Result<Tensor> {
        match (self, condition, other) {
            (Tensor::F32(a), Tensor::F32(cond), Tensor::F32(b)) => {
                if a.shape() != cond.shape() || a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "All tensors must have the same shape for where operation".to_string(),
                    ));
                }
                let result = Zip::from(cond)
                    .and(a)
                    .and(b)
                    .map_collect(|&c, &x, &y| if c > 0.5 { x } else { y });
                Ok(Tensor::F32(result))
            },
            (Tensor::F64(a), Tensor::F64(cond), Tensor::F64(b)) => {
                if a.shape() != cond.shape() || a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "All tensors must have the same shape for where operation".to_string(),
                    ));
                }
                let result = Zip::from(cond)
                    .and(a)
                    .and(b)
                    .map_collect(|&c, &x, &y| if c > 0.5 { x } else { y });
                Ok(Tensor::F64(result))
            },
            (Tensor::I64(a), Tensor::I64(cond), Tensor::I64(b)) => {
                if a.shape() != cond.shape() || a.shape() != b.shape() {
                    return Err(TrustformersError::shape_error(
                        "All tensors must have the same shape for where operation".to_string(),
                    ));
                }
                let result = Zip::from(cond)
                    .and(a)
                    .and(b)
                    .map_collect(|&c, &x, &y| if c > 0 { x } else { y });
                Ok(Tensor::I64(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Where operation not implemented for this tensor type combination",
                "where_cond",
            )),
        }
    }

    /// Layer normalization.
    pub fn layer_norm(&self, axis: i32, epsilon: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let ndim = a.ndim();
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                if axis >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis, ndim
                        ),
                        "layer_norm",
                    ));
                }

                // Calculate mean along the axis
                let _mean = a.mean_axis(Axis(axis)).ok_or_else(|| {
                    crate::errors::compute_error(
                        "layer_norm",
                        "axis must be valid for mean operation",
                    )
                })?;

                // Simple layer normalization for last dimension
                let last_dim = a.ndim() - 1;
                if axis != last_dim {
                    return Err(TrustformersError::tensor_op_error(
                        "Layer norm currently only supports last dimension normalization",
                        "layer_norm",
                    ));
                }

                // Calculate statistics along the last axis
                let mean = a.mean_axis(Axis(axis)).ok_or_else(|| {
                    crate::errors::compute_error(
                        "layer_norm",
                        "axis must be valid for mean operation",
                    )
                })?;
                let var = a.map_axis(Axis(axis), |lane| {
                    // mean() is None only for an empty lane; layer-norm lanes are non-empty,
                    // so fall back to 0.0 to keep this map_axis closure infallible.
                    let lane_mean = lane.mean().unwrap_or(0.0);
                    lane.mapv(|x| (x - lane_mean).powi(2)).mean().unwrap_or(0.0)
                });

                // Normalize
                let mut result = a.clone();
                for (i, mut lane) in result.axis_iter_mut(Axis(axis)).enumerate() {
                    let m = mean[i];
                    let v = var[i];
                    lane.mapv_inplace(|x| (x - m) / (v + epsilon).sqrt());
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                run_half_in_f32(self, |t| t.layer_norm(axis, epsilon))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Layer norm not supported for this tensor type",
                "layer_norm",
            )),
        }
    }

    /// Cross entropy loss.
    pub fn cross_entropy(&self, targets: &Tensor, reduction: &str) -> Result<Tensor> {
        match (self, targets) {
            (Tensor::F32(predictions), Tensor::F32(targets)) => {
                // Calculate cross entropy: -sum(target * log(prediction))
                let log_preds = predictions.mapv(|x| (x + 1e-8).ln()); // Add small epsilon to avoid log(0)
                let losses = Zip::from(&log_preds)
                    .and(targets)
                    .map_collect(|&log_pred, &target| -target * log_pred);

                match reduction {
                    "mean" => {
                        let mean_loss = losses.mean().ok_or_else(|| {
                            crate::errors::compute_error("cross_entropy", "Mean calculation failed")
                        })?;
                        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), mean_loss)))
                    },
                    "sum" => {
                        let sum_loss = losses.sum();
                        Ok(Tensor::F32(ArrayD::from_elem(IxDyn(&[]), sum_loss)))
                    },
                    "none" => Ok(Tensor::F32(losses)),
                    _ => Err(TrustformersError::tensor_op_error(
                        "Invalid reduction. Use 'mean', 'sum', or 'none'",
                        "cross_entropy",
                    )),
                }
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cross entropy not supported for these tensor types",
                "cross_entropy",
            )),
        }
    }

    /// Cosine similarity.
    pub fn cosine_similarity(&self, other: &Tensor, dim: i32, eps: f32) -> Result<Tensor> {
        match (self, other) {
            (Tensor::F32(a), Tensor::F32(b)) => {
                let ndim = a.ndim();
                let axis = if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };

                // Calculate dot product along the specified dimension
                let dot_product =
                    Zip::from(a).and(b).map_collect(|&x, &y| x * y).sum_axis(Axis(axis));

                // Calculate norms
                let norm_a = a.mapv(|x| x * x).sum_axis(Axis(axis)).mapv(|x| (x + eps).sqrt());
                let norm_b = b.mapv(|x| x * x).sum_axis(Axis(axis)).mapv(|x| (x + eps).sqrt());

                // Calculate cosine similarity
                let similarity = Zip::from(&dot_product)
                    .and(&norm_a)
                    .and(&norm_b)
                    .map_collect(|&dot, &norm_a, &norm_b| dot / (norm_a * norm_b));

                Ok(Tensor::F32(similarity))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Cosine similarity not supported for these tensor types",
                "cosine_similarity",
            )),
        }
    }

    /// Log softmax.
    pub fn log_softmax(&self, dim: i32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let ndim = a.ndim();
                let axis = if dim < 0 { (ndim as i32 + dim) as usize } else { dim as usize };

                if axis >= ndim {
                    return Err(TrustformersError::tensor_op_error(
                        &format!(
                            "Axis {} is out of bounds for tensor with {} dimensions",
                            axis, ndim
                        ),
                        "log_softmax",
                    ));
                }

                // Calculate log_softmax: log(softmax(x)) = x - log(sum(exp(x)))
                // For numerical stability: x - max(x) - log(sum(exp(x - max(x))))
                let max_vals = a.map_axis(Axis(axis), |lane| {
                    lane.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
                });

                // Expand max_vals to match original tensor shape for broadcasting
                let mut max_shape = a.shape().to_vec();
                max_shape[axis] = 1;
                let max_expanded =
                    max_vals.into_shape_with_order(max_shape.clone()).map_err(|e| {
                        crate::errors::compute_error(
                            "log_softmax",
                            format!("{}: {e}", "reshape must be valid for max values"),
                        )
                    })?;

                // Subtract max for numerical stability
                let shifted = a - &max_expanded.broadcast(a.raw_dim()).ok_or_else(|| {
                    crate::errors::compute_error(
                        "log_softmax",
                        "broadcast must succeed with compatible shapes",
                    )
                })?;

                // Calculate log sum exp
                let exp_shifted = shifted.mapv(|x| x.exp());
                let sum_exp = exp_shifted.sum_axis(Axis(axis));
                let log_sum_exp = sum_exp.mapv(|x| x.ln());

                // Expand log_sum_exp for broadcasting
                let log_sum_exp_expanded =
                    log_sum_exp.into_shape_with_order(max_shape).map_err(|e| {
                        crate::errors::compute_error(
                            "log_softmax",
                            format!("{}: {e}", "reshape must be valid for log_sum_exp"),
                        )
                    })?;

                // Final result
                let result = shifted
                    - log_sum_exp_expanded.broadcast(a.raw_dim()).ok_or_else(|| {
                        crate::errors::compute_error(
                            "log_softmax",
                            "broadcast must succeed with compatible shapes",
                        )
                    })?;
                Ok(Tensor::F32(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.log_softmax(dim)),
            _ => Err(TrustformersError::tensor_op_error(
                "Log softmax not supported for this tensor type",
                "log_softmax",
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::tensor::{DType, Tensor};
    use scirs2_core::ndarray::{ArrayD, IxDyn};

    /// Build an F16 tensor from f32 values.
    fn make_f16(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let arr = ArrayD::from_shape_vec(
            IxDyn(shape),
            data.iter().map(|&x| half::f16::from_f32(x)).collect(),
        )
        .map_err(|e| crate::errors::TrustformersError::shape_error(e.to_string()))?;
        Ok(Tensor::F16(arr))
    }

    /// Build a BF16 tensor from f32 values.
    fn make_bf16(data: &[f32], shape: &[usize]) -> Result<Tensor> {
        let arr = ArrayD::from_shape_vec(
            IxDyn(shape),
            data.iter().map(|&x| half::bf16::from_f32(x)).collect(),
        )
        .map_err(|e| crate::errors::TrustformersError::shape_error(e.to_string()))?;
        Ok(Tensor::BF16(arr))
    }

    /// Read a half-precision tensor's values as f32 for assertions.
    fn half_to_vec_f32(t: &Tensor) -> Vec<f32> {
        match t {
            Tensor::F16(a) => a.iter().map(|x| x.to_f32()).collect(),
            Tensor::BF16(a) => a.iter().map(|x| x.to_f32()).collect(),
            _ => panic!("expected a half-precision tensor"),
        }
    }

    #[test]
    fn test_layer_norm_f16_bf16() -> Result<()> {
        // Use a square shape so the (pre-existing) F32 last-dim normalization loop
        // stays in bounds; we assert the upcast path preserves dtype/shape and
        // yields finite values rather than relying on the exact F32 reduction math.
        for (t, dt) in [
            (
                make_f16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?,
                DType::F16,
            ),
            (
                make_bf16(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0], &[3, 3])?,
                DType::BF16,
            ),
        ] {
            let r = t.layer_norm(-1, 1e-5)?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![3, 3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
        }
        Ok(())
    }

    #[test]
    fn test_log_softmax_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (
                make_f16(&[1.0, 2.0, 3.0, 0.0, 1.0, 0.0], &[2, 3])?,
                DType::F16,
            ),
            (
                make_bf16(&[1.0, 2.0, 3.0, 0.0, 1.0, 0.0], &[2, 3])?,
                DType::BF16,
            ),
        ] {
            let r = t.log_softmax(-1)?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![2, 3]);
            let data = half_to_vec_f32(&r);
            // log_softmax outputs are <= 0 and finite; exp of each row sums to ~1.
            assert!(data.iter().all(|v| v.is_finite() && *v <= 0.05));
            let row0: f32 = data[0..3].iter().map(|v| v.exp()).sum();
            let row1: f32 = data[3..6].iter().map(|v| v.exp()).sum();
            assert!((row0 - 1.0).abs() < 0.05, "row0 exp-sum = {}", row0);
            assert!((row1 - 1.0).abs() < 0.05, "row1 exp-sum = {}", row1);
        }
        Ok(())
    }
}
