//! Tensor activation functions.
//!
//! This module contains activation functions commonly used in neural networks.
//!
//! # Performance
//!
//! This module uses scirs2-core's SIMD-optimized activation functions for larger tensors:
//! - `simd_gelu` - GELU activation (used in BERT, GPT, etc.)
//! - `simd_swish` - Swish/SiLU activation (used in EfficientNet, GPT-NeoX)
//! - `simd_sigmoid` - Sigmoid activation
//! - `simd_tanh` - Tanh activation
//!
//! For tensors with <256 elements, uses scalar operations to avoid SIMD overhead.

#![allow(deprecated)] // Using rand legacy API, will migrate to scirs2_core

use super::Tensor;
use crate::errors::{Result, TrustformersError};
use scirs2_core::ndarray::{Axis, IxDyn};
use scirs2_core::simd_ops::SimdUnifiedOps;

/// Minimum tensor size to use SIMD operations (avoids overhead for small tensors)
const MIN_SIZE_FOR_SIMD: usize = 256;

/// Upcast a half-precision (F16/BF16) tensor to F32, run `op`, then downcast the
/// F32 result back to the original half-precision dtype.
///
/// Half-precision floats lack the precision and dynamic range required for stable
/// activation math (e.g. `exp` in softmax), so we compute in F32 and round the
/// result back to F16/BF16 so the output dtype matches the input dtype.
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
    /// ReLU activation function.
    ///
    /// # Returns
    ///
    /// A tensor with ReLU applied element-wise.
    pub fn relu(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f32::simd_relu(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result))
                } else {
                    Ok(Tensor::F32(a.mapv(|x| x.max(0.0))))
                }
            },
            Tensor::F64(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f64::simd_relu(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result))
                } else {
                    Ok(Tensor::F64(a.mapv(|x| x.max(0.0))))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.relu()),
            _ => Err(TrustformersError::tensor_op_error(
                "ReLU not supported for this tensor type",
                "relu",
            )),
        }
    }

    /// Sigmoid activation function.
    ///
    /// # Performance
    ///
    /// Uses scirs2-core's SIMD-accelerated sigmoid for tensors with ≥256 elements.
    ///
    /// # Returns
    ///
    /// A tensor with sigmoid applied element-wise.
    pub fn sigmoid(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated sigmoid for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f32::simd_sigmoid(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result))
                } else {
                    // Numerically stable sigmoid implementation for small tensors
                    let result = a.mapv(|x| {
                        if x >= 0.0 {
                            let exp_neg_x = (-x).exp();
                            1.0 / (1.0 + exp_neg_x)
                        } else {
                            let exp_x = x.exp();
                            exp_x / (1.0 + exp_x)
                        }
                    });
                    Ok(Tensor::F32(result))
                }
            },
            Tensor::F64(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated sigmoid for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f64::simd_sigmoid(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result))
                } else {
                    // Numerically stable sigmoid implementation for small tensors
                    let result = a.mapv(|x| {
                        if x >= 0.0 {
                            let exp_neg_x = (-x).exp();
                            1.0 / (1.0 + exp_neg_x)
                        } else {
                            let exp_x = x.exp();
                            exp_x / (1.0 + exp_x)
                        }
                    });
                    Ok(Tensor::F64(result))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.sigmoid()),
            _ => Err(TrustformersError::tensor_op_error(
                "Sigmoid not supported for this tensor type",
                "sigmoid",
            )),
        }
    }

    /// Tanh activation function.
    ///
    /// # Performance
    ///
    /// Uses scirs2-core's SIMD-accelerated tanh for tensors with ≥256 elements.
    ///
    /// # Returns
    ///
    /// A tensor with tanh applied element-wise.
    pub fn tanh(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated tanh for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f32::simd_tanh(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result))
                } else {
                    let result = a.mapv(|x| x.tanh());
                    Ok(Tensor::F32(result))
                }
            },
            Tensor::F64(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated tanh for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f64::simd_tanh(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result))
                } else {
                    let result = a.mapv(|x| x.tanh());
                    Ok(Tensor::F64(result))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.tanh()),
            _ => Err(TrustformersError::tensor_op_error(
                "Tanh not supported for this tensor type",
                "tanh",
            )),
        }
    }

    /// Softmax activation function.
    ///
    /// # Arguments
    ///
    /// * `axis` - The axis along which to apply softmax
    ///
    /// # Returns
    ///
    /// A tensor with softmax applied along the specified axis.
    pub fn softmax(&self, axis: i32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let ndim = a.ndim();
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                if axis >= ndim {
                    return Err(TrustformersError::shape_error(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis, ndim
                    )));
                }

                // Single owned buffer, then one in-place pass per lane.
                //
                // The previous implementation materialised seven full-size arrays
                // (input copy, max, shifted, shifted copy, exp, exp copy, result,
                // result copy); three of those copies were provably pure memcpy
                // because the preceding expression already produced an owned
                // standard-layout array. For [1,32,2048,2048] attention scores
                // that is gigabytes of needless traffic per softmax.
                let mut result = a.as_standard_layout().into_owned();

                for mut lane in result.lanes_mut(Axis(axis)) {
                    let max = lane.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                    // An all-`-inf` lane would make `x - max` NaN; shift by 0 so the
                    // lane degrades to all-zero weights instead of NaN.
                    let max = if max.is_finite() { max } else { 0.0 };

                    let mut sum = 0.0f32;
                    for value in lane.iter_mut() {
                        let exponential = (*value - max).exp();
                        *value = exponential;
                        sum += exponential;
                    }

                    // Protect against division by (sub)normal sums.
                    let inverse_sum =
                        if sum <= f32::MIN_POSITIVE { 1.0 / f32::MIN_POSITIVE } else { 1.0 / sum };
                    for value in lane.iter_mut() {
                        *value *= inverse_sum;
                    }
                }

                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let ndim = a.ndim();
                let axis = if axis < 0 { (ndim as i32 + axis) as usize } else { axis as usize };

                if axis >= ndim {
                    return Err(TrustformersError::shape_error(format!(
                        "Axis {} is out of bounds for tensor with {} dimensions",
                        axis, ndim
                    )));
                }

                // Single owned buffer, then one in-place pass per lane (see the
                // F32 arm for why the seven-buffer version was removed).
                let mut result = a.as_standard_layout().into_owned();

                for mut lane in result.lanes_mut(Axis(axis)) {
                    let max = lane.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x));
                    let max = if max.is_finite() { max } else { 0.0 };

                    let mut sum = 0.0f64;
                    for value in lane.iter_mut() {
                        let exponential = (*value - max).exp();
                        *value = exponential;
                        sum += exponential;
                    }

                    let inverse_sum =
                        if sum <= f64::MIN_POSITIVE { 1.0 / f64::MIN_POSITIVE } else { 1.0 / sum };
                    for value in lane.iter_mut() {
                        *value *= inverse_sum;
                    }
                }

                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.softmax(axis)),
            _ => Err(TrustformersError::tensor_op_error(
                "Softmax not supported for this tensor type",
                "softmax",
            )),
        }
    }

    /// Dropout operation.
    ///
    /// # Arguments
    ///
    /// * `dropout_prob` - Probability of dropping each element
    ///
    /// # Returns
    ///
    /// A tensor with dropout applied.
    pub fn dropout(&self, dropout_prob: f32) -> Result<Tensor> {
        use scirs2_core::random::*;

        if !(0.0..=1.0).contains(&dropout_prob) {
            return Err(TrustformersError::tensor_op_error(
                "Dropout probability must be between 0 and 1",
                "dropout",
            ));
        }

        if dropout_prob == 0.0 {
            return Ok(self.clone());
        }

        match self {
            Tensor::F32(a) => {
                let mut rng = thread_rng();
                let scale = 1.0 / (1.0 - dropout_prob);
                let result =
                    a.mapv(
                        |x| {
                            if rng.random::<f32>() < dropout_prob {
                                0.0
                            } else {
                                x * scale
                            }
                        },
                    );
                Ok(Tensor::F32(result))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Dropout not supported for this tensor type",
                "dropout",
            )),
        }
    }

    /// GELU (Gaussian Error Linear Unit) activation function.
    ///
    /// # Performance
    ///
    /// Uses scirs2-core's SIMD-accelerated GELU for tensors with ≥256 elements.
    /// GELU is widely used in Transformer models (BERT, GPT, etc.).
    ///
    /// # Returns
    ///
    /// A tensor with GELU applied element-wise.
    pub fn gelu(&self) -> Result<Tensor> {
        match self {
            // Metal GPU path - stays on GPU!
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(metal_data) => {
                use crate::gpu_ops::metal::get_metal_backend;
                use crate::tensor::MetalTensorData;

                let backend = get_metal_backend()?;
                let size = metal_data.shape.iter().product();

                let output_buffer_id = backend.gelu_gpu_to_gpu(&metal_data.buffer_id, size)?;

                Ok(Tensor::Metal(MetalTensorData {
                    buffer_id: output_buffer_id,
                    shape: metal_data.shape.clone(),
                    dtype: metal_data.dtype,
                }))
            },
            Tensor::F32(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated GELU for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f32::simd_gelu(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result))
                } else {
                    // Scalar path for small tensors
                    let result = a.mapv(|x| {
                        0.5 * x * (1.0 + (0.7978845608 * (x + 0.044715 * x.powi(3))).tanh())
                    });
                    Ok(Tensor::F32(result))
                }
            },
            Tensor::F64(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated GELU for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f64::simd_gelu(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result))
                } else {
                    // Scalar path for small tensors
                    let result = a.mapv(|x| {
                        0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x.powi(3))).tanh())
                    });
                    Ok(Tensor::F64(result))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.gelu()),
            _ => Err(TrustformersError::tensor_op_error(
                "GELU not supported for this tensor type",
                "gelu",
            )),
        }
    }

    /// Leaky ReLU activation function.
    ///
    /// # Arguments
    ///
    /// * `negative_slope` - The slope for negative values (default: 0.01)
    ///
    /// # Returns
    ///
    /// A tensor with Leaky ReLU applied element-wise.
    pub fn leaky_relu(&self, negative_slope: f32) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| if x > 0.0 { x } else { negative_slope * x });
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let negative_slope = negative_slope as f64;
                let result = a.mapv(|x| if x > 0.0 { x } else { negative_slope * x });
                Ok(Tensor::F64(result))
            },
            Tensor::F16(_) | Tensor::BF16(_) => {
                run_half_in_f32(self, |t| t.leaky_relu(negative_slope))
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Leaky ReLU not supported for this tensor type",
                "leaky_relu",
            )),
        }
    }

    /// SiLU (Sigmoid-Linear Unit) activation function.
    ///
    /// Also known as Swish activation: f(x) = x * sigmoid(x)
    ///
    /// # Performance
    ///
    /// Uses scirs2-core's SIMD-accelerated Swish for tensors with ≥256 elements.
    /// SiLU/Swish is used in EfficientNet, GPT-NeoX, and many modern architectures.
    ///
    /// # Returns
    ///
    /// A tensor with SiLU applied element-wise.
    pub fn silu(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated Swish for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f32::simd_swish(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F32(result))
                } else {
                    // Scalar path for small tensors
                    let result = a.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));
                    Ok(Tensor::F32(result))
                }
            },
            Tensor::F64(a) => {
                let size = a.len();
                if size >= MIN_SIZE_FOR_SIMD {
                    // Use SIMD-accelerated Swish for larger tensors
                    let shape = a.shape().to_vec();
                    let flat = a.as_standard_layout();
                    let flat_view = flat
                        .view()
                        .into_shape_with_order(size)
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    let result_1d = f64::simd_swish(&flat_view);
                    let result = result_1d
                        .into_shape_with_order(IxDyn(&shape))
                        .map_err(|e| TrustformersError::shape_error(e.to_string()))?;
                    Ok(Tensor::F64(result))
                } else {
                    // Scalar path for small tensors
                    let result = a.mapv(|x| x * (1.0 / (1.0 + (-x).exp())));
                    Ok(Tensor::F64(result))
                }
            },
            Tensor::F16(_) | Tensor::BF16(_) => run_half_in_f32(self, |t| t.silu()),
            _ => Err(TrustformersError::tensor_op_error(
                "SiLU not supported for this tensor type",
                "silu",
            )),
        }
    }

    /// Swish activation function (alias for SiLU).
    ///
    /// Swish(x) = x * sigmoid(x) = SiLU(x)
    pub fn swish(&self) -> Result<Tensor> {
        self.silu()
    }
}

#[cfg(test)]
mod tests {
    use crate::errors::Result;
    use crate::tensor::Tensor;

    #[test]
    fn test_relu_positive() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let r = t.relu()?;
        let data = r.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_relu_negative() -> Result<()> {
        let t = Tensor::from_data(vec![-1.0, -2.0, -3.0], &[3])?;
        let r = t.relu()?;
        let data = r.data()?;
        for val in &data {
            assert!(val.abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_relu_mixed() -> Result<()> {
        let t = Tensor::from_data(vec![-2.0, 0.0, 3.0], &[3])?;
        let r = t.relu()?;
        let data = r.data()?;
        assert!(data[0].abs() < 1e-6);
        assert!(data[1].abs() < 1e-6);
        assert!((data[2] - 3.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_sigmoid_zero() -> Result<()> {
        let t = Tensor::from_data(vec![0.0], &[1])?;
        let r = t.sigmoid()?;
        let data = r.data()?;
        assert!((data[0] - 0.5).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_sigmoid_large_positive() -> Result<()> {
        let t = Tensor::from_data(vec![10.0], &[1])?;
        let r = t.sigmoid()?;
        let data = r.data()?;
        assert!((data[0] - 1.0).abs() < 1e-3);
        Ok(())
    }

    #[test]
    fn test_sigmoid_large_negative() -> Result<()> {
        let t = Tensor::from_data(vec![-10.0], &[1])?;
        let r = t.sigmoid()?;
        let data = r.data()?;
        assert!(data[0] < 1e-3);
        Ok(())
    }

    #[test]
    fn test_sigmoid_range() -> Result<()> {
        let t = Tensor::from_data(vec![-5.0, -1.0, 0.0, 1.0, 5.0], &[5])?;
        let r = t.sigmoid()?;
        let data = r.data()?;
        for val in &data {
            assert!(*val >= 0.0 && *val <= 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_tanh_zero() -> Result<()> {
        let t = Tensor::from_data(vec![0.0], &[1])?;
        let r = t.tanh()?;
        let data = r.data()?;
        assert!(data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_tanh_range() -> Result<()> {
        let t = Tensor::from_data(vec![-10.0, -1.0, 0.0, 1.0, 10.0], &[5])?;
        let r = t.tanh()?;
        let data = r.data()?;
        for val in &data {
            assert!(*val >= -1.0 && *val <= 1.0);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_sums_to_one() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let r = t.softmax(0)?;
        let data = r.data()?;
        let sum: f32 = data.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_softmax_positive() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let r = t.softmax(0)?;
        let data = r.data()?;
        for val in &data {
            assert!(*val > 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_ordering() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let r = t.softmax(0)?;
        let data = r.data()?;
        assert!(data[0] < data[1]);
        assert!(data[1] < data[2]);
        Ok(())
    }

    #[test]
    fn test_gelu() -> Result<()> {
        let t = Tensor::from_data(vec![0.0, 1.0, -1.0], &[3])?;
        let r = t.gelu()?;
        let data = r.data()?;
        // GELU(0) = 0
        assert!(data[0].abs() < 1e-4);
        // GELU(1) ~ 0.8413
        assert!((data[1] - 0.8413).abs() < 0.02);
        // GELU(-1) ~ -0.1587
        assert!((data[2] - (-0.1587)).abs() < 0.02);
        Ok(())
    }

    #[test]
    fn test_leaky_relu_positive() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0], &[2])?;
        let r = t.leaky_relu(0.01)?;
        let data = r.data()?;
        assert!((data[0] - 1.0).abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_leaky_relu_negative() -> Result<()> {
        let t = Tensor::from_data(vec![-1.0, -2.0], &[2])?;
        let r = t.leaky_relu(0.1)?;
        let data = r.data()?;
        assert!((data[0] - (-0.1)).abs() < 1e-5);
        assert!((data[1] - (-0.2)).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_silu_zero() -> Result<()> {
        let t = Tensor::from_data(vec![0.0], &[1])?;
        let r = t.silu()?;
        let data = r.data()?;
        // SiLU(0) = 0 * sigmoid(0) = 0 * 0.5 = 0
        assert!(data[0].abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_silu_positive() -> Result<()> {
        let t = Tensor::from_data(vec![2.0], &[1])?;
        let r = t.silu()?;
        let data = r.data()?;
        // SiLU(2) = 2 * sigmoid(2) ~ 2 * 0.88 ~ 1.76
        assert!(data[0] > 1.5 && data[0] < 2.0);
        Ok(())
    }

    #[test]
    fn test_swish_is_silu() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, -1.0], &[3])?;
        let silu = t.silu()?;
        let swish = t.swish()?;
        let silu_data = silu.data()?;
        let swish_data = swish.data()?;
        for i in 0..3 {
            assert!((silu_data[i] - swish_data[i]).abs() < 1e-6);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_2d() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0, 1.0, 2.0, 3.0], &[2, 3])?;
        let r = t.softmax(-1)?;
        assert_eq!(r.shape(), vec![2, 3]);
        Ok(())
    }

    #[test]
    fn test_relu_2d() -> Result<()> {
        let t = Tensor::from_data(vec![-1.0, 2.0, -3.0, 4.0], &[2, 2])?;
        let r = t.relu()?;
        let data = r.data()?;
        assert!(data[0].abs() < 1e-6);
        assert!((data[1] - 2.0).abs() < 1e-6);
        assert!(data[2].abs() < 1e-6);
        assert!((data[3] - 4.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn test_dropout_zero_prob() -> Result<()> {
        let t = Tensor::from_data(vec![1.0, 2.0, 3.0], &[3])?;
        let r = t.dropout(0.0)?;
        let data = r.data()?;
        assert!((data[0] - 1.0).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        Ok(())
    }

    #[test]
    fn test_gelu_2d() -> Result<()> {
        let t = Tensor::from_data(vec![0.0, 1.0, -1.0, 2.0], &[2, 2])?;
        let r = t.gelu()?;
        assert_eq!(r.shape(), vec![2, 2]);
        Ok(())
    }

    // ---- Half-precision (F16 / BF16) upcast-path tests ----

    use crate::tensor::DType;
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
    fn test_relu_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[-1.0, 0.0, 2.5], &[3])?, DType::F16),
            (make_bf16(&[-1.0, 0.0, 2.5], &[3])?, DType::BF16),
        ] {
            let r = t.relu()?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
            assert!(data[0].abs() < 0.05);
            assert!(data[1].abs() < 0.05);
            assert!((data[2] - 2.5).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_sigmoid_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[0.0, 4.0, -4.0], &[3])?, DType::F16),
            (make_bf16(&[0.0, 4.0, -4.0], &[3])?, DType::BF16),
        ] {
            let r = t.sigmoid()?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite() && *v >= 0.0 && *v <= 1.0));
            assert!((data[0] - 0.5).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_tanh_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[0.0, 2.0, -2.0], &[3])?, DType::F16),
            (make_bf16(&[0.0, 2.0, -2.0], &[3])?, DType::BF16),
        ] {
            let r = t.tanh()?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite() && *v >= -1.0 && *v <= 1.0));
            assert!(data[0].abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_softmax_f16_bf16_rows_sum_to_one() -> Result<()> {
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
            let r = t.softmax(-1)?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![2, 3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
            // Each row of 3 elements should sum to approximately 1.0.
            let row0: f32 = data[0..3].iter().sum();
            let row1: f32 = data[3..6].iter().sum();
            assert!((row0 - 1.0).abs() < 0.05, "row0 sum = {}", row0);
            assert!((row1 - 1.0).abs() < 0.05, "row1 sum = {}", row1);
        }
        Ok(())
    }

    #[test]
    fn test_gelu_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[0.0, 1.0, -1.0], &[3])?, DType::F16),
            (make_bf16(&[0.0, 1.0, -1.0], &[3])?, DType::BF16),
        ] {
            let r = t.gelu()?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![3]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
            assert!(data[0].abs() < 0.05);
            assert!((data[1] - 0.8413).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_leaky_relu_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[-1.0, 2.0], &[2])?, DType::F16),
            (make_bf16(&[-1.0, 2.0], &[2])?, DType::BF16),
        ] {
            let r = t.leaky_relu(0.1)?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![2]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
            assert!((data[0] - (-0.1)).abs() < 0.05);
            assert!((data[1] - 2.0).abs() < 0.05);
        }
        Ok(())
    }

    #[test]
    fn test_silu_f16_bf16() -> Result<()> {
        for (t, dt) in [
            (make_f16(&[0.0, 2.0], &[2])?, DType::F16),
            (make_bf16(&[0.0, 2.0], &[2])?, DType::BF16),
        ] {
            let r = t.silu()?;
            assert_eq!(r.dtype(), dt);
            assert_eq!(r.shape(), vec![2]);
            let data = half_to_vec_f32(&r);
            assert!(data.iter().all(|v| v.is_finite()));
            assert!(data[0].abs() < 0.05);
            assert!(data[1] > 1.5 && data[1] < 2.0);
        }
        Ok(())
    }
}
