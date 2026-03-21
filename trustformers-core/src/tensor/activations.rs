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

impl Tensor {
    /// ReLU activation function.
    ///
    /// # Returns
    ///
    /// A tensor with ReLU applied element-wise.
    pub fn relu(&self) -> Result<Tensor> {
        match self {
            Tensor::F32(a) => {
                let result = a.mapv(|x| x.max(0.0));
                Ok(Tensor::F32(result))
            },
            Tensor::F64(a) => {
                let result = a.mapv(|x| x.max(0.0));
                Ok(Tensor::F64(result))
            },
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

                // Ensure contiguous input layout
                let a_contiguous = a.as_standard_layout().to_owned();

                // For numerical stability, subtract max before exp
                let max_vals = a_contiguous.map_axis(Axis(axis), |lane| {
                    lane.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x))
                });

                // Ensure contiguous max_vals and compute shifted values
                let max_vals_contiguous = max_vals.as_standard_layout().to_owned();
                let shifted = &a_contiguous - &max_vals_contiguous.insert_axis(Axis(axis));
                let shifted_contiguous = shifted.as_standard_layout().to_owned();

                // Compute exp and sum with contiguous layout
                let exp_vals = shifted_contiguous.mapv(|x| x.exp());
                let exp_vals_contiguous = exp_vals.as_standard_layout().to_owned();
                let sum_exp = exp_vals_contiguous.sum_axis(Axis(axis));
                let sum_exp_contiguous = sum_exp.as_standard_layout().to_owned();

                // Protect against division by very small numbers
                let protected_sum = sum_exp_contiguous.mapv(|x| {
                    if x <= f32::MIN_POSITIVE {
                        f32::MIN_POSITIVE
                    } else {
                        x
                    }
                });

                // Final result with contiguous layout
                let result = exp_vals_contiguous / protected_sum.insert_axis(Axis(axis));
                let result_contiguous = result.as_standard_layout().to_owned();
                Ok(Tensor::F32(result_contiguous))
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

                // Ensure contiguous input layout
                let a_contiguous = a.as_standard_layout().to_owned();

                let max_vals = a_contiguous.map_axis(Axis(axis), |lane| {
                    lane.iter().fold(f64::NEG_INFINITY, |acc, &x| acc.max(x))
                });

                // Ensure contiguous layouts throughout computation
                let max_vals_contiguous = max_vals.as_standard_layout().to_owned();
                let shifted = &a_contiguous - &max_vals_contiguous.insert_axis(Axis(axis));
                let shifted_contiguous = shifted.as_standard_layout().to_owned();

                let exp_vals = shifted_contiguous.mapv(|x| x.exp());
                let exp_vals_contiguous = exp_vals.as_standard_layout().to_owned();
                let sum_exp = exp_vals_contiguous.sum_axis(Axis(axis));
                let sum_exp_contiguous = sum_exp.as_standard_layout().to_owned();

                // Protect against division by very small numbers
                let protected_sum = sum_exp_contiguous.mapv(|x| {
                    if x <= f64::MIN_POSITIVE {
                        f64::MIN_POSITIVE
                    } else {
                        x
                    }
                });

                let result = exp_vals_contiguous / protected_sum.insert_axis(Axis(axis));
                let result_contiguous = result.as_standard_layout().to_owned();
                Ok(Tensor::F64(result_contiguous))
            },
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
