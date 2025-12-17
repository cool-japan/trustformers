//! SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs
//!
//! SmoothQuant enables 8-bit weight and activation quantization (W8A8) by smoothing
//! activation outliers through mathematically equivalent transformations.

#![allow(unused_variables)] // SmoothQuant implementation

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SmoothQuant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothQuantConfig {
    /// Alpha parameter for balancing weight and activation quantization difficulty
    /// 0.0 = all difficulty on weights, 1.0 = all difficulty on activations
    pub alpha: f32,

    /// Number of calibration samples
    pub num_calibration_samples: usize,

    /// Percentile for activation range calculation
    pub activation_percentile: f32,

    /// Whether to use per-channel quantization
    pub per_channel: bool,

    /// Migration strength for outlier smoothing
    pub migration_strength: f32,

    /// Whether to quantize activations
    pub quantize_activations: bool,
}

impl Default for SmoothQuantConfig {
    fn default() -> Self {
        Self {
            alpha: 0.5, // Balanced migration
            num_calibration_samples: 512,
            activation_percentile: 99.99,
            per_channel: true,
            migration_strength: 0.8,
            quantize_activations: true,
        }
    }
}

/// SmoothQuant quantizer
pub struct SmoothQuantizer {
    config: SmoothQuantConfig,
    /// Smoothing scales for each layer
    smoothing_scales: HashMap<String, Tensor>,
    /// Activation statistics from calibration
    activation_stats: HashMap<String, ActivationStats>,
}

/// Activation statistics for calibration
#[derive(Debug, Clone)]
struct ActivationStats {
    /// Maximum absolute values per channel
    #[allow(dead_code)]
    max_vals: Vec<f32>,
    /// Minimum absolute values per channel
    _min_vals: Vec<f32>,
    /// Percentile values per channel
    percentile_vals: Vec<f32>,
    /// Number of samples
    _num_samples: usize,
}

impl SmoothQuantizer {
    /// Create a new SmoothQuant quantizer
    pub fn new(config: SmoothQuantConfig) -> Self {
        Self {
            config,
            smoothing_scales: HashMap::new(),
            activation_stats: HashMap::new(),
        }
    }

    /// Calibrate the quantizer with sample data
    pub fn calibrate(
        &mut self,
        layer_name: &str,
        activations: &[Tensor],
        weights: &Tensor,
    ) -> Result<()> {
        if activations.is_empty() {
            return Err(TrustformersError::invalid_argument(
                "No calibration data provided".to_string(),
            ));
        }

        // Collect activation statistics
        let stats = self.collect_activation_stats(activations)?;
        self.activation_stats.insert(layer_name.to_string(), stats.clone());

        // Calculate smoothing scales
        let scales = self.calculate_smoothing_scales(&stats, weights)?;
        self.smoothing_scales.insert(layer_name.to_string(), scales);

        Ok(())
    }

    /// Collect activation statistics from calibration data
    fn collect_activation_stats(&self, activations: &[Tensor]) -> Result<ActivationStats> {
        let first_shape = activations[0].shape();
        let num_channels = first_shape[first_shape.len() - 1];

        let mut max_vals = vec![0.0f32; num_channels];
        let mut min_vals = vec![f32::MAX; num_channels];
        let mut all_values: Vec<Vec<f32>> = vec![Vec::new(); num_channels];

        // Collect values per channel
        for activation in activations {
            match activation {
                Tensor::F32(data) => {
                    let values = data.as_slice().ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "Failed to get tensor data",
                            "collect_activation_stats",
                        )
                    })?;

                    // Process each channel
                    for (idx, val) in values.iter().enumerate() {
                        let channel = idx % num_channels;
                        let abs_val = val.abs();
                        max_vals[channel] = max_vals[channel].max(abs_val);
                        min_vals[channel] = min_vals[channel].min(abs_val);
                        all_values[channel].push(abs_val);
                    }
                },
                Tensor::F64(data) => {
                    let values = data.as_slice().ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "Failed to get tensor data",
                            "collect_activation_stats",
                        )
                    })?;

                    // Process each channel (convert to f32)
                    for (idx, val) in values.iter().enumerate() {
                        let channel = idx % num_channels;
                        let abs_val = (*val as f32).abs();
                        max_vals[channel] = max_vals[channel].max(abs_val);
                        min_vals[channel] = min_vals[channel].min(abs_val);
                        all_values[channel].push(abs_val);
                    }
                },
                Tensor::I64(data) => {
                    let values = data.as_slice().ok_or_else(|| {
                        TrustformersError::tensor_op_error(
                            "Failed to get tensor data",
                            "collect_activation_stats",
                        )
                    })?;

                    // Process each channel (convert to f32)
                    for (idx, val) in values.iter().enumerate() {
                        let channel = idx % num_channels;
                        let abs_val = (*val as f32).abs();
                        max_vals[channel] = max_vals[channel].max(abs_val);
                        min_vals[channel] = min_vals[channel].min(abs_val);
                        all_values[channel].push(abs_val);
                    }
                },
                Tensor::F16(data) => {
                    // Convert F16 to F32 for processing
                    for (idx, val) in data.iter().enumerate() {
                        let channel = idx % num_channels;
                        let abs_val = f32::from(*val).abs();
                        max_vals[channel] = max_vals[channel].max(abs_val);
                        min_vals[channel] = min_vals[channel].min(abs_val);
                        all_values[channel].push(abs_val);
                    }
                },
                Tensor::BF16(data) => {
                    // Convert BF16 to F32 for processing
                    for (idx, val) in data.iter().enumerate() {
                        let channel = idx % num_channels;
                        let abs_val = f32::from(*val).abs();
                        max_vals[channel] = max_vals[channel].max(abs_val);
                        min_vals[channel] = min_vals[channel].min(abs_val);
                        all_values[channel].push(abs_val);
                    }
                },
                Tensor::CF16(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex F16 tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                Tensor::CBF16(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex BF16 tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                Tensor::C32(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex32 tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                Tensor::C64(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Complex64 tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                Tensor::Sparse(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Sparse tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
                #[cfg(feature = "cuda")]
                _ => {
                    return Err(TrustformersError::tensor_op_error(
                        "CUDA tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
                #[cfg(feature = "torch")]
                Tensor::Torch(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Torch tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
                #[cfg(feature = "candle")]
                Tensor::Candle(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Candle tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
                #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
                #[cfg(all(target_os = "macos", feature = "metal"))]
                Tensor::Metal(_) => {
                    return Err(TrustformersError::tensor_op_error(
                        "Metal tensors not yet supported for calibration",
                        "collect_activation_stats",
                    ));
                },
            }
        }

        // Calculate percentiles
        let mut percentile_vals = vec![0.0f32; num_channels];
        for (channel, values) in all_values.iter_mut().enumerate() {
            values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let percentile_idx = ((values.len() as f32 * self.config.activation_percentile / 100.0)
                as usize)
                .min(values.len() - 1);
            percentile_vals[channel] = values[percentile_idx];
        }

        Ok(ActivationStats {
            max_vals,
            _min_vals: min_vals,
            percentile_vals,
            _num_samples: activations.len(),
        })
    }

    /// Calculate smoothing scales based on activation statistics and weights
    fn calculate_smoothing_scales(
        &self,
        stats: &ActivationStats,
        weights: &Tensor,
    ) -> Result<Tensor> {
        let weight_shape = weights.shape();
        let out_channels = weight_shape[0];
        let in_channels = weight_shape[weight_shape.len() - 1];

        // Get weight statistics
        let weight_scales = self.calculate_weight_scales(weights)?;

        // Calculate smoothing scales using the SmoothQuant formula
        let mut scales = vec![1.0f32; in_channels];

        for i in 0..in_channels {
            let act_scale = stats.percentile_vals[i];
            let weight_scale = weight_scales[i];

            if act_scale > 0.0 && weight_scale > 0.0 {
                // SmoothQuant formula: s = (act_scale^alpha) / (weight_scale^(1-alpha))
                let s = (act_scale.powf(self.config.alpha)
                    / weight_scale.powf(1.0 - self.config.alpha))
                    * self.config.migration_strength;
                scales[i] = s.clamp(0.1, 10.0); // Clamp to reasonable range
            }
        }

        Tensor::from_vec(scales, &[in_channels])
    }

    /// Calculate per-channel weight scales
    fn calculate_weight_scales(&self, weights: &Tensor) -> Result<Vec<f32>> {
        let shape = weights.shape();
        let in_channels = shape[shape.len() - 1];
        let mut scales = vec![0.0f32; in_channels];

        match weights {
            Tensor::F32(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get weight data",
                        "calculate_weight_scales",
                    )
                })?;

                // Calculate per-input-channel maximum absolute values
                for (idx, val) in values.iter().enumerate() {
                    let channel = idx % in_channels;
                    scales[channel] = scales[channel].max(val.abs());
                }
            },
            Tensor::F64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get weight data",
                        "calculate_weight_scales",
                    )
                })?;

                // Calculate per-input-channel maximum absolute values (convert to f32)
                for (idx, val) in values.iter().enumerate() {
                    let channel = idx % in_channels;
                    scales[channel] = scales[channel].max((*val as f32).abs());
                }
            },
            Tensor::I64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get weight data",
                        "calculate_weight_scales",
                    )
                })?;

                // Calculate per-input-channel maximum absolute values (convert to f32)
                for (idx, val) in values.iter().enumerate() {
                    let channel = idx % in_channels;
                    scales[channel] = scales[channel].max((*val as f32).abs());
                }
            },
            Tensor::F16(data) => {
                // Convert F16 to F32 for processing
                for (idx, val) in data.iter().enumerate() {
                    let channel = idx % in_channels;
                    scales[channel] = scales[channel].max(f32::from(*val).abs());
                }
            },
            Tensor::BF16(data) => {
                // Convert BF16 to F32 for processing
                for (idx, val) in data.iter().enumerate() {
                    let channel = idx % in_channels;
                    scales[channel] = scales[channel].max(f32::from(*val).abs());
                }
            },
            Tensor::CF16(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Complex F16 tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            Tensor::CBF16(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Complex BF16 tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            Tensor::C32(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Complex32 tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            Tensor::C64(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Complex64 tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            Tensor::Sparse(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Sparse tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Torch tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Candle tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "Metal tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => {
                return Err(TrustformersError::tensor_op_error(
                    "CUDA tensors not yet supported for weight scaling",
                    "calculate_weight_scales",
                ));
            },
        }

        Ok(scales)
    }

    /// Apply smoothing to weights
    pub fn smooth_weights(&self, layer_name: &str, weights: &Tensor) -> Result<Tensor> {
        let scales = self.smoothing_scales.get(layer_name).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("No smoothing scales found for layer {}", layer_name),
                "smooth_weights",
            )
        })?;

        // Apply smoothing: W_smooth = W / s
        // For 2D weights [rows, cols], manually broadcast scales
        let weight_shape = weights.shape();
        if weight_shape.len() == 2 && scales.shape().len() == 1 {
            let (rows, cols) = (weight_shape[0], weight_shape[1]);
            if scales.shape()[0] == cols {
                // Scale each row by corresponding column scale
                let weight_data = weights.to_vec_f32()?;
                let scale_data = scales.to_vec_f32()?;

                let mut result = Vec::with_capacity(weight_data.len());
                for row in 0..rows {
                    for (col, &scale) in scale_data.iter().enumerate() {
                        let idx = row * cols + col;
                        result.push(weight_data[idx] / scale);
                    }
                }

                Tensor::from_vec(result, &[rows, cols])
            } else if scales.shape()[0] == rows {
                // Scale each column by corresponding row scale
                let weight_data = weights.to_vec_f32()?;
                let scale_data = scales.to_vec_f32()?;

                let mut result = Vec::with_capacity(weight_data.len());
                for (row, &scale) in scale_data.iter().enumerate() {
                    for col in 0..cols {
                        let idx = row * cols + col;
                        result.push(weight_data[idx] / scale);
                    }
                }

                Tensor::from_vec(result, &[rows, cols])
            } else {
                Err(TrustformersError::tensor_op_error(
                    &format!(
                        "Scale dimensions {} don't match any weight dimension [{}, {}]",
                        scales.shape()[0],
                        rows,
                        cols
                    ),
                    "smoothquant weight scaling",
                ))
            }
        } else {
            // Fallback for other shapes
            weights.div(scales)
        }
    }

    /// Apply smoothing to activations
    pub fn smooth_activations(&self, layer_name: &str, activations: &Tensor) -> Result<Tensor> {
        let scales = self.smoothing_scales.get(layer_name).ok_or_else(|| {
            TrustformersError::tensor_op_error(
                &format!("No smoothing scales found for layer {}", layer_name),
                "smooth_weights",
            )
        })?;

        // Apply smoothing: X_smooth = X * s
        activations.mul(scales)
    }

    /// Quantize smoothed weights to INT8
    pub fn quantize_weights(&self, weights: &Tensor) -> Result<QuantizedTensor> {
        // Calculate quantization scale
        let (min_val, max_val) = self.get_tensor_range(weights)?;
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as u8;

        // Quantize
        let quantized =
            weights.sub_scalar(min_val)?.div_scalar(scale)?.round()?.clamp(0.0, 255.0)?;

        Ok(QuantizedTensor {
            data: self.tensor_to_u8(quantized)?,
            scale,
            zero_point,
            shape: weights.shape().to_vec(),
        })
    }

    /// Quantize smoothed activations to INT8
    pub fn quantize_activations(&self, activations: &Tensor) -> Result<QuantizedTensor> {
        if !self.config.quantize_activations {
            return Err(TrustformersError::invalid_config(
                "Activation quantization is disabled".to_string(),
            ));
        }

        // Dynamic quantization for activations
        let (min_val, max_val) = self.get_tensor_range(activations)?;
        let scale = (max_val - min_val) / 255.0;
        let zero_point = (-min_val / scale).round() as u8;

        let quantized =
            activations.sub_scalar(min_val)?.div_scalar(scale)?.round()?.clamp(0.0, 255.0)?;

        Ok(QuantizedTensor {
            data: self.tensor_to_u8(quantized)?,
            scale,
            zero_point,
            shape: activations.shape().to_vec(),
        })
    }

    /// Get min/max range of tensor
    fn get_tensor_range(&self, tensor: &Tensor) -> Result<(f32, f32)> {
        match tensor {
            Tensor::F32(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                let min = values.iter().fold(f32::MAX, |a, &b| a.min(b));
                let max = values.iter().fold(f32::MIN, |a, &b| a.max(b));

                Ok((min, max))
            },
            Tensor::F64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                let min = values.iter().fold(f64::MAX, |a, &b| a.min(b)) as f32;
                let max = values.iter().fold(f64::MIN, |a, &b| a.max(b)) as f32;

                Ok((min, max))
            },
            Tensor::I64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                let min = values.iter().fold(i64::MAX, |a, &b| a.min(b)) as f32;
                let max = values.iter().fold(i64::MIN, |a, &b| a.max(b)) as f32;

                Ok((min, max))
            },
            Tensor::C32(_) => Err(TrustformersError::tensor_op_error(
                "Complex32 tensors not yet supported for range calculation",
                "calculate_range",
            )),
            Tensor::C64(_) => Err(TrustformersError::tensor_op_error(
                "Complex64 tensors not yet supported for range calculation",
                "calculate_range",
            )),
            Tensor::F16(data) => {
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(f32::from(b)));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(f32::from(b)));
                Ok((min, max))
            },
            Tensor::BF16(data) => {
                let min = data.iter().fold(f32::INFINITY, |a, &b| a.min(f32::from(b)));
                let max = data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(f32::from(b)));
                Ok((min, max))
            },
            Tensor::CF16(_) => Err(TrustformersError::tensor_op_error(
                "CF16 tensors not yet supported for range calculation",
                "calculate_range",
            )),
            Tensor::CBF16(_) => Err(TrustformersError::tensor_op_error(
                "CBF16 tensors not yet supported for range calculation",
                "calculate_range",
            )),
            Tensor::Sparse(_) => Err(TrustformersError::tensor_op_error(
                "Sparse tensors not yet supported for range calculation",
                "calculate_range",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "cuda")]
            _ => Err(TrustformersError::tensor_op_error(
                "CUDA tensors not yet supported",
                "calculate_range",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => Err(TrustformersError::tensor_op_error(
                "Torch tensors not yet supported",
                "calculate_range",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => Err(TrustformersError::tensor_op_error(
                "Candle tensors not yet supported",
                "calculate_range",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(_) => Err(TrustformersError::tensor_op_error(
                "Metal tensors not yet supported",
                "calculate_range",
            )),
        }
    }

    /// Convert tensor to u8 vector
    fn tensor_to_u8(&self, tensor: Tensor) -> Result<Vec<u8>> {
        match tensor {
            Tensor::F32(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                Ok(values.iter().map(|&v| v as u8).collect())
            },
            Tensor::F64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                Ok(values.iter().map(|&v| v as f32 as u8).collect())
            },
            Tensor::I64(data) => {
                let values = data.as_slice().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "Failed to get tensor data",
                        "collect_activation_stats",
                    )
                })?;

                Ok(values.iter().map(|&v| v as f32 as u8).collect())
            },
            Tensor::C32(_) => Err(TrustformersError::tensor_op_error(
                "Complex32 tensors not yet supported for conversion to u8",
                "tensor_to_u8",
            )),
            Tensor::C64(_) => Err(TrustformersError::tensor_op_error(
                "Complex64 tensors not yet supported for conversion to u8",
                "tensor_to_u8",
            )),
            Tensor::F16(data) => Ok(data.iter().map(|&v| f32::from(v) as u8).collect()),
            Tensor::BF16(data) => Ok(data.iter().map(|&v| f32::from(v) as u8).collect()),
            Tensor::CF16(_) => Err(TrustformersError::tensor_op_error(
                "CF16 tensors not yet supported for conversion to u8",
                "tensor_to_u8",
            )),
            Tensor::CBF16(_) => Err(TrustformersError::tensor_op_error(
                "CBF16 tensors not yet supported for conversion to u8",
                "tensor_to_u8",
            )),
            Tensor::Sparse(_) => Err(TrustformersError::tensor_op_error(
                "Sparse tensors not yet supported for conversion to u8",
                "tensor_to_u8",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "cuda")]
            _ => Err(TrustformersError::tensor_op_error(
                "CUDA tensors not yet supported",
                "tensor_to_u8",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => Err(TrustformersError::tensor_op_error(
                "Torch tensors not yet supported",
                "tensor_to_u8",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => Err(TrustformersError::tensor_op_error(
                "Candle tensors not yet supported",
                "tensor_to_u8",
            )),
            #[allow(unreachable_patterns)] // Feature-gated patterns after catch-all
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(_) => Err(TrustformersError::tensor_op_error(
                "Metal tensors not yet supported",
                "tensor_to_u8",
            )),
        }
    }

    /// Apply SmoothQuant to a linear layer
    pub fn quantize_linear_layer(
        &mut self,
        layer_name: &str,
        weights: &Tensor,
        calibration_data: &[Tensor],
    ) -> Result<SmoothQuantizedLinear> {
        // Calibrate with activation data
        self.calibrate(layer_name, calibration_data, weights)?;

        // Smooth and quantize weights
        let smoothed_weights = self.smooth_weights(layer_name, weights)?;
        let quantized_weights = self.quantize_weights(&smoothed_weights)?;

        // Get smoothing scales for runtime
        let smoothing_scales = self
            .smoothing_scales
            .get(layer_name)
            .ok_or_else(|| {
                TrustformersError::tensor_op_error("Missing smoothing scales", "smooth_weights")
            })?
            .clone();

        Ok(SmoothQuantizedLinear {
            quantized_weights,
            smoothing_scales,
            quantize_activations: self.config.quantize_activations,
        })
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data as u8
    pub data: Vec<u8>,
    /// Scale factor
    pub scale: f32,
    /// Zero point
    pub zero_point: u8,
    /// Original shape
    pub shape: Vec<usize>,
}

impl QuantizedTensor {
    /// Dequantize back to f32 tensor
    pub fn dequantize(&self) -> Result<Tensor> {
        let values: Vec<f32> = self
            .data
            .iter()
            .map(|&v| (v as f32 - self.zero_point as f32) * self.scale)
            .collect();

        Tensor::from_vec(values, &self.shape)
    }
}

/// SmoothQuant quantized linear layer
#[derive(Debug, Clone)]
pub struct SmoothQuantizedLinear {
    /// Quantized weights
    pub quantized_weights: QuantizedTensor,
    /// Smoothing scales for activations
    pub smoothing_scales: Tensor,
    /// Whether to quantize activations
    pub quantize_activations: bool,
}

impl SmoothQuantizedLinear {
    /// Forward pass with optional activation quantization
    pub fn forward(&self, input: &Tensor) -> Result<Tensor> {
        // Apply smoothing to input
        let smoothed_input = input.mul(&self.smoothing_scales)?;

        // Optionally quantize activations
        let processed_input = if self.quantize_activations {
            // Quantize and dequantize for INT8 computation
            let quantizer = SmoothQuantizer::new(SmoothQuantConfig::default());
            let quantized = quantizer.quantize_activations(&smoothed_input)?;
            quantized.dequantize()?
        } else {
            smoothed_input
        };

        // Dequantize weights
        let weights = self.quantized_weights.dequantize()?;

        // Perform linear operation
        processed_input.matmul(&weights.t()?)
    }
}

/// Migration analyzer for finding optimal alpha values
pub struct MigrationAnalyzer {
    /// Range of alpha values to test
    alpha_range: Vec<f32>,
    /// Metric to optimize (e.g., "perplexity", "accuracy")
    #[allow(dead_code)]
    metric: String,
}

impl MigrationAnalyzer {
    /// Create a new migration analyzer
    pub fn new(metric: &str) -> Self {
        Self {
            alpha_range: vec![0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            metric: metric.to_string(),
        }
    }

    /// Find optimal alpha value for a layer
    pub fn find_optimal_alpha(
        &self,
        weights: &Tensor,
        calibration_data: &[Tensor],
        eval_fn: impl Fn(&SmoothQuantizedLinear) -> f32,
    ) -> Result<f32> {
        let mut best_alpha = 0.5;
        let mut best_score = f32::MIN;

        for &alpha in &self.alpha_range {
            let config = SmoothQuantConfig {
                alpha,
                ..Default::default()
            };

            let mut quantizer = SmoothQuantizer::new(config);
            let quantized =
                quantizer.quantize_linear_layer("test_layer", weights, calibration_data)?;

            let score = eval_fn(&quantized);
            if score > best_score {
                best_score = score;
                best_alpha = alpha;
            }
        }

        Ok(best_alpha)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_smoothquant_calibration() {
        let config = SmoothQuantConfig::default();
        let mut quantizer = SmoothQuantizer::new(config);

        // Create test data
        let weights = Tensor::randn(&[64, 64]).unwrap();
        let activations = vec![
            Tensor::randn(&[32, 64]).unwrap(),
            Tensor::randn(&[32, 64]).unwrap(),
        ];

        // Calibrate
        quantizer.calibrate("test_layer", &activations, &weights).unwrap();

        // Check that smoothing scales were calculated
        assert!(quantizer.smoothing_scales.contains_key("test_layer"));
    }

    #[test]
    fn test_weight_smoothing() {
        let config = SmoothQuantConfig::default();
        let mut quantizer = SmoothQuantizer::new(config);

        // Setup
        let weights = Tensor::ones(&[64, 64]).unwrap();
        let scales = Tensor::from_vec(vec![2.0; 64], &[64]).unwrap();
        quantizer.smoothing_scales.insert("test_layer".to_string(), scales);

        // Apply smoothing
        let smoothed = quantizer.smooth_weights("test_layer", &weights).unwrap();

        // Verify
        match smoothed {
            Tensor::F32(data) => {
                let values = data.as_slice().unwrap();
                assert!((values[0] - 0.5).abs() < 1e-5);
            },
            Tensor::F64(data) => {
                let values = data.as_slice().unwrap();
                assert!((values[0] as f32 - 0.5).abs() < 1e-5);
            },
            Tensor::I64(data) => {
                let values = data.as_slice().unwrap();
                assert!((values[0] as f32 - 0.5).abs() < 1e-5);
            },
            Tensor::F16(_) => panic!("F16 tensor type not expected in smoothing test"),
            Tensor::BF16(_) => panic!("BF16 tensor type not expected in smoothing test"),
            Tensor::C32(_) => panic!("C32 tensor type not expected in smoothing test"),
            Tensor::C64(_) => panic!("C64 tensor type not expected in smoothing test"),
            Tensor::CF16(_) => panic!("CF16 tensor type not expected in smoothing test"),
            Tensor::CBF16(_) => panic!("CBF16 tensor type not expected in smoothing test"),
            Tensor::Sparse(_) => panic!("Sparse tensor type not expected in smoothing test"),
            #[cfg(all(target_os = "macos", feature = "metal"))]
            Tensor::Metal(_) => panic!("Metal tensor type not expected in smoothing test"),
            #[cfg(feature = "cuda")]
            Tensor::CUDA(_) => panic!("CUDA tensor type not expected in smoothing test"),
            #[cfg(feature = "torch")]
            Tensor::Torch(_) => panic!("Unexpected Torch tensor type"),
            #[cfg(feature = "candle")]
            Tensor::Candle(_) => panic!("Unexpected Candle tensor type"),
        }
    }

    #[test]
    fn test_quantization() {
        let quantizer = SmoothQuantizer::new(SmoothQuantConfig::default());

        // Create test tensor
        let tensor = Tensor::from_vec(vec![0.0, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();

        // Quantize
        let quantized = quantizer.quantize_weights(&tensor).unwrap();

        // Verify
        assert_eq!(quantized.shape, vec![5]);
        assert_eq!(quantized.data.len(), 5);

        // Dequantize and check reconstruction
        let reconstructed = quantized.dequantize().unwrap();
        match (tensor, reconstructed) {
            (Tensor::F32(orig), Tensor::F32(recon)) => {
                let orig_vals = orig.as_slice().unwrap();
                let recon_vals = recon.as_slice().unwrap();

                for (o, r) in orig_vals.iter().zip(recon_vals.iter()) {
                    assert!((o - r).abs() < 0.1); // Allow small quantization error
                }
            },
            _ => panic!("Unexpected tensor types"),
        }
    }

    #[test]
    fn test_f64_tensor_support() {
        use crate::tensor::DType;
        let quantizer = SmoothQuantizer::new(SmoothQuantConfig::default());

        // Create F64 test tensor by converting from F32
        let base_tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let tensor = base_tensor.to_dtype(DType::F64).unwrap();

        // Test get_tensor_range with F64
        let (min, max) = quantizer.get_tensor_range(&tensor).unwrap();
        assert!((min - 0.0).abs() < 1e-5);
        assert!((max - 4.0).abs() < 1e-5);

        // Test tensor_to_u8 with F64
        let quantized = tensor.add_scalar(1.0).unwrap().clamp(0.0, 255.0).unwrap();
        let u8_data = quantizer.tensor_to_u8(quantized).unwrap();
        assert_eq!(u8_data, vec![1, 2, 3, 4, 5]);
    }

    #[test]
    fn test_i64_tensor_support() {
        use crate::tensor::DType;
        let quantizer = SmoothQuantizer::new(SmoothQuantConfig::default());

        // Create I64 test tensor by converting from F32
        let base_tensor = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0, 4.0], &[5]).unwrap();
        let tensor = base_tensor.to_dtype(DType::I64).unwrap();

        // Test get_tensor_range with I64
        let (min, max) = quantizer.get_tensor_range(&tensor).unwrap();
        assert!((min - 0.0).abs() < 1e-5);
        assert!((max - 4.0).abs() < 1e-5);

        // Test tensor_to_u8 with I64 - convert to F32 for clamp operation
        let f32_tensor = tensor.to_dtype(DType::F32).unwrap();
        let quantized = f32_tensor.add_scalar(1.0).unwrap().clamp(0.0, 255.0).unwrap();
        let u8_data = quantizer.tensor_to_u8(quantized).unwrap();
        assert_eq!(u8_data, vec![1, 2, 3, 4, 5]);
    }
}
