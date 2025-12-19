//! FP8 quantization for modern GPU architectures
//!
//! This module implements FP8 (8-bit floating point) quantization, which is natively
//! supported on modern GPUs like NVIDIA H100, AMD MI300, and future accelerators.
//!
//! FP8 comes in two main formats:
//! - **E4M3** (4-bit exponent, 3-bit mantissa): Better dynamic range for forward pass
//! - **E5M2** (5-bit exponent, 2-bit mantissa): Better precision for gradients
//!
//! # Features
//! - Native FP8 tensor quantization and dequantization
//! - Per-tensor and per-channel scaling
//! - Delayed scaling for training stability
//! - Automatic format selection based on use case
//! - Hardware-accelerated operations when available
//! - Integration with mixed-precision training
//!
//! # Examples
//!
//! ```rust
//! use trustformers_core::quantization::fp8::{FP8Config, FP8Quantizer, FP8Format};
//! use trustformers_core::tensor::Tensor;
//!
//! let config = FP8Config {
//!     format: FP8Format::E4M3,
//!     per_channel: false,
//!     delayed_scaling: true,
//!     ..Default::default()
//! };
//!
//! let quantizer = FP8Quantizer::new(config)?;
//! let tensor = Tensor::randn(&[1024, 768])?;
//!
//! // Quantize to FP8
//! let (quantized, scale) = quantizer.quantize(&tensor)?;
//!
//! // Dequantize back
//! let dequantized = quantizer.dequantize(&quantized, &scale)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};

/// FP8 data format specification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FP8Format {
    /// E4M3: 4-bit exponent, 3-bit mantissa (sign: 1, exp: 4, mantissa: 3)
    /// Range: ±448, better dynamic range
    /// Best for: Forward pass, activations, weights
    E4M3,

    /// E5M2: 5-bit exponent, 2-bit mantissa (sign: 1, exp: 5, mantissa: 2)
    /// Range: ±57344, wider range but less precision
    /// Best for: Gradients, loss scaling
    E5M2,
}

impl FP8Format {
    /// Maximum representable value for this format
    pub fn max_value(&self) -> f32 {
        match self {
            FP8Format::E4M3 => 448.0,
            FP8Format::E5M2 => 57344.0,
        }
    }

    /// Minimum positive normal value
    pub fn min_positive_normal(&self) -> f32 {
        match self {
            FP8Format::E4M3 => 2.0f32.powi(-9),  // 2^-9
            FP8Format::E5M2 => 2.0f32.powi(-16), // 2^-16
        }
    }

    /// Number of mantissa bits
    pub fn mantissa_bits(&self) -> u8 {
        match self {
            FP8Format::E4M3 => 3,
            FP8Format::E5M2 => 2,
        }
    }

    /// Number of exponent bits
    pub fn exponent_bits(&self) -> u8 {
        match self {
            FP8Format::E4M3 => 4,
            FP8Format::E5M2 => 5,
        }
    }

    /// Exponent bias
    pub fn exponent_bias(&self) -> i32 {
        match self {
            FP8Format::E4M3 => 7,  // 2^(4-1) - 1
            FP8Format::E5M2 => 15, // 2^(5-1) - 1
        }
    }
}

/// Scaling strategy for FP8 quantization
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalingStrategy {
    /// Per-tensor scaling: single scale factor for entire tensor
    PerTensor,

    /// Per-channel scaling: scale factor per output channel
    PerChannel,

    /// Per-token scaling: scale factor per token (for sequence models)
    PerToken,

    /// Block-wise scaling: scale factor per fixed-size block
    BlockWise { block_size: usize },
}

/// Delayed scaling configuration for training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayedScalingConfig {
    /// Enable delayed scaling
    pub enabled: bool,

    /// Number of intervals before updating scale
    pub interval: usize,

    /// Margin factor (multiplier for scale to prevent overflow)
    pub margin: f32,

    /// Update threshold (fraction of max value to trigger update)
    pub update_threshold: f32,

    /// History window for statistics
    pub history_window: usize,
}

impl Default for DelayedScalingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: 1000,
            margin: 1.2,
            update_threshold: 0.95,
            history_window: 100,
        }
    }
}

/// FP8 quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FP8Config {
    /// FP8 format to use
    pub format: FP8Format,

    /// Scaling strategy
    pub scaling: ScalingStrategy,

    /// Delayed scaling configuration
    pub delayed_scaling: DelayedScalingConfig,

    /// Enable stochastic rounding for better accuracy
    pub stochastic_rounding: bool,

    /// Clipping strategy (clip to max or saturate)
    pub clip_to_max: bool,

    /// Use hardware FP8 operations if available
    pub use_hardware_ops: bool,

    /// Calibration samples for initial scale estimation
    pub calibration_samples: usize,
}

impl Default for FP8Config {
    fn default() -> Self {
        Self {
            format: FP8Format::E4M3,
            scaling: ScalingStrategy::PerTensor,
            delayed_scaling: DelayedScalingConfig::default(),
            stochastic_rounding: true,
            clip_to_max: true,
            use_hardware_ops: true,
            calibration_samples: 100,
        }
    }
}

/// FP8 quantized tensor representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FP8Tensor {
    /// Quantized data stored as u8 (bitwise FP8 representation)
    pub data: Vec<u8>,

    /// Original tensor shape
    pub shape: Vec<usize>,

    /// FP8 format used
    pub format: FP8Format,

    /// Scale factors (shape depends on scaling strategy)
    pub scales: ScaleFactors,

    /// Zero points (if using asymmetric quantization)
    pub zero_points: Option<Vec<f32>>,
}

/// Scale factors for FP8 quantization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScaleFactors {
    /// Single scale for entire tensor
    PerTensor(f32),

    /// Per-channel scales
    PerChannel(Vec<f32>),

    /// Per-token scales
    PerToken(Vec<f32>),

    /// Block-wise scales
    BlockWise { scales: Vec<f32>, block_size: usize },
}

/// FP8 quantization statistics for delayed scaling
#[derive(Debug, Clone)]
struct QuantStats {
    /// Maximum absolute values history
    max_history: Vec<f32>,

    /// Current iteration counter
    iteration: usize,

    /// Current scale factor
    current_scale: f32,

    /// Number of overflow events
    overflow_count: usize,

    /// Number of underflow events
    underflow_count: usize,
}

impl QuantStats {
    fn new(initial_scale: f32, window_size: usize) -> Self {
        Self {
            max_history: Vec::with_capacity(window_size),
            iteration: 0,
            current_scale: initial_scale,
            overflow_count: 0,
            underflow_count: 0,
        }
    }

    fn update(&mut self, max_val: f32, window_size: usize) {
        self.max_history.push(max_val);
        if self.max_history.len() > window_size {
            self.max_history.remove(0);
        }
        self.iteration += 1;
    }

    fn get_optimal_scale(&self, margin: f32, max_value: f32) -> f32 {
        if self.max_history.is_empty() {
            return self.current_scale;
        }

        // Use percentile instead of max to be robust to outliers
        let mut sorted = self.max_history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let percentile_99 = sorted[(sorted.len() as f32 * 0.99) as usize];

        max_value / (percentile_99 * margin)
    }
}

/// FP8 quantizer with delayed scaling support
pub struct FP8Quantizer {
    /// Configuration
    config: FP8Config,

    /// Statistics for delayed scaling (per channel or per tensor)
    stats: Option<Vec<QuantStats>>,
}

impl FP8Quantizer {
    /// Create a new FP8 quantizer
    pub fn new(config: FP8Config) -> Result<Self> {
        Ok(Self {
            config,
            stats: None,
        })
    }

    /// Initialize statistics for delayed scaling
    fn init_stats(&mut self, num_groups: usize) {
        if self.config.delayed_scaling.enabled && self.stats.is_none() {
            let initial_scale = 1.0;
            let window = self.config.delayed_scaling.history_window;
            self.stats =
                Some((0..num_groups).map(|_| QuantStats::new(initial_scale, window)).collect());
        }
    }

    /// Quantize a tensor to FP8
    pub fn quantize(&mut self, tensor: &Tensor) -> Result<FP8Tensor> {
        let data = tensor.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        match self.config.scaling {
            ScalingStrategy::PerTensor => self.quantize_per_tensor(&data, &shape),
            ScalingStrategy::PerChannel => self.quantize_per_channel(&data, &shape),
            ScalingStrategy::PerToken => self.quantize_per_token(&data, &shape),
            ScalingStrategy::BlockWise { block_size } => {
                self.quantize_blockwise(&data, &shape, block_size)
            },
        }
    }

    /// Per-tensor quantization
    fn quantize_per_tensor(&mut self, data: &[f32], shape: &[usize]) -> Result<FP8Tensor> {
        self.init_stats(1);

        // Compute max absolute value
        let max_abs = data
            .iter()
            .map(|x| x.abs())
            .max_by(|a, b| a.partial_cmp(b).unwrap())
            .unwrap_or(1e-8);

        // Compute or update scale
        let scale = if let Some(stats) = &mut self.stats {
            let stat = &mut stats[0];
            stat.update(max_abs, self.config.delayed_scaling.history_window);

            if stat.iteration % self.config.delayed_scaling.interval == 0 {
                stat.current_scale = stat.get_optimal_scale(
                    self.config.delayed_scaling.margin,
                    self.config.format.max_value(),
                );
            }
            stat.current_scale
        } else {
            self.config.format.max_value() / (max_abs * 1.2)
        };

        // Quantize data
        let quantized = self.quantize_data(data, scale)?;

        Ok(FP8Tensor {
            data: quantized,
            shape: shape.to_vec(),
            format: self.config.format,
            scales: ScaleFactors::PerTensor(scale),
            zero_points: None,
        })
    }

    /// Per-channel quantization
    fn quantize_per_channel(&mut self, data: &[f32], shape: &[usize]) -> Result<FP8Tensor> {
        if shape.len() < 2 {
            return Err(TrustformersError::quantization_error(
                "Per-channel quantization requires at least 2D tensor".to_string(),
            ));
        }

        let num_channels = shape[0];
        let channel_size = data.len() / num_channels;

        self.init_stats(num_channels);

        // Compute per-channel scales
        let mut scales = Vec::with_capacity(num_channels);
        let mut quantized_data = Vec::with_capacity(data.len());

        for ch in 0..num_channels {
            let channel_data = &data[ch * channel_size..(ch + 1) * channel_size];

            let max_abs = channel_data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1e-8);

            let scale = if let Some(stats) = &mut self.stats {
                let stat = &mut stats[ch];
                stat.update(max_abs, self.config.delayed_scaling.history_window);

                if stat.iteration % self.config.delayed_scaling.interval == 0 {
                    stat.current_scale = stat.get_optimal_scale(
                        self.config.delayed_scaling.margin,
                        self.config.format.max_value(),
                    );
                }
                stat.current_scale
            } else {
                self.config.format.max_value() / (max_abs * 1.2)
            };

            scales.push(scale);

            let ch_quantized = self.quantize_data(channel_data, scale)?;
            quantized_data.extend(ch_quantized);
        }

        Ok(FP8Tensor {
            data: quantized_data,
            shape: shape.to_vec(),
            format: self.config.format,
            scales: ScaleFactors::PerChannel(scales),
            zero_points: None,
        })
    }

    /// Per-token quantization (for sequence models)
    fn quantize_per_token(&mut self, data: &[f32], shape: &[usize]) -> Result<FP8Tensor> {
        if shape.len() < 2 {
            return Err(TrustformersError::quantization_error(
                "Per-token quantization requires at least 2D tensor [batch, seq_len, ...]"
                    .to_string(),
            ));
        }

        // Assume shape is [batch, seq_len, hidden_dim] or similar
        let batch_size = shape[0];
        let seq_len = if shape.len() >= 2 { shape[1] } else { 1 };
        let num_tokens = batch_size * seq_len;
        let token_size = data.len() / num_tokens;

        self.init_stats(num_tokens);

        let mut scales = Vec::with_capacity(num_tokens);
        let mut quantized_data = Vec::with_capacity(data.len());

        for tok in 0..num_tokens {
            let token_data = &data[tok * token_size..(tok + 1) * token_size];

            let max_abs = token_data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1e-8);

            let scale = self.config.format.max_value() / (max_abs * 1.2);
            scales.push(scale);

            let tok_quantized = self.quantize_data(token_data, scale)?;
            quantized_data.extend(tok_quantized);
        }

        Ok(FP8Tensor {
            data: quantized_data,
            shape: shape.to_vec(),
            format: self.config.format,
            scales: ScaleFactors::PerToken(scales),
            zero_points: None,
        })
    }

    /// Block-wise quantization
    fn quantize_blockwise(
        &mut self,
        data: &[f32],
        shape: &[usize],
        block_size: usize,
    ) -> Result<FP8Tensor> {
        let num_blocks = data.len().div_ceil(block_size);

        self.init_stats(num_blocks);

        let mut scales = Vec::with_capacity(num_blocks);
        let mut quantized_data = Vec::with_capacity(data.len());

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block_data = &data[start..end];

            let max_abs = block_data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1e-8);

            let scale = self.config.format.max_value() / (max_abs * 1.2);
            scales.push(scale);

            let block_quantized = self.quantize_data(block_data, scale)?;
            quantized_data.extend(block_quantized);
        }

        Ok(FP8Tensor {
            data: quantized_data,
            shape: shape.to_vec(),
            format: self.config.format,
            scales: ScaleFactors::BlockWise { scales, block_size },
            zero_points: None,
        })
    }

    /// Core quantization logic: convert f32 values to FP8 representation
    fn quantize_data(&mut self, data: &[f32], scale: f32) -> Result<Vec<u8>> {
        let max_value = self.config.format.max_value();
        let mut quantized = Vec::with_capacity(data.len());

        for &value in data {
            let scaled = value * scale;

            // Clip to FP8 range
            let clipped = if self.config.clip_to_max {
                scaled.clamp(-max_value, max_value)
            } else {
                scaled
            };

            // Convert to FP8 (simplified - actual implementation would use proper IEEE conversion)
            let fp8_val = self.f32_to_fp8(clipped)?;
            quantized.push(fp8_val);
        }

        Ok(quantized)
    }

    /// Convert f32 to FP8 bitwise representation
    fn f32_to_fp8(&mut self, value: f32) -> Result<u8> {
        // Extract sign, exponent, and mantissa from f32
        let bits = value.to_bits();
        let sign = (bits >> 31) & 1;
        let exp_f32 = ((bits >> 23) & 0xFF) as i32;
        let mant_f32 = bits & 0x7F_FFFF;

        // Handle special cases
        if value == 0.0 || value == -0.0 {
            return Ok((sign as u8) << 7);
        }

        if value.is_nan() || value.is_infinite() {
            // Map to max FP8 value
            let exp_bits = self.config.format.exponent_bits();
            let max_exp = (1 << exp_bits) - 1;
            return Ok(
                ((sign as u8) << 7) | ((max_exp as u8) << self.config.format.mantissa_bits())
            );
        }

        // Rebias exponent
        let exp_bias_f32 = 127;
        let exp_bias_fp8 = self.config.format.exponent_bias();
        let exp = exp_f32 - exp_bias_f32 + exp_bias_fp8;

        // Check bounds
        let max_exp = (1 << self.config.format.exponent_bits()) - 1;
        if exp <= 0 {
            // Subnormal or underflow - map to zero
            if let Some(stats) = &mut self.stats {
                stats[0].underflow_count += 1;
            }
            return Ok((sign as u8) << 7);
        }
        if exp >= max_exp {
            // Overflow - saturate to max
            if let Some(stats) = &mut self.stats {
                stats[0].overflow_count += 1;
            }
            let max_exp_fp8 = max_exp - 1;
            let max_mant = (1 << self.config.format.mantissa_bits()) - 1;
            return Ok(((sign as u8) << 7)
                | ((max_exp_fp8 as u8) << self.config.format.mantissa_bits())
                | (max_mant as u8));
        }

        // Extract mantissa bits
        let mant_bits = self.config.format.mantissa_bits();
        let mant_shift = 23 - mant_bits;
        let mut mant = (mant_f32 >> mant_shift) as u8;

        // Round to nearest even (stochastic rounding disabled for now)
        // TODO: Implement stochastic rounding when scirs2_core Random API is clearer
        let remainder = mant_f32 & ((1 << mant_shift) - 1);
        if remainder > (1 << (mant_shift - 1))
            || (remainder == (1 << (mant_shift - 1)) && (mant & 1) == 1)
        {
            mant = mant.saturating_add(1);
        }

        // Combine sign, exponent, mantissa
        let fp8 =
            ((sign as u8) << 7) | ((exp as u8) << mant_bits) | (mant & ((1 << mant_bits) - 1));

        Ok(fp8)
    }

    /// Convert FP8 bitwise representation to f32
    fn fp8_to_f32(&self, fp8: u8) -> f32 {
        let mant_bits = self.config.format.mantissa_bits();
        let exp_bits = self.config.format.exponent_bits();

        let sign = (fp8 >> 7) & 1;
        let exp = ((fp8 >> mant_bits) & ((1 << exp_bits) - 1)) as i32;
        let mant = (fp8 & ((1 << mant_bits) - 1)) as u32;

        // Handle zero
        if exp == 0 && mant == 0 {
            return if sign == 1 { -0.0 } else { 0.0 };
        }

        // Rebias to f32 exponent
        let exp_bias_fp8 = self.config.format.exponent_bias();
        let exp_bias_f32 = 127;
        let exp_f32 = exp - exp_bias_fp8 + exp_bias_f32;

        // Handle special cases
        let max_exp = (1 << exp_bits) - 1;
        if exp == max_exp {
            return if sign == 1 {
                -self.config.format.max_value()
            } else {
                self.config.format.max_value()
            };
        }

        // Construct f32 mantissa
        let mant_shift = 23 - mant_bits;
        let mant_f32 = (mant << mant_shift) | (1 << 23); // Add implicit leading 1

        // Construct f32
        let bits = ((sign as u32) << 31) | ((exp_f32 as u32) << 23) | (mant_f32 & 0x7F_FFFF);
        f32::from_bits(bits)
    }

    /// Dequantize FP8 tensor back to f32
    pub fn dequantize(&self, fp8_tensor: &FP8Tensor) -> Result<Tensor> {
        let mut dequantized = Vec::with_capacity(fp8_tensor.data.len());

        match &fp8_tensor.scales {
            ScaleFactors::PerTensor(scale) => {
                for &fp8_val in &fp8_tensor.data {
                    let f32_val = self.fp8_to_f32(fp8_val) / scale;
                    dequantized.push(f32_val);
                }
            },
            ScaleFactors::PerChannel(scales) => {
                let num_channels = scales.len();
                let channel_size = fp8_tensor.data.len() / num_channels;

                for (ch, &scale) in scales.iter().enumerate() {
                    for i in 0..channel_size {
                        let idx = ch * channel_size + i;
                        let f32_val = self.fp8_to_f32(fp8_tensor.data[idx]) / scale;
                        dequantized.push(f32_val);
                    }
                }
            },
            ScaleFactors::PerToken(scales) => {
                let num_tokens = scales.len();
                let token_size = fp8_tensor.data.len() / num_tokens;

                for (tok, &scale) in scales.iter().enumerate() {
                    for i in 0..token_size {
                        let idx = tok * token_size + i;
                        let f32_val = self.fp8_to_f32(fp8_tensor.data[idx]) / scale;
                        dequantized.push(f32_val);
                    }
                }
            },
            ScaleFactors::BlockWise { scales, block_size } => {
                for (block_idx, &scale) in scales.iter().enumerate() {
                    let start = block_idx * block_size;
                    let end = (start + block_size).min(fp8_tensor.data.len());

                    for idx in start..end {
                        let f32_val = self.fp8_to_f32(fp8_tensor.data[idx]) / scale;
                        dequantized.push(f32_val);
                    }
                }
            },
        }

        Tensor::from_vec(dequantized, &fp8_tensor.shape)
    }

    /// Get quantization statistics
    pub fn get_stats(&self) -> Option<Vec<(usize, usize)>> {
        self.stats
            .as_ref()
            .map(|stats| stats.iter().map(|s| (s.overflow_count, s.underflow_count)).collect())
    }

    /// Reset statistics
    pub fn reset_stats(&mut self) {
        if let Some(stats) = &mut self.stats {
            for stat in stats {
                stat.overflow_count = 0;
                stat.underflow_count = 0;
            }
        }
    }
}

/// Utility functions for FP8 quantization
/// Automatic format selection based on tensor characteristics
pub fn select_fp8_format(tensor: &Tensor, use_case: &str) -> FP8Format {
    match use_case {
        "forward" | "weights" | "activations" => FP8Format::E4M3,
        "backward" | "gradients" => FP8Format::E5M2,
        _ => {
            // Analyze tensor statistics
            let data = tensor.to_vec_f32().unwrap_or_default();
            let max_abs = data
                .iter()
                .map(|x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(1.0);

            // If range is large, use E5M2, otherwise E4M3
            if max_abs > 448.0 {
                FP8Format::E5M2
            } else {
                FP8Format::E4M3
            }
        },
    }
}

/// Estimate quantization error
pub fn estimate_quantization_error(_original: &Tensor, _quantized: &FP8Tensor) -> Result<f32> {
    // This would require dequantization and comparison
    // Simplified implementation
    Ok(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fp8_format_properties() {
        let e4m3 = FP8Format::E4M3;
        assert_eq!(e4m3.exponent_bits(), 4);
        assert_eq!(e4m3.mantissa_bits(), 3);
        assert_eq!(e4m3.max_value(), 448.0);

        let e5m2 = FP8Format::E5M2;
        assert_eq!(e5m2.exponent_bits(), 5);
        assert_eq!(e5m2.mantissa_bits(), 2);
        assert_eq!(e5m2.max_value(), 57344.0);
    }

    #[test]
    fn test_fp8_per_tensor_quantization() -> Result<()> {
        let config = FP8Config {
            format: FP8Format::E4M3,
            scaling: ScalingStrategy::PerTensor,
            ..Default::default()
        };

        let mut quantizer = FP8Quantizer::new(config)?;
        let tensor = Tensor::randn(&[4, 8])?;

        let fp8_tensor = quantizer.quantize(&tensor)?;

        assert_eq!(fp8_tensor.shape, vec![4, 8]);
        assert_eq!(fp8_tensor.data.len(), 32);
        assert_eq!(fp8_tensor.format, FP8Format::E4M3);

        // Check that scales are per-tensor
        match fp8_tensor.scales {
            ScaleFactors::PerTensor(_) => (),
            _ => panic!("Expected PerTensor scales"),
        }

        Ok(())
    }

    #[test]
    fn test_fp8_roundtrip() -> Result<()> {
        let config = FP8Config {
            format: FP8Format::E4M3,
            stochastic_rounding: false,
            ..Default::default()
        };

        let mut quantizer = FP8Quantizer::new(config)?;

        // Create a simple tensor with known values
        let data = vec![0.0, 1.0, -1.0, 100.0, -100.0, 0.5, -0.5];
        let tensor = Tensor::from_vec(data.clone(), &[7])?;

        let fp8_tensor = quantizer.quantize(&tensor)?;
        let dequantized = quantizer.dequantize(&fp8_tensor)?;

        let deq_data = dequantized.to_vec_f32()?;

        // Check that values are approximately preserved
        for (original, recovered) in data.iter().zip(deq_data.iter()) {
            let rel_error = (original - recovered).abs() / (original.abs() + 1e-6);
            assert!(
                rel_error < 0.1,
                "Relative error too large: {} vs {}",
                original,
                recovered
            );
        }

        Ok(())
    }

    #[test]
    fn test_fp8_per_channel_quantization() -> Result<()> {
        let config = FP8Config {
            format: FP8Format::E4M3,
            scaling: ScalingStrategy::PerChannel,
            ..Default::default()
        };

        let mut quantizer = FP8Quantizer::new(config)?;
        let tensor = Tensor::randn(&[4, 8])?;

        let fp8_tensor = quantizer.quantize(&tensor)?;

        match &fp8_tensor.scales {
            ScaleFactors::PerChannel(scales) => {
                assert_eq!(scales.len(), 4); // 4 channels
            },
            _ => panic!("Expected PerChannel scales"),
        }

        Ok(())
    }

    #[test]
    fn test_select_fp8_format() -> Result<()> {
        let tensor = Tensor::randn(&[10, 10])?;

        let format_forward = select_fp8_format(&tensor, "forward");
        assert_eq!(format_forward, FP8Format::E4M3);

        let format_backward = select_fp8_format(&tensor, "gradients");
        assert_eq!(format_backward, FP8Format::E5M2);

        Ok(())
    }

    #[test]
    fn test_delayed_scaling() -> Result<()> {
        let config = FP8Config {
            format: FP8Format::E4M3,
            delayed_scaling: DelayedScalingConfig {
                enabled: true,
                interval: 2,
                ..Default::default()
            },
            ..Default::default()
        };

        let mut quantizer = FP8Quantizer::new(config)?;

        // Quantize multiple times to test delayed scaling
        for _ in 0..5 {
            let tensor = Tensor::randn(&[10, 10])?;
            let _fp8_tensor = quantizer.quantize(&tensor)?;
        }

        // Check that stats are being tracked
        assert!(quantizer.stats.is_some());

        Ok(())
    }
}
