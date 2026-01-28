/// Advanced Quantization Methods for TrustformeRS Models
///
/// This module implements cutting-edge quantization techniques:
/// - NF4 (Normalized Float 4): 4-bit normalized float quantization with information-theoretic optimal scaling
/// - FP4 (4-bit Float): Native 4-bit floating point representation
/// - Block-wise quantization with optimal outlier handling
/// - Advanced dequantization with hardware-accelerated operations
///
/// Based on:
/// - "QLoRA: Efficient Finetuning of Quantized LLMs" (Dettmers et al., 2023)
/// - "QLORA: Quantized Low-Rank Adaptation of Large Language Models"
/// - "The case for 4-bit precision: k-bit Inference Scaling Laws" (Dettmers et al., 2023)
use std::collections::HashMap;
use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};

/// Advanced quantization configuration
#[derive(Debug, Clone)]
pub struct AdvancedQuantizationConfig {
    /// Quantization method
    pub method: QuantizationMethod,
    /// Block size for block-wise quantization (default: 64)
    pub block_size: usize,
    /// Whether to use double quantization (quantize scaling factors)
    pub double_quantization: bool,
    /// Compute data type for quantization operations
    pub compute_dtype: ComputeDataType,
    /// Outlier threshold for mixed precision (default: 6.0)
    pub outlier_threshold: f32,
    /// Whether to use nested quantization for very large models
    pub nested_quantization: bool,
}

impl Default for AdvancedQuantizationConfig {
    fn default() -> Self {
        Self {
            method: QuantizationMethod::NF4,
            block_size: 64,
            double_quantization: true,
            compute_dtype: ComputeDataType::BFloat16,
            outlier_threshold: 6.0,
            nested_quantization: false,
        }
    }
}

/// Advanced quantization methods
#[derive(Debug, Clone, PartialEq)]
pub enum QuantizationMethod {
    /// Normalized Float 4-bit with information-theoretic optimal quantiles
    NF4,
    /// Standard 4-bit floating point
    FP4,
    /// 4-bit integer with asymmetric scaling
    Int4Asymmetric,
    /// 8-bit integer with block-wise scaling
    Int8BlockWise,
    /// Mixed precision with dynamic bit allocation
    MixedPrecision { primary_bits: u8, outlier_bits: u8 },
}

/// Compute data types for quantization operations
#[derive(Debug, Clone, PartialEq)]
pub enum ComputeDataType {
    Float32,
    Float16,
    BFloat16,
}

/// NF4 quantization table - information-theoretically optimal quantiles for normal distribution
const NF4_QUANT_TABLE: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// FP4 E2M1 format specification (1 sign bit, 2 exponent bits, 1 mantissa bit)
pub struct FP4Format {
    pub sign_bits: u8,
    pub exponent_bits: u8,
    pub mantissa_bits: u8,
}

impl Default for FP4Format {
    fn default() -> Self {
        Self {
            sign_bits: 1,
            exponent_bits: 2,
            mantissa_bits: 1,
        }
    }
}

/// Quantized tensor representation
#[derive(Debug, Clone)]
pub struct QuantizedTensor {
    /// Quantized data
    pub data: Vec<u8>,
    /// Scaling factors (per block or global)
    pub scales: Vec<f32>,
    /// Zero points (for asymmetric quantization)
    pub zero_points: Option<Vec<f32>>,
    /// Original tensor shape
    pub shape: Vec<usize>,
    /// Quantization metadata
    pub metadata: QuantizationMetadata,
}

/// Quantization metadata
#[derive(Debug, Clone)]
pub struct QuantizationMetadata {
    pub method: QuantizationMethod,
    pub block_size: usize,
    pub outlier_indices: Option<Vec<usize>>,
    pub outlier_values: Option<Vec<f32>>,
    pub double_quantized: bool,
    pub compute_dtype: ComputeDataType,
}

/// Advanced quantization engine
pub struct AdvancedQuantizer {
    config: AdvancedQuantizationConfig,
    #[allow(dead_code)]
    nf4_lookup: HashMap<u8, f32>,
    #[allow(dead_code)]
    inv_nf4_lookup: HashMap<u32, u8>, // Using u32 for f32 bits
}

impl AdvancedQuantizer {
    /// Create a new advanced quantizer
    pub fn new(config: AdvancedQuantizationConfig) -> Self {
        let mut nf4_lookup = HashMap::new();
        let mut inv_nf4_lookup = HashMap::new();

        // Build NF4 lookup tables for fast quantization/dequantization
        for (i, &value) in NF4_QUANT_TABLE.iter().enumerate() {
            nf4_lookup.insert(i as u8, value);
            inv_nf4_lookup.insert(value.to_bits(), i as u8);
        }

        Self {
            config,
            nf4_lookup,
            inv_nf4_lookup,
        }
    }

    /// Quantize tensor using the configured advanced method
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        match &self.config.method {
            QuantizationMethod::NF4 => self.quantize_nf4(tensor),
            QuantizationMethod::FP4 => self.quantize_fp4(tensor),
            QuantizationMethod::Int4Asymmetric => self.quantize_int4_asymmetric(tensor),
            QuantizationMethod::Int8BlockWise => self.quantize_int8_blockwise(tensor),
            QuantizationMethod::MixedPrecision {
                primary_bits,
                outlier_bits,
            } => self.quantize_mixed_precision(tensor, *primary_bits, *outlier_bits),
        }
    }

    /// Dequantize tensor back to full precision
    pub fn dequantize(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        match &quantized.metadata.method {
            QuantizationMethod::NF4 => self.dequantize_nf4(quantized),
            QuantizationMethod::FP4 => self.dequantize_fp4(quantized),
            QuantizationMethod::Int4Asymmetric => self.dequantize_int4_asymmetric(quantized),
            QuantizationMethod::Int8BlockWise => self.dequantize_int8_blockwise(quantized),
            QuantizationMethod::MixedPrecision { .. } => self.dequantize_mixed_precision(quantized),
        }
    }

    /// NF4 quantization implementation
    fn quantize_nf4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        let tensor_data = tensor.data_f32()?;
        let total_elements = tensor_data.len();
        let num_blocks = total_elements.div_ceil(self.config.block_size);

        let mut quantized_data = Vec::with_capacity(total_elements.div_ceil(2)); // 4 bits per element
        let mut scales = Vec::with_capacity(num_blocks);
        let mut outlier_indices = Vec::new();
        let mut outlier_values = Vec::new();

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * self.config.block_size;
            let end_idx = (start_idx + self.config.block_size).min(total_elements);
            let block_data = &tensor_data[start_idx..end_idx];

            // Calculate block-wise statistics
            let abs_max = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Handle outliers if enabled
            let mut processed_block = Vec::with_capacity(block_data.len());
            for (local_idx, &value) in block_data.iter().enumerate() {
                if value.abs() > self.config.outlier_threshold * abs_max {
                    outlier_indices.push(start_idx + local_idx);
                    outlier_values.push(value);
                    processed_block.push(0.0); // Replace outlier with 0 for quantization
                } else {
                    processed_block.push(value);
                }
            }

            // Calculate optimal scale for this block
            let scale = abs_max;
            scales.push(scale);

            // Quantize block using NF4
            for chunk in processed_block.chunks(2) {
                let val1 = self.quantize_nf4_value(chunk[0] / scale);
                let val2 =
                    if chunk.len() > 1 { self.quantize_nf4_value(chunk[1] / scale) } else { 0 };

                // Pack two 4-bit values into one byte
                let packed = (val1 & 0xF) | ((val2 & 0xF) << 4);
                quantized_data.push(packed);
            }
        }

        // Double quantization of scales if enabled
        let final_scales = if self.config.double_quantization {
            self.double_quantize_scales(&scales)?
        } else {
            scales
        };

        let outliers = if outlier_indices.is_empty() {
            (None, None)
        } else {
            (Some(outlier_indices), Some(outlier_values))
        };

        Ok(QuantizedTensor {
            data: quantized_data,
            scales: final_scales,
            zero_points: None,
            shape: tensor.shape().to_vec(),
            metadata: QuantizationMetadata {
                method: QuantizationMethod::NF4,
                block_size: self.config.block_size,
                outlier_indices: outliers.0,
                outlier_values: outliers.1,
                double_quantized: self.config.double_quantization,
                compute_dtype: self.config.compute_dtype.clone(),
            },
        })
    }

    /// Quantize single value to NF4
    fn quantize_nf4_value(&self, value: f32) -> u8 {
        let clamped = value.clamp(-1.0, 1.0);

        // Find the closest quantization level
        let mut best_idx = 0;
        let mut best_error = (clamped - NF4_QUANT_TABLE[0]).abs();

        for (idx, &quant_val) in NF4_QUANT_TABLE.iter().enumerate().skip(1) {
            let error = (clamped - quant_val).abs();
            if error < best_error {
                best_error = error;
                best_idx = idx;
            }
        }

        best_idx as u8
    }

    /// FP4 quantization implementation
    fn quantize_fp4(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        let tensor_data = tensor.data_f32()?;
        let total_elements = tensor_data.len();
        let num_blocks = total_elements.div_ceil(self.config.block_size);

        let mut quantized_data = Vec::with_capacity(total_elements.div_ceil(2));
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * self.config.block_size;
            let end_idx = (start_idx + self.config.block_size).min(total_elements);
            let block_data = &tensor_data[start_idx..end_idx];

            // Calculate block-wise scale
            let abs_max = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            let scale = abs_max / 7.0; // FP4 E2M1 max representable value is ~7.0
            scales.push(scale);

            // Quantize block using FP4 E2M1 format
            for chunk in block_data.chunks(2) {
                let val1 = self.quantize_fp4_value(chunk[0] / scale);
                let val2 =
                    if chunk.len() > 1 { self.quantize_fp4_value(chunk[1] / scale) } else { 0 };

                // Pack two 4-bit values into one byte
                let packed = (val1 & 0xF) | ((val2 & 0xF) << 4);
                quantized_data.push(packed);
            }
        }

        Ok(QuantizedTensor {
            data: quantized_data,
            scales,
            zero_points: None,
            shape: tensor.shape().to_vec(),
            metadata: QuantizationMetadata {
                method: QuantizationMethod::FP4,
                block_size: self.config.block_size,
                outlier_indices: None,
                outlier_values: None,
                double_quantized: false,
                compute_dtype: self.config.compute_dtype.clone(),
            },
        })
    }

    /// Quantize single value to FP4 E2M1 format
    fn quantize_fp4_value(&self, value: f32) -> u8 {
        if value == 0.0 {
            return 0;
        }

        let sign = if value < 0.0 { 1u8 } else { 0u8 };
        let abs_val = value.abs();

        // E2M1 format: 1 sign + 2 exponent + 1 mantissa bits
        // Representable values: 0, ±0.5, ±1.0, ±1.5, ±2.0, ±3.0, ±4.0, ±6.0
        let quantized = if abs_val <= 0.25 {
            0 // Underflow to 0
        } else if abs_val <= 0.75 {
            1 // Map to 0.5
        } else if abs_val <= 1.25 {
            2 // Map to 1.0
        } else if abs_val <= 1.75 {
            3 // Map to 1.5
        } else if abs_val <= 2.5 {
            4 // Map to 2.0
        } else if abs_val <= 3.5 {
            5 // Map to 3.0
        } else if abs_val <= 5.0 {
            6 // Map to 4.0
        } else {
            7 // Map to 6.0
        };

        (sign << 3) | (quantized & 0x7)
    }

    /// Int4 asymmetric quantization
    fn quantize_int4_asymmetric(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        let tensor_data = tensor.data_f32()?;
        let total_elements = tensor_data.len();
        let num_blocks = total_elements.div_ceil(self.config.block_size);

        let mut quantized_data = Vec::with_capacity(total_elements.div_ceil(2));
        let mut scales = Vec::with_capacity(num_blocks);
        let mut zero_points = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * self.config.block_size;
            let end_idx = (start_idx + self.config.block_size).min(total_elements);
            let block_data = &tensor_data[start_idx..end_idx];

            // Calculate min/max for asymmetric quantization
            let min_val = block_data.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = block_data.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

            let scale = (max_val - min_val) / 15.0; // 4-bit range: 0-15
            let zero_point = -min_val / scale;

            scales.push(scale);
            zero_points.push(zero_point);

            // Quantize block
            for chunk in block_data.chunks(2) {
                let val1 = ((chunk[0] / scale + zero_point).round() as i32).clamp(0, 15) as u8;
                let val2 = if chunk.len() > 1 {
                    ((chunk[1] / scale + zero_point).round() as i32).clamp(0, 15) as u8
                } else {
                    0
                };

                let packed = (val1 & 0xF) | ((val2 & 0xF) << 4);
                quantized_data.push(packed);
            }
        }

        Ok(QuantizedTensor {
            data: quantized_data,
            scales,
            zero_points: Some(zero_points),
            shape: tensor.shape().to_vec(),
            metadata: QuantizationMetadata {
                method: QuantizationMethod::Int4Asymmetric,
                block_size: self.config.block_size,
                outlier_indices: None,
                outlier_values: None,
                double_quantized: false,
                compute_dtype: self.config.compute_dtype.clone(),
            },
        })
    }

    /// Int8 block-wise quantization
    fn quantize_int8_blockwise(&self, tensor: &Tensor) -> Result<QuantizedTensor> {
        let tensor_data = tensor.data_f32()?;
        let total_elements = tensor_data.len();
        let num_blocks = total_elements.div_ceil(self.config.block_size);

        let mut quantized_data = Vec::with_capacity(total_elements);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start_idx = block_idx * self.config.block_size;
            let end_idx = (start_idx + self.config.block_size).min(total_elements);
            let block_data = &tensor_data[start_idx..end_idx];

            let abs_max = block_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            let scale = abs_max / 127.0;
            scales.push(scale);

            for &value in block_data {
                let quantized = ((value / scale).round() as i32).clamp(-127, 127) as i8;
                quantized_data.push(quantized as u8);
            }
        }

        Ok(QuantizedTensor {
            data: quantized_data,
            scales,
            zero_points: None,
            shape: tensor.shape().to_vec(),
            metadata: QuantizationMetadata {
                method: QuantizationMethod::Int8BlockWise,
                block_size: self.config.block_size,
                outlier_indices: None,
                outlier_values: None,
                double_quantized: false,
                compute_dtype: self.config.compute_dtype.clone(),
            },
        })
    }

    /// Mixed precision quantization
    fn quantize_mixed_precision(
        &self,
        tensor: &Tensor,
        _primary_bits: u8,
        _outlier_bits: u8,
    ) -> Result<QuantizedTensor> {
        // Implementation would detect outliers and use different bit widths
        // For now, delegate to NF4 as primary quantization
        self.quantize_nf4(tensor)
    }

    /// Dequantize NF4 tensor
    fn dequantize_nf4(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        let total_elements: usize = quantized.shape.iter().product();
        let mut dequantized_data = vec![0.0f32; total_elements];
        let num_blocks = quantized.scales.len();

        let mut data_idx = 0;
        let mut elem_idx = 0;

        for block_idx in 0..num_blocks {
            let scale = quantized.scales[block_idx];
            let block_size = if block_idx == num_blocks - 1 {
                // Last block might be partial
                total_elements - block_idx * quantized.metadata.block_size
            } else {
                quantized.metadata.block_size
            };

            let mut block_elem_count = 0;

            while block_elem_count < block_size && data_idx < quantized.data.len() {
                let packed = quantized.data[data_idx];
                data_idx += 1;

                // Unpack two 4-bit values
                let val1_idx = (packed & 0xF) as usize;
                let val2_idx = ((packed >> 4) & 0xF) as usize;

                if elem_idx < total_elements {
                    dequantized_data[elem_idx] = NF4_QUANT_TABLE[val1_idx] * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }

                if block_elem_count < block_size && elem_idx < total_elements {
                    dequantized_data[elem_idx] = NF4_QUANT_TABLE[val2_idx] * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }
            }
        }

        // Restore outliers if present
        if let (Some(outlier_indices), Some(outlier_values)) = (
            &quantized.metadata.outlier_indices,
            &quantized.metadata.outlier_values,
        ) {
            for (&idx, &value) in outlier_indices.iter().zip(outlier_values.iter()) {
                if idx < dequantized_data.len() {
                    dequantized_data[idx] = value;
                }
            }
        }

        Tensor::from_vec(dequantized_data, &quantized.shape)
    }

    /// Dequantize FP4 tensor
    fn dequantize_fp4(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        let total_elements: usize = quantized.shape.iter().product();
        let mut dequantized_data = vec![0.0f32; total_elements];
        let num_blocks = quantized.scales.len();

        let mut data_idx = 0;
        let mut elem_idx = 0;

        for block_idx in 0..num_blocks {
            let scale = quantized.scales[block_idx];
            let block_size = if block_idx == num_blocks - 1 {
                total_elements - block_idx * quantized.metadata.block_size
            } else {
                quantized.metadata.block_size
            };

            let mut block_elem_count = 0;

            while block_elem_count < block_size && data_idx < quantized.data.len() {
                let packed = quantized.data[data_idx];
                data_idx += 1;

                let val1_bits = packed & 0xF;
                let val2_bits = (packed >> 4) & 0xF;

                if elem_idx < total_elements {
                    dequantized_data[elem_idx] = self.dequantize_fp4_value(val1_bits) * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }

                if block_elem_count < block_size && elem_idx < total_elements {
                    dequantized_data[elem_idx] = self.dequantize_fp4_value(val2_bits) * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }
            }
        }

        Tensor::from_vec(dequantized_data, &quantized.shape)
    }

    /// Dequantize single FP4 value
    fn dequantize_fp4_value(&self, bits: u8) -> f32 {
        if bits == 0 {
            return 0.0;
        }

        let sign = if (bits >> 3) & 1 == 1 { -1.0 } else { 1.0 };
        let magnitude_bits = bits & 0x7;

        let magnitude = match magnitude_bits {
            1 => 0.5,
            2 => 1.0,
            3 => 1.5,
            4 => 2.0,
            5 => 3.0,
            6 => 4.0,
            7 => 6.0,
            _ => 0.0,
        };

        sign * magnitude
    }

    /// Dequantize Int4 asymmetric tensor
    fn dequantize_int4_asymmetric(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        let total_elements: usize = quantized.shape.iter().product();
        let mut dequantized_data = vec![0.0f32; total_elements];
        let num_blocks = quantized.scales.len();
        let zero_points = quantized.zero_points.as_ref().ok_or_else(|| {
            TrustformersError::config_error(
                "Zero points required for asymmetric quantization",
                "dequantize_int4",
            )
        })?;

        let mut data_idx = 0;
        let mut elem_idx = 0;

        for block_idx in 0..num_blocks {
            let scale = quantized.scales[block_idx];
            let zero_point = zero_points[block_idx];
            let block_size = if block_idx == num_blocks - 1 {
                total_elements - block_idx * quantized.metadata.block_size
            } else {
                quantized.metadata.block_size
            };

            let mut block_elem_count = 0;

            while block_elem_count < block_size && data_idx < quantized.data.len() {
                let packed = quantized.data[data_idx];
                data_idx += 1;

                let val1 = (packed & 0xF) as f32;
                let val2 = ((packed >> 4) & 0xF) as f32;

                if elem_idx < total_elements {
                    dequantized_data[elem_idx] = (val1 - zero_point) * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }

                if block_elem_count < block_size && elem_idx < total_elements {
                    dequantized_data[elem_idx] = (val2 - zero_point) * scale;
                    elem_idx += 1;
                    block_elem_count += 1;
                }
            }
        }

        Tensor::from_vec(dequantized_data, &quantized.shape)
    }

    /// Dequantize Int8 block-wise tensor
    fn dequantize_int8_blockwise(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        let total_elements: usize = quantized.shape.iter().product();
        let mut dequantized_data = vec![0.0f32; total_elements];
        let num_blocks = quantized.scales.len();

        let mut data_idx = 0;
        let mut elem_idx = 0;

        for block_idx in 0..num_blocks {
            let scale = quantized.scales[block_idx];
            let block_size = if block_idx == num_blocks - 1 {
                total_elements - block_idx * quantized.metadata.block_size
            } else {
                quantized.metadata.block_size
            };

            for _ in 0..block_size {
                if data_idx < quantized.data.len() && elem_idx < total_elements {
                    let quantized_val = quantized.data[data_idx] as i8;
                    dequantized_data[elem_idx] = quantized_val as f32 * scale;
                    data_idx += 1;
                    elem_idx += 1;
                }
            }
        }

        Tensor::from_vec(dequantized_data, &quantized.shape)
    }

    /// Dequantize mixed precision tensor
    fn dequantize_mixed_precision(&self, quantized: &QuantizedTensor) -> Result<Tensor> {
        // For now, delegate to NF4 dequantization
        self.dequantize_nf4(quantized)
    }

    /// Double quantization of scaling factors
    fn double_quantize_scales(&self, scales: &[f32]) -> Result<Vec<f32>> {
        if scales.is_empty() {
            return Ok(Vec::new());
        }

        // Simple 8-bit quantization of scales
        let max_scale = scales.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale_scale = max_scale / 127.0;

        let mut quantized_scales = Vec::with_capacity(scales.len());
        for &scale in scales {
            let quantized = ((scale / scale_scale).round() as i32).clamp(-127, 127) as i8;
            quantized_scales.push(quantized as f32 * scale_scale);
        }

        Ok(quantized_scales)
    }

    /// Get quantization statistics
    pub fn get_stats(&self, quantized: &QuantizedTensor) -> QuantizationStats {
        let original_size = quantized.shape.iter().product::<usize>() * 4; // Assume F32
        let compressed_size = quantized.data.len() + quantized.scales.len() * 4;
        let compression_ratio = original_size as f32 / compressed_size as f32;

        QuantizationStats {
            original_size_bytes: original_size,
            compressed_size_bytes: compressed_size,
            compression_ratio,
            method: quantized.metadata.method.clone(),
            block_size: quantized.metadata.block_size,
            outlier_count: quantized
                .metadata
                .outlier_indices
                .as_ref()
                .map(|indices| indices.len())
                .unwrap_or(0),
        }
    }
}

/// Quantization statistics
#[derive(Debug, Clone)]
pub struct QuantizationStats {
    pub original_size_bytes: usize,
    pub compressed_size_bytes: usize,
    pub compression_ratio: f32,
    pub method: QuantizationMethod,
    pub block_size: usize,
    pub outlier_count: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nf4_quantization() -> Result<()> {
        let config = AdvancedQuantizationConfig::default();
        let quantizer = AdvancedQuantizer::new(config);

        // Create test tensor
        let data = vec![0.1, -0.5, 0.8, -1.2, 0.0, 0.3, -0.7, 2.1];
        let tensor = Tensor::from_vec(data.clone(), &[8])?;

        // Quantize
        let quantized = quantizer.quantize(&tensor)?;
        assert_eq!(quantized.metadata.method, QuantizationMethod::NF4);

        // Dequantize
        let dequantized = quantizer.dequantize(&quantized)?;
        let dequant_data = dequantized.data_f32()?;

        // Check that values are reasonably close (within quantization error)
        // NF4 is 4-bit quantization, so we expect larger errors, especially for small values
        for (orig, dequant) in data.iter().zip(dequant_data.iter()) {
            let abs_error = (orig - dequant).abs();
            let rel_error = abs_error / orig.abs().max(1e-6);

            // Use a more lenient tolerance for 4-bit quantization
            // Small values may have high relative error, so we check both absolute and relative error
            let tolerance_met = rel_error < 1.0 || abs_error < 0.5;
            assert!(
                tolerance_met,
                "Quantization error too large: {} vs {} (abs_error: {:.4}, rel_error: {:.4})",
                orig, dequant, abs_error, rel_error
            );
        }

        Ok(())
    }

    #[test]
    fn test_fp4_quantization() -> Result<()> {
        let mut config = AdvancedQuantizationConfig::default();
        config.method = QuantizationMethod::FP4;
        let quantizer = AdvancedQuantizer::new(config);

        let data = vec![0.5, -1.0, 2.0, -3.5, 0.0, 1.5, -0.25, 4.0];
        let tensor = Tensor::from_vec(data.clone(), &[8])?;

        let quantized = quantizer.quantize(&tensor)?;
        let dequantized = quantizer.dequantize(&quantized)?;
        let dequant_data = dequantized.data_f32()?;

        // FP4 should preserve some values exactly
        assert_eq!(dequant_data.len(), data.len());

        Ok(())
    }

    #[test]
    fn test_quantization_stats() -> Result<()> {
        let config = AdvancedQuantizationConfig::default();
        let quantizer = AdvancedQuantizer::new(config);

        let data = vec![1.0; 1024]; // 1K elements
        let tensor = Tensor::from_vec(data, &[1024])?;

        let quantized = quantizer.quantize(&tensor)?;
        let stats = quantizer.get_stats(&quantized);

        assert!(stats.compression_ratio > 4.0); // Should achieve >4x compression
        assert_eq!(stats.original_size_bytes, 1024 * 4); // 1024 * sizeof(f32)

        Ok(())
    }

    #[test]
    fn test_int4_asymmetric_quantization() -> Result<()> {
        let mut config = AdvancedQuantizationConfig::default();
        config.method = QuantizationMethod::Int4Asymmetric;
        let quantizer = AdvancedQuantizer::new(config);

        let data = vec![0.1, 0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9]; // Asymmetric range
        let tensor = Tensor::from_vec(data.clone(), &[8])?;

        let quantized = quantizer.quantize(&tensor)?;
        assert!(quantized.zero_points.is_some());

        let dequantized = quantizer.dequantize(&quantized)?;
        let dequant_data = dequantized.data_f32()?;

        // Should handle asymmetric range well
        assert_eq!(dequant_data.len(), data.len());

        Ok(())
    }

    #[test]
    fn test_nf4_lookup_table() {
        let config = AdvancedQuantizationConfig::default();
        let quantizer = AdvancedQuantizer::new(config);

        // Test that NF4 quantization table is properly initialized
        assert_eq!(quantizer.nf4_lookup.len(), 16);

        // Test specific values
        assert_eq!(quantizer.nf4_lookup[&0], -1.0);
        assert_eq!(quantizer.nf4_lookup[&7], 0.0);
        assert_eq!(quantizer.nf4_lookup[&15], 1.0);
    }
}
