//! Advanced GGML quantization formats: Q5 and Q6 variants
//!
//! This module implements higher-precision quantization formats that offer
//! better quality than Q4 variants while maintaining good compression ratios.

use crate::errors::Result;
use crate::tensor::Tensor;
use anyhow::anyhow;
use serde::{Deserialize, Serialize};

/// GGML quantization type with Q5 and Q6 variants
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GGMLQuantType {
    /// 5-bit quantization (5.5 bits per weight)
    Q5_0,
    /// 5-bit quantization with 16-bit scales (5.5 bits per weight)
    Q5_1,
    /// 5-bit quantization with 6-bit scales (5.5 bits per weight)
    Q5K,
    /// 6-bit quantization (6.5 bits per weight)
    Q6K,
}

impl GGMLQuantType {
    /// Get block size for this quantization type
    pub fn block_size(&self) -> usize {
        match self {
            Self::Q5_0 | Self::Q5_1 => 32,
            Self::Q5K | Self::Q6K => 256,
        }
    }

    /// Get bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            Self::Q5_0 | Self::Q5_1 | Self::Q5K => 5.5,
            Self::Q6K => 6.5625,
        }
    }
}

/// Q5_0 quantization block (32 weights packed into 22 bytes)
#[derive(Debug, Clone)]
pub struct BlockQ5_0 {
    /// Scale factor (FP16)
    pub d: F16,
    /// High bits for 32 values (4 bytes)
    pub qh: [u8; 4],
    /// Low 4 bits for 32 values (16 bytes)
    pub qs: [u8; 16],
}

/// Q5_1 quantization block (32 weights packed into 24 bytes)
#[derive(Debug, Clone)]
pub struct BlockQ5_1 {
    /// Scale factor (FP16)
    pub d: F16,
    /// Minimum value (FP16)
    pub m: F16,
    /// High bits for 32 values (4 bytes)
    pub qh: [u8; 4],
    /// Low 4 bits for 32 values (16 bytes)
    pub qs: [u8; 16],
}

/// Q5K quantization block (256 weights in super-blocks)
#[derive(Debug, Clone)]
#[allow(dead_code)] // Reserved for future GGML Q5K quantization support
pub struct BlockQ5K {
    /// Scale factors (8x FP16)
    pub d: [F16; 8],
    /// Minimum values (8x FP16)
    pub dmin: [F16; 8],
    /// 6-bit scales (12 bytes)
    pub scales: [u8; 12],
    /// High bits (32 bytes)
    pub qh: [u8; 32],
    /// Low 4 bits (128 bytes)
    pub qs: [u8; 128],
}

/// Q6K quantization block (256 weights in super-blocks)
#[derive(Debug, Clone)]
pub struct BlockQ6K {
    /// Scale factor (FP16)
    pub d: F16,
    /// 8-bit scales (16 values)
    pub scales: [u8; 16],
    /// Low 4 bits (128 bytes)
    pub ql: [u8; 128],
    /// High 2 bits (64 bytes)
    pub qh: [u8; 64],
}

/// Helper type for f16 (using u16 storage)
type F16 = u16;

/// Convert f32 to f16
fn f32_to_f16(val: f32) -> F16 {
    // Simplified conversion - in production use half crate
    let bits = val.to_bits();
    let sign = (bits >> 31) & 0x1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let frac = bits & 0x7FFFFF;

    if exp == 0xFF {
        // Inf or NaN
        ((sign << 15) | (0x1F << 10) | (frac >> 13)) as u16
    } else if exp == 0 {
        // Zero or denormal
        (sign << 15) as u16
    } else {
        let new_exp = exp - 127 + 15;
        if new_exp >= 0x1F {
            // Overflow to inf
            ((sign << 15) | (0x1F << 10)) as u16
        } else if new_exp <= 0 {
            // Underflow to zero
            (sign << 15) as u16
        } else {
            ((sign << 15) | ((new_exp as u32) << 10) | (frac >> 13)) as u16
        }
    }
}

/// Convert f16 to f32
fn f16_to_f32(val: F16) -> f32 {
    let sign = (val >> 15) & 0x1;
    let exp = (val >> 10) & 0x1F;
    let frac = val & 0x3FF;

    if exp == 0x1F {
        // Inf or NaN
        f32::from_bits(((sign as u32) << 31) | (0xFF << 23) | ((frac as u32) << 13))
    } else if exp == 0 {
        // Zero or denormal
        f32::from_bits((sign as u32) << 31)
    } else {
        let new_exp = (exp as i32) - 15 + 127;
        f32::from_bits(((sign as u32) << 31) | ((new_exp as u32) << 23) | ((frac as u32) << 13))
    }
}

/// Quantize tensor to Q5_0 format
pub fn quantize_q5_0(tensor: &Tensor) -> Result<Vec<BlockQ5_0>> {
    let values: Vec<f32> = match tensor {
        Tensor::F32(data) => {
            data.as_slice().ok_or_else(|| anyhow!("Failed to get tensor data"))?.to_vec()
        },
        Tensor::F64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::F16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::BF16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::I64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::C32(_) => {
            return Err(anyhow!("Complex32 tensors not yet supported for quantization").into())
        },
        Tensor::C64(_) => {
            return Err(anyhow!("Complex64 tensors not yet supported for quantization").into())
        },
        Tensor::CF16(_) => {
            return Err(anyhow!("Complex16 tensors not yet supported for quantization").into())
        },
        Tensor::CBF16(_) => {
            return Err(
                anyhow!("Complex BFloat16 tensors not yet supported for quantization").into(),
            )
        },
        Tensor::Sparse(_) => {
            return Err(anyhow!("Sparse tensors not yet supported for quantization").into())
        },
        #[cfg(feature = "torch")]
        Tensor::Torch(_) => return Err(anyhow!("Torch tensors not yet supported").into()),
        #[cfg(feature = "candle")]
        Tensor::Candle(_) => return Err(anyhow!("Candle tensors not yet supported").into()),
        #[cfg(feature = "metal")]
        Tensor::Metal(_) => return Err(anyhow!("Metal tensors not yet supported for quantization").into()),
    };

    let n = values.len();
    let nb = n / 32; // Number of blocks
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let start = i * 32;
        let block_data = &values[start..start + 32];

        // Find absolute maximum for symmetric quantization
        let amax = block_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));

        // Calculate scale for Q5_0 (symmetric, 5 bits signed)
        let scale = amax / 15.0;
        let iscale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        // Quantize values
        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for j in 0..32 {
            let val = block_data[j];
            let qi = (val * iscale + 15.5) as i32;
            let qi = qi.clamp(0, 31) as u8;

            // Store low 4 bits
            if j % 2 == 0 {
                qs[j / 2] = qi & 0xF;
            } else {
                qs[j / 2] |= (qi & 0xF) << 4;
            }

            // Store high bit
            if qi & 0x10 != 0 {
                qh[j / 8] |= 1 << (j % 8);
            }
        }

        blocks.push(BlockQ5_0 {
            d: f32_to_f16(scale),
            qh,
            qs,
        });
    }

    Ok(blocks)
}

/// Dequantize Q5_0 blocks back to f32
pub fn dequantize_q5_0(blocks: &[BlockQ5_0], shape: &[usize]) -> Result<Tensor> {
    let n = blocks.len() * 32;
    let mut values = vec![0.0f32; n];

    for (i, block) in blocks.iter().enumerate() {
        let scale = f16_to_f32(block.d);
        let offset = i * 32;

        for j in 0..32 {
            // Extract 5-bit value
            let ql = if j % 2 == 0 { block.qs[j / 2] & 0xF } else { block.qs[j / 2] >> 4 };

            let qh = if block.qh[j / 8] & (1 << (j % 8)) != 0 { 0x10 } else { 0x00 };

            let qi = ql | qh;
            values[offset + j] = ((qi as f32) - 15.0) * scale;
        }
    }

    Tensor::from_vec(values, shape)
}

/// Quantize tensor to Q5_1 format
pub fn quantize_q5_1(tensor: &Tensor) -> Result<Vec<BlockQ5_1>> {
    let values: Vec<f32> = match tensor {
        Tensor::F32(data) => {
            data.as_slice().ok_or_else(|| anyhow!("Failed to get tensor data"))?.to_vec()
        },
        Tensor::F64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::F16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::BF16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::I64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::C32(_) => {
            return Err(anyhow!("Complex32 tensors not yet supported for quantization").into())
        },
        Tensor::C64(_) => {
            return Err(anyhow!("Complex64 tensors not yet supported for quantization").into())
        },
        Tensor::CF16(_) => {
            return Err(anyhow!("Complex16 tensors not yet supported for quantization").into())
        },
        Tensor::CBF16(_) => {
            return Err(
                anyhow!("Complex BFloat16 tensors not yet supported for quantization").into(),
            )
        },
        Tensor::Sparse(_) => {
            return Err(anyhow!("Sparse tensors not yet supported for quantization").into())
        },
        #[cfg(feature = "torch")]
        Tensor::Torch(_) => return Err(anyhow!("Torch tensors not yet supported").into()),
        #[cfg(feature = "candle")]
        Tensor::Candle(_) => return Err(anyhow!("Candle tensors not yet supported").into()),
        #[cfg(feature = "metal")]
        Tensor::Metal(_) => return Err(anyhow!("Metal tensors not yet supported for quantization").into()),
    };

    let n = values.len();
    let nb = n / 32;
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let start = i * 32;
        let block_data = &values[start..start + 32];

        // Find min and max
        let min = block_data.iter().fold(f32::MAX, |a, &b| a.min(b));
        let max = block_data.iter().fold(f32::MIN, |a, &b| a.max(b));

        // Calculate scale
        let scale = (max - min) / 31.0;
        let iscale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        // Quantize values
        let mut qs = [0u8; 16];
        let mut qh = [0u8; 4];

        for j in 0..32 {
            let val = block_data[j];
            let qi = ((val - min) * iscale + 0.5) as i32;
            let qi = qi.clamp(0, 31) as u8;

            // Store low 4 bits
            if j % 2 == 0 {
                qs[j / 2] = qi & 0xF;
            } else {
                qs[j / 2] |= (qi & 0xF) << 4;
            }

            // Store high bit
            if qi & 0x10 != 0 {
                qh[j / 8] |= 1 << (j % 8);
            }
        }

        blocks.push(BlockQ5_1 {
            d: f32_to_f16(scale),
            m: f32_to_f16(min),
            qh,
            qs,
        });
    }

    Ok(blocks)
}

/// Quantize tensor to Q6K format
pub fn quantize_q6_k(tensor: &Tensor) -> Result<Vec<BlockQ6K>> {
    let values: Vec<f32> = match tensor {
        Tensor::F32(data) => {
            data.as_slice().ok_or_else(|| anyhow!("Failed to get tensor data"))?.to_vec()
        },
        Tensor::F64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::F16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::BF16(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v.to_f32())
            .collect(),
        Tensor::I64(data) => data
            .as_slice()
            .ok_or_else(|| anyhow!("Failed to get tensor data"))?
            .iter()
            .map(|&v| v as f32)
            .collect(),
        Tensor::C32(_) => {
            return Err(anyhow!("Complex32 tensors not yet supported for quantization").into())
        },
        Tensor::C64(_) => {
            return Err(anyhow!("Complex64 tensors not yet supported for quantization").into())
        },
        Tensor::CF16(_) => {
            return Err(anyhow!("Complex16 tensors not yet supported for quantization").into())
        },
        Tensor::CBF16(_) => {
            return Err(
                anyhow!("Complex BFloat16 tensors not yet supported for quantization").into(),
            )
        },
        Tensor::Sparse(_) => {
            return Err(anyhow!("Sparse tensors not yet supported for quantization").into())
        },
        #[cfg(feature = "torch")]
        Tensor::Torch(_) => return Err(anyhow!("Torch tensors not yet supported").into()),
        #[cfg(feature = "candle")]
        Tensor::Candle(_) => return Err(anyhow!("Candle tensors not yet supported").into()),
        #[cfg(feature = "metal")]
        Tensor::Metal(_) => return Err(anyhow!("Metal tensors not yet supported for quantization").into()),
    };

    let n = values.len();
    let nb = n / 256; // Super-blocks of 256
    let mut blocks = Vec::with_capacity(nb);

    for i in 0..nb {
        let start = i * 256;
        let block_data = &values[start..start + 256];

        // Find global scale
        let max = block_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
        let scale = max / 127.0;
        let iscale = if scale != 0.0 { 1.0 / scale } else { 0.0 };

        let mut scales = [0u8; 16];
        let mut ql = [0u8; 128];
        let mut qh = [0u8; 64];

        // Process 16 sub-blocks of 16 values each
        for (sb, scale_ref) in scales.iter_mut().enumerate() {
            let sb_start = sb * 16;
            let sb_data = &block_data[sb_start..sb_start + 16];

            // Find sub-block scale
            let sb_max = sb_data.iter().fold(0.0f32, |a, &b| a.max(b.abs()));
            let sb_scale = sb_max * iscale;
            *scale_ref = (sb_scale * 63.0 + 0.5) as u8;

            // Quantize sub-block
            for (j, &val) in sb_data.iter().enumerate() {
                let scaled = val * iscale * 63.0 / (*scale_ref as f32 + 1e-10);
                let qi = (scaled + 32.5) as i32;
                let qi = qi.clamp(0, 63) as u8;

                // Store low 4 bits
                let idx = sb_start + j;
                if idx % 2 == 0 {
                    ql[idx / 2] = qi & 0xF;
                } else {
                    ql[idx / 2] |= (qi & 0xF) << 4;
                }

                // Store high 2 bits
                let qh_idx = idx / 4;
                let qh_shift = (idx % 4) * 2;
                qh[qh_idx] |= ((qi >> 4) & 0x3) << qh_shift;
            }
        }

        blocks.push(BlockQ6K {
            d: f32_to_f16(scale),
            scales,
            ql,
            qh,
        });
    }

    Ok(blocks)
}

/// Advanced GGML quantizer supporting Q5 and Q6 variants
pub struct AdvancedGGMLQuantizer {
    pub quant_type: GGMLQuantType,
}

impl AdvancedGGMLQuantizer {
    /// Create a new advanced GGML quantizer
    pub fn new(quant_type: GGMLQuantType) -> Self {
        Self { quant_type }
    }

    /// Quantize a tensor
    pub fn quantize(&self, tensor: &Tensor) -> Result<QuantizedGGMLTensor> {
        let shape = tensor.shape().to_vec();

        let data = match self.quant_type {
            GGMLQuantType::Q5_0 => {
                let blocks = quantize_q5_0(tensor)?;
                GGMLQuantData::Q5_0(blocks)
            },
            GGMLQuantType::Q5_1 => {
                let blocks = quantize_q5_1(tensor)?;
                GGMLQuantData::Q5_1(blocks)
            },
            GGMLQuantType::Q5K => {
                // For now, fall back to Q5_0 for Q5K
                // Full Q5K implementation would be more complex
                let blocks = quantize_q5_0(tensor)?;
                GGMLQuantData::Q5_0(blocks)
            },
            GGMLQuantType::Q6K => {
                let blocks = quantize_q6_k(tensor)?;
                GGMLQuantData::Q6K(blocks)
            },
        };

        Ok(QuantizedGGMLTensor {
            data,
            shape,
            quant_type: self.quant_type,
        })
    }

    /// Calculate compression ratio
    pub fn compression_ratio(&self, original_size: usize) -> f32 {
        let bits_per_weight = self.quant_type.bits_per_weight();
        let compressed_bits = (original_size as f32) * bits_per_weight;
        let original_bits = (original_size * 32) as f32; // 32 bits per f32
        original_bits / compressed_bits
    }
}

/// Quantized GGML tensor
#[derive(Debug, Clone)]
pub struct QuantizedGGMLTensor {
    pub data: GGMLQuantData,
    pub shape: Vec<usize>,
    pub quant_type: GGMLQuantType,
}

/// GGML quantized data variants
#[derive(Debug, Clone)]
pub enum GGMLQuantData {
    Q5_0(Vec<BlockQ5_0>),
    Q5_1(Vec<BlockQ5_1>),
    Q6K(Vec<BlockQ6K>),
}

impl QuantizedGGMLTensor {
    /// Dequantize back to f32 tensor
    pub fn dequantize(&self) -> Result<Tensor> {
        match &self.data {
            GGMLQuantData::Q5_0(blocks) => dequantize_q5_0(blocks, &self.shape),
            GGMLQuantData::Q5_1(blocks) => {
                // Implement Q5_1 dequantization
                let n = blocks.len() * 32;
                let mut values = vec![0.0f32; n];

                for (i, block) in blocks.iter().enumerate() {
                    let scale = f16_to_f32(block.d);
                    let min = f16_to_f32(block.m);
                    let offset = i * 32;

                    for j in 0..32 {
                        let ql =
                            if j % 2 == 0 { block.qs[j / 2] & 0xF } else { block.qs[j / 2] >> 4 };

                        let qh = if block.qh[j / 8] & (1 << (j % 8)) != 0 { 0x10 } else { 0x00 };

                        let qi = ql | qh;
                        values[offset + j] = min + (qi as f32) * scale;
                    }
                }

                Tensor::from_vec(values, &self.shape)
            },
            GGMLQuantData::Q6K(blocks) => {
                // Implement Q6K dequantization
                let n = blocks.len() * 256;
                let mut values = vec![0.0f32; n];

                for (i, block) in blocks.iter().enumerate() {
                    let scale = f16_to_f32(block.d);
                    let offset = i * 256;

                    for sb in 0..16 {
                        let sb_scale = (block.scales[sb] as f32) / 63.0;
                        let sb_start = sb * 16;

                        for j in 0..16 {
                            let idx = sb_start + j;

                            // Extract 6-bit value
                            let ql = if idx % 2 == 0 {
                                block.ql[idx / 2] & 0xF
                            } else {
                                block.ql[idx / 2] >> 4
                            };

                            let qh_idx = idx / 4;
                            let qh_shift = (idx % 4) * 2;
                            let qh = (block.qh[qh_idx] >> qh_shift) & 0x3;

                            let qi = ql | (qh << 4);
                            values[offset + idx] = (qi as f32 - 32.0) * scale * sb_scale;
                        }
                    }
                }

                Tensor::from_vec(values, &self.shape)
            },
        }
    }

    /// Get memory usage in bytes
    pub fn memory_usage(&self) -> usize {
        match &self.data {
            GGMLQuantData::Q5_0(blocks) => blocks.len() * 22,
            GGMLQuantData::Q5_1(blocks) => blocks.len() * 24,
            GGMLQuantData::Q6K(blocks) => blocks.len() * 210,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q5_0_quantization() {
        // Create test tensor
        let values: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let tensor = Tensor::from_vec(values.clone(), &[64]).unwrap();

        // Quantize
        let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
        let quantized = quantizer.quantize(&tensor).unwrap();

        // Check compression
        let compression = quantizer.compression_ratio(64);
        assert!(compression > 5.0 && compression < 6.0);

        // Dequantize and check error
        let dequantized = quantized.dequantize().unwrap();
        match dequantized {
            Tensor::F32(data) => {
                let deq_values = data.as_slice().unwrap();
                let max_error = values
                    .iter()
                    .zip(deq_values.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |a, b| a.max(b));

                assert!(max_error < 0.25); // Reasonable error for 5-bit quantization
            },
            _ => panic!("Unexpected tensor type"),
        }
    }

    #[test]
    fn test_q6_k_quantization() {
        // Create larger test tensor for super-blocks
        let values: Vec<f32> = (0..512).map(|i| (i as f32 * 0.01).sin()).collect();
        let tensor = Tensor::from_vec(values.clone(), &[512]).unwrap();

        // Quantize
        let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q6K);
        let quantized = quantizer.quantize(&tensor).unwrap();

        // Check memory usage
        let memory = quantized.memory_usage();
        assert!(memory < values.len() * 4); // Should be compressed

        // Dequantize and check
        let dequantized = quantized.dequantize().unwrap();
        match dequantized {
            Tensor::F32(data) => {
                let deq_values = data.as_slice().unwrap();
                assert_eq!(deq_values.len(), values.len());

                // Check reconstruction quality
                let mse: f32 =
                    values.iter().zip(deq_values.iter()).map(|(a, b)| (a - b).powi(2)).sum::<f32>()
                        / values.len() as f32;

                assert!(mse < 0.01); // Good quality for 6-bit
            },
            _ => panic!("Unexpected tensor type"),
        }
    }

    #[test]
    fn test_f64_i64_tensor_support() {
        use crate::tensor::DType;

        // Test F64 support
        let values_f32: Vec<f32> = (0..64).map(|i| i as f32 * 0.1).collect();
        let base_tensor_f64 = Tensor::from_vec(values_f32.clone(), &[64]).unwrap();
        let tensor_f64 = base_tensor_f64.to_dtype(DType::F64).unwrap();

        let quantizer = AdvancedGGMLQuantizer::new(GGMLQuantType::Q5_0);
        let quantized_f64 = quantizer.quantize(&tensor_f64).unwrap();
        let dequantized_f64 = quantized_f64.dequantize().unwrap();

        match dequantized_f64 {
            Tensor::F32(data) => {
                let deq_values = data.as_slice().unwrap();
                let max_error = values_f32
                    .iter()
                    .zip(deq_values.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |a, b| a.max(b));
                assert!(max_error < 0.25); // Reasonable error for F64->F32 conversion
            },
            _ => panic!("Unexpected tensor type"),
        }

        // Test I64 support
        let values_i32: Vec<f32> = (0..64).map(|i| i as f32).collect();
        let base_tensor_i64 = Tensor::from_vec(values_i32.clone(), &[64]).unwrap();
        let tensor_i64 = base_tensor_i64.to_dtype(DType::I64).unwrap();

        let quantized_i64 = quantizer.quantize(&tensor_i64).unwrap();
        let dequantized_i64 = quantized_i64.dequantize().unwrap();

        match dequantized_i64 {
            Tensor::F32(data) => {
                let deq_values = data.as_slice().unwrap();
                let max_error = values_i32
                    .iter()
                    .zip(deq_values.iter())
                    .map(|(a, b)| (a - b).abs())
                    .fold(0.0f32, |a, b| a.max(b));
                assert!(max_error < 2.1); // Reasonable error for I64->F32 conversion
            },
            _ => panic!("Unexpected tensor type"),
        }
    }
}
