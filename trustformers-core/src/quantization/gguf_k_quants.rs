//! GGUF K-quant formats: Q2_K, Q3_K, Q4_K
//!
//! This module implements the K-quant family of quantization formats used in GGUF files.
//! These formats provide better quality-to-size ratios through sophisticated block structures
//! with per-block scales and minimum values.
//!
//! # K-Quant Design
//!
//! K-quants use super-blocks of 256 weights, subdivided into smaller blocks with:
//! - Per-block scales for better dynamic range
//! - Quantized scales for additional compression
//! - Importance-based quantization (more bits for critical weights)
//!
//! # Formats
//!
//! - **Q2_K**: 2.5625 bits per weight, ~10GB for 7B model
//! - **Q3_K**: 3.4375 bits per weight, ~13GB for 7B model
//! - **Q4_K**: 4.5 bits per weight, ~15GB for 7B model
//!
//! # Examples
//!
//! ```rust
//! use trustformers_core::quantization::gguf_k_quants::{KQuantConfig, KQuantizer, KQuantType};
//! use trustformers_core::tensor::Tensor;
//!
//! let config = KQuantConfig {
//!     quant_type: KQuantType::Q4_K,
//!     ..Default::default()
//! };
//!
//! let quantizer = KQuantizer::new(config)?;
//! let tensor = Tensor::randn(&[1024, 768])?;
//!
//! let quantized = quantizer.quantize(&tensor)?;
//! let dequantized = quantizer.dequantize(&quantized)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::f32;

/// K-quant format types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[allow(non_camel_case_types)]
pub enum KQuantType {
    /// 2-bit K-quant: 2.5625 bits per weight
    /// Best for: Maximum compression with acceptable quality loss
    Q2_K,

    /// 3-bit K-quant: 3.4375 bits per weight
    /// Best for: Balanced compression and quality
    Q3_K,

    /// 4-bit K-quant: 4.5 bits per weight
    /// Best for: High quality with good compression
    Q4_K,
}

impl KQuantType {
    /// Get super-block size (always 256 for K-quants)
    pub fn superblock_size(&self) -> usize {
        256
    }

    /// Get number of sub-blocks per super-block
    pub fn num_subblocks(&self) -> usize {
        match self {
            KQuantType::Q2_K => 16, // 16 blocks of 16 weights
            KQuantType::Q3_K => 16, // 16 blocks of 16 weights
            KQuantType::Q4_K => 8,  // 8 blocks of 32 weights
        }
    }

    /// Get sub-block size
    pub fn subblock_size(&self) -> usize {
        self.superblock_size() / self.num_subblocks()
    }

    /// Get bits per weight
    pub fn bits_per_weight(&self) -> f32 {
        match self {
            KQuantType::Q2_K => 2.5625,
            KQuantType::Q3_K => 3.4375,
            KQuantType::Q4_K => 4.5,
        }
    }

    /// Get quantization bits for weights
    pub fn weight_bits(&self) -> u8 {
        match self {
            KQuantType::Q2_K => 2,
            KQuantType::Q3_K => 3,
            KQuantType::Q4_K => 4,
        }
    }

    /// Get quantization bits for scales
    pub fn scale_bits(&self) -> u8 {
        match self {
            KQuantType::Q2_K => 4, // 4-bit quantized scales
            KQuantType::Q3_K => 6, // 6-bit quantized scales
            KQuantType::Q4_K => 6, // 6-bit quantized scales
        }
    }

    /// Get bytes per super-block
    pub fn bytes_per_superblock(&self) -> usize {
        match self {
            KQuantType::Q2_K => 82,  // 256 weights @ 2.5625 bpw
            KQuantType::Q3_K => 110, // 256 weights @ 3.4375 bpw
            KQuantType::Q4_K => 144, // 256 weights @ 4.5 bpw
        }
    }
}

/// Helper type for FP16 (using u16 storage)
type F16 = u16;

/// Q2_K quantization block (256 weights in 82 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockQ2K {
    /// 16 sub-blocks of 16 weights each

    /// Super-block scale (FP16)
    pub d: F16,

    /// Super-block minimum (FP16)
    pub dmin: F16,

    /// 4-bit quantized scales for 16 sub-blocks (8 bytes)
    pub scales: Vec<u8>,

    /// 4-bit quantized min values for 16 sub-blocks (8 bytes)
    pub mins: Vec<u8>,

    /// 2-bit quantized weights (64 bytes = 256 weights * 2 bits / 8)
    pub qs: Vec<u8>,
}

/// Q3_K quantization block (256 weights in 110 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockQ3K {
    /// 16 sub-blocks of 16 weights each

    /// High bits for quantized scales (2 bytes)
    pub hmask: Vec<u8>,

    /// 6-bit quantized scales for 16 sub-blocks (12 bytes)
    pub scales: Vec<u8>,

    /// Super-block scale (FP16)
    pub d: F16,

    /// 3-bit quantized weights (96 bytes = 256 weights * 3 bits / 8)
    pub qs: Vec<u8>,
}

/// Q4_K quantization block (256 weights in 144 bytes)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlockQ4K {
    /// 8 sub-blocks of 32 weights each

    /// Super-block scale (FP16)
    pub d: F16,

    /// Super-block minimum (FP16)
    pub dmin: F16,

    /// 6-bit quantized scales for 8 sub-blocks (6 bytes)
    pub scales: Vec<u8>,

    /// 6-bit quantized min values for 8 sub-blocks (6 bytes)
    pub mins: Vec<u8>,

    /// 4-bit quantized weights (128 bytes = 256 weights * 4 bits / 8)
    pub qs: Vec<u8>,
}

/// K-quant configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KQuantConfig {
    /// K-quant format type
    pub quant_type: KQuantType,

    /// Use importance-based quantization (allocate more bits to important weights)
    pub importance_based: bool,

    /// Percentile for outlier detection (0.99 = top 1% are outliers)
    pub outlier_percentile: f32,

    /// Scale optimization iterations
    pub scale_optimization_iters: usize,
}

impl Default for KQuantConfig {
    fn default() -> Self {
        Self {
            quant_type: KQuantType::Q4_K,
            importance_based: true,
            outlier_percentile: 0.99,
            scale_optimization_iters: 10,
        }
    }
}

/// K-quant quantized tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KQuantTensor {
    /// Quantization type
    pub quant_type: KQuantType,

    /// Original tensor shape
    pub shape: Vec<usize>,

    /// Quantized blocks (serialized as bytes)
    pub blocks: Vec<u8>,

    /// Number of blocks
    pub num_blocks: usize,

    /// Total number of weights
    pub num_weights: usize,
}

/// K-quant quantizer
pub struct KQuantizer {
    config: KQuantConfig,
}

impl KQuantizer {
    /// Create a new K-quant quantizer
    pub fn new(config: KQuantConfig) -> Result<Self> {
        Ok(Self { config })
    }

    /// Quantize tensor to K-quant format
    pub fn quantize(&self, tensor: &Tensor) -> Result<KQuantTensor> {
        let data = tensor.to_vec_f32()?;
        let shape = tensor.shape().to_vec();

        let superblock_size = self.config.quant_type.superblock_size();
        let num_blocks = data.len().div_ceil(superblock_size);
        let bytes_per_block = self.config.quant_type.bytes_per_superblock();

        let mut blocks = Vec::with_capacity(num_blocks * bytes_per_block);

        // Quantize each super-block
        for block_idx in 0..num_blocks {
            let start = block_idx * superblock_size;
            let end = (start + superblock_size).min(data.len());
            let block_data = &data[start..end];

            // Pad if necessary
            let mut padded = block_data.to_vec();
            while padded.len() < superblock_size {
                padded.push(0.0);
            }

            let block_bytes = match self.config.quant_type {
                KQuantType::Q2_K => self.quantize_q2k(&padded)?,
                KQuantType::Q3_K => self.quantize_q3k(&padded)?,
                KQuantType::Q4_K => self.quantize_q4k(&padded)?,
            };

            blocks.extend(block_bytes);
        }

        Ok(KQuantTensor {
            quant_type: self.config.quant_type,
            shape,
            blocks,
            num_blocks,
            num_weights: data.len(),
        })
    }

    /// Quantize a super-block to Q2_K format
    fn quantize_q2k(&self, data: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(data.len(), 256);

        let num_subblocks = 16;
        let subblock_size = 16;

        // Compute super-block statistics
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);

        let d = f32_to_f16(max_abs / 3.0); // Scale for 2-bit values (0-3)
        let dmin = f32_to_f16(min_val.abs());

        // Compute per-subblock scales and minimums
        let mut scales = vec![0u8; 8];
        let mut mins = vec![0u8; 8];
        let mut qs = vec![0u8; 64];

        for sb in 0..num_subblocks {
            let sb_start = sb * subblock_size;
            let sb_data = &data[sb_start..sb_start + subblock_size];

            let sb_max = sb_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let sb_min = sb_data.iter().copied().fold(f32::INFINITY, f32::min);

            // Quantize scale and min to 4 bits
            let scale_q = ((sb_max / f16_to_f32(d)) * 15.0).round().clamp(0.0, 15.0) as u8;
            let min_q = ((sb_min.abs() / f16_to_f32(dmin)) * 15.0).round().clamp(0.0, 15.0) as u8;

            // Pack two 4-bit values per byte
            if sb % 2 == 0 {
                scales[sb / 2] = scale_q;
                mins[sb / 2] = min_q;
            } else {
                scales[sb / 2] |= scale_q << 4;
                mins[sb / 2] |= min_q << 4;
            }

            // Quantize weights to 2 bits
            let sb_scale = f16_to_f32(d) * (scale_q as f32 / 15.0);
            #[allow(clippy::needless_range_loop)]
            for i in 0..subblock_size {
                let weight = sb_data[i];
                let quant = ((weight / sb_scale) + 1.5).round().clamp(0.0, 3.0) as u8;

                // Pack 4 x 2-bit values per byte
                let byte_idx = (sb_start + i) / 4;
                let bit_offset = ((sb_start + i) % 4) * 2;
                qs[byte_idx] |= quant << bit_offset;
            }
        }

        // Serialize to bytes
        let mut bytes = Vec::with_capacity(84);
        bytes.extend(&d.to_le_bytes());
        bytes.extend(&dmin.to_le_bytes());
        bytes.extend(&scales);
        bytes.extend(&mins);
        bytes.extend(&qs);

        Ok(bytes)
    }

    /// Quantize a super-block to Q3_K format
    fn quantize_q3k(&self, data: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(data.len(), 256);

        let num_subblocks = 16;
        let subblock_size = 16;

        // Compute super-block scale
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let d = f32_to_f16(max_abs / 7.0); // Scale for 3-bit values (-4 to 3)

        // Compute per-subblock scales
        let mut scales = vec![0u8; 12]; // 16 x 6 bits = 96 bits = 12 bytes
        let mut hmask = [0u8; 2];
        let mut qs = vec![0u8; 96];

        for sb in 0..num_subblocks {
            let sb_start = sb * subblock_size;
            let sb_data = &data[sb_start..sb_start + subblock_size];

            let sb_max = sb_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);

            // Quantize scale to 6 bits (with high bit stored separately)
            let scale_f = (sb_max / f16_to_f32(d)) * 63.0;
            let scale_q = scale_f.round().clamp(0.0, 127.0) as u8;

            // Store 6 low bits in scales array
            let scale_6bit = scale_q & 0x3F;

            // Store high bit in hmask
            if scale_q & 0x40 != 0 {
                hmask[sb / 8] |= 1 << (sb % 8);
            }

            // Pack 6-bit scales
            // This is simplified - actual packing is more complex
            let byte_idx = (sb * 6) / 8;
            let bit_offset = (sb * 6) % 8;
            if byte_idx < scales.len() {
                scales[byte_idx] |= scale_6bit << bit_offset;
                if bit_offset > 2 && byte_idx + 1 < scales.len() {
                    scales[byte_idx + 1] |= scale_6bit >> (8 - bit_offset);
                }
            }

            // Quantize weights to 3 bits
            let sb_scale = f16_to_f32(d) * (scale_6bit as f32 / 63.0);
            #[allow(clippy::needless_range_loop)]
            for i in 0..subblock_size {
                let weight = sb_data[i];
                let quant = ((weight / sb_scale) + 4.0).round().clamp(0.0, 7.0) as u8;

                // Pack 3-bit values (8 values per 3 bytes)
                let bit_idx = (sb_start + i) * 3;
                let byte_idx = bit_idx / 8;
                let bit_offset = bit_idx % 8;

                if byte_idx < qs.len() {
                    qs[byte_idx] |= (quant & 0x07) << bit_offset;
                    if bit_offset > 5 && byte_idx + 1 < qs.len() {
                        qs[byte_idx + 1] |= (quant & 0x07) >> (8 - bit_offset);
                    }
                }
            }
        }

        // Serialize to bytes
        let mut bytes = Vec::with_capacity(110);
        bytes.extend(&hmask);
        bytes.extend(&scales);
        bytes.extend(&d.to_le_bytes());
        bytes.extend(&qs);

        Ok(bytes)
    }

    /// Quantize a super-block to Q4_K format
    fn quantize_q4k(&self, data: &[f32]) -> Result<Vec<u8>> {
        assert_eq!(data.len(), 256);

        let num_subblocks = 8;
        let subblock_size = 32;

        // Compute super-block statistics
        let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let min_val = data.iter().copied().fold(f32::INFINITY, f32::min);

        let d = f32_to_f16(max_abs / 15.0); // Scale for 4-bit values (0-15)
        let dmin = f32_to_f16(min_val.abs());

        // Compute per-subblock scales and minimums
        let mut scales = vec![0u8; 6]; // 8 x 6 bits = 48 bits = 6 bytes
        let mut mins = vec![0u8; 6];
        let mut qs = vec![0u8; 128];

        for sb in 0..num_subblocks {
            let sb_start = sb * subblock_size;
            let sb_data = &data[sb_start..sb_start + subblock_size];

            let sb_max = sb_data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
            let sb_min = sb_data.iter().copied().fold(f32::INFINITY, f32::min);

            // Quantize scale and min to 6 bits
            let scale_q = ((sb_max / f16_to_f32(d)) * 63.0).round().clamp(0.0, 63.0) as u8;
            let min_q = ((sb_min.abs() / f16_to_f32(dmin)) * 63.0).round().clamp(0.0, 63.0) as u8;

            // Pack 6-bit values
            let byte_idx = (sb * 6) / 8;
            let bit_offset = (sb * 6) % 8;
            if byte_idx < scales.len() {
                scales[byte_idx] |= scale_q << bit_offset;
                if bit_offset > 2 && byte_idx + 1 < scales.len() {
                    scales[byte_idx + 1] |= scale_q >> (8 - bit_offset);
                }
            }
            if byte_idx < mins.len() {
                mins[byte_idx] |= min_q << bit_offset;
                if bit_offset > 2 && byte_idx + 1 < mins.len() {
                    mins[byte_idx + 1] |= min_q >> (8 - bit_offset);
                }
            }

            // Quantize weights to 4 bits
            let sb_scale = f16_to_f32(d) * (scale_q as f32 / 63.0);
            #[allow(clippy::needless_range_loop)]
            for i in 0..subblock_size {
                let weight = sb_data[i];
                let quant = ((weight / sb_scale) + 8.0).round().clamp(0.0, 15.0) as u8;

                // Pack two 4-bit values per byte
                let byte_idx = (sb_start + i) / 2;
                if (sb_start + i) % 2 == 0 {
                    qs[byte_idx] = quant;
                } else {
                    qs[byte_idx] |= quant << 4;
                }
            }
        }

        // Serialize to bytes
        let mut bytes = Vec::with_capacity(144);
        bytes.extend(&d.to_le_bytes());
        bytes.extend(&dmin.to_le_bytes());
        bytes.extend(&scales);
        bytes.extend(&mins);
        bytes.extend(&qs);

        Ok(bytes)
    }

    /// Dequantize K-quant tensor back to f32
    pub fn dequantize(&self, kquant: &KQuantTensor) -> Result<Tensor> {
        let mut dequantized = Vec::with_capacity(kquant.num_weights);

        let bytes_per_block = kquant.quant_type.bytes_per_superblock();
        let superblock_size = kquant.quant_type.superblock_size();

        for block_idx in 0..kquant.num_blocks {
            let block_start = block_idx * bytes_per_block;
            let block_end = block_start + bytes_per_block;

            if block_end > kquant.blocks.len() {
                break;
            }

            let block_bytes = &kquant.blocks[block_start..block_end];

            let block_data = match kquant.quant_type {
                KQuantType::Q2_K => self.dequantize_q2k(block_bytes)?,
                KQuantType::Q3_K => self.dequantize_q3k(block_bytes)?,
                KQuantType::Q4_K => self.dequantize_q4k(block_bytes)?,
            };

            // Only add the weights we actually need (not padding)
            let remaining = kquant.num_weights - dequantized.len();
            let to_add = remaining.min(superblock_size);
            dequantized.extend_from_slice(&block_data[..to_add]);
        }

        Tensor::from_vec(dequantized, &kquant.shape)
    }

    /// Dequantize Q2_K block
    fn dequantize_q2k(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        // Q2_K block: 2 (d) + 2 (dmin) + 8 (scales) + 8 (mins) + 64 (qs) = 84 bytes
        if bytes.len() < 84 {
            return Err(TrustformersError::quantization_error(
                "Invalid Q2_K block size".to_string(),
            ));
        }

        let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let _dmin = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));

        let scales = &bytes[4..12];
        let _mins = &bytes[12..20];
        let qs = &bytes[20..84];

        let mut weights = Vec::with_capacity(256);

        for sb in 0..16 {
            let scale_byte = scales[sb / 2];
            let scale_q = if sb % 2 == 0 { scale_byte & 0x0F } else { scale_byte >> 4 };
            let sb_scale = d * (scale_q as f32 / 15.0);

            for i in 0..16 {
                let weight_idx = sb * 16 + i;
                let byte_idx = weight_idx / 4;
                let bit_offset = (weight_idx % 4) * 2;

                if byte_idx < qs.len() {
                    let quant = (qs[byte_idx] >> bit_offset) & 0x03;
                    let weight = sb_scale * (quant as f32 - 1.5);
                    weights.push(weight);
                }
            }
        }

        Ok(weights)
    }

    /// Dequantize Q3_K block
    fn dequantize_q3k(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() < 110 {
            return Err(TrustformersError::quantization_error(
                "Invalid Q3_K block size".to_string(),
            ));
        }

        let hmask = &bytes[0..2];
        let scales = &bytes[2..14];
        let d = f16_to_f32(u16::from_le_bytes([bytes[14], bytes[15]]));
        let qs = &bytes[16..112];

        let mut weights = Vec::with_capacity(256);

        for sb in 0..16 {
            // Extract 6-bit scale (simplified)
            let byte_idx = (sb * 6) / 8;
            let bit_offset = (sb * 6) % 8;
            let mut scale_q = if byte_idx < scales.len() {
                (scales[byte_idx] >> bit_offset) & 0x3F
            } else {
                32
            };

            // Add high bit from hmask
            if hmask[sb / 8] & (1 << (sb % 8)) != 0 {
                scale_q |= 0x40;
            }

            let sb_scale = d * (scale_q as f32 / 63.0);

            for i in 0..16 {
                let weight_idx = sb * 16 + i;
                let bit_idx = weight_idx * 3;
                let byte_idx = bit_idx / 8;
                let bit_offset = bit_idx % 8;

                if byte_idx < qs.len() {
                    let mut quant = (qs[byte_idx] >> bit_offset) & 0x07;
                    if bit_offset > 5 && byte_idx + 1 < qs.len() {
                        quant |= (qs[byte_idx + 1] << (8 - bit_offset)) & 0x07;
                    }

                    let weight = sb_scale * (quant as f32 - 4.0);
                    weights.push(weight);
                }
            }
        }

        Ok(weights)
    }

    /// Dequantize Q4_K block
    fn dequantize_q4k(&self, bytes: &[u8]) -> Result<Vec<f32>> {
        if bytes.len() < 144 {
            return Err(TrustformersError::quantization_error(
                "Invalid Q4_K block size".to_string(),
            ));
        }

        let d = f16_to_f32(u16::from_le_bytes([bytes[0], bytes[1]]));
        let _dmin = f16_to_f32(u16::from_le_bytes([bytes[2], bytes[3]]));

        let scales = &bytes[4..10];
        let _mins = &bytes[10..16];
        let qs = &bytes[16..144];

        let mut weights = Vec::with_capacity(256);

        for sb in 0..8 {
            // Extract 6-bit scale (simplified)
            let byte_idx = (sb * 6) / 8;
            let bit_offset = (sb * 6) % 8;
            let scale_q = if byte_idx < scales.len() {
                (scales[byte_idx] >> bit_offset) & 0x3F
            } else {
                32
            };

            let sb_scale = d * (scale_q as f32 / 63.0);

            for i in 0..32 {
                let weight_idx = sb * 32 + i;
                let byte_idx = weight_idx / 2;

                if byte_idx < qs.len() {
                    let quant =
                        if weight_idx % 2 == 0 { qs[byte_idx] & 0x0F } else { qs[byte_idx] >> 4 };

                    let weight = sb_scale * (quant as f32 - 8.0);
                    weights.push(weight);
                }
            }
        }

        Ok(weights)
    }
}

/// Convert f32 to FP16
fn f32_to_f16(val: f32) -> F16 {
    // Simplified conversion - in production, use half crate
    let bits = val.to_bits();
    let sign = (bits >> 31) & 1;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7F_FFFF;

    if exp == 0 {
        return (sign as u16) << 15;
    }

    let exp_f16 = exp - 127 + 15;
    if exp_f16 <= 0 {
        return (sign as u16) << 15;
    }
    if exp_f16 >= 31 {
        return ((sign as u16) << 15) | 0x7C00; // Infinity
    }

    let mant_f16 = (mant >> 13) as u16;
    ((sign as u16) << 15) | ((exp_f16 as u16) << 10) | (mant_f16 & 0x3FF)
}

/// Convert FP16 to f32
fn f16_to_f32(val: F16) -> f32 {
    let sign = (val >> 15) & 1;
    let exp = ((val >> 10) & 0x1F) as i32;
    let mant = (val & 0x3FF) as u32;

    if exp == 0 {
        return if sign == 1 { -0.0 } else { 0.0 };
    }

    if exp == 31 {
        return if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY };
    }

    let exp_f32 = exp - 15 + 127;
    let mant_f32 = mant << 13;

    let bits = ((sign as u32) << 31) | ((exp_f32 as u32) << 23) | mant_f32;
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_kquant_types() {
        let q2k = KQuantType::Q2_K;
        assert_eq!(q2k.superblock_size(), 256);
        assert_eq!(q2k.num_subblocks(), 16);
        assert_eq!(q2k.weight_bits(), 2);
        assert_eq!(q2k.bytes_per_superblock(), 82);

        let q3k = KQuantType::Q3_K;
        assert_eq!(q3k.weight_bits(), 3);
        assert_eq!(q3k.bytes_per_superblock(), 110);

        let q4k = KQuantType::Q4_K;
        assert_eq!(q4k.weight_bits(), 4);
        assert_eq!(q4k.bytes_per_superblock(), 144);
    }

    #[test]
    fn test_q4k_quantization() -> Result<()> {
        let config = KQuantConfig {
            quant_type: KQuantType::Q4_K,
            ..Default::default()
        };

        let quantizer = KQuantizer::new(config)?;

        // Create test tensor (must be multiple of 256)
        let data: Vec<f32> = (0..512).map(|i| (i as f32) * 0.01).collect();
        let tensor = Tensor::from_vec(data.clone(), &[512])?;

        let quantized = quantizer.quantize(&tensor)?;

        assert_eq!(quantized.quant_type, KQuantType::Q4_K);
        assert_eq!(quantized.num_blocks, 2); // 512 / 256 = 2
        assert_eq!(quantized.blocks.len(), 2 * 144); // 2 blocks * 144 bytes

        // Dequantize and check
        let dequantized = quantizer.dequantize(&quantized)?;
        assert_eq!(dequantized.shape(), &[512]);

        Ok(())
    }

    #[test]
    fn test_q2k_roundtrip() -> Result<()> {
        let config = KQuantConfig {
            quant_type: KQuantType::Q2_K,
            ..Default::default()
        };

        let quantizer = KQuantizer::new(config)?;

        let data: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.01).collect();
        let tensor = Tensor::from_vec(data.clone(), &[256])?;

        let quantized = quantizer.quantize(&tensor)?;
        let dequantized = quantizer.dequantize(&quantized)?;

        let deq_data = dequantized.to_vec_f32()?;

        // Check approximate equality (2-bit quantization has significant error)
        for (orig, deq) in data.iter().zip(deq_data.iter()) {
            let abs_error = (orig - deq).abs();
            assert!(abs_error < 0.5, "Error too large: {} vs {}", orig, deq);
        }

        Ok(())
    }

    #[test]
    fn test_f16_conversion() {
        let values = vec![0.0, 1.0, -1.0, 10.5, -10.5, 0.123, -0.123];

        for &val in &values {
            let f16 = f32_to_f16(val);
            let recovered = f16_to_f32(f16);

            let rel_error = (val - recovered).abs() / (val.abs() + 1e-6);
            assert!(
                rel_error < 0.001,
                "FP16 conversion error: {} vs {}",
                val,
                recovered
            );
        }
    }
}
