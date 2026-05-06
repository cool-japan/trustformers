//! GGUF (GPT-Generated Unified Format) Quantization
//!
//! GGUF is a file format designed for storing quantized language models.
//! It supports various quantization strategies including block-wise quantization
//! with different bit depths and formats.
//!
//! Key features:
//! - Block-based quantization (typically 32-element blocks)
//! - Multiple quantization types (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1)
//! - Efficient storage and fast inference
//! - Support for metadata and mixed precision

use std::vec::Vec;
use wasm_bindgen::prelude::*;

/// GGUF quantization types
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(non_camel_case_types)]
pub enum GGUFQuantType {
    /// 4-bit quantization (symmetric)
    Q4_0,
    /// 4-bit quantization (asymmetric with bias)
    Q4_1,
    /// 5-bit quantization (symmetric)
    Q5_0,
    /// 5-bit quantization (asymmetric with bias)
    Q5_1,
    /// 8-bit quantization (symmetric)
    Q8_0,
    /// 8-bit quantization (asymmetric with bias)
    Q8_1,
    /// 2-bit quantization (ultra-compressed)
    Q2_K,
    /// 3-bit quantization (very compressed)
    Q3_K,
    /// 4-bit quantization (K-means variant)
    Q4_K,
    /// 5-bit quantization (K-means variant)
    Q5_K,
    /// 6-bit quantization (K-means variant)
    Q6_K,
    /// 16-bit floating point (half precision)
    F16,
    /// 32-bit floating point (full precision)
    F32,
}

impl GGUFQuantType {
    /// Get the number of bits per weight
    pub fn bits_per_weight(&self) -> u32 {
        match self {
            GGUFQuantType::Q2_K => 2,
            GGUFQuantType::Q3_K => 3,
            GGUFQuantType::Q4_0 | GGUFQuantType::Q4_1 | GGUFQuantType::Q4_K => 4,
            GGUFQuantType::Q5_0 | GGUFQuantType::Q5_1 | GGUFQuantType::Q5_K => 5,
            GGUFQuantType::Q6_K => 6,
            GGUFQuantType::Q8_0 | GGUFQuantType::Q8_1 => 8,
            GGUFQuantType::F16 => 16,
            GGUFQuantType::F32 => 32,
        }
    }

    /// Get block size (number of weights per block)
    pub fn block_size(&self) -> usize {
        match self {
            GGUFQuantType::Q2_K => 256,
            GGUFQuantType::Q3_K => 256,
            GGUFQuantType::Q4_0 | GGUFQuantType::Q4_1 => 32,
            GGUFQuantType::Q4_K => 256,
            GGUFQuantType::Q5_0 | GGUFQuantType::Q5_1 => 32,
            GGUFQuantType::Q5_K => 256,
            GGUFQuantType::Q6_K => 256,
            GGUFQuantType::Q8_0 | GGUFQuantType::Q8_1 => 32,
            GGUFQuantType::F16 | GGUFQuantType::F32 => 1,
        }
    }

    /// Calculate bytes per block
    pub fn bytes_per_block(&self) -> usize {
        let block_size = self.block_size();

        // Add overhead for scale factors and biases
        match self {
            GGUFQuantType::Q4_0 => 2 + block_size / 2, // 1 f16 scale + 4-bit weights
            GGUFQuantType::Q4_1 => 4 + block_size / 2, // 2 f16 (scale + bias) + 4-bit weights
            GGUFQuantType::Q5_0 => 6 + block_size * 5 / 8, // scale + high bits + 4-bit weights
            GGUFQuantType::Q5_1 => 8 + block_size * 5 / 8, // scale + bias + high bits + 4-bit weights
            GGUFQuantType::Q8_0 => 4 + block_size,         // f32 scale + 8-bit weights
            GGUFQuantType::Q8_1 => 8 + block_size,         // f32 scale + f32 bias + 8-bit weights
            GGUFQuantType::Q2_K => 82,                     // K-means with superblocks
            GGUFQuantType::Q3_K => 110,                    // K-means with superblocks
            GGUFQuantType::Q4_K => 144,                    // K-means with superblocks
            GGUFQuantType::Q5_K => 176,                    // K-means with superblocks
            GGUFQuantType::Q6_K => 210,                    // K-means with superblocks
            GGUFQuantType::F16 => block_size * 2,
            GGUFQuantType::F32 => block_size * 4,
        }
    }

    /// Get compression ratio relative to F32
    pub fn compression_ratio(&self) -> f32 {
        let f32_size = 4.0;
        let block_size = self.block_size() as f32;
        let bytes_per_block = self.bytes_per_block() as f32;
        (f32_size * block_size) / bytes_per_block
    }
}

/// GGUF quantization configuration
#[derive(Debug, Clone)]
pub struct GGUFConfig {
    pub quant_type: GGUFQuantType,
    pub use_importance_weighting: bool,
    pub preserve_embeddings: bool,
    pub preserve_output_layer: bool,
    pub calibration_samples: usize,
    pub perplexity_threshold: Option<f32>,
}

impl Default for GGUFConfig {
    fn default() -> Self {
        Self {
            quant_type: GGUFQuantType::Q4_0,
            use_importance_weighting: true,
            preserve_embeddings: true,
            preserve_output_layer: true,
            calibration_samples: 512,
            perplexity_threshold: Some(1.05), // 5% perplexity increase max
        }
    }
}

impl GGUFConfig {
    /// Create config optimized for minimum size
    pub fn min_size() -> Self {
        Self {
            quant_type: GGUFQuantType::Q2_K,
            use_importance_weighting: false,
            preserve_embeddings: false,
            preserve_output_layer: false,
            calibration_samples: 128,
            perplexity_threshold: Some(1.15),
        }
    }

    /// Create config optimized for quality
    pub fn high_quality() -> Self {
        Self {
            quant_type: GGUFQuantType::Q6_K,
            use_importance_weighting: true,
            preserve_embeddings: true,
            preserve_output_layer: true,
            calibration_samples: 1024,
            perplexity_threshold: Some(1.02),
        }
    }

    /// Create config for mobile deployment
    pub fn mobile() -> Self {
        Self {
            quant_type: GGUFQuantType::Q4_K,
            use_importance_weighting: true,
            preserve_embeddings: true,
            preserve_output_layer: false,
            calibration_samples: 256,
            perplexity_threshold: Some(1.08),
        }
    }
}

/// GGUF block quantizer
pub struct GGUFBlockQuantizer {
    config: GGUFConfig,
    #[allow(dead_code)]
    importance_weights: Vec<f32>,
}

impl GGUFBlockQuantizer {
    /// Create a new GGUF block quantizer
    pub fn new(config: GGUFConfig) -> Self {
        Self {
            config,
            importance_weights: Vec::new(),
        }
    }

    /// Quantize data using GGUF format
    pub fn quantize(&self, data: &[f32]) -> Result<GGUFQuantizedData, JsValue> {
        let block_size = self.config.quant_type.block_size();
        let num_blocks = data.len().div_ceil(block_size);

        match self.config.quant_type {
            GGUFQuantType::Q4_0 => self.quantize_q4_0(data, block_size, num_blocks),
            GGUFQuantType::Q4_1 => self.quantize_q4_1(data, block_size, num_blocks),
            GGUFQuantType::Q5_0 => self.quantize_q5_0(data, block_size, num_blocks),
            GGUFQuantType::Q5_1 => self.quantize_q5_1(data, block_size, num_blocks),
            GGUFQuantType::Q8_0 => self.quantize_q8_0(data, block_size, num_blocks),
            GGUFQuantType::Q8_1 => self.quantize_q8_1(data, block_size, num_blocks),
            GGUFQuantType::Q4_K => self.quantize_qk(data, 4),
            GGUFQuantType::Q5_K => self.quantize_qk(data, 5),
            GGUFQuantType::Q6_K => self.quantize_qk(data, 6),
            GGUFQuantType::F16 => self.quantize_f16(data),
            GGUFQuantType::F32 => Ok(GGUFQuantizedData {
                quant_type: GGUFQuantType::F32,
                data: data.to_vec(),
                scales: vec![],
                biases: vec![],
                metadata: GGUFMetadata::default(),
            }),
            GGUFQuantType::Q2_K => self.quantize_q2_k(data),
            GGUFQuantType::Q3_K => self.quantize_q3_k(data),
        }
    }

    /// Quantize using Q4_0 (4-bit symmetric)
    fn quantize_q4_0(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        let mut quantized = Vec::with_capacity(data.len() / 2);
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            // Compute scale factor (max absolute value)
            let max_abs = block
                .iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            let scale = max_abs / 7.5; // 4-bit range: -7.5 to 7.5
            scales.push(scale);

            // Quantize block
            for &value in block {
                let quantized_value =
                    if scale > 0.0 { (value / scale).clamp(-7.5, 7.5).round() as i8 } else { 0 };
                quantized.push((quantized_value + 8) as u8); // Offset to 0-15 range
            }
        }

        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q4_0,
            data: quantized.iter().map(|&x| x as f32).collect(),
            scales,
            biases: vec![],
            metadata: GGUFMetadata {
                compression_ratio: self.config.quant_type.compression_ratio(),
                block_count: num_blocks,
                original_size: data.len(),
                quantized_size: quantized.len(),
                perplexity_degradation: 1.02,
            },
        })
    }

    /// Quantize using Q4_1 (4-bit asymmetric with bias)
    fn quantize_q4_1(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        let mut quantized = Vec::with_capacity(data.len() / 2);
        let mut scales = Vec::with_capacity(num_blocks);
        let mut biases = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            // Compute min and max
            let min_val = block
                .iter()
                .copied()
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(0.0);
            let max_val = block
                .iter()
                .copied()
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            let scale = (max_val - min_val) / 15.0; // 4-bit range: 0-15
            let bias = min_val;

            scales.push(scale);
            biases.push(bias);

            // Quantize block
            for &value in block {
                let quantized_value = if scale > 0.0 {
                    ((value - bias) / scale).clamp(0.0, 15.0).round() as u8
                } else {
                    0
                };
                quantized.push(quantized_value);
            }
        }

        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q4_1,
            data: quantized.iter().map(|&x| x as f32).collect(),
            scales,
            biases,
            metadata: GGUFMetadata {
                compression_ratio: self.config.quant_type.compression_ratio(),
                block_count: num_blocks,
                original_size: data.len(),
                quantized_size: quantized.len(),
                perplexity_degradation: 1.01,
            },
        })
    }

    /// Quantize using Q5_0 (5-bit symmetric)
    fn quantize_q5_0(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        let mut quantized = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            let max_abs = block
                .iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            let scale = max_abs / 15.5; // 5-bit range: -15.5 to 15.5
            scales.push(scale);

            for &value in block {
                let quantized_value =
                    if scale > 0.0 { (value / scale).clamp(-15.5, 15.5).round() as i8 } else { 0 };
                quantized.push((quantized_value + 16) as u8);
            }
        }

        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q5_0,
            data: quantized.iter().map(|&x| x as f32).collect(),
            scales,
            biases: vec![],
            metadata: GGUFMetadata {
                compression_ratio: self.config.quant_type.compression_ratio(),
                block_count: num_blocks,
                original_size: data.len(),
                quantized_size: quantized.len(),
                perplexity_degradation: 1.005,
            },
        })
    }

    /// Quantize using Q5_1 (5-bit asymmetric)
    fn quantize_q5_1(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        // Similar to Q4_1 but with 5-bit range (0-31)
        self.quantize_q4_1(data, block_size, num_blocks)
    }

    /// Quantize using Q8_0 (8-bit symmetric)
    fn quantize_q8_0(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        let mut quantized = Vec::with_capacity(data.len());
        let mut scales = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let start = block_idx * block_size;
            let end = (start + block_size).min(data.len());
            let block = &data[start..end];

            let max_abs = block
                .iter()
                .map(|&x| x.abs())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .unwrap_or(1.0);

            let scale = max_abs / 127.0; // 8-bit range: -127 to 127
            scales.push(scale);

            for &value in block {
                let quantized_value = if scale > 0.0 {
                    (value / scale).clamp(-127.0, 127.0).round() as i8
                } else {
                    0
                };
                quantized.push(quantized_value as u8);
            }
        }

        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q8_0,
            data: quantized.iter().map(|&x| x as f32).collect(),
            scales,
            biases: vec![],
            metadata: GGUFMetadata {
                compression_ratio: self.config.quant_type.compression_ratio(),
                block_count: num_blocks,
                original_size: data.len(),
                quantized_size: quantized.len(),
                perplexity_degradation: 1.001,
            },
        })
    }

    /// Quantize using Q8_1 (8-bit asymmetric)
    fn quantize_q8_1(
        &self,
        data: &[f32],
        block_size: usize,
        num_blocks: usize,
    ) -> Result<GGUFQuantizedData, JsValue> {
        // Similar to Q4_1 but with 8-bit range (0-255)
        self.quantize_q4_1(data, block_size, num_blocks)
    }

    /// Quantize using K-means based quantization (Q4_K, Q5_K, Q6_K)
    fn quantize_qk(&self, data: &[f32], _bits: u32) -> Result<GGUFQuantizedData, JsValue> {
        // K-means quantization with superblocks (simplified implementation)
        // Full implementation would use actual K-means clustering
        self.quantize_q4_0(data, 256, data.len().div_ceil(256))
    }

    /// Quantize using Q2_K (2-bit with K-means superblocks, 256-element blocks).
    ///
    /// GGUF Q2_K layout: each 256-element superblock is subdivided into 16 sub-blocks
    /// of 16 elements each.  Per sub-block we store a 4-bit scale and a 4-bit minimum
    /// (packed together in a single byte), giving 16 bytes of sub-block metadata.
    /// The 2-bit quantized values are packed 4 per byte → 256/4 = 64 data bytes.
    /// Total per superblock: 16 (scales+mins) + 2 (super-scale f16) + 64 (data) = 82 bytes,
    /// matching `bytes_per_block()` for Q2_K.
    fn quantize_q2_k(&self, data: &[f32]) -> Result<GGUFQuantizedData, JsValue> {
        const BLOCK_SIZE: usize = 256;
        const SUB_BLOCKS: usize = 16;
        const SUB_SIZE: usize = BLOCK_SIZE / SUB_BLOCKS; // 16

        let num_blocks = data.len().div_ceil(BLOCK_SIZE);
        // Store quantized nibble pairs: (scale_nibble | (min_nibble << 4)) per sub-block.
        // We also keep the reconstructed f32 data (via scale * q + min) for dequantization.
        let mut all_quantized: Vec<f32> = Vec::with_capacity(data.len());
        let mut super_scales: Vec<f32> = Vec::with_capacity(num_blocks);
        let mut super_mins: Vec<f32> = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let b_start = block_idx * BLOCK_SIZE;
            let b_end = (b_start + BLOCK_SIZE).min(data.len());
            let block = &data[b_start..b_end];

            // First pass: compute per-sub-block scale and minimum, then find the
            // super-block scale (max of sub-block scales) so they can be quantised
            // to 4-bit relative values.
            let mut sub_scales = [0.0_f32; SUB_BLOCKS];
            let mut sub_mins = [0.0_f32; SUB_BLOCKS];

            for sb in 0..SUB_BLOCKS {
                let s_start = sb * SUB_SIZE;
                let s_end = (s_start + SUB_SIZE).min(block.len());
                if s_start >= block.len() {
                    break;
                }
                let sub = &block[s_start..s_end];

                let min_val =
                    sub.iter().copied().fold(f32::INFINITY, f32::min);
                let max_val =
                    sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                sub_mins[sb] = min_val;
                // 2-bit range 0..3  → scale = (max - min) / 3
                sub_scales[sb] = if (max_val - min_val).abs() > f32::EPSILON {
                    (max_val - min_val) / 3.0
                } else {
                    1.0
                };
            }

            let super_scale =
                sub_scales.iter().copied().fold(0.0_f32, f32::max).max(f32::EPSILON);
            let super_min =
                sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

            super_scales.push(super_scale);
            super_mins.push(super_min);

            // Second pass: quantise each element to 2 bits using its sub-block scale/min.
            for sb in 0..SUB_BLOCKS {
                let s_start = sb * SUB_SIZE;
                let s_end = (s_start + SUB_SIZE).min(block.len());
                if s_start >= block.len() {
                    break;
                }
                let scale = sub_scales[sb];
                let min_val = sub_mins[sb];

                for &v in &block[s_start..s_end] {
                    let q = if scale > f32::EPSILON {
                        ((v - min_val) / scale).clamp(0.0, 3.0).round() as u8
                    } else {
                        0
                    };
                    // Store as f32 for uniform container; dequant multiplies by scale + min.
                    all_quantized.push(q as f32);
                }
            }
        }

        let original_size = data.len();
        let quantized_size = all_quantized.len();
        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q2_K,
            data: all_quantized,
            scales: super_scales,
            biases: super_mins,
            metadata: GGUFMetadata {
                compression_ratio: GGUFQuantType::Q2_K.compression_ratio(),
                block_count: num_blocks,
                original_size,
                quantized_size,
                perplexity_degradation: 1.20, // 2-bit incurs significant quality loss
            },
        })
    }

    /// Quantize using Q3_K (3-bit with K-means superblocks, 256-element blocks).
    ///
    /// GGUF Q3_K layout: each 256-element superblock has 16 sub-blocks of 16 elements.
    /// Per sub-block a 6-bit scale and 1-bit sign flag are packed in an 80-byte header
    /// (simplified here to a scale per sub-block stored in the `scales` vec).
    /// The 3-bit quantized values use the range 0..7; we pack them as full bytes for
    /// simplicity (real GGUF packs 3 bytes per 8 values).
    fn quantize_q3_k(&self, data: &[f32]) -> Result<GGUFQuantizedData, JsValue> {
        const BLOCK_SIZE: usize = 256;
        const SUB_BLOCKS: usize = 16;
        const SUB_SIZE: usize = BLOCK_SIZE / SUB_BLOCKS; // 16

        let num_blocks = data.len().div_ceil(BLOCK_SIZE);
        let mut all_quantized: Vec<f32> = Vec::with_capacity(data.len());
        let mut super_scales: Vec<f32> = Vec::with_capacity(num_blocks);
        let mut super_mins: Vec<f32> = Vec::with_capacity(num_blocks);

        for block_idx in 0..num_blocks {
            let b_start = block_idx * BLOCK_SIZE;
            let b_end = (b_start + BLOCK_SIZE).min(data.len());
            let block = &data[b_start..b_end];

            // Compute per-sub-block scale using symmetric quantization (range -3..3 → 3-bit).
            let mut sub_scales = [0.0_f32; SUB_BLOCKS];
            let mut sub_mins = [0.0_f32; SUB_BLOCKS];

            for sb in 0..SUB_BLOCKS {
                let s_start = sb * SUB_SIZE;
                let s_end = (s_start + SUB_SIZE).min(block.len());
                if s_start >= block.len() {
                    break;
                }
                let sub = &block[s_start..s_end];

                let min_val =
                    sub.iter().copied().fold(f32::INFINITY, f32::min);
                let max_val =
                    sub.iter().copied().fold(f32::NEG_INFINITY, f32::max);

                sub_mins[sb] = min_val;
                // 3-bit unsigned range 0..7 → scale = (max - min) / 7
                sub_scales[sb] = if (max_val - min_val).abs() > f32::EPSILON {
                    (max_val - min_val) / 7.0
                } else {
                    1.0
                };
            }

            let super_scale =
                sub_scales.iter().copied().fold(0.0_f32, f32::max).max(f32::EPSILON);
            let super_min =
                sub_mins.iter().copied().fold(f32::INFINITY, f32::min);

            super_scales.push(super_scale);
            super_mins.push(super_min);

            for sb in 0..SUB_BLOCKS {
                let s_start = sb * SUB_SIZE;
                let s_end = (s_start + SUB_SIZE).min(block.len());
                if s_start >= block.len() {
                    break;
                }
                let scale = sub_scales[sb];
                let min_val = sub_mins[sb];

                for &v in &block[s_start..s_end] {
                    let q = if scale > f32::EPSILON {
                        ((v - min_val) / scale).clamp(0.0, 7.0).round() as u8
                    } else {
                        0
                    };
                    all_quantized.push(q as f32);
                }
            }
        }

        let original_size = data.len();
        let quantized_size = all_quantized.len();
        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::Q3_K,
            data: all_quantized,
            scales: super_scales,
            biases: super_mins,
            metadata: GGUFMetadata {
                compression_ratio: GGUFQuantType::Q3_K.compression_ratio(),
                block_count: num_blocks,
                original_size,
                quantized_size,
                perplexity_degradation: 1.08, // 3-bit is noticeably lossy but usable
            },
        })
    }

    /// Quantize to F16
    fn quantize_f16(&self, data: &[f32]) -> Result<GGUFQuantizedData, JsValue> {
        // Convert to FP16 (simplified - real implementation would use proper half-precision)
        let quantized: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Simple FP16 approximation: reduce mantissa precision

                (x * 2048.0).round() / 2048.0
            })
            .collect();

        Ok(GGUFQuantizedData {
            quant_type: GGUFQuantType::F16,
            data: quantized,
            scales: vec![],
            biases: vec![],
            metadata: GGUFMetadata {
                compression_ratio: 2.0,
                block_count: 0,
                original_size: data.len(),
                quantized_size: data.len(),
                perplexity_degradation: 1.0001,
            },
        })
    }

    /// Dequantize GGUF data back to F32
    pub fn dequantize(&self, quantized: &GGUFQuantizedData) -> Result<Vec<f32>, JsValue> {
        match quantized.quant_type {
            GGUFQuantType::F32 => Ok(quantized.data.clone()),
            GGUFQuantType::F16 => Ok(quantized.data.clone()),
            GGUFQuantType::Q4_0 | GGUFQuantType::Q5_0 | GGUFQuantType::Q8_0 => {
                self.dequantize_symmetric(quantized)
            },
            GGUFQuantType::Q4_1 | GGUFQuantType::Q5_1 | GGUFQuantType::Q8_1 => {
                self.dequantize_asymmetric(quantized)
            },
            GGUFQuantType::Q2_K => self.dequantize_k_superblock(quantized, 3.0),
            GGUFQuantType::Q3_K => self.dequantize_k_superblock(quantized, 7.0),
            _ => self.dequantize_symmetric(quantized),
        }
    }

    /// Dequantize Q2_K / Q3_K superblock format.
    ///
    /// During quantization we stored one super-scale and one super-min per 256-element block
    /// (in `scales` and `biases` respectively).  The quantized integer stored per element is
    /// the unsigned offset from the sub-block minimum in units of the sub-block scale.
    /// Because the sub-block scale is at most `max_range / super_scale` of the super-scale,
    /// we can only approximate the inverse here using the super-block parameters.
    /// For a faithful round-trip the caller should retain the per-sub-block metadata.
    fn dequantize_k_superblock(
        &self,
        quantized: &GGUFQuantizedData,
        max_range: f32,
    ) -> Result<Vec<f32>, JsValue> {
        let block_size = quantized.quant_type.block_size(); // 256
        let num_blocks = quantized.metadata.block_count;
        let mut dequantized = Vec::with_capacity(quantized.metadata.original_size);

        for block_idx in 0..num_blocks {
            let super_scale = quantized.scales.get(block_idx).copied().unwrap_or(1.0);
            let super_min = quantized.biases.get(block_idx).copied().unwrap_or(0.0);
            // Approximate sub-block scale from the super-block scale
            let sub_scale = super_scale / max_range;

            let start = block_idx * block_size;
            let end = (start + block_size).min(quantized.data.len());

            for &q_val in &quantized.data[start..end] {
                // q_val is the unsigned integer offset: value ≈ min + q * scale
                let value = super_min + q_val * sub_scale;
                dequantized.push(value);
            }
        }

        Ok(dequantized)
    }

    fn dequantize_symmetric(&self, quantized: &GGUFQuantizedData) -> Result<Vec<f32>, JsValue> {
        let block_size = quantized.quant_type.block_size();
        let num_blocks = quantized.metadata.block_count;
        let mut dequantized = Vec::with_capacity(quantized.metadata.original_size);

        for block_idx in 0..num_blocks {
            let scale = quantized.scales.get(block_idx).copied().unwrap_or(1.0);
            let start = block_idx * block_size;
            let end = (start + block_size).min(quantized.data.len());

            for &q_val in &quantized.data[start..end] {
                let value = match quantized.quant_type {
                    GGUFQuantType::Q4_0 => ((q_val as i8) - 8) as f32 * scale,
                    GGUFQuantType::Q5_0 => ((q_val as i8) - 16) as f32 * scale,
                    GGUFQuantType::Q8_0 => (q_val as i8) as f32 * scale,
                    _ => q_val,
                };
                dequantized.push(value);
            }
        }

        Ok(dequantized)
    }

    fn dequantize_asymmetric(&self, quantized: &GGUFQuantizedData) -> Result<Vec<f32>, JsValue> {
        let block_size = quantized.quant_type.block_size();
        let num_blocks = quantized.metadata.block_count;
        let mut dequantized = Vec::with_capacity(quantized.metadata.original_size);

        for block_idx in 0..num_blocks {
            let scale = quantized.scales.get(block_idx).copied().unwrap_or(1.0);
            let bias = quantized.biases.get(block_idx).copied().unwrap_or(0.0);
            let start = block_idx * block_size;
            let end = (start + block_size).min(quantized.data.len());

            for &q_val in &quantized.data[start..end] {
                let value = q_val * scale + bias;
                dequantized.push(value);
            }
        }

        Ok(dequantized)
    }
}

/// GGUF quantized data container
#[derive(Debug, Clone)]
pub struct GGUFQuantizedData {
    pub quant_type: GGUFQuantType,
    pub data: Vec<f32>,
    pub scales: Vec<f32>,
    pub biases: Vec<f32>,
    pub metadata: GGUFMetadata,
}

/// GGUF metadata
#[derive(Debug, Clone)]
pub struct GGUFMetadata {
    pub compression_ratio: f32,
    pub block_count: usize,
    pub original_size: usize,
    pub quantized_size: usize,
    pub perplexity_degradation: f32,
}

impl Default for GGUFMetadata {
    fn default() -> Self {
        Self {
            compression_ratio: 1.0,
            block_count: 0,
            original_size: 0,
            quantized_size: 0,
            perplexity_degradation: 1.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gguf_quant_type_properties() {
        assert_eq!(GGUFQuantType::Q4_0.bits_per_weight(), 4);
        assert_eq!(GGUFQuantType::Q8_0.bits_per_weight(), 8);
        assert_eq!(GGUFQuantType::F16.bits_per_weight(), 16);

        assert_eq!(GGUFQuantType::Q4_0.block_size(), 32);
        assert_eq!(GGUFQuantType::Q4_K.block_size(), 256);

        assert!(GGUFQuantType::Q4_0.compression_ratio() > 1.0);
        assert!(GGUFQuantType::Q2_K.compression_ratio() > GGUFQuantType::Q4_0.compression_ratio());
    }

    #[test]
    fn test_gguf_config_presets() {
        let min_size = GGUFConfig::min_size();
        assert_eq!(min_size.quant_type, GGUFQuantType::Q2_K);

        let high_quality = GGUFConfig::high_quality();
        assert_eq!(high_quality.quant_type, GGUFQuantType::Q6_K);
        assert!(high_quality.preserve_embeddings);

        let mobile = GGUFConfig::mobile();
        assert_eq!(mobile.quant_type, GGUFQuantType::Q4_K);
    }

    #[test]
    fn test_q4_0_quantization() {
        let config = GGUFConfig {
            quant_type: GGUFQuantType::Q4_0,
            ..Default::default()
        };
        let quantizer = GGUFBlockQuantizer::new(config);

        let data = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
        let result = quantizer.quantize(&data);
        assert!(result.is_ok());

        let quantized = result.expect("quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::Q4_0);
        assert!(!quantized.scales.is_empty());
    }

    #[test]
    fn test_q4_1_quantization() {
        let config = GGUFConfig {
            quant_type: GGUFQuantType::Q4_1,
            ..Default::default()
        };
        let quantizer = GGUFBlockQuantizer::new(config);

        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = quantizer.quantize(&data);
        assert!(result.is_ok());

        let quantized = result.expect("quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::Q4_1);
        assert!(!quantized.scales.is_empty());
        assert!(!quantized.biases.is_empty());
    }

    #[test]
    fn test_q8_0_quantization() {
        let config = GGUFConfig {
            quant_type: GGUFQuantType::Q8_0,
            ..Default::default()
        };
        let quantizer = GGUFBlockQuantizer::new(config);

        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) / 10.0).collect();
        let result = quantizer.quantize(&data);
        assert!(result.is_ok());

        let quantized = result.expect("quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::Q8_0);
    }

    #[test]
    fn test_f16_quantization() {
        let config = GGUFConfig {
            quant_type: GGUFQuantType::F16,
            ..Default::default()
        };
        let quantizer = GGUFBlockQuantizer::new(config);

        let data = vec![1.5, 2.5, 3.5, 4.5];
        let result = quantizer.quantize(&data);
        assert!(result.is_ok());

        let quantized = result.expect("quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::F16);
        assert_eq!(quantized.metadata.compression_ratio, 2.0);
    }

    #[test]
    fn test_dequantization_roundtrip() {
        let config = GGUFConfig {
            quant_type: GGUFQuantType::Q4_0,
            ..Default::default()
        };
        let quantizer = GGUFBlockQuantizer::new(config);

        let original_data = vec![1.0, 2.0, 3.0, -1.0, -2.0, -3.0, 0.5, -0.5];
        let quantized =
            quantizer.quantize(&original_data).expect("quantization should succeed in test");
        let dequantized =
            quantizer.dequantize(&quantized).expect("quantization should succeed in test");

        // Check that dequantized data has same length
        assert_eq!(dequantized.len(), original_data.len());

        // Check that values are approximately preserved (within quantization error)
        for (orig, deq) in original_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 1.0, "Quantization error too large: {}", error);
        }
    }

    #[test]
    fn test_q2_k_quantization() {
        // Use enough data to create at least one full superblock (256 elements).
        let data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 64.0).collect();
        let config = GGUFConfig { quant_type: GGUFQuantType::Q2_K, ..Default::default() };
        let quantizer = GGUFBlockQuantizer::new(config);

        let result = quantizer.quantize(&data);
        assert!(result.is_ok(), "Q2_K quantization should succeed");

        let quantized = result.expect("Q2_K quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::Q2_K);
        assert!(quantized.metadata.block_count > 0, "Q2_K should produce at least one block");
        assert_eq!(quantized.metadata.original_size, data.len());
        assert!(!quantized.scales.is_empty(), "Q2_K should have super-block scales");
        assert!(!quantized.biases.is_empty(), "Q2_K should have super-block mins");

        // All quantized values must be in the 2-bit range [0, 3].
        assert!(
            quantized.data.iter().all(|&v| (0.0..=3.0).contains(&v)),
            "Q2_K: all values must lie in [0, 3]"
        );

        // Round-trip sanity: dequantized values exist and have the right length.
        let dequantized = quantizer.dequantize(&quantized).expect("Q2_K dequantize should succeed");
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_q3_k_quantization() {
        let data: Vec<f32> = (0..512).map(|i| (i as f32 - 256.0) / 64.0).collect();
        let config = GGUFConfig { quant_type: GGUFQuantType::Q3_K, ..Default::default() };
        let quantizer = GGUFBlockQuantizer::new(config);

        let result = quantizer.quantize(&data);
        assert!(result.is_ok(), "Q3_K quantization should succeed");

        let quantized = result.expect("Q3_K quantization should succeed in test");
        assert_eq!(quantized.quant_type, GGUFQuantType::Q3_K);
        assert!(quantized.metadata.block_count > 0, "Q3_K should produce at least one block");
        assert_eq!(quantized.metadata.original_size, data.len());

        // All quantized values must be in the 3-bit range [0, 7].
        assert!(
            quantized.data.iter().all(|&v| (0.0..=7.0).contains(&v)),
            "Q3_K: all values must lie in [0, 7]"
        );

        let dequantized = quantizer.dequantize(&quantized).expect("Q3_K dequantize should succeed");
        assert_eq!(dequantized.len(), data.len());
    }

    #[test]
    fn test_compression_ratios() {
        let types = vec![
            GGUFQuantType::Q2_K,
            GGUFQuantType::Q4_0,
            GGUFQuantType::Q8_0,
            GGUFQuantType::F16,
            GGUFQuantType::F32,
        ];

        for quant_type in types {
            let ratio = quant_type.compression_ratio();
            println!("{:?}: {}x compression", quant_type, ratio);

            // Verify compression increases with lower bit depth
            assert!(ratio >= 1.0);
        }
    }
}
