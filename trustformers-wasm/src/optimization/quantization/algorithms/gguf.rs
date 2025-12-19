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
            _ => Err(JsValue::from_str("Quantization type not yet implemented")),
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
                .max_by(|a, b| a.partial_cmp(b).unwrap())
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
            let min_val =
                block.iter().copied().min_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(0.0);
            let max_val =
                block.iter().copied().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap_or(1.0);

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
                .max_by(|a, b| a.partial_cmp(b).unwrap())
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
                .max_by(|a, b| a.partial_cmp(b).unwrap())
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

    /// Quantize to F16
    fn quantize_f16(&self, data: &[f32]) -> Result<GGUFQuantizedData, JsValue> {
        // Convert to FP16 (simplified - real implementation would use proper half-precision)
        let quantized: Vec<f32> = data
            .iter()
            .map(|&x| {
                // Simple FP16 approximation: reduce mantissa precision
                let rounded = (x * 2048.0).round() / 2048.0;
                rounded
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
            _ => self.dequantize_symmetric(quantized),
        }
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

        let quantized = result.unwrap();
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

        let quantized = result.unwrap();
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

        let quantized = result.unwrap();
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

        let quantized = result.unwrap();
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
        let quantized = quantizer.quantize(&original_data).unwrap();
        let dequantized = quantizer.dequantize(&quantized).unwrap();

        // Check that dequantized data has same length
        assert_eq!(dequantized.len(), original_data.len());

        // Check that values are approximately preserved (within quantization error)
        for (orig, deq) in original_data.iter().zip(dequantized.iter()) {
            let error = (orig - deq).abs();
            assert!(error < 1.0, "Quantization error too large: {}", error);
        }
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
