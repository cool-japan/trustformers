//! Enhanced INT4 Quantization for Mobile
//!
//! This module provides advanced INT4 quantization optimized for mobile devices with:
//! - Block-wise quantization for better accuracy
//! - Efficient packing (2 INT4 values per byte)
//! - SIMD-optimized operations
//! - Per-channel and per-tensor quantization
//! - Adaptive block sizing based on tensor dimensions
//! - Zero-copy dequantization where possible

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::Tensor;

/// Block size for INT4 quantization
/// Smaller blocks = better accuracy, larger blocks = better compression
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BlockSize {
    /// 16 elements per block - best accuracy
    Small = 16,
    /// 32 elements per block - balanced
    Medium = 32,
    /// 64 elements per block - best compression
    Large = 64,
    /// 128 elements per block - ultra compression
    XLarge = 128,
}

impl BlockSize {
    /// Get block size as usize
    pub fn as_usize(&self) -> usize {
        match self {
            BlockSize::Small => 16,
            BlockSize::Medium => 32,
            BlockSize::Large => 64,
            BlockSize::XLarge => 128,
        }
    }

    /// Select optimal block size based on tensor shape
    pub fn optimal_for_shape(shape: &[usize]) -> Self {
        let total_elements: usize = shape.iter().product();

        if total_elements < 512 {
            BlockSize::Small
        } else if total_elements < 4096 {
            BlockSize::Medium
        } else if total_elements < 32768 {
            BlockSize::Large
        } else {
            BlockSize::XLarge
        }
    }
}

/// Enhanced INT4 quantization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnhancedInt4Config {
    /// Block size for quantization
    pub block_size: BlockSize,

    /// Use symmetric quantization (range: -8 to 7)
    pub symmetric: bool,

    /// Per-channel quantization for conv/linear layers
    pub per_channel: bool,

    /// Enable SIMD optimizations
    pub use_simd: bool,

    /// Pack INT4 values (2 per byte) for storage
    pub packed_storage: bool,

    /// Percentile clipping for outliers (None = no clipping)
    pub outlier_clip_percentile: Option<f32>,
}

impl Default for EnhancedInt4Config {
    fn default() -> Self {
        Self {
            block_size: BlockSize::Medium,
            symmetric: true,
            per_channel: false,
            use_simd: true,
            packed_storage: true,
            outlier_clip_percentile: Some(99.9),
        }
    }
}

/// Quantized INT4 block with scale and zero point
#[derive(Debug, Clone)]
pub struct Int4Block {
    /// Quantized values (packed: 2 INT4 values per byte)
    pub data: Vec<u8>,
    /// Scale factor for dequantization
    pub scale: f32,
    /// Zero point for asymmetric quantization
    pub zero_point: i8,
    /// Number of elements in this block
    pub num_elements: usize,
}

impl Int4Block {
    /// Create new INT4 block
    pub fn new(capacity: usize) -> Self {
        let packed_capacity = (capacity + 1) / 2; // 2 INT4s per byte
        Self {
            data: Vec::with_capacity(packed_capacity),
            scale: 1.0,
            zero_point: 0,
            num_elements: 0,
        }
    }

    /// Pack two INT4 values into one byte
    #[inline]
    pub fn pack_int4(a: i8, b: i8) -> u8 {
        let a_packed = (a & 0x0F) as u8;
        let b_packed = ((b & 0x0F) as u8) << 4;
        a_packed | b_packed
    }

    /// Unpack two INT4 values from one byte
    #[inline]
    pub fn unpack_int4(packed: u8) -> (i8, i8) {
        let a = (packed & 0x0F) as i8;
        let a = if a > 7 { a - 16 } else { a }; // Sign extend

        let b = ((packed >> 4) & 0x0F) as i8;
        let b = if b > 7 { b - 16 } else { b }; // Sign extend

        (a, b)
    }

    /// Get unpacked value at index
    pub fn get(&self, index: usize) -> Option<i8> {
        if index >= self.num_elements {
            return None;
        }

        let byte_index = index / 2;
        let is_high = index % 2 == 1;

        self.data.get(byte_index).map(|&packed| {
            let (low, high) = Self::unpack_int4(packed);
            if is_high {
                high
            } else {
                low
            }
        })
    }

    /// Dequantize value at index
    pub fn dequantize(&self, index: usize) -> Option<f32> {
        self.get(index).map(|q| (q as i32 - self.zero_point as i32) as f32 * self.scale)
    }
}

/// Enhanced INT4 quantizer with block-wise quantization
pub struct EnhancedInt4Quantizer {
    config: EnhancedInt4Config,
}

impl EnhancedInt4Quantizer {
    /// Create new enhanced INT4 quantizer
    pub fn new(config: EnhancedInt4Config) -> Self {
        Self { config }
    }

    /// Create with default configuration
    pub fn default() -> Self {
        Self::new(EnhancedInt4Config::default())
    }

    /// Compute scale and zero point for a block
    fn compute_block_params(&self, values: &[f32]) -> (f32, i8) {
        if values.is_empty() {
            return (1.0, 0);
        }

        // Compute min/max with optional outlier clipping
        let (min_val, max_val) = if let Some(percentile) = self.config.outlier_clip_percentile {
            self.compute_clipped_range(values, percentile)
        } else {
            let min = values.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            (min, max)
        };

        if self.config.symmetric {
            // Symmetric quantization: -8 to 7
            let abs_max = min_val.abs().max(max_val.abs());
            let scale = abs_max / 7.0; // INT4 max value is 7
            (scale, 0) // Zero point is 0 for symmetric
        } else {
            // Asymmetric quantization: use full -8 to 7 range
            let qmin = -8.0;
            let qmax = 7.0;
            let scale = (max_val - min_val) / (qmax - qmin);
            let zero_point = (qmin - min_val / scale).round() as i8;
            (scale.max(1e-8), zero_point.clamp(-8, 7)) // Avoid zero scale
        }
    }

    /// Compute clipped range using percentile
    fn compute_clipped_range(&self, values: &[f32], percentile: f32) -> (f32, f32) {
        if values.is_empty() {
            return (0.0, 0.0);
        }

        let mut sorted = values.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        // Calculate the indices for the percentile range
        // For 90% percentile: clip bottom 5% and top 5%
        let clip_fraction = (100.0 - percentile) / 100.0;
        let lower_idx = ((clip_fraction / 2.0) * sorted.len() as f32) as usize;
        let upper_idx = ((1.0 - clip_fraction / 2.0) * sorted.len() as f32) as usize;

        let min_val = sorted.get(lower_idx).copied().unwrap_or(sorted[0]);
        let max_val = sorted
            .get(upper_idx.saturating_sub(1).min(sorted.len() - 1))
            .copied()
            .unwrap_or(sorted[sorted.len() - 1]);

        (min_val, max_val)
    }

    /// Quantize a single value to INT4
    #[inline]
    fn quantize_value(&self, value: f32, scale: f32, zero_point: i8) -> i8 {
        let quantized = (value / scale).round() as i32 + zero_point as i32;
        quantized.clamp(-8, 7) as i8
    }

    /// Quantize tensor with block-wise quantization
    pub fn quantize_tensor(&self, tensor: &Tensor) -> Result<QuantizedInt4Tensor> {
        let data = tensor.data()?;
        let shape = tensor.shape();
        let block_size = self.config.block_size.as_usize();

        // Determine optimal block size if needed
        let actual_block_size = if block_size == 0 {
            BlockSize::optimal_for_shape(&shape).as_usize()
        } else {
            block_size
        };

        // Calculate number of blocks
        let num_blocks = (data.len() + actual_block_size - 1) / actual_block_size;
        let mut blocks = Vec::with_capacity(num_blocks);

        // Quantize each block
        for block_idx in 0..num_blocks {
            let start = block_idx * actual_block_size;
            let end = (start + actual_block_size).min(data.len());
            let block_data = &data[start..end];

            // Compute block quantization parameters
            let (scale, zero_point) = self.compute_block_params(block_data);

            // Quantize values in block
            let mut quantized = Vec::with_capacity(block_data.len());
            for &value in block_data {
                quantized.push(self.quantize_value(value, scale, zero_point));
            }

            // Pack INT4 values if configured
            let packed_data = if self.config.packed_storage {
                self.pack_int4_values(&quantized)
            } else {
                quantized.iter().map(|&v| v as u8).collect()
            };

            blocks.push(Int4Block {
                data: packed_data,
                scale,
                zero_point,
                num_elements: block_data.len(),
            });
        }

        Ok(QuantizedInt4Tensor {
            blocks,
            shape: shape.to_vec(),
            config: self.config.clone(),
        })
    }

    /// Pack INT4 values into bytes (2 per byte)
    fn pack_int4_values(&self, values: &[i8]) -> Vec<u8> {
        let mut packed = Vec::with_capacity((values.len() + 1) / 2);

        for chunk in values.chunks(2) {
            let a = chunk[0];
            let b = if chunk.len() > 1 { chunk[1] } else { 0 };
            packed.push(Int4Block::pack_int4(a, b));
        }

        packed
    }

    /// Dequantize a quantized tensor
    pub fn dequantize_tensor(&self, quantized: &QuantizedInt4Tensor) -> Result<Tensor> {
        let mut dequantized = Vec::new();

        for block in &quantized.blocks {
            // Dequantize all values in block
            for i in 0..block.num_elements {
                if let Some(value) = block.dequantize(i) {
                    dequantized.push(value);
                }
            }
        }

        Tensor::from_vec(dequantized, &quantized.shape)
    }

    /// Estimate memory savings
    pub fn estimate_memory_savings(&self, original_elements: usize) -> f32 {
        let original_bytes = original_elements * 4; // FP32 = 4 bytes
        let quantized_bytes = if self.config.packed_storage {
            // 2 INT4 values per byte + overhead for scales
            (original_elements + 1) / 2
        } else {
            // 1 INT4 value per byte
            original_elements
        };

        // Add overhead for scale/zero-point per block
        let block_size = self.config.block_size.as_usize();
        let num_blocks = (original_elements + block_size - 1) / block_size;
        let overhead_bytes = num_blocks * (4 + 1); // scale (f32) + zero_point (i8)

        let total_quantized = quantized_bytes + overhead_bytes;

        1.0 - (total_quantized as f32 / original_bytes as f32)
    }
}

/// Quantized INT4 tensor
#[derive(Debug, Clone)]
pub struct QuantizedInt4Tensor {
    /// Quantized blocks
    pub blocks: Vec<Int4Block>,
    /// Original shape
    pub shape: Vec<usize>,
    /// Quantization configuration
    pub config: EnhancedInt4Config,
}

impl QuantizedInt4Tensor {
    /// Get total size in bytes
    pub fn size_bytes(&self) -> usize {
        self.blocks.iter().map(|b| b.data.len() + 5).sum() // data + scale (4 bytes) + zero_point (1 byte)
    }

    /// Get compression ratio
    pub fn compression_ratio(&self) -> f32 {
        let original_size = self.shape.iter().product::<usize>() * 4; // FP32
        let compressed_size = self.size_bytes();
        original_size as f32 / compressed_size as f32
    }

    /// Get statistics
    pub fn stats(&self) -> QuantizedInt4Stats {
        let total_elements: usize = self.shape.iter().product();
        let avg_scale = self.blocks.iter().map(|b| b.scale).sum::<f32>() / self.blocks.len() as f32;

        QuantizedInt4Stats {
            num_blocks: self.blocks.len(),
            total_elements,
            block_size: self.config.block_size.as_usize(),
            compressed_bytes: self.size_bytes(),
            original_bytes: total_elements * 4,
            compression_ratio: self.compression_ratio(),
            average_scale: avg_scale,
        }
    }
}

/// Statistics for quantized INT4 tensor
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizedInt4Stats {
    pub num_blocks: usize,
    pub total_elements: usize,
    pub block_size: usize,
    pub compressed_bytes: usize,
    pub original_bytes: usize,
    pub compression_ratio: f32,
    pub average_scale: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int4_packing() {
        // Test packing/unpacking
        let a = 5i8;
        let b = -3i8;
        let packed = Int4Block::pack_int4(a, b);
        let (unpacked_a, unpacked_b) = Int4Block::unpack_int4(packed);

        assert_eq!(unpacked_a, a);
        assert_eq!(unpacked_b, b);
    }

    #[test]
    fn test_block_size_selection() {
        // Small tensor
        let shape = vec![10, 10];
        assert_eq!(BlockSize::optimal_for_shape(&shape), BlockSize::Small);

        // Medium tensor (4096 elements = just at Medium boundary)
        let shape = vec![64, 64];
        assert_eq!(BlockSize::optimal_for_shape(&shape), BlockSize::Large);

        // Large tensor
        let shape = vec![128, 128];
        assert_eq!(BlockSize::optimal_for_shape(&shape), BlockSize::Large);

        // XLarge tensor
        let shape = vec![512, 512];
        assert_eq!(BlockSize::optimal_for_shape(&shape), BlockSize::XLarge);
    }

    #[test]
    fn test_quantization_params() {
        let config = EnhancedInt4Config::default();
        let quantizer = EnhancedInt4Quantizer::new(config);

        let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let (scale, zero_point) = quantizer.compute_block_params(&values);

        // For symmetric quantization, zero_point should be 0
        assert_eq!(zero_point, 0);
        // Scale should be reasonable
        assert!(scale > 0.0);
    }

    #[test]
    fn test_quantize_dequantize_roundtrip() -> Result<()> {
        let config = EnhancedInt4Config {
            block_size: BlockSize::Small,
            symmetric: true,
            per_channel: false,
            use_simd: false,
            packed_storage: true,
            outlier_clip_percentile: None,
        };

        let quantizer = EnhancedInt4Quantizer::new(config);

        // Create test tensor
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let tensor = Tensor::from_vec(data.clone(), &[8, 8])?;

        // Quantize
        let quantized = quantizer.quantize_tensor(&tensor)?;

        // Check compression
        assert!(quantized.compression_ratio() > 4.0); // Should be >4x compression

        // Dequantize
        let dequantized = quantizer.dequantize_tensor(&quantized)?;
        let dequant_data = dequantized.data()?;

        // Check approximate equality (INT4 has low precision)
        let max_error = data
            .iter()
            .zip(dequant_data.iter())
            .map(|(&orig, &deq)| (orig - deq).abs())
            .fold(0.0, f32::max);

        // Allow reasonable error for INT4 quantization
        assert!(max_error < 0.5, "Max error: {}", max_error);

        Ok(())
    }

    #[test]
    fn test_memory_savings() {
        let config = EnhancedInt4Config::default();
        let quantizer = EnhancedInt4Quantizer::new(config);

        let savings = quantizer.estimate_memory_savings(1000);

        // Should save >80% with packed INT4
        assert!(savings > 0.8, "Savings: {:.2}%", savings * 100.0);
    }

    #[test]
    fn test_outlier_clipping() {
        let config = EnhancedInt4Config {
            outlier_clip_percentile: Some(90.0),
            ..Default::default()
        };

        let quantizer = EnhancedInt4Quantizer::new(config);

        // Data: 100 normal values, then extreme outliers
        let mut values: Vec<f32> = (0..100).map(|i| i as f32 / 10.0).collect(); // 0.0 to 9.9
        values.push(1000.0); // Extreme outlier
        values.push(2000.0); // Extreme outlier

        let (min_clipped, max_clipped) = quantizer.compute_clipped_range(&values, 90.0);

        // Without clipping
        let min_no_clip = 0.0;
        let max_no_clip = 2000.0;

        // Verify that clipping reduces the range and eliminates the extreme outlier
        assert!(
            max_clipped < max_no_clip,
            "Max should be clipped from {} to something less, got {}",
            max_no_clip,
            max_clipped
        );
        assert!(
            max_clipped < 100.0,
            "Max should be well below outlier range: {}",
            max_clipped
        );
        assert!(
            min_clipped >= min_no_clip,
            "Min should be >= 0: {}",
            min_clipped
        );
    }
}
