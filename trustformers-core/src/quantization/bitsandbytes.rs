//! Bitsandbytes-compatible quantization for TrustformeRS
//!
//! This module provides compatibility with the bitsandbytes library,
//! enabling efficient 8-bit and 4-bit quantization methods including:
//! - Linear quantization (INT8)
//! - Dynamic tree quantization
//! - Block-wise quantization
//! - Stochastic quantization

#![allow(unused_variables)] // BitsAndBytes quantization

use crate::{
    errors::{invalid_input, Result},
    tensor::{DType, Tensor},
};
use std::collections::HashMap;

/// Quantization configuration compatible with bitsandbytes
#[derive(Debug, Clone)]
pub struct BitsAndBytesConfig {
    /// Quantization bit width (4 or 8)
    pub bits: u8,
    /// Use dynamic tree quantization
    pub dynamic_tree: bool,
    /// Block size for block-wise quantization
    pub block_size: usize,
    /// Use stochastic quantization
    pub stochastic: bool,
    /// Percentile for outlier detection
    pub outlier_threshold: f32,
    /// Use nested quantization for scales
    pub nested_quantization: bool,
}

impl Default for BitsAndBytesConfig {
    fn default() -> Self {
        Self {
            bits: 8,
            dynamic_tree: false,
            block_size: 256,
            stochastic: false,
            outlier_threshold: 0.99,
            nested_quantization: false,
        }
    }
}

/// Quantization state for bitsandbytes compatibility
#[derive(Debug, Clone)]
pub struct QuantState {
    /// Quantized data
    pub data: Tensor,
    /// Scale factors
    pub scale: Tensor,
    /// Zero points (optional for symmetric quantization)
    pub zero_point: Option<Tensor>,
    /// Outlier indices for mixed precision
    pub outliers: Option<Vec<usize>>,
    /// Original data type
    pub original_dtype: DType,
    /// Block sizes used for quantization
    pub block_sizes: Vec<usize>,
    /// Original tensor shape (before quantization)
    pub original_shape: Vec<usize>,
}

/// Linear 8-bit quantization (LLM.int8())
pub fn quantize_int8(tensor: &Tensor, config: &BitsAndBytesConfig) -> Result<QuantState> {
    let original_dtype = tensor.dtype();
    let shape = tensor.shape();

    // Flatten tensor for processing
    let total_elements = tensor.shape().iter().product::<usize>();
    let flattened = tensor.reshape(&[total_elements])?;
    let num_elements = flattened.shape()[0];

    // Calculate block-wise statistics
    let num_blocks = num_elements.div_ceil(config.block_size);
    let mut scales = Vec::with_capacity(num_blocks);
    let mut zero_points = Vec::with_capacity(num_blocks);
    let mut quantized_blocks = Vec::new();
    let mut outlier_indices = Vec::new();

    for block_idx in 0..num_blocks {
        let start = block_idx * config.block_size;
        let end = std::cmp::min(start + config.block_size, num_elements);
        let block = flattened.slice_ranges(&[(start, end)])?;

        // Calculate block statistics
        let (min_val, max_val) = block.min_max()?;

        // Detect outliers using percentile threshold
        if config.outlier_threshold < 1.0 {
            let sorted = block.sort()?;
            let lower_idx = ((1.0 - config.outlier_threshold) * (end - start) as f32) as usize;
            let upper_idx = (config.outlier_threshold * (end - start) as f32) as usize;

            let lower_bound = sorted.get_float(lower_idx)?;
            let upper_bound = sorted.get_float(upper_idx)?;

            // Mark outliers
            for i in start..end {
                let val = flattened.get_float(i)?;
                if val < lower_bound || val > upper_bound {
                    outlier_indices.push(i);
                }
            }
        }

        // Calculate scale and zero point
        let scale = (max_val - min_val) / 255.0;
        let zero_point = -min_val / scale;

        scales.push(scale);
        zero_points.push(zero_point);

        // Quantize block
        let quantized = block.sub_scalar(min_val)?.div_scalar(scale)?.round()?.clamp(0.0, 255.0)?;

        quantized_blocks.push(quantized);
    }

    // Concatenate quantized blocks
    let quantized_data =
        Tensor::concat(&quantized_blocks, 0)?.to_dtype(DType::I64)?.reshape(&shape)?;

    // Create scale and zero point tensors
    let scale_tensor = Tensor::from_vec(scales, &[num_blocks])?;
    let zero_point_tensor = Tensor::from_vec(zero_points, &[num_blocks])?;

    // Apply nested quantization to scales if requested
    let final_scale = if config.nested_quantization {
        quantize_scales(&scale_tensor, 8)?
    } else {
        scale_tensor
    };

    Ok(QuantState {
        data: quantized_data,
        scale: final_scale,
        zero_point: Some(zero_point_tensor),
        outliers: if outlier_indices.is_empty() { None } else { Some(outlier_indices) },
        original_dtype,
        block_sizes: vec![config.block_size],
        original_shape: shape.to_vec(),
    })
}

/// 4-bit quantization (NF4/FP4)
pub fn quantize_4bit(tensor: &Tensor, config: &BitsAndBytesConfig) -> Result<QuantState> {
    let original_dtype = tensor.dtype();
    let shape = tensor.shape();

    // Use smaller block size for 4-bit quantization
    let block_size = config.block_size / 2;
    let total_elements = tensor.shape().iter().product::<usize>();
    let flattened = tensor.reshape(&[total_elements])?;
    let num_elements = flattened.shape()[0];
    let num_blocks = num_elements.div_ceil(block_size);

    let mut scales = Vec::with_capacity(num_blocks);
    let mut quantized_blocks = Vec::new();

    // NF4 quantization levels (normalized float 4-bit)
    let nf4_levels = get_nf4_quantization_levels();

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = std::cmp::min(start + block_size, num_elements);
        let block = flattened.slice_ranges(&[(start, end)])?;

        // Normalize block
        let mean = block.mean()?;
        let std = block.std()?;
        let mean_scalar = mean.get_float(0)?;
        let std_scalar = std.get_float(0)?;
        let normalized = block.sub_scalar(mean_scalar)?.div_scalar(std_scalar + 1e-8)?;

        // Find scale for mapping to NF4 levels
        let abs_max = normalized.abs()?.max_value()?;
        let scale = abs_max.get_float(0)?;
        scales.push(scale);

        // Quantize to nearest NF4 level
        let mut quantized_values = Vec::with_capacity(end - start);
        for i in 0..(end - start) {
            let val = normalized.get_float(i)? / scale;
            let quantized_idx = find_nearest_nf4_level(val, &nf4_levels);
            quantized_values.push(quantized_idx as f32);
        }

        let quantized = Tensor::from_vec(quantized_values, &[end - start])?;
        quantized_blocks.push(quantized);
    }

    // Pack 4-bit values into bytes
    let quantized_concat = Tensor::concat(&quantized_blocks, 0)?;
    let packed_data = pack_4bit_tensor(&quantized_concat)?;

    let scale_tensor = Tensor::from_vec(scales, &[num_blocks])?;

    Ok(QuantState {
        data: packed_data,
        scale: scale_tensor,
        zero_point: None,
        outliers: None,
        original_dtype,
        block_sizes: vec![block_size],
        original_shape: shape.to_vec(),
    })
}

/// Dynamic tree quantization
pub fn quantize_dynamic_tree(tensor: &Tensor, config: &BitsAndBytesConfig) -> Result<QuantState> {
    // Build quantization tree based on data distribution
    let total_elements = tensor.shape().iter().product::<usize>();
    let flattened = tensor.reshape(&[total_elements])?;
    let histogram = build_histogram(&flattened, 256)?;
    let tree = build_quantization_tree(&histogram, config.bits)?;

    // Map values through tree
    let quantized = apply_tree_quantization(&flattened, &tree)?;

    // Store tree structure as scale information
    let scale_data = serialize_tree(&tree)?;

    Ok(QuantState {
        data: quantized.reshape(&tensor.shape())?,
        scale: scale_data,
        zero_point: None,
        outliers: None,
        original_dtype: tensor.dtype(),
        block_sizes: vec![],
        original_shape: tensor.shape().to_vec(),
    })
}

/// Dequantize tensor from bitsandbytes format
pub fn dequantize_bitsandbytes(state: &QuantState, config: &BitsAndBytesConfig) -> Result<Tensor> {
    match config.bits {
        8 => dequantize_int8(state),
        4 => dequantize_4bit(state),
        _ => Err(invalid_input(format!(
            "Unsupported bit width: {}",
            config.bits
        ))),
    }
}

/// Dequantize INT8 tensor
fn dequantize_int8(state: &QuantState) -> Result<Tensor> {
    let shape = state.data.shape();
    let total_elements = state.data.shape().iter().product::<usize>();
    let flattened = state.data.reshape(&[total_elements])?;
    let num_elements = flattened.shape()[0];

    // Get block size from state
    let block_size = state.block_sizes.first().copied().unwrap_or(256);
    let num_blocks = num_elements.div_ceil(block_size);

    let mut dequantized_blocks = Vec::new();

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = std::cmp::min(start + block_size, num_elements);
        let block = flattened.slice_ranges(&[(start, end)])?;

        // Get scale and zero point for this block
        let scale = state.scale.get_float(block_idx)?;
        let zero_point = state
            .zero_point
            .as_ref()
            .map(|zp| zp.get_float(block_idx))
            .transpose()?
            .unwrap_or(0.0);

        // Dequantize block
        let dequantized = block.to_dtype(DType::F32)?.sub_scalar(zero_point)?.scalar_mul(scale)?;

        dequantized_blocks.push(dequantized);
    }

    // Concatenate and reshape
    Tensor::concat(&dequantized_blocks, 0)?
        .reshape(&shape)?
        .to_dtype(state.original_dtype)
}

/// Dequantize 4-bit tensor
fn dequantize_4bit(state: &QuantState) -> Result<Tensor> {
    // Unpack 4-bit values
    let unpacked = unpack_4bit_tensor(&state.data)?;
    let nf4_levels = get_nf4_quantization_levels();

    let original_shape = &state.original_shape;
    let block_size = state.block_sizes.first().copied().unwrap_or(128);
    let num_elements = unpacked.shape()[0];
    let num_blocks = num_elements.div_ceil(block_size);

    let mut dequantized_blocks = Vec::new();

    for block_idx in 0..num_blocks {
        let start = block_idx * block_size;
        let end = std::cmp::min(start + block_size, num_elements);
        let block = unpacked.slice(0, start, end)?;

        let scale = state.scale.get_float(block_idx)?;

        // Map from NF4 indices to values
        let mut values = Vec::with_capacity(end - start);
        for i in 0..(end - start) {
            let idx = block.get_float(i)? as usize;
            let nf4_value = nf4_levels[idx];
            values.push(nf4_value * scale);
        }

        let dequantized = Tensor::from_vec(values, &[end - start])?;
        dequantized_blocks.push(dequantized);
    }

    Tensor::concat(&dequantized_blocks, 0)?
        .reshape(original_shape)?
        .to_dtype(state.original_dtype)
}

/// Get NF4 quantization levels
fn get_nf4_quantization_levels() -> Vec<f32> {
    vec![
        -1.0,
        -0.6961928009986877,
        -0.5250730514526367,
        -0.39491748809814453,
        -0.28444138169288635,
        -0.18477343022823334,
        -0.09105003625154495,
        0.0,
        0.07958029955625534,
        0.16093020141124725,
        0.24611230194568634,
        0.33791524171829224,
        0.44070982933044434,
        0.5626170039176941,
        0.7229568362236023,
        1.0,
    ]
}

/// Find nearest NF4 quantization level
fn find_nearest_nf4_level(value: f32, levels: &[f32]) -> usize {
    let mut min_dist = f32::INFINITY;
    let mut best_idx = 0;

    for (idx, &level) in levels.iter().enumerate() {
        let dist = (value - level).abs();
        if dist < min_dist {
            min_dist = dist;
            best_idx = idx;
        }
    }

    best_idx
}

/// Pack 4-bit values into bytes
fn pack_4bit_tensor(tensor: &Tensor) -> Result<Tensor> {
    let data = tensor.to_vec_f32()?;
    let mut packed = Vec::with_capacity(data.len().div_ceil(2));

    for i in (0..data.len()).step_by(2) {
        let low = data[i] as u8 & 0x0F;
        let high = if i + 1 < data.len() { (data[i + 1] as u8 & 0x0F) << 4 } else { 0 };
        packed.push(low | high);
    }

    let packed_f32: Vec<f32> = packed.into_iter().map(|x| x as f32).collect();
    let len = packed_f32.len();
    Tensor::from_vec(packed_f32, &[len])
}

/// Unpack 4-bit values from bytes
fn unpack_4bit_tensor(tensor: &Tensor) -> Result<Tensor> {
    let packed = tensor.to_vec_u8()?;
    let mut unpacked = Vec::with_capacity(packed.len() * 2);

    for byte in packed {
        unpacked.push((byte & 0x0F) as f32);
        unpacked.push(((byte >> 4) & 0x0F) as f32);
    }

    let len = unpacked.len();
    Tensor::from_vec(unpacked, &[len])
}

/// Build histogram for dynamic quantization
fn build_histogram(tensor: &Tensor, bins: usize) -> Result<Vec<f32>> {
    let data = tensor.to_vec_f32()?;
    let (min_val, max_val) = tensor.min_max()?;
    let range = max_val - min_val;
    let bin_width = range / bins as f32;

    let mut histogram = vec![0.0; bins];

    for &value in &data {
        let bin_idx = ((value - min_val) / bin_width).floor() as usize;
        let bin_idx = bin_idx.min(bins - 1);
        histogram[bin_idx] += 1.0;
    }

    // Normalize
    let total: f32 = histogram.iter().sum();
    for count in &mut histogram {
        *count /= total;
    }

    Ok(histogram)
}

/// Quantization tree node
#[derive(Debug, Clone)]
struct TreeNode {
    threshold: f32,
    left: Option<Box<TreeNode>>,
    right: Option<Box<TreeNode>>,
    value: Option<u8>,
}

/// Build quantization tree from histogram
fn build_quantization_tree(histogram: &[f32], bits: u8) -> Result<TreeNode> {
    // Simplified tree building - in practice, this would use entropy-based splitting
    let levels = 1 << bits;
    let mut thresholds = Vec::with_capacity(levels - 1);

    // Create uniform thresholds for now
    for i in 1..levels {
        thresholds.push(i as f32 / levels as f32);
    }

    // Build binary tree
    fn build_node(thresholds: &[f32], start: usize, end: usize) -> TreeNode {
        if start >= end {
            TreeNode {
                threshold: 0.0,
                left: None,
                right: None,
                value: Some(start as u8),
            }
        } else {
            let mid = (start + end) / 2;
            TreeNode {
                threshold: thresholds[mid],
                left: Some(Box::new(build_node(thresholds, start, mid))),
                right: Some(Box::new(build_node(thresholds, mid + 1, end))),
                value: None,
            }
        }
    }

    Ok(build_node(&thresholds, 0, levels - 1))
}

/// Apply tree quantization
fn apply_tree_quantization(tensor: &Tensor, tree: &TreeNode) -> Result<Tensor> {
    let data = tensor.to_vec_f32()?;
    let mut quantized = Vec::with_capacity(data.len());

    for &value in &data {
        let quantized_value = traverse_tree(value, tree);
        quantized.push(quantized_value as f32);
    }

    Tensor::from_vec(quantized, &tensor.shape())
}

/// Traverse quantization tree
fn traverse_tree(value: f32, node: &TreeNode) -> u8 {
    if let Some(leaf_value) = node.value {
        leaf_value
    } else if value < node.threshold {
        traverse_tree(
            value,
            node.left.as_ref().expect("non-leaf node must have left child"),
        )
    } else {
        traverse_tree(
            value,
            node.right.as_ref().expect("non-leaf node must have right child"),
        )
    }
}

/// Serialize tree structure
fn serialize_tree(tree: &TreeNode) -> Result<Tensor> {
    // Simplified serialization - store thresholds in order
    let mut thresholds = Vec::new();
    collect_thresholds(tree, &mut thresholds);
    let len = thresholds.len();
    Tensor::from_vec(thresholds, &[len])
}

/// Collect thresholds from tree
fn collect_thresholds(node: &TreeNode, thresholds: &mut Vec<f32>) {
    if node.value.is_none() {
        thresholds.push(node.threshold);
        if let Some(left) = &node.left {
            collect_thresholds(left, thresholds);
        }
        if let Some(right) = &node.right {
            collect_thresholds(right, thresholds);
        }
    }
}

/// Quantize scale factors using nested quantization
fn quantize_scales(scales: &Tensor, bits: u8) -> Result<Tensor> {
    // Simple uniform quantization for scales
    let (min_val, max_val) = scales.min_max()?;
    let levels = (1 << bits) as f32;
    let scale = (max_val - min_val) / (levels - 1.0);

    scales.sub_scalar(min_val)?.div_scalar(scale)?.round()?.clamp(0.0, levels - 1.0)
}

/// Convert TrustformeRS tensor to bitsandbytes-compatible format
pub fn to_bitsandbytes_format(
    tensor: &Tensor,
    config: &BitsAndBytesConfig,
) -> Result<HashMap<String, Tensor>> {
    let state = match config.bits {
        8 => quantize_int8(tensor, config)?,
        4 => quantize_4bit(tensor, config)?,
        _ => {
            return Err(invalid_input(format!(
                "Unsupported bit width: {}",
                config.bits
            )))
        },
    };

    let mut result = HashMap::new();
    result.insert("data".to_string(), state.data);
    result.insert("scale".to_string(), state.scale);

    if let Some(zero_point) = state.zero_point {
        result.insert("zero_point".to_string(), zero_point);
    }

    if let Some(outliers) = state.outliers {
        let outlier_tensor = Tensor::from_vec(
            outliers.iter().map(|&idx| idx as f32).collect(),
            &[outliers.len()],
        )?;
        result.insert("outliers".to_string(), outlier_tensor);
    }

    Ok(result)
}

/// Convert from bitsandbytes format to TrustformeRS tensor
pub fn from_bitsandbytes_format(
    data: HashMap<String, Tensor>,
    config: &BitsAndBytesConfig,
) -> Result<Tensor> {
    let quantized_data = data
        .get("data")
        .ok_or_else(|| invalid_input("Missing 'data' tensor".to_string()))?;
    let scale = data
        .get("scale")
        .ok_or_else(|| invalid_input("Missing 'scale' tensor".to_string()))?;
    let zero_point = data.get("zero_point");
    let outliers = data
        .get("outliers")
        .map(|t| t.to_vec_f32().map(|v| v.iter().map(|&x| x as usize).collect()))
        .transpose()?;

    let state = QuantState {
        data: quantized_data.clone(),
        scale: scale.clone(),
        zero_point: zero_point.cloned(),
        outliers,
        original_dtype: DType::F32,
        block_sizes: vec![config.block_size],
        original_shape: quantized_data.shape().to_vec(),
    };

    dequantize_bitsandbytes(&state, config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_int8_quantization() -> Result<()> {
        let tensor = Tensor::randn(&[64, 64])?;
        let config = BitsAndBytesConfig::default();

        let state = quantize_int8(&tensor, &config)?;
        let dequantized = dequantize_int8(&state)?;

        // Check shape preservation
        assert_eq!(tensor.shape(), dequantized.shape());

        // Check reconstruction error
        let error = tensor.sub(&dequantized)?.abs()?.mean()?;
        let error_val = error.get_float(0)?;
        assert!(
            error_val < 0.1,
            "Reconstruction error too high: {}",
            error_val
        );

        Ok(())
    }

    #[test]
    fn test_4bit_quantization() -> Result<()> {
        let tensor = Tensor::randn(&[32, 32])?;
        let config = BitsAndBytesConfig {
            bits: 4,
            ..Default::default()
        };

        let state = quantize_4bit(&tensor, &config)?;
        let dequantized = dequantize_4bit(&state)?;

        assert_eq!(tensor.shape(), dequantized.shape());
        Ok(())
    }

    #[test]
    fn test_bitsandbytes_format_conversion() -> Result<()> {
        let tensor = Tensor::randn(&[128, 128])?;
        let config = BitsAndBytesConfig::default();

        let bnb_format = to_bitsandbytes_format(&tensor, &config)?;
        let reconstructed = from_bitsandbytes_format(bnb_format, &config)?;

        assert_eq!(tensor.shape(), reconstructed.shape());
        Ok(())
    }
}
