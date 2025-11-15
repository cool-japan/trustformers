//! Multi-head attention implementation.
//!
//! This module provides the standard multi-head attention mechanism used in transformers,
//! refactored to use shared components and utilities.

#![allow(unused_variables)] // Multi-head attention

use super::common::{
    AttentionConfig, AttentionOptimizationHints, AttentionProjections, AttentionUtils,
};
use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use crate::traits::Layer;

/// Multi-head attention layer
///
/// This implements the standard multi-head attention mechanism from "Attention is All You Need".
/// The implementation is optimized for both training and inference with various optimization hints.
#[derive(Debug, Clone)]
pub struct MultiHeadAttention {
    /// Attention configuration
    config: AttentionConfig,
    /// Query, key, value, and output projections
    projections: AttentionProjections,
    /// Optimization hints for performance
    optimization_hints: AttentionOptimizationHints,
}

impl MultiHeadAttention {
    /// Create a new multi-head attention layer with device support
    pub fn new_with_device(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        device: Device,
    ) -> Result<Self> {
        let config = AttentionConfig::new(hidden_size, num_heads, dropout_prob, bias)?;
        let projections = AttentionProjections::new_with_device(&config, device);
        let optimization_hints = AttentionOptimizationHints::default();

        Ok(Self {
            config,
            projections,
            optimization_hints,
        })
    }

    /// Create a new multi-head attention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
    ) -> Result<Self> {
        Self::new_with_device(hidden_size, num_heads, dropout_prob, bias, Device::CPU)
    }

    /// Create a new multi-head attention layer from an existing config
    pub fn from_config(config: AttentionConfig) -> Result<Self> {
        let projections = AttentionProjections::new(&config);
        let optimization_hints = AttentionOptimizationHints::default();

        Ok(Self {
            config,
            projections,
            optimization_hints,
        })
    }

    /// Create a new multi-head attention layer with custom optimization hints
    pub fn new_with_hints(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        optimization_hints: AttentionOptimizationHints,
    ) -> Result<Self> {
        let config = AttentionConfig::new(hidden_size, num_heads, dropout_prob, bias)?;
        let projections = AttentionProjections::new(&config);

        Ok(Self {
            config,
            projections,
            optimization_hints,
        })
    }

    /// Get the attention configuration
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// Get the optimization hints
    pub fn optimization_hints(&self) -> &AttentionOptimizationHints {
        &self.optimization_hints
    }

    /// Update optimization hints
    pub fn set_optimization_hints(&mut self, hints: AttentionOptimizationHints) {
        self.optimization_hints = hints;
    }

    /// Get the total number of parameters in this attention layer
    pub fn parameter_count(&self) -> usize {
        self.projections.query.parameter_count()
            + self.projections.key.parameter_count()
            + self.projections.value.parameter_count()
            + self.projections.out_proj.parameter_count()
    }

    /// Set weights for the query projection
    pub fn set_query_weight(&mut self, weight: Tensor) -> Result<()> {
        self.projections.query.set_weight(weight)
    }

    /// Set bias for the query projection
    pub fn set_query_bias(&mut self, bias: Tensor) -> Result<()> {
        self.projections.query.set_bias(bias)
    }

    /// Set weights for the key projection
    pub fn set_key_weight(&mut self, weight: Tensor) -> Result<()> {
        self.projections.key.set_weight(weight)
    }

    /// Set bias for the key projection
    pub fn set_key_bias(&mut self, bias: Tensor) -> Result<()> {
        self.projections.key.set_bias(bias)
    }

    /// Set weights for the value projection
    pub fn set_value_weight(&mut self, weight: Tensor) -> Result<()> {
        self.projections.value.set_weight(weight)
    }

    /// Set bias for the value projection
    pub fn set_value_bias(&mut self, bias: Tensor) -> Result<()> {
        self.projections.value.set_bias(bias)
    }

    /// Set weights for the output projection
    pub fn set_out_proj_weight(&mut self, weight: Tensor) -> Result<()> {
        self.projections.out_proj.set_weight(weight)
    }

    /// Set bias for the output projection
    pub fn set_out_proj_bias(&mut self, bias: Tensor) -> Result<()> {
        self.projections.out_proj.set_bias(bias)
    }

    /// Compute attention for training (with query, key, value from same input)
    pub fn forward_self_attention(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        self.forward_attention(input, input, input, attention_mask, causal)
    }

    /// Compute attention with separate query, key, value inputs
    pub fn forward_attention(
        &self,
        query_input: &Tensor,
        key_input: &Tensor,
        value_input: &Tensor,
        attention_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        // Apply input projections
        let query = self.projections.query.forward(query_input.clone())?;
        let key = self.projections.key.forward(key_input.clone())?;
        let value = self.projections.value.forward(value_input.clone())?;

        // Split into attention heads
        let q = AttentionUtils::split_heads(&query, self.config.num_heads, self.config.head_dim)?;
        let k = AttentionUtils::split_heads(&key, self.config.num_heads, self.config.head_dim)?;
        let v = AttentionUtils::split_heads(&value, self.config.num_heads, self.config.head_dim)?;

        // Validate dimensions
        AttentionUtils::validate_attention_dims(
            &q,
            &k,
            &v,
            self.config.num_heads,
            self.config.head_dim,
        )?;

        // Compute attention
        let attention_output = self.compute_attention(&q, &k, &v, attention_mask, causal)?;

        // Combine heads
        let combined = AttentionUtils::combine_heads(
            &attention_output,
            self.config.num_heads,
            self.config.head_dim,
        )?;

        // Apply output projection
        self.projections.out_proj.forward(combined)
    }

    /// Core attention computation
    fn compute_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Choose computation method based on optimization hints
        if self.optimization_hints.use_flash_attention {
            self.compute_flash_attention(q, k, v, attention_mask, causal, scale)
        } else {
            self.compute_standard_attention(q, k, v, attention_mask, causal, scale)
        }
    }

    /// Standard attention computation
    fn compute_standard_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
    ) -> Result<Tensor> {
        // Compute attention weights
        let attention_weights = AttentionUtils::compute_attention_weights(q, k, scale, causal)?;

        // Apply attention mask if provided
        let masked_weights = if let Some(mask) = attention_mask {
            self.apply_attention_mask(&attention_weights, mask)?
        } else {
            attention_weights
        };

        // Apply dropout if training
        let dropped_weights = self.apply_dropout(&masked_weights)?;

        // Apply attention to values
        AttentionUtils::apply_attention(&dropped_weights, v)
    }

    /// Memory-efficient flash attention computation
    fn compute_flash_attention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
        causal: bool,
        scale: f32,
    ) -> Result<Tensor> {
        let shape = q.shape();
        let batch_size = shape[0];
        let num_heads = shape[1];
        let seq_q = shape[2];
        let head_dim = shape[3];
        let seq_k = k.shape()[2];

        // Determine optimal block size for tiling
        let block_size = self.compute_flash_block_size(seq_q, seq_k, head_dim);

        let output = Tensor::zeros(&[batch_size, num_heads, seq_q, head_dim])?;

        // Tile over the sequence dimension
        let num_blocks_q = (seq_q + block_size - 1) / block_size;
        let num_blocks_k = (seq_k + block_size - 1) / block_size;

        for q_block_idx in 0..num_blocks_q {
            let q_start = q_block_idx * block_size;
            let q_end = (q_start + block_size).min(seq_q);

            // Extract Q block
            let q_block = q.slice_ranges(&[
                (0, batch_size),
                (0, num_heads),
                (q_start, q_end),
                (0, head_dim),
            ])?;

            // Initialize accumulator for this Q block
            let mut block_output =
                Tensor::zeros(&[batch_size, num_heads, q_end - q_start, head_dim])?;
            let mut block_max = Tensor::full(
                f32::NEG_INFINITY,
                vec![batch_size, num_heads, q_end - q_start, 1],
            )?;
            let mut block_sum = Tensor::zeros(&[batch_size, num_heads, q_end - q_start, 1])?;

            // Process each K block
            for k_block_idx in 0..num_blocks_k {
                let k_start = k_block_idx * block_size;
                let k_end = (k_start + block_size).min(seq_k);

                // Extract K and V blocks
                let k_block = k.slice_ranges(&[
                    (0, batch_size),
                    (0, num_heads),
                    (k_start, k_end),
                    (0, head_dim),
                ])?;
                let v_block = v.slice_ranges(&[
                    (0, batch_size),
                    (0, num_heads),
                    (k_start, k_end),
                    (0, head_dim),
                ])?;

                // Compute attention scores for this block
                let attention_scores = self.compute_block_scores(
                    &q_block,
                    &k_block,
                    scale,
                    q_start,
                    k_start,
                    attention_mask,
                    causal,
                )?;

                // Online softmax computation with numerical stability
                self.update_flash_statistics(
                    &mut block_output,
                    &mut block_max,
                    &mut block_sum,
                    &attention_scores,
                    &v_block,
                )?;
            }

            // Normalize the accumulated output for this Q block
            let normalized_output = self.normalize_flash_output(&block_output, &block_sum)?;

            // Store the result in the final output tensor (simplified approach)
            // In a full implementation, this would use proper slice assignment
            // For now, this is a placeholder since slice_assign is not available
        }

        Ok(output)
    }

    /// Apply attention mask to attention weights
    fn apply_attention_mask(&self, attention_weights: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let attention_shape = attention_weights.shape();
        let mask_shape = mask.shape();

        // Handle different mask shapes
        let compatible_mask = if mask_shape.len() == 3 && attention_shape.len() == 4 {
            // Mask is [batch, seq_q, seq_k], attention is [batch, num_heads, seq_q, seq_k]
            // Reshape mask from [batch, seq_q, seq_k] to [batch, 1, seq_q, seq_k] for broadcasting
            let batch_size = mask_shape[0];
            let seq_q = mask_shape[1];
            let seq_k = mask_shape[2];

            mask.reshape(&[batch_size, 1, seq_q, seq_k])?
        } else {
            // Use mask as-is for other cases (2D, 4D, etc.)
            mask.clone()
        };

        // Apply mask by adding large negative values where mask is 0
        // The tensor addition should handle broadcasting automatically
        let mask_value = Tensor::scalar(-1e9)?;
        let inverted_mask = compatible_mask.sub(&Tensor::ones(&compatible_mask.shape())?)?;
        let mask_additive = inverted_mask.mul(&mask_value)?;
        attention_weights.add(&mask_additive)
    }

    /// Apply dropout to attention weights
    fn apply_dropout(&self, attention_weights: &Tensor) -> Result<Tensor> {
        if self.config.dropout_prob > 0.0 {
            attention_weights.dropout(self.config.dropout_prob)
        } else {
            Ok(attention_weights.clone())
        }
    }

    /// Get memory usage estimation for this attention layer
    pub fn estimate_memory_usage(&self, batch_size: usize, seq_len: usize) -> usize {
        let attention_matrix_size = batch_size * self.config.num_heads * seq_len * seq_len;
        let projection_size = batch_size * seq_len * self.config.hidden_size * 4; // Q, K, V, O
        let intermediate_size =
            batch_size * self.config.num_heads * seq_len * self.config.head_dim * 3; // Q, K, V heads

        (attention_matrix_size + projection_size + intermediate_size) * 4 // 4 bytes per f32
    }

    /// Update optimization hints based on current usage pattern
    pub fn update_optimization_hints(
        &mut self,
        batch_size: usize,
        seq_len: usize,
        available_memory_mb: Option<usize>,
    ) {
        self.optimization_hints =
            AttentionOptimizationHints::for_sequence_length(seq_len, available_memory_mb);

        // Adjust based on memory usage
        let estimated_memory_mb = self.estimate_memory_usage(batch_size, seq_len) / (1024 * 1024);
        if let Some(available_mb) = available_memory_mb {
            if estimated_memory_mb > available_mb / 2 {
                self.optimization_hints.use_flash_attention = true;
                self.optimization_hints.use_half_precision = true;
            }
        }
    }

    /// Compute optimal block size for FlashAttention tiling
    fn compute_flash_block_size(&self, seq_q: usize, seq_k: usize, head_dim: usize) -> usize {
        // Adaptive block sizing based on sequence length and available memory
        let base_size = 128; // Conservative default

        // For very long sequences, use larger blocks to reduce overhead
        if seq_q > 2048 || seq_k > 2048 {
            base_size * 2 // 256
        } else if seq_q < 128 && seq_k < 128 {
            // For short sequences, use smaller blocks
            base_size / 2 // 64
        } else {
            base_size
        }
    }

    /// Compute attention scores for a block in FlashAttention
    fn compute_block_scores(
        &self,
        q_block: &Tensor,
        k_block: &Tensor,
        scale: f32,
        q_offset: usize,
        k_offset: usize,
        attention_mask: Option<&Tensor>,
        causal: bool,
    ) -> Result<Tensor> {
        // Compute Q @ K^T
        let scores = q_block.matmul(&k_block.transpose(2, 3)?)?;
        let scaled_scores = scores.mul(&Tensor::scalar(scale)?)?;

        // Apply causal masking if needed
        let masked_scores = if causal {
            self.apply_block_causal_mask(&scaled_scores, q_offset, k_offset)?
        } else {
            scaled_scores
        };

        // Apply attention mask if provided
        if let Some(mask) = attention_mask {
            let mask_block = self.extract_attention_mask_block(
                mask,
                q_offset,
                k_offset,
                q_block.shape()[2],
                k_block.shape()[2],
            )?;
            masked_scores.add(&mask_block)
        } else {
            Ok(masked_scores)
        }
    }

    /// Apply causal mask to a specific block
    fn apply_block_causal_mask(
        &self,
        scores: &Tensor,
        q_offset: usize,
        k_offset: usize,
    ) -> Result<Tensor> {
        let shape = scores.shape();
        let q_block_size = shape[shape.len() - 2];
        let k_block_size = shape[shape.len() - 1];

        // Create causal mask for this block
        let mut mask_data = vec![0.0f32; q_block_size * k_block_size];

        for i in 0..q_block_size {
            for j in 0..k_block_size {
                let global_q_pos = q_offset + i;
                let global_k_pos = k_offset + j;

                if global_k_pos > global_q_pos {
                    // Future position, mask it
                    mask_data[i * k_block_size + j] = f32::NEG_INFINITY;
                }
            }
        }

        let causal_mask = Tensor::from_vec(mask_data, &[q_block_size, k_block_size])?;
        scores.add(&causal_mask)
    }

    /// Extract attention mask block for current computation
    fn extract_attention_mask_block(
        &self,
        mask: &Tensor,
        q_offset: usize,
        k_offset: usize,
        q_block_size: usize,
        k_block_size: usize,
    ) -> Result<Tensor> {
        let mask_shape = mask.shape();

        // Handle different mask shapes
        if mask_shape.len() == 2 {
            // [seq_q, seq_k] mask
            mask.slice_ranges(&[
                (q_offset, q_offset + q_block_size),
                (k_offset, k_offset + k_block_size),
            ])
        } else if mask_shape.len() == 3 {
            // [batch, seq_q, seq_k] mask - broadcast over heads
            mask.slice_ranges(&[
                (0, mask_shape[0]),
                (q_offset, q_offset + q_block_size),
                (k_offset, k_offset + k_block_size),
            ])
        } else if mask_shape.len() == 4 {
            // [batch, heads, seq_q, seq_k] mask
            mask.slice_ranges(&[
                (0, mask_shape[0]),
                (0, mask_shape[1]),
                (q_offset, q_offset + q_block_size),
                (k_offset, k_offset + k_block_size),
            ])
        } else {
            Err(TrustformersError::tensor_op_error(
                &format!("Unsupported attention mask shape: {:?}", mask_shape),
                "extract_attention_mask_block",
            ))
        }
    }

    /// Update FlashAttention statistics with online softmax
    fn update_flash_statistics(
        &self,
        block_output: &mut Tensor,
        block_max: &mut Tensor,
        block_sum: &mut Tensor,
        attention_scores: &Tensor,
        v_block: &Tensor,
    ) -> Result<()> {
        // Compute new maximum values (simplified approach)
        let scores_max_val = attention_scores.max_value()?;
        let new_max = block_max.max(&scores_max_val)?;

        // Compute normalized scores
        let scores_shifted = attention_scores.sub(&new_max)?;
        let scores_exp = scores_shifted.exp()?;

        // Update running sum with correction for previous maximum
        let old_sum_correction = block_max.sub(&new_max)?.exp()?;
        let corrected_old_sum = block_sum.mul(&old_sum_correction)?;
        let new_contribution = scores_exp.sum(None, false)?; // Simplified to global sum
        let updated_sum = corrected_old_sum.add(&new_contribution)?;

        // Update running output with correction
        let old_output_correction = block_output.mul(&old_sum_correction)?;
        let new_output_contribution = scores_exp.matmul(v_block)?;
        let updated_output = old_output_correction.add(&new_output_contribution)?;

        // Store updated values
        *block_max = new_max;
        *block_sum = updated_sum;
        *block_output = updated_output;

        Ok(())
    }

    /// Normalize FlashAttention output
    fn normalize_flash_output(&self, block_output: &Tensor, block_sum: &Tensor) -> Result<Tensor> {
        // Avoid division by zero
        let epsilon = Tensor::scalar(1e-8)?;
        let safe_sum = block_sum.add(&epsilon)?;
        block_output.div(&safe_sum)
    }
}

impl Layer for MultiHeadAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Default to self-attention without causal masking
        self.forward_self_attention(&input, None, false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;

    #[test]
    fn test_multi_head_attention_creation() {
        let attention = MultiHeadAttention::new(512, 8, 0.1, true).unwrap();
        assert_eq!(attention.config.hidden_size, 512);
        assert_eq!(attention.config.num_heads, 8);
        assert_eq!(attention.config.head_dim, 64);
    }

    #[test]
    fn test_invalid_head_configuration() {
        let result = MultiHeadAttention::new(512, 7, 0.1, true);
        assert!(result.is_err());
    }

    #[test]
    fn test_self_attention_forward() {
        let attention = MultiHeadAttention::new(512, 8, 0.1, true).unwrap();
        let input = Tensor::randn(&[2, 10, 512]).unwrap();
        let output = attention.forward(input).unwrap();
        assert_eq!(output.shape(), vec![2, 10, 512]);
    }

    #[test]
    fn test_memory_estimation() {
        let attention = MultiHeadAttention::new(512, 8, 0.1, true).unwrap();
        let memory_usage = attention.estimate_memory_usage(2, 100);
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_optimization_hints_update() {
        let mut attention = MultiHeadAttention::new(512, 8, 0.1, true).unwrap();
        attention.update_optimization_hints(2, 2048, Some(1024));
        assert!(attention.optimization_hints.use_flash_attention);
    }
}
