//! Multi-head attention implementation.
//!
//! This module provides the standard multi-head attention mechanism used in transformers,
//! refactored to use shared components and utilities.

use super::common::{
    AttentionConfig, AttentionOptimizationHints, AttentionProjections, AttentionUtils,
};
use super::flash_kernel::{flash_attention, FlashParams};
use crate::device::Device;
use crate::errors::Result;
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

    /// Whether the layer is in training mode (which enables attention dropout).
    pub fn is_training(&self) -> bool {
        self.config.training
    }

    /// Switch between training and inference mode.
    ///
    /// Attention dropout is only applied in training mode, mirroring
    /// [`crate::layers::Dropout`].
    pub fn set_training(&mut self, training: bool) {
        self.config.training = training;
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
        // Pre-softmax scores, with causal masking already applied.
        let scores = AttentionUtils::compute_attention_scores(q, k, scale, causal)?;

        // Apply the attention mask *before* the softmax - adding a large
        // negative penalty to an already-normalised probability is meaningless.
        let masked_scores = match attention_mask {
            Some(mask) => AttentionUtils::apply_attention_mask_to_scores(&scores, mask)?,
            None => scores,
        };

        let attention_weights = masked_scores.softmax(-1)?;

        // Apply dropout if training
        let dropped_weights = self.apply_dropout(&attention_weights)?;

        // Apply attention to values
        AttentionUtils::apply_attention(&dropped_weights, v)
    }

    /// Memory-efficient FlashAttention computation.
    ///
    /// Delegates to the shared block-tiled kernel
    /// ([`super::flash_kernel`]), which keeps per-query-row running softmax
    /// statistics and never materialises the full attention matrix.
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
        let seq_q = shape[2];
        let head_dim = shape[3];
        let seq_k = k.shape()[2];

        let block_size = self.compute_flash_block_size(seq_q, seq_k, head_dim);
        let dropout = if self.config.training { Some(self.config.dropout_prob) } else { None };
        let params = FlashParams {
            scale,
            causal,
            block_q: block_size,
            block_k: block_size,
            dropout: None,
        }
        .with_dropout(dropout)?;
        flash_attention(q, k, v, attention_mask, &params)
    }

    /// Apply dropout to attention weights.
    ///
    /// Dropout is a training-time regulariser: applying it at inference would
    /// make the logits stochastic, so it is gated on
    /// [`MultiHeadAttention::is_training`] exactly like
    /// [`crate::layers::Dropout`].
    fn apply_dropout(&self, attention_weights: &Tensor) -> Result<Tensor> {
        if self.config.training && self.config.dropout_prob > 0.0 {
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

    /// Compute optimal block size for FlashAttention tiling.
    ///
    /// Starts from [`AttentionOptimizationHints::block_size`] and adapts it to
    /// the sequence lengths. The result is always a valid tile size: at least
    /// 1, never longer than the longest sequence.
    ///
    /// `_head_dim` is accepted for symmetry with the other block-size helpers;
    /// the heuristic currently depends only on the sequence lengths.
    fn compute_flash_block_size(&self, seq_q: usize, seq_k: usize, _head_dim: usize) -> usize {
        let base_size = self.optimization_hints.block_size.max(1);

        let adaptive_size = if seq_q > 2048 || seq_k > 2048 {
            // For very long sequences, use larger blocks to reduce overhead
            base_size * 2
        } else if seq_q < 128 && seq_k < 128 {
            // For short sequences, use smaller blocks for better granularity
            (base_size / 2).max(1)
        } else {
            base_size
        };

        adaptive_size.clamp(1, seq_q.max(seq_k).max(1))
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
        let attention =
            MultiHeadAttention::new(512, 8, 0.1, true).expect("operation failed in test");
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
        let attention =
            MultiHeadAttention::new(512, 8, 0.1, true).expect("operation failed in test");
        let input = Tensor::randn(&[2, 10, 512]).expect("Failed to create random tensor");
        let output = attention.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![2, 10, 512]);
    }

    #[test]
    fn test_memory_estimation() {
        let attention =
            MultiHeadAttention::new(512, 8, 0.1, true).expect("operation failed in test");
        let memory_usage = attention.estimate_memory_usage(2, 100);
        assert!(memory_usage > 0);
    }

    /// Deterministic pseudo-random tensor so assertions are reproducible.
    fn deterministic(shape: &[usize], seed: u32) -> Tensor {
        let count: usize = shape.iter().product();
        let mut state = seed.wrapping_mul(2_654_435_761).wrapping_add(12_345);
        let mut data = Vec::with_capacity(count);
        for _ in 0..count {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let unit = (state >> 8) as f32 / (1u32 << 24) as f32;
            data.push(unit * 2.0 - 1.0);
        }
        Tensor::from_vec(data, shape).expect("test tensor shape must be valid")
    }

    fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "output length mismatch");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
    }

    fn attention_with_weights(hidden_size: usize, num_heads: usize) -> MultiHeadAttention {
        let mut attention = MultiHeadAttention::new(hidden_size, num_heads, 0.0, true)
            .expect("construction must succeed");
        attention
            .set_query_weight(deterministic(&[hidden_size, hidden_size], 1))
            .expect("q weight");
        attention
            .set_key_weight(deterministic(&[hidden_size, hidden_size], 2))
            .expect("k weight");
        attention
            .set_value_weight(deterministic(&[hidden_size, hidden_size], 3))
            .expect("v weight");
        attention
            .set_out_proj_weight(deterministic(&[hidden_size, hidden_size], 4))
            .expect("o weight");
        attention.set_query_bias(deterministic(&[hidden_size], 5)).expect("q bias");
        attention.set_key_bias(deterministic(&[hidden_size], 6)).expect("k bias");
        attention.set_value_bias(deterministic(&[hidden_size], 7)).expect("v bias");
        attention.set_out_proj_bias(deterministic(&[hidden_size], 8)).expect("o bias");
        attention
    }

    #[test]
    fn flash_path_matches_standard_path() {
        // Regression test: the flash branch used to compute every block and
        // then return the all-zero tensor it had allocated up front.
        for causal in [false, true] {
            let hidden_size = 64;
            let attention = attention_with_weights(hidden_size, 8);
            let input = deterministic(&[2, 21, hidden_size], 9);

            let standard = attention
                .forward_self_attention(&input, None, causal)
                .expect("standard forward failed");

            let mut flash_attention = attention.clone();
            // A small tile forces several Q/K blocks for a 21-token sequence.
            flash_attention.set_optimization_hints(AttentionOptimizationHints {
                use_flash_attention: true,
                block_size: 8,
                ..Default::default()
            });
            let flash = flash_attention
                .forward_self_attention(&input, None, causal)
                .expect("flash forward failed");

            let difference = max_abs_difference(
                &standard.data().expect("standard data"),
                &flash.data().expect("flash data"),
            );
            assert!(
                difference < 1e-4,
                "flash and standard attention disagree by {difference} (causal={causal})"
            );
            assert!(
                flash.data().expect("flash data").iter().any(|x| x.abs() > 1e-6),
                "flash attention must not return zeros"
            );
        }
    }

    #[test]
    fn flash_path_honours_the_attention_mask() {
        let hidden_size = 32;
        let seq_len = 12;
        let attention = attention_with_weights(hidden_size, 4);
        let input = deterministic(&[1, seq_len, hidden_size], 10);

        let mut keep = vec![1.0f32; seq_len];
        keep[4] = 0.0;
        keep[9] = 0.0;
        let mask = Tensor::from_vec(keep, &[1, 1, 1, seq_len]).expect("mask shape");

        let standard = attention
            .forward_self_attention(&input, Some(&mask), false)
            .expect("standard forward failed");

        let mut flash_attention = attention.clone();
        flash_attention.set_optimization_hints(AttentionOptimizationHints {
            use_flash_attention: true,
            block_size: 10,
            ..Default::default()
        });
        let flash = flash_attention
            .forward_self_attention(&input, Some(&mask), false)
            .expect("flash forward failed");

        let difference = max_abs_difference(
            &standard.data().expect("standard data"),
            &flash.data().expect("flash data"),
        );
        assert!(
            difference < 1e-4,
            "masked flash attention disagrees with the standard path by {difference}"
        );

        let unmasked = attention
            .forward_self_attention(&input, None, false)
            .expect("unmasked forward failed");
        assert!(
            max_abs_difference(
                &standard.data().expect("standard data"),
                &unmasked.data().expect("unmasked data")
            ) > 1e-3,
            "the mask must actually change the result"
        );
    }

    #[test]
    fn causal_attention_ignores_future_tokens() {
        let hidden_size = 32;
        let seq_len = 10;
        let attention = attention_with_weights(hidden_size, 4);

        let base_input = deterministic(&[1, seq_len, hidden_size], 11);
        let mut perturbed = base_input.data().expect("input data");
        for position in 5..seq_len {
            for feature in 0..hidden_size {
                perturbed[position * hidden_size + feature] += 2.5;
            }
        }
        let perturbed_input = Tensor::from_vec(perturbed, &[1, seq_len, hidden_size])
            .expect("perturbed tensor shape");

        let base = attention
            .forward_self_attention(&base_input, None, true)
            .expect("base forward failed");
        let changed = attention
            .forward_self_attention(&perturbed_input, None, true)
            .expect("perturbed forward failed");

        let prefix = 5 * hidden_size;
        let base_data = base.data().expect("base data");
        let changed_data = changed.data().expect("changed data");
        assert!(
            max_abs_difference(&base_data[..prefix], &changed_data[..prefix]) < 1e-4,
            "causal attention leaked information from future tokens"
        );
    }

    #[test]
    fn dropout_is_inert_at_inference_and_active_in_training() {
        let hidden_size = 32;
        let mut attention =
            MultiHeadAttention::new(hidden_size, 4, 0.5, false).expect("construction must succeed");
        let input = deterministic(&[1, 8, hidden_size], 12);

        assert!(!attention.is_training());
        let first = attention
            .forward_self_attention(&input, None, false)
            .expect("first inference pass");
        let second = attention
            .forward_self_attention(&input, None, false)
            .expect("second inference pass");
        assert!(
            max_abs_difference(&first.data().expect("data"), &second.data().expect("data")) < 1e-6,
            "inference must be deterministic with dropout_prob > 0"
        );

        attention.set_training(true);
        let training =
            attention.forward_self_attention(&input, None, false).expect("training pass");
        assert!(
            max_abs_difference(
                &first.data().expect("data"),
                &training.data().expect("data")
            ) > 1e-4,
            "training mode must actually apply dropout"
        );
    }

    #[test]
    fn test_optimization_hints_update() {
        let mut attention =
            MultiHeadAttention::new(512, 8, 0.1, true).expect("operation failed in test");
        attention.update_optimization_hints(2, 2048, Some(1024));
        assert!(attention.optimization_hints.use_flash_attention);
    }
}
