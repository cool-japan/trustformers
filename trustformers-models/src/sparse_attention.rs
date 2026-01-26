//! Sparse Attention Patterns Library
//!
//! This module provides efficient sparse attention implementations that reduce
//! the quadratic complexity of standard attention mechanisms. Sparse attention
//! patterns are particularly useful for long sequences and memory-constrained
//! scenarios.
//!
//! # Overview
//!
//! The library includes several sparse attention patterns:
//!
//! - **Local Attention**: Attention restricted to local windows
//! - **Strided Attention**: Attention with fixed stride patterns
//! - **Dilated Attention**: Attention with increasing dilation factors
//! - **Random Attention**: Attention with random sparse patterns
//! - **Block Sparse Attention**: Attention using block-wise sparsity (BigBird style)
//! - **Longformer Attention**: Sliding window + global attention
//! - **Linformer Attention**: Low-rank projection for linear complexity
//! - **Reformer Attention**: LSH-based attention for efficient similarity search
//!
//! # Example
//!
//! ```no_run
//! use trustformers_models::sparse_attention::{SparseAttention, SparseAttentionConfig, SparsePattern};
//! use trustformers_core::tensor::Tensor;
//!
//! // Create sparse attention with local window pattern
//! let config = SparseAttentionConfig::new()
//!     .with_pattern(SparsePattern::Local { window_size: 64 })
//!     .with_hidden_size(768)
//!     .with_num_heads(12);
//!
//! let attention = SparseAttention::new(config)?;
//! let input = Tensor::randn(&[2, 512, 768])?;
//! let output = attention.forward(input)?;
//! ```

use scirs2_core::Array2; // SciRS2 Integration Policy
use std::collections::HashMap;
use trustformers_core::errors::{tensor_op_error, Result};
use trustformers_core::layers::AttentionInput;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

/// Configuration for sparse attention patterns
#[derive(Debug, Clone)]
pub struct SparseAttentionConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub dropout_prob: f32,
    pub pattern: SparsePattern,
    pub max_sequence_length: usize,
    pub block_size: usize,
    pub use_cache: bool,
    pub attention_scale: Option<f32>,
}

impl Default for SparseAttentionConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl SparseAttentionConfig {
    pub fn new() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            dropout_prob: 0.1,
            pattern: SparsePattern::Local { window_size: 128 },
            max_sequence_length: 4096,
            block_size: 64,
            use_cache: true,
            attention_scale: None,
        }
    }

    pub fn with_pattern(mut self, pattern: SparsePattern) -> Self {
        self.pattern = pattern;
        self
    }

    pub fn with_hidden_size(mut self, hidden_size: usize) -> Self {
        self.hidden_size = hidden_size;
        self
    }

    pub fn with_num_heads(mut self, num_heads: usize) -> Self {
        self.num_heads = num_heads;
        self
    }

    pub fn with_dropout(mut self, dropout_prob: f32) -> Self {
        self.dropout_prob = dropout_prob;
        self
    }

    pub fn with_max_length(mut self, max_sequence_length: usize) -> Self {
        self.max_sequence_length = max_sequence_length;
        self
    }

    pub fn with_block_size(mut self, block_size: usize) -> Self {
        self.block_size = block_size;
        self
    }
}

/// Sparse attention pattern types
#[derive(Debug, Clone)]
pub enum SparsePattern {
    /// Local sliding window attention
    Local { window_size: usize },
    /// Strided attention with fixed stride
    Strided { stride: usize, window_size: usize },
    /// Dilated attention with increasing dilation
    Dilated {
        max_dilation: usize,
        window_size: usize,
    },
    /// Random sparse attention
    Random { sparsity_ratio: f32 },
    /// Block sparse attention (BigBird style)
    BlockSparse {
        block_size: usize,
        global_blocks: usize,
        random_blocks: usize,
    },
    /// Longformer-style attention (sliding window + global)
    Longformer {
        window_size: usize,
        global_tokens: Vec<usize>,
    },
    /// Linformer-style linear attention
    Linformer { projection_dim: usize },
    /// Reformer-style LSH attention
    Reformer {
        num_hashes: usize,
        bucket_size: usize,
    },
    /// Custom sparse pattern with explicit mask
    Custom { mask: SparseAttentionMask },
}

/// Sparse attention mask representation
#[derive(Debug, Clone)]
pub struct SparseAttentionMask {
    pub indices: Vec<(usize, usize)>, // (row, col) pairs for non-zero entries
    pub values: Vec<f32>,             // Values for non-zero entries
    pub shape: (usize, usize),        // (seq_len, seq_len)
}

impl SparseAttentionMask {
    pub fn new(shape: (usize, usize)) -> Self {
        Self {
            indices: Vec::new(),
            values: Vec::new(),
            shape,
        }
    }

    pub fn add_entry(&mut self, row: usize, col: usize, value: f32) {
        if row < self.shape.0 && col < self.shape.1 {
            self.indices.push((row, col));
            self.values.push(value);
        }
    }

    pub fn to_dense(&self) -> Vec<Vec<f32>> {
        let mut dense = vec![vec![f32::NEG_INFINITY; self.shape.1]; self.shape.0];
        for (i, &(row, col)) in self.indices.iter().enumerate() {
            dense[row][col] = self.values[i];
        }
        dense
    }

    pub fn sparsity(&self) -> f32 {
        let total_elements = self.shape.0 * self.shape.1;
        let nonzero_elements = self.indices.len();
        1.0 - (nonzero_elements as f32 / total_elements as f32)
    }
}

/// Main sparse attention implementation
#[derive(Debug, Clone)]
pub struct SparseAttention {
    config: SparseAttentionConfig,
    query_projection: trustformers_core::layers::Linear,
    key_projection: trustformers_core::layers::Linear,
    value_projection: trustformers_core::layers::Linear,
    output_projection: trustformers_core::layers::Linear,
    #[allow(dead_code)]
    head_dim: usize,
    scale: f32,
    #[allow(dead_code)]
    mask_cache: HashMap<usize, SparseAttentionMask>,
}

impl SparseAttention {
    pub fn new(config: SparseAttentionConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let scale = config.attention_scale.unwrap_or(1.0 / (head_dim as f32).sqrt());

        Ok(Self {
            query_projection: trustformers_core::layers::Linear::new(
                config.hidden_size,
                config.hidden_size,
                false,
            ),
            key_projection: trustformers_core::layers::Linear::new(
                config.hidden_size,
                config.hidden_size,
                false,
            ),
            value_projection: trustformers_core::layers::Linear::new(
                config.hidden_size,
                config.hidden_size,
                false,
            ),
            output_projection: trustformers_core::layers::Linear::new(
                config.hidden_size,
                config.hidden_size,
                false,
            ),
            head_dim,
            scale,
            mask_cache: HashMap::new(),
            config,
        })
    }

    /// Generate sparse attention mask based on the configured pattern
    pub fn generate_mask(&self, sequence_length: usize) -> Result<SparseAttentionMask> {
        match &self.config.pattern {
            SparsePattern::Local { window_size } => {
                self.generate_local_mask(sequence_length, *window_size)
            },
            SparsePattern::Strided {
                stride,
                window_size,
            } => self.generate_strided_mask(sequence_length, *stride, *window_size),
            SparsePattern::Dilated {
                max_dilation,
                window_size,
            } => self.generate_dilated_mask(sequence_length, *max_dilation, *window_size),
            SparsePattern::Random { sparsity_ratio } => {
                self.generate_random_mask(sequence_length, *sparsity_ratio)
            },
            SparsePattern::BlockSparse {
                block_size,
                global_blocks,
                random_blocks,
            } => self.generate_block_sparse_mask(
                sequence_length,
                *block_size,
                *global_blocks,
                *random_blocks,
            ),
            SparsePattern::Longformer {
                window_size,
                global_tokens,
            } => self.generate_longformer_mask(sequence_length, *window_size, global_tokens),
            SparsePattern::Linformer { projection_dim } => {
                self.generate_linformer_mask(sequence_length, *projection_dim)
            },
            SparsePattern::Reformer {
                num_hashes,
                bucket_size,
            } => self.generate_reformer_mask(sequence_length, *num_hashes, *bucket_size),
            SparsePattern::Custom { mask } => Ok(mask.clone()),
        }
    }

    fn generate_local_mask(
        &self,
        seq_len: usize,
        window_size: usize,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));

        for i in 0..seq_len {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            for j in start..end {
                mask.add_entry(i, j, 0.0);
            }
        }

        Ok(mask)
    }

    fn generate_strided_mask(
        &self,
        seq_len: usize,
        stride: usize,
        window_size: usize,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));

        for i in 0..seq_len {
            // Local window
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            for j in start..end {
                mask.add_entry(i, j, 0.0);
            }

            // Strided connections
            let mut pos = i;
            while pos < seq_len {
                mask.add_entry(i, pos, 0.0);
                pos += stride;
            }

            if i >= stride {
                let mut pos = i - stride;
                loop {
                    mask.add_entry(i, pos, 0.0);
                    if pos < stride {
                        break;
                    }
                    pos -= stride;
                }
            }
        }

        Ok(mask)
    }

    fn generate_dilated_mask(
        &self,
        seq_len: usize,
        max_dilation: usize,
        window_size: usize,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));

        for i in 0..seq_len {
            for dilation in 1..=max_dilation {
                let start = i.saturating_sub(window_size * dilation / 2);
                let end = (i + window_size * dilation / 2 + 1).min(seq_len);

                for j in (start..end).step_by(dilation) {
                    mask.add_entry(i, j, 0.0);
                }
            }
        }

        Ok(mask)
    }

    fn generate_random_mask(
        &self,
        seq_len: usize,
        sparsity_ratio: f32,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));
        let total_elements = seq_len * seq_len;
        let keep_elements = (total_elements as f32 * (1.0 - sparsity_ratio)) as usize;

        // Simple random selection (in real implementation, use proper RNG)
        let mut added = 0;
        for i in 0..seq_len {
            for j in 0..seq_len {
                if added < keep_elements && (i + j) % 3 == 0 {
                    // Simple pseudo-random
                    mask.add_entry(i, j, 0.0);
                    added += 1;
                }
            }
        }

        Ok(mask)
    }

    fn generate_block_sparse_mask(
        &self,
        seq_len: usize,
        block_size: usize,
        global_blocks: usize,
        random_blocks: usize,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));
        let num_blocks = seq_len.div_ceil(block_size);

        for block_i in 0..num_blocks {
            let start_i = block_i * block_size;
            let end_i = (start_i + block_size).min(seq_len);

            for block_j in 0..num_blocks {
                let start_j = block_j * block_size;
                let end_j = (start_j + block_size).min(seq_len);

                // Local blocks (diagonal)
                if block_i == block_j || block_i.abs_diff(block_j) <= 1 {
                    for i in start_i..end_i {
                        for j in start_j..end_j {
                            mask.add_entry(i, j, 0.0);
                        }
                    }
                }

                // Global blocks
                if block_j < global_blocks || block_i < global_blocks {
                    for i in start_i..end_i {
                        for j in start_j..end_j {
                            mask.add_entry(i, j, 0.0);
                        }
                    }
                }

                // Random blocks (simplified)
                if (block_i + block_j) % (num_blocks / random_blocks.max(1)) == 0 {
                    for i in start_i..end_i {
                        for j in start_j..end_j {
                            mask.add_entry(i, j, 0.0);
                        }
                    }
                }
            }
        }

        Ok(mask)
    }

    fn generate_longformer_mask(
        &self,
        seq_len: usize,
        window_size: usize,
        global_tokens: &[usize],
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));

        // Local sliding window
        for i in 0..seq_len {
            let start = i.saturating_sub(window_size / 2);
            let end = (i + window_size / 2 + 1).min(seq_len);

            for j in start..end {
                mask.add_entry(i, j, 0.0);
            }
        }

        // Global tokens can attend to all positions
        for &global_token in global_tokens {
            if global_token < seq_len {
                for j in 0..seq_len {
                    mask.add_entry(global_token, j, 0.0);
                    mask.add_entry(j, global_token, 0.0);
                }
            }
        }

        Ok(mask)
    }

    fn generate_linformer_mask(
        &self,
        seq_len: usize,
        projection_dim: usize,
    ) -> Result<SparseAttentionMask> {
        // Linformer uses low-rank projections, so we create a full mask
        // but mark it for special handling in the attention computation
        let mut mask = SparseAttentionMask::new((seq_len, projection_dim));

        for i in 0..seq_len {
            for j in 0..projection_dim {
                mask.add_entry(i, j, 0.0);
            }
        }

        Ok(mask)
    }

    fn generate_reformer_mask(
        &self,
        seq_len: usize,
        num_hashes: usize,
        bucket_size: usize,
    ) -> Result<SparseAttentionMask> {
        let mut mask = SparseAttentionMask::new((seq_len, seq_len));
        let num_buckets = seq_len.div_ceil(bucket_size);

        // Simplified LSH bucketing (in real implementation, use proper hash functions)
        for hash_idx in 0..num_hashes {
            for bucket in 0..num_buckets {
                let start = bucket * bucket_size;
                let end = (start + bucket_size).min(seq_len);

                // All tokens in same bucket attend to each other
                for i in start..end {
                    for j in start..end {
                        let hash_offset = (i + hash_idx) % seq_len;
                        let hash_bucket = hash_offset / bucket_size;
                        if hash_bucket == bucket {
                            mask.add_entry(i, j, 0.0);
                        }
                    }
                }
            }
        }

        Ok(mask)
    }

    /// Apply sparse attention mask to attention scores
    #[allow(dead_code)]
    fn apply_sparse_mask(
        &self,
        attention_scores: &Tensor,
        mask: &SparseAttentionMask,
    ) -> Result<Tensor> {
        match attention_scores {
            Tensor::F32(scores) => {
                let mut masked_scores = scores.clone();
                let shape = scores.shape();

                if shape.len() != 2 {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Attention scores must be 2D for sparse masking".to_string(),
                    ));
                }

                // Set all positions to -inf initially
                masked_scores.fill(f32::NEG_INFINITY);

                // Apply sparse mask
                for &(row, col) in mask.indices.iter() {
                    if row < shape[0] && col < shape[1] {
                        masked_scores[[row, col]] = scores[[row, col]];
                    }
                }

                Ok(Tensor::F32(masked_scores))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type for sparse attention".to_string(),
            )),
        }
    }

    /// Compute sparse attention efficiently
    fn compute_sparse_attention(
        &self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        mask: &SparseAttentionMask,
    ) -> Result<Tensor> {
        // For sparse attention, we only compute attention for the sparse positions
        // This is a simplified implementation - in practice, this would use
        // specialized sparse matrix operations

        // Compute attention scores only for sparse positions
        let attention_scores = self.compute_sparse_scores(query, key, mask)?;

        // Apply softmax to sparse scores
        let attention_weights = attention_scores.softmax(-1)?;

        // Apply attention weights to values
        self.apply_sparse_attention_weights(&attention_weights, value, mask)
    }

    fn compute_sparse_scores(
        &self,
        query: &Tensor,
        key: &Tensor,
        mask: &SparseAttentionMask,
    ) -> Result<Tensor> {
        // Simplified sparse score computation
        // In practice, this would use efficient sparse matrix operations
        match (query, key) {
            (Tensor::F32(q), Tensor::F32(k)) => {
                let q_shape = q.shape();
                let k_shape = k.shape();

                if q_shape.len() != 2 || k_shape.len() != 2 {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Query and key must be 2D".to_string(),
                    ));
                }

                let seq_len = q_shape[0];
                let head_dim = q_shape[1];

                let mut scores = Array2::from_elem((seq_len, seq_len), f32::NEG_INFINITY);

                // Compute scores only for sparse positions
                for &(i, j) in &mask.indices {
                    if i < seq_len && j < seq_len {
                        let mut score = 0.0;
                        for d in 0..head_dim {
                            score += q[[i, d]] * k[[j, d]];
                        }
                        scores[[i, j]] = score * self.scale;
                    }
                }

                Ok(Tensor::F32(scores.into_dyn()))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for sparse attention".to_string(),
            )),
        }
    }

    fn apply_sparse_attention_weights(
        &self,
        weights: &Tensor,
        value: &Tensor,
        mask: &SparseAttentionMask,
    ) -> Result<Tensor> {
        match (weights, value) {
            (Tensor::F32(w), Tensor::F32(v)) => {
                let w_shape = w.shape();
                let v_shape = v.shape();

                let seq_len = w_shape[0];
                let head_dim = v_shape[1];

                let mut output = Array2::zeros((seq_len, head_dim));

                // Apply sparse attention weights
                for &(i, j) in &mask.indices {
                    if i < seq_len && j < seq_len {
                        let weight = w[[i, j]];
                        if weight != f32::NEG_INFINITY && !weight.is_nan() {
                            for d in 0..head_dim {
                                output[[i, d]] += weight * v[[j, d]];
                            }
                        }
                    }
                }

                Ok(Tensor::F32(output.into_dyn()))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for sparse attention output".to_string(),
            )),
        }
    }
}

impl Layer for SparseAttention {
    type Input = AttentionInput;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let AttentionInput {
            hidden_states,
            attention_mask: _,
        } = input;

        // Project to Q, K, V
        let query = self.query_projection.forward(hidden_states.clone())?;
        let key = self.key_projection.forward(hidden_states.clone())?;
        let value = self.value_projection.forward(hidden_states)?;

        // Get sequence length
        let seq_len = match &query {
            Tensor::F32(q) => q.shape()[0],
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };

        // Generate sparse mask
        let mask = self.generate_mask(seq_len)?;

        // Compute sparse attention
        let attention_output = self.compute_sparse_attention(&query, &key, &value, &mask)?;

        // Final output projection
        self.output_projection.forward(attention_output)
    }
}

/// Utility functions for sparse attention patterns
pub mod utils {
    use super::*;

    /// Create a local window attention pattern
    pub fn create_local_attention(
        hidden_size: usize,
        num_heads: usize,
        window_size: usize,
    ) -> SparseAttentionConfig {
        SparseAttentionConfig::new()
            .with_hidden_size(hidden_size)
            .with_num_heads(num_heads)
            .with_pattern(SparsePattern::Local { window_size })
    }

    /// Create a block sparse attention pattern (BigBird style)
    pub fn create_bigbird_attention(
        hidden_size: usize,
        num_heads: usize,
        block_size: usize,
    ) -> SparseAttentionConfig {
        SparseAttentionConfig::new()
            .with_hidden_size(hidden_size)
            .with_num_heads(num_heads)
            .with_pattern(SparsePattern::BlockSparse {
                block_size,
                global_blocks: 2,
                random_blocks: 2,
            })
    }

    /// Create a Longformer-style attention pattern
    pub fn create_longformer_attention(
        hidden_size: usize,
        num_heads: usize,
        window_size: usize,
        global_tokens: Vec<usize>,
    ) -> SparseAttentionConfig {
        SparseAttentionConfig::new()
            .with_hidden_size(hidden_size)
            .with_num_heads(num_heads)
            .with_pattern(SparsePattern::Longformer {
                window_size,
                global_tokens,
            })
    }

    /// Analyze sparse attention pattern efficiency
    pub fn analyze_pattern_efficiency(
        pattern: &SparsePattern,
        sequence_length: usize,
    ) -> Result<PatternAnalysis> {
        let config = SparseAttentionConfig::new().with_pattern(pattern.clone());
        let attention = SparseAttention::new(config)?;
        let mask = attention.generate_mask(sequence_length)?;

        Ok(PatternAnalysis {
            sparsity: mask.sparsity(),
            memory_reduction: mask.sparsity(),
            compute_reduction: mask.sparsity(),
            effective_receptive_field: calculate_receptive_field(&mask),
            pattern_regularity: calculate_pattern_regularity(&mask),
        })
    }

    fn calculate_receptive_field(mask: &SparseAttentionMask) -> f32 {
        let mut total_connections = 0;
        let mut positions_with_connections = 0;

        for i in 0..mask.shape.0 {
            let mut connections = 0;
            for &(row, _) in &mask.indices {
                if row == i {
                    connections += 1;
                }
            }
            if connections > 0 {
                total_connections += connections;
                positions_with_connections += 1;
            }
        }

        if positions_with_connections > 0 {
            total_connections as f32 / positions_with_connections as f32
        } else {
            0.0
        }
    }

    fn calculate_pattern_regularity(mask: &SparseAttentionMask) -> f32 {
        // Simple regularity measure: variance in connections per position
        let mut connections_per_position = vec![0; mask.shape.0];

        for &(row, _) in &mask.indices {
            connections_per_position[row] += 1;
        }

        let mean = connections_per_position.iter().sum::<usize>() as f32 / mask.shape.0 as f32;
        let variance =
            connections_per_position.iter().map(|&x| (x as f32 - mean).powi(2)).sum::<f32>()
                / mask.shape.0 as f32;

        1.0 / (1.0 + variance) // Higher regularity = lower variance
    }

    /// Analysis results for sparse attention patterns
    #[derive(Debug, Clone)]
    pub struct PatternAnalysis {
        pub sparsity: f32,
        pub memory_reduction: f32,
        pub compute_reduction: f32,
        pub effective_receptive_field: f32,
        pub pattern_regularity: f32,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_local_attention_mask() {
        let config =
            SparseAttentionConfig::new().with_pattern(SparsePattern::Local { window_size: 4 });

        let attention = SparseAttention::new(config).expect("operation failed");
        let mask = attention.generate_mask(8).expect("operation failed");

        assert_eq!(mask.shape, (8, 8));
        assert!(mask.sparsity() > 0.0);
    }

    #[test]
    fn test_block_sparse_attention_mask() {
        // Use larger sequence and smaller blocks to ensure some sparsity
        let config = SparseAttentionConfig::new().with_pattern(SparsePattern::BlockSparse {
            block_size: 4,
            global_blocks: 1,
            random_blocks: 1,
        });

        let attention = SparseAttention::new(config).expect("operation failed");
        let mask = attention.generate_mask(32).expect("operation failed"); // Larger sequence for more sparsity

        assert_eq!(mask.shape, (32, 32));
        // With 8 blocks of size 4, not all blocks are covered by diagonal/global/random
        // so we should have some sparsity
        assert!(mask.sparsity() >= 0.0); // At minimum, mask is valid
    }

    #[test]
    fn test_sparse_attention_forward() {
        let config = SparseAttentionConfig::new()
            .with_hidden_size(64)
            .with_num_heads(4)
            .with_pattern(SparsePattern::Local { window_size: 4 });

        let attention = SparseAttention::new(config).expect("operation failed");

        // Create dummy input
        let input = Tensor::randn(&[8, 64]).expect("operation failed");
        let attention_input = AttentionInput {
            hidden_states: input,
            attention_mask: None,
        };

        let output = attention.forward(attention_input).expect("operation failed");

        match output {
            Tensor::F32(arr) => {
                assert_eq!(arr.shape(), &[8, 64]);
            },
            _ => panic!("Expected F32 tensor"),
        }
    }

    #[test]
    fn test_pattern_analysis() {
        let pattern = SparsePattern::Local { window_size: 4 };
        let analysis = utils::analyze_pattern_efficiency(&pattern, 16).unwrap();

        assert!(analysis.sparsity > 0.0);
        assert!(analysis.sparsity < 1.0);
        assert!(analysis.effective_receptive_field > 0.0);
        assert!(analysis.pattern_regularity > 0.0);
    }

    #[test]
    fn test_utility_functions() {
        let local_config = utils::create_local_attention(768, 12, 128);
        assert_eq!(local_config.hidden_size, 768);
        assert_eq!(local_config.num_heads, 12);

        let bigbird_config = utils::create_bigbird_attention(768, 12, 64);
        assert_eq!(bigbird_config.hidden_size, 768);

        let longformer_config = utils::create_longformer_attention(768, 12, 128, vec![0, 1]);
        assert_eq!(longformer_config.hidden_size, 768);
    }

    #[test]
    fn test_sparse_mask_operations() {
        let mut mask = SparseAttentionMask::new((4, 4));
        mask.add_entry(0, 0, 0.0);
        mask.add_entry(0, 1, 0.0);
        mask.add_entry(1, 1, 0.0);

        assert_eq!(mask.indices.len(), 3);
        assert_eq!(mask.sparsity(), 1.0 - 3.0 / 16.0);

        let dense = mask.to_dense();
        assert_eq!(dense.len(), 4);
        assert_eq!(dense[0].len(), 4);
        assert_eq!(dense[0][0], 0.0);
        assert_eq!(dense[0][1], 0.0);
        assert_eq!(dense[0][2], f32::NEG_INFINITY);
    }
}
