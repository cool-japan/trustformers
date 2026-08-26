//! FlashAttention layer.
//!
//! This module provides memory-efficient attention computation using the
//! FlashAttention algorithm. The full `seq_q x seq_k` attention matrix is never
//! materialised: memory scales with the tile size instead of `O(N^2)`.
//!
//! The numerical core lives in `super::flash_kernel` and is shared with
//! [`super::multi_head::MultiHeadAttention`]'s memory-efficient path.

use super::common::{AttentionConfig, AttentionProjections, AttentionUtils};
use super::flash_kernel::{flash_attention, FlashParams};
use crate::errors::Result;
use crate::tensor::Tensor;
use crate::traits::Layer;

/// FlashAttention: Memory-efficient attention computation
///
/// This implements the FlashAttention algorithm which reduces memory complexity
/// from O(N²) to O(N) by computing attention in blocks and not materializing
/// the full attention matrix.
///
/// Reference: FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness
/// <https://arxiv.org/abs/2205.14135>
///
/// # FlashAttention-1 vs FlashAttention-2
///
/// Both versions run the *same* CPU kernel (`super::flash_kernel`), which
/// already uses the FlashAttention-2 style deferred normalisation (the output
/// accumulator is divided by the running row sum once, after the last key
/// block, rather than after every block). The only difference between the two
/// settings on CPU is work partitioning: FlashAttention-1 uses the configured
/// fixed [`FlashAttention::block_size`], while FlashAttention-2 derives an
/// adaptive tile size from the sequence lengths. No further algorithmic
/// difference is claimed or implemented.
#[derive(Debug, Clone)]
pub struct FlashAttention {
    /// Attention configuration
    config: AttentionConfig,
    /// Query, key, value, and output projections
    projections: AttentionProjections,
    /// Block size for tiled computation
    block_size: usize,
    /// Whether to use causal masking
    causal: bool,
    /// Whether to use FlashAttention-2 work partitioning
    use_flash_attention_2: bool,
}

impl FlashAttention {
    /// Create a new FlashAttention layer
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        block_size: Option<usize>,
        causal: bool,
    ) -> Result<Self> {
        let config = AttentionConfig::new(hidden_size, num_heads, dropout_prob, bias)?;
        let projections = AttentionProjections::new(&config);

        let block_size = block_size.unwrap_or(AttentionUtils::compute_block_size(
            1024,
            config.head_dim,
            None,
        ));

        Ok(Self {
            config,
            projections,
            block_size,
            causal,
            use_flash_attention_2: true,
        })
    }

    /// Create a new FlashAttention layer with version control
    pub fn new_with_version(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
        block_size: Option<usize>,
        causal: bool,
        use_flash_attention_2: bool,
    ) -> Result<Self> {
        let mut flash_attention = Self::new(
            hidden_size,
            num_heads,
            dropout_prob,
            bias,
            block_size,
            causal,
        )?;
        flash_attention.use_flash_attention_2 = use_flash_attention_2;
        Ok(flash_attention)
    }

    /// Get the attention configuration
    pub fn config(&self) -> &AttentionConfig {
        &self.config
    }

    /// The four projection layers (`query`, `key`, `value`, `out_proj`).
    ///
    /// Exposed so a model can publish this attention block's parameters through
    /// [`Model::named_tensors`](crate::traits::Model::named_tensors) without the
    /// attention layer having to know any checkpoint naming convention.
    pub fn projections(&self) -> &AttentionProjections {
        &self.projections
    }

    /// Mutable counterpart of [`FlashAttention::projections`].
    ///
    /// The four projections are separate fields of [`AttentionProjections`], so a
    /// caller can borrow each of them mutably at once through this single handle —
    /// which is what building a `Vec<(String, &mut Tensor)>` for
    /// [`Model::named_tensors_mut`](crate::traits::Model::named_tensors_mut)
    /// requires.
    pub fn projections_mut(&mut self) -> &mut AttentionProjections {
        &mut self.projections
    }

    /// Get the block size used for tiled computation
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Set the block size for tiled computation
    pub fn set_block_size(&mut self, block_size: usize) {
        self.block_size = block_size;
    }

    /// Check if using FlashAttention-2 work partitioning
    pub fn is_using_flash_attention_2(&self) -> bool {
        self.use_flash_attention_2
    }

    /// Enable or disable FlashAttention-2 work partitioning
    pub fn set_flash_attention_2(&mut self, enabled: bool) {
        self.use_flash_attention_2 = enabled;
    }

    /// Whether the layer is in training mode (which enables attention dropout).
    pub fn is_training(&self) -> bool {
        self.config.training
    }

    /// Switch between training and inference mode.
    ///
    /// Attention dropout is applied to the tile probabilities only in training
    /// mode, so inference stays deterministic even for a configuration with a
    /// non-zero `dropout_prob`.
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

    /// Compute self-attention using FlashAttention algorithm
    pub fn forward_self_attention(
        &self,
        input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.forward_attention(input, input, input, attention_mask)
    }

    /// Compute attention with separate query, key, value inputs
    pub fn forward_attention(
        &self,
        query_input: &Tensor,
        key_input: &Tensor,
        value_input: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Apply input projections
        // `forward_ref` borrows: self-attention passes the *same* hidden-state
        // tensor three times, and the owning `forward` would deep-copy
        // `[batch, seq, hidden]` once per projection.
        let query = self.projections.query.forward_ref(query_input)?;
        let key = self.projections.key.forward_ref(key_input)?;
        let value = self.projections.value.forward_ref(value_input)?;

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

        // Compute FlashAttention
        let attention_output = if self.use_flash_attention_2 {
            self.compute_flash_attention_2(&q, &k, &v, attention_mask)?
        } else {
            self.compute_flash_attention_1(&q, &k, &v, attention_mask)?
        };

        // Combine heads
        let combined = AttentionUtils::combine_heads(
            &attention_output,
            self.config.num_heads,
            self.config.head_dim,
        )?;

        // Apply output projection
        self.projections.out_proj.forward(combined)
    }

    /// FlashAttention-1: fixed-size tiling.
    fn compute_flash_attention_1(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let params = FlashParams::new(self.config.head_dim, self.causal, self.block_size)
            .with_dropout(self.active_dropout())?;
        flash_attention(q, k, v, attention_mask, &params)
    }

    /// FlashAttention-2: adaptive tiling derived from the sequence lengths.
    fn compute_flash_attention_2(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let q_shape = q.shape();
        let seq_q = q_shape[q_shape.len() - 2];
        let seq_k = {
            let k_shape = k.shape();
            k_shape[k_shape.len() - 2]
        };
        let block_size = self.compute_adaptive_block_size(seq_q, seq_k);
        let params = FlashParams::new(self.config.head_dim, self.causal, block_size)
            .with_dropout(self.active_dropout())?;
        flash_attention(q, k, v, attention_mask, &params)
    }

    /// Attention dropout probability in effect for the current mode.
    fn active_dropout(&self) -> Option<f32> {
        if self.config.training {
            Some(self.config.dropout_prob)
        } else {
            None
        }
    }

    /// Estimate memory usage for FlashAttention
    pub fn estimate_memory_usage(&self, batch_size: usize, seq_len: usize) -> usize {
        // FlashAttention memory usage is O(N) instead of O(N²)
        let projection_memory = batch_size * seq_len * self.config.hidden_size * 4; // Q, K, V, O
        let block_memory = batch_size * self.config.num_heads * self.block_size * self.block_size;
        let intermediate_memory =
            batch_size * self.config.num_heads * seq_len * self.config.head_dim * 3;

        (projection_memory + block_memory + intermediate_memory) * 4 // 4 bytes per f32
    }

    /// Compute optimal block size for current sequence length
    pub fn compute_optimal_block_size(
        &self,
        seq_len: usize,
        available_memory_mb: Option<usize>,
    ) -> usize {
        AttentionUtils::compute_block_size(seq_len, self.config.head_dim, available_memory_mb)
    }

    /// Update block size based on sequence length and available memory
    pub fn update_block_size(&mut self, seq_len: usize, available_memory_mb: Option<usize>) {
        self.block_size = self.compute_optimal_block_size(seq_len, available_memory_mb);
    }

    /// Compute the adaptive tile size used by the FlashAttention-2 partitioning.
    ///
    /// The result is always at least 1 and never larger than the longest of the
    /// two sequences, so it is a valid tile size for any input.
    fn compute_adaptive_block_size(&self, seq_q: usize, seq_k: usize) -> usize {
        let base_block_size = self.block_size.max(1);

        let adaptive_size = if seq_q > 4096 || seq_k > 4096 {
            // For very long sequences, use larger blocks to reduce overhead
            (base_block_size * 2).min(1024)
        } else if seq_q < 128 && seq_k < 128 {
            // For short sequences, use smaller blocks for better granularity
            (base_block_size / 2).max(1)
        } else {
            base_block_size
        };

        adaptive_size.clamp(1, seq_q.max(seq_k).max(1))
    }
}

impl Layer for FlashAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Default to self-attention
        self.forward_self_attention(&input, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::attention::multi_head::MultiHeadAttention;
    use crate::tensor::Tensor;

    fn max_abs_difference(a: &[f32], b: &[f32]) -> f32 {
        assert_eq!(a.len(), b.len(), "output length mismatch");
        a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()))
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

    /// Query/key/value/output projection weights and biases.
    fn projection_parameters(hidden_size: usize, seed: u32) -> ([Tensor; 4], [Tensor; 4]) {
        let weights = [
            deterministic(&[hidden_size, hidden_size], seed),
            deterministic(&[hidden_size, hidden_size], seed + 1),
            deterministic(&[hidden_size, hidden_size], seed + 2),
            deterministic(&[hidden_size, hidden_size], seed + 3),
        ];
        let biases = [
            deterministic(&[hidden_size], seed + 4),
            deterministic(&[hidden_size], seed + 5),
            deterministic(&[hidden_size], seed + 6),
            deterministic(&[hidden_size], seed + 7),
        ];
        (weights, biases)
    }

    fn apply_parameters(
        attention: &mut FlashAttention,
        weights: &[Tensor; 4],
        biases: &[Tensor; 4],
    ) {
        attention.set_query_weight(weights[0].clone()).expect("q weight");
        attention.set_key_weight(weights[1].clone()).expect("k weight");
        attention.set_value_weight(weights[2].clone()).expect("v weight");
        attention.set_out_proj_weight(weights[3].clone()).expect("o weight");
        attention.set_query_bias(biases[0].clone()).expect("q bias");
        attention.set_key_bias(biases[1].clone()).expect("k bias");
        attention.set_value_bias(biases[2].clone()).expect("v bias");
        attention.set_out_proj_bias(biases[3].clone()).expect("o bias");
    }

    #[test]
    fn test_flash_attention_creation() {
        let attention =
            FlashAttention::new(512, 8, 0.1, true, None, false).expect("operation failed in test");
        assert_eq!(attention.config.hidden_size, 512);
        assert_eq!(attention.config.num_heads, 8);
        assert_eq!(attention.config.head_dim, 64);
    }

    #[test]
    fn test_flash_attention_with_custom_block_size() {
        let attention = FlashAttention::new(512, 8, 0.1, true, Some(128), false)
            .expect("operation failed in test");
        assert_eq!(attention.block_size(), 128);
    }

    #[test]
    fn test_flash_attention_2_version() {
        let attention = FlashAttention::new_with_version(512, 8, 0.1, true, None, false, true)
            .expect("operation failed in test");
        assert!(attention.is_using_flash_attention_2());
    }

    #[test]
    fn test_flash_attention_forward() {
        let attention =
            FlashAttention::new(512, 8, 0.1, true, None, false).expect("operation failed in test");
        let input = deterministic(&[2, 10, 512], 1);
        let output = attention.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![2, 10, 512]);
    }

    #[test]
    fn test_memory_estimation() {
        let attention =
            FlashAttention::new(512, 8, 0.1, true, None, false).expect("operation failed in test");
        let memory_usage = attention.estimate_memory_usage(2, 1000);
        assert!(memory_usage > 0);
    }

    #[test]
    fn test_optimal_block_size_computation() {
        let attention =
            FlashAttention::new(512, 8, 0.1, true, None, false).expect("operation failed in test");
        let block_size = attention.compute_optimal_block_size(2048, Some(1024));
        assert!(block_size > 0);
        assert!(block_size <= 512);
    }

    #[test]
    fn test_causal_attention() {
        let attention =
            FlashAttention::new(512, 8, 0.1, true, None, true).expect("operation failed in test");
        let input = deterministic(&[2, 10, 512], 2);
        let output = attention.forward(input).expect("Forward pass failed");
        assert_eq!(output.shape(), vec![2, 10, 512]);
    }

    #[test]
    fn forward_output_is_not_constant() {
        // Regression test: the previous implementation never wrote the block
        // results into the output tensor, so `forward` returned the broadcast
        // output-projection bias for every input.
        let attention =
            FlashAttention::new(64, 4, 0.0, true, Some(8), false).expect("construction failed");
        let first =
            attention.forward(deterministic(&[1, 12, 64], 3)).expect("first forward failed");
        let second = attention
            .forward(deterministic(&[1, 12, 64], 4))
            .expect("second forward failed");

        let first_data = first.data().expect("data");
        let second_data = second.data().expect("data");
        assert!(
            max_abs_difference(&first_data, &second_data) > 1e-3,
            "FlashAttention output must depend on its input"
        );

        // ... and it must not be constant across positions either.
        let row0 = &first_data[0..64];
        let row1 = &first_data[64..128];
        assert!(
            max_abs_difference(row0, row1) > 1e-4,
            "FlashAttention output must vary across sequence positions"
        );
    }

    #[test]
    fn flash_attention_matches_standard_attention() {
        // FlashAttention and MultiHeadAttention share the same projection
        // shapes; copying the weights across lets us compare the attention
        // maths directly.
        let hidden_size = 64;
        let num_heads = 4;
        let mut flash = FlashAttention::new(hidden_size, num_heads, 0.0, true, Some(5), false)
            .expect("flash construction failed");
        let mut standard = MultiHeadAttention::new(hidden_size, num_heads, 0.0, true)
            .expect("standard construction failed");

        let (weights, biases) = projection_parameters(hidden_size, 5);

        apply_parameters(&mut flash, &weights, &biases);
        standard.set_query_weight(weights[0].clone()).expect("q weight");
        standard.set_key_weight(weights[1].clone()).expect("k weight");
        standard.set_value_weight(weights[2].clone()).expect("v weight");
        standard.set_out_proj_weight(weights[3].clone()).expect("o weight");
        standard.set_query_bias(biases[0].clone()).expect("q bias");
        standard.set_key_bias(biases[1].clone()).expect("k bias");
        standard.set_value_bias(biases[2].clone()).expect("v bias");
        standard.set_out_proj_bias(biases[3].clone()).expect("o bias");

        let input = deterministic(&[2, 13, hidden_size], 13);
        let flash_output =
            flash.forward_self_attention(&input, None).expect("flash forward failed");
        let standard_output = standard
            .forward_self_attention(&input, None, false)
            .expect("standard forward failed");

        let difference = max_abs_difference(
            &flash_output.data().expect("flash data"),
            &standard_output.data().expect("standard data"),
        );
        assert!(
            difference < 1e-4,
            "FlashAttention disagrees with standard attention by {difference}"
        );
    }

    #[test]
    fn flash_attention_1_and_2_agree() {
        for causal in [false, true] {
            let hidden_size = 32;
            let num_heads = 4;
            let mut version_1 = FlashAttention::new_with_version(
                hidden_size,
                num_heads,
                0.0,
                true,
                Some(4),
                causal,
                false,
            )
            .expect("v1 construction failed");
            let mut version_2 = FlashAttention::new_with_version(
                hidden_size,
                num_heads,
                0.0,
                true,
                Some(4),
                causal,
                true,
            )
            .expect("v2 construction failed");

            let (weights, biases) = projection_parameters(hidden_size, 14);
            apply_parameters(&mut version_1, &weights, &biases);
            apply_parameters(&mut version_2, &weights, &biases);

            let input = deterministic(&[1, 17, hidden_size], 22);
            let output_1 =
                version_1.forward_self_attention(&input, None).expect("v1 forward failed");
            let output_2 =
                version_2.forward_self_attention(&input, None).expect("v2 forward failed");

            let difference = max_abs_difference(
                &output_1.data().expect("v1 data"),
                &output_2.data().expect("v2 data"),
            );
            assert!(
                difference < 1e-4,
                "FlashAttention-1 and -2 disagree by {difference} (causal={causal})"
            );
        }
    }

    #[test]
    fn causal_flash_attention_ignores_future_tokens() {
        // Regression test for the block causal mask that used to be a no-op.
        let hidden_size = 32;
        let attention = FlashAttention::new(hidden_size, 4, 0.0, false, Some(3), true)
            .expect("construction failed");

        let base_input = deterministic(&[1, 10, hidden_size], 23);
        let mut perturbed_data = base_input.data().expect("input data");
        for position in 5..10 {
            for feature in 0..hidden_size {
                perturbed_data[position * hidden_size + feature] += 2.0;
            }
        }
        let perturbed_input = Tensor::from_vec(perturbed_data, &[1, 10, hidden_size])
            .expect("perturbed tensor shape");

        let base = attention
            .forward_self_attention(&base_input, None)
            .expect("base forward failed");
        let perturbed = attention
            .forward_self_attention(&perturbed_input, None)
            .expect("perturbed forward failed");

        let base_data = base.data().expect("base data");
        let perturbed_data = perturbed.data().expect("perturbed data");
        let prefix = 5 * hidden_size;
        assert!(
            max_abs_difference(&base_data[..prefix], &perturbed_data[..prefix]) < 1e-4,
            "causal FlashAttention leaked information from future tokens"
        );
        assert!(
            max_abs_difference(&base_data[prefix..], &perturbed_data[prefix..]) > 1e-3,
            "perturbation had no effect at all"
        );
    }
}
