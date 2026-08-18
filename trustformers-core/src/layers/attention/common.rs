//! Common attention utilities and shared components.
//!
//! This module contains shared functionality used across different attention implementations
//! to reduce code duplication and improve maintainability.

use super::mask::MaskView;
use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::layers::Linear;
use crate::tensor::Tensor;
use scirs2_core::ndarray::Axis;

/// Shared configuration for attention layers
#[derive(Debug, Clone)]
pub struct AttentionConfig {
    /// Number of attention heads
    pub num_heads: usize,
    /// Hidden size (must be divisible by num_heads)
    pub hidden_size: usize,
    /// Dimension of each attention head
    pub head_dim: usize,
    /// Dropout probability
    pub dropout_prob: f32,
    /// Whether to use bias in linear layers
    pub bias: bool,
    /// Maximum sequence length for optimizations
    pub max_seq_len: Option<usize>,
    /// Whether the layer is in training mode.
    ///
    /// Attention dropout is applied **only** when this is `true`; the default
    /// is `false` so that a model loaded from a checkpoint with a non-zero
    /// `dropout_prob` produces deterministic inference logits.
    pub training: bool,
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
    ) -> Result<Self> {
        if !hidden_size.is_multiple_of(num_heads) {
            return Err(TrustformersError::config_error(
                &format!(
                    "hidden_size {} must be divisible by num_heads {}",
                    hidden_size, num_heads
                ),
                "AttentionConfig::new",
            ));
        }

        let head_dim = hidden_size / num_heads;

        Ok(Self {
            num_heads,
            hidden_size,
            head_dim,
            dropout_prob,
            bias,
            max_seq_len: None,
            training: false,
        })
    }

    /// Set maximum sequence length for optimizations
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = Some(max_seq_len);
        self
    }

    /// Enable or disable training mode (which gates attention dropout).
    pub fn with_training(mut self, training: bool) -> Self {
        self.training = training;
        self
    }
}

/// Shared attention components used across different attention implementations
#[derive(Debug, Clone)]
pub struct AttentionProjections {
    /// Query projection layer
    pub query: Linear,
    /// Key projection layer
    pub key: Linear,
    /// Value projection layer
    pub value: Linear,
    /// Output projection layer
    pub out_proj: Linear,
}

impl AttentionProjections {
    /// Create new attention projections with device support
    pub fn new_with_device(config: &AttentionConfig, device: Device) -> Self {
        Self {
            query: Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ),
            key: Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ),
            value: Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ),
            out_proj: Linear::new_with_device(
                config.hidden_size,
                config.hidden_size,
                config.bias,
                device,
            ),
        }
    }

    /// Create new attention projections from configuration
    pub fn new(config: &AttentionConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    /// Append all four projections' parameters to `into`.
    ///
    /// `names` supplies the checkpoint sub-path of each projection in
    /// `(query, key, value, out_proj)` order, because the spelling differs per
    /// architecture (BERT's `attention.self.query` vs DistilBERT's
    /// `attention.q_lin`). Each name is joined to `prefix` with a `.`, and each
    /// projection then contributes `.weight` (plus `.bias` when it has one) via
    /// [`Linear::collect_named_parameters`].
    pub fn collect_named_parameters<'a>(
        &'a self,
        prefix: &str,
        names: [&str; 4],
        into: &mut Vec<(String, &'a Tensor)>,
    ) {
        let layers = [&self.query, &self.key, &self.value, &self.out_proj];
        for (layer, name) in layers.into_iter().zip(names) {
            layer.collect_named_parameters(&join_name(prefix, name), into);
        }
    }

    /// Mutable counterpart of [`AttentionProjections::collect_named_parameters`].
    ///
    /// The four projections are separate fields, so all four can be borrowed
    /// mutably at once.
    pub fn collect_named_parameters_mut<'a>(
        &'a mut self,
        prefix: &str,
        names: [&str; 4],
        into: &mut Vec<(String, &'a mut Tensor)>,
    ) {
        let layers = [
            &mut self.query,
            &mut self.key,
            &mut self.value,
            &mut self.out_proj,
        ];
        for (layer, name) in layers.into_iter().zip(names) {
            layer.collect_named_parameters_mut(&join_name(prefix, name), into);
        }
    }
}

/// Join a checkpoint prefix and a relative name with a `.`, tolerating an empty
/// prefix (a model whose parameters sit at the checkpoint root).
pub fn join_name(prefix: &str, name: &str) -> String {
    if prefix.is_empty() {
        name.to_string()
    } else {
        format!("{prefix}.{name}")
    }
}

/// Utilities for attention computation
pub struct AttentionUtils;

impl AttentionUtils {
    /// Split tensor into multiple attention heads
    ///
    /// Converts from [batch, seq_len, hidden_size] to [batch, num_heads, seq_len, head_dim]
    pub fn split_heads(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 3 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Input tensor must have 3 dimensions for split_heads, got {}",
                    shape.len()
                ),
                "AttentionUtils::split_heads",
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_size = shape[2];

        if hidden_size != num_heads * head_dim {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "hidden_size {} must equal num_heads * head_dim ({})",
                    hidden_size,
                    num_heads * head_dim
                ),
                "AttentionUtils::split_heads",
            ));
        }

        // Ensure input tensor has contiguous layout
        let input_contiguous = match tensor {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => tensor.clone(),
        };

        // Reshape to [batch, seq_len, num_heads, head_dim]
        let reshaped = input_contiguous.reshape(&[batch_size, seq_len, num_heads, head_dim])?;

        // Transpose to [batch, num_heads, seq_len, head_dim] (swap dims 1 and 2)
        let transposed = reshaped.transpose(1, 2)?;

        // Ensure final result has contiguous layout
        match transposed {
            Tensor::F32(a) => Ok(Tensor::F32(a.as_standard_layout().to_owned())),
            Tensor::F64(a) => Ok(Tensor::F64(a.as_standard_layout().to_owned())),
            _ => Ok(transposed),
        }
    }

    /// Combine multiple attention heads back into hidden dimension
    ///
    /// Converts from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, hidden_size]
    pub fn combine_heads(tensor: &Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
        let shape = tensor.shape();
        if shape.len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Input tensor must have 4 dimensions for combine_heads, got {}",
                    shape.len()
                ),
                "AttentionUtils::combine_heads",
            ));
        }

        let batch_size = shape[0];
        let seq_len = shape[2];
        let hidden_size = num_heads * head_dim;

        // Ensure input tensor has contiguous layout
        let input_contiguous = match tensor {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => tensor.clone(),
        };

        // Transpose from [batch, num_heads, seq_len, head_dim] to [batch, seq_len, num_heads, head_dim] (swap dims 1 and 2)
        let transposed = input_contiguous.transpose(1, 2)?;

        // Ensure intermediate result has contiguous layout before reshape
        let transposed_contiguous = match transposed {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => transposed,
        };

        // Reshape to [batch, seq_len, hidden_size]
        let reshaped = transposed_contiguous.reshape(&[batch_size, seq_len, hidden_size])?;

        // Ensure final result has contiguous layout
        match reshaped {
            Tensor::F32(a) => Ok(Tensor::F32(a.as_standard_layout().to_owned())),
            Tensor::F64(a) => Ok(Tensor::F64(a.as_standard_layout().to_owned())),
            _ => Ok(reshaped),
        }
    }

    /// Apply causal mask to attention scores
    ///
    /// Sets attention scores to `-inf` for key positions that lie strictly in
    /// the future of their query position. Works for any rank `>= 2`; the
    /// masked extent is taken from the last two dimensions of the tensor.
    ///
    /// `_seq_len` is retained for API compatibility - the mask geometry is
    /// derived from the tensor shape itself.
    pub fn apply_causal_mask(attention_scores: &Tensor, _seq_len: usize) -> Result<Tensor> {
        let mut result = attention_scores.clone();
        let shape = attention_scores.shape();

        // Validate input tensor has at least 2 dimensions for the sequence length
        // For attention scores, shape is typically [batch, num_heads, seq_q, seq_k]
        if shape.len() < 2 {
            return Err(TrustformersError::tensor_op_error(
                &format!("Invalid attention scores shape for causal masking. Expected at least 2 dimensions, got shape: {:?}",
                    shape),
                "apply_causal_mask"
            ));
        }

        let seq_q = shape[shape.len() - 2];
        let seq_k = shape[shape.len() - 1];

        match &mut result {
            Tensor::F32(scores) => {
                let last_axis = Axis(scores.ndim() - 1);
                // Lanes along the last axis are visited in row-major order of
                // the remaining axes, so the query index simply cycles with
                // period `seq_q`. Each masked span is one contiguous fill
                // instead of `seq_q * seq_k` bounds-checked index computations.
                for (position, mut row) in scores.lanes_mut(last_axis).into_iter().enumerate() {
                    let query = position % seq_q;
                    if query + 1 < seq_k {
                        row.slice_mut(scirs2_core::ndarray::s![query + 1..])
                            .fill(f32::NEG_INFINITY);
                    }
                }
            },
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Causal masking only supports F32 tensors currently",
                    "apply_causal_mask",
                ));
            },
        }

        Ok(result)
    }

    /// Add an attention mask to pre-softmax scores.
    ///
    /// See [`super::mask`] for the supported mask shapes and the keep/additive
    /// convention. Masking **must** happen before the softmax - adding a large
    /// negative value to a probability is meaningless.
    pub fn apply_attention_mask_to_scores(scores: &Tensor, mask: &Tensor) -> Result<Tensor> {
        let shape = scores.shape();
        if shape.len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Attention scores must be 4-D [batch, heads, seq_q, seq_k], got {:?}",
                    shape
                ),
                "apply_attention_mask_to_scores",
            ));
        }
        let (batch, heads, seq_q, seq_k) = (shape[0], shape[1], shape[2], shape[3]);
        let view = MaskView::new(mask, batch, heads, seq_q, seq_k)?;

        let mut result = scores.clone();
        match &mut result {
            Tensor::F32(values) => {
                for b in 0..batch {
                    for h in 0..heads {
                        for i in 0..seq_q {
                            for j in 0..seq_k {
                                let penalty = view.additive(b, h, i, j);
                                if penalty != 0.0 {
                                    values[[b, h, i, j]] += penalty;
                                }
                            }
                        }
                    }
                }
                Ok(result)
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Attention masking only supports F32 tensors currently",
                "apply_attention_mask_to_scores",
            )),
        }
    }

    /// Compute pre-softmax attention scores `scale * Q K^T` with optional
    /// causal masking.
    pub fn compute_attention_scores(
        q: &Tensor,
        k: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        // Ensure q and k have contiguous layouts
        let q_contiguous = match q {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => q.clone(),
        };

        let k_contiguous = match k {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => k.clone(),
        };

        // Compute attention scores: Q @ K^T
        let k_transposed = k_contiguous.transpose(2, 3)?;
        let attention_scores = q_contiguous.matmul(&k_transposed)?;

        // Scale by sqrt(head_dim)
        let scaled_scores = attention_scores.scalar_mul(scale)?;

        // Apply causal mask if needed
        if causal {
            let seq_len = q.shape()[2];
            Self::apply_causal_mask(&scaled_scores, seq_len)
        } else {
            Ok(scaled_scores)
        }
    }

    /// Compute attention weights using scaled dot-product
    pub fn compute_attention_weights(
        q: &Tensor,
        k: &Tensor,
        scale: f32,
        causal: bool,
    ) -> Result<Tensor> {
        let scores = Self::compute_attention_scores(q, k, scale, causal)?;
        scores.softmax(-1)
    }

    /// Apply attention weights to values
    pub fn apply_attention(attention_weights: &Tensor, values: &Tensor) -> Result<Tensor> {
        // Ensure both attention_weights and values have contiguous layouts
        let weights_contiguous = match attention_weights {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => attention_weights.clone(),
        };

        let values_contiguous = match values {
            Tensor::F32(a) => Tensor::F32(a.as_standard_layout().to_owned()),
            Tensor::F64(a) => Tensor::F64(a.as_standard_layout().to_owned()),
            _ => values.clone(),
        };

        // Perform matrix multiplication with contiguous tensors
        let result = weights_contiguous.matmul(&values_contiguous)?;

        // Ensure result also has contiguous layout
        match result {
            Tensor::F32(a) => Ok(Tensor::F32(a.as_standard_layout().to_owned())),
            Tensor::F64(a) => Ok(Tensor::F64(a.as_standard_layout().to_owned())),
            _ => Ok(result),
        }
    }

    /// Compute optimal block size for memory-efficient attention
    ///
    /// `_head_dim` is accepted for API symmetry with the other block-size
    /// helpers; the current heuristic only depends on the sequence length and
    /// the memory budget.
    pub fn compute_block_size(
        seq_len: usize,
        _head_dim: usize,
        available_memory_mb: Option<usize>,
    ) -> usize {
        let default_block_size = 256;

        if let Some(mem_mb) = available_memory_mb {
            // Estimate memory usage and compute optimal block size
            let mem_bytes = mem_mb * 1024 * 1024;
            let element_size = 4; // f32 size
            let attention_memory_per_block = default_block_size * default_block_size * element_size;
            let max_blocks = mem_bytes / attention_memory_per_block;

            if max_blocks > 0 {
                (seq_len / max_blocks.max(1)).clamp(32, 512)
            } else {
                default_block_size
            }
        } else {
            default_block_size
        }
    }

    /// Validate attention tensor dimensions
    pub fn validate_attention_dims(
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        expected_num_heads: usize,
        expected_head_dim: usize,
    ) -> Result<()> {
        let q_shape = q.shape();
        let k_shape = k.shape();
        let v_shape = v.shape();

        // Check that all tensors have 4 dimensions [batch, heads, seq_len, head_dim]
        if q_shape.len() != 4 || k_shape.len() != 4 || v_shape.len() != 4 {
            return Err(TrustformersError::tensor_op_error(
                "Q, K, V tensors must have 4 dimensions [batch, heads, seq_len, head_dim]",
                "AttentionUtils::validate_attention_dims",
            ));
        }

        // Check batch size consistency
        if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
            return Err(TrustformersError::tensor_op_error(
                "Q, K, V tensors must have the same batch size",
                "AttentionUtils::validate_attention_dims",
            ));
        }

        // Check number of heads
        if q_shape[1] != expected_num_heads
            || k_shape[1] != expected_num_heads
            || v_shape[1] != expected_num_heads
        {
            return Err(TrustformersError::tensor_op_error(
                &format!("Q, K, V tensors must have {} heads", expected_num_heads),
                "AttentionUtils::validate_attention_dims",
            ));
        }

        // Check head dimension
        if q_shape[3] != expected_head_dim
            || k_shape[3] != expected_head_dim
            || v_shape[3] != expected_head_dim
        {
            return Err(TrustformersError::tensor_op_error(
                &format!(
                    "Q, K, V tensors must have head dimension {}",
                    expected_head_dim
                ),
                "AttentionUtils::validate_attention_dims",
            ));
        }

        // Check key and value sequence length consistency
        if k_shape[2] != v_shape[2] {
            return Err(TrustformersError::tensor_op_error(
                "Key and Value tensors must have the same sequence length",
                "AttentionUtils::validate_attention_dims",
            ));
        }

        Ok(())
    }
}

/// Performance optimization hints for attention computation
#[derive(Debug, Clone)]
pub struct AttentionOptimizationHints {
    /// Whether to use flash attention for memory efficiency
    pub use_flash_attention: bool,
    /// Whether to use paged attention for inference
    pub use_paged_attention: bool,
    /// Block size for tiled attention computation
    pub block_size: usize,
    /// Whether to fuse operations where possible
    pub fuse_operations: bool,
    /// Whether to use half precision for intermediate calculations
    pub use_half_precision: bool,
}

impl Default for AttentionOptimizationHints {
    /// Defaults tuned for short sequences.
    ///
    /// FlashAttention is *off* by default: its tiling only pays for itself once
    /// the `seq_q x seq_k` score matrix stops fitting comfortably in cache, and
    /// for the short sequences that dominate the default path the dense kernel
    /// is faster. [`AttentionOptimizationHints::for_sequence_length`] turns it
    /// on for sequences longer than 512, and it can always be forced on
    /// explicitly.
    fn default() -> Self {
        Self {
            use_flash_attention: false,
            use_paged_attention: false,
            block_size: 256,
            fuse_operations: true,
            use_half_precision: false,
        }
    }
}

impl AttentionOptimizationHints {
    /// Create optimization hints based on sequence length and available memory
    pub fn for_sequence_length(seq_len: usize, available_memory_mb: Option<usize>) -> Self {
        Self {
            use_flash_attention: seq_len > 512, // Use flash attention for longer sequences
            use_paged_attention: seq_len > 2048, // Use paged attention for very long sequences during inference
            block_size: AttentionUtils::compute_block_size(seq_len, 64, available_memory_mb), // Compute optimal block size
            use_half_precision: seq_len > 4096, // Use half precision for very long sequences to save memory
            ..Default::default()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(shape: &[usize]) -> Tensor {
        let count: usize = shape.iter().product();
        let data: Vec<f32> = (0..count).map(|x| x as f32 + 1.0).collect();
        Tensor::from_vec(data, shape).expect("test tensor shape must be valid")
    }

    #[test]
    fn causal_mask_blanks_only_the_strict_upper_triangle() {
        let masked = AttentionUtils::apply_causal_mask(&scores(&[2, 3, 4, 4]), 4)
            .expect("causal masking must succeed");
        let Tensor::F32(values) = &masked else {
            panic!("expected an F32 tensor");
        };
        for b in 0..2 {
            for h in 0..3 {
                for i in 0..4 {
                    for j in 0..4 {
                        let value = values[[b, h, i, j]];
                        if j > i {
                            assert_eq!(
                                value,
                                f32::NEG_INFINITY,
                                "future position ({i}, {j}) must be masked"
                            );
                        } else {
                            assert!(
                                value.is_finite(),
                                "past position ({i}, {j}) must be preserved"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn causal_mask_handles_rectangular_scores() {
        let masked = AttentionUtils::apply_causal_mask(&scores(&[1, 1, 3, 5]), 3)
            .expect("causal masking must succeed");
        let Tensor::F32(values) = &masked else {
            panic!("expected an F32 tensor");
        };
        for i in 0..3 {
            for j in 0..5 {
                let value = values[[0, 0, i, j]];
                assert_eq!(
                    j > i,
                    value == f32::NEG_INFINITY,
                    "unexpected mask state at ({i}, {j})"
                );
            }
        }
    }

    #[test]
    fn causal_mask_leaves_the_input_untouched() {
        let original = scores(&[1, 1, 3, 3]);
        let _ = AttentionUtils::apply_causal_mask(&original, 3).expect("causal masking");
        let data = original.data().expect("input data");
        assert!(
            data.iter().all(|x| x.is_finite()),
            "apply_causal_mask must not mutate its input"
        );
    }

    #[test]
    fn keep_mask_zeroes_out_masked_keys_before_softmax() {
        // A keep mask of 0 must *subtract* from the score, not add to it.
        let raw =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[1, 1, 2, 2]).expect("score tensor shape");
        let mask = Tensor::from_vec(vec![1.0, 0.0], &[1, 1, 1, 2]).expect("mask shape");
        let masked = AttentionUtils::apply_attention_mask_to_scores(&raw, &mask)
            .expect("masking must succeed");
        let data = masked.data().expect("masked data");
        assert_eq!(data[0], 1.0);
        assert_eq!(data[1], f32::NEG_INFINITY);
        assert_eq!(data[2], 3.0);
        assert_eq!(data[3], f32::NEG_INFINITY);
    }

    #[test]
    fn training_defaults_to_disabled() {
        let config = AttentionConfig::new(64, 8, 0.5, true).expect("valid config");
        assert!(!config.training, "inference must be the default mode");
        assert!(config.with_training(true).training);
    }
}
