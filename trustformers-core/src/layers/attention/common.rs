//! Common attention utilities and shared components.
//!
//! This module contains shared functionality used across different attention implementations
//! to reduce code duplication and improve maintainability.

#![allow(unused_variables)] // Attention implementation with reserved parameters

use crate::device::Device;
use crate::errors::{Result, TrustformersError};
use crate::layers::Linear;
use crate::tensor::Tensor;

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
}

impl AttentionConfig {
    /// Create a new attention configuration
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        dropout_prob: f32,
        bias: bool,
    ) -> Result<Self> {
        if hidden_size % num_heads != 0 {
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
        })
    }

    /// Set maximum sequence length for optimizations
    pub fn with_max_seq_len(mut self, max_seq_len: usize) -> Self {
        self.max_seq_len = Some(max_seq_len);
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
            query: Linear::new_with_device(config.hidden_size, config.hidden_size, config.bias, device),
            key: Linear::new_with_device(config.hidden_size, config.hidden_size, config.bias, device),
            value: Linear::new_with_device(config.hidden_size, config.hidden_size, config.bias, device),
            out_proj: Linear::new_with_device(config.hidden_size, config.hidden_size, config.bias, device),
        }
    }

    /// Create new attention projections from configuration
    pub fn new(config: &AttentionConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
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
    /// Sets attention scores to -infinity for positions that should be masked
    pub fn apply_causal_mask(attention_scores: &Tensor, seq_len: usize) -> Result<Tensor> {
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

        // For causal masking, we typically expect seq_q == seq_k == seq_len (self-attention)
        // But let's be more flexible and use the actual dimensions
        let actual_seq_len = seq_q.min(seq_k);

        // Create causal mask tensor - lower triangular matrix
        let mut causal_mask_data = vec![0.0f32; seq_q * seq_k];
        for i in 0..seq_q {
            for j in 0..seq_k {
                if j > i {
                    // Upper triangular: mask out future positions
                    causal_mask_data[i * seq_k + j] = f32::NEG_INFINITY;
                } else {
                    // Lower triangular + diagonal: allow past and current positions
                    causal_mask_data[i * seq_k + j] = 0.0;
                }
            }
        }

        // Create causal mask tensor with shape [seq_q, seq_k]
        let causal_mask = Tensor::from_vec(causal_mask_data, &[seq_q, seq_k])?;

        // Apply causal mask element-wise - we need to mask the attention scores directly
        // For now, implement a simple approach that works with the tensor structure
        match (&mut result, &causal_mask) {
            (Tensor::F32(ref mut scores), Tensor::F32(mask)) => {
                let scores_shape = scores.shape();
                let batch_size = scores_shape[0];
                let num_heads = scores_shape[1];

                // Apply mask to each batch and head
                for b in 0..batch_size {
                    for h in 0..num_heads {
                        for i in 0..seq_q {
                            for j in 0..seq_k {
                                if j > i {
                                    // Mask future tokens
                                    scores[[b, h, i, j]] = f32::NEG_INFINITY;
                                }
                            }
                        }
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

    /// Compute attention weights using scaled dot-product
    pub fn compute_attention_weights(
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
        let masked_scores = if causal {
            let seq_len = q.shape()[2];
            Self::apply_causal_mask(&scaled_scores, seq_len)?
        } else {
            scaled_scores
        };

        // Apply softmax to get attention weights
        masked_scores.softmax(-1)
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
    pub fn compute_block_size(
        seq_len: usize,
        head_dim: usize,
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
    fn default() -> Self {
        Self {
            use_flash_attention: false, // Temporarily disabled for testing
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
