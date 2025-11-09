//! Neural network layers for transformer architectures.
//!
//! This module provides the fundamental building blocks for constructing
//! transformer models. Each layer implements the [`Layer`](crate::traits::Layer)
//! trait and can be composed to create complete architectures.
//!
//! # Overview
//!
//! The layers module includes:
//!
//! - **Basic Layers**:
//!   - [`Linear`]: Fully connected linear transformation
//!   - [`Embedding`]: Token and position embeddings
//!   - [`LayerNorm`]: Layer normalization for training stability
//!   - [`FeedForward`]: Position-wise feed-forward networks
//!
//! - **Attention Mechanisms**:
//!   - [`MultiHeadAttention`]: Standard multi-head self/cross attention
//!   - [`FlashAttention`]: Memory-efficient attention with O(N) complexity
//!   - [`PagedAttention`]: Paged KV cache for efficient inference
//!   - [`SDPA`]: Scaled Dot-Product Attention with optimizations
//!   - [`MultiQueryAttention`]: Efficient attention with shared KV heads
//!   - [`GroupedQueryAttention`]: Balance between MHA and MQA
//!
//! # Example
//!
//! ```no_run
//! use trustformers_core::layers::{Linear, LayerNorm, MultiHeadAttention};
//! use trustformers_core::tensor::Tensor;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create layers for a transformer block
//! let attention = MultiHeadAttention::new(768, 12, 0.1)?;
//! let norm1 = LayerNorm::new(vec![768], 1e-5)?;
//! let ffn = Linear::new(768, 3072, true)?;
//! let norm2 = LayerNorm::new(vec![768], 1e-5)?;
//!
//! // Use in a forward pass
//! let input = Tensor::randn(&[2, 128, 768])?;
//! let attended = attention.forward(input)?;
//! # Ok(())
//! # }
//! ```
//!
//! # Performance Notes
//!
//! - Use `FlashAttention` for long sequences to reduce memory usage
//! - `PagedAttention` is optimal for inference with KV caching
//! - SIMD operations are used throughout for better CPU performance
//! - GPU acceleration is available with appropriate features enabled

pub mod attention;
pub mod conv2d;
pub mod dropout;
pub mod embedding;
pub mod feedforward;
pub mod flash_attention;
pub mod layernorm;
pub mod linear;
pub mod paged_attention;
pub mod sdpa;

#[cfg(test)]
mod tests;

pub use attention::{
    AttentionConfig, AttentionOptimizationHints, AttentionProjections, AttentionUtils,
    FlashAttention, MultiHeadAttention,
};
pub use conv2d::Conv2d;

// Keep AttentionInput for backward compatibility
#[derive(Debug, Clone)]
pub struct AttentionInput {
    pub hidden_states: crate::tensor::Tensor,
    pub attention_mask: Option<crate::tensor::Tensor>,
}

impl AttentionInput {
    pub fn new(hidden_states: crate::tensor::Tensor) -> Self {
        Self {
            hidden_states,
            attention_mask: None,
        }
    }

    pub fn with_mask(
        hidden_states: crate::tensor::Tensor,
        attention_mask: crate::tensor::Tensor,
    ) -> Self {
        Self {
            hidden_states,
            attention_mask: Some(attention_mask),
        }
    }
}
pub use dropout::Dropout;
pub use embedding::Embedding;
pub use feedforward::FeedForward;
pub use flash_attention::{FlashAttentionInput, GroupedQueryAttention, MultiQueryAttention};
pub use layernorm::{LayerNorm, RMSNorm};
pub use linear::Linear;
pub use paged_attention::{KVCache, MemoryStats, PagedAttention, PagedAttentionInput};
pub use sdpa::SDPA;
