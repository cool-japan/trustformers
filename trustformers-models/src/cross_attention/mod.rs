//! # Cross-Attention Variants
//!
//! This module implements various cross-attention mechanisms that can be used
//! across different transformer architectures for enhanced model capabilities.
//!
//! ## Cross-Attention Types
//!
//! Cross-attention enables models to attend to information from different sequences
//! or modalities. This module provides several variants:
//!
//! - **Standard Cross-Attention**: Basic cross-attention mechanism
//! - **Multi-Head Cross-Attention**: Parallel attention heads for different representations
//! - **Sparse Cross-Attention**: Efficient attention with sparsity patterns
//! - **Hierarchical Cross-Attention**: Multi-scale attention across different levels
//! - **Adaptive Cross-Attention**: Dynamic attention based on input characteristics
//! - **Gated Cross-Attention**: Controllable attention with gating mechanisms
//!
//! ## Use Cases
//!
//! Cross-attention is essential for:
//! - **Encoder-Decoder Models**: Decoder attending to encoder states
//! - **Multimodal Models**: Text attending to vision features
//! - **Retrieval-Augmented Models**: Attending to retrieved knowledge
//! - **Memory-Augmented Models**: Attending to external memory
//! - **Multi-Document Models**: Attending across multiple documents
//!
//! ## Architecture Overview
//!
//! Cross-attention computes attention between:
//! - Query (Q): From the target sequence
//! - Key (K) and Value (V): From the source sequence
//!
//! This allows the model to selectively focus on relevant parts of the source
//! when processing each element of the target sequence.
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use trustformers_models::cross_attention::{CrossAttention, CrossAttentionConfig};
//!
//! let config = CrossAttentionConfig {
//!     hidden_size: 512,
//!     num_heads: 8,
//!     attention_dropout: 0.1,
//!     ..Default::default()
//! };
//!
//! let cross_attn = CrossAttention::new(config)?;
//!
//! // Query from target sequence, Key/Value from source sequence
//! let output = cross_attn.forward(query_states, key_states, value_states)?;
//! ```

pub mod config;
pub mod layers;
pub mod utils;

pub use config::CrossAttentionConfig;
pub use layers::{
    AdaptiveCrossAttention, CrossAttention, GatedCrossAttention, HierarchicalCrossAttention,
    MultiHeadCrossAttention, SparseCrossAttention,
};
pub use utils::{
    compute_attention_stats, create_attention_mask, create_hierarchical_mask, create_sparse_mask,
    CrossAttentionOutput,
};
