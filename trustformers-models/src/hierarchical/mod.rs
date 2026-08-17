//! # Hierarchical Transformers
//!
//! This module implements hierarchical transformer architectures that process
//! sequences at multiple scales, enabling efficient handling of long sequences
//! and hierarchical pattern recognition.
//!
//! ## Key Concepts
//!
//! Hierarchical transformers operate on the principle of multi-scale processing:
//! - **Local Processing**: Fine-grained attention at token level
//! - **Regional Processing**: Medium-scale attention over token groups
//! - **Global Processing**: Coarse-grained attention over entire sequence
//!
//! ## Architecture Variants
//!
//! This module provides several hierarchical transformer variants:
//! - **Hierarchical Attention**: Multi-level attention with different scales
//! - **Pyramid Transformer**: Progressively coarsening representations
//! - **Nested Transformer**: Hierarchical encoder-decoder structures
//! - **Tree Transformer**: Tree-structured attention patterns
//! - **Hierarchical Memory**: Multi-scale memory mechanisms
//!
//! ## Applications
//!
//! Hierarchical transformers are particularly useful for:
//! - **Long Document Processing**: Efficient attention over very long sequences
//! - **Image Processing**: Multi-scale visual feature extraction
//! - **Speech Recognition**: Hierarchical audio pattern recognition
//! - **Code Understanding**: Multi-level program structure analysis
//! - **Scientific Documents**: Hierarchical text structure processing
//!
//! ## Performance Benefits
//!
//! - **Reduced Complexity**: O(n log n) instead of O(n²) for attention
//! - **Better Inductive Biases**: Natural hierarchical structure modeling
//! - **Improved Generalization**: Multi-scale feature learning
//! - **Memory Efficiency**: Hierarchical memory usage patterns
//!
//! ## Example Usage
//!
//! ```rust,no_run
//! use trustformers_models::hierarchical::{HierarchicalTransformer, HierarchicalConfig};
//! use trustformers_core::traits::Model;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let config = HierarchicalConfig {
//!     hidden_size: 768,
//!     num_levels: 4,
//!     reduction_factor: 2,
//!     num_heads: 12,
//!     ..Default::default()
//! };
//!
//! # let vocab_size = 32000;
//! let model = HierarchicalTransformer::new(config, vocab_size)?;
//! # let input_ids: Vec<u32> = vec![1, 2, 3, 4, 5];
//! let output = model.forward(input_ids)?;
//! # let _ = output;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod layers;
pub mod models;
pub mod utils;

#[cfg(test)]
mod tests;

pub use config::{HierarchicalConfig, HierarchicalType, ReductionMethod};
pub use layers::{
    HierarchicalAttention, HierarchicalEncoder, HierarchicalLayer, NestedTransformerLayer,
    PyramidLayer, TreeAttention,
};
pub use models::{
    HierarchicalForLanguageModeling, HierarchicalForSequenceClassification,
    HierarchicalTransformer, NestedTransformer, PyramidTransformer, TreeTransformer,
};
pub use utils::{
    build_hierarchy, compute_hierarchical_positions, create_tree_mask, hierarchical_pooling,
    hierarchical_upsampling, HierarchicalOutput,
};
