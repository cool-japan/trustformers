//! Recursive Transformers for Long Sequences
//!
//! This module implements recursive transformer architectures that can efficiently
//! process very long sequences by applying transformations hierarchically and
//! maintaining memory across recursive calls.
//!
//! # Architecture Overview
//!
//! Recursive transformers process sequences through several strategies:
//! - **Divide and Conquer**: Split long sequences into chunks and process recursively
//! - **Hierarchical Processing**: Multi-level attention at different granularities
//! - **Universal Transformers**: Repeated application of the same layer with adaptive computation
//! - **Memory Augmentation**: External memory for maintaining context across chunks
//! - **Adaptive Depth**: Dynamic determination of recursion depth based on complexity
//!
//! # Key Features
//!
//! - **Scalable Sequence Processing**: Handle sequences much longer than typical transformers
//! - **Memory Efficient**: Gradient checkpointing and memory compression
//! - **Adaptive Computation**: Variable depth and computation time based on input complexity
//! - **Hierarchical Attention**: Multi-scale processing for better long-range dependencies
//! - **Multiple Task Heads**: Support for language modeling and classification
//!
//! # Use Cases
//!
//! - **Long Document Processing**: Books, research papers, legal documents
//! - **Code Understanding**: Large codebases with hierarchical structure
//! - **Memory-Efficient Processing**: When GPU memory is limited
//! - **Adaptive Computation**: When computational budget varies by input
//!
//! # References
//!
//! - "Universal Transformers" (Dehghani et al., 2018)
//! - "Adaptive Computation Time for Recurrent Neural Networks" (Graves, 2016)
//! - "Hierarchical Transformers for Long Document Classification" (Various)

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{MemoryUpdateStrategy, RecursionStrategy, RecursiveConfig};
pub use model::{
    DepthPredictor, HierarchyManager, MemoryManager, MemoryState, RecursiveForCausalLM,
    RecursiveForSequenceClassification, RecursiveInput, RecursiveOutput, RecursiveTransformer,
    UniversalController,
};

use trustformers_core::errors::Result;

/// Create a recursive transformer for long documents
pub fn long_document() -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::long_document();
    RecursiveTransformer::new(config)
}

/// Create a Universal Transformer with adaptive computation
pub fn universal() -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::universal();
    RecursiveTransformer::new(config)
}

/// Create a memory-efficient recursive transformer
pub fn memory_efficient() -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::memory_efficient();
    RecursiveTransformer::new(config)
}

/// Create a hierarchical recursive transformer
pub fn hierarchical() -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::hierarchical();
    RecursiveTransformer::new(config)
}

/// Create a recursive transformer for code understanding
pub fn code_understanding() -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::code_understanding();
    RecursiveTransformer::new(config)
}

/// Create a recursive transformer from a pretrained model name
///
/// # Supported Models
///
/// - "recursive-long-document" - For processing long documents
/// - "recursive-universal" - Universal Transformer with ACT
/// - "recursive-memory-efficient" - Optimized for low memory usage
/// - "recursive-hierarchical" - Multi-level hierarchical processing
/// - "recursive-code" - Specialized for code understanding
pub fn from_pretrained(model_name: &str) -> Result<RecursiveTransformer> {
    let config = RecursiveConfig::from_pretrained_name(model_name).ok_or_else(|| {
        trustformers_core::errors::TrustformersError::invalid_config(format!(
            "Unknown model name: {}",
            model_name
        ))
    })?;

    RecursiveTransformer::new(config)
}

/// Create a recursive transformer for causal language modeling
pub fn for_causal_lm(config: RecursiveConfig) -> Result<RecursiveForCausalLM> {
    RecursiveForCausalLM::new(config)
}

/// Create a recursive transformer for sequence classification
pub fn for_sequence_classification(
    config: RecursiveConfig,
    num_labels: usize,
) -> Result<RecursiveForSequenceClassification> {
    RecursiveForSequenceClassification::new(config, num_labels)
}

/// Available model configurations
pub fn available_models() -> Vec<&'static str> {
    vec![
        "recursive-long-document",
        "recursive-universal",
        "recursive-memory-efficient",
        "recursive-hierarchical",
        "recursive-code",
    ]
}

/// Model capabilities and recommended use cases
pub fn model_info(model_name: &str) -> Option<ModelInfo> {
    match model_name {
        "recursive-long-document" => Some(ModelInfo {
            name: "Recursive Long Document",
            description: "Optimized for processing very long documents with hierarchical chunking",
            use_cases: vec!["Book processing", "Legal documents", "Research papers"],
            max_sequence_length: 32768,
            memory_efficient: true,
            adaptive_depth: true,
        }),
        "recursive-universal" => Some(ModelInfo {
            name: "Universal Transformer",
            description: "Recurrent transformer with adaptive computation time",
            use_cases: vec![
                "Variable complexity tasks",
                "Adaptive reasoning",
                "Few-shot learning",
            ],
            max_sequence_length: 16384,
            memory_efficient: false,
            adaptive_depth: true,
        }),
        "recursive-memory-efficient" => Some(ModelInfo {
            name: "Memory Efficient Recursive",
            description: "Optimized for minimal memory usage with gradient checkpointing",
            use_cases: vec![
                "Low-resource environments",
                "Large batch processing",
                "Mobile deployment",
            ],
            max_sequence_length: 8192,
            memory_efficient: true,
            adaptive_depth: false,
        }),
        "recursive-hierarchical" => Some(ModelInfo {
            name: "Hierarchical Recursive",
            description: "Multi-level processing with hierarchical attention patterns",
            use_cases: vec!["Structured documents", "Code analysis", "Nested data"],
            max_sequence_length: 16384,
            memory_efficient: true,
            adaptive_depth: true,
        }),
        "recursive-code" => Some(ModelInfo {
            name: "Code Understanding Recursive",
            description:
                "Specialized for understanding large codebases with hierarchical structure",
            use_cases: vec!["Code completion", "Bug detection", "Code summarization"],
            max_sequence_length: 8192,
            memory_efficient: true,
            adaptive_depth: true,
        }),
        _ => None,
    }
}

/// Information about a specific model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub use_cases: Vec<&'static str>,
    pub max_sequence_length: usize,
    pub memory_efficient: bool,
    pub adaptive_depth: bool,
}

/// Utility functions for working with recursive transformers
/// Create memory state for a given batch and configuration
pub fn create_memory_state(batch_size: usize, config: &RecursiveConfig) -> Result<MemoryState> {
    MemoryState::new(batch_size, config.memory_size, config.hidden_size)
}

/// Calculate optimal chunk size for a given sequence length and memory constraints
pub fn optimal_chunk_size(
    sequence_length: usize,
    memory_limit_mb: usize,
    hidden_size: usize,
) -> usize {
    // Simplified calculation - in practice would consider attention memory complexity
    let memory_per_token = hidden_size * 4; // Approximate bytes per token
    let available_memory_bytes = memory_limit_mb * 1024 * 1024;
    let max_tokens = available_memory_bytes / memory_per_token;

    std::cmp::min(max_tokens, sequence_length / 4) // Use at most 1/4 of sequence as chunk
}

/// Estimate memory usage for a given configuration
pub fn estimate_memory_usage(config: &RecursiveConfig, sequence_length: usize) -> usize {
    let batch_size = 1; // Assume single batch for estimation

    // Embedding memory
    let embedding_memory = config.vocab_size * config.hidden_size * 4; // 4 bytes per float

    // Attention memory (quadratic in sequence length for full attention)
    let chunk_size = std::cmp::min(sequence_length, config.chunk_size);
    let attention_memory = batch_size * config.num_attention_heads * chunk_size * chunk_size * 4;

    // Hidden states memory
    let hidden_memory = batch_size * sequence_length * config.hidden_size * 4;

    // Recursive memory
    let recursive_memory = config.memory_size * config.hidden_size * 4;

    (embedding_memory + attention_memory + hidden_memory + recursive_memory) / (1024 * 1024)
    // Convert to MB
}

/// Performance tips for different use cases
pub fn performance_tips() -> Vec<&'static str> {
    vec![
        "Use gradient checkpointing for memory-constrained scenarios",
        "Enable memory compression for very long sequences",
        "Use adaptive depth for variable complexity inputs",
        "Tune chunk size based on your memory constraints",
        "Enable flash attention for better performance",
        "Use hierarchical attention for structured data",
        "Consider Universal Transformer for reasoning tasks",
        "Use smaller models for mobile/edge deployment",
    ]
}

/// Common configuration presets
pub struct ConfigPresets;

impl ConfigPresets {
    /// Configuration for processing books/novels
    pub fn book_processing() -> RecursiveConfig {
        let mut config = RecursiveConfig::long_document();
        config.chunk_size = 2048;
        config.overlap_size = 256;
        config.recursion_depth = 4;
        config.use_hierarchical_attention = true;
        config
    }

    /// Configuration for code analysis
    pub fn code_analysis() -> RecursiveConfig {
        let mut config = RecursiveConfig::code_understanding();
        config.hierarchy_levels = 4; // File -> Class -> Method -> Statement
        config.use_adaptive_depth = true;
        config.max_depth = 6;
        config
    }

    /// Configuration for legal document processing
    pub fn legal_documents() -> RecursiveConfig {
        let mut config = RecursiveConfig::long_document();
        config.chunk_size = 1024;
        config.memory_size = 2048;
        config.use_memory_compression = true;
        config.compression_ratio = 0.3;
        config
    }

    /// Configuration for research paper processing
    pub fn research_papers() -> RecursiveConfig {
        let mut config = RecursiveConfig::hierarchical();
        config.hierarchy_levels = 3; // Paper -> Section -> Paragraph
        config.level_compression_ratios = vec![1.0, 0.6, 0.3];
        config.cross_level_attention = true;
        config
    }

    /// Configuration for mobile/edge deployment
    pub fn mobile_deployment() -> RecursiveConfig {
        let mut config = RecursiveConfig::memory_efficient();
        config.hidden_size = 384;
        config.intermediate_size = 1536;
        config.num_attention_heads = 6;
        config.chunk_size = 256;
        config.memory_size = 256;
        config
    }
}
