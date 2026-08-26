//! Context summarization module for conversational AI pipeline.
//!
//! This module provides comprehensive conversation context summarization capabilities,
//! including multiple summarization strategies, quality assessment, token management,
//! and optimization for different conversation types and requirements.
//!
//! # Features
//!
//! - **Multiple Strategies**: Extractive, abstractive, and hybrid summarization
//! - **Context Compression**: Intelligent compression while preserving key information
//! - **Quality Assessment**: Automatic summary quality scoring and validation
//! - **Token Management**: Precise token counting and context window management
//! - **Adaptive Algorithms**: Different algorithms for different conversation types
//! - **Performance Optimization**: Efficient summarization with minimal latency
//! - **Error Recovery**: Robust error handling and fallback mechanisms
//!
//! Split into cohesive submodules: [`types`] (result/weight/threshold data types and
//! the public `SummarizationEngine`/`SummarizationMetadata` aliases), [`engine`] (the
//! `ContextSummarizer` engine itself) and [`factory`] (config validation and
//! pre-configured constructors).

pub mod engine;
pub mod factory;
#[cfg(test)]
mod tests;
pub mod types;

// Re-export all types
pub use engine::*;
pub use factory::*;
pub use types::*;
