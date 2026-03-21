//! Tokenizer Training Module Organization
//!
//! This module provides comprehensive tokenizer training capabilities including
//! BPE, WordPiece, and Unigram algorithms, advanced metrics, corpus processing,
//! distributed training, and comprehensive analysis tools.

/// Core training algorithms (BPE, WordPiece, Unigram)
pub mod algorithms;

/// Configuration structures and utilities
pub mod config;

/// Training metrics and incremental training
pub mod metrics;

/// Corpus processing and streaming utilities
pub mod corpus;

/// Distributed and concurrent training coordination
pub mod distributed;

/// Advanced analysis tools (coverage, language detection, distribution)
pub mod analysis;

/// Training utilities and optimization helpers
pub mod utils;

// Re-export all public types for backward compatibility
pub use self::algorithms::*;
pub use self::analysis::*;
pub use self::config::*;
pub use self::corpus::*;
pub use self::distributed::*;
pub use self::metrics::*;
pub use self::utils::*;
