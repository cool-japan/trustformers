//! # QueryEncryptionConfig - Trait Implementations
//!
//! This module contains trait implementations for `QueryEncryptionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{QueryEncryptionConfig, QueryParsingMode};

impl Default for QueryEncryptionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            encrypt_parameters: true,
            encrypt_results: false,
            parsing_mode: QueryParsingMode::ParameterOnly,
        }
    }
}

