//! # CompressionConfig - Trait Implementations
//!
//! This module contains trait implementations for `CompressionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{CompressionAlgorithm, CompressionConfig};

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            algorithm: CompressionAlgorithm::Lz4,
            level: 4,
            min_size: 1024,
            enable_stats: true,
            timeout: Duration::from_secs(5),
        }
    }
}
