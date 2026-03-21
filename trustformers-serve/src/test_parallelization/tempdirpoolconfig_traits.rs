//! # TempDirPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `TempDirPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{TempDirCleanupStrategy, TempDirPoolConfig};

impl Default for TempDirPoolConfig {
    fn default() -> Self {
        Self {
            base_dir: "/tmp/trustformers_tests".to_string(),
            max_directories: 100,
            cleanup_strategy: TempDirCleanupStrategy::AtEnd,
            size_limit_mb: Some(1024),
        }
    }
}
