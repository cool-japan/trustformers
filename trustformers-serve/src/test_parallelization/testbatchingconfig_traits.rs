//! # TestBatchingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TestBatchingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{BatchingStrategy, TestBatchingConfig};

impl Default for TestBatchingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            optimal_batch_size: 4,
            max_batch_size: 8,
            batching_strategy: BatchingStrategy::ByResource,
            batch_timeout: Duration::from_secs(300),
        }
    }
}
