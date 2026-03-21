//! # BatchConfig - Trait Implementations
//!
//! This module contains trait implementations for `BatchConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::BatchConfig;

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 512,
            max_batch_timeout: Duration::from_secs(5),
            max_queue_size: 2048,
        }
    }
}
