//! # StreamingConfig - Trait Implementations
//!
//! This module contains trait implementations for `StreamingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::StreamingConfig;

impl Default for StreamingConfig {
    fn default() -> Self {
        Self {
            buffer_size: 10000,
            worker_count: 4,
            backpressure_threshold: 0.8,
            flow_control_enabled: true,
            adaptive_processing: true,
        }
    }
}
