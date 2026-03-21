//! # WarmupOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `WarmupOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::WarmupOptimizationConfig;

impl Default for WarmupOptimizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            warmup_iterations: 3,
            warmup_timeout: Duration::from_secs(30),
            cache_warmup: true,
            parallel_warmup: true,
        }
    }
}
