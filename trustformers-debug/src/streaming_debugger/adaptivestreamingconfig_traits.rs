//! # AdaptiveStreamingConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveStreamingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for AdaptiveStreamingConfig {
    fn default() -> Self {
        Self {
            enable_adaptive_streaming: true,
            min_quality: 0.1,
            max_quality: 1.0,
            bandwidth_threshold: 1_000_000,
            latency_threshold_ms: 100,
            quality_step: 0.1,
            monitoring_interval_ms: 1000,
        }
    }
}
