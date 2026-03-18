//! # AdaptiveSchedulingParams - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveSchedulingParams`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AdaptiveSchedulingParams;

impl Default for AdaptiveSchedulingParams {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            adaptation_interval: Duration::from_secs(300),
            history_window: 100,
            min_confidence: 0.7,
            max_adaptation_rate: 0.3,
        }
    }
}
