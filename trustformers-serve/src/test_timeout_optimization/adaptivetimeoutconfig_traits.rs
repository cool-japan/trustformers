//! # AdaptiveTimeoutConfig - Trait Implementations
//!
//! This module contains trait implementations for `AdaptiveTimeoutConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::AdaptiveTimeoutConfig;

impl Default for AdaptiveTimeoutConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            learning_rate: 0.1,
            min_multiplier: 0.5,
            max_multiplier: 3.0,
            history_window: 10,
            success_threshold: 0.9,
            failure_threshold: 0.2,
            escalation_steps: vec![0.7, 0.85, 1.0],
        }
    }
}
