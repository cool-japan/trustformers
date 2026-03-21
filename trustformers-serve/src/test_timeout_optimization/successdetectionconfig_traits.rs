//! # SuccessDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `SuccessDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::SuccessDetectionConfig;

impl Default for SuccessDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            confidence_threshold: 0.95,
            min_execution_time: Duration::from_millis(100),
        }
    }
}
