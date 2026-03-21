//! # EarlyTerminationConfig - Trait Implementations
//!
//! This module contains trait implementations for `EarlyTerminationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{EarlyTerminationConfig, FastFailConfig, SuccessDetectionConfig};

impl Default for EarlyTerminationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            progress_check_interval: Duration::from_millis(500),
            min_progress_rate: 0.1,
            success_detection: SuccessDetectionConfig::default(),
            fast_fail: FastFailConfig::default(),
        }
    }
}
