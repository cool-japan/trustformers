//! # TimeoutLoggingConfig - Trait Implementations
//!
//! This module contains trait implementations for `TimeoutLoggingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::TimeoutLoggingConfig;

impl Default for TimeoutLoggingConfig {
    fn default() -> Self {
        Self {
            log_warnings: true,
            log_failures: true,
            log_adjustments: true,
            log_early_terminations: true,
        }
    }
}
