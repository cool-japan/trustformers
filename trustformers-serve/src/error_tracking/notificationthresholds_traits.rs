//! # NotificationThresholds - Trait Implementations
//!
//! This module contains trait implementations for `NotificationThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::NotificationThresholds;

impl Default for NotificationThresholds {
    fn default() -> Self {
        Self {
            critical_errors_per_minute: 10.0,
            high_error_rate_per_minute: 50.0,
            new_error_types_per_hour: 5,
            error_spike_threshold_percent: 200.0,
            spike_threshold: 10,
            consecutive_threshold: 5,
        }
    }
}
