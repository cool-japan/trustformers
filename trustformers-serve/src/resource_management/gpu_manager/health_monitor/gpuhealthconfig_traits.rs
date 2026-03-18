//! # GpuHealthConfig - Trait Implementations
//!
//! This module contains trait implementations for `GpuHealthConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use chrono::Duration as ChronoDuration;

use super::types::GpuHealthConfig;

impl Default for GpuHealthConfig {
    fn default() -> Self {
        Self {
            check_interval: Duration::from_secs(30),
            temperature_threshold: 85.0,
            memory_threshold: 95.0,
            utilization_threshold: 98.0,
            power_threshold: 400.0,
            health_score_threshold: 0.7,
            consecutive_checks_threshold: 3,
            enable_prediction: true,
            history_retention: ChronoDuration::days(7),
            analytics_interval: Duration::from_secs(300),
        }
    }
}
