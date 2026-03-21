//! # ThresholdMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `ThresholdMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::ThresholdMonitorConfig;

impl Default for ThresholdMonitorConfig {
    fn default() -> Self {
        Self {
            enable_monitoring: true,
            monitoring_interval: Duration::from_secs(30),
            enable_adaptive_thresholds: true,
            enable_alert_suppression: true,
            enable_alert_correlation: true,
            enable_performance_analysis: true,
            max_alert_history: 10000,
            alert_processing_timeout: Duration::from_secs(30),
            enable_detailed_logging: false,
        }
    }
}
