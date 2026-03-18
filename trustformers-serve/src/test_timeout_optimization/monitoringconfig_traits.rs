//! # MonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{MonitoringConfig, TimeoutLoggingConfig};

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            collection_interval: Duration::from_millis(100),
            tracked_percentiles: vec![50.0, 90.0, 95.0, 99.0],
            regression_threshold: 0.2,
            timeout_logging: TimeoutLoggingConfig::default(),
        }
    }
}
