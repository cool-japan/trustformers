//! # IoMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `IoMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::IoMonitorConfig;

impl Default for IoMonitorConfig {
    fn default() -> Self {
        Self {
            per_device_monitoring: true,
            latency_tracking: true,
            queue_depth_monitoring: true,
            device_health_monitoring: true,
            latency_threshold: Duration::from_millis(100),
            health_check_interval: Duration::from_secs(60),
        }
    }
}
