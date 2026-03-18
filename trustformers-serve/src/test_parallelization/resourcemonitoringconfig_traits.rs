//! # ResourceMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ResourceAlertConfig, ResourceMonitoringConfig, ResourceUsageThresholds};

impl Default for ResourceMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(1),
            usage_thresholds: ResourceUsageThresholds::default(),
            alerts: ResourceAlertConfig::default(),
        }
    }
}
