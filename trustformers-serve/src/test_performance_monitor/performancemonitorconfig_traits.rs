//! # PerformanceMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    AlertingConfig, DashboardConfig, DataRetentionConfig, PerformanceMonitorConfig, ReportingConfig,
};

impl Default for PerformanceMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(60),
            dashboard: DashboardConfig::default(),
            alerting: AlertingConfig::default(),
            reporting: ReportingConfig::default(),
            retention: DataRetentionConfig::default(),
        }
    }
}
