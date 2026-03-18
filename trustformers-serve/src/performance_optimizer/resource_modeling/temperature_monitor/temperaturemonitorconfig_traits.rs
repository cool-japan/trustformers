//! # TemperatureMonitorConfig - Trait Implementations
//!
//! This module contains trait implementations for `TemperatureMonitorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{TemperatureMonitorConfig, TemperatureThresholds};

impl Default for TemperatureMonitorConfig {
    fn default() -> Self {
        Self {
            monitoring_interval: Duration::from_secs(1),
            thresholds: TemperatureThresholds::default(),
            enable_predictive_throttling: true,
            enable_cooling_control: true,
            enable_thermal_analysis: true,
            max_history_size: 3600,
            calibration_interval: Duration::from_secs(86400),
            enable_alerting: true,
            alert_escalation_timeout: Duration::from_secs(5 * 60),
            enable_heat_analysis: true,
        }
    }
}
