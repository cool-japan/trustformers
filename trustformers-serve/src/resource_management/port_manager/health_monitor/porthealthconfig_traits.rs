//! # PortHealthConfig - Trait Implementations
//!
//! This module contains trait implementations for `PortHealthConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::*;

use super::types::PortHealthConfig;

impl Default for PortHealthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_interval: Duration::from_secs(60),
            history_size: 1000,
            enable_alerts: true,
            enable_trend_analysis: true,
            enable_predictive_indicators: true,
            check_timeout: Duration::from_secs(10),
            alert_throttle_duration: Duration::from_secs(300),
            enable_detailed_logging: false,
        }
    }
}

