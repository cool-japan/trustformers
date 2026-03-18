//! # PerformanceConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::*;

use super::types::{PerformanceAlertThresholds, PerformanceConfig};

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval: Duration::from_secs(300),
            history_size: 288,
            enable_detailed_timing: true,
            enable_percentile_tracking: true,
            max_timing_samples: 10000,
            enable_trend_analysis: true,
            trend_window_size: 12,
            enable_memory_optimization: true,
            cleanup_interval: Duration::from_secs(3600),
            alert_thresholds: PerformanceAlertThresholds::default(),
        }
    }
}

