//! # MonitoringStatus - Trait Implementations
//!
//! This module contains trait implementations for `MonitoringStatus`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::MonitoringStatus;

impl Default for MonitoringStatus {
    fn default() -> Self {
        Self {
            active_runs: 0,
            total_runs_monitored: 0,
            average_completion_time: Duration::from_secs(0),
            current_peak_cpu: 0.0,
            current_peak_memory: 0.0,
        }
    }
}
