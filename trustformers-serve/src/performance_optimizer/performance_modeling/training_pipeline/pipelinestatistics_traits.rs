//! # PipelineStatistics - Trait Implementations
//!
//! This module contains trait implementations for `PipelineStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{MonitoringStatus, PipelineStatistics};

impl Default for PipelineStatistics {
    fn default() -> Self {
        Self {
            total_runs: 0,
            successful_runs: 0,
            success_rate: 0.0,
            average_duration: Duration::from_secs(0),
            best_validation_score: 0.0,
            last_run: None,
            current_status: MonitoringStatus::default(),
        }
    }
}
