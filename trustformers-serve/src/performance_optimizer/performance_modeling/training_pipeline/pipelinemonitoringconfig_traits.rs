//! # PipelineMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `PipelineMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PipelineMonitoringConfig;

impl Default for PipelineMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_detailed_logging: true,
            enable_resource_monitoring: true,
            progress_interval: Duration::from_secs(30),
            enable_early_stopping: true,
            early_stopping_patience: 10,
        }
    }
}
