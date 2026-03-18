//! # DataRetentionConfig - Trait Implementations
//!
//! This module contains trait implementations for `DataRetentionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::DataRetentionConfig;

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            execution_results_retention: Duration::from_secs(7 * 24 * 3600),
            performance_metrics_retention: Duration::from_secs(30 * 24 * 3600),
            alert_history_retention: Duration::from_secs(90 * 24 * 3600),
            cleanup_interval: Duration::from_secs(24 * 3600),
        }
    }
}
