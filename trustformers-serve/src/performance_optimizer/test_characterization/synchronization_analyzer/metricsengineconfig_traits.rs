//! # MetricsEngineConfig - Trait Implementations
//!
//! This module contains trait implementations for `MetricsEngineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MetricsEngineConfig;

impl Default for MetricsEngineConfig {
    fn default() -> Self {
        Self {
            real_time_collection: true,
            retention_period_days: 30,
            sampling_rate: 1.0,
            trend_analysis: true,
            incident_tracking: true,
            database_persistence: false,
        }
    }
}
