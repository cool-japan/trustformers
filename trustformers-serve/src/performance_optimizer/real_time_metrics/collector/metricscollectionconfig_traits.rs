//! # MetricsCollectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `MetricsCollectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::MetricsCollectionConfig;

impl Default for MetricsCollectionConfig {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_millis(100),
            min_interval: Duration::from_millis(10),
            max_interval: Duration::from_secs(1),
            history_buffer_size: 1000,
            adaptive_sampling: true,
            high_precision_mode: false,
            batch_size: 10,
            compression_enabled: false,
            collection_timeout: Duration::from_secs(5),
            resource_monitoring: true,
            custom_metrics: false,
            stream_publishing: true,
            error_recovery_enabled: true,
            quality_monitoring_enabled: true,
        }
    }
}
