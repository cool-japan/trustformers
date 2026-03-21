//! # DataCollectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `DataCollectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::DataCollectionConfig;

impl Default for DataCollectionConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_millis(100),
            buffer_size_limit: 10 * 1024 * 1024,
            enable_realtime: true,
            sampling_rate: 1.0,
            retention_period: Duration::from_secs(3600),
            enable_compression: false,
            collection_timeout: Duration::from_secs(5),
        }
    }
}
