//! # QueueManagementConfig - Trait Implementations
//!
//! This module contains trait implementations for `QueueManagementConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::QueueManagementConfig;

impl Default for QueueManagementConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 1000,
            queue_timeout: Duration::from_secs(3600),
            priority_boost_interval: Duration::from_secs(300),
            starvation_prevention: true,
            compaction_interval: Duration::from_secs(60),
        }
    }
}
