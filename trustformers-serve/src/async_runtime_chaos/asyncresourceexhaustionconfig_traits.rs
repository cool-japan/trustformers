//! # AsyncResourceExhaustionConfig - Trait Implementations
//!
//! This module contains trait implementations for `AsyncResourceExhaustionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AsyncResourceExhaustionConfig;

impl Default for AsyncResourceExhaustionConfig {
    fn default() -> Self {
        Self {
            total_tasks: 50,
            max_concurrent_resources: 10,
            resource_timeout: Duration::from_millis(200),
            work_duration: Duration::from_millis(100),
        }
    }
}
