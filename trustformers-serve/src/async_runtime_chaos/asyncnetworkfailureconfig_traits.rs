//! # AsyncNetworkFailureConfig - Trait Implementations
//!
//! This module contains trait implementations for `AsyncNetworkFailureConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AsyncNetworkFailureConfig;

impl Default for AsyncNetworkFailureConfig {
    fn default() -> Self {
        Self {
            concurrent_operations: 20,
            max_retries: 3,
            base_backoff_ms: 100,
            failure_start_delay: Duration::from_millis(100),
            failure_duration: Duration::from_secs(2),
        }
    }
}
