//! # TaskCancellationConfig - Trait Implementations
//!
//! This module contains trait implementations for `TaskCancellationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{CancellationStrategy, TaskCancellationConfig};

impl Default for TaskCancellationConfig {
    fn default() -> Self {
        Self {
            task_count: 100,
            task_duration: Duration::from_secs(10),
            cancellation_delay: Duration::from_millis(500),
            cancellation_strategy: CancellationStrategy::BroadcastCancel,
            graceful_timeout: Duration::from_secs(2),
            completion_timeout: Duration::from_secs(5),
        }
    }
}
