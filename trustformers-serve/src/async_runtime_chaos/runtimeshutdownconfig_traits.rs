//! # RuntimeShutdownConfig - Trait Implementations
//!
//! This module contains trait implementations for `RuntimeShutdownConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::RuntimeShutdownConfig;

impl Default for RuntimeShutdownConfig {
    fn default() -> Self {
        Self {
            worker_threads: 4,
            concurrent_operations: 50,
            operation_duration: Duration::from_secs(30),
            startup_delay: Duration::from_millis(200),
            graceful_shutdown_timeout: Duration::from_secs(5),
        }
    }
}
