//! # AsyncMemoryPressureConfig - Trait Implementations
//!
//! This module contains trait implementations for `AsyncMemoryPressureConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AsyncMemoryPressureConfig;

impl Default for AsyncMemoryPressureConfig {
    fn default() -> Self {
        Self {
            memory_pressure_mb: 500,
            concurrent_async_tasks: 20,
            pressure_duration: Duration::from_secs(5),
        }
    }
}
