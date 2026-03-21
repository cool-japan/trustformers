//! # DeadlockConfig - Trait Implementations
//!
//! This module contains trait implementations for `DeadlockConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::DeadlockConfig;

impl Default for DeadlockConfig {
    fn default() -> Self {
        Self {
            deadlock_delay: Duration::from_millis(100),
            deadlock_timeout: Duration::from_secs(2),
            detection_timeout: Duration::from_secs(5),
        }
    }
}
