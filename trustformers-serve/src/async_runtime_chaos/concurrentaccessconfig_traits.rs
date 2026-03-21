//! # ConcurrentAccessConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConcurrentAccessConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ConcurrentAccessConfig, ConcurrentAccessPattern};

impl Default for ConcurrentAccessConfig {
    fn default() -> Self {
        Self {
            concurrent_tasks: 20,
            operations_per_task: 50,
            access_pattern: ConcurrentAccessPattern::Mixed,
        }
    }
}
