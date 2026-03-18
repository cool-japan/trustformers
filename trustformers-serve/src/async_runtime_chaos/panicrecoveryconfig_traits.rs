//! # PanicRecoveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `PanicRecoveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{PanicRecoveryConfig, PanicType};

impl Default for PanicRecoveryConfig {
    fn default() -> Self {
        Self {
            total_tasks: 50,
            panic_task_count: 10,
            panic_type: PanicType::Immediate,
        }
    }
}
