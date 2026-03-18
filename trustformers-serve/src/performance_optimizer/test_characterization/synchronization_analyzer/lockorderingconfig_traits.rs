//! # LockOrderingConfig - Trait Implementations
//!
//! This module contains trait implementations for `LockOrderingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{LockOrderingConfig, OrderingAlgorithmType};

impl Default for LockOrderingConfig {
    fn default() -> Self {
        Self {
            consistency_enforcement: true,
            performance_optimization: true,
            dynamic_adaptation: true,
            violation_sensitivity: 0.80,
            history_tracking: true,
            algorithm_preference: OrderingAlgorithmType::DeadlockFree,
        }
    }
}
