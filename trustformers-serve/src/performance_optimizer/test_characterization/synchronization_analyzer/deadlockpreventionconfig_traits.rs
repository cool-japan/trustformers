//! # DeadlockPreventionConfig - Trait Implementations
//!
//! This module contains trait implementations for `DeadlockPreventionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DeadlockPreventionConfig, PreventionStrategyType};

impl Default for DeadlockPreventionConfig {
    fn default() -> Self {
        Self {
            strategy_priority: vec![
                PreventionStrategyType::LockOrdering,
                PreventionStrategyType::TimeoutBased,
                PreventionStrategyType::LockHierarchy,
                PreventionStrategyType::ResourceOrdering,
                PreventionStrategyType::Dynamic,
            ],
            timeout_prevention: true,
            default_timeout_ms: 5000,
            hierarchy_enforcement: true,
            dynamic_adaptation: true,
            statistics_collection: true,
        }
    }
}
