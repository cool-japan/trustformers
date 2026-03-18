//! # DeadlockPreventionEngine - Trait Implementations
//!
//! This module contains trait implementations for `DeadlockPreventionEngine`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::DeadlockPreventionEngine;

impl std::fmt::Debug for DeadlockPreventionEngine {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DeadlockPreventionEngine")
            .field("config", &self.config)
            .field("ordering_algorithms", &"<dyn trait objects>")
            .field("timeout_manager", &self.timeout_manager)
            .field("hierarchy_enforcer", &self.hierarchy_enforcer)
            .field("allocation_orderer", &self.allocation_orderer)
            .field("dynamic_strategy_manager", &self.dynamic_strategy_manager)
            .field("prevention_statistics", &self.prevention_statistics)
            .finish()
    }
}
