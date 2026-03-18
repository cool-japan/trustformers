//! # ProfilingStageExecutor - Trait Implementations
//!
//! This module contains trait implementations for `ProfilingStageExecutor`.
//!
//! ## Implemented Traits
//!
//! - `Debug`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ProfilingStageExecutor;

impl std::fmt::Debug for ProfilingStageExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ProfilingStageExecutor")
            .field("config", &self.config)
            .field("stages", &"<trait objects>")
            .field("scheduler", &self.scheduler)
            .field("dependencies", &self.dependencies)
            .field("execution_state", &self.execution_state)
            .field("metrics", &self.metrics)
            .finish()
    }
}
