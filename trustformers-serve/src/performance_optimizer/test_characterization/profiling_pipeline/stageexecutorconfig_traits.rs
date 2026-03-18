//! # StageExecutorConfig - Trait Implementations
//!
//! This module contains trait implementations for `StageExecutorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{StageExecutorConfig, StageResourceAllocation};

impl Default for StageExecutorConfig {
    fn default() -> Self {
        Self {
            max_parallel_stages: 4,
            stage_timeout: Duration::from_secs(120),
            retry_attempts: 3,
            enable_dependency_checking: true,
            enable_stage_caching: true,
            resource_allocation: StageResourceAllocation {
                cpu_cores_per_stage: 1,
                memory_per_stage: 512 * 1024 * 1024,
                io_quota_per_stage: 10 * 1024 * 1024,
            },
        }
    }
}
