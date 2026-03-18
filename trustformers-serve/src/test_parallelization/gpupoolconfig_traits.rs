//! # GpuPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `GpuPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GpuPoolConfig, GpuSharingStrategy};

impl Default for GpuPoolConfig {
    fn default() -> Self {
        Self {
            device_ids: vec![0],
            memory_limit_mb: Some(2048),
            utilization_limit: Some(0.8),
            sharing_strategy: GpuSharingStrategy::SharedMemory,
        }
    }
}
