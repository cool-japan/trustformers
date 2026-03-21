//! # ResourceLimits - Trait Implementations
//!
//! This module contains trait implementations for `ResourceLimits`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ResourceLimits;

impl Default for ResourceLimits {
    fn default() -> Self {
        let cpu_count = num_cpus::get();
        Self {
            cpu_intensive_tests: cpu_count / 2,
            memory_intensive_tests: 2,
            gpu_tests: 1,
            network_tests: 4,
            filesystem_tests: 8,
            database_tests: 2,
            total_memory_limit_mb: Some(8192),
            total_cpu_limit_percent: Some(80.0),
        }
    }
}
