//! # MemoryOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::MemoryOptimizationConfig;

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            memory_aware_scheduling: true,
            memory_warning_threshold: 0.7,
            memory_throttling_threshold: 0.85,
            gc_hints: true,
            cleanup_between_tests: true,
        }
    }
}
