//! # IntelligentBufferingConfig - Trait Implementations
//!
//! This module contains trait implementations for `IntelligentBufferingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for IntelligentBufferingConfig {
    fn default() -> Self {
        Self {
            enable_intelligent_buffering: true,
            buffer_strategy: BufferStrategy::Adaptive,
            min_buffer_size: 100,
            max_buffer_size: 10000,
            utilization_threshold: 0.8,
            adjustment_factor: 1.5,
            enable_priority_buffering: true,
            memory_pressure_threshold: 0.9,
        }
    }
}
