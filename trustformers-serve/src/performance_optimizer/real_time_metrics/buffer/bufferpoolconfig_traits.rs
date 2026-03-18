//! # BufferPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `BufferPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{BufferPoolConfig, PreallocationStrategy};

impl Default for BufferPoolConfig {
    fn default() -> Self {
        Self {
            max_buffers: 100,
            min_buffers: 10,
            default_capacity: 1024,
            cleanup_interval: Duration::from_secs(60),
            max_idle_time: Duration::from_secs(300),
            monitoring_enabled: true,
            preallocation_strategy: PreallocationStrategy::Lazy,
        }
    }
}
