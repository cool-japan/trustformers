//! # BufferManagerConfig - Trait Implementations
//!
//! This module contains trait implementations for `BufferManagerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::BufferManagerConfig;

impl Default for BufferManagerConfig {
    fn default() -> Self {
        Self {
            max_managed_buffers: 1000,
            default_buffer_capacity: 1024,
            auto_cleanup: true,
            cleanup_interval: Duration::from_secs(300),
            monitoring_enabled: true,
        }
    }
}
