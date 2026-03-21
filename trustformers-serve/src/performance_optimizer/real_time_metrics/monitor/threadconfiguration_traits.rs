//! # ThreadConfiguration - Trait Implementations
//!
//! This module contains trait implementations for `ThreadConfiguration`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::{RetryConfiguration, ThreadConfiguration, ThreadPriority, ThreadResourceLimits};

impl Default for ThreadConfiguration {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_millis(100),
            timeout: Duration::from_secs(5),
            buffer_size: 1000,
            priority: ThreadPriority::Normal,
            resource_limits: ThreadResourceLimits {
                max_memory: 50 * 1024 * 1024,
                max_cpu: 0.05,
                max_iops: 500,
                max_bandwidth: 5 * 1024 * 1024,
            },
            retry_config: RetryConfiguration {
                max_attempts: 3,
                base_delay: Duration::from_millis(100),
                max_delay: Duration::from_secs(1),
                backoff_multiplier: 2.0,
                jitter_enabled: true,
            },
        }
    }
}
