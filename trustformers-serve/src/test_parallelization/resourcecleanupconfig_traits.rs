//! # ResourceCleanupConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceCleanupConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::{CleanupStrategy, ResourceCleanupConfig};

impl Default for ResourceCleanupConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cleanup_interval: Duration::from_secs(300),
            cleanup_strategies: HashMap::from([
                ("network_ports".to_string(), CleanupStrategy::Immediate),
                (
                    "temp_directories".to_string(),
                    CleanupStrategy::Deferred(Duration::from_secs(600)),
                ),
                ("gpu_memory".to_string(), CleanupStrategy::Immediate),
                (
                    "database_connections".to_string(),
                    CleanupStrategy::Immediate,
                ),
            ]),
            force_cleanup_on_shutdown: true,
        }
    }
}
