//! # ConflictDetectorConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConflictDetectorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::*;
use crate::resource_management::types::*;

use super::types::ConflictDetectorConfig;

impl Default for ConflictDetectorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            enable_auto_resolution: true,
            max_alternative_search: 100,
            resolution_timeout: Duration::from_secs(5),
            enable_priority_resolution: true,
            max_history_size: 10000,
            enable_detailed_logging: true,
        }
    }
}

