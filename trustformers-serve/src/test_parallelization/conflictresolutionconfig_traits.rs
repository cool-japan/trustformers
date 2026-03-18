//! # ConflictResolutionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConflictResolutionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    ConflictDetectionStrategy, ConflictResolutionConfig, ConflictResolutionStrategy,
};

impl Default for ConflictResolutionConfig {
    fn default() -> Self {
        Self {
            detection_strategy: ConflictDetectionStrategy::Hybrid,
            resolution_strategy: ConflictResolutionStrategy::Queue,
            conflict_timeout: Duration::from_secs(60),
            max_resolution_attempts: 3,
        }
    }
}
