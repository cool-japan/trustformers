//! # LockDependencyAnalyzerConfig - Trait Implementations
//!
//! This module contains trait implementations for `LockDependencyAnalyzerConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::LockDependencyAnalyzerConfig;

impl Default for LockDependencyAnalyzerConfig {
    fn default() -> Self {
        Self {
            max_graph_depth: 15,
            circular_detection_threshold: 0.90,
            dependency_strength_analysis: true,
            temporal_tracking_window: 300,
            cache_expiration_seconds: 1800,
        }
    }
}
