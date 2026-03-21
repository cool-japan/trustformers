//! # AnalysisQualityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisQualityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::types::*;

use std::time::Duration;

impl Default for AnalysisQualityThresholds {
    fn default() -> Self {
        Self {
            min_dependency_accuracy: 0.8,
            min_conflict_accuracy: 0.85,
            min_grouping_quality: 0.7,
            max_analysis_time: Duration::from_secs(120),
        }
    }
}
