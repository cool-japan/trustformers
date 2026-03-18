//! # QualityThresholds - Trait Implementations
//!
//! This module contains trait implementations for `QualityThresholds`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::QualityThresholds;

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_completeness: 0.95,
            max_staleness: Duration::from_secs(300),
            min_sample_size: 30,
            max_missing_ratio: 0.05,
            min_confidence: 0.8,
        }
    }
}
