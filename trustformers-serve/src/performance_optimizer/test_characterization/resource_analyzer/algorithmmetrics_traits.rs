//! # AlgorithmMetrics - Trait Implementations
//!
//! This module contains trait implementations for `AlgorithmMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::{Duration, Instant};

use super::types::AlgorithmMetrics;

impl Default for AlgorithmMetrics {
    fn default() -> Self {
        Self {
            total_calculations: 0,
            total_duration: Duration::from_secs(0),
            average_duration: Duration::from_secs(0),
            last_calculation: Instant::now(),
            last_quality_score: 0.0,
        }
    }
}
