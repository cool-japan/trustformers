//! # StatisticalAnalyzerStats - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAnalyzerStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use prometheus::core::{Atomic, AtomicF64};
use std::sync::atomic::AtomicU64;

use super::types::StatisticalAnalyzerStats;

impl Default for StatisticalAnalyzerStats {
    fn default() -> Self {
        Self {
            analyses_performed: AtomicU64::new(0),
            avg_processing_time: AtomicF64::new(0.0),
            processing_errors: AtomicU64::new(0),
        }
    }
}
