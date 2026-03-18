//! # AnalyticsStats - Trait Implementations
//!
//! This module contains trait implementations for `AnalyticsStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use prometheus::core::{Atomic, AtomicF64};
use std::sync::atomic::AtomicU64;

use super::types::AnalyticsStats;

impl Default for AnalyticsStats {
    fn default() -> Self {
        Self {
            analyses_performed: AtomicU64::new(0),
            avg_analysis_duration: AtomicF64::new(0.0),
            cache_hit_rate: AtomicF64::new(0.0),
            analysis_errors: AtomicU64::new(0),
            memory_usage: AtomicU64::new(0),
            cpu_usage: AtomicF64::new(0.0),
        }
    }
}
