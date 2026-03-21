//! # FrameworkStats - Trait Implementations
//!
//! This module contains trait implementations for `FrameworkStats`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, AtomicUsize};
use std::time::Instant;

use super::types::FrameworkStats;

impl Default for FrameworkStats {
    fn default() -> Self {
        Self {
            total_tests: AtomicUsize::new(0),
            optimized_tests: AtomicUsize::new(0),
            total_time_saved: AtomicU64::new(0),
            uptime_start: Instant::now(),
            active_test_count: AtomicUsize::new(0),
            peak_concurrent_tests: AtomicUsize::new(0),
        }
    }
}

impl Clone for FrameworkStats {
    fn clone(&self) -> Self {
        Self {
            total_tests: AtomicUsize::new(
                self.total_tests.load(std::sync::atomic::Ordering::Relaxed),
            ),
            optimized_tests: AtomicUsize::new(
                self.optimized_tests.load(std::sync::atomic::Ordering::Relaxed),
            ),
            total_time_saved: AtomicU64::new(
                self.total_time_saved.load(std::sync::atomic::Ordering::Relaxed),
            ),
            uptime_start: self.uptime_start,
            active_test_count: AtomicUsize::new(
                self.active_test_count.load(std::sync::atomic::Ordering::Relaxed),
            ),
            peak_concurrent_tests: AtomicUsize::new(
                self.peak_concurrent_tests.load(std::sync::atomic::Ordering::Relaxed),
            ),
        }
    }
}
