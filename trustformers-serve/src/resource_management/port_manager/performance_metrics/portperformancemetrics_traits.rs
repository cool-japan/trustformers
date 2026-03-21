//! # PortPerformanceMetrics - Trait Implementations
//!
//! This module contains trait implementations for `PortPerformanceMetrics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::{Arc, atomic::AtomicU64};

use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use super::types::*;

use super::types::{PerformanceConfig, PerformanceTrends, PortPerformanceMetrics};

impl Default for PortPerformanceMetrics {
    fn default() -> Self {
        let now = Utc::now();
        Self {
            total_allocations: AtomicU64::new(0),
            total_deallocations: AtomicU64::new(0),
            total_reservations: AtomicU64::new(0),
            total_conflicts: AtomicU64::new(0),
            total_allocation_failures: AtomicU64::new(0),
            total_deallocation_failures: AtomicU64::new(0),
            cumulative_allocation_time_ns: AtomicU64::new(0),
            cumulative_deallocation_time_ns: AtomicU64::new(0),
            performance_history: Arc::new(Mutex::new(Vec::new())),
            config: Arc::new(RwLock::new(PerformanceConfig::default())),
            start_time: now,
            last_snapshot_time: Arc::new(Mutex::new(now)),
            allocation_times_ns: Arc::new(Mutex::new(Vec::new())),
            deallocation_times_ns: Arc::new(Mutex::new(Vec::new())),
            performance_trends: Arc::new(Mutex::new(PerformanceTrends::default())),
        }
    }
}

