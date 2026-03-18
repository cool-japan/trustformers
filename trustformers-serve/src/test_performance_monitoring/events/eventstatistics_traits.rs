//! # EventStatistics - Trait Implementations
//!
//! This module contains trait implementations for `EventStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use parking_lot::RwLock as ParkingLotRwLock;
use std::sync::atomic::{AtomicU64, Ordering};

use super::types::EventStatistics;

impl Clone for EventStatistics {
    fn clone(&self) -> Self {
        Self {
            total_events_processed: AtomicU64::new(
                self.total_events_processed.load(Ordering::Relaxed),
            ),
            events_by_type: ParkingLotRwLock::new(self.events_by_type.read().clone()),
            events_by_severity: ParkingLotRwLock::new(self.events_by_severity.read().clone()),
            average_processing_time: AtomicU64::new(
                self.average_processing_time.load(Ordering::Relaxed),
            ),
            current_event_rate: AtomicU64::new(self.current_event_rate.load(Ordering::Relaxed)),
            peak_event_rate: AtomicU64::new(self.peak_event_rate.load(Ordering::Relaxed)),
            error_counts: ParkingLotRwLock::new(self.error_counts.read().clone()),
        }
    }
}
