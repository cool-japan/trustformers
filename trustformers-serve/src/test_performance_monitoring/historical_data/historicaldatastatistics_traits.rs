//! # HistoricalDataStatistics - Trait Implementations
//!
//! This module contains trait implementations for `HistoricalDataStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Clone`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use super::types::HistoricalDataStatistics;

impl Clone for HistoricalDataStatistics {
    fn clone(&self) -> Self {
        Self {
            total_time_series: AtomicU64::new(self.total_time_series.load(Ordering::Relaxed)),
            total_data_points: AtomicU64::new(self.total_data_points.load(Ordering::Relaxed)),
            total_storage_bytes: AtomicU64::new(self.total_storage_bytes.load(Ordering::Relaxed)),
            compression_ratio: AtomicU64::new(self.compression_ratio.load(Ordering::Relaxed)),
            query_performance: Arc::clone(&self.query_performance),
            storage_efficiency: Arc::clone(&self.storage_efficiency),
            retention_metrics: Arc::clone(&self.retention_metrics),
        }
    }
}
