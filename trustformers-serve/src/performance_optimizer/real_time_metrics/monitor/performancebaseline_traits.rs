//! # PerformanceBaseline - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceBaseline`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::types::ConfidenceIntervals;
use super::types::*;
use chrono::Utc;

use super::types::{BaselineValidationStatus, PerformanceBaseline, VariabilityBounds};

impl Default for PerformanceBaseline {
    /// A baseline that has observed nothing yet.
    ///
    /// ## Changed in 0.2.1
    ///
    /// The default used to describe a fully established baseline that had never
    /// seen a sample: throughput 100.0, latency 50ms, CPU 0.5, memory 0.6,
    /// `sample_size`/`sample_count` 1000, `quality_score` 0.9,
    /// `stability_score` 0.8, `confidence_level` 0.95 and
    /// `validation_status: Valid`. `BaselineManager::new` installs this default,
    /// and `ParallelPerformanceMonitor::get_monitoring_status` publishes
    /// `get_validation_status()` -- so a freshly started monitor reported a
    /// *valid* baseline backed by a thousand imaginary samples, and
    /// `check_deviation` judged live metrics against 100.0 req/s.
    ///
    /// Zero samples, `Pending` status and zero confidence are what is actually
    /// true before anything is measured.
    fn default() -> Self {
        let now = Utc::now();
        Self {
            timestamp: now,
            baseline_throughput: 0.0,
            baseline_latency: Duration::ZERO,
            baseline_cpu: 0.0,
            baseline_memory: 0.0,
            variability_bounds: VariabilityBounds::default(),
            confidence_intervals: ConfidenceIntervals::default(),
            quality_score: 0.0,
            sample_size: 0,
            stability_score: 0.0,
            adaptation_rate: 0.1,
            version: 0,
            validation_status: BaselineValidationStatus::Pending,
            throughput_baseline: 0.0,
            latency_baseline: Duration::ZERO,
            cpu_baseline: 0.0,
            memory_baseline: 0.0,
            established_at: now,
            last_updated: now,
            sample_count: 0,
            confidence_level: 0.0,
        }
    }
}
