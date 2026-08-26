//! Data Structure Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, Instant},
};

// Import common types
use super::common::AtomicF32;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// Import enums
use super::enums::TrendDirection;

// Import types from sibling modules
use super::metrics::QualityMetrics;
use super::statistics::{DistributionAnalysis, TrendAnalysis};

// =============================================================================
// DATA STRUCTURE TYPES
// =============================================================================

/// Basic data point structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

// DATA STRUCTURE TYPES
// =============================================================================

/// Timestamped metrics data point
///
/// Individual metrics measurement with precise timestamp and comprehensive
/// performance data for real-time analysis and historical tracking.
#[derive(Debug, Clone)]
pub struct TimestampedMetrics {
    /// Measurement timestamp
    pub timestamp: DateTime<Utc>,

    /// High-precision timestamp for sub-second accuracy
    pub precise_timestamp: Instant,

    /// Performance metrics
    pub metrics: RealTimeMetrics,

    /// System state snapshot
    pub system_state: SystemState,

    /// Measurement quality score
    pub quality_score: f32,

    /// Collection source
    pub source: String,

    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

impl TimestampedMetrics {
    /// Create new timestamped metrics
    pub fn new(metrics: RealTimeMetrics, system_state: SystemState, source: String) -> Self {
        Self {
            timestamp: Utc::now(),
            precise_timestamp: Instant::now(),
            metrics,
            system_state,
            quality_score: 1.0,
            source,
            metadata: HashMap::new(),
        }
    }

    /// Get age of metrics
    pub fn age(&self) -> Duration {
        self.precise_timestamp.elapsed()
    }

    /// Check if metrics are fresh
    pub fn is_fresh(&self, max_age: Duration) -> bool {
        self.age() <= max_age
    }
}

impl Default for TimestampedMetrics {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            precise_timestamp: Instant::now(),
            metrics: RealTimeMetrics::default(),
            system_state: SystemState::default(),
            quality_score: 1.0,
            source: String::new(),
            metadata: HashMap::new(),
        }
    }
}

// `CircularBuffer` was deleted from this module in 0.2.1. It was a shadow copy
// of `collector::types::CircularBuffer` -- the one that is actually used, held
// behind a `Mutex` by `RealTimeMetricsCollector` -- and nothing in the crate
// ever constructed this one. It was also unsound: `insert(&self, item: T)` cast
// the buffer's `*const` to `*mut` and wrote through it from a shared reference,
// with a comment conceding that "in a real implementation, we'd use proper
// atomic operations or locks". `real_time_metrics::CircularBuffer` now names
// the collector's implementation.

/// Buffer performance statistics
///
/// Statistics for monitoring circular buffer performance and optimization.
#[derive(Debug, Default)]
pub struct BufferStatistics {
    /// Total insertions
    pub total_insertions: AtomicU64,

    /// Buffer overwrites
    pub overwrites: AtomicU64,

    /// Average insertion time (nanoseconds)
    pub avg_insertion_time: AtomicF32,

    /// Memory usage (bytes)
    pub memory_usage: AtomicU64,
}

impl BufferStatistics {
    /// Get insertion rate
    pub fn insertion_rate(&self) -> f64 {
        let _total = self.total_insertions.load(Ordering::Acquire);
        let avg_time = self.avg_insertion_time.load(Ordering::Acquire);
        if avg_time > 0.0 {
            1_000_000_000.0 / avg_time as f64
        } else {
            0.0
        }
    }

    /// Get overwrite rate
    pub fn overwrite_rate(&self) -> f64 {
        let total = self.total_insertions.load(Ordering::Acquire);
        let overwrites = self.overwrites.load(Ordering::Acquire);
        if total > 0 {
            overwrites as f64 / total as f64
        } else {
            0.0
        }
    }
}

// `AggregationWindow` was deleted from this module in 0.2.1. Like
// `CircularBuffer` above it was a shadow copy -- `aggregator::types::AggregationWindow`
// is the one the aggregator actually builds -- and nothing constructed this one.
// Its `update_statistics` also published invented numbers: `min: 0.0`/`max: 0.0`
// under "would need to track actual min", `outlier_count: 0`,
// `efficiency_trend: Stable`, and a `QualityMetrics` whose consistency (0.9),
// timeliness (0.95) and outlier percentage (0.05) were constants labelled
// "Simplified". `real_time_metrics::AggregationWindow` now names the
// aggregator's implementation.

/// Comprehensive statistical calculations for windows
///
/// Advanced statistical analysis including descriptive statistics,
/// distribution analysis, and trend detection for optimization insights.
/// This is the canonical WindowStatistics type used throughout the real-time metrics system.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowStatistics {
    /// Number of data points
    pub count: usize,

    /// Timestamp of statistics calculation
    pub calculated_at: DateTime<Utc>,

    /// Mean value (aggregate across all metrics)
    pub mean: f64,

    /// Standard deviation (aggregate across all metrics)
    pub std_dev: f64,

    /// Minimum value (aggregate across all metrics)
    pub min: f64,

    /// Maximum value (aggregate across all metrics)
    pub max: f64,

    /// Number of outliers detected
    pub outlier_count: usize,

    /// Mean throughput
    pub mean_throughput: f64,

    /// Throughput standard deviation
    pub throughput_std_dev: f64,

    /// Mean latency
    pub mean_latency: Duration,

    /// Latency percentiles
    pub latency_percentiles: HashMap<u8, Duration>,

    /// Mean CPU utilization
    pub mean_cpu_utilization: f32,

    /// Mean memory utilization
    pub mean_memory_utilization: f32,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Trend analysis
    pub trend_analysis: TrendAnalysis,

    /// Distribution analysis
    pub distribution_analysis: DistributionAnalysis,

    /// Efficiency trend
    pub efficiency_trend: TrendDirection,

    /// Variability coefficient
    pub variability_coefficient: f32,
}

impl Default for WindowStatistics {
    fn default() -> Self {
        Self {
            count: 0,
            calculated_at: Utc::now(),
            mean: 0.0,
            std_dev: 0.0,
            min: 0.0,
            max: 0.0,
            outlier_count: 0,
            mean_throughput: 0.0,
            throughput_std_dev: 0.0,
            mean_latency: Duration::ZERO,
            latency_percentiles: HashMap::new(),
            mean_cpu_utilization: 0.0,
            mean_memory_utilization: 0.0,
            quality_metrics: QualityMetrics::default(),
            trend_analysis: TrendAnalysis::default(),
            distribution_analysis: DistributionAnalysis::default(),
            efficiency_trend: TrendDirection::Stable,
            variability_coefficient: 0.0,
        }
    }
}

/// Comprehensive throughput statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThroughputStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
    pub percentiles: HashMap<u8, f64>,
    pub variance: f64,
    pub skewness: f64,
    pub kurtosis: f64,
    pub coefficient_of_variation: f64,
}

/// Comprehensive latency statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyStatistics {
    pub mean: Duration,
    pub median: Duration,
    pub std_dev: Duration,
    pub min: Duration,
    pub max: Duration,
    pub percentiles: HashMap<u8, Duration>,
    pub variance: Duration,
    pub tail_latency: HashMap<String, Duration>,
}

impl Default for LatencyStatistics {
    fn default() -> Self {
        Self {
            mean: Duration::ZERO,
            median: Duration::ZERO,
            std_dev: Duration::ZERO,
            min: Duration::ZERO,
            max: Duration::ZERO,
            percentiles: HashMap::new(),
            variance: Duration::ZERO,
            tail_latency: HashMap::new(),
        }
    }
}

/// Resource utilization statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UtilizationStatistics {
    pub mean: f32,
    pub median: f32,
    pub std_dev: f32,
    pub min: f32,
    pub max: f32,
    pub percentiles: HashMap<u8, f32>,
    pub peak_usage: f32,
    pub utilization_efficiency: f32,
}

/// Efficiency metrics for performance analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EfficiencyMetrics {
    pub throughput_per_cpu: f64,
    pub throughput_per_memory: f64,
    pub resource_efficiency_score: f32,
    pub performance_efficiency_index: f32,
    pub energy_efficiency_estimate: f32,
}

/// Variability measures for stability analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VariabilityMeasures {
    pub coefficient_of_variation: f32,
    pub range_to_mean_ratio: f32,
    pub interquartile_range: f64,
    pub mean_absolute_deviation: f64,
    pub stability_index: f32,
}

// =============================================================================
