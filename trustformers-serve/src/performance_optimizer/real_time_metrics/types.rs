//! Comprehensive Types Module for Real-Time Metrics System
//!
//! This module contains all 147+ types extracted from the real-time metrics system,
//! organized into logical categories for optimal maintainability and comprehension.
//! Each type includes comprehensive documentation and appropriate traits and implementations.
//!
//! ## Type Categories
//!
//! - **Core Configuration Types**: System-wide configuration structures
//! - **Data Structure Types**: Core data containers and buffers
//! - **Metrics and Analysis Types**: Performance measurement and analysis structures
//! - **Monitoring Types**: Real-time monitoring and thread management
//! - **Alerting Types**: Alert generation and threshold management
//! - **Enums**: Classification and state enumerations
//! - **Error Types**: Comprehensive error handling
//! - **Trait Definitions**: Key interfaces and abstractions
//! - **Additional Support Types**: Utility and helper structures

use anyhow::Result;
use chrono::{DateTime, Utc};
use num_cpus;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, AtomicU32, AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};

/// Atomic wrapper for f32 values using u32 bit manipulation
#[derive(Debug)]
pub struct AtomicF32 {
    inner: AtomicU32,
}

impl AtomicF32 {
    pub fn new(value: f32) -> Self {
        Self {
            inner: AtomicU32::new(value.to_bits()),
        }
    }

    pub fn load(&self, order: Ordering) -> f32 {
        f32::from_bits(self.inner.load(order))
    }

    pub fn store(&self, value: f32, order: Ordering) {
        self.inner.store(value.to_bits(), order)
    }

    pub fn swap(&self, value: f32, order: Ordering) -> f32 {
        f32::from_bits(self.inner.swap(value.to_bits(), order))
    }

    pub fn compare_exchange(
        &self,
        current: f32,
        new: f32,
        success: Ordering,
        failure: Ordering,
    ) -> Result<f32, f32> {
        match self.inner.compare_exchange(current.to_bits(), new.to_bits(), success, failure) {
            Ok(old) => Ok(f32::from_bits(old)),
            Err(old) => Err(f32::from_bits(old)),
        }
    }

    /// Atomically adds a value to the current value and returns the previous value
    pub fn fetch_add(&self, value: f32, order: Ordering) -> f32 {
        loop {
            let current = self.load(Ordering::Relaxed);
            let new = current + value;
            match self.compare_exchange(current, new, order, Ordering::Relaxed) {
                Ok(prev) => return prev,
                Err(_) => continue, // Retry if compare_exchange failed
            }
        }
    }

    /// Atomically subtracts a value from the current value and returns the previous value
    pub fn fetch_sub(&self, value: f32, order: Ordering) -> f32 {
        self.fetch_add(-value, order)
    }
}

impl Default for AtomicF32 {
    fn default() -> Self {
        Self::new(0.0)
    }
}
use thiserror;
use tokio::task::JoinHandle;

// Import types from the parent types module
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RecommendedAction, SystemState, TestCharacteristics,
};

// Re-export RealTimeMetrics from parent module for local use
pub use super::super::types::RealTimeMetrics;

// Define missing types that are referenced but not found
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FeedbackType {
    Performance,
    Resource,
    Quality,
    Error,
    Warning,
    Success,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

// Import SeverityLevel from pattern engine to avoid conflicts
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

// Additional missing types that are frequently referenced
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CleanupPriority {
    Low,
    Normal,
    High,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub metadata: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RiskType {
    Performance,
    Security,
    Resource,
    Operational,
    Financial,
    Technical,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MemoryType {
    Heap,
    Stack,
    Static,
    Shared,
    GPU,
    Cache,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ImpactArea {
    Performance,
    Reliability,
    Security,
    UserExperience,
    ResourceUtilization,
    Cost,
}

// =============================================================================
// CORE CONFIGURATION TYPES
// =============================================================================

/// Configuration for metrics collection system
///
/// Comprehensive configuration for real-time metrics collection including
/// sampling rates, buffer management, and processing options for optimal
/// performance with minimal system overhead.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsCollectionConfig {
    /// Base collection interval
    pub base_interval: Duration,

    /// Minimum collection interval
    pub min_interval: Duration,

    /// Maximum collection interval
    pub max_interval: Duration,

    /// History buffer size
    pub history_buffer_size: usize,

    /// Adaptive sampling enabled
    pub adaptive_sampling: bool,

    /// High precision mode
    pub high_precision_mode: bool,

    /// Batch processing size
    pub batch_size: usize,

    /// Compression enabled
    pub compression_enabled: bool,

    /// Collection timeout
    pub collection_timeout: Duration,

    /// Resource monitoring enabled
    pub resource_monitoring: bool,

    /// Custom metrics enabled
    pub custom_metrics: bool,

    /// Stream publishing enabled
    pub stream_publishing: bool,
}

impl Default for MetricsCollectionConfig {
    fn default() -> Self {
        Self {
            base_interval: Duration::from_millis(100),
            min_interval: Duration::from_millis(10),
            max_interval: Duration::from_secs(1),
            history_buffer_size: 10000,
            adaptive_sampling: true,
            high_precision_mode: false,
            batch_size: 100,
            compression_enabled: false,
            collection_timeout: Duration::from_secs(5),
            resource_monitoring: true,
            custom_metrics: false,
            stream_publishing: true,
        }
    }
}

/// Configuration for parallel performance monitoring
///
/// Advanced configuration for parallel performance monitoring including
/// thread allocation, monitoring strategies, and analysis parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitorConfiguration {
    /// Number of monitor threads
    pub thread_count: usize,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Data aggregation window
    pub aggregation_window: Duration,

    /// Trend analysis window
    pub trend_window: Duration,

    /// Anomaly detection sensitivity
    pub anomaly_sensitivity: f32,

    /// Baseline update interval
    pub baseline_update_interval: Duration,

    /// Event broadcasting enabled
    pub event_broadcasting: bool,

    /// Performance baseline enabled
    pub baseline_enabled: bool,

    /// Advanced analytics enabled
    pub advanced_analytics: bool,

    /// Real-time processing enabled
    pub realtime_processing: bool,
}

impl Default for MonitorConfiguration {
    fn default() -> Self {
        Self {
            thread_count: num_cpus::get(),
            monitoring_interval: Duration::from_millis(50),
            aggregation_window: Duration::from_secs(60),
            trend_window: Duration::from_secs(300),
            anomaly_sensitivity: 0.8,
            baseline_update_interval: Duration::from_secs(3600),
            event_broadcasting: true,
            baseline_enabled: true,
            advanced_analytics: true,
            realtime_processing: true,
        }
    }
}

/// Configuration for data aggregation operations
///
/// Detailed configuration for real-time data aggregation including window
/// specifications, statistical analysis, and processing optimizations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation windows
    pub windows: Vec<Duration>,

    /// Statistical analysis enabled
    pub statistical_analysis: bool,

    /// Trend detection enabled
    pub trend_detection: bool,

    /// Outlier removal enabled
    pub outlier_removal: bool,

    /// Cache size limit
    pub cache_size_limit: usize,

    /// Processing batch size
    pub processing_batch_size: usize,

    /// Parallel processing enabled
    pub parallel_processing: bool,

    /// Compression level
    pub compression_level: u8,

    /// Quality control enabled
    pub quality_control: bool,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            windows: vec![
                Duration::from_secs(5),
                Duration::from_secs(30),
                Duration::from_secs(300),
                Duration::from_secs(3600),
            ],
            statistical_analysis: true,
            trend_detection: true,
            outlier_removal: true,
            cache_size_limit: 50000,
            processing_batch_size: 500,
            parallel_processing: true,
            compression_level: 3,
            quality_control: true,
        }
    }
}

/// Configuration for optimization engine behavior
///
/// Comprehensive configuration for the live optimization engine including
/// algorithm selection, recommendation generation, and confidence scoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationEngineConfig {
    /// Recommendation generation interval
    pub generation_interval: Duration,

    /// Minimum confidence threshold
    pub min_confidence_threshold: f32,

    /// Maximum recommendations per interval
    pub max_recommendations: usize,

    /// Analysis window size
    pub analysis_window: Duration,

    /// Prediction horizon
    pub prediction_horizon: Duration,

    /// Conservative mode enabled
    pub conservative_mode: bool,

    /// Machine learning enabled
    pub ml_enabled: bool,

    /// Real-time adaptation enabled
    pub realtime_adaptation: bool,

    /// Multi-objective optimization
    pub multi_objective: bool,
}

impl Default for OptimizationEngineConfig {
    fn default() -> Self {
        Self {
            generation_interval: Duration::from_secs(30),
            min_confidence_threshold: 0.7,
            max_recommendations: 10,
            analysis_window: Duration::from_secs(300),
            prediction_horizon: Duration::from_secs(600),
            conservative_mode: false,
            ml_enabled: true,
            realtime_adaptation: true,
            multi_objective: true,
        }
    }
}

/// Threshold configuration for performance monitoring
///
/// Comprehensive threshold configuration with multiple threshold levels,
/// adaptive capabilities, and intelligent alerting policies.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    /// Threshold name
    pub name: String,

    /// Metric being monitored
    pub metric: String,

    /// Warning threshold
    pub warning_threshold: f64,

    /// Critical threshold
    pub critical_threshold: f64,

    /// Threshold direction (above/below)
    pub direction: ThresholdDirection,

    /// Adaptive threshold enabled
    pub adaptive: bool,

    /// Evaluation window
    pub evaluation_window: Duration,

    /// Minimum trigger count
    pub min_trigger_count: usize,

    /// Alert cooldown period
    pub cooldown_period: Duration,

    /// Escalation policy
    pub escalation_policy: String,
}

impl Default for ThresholdConfig {
    fn default() -> Self {
        Self {
            name: "default_threshold".to_string(),
            metric: "throughput".to_string(),
            warning_threshold: 80.0,
            critical_threshold: 95.0,
            direction: ThresholdDirection::Above,
            adaptive: true,
            evaluation_window: Duration::from_secs(60),
            min_trigger_count: 3,
            cooldown_period: Duration::from_secs(300),
            escalation_policy: "standard".to_string(),
        }
    }
}

// =============================================================================
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

/// Circular buffer for efficient metrics storage
///
/// High-performance circular buffer optimized for real-time metrics storage
/// with constant-time insertion and efficient memory management.
#[derive(Debug)]
pub struct CircularBuffer<T> {
    /// Buffer data
    buffer: Vec<Option<T>>,

    /// Current write position
    write_pos: AtomicUsize,

    /// Current size
    size: AtomicUsize,

    /// Maximum capacity
    capacity: usize,

    /// Buffer statistics
    stats: BufferStatistics,
}

impl<T> CircularBuffer<T> {
    /// Create new circular buffer
    pub fn new(capacity: usize) -> Self {
        Self {
            buffer: (0..capacity).map(|_| None).collect(),
            write_pos: AtomicUsize::new(0),
            size: AtomicUsize::new(0),
            capacity,
            stats: BufferStatistics::default(),
        }
    }

    /// Insert item into buffer
    pub fn insert(&self, item: T) {
        let start_time = Instant::now();

        let pos = self.write_pos.fetch_add(1, Ordering::AcqRel) % self.capacity;

        // This is unsafe but necessary for performance - we need proper synchronization
        // In a real implementation, we'd use proper atomic operations or locks
        unsafe {
            let buffer_ptr = self.buffer.as_ptr() as *mut Option<T>;
            let slot = buffer_ptr.add(pos);
            std::ptr::write(slot, Some(item));
        }

        let current_size = self.size.load(Ordering::Acquire);
        if current_size < self.capacity {
            self.size.fetch_add(1, Ordering::AcqRel);
        } else {
            self.stats.overwrites.fetch_add(1, Ordering::AcqRel);
        }

        self.stats.total_insertions.fetch_add(1, Ordering::AcqRel);

        let insertion_time = start_time.elapsed().as_nanos() as f32;
        // Update running average (simplified)
        let current_avg = self.stats.avg_insertion_time.load(Ordering::Acquire);
        let new_avg = (current_avg * 0.9) + (insertion_time * 0.1);
        self.stats.avg_insertion_time.store(new_avg, Ordering::Release);
    }

    /// Get current buffer size
    pub fn size(&self) -> usize {
        self.size.load(Ordering::Acquire)
    }

    /// Get buffer capacity
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Check if buffer is full
    pub fn is_full(&self) -> bool {
        self.size.load(Ordering::Acquire) >= self.capacity
    }

    /// Get buffer statistics
    pub fn stats(&self) -> &BufferStatistics {
        &self.stats
    }
}

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

/// Aggregation window for time-based analysis
///
/// Time-based aggregation window with statistical analysis and trend detection
/// capabilities for real-time performance insights.
#[derive(Debug)]
pub struct AggregationWindow {
    /// Window duration
    pub duration: Duration,

    /// Data points in window
    pub data_points: VecDeque<TimestampedMetrics>,

    /// Window statistics
    pub statistics: WindowStatistics,

    /// Last update timestamp
    pub last_update: DateTime<Utc>,

    /// Window full indicator
    pub is_full: bool,
}

impl AggregationWindow {
    /// Create new aggregation window
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            data_points: VecDeque::new(),
            statistics: WindowStatistics::default(),
            last_update: Utc::now(),
            is_full: false,
        }
    }

    /// Add data point to window
    pub fn add_data_point(&mut self, point: TimestampedMetrics) {
        let now = Utc::now();
        let cutoff = now
            - chrono::Duration::from_std(self.duration)
                .unwrap_or_else(|_| chrono::Duration::seconds(60));

        // Remove old data points
        while let Some(front) = self.data_points.front() {
            if front.timestamp < cutoff {
                self.data_points.pop_front();
            } else {
                break;
            }
        }

        // Add new data point
        self.data_points.push_back(point);
        self.last_update = now;

        // Update statistics
        self.update_statistics();

        // Check if window is full (has data across the entire duration)
        if let (Some(front), Some(back)) = (self.data_points.front(), self.data_points.back()) {
            let window_span = back.timestamp - front.timestamp;
            let duration_chrono = chrono::Duration::from_std(self.duration)
                .unwrap_or_else(|_| chrono::Duration::seconds(60));
            self.is_full = window_span >= duration_chrono * 8 / 10; // 80% coverage
        }
    }

    /// Update window statistics
    fn update_statistics(&mut self) {
        if self.data_points.is_empty() {
            return;
        }

        // Calculate basic statistics
        let count = self.data_points.len();
        let throughputs: Vec<f64> = self.data_points.iter().map(|p| p.metrics.throughput).collect();

        let mean_throughput = throughputs.iter().sum::<f64>() / count as f64;
        let throughput_variance: f64 =
            throughputs.iter().map(|t| (t - mean_throughput).powi(2)).sum::<f64>() / count as f64;
        let throughput_std_dev = throughput_variance.sqrt();

        // Calculate latency statistics
        let latencies: Vec<Duration> = self.data_points.iter().map(|p| p.metrics.latency).collect();

        let mean_latency_nanos: f64 =
            latencies.iter().map(|l| l.as_nanos() as f64).sum::<f64>() / count as f64;
        let mean_latency = Duration::from_nanos(mean_latency_nanos as u64);

        // Calculate CPU and memory utilization
        let mean_cpu: f32 =
            self.data_points.iter().map(|p| p.metrics.cpu_utilization).sum::<f32>() / count as f32;

        let mean_memory: f32 =
            self.data_points.iter().map(|p| p.metrics.memory_utilization).sum::<f32>()
                / count as f32;

        // Calculate percentiles (simplified)
        let mut sorted_latencies = latencies;
        sorted_latencies.sort();
        let mut percentiles = HashMap::new();
        if !sorted_latencies.is_empty() {
            percentiles.insert(50, sorted_latencies[count * 50 / 100]);
            percentiles.insert(90, sorted_latencies[count * 90 / 100]);
            percentiles.insert(95, sorted_latencies[count * 95 / 100]);
            percentiles.insert(99, sorted_latencies[count * 99 / 100]);
        }

        // Update statistics
        self.statistics = WindowStatistics {
            count,
            calculated_at: Utc::now(),
            mean: mean_throughput,
            std_dev: throughput_std_dev,
            min: 0.0, // Would need to track actual min
            max: 0.0, // Would need to track actual max
            outlier_count: 0,
            mean_throughput,
            throughput_std_dev,
            mean_latency,
            latency_percentiles: percentiles,
            mean_cpu_utilization: mean_cpu,
            mean_memory_utilization: mean_memory,
            quality_metrics: self.calculate_quality_metrics(),
            trend_analysis: TrendAnalysis::default(),
            distribution_analysis: DistributionAnalysis::default(),
            efficiency_trend: TrendDirection::Stable, // Simplified
            variability_coefficient: if mean_throughput > 0.0 {
                (throughput_std_dev / mean_throughput) as f32
            } else {
                0.0
            },
        };
    }

    /// Calculate quality metrics for the window
    fn calculate_quality_metrics(&self) -> QualityMetrics {
        if self.data_points.is_empty() {
            return QualityMetrics::default();
        }

        let count = self.data_points.len() as f32;
        let expected_points = (self.duration.as_secs() * 10) as f32; // Assuming 10Hz base rate
        let completeness_score = (count / expected_points).min(1.0);

        // Calculate accuracy based on quality scores
        let accuracy_score: f32 =
            self.data_points.iter().map(|p| p.quality_score).sum::<f32>() / count;

        // Simplified quality metrics
        QualityMetrics {
            overall_score: (completeness_score + accuracy_score) / 2.0,
            completeness_score,
            accuracy_score,
            consistency_score: 0.9,   // Simplified
            timeliness_score: 0.95,   // Simplified
            outlier_percentage: 0.05, // Simplified
            missing_data_percentage: (1.0 - completeness_score) * 100.0,
            quality_trend: TrendDirection::Stable,
        }
    }

    /// Get window span
    pub fn get_span(&self) -> Option<Duration> {
        if let (Some(front), Some(back)) = (self.data_points.front(), self.data_points.back()) {
            let span = back.timestamp - front.timestamp;
            span.to_std().ok()
        } else {
            None
        }
    }

    /// Check if window has enough data
    pub fn has_sufficient_data(&self) -> bool {
        self.data_points.len() >= 10 && self.is_full
    }
}

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
// METRICS AND ANALYSIS TYPES
// =============================================================================

/// Data quality assessment metrics
///
/// Comprehensive assessment of data quality including completeness, accuracy,
/// consistency, timeliness, and trend analysis for informed decision making.
/// This is the canonical QualityMetrics used in WindowStatistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f32,

    /// Data completeness score (0.0 to 1.0)
    pub completeness_score: f32,

    /// Data accuracy score (0.0 to 1.0)
    pub accuracy_score: f32,

    /// Data consistency score (0.0 to 1.0)
    pub consistency_score: f32,

    /// Data timeliness score (0.0 to 1.0)
    pub timeliness_score: f32,

    /// Outlier percentage
    pub outlier_percentage: f32,

    /// Missing data percentage
    pub missing_data_percentage: f32,

    /// Quality trend direction
    pub quality_trend: TrendDirection,
}

impl Default for QualityMetrics {
    fn default() -> Self {
        Self {
            overall_score: 1.0,
            completeness_score: 1.0,
            accuracy_score: 1.0,
            consistency_score: 1.0,
            timeliness_score: 1.0,
            outlier_percentage: 0.0,
            missing_data_percentage: 0.0,
            quality_trend: TrendDirection::Stable,
        }
    }
}

/// Aggregation result with comprehensive analysis
///
/// Complete aggregation result including statistical analysis, trends,
/// insights, and recommendations based on real-time data processing.
#[derive(Debug, Clone)]
pub struct AggregationResult {
    /// Aggregation timestamp
    pub timestamp: DateTime<Utc>,

    /// Aggregation window
    pub window: Duration,

    /// Statistical summary
    pub statistics: WindowStatistics,

    /// Trend analysis
    pub trends: Vec<PerformanceTrend>,

    /// Performance insights
    pub insights: Vec<PerformanceInsight>,

    /// Recommendations
    pub recommendations: Vec<RecommendedAction>,

    /// Confidence score
    pub confidence: f32,

    /// Processing metadata
    pub metadata: HashMap<String, String>,

    /// Window duration
    pub window_duration: Duration,

    /// Data point count
    pub data_point_count: usize,

    /// Quality score
    pub quality_score: f32,

    /// Trend analysis (detailed)
    pub trend_analysis: String,
}

impl AggregationResult {
    /// Create new aggregation result
    pub fn new(window: Duration, statistics: WindowStatistics) -> Self {
        Self {
            timestamp: Utc::now(),
            window,
            statistics,
            trends: Vec::new(),
            insights: Vec::new(),
            recommendations: Vec::new(),
            confidence: 1.0,
            metadata: HashMap::new(),
            window_duration: window,
            data_point_count: 0, // Will be updated as data is aggregated
            quality_score: 1.0,  // High quality by default
            trend_analysis: String::new(), // Will be populated by trend analysis
        }
    }

    /// Add insight to result
    pub fn add_insight(&mut self, insight: PerformanceInsight) {
        self.insights.push(insight);
    }

    /// Add recommendation to result
    pub fn add_recommendation(&mut self, recommendation: RecommendedAction) {
        self.recommendations.push(recommendation);
    }

    /// Check if result has critical insights
    pub fn has_critical_insights(&self) -> bool {
        self.insights.iter().any(|i| i.severity == SeverityLevel::Critical)
    }
}

/// Performance insight from real-time analysis
///
/// Actionable performance insight derived from real-time data analysis
/// with severity assessment and recommended actions.
#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    /// Insight type
    pub insight_type: InsightType,

    /// Insight description
    pub description: String,

    /// Severity level
    pub severity: SeverityLevel,

    /// Confidence score
    pub confidence: f32,

    /// Supporting data
    pub supporting_data: HashMap<String, f64>,

    /// Recommended actions
    pub actions: Vec<RecommendedAction>,

    /// Impact assessment
    pub impact: ImpactAssessment,
}

impl PerformanceInsight {
    /// Create new performance insight
    pub fn new(insight_type: InsightType, description: String, severity: SeverityLevel) -> Self {
        Self {
            insight_type,
            description,
            severity,
            confidence: 0.0,
            supporting_data: HashMap::new(),
            actions: Vec::new(),
            impact: ImpactAssessment::default(),
        }
    }

    /// Add supporting data
    pub fn add_data(&mut self, key: String, value: f64) {
        self.supporting_data.insert(key, value);
    }

    /// Add recommended action
    pub fn add_action(&mut self, action: RecommendedAction) {
        self.actions.push(action);
    }

    /// Check if insight requires immediate attention
    pub fn requires_immediate_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Impact assessment for insights and recommendations
///
/// Comprehensive assessment of potential impact including performance,
/// resources, costs, and risks for informed decision making.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAssessment {
    /// Performance impact estimate
    pub performance_impact: f32,

    /// Resource impact estimate
    pub resource_impact: f32,

    /// Implementation complexity
    pub complexity: f32,

    /// Risk assessment
    pub risk_level: f32,

    /// Estimated benefit
    pub estimated_benefit: f32,

    /// Time to implementation
    pub implementation_time: Duration,
}

impl Default for ImpactAssessment {
    fn default() -> Self {
        Self {
            performance_impact: 0.0,
            resource_impact: 0.0,
            complexity: 0.5,
            risk_level: 0.3,
            estimated_benefit: 0.0,
            implementation_time: Duration::from_secs(300),
        }
    }
}

impl ImpactAssessment {
    /// Calculate overall impact score
    pub fn overall_score(&self) -> f32 {
        let benefit_score = self.estimated_benefit;
        let cost_score = (self.complexity + self.risk_level + self.resource_impact.abs()) / 3.0;
        (benefit_score - cost_score).max(0.0)
    }

    /// Check if implementation is recommended
    pub fn is_recommended(&self) -> bool {
        self.overall_score() > 0.5 && self.risk_level < 0.7
    }
}

/// Performance baseline for comparison
///
/// Established performance baseline with statistical characteristics
/// for detecting deviations and performance changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline timestamp
    pub timestamp: DateTime<Utc>,

    /// Baseline throughput
    pub baseline_throughput: f64,

    /// Baseline latency
    pub baseline_latency: Duration,

    /// Baseline CPU utilization
    pub baseline_cpu: f32,

    /// Baseline memory utilization
    pub baseline_memory: f32,

    /// Baseline variability
    pub variability_bounds: VariabilityBounds,

    /// Confidence intervals
    pub confidence_intervals: ConfidenceIntervals,

    /// Baseline quality score
    pub quality_score: f32,
}

impl PerformanceBaseline {
    /// Create new baseline from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        Self {
            timestamp: Utc::now(),
            baseline_throughput: stats.mean_throughput,
            baseline_latency: stats.mean_latency,
            baseline_cpu: stats.mean_cpu_utilization,
            baseline_memory: stats.mean_memory_utilization,
            variability_bounds: VariabilityBounds::from_statistics(stats),
            confidence_intervals: ConfidenceIntervals::from_statistics(stats),
            quality_score: stats.quality_metrics.overall_score,
        }
    }

    /// Check if current metrics deviate from baseline
    pub fn check_deviation(&self, metrics: &RealTimeMetrics) -> bool {
        let throughput_ok = self.variability_bounds.throughput_bounds.0 <= metrics.throughput
            && metrics.throughput <= self.variability_bounds.throughput_bounds.1;

        let cpu_ok = self.variability_bounds.cpu_bounds.0 <= metrics.cpu_utilization
            && metrics.cpu_utilization <= self.variability_bounds.cpu_bounds.1;

        let memory_ok = self.variability_bounds.memory_bounds.0 <= metrics.memory_utilization
            && metrics.memory_utilization <= self.variability_bounds.memory_bounds.1;

        !(throughput_ok && cpu_ok && memory_ok)
    }

    /// Get baseline age
    pub fn age(&self) -> Duration {
        let now = Utc::now();
        (now - self.timestamp).to_std().unwrap_or(Duration::from_secs(0))
    }

    /// Check if baseline needs update
    pub fn needs_update(&self, max_age: Duration) -> bool {
        self.age() > max_age
    }
}

/// Variability bounds for performance baseline
///
/// Statistical bounds defining normal variability ranges for performance
/// metrics to distinguish normal fluctuations from significant changes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VariabilityBounds {
    /// Throughput bounds (min, max)
    pub throughput_bounds: (f64, f64),

    /// Latency bounds (min, max)
    pub latency_bounds: (Duration, Duration),

    /// CPU utilization bounds (min, max)
    pub cpu_bounds: (f32, f32),

    /// Memory utilization bounds (min, max)
    pub memory_bounds: (f32, f32),

    /// Efficiency bounds (min, max)
    pub efficiency_bounds: (f32, f32),
}

impl VariabilityBounds {
    /// Create bounds from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        let throughput_margin = stats.throughput_std_dev * 2.0; // 2 sigma
        let cpu_margin = stats.mean_cpu_utilization * 0.1; // 10% margin
        let memory_margin = stats.mean_memory_utilization * 0.1; // 10% margin
        let latency_margin = stats.mean_latency / 10; // 10% margin

        Self {
            throughput_bounds: (
                (stats.mean_throughput - throughput_margin).max(0.0),
                stats.mean_throughput + throughput_margin,
            ),
            latency_bounds: (
                stats.mean_latency.saturating_sub(latency_margin),
                stats.mean_latency + latency_margin,
            ),
            cpu_bounds: (
                (stats.mean_cpu_utilization - cpu_margin).max(0.0),
                (stats.mean_cpu_utilization + cpu_margin).min(100.0),
            ),
            memory_bounds: (
                (stats.mean_memory_utilization - memory_margin).max(0.0),
                (stats.mean_memory_utilization + memory_margin).min(100.0),
            ),
            efficiency_bounds: (0.5, 1.0), // Simplified
        }
    }
}

/// Confidence intervals for baseline metrics
///
/// Statistical confidence intervals for baseline performance metrics
/// to support reliable anomaly detection and performance comparison.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConfidenceIntervals {
    /// Confidence level (e.g., 0.95 for 95%)
    pub confidence_level: f32,

    /// Throughput confidence interval
    pub throughput_interval: (f64, f64),

    /// Latency confidence interval
    pub latency_interval: (Duration, Duration),

    /// CPU utilization confidence interval
    pub cpu_interval: (f32, f32),

    /// Memory utilization confidence interval
    pub memory_interval: (f32, f32),

    /// Network throughput confidence interval
    pub network_interval: (f64, f64),

    /// I/O operations confidence interval
    pub io_interval: (f64, f64),

    /// Response time confidence interval
    pub response_time_interval: (Duration, Duration),

    /// Error rate confidence interval
    pub error_rate_interval: (f32, f32),

    /// Statistical method used for calculation
    pub method: ConfidenceMethod,

    /// Mean lower bound
    pub mean_lower: f64,

    /// Mean upper bound
    pub mean_upper: f64,

    /// Variance lower bound
    pub variance_lower: f64,

    /// Variance upper bound
    pub variance_upper: f64,
}

impl ConfidenceIntervals {
    /// Create confidence intervals from window statistics
    pub fn from_statistics(stats: &WindowStatistics) -> Self {
        // Simplified confidence interval calculation (assuming normal distribution)
        let confidence_level = 95.0;
        let z_score = 1.96; // for 95% confidence

        let throughput_margin = (stats.throughput_std_dev / (stats.count as f64).sqrt()) * z_score;
        let cpu_margin = stats.mean_cpu_utilization * 0.05; // Simplified
        let memory_margin = stats.mean_memory_utilization * 0.05; // Simplified
        let latency_margin = stats.mean_latency / 20; // Simplified

        Self {
            confidence_level,
            throughput_interval: (
                stats.mean_throughput - throughput_margin,
                stats.mean_throughput + throughput_margin,
            ),
            latency_interval: (
                stats.mean_latency.saturating_sub(latency_margin),
                stats.mean_latency + latency_margin,
            ),
            cpu_interval: (
                (stats.mean_cpu_utilization - cpu_margin).max(0.0),
                (stats.mean_cpu_utilization + cpu_margin).min(100.0),
            ),
            memory_interval: (
                (stats.mean_memory_utilization - memory_margin).max(0.0),
                (stats.mean_memory_utilization + memory_margin).min(100.0),
            ),
            network_interval: (2_000_000.0, 8_000_000.0), // Simplified default
            io_interval: (200.0, 800.0),                  // Simplified default
            response_time_interval: (Duration::from_millis(15), Duration::from_millis(85)),
            error_rate_interval: (0.0, 3.0),
            method: ConfidenceMethod::TDistribution,
            mean_lower: stats.mean_throughput - throughput_margin,
            mean_upper: stats.mean_throughput + throughput_margin,
            variance_lower: (stats.throughput_std_dev * stats.throughput_std_dev * 0.8),
            variance_upper: (stats.throughput_std_dev * stats.throughput_std_dev * 1.2),
        }
    }
}

impl Default for ConfidenceIntervals {
    fn default() -> Self {
        Self {
            confidence_level: 0.95,
            throughput_interval: (90.0, 110.0),
            latency_interval: (Duration::from_millis(45), Duration::from_millis(55)),
            cpu_interval: (0.35, 0.65),
            memory_interval: (0.55, 0.75),
            network_interval: (2_000_000.0, 8_000_000.0), // 2MB/s to 8MB/s
            io_interval: (200.0, 800.0),                  // 200 to 800 IOPS
            response_time_interval: (Duration::from_millis(15), Duration::from_millis(85)),
            error_rate_interval: (0.0, 3.0), // 0% to 3%
            method: ConfidenceMethod::TDistribution,
            mean_lower: 90.0,
            mean_upper: 110.0,
            variance_lower: 1.0,
            variance_upper: 5.0,
        }
    }
}

// =============================================================================
// MONITORING TYPES
// =============================================================================

/// Monitor thread for parallel performance monitoring
///
/// Individual monitoring thread responsible for specific aspects of performance
/// monitoring with dedicated responsibilities and optimized data collection.
#[derive(Debug)]
pub struct MonitorThread {
    /// Thread identifier
    pub id: String,

    /// Thread handle
    pub handle: JoinHandle<()>,

    /// Monitoring scope
    pub scope: MonitoringScope,

    /// Thread statistics
    pub stats: ThreadStatistics,

    /// Last activity timestamp
    pub last_activity: Arc<RwLock<DateTime<Utc>>>,

    /// Thread health status
    pub health_status: Arc<AtomicBool>,
}

impl MonitorThread {
    /// Create new monitor thread
    pub fn new(id: String, handle: JoinHandle<()>, scope: MonitoringScope) -> Self {
        Self {
            id,
            handle,
            scope,
            stats: ThreadStatistics::default(),
            last_activity: Arc::new(RwLock::new(Utc::now())),
            health_status: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Update last activity
    pub fn update_activity(&self) {
        *self.last_activity.write() = Utc::now();
    }

    /// Check if thread is healthy
    pub fn is_healthy(&self) -> bool {
        self.health_status.load(Ordering::Acquire)
    }

    /// Get time since last activity
    pub fn time_since_activity(&self) -> Duration {
        let now = Utc::now();
        let last_activity = *self.last_activity.read();
        (now - last_activity).to_std().unwrap_or(Duration::from_secs(0))
    }
}

/// Thread performance statistics
///
/// Performance statistics for individual monitoring threads including
/// collection rates, processing times, and resource utilization.
#[derive(Debug, Default)]
pub struct ThreadStatistics {
    /// Data points collected
    pub data_points_collected: AtomicU64,

    /// Average collection time (nanoseconds)
    pub avg_collection_time: AtomicF32,

    /// Processing errors
    pub processing_errors: AtomicU64,

    /// Thread CPU usage
    pub cpu_usage: AtomicF32,

    /// Thread memory usage (bytes)
    pub memory_usage: AtomicU64,
}

impl ThreadStatistics {
    /// Get collection rate (points per second)
    pub fn collection_rate(&self) -> f64 {
        let _total_points = self.data_points_collected.load(Ordering::Acquire);
        let avg_time_nanos = self.avg_collection_time.load(Ordering::Acquire);

        if avg_time_nanos > 0.0 {
            1_000_000_000.0 / avg_time_nanos as f64
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        let total_points = self.data_points_collected.load(Ordering::Acquire);
        let errors = self.processing_errors.load(Ordering::Acquire);

        if total_points > 0 {
            errors as f64 / total_points as f64
        } else {
            0.0
        }
    }

    /// Update collection time
    pub fn update_collection_time(&self, time_nanos: f32) {
        let current_avg = self.avg_collection_time.load(Ordering::Acquire);
        let new_avg = if current_avg == 0.0 {
            time_nanos
        } else {
            (current_avg * 0.9) + (time_nanos * 0.1) // Exponential moving average
        };
        self.avg_collection_time.store(new_avg, Ordering::Release);
    }
}

/// Monitoring event for system communication
///
/// Event structure for communicating monitoring information, alerts,
/// and system status updates across monitoring components.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,

    /// Event type
    pub event_type: MonitoringEventType,

    /// Event source
    pub source: String,

    /// Event data
    pub data: HashMap<String, String>,

    /// Event severity
    pub severity: SeverityLevel,

    /// Event metadata
    pub metadata: HashMap<String, String>,
}

impl MonitoringEvent {
    /// Create new monitoring event
    pub fn new(event_type: MonitoringEventType, source: String, severity: SeverityLevel) -> Self {
        Self {
            timestamp: Utc::now(),
            event_type,
            source,
            data: HashMap::new(),
            severity,
            metadata: HashMap::new(),
        }
    }

    /// Add data to event
    pub fn add_data(&mut self, key: String, value: String) {
        self.data.insert(key, value);
    }

    /// Add metadata to event
    pub fn add_metadata(&mut self, key: String, value: String) {
        self.metadata.insert(key, value);
    }

    /// Check if event requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

// =============================================================================
// ALERTING TYPES
// =============================================================================

/// Alert event for threshold violations
///
/// Comprehensive alert event with detailed information about threshold
/// violations, context, and recommended actions.
#[derive(Debug, Clone)]
pub struct AlertEvent {
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Alert ID
    pub alert_id: String,

    /// Threshold configuration
    pub threshold: ThresholdConfig,

    /// Current value
    pub current_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message
    pub message: String,

    /// Context information
    pub context: HashMap<String, String>,

    /// Recommended actions
    pub actions: Vec<RecommendedAction>,

    /// Alert metadata
    pub metadata: HashMap<String, String>,

    /// Correlation ID for related alerts
    pub correlation_id: Option<String>,

    /// Suppression information
    pub suppression_info: Option<SuppressionInfo>,
}

impl AlertEvent {
    /// Create new alert event
    pub fn new(
        alert_id: String,
        threshold: ThresholdConfig,
        current_value: f64,
        threshold_value: f64,
        severity: SeverityLevel,
        message: String,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            alert_id,
            threshold,
            current_value,
            threshold_value,
            severity,
            message,
            context: HashMap::new(),
            actions: Vec::new(),
            metadata: HashMap::new(),
            correlation_id: None,
            suppression_info: None,
        }
    }

    /// Add context information
    pub fn add_context(&mut self, key: String, value: String) {
        self.context.insert(key, value);
    }

    /// Add recommended action
    pub fn add_action(&mut self, action: RecommendedAction) {
        self.actions.push(action);
    }

    /// Check if alert is critical
    pub fn is_critical(&self) -> bool {
        self.severity == SeverityLevel::Critical
    }

    /// Get alert age
    pub fn age(&self) -> Duration {
        let now = Utc::now();
        (now - self.timestamp).to_std().unwrap_or(Duration::from_secs(0))
    }
}

/// Alert suppression information
#[derive(Debug, Clone)]
pub struct SuppressionInfo {
    /// Suppression reason
    pub reason: String,

    /// Suppression start time
    pub start_time: DateTime<Utc>,

    /// Suppression duration
    pub duration: Duration,

    /// Suppressed alert count
    pub suppressed_count: u32,
}

/// Threshold monitoring state
///
/// Current state of threshold monitoring system including active alerts,
/// monitoring statistics, and system health information.
#[derive(Debug, Default)]
pub struct ThresholdMonitoringState {
    /// Active alerts
    pub active_alerts: HashMap<String, AlertEvent>,

    /// Alert counts by severity
    pub alert_counts: HashMap<SeverityLevel, u64>,

    /// Last evaluation timestamp
    pub last_evaluation: Option<DateTime<Utc>>,

    /// Total evaluations performed
    pub total_evaluations: u64,

    /// Total alerts generated
    pub total_alerts: u64,

    /// System health status
    pub system_healthy: bool,
}

impl ThresholdMonitoringState {
    /// Add alert to state
    pub fn add_alert(&mut self, alert: AlertEvent) {
        *self.alert_counts.entry(alert.severity).or_insert(0) += 1;
        self.total_alerts += 1;
        self.active_alerts.insert(alert.alert_id.clone(), alert);
        self.update_health_status();
    }

    /// Remove alert from state
    pub fn remove_alert(&mut self, alert_id: &str) -> Option<AlertEvent> {
        let removed = self.active_alerts.remove(alert_id);
        if let Some(ref alert) = removed {
            if let Some(count) = self.alert_counts.get_mut(&alert.severity) {
                *count = count.saturating_sub(1);
            }
        }
        self.update_health_status();
        removed
    }

    /// Update system health status
    fn update_health_status(&mut self) {
        let critical_count = self.alert_counts.get(&SeverityLevel::Critical).copied().unwrap_or(0);
        let high_count = self.alert_counts.get(&SeverityLevel::High).copied().unwrap_or(0);

        self.system_healthy = critical_count == 0 && high_count < 5;
    }

    /// Get alert count for severity level
    pub fn get_alert_count(&self, severity: &SeverityLevel) -> u64 {
        self.alert_counts.get(severity).copied().unwrap_or(0)
    }

    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.system_healthy
    }
}

/// Evaluation statistics for monitoring
///
/// Statistics for threshold evaluations and system performance monitoring.
#[derive(Debug, Default, Clone)]
pub struct EvaluationStatistics {
    /// Total evaluations performed
    pub total_evaluations: u64,

    /// Threshold violations detected
    pub violations_detected: u64,

    /// False positives
    pub false_positives: u64,

    /// Average evaluation time (microseconds)
    pub avg_evaluation_time: f32,

    /// Evaluation errors
    pub evaluation_errors: u64,
}

impl EvaluationStatistics {
    /// Get violation rate
    pub fn violation_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.violations_detected as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }

    /// Get false positive rate
    pub fn false_positive_rate(&self) -> f64 {
        if self.violations_detected > 0 {
            self.false_positives as f64 / self.violations_detected as f64
        } else {
            0.0
        }
    }

    /// Get error rate
    pub fn error_rate(&self) -> f64 {
        if self.total_evaluations > 0 {
            self.evaluation_errors as f64 / self.total_evaluations as f64
        } else {
            0.0
        }
    }

    /// Update evaluation time
    pub fn update_evaluation_time(&mut self, time_micros: f32) {
        if self.avg_evaluation_time == 0.0 {
            self.avg_evaluation_time = time_micros;
        } else {
            self.avg_evaluation_time = (self.avg_evaluation_time * 0.9) + (time_micros * 0.1);
        }
    }
}

// =============================================================================
// ERROR TYPES
// =============================================================================

/// Processing error information
///
/// Information about processing errors including type, severity,
/// and recovery suggestions for robust error handling.
#[derive(Debug, Clone, thiserror::Error)]
#[error("Processing error: {message}")]
pub struct ProcessingError {
    /// Error type
    pub error_type: String,

    /// Error message
    pub message: String,

    /// Error severity
    pub severity: SeverityLevel,

    /// Recovery suggestions
    pub recovery: Vec<String>,

    /// Error timestamp
    pub timestamp: DateTime<Utc>,

    /// Error context
    pub context: HashMap<String, String>,
}

impl ProcessingError {
    /// Create new processing error
    pub fn new(error_type: String, message: String, severity: SeverityLevel) -> Self {
        Self {
            error_type,
            message,
            severity,
            recovery: Vec::new(),
            timestamp: Utc::now(),
            context: HashMap::new(),
        }
    }

    /// Add recovery suggestion
    pub fn add_recovery(&mut self, suggestion: String) {
        self.recovery.push(suggestion);
    }

    /// Add context information
    pub fn add_context(&mut self, key: String, value: String) {
        self.context.insert(key, value);
    }

    /// Check if error is critical
    pub fn is_critical(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Comprehensive error types for the real-time metrics system
#[derive(Debug, thiserror::Error)]
pub enum RealTimeMetricsError {
    /// Configuration error
    #[error("Configuration error: {message}")]
    Configuration { message: String },

    /// Data collection error
    #[error("Data collection error: {message}")]
    DataCollection { message: String },

    /// Processing error
    #[error("Processing error: {message}")]
    Processing { message: String },

    /// Threshold evaluation error
    #[error("Threshold evaluation error: {message}")]
    ThresholdEvaluation { message: String },

    /// Quality control error
    #[error("Quality control error: {message}")]
    QualityControl { message: String },

    /// Optimization error
    #[error("Optimization error: {message}")]
    Optimization { message: String },

    /// I/O error
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization error
    #[error("Serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    /// Timeout error
    #[error("Timeout error: operation timed out after {duration:?}")]
    Timeout { duration: Duration },

    /// Resource exhaustion error
    #[error("Resource exhaustion: {resource} limit exceeded")]
    ResourceExhaustion { resource: String },

    /// Invalid state error
    #[error("Invalid state: {message}")]
    InvalidState { message: String },
}

impl RealTimeMetricsError {
    /// Get error severity level
    pub fn severity(&self) -> SeverityLevel {
        match self {
            RealTimeMetricsError::Configuration { .. } => SeverityLevel::High,
            RealTimeMetricsError::DataCollection { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::Processing { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::ThresholdEvaluation { .. } => SeverityLevel::Low,
            RealTimeMetricsError::QualityControl { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::Optimization { .. } => SeverityLevel::Low,
            RealTimeMetricsError::Io(_) => SeverityLevel::High,
            RealTimeMetricsError::Serialization(_) => SeverityLevel::Medium,
            RealTimeMetricsError::Timeout { .. } => SeverityLevel::Medium,
            RealTimeMetricsError::ResourceExhaustion { .. } => SeverityLevel::Critical,
            RealTimeMetricsError::InvalidState { .. } => SeverityLevel::High,
        }
    }

    /// Check if error is recoverable
    pub fn is_recoverable(&self) -> bool {
        !matches!(
            self,
            RealTimeMetricsError::Configuration { .. }
                | RealTimeMetricsError::InvalidState { .. }
                | RealTimeMetricsError::ResourceExhaustion { .. }
        )
    }
}

/// Error handling policy for processing
///
/// Policy for handling errors during processing including retry strategies,
/// fallback mechanisms, and error escalation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ErrorHandlingPolicy {
    /// Fail fast on any error
    FailFast,

    /// Retry with exponential backoff
    RetryWithBackoff {
        max_retries: usize,
        initial_delay: Duration,
        max_delay: Duration,
    },

    /// Continue processing with logging
    ContinueWithLogging,

    /// Circuit breaker pattern
    CircuitBreaker {
        failure_threshold: usize,
        timeout: Duration,
    },

    /// Custom error handling
    Custom(String),
}

impl Default for ErrorHandlingPolicy {
    fn default() -> Self {
        ErrorHandlingPolicy::RetryWithBackoff {
            max_retries: 3,
            initial_delay: Duration::from_millis(100),
            max_delay: Duration::from_secs(10),
        }
    }
}

// =============================================================================
// TRAIT DEFINITIONS
// =============================================================================

/// Statistical processor trait for data analysis
///
/// Interface for statistical processors that analyze streaming metrics data
/// and generate statistical insights and analysis results.
pub trait StatisticalProcessor: Send + Sync {
    /// Process metrics data and generate statistics
    fn process(
        &self,
        data: &[TimestampedMetrics],
    ) -> Result<StatisticalResult, RealTimeMetricsError>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle data
    fn can_process(&self, data_type: &str) -> bool;

    /// Get processor configuration
    fn config(&self) -> &dyn std::any::Any;

    /// Reset processor state
    fn reset(&mut self);

    /// Get processor statistics
    fn statistics(&self) -> ProcessorStatistics;
}

/// Threshold evaluator trait for threshold monitoring
///
/// Interface for threshold evaluators that assess metric values against
/// configured thresholds and generate alerts when violations occur.
pub trait ThresholdEvaluator: Send + Sync {
    /// Evaluate threshold against current value
    fn evaluate(
        &self,
        config: &ThresholdConfig,
        value: f64,
    ) -> Result<ThresholdEvaluation, RealTimeMetricsError>;

    /// Get evaluator name
    fn name(&self) -> &str;

    /// Check if evaluator supports threshold type
    fn supports_threshold(&self, threshold_type: &str) -> bool;

    /// Update evaluator with historical data
    fn update_history(&mut self, data: &[TimestampedMetrics]);

    /// Get evaluator configuration
    fn configuration(&self) -> &dyn std::any::Any;
}

/// Live optimization algorithm trait
///
/// Interface for live optimization algorithms that analyze real-time
/// performance data and generate optimization recommendations.
pub trait LiveOptimizationAlgorithm: Send + Sync {
    /// Generate optimization recommendations
    fn optimize(
        &self,
        metrics: &RealTimeMetrics,
        history: &[TimestampedMetrics],
        context: &OptimizationContext,
    ) -> Result<Vec<OptimizationRecommendation>, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm confidence for current data
    fn confidence(&self, data_quality: f32) -> f32;

    /// Check if algorithm is applicable
    fn is_applicable(&self, context: &OptimizationContext) -> bool;

    /// Update algorithm with feedback
    fn update_with_feedback(
        &mut self,
        feedback: &PerformanceFeedback,
    ) -> Result<(), RealTimeMetricsError>;

    /// Get algorithm statistics
    fn statistics(&self) -> AlgorithmStatistics;
}

/// Sample rate adjustment algorithm trait
///
/// Interface for algorithms that control adaptive sample rate adjustment
/// based on system conditions and performance requirements.
pub trait SampleRateAlgorithm: Send + Sync {
    /// Calculate optimal sample rate
    fn calculate_rate(
        &self,
        current_load: f32,
        target_accuracy: f32,
        resource_availability: f32,
    ) -> Result<f32, RealTimeMetricsError>;

    /// Get algorithm name
    fn name(&self) -> &str;

    /// Get algorithm parameters
    fn parameters(&self) -> HashMap<String, f64>;

    /// Update algorithm state
    fn update_state(&mut self, metrics: &RealTimeMetrics);

    /// Reset algorithm to initial state
    fn reset(&mut self);
}

/// Quality checker trait for data validation
///
/// Interface for quality checkers that validate data quality throughout
/// the processing pipeline.
pub trait QualityChecker: Send + Sync {
    /// Check data quality
    fn check(
        &self,
        data: &[TimestampedMetrics],
    ) -> Result<QualityCheckResult, RealTimeMetricsError>;

    /// Get checker name
    fn name(&self) -> &str;

    /// Get quality standards
    fn standards(&self) -> &QualityStandards;

    /// Update checker configuration
    fn update_standards(&mut self, standards: QualityStandards);

    /// Get checker statistics
    fn statistics(&self) -> CheckerStatistics;
}

/// Pipeline stage trait for processing pipeline
///
/// Interface for pipeline stages that process data in the metrics processing pipeline.
pub trait PipelineStage: Send + Sync {
    /// Process pipeline input and generate output
    fn process(&self, input: PipelineInput) -> Result<PipelineOutput, RealTimeMetricsError>;

    /// Get stage name
    fn name(&self) -> &str;

    /// Get stage configuration
    fn configuration(&self) -> &dyn std::any::Any;

    /// Get stage statistics
    fn statistics(&self) -> PipelineStageStats;

    /// Check if stage can process input type
    fn can_process(&self, input_type: &str) -> bool;
}

// =============================================================================
// ADDITIONAL SUPPORT TYPES
// =============================================================================

/// Statistics for statistical processors
#[derive(Debug, Clone, Default)]
pub struct ProcessorStatistics {
    /// Total data points processed
    pub total_processed: u64,

    /// Processing errors
    pub errors: u64,

    /// Average processing time (microseconds)
    pub avg_processing_time: f32,

    /// Last processing timestamp
    pub last_processing: Option<DateTime<Utc>>,
}

/// Statistics for optimization algorithms
#[derive(Debug, Clone, Default)]
pub struct AlgorithmStatistics {
    /// Total recommendations generated
    pub recommendations_generated: u64,

    /// Average confidence score
    pub avg_confidence: f32,

    /// Algorithm accuracy (successful recommendations)
    pub accuracy: f32,

    /// Processing time statistics
    pub processing_time: ProcessorStatistics,
}

/// Statistics for quality checkers
#[derive(Debug, Clone, Default)]
pub struct CheckerStatistics {
    /// Total checks performed
    pub total_checks: u64,

    /// Quality issues found
    pub issues_found: u64,

    /// Average quality score
    pub avg_quality_score: f32,

    /// Check processing time
    pub avg_check_time: f32,
}

/// Quality requirements for processing
///
/// Quality requirements and thresholds for data processing operations
/// to ensure reliable and accurate results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityRequirements {
    /// Minimum accuracy required
    pub min_accuracy: f32,

    /// Maximum acceptable error rate
    pub max_error_rate: f32,

    /// Required completeness
    pub required_completeness: f32,

    /// Consistency requirements
    pub consistency_requirements: HashMap<String, f32>,

    /// Maximum data age allowed
    pub max_data_age: Duration,
}

impl Default for QualityRequirements {
    fn default() -> Self {
        Self {
            min_accuracy: 0.95,
            max_error_rate: 0.05,
            required_completeness: 0.9,
            consistency_requirements: HashMap::new(),
            max_data_age: Duration::from_secs(300),
        }
    }
}

/// Pipeline configuration
///
/// Configuration for the processing pipeline including parallelism,
/// buffer sizes, and processing parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Parallel processing threads
    pub parallel_threads: usize,

    /// Buffer size per stage
    pub buffer_size: usize,

    /// Processing timeout
    pub timeout: Duration,

    /// Error handling policy
    pub error_handling: ErrorHandlingPolicy,

    /// Quality control enabled
    pub quality_control: bool,

    /// Pipeline stages configuration
    pub stages: Vec<String>,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            parallel_threads: num_cpus::get(),
            buffer_size: 1000,
            timeout: Duration::from_secs(30),
            error_handling: ErrorHandlingPolicy::default(),
            quality_control: true,
            stages: vec![
                "validation".to_string(),
                "filtering".to_string(),
                "aggregation".to_string(),
                "analysis".to_string(),
            ],
        }
    }
}

/// Pipeline performance statistics
///
/// Performance statistics for the processing pipeline including throughput,
/// latency, and resource utilization metrics.
#[derive(Debug, Default)]
pub struct PipelineStatistics {
    /// Total items processed
    pub items_processed: AtomicU64,

    /// Processing throughput (items/sec)
    pub throughput: AtomicF32,

    /// Average processing latency (microseconds)
    pub avg_latency: AtomicF32,

    /// Error rate
    pub error_rate: AtomicF32,

    /// Resource utilization
    pub resource_utilization: AtomicF32,

    /// Pipeline uptime
    pub uptime: AtomicU64,
}

impl PipelineStatistics {
    /// Get current throughput
    pub fn current_throughput(&self) -> f32 {
        self.throughput.load(Ordering::Acquire)
    }

    /// Get current error rate
    pub fn current_error_rate(&self) -> f32 {
        self.error_rate.load(Ordering::Acquire)
    }

    /// Update throughput
    pub fn update_throughput(&self, throughput: f32) {
        self.throughput.store(throughput, Ordering::Release);
    }
}

/// Pipeline input data structure
///
/// Input data structure for pipeline stages containing metrics data
/// and processing context.
#[derive(Debug, Clone)]
pub struct PipelineInput {
    /// Input data
    pub data: Vec<TimestampedMetrics>,

    /// Processing context
    pub context: ProcessingContext,

    /// Input metadata
    pub metadata: HashMap<String, String>,
}

/// Pipeline output data structure
///
/// Output data structure from pipeline stages containing processed results
/// and updated context.
#[derive(Debug, Clone)]
pub struct PipelineOutput {
    /// Output data
    pub data: Vec<TimestampedMetrics>,

    /// Processing results
    pub results: ProcessingResults,

    /// Output metadata
    pub metadata: HashMap<String, String>,
}

/// Processing context for pipeline operations
///
/// Context information maintained throughout pipeline processing including
/// configuration, state, and processing metadata.
#[derive(Debug, Clone)]
pub struct ProcessingContext {
    /// Processing configuration
    pub config: PipelineConfig,

    /// Current processing stage
    pub current_stage: String,

    /// Processing timestamp
    pub timestamp: DateTime<Utc>,

    /// Quality requirements
    pub quality_requirements: QualityRequirements,

    /// Context metadata
    pub metadata: HashMap<String, String>,
}

impl ProcessingContext {
    /// Create new processing context
    pub fn new(config: PipelineConfig) -> Self {
        Self {
            config,
            current_stage: "initialization".to_string(),
            timestamp: Utc::now(),
            quality_requirements: QualityRequirements::default(),
            metadata: HashMap::new(),
        }
    }

    /// Update current stage
    pub fn update_stage(&mut self, stage: String) {
        self.current_stage = stage;
        self.timestamp = Utc::now();
    }
}

/// Processing results from pipeline operations
///
/// Results from pipeline processing including statistics, quality metrics,
/// and any processing errors or warnings.
#[derive(Debug, Clone)]
pub struct ProcessingResults {
    /// Processing statistics
    pub statistics: ProcessorStatistics,

    /// Quality metrics
    pub quality_metrics: QualityMetrics,

    /// Processing errors
    pub errors: Vec<ProcessingError>,

    /// Result metadata
    pub metadata: HashMap<String, String>,
}

impl Default for ProcessingResults {
    fn default() -> Self {
        Self {
            statistics: ProcessorStatistics::default(),
            quality_metrics: QualityMetrics::default(),
            errors: Vec::new(),
            metadata: HashMap::new(),
        }
    }
}

/// Quality check result
///
/// Result from quality checking including pass/fail status, quality scores,
/// and detailed findings.
#[derive(Debug, Clone)]
pub struct QualityCheckResult {
    /// Check passed
    pub passed: bool,

    /// Quality score (0.0 to 1.0)
    pub score: f32,

    /// Quality metrics
    pub metrics: QualityMetrics,

    /// Quality issues found
    pub issues: Vec<QualityIssue>,

    /// Check metadata
    pub metadata: HashMap<String, String>,
}

impl QualityCheckResult {
    /// Create successful quality check result
    pub fn success(score: f32, metrics: QualityMetrics) -> Self {
        Self {
            passed: true,
            score,
            metrics,
            issues: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Create failed quality check result
    pub fn failure(score: f32, metrics: QualityMetrics, issues: Vec<QualityIssue>) -> Self {
        Self {
            passed: false,
            score,
            metrics,
            issues,
            metadata: HashMap::new(),
        }
    }

    /// Check if result has critical issues
    pub fn has_critical_issues(&self) -> bool {
        self.issues.iter().any(|issue| issue.severity == SeverityLevel::Critical)
    }
}

/// Quality issue information
///
/// Information about quality issues found during quality checking
/// including type, severity, and resolution recommendations.
#[derive(Debug, Clone)]
pub struct QualityIssue {
    /// Issue type
    pub issue_type: QualityIssueType,

    /// Issue severity
    pub severity: SeverityLevel,

    /// Issue description
    pub description: String,

    /// Affected data count
    pub affected_count: usize,

    /// Resolution recommendations
    pub recommendations: Vec<String>,

    /// Issue timestamp
    pub timestamp: DateTime<Utc>,
}

impl QualityIssue {
    /// Create new quality issue
    pub fn new(
        issue_type: QualityIssueType,
        severity: SeverityLevel,
        description: String,
        affected_count: usize,
    ) -> Self {
        Self {
            issue_type,
            severity,
            description,
            affected_count,
            recommendations: Vec::new(),
            timestamp: Utc::now(),
        }
    }

    /// Add recommendation
    pub fn add_recommendation(&mut self, recommendation: String) {
        self.recommendations.push(recommendation);
    }

    /// Check if issue requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

/// Quality standards for data processing
///
/// Standards and thresholds for data quality assessment and validation
/// throughout the processing pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityStandards {
    /// Minimum completeness threshold
    pub min_completeness: f32,

    /// Maximum staleness allowed
    pub max_staleness: Duration,

    /// Outlier detection threshold
    pub outlier_threshold: f32,

    /// Consistency requirements
    pub consistency_threshold: f32,

    /// Accuracy requirements
    pub accuracy_threshold: f32,

    /// Maximum error rate allowed
    pub max_error_rate: f32,
}

impl Default for QualityStandards {
    fn default() -> Self {
        Self {
            min_completeness: 0.9,
            max_staleness: Duration::from_secs(300),
            outlier_threshold: 2.5, // 2.5 standard deviations
            consistency_threshold: 0.95,
            accuracy_threshold: 0.95,
            max_error_rate: 0.05,
        }
    }
}

/// Quality violation information
///
/// Information about quality violations including details, impact,
/// and corrective actions taken.
#[derive(Debug, Clone)]
pub struct QualityViolation {
    /// Violation timestamp
    pub timestamp: DateTime<Utc>,

    /// Violation type
    pub violation_type: QualityIssueType,

    /// Violation severity
    pub severity: SeverityLevel,

    /// Impact assessment
    pub impact: f32,

    /// Corrective actions taken
    pub actions_taken: Vec<String>,

    /// Resolution status
    pub resolved: bool,

    /// Resolution timestamp
    pub resolution_timestamp: Option<DateTime<Utc>>,
}

impl QualityViolation {
    /// Create new quality violation
    pub fn new(violation_type: QualityIssueType, severity: SeverityLevel, impact: f32) -> Self {
        Self {
            timestamp: Utc::now(),
            violation_type,
            severity,
            impact,
            actions_taken: Vec::new(),
            resolved: false,
            resolution_timestamp: None,
        }
    }

    /// Add corrective action
    pub fn add_action(&mut self, action: String) {
        self.actions_taken.push(action);
    }

    /// Mark violation as resolved
    pub fn resolve(&mut self) {
        self.resolved = true;
        self.resolution_timestamp = Some(Utc::now());
    }

    /// Get time to resolution
    pub fn time_to_resolution(&self) -> Option<Duration> {
        self.resolution_timestamp.map(|resolution| {
            (resolution - self.timestamp).to_std().unwrap_or(Duration::from_secs(0))
        })
    }
}

/// Quality control configuration
///
/// Configuration for quality control operations including checking
/// frequency, standards enforcement, and violation handling.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QualityControlConfig {
    /// Quality checking enabled
    pub enabled: bool,

    /// Checking frequency
    pub check_frequency: Duration,

    /// Automatic correction enabled
    pub auto_correction: bool,

    /// Violation alert threshold
    pub alert_threshold: f32,

    /// Standards enforcement level
    pub enforcement_level: EnforcementLevel,

    /// Maximum violations before escalation
    pub max_violations: usize,
}

impl Default for QualityControlConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            check_frequency: Duration::from_secs(30),
            auto_correction: true,
            alert_threshold: 0.8,
            enforcement_level: EnforcementLevel::Moderate,
            max_violations: 10,
        }
    }
}

// =============================================================================
// ENUMS
// =============================================================================

/// Types of performance insights
///
/// Classification of different types of performance insights that can be
/// generated from real-time data analysis.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance degradation detected
    PerformanceDegradation,

    /// Resource bottleneck identified
    ResourceBottleneck,

    /// Optimization opportunity found
    OptimizationOpportunity,

    /// Anomalous behavior detected
    AnomalousBehavior,

    /// Trend change identified
    TrendChange,

    /// Threshold violation
    ThresholdViolation,

    /// Capacity planning insight
    CapacityPlanning,

    /// Efficiency improvement
    EfficiencyImprovement,

    /// Custom insight
    Custom(String),
}

impl std::fmt::Display for InsightType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InsightType::PerformanceDegradation => write!(f, "Performance Degradation"),
            InsightType::ResourceBottleneck => write!(f, "Resource Bottleneck"),
            InsightType::OptimizationOpportunity => write!(f, "Optimization Opportunity"),
            InsightType::AnomalousBehavior => write!(f, "Anomalous Behavior"),
            InsightType::TrendChange => write!(f, "Trend Change"),
            InsightType::ThresholdViolation => write!(f, "Threshold Violation"),
            InsightType::CapacityPlanning => write!(f, "Capacity Planning"),
            InsightType::EfficiencyImprovement => write!(f, "Efficiency Improvement"),
            InsightType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

// SeverityLevel is imported from pattern_engine module - see line 115

/// Scope of monitoring for individual threads
///
/// Defines the specific monitoring responsibilities and data collection
/// scope for individual monitoring threads.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitoringScope {
    /// CPU performance monitoring
    CpuMonitoring,

    /// Memory performance monitoring
    MemoryMonitoring,

    /// I/O performance monitoring
    IoMonitoring,

    /// Network performance monitoring
    NetworkMonitoring,

    /// Application-level monitoring
    ApplicationMonitoring,

    /// System-level monitoring
    SystemMonitoring,

    /// Thread-level monitoring
    ThreadMonitoring,

    /// Process-level monitoring
    ProcessMonitoring,

    /// Custom monitoring scope
    Custom(String),
}

impl std::fmt::Display for MonitoringScope {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitoringScope::CpuMonitoring => write!(f, "CPU Monitoring"),
            MonitoringScope::MemoryMonitoring => write!(f, "Memory Monitoring"),
            MonitoringScope::IoMonitoring => write!(f, "I/O Monitoring"),
            MonitoringScope::NetworkMonitoring => write!(f, "Network Monitoring"),
            MonitoringScope::ApplicationMonitoring => write!(f, "Application Monitoring"),
            MonitoringScope::SystemMonitoring => write!(f, "System Monitoring"),
            MonitoringScope::ThreadMonitoring => write!(f, "Thread Monitoring"),
            MonitoringScope::ProcessMonitoring => write!(f, "Process Monitoring"),
            MonitoringScope::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Types of monitoring events
///
/// Classification of different monitoring events for system communication
/// and event-driven processing.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MonitoringEventType {
    /// Metrics collected
    MetricsCollected,

    /// Threshold exceeded
    ThresholdExceeded,

    /// Anomaly detected
    AnomalyDetected,

    /// System state changed
    SystemStateChanged,

    /// Performance degraded
    PerformanceDegraded,

    /// Performance improved
    PerformanceImproved,

    /// Optimization applied
    OptimizationApplied,

    /// Configuration changed
    ConfigurationChanged,

    /// Error occurred
    ErrorOccurred,

    /// Warning issued
    WarningIssued,

    /// System started
    SystemStarted,

    /// System stopped
    SystemStopped,

    /// Monitoring started
    MonitoringStarted,

    /// Monitoring shutdown
    MonitoringShutdown,

    /// Custom event
    Custom(String),
}

impl std::fmt::Display for MonitoringEventType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            MonitoringEventType::MetricsCollected => write!(f, "Metrics Collected"),
            MonitoringEventType::ThresholdExceeded => write!(f, "Threshold Exceeded"),
            MonitoringEventType::AnomalyDetected => write!(f, "Anomaly Detected"),
            MonitoringEventType::SystemStateChanged => write!(f, "System State Changed"),
            MonitoringEventType::PerformanceDegraded => write!(f, "Performance Degraded"),
            MonitoringEventType::PerformanceImproved => write!(f, "Performance Improved"),
            MonitoringEventType::OptimizationApplied => write!(f, "Optimization Applied"),
            MonitoringEventType::ConfigurationChanged => write!(f, "Configuration Changed"),
            MonitoringEventType::ErrorOccurred => write!(f, "Error Occurred"),
            MonitoringEventType::WarningIssued => write!(f, "Warning Issued"),
            MonitoringEventType::SystemStarted => write!(f, "System Started"),
            MonitoringEventType::SystemStopped => write!(f, "System Stopped"),
            MonitoringEventType::MonitoringStarted => write!(f, "Monitoring Started"),
            MonitoringEventType::MonitoringShutdown => write!(f, "Monitoring Shutdown"),
            MonitoringEventType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Direction for threshold evaluation
///
/// Specifies whether threshold violations occur when values are above
/// or below the configured threshold levels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ThresholdDirection {
    /// Violation when value exceeds threshold
    Above,

    /// Violation when value falls below threshold
    Below,

    /// Violation when value is outside range
    OutsideRange { min: f64, max: f64 },

    /// Violation when value is inside range (for inverted logic)
    InsideRange { min: f64, max: f64 },
}

impl std::fmt::Display for ThresholdDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ThresholdDirection::Above => write!(f, "Above"),
            ThresholdDirection::Below => write!(f, "Below"),
            ThresholdDirection::OutsideRange { min, max } => {
                write!(f, "Outside [{}, {}]", min, max)
            },
            ThresholdDirection::InsideRange { min, max } => write!(f, "Inside [{}, {}]", min, max),
        }
    }
}

impl ThresholdDirection {
    /// Check if value violates threshold
    pub fn is_violation(&self, value: f64, threshold: f64) -> bool {
        match self {
            ThresholdDirection::Above => value > threshold,
            ThresholdDirection::Below => value < threshold,
            ThresholdDirection::OutsideRange { min, max } => value < *min || value > *max,
            ThresholdDirection::InsideRange { min, max } => value >= *min && value <= *max,
        }
    }
}

// ActionType is imported from parent types module - see line 86

/// Types of quality issues
///
/// Classification of different types of quality issues that can be
/// detected during quality checking.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QualityIssueType {
    /// Missing data
    MissingData,

    /// Inconsistent data
    InconsistentData,

    /// Outlier data
    OutlierData,

    /// Stale data
    StaleData,

    /// Corrupted data
    CorruptedData,

    /// Duplicate data
    DuplicateData,

    /// Invalid format
    InvalidFormat,

    /// Schema mismatch
    SchemaMismatch,

    /// Custom issue
    Custom(String),
}

impl std::fmt::Display for QualityIssueType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QualityIssueType::MissingData => write!(f, "Missing Data"),
            QualityIssueType::InconsistentData => write!(f, "Inconsistent Data"),
            QualityIssueType::OutlierData => write!(f, "Outlier Data"),
            QualityIssueType::StaleData => write!(f, "Stale Data"),
            QualityIssueType::CorruptedData => write!(f, "Corrupted Data"),
            QualityIssueType::DuplicateData => write!(f, "Duplicate Data"),
            QualityIssueType::InvalidFormat => write!(f, "Invalid Format"),
            QualityIssueType::SchemaMismatch => write!(f, "Schema Mismatch"),
            QualityIssueType::Custom(name) => write!(f, "Custom: {}", name),
        }
    }
}

/// Quality enforcement levels
///
/// Different levels of quality enforcement with varying strictness
/// and impact on processing operations.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EnforcementLevel {
    /// Advisory only (warnings)
    Advisory,

    /// Moderate enforcement (warnings and corrections)
    Moderate,

    /// Strict enforcement (block processing on violations)
    Strict,

    /// Emergency mode (immediate escalation)
    Emergency,
}

impl std::fmt::Display for EnforcementLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EnforcementLevel::Advisory => write!(f, "Advisory"),
            EnforcementLevel::Moderate => write!(f, "Moderate"),
            EnforcementLevel::Strict => write!(f, "Strict"),
            EnforcementLevel::Emergency => write!(f, "Emergency"),
        }
    }
}

/// Types of optimization objectives
///
/// Different types of optimization objectives that can be pursued
/// simultaneously in multi-objective optimization.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ObjectiveType {
    /// Maximize throughput
    MaximizeThroughput,

    /// Minimize latency
    MinimizeLatency,

    /// Minimize resource usage
    MinimizeResourceUsage,

    /// Maximize efficiency
    MaximizeEfficiency,

    /// Minimize cost
    MinimizeCost,

    /// Minimize energy consumption
    MinimizeEnergy,

    /// Maximize reliability
    MaximizeReliability,

    /// Custom objective
    Custom {
        name: String,
        direction: OptimizationDirection,
    },
}

impl std::fmt::Display for ObjectiveType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ObjectiveType::MaximizeThroughput => write!(f, "Maximize Throughput"),
            ObjectiveType::MinimizeLatency => write!(f, "Minimize Latency"),
            ObjectiveType::MinimizeResourceUsage => write!(f, "Minimize Resource Usage"),
            ObjectiveType::MaximizeEfficiency => write!(f, "Maximize Efficiency"),
            ObjectiveType::MinimizeCost => write!(f, "Minimize Cost"),
            ObjectiveType::MinimizeEnergy => write!(f, "Minimize Energy"),
            ObjectiveType::MaximizeReliability => write!(f, "Maximize Reliability"),
            ObjectiveType::Custom { name, direction } => {
                write!(f, "Custom: {} ({:?})", name, direction)
            },
        }
    }
}

/// Optimization direction
///
/// Direction for optimization objectives (minimize or maximize).
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum OptimizationDirection {
    /// Minimize the objective
    Minimize,

    /// Maximize the objective
    Maximize,
}

impl std::fmt::Display for OptimizationDirection {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptimizationDirection::Minimize => write!(f, "Minimize"),
            OptimizationDirection::Maximize => write!(f, "Maximize"),
        }
    }
}

// =============================================================================
// ADDITIONAL OPTIMIZATION AND STATISTICAL TYPES
// =============================================================================

/// Threshold evaluation result
///
/// Result from threshold evaluation including violation status
/// and contextual information.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdEvaluation {
    /// Evaluation timestamp
    pub timestamp: DateTime<Utc>,

    /// Threshold violated
    pub violated: bool,

    /// Violation severity
    pub severity: SeverityLevel,

    /// Current value
    pub current_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Evaluation confidence
    pub confidence: f32,

    /// Evaluation metadata
    pub metadata: HashMap<String, String>,
}

impl ThresholdEvaluation {
    /// Create new threshold evaluation
    pub fn new(violated: bool, current_value: f64, threshold_value: f64) -> Self {
        let severity = if violated {
            if (current_value - threshold_value).abs() > threshold_value * 0.5 {
                SeverityLevel::Critical
            } else if (current_value - threshold_value).abs() > threshold_value * 0.2 {
                SeverityLevel::High
            } else {
                SeverityLevel::Medium
            }
        } else {
            SeverityLevel::Info
        };

        Self {
            timestamp: Utc::now(),
            violated,
            severity,
            current_value,
            threshold_value,
            confidence: 1.0,
            metadata: HashMap::new(),
        }
    }

    /// Get violation magnitude
    pub fn violation_magnitude(&self) -> f64 {
        if self.violated {
            (self.current_value - self.threshold_value).abs()
        } else {
            0.0
        }
    }

    /// Get violation percentage
    pub fn violation_percentage(&self) -> f64 {
        if self.threshold_value != 0.0 {
            self.violation_magnitude() / self.threshold_value.abs() * 100.0
        } else {
            0.0
        }
    }
}

/// Optimization context for algorithms
///
/// Context information provided to optimization algorithms including
/// system state, constraints, and optimization objectives.
#[derive(Debug, Clone)]
pub struct OptimizationContext {
    /// Current system state
    pub system_state: SystemState,

    /// Test characteristics
    pub test_characteristics: TestCharacteristics,

    /// Optimization objectives
    pub objectives: Vec<OptimizationObjective>,

    /// Constraints
    pub constraints: HashMap<String, f64>,

    /// Historical performance
    pub historical_performance: Vec<PerformanceDataPoint>,

    /// Context metadata
    pub metadata: HashMap<String, String>,

    /// Optimization window
    pub optimization_window: Duration,

    /// Maximum optimization time
    pub max_optimization_time: Duration,
}

impl OptimizationContext {
    /// Create new optimization context
    pub fn new(system_state: SystemState, test_characteristics: TestCharacteristics) -> Self {
        Self {
            system_state,
            test_characteristics,
            objectives: Vec::new(),
            constraints: HashMap::new(),
            historical_performance: Vec::new(),
            metadata: HashMap::new(),
            optimization_window: Duration::from_secs(300),
            max_optimization_time: Duration::from_secs(60),
        }
    }

    /// Add optimization objective
    pub fn add_objective(&mut self, objective: OptimizationObjective) {
        self.objectives.push(objective);
    }

    /// Add constraint
    pub fn add_constraint(&mut self, name: String, value: f64) {
        self.constraints.insert(name, value);
    }

    /// Check if context has conflicting objectives
    pub fn has_conflicting_objectives(&self) -> bool {
        // Simplified conflict detection
        let has_minimize = self.objectives.iter().any(|obj| {
            matches!(
                obj.objective_type,
                ObjectiveType::MinimizeLatency
                    | ObjectiveType::MinimizeResourceUsage
                    | ObjectiveType::MinimizeCost
            )
        });
        let has_maximize = self.objectives.iter().any(|obj| {
            matches!(
                obj.objective_type,
                ObjectiveType::MaximizeThroughput
                    | ObjectiveType::MaximizeEfficiency
                    | ObjectiveType::MaximizeReliability
            )
        });
        has_minimize && has_maximize
    }
}

/// Optimization objective
///
/// Individual optimization objective with weight, target, and constraints
/// for multi-objective optimization scenarios.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationObjective {
    /// Objective name
    pub name: String,

    /// Objective type
    pub objective_type: ObjectiveType,

    /// Target value
    pub target: f64,

    /// Objective weight
    pub weight: f32,

    /// Constraints
    pub constraints: HashMap<String, f64>,

    /// Priority level
    pub priority: u8,

    /// Tolerance range
    pub tolerance: f64,
}

impl OptimizationObjective {
    /// Create new optimization objective
    pub fn new(name: String, objective_type: ObjectiveType, target: f64, weight: f32) -> Self {
        Self {
            name,
            objective_type,
            target,
            weight,
            constraints: HashMap::new(),
            priority: 1,
            tolerance: 0.05, // 5% tolerance
        }
    }

    /// Check if current value meets objective
    pub fn is_met(&self, current_value: f64) -> bool {
        let diff = (current_value - self.target).abs();
        diff <= self.tolerance * self.target.abs()
    }

    /// Calculate objective score
    pub fn score(&self, current_value: f64) -> f32 {
        let diff = (current_value - self.target).abs();
        let normalized_diff = if self.target != 0.0 { diff / self.target.abs() } else { diff };

        (1.0 - normalized_diff as f32).clamp(0.0, 1.0)
    }
}

/// Optimization recommendation
///
/// Recommendation generated by optimization algorithms including
/// actions, expected impact, and confidence scores.
#[derive(Debug, Clone)]
pub struct OptimizationRecommendation {
    /// Recommendation ID
    pub id: String,

    /// Recommendation timestamp
    pub timestamp: DateTime<Utc>,

    /// Recommended actions
    pub actions: Vec<RecommendedAction>,

    /// Expected performance impact
    pub expected_impact: ImpactAssessment,

    /// Recommendation confidence
    pub confidence: f32,

    /// Supporting analysis
    pub analysis: String,

    /// Risk assessment
    pub risks: Vec<RiskFactor>,

    /// Implementation priority
    pub priority: u8,

    /// Expected implementation time
    pub implementation_time: Duration,
}

impl OptimizationRecommendation {
    /// Create new optimization recommendation
    pub fn new(id: String, actions: Vec<RecommendedAction>, confidence: f32) -> Self {
        Self {
            id,
            timestamp: Utc::now(),
            actions,
            expected_impact: ImpactAssessment::default(),
            confidence,
            analysis: String::new(),
            risks: Vec::new(),
            priority: 1,
            implementation_time: Duration::from_secs(300),
        }
    }

    /// Add risk factor
    pub fn add_risk(&mut self, risk: RiskFactor) {
        self.risks.push(risk);
    }

    /// Check if recommendation is high priority
    pub fn is_high_priority(&self) -> bool {
        self.priority >= 3 && self.confidence >= 0.8
    }

    /// Calculate overall recommendation score
    pub fn overall_score(&self) -> f32 {
        let impact_score = self.expected_impact.overall_score();
        let confidence_weight = self.confidence;
        let priority_weight = self.priority as f32 / 5.0;

        (impact_score * confidence_weight * priority_weight).min(1.0)
    }
}

/// Risk factor for recommendations
///
/// Risk factor associated with implementing optimization recommendations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFactor {
    /// Risk type
    pub risk_type: String,

    /// Risk description
    pub description: String,

    /// Risk probability (0.0 to 1.0)
    pub probability: f32,

    /// Risk impact (0.0 to 1.0)
    pub impact: f32,

    /// Risk severity
    pub severity: SeverityLevel,

    /// Mitigation strategies
    pub mitigation: Vec<String>,
}

impl RiskFactor {
    /// Create new risk factor
    pub fn new(risk_type: String, description: String, probability: f32, impact: f32) -> Self {
        let severity = match probability * impact {
            x if x >= 0.7 => SeverityLevel::Critical,
            x if x >= 0.5 => SeverityLevel::High,
            x if x >= 0.3 => SeverityLevel::Medium,
            x if x >= 0.1 => SeverityLevel::Low,
            _ => SeverityLevel::Info,
        };

        Self {
            risk_type,
            description,
            probability,
            impact,
            severity,
            mitigation: Vec::new(),
        }
    }

    /// Calculate risk score
    pub fn risk_score(&self) -> f32 {
        self.probability * self.impact
    }

    /// Add mitigation strategy
    pub fn add_mitigation(&mut self, strategy: String) {
        self.mitigation.push(strategy);
    }
}

/// Statistical result from processing
///
/// Statistical analysis result including various statistical measures
/// and analysis metadata.
#[derive(Debug, Clone)]
pub struct StatisticalResult {
    /// Result timestamp
    pub timestamp: DateTime<Utc>,

    /// Statistical measures
    pub measures: HashMap<String, f64>,

    /// Distribution analysis
    pub distribution: DistributionAnalysis,

    /// Correlation analysis
    pub correlations: HashMap<String, f32>,

    /// Trend analysis
    pub trends: Vec<TrendAnalysis>,

    /// Analysis confidence
    pub confidence: f32,

    /// Sample size
    pub sample_size: usize,

    /// Analysis window
    pub analysis_window: Duration,
}

impl StatisticalResult {
    /// Create new statistical result
    pub fn new(sample_size: usize, analysis_window: Duration) -> Self {
        Self {
            timestamp: Utc::now(),
            measures: HashMap::new(),
            distribution: DistributionAnalysis::default(),
            correlations: HashMap::new(),
            trends: Vec::new(),
            confidence: 0.0,
            sample_size,
            analysis_window,
        }
    }

    /// Add statistical measure
    pub fn add_measure(&mut self, name: String, value: f64) {
        self.measures.insert(name, value);
    }
}

impl Default for StatisticalResult {
    fn default() -> Self {
        Self {
            timestamp: Utc::now(),
            measures: HashMap::new(),
            distribution: DistributionAnalysis::default(),
            correlations: HashMap::new(),
            trends: Vec::new(),
            confidence: 0.0,
            sample_size: 0,
            analysis_window: Duration::from_secs(60),
        }
    }
}

impl StatisticalResult {
    /// Get measure value
    pub fn get_measure(&self, name: &str) -> Option<f64> {
        self.measures.get(name).copied()
    }

    /// Check if result is statistically significant
    pub fn is_significant(&self) -> bool {
        self.confidence >= 0.95 && self.sample_size >= 30
    }
}

/// Distribution analysis results
///
/// Analysis of data distribution including type, parameters, and
/// goodness of fit measures.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributionAnalysis {
    /// Distribution type
    pub distribution_type: String,

    /// Distribution parameters
    pub parameters: HashMap<String, f64>,

    /// Goodness of fit score
    pub goodness_of_fit: f32,

    /// Statistical tests
    pub tests: HashMap<String, f64>,

    /// Confidence level
    pub confidence_level: f32,
}

impl Default for DistributionAnalysis {
    fn default() -> Self {
        Self {
            distribution_type: "normal".to_string(),
            parameters: HashMap::new(),
            goodness_of_fit: 0.0,
            tests: HashMap::new(),
            confidence_level: 0.95,
        }
    }
}

impl DistributionAnalysis {
    /// Check if distribution is normal
    pub fn is_normal(&self) -> bool {
        self.distribution_type == "normal" && self.goodness_of_fit >= 0.8
    }

    /// Get distribution parameter
    pub fn get_parameter(&self, name: &str) -> Option<f64> {
        self.parameters.get(name).copied()
    }

    /// Add statistical test result
    pub fn add_test(&mut self, test_name: String, p_value: f64) {
        self.tests.insert(test_name, p_value);
    }
}

/// Trend analysis results
///
/// Analysis of trends in the data including direction, strength,
/// and statistical significance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    /// Trend metric
    pub metric: String,

    /// Trend direction
    pub direction: TrendDirection,

    /// Trend strength
    pub strength: f32,

    /// Statistical significance
    pub significance: f32,

    /// Trend duration
    #[serde(skip)]
    pub duration: Duration,

    /// Slope coefficient
    pub slope: f64,

    /// R-squared value
    pub r_squared: f32,

    /// Confidence level of the trend analysis
    pub confidence: f32,
}

impl Default for TrendAnalysis {
    fn default() -> Self {
        Self {
            metric: String::new(),
            direction: TrendDirection::Unknown,
            strength: 0.0,
            significance: 0.0,
            duration: Duration::from_secs(0),
            slope: 0.0,
            r_squared: 0.0,
            confidence: 0.0,
        }
    }
}

impl TrendAnalysis {
    /// Create new trend analysis
    pub fn new(
        metric: String,
        direction: TrendDirection,
        strength: f32,
        duration: Duration,
    ) -> Self {
        Self {
            metric,
            direction,
            strength,
            significance: 0.0,
            duration,
            slope: 0.0,
            r_squared: 0.0,
            confidence: 0.5,
        }
    }

    /// Check if trend is statistically significant
    pub fn is_significant(&self) -> bool {
        self.significance >= 0.95 && self.strength >= 0.3
    }

    /// Check if trend is strong
    pub fn is_strong(&self) -> bool {
        self.strength >= 0.7 && self.r_squared >= 0.5
    }
}

// =============================================================================
// ADDITIONAL HELPER TYPES AND CONFIGURATION
// =============================================================================

/// Sample rate configuration
///
/// Configuration for adaptive sample rate control including bounds,
/// adjustment policies, and performance targets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SampleRateConfig {
    /// Minimum sample rate
    pub min_rate: f32,

    /// Maximum sample rate
    pub max_rate: f32,

    /// Target accuracy
    pub target_accuracy: f32,

    /// Adjustment sensitivity
    pub adjustment_sensitivity: f32,

    /// Rate adjustment interval
    pub adjustment_interval: Duration,

    /// Stability threshold
    pub stability_threshold: f32,

    /// Adaptive mode enabled
    pub adaptive_mode: bool,
}

impl Default for SampleRateConfig {
    fn default() -> Self {
        Self {
            min_rate: 0.1,
            max_rate: 100.0,
            target_accuracy: 0.95,
            adjustment_sensitivity: 0.1,
            adjustment_interval: Duration::from_secs(30),
            stability_threshold: 0.05,
            adaptive_mode: true,
        }
    }
}

/// Rate adjustment record
///
/// Record of sample rate adjustments including reasons, effectiveness,
/// and performance impact.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateAdjustment {
    /// Adjustment timestamp
    pub timestamp: DateTime<Utc>,

    /// Previous rate
    pub previous_rate: f32,

    /// New rate
    pub new_rate: f32,

    /// Adjustment reason
    pub reason: String,

    /// Effectiveness score
    pub effectiveness: Option<f32>,

    /// Performance before adjustment
    pub performance_before: Option<PerformanceMeasurement>,

    /// Performance after adjustment
    pub performance_after: Option<PerformanceMeasurement>,
}

impl RateAdjustment {
    /// Create new rate adjustment
    pub fn new(previous_rate: f32, new_rate: f32, reason: String) -> Self {
        Self {
            timestamp: Utc::now(),
            previous_rate,
            new_rate,
            reason,
            effectiveness: None,
            performance_before: None,
            performance_after: None,
        }
    }

    /// Calculate adjustment magnitude
    pub fn adjustment_magnitude(&self) -> f32 {
        (self.new_rate - self.previous_rate).abs()
    }

    /// Get adjustment direction
    pub fn adjustment_direction(&self) -> &'static str {
        if self.new_rate > self.previous_rate {
            "increase"
        } else if self.new_rate < self.previous_rate {
            "decrease"
        } else {
            "no_change"
        }
    }
}

/// Rate controller statistics
///
/// Performance statistics for the sample rate controller including
/// adjustment frequency and effectiveness metrics.
#[derive(Debug, Default)]
pub struct RateControllerStats {
    /// Total adjustments made
    pub total_adjustments: AtomicU64,

    /// Average adjustment effectiveness
    pub avg_effectiveness: AtomicF32,

    /// Current rate stability
    pub rate_stability: AtomicF32,

    /// Controller accuracy
    pub accuracy: AtomicF32,

    /// Last adjustment timestamp
    pub last_adjustment: parking_lot::Mutex<Option<DateTime<Utc>>>,
}

impl RateControllerStats {
    /// Update effectiveness
    pub fn update_effectiveness(&self, effectiveness: f32) {
        let current_avg = self.avg_effectiveness.load(Ordering::Acquire);
        let new_avg = if current_avg == 0.0 {
            effectiveness
        } else {
            (current_avg * 0.9) + (effectiveness * 0.1)
        };
        self.avg_effectiveness.store(new_avg, Ordering::Release);
    }

    /// Get adjustment frequency (adjustments per hour)
    pub fn adjustment_frequency(&self) -> f64 {
        if let Some(last_adjustment) = *self.last_adjustment.lock() {
            let hours_since = (Utc::now() - last_adjustment).num_hours() as f64;
            if hours_since > 0.0 {
                self.total_adjustments.load(Ordering::Acquire) as f64 / hours_since
            } else {
                0.0
            }
        } else {
            0.0
        }
    }

    /// Record new adjustment
    pub fn record_adjustment(&self) {
        self.total_adjustments.fetch_add(1, Ordering::AcqRel);
        *self.last_adjustment.lock() = Some(Utc::now());
    }
}

/// Overhead measurement for impact monitoring
///
/// Measurement of performance overhead caused by metrics collection
/// and monitoring operations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OverheadMeasurement {
    /// Measurement timestamp
    pub timestamp: DateTime<Utc>,

    /// CPU overhead percentage
    pub cpu_overhead: f32,

    /// Memory overhead (bytes)
    pub memory_overhead: u64,

    /// I/O overhead
    pub io_overhead: f32,

    /// Network overhead
    pub network_overhead: f32,

    /// Collection latency overhead
    pub latency_overhead: Duration,

    /// Throughput impact
    pub throughput_impact: f32,
}

impl OverheadMeasurement {
    /// Create new overhead measurement
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            cpu_overhead: 0.0,
            memory_overhead: 0,
            io_overhead: 0.0,
            network_overhead: 0.0,
            latency_overhead: Duration::from_micros(0),
            throughput_impact: 0.0,
        }
    }

    /// Calculate overall overhead score
    pub fn overall_overhead(&self) -> f32 {
        (self.cpu_overhead * 0.3
            + (self.memory_overhead as f32 / 1_000_000.0) * 0.2
            + self.io_overhead * 0.2
            + self.network_overhead * 0.1
            + (self.latency_overhead.as_micros() as f32 / 1000.0) * 0.1
            + self.throughput_impact * 0.1)
            .min(100.0)
    }

    /// Check if overhead is acceptable
    pub fn is_acceptable(&self, max_cpu: f32, max_memory: u64, max_throughput_impact: f32) -> bool {
        self.cpu_overhead <= max_cpu
            && self.memory_overhead <= max_memory
            && self.throughput_impact <= max_throughput_impact
    }
}

impl Default for OverheadMeasurement {
    fn default() -> Self {
        Self::new()
    }
}

/// Impact analysis for performance monitoring
///
/// Analysis of the impact of monitoring operations on system performance.
#[derive(Debug, Clone)]
pub struct ImpactAnalysis {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,

    /// Overhead measurements
    pub overhead_measurements: Vec<OverheadMeasurement>,

    /// Performance baseline without monitoring
    pub baseline_without_monitoring: Option<PerformanceBaseline>,

    /// Performance baseline with monitoring
    pub baseline_with_monitoring: Option<PerformanceBaseline>,

    /// Impact severity
    pub impact_severity: SeverityLevel,

    /// Recommendations for impact reduction
    pub recommendations: Vec<String>,
}

impl ImpactAnalysis {
    /// Create new impact analysis
    pub fn new() -> Self {
        Self {
            timestamp: Utc::now(),
            overhead_measurements: Vec::new(),
            baseline_without_monitoring: None,
            baseline_with_monitoring: None,
            impact_severity: SeverityLevel::Info,
            recommendations: Vec::new(),
        }
    }

    /// Add overhead measurement
    pub fn add_measurement(&mut self, measurement: OverheadMeasurement) {
        self.overhead_measurements.push(measurement);
        self.update_severity();
    }

    /// Update impact severity based on measurements
    fn update_severity(&mut self) {
        if self.overhead_measurements.is_empty() {
            return;
        }

        let avg_overhead: f32 =
            self.overhead_measurements.iter().map(|m| m.overall_overhead()).sum::<f32>()
                / self.overhead_measurements.len() as f32;

        self.impact_severity = match avg_overhead {
            x if x >= 20.0 => SeverityLevel::Critical,
            x if x >= 10.0 => SeverityLevel::High,
            x if x >= 5.0 => SeverityLevel::Medium,
            x if x >= 2.0 => SeverityLevel::Low,
            _ => SeverityLevel::Info,
        };
    }

    /// Get average overhead
    pub fn average_overhead(&self) -> f32 {
        if self.overhead_measurements.is_empty() {
            0.0
        } else {
            self.overhead_measurements.iter().map(|m| m.overall_overhead()).sum::<f32>()
                / self.overhead_measurements.len() as f32
        }
    }
}

impl Default for ImpactAnalysis {
    fn default() -> Self {
        Self::new()
    }
}

/// Impact monitor configuration
///
/// Configuration for monitoring the impact of metrics collection on system performance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactMonitorConfig {
    /// Impact monitoring enabled
    pub enabled: bool,

    /// Monitoring interval
    pub monitoring_interval: Duration,

    /// Maximum acceptable CPU overhead
    pub max_cpu_overhead: f32,

    /// Maximum acceptable memory overhead
    pub max_memory_overhead: u64,

    /// Maximum acceptable throughput impact
    pub max_throughput_impact: f32,

    /// Alert on high impact
    pub alert_on_high_impact: bool,

    /// Auto-adjustment enabled
    pub auto_adjustment: bool,
}

impl Default for ImpactMonitorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval: Duration::from_secs(60),
            max_cpu_overhead: 5.0,
            max_memory_overhead: 100_000_000, // 100MB
            max_throughput_impact: 2.0,
            alert_on_high_impact: true,
            auto_adjustment: true,
        }
    }
}

/// Impact alert for high overhead conditions
///
/// Alert generated when monitoring overhead exceeds acceptable thresholds.
#[derive(Debug, Clone)]
pub struct ImpactAlert {
    /// Alert timestamp
    pub timestamp: DateTime<Utc>,

    /// Alert ID
    pub alert_id: String,

    /// Impact measurement
    pub measurement: OverheadMeasurement,

    /// Alert severity
    pub severity: SeverityLevel,

    /// Alert message
    pub message: String,

    /// Recommended actions
    pub actions: Vec<String>,

    /// Alert metadata
    pub metadata: HashMap<String, String>,
}

impl ImpactAlert {
    /// Create new impact alert
    pub fn new(
        alert_id: String,
        measurement: OverheadMeasurement,
        severity: SeverityLevel,
        message: String,
    ) -> Self {
        Self {
            timestamp: Utc::now(),
            alert_id,
            measurement,
            severity,
            message,
            actions: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// Add recommended action
    pub fn add_action(&mut self, action: String) {
        self.actions.push(action);
    }

    /// Check if alert requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self.severity, SeverityLevel::High | SeverityLevel::Critical)
    }
}

// ============================================================================
// Streaming Aggregation Types
// ============================================================================

/// Anomaly tracker for detecting anomalous data patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyTracker {
    /// Anomaly detection enabled
    pub enabled: bool,
    /// Anomaly threshold
    pub threshold: f64,
    /// Detected anomalies count
    pub anomalies_detected: u64,
}

impl Default for AnomalyTracker {
    fn default() -> Self {
        Self {
            enabled: true,
            threshold: 3.0,
            anomalies_detected: 0,
        }
    }
}

/// Stream statistics for real-time data streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamStatistics {
    /// Total items processed
    pub items_processed: u64,
    /// Processing rate (items/sec)
    pub processing_rate: f64,
    /// Average latency
    pub avg_latency: Duration,
    /// Stream start time
    pub stream_start: DateTime<Utc>,
}

impl Default for StreamStatistics {
    fn default() -> Self {
        Self {
            items_processed: 0,
            processing_rate: 0.0,
            avg_latency: Duration::from_secs(0),
            stream_start: Utc::now(),
        }
    }
}

/// Backpressure controller for stream flow control
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackpressureController {
    /// Backpressure enabled
    pub enabled: bool,
    /// Current backpressure level (0.0-1.0)
    pub pressure_level: f64,
    /// Buffer high watermark
    pub high_watermark: usize,
    /// Buffer low watermark
    pub low_watermark: usize,
}

impl Default for BackpressureController {
    fn default() -> Self {
        Self {
            enabled: true,
            pressure_level: 0.0,
            high_watermark: 10000,
            low_watermark: 1000,
        }
    }
}

/// Flow controller for managing data flow rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlowController {
    /// Target throughput (items/sec)
    pub target_throughput: f64,
    /// Current throughput (items/sec)
    pub current_throughput: f64,
    /// Throttling enabled
    pub throttling_enabled: bool,
}

impl Default for FlowController {
    fn default() -> Self {
        Self {
            target_throughput: 1000.0,
            current_throughput: 0.0,
            throttling_enabled: false,
        }
    }
}

/// Publishing statistics for result publication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PublishingStatistics {
    /// Total results published
    pub results_published: u64,
    /// Publishing rate (results/sec)
    pub publishing_rate: f64,
    /// Failed publications
    pub failed_publications: u64,
    /// Average publish latency
    pub avg_publish_latency: Duration,
}

impl Default for PublishingStatistics {
    fn default() -> Self {
        Self {
            results_published: 0,
            publishing_rate: 0.0,
            failed_publications: 0,
            avg_publish_latency: Duration::from_secs(0),
        }
    }
}

impl PublishingStatistics {
    /// Create new publishing statistics with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Result formatter for formatting aggregation results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultFormatter {
    /// Output format
    pub format: String,
    /// Include metadata
    pub include_metadata: bool,
    /// Precision for floating point values
    pub precision: u8,
}

impl Default for ResultFormatter {
    fn default() -> Self {
        Self {
            format: "json".to_string(),
            include_metadata: true,
            precision: 6,
        }
    }
}

impl ResultFormatter {
    /// Create new result formatter with default values
    pub fn new() -> Self {
        Self::default()
    }
}

// ============================================================================
// Configuration Types
// ============================================================================

/// Coordination configuration for multi-aggregator coordination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinationConfig {
    /// Coordination enabled
    pub enabled: bool,
    /// Coordination protocol
    pub protocol: String,
    /// Sync interval
    pub sync_interval: Duration,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            protocol: "raft".to_string(),
            sync_interval: Duration::from_secs(5),
        }
    }
}

/// Compression configuration for data compression
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Compression enabled
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: Compression,
    /// Compression level (1-9)
    pub level: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: Compression::Gzip,
            level: 6,
        }
    }
}

/// Compression statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionStatistics {
    /// Total bytes compressed
    pub bytes_compressed: u64,
    /// Original size
    pub original_size: u64,
    /// Compressed size
    pub compressed_size: u64,
    /// Compression ratio
    pub compression_ratio: f64,
}

impl Default for CompressionStatistics {
    fn default() -> Self {
        Self {
            bytes_compressed: 0,
            original_size: 0,
            compressed_size: 0,
            compression_ratio: 1.0,
        }
    }
}

impl CompressionStatistics {
    /// Create new compression statistics with default values
    pub fn new() -> Self {
        Self::default()
    }
}

/// Compression algorithm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Compression {
    /// No compression
    None,
    /// Gzip compression
    Gzip,
    /// Zstd compression
    Zstd,
    /// LZ4 compression
    Lz4,
}

/// Recommendation type for optimization recommendations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RecommendationType {
    /// Increase parallelism
    IncreaseParallelism { target_level: usize },
    /// Decrease parallelism
    DecreaseParallelism { target_level: usize },
    /// Adjust buffer size
    AdjustBuffer,
    /// Enable compression
    EnableCompression,
    /// Disable compression
    DisableCompression,
    /// Scale resources
    ScaleResources,
}

/// Confidence calculation method
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfidenceMethod {
    /// Bootstrap confidence intervals
    Bootstrap,
    /// T-distribution method
    TDistribution,
    /// Normal distribution method
    Normal,
    /// Non-parametric method
    NonParametric,
    /// Bayesian confidence intervals
    Bayesian,
    /// Frequentist confidence intervals
    Frequentist,
    /// Simple standard error
    StandardError,
}

/// Pipeline stage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipelineStageStats {
    /// Stage name
    pub stage_name: String,
    /// Items processed
    pub items_processed: u64,
    /// Processing time
    pub processing_time: Duration,
    /// Throughput (items/sec)
    pub throughput: f64,
    /// Error count
    pub errors: u64,
}

impl Default for PipelineStageStats {
    fn default() -> Self {
        Self {
            stage_name: String::new(),
            items_processed: 0,
            processing_time: Duration::from_secs(0),
            throughput: 0.0,
            errors: 0,
        }
    }
}

// ============================================================================
// Aggregator Types
// ============================================================================

/// Streaming worker for parallel processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamingWorker {
    pub worker_id: usize,
    pub tasks_processed: u64,
    pub active: bool,
}

/// Formatted aggregation result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FormattedResult {
    pub format: String,
    pub data: String,
    pub timestamp: DateTime<Utc>,
}

/// Histogram bin
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistogramBin {
    pub lower_bound: f64,
    pub upper_bound: f64,
    pub count: u64,
    pub frequency: f64,
}

/// Validation rule for data quality
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationRule {
    pub rule_name: String,
    pub rule_type: String,
    pub parameters: std::collections::HashMap<String, String>,
    pub severity: String,
}

/// Basic statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BasicStatistics {
    pub count: u64,
    pub sum: f64,
    pub mean: f64,
    pub min: f64,
    pub max: f64,
    pub std_dev: f64,
}

/// Advanced statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AdvancedStatistics {
    pub skewness: f64,
    pub kurtosis: f64,
    pub percentiles: std::collections::HashMap<String, f64>,
    pub variance: f64,
    pub median: f64,
}

/// Stage metrics for pipeline stages
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageMetrics {
    pub stage_name: String,
    pub duration: Duration,
    pub throughput: f64,
    pub success_rate: f64,
}

/// Quality criteria for data validation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityCriteria {
    pub completeness_threshold: f64,
    pub accuracy_threshold: f64,
    pub consistency_threshold: f64,
}

/// Outlier detection result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutlierResult {
    pub is_outlier: bool,
    pub score: f64,
    pub method: String,
    pub confidence: f64,
}

/// Outlier detection parameters
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutlierParameters {
    pub method: String,
    pub threshold: f64,
    pub window_size: usize,
}

/// Compressed data container
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    pub algorithm: String,
    pub data: Vec<u8>,
    pub original_size: usize,
    pub compressed_size: usize,
}

/// Aggregator performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatorPerformanceMetrics {
    pub throughput: f64,
    pub latency_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_usage_percent: f64,
    pub processing_rate: f64,
    pub processing_latency_micros: f64,
    pub queue_depth: usize,
    /// Memory usage in bytes
    pub memory_usage_bytes: u64,
    /// Total error count
    pub error_count: usize,
    /// Number of windows processed
    pub window_count: usize,
}

/// Aggregation metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregationMetadata {
    pub aggregation_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub data_points: u64,
    pub aggregation_method: String,
    pub version: u64,
    pub processor_versions: Vec<String>,
    pub quality_checks_performed: Vec<String>,
    pub statistical_methods_used: Vec<String>,
    pub data_source_info: HashMap<String, String>,
}

impl Default for ValidationRule {
    fn default() -> Self {
        Self {
            rule_name: String::new(),
            rule_type: String::new(),
            parameters: std::collections::HashMap::new(),
            severity: "medium".to_string(),
        }
    }
}

impl Default for HistogramBin {
    fn default() -> Self {
        Self {
            lower_bound: 0.0,
            upper_bound: 0.0,
            count: 0,
            frequency: 0.0,
        }
    }
}

impl Default for CompressedData {
    fn default() -> Self {
        Self {
            algorithm: "none".to_string(),
            data: Vec::new(),
            original_size: 0,
            compressed_size: 0,
        }
    }
}
