//! Port Performance Metrics Module for TrustformeRS
//!
//! This module provides comprehensive performance tracking and analysis capabilities
//! for network port management operations. It includes advanced metrics collection,
//! statistical analysis, trend tracking, and performance reporting features.
//!
//! # Features
//!
//! - **Real-time Metrics Collection**: Track allocation/deallocation operations in real-time
//! - **Performance Snapshots**: Create point-in-time performance snapshots
//! - **Statistical Analysis**: Compute averages, percentiles, and performance trends
//! - **Historical Tracking**: Maintain performance history for trend analysis
//! - **Performance Reporting**: Generate detailed performance reports and analytics
//! - **Configurable Monitoring**: Flexible configuration for different monitoring needs
//! - **Thread-Safe Operations**: All operations are thread-safe for concurrent access
//! - **Memory Efficient**: Efficient storage and retrieval of performance data
//!
//! # Usage
//!
//! ```rust
//! use trustformers_serve::resource_management::port_manager::performance_metrics::PortPerformanceMetrics;
//!
//! # async fn example() -> anyhow::Result<()> {
//! // Create a new performance metrics tracker
//! let metrics = PortPerformanceMetrics::new().await?;
//!
//! // Record performance metrics
//! metrics.record_allocation_success(std::time::Duration::from_millis(25)).await;
//! metrics.record_conflict().await;
//!
//! // Get current performance snapshot
//! let snapshot = metrics.get_current_snapshot().await;
//! println!("Average allocation time: {:.2}ms", snapshot.avg_allocation_time_ms);
//!
//! // Generate detailed performance report
//! let report = metrics.generate_performance_report().await;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```

use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::{debug, info, instrument, warn};

use super::types::*;

/// Performance metrics collection for port management
///
/// This struct provides comprehensive tracking of port management performance
/// including allocation times, success rates, conflict rates, and historical trends.
/// All operations are thread-safe and designed for high-concurrency environments.
#[derive(Debug)]
pub struct PortPerformanceMetrics {
    /// Total allocations performed
    total_allocations: AtomicU64,

    /// Total deallocations performed
    total_deallocations: AtomicU64,

    /// Total reservations made
    total_reservations: AtomicU64,

    /// Total conflicts detected
    total_conflicts: AtomicU64,

    /// Total allocation failures
    total_allocation_failures: AtomicU64,

    /// Total deallocation failures
    total_deallocation_failures: AtomicU64,

    /// Cumulative allocation time in nanoseconds
    cumulative_allocation_time_ns: AtomicU64,

    /// Cumulative deallocation time in nanoseconds
    cumulative_deallocation_time_ns: AtomicU64,

    /// Performance history with detailed snapshots
    performance_history: Arc<Mutex<Vec<PerformanceSnapshot>>>,

    /// Performance configuration
    config: Arc<RwLock<PerformanceConfig>>,

    /// Metrics collection start time for rate calculations
    start_time: DateTime<Utc>,

    /// Last snapshot timestamp for interval calculations
    last_snapshot_time: Arc<Mutex<DateTime<Utc>>>,

    /// Detailed timing measurements for percentile calculations
    allocation_times_ns: Arc<Mutex<Vec<u64>>>,

    /// Deallocation timing measurements
    deallocation_times_ns: Arc<Mutex<Vec<u64>>>,

    /// Performance trends analysis data
    performance_trends: Arc<Mutex<PerformanceTrends>>,
}

/// Performance snapshot for historical analysis and reporting
///
/// Contains comprehensive performance metrics captured at a specific point in time,
/// including operational rates, timing statistics, and system utilization data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// Operations per second during this period
    pub ops_per_second: f64,

    /// Average allocation time in milliseconds
    pub avg_allocation_time_ms: f64,

    /// Median allocation time in milliseconds
    pub median_allocation_time_ms: f64,

    /// 95th percentile allocation time in milliseconds
    pub p95_allocation_time_ms: f64,

    /// 99th percentile allocation time in milliseconds
    pub p99_allocation_time_ms: f64,

    /// Average deallocation time in milliseconds
    pub avg_deallocation_time_ms: f64,

    /// Success rate percentage for allocations
    pub success_rate_percent: f64,

    /// Conflict rate percentage
    pub conflict_rate_percent: f64,

    /// Utilization percentage at snapshot time
    pub utilization_percent: f32,

    /// Throughput in operations per minute
    pub throughput_ops_per_minute: f64,

    /// Total allocations at snapshot time
    pub total_allocations: u64,

    /// Total deallocations at snapshot time
    pub total_deallocations: u64,

    /// Total conflicts at snapshot time
    pub total_conflicts: u64,

    /// Additional custom metrics
    pub metrics: HashMap<String, f64>,
}

/// Configuration for performance monitoring and collection
///
/// Provides fine-grained control over performance monitoring behavior,
/// data retention policies, and analysis capabilities.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceConfig {
    /// Enable performance monitoring
    pub enabled: bool,

    /// Snapshot collection interval
    pub snapshot_interval: Duration,

    /// Number of snapshots to keep in history
    pub history_size: usize,

    /// Enable detailed timing measurements
    pub enable_detailed_timing: bool,

    /// Enable percentile calculations
    pub enable_percentile_tracking: bool,

    /// Maximum number of timing samples to keep for percentiles
    pub max_timing_samples: usize,

    /// Enable trend analysis
    pub enable_trend_analysis: bool,

    /// Trend analysis window size in snapshots
    pub trend_window_size: usize,

    /// Enable memory optimization for large datasets
    pub enable_memory_optimization: bool,

    /// Automatic cleanup interval for old data
    pub cleanup_interval: Duration,

    /// Alert thresholds for performance degradation
    pub alert_thresholds: PerformanceAlertThresholds,
}

impl Default for PerformanceConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            snapshot_interval: Duration::from_secs(300), // 5 minutes
            history_size: 288, // 24 hours worth of 5-minute snapshots
            enable_detailed_timing: true,
            enable_percentile_tracking: true,
            max_timing_samples: 10000,
            enable_trend_analysis: true,
            trend_window_size: 12, // 1 hour worth of 5-minute snapshots
            enable_memory_optimization: true,
            cleanup_interval: Duration::from_secs(3600), // 1 hour
            alert_thresholds: PerformanceAlertThresholds::default(),
        }
    }
}

/// Alert thresholds for performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlertThresholds {
    /// Maximum acceptable average allocation time (ms)
    pub max_avg_allocation_time_ms: f64,

    /// Maximum acceptable P95 allocation time (ms)
    pub max_p95_allocation_time_ms: f64,

    /// Minimum acceptable success rate (%)
    pub min_success_rate_percent: f64,

    /// Maximum acceptable conflict rate (%)
    pub max_conflict_rate_percent: f64,

    /// Minimum acceptable throughput (ops/min)
    pub min_throughput_ops_per_minute: f64,
}

impl Default for PerformanceAlertThresholds {
    fn default() -> Self {
        Self {
            max_avg_allocation_time_ms: 100.0,
            max_p95_allocation_time_ms: 500.0,
            min_success_rate_percent: 95.0,
            max_conflict_rate_percent: 5.0,
            min_throughput_ops_per_minute: 60.0,
        }
    }
}

/// Performance trends analysis data
#[derive(Debug, Clone)]
pub struct PerformanceTrends {
    /// Allocation time trend (positive = increasing, negative = decreasing)
    pub allocation_time_trend: f64,

    /// Success rate trend
    pub success_rate_trend: f64,

    /// Throughput trend
    pub throughput_trend: f64,

    /// Conflict rate trend
    pub conflict_rate_trend: f64,

    /// Overall performance score (0-100, higher is better)
    pub overall_performance_score: f64,

    /// Last update timestamp
    pub last_updated: DateTime<Utc>,
}

impl Default for PerformanceTrends {
    fn default() -> Self {
        Self {
            allocation_time_trend: 0.0,
            success_rate_trend: 0.0,
            throughput_trend: 0.0,
            conflict_rate_trend: 0.0,
            overall_performance_score: 100.0,
            last_updated: Utc::now(),
        }
    }
}

/// Performance analysis result with insights and recommendations
#[derive(Debug, Clone)]
pub struct PerformanceAnalysis {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,

    /// Overall system performance grade (A-F)
    pub performance_grade: PerformanceGrade,

    /// Key performance insights
    pub insights: Vec<PerformanceInsight>,

    /// Performance recommendations
    pub recommendations: Vec<PerformanceRecommendation>,

    /// Predicted future performance
    pub performance_forecast: PerformanceForecast,

    /// Anomaly detection results
    pub anomalies: Vec<PerformanceAnomaly>,
}

/// Performance grade classification
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PerformanceGrade {
    Excellent, // A
    Good,      // B
    Fair,      // C
    Poor,      // D
    Critical,  // F
}

/// Performance insight with detailed analysis
#[derive(Debug, Clone)]
pub struct PerformanceInsight {
    /// Insight category
    pub category: InsightCategory,

    /// Insight description
    pub description: String,

    /// Impact level
    pub impact: ImpactLevel,

    /// Supporting metrics
    pub metrics: HashMap<String, f64>,
}

/// Categories for performance insights
#[derive(Debug, Clone)]
pub enum InsightCategory {
    AllocationPerformance,
    ConflictPatterns,
    ThroughputTrends,
    ResourceUtilization,
    SystemStability,
}

/// Impact level classification
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ImpactLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance recommendation
#[derive(Debug, Clone)]
pub struct PerformanceRecommendation {
    /// Recommendation type
    pub recommendation_type: RecommendationType,

    /// Recommendation description
    pub description: String,

    /// Expected impact
    pub expected_impact: ImpactLevel,

    /// Implementation priority
    pub priority: Priority,

    /// Estimated effort required
    pub effort_estimate: EffortEstimate,
}

/// Types of performance recommendations
#[derive(Debug, Clone)]
pub enum RecommendationType {
    ConfigurationTuning,
    ResourceAllocation,
    AlgorithmOptimization,
    CapacityPlanning,
    SystemUpgrade,
    ProcessImprovement,
}

/// Priority levels for recommendations
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Low,
    Medium,
    High,
    Urgent,
}

/// Effort estimation for recommendations
#[derive(Debug, Clone)]
pub enum EffortEstimate {
    Low,    // Hours
    Medium, // Days
    High,   // Weeks
    Major,  // Months
}

/// Performance forecast data
#[derive(Debug, Clone)]
pub struct PerformanceForecast {
    /// Forecast period in hours
    pub forecast_hours: u32,

    /// Predicted performance metrics
    pub predicted_metrics: HashMap<String, f64>,

    /// Confidence level (0-100)
    pub confidence_level: f64,

    /// Forecast method used
    pub forecast_method: String,
}

/// Performance anomaly detection result
#[derive(Debug, Clone)]
pub struct PerformanceAnomaly {
    /// Anomaly timestamp
    pub timestamp: DateTime<Utc>,

    /// Anomaly type
    pub anomaly_type: AnomalyType,

    /// Anomaly description
    pub description: String,

    /// Severity level
    pub severity: ImpactLevel,

    /// Affected metrics
    pub affected_metrics: Vec<String>,

    /// Anomaly score (0-100, higher = more anomalous)
    pub anomaly_score: f64,
}

/// Types of performance anomalies
#[derive(Debug, Clone)]
pub enum AnomalyType {
    SuddenPerformanceDrop,
    UnusualAllocationPattern,
    ConflictSpike,
    ThroughputDegradation,
    MemoryLeakSuspicion,
    ResourceContention,
}

impl PortPerformanceMetrics {
    /// Create a new performance metrics tracker with comprehensive monitoring capabilities
    ///
    /// # Returns
    ///
    /// A new PortPerformanceMetrics instance ready for performance tracking
    ///
    /// # Errors
    ///
    /// Returns an error if initialization fails
    #[instrument]
    pub async fn new() -> PortManagementResult<Self> {
        let now = Utc::now();

        Ok(Self {
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
        })
    }

    /// Record a successful allocation with detailed timing
    ///
    /// # Arguments
    ///
    /// * `allocation_time` - Duration of the allocation operation
    #[instrument(skip(self))]
    pub async fn record_allocation_success(&self, allocation_time: Duration) {
        self.total_allocations.fetch_add(1, Ordering::Relaxed);
        let time_ns = allocation_time.as_nanos() as u64;
        self.cumulative_allocation_time_ns.fetch_add(time_ns, Ordering::Relaxed);

        let config = self.config.read();
        if config.enable_detailed_timing {
            let mut times = self.allocation_times_ns.lock();
            times.push(time_ns);

            // Memory optimization: keep only the most recent samples
            if config.enable_memory_optimization && times.len() > config.max_timing_samples {
                let keep_count = config.max_timing_samples * 3 / 4; // Keep 75% when cleaning up
                times.drain(0..times.len() - keep_count);
            }
        }

        debug!("Recorded allocation success: {:?}", allocation_time);
    }

    /// Record an allocation failure
    #[instrument(skip(self))]
    pub async fn record_allocation_failure(&self) {
        self.total_allocation_failures.fetch_add(1, Ordering::Relaxed);
        debug!("Recorded allocation failure");
    }

    /// Record a successful deallocation with timing
    ///
    /// # Arguments
    ///
    /// * `deallocation_time` - Duration of the deallocation operation
    #[instrument(skip(self))]
    pub async fn record_deallocation_success(&self, deallocation_time: Duration) {
        self.total_deallocations.fetch_add(1, Ordering::Relaxed);
        let time_ns = deallocation_time.as_nanos() as u64;
        self.cumulative_deallocation_time_ns.fetch_add(time_ns, Ordering::Relaxed);

        let config = self.config.read();
        if config.enable_detailed_timing {
            let mut times = self.deallocation_times_ns.lock();
            times.push(time_ns);

            // Memory optimization
            if config.enable_memory_optimization && times.len() > config.max_timing_samples {
                let keep_count = config.max_timing_samples * 3 / 4;
                times.drain(0..times.len() - keep_count);
            }
        }

        debug!("Recorded deallocation success: {:?}", deallocation_time);
    }

    /// Record a deallocation failure
    #[instrument(skip(self))]
    pub async fn record_deallocation_failure(&self) {
        self.total_deallocation_failures.fetch_add(1, Ordering::Relaxed);
        debug!("Recorded deallocation failure");
    }

    /// Record a conflict event
    #[instrument(skip(self))]
    pub async fn record_conflict(&self) {
        self.total_conflicts.fetch_add(1, Ordering::Relaxed);
        debug!("Recorded conflict event");
    }

    /// Record a reservation event
    #[instrument(skip(self))]
    pub async fn record_reservation(&self) {
        self.total_reservations.fetch_add(1, Ordering::Relaxed);
        debug!("Recorded reservation event");
    }

    /// Get current performance snapshot with comprehensive metrics
    ///
    /// # Returns
    ///
    /// Current performance snapshot with detailed metrics
    #[instrument(skip(self))]
    pub async fn get_current_snapshot(&self) -> PerformanceSnapshot {
        let total_allocations = self.total_allocations.load(Ordering::Relaxed);
        let total_deallocations = self.total_deallocations.load(Ordering::Relaxed);
        let total_conflicts = self.total_conflicts.load(Ordering::Relaxed);
        let total_failures = self.total_allocation_failures.load(Ordering::Relaxed);
        let total_time_ns = self.cumulative_allocation_time_ns.load(Ordering::Relaxed);
        let dealloc_time_ns = self.cumulative_deallocation_time_ns.load(Ordering::Relaxed);

        let now = Utc::now();
        let elapsed_seconds = (now - self.start_time).num_seconds() as f64;

        // Calculate basic metrics
        let avg_allocation_time_ms = if total_allocations > 0 {
            (total_time_ns as f64 / total_allocations as f64) / 1_000_000.0
        } else {
            0.0
        };

        let avg_deallocation_time_ms = if total_deallocations > 0 {
            (dealloc_time_ns as f64 / total_deallocations as f64) / 1_000_000.0
        } else {
            0.0
        };

        let success_rate_percent = if total_allocations + total_failures > 0 {
            (total_allocations as f64 / (total_allocations + total_failures) as f64) * 100.0
        } else {
            100.0
        };

        let conflict_rate_percent = if total_allocations > 0 {
            (total_conflicts as f64 / total_allocations as f64) * 100.0
        } else {
            0.0
        };

        let ops_per_second = if elapsed_seconds > 0.0 {
            (total_allocations + total_deallocations) as f64 / elapsed_seconds
        } else {
            0.0
        };

        let throughput_ops_per_minute = ops_per_second * 60.0;

        // Calculate percentiles if detailed timing is enabled
        let (median_allocation_time_ms, p95_allocation_time_ms, p99_allocation_time_ms) =
            self.calculate_timing_percentiles().await;

        let mut metrics = HashMap::new();
        metrics.insert("total_operations".to_string(), (total_allocations + total_deallocations) as f64);
        metrics.insert("failure_rate_percent".to_string(),
            if total_allocations + total_failures > 0 {
                (total_failures as f64 / (total_allocations + total_failures) as f64) * 100.0
            } else {
                0.0
            }
        );

        PerformanceSnapshot {
            timestamp: now,
            ops_per_second,
            avg_allocation_time_ms,
            median_allocation_time_ms,
            p95_allocation_time_ms,
            p99_allocation_time_ms,
            avg_deallocation_time_ms,
            success_rate_percent,
            conflict_rate_percent,
            utilization_percent: 0.0, // Would be provided from external source
            throughput_ops_per_minute,
            total_allocations,
            total_deallocations,
            total_conflicts,
            metrics,
        }
    }

    /// Calculate timing percentiles from detailed measurements
    async fn calculate_timing_percentiles(&self) -> (f64, f64, f64) {
        let config = self.config.read();
        if !config.enable_percentile_tracking {
            return (0.0, 0.0, 0.0);
        }

        let times = self.allocation_times_ns.lock();
        if times.is_empty() {
            return (0.0, 0.0, 0.0);
        }

        let mut sorted_times: Vec<u64> = times.clone();
        sorted_times.sort_unstable();

        let len = sorted_times.len();
        let median_idx = len / 2;
        let p95_idx = (len as f64 * 0.95) as usize;
        let p99_idx = (len as f64 * 0.99) as usize;

        let median = if len % 2 == 0 && len > 1 {
            (sorted_times[median_idx - 1] + sorted_times[median_idx]) as f64 / 2.0
        } else {
            sorted_times[median_idx] as f64
        };

        let p95 = sorted_times[p95_idx.min(len - 1)] as f64;
        let p99 = sorted_times[p99_idx.min(len - 1)] as f64;

        (
            median / 1_000_000.0, // Convert to milliseconds
            p95 / 1_000_000.0,
            p99 / 1_000_000.0,
        )
    }

    /// Create and store a performance snapshot
    ///
    /// # Returns
    ///
    /// The created snapshot
    #[instrument(skip(self))]
    pub async fn create_snapshot(&self) -> PerformanceSnapshot {
        let snapshot = self.get_current_snapshot().await;

        let config = self.config.read();
        let mut history = self.performance_history.lock();

        history.push(snapshot.clone());

        // Maintain history size limit
        if history.len() > config.history_size {
            let remove_count = history.len() - config.history_size;
            history.drain(0..remove_count);
        }

        // Update last snapshot time
        *self.last_snapshot_time.lock() = snapshot.timestamp;

        // Update trends if enabled
        if config.enable_trend_analysis {
            self.update_performance_trends(&history, &config).await;
        }

        info!("Created performance snapshot with {:.2}ms avg allocation time",
              snapshot.avg_allocation_time_ms);

        snapshot
    }

    /// Update performance trends analysis
    async fn update_performance_trends(&self, history: &[PerformanceSnapshot], config: &PerformanceConfig) {
        if history.len() < 2 {
            return;
        }

        let window_size = config.trend_window_size.min(history.len());
        let recent_snapshots = &history[history.len() - window_size..];

        if recent_snapshots.len() < 2 {
            return;
        }

        // Calculate trends using linear regression
        let allocation_time_trend = self.calculate_trend(
            &recent_snapshots.iter().map(|s| s.avg_allocation_time_ms).collect::<Vec<_>>()
        );

        let success_rate_trend = self.calculate_trend(
            &recent_snapshots.iter().map(|s| s.success_rate_percent).collect::<Vec<_>>()
        );

        let throughput_trend = self.calculate_trend(
            &recent_snapshots.iter().map(|s| s.throughput_ops_per_minute).collect::<Vec<_>>()
        );

        let conflict_rate_trend = self.calculate_trend(
            &recent_snapshots.iter().map(|s| s.conflict_rate_percent).collect::<Vec<_>>()
        );

        // Calculate overall performance score (0-100)
        let latest = &recent_snapshots[recent_snapshots.len() - 1];
        let performance_score = self.calculate_performance_score(latest);

        let mut trends = self.performance_trends.lock();
        trends.allocation_time_trend = allocation_time_trend;
        trends.success_rate_trend = success_rate_trend;
        trends.throughput_trend = throughput_trend;
        trends.conflict_rate_trend = conflict_rate_trend;
        trends.overall_performance_score = performance_score;
        trends.last_updated = Utc::now();
    }

    /// Calculate trend using simple linear regression
    fn calculate_trend(&self, values: &[f64]) -> f64 {
        if values.len() < 2 {
            return 0.0;
        }

        let n = values.len() as f64;
        let x_sum: f64 = (0..values.len()).map(|i| i as f64).sum();
        let y_sum: f64 = values.iter().sum();
        let xy_sum: f64 = values.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
        let x2_sum: f64 = (0..values.len()).map(|i| (i as f64).powi(2)).sum();

        let denominator = n * x2_sum - x_sum * x_sum;
        if denominator.abs() < f64::EPSILON {
            return 0.0;
        }

        (n * xy_sum - x_sum * y_sum) / denominator
    }

    /// Calculate overall performance score
    fn calculate_performance_score(&self, snapshot: &PerformanceSnapshot) -> f64 {
        let config = self.config.read();
        let thresholds = &config.alert_thresholds;

        let mut score = 100.0;

        // Allocation time score (0-25 points)
        if snapshot.avg_allocation_time_ms > thresholds.max_avg_allocation_time_ms {
            let ratio = snapshot.avg_allocation_time_ms / thresholds.max_avg_allocation_time_ms;
            score -= (ratio - 1.0) * 25.0;
        }

        // Success rate score (0-25 points)
        if snapshot.success_rate_percent < thresholds.min_success_rate_percent {
            let diff = thresholds.min_success_rate_percent - snapshot.success_rate_percent;
            score -= diff * 0.25; // 1 point per 1% below threshold
        }

        // Conflict rate score (0-25 points)
        if snapshot.conflict_rate_percent > thresholds.max_conflict_rate_percent {
            let ratio = snapshot.conflict_rate_percent / thresholds.max_conflict_rate_percent;
            score -= (ratio - 1.0) * 25.0;
        }

        // Throughput score (0-25 points)
        if snapshot.throughput_ops_per_minute < thresholds.min_throughput_ops_per_minute {
            let ratio = snapshot.throughput_ops_per_minute / thresholds.min_throughput_ops_per_minute;
            score -= (1.0 - ratio) * 25.0;
        }

        score.clamp(0.0, 100.0)
    }

    /// Get performance history
    ///
    /// # Returns
    ///
    /// Vector of historical performance snapshots
    pub async fn get_performance_history(&self) -> Vec<PerformanceSnapshot> {
        let history = self.performance_history.lock();
        history.clone()
    }

    /// Get performance trends
    ///
    /// # Returns
    ///
    /// Current performance trends analysis
    pub async fn get_performance_trends(&self) -> PerformanceTrends {
        let trends = self.performance_trends.lock();
        trends.clone()
    }

    /// Generate comprehensive performance report
    ///
    /// # Returns
    ///
    /// Detailed performance report string
    #[instrument(skip(self))]
    pub async fn generate_performance_report(&self) -> String {
        let snapshot = self.get_current_snapshot().await;
        let trends = self.get_performance_trends().await;
        let elapsed = Utc::now() - self.start_time;

        format!(
            "Performance Report - Port Management System\n\
             ==========================================\n\
             Generated: {} UTC\n\
             Monitoring Duration: {} hours\n\n\
             Current Performance Metrics:\n\
             - Average Allocation Time: {:.2}ms\n\
             - Median Allocation Time: {:.2}ms\n\
             - P95 Allocation Time: {:.2}ms\n\
             - P99 Allocation Time: {:.2}ms\n\
             - Average Deallocation Time: {:.2}ms\n\
             - Success Rate: {:.1}%\n\
             - Conflict Rate: {:.1}%\n\
             - Throughput: {:.1} ops/min\n\
             - Operations per Second: {:.2}\n\n\
             Cumulative Statistics:\n\
             - Total Allocations: {}\n\
             - Total Deallocations: {}\n\
             - Total Conflicts: {}\n\
             - Total Operations: {}\n\n\
             Performance Trends:\n\
             - Allocation Time Trend: {:.3}ms/snapshot\n\
             - Success Rate Trend: {:.3}%/snapshot\n\
             - Throughput Trend: {:.3} ops/min/snapshot\n\
             - Overall Performance Score: {:.1}/100\n\n\
             Performance Grade: {:?}\n\
             ==========================================",
            Utc::now().format("%Y-%m-%d %H:%M:%S"),
            elapsed.num_hours(),
            snapshot.avg_allocation_time_ms,
            snapshot.median_allocation_time_ms,
            snapshot.p95_allocation_time_ms,
            snapshot.p99_allocation_time_ms,
            snapshot.avg_deallocation_time_ms,
            snapshot.success_rate_percent,
            snapshot.conflict_rate_percent,
            snapshot.throughput_ops_per_minute,
            snapshot.ops_per_second,
            snapshot.total_allocations,
            snapshot.total_deallocations,
            snapshot.total_conflicts,
            snapshot.total_allocations + snapshot.total_deallocations,
            trends.allocation_time_trend,
            trends.success_rate_trend,
            trends.throughput_trend,
            trends.overall_performance_score,
            self.calculate_performance_grade(trends.overall_performance_score)
        )
    }

    /// Calculate performance grade based on score
    fn calculate_performance_grade(&self, score: f64) -> PerformanceGrade {
        match score {
            s if s >= 90.0 => PerformanceGrade::Excellent,
            s if s >= 80.0 => PerformanceGrade::Good,
            s if s >= 70.0 => PerformanceGrade::Fair,
            s if s >= 60.0 => PerformanceGrade::Poor,
            _ => PerformanceGrade::Critical,
        }
    }

    /// Perform comprehensive performance analysis
    ///
    /// # Returns
    ///
    /// Detailed performance analysis with insights and recommendations
    #[instrument(skip(self))]
    pub async fn analyze_performance(&self) -> PerformanceAnalysis {
        let snapshot = self.get_current_snapshot().await;
        let trends = self.get_performance_trends().await;
        let history = self.get_performance_history().await;

        let performance_grade = self.calculate_performance_grade(trends.overall_performance_score);
        let insights = self.generate_performance_insights(&snapshot, &trends, &history).await;
        let recommendations = self.generate_performance_recommendations(&snapshot, &trends).await;
        let forecast = self.generate_performance_forecast(&history).await;
        let anomalies = self.detect_performance_anomalies(&history).await;

        PerformanceAnalysis {
            timestamp: Utc::now(),
            performance_grade,
            insights,
            recommendations,
            performance_forecast: forecast,
            anomalies,
        }
    }

    /// Generate performance insights
    async fn generate_performance_insights(
        &self,
        snapshot: &PerformanceSnapshot,
        trends: &PerformanceTrends,
        history: &[PerformanceSnapshot],
    ) -> Vec<PerformanceInsight> {
        let mut insights = Vec::new();
        let config = self.config.read();

        // Allocation performance insights
        if snapshot.avg_allocation_time_ms > config.alert_thresholds.max_avg_allocation_time_ms {
            let mut metrics = HashMap::new();
            metrics.insert("current_time_ms".to_string(), snapshot.avg_allocation_time_ms);
            metrics.insert("threshold_ms".to_string(), config.alert_thresholds.max_avg_allocation_time_ms);

            insights.push(PerformanceInsight {
                category: InsightCategory::AllocationPerformance,
                description: format!(
                    "Allocation time ({:.2}ms) exceeds threshold ({:.2}ms)",
                    snapshot.avg_allocation_time_ms,
                    config.alert_thresholds.max_avg_allocation_time_ms
                ),
                impact: if snapshot.avg_allocation_time_ms > config.alert_thresholds.max_avg_allocation_time_ms * 2.0 {
                    ImpactLevel::High
                } else {
                    ImpactLevel::Medium
                },
                metrics,
            });
        }

        // Trend insights
        if trends.allocation_time_trend > 1.0 {
            let mut metrics = HashMap::new();
            metrics.insert("trend_ms_per_snapshot".to_string(), trends.allocation_time_trend);

            insights.push(PerformanceInsight {
                category: InsightCategory::AllocationPerformance,
                description: "Allocation time is trending upward, indicating potential performance degradation".to_string(),
                impact: ImpactLevel::Medium,
                metrics,
            });
        }

        // Conflict pattern insights
        if snapshot.conflict_rate_percent > config.alert_thresholds.max_conflict_rate_percent {
            let mut metrics = HashMap::new();
            metrics.insert("current_rate_percent".to_string(), snapshot.conflict_rate_percent);
            metrics.insert("threshold_percent".to_string(), config.alert_thresholds.max_conflict_rate_percent);

            insights.push(PerformanceInsight {
                category: InsightCategory::ConflictPatterns,
                description: format!(
                    "Conflict rate ({:.1}%) is above threshold ({:.1}%)",
                    snapshot.conflict_rate_percent,
                    config.alert_thresholds.max_conflict_rate_percent
                ),
                impact: ImpactLevel::Medium,
                metrics,
            });
        }

        // Throughput insights
        if snapshot.throughput_ops_per_minute < config.alert_thresholds.min_throughput_ops_per_minute {
            let mut metrics = HashMap::new();
            metrics.insert("current_throughput".to_string(), snapshot.throughput_ops_per_minute);
            metrics.insert("threshold_throughput".to_string(), config.alert_thresholds.min_throughput_ops_per_minute);

            insights.push(PerformanceInsight {
                category: InsightCategory::ThroughputTrends,
                description: "System throughput is below expected levels".to_string(),
                impact: ImpactLevel::Medium,
                metrics,
            });
        }

        insights
    }

    /// Generate performance recommendations
    async fn generate_performance_recommendations(
        &self,
        snapshot: &PerformanceSnapshot,
        trends: &PerformanceTrends,
    ) -> Vec<PerformanceRecommendation> {
        let mut recommendations = Vec::new();
        let config = self.config.read();

        // High allocation time recommendations
        if snapshot.avg_allocation_time_ms > config.alert_thresholds.max_avg_allocation_time_ms {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::AlgorithmOptimization,
                description: "Consider optimizing port allocation algorithms or implementing caching".to_string(),
                expected_impact: ImpactLevel::High,
                priority: Priority::High,
                effort_estimate: EffortEstimate::Medium,
            });
        }

        // High conflict rate recommendations
        if snapshot.conflict_rate_percent > config.alert_thresholds.max_conflict_rate_percent {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::ConfigurationTuning,
                description: "Increase port pool size or implement better conflict avoidance strategies".to_string(),
                expected_impact: ImpactLevel::Medium,
                priority: Priority::Medium,
                effort_estimate: EffortEstimate::Low,
            });
        }

        // Degrading trends
        if trends.allocation_time_trend > 1.0 && trends.throughput_trend < -1.0 {
            recommendations.push(PerformanceRecommendation {
                recommendation_type: RecommendationType::SystemUpgrade,
                description: "Performance is degrading over time, consider system capacity upgrade".to_string(),
                expected_impact: ImpactLevel::High,
                priority: Priority::High,
                effort_estimate: EffortEstimate::High,
            });
        }

        recommendations
    }

    /// Generate performance forecast
    async fn generate_performance_forecast(&self, history: &[PerformanceSnapshot]) -> PerformanceForecast {
        if history.len() < 3 {
            return PerformanceForecast {
                forecast_hours: 24,
                predicted_metrics: HashMap::new(),
                confidence_level: 0.0,
                forecast_method: "Insufficient data".to_string(),
            };
        }

        let mut predicted_metrics = HashMap::new();

        // Simple linear extrapolation for demonstration
        let recent_snapshots = &history[history.len().saturating_sub(10)..];
        let allocation_times: Vec<f64> = recent_snapshots.iter().map(|s| s.avg_allocation_time_ms).collect();
        let throughputs: Vec<f64> = recent_snapshots.iter().map(|s| s.throughput_ops_per_minute).collect();

        let allocation_trend = self.calculate_trend(&allocation_times);
        let throughput_trend = self.calculate_trend(&throughputs);

        // Predict 24 hours ahead (assuming 5-minute snapshots = 288 snapshots)
        let prediction_steps = 288.0;
        let current_allocation_time = allocation_times.last().copied().unwrap_or(0.0);
        let current_throughput = throughputs.last().copied().unwrap_or(0.0);

        predicted_metrics.insert(
            "avg_allocation_time_ms".to_string(),
            current_allocation_time + allocation_trend * prediction_steps,
        );
        predicted_metrics.insert(
            "throughput_ops_per_minute".to_string(),
            current_throughput + throughput_trend * prediction_steps,
        );

        // Confidence decreases with prediction distance and trend volatility
        let confidence_level = (100.0 - (allocation_trend.abs() * 10.0)).clamp(20.0, 95.0);

        PerformanceForecast {
            forecast_hours: 24,
            predicted_metrics,
            confidence_level,
            forecast_method: "Linear extrapolation".to_string(),
        }
    }

    /// Detect performance anomalies
    async fn detect_performance_anomalies(&self, history: &[PerformanceSnapshot]) -> Vec<PerformanceAnomaly> {
        let mut anomalies = Vec::new();

        if history.len() < 10 {
            return anomalies;
        }

        // Simple anomaly detection based on statistical thresholds
        let recent_window = &history[history.len().saturating_sub(10)..];
        let older_window = &history[history.len().saturating_sub(20)..history.len().saturating_sub(10)];

        if recent_window.is_empty() || older_window.is_empty() {
            return anomalies;
        }

        // Calculate averages
        let recent_avg_time: f64 = recent_window.iter().map(|s| s.avg_allocation_time_ms).sum::<f64>() / recent_window.len() as f64;
        let older_avg_time: f64 = older_window.iter().map(|s| s.avg_allocation_time_ms).sum::<f64>() / older_window.len() as f64;

        // Detect sudden performance drops
        if recent_avg_time > older_avg_time * 1.5 {
            anomalies.push(PerformanceAnomaly {
                timestamp: Utc::now(),
                anomaly_type: AnomalyType::SuddenPerformanceDrop,
                description: format!(
                    "Allocation time increased significantly: {:.2}ms -> {:.2}ms",
                    older_avg_time, recent_avg_time
                ),
                severity: ImpactLevel::High,
                affected_metrics: vec!["avg_allocation_time_ms".to_string()],
                anomaly_score: ((recent_avg_time - older_avg_time) / older_avg_time * 100.0).min(100.0),
            });
        }

        // Detect throughput degradation
        let recent_avg_throughput: f64 = recent_window.iter().map(|s| s.throughput_ops_per_minute).sum::<f64>() / recent_window.len() as f64;
        let older_avg_throughput: f64 = older_window.iter().map(|s| s.throughput_ops_per_minute).sum::<f64>() / older_window.len() as f64;

        if recent_avg_throughput < older_avg_throughput * 0.7 {
            anomalies.push(PerformanceAnomaly {
                timestamp: Utc::now(),
                anomaly_type: AnomalyType::ThroughputDegradation,
                description: format!(
                    "Throughput decreased significantly: {:.1} -> {:.1} ops/min",
                    older_avg_throughput, recent_avg_throughput
                ),
                severity: ImpactLevel::High,
                affected_metrics: vec!["throughput_ops_per_minute".to_string()],
                anomaly_score: ((older_avg_throughput - recent_avg_throughput) / older_avg_throughput * 100.0).min(100.0),
            });
        }

        anomalies
    }

    /// Update performance configuration
    ///
    /// # Arguments
    ///
    /// * `new_config` - New performance configuration
    #[instrument(skip(self, new_config))]
    pub async fn update_config(&self, new_config: PerformanceConfig) {
        let mut config = self.config.write();
        *config = new_config;
        info!("Updated performance metrics configuration");
    }

    /// Get current configuration
    ///
    /// # Returns
    ///
    /// Current performance configuration
    pub async fn get_config(&self) -> PerformanceConfig {
        let config = self.config.read();
        config.clone()
    }

    /// Reset all metrics and history
    ///
    /// Useful for testing or when starting fresh monitoring
    #[instrument(skip(self))]
    pub async fn reset_metrics(&self) {
        self.total_allocations.store(0, Ordering::Relaxed);
        self.total_deallocations.store(0, Ordering::Relaxed);
        self.total_reservations.store(0, Ordering::Relaxed);
        self.total_conflicts.store(0, Ordering::Relaxed);
        self.total_allocation_failures.store(0, Ordering::Relaxed);
        self.total_deallocation_failures.store(0, Ordering::Relaxed);
        self.cumulative_allocation_time_ns.store(0, Ordering::Relaxed);
        self.cumulative_deallocation_time_ns.store(0, Ordering::Relaxed);

        self.performance_history.lock().clear();
        self.allocation_times_ns.lock().clear();
        self.deallocation_times_ns.lock().clear();

        *self.performance_trends.lock() = PerformanceTrends::default();
        *self.last_snapshot_time.lock() = Utc::now();

        info!("Reset all performance metrics and history");
    }

    /// Cleanup old data based on configuration
    ///
    /// # Returns
    ///
    /// Number of items cleaned up
    #[instrument(skip(self))]
    pub async fn cleanup_old_data(&self) -> usize {
        let config = self.config.read();
        if !config.enable_memory_optimization {
            return 0;
        }

        let mut total_cleaned = 0;

        // Cleanup timing samples
        let mut allocation_times = self.allocation_times_ns.lock();
        if allocation_times.len() > config.max_timing_samples {
            let keep_count = config.max_timing_samples * 3 / 4;
            let removed = allocation_times.len() - keep_count;
            allocation_times.drain(0..removed);
            total_cleaned += removed;
        }

        let mut deallocation_times = self.deallocation_times_ns.lock();
        if deallocation_times.len() > config.max_timing_samples {
            let keep_count = config.max_timing_samples * 3 / 4;
            let removed = deallocation_times.len() - keep_count;
            deallocation_times.drain(0..removed);
            total_cleaned += removed;
        }

        // Cleanup history
        let mut history = self.performance_history.lock();
        if history.len() > config.history_size {
            let remove_count = history.len() - config.history_size;
            history.drain(0..remove_count);
            total_cleaned += remove_count;
        }

        if total_cleaned > 0 {
            info!("Cleaned up {} old performance data items", total_cleaned);
        }

        total_cleaned
    }

    /// Export performance data for external analysis
    ///
    /// # Returns
    ///
    /// JSON string containing all performance data
    #[instrument(skip(self))]
    pub async fn export_performance_data(&self) -> Result<String> {
        let snapshot = self.get_current_snapshot().await;
        let trends = self.get_performance_trends().await;
        let history = self.get_performance_history().await;
        let analysis = self.analyze_performance().await;

        let export_data = serde_json::json!({
            "timestamp": Utc::now(),
            "current_snapshot": snapshot,
            "trends": {
                "allocation_time_trend": trends.allocation_time_trend,
                "success_rate_trend": trends.success_rate_trend,
                "throughput_trend": trends.throughput_trend,
                "conflict_rate_trend": trends.conflict_rate_trend,
                "overall_performance_score": trends.overall_performance_score,
                "last_updated": trends.last_updated
            },
            "history": history,
            "analysis": {
                "performance_grade": format!("{:?}", analysis.performance_grade),
                "insights_count": analysis.insights.len(),
                "recommendations_count": analysis.recommendations.len(),
                "anomalies_count": analysis.anomalies.len(),
                "forecast_confidence": analysis.performance_forecast.confidence_level
            }
        });

        serde_json::to_string_pretty(&export_data)
            .context("Failed to serialize performance data")
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;
    use tokio::test;

    #[test]
    async fn test_performance_metrics_initialization() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();
        let snapshot = metrics.get_current_snapshot().await;

        assert_eq!(snapshot.total_allocations, 0);
        assert_eq!(snapshot.total_deallocations, 0);
        assert_eq!(snapshot.avg_allocation_time_ms, 0.0);
        assert_eq!(snapshot.success_rate_percent, 100.0);
    }

    #[test]
    async fn test_allocation_success_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_allocation_success(Duration::from_millis(75)).await;

        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 2);
        assert_eq!(snapshot.avg_allocation_time_ms, 62.5);
        assert_eq!(snapshot.success_rate_percent, 100.0);
    }

    #[test]
    async fn test_allocation_failure_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_allocation_failure().await;

        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 1);
        assert_eq!(snapshot.success_rate_percent, 50.0);
    }

    #[test]
    async fn test_conflict_recording() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_conflict().await;

        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_conflicts, 1);
        assert_eq!(snapshot.conflict_rate_percent, 100.0);
    }

    #[test]
    async fn test_snapshot_creation_and_history() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        let snapshot1 = metrics.create_snapshot().await;

        metrics.record_allocation_success(Duration::from_millis(100)).await;
        let snapshot2 = metrics.create_snapshot().await;

        let history = metrics.get_performance_history().await;
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].total_allocations, 1);
        assert_eq!(history[1].total_allocations, 2);
    }

    #[test]
    async fn test_performance_trends_calculation() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        // Create some data with increasing allocation times
        for i in 1..=5 {
            metrics.record_allocation_success(Duration::from_millis(i * 10)).await;
            metrics.create_snapshot().await;
        }

        let trends = metrics.get_performance_trends().await;
        assert!(trends.allocation_time_trend > 0.0); // Should show increasing trend
    }

    #[test]
    async fn test_percentile_calculations() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        // Record various allocation times
        let times = vec![10, 20, 30, 40, 50, 60, 70, 80, 90, 100];
        for time in times {
            metrics.record_allocation_success(Duration::from_millis(time)).await;
        }

        let snapshot = metrics.get_current_snapshot().await;
        assert!(snapshot.median_allocation_time_ms > 0.0);
        assert!(snapshot.p95_allocation_time_ms >= snapshot.median_allocation_time_ms);
        assert!(snapshot.p99_allocation_time_ms >= snapshot.p95_allocation_time_ms);
    }

    #[test]
    async fn test_performance_report_generation() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_deallocation_success(Duration::from_millis(25)).await;
        metrics.record_conflict().await;

        let report = metrics.generate_performance_report().await;
        assert!(report.contains("Performance Report"));
        assert!(report.contains("Average Allocation Time"));
        assert!(report.contains("Success Rate"));
        assert!(report.contains("Conflict Rate"));
    }

    #[test]
    async fn test_performance_analysis() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        // Generate some performance data
        for i in 1..=10 {
            metrics.record_allocation_success(Duration::from_millis(i * 20)).await;
            metrics.create_snapshot().await;
        }

        let analysis = metrics.analyze_performance().await;
        assert!(!matches!(analysis.performance_grade, PerformanceGrade::Critical));
        // With increasing allocation times, we should have some insights
    }

    #[test]
    async fn test_configuration_update() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        let mut new_config = PerformanceConfig::default();
        new_config.history_size = 50;
        new_config.enable_detailed_timing = false;

        metrics.update_config(new_config.clone()).await;
        let updated_config = metrics.get_config().await;

        assert_eq!(updated_config.history_size, 50);
        assert!(!updated_config.enable_detailed_timing);
    }

    #[test]
    async fn test_metrics_reset() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.record_conflict().await;
        metrics.create_snapshot().await;

        metrics.reset_metrics().await;

        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 0);
        assert_eq!(snapshot.total_conflicts, 0);

        let history = metrics.get_performance_history().await;
        assert!(history.is_empty());
    }

    #[test]
    async fn test_data_cleanup() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        // Generate a lot of data
        for _ in 0..1000 {
            metrics.record_allocation_success(Duration::from_millis(50)).await;
        }

        let cleaned = metrics.cleanup_old_data().await;
        assert!(cleaned > 0);
    }

    #[test]
    async fn test_data_export() {
        let metrics = PortPerformanceMetrics::new().await.unwrap();

        metrics.record_allocation_success(Duration::from_millis(50)).await;
        metrics.create_snapshot().await;

        let export_data = metrics.export_performance_data().await.unwrap();
        assert!(export_data.contains("current_snapshot"));
        assert!(export_data.contains("trends"));
        assert!(export_data.contains("history"));
    }

    #[test]
    async fn test_concurrent_metrics_recording() {
        use std::sync::Arc;
        use tokio::task;

        let metrics = Arc::new(PortPerformanceMetrics::new().await.unwrap());

        // Spawn multiple tasks recording metrics concurrently
        let mut handles = vec![];
        for i in 0..10 {
            let metrics_clone = Arc::clone(&metrics);
            let handle = task::spawn(async move {
                for _ in 0..10 {
                    metrics_clone.record_allocation_success(Duration::from_millis(i * 10 + 50)).await;
                }
            });
            handles.push(handle);
        }

        // Wait for all tasks to complete
        for handle in handles {
            handle.await.unwrap();
        }

        let snapshot = metrics.get_current_snapshot().await;
        assert_eq!(snapshot.total_allocations, 100); // 10 tasks * 10 allocations each
    }
}