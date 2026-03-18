//! Metrics and statistics types for test characterization

use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};

use super::super::analysis::TrendDirection;

pub struct BasicStatistics {
    pub count: usize,
    pub sum: f64,
    pub min: f64,
    pub max: f64,
}

pub struct AdvancedStatistics {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<u8, f64>,
}

pub struct AggregatorPerformanceMetrics {
    pub throughput: f64,
    pub latency: f64,
    pub memory_usage: f64,
}

pub struct CurrentPerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub error_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DurationStatistics {
    /// Minimum duration observed
    #[serde(skip)]
    pub min: Duration,
    /// Maximum duration observed
    #[serde(skip)]
    pub max: Duration,
    /// Average duration
    #[serde(skip)]
    pub mean: Duration,
    /// Median duration
    #[serde(skip)]
    pub median: Duration,
    /// Standard deviation
    #[serde(skip)]
    pub std_dev: Duration,
    /// 95th percentile
    #[serde(skip)]
    pub p95: Duration,
    /// 99th percentile
    #[serde(skip)]
    pub p99: Duration,
    /// Number of samples
    pub sample_count: usize,
    /// Variance in durations
    pub variance: f64,
    /// Trend over time
    pub trend: TrendDirection,
}

impl Default for DurationStatistics {
    fn default() -> Self {
        Self {
            min: Duration::from_secs(0),
            max: Duration::from_secs(0),
            mean: Duration::from_secs(0),
            median: Duration::from_secs(0),
            std_dev: Duration::from_secs(0),
            p95: Duration::from_secs(0),
            p99: Duration::from_secs(0),
            sample_count: 0,
            variance: 0.0,
            trend: TrendDirection::Stable,
        }
    }
}

pub struct ErrorAnalysis {
    pub error_count: usize,
    pub error_rate: f64,
    pub error_types: HashMap<String, usize>,
}

pub struct EvaluationMetrics {
    pub accuracy: f64,
    pub precision: f64,
    pub recall: f64,
}

#[derive(Debug, Clone)]
pub struct MatchQualityMetrics {
    /// Accuracy score
    pub accuracy: f64,
    /// Precision score
    pub precision: f64,
    /// Recall score
    pub recall: f64,
    /// F1 score
    pub f1_score: f64,
    /// Specificity score
    pub specificity: f64,
    /// Matthews correlation coefficient
    pub mcc: f64,
    /// Area under ROC curve
    pub auc_roc: f64,
    /// False positive rate
    pub false_positive_rate: f64,
    /// False negative rate
    pub false_negative_rate: f64,
    /// Overall quality score
    pub overall_quality: f64,
}

pub struct Percentiles {
    pub p50: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

pub struct PipelineStageStats {
    pub stage_name: String,
    pub execution_time: Duration,
    pub success_rate: f64,
}

#[derive(Debug, Clone)]
pub struct ProcessingMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub error_rate: f64,
    pub processed_count: Arc<AtomicUsize>,
}

impl ProcessingMetrics {
    /// Create a new ProcessingMetrics with zero values
    pub fn new() -> Self {
        Self {
            throughput: 0.0,
            latency: Duration::ZERO,
            error_rate: 0.0,
            processed_count: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Increment the processed points counter (thread-safe)
    pub fn increment_processed_points(&self) {
        self.processed_count.fetch_add(1, Ordering::SeqCst);
    }

    /// Get the current processed count
    pub fn get_processed_count(&self) -> usize {
        self.processed_count.load(Ordering::SeqCst)
    }
}

impl Default for ProcessingMetrics {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMetrics {
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub metrics: HashMap<String, f64>,
}

impl RealTimeMetrics {
    /// Create a new RealTimeMetrics with current timestamp
    pub fn new() -> Self {
        Self {
            timestamp: chrono::Utc::now(),
            metrics: HashMap::new(),
        }
    }

    /// Merge resource metrics from another RealTimeMetrics
    pub fn merge_resource_metrics(&mut self, other: &Self) {
        for (key, value) in &other.metrics {
            self.metrics.entry(key.clone()).and_modify(|v| *v += value).or_insert(*value);
        }
        // Update timestamp to latest
        if other.timestamp > self.timestamp {
            self.timestamp = other.timestamp;
        }
    }
}

impl Default for RealTimeMetrics {
    fn default() -> Self {
        Self::new()
    }
}

pub struct RegressionAnalysis {
    pub model_type: String,
    pub r_squared: f64,
    pub coefficients: Vec<f64>,
}

pub struct RegressionAnalysisResult {
    pub r_squared: f64,
    pub adjusted_r_squared: f64,
    pub coefficients: HashMap<String, f64>,
}

pub struct ServiceOperationMetadata {
    pub operation_id: String,
    pub service_name: String,
    pub operation_type: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
    pub duration: Duration,
}

pub struct StageMetrics {
    pub stage_name: String,
    pub duration: Duration,
    pub throughput: f64,
    pub error_count: usize,
}

#[derive(Debug, Clone)]
pub struct StreamStatistics {
    /// Total samples processed
    pub samples_processed: u64,
    /// Processing rate (samples/sec)
    pub processing_rate: f64,
    /// Data quality score
    pub quality_score: f64,
    /// Error count
    pub error_count: usize,
    /// Drop count
    pub drop_count: usize,
    /// Average latency
    pub avg_latency: Duration,
    /// Throughput
    pub throughput: f64,
    /// Resource utilization
    pub resource_utilization: HashMap<String, f64>,
    /// Stream efficiency
    pub efficiency: f64,
    /// Uptime percentage
    pub uptime: f64,
}

impl Default for StreamStatistics {
    fn default() -> Self {
        Self {
            samples_processed: 0,
            processing_rate: 0.0,
            quality_score: 1.0,
            error_count: 0,
            drop_count: 0,
            avg_latency: Duration::ZERO,
            throughput: 0.0,
            resource_utilization: HashMap::new(),
            efficiency: 1.0,
            uptime: 100.0,
        }
    }
}

pub struct UsageStatistics {
    pub total_requests: usize,
    pub active_users: usize,
    pub resource_consumption: HashMap<String, f64>,
    pub peak_usage: f64,
    pub average_usage: f64,
}

#[derive(Debug, Clone, Default)]
pub struct ComprehensiveResourceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub io_usage: f64,
}

pub struct ConfidenceScores {
    pub overall_confidence: f64,
    pub component_scores: HashMap<String, f64>,
}

pub struct DirectoryUsageStatistics {
    pub total_size: u64,
    pub average_file_size: u64,
    pub largest_file: String,
}

pub struct SeverityDistribution {
    pub distribution: HashMap<String, usize>,
    pub total_count: usize,
    pub most_common_severity: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestMetadata {
    pub test_id: String,
    pub test_name: String,
    pub test_suite: String,
    pub tags: Vec<String>,
    pub author: String,
    pub created_at: chrono::DateTime<chrono::Utc>,
    pub description: String,
}

#[derive(Debug, Clone, Default)]
pub struct TestProfile {
    pub resource_metrics: HashMap<String, f64>,
}

pub struct CleanupStatistics {
    pub total_cleanups: usize,
    pub items_removed: usize,
    pub bytes_freed: u64,
}

#[derive(Debug, Clone)]
pub struct CollectionCounters {
    pub total_collected: usize,
    pub successful: usize,
    pub failed: usize,
}

impl CollectionCounters {
    /// Create a new CollectionCounters with zero counts
    pub fn new() -> Self {
        Self {
            total_collected: 0,
            successful: 0,
            failed: 0,
        }
    }

    /// Increment the collection counters
    pub fn increment_collections(&mut self) {
        self.total_collected += 1;
        self.successful += 1;
    }
}

impl Default for CollectionCounters {
    fn default() -> Self {
        Self::new()
    }
}
