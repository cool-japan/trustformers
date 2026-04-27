//! Aggregator and Configuration Types

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

// Import common types

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coordination_config_default() {
        let config = CoordinationConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.protocol, "raft");
        assert_eq!(config.sync_interval, Duration::from_secs(5));
    }

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.level, 6);
    }

    #[test]
    fn test_compression_statistics_default() {
        let stats = CompressionStatistics::default();
        assert_eq!(stats.bytes_compressed, 0);
        assert!((stats.compression_ratio - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_compression_statistics_new() {
        let stats = CompressionStatistics::new();
        assert_eq!(stats.original_size, 0);
    }

    #[test]
    fn test_quality_criteria_default() {
        let criteria = QualityCriteria::default();
        assert!((criteria.completeness_threshold - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_outlier_result_default() {
        let result = OutlierResult::default();
        assert!(!result.is_outlier);
        assert!((result.score - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_validation_rule_default() {
        let rule = ValidationRule::default();
        assert!(rule.rule_name.is_empty());
        assert_eq!(rule.severity, "medium");
    }

    #[test]
    fn test_histogram_bin_default() {
        let bin = HistogramBin::default();
        assert!((bin.lower_bound - 0.0).abs() < 1e-9);
        assert_eq!(bin.count, 0);
    }

    #[test]
    fn test_compressed_data_default() {
        let data = CompressedData::default();
        assert_eq!(data.algorithm, "none");
        assert!(data.data.is_empty());
        assert_eq!(data.original_size, 0);
    }

    #[test]
    fn test_aggregator_performance_metrics_default() {
        let m = AggregatorPerformanceMetrics::default();
        assert!((m.throughput - 0.0).abs() < 1e-9);
        assert_eq!(m.error_count, 0);
        assert_eq!(m.window_count, 0);
    }

    #[test]
    fn test_aggregation_metadata_default() {
        let m = AggregationMetadata::default();
        assert!(m.aggregation_id.is_empty());
        assert_eq!(m.data_points, 0);
    }
}
