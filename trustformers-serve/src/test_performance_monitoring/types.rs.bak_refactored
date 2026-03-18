//! Core Types and Configuration Structures
//!
//! This module contains all the fundamental types, enums, and configuration structures
//! used throughout the test performance monitoring system.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    fmt,
    sync::atomic::{AtomicBool, AtomicU64},
    time::Duration,
};

// Re-export types from other modules
pub use super::historical_data::RetentionPolicy;
pub use super::reporting::Report;

/// Main configuration for test performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestPerformanceMonitoringConfig {
    /// Enable real-time monitoring
    pub enable_real_time: bool,
    /// Monitoring interval
    pub monitoring_interval: Duration,
    /// Metrics retention period
    pub retention_period: Duration,
    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,
    /// Report configuration
    pub report_config: ReportConfig,
    /// Stream configuration
    pub stream_config: StreamConfig,
    /// Monitoring detailed configuration
    pub monitoring_config: MonitoringConfig,
    /// Data retention configuration
    pub data_retention_config: DataRetentionConfig,
}

impl Default for TestPerformanceMonitoringConfig {
    fn default() -> Self {
        Self {
            enable_real_time: true,
            monitoring_interval: Duration::from_secs(5),
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            alert_thresholds: AlertThresholds::default(),
            report_config: ReportConfig::default(),
            stream_config: StreamConfig::default(),
            monitoring_config: MonitoringConfig::default(),
            data_retention_config: DataRetentionConfig::default(),
        }
    }
}

/// Alert threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    pub cpu_usage_threshold: f64,
    pub memory_usage_threshold: f64,
    pub execution_time_threshold: Duration,
    pub failure_rate_threshold: f64,
    pub throughput_threshold: f64,
    pub resource_pressure_threshold: PressureLevel,
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            cpu_usage_threshold: 0.8,                           // 80%
            memory_usage_threshold: 0.85,                       // 85%
            execution_time_threshold: Duration::from_secs(300), // 5 minutes
            failure_rate_threshold: 0.1,                        // 10%
            throughput_threshold: 1.0,                          // 1 test per second minimum
            resource_pressure_threshold: PressureLevel::High,
        }
    }
}

/// Report generation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportConfig {
    pub generate_detailed_reports: bool,
    pub include_historical_data: bool,
    pub export_formats: Vec<String>,
    pub auto_generate_interval: Option<Duration>,
    pub report_sections: Vec<ReportSection>,
}

impl Default for ReportConfig {
    fn default() -> Self {
        Self {
            generate_detailed_reports: true,
            include_historical_data: true,
            export_formats: vec!["json".to_string(), "html".to_string()],
            auto_generate_interval: Some(Duration::from_secs(3600)), // Hourly
            report_sections: vec![
                ReportSection::Summary,
                ReportSection::TestExecution,
                ReportSection::ResourceUsage,
                ReportSection::Performance,
            ],
        }
    }
}

/// Pressure levels for resource monitoring
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize, Default)]
pub enum PressureLevel {
    #[default]
    Low,
    Medium,
    High,
    Critical,
}

impl fmt::Display for PressureLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PressureLevel::Low => write!(f, "Low"),
            PressureLevel::Medium => write!(f, "Medium"),
            PressureLevel::High => write!(f, "High"),
            PressureLevel::Critical => write!(f, "Critical"),
        }
    }
}

/// Trend directions for performance analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub enum TrendDirection {
    Improving,
    Stable,
    Degrading,
    #[default]
    Unknown,
    Increasing,
    Decreasing,
}

impl fmt::Display for TrendDirection {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TrendDirection::Improving => write!(f, "Improving"),
            TrendDirection::Stable => write!(f, "Stable"),
            TrendDirection::Degrading => write!(f, "Degrading"),
            TrendDirection::Unknown => write!(f, "Unknown"),
            TrendDirection::Increasing => write!(f, "Increasing"),
            TrendDirection::Decreasing => write!(f, "Decreasing"),
        }
    }
}

/// Event severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl fmt::Display for EventSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EventSeverity::Info => write!(f, "Info"),
            EventSeverity::Warning => write!(f, "Warning"),
            EventSeverity::Error => write!(f, "Error"),
            EventSeverity::Critical => write!(f, "Critical"),
        }
    }
}

/// Stream configuration for performance data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StreamConfig {
    /// Maximum buffer size for the stream
    pub max_buffer_size: usize,
    /// Batch size for stream processing
    pub batch_size: usize,
    /// Stream compression settings
    pub enable_compression: bool,
    /// Rate limiting configuration
    pub rate_limiting: Option<RateLimitingStrategy>,
    /// Stream priority levels
    pub priority_levels: Vec<SubscriptionPriority>,
}

impl Default for StreamConfig {
    fn default() -> Self {
        Self {
            max_buffer_size: 10000,
            batch_size: 100,
            enable_compression: true,
            rate_limiting: Some(RateLimitingStrategy::TokenBucket {
                capacity: 1000,
                refill_rate: 100,
            }),
            priority_levels: vec![
                SubscriptionPriority::Low,
                SubscriptionPriority::Medium,
                SubscriptionPriority::High,
                SubscriptionPriority::Critical,
            ],
        }
    }
}

/// Rate limiting strategies for streams
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitingStrategy {
    /// Token bucket algorithm
    TokenBucket { capacity: u64, refill_rate: u64 },
    /// Fixed window rate limiting
    FixedWindow {
        window_size: Duration,
        max_requests: u64,
    },
    /// Sliding window rate limiting
    SlidingWindow {
        window_size: Duration,
        max_requests: u64,
    },
    /// No rate limiting
    None,
}

/// Monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringConfig {
    /// Collection interval
    pub collection_interval: Duration,
    /// Buffer size for metrics
    pub metrics_buffer_size: usize,
    /// Performance thresholds
    pub performance_thresholds: Vec<PerformanceThreshold>,
    /// Anomaly detection configuration
    pub anomaly_detection: AnomalyDetectionConfig,
    /// Enable detailed tracing
    pub enable_detailed_tracing: bool,
}

impl Default for MonitoringConfig {
    fn default() -> Self {
        Self {
            collection_interval: Duration::from_millis(1000),
            metrics_buffer_size: 1000,
            performance_thresholds: vec![
                PerformanceThreshold {
                    metric_name: "cpu_usage".to_string(),
                    threshold_value: 0.8,
                    comparison_operator: "greater_than".to_string(),
                    action: ThresholdAction::Alert,
                },
                PerformanceThreshold {
                    metric_name: "memory_usage".to_string(),
                    threshold_value: 0.85,
                    comparison_operator: "greater_than".to_string(),
                    action: ThresholdAction::Alert,
                },
            ],
            anomaly_detection: AnomalyDetectionConfig::default(),
            enable_detailed_tracing: false,
        }
    }
}

/// Performance threshold definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    /// Name of the metric to monitor
    pub metric_name: String,
    /// Threshold value
    pub threshold_value: f64,
    /// Comparison operator (greater_than, less_than, equals)
    pub comparison_operator: String,
    /// Action to take when threshold is exceeded
    pub action: ThresholdAction,
}

/// Actions to take when thresholds are exceeded
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdAction {
    /// Log the event
    Log,
    /// Send an alert
    Alert,
    /// Trigger optimization
    Optimize,
    /// Scale resources
    Scale,
    /// Stop monitoring
    Stop,
    /// Custom action with command
    Custom(String),
}

/// Anomaly detection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyDetectionConfig {
    /// Enable anomaly detection
    pub enabled: bool,
    /// Detection algorithm to use
    pub algorithm: AnomalyDetectionAlgorithm,
    /// Sensitivity threshold
    pub sensitivity: f64,
    /// Training window size
    pub training_window_size: usize,
    /// Scoring method
    pub scoring_method: AnomalyScoringMethod,
}

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: AnomalyDetectionAlgorithm::StatisticalOutlier,
            sensitivity: 0.95,
            training_window_size: 100,
            scoring_method: AnomalyScoringMethod::ZScore,
        }
    }
}

/// Anomaly detection algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyDetectionAlgorithm {
    /// Statistical outlier detection
    StatisticalOutlier,
    /// Isolation forest
    IsolationForest,
    /// Moving average based
    MovingAverage,
    /// Seasonal decomposition
    SeasonalDecomposition,
    /// Machine learning based
    MachineLearning,
}

/// Anomaly scoring methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyScoringMethod {
    /// Z-score based scoring
    ZScore,
    /// Percentile based scoring
    Percentile,
    /// IQR based scoring
    IQR,
    /// Custom scoring function
    Custom,
}

/// Data retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataRetentionConfig {
    /// Raw data retention period
    pub raw_data_retention: Duration,
    /// Aggregated data retention period
    pub aggregated_data_retention: Duration,
    /// Compression configuration
    pub compression: CompressionConfig,
    /// Archival configuration
    pub archival: ArchivalConfig,
    /// Cleanup schedule
    pub cleanup_schedule: CleanupSchedule,
}

impl Default for DataRetentionConfig {
    fn default() -> Self {
        Self {
            raw_data_retention: Duration::from_secs(24 * 3600), // 1 day
            aggregated_data_retention: Duration::from_secs(30 * 24 * 3600), // 30 days
            compression: CompressionConfig::default(),
            archival: ArchivalConfig::default(),
            cleanup_schedule: CleanupSchedule::default(),
        }
    }
}

/// Compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    /// Enable compression
    pub enabled: bool,
    /// Compression algorithm
    pub algorithm: CompressionAlgorithm,
    /// Compression level (1-9)
    pub compression_level: u8,
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: CompressionAlgorithm::Gzip,
            compression_level: 6,
        }
    }
}

/// Compression algorithms
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CompressionAlgorithm {
    Gzip,
    Zstd,
    Lz4,
    Snappy,
}

/// Archival configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArchivalConfig {
    /// Enable archival
    pub enabled: bool,
    /// Storage type for archival
    pub storage_type: ArchivalStorageType,
    /// Archival schedule
    pub schedule: ArchivalSchedule,
    /// Archival format
    pub format: ArchivalFormat,
}

impl Default for ArchivalConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            storage_type: ArchivalStorageType::LocalFileSystem,
            schedule: ArchivalSchedule::Weekly,
            format: ArchivalFormat::CompressedJson,
        }
    }
}

/// Archival storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivalStorageType {
    LocalFileSystem,
    S3,
    GCS,
    Azure,
}

/// Archival schedules
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivalSchedule {
    Daily,
    Weekly,
    Monthly,
    Manual,
}

/// Archival formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArchivalFormat {
    Json,
    CompressedJson,
    Parquet,
    Csv,
}

/// Cleanup schedule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CleanupSchedule {
    /// Cleanup interval
    pub interval: Duration,
    /// Enable automatic cleanup
    pub auto_cleanup: bool,
}

impl Default for CleanupSchedule {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(24 * 3600), // Daily
            auto_cleanup: true,
        }
    }
}

/// Subscription priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SubscriptionPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Report sections that can be included
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReportSection {
    Summary,
    TestExecution,
    ResourceUsage,
    Performance,
    Alerts,
    Trends,
    Recommendations,
}

/// Report types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportType {
    Summary,
    Detailed,
    Performance,
    Historical,
    Comparative,
    Custom(String),
}

/// Report formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReportFormat {
    Json,
    Html,
    Pdf,
    Csv,
    Xml,
}

/// Time range specification for reports
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

/// Aggregation levels for data processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationLevel {
    Raw,
    Minute,
    Hour,
    Day,
    Week,
    Month,
}

/// Common metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    Integer(i64),
    Float(f64),
    Boolean(bool),
    String(String),
    Duration(Duration),
    Timestamp(DateTime<Utc>),
    Array(Vec<MetricValue>),
    Object(HashMap<String, MetricValue>),
}

/// Threshold types for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdType {
    Absolute,
    Percentage,
    Relative,
    Trend,
}

/// Anomaly types for detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnomalyType {
    Outlier,
    Drift,
    Spike,
    Drop,
    Seasonal,
    Trend,
}

/// Alert types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    Performance,
    Resource,
    Error,
    Threshold,
    Anomaly,
    System,
}

/// Alert severity levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

/// Utility trait for metric conversion
pub trait IntoMetricValue {
    fn into_metric_value(self) -> MetricValue;
}

impl IntoMetricValue for i64 {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Integer(self)
    }
}

impl IntoMetricValue for f64 {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Float(self)
    }
}

impl IntoMetricValue for bool {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Boolean(self)
    }
}

impl IntoMetricValue for String {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::String(self)
    }
}

impl IntoMetricValue for Duration {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Duration(self)
    }
}

impl IntoMetricValue for DateTime<Utc> {
    fn into_metric_value(self) -> MetricValue {
        MetricValue::Timestamp(self)
    }
}

/// Event management configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventConfig {
    /// Channel capacity for event broadcasting
    pub channel_capacity: usize,
    /// Event storage configuration
    pub storage_config: EventStorageConfig,
    /// Event buffer size
    pub buffer_size: usize,
    /// Enable compression for stored events
    pub compression_enabled: bool,
    /// Event indexing configuration
    pub indexing_config: EventIndexingConfig,
    /// Event retention configuration
    pub retention_config: EventRetentionConfig,
    /// Rate limiting configuration
    pub rate_limit_config: RateLimitConfig,
    /// Event correlation configuration
    pub correlation_config: CorrelationConfig,
    /// Pattern matching configuration
    pub pattern_config: PatternConfig,
    /// Aggregation configuration
    pub aggregation_config: AggregationConfig,
    /// Event enrichment configuration
    pub enrichment_config: EnrichmentConfig,
}

/// Query parameters for event retrieval
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventQuery {
    /// Filter criteria
    pub filters: std::collections::HashMap<String, String>,
    /// Time range for query
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    /// Maximum number of events to return
    pub limit: Option<usize>,
    /// Event types to include
    pub event_types: Option<Vec<String>>,
    /// Sort order
    pub sort_order: Option<String>,
}

impl Default for EventConfig {
    fn default() -> Self {
        Self {
            channel_capacity: 1000,
            storage_config: EventStorageConfig::default(),
            buffer_size: 10000,
            compression_enabled: true,
            indexing_config: EventIndexingConfig::default(),
            retention_config: EventRetentionConfig::default(),
            rate_limit_config: RateLimitConfig::default(),
            correlation_config: CorrelationConfig::default(),
            pattern_config: PatternConfig::default(),
            aggregation_config: AggregationConfig::default(),
            enrichment_config: EnrichmentConfig::default(),
        }
    }
}

/// Event storage configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventStorageConfig {
    /// Storage type
    pub storage_type: EventStorageType,
    /// Maximum storage size in bytes
    pub max_storage_size: usize,
    /// Storage path for file-based storage
    pub storage_path: Option<String>,
    /// Enable persistent storage
    pub persistent: bool,
}

impl Default for EventStorageConfig {
    fn default() -> Self {
        Self {
            storage_type: EventStorageType::Memory,
            max_storage_size: 100 * 1024 * 1024, // 100 MB
            storage_path: None,
            persistent: false,
        }
    }
}

/// Event storage types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventStorageType {
    /// In-memory storage
    Memory,
    /// File-based storage
    File,
    /// Database storage
    Database,
}

/// Event indexing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventIndexingConfig {
    /// Enable indexing
    pub enabled: bool,
    /// Index fields
    pub indexed_fields: Vec<String>,
    /// Index rebuild interval
    pub rebuild_interval: Duration,
}

impl Default for EventIndexingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            indexed_fields: vec![
                "event_type".to_string(),
                "test_id".to_string(),
                "timestamp".to_string(),
            ],
            rebuild_interval: Duration::from_secs(3600), // 1 hour
        }
    }
}

/// Event retention configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRetentionConfig {
    /// Retention period
    pub retention_period: Duration,
    /// Maximum events to retain
    pub max_events: usize,
    /// Cleanup interval
    pub cleanup_interval: Duration,
}

impl Default for EventRetentionConfig {
    fn default() -> Self {
        Self {
            retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            max_events: 1000000,                                  // 1 million events
            cleanup_interval: Duration::from_secs(3600),          // 1 hour
        }
    }
}

/// Rate limiting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    /// Enable rate limiting
    pub enabled: bool,
    /// Maximum events per second
    pub max_events_per_second: u64,
    /// Burst capacity
    pub burst_capacity: u64,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_events_per_second: 1000,
            burst_capacity: 2000,
        }
    }
}

/// Event correlation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationConfig {
    /// Enable correlation
    pub enabled: bool,
    /// Correlation window
    pub correlation_window: Duration,
    /// Correlation fields
    pub correlation_fields: Vec<String>,
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            correlation_window: Duration::from_secs(60), // 1 minute
            correlation_fields: vec!["test_id".to_string(), "session_id".to_string()],
        }
    }
}

/// Pattern matching configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternConfig {
    /// Enable pattern matching
    pub enabled: bool,
    /// Pattern rules
    pub pattern_rules: Vec<String>,
    /// Pattern timeout
    pub pattern_timeout: Duration,
}

impl Default for PatternConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            pattern_rules: Vec::new(),
            pattern_timeout: Duration::from_secs(30),
        }
    }
}

/// Event enrichment configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnrichmentConfig {
    /// Enable enrichment
    pub enabled: bool,
    /// Enrichment sources
    pub enrichment_sources: Vec<String>,
    /// Enrichment timeout
    pub enrichment_timeout: Duration,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            enrichment_sources: Vec::new(),
            enrichment_timeout: Duration::from_secs(5),
        }
    }
}

/// Event dispatch statistics
#[derive(Debug, Default)]
pub struct DispatchStatistics {
    pub total_events: AtomicU64,
    pub processed_events: AtomicU64,
    pub failed_events: AtomicU64,
    pub avg_processing_time: Duration,
}

impl DispatchStatistics {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Subscriber registry for event management
#[derive(Debug, Default)]
pub struct SubscriberRegistry {
    subscribers: HashMap<String, Vec<String>>,
    active_subscriptions: HashMap<String, AtomicBool>,
}

impl SubscriberRegistry {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Query performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryPerformanceMetrics {
    pub avg_query_time: Duration,
    pub query_count: u64,
    pub cache_hit_rate: f64,
}

/// Storage efficiency metrics
#[derive(Debug, Clone, Default)]
pub struct StorageEfficiencyMetrics {
    pub compression_ratio: f64,
    pub storage_used: u64,
    pub storage_capacity: u64,
}

/// Data retention metrics
#[derive(Debug, Clone, Default)]
pub struct RetentionMetrics {
    pub retention_period: Duration,
    pub cleanup_frequency: Duration,
    pub purged_records: u64,
}

/// Export manager for data export operations
#[derive(Debug)]
pub struct ExportManager {
    export_format: String,
    destination: String,
}

impl ExportManager {
    pub fn new(_config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            export_format: "json".to_string(),
            destination: "/tmp/exports".to_string(),
        }
    }

    pub fn from_report_config(config: &ReportConfig) -> Self {
        let export_format =
            config.export_formats.first().cloned().unwrap_or_else(|| "json".to_string());

        Self {
            export_format,
            destination: "/tmp/reports".to_string(),
        }
    }

    /// Export a report
    /// TODO: Implement actual report export logic
    pub async fn export_report(
        &self,
        _report: &Report,
        _format: ReportFormat,
    ) -> Result<ExportResult, anyhow::Error> {
        Ok(ExportResult {
            export_id: format!("export_{}", chrono::Utc::now().timestamp()),
            file_path: format!("{}/report_stub", self.destination),
            format: self.export_format.clone(),
            file_size: 0,
            exported_at: chrono::Utc::now(),
        })
    }
}

/// Report storage manager
#[derive(Debug)]
pub struct ReportStorage {
    storage_path: String,
    retention_policy: Duration,
}

impl ReportStorage {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            storage_path: "/tmp/reports".to_string(),
            retention_policy: config.retention_period,
        }
    }

    pub fn from_report_config(config: &ReportConfig) -> Self {
        let retention_policy =
            config.auto_generate_interval.unwrap_or_else(|| Duration::from_secs(24 * 3600));

        Self {
            storage_path: "/tmp/reports".to_string(),
            retention_policy,
        }
    }

    /// Get a report by ID
    /// TODO: Implement actual report retrieval from storage
    pub async fn get_report(&self, _report_id: &str) -> Result<Report, anyhow::Error> {
        Ok(Report {
            report_id: "stub".to_string(),
            test_id: "stub".to_string(),
            report_type: super::reporting::ReportType::Summary,
            content: "Stub report".to_string(),
            generated_at: chrono::Utc::now(),
            metadata: std::collections::HashMap::new(),
        })
    }
}

/// Security classification levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SecurityClassification {
    Public,
    Internal,
    Confidential,
    Restricted,
}

/// Time series data types
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeSeriesDataType {
    Numeric,
    String,
    Boolean,
    Struct,
}

/// Time series index for efficient data access
#[derive(Debug, Clone, Default)]
pub struct TimeSeriesIndex {
    index_entries: HashMap<String, u64>,
    last_updated: Option<DateTime<Utc>>,
}

/// Compression information for stored data
#[derive(Debug, Clone, Default)]
pub struct CompressionInfo {
    pub algorithm: String,
    pub compression_ratio: f64,
    pub original_size: u64,
    pub compressed_size: u64,
}

/// Database statistics
#[derive(Debug, Clone, Default)]
pub struct DatabaseStatistics {
    pub query_count: u64,
    pub avg_query_time: Duration,
    pub connection_count: u32,
    pub cache_hit_rate: f64,
}

/// Resolution method for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ResolutionMethod {
    Manual,
    Automatic,
    Escalation,
    Timeout,
}

/// Rule executor for alerting system
#[derive(Debug)]
pub struct RuleExecutor {
    config: TestPerformanceMonitoringConfig,
}

impl RuleExecutor {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Condition evaluator for alert rules
#[derive(Debug)]
pub struct ConditionEvaluator {
    config: TestPerformanceMonitoringConfig,
}

impl ConditionEvaluator {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Rule scheduler for scheduled evaluations
#[derive(Debug)]
pub struct RuleScheduler {
    config: TestPerformanceMonitoringConfig,
}

impl RuleScheduler {
    pub fn new(config: &TestPerformanceMonitoringConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Evaluation context for rule execution
#[derive(Debug, Default)]
pub struct EvaluationContext {
    pub variables: HashMap<String, MetricValue>,
    pub timestamp: Option<DateTime<Utc>>,
}

impl EvaluationContext {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Alert category classification
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AlertCategory {
    Performance,
    Resource,
    Error,
    Security,
    Availability,
}

/// Outlier detection configuration
#[derive(Debug, Clone, Default)]
pub struct OutlierDetectionConfig {
    pub threshold: f64,
    pub window_size: Duration,
    pub sensitivity: f64,
}

/// Outlier type classification
#[derive(Debug, Clone)]
pub enum OutlierType {
    Low,
    High,
    Anomalous,
}

/// Layout engine for dashboard rendering
#[derive(Debug)]
pub struct LayoutEngine {
    config: DashboardConfig,
}

impl LayoutEngine {
    pub fn new(config: &DashboardConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Update scheduler for dashboard updates
#[derive(Debug)]
pub struct UpdateScheduler {
    config: DashboardConfig,
}

impl UpdateScheduler {
    pub fn new(config: &DashboardConfig) -> Self {
        Self {
            config: config.clone(),
        }
    }
}

/// Rate limit state for event processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RateLimitState {
    Normal,
    Limited,
    Throttled,
    Blocked,
}

/// Recovery tracker for error recovery
#[derive(Debug, Default)]
pub struct RecoveryTracker {
    pub recovery_attempts: u32,
    pub success_count: u32,
    pub last_recovery: Option<DateTime<Utc>>,
}

/// Context value for error contexts
#[derive(Debug, Clone)]
pub enum ContextValue {
    String(String),
    Number(f64),
    Boolean(bool),
    List(Vec<ContextValue>),
}

/// Resource thresholds for monitoring
#[derive(Debug, Clone, Default)]
pub struct ResourceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub disk_threshold: f64,
    pub network_threshold: f64,
}

/// Enhanced latency processor for feedback systems
#[derive(Debug, Default)]
pub struct EnhancedLatencyProcessor {
    pub processing_times: Vec<Duration>,
    pub avg_latency: Duration,
    pub max_latency: Duration,
}

impl EnhancedLatencyProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Enhanced resource utilization processor for feedback systems
#[derive(Debug, Default)]
pub struct EnhancedResourceUtilizationProcessor {
    pub cpu_utilization: f64,
    pub memory_utilization: f64,
    pub gpu_utilization: f64,
    pub network_utilization: f64,
}

impl EnhancedResourceUtilizationProcessor {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Recovery action for alert recovery
#[derive(Debug, Clone)]
pub enum RecoveryAction {
    Restart,
    Scale,
    Notify,
    Custom(String),
}

/// Flap detection for alert stability
#[derive(Debug, Clone, Default)]
pub struct FlapDetection {
    pub enabled: bool,
    pub threshold: u32,
    pub window: Duration,
}

/// Aggregation manager for data aggregation
#[derive(Debug, Default)]
pub struct AggregationManager {
    pub aggregated_metrics: HashMap<String, f64>,
    pub last_aggregation: Option<DateTime<Utc>>,
}

impl AggregationManager {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Adaptive learning orchestrator for performance modeling
#[derive(Debug)]
pub struct AdaptiveLearningOrchestrator {
    pub learning_rate: f64,
    pub model_updates: u64,
    pub last_update: Option<DateTime<Utc>>,
}

impl AdaptiveLearningOrchestrator {
    pub async fn new(_config: &TestPerformanceMonitoringConfig) -> anyhow::Result<Self> {
        Ok(Self {
            learning_rate: 0.01,
            model_updates: 0,
            last_update: None,
        })
    }
}

// =============================================================================
// Additional Types for lib.rs Imports
// =============================================================================

/// Current performance metrics snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CurrentPerformanceMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub active_tests: usize,
    pub throughput: f64,
}

/// Performance event data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceEventData {
    pub event_type: String,
    pub metrics: HashMap<String, MetricValue>,
    pub severity: EventSeverity,
    pub timestamp: DateTime<Utc>,
}

/// Performance report (alias to avoid conflicts with trustformers_core)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub report_id: String,
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub metrics: HashMap<String, MetricValue>,
    pub summary: String,
}

/// Performance data stream
#[derive(Debug, Clone)]
pub struct PerformanceStream {
    pub stream_id: String,
    pub start_time: DateTime<Utc>,
    pub metrics_buffer: Vec<CurrentPerformanceMetrics>,
}

/// Report generation request
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportRequest {
    pub template_id: String,
    pub parameters: HashMap<String, String>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub format: String,
}

/// Report summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReportSummary {
    pub report_id: String,
    pub title: String,
    pub generated_at: DateTime<Utc>,
    pub record_count: usize,
    pub key_findings: Vec<String>,
}

/// System resource metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemResourceMetrics {
    pub timestamp: DateTime<Utc>,
    pub cpu_percent: f64,
    pub memory_bytes: u64,
    pub memory_percent: f64,
    pub disk_io_read: u64,
    pub disk_io_write: u64,
    pub network_rx: u64,
    pub network_tx: u64,
}

/// Test execution metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestExecutionMetrics {
    pub test_id: String,
    pub execution_time: Duration,
    pub success: bool,
    pub cpu_time: Duration,
    pub memory_peak: u64,
    pub io_operations: u64,
}

/// Timestamped metrics (alias to real_time_metrics type)
pub use crate::performance_optimizer::real_time_metrics::TimestampedMetrics;

// ============================================================================
// Alert and Status Enum Types
// ============================================================================

/// Alert status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert is acknowledged
    Acknowledged,
    /// Alert is resolved
    Resolved,
    /// Alert is suppressed
    Suppressed,
    /// Alert is expired
    Expired,
}

/// Health status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is healthy
    Healthy,
    /// System is degraded
    Degraded,
    /// System is unhealthy
    Unhealthy,
    /// System is critical
    Critical,
    /// Status unknown
    Unknown,
}

/// Impact level enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ImpactLevel {
    /// Low impact
    Low,
    /// Medium impact
    Medium,
    /// High impact
    High,
    /// Critical impact
    Critical,
}

/// Business impact enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BusinessImpact {
    /// No business impact
    None,
    /// Minor business impact
    Minor,
    /// Moderate business impact
    Moderate,
    /// Major business impact
    Major,
    /// Severe business impact
    Severe,
}

/// Insight type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InsightType {
    /// Performance insight
    Performance,
    /// Anomaly insight
    Anomaly,
    /// Trend insight
    Trend,
    /// Pattern insight
    Pattern,
    /// Optimization insight
    Optimization,
    /// Risk insight
    Risk,
}

/// Evidence type enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EvidenceType {
    /// Statistical evidence
    Statistical,
    /// Historical evidence
    Historical,
    /// Comparative evidence
    Comparative,
    /// Experimental evidence
    Experimental,
    /// Observational evidence
    Observational,
}

/// Statistical method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StatisticalMethod {
    /// Mean calculation
    Mean,
    /// Median calculation
    Median,
    /// Standard deviation
    StandardDeviation,
    /// Percentile calculation
    Percentile,
    /// Regression analysis
    Regression,
    /// Correlation analysis
    Correlation,
}

/// Outlier detection method enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierDetectionMethod {
    /// Z-score method
    ZScore,
    /// IQR method
    IQR,
    /// MAD method (Median Absolute Deviation)
    MAD,
    /// Isolation forest
    IsolationForest,
    /// LOF (Local Outlier Factor)
    LOF,
}

/// Test phase enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestPhase {
    /// Setup phase
    Setup,
    /// Execution phase
    Execution,
    /// Teardown phase
    Teardown,
    /// Validation phase
    Validation,
    /// Cleanup phase
    Cleanup,
}

/// User preferences for test performance monitoring
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UserPreferences {
    /// Enable email notifications
    pub email_enabled: bool,
    /// Enable push notifications
    pub push_enabled: bool,
    /// Notification frequency in seconds
    pub notification_frequency: u64,
    /// Theme preference
    pub theme: String,
    /// Dashboard layout preference
    pub dashboard_layout: String,
    /// Auto-refresh interval
    pub auto_refresh_interval: Duration,
}

impl Default for UserPreferences {
    fn default() -> Self {
        Self {
            email_enabled: true,
            push_enabled: false,
            notification_frequency: 300, // 5 minutes
            theme: "light".to_string(),
            dashboard_layout: "default".to_string(),
            auto_refresh_interval: Duration::from_secs(30),
        }
    }
}

/// Aggregation configuration for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregationConfig {
    /// Aggregation window size
    pub window_size: Duration,
    /// Aggregation method
    pub method: StatisticalMethod,
    /// Enable percentile calculation
    pub enable_percentiles: bool,
    /// Percentile values to calculate
    pub percentiles: Vec<f64>,
    /// Enable outlier detection
    pub enable_outlier_detection: bool,
    /// Outlier detection method
    pub outlier_method: OutlierDetectionMethod,
}

impl Default for AggregationConfig {
    fn default() -> Self {
        Self {
            window_size: Duration::from_secs(60),
            method: StatisticalMethod::Mean,
            enable_percentiles: true,
            percentiles: vec![50.0, 90.0, 95.0, 99.0],
            enable_outlier_detection: false,
            outlier_method: OutlierDetectionMethod::ZScore,
        }
    }
}

// ============================================================================
// Alerting System Types
// ============================================================================

/// Alert configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertConfig {
    pub enabled: bool,
    pub evaluation_interval: Duration,
    pub notification_channels: Vec<String>,
}

/// Subscription configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionConfig {
    /// Enable subscription system
    pub enabled: bool,
    /// Maximum subscriptions per user
    pub max_subscriptions_per_user: usize,
    /// Subscription retention period
    pub retention_period: Duration,
    /// Enable subscription analytics
    pub analytics_enabled: bool,
}

/// Suppression configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuppressionConfig {
    pub enabled: bool,
    pub duration: Duration,
    pub rules: Vec<String>,
}

/// Alert rule metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertRuleMetadata {
    pub rule_id: String,
    pub rule_name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
}

/// Threshold value for alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdValue {
    Absolute(f64),
    Percentage(f64),
    Dynamic(String),
}

/// Aggregation method for metrics
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Percentile,
}

/// Condition context for evaluation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConditionContext {
    pub variables: std::collections::HashMap<String, f64>,
    pub timestamp: DateTime<Utc>,
}

/// Aggregation scope
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationScope {
    Test,
    Suite,
    Global,
    Custom,
}

/// Dynamic threshold method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynamicThresholdMethod {
    Statistical,
    MachineLearning,
    Adaptive,
    Baseline,
}

/// Learning algorithm for adaptive thresholds
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LearningAlgorithm {
    LinearRegression,
    ARIMA,
    ExponentialSmoothing,
    NeuralNetwork,
}

/// Outlier handling strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutlierHandling {
    Ignore,
    Remove,
    Dampen,
    Flag,
}

/// Convergence criteria for learning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConvergenceCriteria {
    pub max_iterations: usize,
    pub tolerance: f64,
    pub min_improvement: f64,
}

/// Deviation type for baseline comparison
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviationType {
    Absolute,
    Relative,
    StandardDeviation,
    Percentage,
}

/// Baseline update strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineUpdateStrategy {
    Fixed,
    RollingWindow,
    ExponentialDecay,
    Adaptive,
}

/// Monitoring scheduler
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MonitoringScheduler {
    pub enabled: bool,
    pub interval: Duration,
    pub max_concurrent: usize,
}

/// Threshold cache
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThresholdCache {
    pub enabled: bool,
    pub ttl: Duration,
    pub max_entries: usize,
}

/// Evaluation metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationMetrics {
    pub evaluations_count: u64,
    pub avg_duration_ms: f64,
    pub success_rate: f64,
}

/// Real-time processor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RealTimeProcessor {
    pub enabled: bool,
    pub buffer_size: usize,
    pub flush_interval: Duration,
}

/// Threshold evaluator type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdEvaluatorType {
    Static,
    Dynamic,
    Adaptive,
    ML,
}

/// Evaluation cost metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationCost {
    pub cpu_ms: f64,
    pub memory_bytes: u64,
    pub io_ops: u64,
}

/// Evaluation metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EvaluationMetadata {
    pub evaluator_type: String,
    pub evaluation_time: DateTime<Utc>,
    pub data_points_evaluated: usize,
}

/// Impact assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImpactAssessment {
    pub severity: String,
    pub affected_systems: Vec<String>,
    pub estimated_cost: f64,
    pub impact_level: String,
    pub affected_users: usize,
    pub business_impact: String,
    pub estimated_downtime: Duration,
    pub financial_impact: f64,
}

/// Suppression condition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuppressionCondition {
    pub condition_type: String,
    pub parameters: std::collections::HashMap<String, String>,
}

/// Dependency status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DependencyStatus {
    Resolved,
    Pending,
    Failed,
    Unknown,
}

/// Change type for alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChangeType {
    Deployment,
    Configuration,
    Infrastructure,
    Code,
}

/// Action type for remediation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionType {
    Restart,
    Scale,
    Rollback,
    Notify,
    Custom,
}

/// Action priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActionPriority {
    Low,
    Medium,
    High,
    Critical,
}

/// Estimated effort
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EstimatedEffort {
    pub person_hours: f64,
    pub complexity: String,
}

/// Expected impact
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExpectedImpact {
    pub improvement_percentage: f64,
    pub affected_metrics: Vec<String>,
}

/// Escalation executor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationExecutor {
    pub executor_id: String,
    pub actions_executed: u64,
}

/// Escalation scheduler
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationScheduler {
    pub schedule_interval: Duration,
    pub max_retries: u32,
}

/// Escalation event
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub level: u8,
}

/// Escalation metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationMetrics {
    pub total_escalations: u64,
    pub avg_response_time: Duration,
}

/// Escalation condition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationCondition {
    pub condition: String,
    pub threshold: f64,
}

/// Business hours configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BusinessHoursConfig {
    pub timezone: String,
    pub start_hour: u8,
    pub end_hour: u8,
    pub days_of_week: Vec<u8>,
}

/// Notification target
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationTarget {
    pub target_id: String,
    pub target_type: String,
    pub contact_info: String,
}

/// Escalation criteria
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationCriteria {
    pub criteria_type: String,
    pub threshold: f64,
    pub duration: Duration,
}

/// Automatic action
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutomaticAction {
    pub action_type: String,
    pub parameters: std::collections::HashMap<String, String>,
}

/// Notification rate limiter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationRateLimiter {
    pub max_per_minute: u32,
    pub burst_size: u32,
}

/// Template engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateEngine {
    pub template_dir: String,
    pub cache_enabled: bool,
}

/// Delivery tracker
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeliveryTracker {
    pub tracking_enabled: bool,
    pub delivery_log: Vec<String>,
}

/// Notification metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationMetrics {
    pub sent_count: u64,
    pub failed_count: u64,
    pub avg_delivery_time_ms: f64,
}

/// Notification type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationType {
    Email,
    Sms,
    Slack,
    Webhook,
    PagerDuty,
}

/// Notification priority
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationPriority {
    Low,
    Normal,
    High,
    Urgent,
}

/// Delivery requirements
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeliveryRequirements {
    pub require_acknowledgment: bool,
    pub max_delivery_time: Duration,
}

/// Delivery result
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryResult {
    Success,
    Failed,
    Pending,
    Throttled,
}

/// Notification error
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NotificationError {
    pub error_type: String,
    pub message: String,
    pub retry_possible: bool,
}

/// Notification channel type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NotificationChannelType {
    Email,
    Sms,
    Slack,
    Webhook,
    PagerDuty,
    Custom,
}

/// Delivery capabilities
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeliveryCapabilities {
    pub supports_attachments: bool,
    pub supports_html: bool,
    pub max_message_size: usize,
}

/// Rate limits
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateLimits {
    pub requests_per_second: u32,
    pub burst_size: u32,
}

/// SMTP configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub use_tls: bool,
}

/// Email template configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmailTemplateConfig {
    pub template_path: String,
    pub default_subject: String,
}

/// Email rate limiter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmailRateLimiter {
    pub max_per_hour: u32,
    pub burst_size: u32,
}

/// SMS provider configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SmsProviderConfig {
    pub provider: String,
    pub api_key: String,
    pub sender_number: String,
}

/// SMS limits
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SmsLimits {
    pub max_length: usize,
    pub max_per_day: u32,
}

/// Slack workspace configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlackWorkspaceConfig {
    pub workspace_id: String,
    pub bot_token: String,
}

/// Slack bot configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SlackBotConfig {
    pub bot_name: String,
    pub default_channel: String,
}

/// Webhook endpoint configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebhookEndpointConfig {
    pub url: String,
    pub method: String,
    pub headers: std::collections::HashMap<String, String>,
}

/// Webhook authentication
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebhookAuthentication {
    pub auth_type: String,
    pub credentials: String,
}

/// Webhook retry configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WebhookRetryConfig {
    pub max_retries: u32,
    pub retry_delay: Duration,
}

/// Payload format
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PayloadFormat {
    Json,
    Xml,
    FormUrlEncoded,
    Custom,
}

/// Historical alert
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistoricalAlert {
    pub alert_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: String,
}

impl Default for ThresholdValue {
    fn default() -> Self {
        Self::Absolute(0.0)
    }
}

// ============================================================================
// Historical Data and Storage Types
// ============================================================================

/// Filter value for queries
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FilterValue {
    String(String),
    Number(f64),
    Boolean(bool),
    Range(f64, f64),
}

/// Storage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageStatistics {
    pub total_size_bytes: u64,
    pub compressed_size_bytes: u64,
    pub record_count: u64,
    pub compression_ratio: f64,
}

/// Optimization result for storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationResult {
    pub success: bool,
    pub space_saved_bytes: u64,
    pub optimization_time_ms: f64,
    pub compression_savings: u64,
    pub storage_optimization: u64,
    pub retention_cleanup: u64,
    pub total_space_saved: u64,
    pub optimization_time: Duration,
}

/// Archival data container
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalData {
    pub data_id: String,
    pub archived_at: DateTime<Utc>,
    pub data_size_bytes: u64,
}

/// Archival operation result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalResult {
    pub success: bool,
    pub archived_count: usize,
    pub total_size_bytes: u64,
}

/// Verification result for archived data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VerificationResult {
    pub verified: bool,
    pub checksum_match: bool,
    pub errors: Vec<String>,
}

/// Archival filter criteria
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalFilter {
    pub age_threshold: Duration,
    pub size_threshold: u64,
    pub include_patterns: Vec<String>,
}

/// Archival metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalMetadata {
    pub archive_id: String,
    pub created_at: DateTime<Utc>,
    pub record_count: u64,
    pub checksum: String,
}

/// Metric index for efficient lookup
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricIndex {
    pub metric_name: String,
    pub indexed_fields: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

/// Tag index for categorization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TagIndex {
    pub tag_name: String,
    pub tag_values: Vec<String>,
    pub usage_count: u64,
}

/// Bloom filter for membership testing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BloomFilter {
    pub size_bits: usize,
    pub hash_count: usize,
    pub false_positive_rate: f64,
}

/// Index statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexStatistics {
    pub index_count: usize,
    pub total_entries: u64,
    pub index_size_bytes: u64,
}

/// Time bucket for aggregation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeBucket {
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub bucket_size: Duration,
}

/// Quality issue type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum QualityIssueType {
    MissingData,
    InvalidData,
    DuplicateData,
    OutOfRange,
    Inconsistent,
}

/// Percentiles for statistical analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Percentiles {
    pub p1: f64,
    pub p5: f64,
    pub p10: f64,
    pub p25: f64,
    pub p50: f64,
    pub p75: f64,
    pub p90: f64,
    pub p95: f64,
    pub p99: f64,
}

/// Change point in time series
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ChangePoint {
    pub timestamp: DateTime<Utc>,
    pub confidence: f64,
    pub change_magnitude: f64,
}

/// Query result metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryResultMetadata {
    pub query_time_ms: f64,
    pub result_count: usize,
    pub cache_hit: bool,
}

/// Statistical summary
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatisticalSummary {
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub min: f64,
    pub max: f64,
}

/// Historical data configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalDataConfig {
    pub retention_days: u32,
    pub aggregation_interval: Duration,
    pub compression_enabled: bool,
    pub indexing_config: IndexingConfig,
    pub partitioning_strategy: PartitioningStrategy,
    pub storage_optimization: StorageOptimization,
    pub cache_config: HistoricalCacheConfig,
}

impl Default for HistoricalDataConfig {
    fn default() -> Self {
        Self {
            retention_days: 30,
            aggregation_interval: Duration::from_secs(3600),
            compression_enabled: true,
            indexing_config: IndexingConfig {
                index_type: "time_series".to_string(),
                fields: vec!["timestamp".to_string(), "test_id".to_string()],
                refresh_interval: Duration::from_secs(300),
            },
            partitioning_strategy: PartitioningStrategy {
                strategy_type: "time_window".to_string(),
                partition_key: "date".to_string(),
                partition_count: 14,
            },
            storage_optimization: StorageOptimization::default(),
            cache_config: HistoricalCacheConfig::default(),
        }
    }
}

/// Cache configuration for historical data operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HistoricalCacheConfig {
    pub enabled: bool,
    pub min_execution_time_ms: u64,
    pub max_cacheable_size_mb: u64,
    pub max_entries: usize,
    pub ttl: Duration,
}

impl Default for HistoricalCacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_execution_time_ms: 250,
            max_cacheable_size_mb: 512,
            max_entries: 10_000,
            ttl: Duration::from_secs(900),
        }
    }
}

/// Time series filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeSeriesFilter {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
    pub metric_names: Vec<String>,
}

/// Archive request
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchiveRequest {
    pub request_id: String,
    pub filter: ArchivalFilter,
    pub destination: String,
}

/// Storage optimization result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageOptimizationResult {
    pub optimized_size_bytes: u64,
    pub space_saved_percent: f64,
    pub optimization_method: String,
    pub space_reclaimed: u64,
    pub optimization_time: Duration,
    pub operations_performed: usize,
}

// ============================================================================
// Analytics and Insights Types
// ============================================================================

/// Test status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TestStatus {
    Pending,
    Running,
    Passed,
    Failed,
    Skipped,
    Cancelled,
}

/// Trend analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendAnalysis {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub confidence: f64,
}

/// Optimization category
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationCategory {
    Performance,
    Resource,
    Reliability,
    Efficiency,
    Quality,
}

/// Effort level for recommendations
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EffortLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Bottleneck type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BottleneckType {
    CPU,
    Memory,
    IO,
    Network,
    Lock,
    Database,
}

/// Complexity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ComplexityLevel {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Risk level assessment
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RiskLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Risk factor
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RiskFactor {
    pub factor_name: String,
    pub severity: f64,
    pub probability: f64,
}

/// Stability assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StabilityAssessment {
    pub stability_score: f64,
    pub variance: f64,
    pub trend: String,
}

/// Detection method for anomalies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DetectionMethod {
    Statistical,
    MachineLearning,
    RuleBased,
    Hybrid,
}

/// Failure type classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FailureType {
    Timeout,
    Error,
    Crash,
    AssertionFailure,
    ResourceExhaustion,
    Unknown,
}

/// Indicator status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndicatorStatus {
    Normal,
    Warning,
    Critical,
    Unknown,
}

/// Criticality level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum CriticalityLevel {
    Low,
    Medium,
    High,
    Critical,
}

/// Performance thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    pub cpu_threshold: f64,
    pub memory_threshold: f64,
    pub latency_threshold_ms: f64,
    pub max_execution_time: Duration,
    pub max_memory_usage: u64,
    pub max_cpu_usage: f64,
}

impl Default for PerformanceThresholds {
    fn default() -> Self {
        Self {
            cpu_threshold: 0.8,
            memory_threshold: 0.85,
            latency_threshold_ms: 1000.0,
            max_execution_time: Duration::from_secs(300),
            max_memory_usage: 1024 * 1024 * 1024,
            max_cpu_usage: 85.0,
        }
    }
}

/// System pressure thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemPressureThresholds {
    pub cpu_pressure_threshold: f64,
    pub memory_pressure_threshold: f64,
    pub disk_pressure_threshold: f64,
    pub file_descriptor_threshold: u32,
}

impl Default for SystemPressureThresholds {
    fn default() -> Self {
        Self {
            cpu_pressure_threshold: 80.0,
            memory_pressure_threshold: 0.85,
            disk_pressure_threshold: 80.0,
            file_descriptor_threshold: 10_000,
        }
    }
}

/// Real-time monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RealTimeMonitoringConfig {
    pub enabled: bool,
    pub sampling_interval_ms: u64,
    pub alert_threshold: f64,
    pub stream_config: StreamConfig,
    pub compression_enabled: bool,
    pub buffer_size: usize,
    pub monitoring_interval: Duration,
}

impl Default for RealTimeMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sampling_interval_ms: 1000,
            alert_threshold: 0.8,
            stream_config: StreamConfig::default(),
            compression_enabled: true,
            buffer_size: 10_000,
            monitoring_interval: Duration::from_secs(5),
        }
    }
}

// ============================================================================
// Subscription and Notification Types
// ============================================================================

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertInfo {
    pub alert_id: String,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
}

/// Subscription type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SubscriptionType {
    Alert,
    Event,
    Metric,
    Report,
    All,
}

/// Subscription filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionFilter {
    pub filter_type: String,
    pub patterns: Vec<String>,
    pub exclude_patterns: Vec<String>,
}

/// Rate limit configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RateLimit {
    pub max_requests: u32,
    pub time_window: Duration,
    pub burst_size: u32,
}

// ============================================================================
// Default Implementations
// ============================================================================

impl Default for FilterValue {
    fn default() -> Self {
        Self::String(String::new())
    }
}

impl Default for QualityIssueType {
    fn default() -> Self {
        Self::MissingData
    }
}

impl Default for TestStatus {
    fn default() -> Self {
        Self::Pending
    }
}

impl Default for OptimizationCategory {
    fn default() -> Self {
        Self::Performance
    }
}

impl Default for EffortLevel {
    fn default() -> Self {
        Self::Medium
    }
}

impl Default for BottleneckType {
    fn default() -> Self {
        Self::CPU
    }
}

impl Default for ComplexityLevel {
    fn default() -> Self {
        Self::Moderate
    }
}

impl Default for RiskLevel {
    fn default() -> Self {
        Self::Low
    }
}

impl Default for DetectionMethod {
    fn default() -> Self {
        Self::Statistical
    }
}

impl Default for FailureType {
    fn default() -> Self {
        Self::Unknown
    }
}

impl Default for IndicatorStatus {
    fn default() -> Self {
        Self::Normal
    }
}

impl Default for CriticalityLevel {
    fn default() -> Self {
        Self::Medium
    }
}

impl Default for SubscriptionType {
    fn default() -> Self {
        Self::All
    }
}

// ============================================================================
// Additional Alert and Correlation Types
// ============================================================================

/// Comparison operator for alerts
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Alert index for efficient lookup
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertIndex {
    pub index_name: String,
    pub indexed_fields: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

/// Alert retention policy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertRetentionPolicy {
    pub retention_days: u32,
    pub archive_after_days: u32,
    pub auto_cleanup: bool,
}

/// Alert query
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertQuery {
    pub filters: std::collections::HashMap<String, String>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub limit: Option<usize>,
}

/// Alert storage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertStorageStatistics {
    pub total_alerts: u64,
    pub active_alerts: u64,
    pub storage_size_bytes: u64,
}

/// Correlation engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationEngine {
    pub engine_id: String,
    pub correlation_window: Duration,
    pub enabled: bool,
}

/// Correlation cache
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationCache {
    pub cache_size: usize,
    pub ttl: Duration,
    pub hit_rate: f64,
}

/// Temporal correlator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalCorrelator {
    pub correlation_threshold: f64,
    pub time_window: Duration,
}

/// Spatial correlator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SpatialCorrelator {
    pub correlation_threshold: f64,
    pub scope: String,
}

/// Group type for alert grouping
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupType {
    BySource,
    BySeverity,
    ByCategory,
    ByTimeWindow,
    Custom,
}

/// Group status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GroupStatus {
    Active,
    Resolved,
    Suppressed,
    Escalated,
}

/// Active suppression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ActiveSuppression {
    pub suppression_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub reason: String,
}

/// Suppression scheduler
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SuppressionScheduler {
    pub schedule_interval: Duration,
    pub max_suppressions: usize,
}

/// Dynamic suppression engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DynamicSuppressionEngine {
    pub enabled: bool,
    pub learning_rate: f64,
    pub adaptation_window: Duration,
}

/// Suppression level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum SuppressionLevel {
    Low,
    Medium,
    High,
    Complete,
}

/// Maintenance notification configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MaintenanceNotificationConfig {
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub advance_notice_hours: u32,
}

/// Auto recovery engine
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AutoRecoveryEngine {
    pub enabled: bool,
    pub max_retries: u32,
    pub retry_delay: Duration,
}

/// Recovery condition type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecoveryConditionType {
    MetricReturnsToNormal,
    ManualResolution,
    Timeout,
    AutoRecovery,
}

/// Metric criteria for alerts
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetricCriteria {
    pub metric_name: String,
    pub operator: String,
    pub threshold: f64,
}

/// Validation check
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ValidationCheck {
    pub check_name: String,
    pub check_type: String,
    pub enabled: bool,
}

/// Alert filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertFilter {
    pub filter_expression: String,
    pub include_resolved: bool,
    pub severity_filter: Option<String>,
}

// ============================================================================
// Network and IO Types
// ============================================================================

/// IO bottleneck detection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoBottleneck {
    pub detected: bool,
    pub bottleneck_type: String,
    pub severity: f64,
}

/// Network pattern detection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkPattern {
    pub pattern_type: String,
    pub confidence: f64,
    pub detected_at: DateTime<Utc>,
}

/// Connection characteristics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConnectionCharacteristics {
    pub connection_type: String,
    pub latency_ms: f64,
    pub throughput_mbps: f64,
}

/// Bandwidth utilization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BandwidthUtilization {
    pub current_mbps: f64,
    pub available_mbps: f64,
    pub utilization_percent: f64,
}

/// Network latency profile
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkLatencyProfile {
    pub avg_latency_ms: f64,
    pub p50_latency_ms: f64,
    pub p95_latency_ms: f64,
    pub p99_latency_ms: f64,
}

/// Network reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct NetworkReliability {
    pub packet_loss_rate: f64,
    pub retransmission_rate: f64,
    pub connection_stability: f64,
}

/// Trend type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Trend {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
    Unknown,
}

/// Failure pattern analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailurePatternAnalysis {
    pub pattern_detected: bool,
    pub pattern_type: String,
    pub frequency: f64,
}

/// Recovery characteristics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecoveryCharacteristics {
    pub recovery_time_ms: f64,
    pub success_rate: f64,
    pub retry_count: u32,
}

/// Stability indicators
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StabilityIndicators {
    pub stability_score: f64,
    pub variance: f64,
    pub trend: String,
}

impl Default for ComparisonOperator {
    fn default() -> Self {
        Self::Equal
    }
}

impl Default for GroupType {
    fn default() -> Self {
        Self::BySource
    }
}

impl Default for GroupStatus {
    fn default() -> Self {
        Self::Active
    }
}

impl Default for SuppressionLevel {
    fn default() -> Self {
        Self::Low
    }
}

impl Default for RecoveryConditionType {
    fn default() -> Self {
        Self::MetricReturnsToNormal
    }
}

impl Default for Trend {
    fn default() -> Self {
        Self::Unknown
    }
}

// ============================================================================
// Analytics System Types
// ============================================================================

/// Analytics configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AnalyticsConfig {
    pub enabled: bool,
    pub analysis_interval: Duration,
    pub retention_days: u32,
}

/// Trend detection configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendDetectionConfig {
    pub enabled: bool,
    pub window_size: usize,
    pub sensitivity: f64,
}

/// Sensitivity level
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SensitivityLevel {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Analysis depth
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AnalysisDepth {
    Shallow,
    Normal,
    Deep,
    Comprehensive,
}

/// Optimization strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Performance,
    ResourceEfficiency,
    Balanced,
    Custom,
}

/// Comparison method
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComparisonMethod {
    Absolute,
    Relative,
    Percentage,
    Statistical,
}

/// Drift detection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DriftDetection {
    pub enabled: bool,
    pub threshold: f64,
    pub window_size: usize,
}

/// Baseline type
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BaselineType {
    Static,
    Rolling,
    Adaptive,
    Seasonal,
}

/// Seasonal pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SeasonalPattern {
    pub period: usize,
    pub amplitude: f64,
    pub detected: bool,
}

/// Growth pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GrowthPattern {
    pub growth_rate: f64,
    pub trend: String,
    pub confidence: f64,
}

/// Leak indicator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LeakIndicator {
    pub detected: bool,
    pub leak_rate: f64,
    pub confidence: f64,
}

/// Garbage collection impact metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GcImpactMetrics {
    pub gc_count: u64,
    pub total_pause_time_ms: f64,
    pub avg_pause_time_ms: f64,
}

/// Thread utilization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ThreadUtilization {
    pub thread_count: usize,
    pub utilization_percent: f64,
    pub blocked_threads: usize,
}

/// CPU-bound phase
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CpuBoundPhase {
    pub duration: Duration,
    pub cpu_usage: f64,
    pub start_time: DateTime<Utc>,
}

/// I/O pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IoPattern {
    pub read_ops: u64,
    pub write_ops: u64,
    pub pattern_type: String,
}

/// Latency characteristics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LatencyCharacteristics {
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

/// Disk usage pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DiskUsagePattern {
    pub reads_per_sec: f64,
    pub writes_per_sec: f64,
    pub usage_trend: String,
}

// ============================================================================
// Test Independence Analyzer Types
// ============================================================================

/// Ordering constraint for test execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrderingConstraint {
    pub before_test: String,
    pub after_test: String,
    pub constraint_type: String,
    pub reason: String,
}

/// Cached sharing capability
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CachedSharingCapability {
    pub can_share: bool,
    pub sharing_mode: String,
    pub cache_timestamp: DateTime<Utc>,
    /// Sharing capability result
    pub result: String,
    /// Cached at timestamp
    pub cached_at: DateTime<Utc>,
    /// Confidence score
    pub confidence: f64,
}

impl CachedSharingCapability {
    /// Check if cached data is still valid (within 5 minutes)
    pub fn is_valid(&self) -> bool {
        let now = chrono::Utc::now();
        let age = now.signed_duration_since(self.cached_at);
        age.num_seconds() < 300 // Valid for 5 minutes
    }
}

impl Default for SensitivityLevel {
    fn default() -> Self {
        Self::Medium
    }
}

impl Default for AnalysisDepth {
    fn default() -> Self {
        Self::Normal
    }
}

impl Default for OptimizationStrategy {
    fn default() -> Self {
        Self::Balanced
    }
}

impl Default for ComparisonMethod {
    fn default() -> Self {
        Self::Relative
    }
}

impl Default for BaselineType {
    fn default() -> Self {
        Self::Rolling
    }
}

impl Default for OrderingConstraint {
    fn default() -> Self {
        Self {
            before_test: String::new(),
            after_test: String::new(),
            constraint_type: "temporal".to_string(),
            reason: String::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = TestPerformanceMonitoringConfig::default();
        assert!(config.enable_real_time);
        assert_eq!(config.monitoring_interval, Duration::from_secs(5));
    }

    #[test]
    fn test_pressure_level_ordering() {
        assert!(PressureLevel::Low < PressureLevel::Medium);
        assert!(PressureLevel::Medium < PressureLevel::High);
        assert!(PressureLevel::High < PressureLevel::Critical);
    }

    #[test]
    fn test_alert_thresholds_default() {
        let thresholds = AlertThresholds::default();
        assert_eq!(thresholds.cpu_usage_threshold, 0.8);
        assert_eq!(thresholds.memory_usage_threshold, 0.85);
    }

    #[test]
    fn test_metric_value_conversion() {
        let int_val: MetricValue = 42i64.into_metric_value();
        matches!(int_val, MetricValue::Integer(42));

        let float_val: MetricValue = 3.14f64.into_metric_value();
        matches!(float_val, MetricValue::Float(f) if (f - 3.14).abs() < f64::EPSILON);

        let bool_val: MetricValue = true.into_metric_value();
        matches!(bool_val, MetricValue::Boolean(true));
    }

    #[test]
    fn test_serialization() {
        let config = TestPerformanceMonitoringConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: TestPerformanceMonitoringConfig =
            serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.enable_real_time, deserialized.enable_real_time);
    }
}

// =============================================================================
// DASHBOARD AND WIDGET TYPES
// =============================================================================

/// Dashboard configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardConfig {
    pub dashboard_id: String,
    pub layout: DashboardLayout,
    pub refresh_interval: Duration,
    pub filters: Vec<DashboardFilter>,
}

/// Dashboard layout configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardLayout {
    pub grid_columns: u32,
    pub grid_rows: u32,
    pub widgets: Vec<WidgetPosition>,
}

/// Dashboard filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardFilter {
    pub field: String,
    pub operator: FilterOperator,
    pub value: FilterValue,
}

/// Filter operator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum FilterOperator {
    #[default]
    Equals,
    NotEquals,
    Contains,
    GreaterThan,
    LessThan,
    Between,
}

/// Widget position in dashboard
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetPosition {
    pub widget_id: String,
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Widget size
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetSize {
    pub width: u32,
    pub height: u32,
}

/// Widget configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetConfiguration {
    pub widget_type: String,
    pub data_source: DataSource,
    pub refresh_rate: Duration,
    pub filters: Vec<WidgetFilter>,
}

/// Widget filter
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetFilter {
    pub field: String,
    pub value: FilterValue,
}

/// Widget factory for creating widgets
#[derive(Debug, Clone)]
pub struct WidgetFactory {
    pub widget_types: Vec<String>,
}

/// Widget definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetDefinition {
    pub widget_id: String,
    pub widget_type: String,
    pub configuration: WidgetConfiguration,
}

/// Widget data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct WidgetData {
    pub widget_id: String,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// Widget updater
#[derive(Debug, Clone)]
pub struct WidgetUpdater {
    pub update_interval: Duration,
}

/// Dashboard permissions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardPermissions {
    pub owner: String,
    pub viewers: Vec<String>,
    pub editors: Vec<String>,
}

/// Dashboard subscription
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DashboardSubscription {
    pub subscription_id: String,
    pub dashboard_id: String,
    pub user_id: String,
    pub notification_config: SubscriptionConfig,
}

/// Visualization configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisualizationConfig {
    pub chart_type: String,
    pub color_scheme: Vec<String>,
    pub axes_config: serde_json::Value,
}

/// Data source
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataSource {
    pub source_type: String,
    pub query: String,
    pub parameters: std::collections::HashMap<String, String>,
}

// =============================================================================
// ANALYTICS TYPES
// =============================================================================

/// Outlier analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OutlierAnalysis {
    pub outliers: Vec<Outlier>,
    pub method: String,
    pub threshold: f64,
}

/// Outlier data point
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Outlier {
    pub value: f64,
    pub score: f64,
    pub timestamp: DateTime<Utc>,
}

/// Correlation analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationAnalysis {
    pub correlation_matrix: Vec<Vec<f64>>,
    pub variables: Vec<String>,
    pub method: String,
}

/// Regression analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegressionAnalysis {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub predictions: Vec<f64>,
}

/// Trend components
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TrendComponents {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

/// Cyclical pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CyclicalPattern {
    pub period: Duration,
    pub amplitude: f64,
    pub phase: f64,
}

/// Severity distribution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SeverityDistribution {
    pub critical: u64,
    pub high: u64,
    pub medium: u64,
    pub low: u64,
}

/// Temporal anomaly pattern
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalAnomalyPattern {
    pub pattern_type: String,
    pub start_time: DateTime<Utc>,
    pub duration: Duration,
    pub severity: f64,
}

/// False positive assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FalsePositiveAssessment {
    pub false_positive_rate: f64,
    pub confidence: f64,
    pub evidence: Vec<String>,
}

/// Bottleneck analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BottleneckAnalysisResult {
    pub bottlenecks: Vec<String>,
    pub severity_scores: Vec<f64>,
    pub recommendations: Vec<String>,
}

/// Resource efficiency analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceEfficiencyAnalysis {
    pub efficiency_score: f64,
    pub resource_usage: std::collections::HashMap<String, f64>,
    pub optimization_opportunities: Vec<ImprovementOpportunity>,
}

/// Improvement opportunity
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementOpportunity {
    pub opportunity_type: String,
    pub potential_gain: f64,
    pub implementation_cost: f64,
}

/// Cost benefit analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostBenefitAnalysis {
    pub benefits: f64,
    pub costs: f64,
    pub roi: f64,
    pub payback_period: Duration,
}

/// Optimization risk assessment
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OptimizationRiskAssessment {
    pub risk_level: String,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

/// Comparison type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ComparisonType {
    #[default]
    Absolute,
    Relative,
    Percentage,
}

/// Statistical significance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StatisticalSignificance {
    pub p_value: f64,
    pub confidence_level: f64,
    pub is_significant: bool,
}

/// Regression analysis result with metadata
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RegressionAnalysisResult {
    pub analysis: RegressionAnalysis,
    pub model_quality: f64,
    pub metadata: std::collections::HashMap<String, String>,
}

/// Improvement analysis result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImprovementAnalysisResult {
    pub baseline: f64,
    pub current: f64,
    pub improvement_percentage: f64,
    pub opportunities: Vec<ImprovementOpportunity>,
}

/// Evidence for analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Evidence {
    pub evidence_type: String,
    pub data: serde_json::Value,
    pub confidence: f64,
    pub description: String,
}

/// Impact analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImpactAnalysis {
    pub impact_score: f64,
    pub affected_areas: Vec<String>,
    pub severity: String,
}

/// Implementation roadmap
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ImplementationRoadmap {
    pub phases: Vec<String>,
    pub timeline: Duration,
    pub dependencies: Vec<String>,
}

// =============================================================================
// STORAGE AND RETENTION TYPES
// =============================================================================

// RetentionPolicy is defined in historical_data.rs with more complete fields

/// Partition key for data sharding
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitionKey {
    pub key_type: String,
    pub key_value: String,
}

/// Cleanup scheduler for maintenance tasks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CleanupScheduler {
    pub schedule_interval: Duration,
    pub cleanup_rules: Vec<String>,
    pub enabled: bool,
}

// OrderingConstraint is already defined at line 2943

/// Dependency analyzer type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DependencyAnalyzer {
    pub analysis_depth: u32,
    pub detected_dependencies: Vec<String>,
}

/// Synchronization analyzer
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SynchronizationAnalyzer {
    pub sync_points: Vec<String>,
    pub contention_detected: bool,
}

// =============================================================================
// EVENT SYSTEM TYPES
// =============================================================================

/// Update type for events
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum UpdateType {
    #[default]
    Add,
    Update,
    Delete,
}

/// Update data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UpdateData {
    pub update_type: UpdateType,
    pub data: serde_json::Value,
    pub timestamp: DateTime<Utc>,
}

/// Queued event
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueuedEvent {
    pub event_id: String,
    pub priority: i32,
    pub timestamp: DateTime<Utc>,
}

/// Dispatch worker
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DispatchWorker {
    pub worker_id: String,
    pub status: String,
}

/// Processing rule
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ProcessingRule {
    pub rule_id: String,
    pub condition: String,
    pub action: String,
}

// =============================================================================
// ALERT SYSTEM TYPES
// =============================================================================

/// Threshold direction
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum ThresholdDirection {
    #[default]
    Above,
    Below,
}

/// Alert history entry
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlertHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub alert_type: String,
    pub severity: String,
}

/// Availability status
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AvailabilityStatus {
    #[default]
    Available,
    Unavailable,
    Degraded,
}

/// Health indicator
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthIndicator {
    pub indicator_name: String,
    pub current_value: f64,
    pub threshold: f64,
}

/// Compliance flag
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceFlag {
    pub flag_name: String,
    pub is_compliant: bool,
}

// =============================================================================
// QUERY AND FILTERING TYPES
// =============================================================================

/// Predicate type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PredicateType {
    #[default]
    Equals,
    NotEquals,
    GreaterThan,
    LessThan,
}

/// Backoff strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct BackoffStrategy {
    pub initial_delay: Duration,
    pub max_delay: Duration,
    pub multiplier: f64,
}

/// Retry predicate
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetryPredicate {
    pub max_attempts: u32,
    pub backoff: BackoffStrategy,
}

/// Subscription error
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionError {
    pub error_code: String,
    pub message: String,
}

/// Group config
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GroupConfig {
    pub group_id: String,
    pub members: Vec<String>,
}

/// Load balancing strategy
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LoadBalancingStrategy {
    #[default]
    RoundRobin,
    LeastConnections,
    Random,
}

/// Failover config
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct FailoverConfig {
    pub enabled: bool,
    pub retry_count: u32,
}

// =============================================================================
// CORRELATION AND PATTERN TYPES
// =============================================================================

/// Correlation context
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationContext {
    pub context_id: String,
    pub data: HashMap<String, String>,
}

/// Correlation statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationStatistics {
    pub correlation_count: u64,
    pub avg_correlation_time: Duration,
}

/// Correlation logic
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationLogic {
    pub logic_type: String,
    pub parameters: HashMap<String, String>,
}

/// Correlation action
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationAction {
    pub action_type: String,
    pub target: String,
}

/// Pattern match state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PatternMatchState {
    pub matched: bool,
    pub partial_matches: Vec<String>,
}

/// Matching algorithm
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MatchingAlgorithm {
    pub algorithm_name: String,
    pub parameters: HashMap<String, String>,
}

/// Event matcher
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventMatcher {
    pub matcher_id: String,
    pub pattern: String,
}

/// Time constraint
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TimeConstraint {
    pub start_time: Option<DateTime<Utc>>,
    pub end_time: Option<DateTime<Utc>>,
}

/// Context requirement
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContextRequirement {
    pub required_fields: Vec<String>,
    pub optional_fields: Vec<String>,
}

/// Aggregation rule
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregationRule {
    pub rule_id: String,
    pub aggregation_type: String,
    pub window_size: Duration,
}

// ============================================================================
// Storage and Lifecycle Management Types
// ============================================================================

/// Aggregation window for event aggregation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregationWindow {
    pub window_id: String,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub window_size: Duration,
}

/// Aggregated event data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregatedEvent {
    pub event_id: String,
    pub event_type: String,
    pub count: usize,
    pub aggregation_time: DateTime<Utc>,
    pub data: HashMap<String, serde_json::Value>,
}

/// Aggregation scheduler
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AggregationScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub window_size: Duration,
}

/// Enrichment data for events
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnrichmentData {
    pub source: String,
    pub data: HashMap<String, serde_json::Value>,
    pub timestamp: DateTime<Utc>,
}

/// Cost of enrichment operation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnrichmentCost {
    pub cpu_time: Duration,
    pub memory_bytes: usize,
    pub io_operations: usize,
}

/// Criteria for data deletion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeletionCriteria {
    pub age: Duration,
    pub size_threshold: usize,
    pub access_count: usize,
}

/// Event index for fast lookup
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventIndex {
    pub index_id: String,
    pub indexed_fields: Vec<String>,
    pub index_type: String,
}

/// Configuration for indexing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IndexingConfig {
    pub index_type: String,
    pub fields: Vec<String>,
    pub refresh_interval: Duration,
}

/// Statistics about data retention
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetentionStatistics {
    pub total_size: usize,
    pub retained_count: usize,
    pub deleted_count: usize,
    pub last_cleanup: DateTime<Utc>,
}

/// Strategy for data partitioning
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitioningStrategy {
    pub strategy_type: String,
    pub partition_key: String,
    pub partition_count: usize,
}

/// Storage optimization settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageOptimization {
    pub compression_enabled: bool,
    pub deduplication_enabled: bool,
    pub archival_enabled: bool,
}

/// Metadata about compression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompressionMetadata {
    pub algorithm: String,
    pub compression_ratio: f64,
    pub compressed_size: usize,
    pub original_size: usize,
}

/// Partitioning scheme definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitioningScheme {
    pub scheme_id: String,
    pub partition_key: String,
    pub partition_count: usize,
    pub strategy: String,
}

/// Index for a partition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitionIndex {
    pub partition_id: String,
    pub index_type: String,
    pub indexed_fields: Vec<String>,
}

/// Statistics about a partition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PartitionStatistics {
    pub partition_id: String,
    pub size: usize,
    pub record_count: usize,
    pub last_modified: DateTime<Utc>,
}

/// Status of a partition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum PartitionStatus {
    #[default]
    Active,
    Archived,
    Deleting,
    Error,
}

/// Statistics about event streams
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StreamStatistics {
    pub stream_id: String,
    pub event_count: usize,
    pub bytes_processed: usize,
    pub throughput: f64,
}

/// Executor for retention policies
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetentionExecutor {
    pub executor_id: String,
    pub schedule: String,
    pub last_run: Option<DateTime<Utc>>,
}

/// Manager for compliance requirements
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceManager {
    pub manager_id: String,
    pub compliance_rules: Vec<String>,
    pub audit_log_enabled: bool,
}

/// Strategy for data deletion
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeletionStrategy {
    pub strategy_type: String,
    pub criteria: Vec<String>,
    pub dry_run: bool,
}

/// Compliance requirement definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceRequirement {
    pub requirement_id: String,
    pub description: String,
    pub enforcement_level: String,
}

/// Cost optimization settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostOptimization {
    pub optimization_type: String,
    pub target_cost: f64,
    pub enabled: bool,
}

/// Storage class for data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum StorageClass {
    #[default]
    Hot,
    Warm,
    Cold,
    Archive,
}

/// Access frequency tracking
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessFrequency {
    pub access_count: usize,
    pub last_access: DateTime<Utc>,
    pub frequency_score: f64,
}

/// Availability level requirement
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AvailabilityLevel {
    #[default]
    High,
    Medium,
    Low,
    Archive,
}

/// Condition for lifecycle transitions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleCondition {
    pub condition_type: String,
    pub threshold: f64,
    pub duration: Duration,
}

/// Action to take in lifecycle
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleAction {
    pub action_type: String,
    pub parameters: HashMap<String, String>,
    pub priority: i32,
}

/// Scheduler for compression tasks
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompressionScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub compression_level: i32,
}

/// Statistics about compression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompressionStatistics {
    pub total_compressed: usize,
    pub total_saved: usize,
    pub compression_ratio: f64,
    pub last_compression: DateTime<Utc>,
}

/// Trigger for compression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CompressionTrigger {
    pub trigger_type: String,
    pub threshold: f64,
    pub enabled: bool,
}

/// Algorithm selection for compression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AlgorithmSelection {
    pub algorithm: String,
    pub level: i32,
    pub auto_select: bool,
}

/// Compression level setting
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum CompressionLevel {
    #[default]
    None,
    Fast,
    Standard,
    Maximum,
}

/// Quality settings for compression
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualitySettings {
    pub quality_level: i32,
    pub preserve_metadata: bool,
    pub verify_integrity: bool,
}

/// Performance targets for operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceTargets {
    pub max_latency: Duration,
    pub min_throughput: f64,
    pub max_cpu_usage: f64,
}

/// Scheduler for archival operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalScheduler {
    pub scheduler_id: String,
    pub schedule: String,
    pub retention_period: Duration,
}

/// Index for archived data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalIndex {
    pub archive_id: String,
    pub indexed_fields: Vec<String>,
    pub index_type: String,
}

/// Cache for retrieval operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalCache {
    pub cache_id: String,
    pub max_size: usize,
    pub ttl: Duration,
}

/// Trigger for archival
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ArchivalTrigger {
    pub trigger_type: String,
    pub age_threshold: Duration,
    pub size_threshold: usize,
}

/// Metadata preservation settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MetadataPreservation {
    pub preserve_timestamps: bool,
    pub preserve_permissions: bool,
    pub preserve_attributes: bool,
}

/// Options for data retrieval
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RetrievalOptions {
    pub priority: i32,
    pub timeout: Duration,
    pub decompress: bool,
}

/// Tracker for lifecycle states
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleStateTracker {
    pub state_id: String,
    pub current_state: String,
    pub last_transition: DateTime<Utc>,
}

/// Executor for lifecycle transitions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransitionExecutor {
    pub executor_id: String,
    pub transition_type: String,
    pub status: String,
}

/// Manager for lifecycle events
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleEventManager {
    pub manager_id: String,
    pub event_handlers: Vec<String>,
    pub enabled: bool,
}

/// Optimizer for cost management
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostOptimizer {
    pub optimizer_id: String,
    pub optimization_strategy: String,
    pub target_cost: f64,
}

/// Rule for lifecycle transitions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TransitionRule {
    pub rule_id: String,
    pub source_state: String,
    pub target_state: String,
    pub conditions: Vec<String>,
}

/// Constraint on cost
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostConstraint {
    pub constraint_type: String,
    pub max_cost: f64,
    pub enforcement_level: String,
}

/// Rule for compliance
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComplianceRule {
    pub rule_id: String,
    pub description: String,
    pub enforcement_level: String,
    pub audit_required: bool,
}

/// Configuration for lifecycle monitoring
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct LifecycleMonitoringConfig {
    pub monitoring_enabled: bool,
    pub alert_threshold: f64,
    pub check_interval: Duration,
}

/// Type of lifecycle stage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum LifecycleStageType {
    #[default]
    Active,
    Transitioning,
    Archived,
    Deleted,
}

/// Requirements for storage
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StorageRequirements {
    pub min_capacity: usize,
    pub storage_class: String,
    pub redundancy_level: i32,
}

/// Requirements for access patterns
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct AccessPatternRequirements {
    pub expected_iops: f64,
    pub access_frequency: String,
    pub latency_requirement: Duration,
}

/// Targets for cost optimization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CostTargets {
    pub max_monthly_cost: f64,
    pub cost_per_gb: f64,
    pub optimization_priority: i32,
}

/// Requirements for data quality
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QualityRequirements {
    pub min_quality_score: f64,
    pub validation_required: bool,
    pub integrity_checks: bool,
}

// ============================================================================
// Query and Analysis Types
// ============================================================================

/// Parser for query expressions
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryParser {
    pub parser_id: String,
    pub syntax_version: String,
    pub strict_mode: bool,
}

/// Optimizer for query execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryOptimizer {
    pub optimizer_id: String,
    pub optimization_level: i32,
    pub cost_model: String,
}

/// Engine for query execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryExecutionEngine {
    pub engine_id: String,
    pub max_parallelism: usize,
    pub timeout: Duration,
}

/// Cache for query results
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryResultCache {
    pub cache_id: String,
    pub max_size: usize,
    pub ttl: Duration,
}

/// Statistics about query execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QueryStatistics {
    pub execution_time: Duration,
    pub rows_scanned: usize,
    pub rows_returned: usize,
    pub cache_hit: bool,
}

/// Specification for sorting
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SortingSpec {
    pub field: String,
    pub order: String,
    pub nulls_first: bool,
}

/// Format for query output
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum OutputFormat {
    #[default]
    Json,
    Csv,
    Parquet,
    Avro,
}

/// Type of aggregation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum AggregationType {
    #[default]
    Sum,
    Count,
    Avg,
    Min,
    Max,
}

/// Condition for HAVING clause
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HavingCondition {
    pub field: String,
    pub operator: String,
    pub value: serde_json::Value,
}

// ============================================================================
// Report Generation Types
// ============================================================================

/// Validator for report templates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemplateValidator {
    pub validator_id: String,
    pub validation_rules: Vec<String>,
    pub strict_mode: bool,
}

/// Custom report template
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CustomTemplate {
    pub template_id: String,
    pub template_name: String,
    pub content: String,
    pub variables: HashMap<String, String>,
}

/// Styling for reports
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportStyling {
    pub theme: String,
    pub colors: HashMap<String, String>,
    pub fonts: HashMap<String, String>,
}

/// Parameter for report generation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportParameter {
    pub param_name: String,
    pub param_type: String,
    pub default_value: Option<String>,
    pub required: bool,
}

/// Metadata for report templates
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportTemplateMetadata {
    pub template_id: String,
    pub version: String,
    pub author: String,
    pub created_at: DateTime<Utc>,
}

/// Type of report section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SectionType {
    #[default]
    Summary,
    Details,
    Chart,
    Table,
    Text,
}

/// Query for report data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataQuery {
    pub query_id: String,
    pub query_text: String,
    pub parameters: HashMap<String, String>,
}

/// Layout for report sections
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SectionLayout {
    pub layout_type: String,
    pub columns: usize,
    pub spacing: i32,
}

/// Conditional display rules
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ConditionalDisplay {
    pub condition: String,
    pub show_if_true: bool,
    pub fallback_content: Option<String>,
}

/// Aggregator for report data
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DataAggregator {
    pub aggregator_id: String,
    pub aggregation_type: String,
    pub group_by: Vec<String>,
}

/// Engine for data visualization
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VisualizationEngine {
    pub engine_id: String,
    pub supported_types: Vec<String>,
    pub rendering_options: HashMap<String, String>,
}

/// Processor for report content
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ContentProcessor {
    pub processor_id: String,
    pub transformations: Vec<String>,
    pub filters: Vec<String>,
}

/// Metadata for generated reports
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportMetadata {
    pub report_id: String,
    pub generated_at: DateTime<Utc>,
    pub generator: String,
    pub version: String,
}

/// Options for report export
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportOption {
    pub format: String,
    pub compression: bool,
    pub encryption: bool,
}

/// Content of a report section
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SectionContent {
    pub content_type: String,
    pub data: serde_json::Value,
    pub metadata: HashMap<String, String>,
}

/// Visualization definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Visualization {
    pub viz_id: String,
    pub viz_type: String,
    pub data_source: String,
    pub config: HashMap<String, String>,
}

/// Insight from report analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportInsight {
    pub insight_id: String,
    pub description: String,
    pub importance: f64,
    pub action_items: Vec<String>,
}

/// Recommendation from analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Recommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: i32,
    pub estimated_impact: f64,
}

/// Engine for report scheduling
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SchedulerEngine {
    pub engine_id: String,
    pub max_concurrent: usize,
    pub retry_policy: String,
}

/// Manager for report notifications
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportNotificationManager {
    pub manager_id: String,
    pub notification_channels: Vec<String>,
    pub throttle_config: HashMap<String, String>,
}

/// Method for report delivery
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum DeliveryMethod {
    #[default]
    Email,
    Slack,
    Webhook,
    FileSystem,
}

/// Result of report export
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExportResult {
    pub export_id: String,
    pub format: String,
    pub file_path: String,
    pub file_size: usize,
    pub exported_at: DateTime<Utc>,
}

/// Summary of component health
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ComponentHealthSummary {
    pub component_id: String,
    pub health_status: String,
    pub issues_count: usize,
    pub last_check: DateTime<Utc>,
}

/// Critical issue in system
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CriticalIssue {
    pub issue_id: String,
    pub severity: String,
    pub description: String,
    pub affected_components: Vec<String>,
}

/// Recommendation for health improvement
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HealthRecommendation {
    pub recommendation_id: String,
    pub title: String,
    pub description: String,
    pub priority: i32,
    pub estimated_impact: String,
}

/// Metadata for service operations
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ServiceOperationMetadata {
    pub operation_id: String,
    pub operation_type: String,
    pub timestamp: DateTime<Utc>,
    pub duration: Duration,
}

/// Information about test execution
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestExecutionInfo {
    pub test_id: String,
    pub test_name: String,
    pub test_suite: Option<String>,
    pub start_time: DateTime<Utc>,
    pub end_time: Option<DateTime<Utc>>,
    pub status: String,
    pub configuration: std::collections::HashMap<String, String>,
    pub expected_duration: Option<Duration>,
    pub resource_requirements: Option<std::collections::HashMap<String, String>>,
}

/// Schedule for report generation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ReportSchedule {
    pub schedule_id: String,
    pub cron_expression: String,
    pub enabled: bool,
    pub last_run: Option<DateTime<Utc>>,
}

/// Preferences for report delivery
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct DeliveryPreferences {
    pub delivery_method: String,
    pub recipients: Vec<String>,
    pub format: String,
    pub notification_enabled: bool,
}

/// Filter for test selection
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TestFilter {
    pub filter_id: String,
    pub criteria: HashMap<String, String>,
    pub include_pattern: Option<String>,
    pub exclude_pattern: Option<String>,
}

/// Preferences for alert escalation
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EscalationPreferences {
    pub escalation_enabled: bool,
    pub escalation_delay: Duration,
    pub escalation_levels: Vec<String>,
    pub notification_channels: Vec<String>,
}

// ============================================================================
// Notification Settings Types
// ============================================================================

/// Email notification settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EmailNotificationSettings {
    pub enabled: bool,
    pub recipients: Vec<String>,
    pub subject_template: String,
    pub body_template: String,
    pub smtp_server: String,
    pub smtp_port: u16,
}

/// SMS notification settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SmsNotificationSettings {
    pub enabled: bool,
    pub phone_numbers: Vec<String>,
    pub message_template: String,
    pub provider: String,
    pub api_key: String,
}

/// Push notification settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PushNotificationSettings {
    pub enabled: bool,
    pub device_tokens: Vec<String>,
    pub title_template: String,
    pub body_template: String,
    pub priority: i32,
}

/// In-app notification settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct InAppNotificationSettings {
    pub enabled: bool,
    pub user_ids: Vec<String>,
    pub notification_type: String,
    pub auto_dismiss_duration: Option<Duration>,
}

/// Quiet hours configuration
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct QuietHours {
    pub enabled: bool,
    pub start_time: String,
    pub end_time: String,
    pub timezone: String,
    pub days_of_week: Vec<String>,
}

/// Notification frequency settings
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum NotificationFrequency {
    #[default]
    Immediate,
    Batched,
    Hourly,
    Daily,
    Weekly,
}

/// User role definition
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum UserRole {
    #[default]
    Admin,
    Developer,
    Viewer,
    Guest,
}

/// Subscription template type
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub enum SubscriptionTemplateType {
    #[default]
    Custom,
    Alert,
    Report,
    Notification,
}

/// Subscription metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SubscriptionMetrics {
    pub total_subscriptions: usize,
    pub active_subscriptions: usize,
    pub notifications_sent: usize,
    pub notifications_failed: usize,
    pub last_notification: Option<DateTime<Utc>>,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct UsageStatistics {
    pub total_requests: usize,
    pub total_bytes: usize,
    pub average_latency: Duration,
    pub peak_throughput: f64,
    pub error_rate: f64,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PerformanceMetrics {
    pub cpu_usage: f64,
    pub memory_usage: f64,
    pub disk_usage: f64,
    pub network_throughput: f64,
    pub request_latency: Duration,
}

// ============================================================================
// Test Characterization Result Types
// ============================================================================

/// Sequential I/O test result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SequentialIoResult {
    pub throughput: f64,
    pub latency: Duration,
    pub block_size: usize,
    pub total_bytes: usize,
    pub operation_type: String,
}

/// Random I/O test result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RandomIoResult {
    pub iops: f64,
    pub latency: Duration,
    pub queue_depth: usize,
    pub total_operations: usize,
    pub operation_type: String,
}

/// Goodness of fit test result
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GoodnessOfFit {
    pub test_statistic: f64,
    pub p_value: f64,
    pub test_type: String,
    pub distribution: String,
    pub sample_size: usize,
}

/// Correlation network for dependency analysis
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CorrelationNetwork {
    pub nodes: Vec<String>,
    pub edges: Vec<(String, String, f64)>,
    pub correlation_threshold: f64,
    pub network_density: f64,
}
