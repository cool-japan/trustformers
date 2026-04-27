//! Config Type Definitions

use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::dashboard::{DashboardFilter, DashboardLayout, WidgetFilter};
use super::enums::{
    AnomalyDetectionAlgorithm, AnomalyScoringMethod, ArchivalFormat, ArchivalSchedule,
    ArchivalStorageType, CompressionAlgorithm, EventStorageType, OutlierDetectionMethod,
    RateLimitingStrategy, ReportSection, StatisticalMethod, SubscriptionPriority, ThresholdAction,
};
use super::storage::{PartitioningStrategy, StorageOptimization};
use super::thresholds::{AlertThresholds, PerformanceThreshold};
use super::utilities::{CleanupSchedule, DataSource};

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct OutlierDetectionConfig {
    pub threshold: f64,
    pub window_size: Duration,
    pub sensitivity: f64,
}

impl Default for OutlierDetectionConfig {
    fn default() -> Self {
        Self {
            threshold: 2.0,
            window_size: Duration::from_secs(300),
            sensitivity: 0.95,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AlertConfig {
    pub enabled: bool,
    pub evaluation_interval: Duration,
    pub notification_channels: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SuppressionConfig {
    pub enabled: bool,
    pub duration: Duration,
    pub rules: Vec<String>,
}

impl Default for SuppressionConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            duration: Duration::from_secs(3600),
            rules: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BusinessHoursConfig {
    pub timezone: String,
    pub start_hour: u8,
    pub end_hour: u8,
    pub days_of_week: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmtpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub use_tls: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmailTemplateConfig {
    pub template_path: String,
    pub default_subject: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmsProviderConfig {
    pub provider: String,
    pub api_key: String,
    pub sender_number: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlackWorkspaceConfig {
    pub workspace_id: String,
    pub bot_token: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SlackBotConfig {
    pub bot_name: String,
    pub default_channel: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebhookEndpointConfig {
    pub url: String,
    pub method: String,
    pub headers: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WebhookRetryConfig {
    pub max_retries: u32,
    pub retry_delay: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MaintenanceNotificationConfig {
    pub enabled: bool,
    pub notification_channels: Vec<String>,
    pub advance_notice_hours: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    pub enabled: bool,
    pub analysis_interval: Duration,
    pub retention_days: u32,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrendDetectionConfig {
    pub enabled: bool,
    pub window_size: usize,
    pub sensitivity: f64,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct DashboardConfig {
    pub dashboard_id: String,
    pub layout: DashboardLayout,
    pub refresh_interval: Duration,
    pub filters: Vec<DashboardFilter>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct WidgetConfiguration {
    pub widget_type: String,
    pub data_source: DataSource,
    pub refresh_rate: Duration,
    pub filters: Vec<WidgetFilter>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualizationConfig {
    pub chart_type: String,
    pub color_scheme: Vec<String>,
    pub axes_config: serde_json::Value,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GroupConfig {
    pub group_id: String,
    pub members: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FailoverConfig {
    pub enabled: bool,
    pub retry_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IndexingConfig {
    pub index_type: String,
    pub fields: Vec<String>,
    pub refresh_interval: Duration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QualitySettings {
    pub quality_level: i32,
    pub preserve_metadata: bool,
    pub verify_integrity: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LifecycleMonitoringConfig {
    pub monitoring_enabled: bool,
    pub alert_threshold: f64,
    pub check_interval: Duration,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeliveryPreferences {
    pub delivery_method: String,
    pub recipients: Vec<String>,
    pub format: String,
    pub notification_enabled: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EscalationPreferences {
    pub escalation_enabled: bool,
    pub escalation_delay: Duration,
    pub escalation_levels: Vec<String>,
    pub notification_channels: Vec<String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EmailNotificationSettings {
    pub enabled: bool,
    pub recipients: Vec<String>,
    pub subject_template: String,
    pub body_template: String,
    pub smtp_server: String,
    pub smtp_port: u16,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SmsNotificationSettings {
    pub enabled: bool,
    pub phone_numbers: Vec<String>,
    pub message_template: String,
    pub provider: String,
    pub api_key: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PushNotificationSettings {
    pub enabled: bool,
    pub device_tokens: Vec<String>,
    pub title_template: String,
    pub body_template: String,
    pub priority: i32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InAppNotificationSettings {
    pub enabled: bool,
    pub user_ids: Vec<String>,
    pub notification_type: String,
    pub auto_dismiss_duration: Option<Duration>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_test_performance_monitoring_config_default() {
        let config = TestPerformanceMonitoringConfig::default();
        assert!(config.enable_real_time);
        assert_eq!(config.monitoring_interval, Duration::from_secs(5));
        assert_eq!(config.retention_period, Duration::from_secs(7 * 24 * 3600));
    }

    #[test]
    fn test_report_config_default() {
        let config = ReportConfig::default();
        assert!(config.generate_detailed_reports);
        assert!(config.include_historical_data);
        assert_eq!(config.export_formats.len(), 2);
        assert!(config.auto_generate_interval.is_some());
        assert_eq!(config.report_sections.len(), 4);
    }

    #[test]
    fn test_stream_config_default() {
        let config = StreamConfig::default();
        assert_eq!(config.max_buffer_size, 10000);
        assert_eq!(config.batch_size, 100);
        assert!(config.enable_compression);
        assert!(config.rate_limiting.is_some());
        assert_eq!(config.priority_levels.len(), 4);
    }

    #[test]
    fn test_monitoring_config_default() {
        let config = MonitoringConfig::default();
        assert_eq!(config.collection_interval, Duration::from_millis(1000));
        assert_eq!(config.metrics_buffer_size, 1000);
        assert_eq!(config.performance_thresholds.len(), 2);
        assert!(!config.enable_detailed_tracing);
    }

    #[test]
    fn test_anomaly_detection_config_default() {
        let config = AnomalyDetectionConfig::default();
        assert!(config.enabled);
        assert!((config.sensitivity - 0.95).abs() < 1e-9);
        assert_eq!(config.training_window_size, 100);
    }

    #[test]
    fn test_data_retention_config_default() {
        let config = DataRetentionConfig::default();
        assert_eq!(config.raw_data_retention, Duration::from_secs(24 * 3600));
        assert_eq!(
            config.aggregated_data_retention,
            Duration::from_secs(30 * 24 * 3600)
        );
    }

    #[test]
    fn test_compression_config_default() {
        let config = CompressionConfig::default();
        assert!(config.enabled);
        assert_eq!(config.compression_level, 6);
    }

    #[test]
    fn test_archival_config_default() {
        let config = ArchivalConfig::default();
        assert!(!config.enabled);
    }

    #[test]
    fn test_event_storage_config_default() {
        let config = EventStorageConfig::default();
        assert_eq!(config.max_storage_size, 100 * 1024 * 1024);
        assert!(config.storage_path.is_none());
        assert!(!config.persistent);
    }

    #[test]
    fn test_event_indexing_config_default() {
        let config = EventIndexingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.indexed_fields.len(), 3);
    }

    #[test]
    fn test_event_retention_config_default() {
        let config = EventRetentionConfig::default();
        assert_eq!(config.max_events, 1000000);
    }

    #[test]
    fn test_rate_limit_config_default() {
        let config = RateLimitConfig::default();
        assert!(config.enabled);
        assert_eq!(config.max_events_per_second, 1000);
        assert_eq!(config.burst_capacity, 2000);
    }

    #[test]
    fn test_correlation_config_default() {
        let config = CorrelationConfig::default();
        assert!(config.enabled);
        assert_eq!(config.correlation_window, Duration::from_secs(60));
        assert_eq!(config.correlation_fields.len(), 2);
    }

    #[test]
    fn test_pattern_config_default() {
        let config = PatternConfig::default();
        assert!(config.enabled);
        assert!(config.pattern_rules.is_empty());
    }

    #[test]
    fn test_enrichment_config_default() {
        let config = EnrichmentConfig::default();
        assert!(!config.enabled);
        assert!(config.enrichment_sources.is_empty());
    }

    #[test]
    fn test_outlier_detection_config_default() {
        let config = OutlierDetectionConfig::default();
        assert!((config.threshold - 2.0).abs() < 1e-9);
        assert!((config.sensitivity - 0.95).abs() < 1e-9);
    }

    #[test]
    fn test_user_preferences_default() {
        let prefs = UserPreferences::default();
        assert!(prefs.email_enabled);
        assert!(!prefs.push_enabled);
        assert_eq!(prefs.notification_frequency, 300);
        assert_eq!(prefs.theme, "light");
    }

    #[test]
    fn test_aggregation_config_default() {
        let config = AggregationConfig::default();
        assert_eq!(config.window_size, Duration::from_secs(60));
        assert!(config.enable_percentiles);
        assert_eq!(config.percentiles.len(), 4);
        assert!(!config.enable_outlier_detection);
    }

    #[test]
    fn test_suppression_config_default() {
        let config = SuppressionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.duration, Duration::from_secs(3600));
        assert!(config.rules.is_empty());
    }

    #[test]
    fn test_historical_data_config_default() {
        let config = HistoricalDataConfig::default();
        assert_eq!(config.retention_days, 30);
        assert!(config.compression_enabled);
    }

    #[test]
    fn test_historical_cache_config_default() {
        let config = HistoricalCacheConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_execution_time_ms, 250);
        assert_eq!(config.max_cacheable_size_mb, 512);
        assert_eq!(config.max_entries, 10_000);
    }

    #[test]
    fn test_real_time_monitoring_config_default() {
        let config = RealTimeMonitoringConfig::default();
        assert!(config.enabled);
        assert_eq!(config.sampling_interval_ms, 1000);
        assert!((config.alert_threshold - 0.8).abs() < 1e-9);
        assert!(config.compression_enabled);
        assert_eq!(config.buffer_size, 10_000);
    }

    #[test]
    fn test_alert_config_default() {
        let config = AlertConfig::default();
        assert!(!config.enabled);
        assert!(config.notification_channels.is_empty());
    }

    #[test]
    fn test_subscription_config_default() {
        let config = SubscriptionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.max_subscriptions_per_user, 0);
    }

    #[test]
    fn test_quality_settings_default() {
        let settings = QualitySettings::default();
        assert_eq!(settings.quality_level, 0);
        assert!(!settings.preserve_metadata);
        assert!(!settings.verify_integrity);
    }

    #[test]
    fn test_trend_detection_config_default() {
        let config = TrendDetectionConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.window_size, 0);
    }

    #[test]
    fn test_analytics_config_default() {
        let config = AnalyticsConfig::default();
        assert!(!config.enabled);
        assert_eq!(config.retention_days, 0);
    }

    #[test]
    fn test_dashboard_config_default() {
        let config = DashboardConfig::default();
        assert!(config.dashboard_id.is_empty());
        assert!(config.filters.is_empty());
    }
}
