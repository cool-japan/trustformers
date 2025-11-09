//! Environment configuration and resource management

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

/// Environment resource limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentResourceLimits {
    /// Maximum CPU cores
    pub max_cpu_cores: Option<usize>,

    /// Maximum memory (MB)
    pub max_memory_mb: Option<u64>,

    /// Maximum parallel tests
    pub max_parallel_tests: Option<usize>,

    /// Maximum execution time
    pub max_execution_time: Option<Duration>,

    /// I/O limits
    pub io_limits: Option<IoLimits>,

    /// Network limits
    pub network_limits: Option<NetworkLimits>,

    /// Custom limits
    pub custom_limits: HashMap<String, serde_json::Value>,
}

/// I/O limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoLimits {
    /// Maximum file operations per second
    pub max_file_ops_per_sec: Option<u64>,

    /// Maximum disk usage (MB)
    pub max_disk_usage_mb: Option<u64>,

    /// Maximum temporary files
    pub max_temp_files: Option<usize>,

    /// Maximum file size (MB)
    pub max_file_size_mb: Option<u64>,
}

/// Network limits
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkLimits {
    /// Maximum network requests per second
    pub max_requests_per_sec: Option<u64>,

    /// Maximum bandwidth (MB/s)
    pub max_bandwidth_mbps: Option<f64>,

    /// Maximum concurrent connections
    pub max_concurrent_connections: Option<usize>,

    /// Request timeout
    pub request_timeout: Option<Duration>,
}

/// Environment optimization settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentOptimizationSettings {
    /// Auto-detect optimal parallelism
    pub auto_parallelism: bool,

    /// Adaptive resource allocation
    pub adaptive_resources: bool,

    /// Performance-based optimization
    pub performance_optimization: bool,

    /// Environment-specific tuning
    pub environment_tuning: bool,

    /// Optimization aggressiveness (0.0 to 1.0)
    pub optimization_aggressiveness: f32,

    /// Learning enabled
    pub learning_enabled: bool,

    /// Custom optimization rules
    pub custom_rules: Vec<OptimizationRule>,
}

/// Environment security settings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentSecuritySettings {
    /// Secure configuration storage
    pub secure_config_storage: bool,

    /// Encrypt sensitive data
    pub encrypt_sensitive_data: bool,

    /// Access control enabled
    pub access_control: bool,

    /// Audit logging
    pub audit_logging: bool,

    /// Security policies
    pub security_policies: Vec<SecurityPolicy>,

    /// Compliance standards
    pub compliance_standards: Vec<ComplianceStandard>,
}

/// Environment monitoring configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentMonitoringConfig {
    /// Metrics collection enabled
    pub metrics_enabled: bool,

    /// Logging level
    pub log_level: LogLevel,

    /// Performance monitoring
    pub performance_monitoring: bool,

    /// Resource monitoring
    pub resource_monitoring: bool,

    /// Alert thresholds
    pub alert_thresholds: AlertThresholds,

    /// Monitoring intervals
    pub monitoring_intervals: MonitoringIntervals,

    /// Export targets
    pub export_targets: Vec<ExportTarget>,
}

/// Logging levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LogLevel {
    Error,
    Warn,
    Info,
    Debug,
    Trace,
}

/// Alert thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertThresholds {
    /// CPU usage threshold (%)
    pub cpu_usage_percent: Option<f64>,

    /// Memory usage threshold (%)
    pub memory_usage_percent: Option<f64>,

    /// Execution time threshold (seconds)
    pub execution_time_seconds: Option<f64>,

    /// Error rate threshold (%)
    pub error_rate_percent: Option<f64>,

    /// Custom thresholds
    pub custom_thresholds: HashMap<String, f64>,
}

/// Monitoring intervals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringIntervals {
    /// Metrics collection interval (seconds)
    pub metrics_interval_seconds: u64,

    /// Health check interval (seconds)
    pub health_check_interval_seconds: u64,

    /// Resource usage interval (seconds)
    pub resource_usage_interval_seconds: u64,

    /// Performance metrics interval (seconds)
    pub performance_interval_seconds: u64,
}

/// Export target configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportTarget {
    /// Target name
    pub name: String,

    /// Target type
    pub target_type: ExportTargetType,

    /// Connection configuration
    pub connection: ConnectionConfig,

    /// Export format
    pub format: ExportFormat,

    /// Export filters
    pub filters: Vec<ExportFilter>,

    /// Retry configuration
    pub retry_config: RetryConfig,

    /// Enabled
    pub enabled: bool,
}

/// Export target types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportTargetType {
    /// Prometheus metrics
    Prometheus,

    /// Grafana dashboard
    Grafana,

    /// ElasticSearch
    ElasticSearch,

    /// InfluxDB
    InfluxDb,

    /// DataDog
    DataDog,

    /// New Relic
    NewRelic,

    /// CloudWatch
    CloudWatch,

    /// Azure Monitor
    AzureMonitor,

    /// Google Cloud Monitoring
    GoogleCloudMonitoring,

    /// File export
    File { path: String },

    /// HTTP endpoint
    Http { url: String },

    /// Custom target
    Custom(String),
}

/// Connection configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConnectionConfig {
    /// Endpoint URL
    pub endpoint: Option<String>,

    /// Authentication
    pub auth: Option<AuthConfig>,

    /// TLS configuration
    pub tls: Option<TlsConfig>,

    /// Connection timeout (seconds)
    pub timeout_seconds: Option<u64>,

    /// Custom headers
    pub headers: HashMap<String, String>,

    /// Custom parameters
    pub parameters: HashMap<String, String>,
}

/// Authentication configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AuthConfig {
    /// No authentication
    None,

    /// API key authentication
    ApiKey { key: String },

    /// Bearer token authentication
    Bearer { token: String },

    /// Basic authentication
    Basic { username: String, password: String },

    /// OAuth2 authentication
    OAuth2 {
        client_id: String,
        client_secret: String,
        token_url: String,
    },

    /// Custom authentication
    Custom(HashMap<String, String>),
}

/// Retry configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryConfig {
    /// Maximum retry attempts
    pub max_attempts: usize,

    /// Initial retry delay (seconds)
    pub initial_delay_seconds: f64,

    /// Maximum retry delay (seconds)
    pub max_delay_seconds: f64,

    /// Backoff multiplier
    pub backoff_multiplier: f64,

    /// Jitter enabled
    pub jitter: bool,
}

/// TLS configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TlsConfig {
    /// Certificate path
    pub cert_path: Option<String>,

    /// Private key path
    pub key_path: Option<String>,

    /// CA certificate path
    pub ca_path: Option<String>,

    /// Verify certificates
    pub verify_certs: bool,

    /// Server name for SNI
    pub server_name: Option<String>,
}

/// Export formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFormat {
    Json,
    Yaml,
    Toml,
    Csv,
    Prometheus,
    InfluxLineProtocol,
    Custom(String),
}

/// Export filter
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportFilter {
    /// Filter name
    pub name: String,

    /// Filter type
    pub filter_type: ExportFilterType,

    /// Filter pattern
    pub pattern: String,

    /// Include or exclude
    pub include: bool,
}

/// Export filter types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ExportFilterType {
    /// Metric name filter
    MetricName,

    /// Label filter
    Label { key: String },

    /// Value filter
    Value,

    /// Regex filter
    Regex,

    /// Custom filter
    Custom(String),
}

/// Notification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Notifications enabled
    pub enabled: bool,

    /// Notification channels
    pub channels: Vec<NotificationChannel>,

    /// Notification templates
    pub templates: Vec<NotificationTemplate>,

    /// Notification rules
    pub rules: Vec<NotificationRule>,
}

/// Notification channel
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel name
    pub name: String,

    /// Channel type
    pub channel_type: NotificationChannelType,

    /// Channel configuration
    pub config: NotificationChannelConfig,

    /// Channel enabled
    pub enabled: bool,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationChannelType {
    /// Email notifications
    Email { smtp_server: String, port: u16 },

    /// Slack notifications
    Slack { webhook_url: String },

    /// Microsoft Teams notifications
    Teams { webhook_url: String },

    /// Discord notifications
    Discord { webhook_url: String },

    /// SMS notifications
    Sms { provider: String, api_key: String },

    /// Webhook notifications
    Webhook { url: String },

    /// Custom notification
    Custom(String),
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannelConfig {
    /// Authentication settings
    pub auth: Option<HashMap<String, String>>,

    /// Custom headers
    pub headers: HashMap<String, String>,

    /// Timeout settings
    pub timeout_seconds: Option<u64>,

    /// Retry settings
    pub retry_config: Option<RetryConfig>,
}

/// Notification template
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationTemplate {
    /// Template name
    pub name: String,

    /// Template subject
    pub subject: String,

    /// Template body
    pub body: String,

    /// Template format
    pub format: NotificationFormat,

    /// Template variables
    pub variables: HashMap<String, String>,
}

/// Notification formats
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationFormat {
    Text,
    Html,
    Markdown,
    Json,
    Custom(String),
}

/// Notification rule
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationRule {
    /// Rule name
    pub name: String,

    /// Rule condition
    pub condition: String,

    /// Notification triggers
    pub triggers: Vec<NotificationTrigger>,

    /// Target channels
    pub channels: Vec<String>,

    /// Rule enabled
    pub enabled: bool,
}

/// Notification triggers
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NotificationTrigger {
    /// Test started
    TestStarted,

    /// Test completed
    TestCompleted,

    /// Test failed
    TestFailed,

    /// Optimization started
    OptimizationStarted,

    /// Optimization completed
    OptimizationCompleted,

    /// Error occurred
    Error,

    /// Custom trigger
    Custom(String),
}

// Forward declarations
use super::optimization::OptimizationRule;
use super::security::{ComplianceStandard, SecurityPolicy};
