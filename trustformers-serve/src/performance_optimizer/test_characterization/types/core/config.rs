//! Configuration types for test characterization

use serde::{Deserialize, Serialize};
use std::{collections::HashMap, time::Duration};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub retention_days: u32,
}

#[derive(Debug, Clone)]
pub struct EstimationConfig {
    pub safety_margin: f64,
    pub history_retention_limit: usize,
}

#[derive(Debug, Clone)]
pub struct MonitoringConfig {
    /// Event capture enabled
    pub enable_event_capture: bool,
    /// Performance tracking interval
    pub tracking_interval: Duration,
    /// Alert generation enabled
    pub enable_alerts: bool,
    /// Trend analysis depth
    pub trend_analysis_depth: usize,
    /// Monitoring buffer size
    pub buffer_size: usize,
    /// Real-time processing enabled
    pub enable_real_time_processing: bool,
    /// Historical data retention
    pub historical_retention: Duration,
    /// Alert threshold sensitivity
    pub alert_sensitivity: f64,
    /// Performance baseline period
    pub baseline_period: Duration,
    /// Quality assessment frequency
    pub quality_frequency: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PipelineConfig {
    /// Number of worker threads
    pub worker_threads: usize,
    /// Task queue capacity
    pub task_queue_capacity: usize,
    /// Enable parallel execution
    pub enable_parallel_execution: bool,
    /// Resource pool size
    pub resource_pool_size: usize,
    /// Priority scheduling enabled
    pub enable_priority_scheduling: bool,
    /// Quality assurance level
    pub quality_assurance_level: u32,
    /// Result aggregation strategy
    pub aggregation_strategy: String,
    /// Conflict resolution enabled
    pub enable_conflict_resolution: bool,
    /// Pipeline monitoring interval
    pub monitoring_interval: Duration,
    /// Maximum pipeline duration
    pub max_pipeline_duration: Duration,
    /// Error recovery enabled
    pub enable_error_recovery: bool,
    /// Checkpoint interval
    pub checkpoint_interval: Duration,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RealTimeProfilerConfig {
    /// Sampling frequency
    pub sampling_frequency: u64,
    /// Sampling rate
    pub sampling_rate: f64,
    /// Buffer size for samples
    pub buffer_size: usize,
    /// Enable streaming analysis
    pub enable_streaming_analysis: bool,
    /// Analysis window size
    pub analysis_window_size: usize,
    /// Anomaly detection sensitivity
    pub anomaly_sensitivity: f64,
    /// Enable adaptive optimization
    pub enable_adaptive_optimization: bool,
    /// Quality monitoring interval
    pub quality_monitoring_interval: Duration,
    /// Dashboard update frequency
    pub dashboard_update_frequency: Duration,
    /// Alert threshold configuration
    pub alert_thresholds: HashMap<String, f64>,
    /// Stream retention period
    pub stream_retention_period: Duration,
    /// Metrics configuration
    pub metrics_config: String,
    /// Streaming configuration
    pub streaming_config: String,
    /// Optimization configuration
    pub optimization_config: String,
    /// Processing configuration
    pub processing_config: String,
    /// Anomaly configuration
    pub anomaly_config: String,
    /// Insights configuration
    pub insights_config: String,
    /// Trend configuration
    pub trend_config: String,
    /// Strategy configuration
    pub strategy_config: String,
    /// Reporting configuration
    pub reporting_config: String,
}

#[derive(Debug, Clone)]
pub struct StreamingAnalyzerConfig {
    pub window_size: usize,
    pub update_interval: Duration,
    pub analysis_interval: Duration,
    pub enable_trend_analysis: bool,
    pub anomaly_detection_enabled: bool,
    /// Pattern configuration
    pub pattern_config: String,
    /// Statistics configuration
    pub stats_config: String,
    /// Buffer size for streaming
    pub buffer_size: usize,
}

impl Default for StreamingAnalyzerConfig {
    fn default() -> Self {
        Self {
            window_size: 1000,
            update_interval: Duration::from_millis(100),
            analysis_interval: Duration::from_secs(1),
            enable_trend_analysis: true,
            anomaly_detection_enabled: true,
            pattern_config: String::new(),
            stats_config: String::new(),
            buffer_size: 10000,
        }
    }
}

#[derive(Debug, Clone)]
pub struct TestCharacterizationConfig {
    /// Enable detailed analysis
    pub enable_detailed_analysis: bool,
    /// Maximum analysis duration
    pub max_analysis_duration: Duration,
    /// Resource monitoring interval
    pub resource_monitoring_interval: Duration,
    /// Concurrency analysis depth
    pub concurrency_analysis_depth: u32,
    /// Pattern recognition sensitivity
    pub pattern_recognition_sensitivity: f64,
    /// Enable real-time profiling
    pub enable_real_time_profiling: bool,
    /// Cache size limit
    pub cache_size_limit: usize,
    /// History retention period
    pub history_retention_period: Duration,
    /// Quality threshold
    pub quality_threshold: f64,
    /// Analysis timeout
    pub analysis_timeout: Duration,
    pub analysis_timeout_seconds: u64,
}

pub struct CoordinationConfig {
    pub coordination_enabled: bool,
    pub sync_interval: Duration,
}

pub struct FailoverConfig {
    pub enabled: bool,
    pub failover_timeout: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IsolationRequirements {
    pub process_isolation: bool,
    pub thread_isolation: bool,
    pub memory_isolation: bool,
    pub network_isolation: bool,
    pub filesystem_isolation: bool,
    pub custom_isolation: HashMap<String, bool>,
}

pub struct LifecycleMonitoringConfig {
    pub monitoring_enabled: bool,
    pub check_interval: Duration,
}

pub struct MaintenanceNotificationConfig {
    pub notifications_enabled: bool,
    pub notification_channels: Vec<String>,
    pub alert_threshold: f64,
}

pub struct ModelConfig {
    pub model_type: String,
    pub model_parameters: HashMap<String, f64>,
}

pub struct ModelTypeConfig {
    pub model_type: String,
    pub config_options: HashMap<String, String>,
}

pub struct RealTimeMonitoringConfig {
    pub enabled: bool,
    pub sampling_rate: f64,
    pub retention_period: Duration,
}

pub struct SmtpConfig {
    pub server: String,
    pub port: u16,
    pub username: String,
    pub use_tls: bool,
}

#[derive(Debug, Clone)]
pub struct StreamConfiguration {
    pub buffer_size: usize,
    pub sampling_interval: Duration,
    pub compression_enabled: bool,
    pub retention_policy: String,
}

#[derive(Debug, Clone)]
pub struct StreamQualitySettings {
    pub min_quality_score: f64,
    pub max_error_rate: f64,
    pub quality_check_interval: Duration,
    pub auto_quality_adjust: bool,
}

pub struct ConfigurationChange {
    pub parameter_name: String,
    pub old_value: String,
    pub new_value: String,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

pub struct ConvergenceCriteria {
    pub max_iterations: usize,
    pub tolerance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentalAwareness {
    pub environment_factors: HashMap<String, String>,
    pub awareness_level: f64,
}

pub struct TimeConstraint {
    pub max_duration: Duration,
    pub deadline: Option<chrono::DateTime<chrono::Utc>>,
    pub timeout_action: String,
}

pub struct TimeoutRequirements {
    pub estimation_timeout: Duration,
    pub execution_timeout: Duration,
    pub cleanup_timeout: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct LearningConfiguration {
    /// Learning rate
    pub learning_rate: f64,
    /// Momentum factor
    pub momentum: f64,
    /// Regularization strength
    pub regularization: f64,
    /// Batch size
    pub batch_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Convergence threshold
    pub convergence_threshold: f64,
    /// Feature selection method
    pub feature_selection: String,
    /// Cross-validation folds
    pub cv_folds: usize,
    /// Early stopping criteria
    pub early_stopping: bool,
    /// Hyperparameter search space
    pub hyperparameter_space: HashMap<String, Vec<f64>>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EstimationSafetyConstraints {
    pub max_concurrency: usize,
    pub safety_margin: f64,
    pub timeout: Duration,
}
