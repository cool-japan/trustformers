//! Configuration Types

use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import common types

use num_cpus;

// Import types from sibling modules
use super::enums::ThresholdDirection;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

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
