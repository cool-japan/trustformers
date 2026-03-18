//! Core data structures for threshold monitoring.

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::time::Duration;

// Re-export commonly used types from real_time_metrics/types for use in sub-modules
pub use super::super::types::{
    AlertEvent, RealTimeMetricsError, SeverityLevel, ThresholdConfig, ThresholdDirection,
    TimestampedMetrics,
};

/// Threshold monitoring state
///
/// Current state of threshold monitoring system including active alerts,
/// monitoring statistics, and system health information.
#[derive(Debug, Clone, Default)]
pub struct ThresholdMonitoringState {
    /// Active alerts
    pub active_alerts: HashMap<String, AlertEvent>,

    /// Alert counts by severity
    pub alert_counts: HashMap<SeverityLevel, u64>,

    /// Last evaluation timestamp
    pub last_evaluation: Option<DateTime<Utc>>,

    /// Evaluation performance metrics
    pub evaluation_performance: EvaluationPerformance,

    /// Evaluation statistics
    pub evaluation_stats: EvaluationStatistics,
}

/// Statistics for threshold evaluation
///
/// Performance statistics for threshold evaluation operations including
/// processing times, accuracy metrics, and system performance impact.
#[derive(Debug, Clone, Default)]
pub struct EvaluationStatistics {
    /// Total evaluations performed
    pub total_evaluations: u64,

    /// Average evaluation time
    pub avg_evaluation_time: Duration,

    /// Total alerts generated
    pub total_alerts: u64,

    /// False positive rate
    pub false_positive_rate: f32,

    /// False negative rate
    pub false_negative_rate: f32,

    /// Alert accuracy
    pub alert_accuracy: f32,
}

/// Performance metrics for threshold evaluation
#[derive(Debug, Clone, Default)]
pub struct EvaluationPerformance {
    /// Last evaluation duration
    pub last_evaluation_duration: Duration,

    /// Average evaluation duration
    pub avg_evaluation_duration: Duration,

    /// Peak evaluation duration
    pub peak_evaluation_duration: Duration,

    /// Total evaluation count
    pub total_evaluations: u64,

    /// Evaluations per second
    pub evaluations_per_second: f32,

    /// Memory usage for evaluations
    pub memory_usage_bytes: u64,

    /// CPU overhead percentage
    pub cpu_overhead_percent: f32,
}

/// Threshold evaluation result
///
/// Result of threshold evaluation including violation status, severity,
/// and contextual information.
#[derive(Debug, Clone)]
pub struct ThresholdEvaluation {
    /// Whether threshold was violated
    pub violated: bool,

    /// Evaluation timestamp
    pub timestamp: DateTime<Utc>,

    /// Severity level if violated
    pub severity: SeverityLevel,

    /// Current value
    pub current_value: f64,

    /// Threshold value
    pub threshold_value: f64,

    /// Evaluation confidence
    pub confidence: f32,

    /// Additional context
    pub context: HashMap<String, String>,
}

// SuppressionInfo is now defined in types.rs and imported via use super::super::types::*
// (Removed duplicate definition to avoid E0659 ambiguity errors)

/// Alert correlation information
#[derive(Debug, Clone)]
pub struct CorrelationInfo {
    /// Correlation ID
    pub correlation_id: String,

    /// Related alert IDs
    pub related_alerts: Vec<String>,

    /// Correlation strength (0.0 to 1.0)
    pub correlation_strength: f32,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Time window for correlation
    pub time_window: Duration,
}

/// Types of alert correlation
#[derive(Debug, Clone)]
pub enum CorrelationType {
    /// Temporal correlation (alerts occurring at similar times)
    Temporal,

    /// Causal correlation (one alert likely causing another)
    Causal,

    /// Resource correlation (alerts from same resource)
    Resource,

    /// Pattern correlation (alerts following a known pattern)
    Pattern,

    /// Metric correlation (alerts from correlated metrics)
    Metric,
}
