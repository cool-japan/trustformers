//! Alerting Types

use chrono::{DateTime, Utc};
use std::{collections::HashMap, time::Duration};

// Import common types

// Import types from sibling modules
use super::config::ThresholdConfig;

// Import types from parent modules
pub use super::super::types::{
    ActionType, AdjustmentReason, EstimationAlgorithm, FeedbackProcessor, FeedbackSource,
    OptimizationEventType, PerformanceDataPoint, PerformanceFeedback, PerformanceMeasurement,
    PerformanceTrend, RealTimeMetrics, RecommendedAction, SystemState, TestCharacteristics,
};

// Import SeverityLevel from pattern engine
pub use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;

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
