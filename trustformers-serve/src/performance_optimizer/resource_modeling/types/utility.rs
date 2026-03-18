//! Utility types for resource modeling
//!
//! Supporting types for resource monitoring, alerts, history tracking,
//! and resource constraints.

use super::{
    enums::{HealthStatus, Severity},
    monitoring::TemperatureReading,
};
use chrono::{DateTime, Utc};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, sync::Arc, time::Duration};

/// System resource monitoring coordinator
///
/// Centralized coordinator for monitoring all system resources
/// with configurable monitoring strategies and alert thresholds.
#[derive(Debug)]
pub struct ResourceMonitor {
    /// Monitoring enabled
    pub enabled: Arc<std::sync::atomic::AtomicBool>,

    /// Monitoring interval
    pub interval: Duration,

    /// Alert thresholds
    pub thresholds: HashMap<String, f32>,

    /// Monitoring history
    pub history: Arc<Mutex<Vec<MonitoringSnapshot>>>,
}

/// Point-in-time monitoring snapshot
///
/// Complete system monitoring snapshot including resource utilization,
/// temperature readings, and performance metrics at a specific time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringSnapshot {
    /// Snapshot timestamp
    pub timestamp: DateTime<Utc>,

    /// CPU utilization
    pub cpu_utilization: f32,

    /// Memory utilization
    pub memory_utilization: f32,

    /// I/O utilization
    pub io_utilization: f32,

    /// Network utilization
    pub network_utilization: f32,

    /// Temperature readings
    pub temperature_readings: Vec<TemperatureReading>,

    /// Performance metrics
    pub performance_metrics: HashMap<String, f64>,
}

/// Resource optimization strategy
///
/// Comprehensive optimization strategy including algorithms,
/// parameters, target metrics, and expected outcomes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationStrategy {
    /// Strategy name
    pub name: String,

    /// Strategy description
    pub description: String,

    /// Target resources
    pub target_resources: Vec<String>,

    /// Optimization parameters
    pub parameters: HashMap<String, f64>,

    /// Expected improvement
    pub expected_improvement: f32,

    /// Implementation complexity
    pub complexity: OptimizationComplexity,
}

/// Optimization implementation complexity
///
/// Classification of optimization complexity including implementation
/// effort, risk assessment, and prerequisite requirements.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationComplexity {
    /// Low complexity optimization
    Low,
    /// Medium complexity optimization
    Medium,
    /// High complexity optimization
    High,
    /// Critical complexity optimization
    Critical,
}

/// Performance regression detection
///
/// System for detecting performance regressions including
/// threshold-based detection, trend analysis, and alert generation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionDetection {
    /// Detection enabled
    pub enabled: bool,

    /// Regression thresholds
    pub thresholds: HashMap<String, f32>,

    /// Detection algorithms
    pub algorithms: Vec<String>,

    /// Alert configuration
    pub alert_config: AlertConfiguration,
}

/// Alert configuration and management
///
/// Configuration for alerting system including notification methods,
/// escalation procedures, and alert aggregation rules.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertConfiguration {
    /// Alert enabled
    pub enabled: bool,

    /// Notification methods
    pub notification_methods: Vec<String>,

    /// Alert aggregation window
    pub aggregation_window: Duration,

    /// Escalation thresholds
    pub escalation_thresholds: HashMap<String, u32>,
}

/// System health assessment
///
/// Comprehensive system health assessment including component status,
/// performance indicators, and overall health score.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthAssessment {
    /// Overall health score (0.0 to 1.0)
    pub overall_health: f32,

    /// Component health scores
    pub component_health: HashMap<String, f32>,

    /// Health indicators
    pub health_indicators: Vec<HealthIndicator>,

    /// Assessment timestamp
    pub timestamp: DateTime<Utc>,

    /// Assessment validity
    pub validity_duration: Duration,
}

/// Individual health indicator
///
/// Specific health indicator including metric name, value,
/// status, and severity for health monitoring.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthIndicator {
    /// Indicator name
    pub name: String,

    /// Indicator value
    pub value: f64,

    /// Indicator status
    pub status: HealthStatus,

    /// Indicator severity
    pub severity: Severity,

    /// Indicator description
    pub description: String,
}
