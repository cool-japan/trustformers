//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{Context, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::Arc, time::Duration,
};
// use super::types::*; // Commented to fix circular import

use std::collections::{VecDeque, HashMap};

/// Resource efficiency metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EfficiencyMetrics {
    /// Port allocation efficiency (0.0 - 1.0)
    pub allocation_efficiency: f32,
    /// Resource utilization efficiency (0.0 - 1.0)
    pub utilization_efficiency: f32,
    /// Conflict resolution efficiency (0.0 - 1.0)
    pub conflict_resolution_efficiency: f32,
    /// Overall system efficiency (0.0 - 1.0)
    pub overall_efficiency: f32,
    /// Resource fragmentation level (0.0 - 1.0)
    pub fragmentation_level: f32,
    /// Waste percentage (0.0 - 100.0)
    pub waste_percentage: f32,
}
/// Predictive health indicators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictiveIndicators {
    /// Predicted utilization trend (increasing/decreasing)
    pub utilization_trend: TrendDirection,
    /// Predicted time until capacity exhaustion (if applicable)
    pub time_to_exhaustion: Option<Duration>,
    /// Risk assessment score (0.0 - 1.0)
    pub risk_score: f32,
    /// Anomaly detection confidence (0.0 - 1.0)
    pub anomaly_confidence: f32,
    /// Predicted performance degradation risk (0.0 - 1.0)
    pub degradation_risk: f32,
    /// Maintenance recommendation urgency (0.0 - 1.0)
    pub maintenance_urgency: f32,
}
/// Health event for tracking system health changes over time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortHealthEvent {
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Health status at the time of event
    pub status: HealthStatus,
    /// Event type classification
    pub event_type: HealthEventType,
    /// Metrics captured during this event
    pub metrics: HashMap<String, f64>,
    /// Alerts generated during this event
    pub alerts: Vec<String>,
    /// Additional contextual details
    pub details: HashMap<String, String>,
    /// Event severity level
    pub severity: EventSeverity,
    /// Duration of the event (if applicable)
    pub duration: Option<Duration>,
    /// Related component or subsystem
    pub component: String,
}
/// Performance baseline for health assessment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBaseline {
    /// Baseline allocation time in milliseconds
    pub baseline_allocation_time_ms: f64,
    /// Baseline utilization percentage
    pub baseline_utilization: f32,
    /// Baseline conflict rate per minute
    pub baseline_conflict_rate: f32,
    /// Baseline efficiency metrics
    pub baseline_efficiency: EfficiencyMetrics,
    /// When the baseline was established
    pub established_at: DateTime<Utc>,
    /// Whether the baseline is valid
    pub is_valid: bool,
    /// Number of samples used for baseline
    pub sample_count: usize,
}
/// Health alert thresholds for various metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortHealthThresholds {
    /// Warning threshold for utilization percentage
    pub utilization_warning: f32,
    /// Critical threshold for utilization percentage
    pub utilization_critical: f32,
    /// Warning threshold for conflicts per minute
    pub conflicts_per_minute_warning: f32,
    /// Critical threshold for conflicts per minute
    pub conflicts_per_minute_critical: f32,
    /// Warning threshold for average allocation time (ms)
    pub allocation_time_warning_ms: f64,
    /// Critical threshold for average allocation time (ms)
    pub allocation_time_critical_ms: f64,
    /// Warning threshold for health score
    pub health_score_warning: f32,
    /// Critical threshold for health score
    pub health_score_critical: f32,
    /// Warning threshold for error rate percentage
    pub error_rate_warning: f32,
    /// Critical threshold for error rate percentage
    pub error_rate_critical: f32,
    /// Warning threshold for resource fragmentation
    pub fragmentation_warning: f32,
    /// Critical threshold for resource fragmentation
    pub fragmentation_critical: f32,
}
/// Alert status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertStatus {
    /// Alert is active
    Active,
    /// Alert has been acknowledged
    Acknowledged,
    /// Alert has been resolved
    Resolved,
    /// Alert has been suppressed
    Suppressed,
}
/// Current health status of the port management system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortHealthStatus {
    /// Overall health status level
    pub overall_status: HealthStatus,
    /// Timestamp of last health check
    pub last_check: DateTime<Utc>,
    /// Number of available ports
    pub available_ports: usize,
    /// Number of currently allocated ports
    pub allocated_ports: usize,
    /// Number of reserved ports
    pub reserved_ports: usize,
    /// Current utilization percentage (0.0 - 100.0)
    pub utilization_percent: f32,
    /// Number of conflicts detected in recent check period
    pub recent_conflicts: usize,
    /// Average allocation time in milliseconds
    pub avg_allocation_time_ms: f64,
    /// Currently active alerts
    pub active_alerts: Vec<String>,
    /// Health score (0.0 - 100.0, higher is better)
    pub health_score: f32,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Last significant event timestamp
    pub last_event: Option<DateTime<Utc>>,
    /// Resource efficiency metrics
    pub efficiency_metrics: EfficiencyMetrics,
    /// Predictive indicators
    pub predictive_indicators: PredictiveIndicators,
}
/// Port Health Monitoring System
///
/// Provides comprehensive health monitoring for the port management system,
/// including real-time status assessment, alert generation, and trend analysis.
#[derive(Debug)]
pub struct PortHealthMonitor {
    /// Current health status
    health_status: Arc<Mutex<PortHealthStatus>>,
    /// Health monitoring configuration
    config: Arc<RwLock<PortHealthConfig>>,
    /// Historical health events for trend analysis
    health_history: Arc<Mutex<VecDeque<PortHealthEvent>>>,
    /// Alert threshold configuration
    alert_thresholds: Arc<RwLock<PortHealthThresholds>>,
    /// Active alerts by alert ID
    active_alerts: Arc<Mutex<HashMap<String, PortHealthAlert>>>,
    /// Health trend analysis data
    trend_analysis: Arc<Mutex<HealthTrendAnalysis>>,
    /// Performance baseline for comparison
    performance_baseline: Arc<RwLock<PerformanceBaseline>>,
}
impl PortHealthMonitor {
    /// Create a new health monitor with default configuration
    ///
    /// # Returns
    ///
    /// A new PortHealthMonitor instance or an error if initialization fails
    ///
    /// # Errors
    ///
    /// Returns an error if internal initialization fails
    #[instrument]
    pub async fn new() -> Result<Self, PortManagementError> {
        let health_monitor = Self {
            health_status: Arc::new(
                Mutex::new(PortHealthStatus {
                    overall_status: HealthStatus::Unknown,
                    last_check: Utc::now(),
                    available_ports: 0,
                    allocated_ports: 0,
                    reserved_ports: 0,
                    utilization_percent: 0.0,
                    recent_conflicts: 0,
                    avg_allocation_time_ms: 0.0,
                    active_alerts: Vec::new(),
                    health_score: 100.0,
                    uptime_seconds: 0,
                    last_event: None,
                    efficiency_metrics: EfficiencyMetrics::default(),
                    predictive_indicators: PredictiveIndicators::default(),
                }),
            ),
            config: Arc::new(RwLock::new(PortHealthConfig::default())),
            health_history: Arc::new(Mutex::new(VecDeque::new())),
            alert_thresholds: Arc::new(RwLock::new(PortHealthThresholds::default())),
            active_alerts: Arc::new(Mutex::new(HashMap::new())),
            trend_analysis: Arc::new(Mutex::new(HealthTrendAnalysis::default())),
            performance_baseline: Arc::new(RwLock::new(PerformanceBaseline::default())),
        };
        info!("PortHealthMonitor initialized successfully");
        Ok(health_monitor)
    }
    /// Get current health status
    ///
    /// # Returns
    ///
    /// Current health status snapshot
    #[instrument(skip(self))]
    pub async fn get_health_status(&self) -> PortHealthStatus {
        let status = self.health_status.lock();
        status.clone()
    }
    /// Update health status with comprehensive analysis
    ///
    /// # Arguments
    ///
    /// * `available_ports` - Number of available ports
    /// * `allocated_ports` - Number of allocated ports
    /// * `reserved_ports` - Number of reserved ports
    /// * `recent_conflicts` - Number of recent conflicts
    /// * `avg_allocation_time_ms` - Average allocation time in milliseconds
    #[instrument(skip(self))]
    pub async fn update_health_status(
        &self,
        available_ports: usize,
        allocated_ports: usize,
        reserved_ports: usize,
        recent_conflicts: usize,
        avg_allocation_time_ms: f64,
    ) {
        let thresholds = self.alert_thresholds.read();
        let config = self.config.read();
        if !config.enabled {
            return;
        }
        let total_ports = available_ports + allocated_ports;
        let utilization_percent = if total_ports > 0 {
            (allocated_ports as f32 / total_ports as f32) * 100.0
        } else {
            0.0
        };
        let health_score = self
            .calculate_health_score(
                utilization_percent,
                avg_allocation_time_ms,
                recent_conflicts,
                &thresholds,
            )
            .await;
        let overall_status = self
            .determine_health_status(
                utilization_percent,
                avg_allocation_time_ms,
                health_score,
                &thresholds,
            )
            .await;
        let efficiency_metrics = self
            .calculate_efficiency_metrics(
                available_ports,
                allocated_ports,
                recent_conflicts,
                avg_allocation_time_ms,
            )
            .await;
        let predictive_indicators = self
            .calculate_predictive_indicators(
                utilization_percent,
                avg_allocation_time_ms,
                recent_conflicts,
            )
            .await;
        let active_alerts = self
            .process_alerts(
                utilization_percent,
                avg_allocation_time_ms,
                health_score,
                recent_conflicts,
                &thresholds,
            )
            .await;
        self.update_trend_analysis(
                utilization_percent,
                avg_allocation_time_ms,
                recent_conflicts as f32,
                health_score,
            )
            .await;
        let mut status = self.health_status.lock();
        let previous_status = status.overall_status.clone();
        *status = PortHealthStatus {
            overall_status: overall_status.clone(),
            last_check: Utc::now(),
            available_ports,
            allocated_ports,
            reserved_ports,
            utilization_percent,
            recent_conflicts,
            avg_allocation_time_ms,
            active_alerts: active_alerts.clone(),
            health_score,
            uptime_seconds: status.uptime_seconds + config.check_interval.as_secs(),
            last_event: if previous_status != overall_status {
                Some(Utc::now())
            } else {
                status.last_event
            },
            efficiency_metrics,
            predictive_indicators,
        };
        if previous_status != overall_status {
            self.record_health_event(
                    overall_status.clone(),
                    HealthEventType::StatusChange,
                    EventSeverity::Info,
                    format!(
                        "Health status changed from {:?} to {:?}", previous_status,
                        overall_status
                    ),
                    HashMap::new(),
                )
                .await;
        }
        if config.enable_detailed_logging {
            debug!(
                "Health status updated: {:?}, utilization: {:.1}%, score: {:.1}, alerts: {}",
                overall_status, utilization_percent, health_score, active_alerts.len()
            );
        }
    }
    /// Calculate comprehensive health score
    async fn calculate_health_score(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        recent_conflicts: usize,
        thresholds: &PortHealthThresholds,
    ) -> f32 {
        let mut score = 100.0;
        let utilization_impact = if utilization_percent > thresholds.utilization_critical
        {
            -40.0
        } else if utilization_percent > thresholds.utilization_warning {
            -20.0 * (utilization_percent - thresholds.utilization_warning)
                / (thresholds.utilization_critical - thresholds.utilization_warning)
        } else {
            0.0
        };
        let performance_impact = if avg_allocation_time_ms
            > thresholds.allocation_time_critical_ms
        {
            -25.0
        } else if avg_allocation_time_ms > thresholds.allocation_time_warning_ms {
            -15.0 * (avg_allocation_time_ms - thresholds.allocation_time_warning_ms)
                / (thresholds.allocation_time_critical_ms
                    - thresholds.allocation_time_warning_ms)
        } else {
            0.0
        };
        let conflict_impact = if recent_conflicts as f32
            > thresholds.conflicts_per_minute_critical
        {
            -20.0
        } else if recent_conflicts as f32 > thresholds.conflicts_per_minute_warning {
            -10.0 * (recent_conflicts as f32 - thresholds.conflicts_per_minute_warning)
                / (thresholds.conflicts_per_minute_critical
                    - thresholds.conflicts_per_minute_warning)
        } else {
            0.0
        };
        let stability_impact = self.calculate_stability_impact().await;
        let efficiency_impact = self.calculate_efficiency_impact().await;
        score
            += utilization_impact + performance_impact + conflict_impact
                + stability_impact + efficiency_impact;
        score.clamp(0.0, 100.0)
    }
    /// Calculate stability impact on health score
    async fn calculate_stability_impact(&self) -> f32 {
        let trend_analysis = self.trend_analysis.lock();
        let recent_events = self.health_history.lock();
        let recent_changes = recent_events
            .iter()
            .rev()
            .take(10)
            .filter(|event| matches!(event.event_type, HealthEventType::StatusChange))
            .count();
        match recent_changes {
            0..=1 => 0.0,
            2..=3 => -2.0,
            4..=5 => -5.0,
            _ => -10.0,
        }
    }
    /// Calculate efficiency impact on health score
    async fn calculate_efficiency_impact(&self) -> f32 {
        let baseline = self.performance_baseline.read();
        if !baseline.is_valid {
            return 0.0;
        }
        let current_status = self.health_status.lock();
        let efficiency_ratio = current_status.efficiency_metrics.overall_efficiency
            / baseline.baseline_efficiency.overall_efficiency;
        match efficiency_ratio {
            r if r >= 1.0 => 5.0,
            r if r >= 0.8 => 0.0,
            r if r >= 0.6 => -3.0,
            _ => -8.0,
        }
    }
    /// Determine health status based on multiple factors
    async fn determine_health_status(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        health_score: f32,
        thresholds: &PortHealthThresholds,
    ) -> HealthStatus {
        if utilization_percent >= thresholds.utilization_critical
            || avg_allocation_time_ms >= thresholds.allocation_time_critical_ms
            || health_score <= thresholds.health_score_critical
        {
            return HealthStatus::Critical;
        }
        if health_score <= thresholds.health_score_warning * 0.8 {
            return HealthStatus::Degraded;
        }
        if utilization_percent >= thresholds.utilization_warning
            || avg_allocation_time_ms >= thresholds.allocation_time_warning_ms
            || health_score <= thresholds.health_score_warning
        {
            return HealthStatus::Warning;
        }
        HealthStatus::Healthy
    }
    /// Calculate efficiency metrics
    async fn calculate_efficiency_metrics(
        &self,
        available_ports: usize,
        allocated_ports: usize,
        recent_conflicts: usize,
        avg_allocation_time_ms: f64,
    ) -> EfficiencyMetrics {
        let total_ports = available_ports + allocated_ports;
        let allocation_efficiency = if avg_allocation_time_ms <= 50.0 {
            1.0
        } else if avg_allocation_time_ms <= 200.0 {
            1.0 - (avg_allocation_time_ms - 50.0) / 150.0 * 0.5
        } else {
            0.5
        };
        let utilization_efficiency = if total_ports > 0 {
            let utilization = allocated_ports as f32 / total_ports as f32;
            if utilization <= 0.8 {
                utilization / 0.8
            } else {
                1.0 - (utilization - 0.8) / 0.2 * 0.3
            }
        } else {
            1.0
        };
        let conflict_resolution_efficiency = if recent_conflicts == 0 {
            1.0
        } else if recent_conflicts <= 5 {
            1.0 - recent_conflicts as f32 * 0.1
        } else {
            0.5
        };
        let fragmentation_level = self
            .calculate_fragmentation_level(available_ports, allocated_ports)
            .await;
        let overall_efficiency = (allocation_efficiency + utilization_efficiency
            + conflict_resolution_efficiency) / 3.0;
        let waste_percentage = if total_ports > 0 {
            fragmentation_level * 100.0
        } else {
            0.0
        };
        EfficiencyMetrics {
            allocation_efficiency,
            utilization_efficiency,
            conflict_resolution_efficiency,
            overall_efficiency,
            fragmentation_level,
            waste_percentage,
        }
    }
    /// Calculate resource fragmentation level
    async fn calculate_fragmentation_level(
        &self,
        _available_ports: usize,
        _allocated_ports: usize,
    ) -> f32 {
        0.1
    }
    /// Calculate predictive indicators
    async fn calculate_predictive_indicators(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        recent_conflicts: usize,
    ) -> PredictiveIndicators {
        let trend_analysis = self.trend_analysis.lock();
        let utilization_trend = self.analyze_trend(&trend_analysis.utilization_history);
        let risk_score = self
            .calculate_risk_score(
                utilization_percent,
                avg_allocation_time_ms,
                recent_conflicts,
            );
        let time_to_exhaustion = if matches!(
            utilization_trend, TrendDirection::Increasing
        ) {
            self.estimate_time_to_exhaustion(&trend_analysis.utilization_history)
        } else {
            None
        };
        let anomaly_confidence = self.detect_anomalies(&trend_analysis);
        let degradation_risk = self
            .assess_degradation_risk(
                utilization_percent,
                avg_allocation_time_ms,
                &trend_analysis,
            );
        let maintenance_urgency = self
            .calculate_maintenance_urgency(
                risk_score,
                degradation_risk,
                anomaly_confidence,
            );
        PredictiveIndicators {
            utilization_trend,
            time_to_exhaustion,
            risk_score,
            anomaly_confidence,
            degradation_risk,
            maintenance_urgency,
        }
    }
    /// Analyze trend direction from historical data
    fn analyze_trend(&self, history: &VecDeque<(DateTime<Utc>, f32)>) -> TrendDirection {
        if history.len() < 3 {
            return TrendDirection::Unknown;
        }
        let recent_values: Vec<f32> = history
            .iter()
            .rev()
            .take(5)
            .map(|(_, value)| *value)
            .collect();
        if recent_values.len() < 3 {
            return TrendDirection::Unknown;
        }
        let first = recent_values[recent_values.len() - 1];
        let last = recent_values[0];
        let difference = last - first;
        if difference > 2.0 {
            TrendDirection::Increasing
        } else if difference < -2.0 {
            TrendDirection::Decreasing
        } else {
            TrendDirection::Stable
        }
    }
    /// Calculate overall risk score
    fn calculate_risk_score(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        recent_conflicts: usize,
    ) -> f32 {
        let mut risk = 0.0;
        risk += (utilization_percent / 100.0) * 0.4;
        risk += ((avg_allocation_time_ms / 1000.0).min(1.0)) * 0.3;
        risk += ((recent_conflicts as f32 / 10.0).min(1.0)) * 0.3;
        risk.min(1.0)
    }
    /// Estimate time until resource exhaustion
    fn estimate_time_to_exhaustion(
        &self,
        history: &VecDeque<(DateTime<Utc>, f32)>,
    ) -> Option<Duration> {
        if history.len() < 5 {
            return None;
        }
        let recent_data: Vec<(DateTime<Utc>, f32)> = history
            .iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        let x_values: Vec<i64> = recent_data
            .iter()
            .enumerate()
            .map(|(i, _)| i as i64)
            .collect();
        let y_values: Vec<f32> = recent_data.iter().map(|(_, y)| *y).collect();
        if let Some(slope) = self.calculate_slope(&x_values, &y_values) {
            if slope > 0.1 {
                let current_utilization = y_values[0];
                let remaining_capacity = 100.0 - current_utilization;
                let time_per_unit = Duration::from_secs(3600);
                let estimated_hours = (remaining_capacity / slope) as u64;
                return Some(Duration::from_secs(estimated_hours * 3600));
            }
        }
        None
    }
    /// Calculate slope for trend analysis
    fn calculate_slope(&self, x_values: &[i64], y_values: &[f32]) -> Option<f32> {
        if x_values.len() != y_values.len() || x_values.len() < 2 {
            return None;
        }
        let n = x_values.len() as f32;
        let sum_x: f32 = x_values.iter().map(|&x| x as f32).sum();
        let sum_y: f32 = y_values.iter().sum();
        let sum_xy: f32 = x_values
            .iter()
            .zip(y_values.iter())
            .map(|(&x, &y)| x as f32 * y)
            .sum();
        let sum_x_squared: f32 = x_values.iter().map(|&x| (x as f32).powi(2)).sum();
        let denominator = n * sum_x_squared - sum_x.powi(2);
        if denominator.abs() < f32::EPSILON {
            return None;
        }
        Some((n * sum_xy - sum_x * sum_y) / denominator)
    }
    /// Detect anomalies in the data
    fn detect_anomalies(&self, trend_analysis: &HealthTrendAnalysis) -> f32 {
        if trend_analysis.utilization_history.len() < 10 {
            return 0.0;
        }
        let values: Vec<f32> = trend_analysis
            .utilization_history
            .iter()
            .map(|(_, v)| *v)
            .collect();
        let mean = values.iter().sum::<f32>() / values.len() as f32;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f32>()
            / values.len() as f32;
        let std_dev = variance.sqrt();
        if let Some(&latest_value) = values.last() {
            let deviation = (latest_value - mean).abs();
            if deviation > 2.0 * std_dev {
                return (deviation / (2.0 * std_dev)).min(1.0);
            }
        }
        0.0
    }
    /// Assess degradation risk
    fn assess_degradation_risk(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        _trend_analysis: &HealthTrendAnalysis,
    ) -> f32 {
        let mut risk = 0.0;
        if utilization_percent > 80.0 {
            risk += (utilization_percent - 80.0) / 20.0 * 0.4;
        }
        if avg_allocation_time_ms > 100.0 {
            risk += ((avg_allocation_time_ms - 100.0) / 400.0).min(1.0) * 0.6;
        }
        risk.min(1.0)
    }
    /// Calculate maintenance urgency
    fn calculate_maintenance_urgency(
        &self,
        risk_score: f32,
        degradation_risk: f32,
        anomaly_confidence: f32,
    ) -> f32 {
        (risk_score * 0.4 + degradation_risk * 0.4 + anomaly_confidence * 0.2).min(1.0)
    }
    /// Process and manage alerts
    async fn process_alerts(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        health_score: f32,
        recent_conflicts: usize,
        thresholds: &PortHealthThresholds,
    ) -> Vec<String> {
        let mut alert_summaries = Vec::new();
        let mut active_alerts = self.active_alerts.lock();
        self.check_utilization_alerts(
                utilization_percent,
                thresholds,
                &mut alert_summaries,
                &mut active_alerts,
            )
            .await;
        self.check_performance_alerts(
                avg_allocation_time_ms,
                thresholds,
                &mut alert_summaries,
                &mut active_alerts,
            )
            .await;
        self.check_health_score_alerts(
                health_score,
                thresholds,
                &mut alert_summaries,
                &mut active_alerts,
            )
            .await;
        self.check_conflict_alerts(
                recent_conflicts,
                thresholds,
                &mut alert_summaries,
                &mut active_alerts,
            )
            .await;
        self.cleanup_resolved_alerts(&mut active_alerts).await;
        alert_summaries
    }
    /// Check and manage utilization alerts
    async fn check_utilization_alerts(
        &self,
        utilization_percent: f32,
        thresholds: &PortHealthThresholds,
        alert_summaries: &mut Vec<String>,
        active_alerts: &mut HashMap<String, PortHealthAlert>,
    ) {
        let alert_id = "high_utilization".to_string();
        if utilization_percent >= thresholds.utilization_critical {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::HighUtilization,
                severity: AlertSeverity::Critical,
                title: "Critical Port Utilization".to_string(),
                description: format!(
                    "Port utilization is critically high at {:.1}%", utilization_percent
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "utilization_monitor".to_string(),
                current_value: utilization_percent as f64,
                threshold_value: thresholds.utilization_critical as f64,
                recommended_actions: vec![
                    "Scale up port pool capacity".to_string(),
                    "Review resource allocation patterns".to_string(),
                    "Implement load balancing".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries
                .push(format!("Critical port utilization: {:.1}%", utilization_percent));
        } else if utilization_percent >= thresholds.utilization_warning {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::HighUtilization,
                severity: AlertSeverity::Medium,
                title: "High Port Utilization".to_string(),
                description: format!(
                    "Port utilization is high at {:.1}%", utilization_percent
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "utilization_monitor".to_string(),
                current_value: utilization_percent as f64,
                threshold_value: thresholds.utilization_warning as f64,
                recommended_actions: vec![
                    "Monitor utilization trends".to_string(),
                    "Consider capacity planning".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries
                .push(format!("High port utilization: {:.1}%", utilization_percent));
        } else {
            active_alerts.remove(&alert_id);
        }
    }
    /// Check and manage performance alerts
    async fn check_performance_alerts(
        &self,
        avg_allocation_time_ms: f64,
        thresholds: &PortHealthThresholds,
        alert_summaries: &mut Vec<String>,
        active_alerts: &mut HashMap<String, PortHealthAlert>,
    ) {
        let alert_id = "slow_performance".to_string();
        if avg_allocation_time_ms >= thresholds.allocation_time_critical_ms {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::SlowPerformance,
                severity: AlertSeverity::Critical,
                title: "Critical Performance Degradation".to_string(),
                description: format!(
                    "Allocation time is critically slow at {:.1}ms",
                    avg_allocation_time_ms
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "performance_monitor".to_string(),
                current_value: avg_allocation_time_ms,
                threshold_value: thresholds.allocation_time_critical_ms,
                recommended_actions: vec![
                    "Investigate system bottlenecks".to_string(),
                    "Optimize allocation algorithms".to_string(),
                    "Check system resources".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries
                .push(
                    format!(
                        "Critical performance: {:.1}ms allocation time",
                        avg_allocation_time_ms
                    ),
                );
        } else if avg_allocation_time_ms >= thresholds.allocation_time_warning_ms {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::SlowPerformance,
                severity: AlertSeverity::Medium,
                title: "Performance Warning".to_string(),
                description: format!(
                    "Allocation time is slow at {:.1}ms", avg_allocation_time_ms
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "performance_monitor".to_string(),
                current_value: avg_allocation_time_ms,
                threshold_value: thresholds.allocation_time_warning_ms,
                recommended_actions: vec![
                    "Monitor performance trends".to_string(),
                    "Review allocation patterns".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries
                .push(
                    format!(
                        "Slow performance: {:.1}ms allocation time",
                        avg_allocation_time_ms
                    ),
                );
        } else {
            active_alerts.remove(&alert_id);
        }
    }
    /// Check and manage health score alerts
    async fn check_health_score_alerts(
        &self,
        health_score: f32,
        thresholds: &PortHealthThresholds,
        alert_summaries: &mut Vec<String>,
        active_alerts: &mut HashMap<String, PortHealthAlert>,
    ) {
        let alert_id = "low_health_score".to_string();
        if health_score <= thresholds.health_score_critical {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::LowHealthScore,
                severity: AlertSeverity::Critical,
                title: "Critical System Health".to_string(),
                description: format!(
                    "System health score is critically low at {:.1}", health_score
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "health_monitor".to_string(),
                current_value: health_score as f64,
                threshold_value: thresholds.health_score_critical as f64,
                recommended_actions: vec![
                    "Immediate system investigation required".to_string(),
                    "Check all system components".to_string(),
                    "Consider maintenance window".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries.push(format!("Critical health score: {:.1}", health_score));
        } else if health_score <= thresholds.health_score_warning {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::LowHealthScore,
                severity: AlertSeverity::Medium,
                title: "Low System Health".to_string(),
                description: format!(
                    "System health score is low at {:.1}", health_score
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "health_monitor".to_string(),
                current_value: health_score as f64,
                threshold_value: thresholds.health_score_warning as f64,
                recommended_actions: vec![
                    "Monitor system closely".to_string(), "Review performance metrics"
                    .to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries.push(format!("Low health score: {:.1}", health_score));
        } else {
            active_alerts.remove(&alert_id);
        }
    }
    /// Check and manage conflict alerts
    async fn check_conflict_alerts(
        &self,
        recent_conflicts: usize,
        thresholds: &PortHealthThresholds,
        alert_summaries: &mut Vec<String>,
        active_alerts: &mut HashMap<String, PortHealthAlert>,
    ) {
        let alert_id = "high_conflicts".to_string();
        let conflict_rate = recent_conflicts as f32;
        if conflict_rate >= thresholds.conflicts_per_minute_critical {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::HighConflictRate,
                severity: AlertSeverity::Critical,
                title: "Critical Conflict Rate".to_string(),
                description: format!(
                    "Port conflict rate is critically high: {} conflicts",
                    recent_conflicts
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "conflict_monitor".to_string(),
                current_value: conflict_rate as f64,
                threshold_value: thresholds.conflicts_per_minute_critical as f64,
                recommended_actions: vec![
                    "Investigate conflict sources".to_string(),
                    "Review allocation strategies".to_string(),
                    "Check for resource contention".to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries
                .push(format!("Critical conflicts: {} recent", recent_conflicts));
        } else if conflict_rate >= thresholds.conflicts_per_minute_warning {
            let alert = PortHealthAlert {
                id: alert_id.clone(),
                alert_type: AlertType::HighConflictRate,
                severity: AlertSeverity::Medium,
                title: "High Conflict Rate".to_string(),
                description: format!(
                    "Port conflict rate is high: {} conflicts", recent_conflicts
                ),
                created_at: Utc::now(),
                updated_at: Utc::now(),
                status: AlertStatus::Active,
                source: "conflict_monitor".to_string(),
                current_value: conflict_rate as f64,
                threshold_value: thresholds.conflicts_per_minute_warning as f64,
                recommended_actions: vec![
                    "Monitor conflict patterns".to_string(), "Review test coordination"
                    .to_string(),
                ],
                metadata: HashMap::new(),
            };
            active_alerts.insert(alert_id, alert);
            alert_summaries.push(format!("High conflicts: {} recent", recent_conflicts));
        } else {
            active_alerts.remove(&alert_id);
        }
    }
    /// Clean up resolved alerts
    async fn cleanup_resolved_alerts(
        &self,
        active_alerts: &mut HashMap<String, PortHealthAlert>,
    ) {
        let config = self.config.read();
        let now = Utc::now();
        active_alerts
            .retain(|_id, alert| {
                now
                    .signed_duration_since(alert.updated_at)
                    .to_std()
                    .unwrap_or(Duration::ZERO) < config.alert_throttle_duration
            });
    }
    /// Update trend analysis data
    async fn update_trend_analysis(
        &self,
        utilization_percent: f32,
        avg_allocation_time_ms: f64,
        conflict_rate: f32,
        health_score: f32,
    ) {
        let mut trend_analysis = self.trend_analysis.lock();
        let now = Utc::now();
        trend_analysis.utilization_history.push_back((now, utilization_percent));
        trend_analysis.performance_history.push_back((now, avg_allocation_time_ms));
        trend_analysis.conflict_history.push_back((now, conflict_rate));
        trend_analysis.health_score_history.push_back((now, health_score));
        while trend_analysis.utilization_history.len() > trend_analysis.window_size {
            trend_analysis.utilization_history.pop_front();
        }
        while trend_analysis.performance_history.len() > trend_analysis.window_size {
            trend_analysis.performance_history.pop_front();
        }
        while trend_analysis.conflict_history.len() > trend_analysis.window_size {
            trend_analysis.conflict_history.pop_front();
        }
        while trend_analysis.health_score_history.len() > trend_analysis.window_size {
            trend_analysis.health_score_history.pop_front();
        }
        trend_analysis.last_analysis = now;
    }
    /// Record a health event
    #[instrument(skip(self, details))]
    pub async fn record_health_event(
        &self,
        status: HealthStatus,
        event_type: HealthEventType,
        severity: EventSeverity,
        description: String,
        details: HashMap<String, String>,
    ) {
        let event = PortHealthEvent {
            timestamp: Utc::now(),
            status,
            event_type,
            metrics: HashMap::new(),
            alerts: Vec::new(),
            details,
            severity,
            duration: None,
            component: "port_health_monitor".to_string(),
        };
        let mut history = self.health_history.lock();
        history.push_back(event);
        let config = self.config.read();
        while history.len() > config.history_size {
            history.pop_front();
        }
        debug!("Health event recorded: {}", description);
    }
    /// Get health history
    pub async fn get_health_history(&self) -> Vec<PortHealthEvent> {
        let history = self.health_history.lock();
        history.iter().cloned().collect()
    }
    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<PortHealthAlert> {
        let alerts = self.active_alerts.lock();
        alerts.values().cloned().collect()
    }
    /// Generate comprehensive health report
    #[instrument(skip(self))]
    pub async fn generate_health_report(&self) -> String {
        let status = self.get_health_status().await;
        let alerts = self.get_active_alerts().await;
        let thresholds = self.alert_thresholds.read();
        let mut report = String::new();
        report.push_str("=== PORT HEALTH MONITORING REPORT ===\n\n");
        report.push_str(&format!("Overall Status: {:?}\n", status.overall_status));
        report.push_str(&format!("Health Score: {:.1}/100.0\n", status.health_score));
        report
            .push_str(
                &format!(
                    "Last Check: {}\n", status.last_check.format("%Y-%m-%d %H:%M:%S UTC")
                ),
            );
        report
            .push_str(
                &format!("System Uptime: {} hours\n", status.uptime_seconds / 3600),
            );
        report.push('\n');
        report.push_str("=== RESOURCE STATUS ===\n");
        report.push_str(&format!("Available Ports: {}\n", status.available_ports));
        report.push_str(&format!("Allocated Ports: {}\n", status.allocated_ports));
        report.push_str(&format!("Reserved Ports: {}\n", status.reserved_ports));
        report.push_str(&format!("Utilization: {:.1}%\n", status.utilization_percent));
        report.push('\n');
        report.push_str("=== PERFORMANCE METRICS ===\n");
        report
            .push_str(
                &format!(
                    "Average Allocation Time: {:.1}ms\n", status.avg_allocation_time_ms
                ),
            );
        report.push_str(&format!("Recent Conflicts: {}\n", status.recent_conflicts));
        report
            .push_str(
                &format!(
                    "Allocation Efficiency: {:.1}%\n", status.efficiency_metrics
                    .allocation_efficiency * 100.0
                ),
            );
        report
            .push_str(
                &format!(
                    "Overall Efficiency: {:.1}%\n", status.efficiency_metrics
                    .overall_efficiency * 100.0
                ),
            );
        report.push('\n');
        report.push_str("=== PREDICTIVE INDICATORS ===\n");
        report
            .push_str(
                &format!(
                    "Utilization Trend: {:?}\n", status.predictive_indicators
                    .utilization_trend
                ),
            );
        report
            .push_str(
                &format!(
                    "Risk Score: {:.1}%\n", status.predictive_indicators.risk_score *
                    100.0
                ),
            );
        report
            .push_str(
                &format!(
                    "Degradation Risk: {:.1}%\n", status.predictive_indicators
                    .degradation_risk * 100.0
                ),
            );
        report
            .push_str(
                &format!(
                    "Maintenance Urgency: {:.1}%\n", status.predictive_indicators
                    .maintenance_urgency * 100.0
                ),
            );
        if let Some(time_to_exhaustion) = status.predictive_indicators.time_to_exhaustion
        {
            report
                .push_str(
                    &format!(
                        "Time to Exhaustion: {} hours\n", time_to_exhaustion.as_secs() /
                        3600
                    ),
                );
        }
        report.push('\n');
        if !alerts.is_empty() {
            report.push_str("=== ACTIVE ALERTS ===\n");
            for alert in &alerts {
                report
                    .push_str(
                        &format!(
                            "- [{:?}] {}: {}\n", alert.severity, alert.title, alert
                            .description
                        ),
                    );
            }
            report.push('\n');
        }
        report.push_str("=== THRESHOLD CONFIGURATION ===\n");
        report
            .push_str(
                &format!("Utilization Warning: {:.1}%\n", thresholds.utilization_warning),
            );
        report
            .push_str(
                &format!(
                    "Utilization Critical: {:.1}%\n", thresholds.utilization_critical
                ),
            );
        report
            .push_str(
                &format!(
                    "Allocation Time Warning: {:.1}ms\n", thresholds
                    .allocation_time_warning_ms
                ),
            );
        report
            .push_str(
                &format!(
                    "Allocation Time Critical: {:.1}ms\n", thresholds
                    .allocation_time_critical_ms
                ),
            );
        report
            .push_str(
                &format!(
                    "Health Score Warning: {:.1}\n", thresholds.health_score_warning
                ),
            );
        report
            .push_str(
                &format!(
                    "Health Score Critical: {:.1}\n", thresholds.health_score_critical
                ),
            );
        report.push('\n');
        report.push_str("=== RECOMMENDATIONS ===\n");
        let recommendations = self.generate_recommendations(&status, &thresholds).await;
        for recommendation in recommendations {
            report.push_str(&format!("- {}\n", recommendation));
        }
        report.push_str("\n=== END OF REPORT ===\n");
        report
    }
    /// Generate recommendations based on current health status
    async fn generate_recommendations(
        &self,
        status: &PortHealthStatus,
        thresholds: &PortHealthThresholds,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        if status.utilization_percent > thresholds.utilization_warning {
            recommendations.push("Consider expanding port pool capacity".to_string());
            recommendations.push("Implement port allocation optimization".to_string());
        }
        if status.avg_allocation_time_ms > thresholds.allocation_time_warning_ms {
            recommendations
                .push("Investigate allocation performance bottlenecks".to_string());
            recommendations.push("Consider algorithm optimization".to_string());
        }
        if status.health_score < thresholds.health_score_warning {
            recommendations
                .push("Comprehensive system health review recommended".to_string());
            recommendations.push("Monitor system trends closely".to_string());
        }
        if status.predictive_indicators.risk_score > 0.7 {
            recommendations
                .push(
                    "High risk detected - proactive maintenance recommended".to_string(),
                );
        }
        if status.predictive_indicators.maintenance_urgency > 0.8 {
            recommendations
                .push("Urgent maintenance window should be scheduled".to_string());
        }
        if status.efficiency_metrics.overall_efficiency < 0.7 {
            recommendations
                .push(
                    "System efficiency is low - review resource allocation strategies"
                        .to_string(),
                );
        }
        if recommendations.is_empty() {
            recommendations
                .push("System is operating within normal parameters".to_string());
        }
        recommendations
    }
    /// Update health monitoring configuration
    #[instrument(skip(self, new_config))]
    pub async fn update_config(
        &self,
        new_config: PortHealthConfig,
    ) -> Result<(), PortManagementError> {
        let mut config = self.config.write();
        *config = new_config;
        info!("Health monitoring configuration updated");
        Ok(())
    }
    /// Update alert thresholds
    #[instrument(skip(self, new_thresholds))]
    pub async fn update_thresholds(
        &self,
        new_thresholds: PortHealthThresholds,
    ) -> Result<(), PortManagementError> {
        let mut thresholds = self.alert_thresholds.write();
        *thresholds = new_thresholds;
        info!("Health monitoring thresholds updated");
        Ok(())
    }
    /// Get health monitoring statistics
    pub async fn get_health_statistics(&self) -> HealthStatistics {
        let history = self.health_history.lock();
        let alerts = self.active_alerts.lock();
        let trend_analysis = self.trend_analysis.lock();
        let total_events = history.len();
        let critical_events = history
            .iter()
            .filter(|event| matches!(event.severity, EventSeverity::Critical))
            .count();
        let warning_events = history
            .iter()
            .filter(|event| matches!(event.severity, EventSeverity::Warning))
            .count();
        HealthStatistics {
            total_events,
            critical_events,
            warning_events,
            active_alerts: alerts.len(),
            trend_data_points: trend_analysis.utilization_history.len(),
            last_analysis: trend_analysis.last_analysis,
        }
    }
    /// Establish performance baseline
    #[instrument(skip(self))]
    pub async fn establish_baseline(
        &self,
        allocation_time_ms: f64,
        utilization: f32,
        conflict_rate: f32,
        efficiency: EfficiencyMetrics,
        sample_count: usize,
    ) -> Result<(), PortManagementError> {
        let mut baseline = self.performance_baseline.write();
        *baseline = PerformanceBaseline {
            baseline_allocation_time_ms: allocation_time_ms,
            baseline_utilization: utilization,
            baseline_conflict_rate: conflict_rate,
            baseline_efficiency: efficiency,
            established_at: Utc::now(),
            is_valid: true,
            sample_count,
        };
        info!("Performance baseline established with {} samples", sample_count);
        Ok(())
    }
}
/// Health status levels with detailed descriptions
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// System is operating optimally with no issues detected
    Healthy,
    /// System has minor issues but continues to function normally
    Warning,
    /// System has significant issues that may affect performance
    Degraded,
    /// System has critical issues requiring immediate attention
    Critical,
    /// Health status cannot be determined
    Unknown,
}
/// Health trend analysis data
#[derive(Debug, Clone)]
pub struct HealthTrendAnalysis {
    /// Utilization trend data points
    pub utilization_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Performance trend data points
    pub performance_history: VecDeque<(DateTime<Utc>, f64)>,
    /// Conflict rate trend data points
    pub conflict_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Health score trend data points
    pub health_score_history: VecDeque<(DateTime<Utc>, f32)>,
    /// Trend analysis window size
    pub window_size: usize,
    /// Last trend analysis timestamp
    pub last_analysis: DateTime<Utc>,
}
/// Configuration for health monitoring behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortHealthConfig {
    /// Enable health monitoring
    pub enabled: bool,
    /// Health check interval
    pub check_interval: Duration,
    /// Number of health events to keep in history
    pub history_size: usize,
    /// Enable automatic alert generation
    pub enable_alerts: bool,
    /// Enable trend analysis
    pub enable_trend_analysis: bool,
    /// Enable predictive health indicators
    pub enable_predictive_indicators: bool,
    /// Health check timeout
    pub check_timeout: Duration,
    /// Minimum time between alerts for the same condition
    pub alert_throttle_duration: Duration,
    /// Enable detailed health logging
    pub enable_detailed_logging: bool,
}
/// Health alert structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortHealthAlert {
    /// Unique alert identifier
    pub id: String,
    /// Alert type
    pub alert_type: AlertType,
    /// Alert severity
    pub severity: AlertSeverity,
    /// Alert title/summary
    pub title: String,
    /// Detailed alert description
    pub description: String,
    /// Timestamp when alert was created
    pub created_at: DateTime<Utc>,
    /// Timestamp when alert was last updated
    pub updated_at: DateTime<Utc>,
    /// Alert status
    pub status: AlertStatus,
    /// Source metric or component that triggered the alert
    pub source: String,
    /// Current metric value
    pub current_value: f64,
    /// Threshold value that was crossed
    pub threshold_value: f64,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Alert metadata
    pub metadata: HashMap<String, String>,
}
/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    /// Low severity
    Low,
    /// Medium severity
    Medium,
    /// High severity
    High,
    /// Critical severity
    Critical,
}
/// Event severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Informational event
    Info,
    /// Warning level event
    Warning,
    /// Error level event
    Error,
    /// Critical level event
    Critical,
}
/// Trend direction enumeration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    /// Metric is increasing
    Increasing,
    /// Metric is decreasing
    Decreasing,
    /// Metric is stable
    Stable,
    /// Trend is unknown
    Unknown,
}
/// Types of health events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthEventType {
    /// System status change
    StatusChange,
    /// Alert triggered
    AlertTriggered,
    /// Alert resolved
    AlertResolved,
    /// Threshold crossed
    ThresholdCrossed,
    /// Performance degradation detected
    PerformanceDegradation,
    /// Performance improvement detected
    PerformanceImprovement,
    /// Resource exhaustion warning
    ResourceExhaustion,
    /// System recovery
    SystemRecovery,
    /// Configuration change
    ConfigurationChange,
    /// Anomaly detected
    AnomalyDetected,
}
/// Types of health alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertType {
    /// High resource utilization
    HighUtilization,
    /// Slow performance
    SlowPerformance,
    /// High conflict rate
    HighConflictRate,
    /// Low health score
    LowHealthScore,
    /// High error rate
    HighErrorRate,
    /// Resource fragmentation
    ResourceFragmentation,
    /// System degradation
    SystemDegradation,
    /// Capacity warning
    CapacityWarning,
    /// Anomalous behavior
    AnomalousBehavior,
    /// Custom alert type
    Custom(String),
}
/// Health monitoring statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStatistics {
    /// Total number of health events recorded
    pub total_events: usize,
    /// Number of critical events
    pub critical_events: usize,
    /// Number of warning events
    pub warning_events: usize,
    /// Number of currently active alerts
    pub active_alerts: usize,
    /// Number of trend analysis data points
    pub trend_data_points: usize,
    /// Last trend analysis timestamp
    pub last_analysis: DateTime<Utc>,
}
