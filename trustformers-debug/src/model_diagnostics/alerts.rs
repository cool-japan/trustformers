//! Alert system and diagnostic notifications.
//!
//! This module provides comprehensive alert management for model diagnostics,
//! including threshold-based monitoring, alert prioritization, notification
//! systems, and automated response recommendations.

use anyhow::Result;
use chrono::{DateTime, Duration, Utc};
use std::collections::VecDeque;

use super::types::{
    ConvergenceStatus, LayerActivationStats, ModelDiagnosticAlert, ModelPerformanceMetrics,
    TrainingDynamics, TrainingStability,
};

/// Alert manager for monitoring and managing diagnostic alerts.
#[derive(Debug)]
pub struct AlertManager {
    /// Active alerts
    active_alerts: Vec<ActiveAlert>,
    /// Alert history
    alert_history: VecDeque<HistoricalAlert>,
    /// Alert configuration
    config: AlertConfig,
    /// Alert thresholds
    thresholds: AlertThresholds,
    /// Performance baseline for comparison
    performance_baseline: Option<PerformanceBaseline>,
}

/// Configuration for the alert system.
#[derive(Debug, Clone)]
pub struct AlertConfig {
    /// Maximum number of alerts to keep in history
    pub max_history_size: usize,
    /// Minimum time between duplicate alerts
    pub duplicate_alert_cooldown: Duration,
    /// Alert severity levels to monitor
    pub monitored_severities: Vec<AlertSeverity>,
    /// Enable automatic alert resolution
    pub auto_resolve_alerts: bool,
    /// Alert notification settings
    pub notification_settings: NotificationSettings,
}

/// Alert thresholds for various metrics.
#[derive(Debug, Clone)]
pub struct AlertThresholds {
    /// Performance degradation threshold (percentage)
    pub performance_degradation_percent: f64,
    /// Memory usage threshold (MB)
    pub memory_usage_threshold_mb: f64,
    /// Memory leak detection threshold (MB per step)
    pub memory_leak_threshold_mb_per_step: f64,
    /// Training instability variance threshold
    pub training_instability_variance: f64,
    /// Dead neuron ratio threshold
    pub dead_neuron_ratio_threshold: f64,
    /// Saturated neuron ratio threshold
    pub saturated_neuron_ratio_threshold: f64,
    /// Convergence plateau duration threshold (steps)
    pub plateau_duration_threshold: usize,
    /// Learning rate adjustment threshold
    pub learning_rate_adjustment_threshold: f64,
}

/// Performance baseline for comparison.
#[derive(Debug, Clone)]
pub struct PerformanceBaseline {
    /// Baseline loss value
    pub baseline_loss: f64,
    /// Baseline throughput
    pub baseline_throughput: f64,
    /// Baseline memory usage
    pub baseline_memory_mb: f64,
    /// Baseline accuracy (if available)
    pub baseline_accuracy: Option<f64>,
    /// When baseline was established
    pub established_at: DateTime<Utc>,
}

/// Active alert with current status.
#[derive(Debug, Clone)]
pub struct ActiveAlert {
    /// Alert information
    pub alert: ModelDiagnosticAlert,
    /// Alert severity
    pub severity: AlertSeverity,
    /// When alert was first triggered
    pub triggered_at: DateTime<Utc>,
    /// Number of times alert has been triggered
    pub trigger_count: usize,
    /// Recommended actions
    pub recommended_actions: Vec<String>,
    /// Alert status
    pub status: AlertStatus,
}

/// Historical alert record.
#[derive(Debug, Clone)]
pub struct HistoricalAlert {
    /// Alert information
    pub alert: ModelDiagnosticAlert,
    /// Alert severity
    pub severity: AlertSeverity,
    /// When alert was triggered
    pub triggered_at: DateTime<Utc>,
    /// When alert was resolved
    pub resolved_at: Option<DateTime<Utc>>,
    /// How alert was resolved
    pub resolution_method: Option<String>,
    /// Duration alert was active
    pub duration: Option<Duration>,
}

/// Alert severity levels.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum AlertSeverity {
    /// Informational alerts
    Info,
    /// Warning alerts
    Warning,
    /// Critical alerts requiring immediate attention
    Critical,
    /// Emergency alerts indicating system failure
    Emergency,
}

/// Alert status tracking.
#[derive(Debug, Clone, PartialEq)]
pub enum AlertStatus {
    /// Alert is active and unresolved
    Active,
    /// Alert is acknowledged but not resolved
    Acknowledged,
    /// Alert is being investigated
    InvestigationInProgress,
    /// Alert has been resolved
    Resolved,
    /// Alert was a false positive
    FalsePositive,
}

/// Notification settings for alerts.
#[derive(Debug, Clone)]
pub struct NotificationSettings {
    /// Enable console notifications
    pub console_notifications: bool,
    /// Enable file logging
    pub file_logging: bool,
    /// Log file path for alerts
    pub log_file_path: Option<String>,
    /// Enable webhook notifications
    pub webhook_notifications: bool,
    /// Webhook URL for notifications
    pub webhook_url: Option<String>,
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            max_history_size: 1000,
            duplicate_alert_cooldown: Duration::minutes(5),
            monitored_severities: vec![
                AlertSeverity::Warning,
                AlertSeverity::Critical,
                AlertSeverity::Emergency,
            ],
            auto_resolve_alerts: true,
            notification_settings: NotificationSettings::default(),
        }
    }
}

impl Default for NotificationSettings {
    fn default() -> Self {
        Self {
            console_notifications: true,
            file_logging: false,
            log_file_path: None,
            webhook_notifications: false,
            webhook_url: None,
        }
    }
}

impl Default for AlertThresholds {
    fn default() -> Self {
        Self {
            performance_degradation_percent: 10.0,
            memory_usage_threshold_mb: 8192.0, // 8GB
            memory_leak_threshold_mb_per_step: 1.0,
            training_instability_variance: 0.1,
            dead_neuron_ratio_threshold: 0.1,
            saturated_neuron_ratio_threshold: 0.05,
            plateau_duration_threshold: 100,
            learning_rate_adjustment_threshold: 0.01,
        }
    }
}

impl AlertManager {
    /// Create a new alert manager.
    pub fn new() -> Self {
        Self {
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
            config: AlertConfig::default(),
            thresholds: AlertThresholds::default(),
            performance_baseline: None,
        }
    }

    /// Create alert manager with custom configuration.
    pub fn with_config(config: AlertConfig, thresholds: AlertThresholds) -> Self {
        Self {
            active_alerts: Vec::new(),
            alert_history: VecDeque::new(),
            config,
            thresholds,
            performance_baseline: None,
        }
    }

    /// Set performance baseline for comparison.
    pub fn set_performance_baseline(&mut self, baseline: PerformanceBaseline) {
        self.performance_baseline = Some(baseline);
    }

    /// Establish baseline from current metrics.
    pub fn establish_baseline_from_metrics(&mut self, metrics: &ModelPerformanceMetrics) {
        self.performance_baseline = Some(PerformanceBaseline {
            baseline_loss: metrics.loss,
            baseline_throughput: metrics.throughput_samples_per_sec,
            baseline_memory_mb: metrics.memory_usage_mb,
            baseline_accuracy: metrics.accuracy,
            established_at: Utc::now(),
        });
    }

    /// Process performance metrics and generate alerts.
    pub fn process_performance_metrics(
        &mut self,
        metrics: &ModelPerformanceMetrics,
    ) -> Result<Vec<ModelDiagnosticAlert>> {
        let mut new_alerts = Vec::new();

        // Check for performance degradation
        if let Some(baseline) = &self.performance_baseline {
            let loss_degradation =
                ((metrics.loss - baseline.baseline_loss) / baseline.baseline_loss) * 100.0;
            if loss_degradation > self.thresholds.performance_degradation_percent {
                let alert = ModelDiagnosticAlert::PerformanceDegradation {
                    metric: "loss".to_string(),
                    current: metrics.loss,
                    previous_avg: baseline.baseline_loss,
                    degradation_percent: loss_degradation,
                };
                new_alerts.push(alert);
            }

            let throughput_degradation = ((baseline.baseline_throughput
                - metrics.throughput_samples_per_sec)
                / baseline.baseline_throughput)
                * 100.0;
            if throughput_degradation > self.thresholds.performance_degradation_percent {
                let alert = ModelDiagnosticAlert::PerformanceDegradation {
                    metric: "throughput".to_string(),
                    current: metrics.throughput_samples_per_sec,
                    previous_avg: baseline.baseline_throughput,
                    degradation_percent: throughput_degradation,
                };
                new_alerts.push(alert);
            }
        }

        // Check for memory issues
        if metrics.memory_usage_mb > self.thresholds.memory_usage_threshold_mb {
            let alert = ModelDiagnosticAlert::MemoryLeak {
                current_usage_mb: metrics.memory_usage_mb,
                growth_rate_mb_per_step: 0.0, // Would need historical data to calculate
            };
            new_alerts.push(alert);
        }

        // Process new alerts
        for alert in &new_alerts {
            self.add_alert(alert.clone(), self.determine_alert_severity(alert))?;
        }

        Ok(new_alerts)
    }

    /// Process training dynamics and generate alerts.
    pub fn process_training_dynamics(
        &mut self,
        dynamics: &TrainingDynamics,
    ) -> Result<Vec<ModelDiagnosticAlert>> {
        let mut new_alerts = Vec::new();

        // Check for training instability
        if matches!(
            dynamics.training_stability,
            TrainingStability::Unstable | TrainingStability::HighVariance
        ) {
            let alert = ModelDiagnosticAlert::TrainingInstability {
                variance: 0.0, // Would need to extract from dynamics
                threshold: self.thresholds.training_instability_variance,
            };
            new_alerts.push(alert);
        }

        // Check for convergence issues
        match dynamics.convergence_status {
            ConvergenceStatus::Diverging => {
                let alert = ModelDiagnosticAlert::ConvergenceIssue {
                    issue_type: ConvergenceStatus::Diverging,
                    duration_steps: 0, // Would need historical tracking
                };
                new_alerts.push(alert);
            },
            ConvergenceStatus::Plateau => {
                if let Some(plateau_info) = &dynamics.plateau_detection {
                    if plateau_info.duration_steps > self.thresholds.plateau_duration_threshold {
                        let alert = ModelDiagnosticAlert::ConvergenceIssue {
                            issue_type: ConvergenceStatus::Plateau,
                            duration_steps: plateau_info.duration_steps,
                        };
                        new_alerts.push(alert);
                    }
                }
            },
            _ => {},
        }

        // Process new alerts
        for alert in &new_alerts {
            self.add_alert(alert.clone(), self.determine_alert_severity(alert))?;
        }

        Ok(new_alerts)
    }

    /// Process layer statistics and generate alerts.
    pub fn process_layer_stats(
        &mut self,
        stats: &LayerActivationStats,
    ) -> Result<Vec<ModelDiagnosticAlert>> {
        let mut new_alerts = Vec::new();

        // Check for dead neurons
        if stats.dead_neurons_ratio > self.thresholds.dead_neuron_ratio_threshold {
            let alert = ModelDiagnosticAlert::ArchitecturalConcern {
                concern: format!(
                    "High dead neuron ratio in layer {}: {:.2}%",
                    stats.layer_name,
                    stats.dead_neurons_ratio * 100.0
                ),
                recommendation: "Consider adjusting learning rate or initialization".to_string(),
            };
            new_alerts.push(alert);
        }

        // Check for saturated neurons
        if stats.saturated_neurons_ratio > self.thresholds.saturated_neuron_ratio_threshold {
            let alert = ModelDiagnosticAlert::ArchitecturalConcern {
                concern: format!(
                    "High saturated neuron ratio in layer {}: {:.2}%",
                    stats.layer_name,
                    stats.saturated_neurons_ratio * 100.0
                ),
                recommendation: "Consider adjusting activation function or scaling".to_string(),
            };
            new_alerts.push(alert);
        }

        // Process new alerts
        for alert in &new_alerts {
            self.add_alert(alert.clone(), self.determine_alert_severity(alert))?;
        }

        Ok(new_alerts)
    }

    /// Add a new alert to the system.
    pub fn add_alert(
        &mut self,
        alert: ModelDiagnosticAlert,
        severity: AlertSeverity,
    ) -> Result<()> {
        // Check for duplicate alerts within cooldown period
        if self.is_duplicate_alert(&alert) {
            return Ok(());
        }

        let active_alert = ActiveAlert {
            alert: alert.clone(),
            severity: severity.clone(),
            triggered_at: Utc::now(),
            trigger_count: 1,
            recommended_actions: self.generate_recommended_actions(&alert),
            status: AlertStatus::Active,
        };

        self.active_alerts.push(active_alert);

        // Send notification
        self.send_notification(&alert, &severity)?;

        Ok(())
    }

    /// Resolve an alert.
    pub fn resolve_alert(&mut self, alert_index: usize, resolution_method: String) -> Result<()> {
        if alert_index >= self.active_alerts.len() {
            return Err(anyhow::anyhow!("Invalid alert index"));
        }

        let mut active_alert = self.active_alerts.remove(alert_index);
        active_alert.status = AlertStatus::Resolved;

        let historical_alert = HistoricalAlert {
            alert: active_alert.alert,
            severity: active_alert.severity,
            triggered_at: active_alert.triggered_at,
            resolved_at: Some(Utc::now()),
            resolution_method: Some(resolution_method),
            duration: Some(Utc::now() - active_alert.triggered_at),
        };

        self.add_to_history(historical_alert);
        Ok(())
    }

    /// Get all active alerts.
    pub fn get_active_alerts(&self) -> &[ActiveAlert] {
        &self.active_alerts
    }

    /// Get alerts by severity.
    pub fn get_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<&ActiveAlert> {
        self.active_alerts.iter().filter(|alert| alert.severity == severity).collect()
    }

    /// Get alert statistics.
    pub fn get_alert_statistics(&self) -> AlertStatistics {
        let mut stats = AlertStatistics::default();

        for alert in &self.active_alerts {
            match alert.severity {
                AlertSeverity::Info => stats.info_count += 1,
                AlertSeverity::Warning => stats.warning_count += 1,
                AlertSeverity::Critical => stats.critical_count += 1,
                AlertSeverity::Emergency => stats.emergency_count += 1,
            }
        }

        stats.total_active = self.active_alerts.len();
        stats.total_historical = self.alert_history.len();

        stats
    }

    /// Clear resolved alerts from active list.
    pub fn clear_resolved_alerts(&mut self) {
        let now = Utc::now();
        let mut resolved_alerts = Vec::new();

        self.active_alerts.retain(|alert| {
            if matches!(alert.status, AlertStatus::Resolved) {
                resolved_alerts.push(HistoricalAlert {
                    alert: alert.alert.clone(),
                    severity: alert.severity.clone(),
                    triggered_at: alert.triggered_at,
                    resolved_at: Some(now),
                    resolution_method: Some("Auto-resolved".to_string()),
                    duration: Some(now - alert.triggered_at),
                });
                false
            } else {
                true
            }
        });

        for historical in resolved_alerts {
            self.add_to_history(historical);
        }
    }

    /// Determine alert severity based on alert type.
    fn determine_alert_severity(&self, alert: &ModelDiagnosticAlert) -> AlertSeverity {
        match alert {
            ModelDiagnosticAlert::PerformanceDegradation {
                degradation_percent,
                ..
            } => {
                if *degradation_percent > 50.0 {
                    AlertSeverity::Critical
                } else if *degradation_percent > 25.0 {
                    AlertSeverity::Warning
                } else {
                    AlertSeverity::Info
                }
            },
            ModelDiagnosticAlert::MemoryLeak {
                current_usage_mb, ..
            } => {
                if *current_usage_mb > 16384.0 {
                    // 16GB
                    AlertSeverity::Emergency
                } else if *current_usage_mb > 8192.0 {
                    // 8GB
                    AlertSeverity::Critical
                } else {
                    AlertSeverity::Warning
                }
            },
            ModelDiagnosticAlert::TrainingInstability { .. } => AlertSeverity::Warning,
            ModelDiagnosticAlert::ConvergenceIssue { issue_type, .. } => match issue_type {
                ConvergenceStatus::Diverging => AlertSeverity::Critical,
                ConvergenceStatus::Plateau => AlertSeverity::Warning,
                _ => AlertSeverity::Info,
            },
            ModelDiagnosticAlert::ArchitecturalConcern { .. } => AlertSeverity::Info,
        }
    }

    /// Check if alert is a duplicate within cooldown period.
    fn is_duplicate_alert(&self, alert: &ModelDiagnosticAlert) -> bool {
        let now = Utc::now();
        let cooldown_threshold = now - self.config.duplicate_alert_cooldown;

        self.active_alerts.iter().any(|active| {
            active.triggered_at > cooldown_threshold
                && std::mem::discriminant(&active.alert) == std::mem::discriminant(alert)
        })
    }

    /// Generate recommended actions for an alert.
    fn generate_recommended_actions(&self, alert: &ModelDiagnosticAlert) -> Vec<String> {
        match alert {
            ModelDiagnosticAlert::PerformanceDegradation { metric, .. } => {
                vec![
                    format!("Investigate {} degradation causes", metric),
                    "Check for data quality issues".to_string(),
                    "Review recent configuration changes".to_string(),
                    "Consider adjusting learning rate".to_string(),
                ]
            },
            ModelDiagnosticAlert::MemoryLeak { .. } => {
                vec![
                    "Monitor memory usage patterns".to_string(),
                    "Check for gradient accumulation issues".to_string(),
                    "Review batch size configuration".to_string(),
                    "Consider implementing memory cleanup".to_string(),
                ]
            },
            ModelDiagnosticAlert::TrainingInstability { .. } => {
                vec![
                    "Reduce learning rate".to_string(),
                    "Enable gradient clipping".to_string(),
                    "Check data preprocessing".to_string(),
                    "Consider using learning rate scheduling".to_string(),
                ]
            },
            ModelDiagnosticAlert::ConvergenceIssue { issue_type, .. } => match issue_type {
                ConvergenceStatus::Diverging => vec![
                    "Immediately reduce learning rate".to_string(),
                    "Check gradient magnitudes".to_string(),
                    "Review loss function implementation".to_string(),
                ],
                ConvergenceStatus::Plateau => vec![
                    "Consider learning rate annealing".to_string(),
                    "Try different optimization algorithm".to_string(),
                    "Evaluate model capacity".to_string(),
                ],
                _ => vec!["Monitor training progress".to_string()],
            },
            ModelDiagnosticAlert::ArchitecturalConcern { recommendation, .. } => {
                vec![recommendation.clone()]
            },
        }
    }

    /// Send notification for an alert.
    fn send_notification(
        &self,
        alert: &ModelDiagnosticAlert,
        severity: &AlertSeverity,
    ) -> Result<()> {
        if self.config.notification_settings.console_notifications {
            println!("[{:?}] Alert: {:?}", severity, alert);
        }

        if self.config.notification_settings.file_logging {
            if let Some(log_path) = &self.config.notification_settings.log_file_path {
                // Would implement file logging here
                let _ = log_path; // Suppress unused warning
            }
        }

        if self.config.notification_settings.webhook_notifications {
            if let Some(webhook_url) = &self.config.notification_settings.webhook_url {
                // Would implement webhook notification here
                let _ = webhook_url; // Suppress unused warning
            }
        }

        Ok(())
    }

    /// Add alert to history with size management.
    fn add_to_history(&mut self, historical_alert: HistoricalAlert) {
        self.alert_history.push_back(historical_alert);

        while self.alert_history.len() > self.config.max_history_size {
            self.alert_history.pop_front();
        }
    }
}

/// Alert system statistics.
#[derive(Debug, Default)]
pub struct AlertStatistics {
    /// Number of active info alerts
    pub info_count: usize,
    /// Number of active warning alerts
    pub warning_count: usize,
    /// Number of active critical alerts
    pub critical_count: usize,
    /// Number of active emergency alerts
    pub emergency_count: usize,
    /// Total active alerts
    pub total_active: usize,
    /// Total historical alerts
    pub total_historical: usize,
}

impl Default for AlertManager {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alert_manager_creation() {
        let manager = AlertManager::new();
        assert_eq!(manager.active_alerts.len(), 0);
        assert_eq!(manager.alert_history.len(), 0);
    }

    #[test]
    fn test_add_alert() {
        let mut manager = AlertManager::new();
        let alert = ModelDiagnosticAlert::PerformanceDegradation {
            metric: "loss".to_string(),
            current: 1.5,
            previous_avg: 1.0,
            degradation_percent: 50.0,
        };

        manager.add_alert(alert, AlertSeverity::Warning).unwrap();
        assert_eq!(manager.active_alerts.len(), 1);
    }

    #[test]
    fn test_alert_severity_determination() {
        let manager = AlertManager::new();

        let high_degradation = ModelDiagnosticAlert::PerformanceDegradation {
            metric: "loss".to_string(),
            current: 2.0,
            previous_avg: 1.0,
            degradation_percent: 60.0,
        };

        let severity = manager.determine_alert_severity(&high_degradation);
        assert_eq!(severity, AlertSeverity::Critical);
    }

    #[test]
    fn test_duplicate_alert_detection() {
        let mut manager = AlertManager::new();
        let alert = ModelDiagnosticAlert::TrainingInstability {
            variance: 0.2,
            threshold: 0.1,
        };

        // Add first alert
        manager.add_alert(alert.clone(), AlertSeverity::Warning).unwrap();
        assert_eq!(manager.active_alerts.len(), 1);

        // Try to add duplicate - should be filtered out
        manager.add_alert(alert, AlertSeverity::Warning).unwrap();
        assert_eq!(manager.active_alerts.len(), 1);
    }
}
