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

/// POST a JSON alert payload to `url`, real HTTP delivery.
///
/// [`AlertManager::send_notification`] is a plain synchronous method (see its
/// doc comment: making it `async` would ripple into every caller of the
/// public [`AlertManager::add_alert`] API across the crate), so this cannot
/// reuse the `async` `post_json` helper in `cicd_integration`. The blocking
/// `reqwest` client is run on a dedicated OS thread rather than the caller's
/// thread directly: `reqwest::blocking` internally starts its own Tokio
/// runtime and panics if constructed on a thread that is already driving one
/// (plausible here, since `AlertManager` may be invoked from async training
/// loops elsewhere in the workspace). A fresh `std::thread` has no runtime
/// affiliation, so this is safe regardless of the caller's context.
#[cfg(feature = "http-integrations")]
fn post_json_blocking(url: &str, payload: &serde_json::Value) -> Result<()> {
    let url = url.to_string();
    let payload = payload.clone();
    std::thread::spawn(move || -> Result<()> {
        let client = reqwest::blocking::Client::new();
        let response = client
            .post(&url)
            .json(&payload)
            // Bound the whole request (connect + send + receive): without this,
            // an unresponsive webhook endpoint would hang this spawned thread
            // (and so the `.join()` below, and so `send_notification`'s caller)
            // indefinitely instead of surfacing as a delivery error.
            .timeout(std::time::Duration::from_secs(30))
            .send()
            .map_err(|e| anyhow::anyhow!("webhook delivery failed: {e}"))?;
        let status = response.status();
        if !status.is_success() {
            let body = response.text().unwrap_or_default();
            anyhow::bail!("webhook delivery failed: HTTP {status}: {body}");
        }
        Ok(())
    })
    .join()
    .map_err(|_| anyhow::anyhow!("webhook delivery thread panicked"))?
}

/// Without `http-integrations`, no HTTP client exists in this build: fail
/// honestly instead of pretending to deliver (mirrors
/// `cicd_integration::post_json`'s disabled-feature branch).
#[cfg(not(feature = "http-integrations"))]
fn post_json_blocking(_url: &str, _payload: &serde_json::Value) -> Result<()> {
    anyhow::bail!(
        "HTTP notification delivery is not enabled: rebuild trustformers-debug with \
         `--features http-integrations`"
    )
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
    ///
    /// Each of the three channels below is opt-in via [`NotificationSettings`]
    /// (all default to disabled except `console_notifications`). A channel
    /// that is enabled but cannot actually deliver -- missing config, or a
    /// build without the `http-integrations` feature -- returns `Err` instead
    /// of silently doing nothing: the alert itself was already recorded in
    /// [`Self::add_alert`] before this is called, so a delivery failure here
    /// only means the *notification* did not go out, which the caller must
    /// be able to observe.
    fn send_notification(
        &self,
        alert: &ModelDiagnosticAlert,
        severity: &AlertSeverity,
    ) -> Result<()> {
        if self.config.notification_settings.console_notifications {
            // A caller-opted-in delivery channel in its own right (see the
            // `file_logging`/`webhook_notifications` channels below), not
            // incidental print debugging -- stdout is this channel's actual
            // destination, so `println!` is correct here rather than `tracing`.
            println!("[{:?}] Alert: {:?}", severity, alert);
        }

        if self.config.notification_settings.file_logging {
            let log_path =
                self.config.notification_settings.log_file_path.as_ref().ok_or_else(|| {
                    anyhow::anyhow!("file_logging is enabled but no log_file_path is configured")
                })?;

            use std::io::Write;
            let line = format!("{} [{:?}] {:?}\n", Utc::now().to_rfc3339(), severity, alert);
            let mut file =
                std::fs::OpenOptions::new().create(true).append(true).open(log_path).map_err(
                    |e| anyhow::anyhow!("failed to open alert log file {log_path}: {e}"),
                )?;
            file.write_all(line.as_bytes())
                .map_err(|e| anyhow::anyhow!("failed to write alert log file {log_path}: {e}"))?;
        }

        if self.config.notification_settings.webhook_notifications {
            let webhook_url =
                self.config.notification_settings.webhook_url.as_ref().ok_or_else(|| {
                    anyhow::anyhow!(
                        "webhook_notifications is enabled but no webhook_url is configured"
                    )
                })?;

            let payload = serde_json::json!({
                "severity": format!("{:?}", severity),
                "alert": format!("{:?}", alert),
                "timestamp": Utc::now().to_rfc3339(),
            });
            post_json_blocking(webhook_url, &payload)?;
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

        manager.add_alert(alert, AlertSeverity::Warning).expect("add operation failed");
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
        manager
            .add_alert(alert.clone(), AlertSeverity::Warning)
            .expect("add operation failed");
        assert_eq!(manager.active_alerts.len(), 1);

        // Try to add duplicate - should be filtered out
        manager.add_alert(alert, AlertSeverity::Warning).expect("add operation failed");
        assert_eq!(manager.active_alerts.len(), 1);
    }

    fn sample_alert() -> ModelDiagnosticAlert {
        ModelDiagnosticAlert::PerformanceDegradation {
            metric: "loss".to_string(),
            current: 1.5,
            previous_avg: 1.0,
            degradation_percent: 50.0,
        }
    }

    /// Regression: `file_logging: true` used to silently discard
    /// `log_file_path` (`let _ = log_path;`) and write nothing at all, while
    /// `add_alert` still returned `Ok(())` as if the log had been written.
    /// The configured path must now contain a real, readable line per alert.
    #[test]
    fn test_file_logging_writes_a_real_line_to_the_configured_path() {
        let log_path = std::env::temp_dir().join(format!(
            "trustformers_debug_alert_log_{}.txt",
            std::process::id()
        ));
        let _ = std::fs::remove_file(&log_path); // start clean; ignore "did not exist"

        let mut settings = NotificationSettings::default();
        settings.console_notifications = false;
        settings.file_logging = true;
        settings.log_file_path = Some(log_path.to_string_lossy().into_owned());

        let mut manager = AlertManager::with_config(
            AlertConfig {
                notification_settings: settings,
                ..AlertConfig::default()
            },
            AlertThresholds::default(),
        );

        manager
            .add_alert(sample_alert(), AlertSeverity::Critical)
            .expect("file logging must succeed when a valid path is configured");

        let written = std::fs::read_to_string(&log_path).expect("log file must have been created");
        assert!(
            written.contains("Critical") && written.contains("PerformanceDegradation"),
            "log file must contain a real record of the alert, got: {written:?}"
        );

        let _ = std::fs::remove_file(&log_path);
    }

    /// Regression: `file_logging: true` with `log_file_path: None` used to
    /// silently do nothing and return `Ok(())`. A caller that opted into
    /// file logging but forgot to configure a path deserves a configuration
    /// error, not silent success.
    #[test]
    fn test_file_logging_without_a_path_is_a_configuration_error() {
        let mut settings = NotificationSettings::default();
        settings.console_notifications = false;
        settings.file_logging = true;
        settings.log_file_path = None;

        let mut manager = AlertManager::with_config(
            AlertConfig {
                notification_settings: settings,
                ..AlertConfig::default()
            },
            AlertThresholds::default(),
        );

        let err = manager
            .add_alert(sample_alert(), AlertSeverity::Warning)
            .expect_err("file_logging without a configured path must not silently succeed");
        assert!(err.to_string().contains("log_file_path"));
    }

    /// Regression: `webhook_notifications: true` used to silently discard
    /// `webhook_url` (`let _ = webhook_url;`) and return `Ok(())` without
    /// attempting any delivery. Without the `http-integrations` feature this
    /// crate has no HTTP client at all, so the honest outcome is an error
    /// naming the feature, not a false "delivered" signal.
    #[cfg(not(feature = "http-integrations"))]
    #[test]
    fn test_webhook_without_the_http_feature_is_an_honest_error() {
        let mut settings = NotificationSettings::default();
        settings.console_notifications = false;
        settings.webhook_notifications = true;
        settings.webhook_url = Some("http://127.0.0.1:1/webhook".to_string());

        let mut manager = AlertManager::with_config(
            AlertConfig {
                notification_settings: settings,
                ..AlertConfig::default()
            },
            AlertThresholds::default(),
        );

        let err = manager
            .add_alert(sample_alert(), AlertSeverity::Emergency)
            .expect_err("webhook delivery must not silently pretend to have sent anything");
        assert!(err.to_string().contains("http-integrations"));
    }

    /// Regression: `webhook_notifications: true` with `webhook_url: None`
    /// used to silently do nothing and return `Ok(())`.
    #[test]
    fn test_webhook_without_a_url_is_a_configuration_error() {
        let mut settings = NotificationSettings::default();
        settings.console_notifications = false;
        settings.webhook_notifications = true;
        settings.webhook_url = None;

        let mut manager = AlertManager::with_config(
            AlertConfig {
                notification_settings: settings,
                ..AlertConfig::default()
            },
            AlertThresholds::default(),
        );

        let err = manager
            .add_alert(sample_alert(), AlertSeverity::Warning)
            .expect_err("webhook_notifications without a configured URL must not silently succeed");
        assert!(err.to_string().contains("webhook_url"));
    }
}
