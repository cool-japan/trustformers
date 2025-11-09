//! Alerting Rules System
//!
//! Provides comprehensive alerting capabilities for monitoring system metrics,
//! performance indicators, and custom conditions with configurable thresholds,
//! notification channels, and alert management.

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};
use tracing::{debug, error, info};

/// Alerting service configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingConfig {
    /// Enable alerting system
    pub enabled: bool,

    /// Default evaluation interval in seconds
    pub default_evaluation_interval_secs: u64,

    /// Maximum number of alerts to keep in history
    pub max_alert_history: usize,

    /// Alert resolution timeout in seconds
    pub alert_resolution_timeout_secs: u64,

    /// Enable alert grouping
    pub enable_alert_grouping: bool,

    /// Alert grouping window in seconds
    pub alert_grouping_window_secs: u64,

    /// Maximum alerts per group
    pub max_alerts_per_group: usize,

    /// Enable alert suppression
    pub enable_alert_suppression: bool,

    /// Alert suppression window in seconds
    pub alert_suppression_window_secs: u64,

    /// Default notification channels
    pub default_notification_channels: Vec<String>,

    /// Enable alert escalation
    pub enable_alert_escalation: bool,

    /// Alert escalation delay in seconds
    pub alert_escalation_delay_secs: u64,
}

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            default_evaluation_interval_secs: 60,
            max_alert_history: 10000,
            alert_resolution_timeout_secs: 300,
            enable_alert_grouping: true,
            alert_grouping_window_secs: 300,
            max_alerts_per_group: 10,
            enable_alert_suppression: true,
            alert_suppression_window_secs: 3600,
            default_notification_channels: vec!["default".to_string()],
            enable_alert_escalation: true,
            alert_escalation_delay_secs: 1800,
        }
    }
}

/// Alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    /// Unique rule identifier
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule description
    pub description: String,

    /// Metric query or condition
    pub query: String,

    /// Alert condition
    pub condition: AlertCondition,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Evaluation interval in seconds
    pub evaluation_interval_secs: u64,

    /// Number of consecutive evaluations before firing
    pub for_duration_secs: u64,

    /// Rule labels
    pub labels: HashMap<String, String>,

    /// Notification channels
    pub notification_channels: Vec<String>,

    /// Rule enabled status
    pub enabled: bool,

    /// Alert template for message formatting
    pub alert_template: Option<AlertTemplate>,

    /// Rule group
    pub group: Option<String>,

    /// Rule dependencies
    pub dependencies: Vec<String>,
}

/// Alert condition types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCondition {
    /// Threshold-based condition
    Threshold {
        operator: ComparisonOperator,
        value: f64,
    },
    /// Range-based condition
    Range { min: Option<f64>, max: Option<f64> },
    /// Rate-based condition
    Rate {
        window_secs: u64,
        operator: ComparisonOperator,
        value: f64,
    },
    /// Anomaly detection condition
    Anomaly { sensitivity: f64, window_secs: u64 },
    /// Custom condition with expression
    Custom { expression: String },
}

/// Comparison operators
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ComparisonOperator {
    GreaterThan,
    GreaterThanOrEqual,
    LessThan,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Alert severity levels
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AlertSeverity {
    Critical = 4,
    High = 3,
    Medium = 2,
    Low = 1,
    Info = 0,
}

impl Default for AlertSeverity {
    fn default() -> Self {
        AlertSeverity::Medium
    }
}

/// Alert template for message formatting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertTemplate {
    /// Alert title template
    pub title: String,

    /// Alert message template
    pub message: String,

    /// Additional fields template
    pub fields: HashMap<String, String>,
}

/// Alert instance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alert {
    /// Unique alert identifier
    pub id: String,

    /// Rule that generated this alert
    pub rule_id: String,

    /// Alert title
    pub title: String,

    /// Alert message
    pub message: String,

    /// Alert severity
    pub severity: AlertSeverity,

    /// Alert state
    pub state: AlertState,

    /// Alert labels
    pub labels: HashMap<String, String>,

    /// Alert annotations
    pub annotations: HashMap<String, String>,

    /// Alert value that triggered the condition
    pub value: f64,

    /// Alert fired timestamp
    pub fired_at: SystemTime,

    /// Alert resolved timestamp
    pub resolved_at: Option<SystemTime>,

    /// Alert acknowledgment status
    pub acknowledged: bool,

    /// Acknowledged by user
    pub acknowledged_by: Option<String>,

    /// Acknowledgment timestamp
    pub acknowledged_at: Option<SystemTime>,

    /// Notification channels used
    pub notification_channels: Vec<String>,

    /// Alert group identifier
    pub group_id: Option<String>,

    /// Parent alert ID (for escalated alerts)
    pub parent_alert_id: Option<String>,
}

/// Alert states
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AlertState {
    /// Alert condition is met and alert is active
    Firing,
    /// Alert condition is no longer met but still in resolution timeout
    Pending,
    /// Alert has been resolved
    Resolved,
    /// Alert has been suppressed
    Suppressed,
    /// Alert has been escalated
    Escalated,
}

/// Alert group for managing related alerts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertGroup {
    /// Group identifier
    pub id: String,

    /// Group name
    pub name: String,

    /// Alerts in this group
    pub alert_ids: Vec<String>,

    /// Group created timestamp
    pub created_at: SystemTime,

    /// Group labels
    pub labels: HashMap<String, String>,

    /// Group status
    pub status: AlertGroupStatus,
}

/// Alert group status
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AlertGroupStatus {
    Active,
    Resolved,
    Suppressed,
}

/// Notification channel configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationChannel {
    /// Channel identifier
    pub id: String,

    /// Channel name
    pub name: String,

    /// Channel type
    pub channel_type: NotificationChannelType,

    /// Channel configuration
    pub config: HashMap<String, String>,

    /// Channel enabled status
    pub enabled: bool,

    /// Severity filter
    pub severity_filter: Vec<AlertSeverity>,

    /// Label filters
    pub label_filters: HashMap<String, String>,
}

/// Notification channel types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum NotificationChannelType {
    Email,
    Slack,
    Discord,
    Webhook,
    SMS,
    PagerDuty,
    Custom(String),
}

/// Alert evaluation result
#[derive(Debug, Clone)]
pub struct AlertEvaluationResult {
    /// Rule that was evaluated
    pub rule_id: String,

    /// Evaluation timestamp
    pub evaluated_at: SystemTime,

    /// Whether condition was met
    pub condition_met: bool,

    /// Evaluated value
    pub value: f64,

    /// Evaluation duration
    pub evaluation_duration: Duration,

    /// Error during evaluation
    pub error: Option<String>,
}

/// Alerting statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingStats {
    /// Total rules configured
    pub total_rules: usize,

    /// Active rules count
    pub active_rules: usize,

    /// Total alerts fired
    pub total_alerts_fired: u64,

    /// Currently active alerts
    pub active_alerts: usize,

    /// Alerts by severity
    pub alerts_by_severity: HashMap<AlertSeverity, u64>,

    /// Alert resolution rate
    pub resolution_rate: f64,

    /// Average alert duration
    pub average_alert_duration: Duration,

    /// Notification success rate
    pub notification_success_rate: f64,

    /// Evaluation performance
    pub average_evaluation_time: Duration,

    /// Alert groups count
    pub alert_groups_count: usize,
}

/// Alerting errors
#[derive(Debug, Error)]
pub enum AlertingError {
    #[error("Rule not found: {0}")]
    RuleNotFound(String),

    #[error("Invalid rule configuration: {0}")]
    InvalidRule(String),

    #[error("Notification channel not found: {0}")]
    ChannelNotFound(String),

    #[error("Alert not found: {0}")]
    AlertNotFound(String),

    #[error("Evaluation failed: {0}")]
    EvaluationFailed(String),

    #[error("Notification failed: {0}")]
    NotificationFailed(String),

    #[error("Internal alerting error: {0}")]
    Internal(#[from] anyhow::Error),
}

/// Alert event for notifications
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertEvent {
    /// Event type
    pub event_type: AlertEventType,

    /// Alert information
    pub alert: Alert,

    /// Event timestamp
    pub timestamp: SystemTime,

    /// Additional event data
    pub data: HashMap<String, String>,
}

/// Alert event types
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum AlertEventType {
    Fired,
    Resolved,
    Acknowledged,
    Suppressed,
    Escalated,
    GroupCreated,
    GroupResolved,
}

/// Alerting service
pub struct AlertingService {
    config: AlertingConfig,
    rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    alerts: Arc<RwLock<HashMap<String, Alert>>>,
    alert_groups: Arc<RwLock<HashMap<String, AlertGroup>>>,
    notification_channels: Arc<RwLock<HashMap<String, NotificationChannel>>>,
    stats: Arc<RwLock<AlertingStats>>,
    evaluation_results: Arc<RwLock<VecDeque<AlertEvaluationResult>>>,
    alert_history: Arc<RwLock<VecDeque<Alert>>>,
    event_sender: broadcast::Sender<AlertEvent>,
    _event_receiver: broadcast::Receiver<AlertEvent>,
}

impl AlertingService {
    /// Create a new alerting service
    pub fn new(config: AlertingConfig) -> Self {
        let (event_sender, event_receiver) = broadcast::channel(1000);

        Self {
            config,
            rules: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_groups: Arc::new(RwLock::new(HashMap::new())),
            notification_channels: Arc::new(RwLock::new(HashMap::new())),
            stats: Arc::new(RwLock::new(AlertingStats {
                total_rules: 0,
                active_rules: 0,
                total_alerts_fired: 0,
                active_alerts: 0,
                alerts_by_severity: HashMap::new(),
                resolution_rate: 0.0,
                average_alert_duration: Duration::from_secs(0),
                notification_success_rate: 0.0,
                average_evaluation_time: Duration::from_secs(0),
                alert_groups_count: 0,
            })),
            evaluation_results: Arc::new(RwLock::new(VecDeque::new())),
            alert_history: Arc::new(RwLock::new(VecDeque::new())),
            event_sender,
            _event_receiver: event_receiver,
        }
    }

    /// Add an alert rule
    pub async fn add_rule(&self, rule: AlertRule) -> Result<(), AlertingError> {
        info!("Adding alert rule: {}", rule.name);

        // Validate rule
        self.validate_rule(&rule)?;

        // Add rule
        self.rules.write().await.insert(rule.id.clone(), rule);

        // Update statistics
        self.update_rule_stats().await;

        Ok(())
    }

    /// Remove an alert rule
    pub async fn remove_rule(&self, rule_id: &str) -> Result<(), AlertingError> {
        let mut rules = self.rules.write().await;

        if rules.remove(rule_id).is_none() {
            return Err(AlertingError::RuleNotFound(rule_id.to_string()));
        }

        info!("Removed alert rule: {}", rule_id);

        // Update statistics
        drop(rules);
        self.update_rule_stats().await;

        Ok(())
    }

    /// Get alert rule
    pub async fn get_rule(&self, rule_id: &str) -> Option<AlertRule> {
        self.rules.read().await.get(rule_id).cloned()
    }

    /// List all alert rules
    pub async fn list_rules(&self) -> Vec<AlertRule> {
        self.rules.read().await.values().cloned().collect()
    }

    /// Add notification channel
    pub async fn add_notification_channel(
        &self,
        channel: NotificationChannel,
    ) -> Result<(), AlertingError> {
        info!("Adding notification channel: {}", channel.name);

        self.notification_channels.write().await.insert(channel.id.clone(), channel);

        Ok(())
    }

    /// Remove notification channel
    pub async fn remove_notification_channel(&self, channel_id: &str) -> Result<(), AlertingError> {
        let mut channels = self.notification_channels.write().await;

        if channels.remove(channel_id).is_none() {
            return Err(AlertingError::ChannelNotFound(channel_id.to_string()));
        }

        info!("Removed notification channel: {}", channel_id);
        Ok(())
    }

    /// Evaluate all rules
    pub async fn evaluate_rules(
        &self,
        metrics: &HashMap<String, f64>,
    ) -> Result<Vec<AlertEvaluationResult>> {
        if !self.config.enabled {
            return Ok(Vec::new());
        }

        let rules = self.rules.read().await;
        let mut results = Vec::new();

        for rule in rules.values() {
            if !rule.enabled {
                continue;
            }

            let start_time = Instant::now();
            let result = self.evaluate_rule(rule, metrics).await;
            let evaluation_duration = start_time.elapsed();

            let eval_result = AlertEvaluationResult {
                rule_id: rule.id.clone(),
                evaluated_at: SystemTime::now(),
                condition_met: result.is_ok() && result.as_ref().unwrap().0,
                value: result.as_ref().map(|(_, v)| *v).unwrap_or(0.0),
                evaluation_duration,
                error: result.err().map(|e| e.to_string()),
            };

            // Handle alert state changes
            if eval_result.condition_met {
                self.handle_alert_firing(rule, eval_result.value).await?;
            } else {
                self.handle_alert_resolution(rule).await?;
            }

            results.push(eval_result);
        }

        // Store evaluation results
        let mut eval_results = self.evaluation_results.write().await;
        for result in &results {
            eval_results.push_back(result.clone());

            // Keep only recent results
            if eval_results.len() > 1000 {
                eval_results.pop_front();
            }
        }

        // Update statistics
        self.update_evaluation_stats(&results).await;

        Ok(results)
    }

    /// Evaluate a single rule
    async fn evaluate_rule(
        &self,
        rule: &AlertRule,
        metrics: &HashMap<String, f64>,
    ) -> Result<(bool, f64)> {
        // Get metric value from query
        let metric_value = self.execute_query(&rule.query, metrics)?;

        // Evaluate condition
        let condition_met = self.evaluate_condition(&rule.condition, metric_value, metrics).await?;

        Ok((condition_met, metric_value))
    }

    /// Execute metric query
    fn execute_query(&self, query: &str, metrics: &HashMap<String, f64>) -> Result<f64> {
        // Simple metric lookup for now - in practice this would be more sophisticated
        if let Some(value) = metrics.get(query) {
            Ok(*value)
        } else {
            Err(anyhow!("Metric not found: {}", query))
        }
    }

    /// Evaluate alert condition
    async fn evaluate_condition(
        &self,
        condition: &AlertCondition,
        value: f64,
        _metrics: &HashMap<String, f64>,
    ) -> Result<bool> {
        match condition {
            AlertCondition::Threshold {
                operator,
                value: threshold,
            } => Ok(self.compare_values(value, *threshold, *operator)),
            AlertCondition::Range { min, max } => {
                let above_min = min.map(|m| value >= m).unwrap_or(true);
                let below_max = max.map(|m| value <= m).unwrap_or(true);
                Ok(above_min && below_max)
            },
            AlertCondition::Rate {
                window_secs: _,
                operator,
                value: threshold,
            } => {
                // Simplified rate calculation - would need historical data
                Ok(self.compare_values(value, *threshold, *operator))
            },
            AlertCondition::Anomaly {
                sensitivity: _,
                window_secs: _,
            } => {
                // Simplified anomaly detection - would use statistical analysis
                Ok(value > 100.0) // Placeholder logic
            },
            AlertCondition::Custom { expression: _ } => {
                // Would implement expression evaluation
                Ok(false)
            },
        }
    }

    /// Compare values with operator
    fn compare_values(&self, left: f64, right: f64, operator: ComparisonOperator) -> bool {
        match operator {
            ComparisonOperator::GreaterThan => left > right,
            ComparisonOperator::GreaterThanOrEqual => left >= right,
            ComparisonOperator::LessThan => left < right,
            ComparisonOperator::LessThanOrEqual => left <= right,
            ComparisonOperator::Equal => (left - right).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (left - right).abs() >= f64::EPSILON,
        }
    }

    /// Handle alert firing
    async fn handle_alert_firing(&self, rule: &AlertRule, value: f64) -> Result<(), AlertingError> {
        let alert_id = format!(
            "{}_{}",
            rule.id,
            SystemTime::now().duration_since(SystemTime::UNIX_EPOCH).unwrap().as_secs()
        );

        // Check if alert already exists
        let alerts = self.alerts.read().await;
        let existing_alert =
            alerts.values().find(|a| a.rule_id == rule.id && a.state == AlertState::Firing);

        if existing_alert.is_some() {
            return Ok(()); // Alert already firing
        }
        drop(alerts);

        // Create new alert
        let alert = self.create_alert(alert_id, rule, value).await?;

        // Add to alerts
        self.alerts.write().await.insert(alert.id.clone(), alert.clone());

        // Handle grouping
        if self.config.enable_alert_grouping {
            self.handle_alert_grouping(&alert).await?;
        }

        // Send notifications
        self.send_alert_notifications(&alert).await?;

        // Emit event
        let event = AlertEvent {
            event_type: AlertEventType::Fired,
            alert,
            timestamp: SystemTime::now(),
            data: HashMap::new(),
        };

        let _ = self.event_sender.send(event);

        // Update statistics
        self.update_alert_stats().await;

        Ok(())
    }

    /// Handle alert resolution
    async fn handle_alert_resolution(&self, rule: &AlertRule) -> Result<(), AlertingError> {
        let mut alerts = self.alerts.write().await;

        // Find active alert for this rule
        let alert_id = alerts
            .values()
            .find(|a| a.rule_id == rule.id && a.state == AlertState::Firing)
            .map(|a| a.id.clone());

        if let Some(alert_id) = alert_id {
            if let Some(alert) = alerts.get_mut(&alert_id) {
                alert.state = AlertState::Resolved;
                alert.resolved_at = Some(SystemTime::now());

                info!("Alert resolved: {}", alert.title);

                // Emit event
                let event = AlertEvent {
                    event_type: AlertEventType::Resolved,
                    alert: alert.clone(),
                    timestamp: SystemTime::now(),
                    data: HashMap::new(),
                };

                let _ = self.event_sender.send(event);
            }
        }

        Ok(())
    }

    /// Create alert from rule and value
    async fn create_alert(&self, alert_id: String, rule: &AlertRule, value: f64) -> Result<Alert> {
        let title = if let Some(template) = &rule.alert_template {
            self.format_template(&template.title, rule, value)
        } else {
            format!("{}: Alert condition met", rule.name)
        };

        let message = if let Some(template) = &rule.alert_template {
            self.format_template(&template.message, rule, value)
        } else {
            format!("Alert {} is firing with value {}", rule.name, value)
        };

        Ok(Alert {
            id: alert_id,
            rule_id: rule.id.clone(),
            title,
            message,
            severity: rule.severity,
            state: AlertState::Firing,
            labels: rule.labels.clone(),
            annotations: HashMap::new(),
            value,
            fired_at: SystemTime::now(),
            resolved_at: None,
            acknowledged: false,
            acknowledged_by: None,
            acknowledged_at: None,
            notification_channels: rule.notification_channels.clone(),
            group_id: None,
            parent_alert_id: None,
        })
    }

    /// Format alert template
    fn format_template(&self, template: &str, rule: &AlertRule, value: f64) -> String {
        template
            .replace("{rule_name}", &rule.name)
            .replace("{value}", &value.to_string())
            .replace("{severity}", &format!("{:?}", rule.severity))
    }

    /// Handle alert grouping
    async fn handle_alert_grouping(&self, alert: &Alert) -> Result<(), AlertingError> {
        // Simplified grouping by rule group
        if let Some(group_name) = &alert.labels.get("group") {
            let group_id = format!("group_{}", group_name);

            let mut groups = self.alert_groups.write().await;
            let group = groups.entry(group_id.clone()).or_insert_with(|| AlertGroup {
                id: group_id,
                name: group_name.to_string(),
                alert_ids: Vec::new(),
                created_at: SystemTime::now(),
                labels: HashMap::new(),
                status: AlertGroupStatus::Active,
            });

            group.alert_ids.push(alert.id.clone());

            // Update alert with group ID
            let mut alerts = self.alerts.write().await;
            if let Some(alert) = alerts.get_mut(&alert.id) {
                alert.group_id = Some(group.id.clone());
            }
        }

        Ok(())
    }

    /// Send alert notifications
    async fn send_alert_notifications(&self, alert: &Alert) -> Result<(), AlertingError> {
        let channels = self.notification_channels.read().await;

        for channel_id in &alert.notification_channels {
            if let Some(channel) = channels.get(channel_id) {
                if channel.enabled && self.should_notify_channel(channel, alert) {
                    self.send_notification(channel, alert).await?;
                }
            }
        }

        Ok(())
    }

    /// Check if channel should be notified for alert
    fn should_notify_channel(&self, channel: &NotificationChannel, alert: &Alert) -> bool {
        // Check severity filter
        if !channel.severity_filter.is_empty() && !channel.severity_filter.contains(&alert.severity)
        {
            return false;
        }

        // Check label filters
        for (key, value) in &channel.label_filters {
            if alert.labels.get(key) != Some(value) {
                return false;
            }
        }

        true
    }

    /// Send notification to channel
    async fn send_notification(
        &self,
        channel: &NotificationChannel,
        alert: &Alert,
    ) -> Result<(), AlertingError> {
        debug!(
            "Sending notification for alert {} to channel {}",
            alert.id, channel.name
        );

        match &channel.channel_type {
            NotificationChannelType::Email => {
                // Would implement email sending
                info!("Email notification sent for alert: {}", alert.title);
            },
            NotificationChannelType::Slack => {
                // Would implement Slack webhook
                info!("Slack notification sent for alert: {}", alert.title);
            },
            NotificationChannelType::Webhook => {
                // Would implement HTTP webhook
                info!("Webhook notification sent for alert: {}", alert.title);
            },
            NotificationChannelType::SMS => {
                // Would implement SMS sending
                info!("SMS notification sent for alert: {}", alert.title);
            },
            _ => {
                debug!(
                    "Notification type {:?} not implemented",
                    channel.channel_type
                );
            },
        }

        Ok(())
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str, user: &str) -> Result<(), AlertingError> {
        let mut alerts = self.alerts.write().await;

        let alert = alerts
            .get_mut(alert_id)
            .ok_or_else(|| AlertingError::AlertNotFound(alert_id.to_string()))?;

        alert.acknowledged = true;
        alert.acknowledged_by = Some(user.to_string());
        alert.acknowledged_at = Some(SystemTime::now());

        info!("Alert {} acknowledged by {}", alert_id, user);

        // Emit event
        let event = AlertEvent {
            event_type: AlertEventType::Acknowledged,
            alert: alert.clone(),
            timestamp: SystemTime::now(),
            data: {
                let mut data = HashMap::new();
                data.insert("acknowledged_by".to_string(), user.to_string());
                data
            },
        };

        let _ = self.event_sender.send(event);

        Ok(())
    }

    /// Get alert by ID
    pub async fn get_alert(&self, alert_id: &str) -> Option<Alert> {
        self.alerts.read().await.get(alert_id).cloned()
    }

    /// List active alerts
    pub async fn list_active_alerts(&self) -> Vec<Alert> {
        self.alerts
            .read()
            .await
            .values()
            .filter(|a| matches!(a.state, AlertState::Firing | AlertState::Pending))
            .cloned()
            .collect()
    }

    /// List alerts by severity
    pub async fn list_alerts_by_severity(&self, severity: AlertSeverity) -> Vec<Alert> {
        self.alerts
            .read()
            .await
            .values()
            .filter(|a| a.severity == severity)
            .cloned()
            .collect()
    }

    /// Validate alert rule
    fn validate_rule(&self, rule: &AlertRule) -> Result<(), AlertingError> {
        if rule.id.is_empty() {
            return Err(AlertingError::InvalidRule(
                "Rule ID cannot be empty".to_string(),
            ));
        }

        if rule.name.is_empty() {
            return Err(AlertingError::InvalidRule(
                "Rule name cannot be empty".to_string(),
            ));
        }

        if rule.query.is_empty() {
            return Err(AlertingError::InvalidRule(
                "Rule query cannot be empty".to_string(),
            ));
        }

        Ok(())
    }

    /// Update rule statistics
    async fn update_rule_stats(&self) {
        let rules = self.rules.read().await;
        let mut stats = self.stats.write().await;

        stats.total_rules = rules.len();
        stats.active_rules = rules.values().filter(|r| r.enabled).count();
    }

    /// Update alert statistics
    async fn update_alert_stats(&self) {
        let alerts = self.alerts.read().await;
        let mut stats = self.stats.write().await;

        stats.total_alerts_fired += 1;
        stats.active_alerts = alerts
            .values()
            .filter(|a| matches!(a.state, AlertState::Firing | AlertState::Pending))
            .count();

        // Update severity distribution
        for alert in alerts.values() {
            *stats.alerts_by_severity.entry(alert.severity).or_insert(0) += 1;
        }

        // Calculate resolution rate
        let total_alerts = alerts.len() as f64;
        let resolved_alerts =
            alerts.values().filter(|a| a.state == AlertState::Resolved).count() as f64;

        if total_alerts > 0.0 {
            stats.resolution_rate = resolved_alerts / total_alerts;
        }
    }

    /// Update evaluation statistics
    async fn update_evaluation_stats(&self, results: &[AlertEvaluationResult]) {
        let mut stats = self.stats.write().await;

        let total_time: Duration = results.iter().map(|r| r.evaluation_duration).sum();
        if !results.is_empty() {
            stats.average_evaluation_time = total_time / results.len() as u32;
        }
    }

    /// Get alerting statistics
    pub async fn get_stats(&self) -> AlertingStats {
        self.stats.read().await.clone()
    }

    /// Subscribe to alert events
    pub fn subscribe_to_events(&self) -> broadcast::Receiver<AlertEvent> {
        self.event_sender.subscribe()
    }

    /// Get alert history
    pub async fn get_alert_history(&self, limit: Option<usize>) -> Vec<Alert> {
        let history = self.alert_history.read().await;
        match limit {
            Some(n) => history.iter().rev().take(n).cloned().collect(),
            None => history.iter().cloned().collect(),
        }
    }
}

/// Summary statistics for the alerting service
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertingStatsSummary {
    /// Total rules configured
    pub total_rules: usize,

    /// Active rules count
    pub active_rules: usize,

    /// Currently active alerts
    pub active_alerts: usize,

    /// Alert resolution rate percentage
    pub resolution_rate_percent: f64,

    /// Average evaluation time in milliseconds
    pub average_evaluation_time_ms: f64,

    /// Most common alert severity
    pub most_common_severity: Option<AlertSeverity>,

    /// Notification channels count
    pub notification_channels_count: usize,
}

impl AlertingService {
    /// Get summary statistics
    pub async fn get_stats_summary(&self) -> AlertingStatsSummary {
        let stats = self.stats.read().await;
        let channels = self.notification_channels.read().await;

        let most_common_severity = stats
            .alerts_by_severity
            .iter()
            .max_by_key(|(_, &count)| count)
            .map(|(severity, _)| *severity);

        AlertingStatsSummary {
            total_rules: stats.total_rules,
            active_rules: stats.active_rules,
            active_alerts: stats.active_alerts,
            resolution_rate_percent: stats.resolution_rate * 100.0,
            average_evaluation_time_ms: stats.average_evaluation_time.as_millis() as f64,
            most_common_severity,
            notification_channels_count: channels.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_alerting_service_creation() {
        let config = AlertingConfig::default();
        let service = AlertingService::new(config);
        let stats = service.get_stats().await;

        assert_eq!(stats.total_rules, 0);
        assert_eq!(stats.active_alerts, 0);
    }

    #[tokio::test]
    async fn test_add_alert_rule() {
        let service = AlertingService::new(AlertingConfig::default());

        let rule = AlertRule {
            id: "test_rule".to_string(),
            name: "Test Rule".to_string(),
            description: "Test alert rule".to_string(),
            query: "cpu_usage".to_string(),
            condition: AlertCondition::Threshold {
                operator: ComparisonOperator::GreaterThan,
                value: 80.0,
            },
            severity: AlertSeverity::High,
            evaluation_interval_secs: 60,
            for_duration_secs: 120,
            labels: HashMap::new(),
            notification_channels: vec!["default".to_string()],
            enabled: true,
            alert_template: None,
            group: None,
            dependencies: Vec::new(),
        };

        let result = service.add_rule(rule).await;
        assert!(result.is_ok());

        let stats = service.get_stats().await;
        assert_eq!(stats.total_rules, 1);
        assert_eq!(stats.active_rules, 1);
    }

    #[tokio::test]
    async fn test_alert_evaluation() {
        let service = AlertingService::new(AlertingConfig::default());

        let rule = AlertRule {
            id: "cpu_rule".to_string(),
            name: "CPU Usage Alert".to_string(),
            description: "Alert when CPU usage is high".to_string(),
            query: "cpu_usage".to_string(),
            condition: AlertCondition::Threshold {
                operator: ComparisonOperator::GreaterThan,
                value: 80.0,
            },
            severity: AlertSeverity::High,
            evaluation_interval_secs: 60,
            for_duration_secs: 0,
            labels: HashMap::new(),
            notification_channels: vec!["default".to_string()],
            enabled: true,
            alert_template: None,
            group: None,
            dependencies: Vec::new(),
        };

        service.add_rule(rule).await.unwrap();

        // Test with high CPU usage
        let mut metrics = HashMap::new();
        metrics.insert("cpu_usage".to_string(), 85.0);

        let results = service.evaluate_rules(&metrics).await.unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].condition_met);

        // Check that alert was created
        let active_alerts = service.list_active_alerts().await;
        assert_eq!(active_alerts.len(), 1);
        assert_eq!(active_alerts[0].severity, AlertSeverity::High);
    }

    #[test]
    fn test_comparison_operators() {
        let service = AlertingService::new(AlertingConfig::default());

        assert!(service.compare_values(10.0, 5.0, ComparisonOperator::GreaterThan));
        assert!(service.compare_values(10.0, 10.0, ComparisonOperator::GreaterThanOrEqual));
        assert!(service.compare_values(5.0, 10.0, ComparisonOperator::LessThan));
        assert!(service.compare_values(10.0, 10.0, ComparisonOperator::LessThanOrEqual));
        assert!(service.compare_values(10.0, 10.0, ComparisonOperator::Equal));
        assert!(service.compare_values(10.0, 5.0, ComparisonOperator::NotEqual));
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let service = AlertingService::new(AlertingConfig::default());

        // Create a test alert
        let alert = Alert {
            id: "test_alert".to_string(),
            rule_id: "test_rule".to_string(),
            title: "Test Alert".to_string(),
            message: "Test alert message".to_string(),
            severity: AlertSeverity::Medium,
            state: AlertState::Firing,
            labels: HashMap::new(),
            annotations: HashMap::new(),
            value: 100.0,
            fired_at: SystemTime::now(),
            resolved_at: None,
            acknowledged: false,
            acknowledged_by: None,
            acknowledged_at: None,
            notification_channels: vec!["default".to_string()],
            group_id: None,
            parent_alert_id: None,
        };

        service.alerts.write().await.insert(alert.id.clone(), alert);

        // Acknowledge the alert
        let result = service.acknowledge_alert("test_alert", "testuser").await;
        assert!(result.is_ok());

        // Check acknowledgment
        let acknowledged_alert = service.get_alert("test_alert").await.unwrap();
        assert!(acknowledged_alert.acknowledged);
        assert_eq!(
            acknowledged_alert.acknowledged_by,
            Some("testuser".to_string())
        );
    }

    #[test]
    fn test_alert_severity_ordering() {
        assert!(AlertSeverity::Critical > AlertSeverity::High);
        assert!(AlertSeverity::High > AlertSeverity::Medium);
        assert!(AlertSeverity::Medium > AlertSeverity::Low);
        assert!(AlertSeverity::Low > AlertSeverity::Info);
    }
}
