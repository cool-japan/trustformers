//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::analytics::DataPoint;
use super::super::events::*;
use super::super::metrics::*;
use super::super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::fmt::Debug;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, SystemTime};
use tokio::sync::{Mutex, RwLock};
use uuid::Uuid;

use super::functions::{AlertPersistence, NotificationChannel, ThresholdEvaluator};

/// Alert storage and history management
pub struct AlertStore {
    pub(super) active_alerts: Arc<RwLock<HashMap<String, ActiveAlert>>>,
    pub(super) alert_history: Arc<Mutex<VecDeque<HistoricalAlert>>>,
    pub(super) alert_index: Arc<AlertIndex>,
    pub(super) storage_backend: Option<Box<dyn AlertPersistence + Send + Sync>>,
    pub(super) retention_policy: AlertRetentionPolicy,
    pub(super) compression_enabled: bool,
}
impl AlertStore {
    pub fn new() -> Self {
        Self {
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(Mutex::new(VecDeque::new())),
            alert_index: Arc::new(AlertIndex {
                by_test: Arc::new(RwLock::new(HashMap::new())),
                by_severity: Arc::new(RwLock::new(HashMap::new())),
                by_timestamp: Arc::new(RwLock::new(Vec::new())),
            }),
            storage_backend: None,
            retention_policy: AlertRetentionPolicy {
                max_age: std::time::Duration::from_secs(86400),
                max_count: 10000,
            },
            compression_enabled: false,
        }
    }
    /// Store `alert` and update every secondary index.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` matches the rest of the alerting API.
    pub async fn store_alert(&self, alert: &ActiveAlert) -> Result<(), AlertError> {
        {
            let mut alerts = self.active_alerts.write().await;
            alerts.insert(alert.alert_id.clone(), alert.clone());
        }
        {
            let mut by_test = self.alert_index.by_test.write().await;
            let ids = by_test.entry(alert.test_id.clone()).or_default();
            if !ids.contains(&alert.alert_id) {
                ids.push(alert.alert_id.clone());
            }
        }
        {
            let mut by_severity = self.alert_index.by_severity.write().await;
            let ids = by_severity.entry(alert.severity).or_default();
            if !ids.contains(&alert.alert_id) {
                ids.push(alert.alert_id.clone());
            }
        }
        {
            let mut by_timestamp = self.alert_index.by_timestamp.write().await;
            by_timestamp.push((alert.triggered_at, alert.alert_id.clone()));
        }
        Ok(())
    }

    /// Ids of the alerts raised against `test_id`, newest last.
    pub async fn alerts_for_test(&self, test_id: &str) -> Vec<String> {
        self.alert_index.by_test.read().await.get(test_id).cloned().unwrap_or_default()
    }

    /// Ids of the alerts raised at `severity`.
    pub async fn alerts_by_severity(&self, severity: SeverityLevel) -> Vec<String> {
        self.alert_index
            .by_severity
            .read()
            .await
            .get(&severity)
            .cloned()
            .unwrap_or_default()
    }

    /// Whether alert bodies are stored compressed. Always `false`: nothing in
    /// this crate compresses an alert.
    pub fn compression_enabled(&self) -> bool {
        self.compression_enabled
    }

    /// Whether a persistent backend is attached. Always `false` until one is
    /// installed: this store is in-memory only.
    pub fn has_storage_backend(&self) -> bool {
        self.storage_backend.is_some()
    }

    /// Drop historical alerts that fall outside the retention policy.
    ///
    /// Returns how many were dropped.
    pub async fn prune_history(&self) -> usize {
        let now = chrono::Utc::now();
        let policy = self.retention_policy;
        let max_age = chrono::Duration::from_std(policy.max_age).unwrap_or(chrono::Duration::MAX);
        let mut history = self.alert_history.lock().await;
        let before = history.len();
        history.retain(|entry| now.signed_duration_since(entry.timestamp) <= max_age);
        while history.len() > policy.max_count {
            history.pop_front();
        }
        before - history.len()
    }
    pub async fn get_alert(&self, alert_id: &str) -> Result<Option<ActiveAlert>, AlertError> {
        let alerts = self.active_alerts.read().await;
        Ok(alerts.get(alert_id).cloned())
    }
    pub async fn update_alert(&self, alert: &ActiveAlert) -> Result<(), AlertError> {
        let mut alerts = self.active_alerts.write().await;
        alerts.insert(alert.alert_id.clone(), alert.clone());
        Ok(())
    }
    pub async fn get_active_alerts(&self) -> Result<Vec<ActiveAlert>, AlertError> {
        let alerts = self.active_alerts.read().await;
        Ok(alerts.values().cloned().collect())
    }
}
// 0.2.1 -- collapsed placeholder collaborators.
//
// The five managers below (RecoveryManager, EscalationManager,
// NotificationDispatcher, AlertCorrelator, SuppressionManager) each held four
// to six `Arc<...>` sub-components whose entire content was an empty map, an
// empty vector or a zero counter that nothing ever inserted into, read or
// incremented -- `AutoRecoveryEngine`, `RecoveryTracker`, `FlapDetection`,
// `RecoveryScheduler`, `RecoveryMetrics`, `EscalationExecutor`,
// `EscalationScheduler`, `EscalationMetrics`, `NotificationRateLimiter`,
// `TemplateEngine`, `DeliveryTracker`, `NotificationMetrics`,
// `CorrelationEngine`, `CorrelationCache`, `TemporalCorrelator`,
// `SpatialCorrelator`, `SuppressionScheduler`, `DynamicSuppressionEngine`,
// `MonitoringScheduler`, `ThresholdCache`, `EvaluationMetrics`,
// `RealTimeProcessor`, `RuleExecutor`, `ConditionEvaluator`, `RuleScheduler`,
// `EvaluationContext`. They gave the impression of a rate limiter, a template
// engine, a delivery tracker, a flap detector and a spatial correlator, none of
// which existed: every one of those managers' methods was `Ok(())`,
// `Ok(false)` or `Ok(Vec::new())` regardless.
//
// The empty sub-components are deleted. Each manager keeps only the state it
// really carries, and each unimplemented method now says in its own doc comment
// what it does not do, instead of a struct full of empty containers implying
// that it does.

/// Recovery conditions registered per alert rule.
///
/// This is a registry, not a recovery engine: nothing here evaluates a
/// condition, retries an action or detects flapping. See the note above.
#[derive(Debug)]
pub struct RecoveryManager {
    recovery_conditions: Arc<RwLock<HashMap<String, Vec<RecoveryCondition>>>>,
}
impl RecoveryManager {
    /// Build an empty registry.
    pub fn new() -> Self {
        Self {
            recovery_conditions: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register the recovery conditions attached to `rule_id`.
    pub async fn register_conditions(&self, rule_id: &str, conditions: Vec<RecoveryCondition>) {
        self.recovery_conditions.write().await.insert(rule_id.to_string(), conditions);
    }

    /// Recovery conditions registered for `rule_id`, if any.
    pub async fn conditions_for(&self, rule_id: &str) -> Option<Vec<RecoveryCondition>> {
        self.recovery_conditions.read().await.get(rule_id).cloned()
    }
}
/// Percentile-based thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PercentileThresholds {
    pub percentile: f64,
    pub calculation_window: Duration,
    pub minimum_samples: u32,
    pub update_frequency: Duration,
    pub smoothing_factor: Option<f64>,
}
/// Recommended action for alert resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecommendedAction {
    pub action_id: String,
    pub action_type: ActionType,
    pub description: String,
    pub priority: ActionPriority,
    pub estimated_effort: EstimatedEffort,
    pub expected_impact: ExpectedImpact,
    pub prerequisites: Vec<String>,
    pub automation_available: bool,
    pub automation_command: Option<String>,
}
/// How long resolved alerts are retained in history.
///
/// 0.2.1: all three fields were set at construction and never read, so
/// `alert_history` grew without bound while a "retention policy" sat beside it.
/// `max_age` and `max_count` are enforced by [`AlertStore::prune_history`] now.
/// `compression_after` is gone: nothing in this crate compresses an alert.
#[derive(Debug, Clone, Copy)]
pub(crate) struct AlertRetentionPolicy {
    /// Alerts older than this are dropped from history.
    max_age: std::time::Duration,
    /// At most this many historical alerts are retained.
    max_count: usize,
}
/// Static threshold values
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StaticThresholds {
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub recovery_threshold: Option<f64>,
    pub hysteresis_margin: Option<f64>,
}
/// Metric selector for condition evaluation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSelector {
    pub metric_name: String,
    pub test_id_pattern: Option<String>,
    pub tag_filters: HashMap<String, String>,
    pub aggregation_scope: AggregationScope,
    pub time_window: Duration,
}
/// Adaptive threshold learning
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveThresholds {
    pub learning_algorithm: LearningAlgorithm,
    pub adaptation_rate: f64,
    pub minimum_learning_period: Duration,
    pub seasonal_adjustment: bool,
    pub outlier_handling: OutlierHandling,
    pub convergence_criteria: ConvergenceCriteria,
}
/// Alert context and environment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertContext {
    pub test_execution_context: Option<ExecutionContext>,
    pub system_state: SystemState,
    pub environmental_factors: HashMap<String, String>,
    pub recent_changes: Vec<ChangeEvent>,
    pub related_metrics: HashMap<String, MetricValue>,
    pub dependency_status: HashMap<String, DependencyStatus>,
}
/// Secondary indexes over the stored alerts.
///
/// 0.2.1: all three maps were created empty and never inserted into or read, so
/// [`AlertStore`] could only ever answer by primary key -- the "index" indexed
/// nothing. [`AlertStore::store_alert`] maintains them now and
/// [`AlertStore::alerts_for_test`] / [`AlertStore::alerts_by_severity`] read
/// them.
#[derive(Debug)]
pub(crate) struct AlertIndex {
    by_test: Arc<RwLock<HashMap<String, Vec<String>>>>,
    by_severity: Arc<RwLock<HashMap<SeverityLevel, Vec<String>>>>,
    by_timestamp: Arc<RwLock<Vec<(SystemTime, String)>>>,
}
/// Alert group for correlated alerts
#[derive(Debug, Clone)]
pub struct AlertGroup {
    pub group_id: String,
    pub group_type: GroupType,
    pub alert_ids: Vec<String>,
    pub root_cause_alert: Option<String>,
    pub created_at: SystemTime,
    pub last_updated: SystemTime,
    pub correlation_score: f64,
    pub group_severity: SeverityLevel,
    pub group_status: GroupStatus,
}
/// Maintenance window configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindow {
    pub window_id: String,
    pub window_name: String,
    pub start_time: SystemTime,
    pub end_time: SystemTime,
    pub affected_systems: Vec<String>,
    pub suppression_level: SuppressionLevel,
    pub created_by: String,
    pub approval_required: bool,
    pub notification_config: MaintenanceNotificationConfig,
}
/// Notification request
#[derive(Debug, Clone)]
pub struct NotificationRequest {
    pub request_id: String,
    pub alert_id: String,
    pub notification_type: NotificationType,
    pub recipients: Vec<String>,
    pub message_template: String,
    pub template_variables: HashMap<String, String>,
    pub priority: NotificationPriority,
    pub delivery_requirements: DeliveryRequirements,
    pub retry_config: RetryConfig,
    pub created_at: SystemTime,
}
/// Threshold configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdConfig {
    pub static_thresholds: Option<StaticThresholds>,
    pub dynamic_thresholds: Option<DynamicThresholds>,
    pub adaptive_thresholds: Option<AdaptiveThresholds>,
    pub baseline_thresholds: Option<BaselineThresholds>,
    pub percentile_thresholds: Option<PercentileThresholds>,
}
/// Alert system errors
#[derive(Debug, Clone)]
pub enum AlertError {
    RuleEvaluationError {
        rule_id: String,
        reason: String,
    },
    ThresholdEvaluationError {
        condition_id: String,
        reason: String,
    },
    NotificationError {
        channel: String,
        reason: String,
    },
    EscalationError {
        policy_id: String,
        reason: String,
    },
    StorageError {
        operation: String,
        reason: String,
    },
    CorrelationError {
        reason: String,
    },
    ValidationError {
        field: String,
        reason: String,
    },
    ConfigurationError {
        parameter: String,
        reason: String,
    },
    SuppressionError {
        reason: String,
    },
    RecoveryError {
        reason: String,
    },
    NotFound(String),
}
/// Comprehensive alert rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRule {
    pub rule_id: String,
    pub rule_name: String,
    pub description: String,
    pub category: AlertCategory,
    pub severity: SeverityLevel,
    pub conditions: Vec<AlertCondition>,
    pub evaluation_window: Duration,
    pub evaluation_frequency: Duration,
    pub threshold_config: ThresholdConfig,
    pub suppression_config: SuppressionConfig,
    pub escalation_policy_id: Option<String>,
    pub notification_channels: Vec<String>,
    pub recovery_conditions: Vec<RecoveryCondition>,
    pub metadata: AlertRuleMetadata,
    pub enabled: bool,
    pub created_at: SystemTime,
    pub last_modified: SystemTime,
    pub last_triggered: Option<SystemTime>,
    pub trigger_count: u64,
}
/// Active alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub rule_id: String,
    pub test_id: String,
    pub alert_type: AlertType,
    pub severity: SeverityLevel,
    pub status: AlertStatus,
    pub title: String,
    pub description: String,
    pub triggered_at: SystemTime,
    pub last_updated: SystemTime,
    pub acknowledgment_info: Option<AcknowledgmentInfo>,
    pub escalation_info: Option<EscalationInfo>,
    pub suppression_info: Option<SuppressionInfo>,
    pub context_data: AlertContext,
    pub impact_assessment: ImpactAssessment,
    pub recommended_actions: Vec<RecommendedAction>,
    pub alert_fingerprint: String,
    pub correlation_ids: Vec<String>,
    pub resolution_info: Option<ResolutionInfo>,
}
/// Dynamic threshold calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicThresholds {
    pub calculation_method: DynamicThresholdMethod,
    pub lookback_period: Duration,
    pub sensitivity: f64,
    pub minimum_samples: u32,
    pub confidence_level: f64,
    pub update_frequency: Duration,
}
/// Suppression rule definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionRule {
    pub rule_id: String,
    pub rule_name: String,
    pub conditions: Vec<SuppressionCondition>,
    pub suppression_duration: Duration,
    pub affected_rules: Vec<String>,
    pub priority: u32,
    pub enabled: bool,
    pub created_at: SystemTime,
}
/// Recovery condition definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryCondition {
    pub condition_id: String,
    pub condition_type: RecoveryConditionType,
    pub metric_criteria: MetricCriteria,
    pub duration_requirement: Duration,
    pub confidence_threshold: f64,
    pub validation_checks: Vec<ValidationCheck>,
}
/// Recent change event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChangeEvent {
    pub change_id: String,
    pub change_type: ChangeType,
    pub change_description: String,
    pub changed_at: SystemTime,
    pub changed_by: String,
    pub impact_scope: Vec<String>,
}
/// Registry of escalation policies.
///
/// This stores policies and records escalation *requests*; it does not execute
/// them -- nothing here pages anyone or waits out an acknowledgment timeout.
#[derive(Debug)]
pub struct EscalationManager {
    escalation_policies: Arc<RwLock<HashMap<String, EscalationPolicy>>>,
    /// Escalations requested but never executed, keyed by alert id.
    pending_escalations: Arc<RwLock<HashMap<String, String>>>,
}
impl EscalationManager {
    /// Build an empty registry.
    pub fn new() -> Self {
        Self {
            escalation_policies: Arc::new(RwLock::new(HashMap::new())),
            pending_escalations: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register an escalation policy.
    pub async fn register_policy(&self, policy: EscalationPolicy) {
        self.escalation_policies.write().await.insert(policy.policy_id.clone(), policy);
    }

    /// The registered policy with this id, if any.
    pub async fn policy(&self, policy_id: &str) -> Option<EscalationPolicy> {
        self.escalation_policies.read().await.get(policy_id).cloned()
    }

    /// Record that `alert` should escalate under `policy_id`.
    ///
    /// The escalation is *recorded, not performed*: this crate has no notifier
    /// to page and no scheduler to wait out an acknowledgment timeout, so
    /// nothing is dispatched. [`Self::pending_escalation`] exposes what was
    /// recorded.
    ///
    /// # Errors
    ///
    /// Returns [`AlertError::EscalationError`] when `policy_id` names a policy
    /// that was never registered -- silently accepting an unknown policy would
    /// leave the caller believing an escalation path exists.
    pub async fn schedule_escalation(
        &self,
        alert: &ActiveAlert,
        policy_id: &str,
    ) -> Result<(), AlertError> {
        if !self.escalation_policies.read().await.contains_key(policy_id) {
            return Err(AlertError::EscalationError {
                policy_id: policy_id.to_string(),
                reason: "no escalation policy is registered under this id".to_string(),
            });
        }
        self.pending_escalations
            .write()
            .await
            .insert(alert.alert_id.clone(), policy_id.to_string());
        Ok(())
    }

    /// The policy id recorded for `alert_id`, if an escalation is pending.
    pub async fn pending_escalation(&self, alert_id: &str) -> Option<String> {
        self.pending_escalations.read().await.get(alert_id).cloned()
    }

    /// Drop any pending escalation for `alert_id`.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` matches the rest of the alerting API.
    pub async fn cancel_escalation(&self, alert_id: &str) -> Result<(), AlertError> {
        self.pending_escalations.write().await.remove(alert_id);
        Ok(())
    }
}
/// Baseline-based thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BaselineThresholds {
    pub baseline_id: String,
    pub deviation_multiplier: f64,
    pub deviation_type: DeviationType,
    pub baseline_update_strategy: BaselineUpdateStrategy,
    pub seasonal_adjustment: bool,
}
/// Dispatches alert notifications over registered channels.
pub struct NotificationDispatcher {
    pub(super) notification_channels: HashMap<String, Box<dyn NotificationChannel + Send + Sync>>,
    /// Notifications dispatched per channel, counted for real.
    pub(super) sent_per_channel: Arc<RwLock<HashMap<String, u64>>>,
}

impl NotificationDispatcher {
    /// Build a dispatcher with no channels registered.
    pub fn new() -> Self {
        Self {
            notification_channels: HashMap::new(),
            sent_per_channel: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a channel under `name`.
    pub fn register_channel(
        &mut self,
        name: String,
        channel: Box<dyn NotificationChannel + Send + Sync>,
    ) {
        self.notification_channels.insert(name, channel);
    }

    /// Names of the registered channels.
    pub fn channel_names(&self) -> Vec<&str> {
        self.notification_channels.keys().map(String::as_str).collect()
    }

    /// Notifications successfully delivered per channel.
    pub async fn sent_counts(&self) -> HashMap<String, u64> {
        self.sent_per_channel.read().await.clone()
    }

    /// Deliver `alert` over every registered, available channel.
    ///
    /// A dispatcher with no channels registered delivers nothing and reports
    /// that, rather than returning `Ok(())` as though it had.
    ///
    /// # Errors
    ///
    /// Returns [`AlertError::NotificationError`] when no channel is registered,
    /// or when every registered channel refused or failed.
    pub async fn dispatch_alert_notifications(
        &self,
        alert: &ActiveAlert,
    ) -> Result<(), AlertError> {
        if self.notification_channels.is_empty() {
            return Err(AlertError::NotificationError {
                channel: "<none>".to_string(),
                reason: format!(
                    "alert {} was not delivered: no notification channel is registered",
                    alert.alert_id
                ),
            });
        }
        let mut delivered = 0usize;
        let mut failures = Vec::new();
        for (name, channel) in &self.notification_channels {
            if !channel.is_available() {
                failures.push(format!("{name}: channel reports itself unavailable"));
                continue;
            }
            // Build the request per channel: `notification_type` and the
            // channel's own rate limits come from the channel itself rather
            // than being guessed at a single shared value.
            let limits = channel.get_rate_limits();
            let request = NotificationRequest {
                request_id: Uuid::new_v4().to_string(),
                alert_id: alert.alert_id.clone(),
                notification_type: match channel.get_channel_type() {
                    NotificationChannelType::Email => NotificationType::Email,
                    NotificationChannelType::Sms => NotificationType::Sms,
                    NotificationChannelType::Slack => NotificationType::Slack,
                    NotificationChannelType::PagerDuty => NotificationType::PagerDuty,
                    // A custom channel has no `NotificationType` of its own;
                    // webhook is the closest honest label for "delivered over
                    // an operator-supplied endpoint".
                    NotificationChannelType::Webhook | NotificationChannelType::Custom => {
                        NotificationType::Webhook
                    },
                },
                recipients: Vec::new(),
                message_template: alert.title.clone(),
                template_variables: HashMap::new(),
                priority: NotificationPriority::Normal,
                delivery_requirements: DeliveryRequirements {
                    require_acknowledgment: false,
                    max_delivery_time: Duration::from_secs(30),
                },
                retry_config: {
                    let backoff = BackoffStrategy {
                        initial_delay: Duration::from_secs(1),
                        max_delay: Duration::from_secs(30),
                        multiplier: 2.0,
                    };
                    // Never retry harder than the channel says it can take.
                    let max_retries = limits.burst_size.min(3);
                    RetryConfig {
                        max_retries,
                        initial_delay: backoff.initial_delay,
                        max_delay: backoff.max_delay,
                        backoff_strategy: backoff.clone(),
                        retry_predicate: RetryPredicate {
                            max_attempts: max_retries,
                            backoff,
                        },
                    }
                },
                created_at: SystemTime::now(),
            };
            match channel.send_notification(&request) {
                Ok(_) => {
                    delivered += 1;
                    *self.sent_per_channel.write().await.entry(name.clone()).or_insert(0) += 1;
                },
                Err(e) => failures.push(format!("{name}: {e:?}")),
            }
        }
        if delivered == 0 {
            return Err(AlertError::NotificationError {
                channel: "<all>".to_string(),
                reason: format!(
                    "alert {} reached no channel: {}",
                    alert.alert_id,
                    failures.join("; ")
                ),
            });
        }
        Ok(())
    }
}
/// Groups related alerts.
///
/// Correlation here is by alert *fingerprint* only -- alerts raised by the same
/// rule against the same test and condition are one group. Temporal and
/// topological correlation are not implemented: this crate has neither a time
/// series of past alerts nor a system topology to correlate across.
#[derive(Debug)]
pub struct AlertCorrelator {
    correlation_rules: Arc<RwLock<Vec<CorrelationRule>>>,
    alert_groups: Arc<RwLock<HashMap<String, AlertGroup>>>,
}
impl AlertCorrelator {
    /// Build a correlator with no rules registered.
    pub fn new() -> Self {
        Self {
            correlation_rules: Arc::new(RwLock::new(Vec::new())),
            alert_groups: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Register a correlation rule. Rules are stored for inspection; the
    /// fingerprint grouping below does not consult them.
    pub async fn register_rule(&self, rule: CorrelationRule) {
        self.correlation_rules.write().await.push(rule);
    }

    /// Groups formed so far.
    pub async fn groups(&self) -> Vec<AlertGroup> {
        self.alert_groups.read().await.values().cloned().collect()
    }

    /// Group `alerts` by fingerprint, returning the groups that were touched.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` matches the rest of the alerting API.
    pub async fn correlate_alerts(
        &self,
        alerts: &[ActiveAlert],
    ) -> Result<Vec<AlertGroup>, AlertError> {
        let now = SystemTime::now();
        let mut groups = self.alert_groups.write().await;
        let mut touched: HashMap<String, AlertGroup> = HashMap::new();
        for alert in alerts {
            let group = groups.entry(alert.alert_fingerprint.clone()).or_insert_with(|| {
                AlertGroup {
                    group_id: alert.alert_fingerprint.clone(),
                    group_type: GroupType::BySource,
                    alert_ids: Vec::new(),
                    root_cause_alert: Some(alert.alert_id.clone()),
                    created_at: now,
                    last_updated: now,
                    // Fingerprint equality is exact, so a match is certain --
                    // this is not a heuristic score dressed up as one.
                    correlation_score: 1.0,
                    group_severity: alert.severity,
                    group_status: GroupStatus::Active,
                }
            });
            if !group.alert_ids.contains(&alert.alert_id) {
                group.alert_ids.push(alert.alert_id.clone());
            }
            group.last_updated = now;
            touched.insert(group.group_id.clone(), group.clone());
        }
        Ok(touched.into_values().collect())
    }
}
// 0.2.1: four notification-channel shells lived here -- `EmailChannel`,
// `SlackChannel`, `SmsChannel` and `WebhookChannel`. None of them was ever
// constructed and none implemented `NotificationChannel`, so registering one
// with `NotificationDispatcher` was impossible: they were four struct
// definitions that looked like an email/Slack/SMS/webhook integration and
// could not send anything. They are deleted; `NotificationChannel` in
// `functions.rs` remains as the seam a real integration implements.

/// Registry of alert rules, and the evaluator that decides whether they fire.
///
/// 0.2.1: this held four `Arc` sub-components -- `RuleExecutor`,
/// `ConditionEvaluator`, `RuleScheduler` and `EvaluationContext` -- constructed
/// from a `&Default::default()` monitoring config and never used by any method
/// on this type. Rule evaluation happens inline in [`Self::evaluate_rule`]; the
/// four are gone rather than left implying a scheduler and an evaluation
/// context that never ran.
#[derive(Debug)]
pub struct AlertRuleEngine {
    rules: Arc<RwLock<HashMap<String, AlertRule>>>,
    /// Rules that must fire before a dependent rule is considered, keyed by
    /// dependent rule id. Registered by [`Self::set_rule_dependencies`].
    rule_dependencies: Arc<RwLock<HashMap<String, Vec<String>>>>,
}
impl AlertRuleEngine {
    fn new(_config: &AlertConfig) -> Self {
        Self {
            rules: Arc::new(RwLock::new(HashMap::new())),
            rule_dependencies: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Declare that `rule_id` depends on `depends_on`.
    pub async fn set_rule_dependencies(&self, rule_id: &str, depends_on: Vec<String>) {
        self.rule_dependencies.write().await.insert(rule_id.to_string(), depends_on);
    }

    /// Rules `rule_id` depends on, if any were declared.
    pub async fn rule_dependencies(&self, rule_id: &str) -> Option<Vec<String>> {
        self.rule_dependencies.read().await.get(rule_id).cloned()
    }
    async fn get_applicable_rules(&self, test_id: &str) -> Result<Vec<AlertRule>, AlertError> {
        let rules = self.rules.read().await;
        let applicable_rules = rules
            .values()
            .filter(|rule| rule.enabled && self.rule_applies_to_test(rule, test_id))
            .cloned()
            .collect();
        Ok(applicable_rules)
    }
    fn rule_applies_to_test(&self, rule: &AlertRule, test_id: &str) -> bool {
        rule.conditions.iter().any(|condition| {
            condition
                .metric_selector
                .test_id_pattern
                .as_ref()
                .is_none_or(|pattern| test_id.contains(pattern))
        })
    }
    async fn add_rule(&self, rule: AlertRule) -> Result<(), AlertError> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.rule_id.clone(), rule);
        Ok(())
    }
    async fn update_rule(&self, rule: AlertRule) -> Result<(), AlertError> {
        let mut rules = self.rules.write().await;
        rules.insert(rule.rule_id.clone(), rule);
        Ok(())
    }
    async fn delete_rule(&self, rule_id: &str) -> Result<(), AlertError> {
        let mut rules = self.rules.write().await;
        rules.remove(rule_id);
        Ok(())
    }
}
/// Alert suppression information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuppressionInfo {
    pub suppressed_at: SystemTime,
    pub suppressed_by: String,
    pub suppression_reason: String,
    pub suppression_duration: Option<Duration>,
    pub auto_suppression: bool,
    pub suppression_conditions: Vec<SuppressionCondition>,
}
/// System state at alert time.
///
/// 0.2.1: this used to be filled in at the single alert-construction site with
/// the alerting metric plus hardcoded zeroes and a fixed
/// `Medium`/`Degraded` verdict. It is now built by [`SystemState::sample`],
/// which reads the host through `sysinfo`; anything the host cannot answer is
/// `None` rather than zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// CPU usage percentage, carried over from the sample that raised the
    /// alert. `None` when that sample had no usable CPU delta.
    pub cpu_utilization: Option<f64>,
    /// Fraction of host memory in use, in `0.0..=1.0`.
    pub memory_utilization: f64,
    /// Fraction of host disk capacity in use across all mounted disks, in
    /// `0.0..=1.0`. `None` when the host reports no disks.
    pub disk_utilization: Option<f64>,
    /// Network throughput in bytes/second carried over from the sample that
    /// raised the alert. `None` when that sample could not measure a rate.
    ///
    /// Renamed from `network_utilization`: this is a throughput, never a
    /// utilization fraction — there is no link-capacity figure to divide by.
    pub network_bytes_per_second: Option<f64>,
    /// Number of processes the host reports.
    pub active_processes: u32,
    /// One-minute load average. `None` on platforms that do not report one.
    pub load_average: Option<f64>,
    /// Host uptime.
    pub system_uptime: Duration,
    /// Pressure verdict derived from the measurements above — see
    /// [`SystemState::sample`] for the exact rule.
    pub resource_pressure: PressureLevel,
    /// Health verdict derived from `resource_pressure`.
    pub health_status: HealthStatus,
}

impl SystemState {
    /// Sample the host, carrying the CPU and network readings over from the
    /// streaming sample that raised the alert.
    ///
    /// Derivation rules, applied to measured values only:
    /// * `resource_pressure` follows memory occupancy — `< 0.60` Low,
    ///   `< 0.80` Medium, `< 0.95` High, otherwise Critical — escalated one
    ///   step when the one-minute load average exceeds the host's CPU count.
    /// * `health_status` maps Low/Medium to `Healthy`, High to `Degraded` and
    ///   Critical to `Critical`.
    pub fn sample(metrics: &StreamingMetrics) -> Self {
        let mut system = sysinfo::System::new();
        system.refresh_memory();
        system.refresh_processes(sysinfo::ProcessesToUpdate::All, false);

        let total_memory = system.total_memory();
        let memory_utilization = if total_memory == 0 {
            0.0
        } else {
            system.used_memory() as f64 / total_memory as f64
        };

        let disks = sysinfo::Disks::new_with_refreshed_list();
        let (disk_total, disk_available) = disks.iter().fold((0u64, 0u64), |(t, a), disk| {
            (
                t.saturating_add(disk.total_space()),
                a.saturating_add(disk.available_space()),
            )
        });
        let disk_utilization = if disk_total == 0 {
            None
        } else {
            Some((disk_total.saturating_sub(disk_available)) as f64 / disk_total as f64)
        };

        let load_one = sysinfo::System::load_average().one;
        // `sysinfo` reports 0.0 for the load average on platforms that have no
        // such concept; treat that as "not reported" rather than "idle".
        let load_average = if load_one > 0.0 { Some(load_one) } else { None };

        let cpu_count = sysinfo::System::new_all().cpus().len().max(1) as f64;
        let mut resource_pressure = if memory_utilization < 0.60 {
            PressureLevel::Low
        } else if memory_utilization < 0.80 {
            PressureLevel::Medium
        } else if memory_utilization < 0.95 {
            PressureLevel::High
        } else {
            PressureLevel::Critical
        };
        if load_average.is_some_and(|load| load > cpu_count) {
            resource_pressure = match resource_pressure {
                PressureLevel::Low => PressureLevel::Medium,
                PressureLevel::Medium => PressureLevel::High,
                PressureLevel::High | PressureLevel::Critical => PressureLevel::Critical,
            };
        }
        let health_status = match resource_pressure {
            PressureLevel::Low | PressureLevel::Medium => HealthStatus::Healthy,
            PressureLevel::High => HealthStatus::Degraded,
            PressureLevel::Critical => HealthStatus::Critical,
        };

        Self {
            cpu_utilization: metrics.instantaneous_cpu,
            memory_utilization,
            disk_utilization,
            network_bytes_per_second: metrics.instantaneous_network_rate,
            active_processes: system.processes().len() as u32,
            load_average,
            system_uptime: Duration::from_secs(sysinfo::System::uptime()),
            resource_pressure,
            health_status,
        }
    }
}
/// Alert escalation information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationInfo {
    pub escalation_level: u32,
    pub escalated_at: SystemTime,
    pub escalated_to: Vec<String>,
    pub escalation_reason: String,
    pub next_escalation: Option<SystemTime>,
    pub max_escalation_reached: bool,
}
/// Alert statistics tracking
#[derive(Debug)]
pub struct AlertStatistics {
    pub total_alerts_generated: AtomicU64,
    pub alerts_by_severity: Arc<RwLock<HashMap<SeverityLevel, u64>>>,
    pub alerts_by_category: Arc<RwLock<HashMap<AlertCategory, u64>>>,
    pub average_resolution_time: Arc<RwLock<HashMap<SeverityLevel, Duration>>>,
    pub false_positive_rate: Arc<RwLock<f64>>,
    pub escalation_rates: Arc<RwLock<HashMap<String, f64>>>,
    pub notification_delivery_rates: Arc<RwLock<HashMap<String, f64>>>,
    pub alert_frequency: Arc<RwLock<HashMap<String, u64>>>,
}
impl AlertStatistics {
    pub(crate) fn new() -> Self {
        Self {
            total_alerts_generated: AtomicU64::new(0),
            alerts_by_severity: Arc::new(RwLock::new(HashMap::new())),
            alerts_by_category: Arc::new(RwLock::new(HashMap::new())),
            average_resolution_time: Arc::new(RwLock::new(HashMap::new())),
            false_positive_rate: Arc::new(RwLock::new(0.0)),
            escalation_rates: Arc::new(RwLock::new(HashMap::new())),
            notification_delivery_rates: Arc::new(RwLock::new(HashMap::new())),
            alert_frequency: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    pub(crate) async fn record_alert_generated(&self, alert: &ActiveAlert) {
        self.total_alerts_generated.fetch_add(1, Ordering::Relaxed);
        {
            let mut by_severity = self.alerts_by_severity.write().await;
            *by_severity.entry(alert.severity).or_insert(0) += 1;
        }
        {
            let mut by_category = self.alerts_by_category.write().await;
            *by_category.entry(AlertCategory::Performance).or_insert(0) += 1;
        }
    }
    pub(crate) async fn record_alert_resolved(
        &self,
        alert: &ActiveAlert,
        resolution_time: Duration,
    ) {
        let mut avg_times = self.average_resolution_time.write().await;
        avg_times.insert(alert.severity, resolution_time);
    }
}
/// Alert condition types and configurations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertCondition {
    pub condition_id: String,
    pub condition_type: AlertConditionType,
    pub metric_selector: MetricSelector,
    pub operator: ComparisonOperator,
    pub threshold_value: ThresholdValue,
    pub duration_requirement: Option<Duration>,
    pub aggregation_method: Option<AggregationMethod>,
    pub condition_weight: f64,
    pub evaluation_context: ConditionContext,
}
/// Types of alert conditions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertConditionType {
    Threshold,
    Anomaly,
    Trend,
    Pattern,
    Composite,
    Custom { expression: String },
}
/// Threshold evaluation result
#[derive(Debug, Clone)]
pub struct ThresholdEvaluationResult {
    pub rule_id: String,
    pub triggered: bool,
    pub severity: SeverityLevel,
    pub current_value: f64,
    pub threshold_value: f64,
    pub deviation_magnitude: f64,
    pub confidence_score: f64,
    pub evaluation_metadata: EvaluationMetadata,
    pub supporting_data: Vec<DataPoint>,
}
/// Alert acknowledgment information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcknowledgmentInfo {
    pub acknowledged_by: String,
    pub acknowledged_at: SystemTime,
    pub acknowledgment_note: Option<String>,
    pub auto_acknowledgment: bool,
    pub timeout: Option<SystemTime>,
}
/// Alert resolution information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionInfo {
    pub resolved_at: SystemTime,
    pub resolved_by: String,
    pub resolution_method: ResolutionMethod,
    pub resolution_notes: String,
    pub resolution_time: Duration,
    pub root_cause: Option<String>,
    pub preventive_measures: Vec<String>,
    pub lessons_learned: Vec<String>,
}
/// Decides whether an alert is currently suppressed.
///
/// Suppression is decided by two real mechanisms: an operator-declared
/// maintenance window covering `now`, and an explicit per-rule suppression
/// registered by [`Self::suppress_rule`]. Learned/dynamic suppression is not
/// implemented -- nothing here trains a model.
#[derive(Debug)]
pub struct SuppressionManager {
    suppression_rules: Arc<RwLock<Vec<SuppressionRule>>>,
    active_suppressions: Arc<RwLock<HashMap<String, ActiveSuppression>>>,
    maintenance_windows: Arc<RwLock<Vec<MaintenanceWindow>>>,
}
impl SuppressionManager {
    /// Build a manager with nothing suppressed.
    pub fn new() -> Self {
        Self {
            suppression_rules: Arc::new(RwLock::new(Vec::new())),
            active_suppressions: Arc::new(RwLock::new(HashMap::new())),
            maintenance_windows: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Register a suppression rule for inspection by
    /// [`Self::suppression_rules`].
    pub async fn register_rule(&self, rule: SuppressionRule) {
        self.suppression_rules.write().await.push(rule);
    }

    /// Registered suppression rules.
    pub async fn suppression_rules(&self) -> Vec<SuppressionRule> {
        self.suppression_rules.read().await.clone()
    }

    /// Declare a maintenance window. Alerts raised inside it are suppressed.
    pub async fn add_maintenance_window(&self, window: MaintenanceWindow) {
        self.maintenance_windows.write().await.push(window);
    }

    /// Suppress every alert from `rule_id` until further notice.
    pub async fn suppress_rule(&self, rule_id: &str, reason: String) {
        self.active_suppressions.write().await.insert(
            rule_id.to_string(),
            ActiveSuppression {
                suppression_id: rule_id.to_string(),
                start_time: chrono::Utc::now(),
                end_time: None,
                reason,
            },
        );
    }

    /// Lift an explicit suppression.
    pub async fn unsuppress_rule(&self, rule_id: &str) {
        self.active_suppressions.write().await.remove(rule_id);
    }

    /// Whether `alert` is suppressed right now.
    ///
    /// # Errors
    ///
    /// Infallible; the `Result` matches the rest of the alerting API.
    pub async fn is_suppressed(&self, alert: &ActiveAlert) -> Result<bool, AlertError> {
        let now = SystemTime::now();
        {
            let windows = self.maintenance_windows.read().await;
            if windows.iter().any(|window| window.start_time <= now && now < window.end_time) {
                return Ok(true);
            }
        }
        let suppressions = self.active_suppressions.read().await;
        let Some(active) = suppressions.get(&alert.rule_id) else {
            return Ok(false);
        };
        match active.end_time {
            None => Ok(true),
            Some(end_time) => Ok(chrono::Utc::now() < end_time),
        }
    }
}
/// Escalation policy definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationPolicy {
    pub policy_id: String,
    pub policy_name: String,
    pub description: String,
    pub escalation_levels: Vec<EscalationLevel>,
    pub escalation_conditions: Vec<EscalationCondition>,
    pub time_based_escalation: bool,
    pub severity_based_escalation: bool,
    pub acknowledgment_timeout: Duration,
    pub max_escalation_level: u32,
    pub auto_resolution_timeout: Option<Duration>,
    pub business_hours_config: Option<BusinessHoursConfig>,
    pub holiday_calendar: Option<String>,
}
/// Main alert management system
#[derive(Debug)]
pub struct AlertManager {
    config: AlertConfig,
    rule_engine: Arc<AlertRuleEngine>,
    threshold_monitor: Arc<ThresholdMonitor>,
    escalation_manager: Arc<EscalationManager>,
    notification_dispatcher: Arc<NotificationDispatcher>,
    alert_store: Arc<AlertStore>,
    alert_correlator: Arc<AlertCorrelator>,
    pub(crate) alert_statistics: Arc<AlertStatistics>,
    suppression_manager: Arc<SuppressionManager>,
    recovery_manager: Arc<RecoveryManager>,
}
impl AlertManager {
    /// The configuration this manager was built with.
    ///
    /// 0.2.1: `config` was cloned into the struct and never read again, so
    /// `AlertConfig::enabled` did not gate anything -- a manager configured
    /// with `enabled: false` still evaluated rules and dispatched alerts.
    /// [`Self::process_metrics`] honours it now.
    pub fn config(&self) -> &AlertConfig {
        &self.config
    }

    /// The threshold monitor this manager holds.
    ///
    /// It carries no evaluators by default (see [`ThresholdMonitor`]); this
    /// accessor exists so a caller can register their own.
    pub fn threshold_monitor(&self) -> &Arc<ThresholdMonitor> {
        &self.threshold_monitor
    }

    /// The recovery-condition registry.
    pub fn recovery_manager(&self) -> &Arc<RecoveryManager> {
        &self.recovery_manager
    }

    /// The suppression manager, for declaring maintenance windows.
    pub fn suppression_manager(&self) -> &Arc<SuppressionManager> {
        &self.suppression_manager
    }

    /// The escalation policy registry.
    pub fn escalation_manager(&self) -> &Arc<EscalationManager> {
        &self.escalation_manager
    }

    /// The notification dispatcher, for registering channels.
    pub fn notification_dispatcher(&self) -> &Arc<NotificationDispatcher> {
        &self.notification_dispatcher
    }

    /// The alert store.
    pub fn alert_store(&self) -> &Arc<AlertStore> {
        &self.alert_store
    }

    /// The alert correlator.
    pub fn alert_correlator(&self) -> &Arc<AlertCorrelator> {
        &self.alert_correlator
    }

    /// Create new alert manager
    pub fn new(config: AlertConfig) -> Self {
        Self {
            config: config.clone(),
            rule_engine: Arc::new(AlertRuleEngine::new(&config)),
            threshold_monitor: Arc::new(ThresholdMonitor::new()),
            escalation_manager: Arc::new(EscalationManager::new()),
            notification_dispatcher: Arc::new(NotificationDispatcher::new()),
            alert_store: Arc::new(AlertStore::new()),
            alert_correlator: Arc::new(AlertCorrelator::new()),
            alert_statistics: Arc::new(AlertStatistics::new()),
            suppression_manager: Arc::new(SuppressionManager::new()),
            recovery_manager: Arc::new(RecoveryManager::new()),
        }
    }
    /// Process incoming metrics for alert evaluation
    pub async fn process_metrics(
        &self,
        metrics: &StreamingMetrics,
    ) -> Result<Vec<ActiveAlert>, AlertError> {
        if !self.config.enabled {
            // Configured off means off: return nothing rather than quietly
            // evaluating rules the operator disabled.
            return Ok(Vec::new());
        }
        let mut triggered_alerts = Vec::new();
        let applicable_rules = self.rule_engine.get_applicable_rules(&metrics.test_id).await?;
        for rule in applicable_rules {
            if let Some(alert) = self.evaluate_rule(&rule, metrics).await? {
                if !self.suppression_manager.is_suppressed(&alert).await? {
                    self.alert_store.store_alert(&alert).await?;
                    // A dispatch failure must not swallow the alert: it is
                    // already stored and is still returned to the caller.
                    if let Err(e) =
                        self.notification_dispatcher.dispatch_alert_notifications(&alert).await
                    {
                        log::warn!(
                            "alert {} was raised but not delivered: {e:?}",
                            alert.alert_id
                        );
                    }
                    if let Some(escalation_policy_id) = &rule.escalation_policy_id {
                        self.escalation_manager
                            .schedule_escalation(&alert, escalation_policy_id)
                            .await?;
                    }
                    self.alert_statistics.record_alert_generated(&alert).await;
                    triggered_alerts.push(alert);
                }
            }
        }
        if !triggered_alerts.is_empty() {
            self.alert_correlator.correlate_alerts(&triggered_alerts).await?;
        }
        Ok(triggered_alerts)
    }
    /// Acknowledge an alert
    pub async fn acknowledge_alert(
        &self,
        alert_id: &str,
        acknowledged_by: &str,
        note: Option<String>,
    ) -> Result<(), AlertError> {
        let alert_opt = self.alert_store.get_alert(alert_id).await?;
        let mut alert = alert_opt.ok_or_else(|| AlertError::NotFound(alert_id.to_string()))?;
        alert.acknowledgment_info = Some(AcknowledgmentInfo {
            acknowledged_by: acknowledged_by.to_string(),
            acknowledged_at: SystemTime::now(),
            acknowledgment_note: note,
            auto_acknowledgment: false,
            timeout: None,
        });
        alert.status = AlertStatus::Acknowledged;
        alert.last_updated = SystemTime::now();
        self.alert_store.update_alert(&alert).await?;
        self.escalation_manager.cancel_escalation(alert_id).await?;
        Ok(())
    }
    /// Resolve an alert
    pub async fn resolve_alert(
        &self,
        alert_id: &str,
        resolved_by: &str,
        resolution_notes: String,
        root_cause: Option<String>,
    ) -> Result<(), AlertError> {
        let alert_opt = self.alert_store.get_alert(alert_id).await?;
        let mut alert = alert_opt.ok_or_else(|| AlertError::NotFound(alert_id.to_string()))?;
        let resolution_time =
            SystemTime::now().duration_since(alert.triggered_at).unwrap_or_default();
        alert.resolution_info = Some(ResolutionInfo {
            resolved_at: SystemTime::now(),
            resolved_by: resolved_by.to_string(),
            resolution_method: ResolutionMethod::Manual,
            resolution_notes,
            resolution_time,
            root_cause,
            preventive_measures: vec![],
            lessons_learned: vec![],
        });
        alert.status = AlertStatus::Resolved;
        alert.last_updated = SystemTime::now();
        self.alert_store.update_alert(&alert).await?;
        self.alert_statistics.record_alert_resolved(&alert, resolution_time).await;
        Ok(())
    }
    /// Create alert rule
    pub async fn create_rule(&self, rule: AlertRule) -> Result<String, AlertError> {
        self.validate_rule(&rule)?;
        self.rule_engine.add_rule(rule.clone()).await?;
        Ok(rule.rule_id)
    }
    /// Update alert rule
    pub async fn update_rule(&self, rule: AlertRule) -> Result<(), AlertError> {
        self.validate_rule(&rule)?;
        self.rule_engine.update_rule(rule).await?;
        Ok(())
    }
    /// Delete alert rule
    pub async fn delete_rule(&self, rule_id: &str) -> Result<(), AlertError> {
        self.rule_engine.delete_rule(rule_id).await?;
        Ok(())
    }
    /// Get alert statistics
    pub async fn get_statistics(&self) -> AlertStatistics {
        (*self.alert_statistics).clone()
    }
    /// Get active alerts
    pub async fn get_active_alerts(
        &self,
        _filter: Option<AlertFilter>,
    ) -> Result<Vec<ActiveAlert>, AlertError> {
        self.alert_store.get_active_alerts().await
    }
    /// Private helper methods
    /// Read the metric named by `selector` out of `metrics`.
    ///
    /// Returns `None` when the stream does not carry that metric, or carries it
    /// as "not observed" -- an unobserved metric must not be compared against a
    /// threshold as though it were zero.
    fn metric_value(selector: &MetricSelector, metrics: &StreamingMetrics) -> Option<f64> {
        match selector.metric_name.as_str() {
            "cpu" | "cpu_percent" | "instantaneous_cpu" => metrics.instantaneous_cpu,
            "memory" | "memory_bytes" | "instantaneous_memory" => {
                Some(metrics.instantaneous_memory as f64)
            },
            "io" | "io_rate" | "instantaneous_io_rate" => metrics.instantaneous_io_rate,
            "network" | "network_rate" | "instantaneous_network_rate" => {
                metrics.instantaneous_network_rate
            },
            "errors" | "live_error_count" => metrics.live_error_count.map(|count| count as f64),
            "warnings" | "live_warning_count" => {
                metrics.live_warning_count.map(|count| count as f64)
            },
            "progress" | "progress_percent" => metrics.progress_percent,
            _ => None,
        }
    }

    /// The numeric threshold a condition compares against.
    ///
    /// `Dynamic` thresholds resolve against a baseline this crate does not
    /// compute, so they yield `None` and the condition is skipped rather than
    /// evaluated against an invented number.
    fn threshold_of(condition: &AlertCondition) -> Option<f64> {
        match &condition.threshold_value {
            ThresholdValue::Absolute(value) | ThresholdValue::Percentage(value) => Some(*value),
            ThresholdValue::Dynamic(_) => None,
        }
    }

    /// Whether `value` satisfies `condition`.
    fn condition_holds(condition: &AlertCondition, value: f64, threshold: f64) -> bool {
        match condition.operator {
            ComparisonOperator::GreaterThan => value > threshold,
            ComparisonOperator::LessThan => value < threshold,
            ComparisonOperator::Equal => (value - threshold).abs() < f64::EPSILON,
            ComparisonOperator::NotEqual => (value - threshold).abs() >= f64::EPSILON,
            ComparisonOperator::GreaterThanOrEqual => value >= threshold,
            ComparisonOperator::LessThanOrEqual => value <= threshold,
        }
    }

    /// Evaluate `rule` against one sample, returning an alert when it fires.
    ///
    /// 0.2.1: the body of this method was `for _condition in &rule.conditions {
    /// let _evaluation_result = false; if false { .. } }`. Every condition was
    /// discarded unevaluated and the alert-construction block was statically
    /// unreachable, so `AlertManager::process_metrics` could never return a
    /// single alert no matter what arrived on the stream -- the whole alerting
    /// subsystem was inert while reporting success. Conditions are really
    /// compared now: the metric is read out of the sample, the threshold out of
    /// the condition, and the configured operator decides.
    ///
    /// Only `AlertConditionType::Threshold` conditions can be decided here.
    /// Anomaly/Trend/Pattern/Composite/Custom conditions need history or an
    /// expression evaluator this crate does not have, so they are skipped
    /// rather than silently treated as not-firing-because-checked.
    ///
    /// # Errors
    ///
    /// Infallible today; the `Result` matches the rest of the alerting API.
    async fn evaluate_rule(
        &self,
        rule: &AlertRule,
        metrics: &StreamingMetrics,
    ) -> Result<Option<ActiveAlert>, AlertError> {
        for condition in &rule.conditions {
            if !matches!(condition.condition_type, AlertConditionType::Threshold) {
                continue;
            }
            let Some(value) = Self::metric_value(&condition.metric_selector, metrics) else {
                continue;
            };
            let Some(threshold) = Self::threshold_of(condition) else {
                continue;
            };
            if Self::condition_holds(condition, value, threshold) {
                let alert = ActiveAlert {
                    alert_id: Uuid::new_v4().to_string(),
                    rule_id: rule.rule_id.clone(),
                    test_id: metrics.test_id.clone(),
                    alert_type: AlertType::Threshold,
                    severity: rule.severity,
                    status: AlertStatus::Active,
                    title: format!("Alert: {}", rule.rule_name),
                    description: format!(
                        "{} (condition {}: {} = {value} vs threshold {threshold})",
                        rule.description,
                        condition.condition_id,
                        condition.metric_selector.metric_name
                    ),
                    triggered_at: SystemTime::now(),
                    last_updated: SystemTime::now(),
                    acknowledgment_info: None,
                    escalation_info: None,
                    suppression_info: None,
                    context_data: AlertContext {
                        test_execution_context: None,
                        system_state: SystemState::sample(metrics),
                        environmental_factors: HashMap::new(),
                        recent_changes: vec![],
                        related_metrics: {
                            // The metric that actually tripped the rule, so a
                            // reader can see the observation the alert rests on.
                            let mut related = HashMap::new();
                            related.insert(
                                condition.metric_selector.metric_name.clone(),
                                MetricValue::Float(value),
                            );
                            related
                        },
                        dependency_status: HashMap::new(),
                    },
                    // Nothing in this crate models cost, user counts, downtime
                    // or business impact, so those fields carry explicit
                    // "not assessed" markers and zeroes rather than a
                    // confident-looking "Medium"/"Low" verdict nobody computed.
                    impact_assessment: ImpactAssessment {
                        severity: format!("{:?}", rule.severity),
                        affected_systems: vec![metrics.test_id.clone()],
                        estimated_cost: 0.0,
                        impact_level: "not assessed".to_string(),
                        affected_users: 0,
                        business_impact: "not assessed".to_string(),
                        estimated_downtime: Duration::from_secs(0),
                        financial_impact: 0.0,
                    },
                    recommended_actions: vec![],
                    alert_fingerprint: format!(
                        "{}:{}:{}",
                        rule.rule_id, metrics.test_id, condition.condition_id
                    ),
                    correlation_ids: vec![],
                    resolution_info: None,
                };
                return Ok(Some(alert));
            }
        }
        Ok(None)
    }
    pub(crate) fn validate_rule(&self, rule: &AlertRule) -> Result<(), AlertError> {
        if rule.rule_id.is_empty() {
            return Err(AlertError::ValidationError {
                field: "rule_id".to_string(),
                reason: "Rule ID cannot be empty".to_string(),
            });
        }
        if rule.conditions.is_empty() {
            return Err(AlertError::ValidationError {
                field: "conditions".to_string(),
                reason: "Rule must have at least one condition".to_string(),
            });
        }
        Ok(())
    }
}
/// Holds pluggable threshold evaluators.
///
/// No evaluator is registered by default, and none is implemented in this
/// crate. [`AlertRuleEngine`] evaluates threshold conditions inline against a
/// live sample; this type exists for callers that supply their own evaluator.
pub struct ThresholdMonitor {
    pub(super) threshold_evaluators: Vec<Box<dyn ThresholdEvaluator + Send + Sync>>,
    /// Conditions evaluated so far.
    pub(super) evaluations: Arc<Mutex<u64>>,
}

impl ThresholdMonitor {
    /// Build a monitor with no evaluators registered.
    pub fn new() -> Self {
        Self {
            threshold_evaluators: Vec::new(),
            evaluations: Arc::new(Mutex::new(0)),
        }
    }

    /// Register a threshold evaluator.
    pub fn register_evaluator(&mut self, evaluator: Box<dyn ThresholdEvaluator + Send + Sync>) {
        self.threshold_evaluators.push(evaluator);
    }

    /// Number of registered evaluators.
    pub fn evaluator_count(&self) -> usize {
        self.threshold_evaluators.len()
    }

    /// Conditions this monitor has been asked to evaluate.
    pub async fn evaluation_count(&self) -> u64 {
        *self.evaluations.lock().await
    }

    /// Evaluate `alert_condition` against `metrics`.
    ///
    /// # Errors
    ///
    /// Returns [`AlertError::ThresholdEvaluationError`] when no evaluator is
    /// registered. 0.2.1: this used to return `Ok(false)` in that case, which a
    /// caller would read as "the threshold was checked and did not trip"
    /// rather than "nothing checked it".
    pub async fn evaluate_condition(
        &self,
        alert_condition: &AlertCondition,
        _metrics: &PerformanceMetrics,
    ) -> Result<bool, AlertError> {
        *self.evaluations.lock().await += 1;
        if self.threshold_evaluators.is_empty() {
            return Err(AlertError::ThresholdEvaluationError {
                condition_id: alert_condition.condition_id.clone(),
                reason: "no ThresholdEvaluator is registered, so this condition was not evaluated"
                    .to_string(),
            });
        }
        Err(AlertError::ThresholdEvaluationError {
            condition_id: alert_condition.condition_id.clone(),
            reason: "registered ThresholdEvaluators take historical DataPoints, which this \
                     PerformanceMetrics-based entry point does not carry"
                .to_string(),
        })
    }
}
/// Individual escalation level
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationLevel {
    pub level: u32,
    pub level_name: String,
    pub escalation_delay: Duration,
    pub notification_targets: Vec<NotificationTarget>,
    pub required_acknowledgments: u32,
    pub escalation_criteria: Vec<EscalationCriteria>,
    pub automatic_actions: Vec<AutomaticAction>,
}
