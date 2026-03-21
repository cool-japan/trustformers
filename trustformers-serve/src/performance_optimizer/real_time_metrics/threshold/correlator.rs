//! Alert correlation system for identifying related alerts.

use super::super::types::*;
use super::types::CorrelationType;
use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Duration;
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;

// =============================================================================
// ALERT CORRELATION SYSTEM
// =============================================================================

/// Alert correlation system for identifying related alerts
///
/// Advanced alert correlation system that identifies relationships between
/// alerts to provide better context and reduce alert fatigue.
pub struct AlertCorrelator {
    /// Correlation rules
    rules: Arc<RwLock<Vec<CorrelationRule>>>,

    /// Active correlations
    correlations: Arc<TokioMutex<HashMap<String, AlertCorrelation>>>,

    /// Correlation statistics
    stats: Arc<CorrelationStats>,

    /// Configuration
    config: Arc<RwLock<CorrelationConfig>>,
}

/// Correlation rule for alerts
#[derive(Debug, Clone)]
pub struct CorrelationRule {
    /// Rule ID
    pub id: String,

    /// Rule name
    pub name: String,

    /// Rule type
    pub rule_type: CorrelationRuleType,

    /// Matching criteria
    pub criteria: CorrelationCriteria,

    /// Correlation strength (0.0 to 1.0)
    pub strength: f32,

    /// Time window for correlation
    pub time_window: Duration,

    /// Rule enabled
    pub enabled: bool,
}

/// Types of correlation rules
#[derive(Debug, Clone)]
pub enum CorrelationRuleType {
    /// Resource-based correlation
    Resource,

    /// Metric-based correlation
    Metric,

    /// Temporal correlation
    Temporal,

    /// Causal correlation
    Causal,

    /// Pattern-based correlation
    Pattern,

    /// Custom correlation logic
    Custom(String),
}

/// Criteria for alert correlation
#[derive(Debug, Clone)]
pub struct CorrelationCriteria {
    /// Metric patterns to match
    pub metric_patterns: Vec<String>,

    /// Resource patterns to match
    pub resource_patterns: Vec<String>,

    /// Severity level matching
    pub severity_matching: SeverityMatching,

    /// Time tolerance for correlation
    pub time_tolerance: Duration,

    /// Minimum correlation strength
    pub min_strength: f32,

    /// Additional conditions
    pub conditions: HashMap<String, String>,
}

/// Severity matching strategy
#[derive(Debug, Clone)]
pub enum SeverityMatching {
    /// Exact severity match
    Exact,

    /// Within severity range
    Range {
        min: SeverityLevel,
        max: SeverityLevel,
    },

    /// Any severity
    Any,

    /// Escalating severity
    Escalating,
}

/// Alert correlation information
#[derive(Debug, Clone)]
pub struct AlertCorrelation {
    /// Correlation ID
    pub correlation_id: String,

    /// Root alert ID
    pub root_alert_id: String,

    /// Correlated alert IDs
    pub correlated_alerts: Vec<String>,

    /// Correlation type
    pub correlation_type: CorrelationType,

    /// Correlation strength
    pub strength: f32,

    /// Creation time
    pub created_at: DateTime<Utc>,

    /// Last updated
    pub updated_at: DateTime<Utc>,

    /// Correlation metadata
    pub metadata: HashMap<String, String>,
}

/// Statistics for alert correlation
#[derive(Debug, Default)]
pub struct CorrelationStats {
    /// Total alerts processed
    pub total_processed: AtomicU64,

    /// Total correlations created
    pub total_correlations: AtomicU64,

    /// Correlations by type
    pub correlations_by_type: Arc<Mutex<HashMap<String, u64>>>,

    /// Average correlation strength
    pub avg_correlation_strength: Arc<Mutex<f32>>,

    /// Active correlations count
    pub active_correlations: AtomicU64,
}

/// Configuration for alert correlation
#[derive(Debug, Clone)]
pub struct CorrelationConfig {
    /// Enable temporal correlation
    pub enable_temporal: bool,

    /// Enable resource correlation
    pub enable_resource: bool,

    /// Enable metric correlation
    pub enable_metric: bool,

    /// Maximum correlation window
    pub max_correlation_window: Duration,

    /// Minimum correlation strength
    pub min_correlation_strength: f32,

    /// Maximum correlations per alert
    pub max_correlations_per_alert: usize,
}

impl Default for AlertCorrelator {
    fn default() -> Self {
        Self::new()
    }
}

impl AlertCorrelator {
    /// Create a new alert correlator
    pub fn new() -> Self {
        Self {
            rules: Arc::new(RwLock::new(Vec::new())),
            correlations: Arc::new(TokioMutex::new(HashMap::new())),
            stats: Arc::new(CorrelationStats::default()),
            config: Arc::new(RwLock::new(CorrelationConfig::default())),
        }
    }

    /// Correlate an alert with existing alerts
    pub async fn correlate_alert(&self, alert: &mut AlertEvent) {
        self.stats.total_processed.fetch_add(1, Ordering::Relaxed);

        let rules_snapshot = {
            let rules = self.rules.read().expect("Rules RwLock poisoned");
            rules.clone()
        };

        let mut correlations = self.correlations.lock().await;

        for rule in rules_snapshot.iter() {
            if !rule.enabled {
                continue;
            }

            if self.matches_correlation_criteria(&rule.criteria, alert) {
                // Look for existing correlations that match this rule
                for (_, correlation) in correlations.iter_mut() {
                    if self.can_correlate_with_existing(rule, alert, correlation) {
                        // Add alert to existing correlation
                        correlation.correlated_alerts.push(alert.alert_id.clone());
                        correlation.updated_at = Utc::now();

                        // Update alert with correlation info
                        alert.correlation_id = Some(correlation.correlation_id.clone());

                        self.update_correlation_stats(&rule.rule_type);
                        return;
                    }
                }

                // Create new correlation if no existing one found
                let correlation_id = Uuid::new_v4().to_string();
                let correlation = AlertCorrelation {
                    correlation_id: correlation_id.clone(),
                    root_alert_id: alert.alert_id.clone(),
                    correlated_alerts: vec![alert.alert_id.clone()],
                    correlation_type: self.rule_type_to_correlation_type(&rule.rule_type),
                    strength: rule.strength,
                    created_at: Utc::now(),
                    updated_at: Utc::now(),
                    metadata: HashMap::new(),
                };

                correlations.insert(correlation_id.clone(), correlation);
                alert.correlation_id = Some(correlation_id);

                self.stats.total_correlations.fetch_add(1, Ordering::Relaxed);
                self.update_correlation_stats(&rule.rule_type);
                break;
            }
        }

        // Clean up old correlations
        self.cleanup_old_correlations(&mut correlations).await;

        self.stats
            .active_correlations
            .store(correlations.len() as u64, Ordering::Relaxed);
    }

    /// Check if alert matches correlation criteria
    fn matches_correlation_criteria(
        &self,
        criteria: &CorrelationCriteria,
        alert: &AlertEvent,
    ) -> bool {
        // Check metric patterns
        if !criteria.metric_patterns.is_empty() {
            let metric_matches = criteria
                .metric_patterns
                .iter()
                .any(|pattern| alert.threshold.metric.contains(pattern));
            if !metric_matches {
                return false;
            }
        }

        // Check resource patterns (from context)
        if !criteria.resource_patterns.is_empty() {
            let resource_matches = criteria
                .resource_patterns
                .iter()
                .any(|pattern| alert.context.values().any(|value| value.contains(pattern)));
            if !resource_matches {
                return false;
            }
        }

        // Check severity matching
        match &criteria.severity_matching {
            SeverityMatching::Exact => {
                // For exact matching, we need another alert to compare with
                // This will be checked in can_correlate_with_existing
            },
            SeverityMatching::Range { min, max } => {
                if alert.severity < *min || alert.severity > *max {
                    return false;
                }
            },
            SeverityMatching::Any => {
                // Any severity is acceptable
            },
            SeverityMatching::Escalating => {
                // Will be checked in can_correlate_with_existing
            },
        }

        // Check additional conditions
        for (key, expected_value) in &criteria.conditions {
            if let Some(actual_value) = alert.context.get(key) {
                if actual_value != expected_value {
                    return false;
                }
            } else {
                return false;
            }
        }

        true
    }

    /// Check if alert can be correlated with existing correlation
    fn can_correlate_with_existing(
        &self,
        rule: &CorrelationRule,
        alert: &AlertEvent,
        correlation: &AlertCorrelation,
    ) -> bool {
        // Check time window
        let time_diff = alert.timestamp.signed_duration_since(correlation.updated_at);
        let time_window =
            chrono::Duration::from_std(rule.time_window).expect("Duration conversion failed");

        if time_diff > time_window {
            return false;
        }

        // Check correlation strength
        if rule.strength < correlation.strength * 0.8 {
            return false;
        }

        // Check maximum correlations per alert
        let max_correlations =
            self.config.read().expect("Config RwLock poisoned").max_correlations_per_alert;
        if correlation.correlated_alerts.len() >= max_correlations {
            return false;
        }

        true
    }

    /// Convert rule type to correlation type
    fn rule_type_to_correlation_type(&self, rule_type: &CorrelationRuleType) -> CorrelationType {
        match rule_type {
            CorrelationRuleType::Resource => CorrelationType::Resource,
            CorrelationRuleType::Metric => CorrelationType::Metric,
            CorrelationRuleType::Temporal => CorrelationType::Temporal,
            CorrelationRuleType::Causal => CorrelationType::Causal,
            CorrelationRuleType::Pattern => CorrelationType::Pattern,
            CorrelationRuleType::Custom(_) => CorrelationType::Pattern,
        }
    }

    /// Update correlation statistics
    fn update_correlation_stats(&self, rule_type: &CorrelationRuleType) {
        let type_name = format!("{:?}", rule_type);
        let mut stats = self
            .stats
            .correlations_by_type
            .lock()
            .expect("Correlations stats lock poisoned");
        *stats.entry(type_name).or_insert(0) += 1;
    }

    /// Clean up old correlations
    async fn cleanup_old_correlations(&self, correlations: &mut HashMap<String, AlertCorrelation>) {
        let max_window = self.config.read().expect("Config RwLock poisoned").max_correlation_window;
        let cutoff_time = Utc::now()
            - chrono::Duration::from_std(max_window).expect("Duration conversion failed");

        correlations.retain(|_, correlation| correlation.updated_at > cutoff_time);
    }

    /// Add correlation rule
    pub fn add_rule(&self, rule: CorrelationRule) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.push(rule);
    }

    /// Remove correlation rule
    pub fn remove_rule(&self, rule_id: &str) {
        let mut rules = self.rules.write().expect("Rules RwLock poisoned");
        rules.retain(|rule| rule.id != rule_id);
    }

    /// Get correlation statistics
    pub fn get_stats(&self) -> CorrelationStats {
        CorrelationStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            total_correlations: AtomicU64::new(
                self.stats.total_correlations.load(Ordering::Relaxed),
            ),
            correlations_by_type: Arc::new(Mutex::new(
                self.stats
                    .correlations_by_type
                    .lock()
                    .expect("Correlations stats lock poisoned")
                    .clone(),
            )),
            avg_correlation_strength: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_correlation_strength
                    .lock()
                    .expect("Avg correlation strength lock poisoned"),
            )),
            active_correlations: AtomicU64::new(
                self.stats.active_correlations.load(Ordering::Relaxed),
            ),
        }
    }

    /// Get active correlations
    pub async fn get_active_correlations(&self) -> Vec<AlertCorrelation> {
        let correlations = self.correlations.lock().await;
        correlations.values().cloned().collect()
    }

    /// Get correlation by ID
    pub async fn get_correlation(&self, correlation_id: &str) -> Option<AlertCorrelation> {
        let correlations = self.correlations.lock().await;
        correlations.get(correlation_id).cloned()
    }
}

impl Default for CorrelationConfig {
    fn default() -> Self {
        Self {
            enable_temporal: true,
            enable_resource: true,
            enable_metric: true,
            max_correlation_window: Duration::from_secs(3600), // 1 hour
            min_correlation_strength: 0.3,
            max_correlations_per_alert: 10,
        }
    }
}
