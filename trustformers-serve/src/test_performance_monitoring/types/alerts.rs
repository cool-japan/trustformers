//! Alerts Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::enums::ThresholdValue;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertRuleMetadata {
    pub rule_id: String,
    pub rule_name: String,
    pub description: String,
    pub created_at: DateTime<Utc>,
}

impl Default for AlertRuleMetadata {
    fn default() -> Self {
        Self {
            rule_id: String::new(),
            rule_name: String::new(),
            description: String::new(),
            created_at: Utc::now(),
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EscalationExecutor {
    pub executor_id: String,
    pub actions_executed: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EscalationScheduler {
    pub schedule_interval: Duration,
    pub max_retries: u32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EscalationEvent {
    pub event_id: String,
    pub timestamp: DateTime<Utc>,
    pub level: u8,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct EscalationMetrics {
    pub total_escalations: u64,
    pub avg_response_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCondition {
    pub condition: String,
    pub threshold: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EscalationCriteria {
    pub criteria_type: String,
    pub threshold: f64,
    pub duration: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HistoricalAlert {
    pub alert_id: String,
    pub timestamp: DateTime<Utc>,
    pub status: String,
}

impl Default for ThresholdValue {
    fn default() -> Self {
        Self::Absolute(0.0)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertInfo {
    pub alert_id: String,
    pub alert_type: String,
    pub severity: String,
    pub message: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertIndex {
    pub index_name: String,
    pub indexed_fields: Vec<String>,
    pub last_updated: DateTime<Utc>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertRetentionPolicy {
    pub retention_days: u32,
    pub archive_after_days: u32,
    pub auto_cleanup: bool,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertQuery {
    pub filters: std::collections::HashMap<String, String>,
    pub time_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    pub limit: Option<usize>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertStorageStatistics {
    pub total_alerts: u64,
    pub active_alerts: u64,
    pub storage_size_bytes: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct AlertFilter {
    pub filter_expression: String,
    pub include_resolved: bool,
    pub severity_filter: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertHistoryEntry {
    pub timestamp: DateTime<Utc>,
    pub alert_type: String,
    pub severity: String,
}
