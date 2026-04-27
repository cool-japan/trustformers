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

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::time::Duration;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn test_alert_rule_metadata_default() {
        let m = AlertRuleMetadata::default();
        assert!(m.rule_id.is_empty());
        assert!(m.rule_name.is_empty());
        assert!(m.description.is_empty());
    }

    #[test]
    fn test_alert_rule_metadata_with_values() {
        let now = chrono::Utc::now();
        let m = AlertRuleMetadata {
            rule_id: "rule-001".to_string(),
            rule_name: "CPU Threshold Alert".to_string(),
            description: "Alert when CPU exceeds 80%".to_string(),
            created_at: now,
        };
        assert_eq!(m.rule_id, "rule-001");
        assert_eq!(m.rule_name, "CPU Threshold Alert");
        assert!(!m.description.is_empty());
    }

    #[test]
    fn test_escalation_executor_construction() {
        let ee = EscalationExecutor {
            executor_id: "exec-42".to_string(),
            actions_executed: 10,
        };
        assert_eq!(ee.executor_id, "exec-42");
        assert_eq!(ee.actions_executed, 10);
    }

    #[test]
    fn test_escalation_executor_zero_actions() {
        let ee = EscalationExecutor {
            executor_id: "exec-0".to_string(),
            actions_executed: 0,
        };
        assert_eq!(ee.actions_executed, 0);
    }

    #[test]
    fn test_escalation_scheduler_construction() {
        let es = EscalationScheduler {
            schedule_interval: Duration::from_secs(300),
            max_retries: 3,
        };
        assert_eq!(es.schedule_interval, Duration::from_secs(300));
        assert_eq!(es.max_retries, 3);
    }

    #[test]
    fn test_escalation_event_construction() {
        let now = chrono::Utc::now();
        let ev = EscalationEvent {
            event_id: "evt-123".to_string(),
            timestamp: now,
            level: 2,
        };
        assert_eq!(ev.event_id, "evt-123");
        assert_eq!(ev.level, 2);
    }

    #[test]
    fn test_escalation_metrics_construction() {
        let em = EscalationMetrics {
            total_escalations: 5,
            avg_response_time: Duration::from_secs(120),
        };
        assert_eq!(em.total_escalations, 5);
        assert_eq!(em.avg_response_time, Duration::from_secs(120));
    }

    #[test]
    fn test_escalation_condition_construction() {
        let ec = EscalationCondition {
            condition: "threshold_exceeded".to_string(),
            threshold: 0.9,
        };
        assert_eq!(ec.condition, "threshold_exceeded");
        assert!((ec.threshold - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn test_escalation_criteria_construction() {
        let ec = EscalationCriteria {
            criteria_type: "time_based".to_string(),
            threshold: 0.95,
            duration: Duration::from_secs(60),
        };
        assert_eq!(ec.criteria_type, "time_based");
        assert!(ec.threshold > 0.0);
        assert!(ec.duration > Duration::from_secs(0));
    }

    #[test]
    fn test_historical_alert_construction() {
        let now = chrono::Utc::now();
        let ha = HistoricalAlert {
            alert_id: "alert-001".to_string(),
            timestamp: now,
            status: "resolved".to_string(),
        };
        assert_eq!(ha.alert_id, "alert-001");
        assert_eq!(ha.status, "resolved");
    }

    #[test]
    fn test_threshold_value_default() {
        let tv = super::super::enums::ThresholdValue::default();
        if let super::super::enums::ThresholdValue::Absolute(v) = tv {
            assert_eq!(v, 0.0);
        } else {
            panic!("expected Absolute default");
        }
    }

    #[test]
    fn test_alert_info_construction() {
        let ai = AlertInfo {
            alert_id: "a-1".to_string(),
            alert_type: "performance".to_string(),
            severity: "warning".to_string(),
            message: "CPU usage high".to_string(),
        };
        assert_eq!(ai.alert_id, "a-1");
        assert_eq!(ai.alert_type, "performance");
        assert_eq!(ai.severity, "warning");
    }

    #[test]
    fn test_alert_index_construction() {
        let now = chrono::Utc::now();
        let ai = AlertIndex {
            index_name: "primary_index".to_string(),
            indexed_fields: vec!["timestamp".to_string(), "severity".to_string()],
            last_updated: now,
        };
        assert_eq!(ai.index_name, "primary_index");
        assert_eq!(ai.indexed_fields.len(), 2);
    }

    #[test]
    fn test_alert_retention_policy() {
        let arp = AlertRetentionPolicy {
            retention_days: 30,
            archive_after_days: 7,
            auto_cleanup: true,
        };
        assert_eq!(arp.retention_days, 30);
        assert_eq!(arp.archive_after_days, 7);
        assert!(arp.auto_cleanup);
        assert!(arp.archive_after_days < arp.retention_days);
    }

    #[test]
    fn test_alert_query_empty_filters() {
        let aq = AlertQuery {
            filters: HashMap::new(),
            time_range: None,
            limit: None,
        };
        assert!(aq.filters.is_empty());
        assert!(aq.time_range.is_none());
        assert!(aq.limit.is_none());
    }

    #[test]
    fn test_alert_query_with_filters_and_limit() {
        let mut filters = HashMap::new();
        filters.insert("severity".to_string(), "critical".to_string());
        let aq = AlertQuery {
            filters,
            time_range: None,
            limit: Some(100),
        };
        assert_eq!(aq.filters.len(), 1);
        assert_eq!(aq.limit, Some(100));
    }

    #[test]
    fn test_alert_storage_statistics_defaults() {
        let stats = AlertStorageStatistics {
            total_alerts: 500,
            active_alerts: 10,
            storage_size_bytes: 1024 * 512,
        };
        assert!(stats.total_alerts >= stats.active_alerts);
        assert!(stats.storage_size_bytes > 0);
    }

    #[test]
    fn test_alert_filter_construction() {
        let af = AlertFilter {
            filter_expression: "severity == 'critical'".to_string(),
            include_resolved: false,
            severity_filter: Some("critical".to_string()),
        };
        assert!(!af.filter_expression.is_empty());
        assert!(!af.include_resolved);
        assert!(af.severity_filter.is_some());
    }

    #[test]
    fn test_alert_filter_include_resolved() {
        let af = AlertFilter {
            filter_expression: String::new(),
            include_resolved: true,
            severity_filter: None,
        };
        assert!(af.include_resolved);
        assert!(af.severity_filter.is_none());
    }

    #[test]
    fn test_alert_history_entry_construction() {
        let now = chrono::Utc::now();
        let entry = AlertHistoryEntry {
            timestamp: now,
            alert_type: "resource".to_string(),
            severity: "warning".to_string(),
        };
        assert_eq!(entry.alert_type, "resource");
        assert_eq!(entry.severity, "warning");
    }

    #[test]
    fn test_multiple_escalation_events_ordering() {
        let mut lcg = Lcg::new(99);
        let events: Vec<EscalationEvent> = (0..5)
            .map(|i| EscalationEvent {
                event_id: format!("evt-{}", i),
                timestamp: chrono::Utc::now(),
                level: (lcg.next_f32() * 5.0) as u8,
            })
            .collect();
        assert_eq!(events.len(), 5);
        for e in &events {
            assert!(!e.event_id.is_empty());
        }
    }

    #[test]
    fn test_alert_query_time_range() {
        let start = chrono::Utc::now();
        let end = chrono::Utc::now();
        let aq = AlertQuery {
            filters: HashMap::new(),
            time_range: Some((start, end)),
            limit: Some(50),
        };
        assert!(aq.time_range.is_some());
        assert_eq!(aq.limit, Some(50));
    }
}
