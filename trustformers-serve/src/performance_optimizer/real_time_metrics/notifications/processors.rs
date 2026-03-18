//! # Alert Processors for Notification System
//!
//! This module contains alert processor implementations that convert alerts
//! into appropriate notifications based on alert type and severity.

use super::types::*;
use crate::performance_optimizer::real_time_metrics::types::*;
use anyhow::Result;
use chrono::{Duration as ChronoDuration, Utc};
use std::collections::HashMap;
use std::fmt::Debug;

/// Enhanced trait for alert processors with async support
#[async_trait::async_trait]
pub trait AlertProcessor: Debug {
    /// Process an alert and generate notifications
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle this alert
    fn can_process(&self, alert: &AlertEvent) -> bool;

    /// Get processor priority (higher = processed first)
    fn priority(&self) -> u8 {
        50 // Default priority
    }
}

/// Default alert processor for general alerts
#[derive(Debug)]

pub struct DefaultAlertProcessor {
    name: String,
}

impl DefaultAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "default".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for DefaultAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["log".to_string()],
            recipients: vec!["system".to_string()],
            subject: format!("Alert: {}", alert.threshold.name),
            content: alert.message.clone(),
            priority: match alert.severity {
                SeverityLevel::Critical => NotificationPriority::Critical,
                SeverityLevel::High => NotificationPriority::High,
                SeverityLevel::Medium => NotificationPriority::Normal,
                _ => NotificationPriority::Low,
            },
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            created_at: Utc::now(),
            deadline: None,
            template: Some("default_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("threshold_name".to_string(), alert.threshold.name.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars
            },
            metadata: HashMap::new(),
            tags: vec!["alert".to_string(), "default".to_string()],
            escalation_policy: None,
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, _alert: &AlertEvent) -> bool {
        true // Default processor handles all alerts
    }

    fn priority(&self) -> u8 {
        10 // Low priority - runs last
    }
}

/// Performance-specific alert processor
#[derive(Debug)]

pub struct PerformanceAlertProcessor {
    name: String,
}

impl PerformanceAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "performance".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for PerformanceAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("perf_notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["email".to_string(), "slack".to_string()],
            recipients: vec!["performance-team@company.com".to_string()],
            subject: format!("Performance Alert: {}", alert.threshold.name),
            content: format!("Performance issue detected: {}", alert.message),
            priority: NotificationPriority::High,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(15)),
            template: Some("performance_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("metric".to_string(), alert.threshold.metric.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars.insert("impact".to_string(), "High".to_string());
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("team".to_string(), "performance".to_string());
                metadata.insert("escalation_required".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "performance".to_string(),
                "urgent".to_string(),
            ],
            escalation_policy: Some("performance_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms" | "response_time"
        )
    }

    fn priority(&self) -> u8 {
        80 // High priority
    }
}

/// Resource-specific alert processor
#[derive(Debug)]

pub struct ResourceAlertProcessor {
    name: String,
}

impl ResourceAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "resource".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for ResourceAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("resource_notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["webhook".to_string(), "slack".to_string()],
            recipients: vec!["ops-team".to_string()],
            subject: format!("Resource Alert: {}", alert.threshold.name),
            content: format!("Resource issue detected: {}", alert.message),
            priority: NotificationPriority::High,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(10)),
            template: Some("resource_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("resource_type".to_string(), alert.threshold.metric.clone());
                vars.insert(
                    "utilization".to_string(),
                    format!("{}%", (alert.current_value * 100.0) as u32),
                );
                vars.insert(
                    "threshold".to_string(),
                    format!("{}%", (alert.threshold_value * 100.0) as u32),
                );
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("team".to_string(), "ops".to_string());
                metadata.insert("auto_scale".to_string(), "enabled".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "resource".to_string(),
                "ops".to_string(),
            ],
            escalation_policy: Some("ops_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization" | "disk_usage"
        )
    }

    fn priority(&self) -> u8 {
        70 // High priority
    }
}

/// Critical alert processor for emergency situations
#[derive(Debug)]

pub struct CriticalAlertProcessor {
    name: String,
}

impl CriticalAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "critical".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for CriticalAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        // Critical alerts generate multiple notifications across all channels
        let mut notifications = Vec::new();

        // Immediate notification to all channels
        let immediate_notification = Notification {
            id: format!("critical_immediate_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec![
                "email".to_string(),
                "slack".to_string(),
                "webhook".to_string(),
                "sms".to_string(),
            ],
            recipients: vec![
                "oncall@company.com".to_string(),
                "manager@company.com".to_string(),
                "emergency-contact".to_string(),
            ],
            subject: format!("🚨 CRITICAL ALERT: {}", alert.threshold.name),
            content: format!(
                "CRITICAL SYSTEM ISSUE: {}\n\nImmediate attention required!",
                alert.message
            ),
            priority: NotificationPriority::Emergency,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::ExactlyOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(5)),
            template: Some("critical_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("severity".to_string(), "CRITICAL".to_string());
                vars.insert("metric".to_string(), alert.threshold.metric.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars.insert(
                    "alert_time".to_string(),
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                );
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("priority".to_string(), "emergency".to_string());
                metadata.insert("bypass_rate_limit".to_string(), "true".to_string());
                metadata.insert("require_acknowledgment".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "critical".to_string(),
                "emergency".to_string(),
            ],
            escalation_policy: Some("critical_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        notifications.push(immediate_notification);

        // Follow-up notification for PagerDuty
        let pagerduty_notification = Notification {
            id: format!("critical_pagerduty_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["pagerduty".to_string()],
            recipients: vec!["incident-response".to_string()],
            subject: format!("Critical System Alert - {}", alert.threshold.name),
            content: format!(
                "Critical alert requires immediate incident response: {}",
                alert.message
            ),
            priority: NotificationPriority::Emergency,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::ExactlyOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(2)),
            template: Some("pagerduty_critical".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("incident_key".to_string(), alert.alert_id.clone());
                vars.insert(
                    "service_key".to_string(),
                    "trustformers-critical".to_string(),
                );
                vars.insert("description".to_string(), alert.message.clone());
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("incident_type".to_string(), "critical".to_string());
                metadata.insert("auto_escalate".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "critical".to_string(),
                "pagerduty".to_string(),
            ],
            escalation_policy: Some("incident_response".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        notifications.push(pagerduty_notification);

        Ok(notifications)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(alert.severity, SeverityLevel::Critical)
    }

    fn priority(&self) -> u8 {
        100 // Highest priority - runs first
    }
}
