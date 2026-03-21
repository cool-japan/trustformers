//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::analytics::DataPoint;
use super::super::types::{
    AlertQuery, AlertStorageStatistics, DeliveryCapabilities, DeliveryResult, EvaluationCost,
    NotificationChannelType, NotificationError, RateLimits, ThresholdEvaluatorType,
};
use super::types::{
    ActiveAlert, AlertError, AlertRule, NotificationRequest, ThresholdConfig,
    ThresholdEvaluationResult,
};
use std::fmt::Debug;

/// Threshold evaluation trait
pub trait ThresholdEvaluator {
    fn evaluate_threshold(
        &self,
        rule: &AlertRule,
        current_value: f64,
        historical_data: &[DataPoint],
    ) -> Result<ThresholdEvaluationResult, AlertError>;
    fn get_evaluator_type(&self) -> ThresholdEvaluatorType;
    fn is_applicable(&self, threshold_config: &ThresholdConfig) -> bool;
    fn get_evaluation_cost(&self) -> EvaluationCost;
}
/// Notification channel trait implementation
pub trait NotificationChannel {
    fn send_notification(
        &self,
        request: &NotificationRequest,
    ) -> Result<DeliveryResult, NotificationError>;
    fn get_channel_type(&self) -> NotificationChannelType;
    fn get_channel_name(&self) -> &str;
    fn is_available(&self) -> bool;
    fn get_delivery_capabilities(&self) -> DeliveryCapabilities;
    fn supports_rich_content(&self) -> bool;
    fn get_rate_limits(&self) -> RateLimits;
}
/// Alert persistence trait
pub trait AlertPersistence: Debug {
    fn store_alert(&self, alert: &ActiveAlert) -> Result<(), AlertError>;
    fn update_alert(&self, alert: &ActiveAlert) -> Result<(), AlertError>;
    fn delete_alert(&self, alert_id: &str) -> Result<(), AlertError>;
    fn query_alerts(&self, query: &AlertQuery) -> Result<Vec<ActiveAlert>, AlertError>;
    fn get_alert_statistics(&self) -> AlertStorageStatistics;
}
#[cfg(test)]
mod tests {

    use super::super::types::*;
    use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
    use crate::test_performance_monitoring::types::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    use std::time::{Duration, SystemTime};
    #[test]
    fn test_alert_manager_creation() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);
        assert_eq!(
            manager.alert_statistics.total_alerts_generated.load(Ordering::Relaxed),
            0
        );
    }
    #[test]
    fn test_alert_rule_validation() {
        let config = AlertConfig::default();
        let manager = AlertManager::new(config);
        let valid_rule = AlertRule {
            rule_id: "test-rule".to_string(),
            rule_name: "Test Rule".to_string(),
            description: "Test alert rule".to_string(),
            category: AlertCategory::Performance,
            severity: SeverityLevel::High,
            conditions: vec![AlertCondition {
                condition_id: "condition-1".to_string(),
                condition_type: AlertConditionType::Threshold,
                metric_selector: MetricSelector {
                    metric_name: "execution_time".to_string(),
                    test_id_pattern: Some("test-*".to_string()),
                    tag_filters: HashMap::new(),
                    aggregation_scope: AggregationScope::Test,
                    time_window: Duration::from_secs(5 * 60),
                },
                operator: ComparisonOperator::GreaterThan,
                threshold_value: ThresholdValue::Absolute(10.0),
                duration_requirement: Some(Duration::from_secs(60)),
                aggregation_method: Some(AggregationMethod::Average),
                condition_weight: 1.0,
                evaluation_context: ConditionContext::default(),
            }],
            evaluation_window: Duration::from_secs(5 * 60),
            evaluation_frequency: Duration::from_secs(60),
            threshold_config: ThresholdConfig {
                static_thresholds: Some(StaticThresholds {
                    warning_threshold: 5.0,
                    critical_threshold: 10.0,
                    recovery_threshold: Some(3.0),
                    hysteresis_margin: Some(0.5),
                }),
                dynamic_thresholds: None,
                adaptive_thresholds: None,
                baseline_thresholds: None,
                percentile_thresholds: None,
            },
            suppression_config: SuppressionConfig::default(),
            escalation_policy_id: None,
            notification_channels: vec!["email".to_string()],
            recovery_conditions: vec![],
            metadata: AlertRuleMetadata::default(),
            enabled: true,
            created_at: SystemTime::now(),
            last_modified: SystemTime::now(),
            last_triggered: None,
            trigger_count: 0,
        };
        assert!(manager.validate_rule(&valid_rule).is_ok());
        let invalid_rule = AlertRule {
            rule_id: "".to_string(),
            rule_name: "Invalid Rule".to_string(),
            description: "Invalid alert rule".to_string(),
            category: AlertCategory::Performance,
            severity: SeverityLevel::High,
            conditions: vec![],
            evaluation_window: Duration::from_secs(5 * 60),
            evaluation_frequency: Duration::from_secs(60),
            threshold_config: ThresholdConfig {
                static_thresholds: None,
                dynamic_thresholds: None,
                adaptive_thresholds: None,
                baseline_thresholds: None,
                percentile_thresholds: None,
            },
            suppression_config: SuppressionConfig::default(),
            escalation_policy_id: None,
            notification_channels: vec![],
            recovery_conditions: vec![],
            metadata: AlertRuleMetadata::default(),
            enabled: true,
            created_at: SystemTime::now(),
            last_modified: SystemTime::now(),
            last_triggered: None,
            trigger_count: 0,
        };
        assert!(manager.validate_rule(&invalid_rule).is_err());
    }
    #[tokio::test]
    async fn test_alert_statistics() {
        let stats = AlertStatistics::new();
        assert_eq!(stats.total_alerts_generated.load(Ordering::Relaxed), 0);
        let alert = ActiveAlert {
            alert_id: "test-alert".to_string(),
            rule_id: "test-rule".to_string(),
            test_id: "test-001".to_string(),
            alert_type: AlertType::Threshold,
            severity: SeverityLevel::High,
            status: AlertStatus::Active,
            title: "Test Alert".to_string(),
            description: "Test alert description".to_string(),
            triggered_at: SystemTime::now(),
            last_updated: SystemTime::now(),
            acknowledgment_info: None,
            escalation_info: None,
            suppression_info: None,
            context_data: AlertContext {
                test_execution_context: None,
                system_state: SystemState {
                    cpu_utilization: 50.0,
                    memory_utilization: 60.0,
                    disk_utilization: 70.0,
                    network_utilization: 30.0,
                    active_processes: 100,
                    load_average: 1.5,
                    system_uptime: Duration::from_secs(24 * 3600),
                    resource_pressure: PressureLevel::Medium,
                    health_status: HealthStatus::Degraded,
                },
                environmental_factors: HashMap::new(),
                recent_changes: vec![],
                related_metrics: HashMap::new(),
                dependency_status: HashMap::new(),
            },
            impact_assessment: ImpactAssessment {
                severity: "High".to_string(),
                impact_level: "High".to_string(),
                affected_users: 100,
                affected_systems: vec!["test-system".to_string()],
                business_impact: "Medium".to_string(),
                estimated_downtime: Duration::from_secs(30 * 60),
                financial_impact: 1000.0,
                estimated_cost: 1000.0,
            },
            recommended_actions: vec![],
            alert_fingerprint: "test-fingerprint".to_string(),
            correlation_ids: vec![],
            resolution_info: None,
        };
        stats.record_alert_generated(&alert).await;
        assert_eq!(stats.total_alerts_generated.load(Ordering::Relaxed), 1);
        let resolution_time = Duration::from_secs(15 * 60);
        stats.record_alert_resolved(&alert, resolution_time).await;
        let avg_times = stats.average_resolution_time.read().await;
        assert!(avg_times.contains_key(&SeverityLevel::High));
    }
    #[test]
    fn test_threshold_config_static() {
        let static_config = StaticThresholds {
            warning_threshold: 5.0,
            critical_threshold: 10.0,
            recovery_threshold: Some(3.0),
            hysteresis_margin: Some(0.5),
        };
        assert_eq!(static_config.warning_threshold, 5.0);
        assert_eq!(static_config.critical_threshold, 10.0);
        assert_eq!(static_config.recovery_threshold, Some(3.0));
        assert_eq!(static_config.hysteresis_margin, Some(0.5));
    }
}
