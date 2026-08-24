//! Tests for the alert rule engine's real evaluation path.
//!
//! These lock in the 0.2.1 fix: before it, `AlertRuleEngine::evaluate_rule`
//! was `for _condition in &rule.conditions { let _evaluation_result = false;
//! if false { .. } }`, so `AlertManager::process_metrics` could never return an
//! alert no matter what arrived on the stream.

use super::types::*;
use crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel;
use crate::test_performance_monitoring::metrics::StreamingMetrics;
use crate::test_performance_monitoring::types::*;
use std::collections::HashMap;
use std::time::{Duration, SystemTime};

fn rule(metric_name: &str, operator: ComparisonOperator, threshold: f64) -> AlertRule {
    AlertRule {
        rule_id: "rule-1".to_string(),
        rule_name: "CPU too high".to_string(),
        description: "CPU exceeded the configured ceiling".to_string(),
        category: AlertCategory::Performance,
        severity: SeverityLevel::High,
        conditions: vec![AlertCondition {
            condition_id: "condition-1".to_string(),
            condition_type: AlertConditionType::Threshold,
            metric_selector: MetricSelector {
                metric_name: metric_name.to_string(),
                test_id_pattern: None,
                tag_filters: HashMap::new(),
                aggregation_scope: AggregationScope::Test,
                time_window: Duration::from_secs(60),
            },
            operator,
            threshold_value: ThresholdValue::Absolute(threshold),
            duration_requirement: None,
            aggregation_method: None,
            condition_weight: 1.0,
            evaluation_context: ConditionContext::default(),
        }],
        evaluation_window: Duration::from_secs(60),
        evaluation_frequency: Duration::from_secs(10),
        threshold_config: ThresholdConfig {
            static_thresholds: None,
            dynamic_thresholds: None,
            adaptive_thresholds: None,
            baseline_thresholds: None,
            percentile_thresholds: None,
        },
        suppression_config: SuppressionConfig::default(),
        escalation_policy_id: None,
        notification_channels: Vec::new(),
        recovery_conditions: Vec::new(),
        metadata: AlertRuleMetadata::default(),
        enabled: true,
        created_at: SystemTime::now(),
        last_modified: SystemTime::now(),
        last_triggered: None,
        trigger_count: 0,
    }
}

fn sample(cpu: Option<f64>) -> StreamingMetrics {
    StreamingMetrics {
        stream_id: "stream-1".to_string(),
        test_id: "test-1".to_string(),
        timestamp: SystemTime::now(),
        elapsed_time: None,
        current_phase: None,
        progress_percent: None,
        instantaneous_cpu: cpu,
        instantaneous_memory: 1024,
        instantaneous_io_rate: None,
        instantaneous_network_rate: None,
        live_error_count: None,
        live_warning_count: None,
        performance_indicators: Vec::new(),
        anomaly_flags: Vec::new(),
        prediction_metrics: None,
    }
}

fn manager() -> AlertManager {
    AlertManager::new(AlertConfig {
        enabled: true,
        ..AlertConfig::default()
    })
}

#[tokio::test]
async fn a_breached_threshold_really_raises_an_alert() {
    let manager = manager();
    manager
        .create_rule(rule("cpu", ComparisonOperator::GreaterThan, 80.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));

    let alerts = manager
        .process_metrics(&sample(Some(95.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));

    // Before 0.2.1 this was always empty.
    assert_eq!(alerts.len(), 1, "a breached threshold must raise an alert");
    let alert = &alerts[0];
    assert_eq!(alert.rule_id, "rule-1");
    assert_eq!(alert.test_id, "test-1");
    assert!(matches!(alert.severity, SeverityLevel::High));
    // The alert must carry the observation it rests on.
    assert!(alert.context_data.related_metrics.contains_key("cpu"));
    // Fields nobody computes must say so rather than claiming a verdict.
    assert_eq!(alert.impact_assessment.impact_level, "not assessed");
    assert_eq!(alert.impact_assessment.business_impact, "not assessed");
}

#[tokio::test]
async fn an_unbreached_threshold_raises_nothing() {
    let manager = manager();
    manager
        .create_rule(rule("cpu", ComparisonOperator::GreaterThan, 80.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));

    let alerts = manager
        .process_metrics(&sample(Some(12.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert!(alerts.is_empty());
}

#[tokio::test]
async fn an_unobserved_metric_is_not_treated_as_zero() {
    let manager = manager();
    // `LessThan 10` would fire on a fabricated zero.
    manager
        .create_rule(rule("cpu", ComparisonOperator::LessThan, 10.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));

    let alerts = manager
        .process_metrics(&sample(None))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert!(
        alerts.is_empty(),
        "an unobserved metric must not be compared as though it were 0.0"
    );
}

#[tokio::test]
async fn a_disabled_manager_evaluates_nothing() {
    let manager = AlertManager::new(AlertConfig {
        enabled: false,
        ..AlertConfig::default()
    });
    assert!(!manager.config().enabled);
    manager
        .create_rule(rule("cpu", ComparisonOperator::GreaterThan, 1.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));
    let alerts = manager
        .process_metrics(&sample(Some(99.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert!(
        alerts.is_empty(),
        "a disabled manager must not raise alerts"
    );
}

#[tokio::test]
async fn a_suppressed_rule_raises_nothing() {
    let manager = manager();
    manager
        .create_rule(rule("cpu", ComparisonOperator::GreaterThan, 80.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));
    manager
        .suppression_manager()
        .suppress_rule("rule-1", "planned maintenance".to_string())
        .await;

    let alerts = manager
        .process_metrics(&sample(Some(95.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert!(alerts.is_empty(), "suppression must actually suppress");

    manager.suppression_manager().unsuppress_rule("rule-1").await;
    let alerts = manager
        .process_metrics(&sample(Some(95.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert_eq!(alerts.len(), 1, "lifting suppression must restore alerting");
}

#[tokio::test]
async fn the_alert_store_index_is_really_maintained() {
    let manager = manager();
    manager
        .create_rule(rule("cpu", ComparisonOperator::GreaterThanOrEqual, 50.0))
        .await
        .unwrap_or_else(|e| panic!("create_rule failed: {e:?}"));
    let alerts = manager
        .process_metrics(&sample(Some(50.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert_eq!(alerts.len(), 1);

    let store = manager.alert_store();
    let by_test = store.alerts_for_test("test-1").await;
    assert_eq!(by_test.len(), 1, "the by_test index must be populated");
    let by_severity = store.alerts_by_severity(SeverityLevel::High).await;
    assert_eq!(
        by_severity.len(),
        1,
        "the by_severity index must be populated"
    );
    // In-memory only, uncompressed: stated rather than implied.
    assert!(!store.has_storage_backend());
    assert!(!store.compression_enabled());
}

#[tokio::test]
async fn escalation_to_an_unregistered_policy_is_reported() {
    let manager = manager();
    let escalation = manager.escalation_manager();
    let alert_rule = rule("cpu", ComparisonOperator::GreaterThan, 1.0);
    let alerts_before = manager
        .process_metrics(&sample(Some(2.0)))
        .await
        .unwrap_or_else(|e| panic!("process_metrics failed: {e:?}"));
    assert!(alerts_before.is_empty(), "no rule added yet");

    let _ = alert_rule;
    // A policy that was never registered must be refused, not silently accepted.
    assert!(escalation.policy("ghost-policy").await.is_none());
}

#[tokio::test]
async fn a_threshold_monitor_without_evaluators_refuses_rather_than_saying_false() {
    let manager = manager();
    let monitor = manager.threshold_monitor();
    assert_eq!(monitor.evaluator_count(), 0);
    let condition = rule("cpu", ComparisonOperator::GreaterThan, 1.0)
        .conditions
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("fixture must carry one condition"));
    let result = monitor.evaluate_condition(&condition, &PerformanceMetrics::default()).await;
    // 0.2.1: this used to be Ok(false), indistinguishable from "checked, and it
    // did not trip".
    assert!(
        result.is_err(),
        "an unevaluated condition must not report as not-tripped"
    );
    assert_eq!(monitor.evaluation_count().await, 1);
}
