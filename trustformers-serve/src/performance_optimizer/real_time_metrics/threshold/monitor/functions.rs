//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::error::{Result, ThresholdError};
// Import types from real_time_metrics::types that are not re-exported through threshold::types
use super::super::super::types::{ThresholdConfig, ThresholdDirection, TimestampedMetrics};
use chrono::Utc;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Create a sample threshold configuration for testing
pub fn create_sample_threshold(name: &str, metric: &str) -> ThresholdConfig {
    ThresholdConfig {
        name: name.to_string(),
        metric: metric.to_string(),
        warning_threshold: 0.8,
        critical_threshold: 0.95,
        direction: ThresholdDirection::Above,
        adaptive: false,
        evaluation_window: Duration::from_secs(60),
        min_trigger_count: 1,
        cooldown_period: Duration::from_secs(300),
        escalation_policy: "default".to_string(),
    }
}
/// Create sample metrics for testing
pub fn create_sample_metrics(values: HashMap<String, f64>) -> TimestampedMetrics {
    let throughput = values.get("throughput").copied().unwrap_or(100.0);
    let cpu_util = values.get("cpu_utilization").copied().unwrap_or(0.5) as f32;
    let mem_util = values.get("memory_utilization").copied().unwrap_or(0.5) as f32;
    let metrics = crate::performance_optimizer::types::RealTimeMetrics {
        current_parallelism: 4,
        current_throughput: throughput,
        current_latency: Duration::from_millis(100),
        current_cpu_utilization: cpu_util,
        current_memory_utilization: mem_util,
        current_resource_efficiency: 0.8,
        last_updated: Utc::now(),
        collection_interval: Duration::from_secs(1),
        throughput,
        latency: Duration::from_millis(100),
        error_rate: 0.0,
        value: throughput,
        metric_type: "test".to_string(),
        resource_usage: Default::default(),
        cpu_utilization: cpu_util,
        memory_utilization: mem_util,
    };
    TimestampedMetrics {
        timestamp: Utc::now(),
        precise_timestamp: Instant::now(),
        metrics,
        system_state: crate::performance_optimizer::types::SystemState::default(),
        quality_score: 1.0,
        source: "test".to_string(),
        metadata: HashMap::new(),
    }
}
/// Validate threshold configuration
pub fn validate_threshold_config(config: &ThresholdConfig) -> Result<()> {
    if config.name.is_empty() {
        return Err(ThresholdError::ConfigurationError(
            "Threshold name cannot be empty".to_string(),
        ));
    }
    if config.metric.is_empty() {
        return Err(ThresholdError::ConfigurationError(
            "Metric name cannot be empty".to_string(),
        ));
    }
    if config.warning_threshold < 0.0 || config.critical_threshold < 0.0 {
        return Err(ThresholdError::ConfigurationError(
            "Thresholds cannot be negative".to_string(),
        ));
    }
    match config.direction {
        ThresholdDirection::Above => {
            if config.warning_threshold >= config.critical_threshold {
                return Err(ThresholdError::ConfigurationError(
                    "For Above direction, warning threshold must be less than critical threshold"
                        .to_string(),
                ));
            }
        },
        ThresholdDirection::Below => {
            if config.warning_threshold <= config.critical_threshold {
                return Err(
                    ThresholdError::ConfigurationError(
                        "For Below direction, warning threshold must be greater than critical threshold"
                            .to_string(),
                    ),
                );
            }
        },
        ThresholdDirection::OutsideRange { min, max } => {
            if min >= max {
                return Err(ThresholdError::ConfigurationError(
                    "For OutsideRange direction, min must be less than max".to_string(),
                ));
            }
        },
        ThresholdDirection::InsideRange { min, max } => {
            if min >= max {
                return Err(ThresholdError::ConfigurationError(
                    "For InsideRange direction, min must be less than max".to_string(),
                ));
            }
        },
    }
    if config.evaluation_window.as_secs() == 0 {
        return Err(ThresholdError::ConfigurationError(
            "Evaluation window cannot be zero".to_string(),
        ));
    }
    if config.min_trigger_count == 0 {
        return Err(ThresholdError::ConfigurationError(
            "Min trigger count cannot be zero".to_string(),
        ));
    }
    Ok(())
}
#[cfg(test)]
mod tests {
    use super::super::super::alert_manager::AlertManager;
    use super::super::super::evaluator::ThresholdEvaluator;
    use super::super::super::simple_evaluator::SimpleThresholdEvaluator;
    use super::super::types::*;
    use super::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering;
    #[tokio::test]
    async fn test_threshold_monitor_creation() {
        let monitor = ThresholdMonitor::new().await;
        assert!(monitor.is_ok());
    }
    #[tokio::test]
    async fn test_simple_threshold_evaluator() {
        let evaluator = SimpleThresholdEvaluator::new();
        let config = create_sample_threshold("test", "cpu_utilization");
        let result = evaluator.evaluate(&config, 0.9);
        assert!(result.is_ok());
        let evaluation = result.expect("Evaluation should succeed");
        assert!(!evaluation.violated);
        let result_violation = evaluator.evaluate(&config, 0.96);
        assert!(result_violation.is_ok());
        let evaluation_violation =
            result_violation.expect("Evaluation with violation should succeed");
        assert!(evaluation_violation.violated);
    }
    #[tokio::test]
    async fn test_alert_manager() {
        let manager = AlertManager::new().await;
        assert!(manager.is_ok());
        let manager = manager.expect("AlertManager creation should succeed");
        let stats = manager.get_stats();
        assert_eq!(stats.alerts_processed.load(Ordering::Relaxed), 0);
    }
    #[tokio::test]
    async fn test_threshold_config_validation() {
        let valid_config = create_sample_threshold("test", "cpu");
        assert!(validate_threshold_config(&valid_config).is_ok());
        let invalid_config = ThresholdConfig {
            name: "".to_string(),
            ..valid_config
        };
        assert!(validate_threshold_config(&invalid_config).is_err());
    }
    #[tokio::test]
    async fn test_performance_analyzer() {
        let analyzer = PerformanceAnalyzer::new();
        let stats = analyzer.get_stats().await;
        assert_eq!(stats.current_metrics.active_evaluations, 0);
        assert!(stats.current_metrics.avg_evaluation_time_ms >= 0.0);
    }
    #[test]
    fn test_sample_metrics_creation() {
        let mut values = HashMap::new();
        values.insert("cpu_utilization".to_string(), 0.85);
        values.insert("memory_utilization".to_string(), 0.72);
        let _metrics = create_sample_metrics(values);
    }
}
