//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::{AlertType, EfficiencyMetrics, EventSeverity, HealthEventType, HealthStatus, PortHealthConfig, PortHealthMonitor, PortHealthThresholds, TrendDirection};

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::test;
    #[test]
    async fn test_health_monitor_creation() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        let status = monitor.get_health_status().await;
        assert_eq!(status.overall_status, HealthStatus::Unknown);
        assert_eq!(status.health_score, 100.0);
    }
    #[test]
    async fn test_health_status_update() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        monitor.update_health_status(90, 10, 5, 0, 25.0).await;
        let status = monitor.get_health_status().await;
        assert_eq!(status.available_ports, 90);
        assert_eq!(status.allocated_ports, 10);
        assert_eq!(status.overall_status, HealthStatus::Healthy);
    }
    #[test]
    async fn test_critical_utilization_alert() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        monitor.update_health_status(5, 95, 0, 0, 25.0).await;
        let status = monitor.get_health_status().await;
        assert_eq!(status.overall_status, HealthStatus::Critical);
        assert!(! status.active_alerts.is_empty());
        let alerts = monitor.get_active_alerts().await;
        assert!(! alerts.is_empty());
        assert!(
            alerts.iter().any(| alert | matches!(alert.alert_type,
            AlertType::HighUtilization))
        );
    }
    #[test]
    async fn test_performance_degradation_alert() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        monitor.update_health_status(50, 50, 0, 0, 600.0).await;
        let status = monitor.get_health_status().await;
        assert_eq!(status.overall_status, HealthStatus::Critical);
        let alerts = monitor.get_active_alerts().await;
        assert!(
            alerts.iter().any(| alert | matches!(alert.alert_type,
            AlertType::SlowPerformance))
        );
    }
    #[test]
    async fn test_health_report_generation() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        monitor.update_health_status(80, 20, 5, 2, 75.0).await;
        let report = monitor.generate_health_report().await;
        assert!(report.contains("PORT HEALTH MONITORING REPORT"));
        assert!(report.contains("Overall Status:"));
        assert!(report.contains("Health Score:"));
        assert!(report.contains("RESOURCE STATUS"));
        assert!(report.contains("PERFORMANCE METRICS"));
    }
    #[test]
    async fn test_trend_analysis() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        for i in 0..10 {
            monitor
                .update_health_status(90 - i * 2, 10 + i * 2, 0, 0, 25.0 + i as f64)
                .await;
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let status = monitor.get_health_status().await;
        assert!(
            matches!(status.predictive_indicators.utilization_trend,
            TrendDirection::Increasing | TrendDirection::Stable)
        );
    }
    #[test]
    async fn test_baseline_establishment() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        let efficiency = EfficiencyMetrics::default();
        monitor.establish_baseline(50.0, 60.0, 1.0, efficiency, 100).await.expect("async operation should succeed in test");
        let baseline = monitor.performance_baseline.read();
        assert!(baseline.is_valid);
        assert_eq!(baseline.sample_count, 100);
    }
    #[test]
    async fn test_health_event_recording() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        monitor
            .record_health_event(
                HealthStatus::Warning,
                HealthEventType::AlertTriggered,
                EventSeverity::Warning,
                "Test event".to_string(),
                HashMap::new(),
            )
            .await;
        let history = monitor.get_health_history().await;
        assert_eq!(history.len(), 1);
        assert_eq!(history[0].status, HealthStatus::Warning);
    }
    #[test]
    async fn test_config_update() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        let mut new_config = PortHealthConfig::default();
        new_config.history_size = 500;
        new_config.enable_detailed_logging = true;
        monitor.update_config(new_config).await.expect("async operation should succeed in test");
        let config = monitor.config.read();
        assert_eq!(config.history_size, 500);
        assert!(config.enable_detailed_logging);
    }
    #[test]
    async fn test_threshold_update() {
        let monitor = PortHealthMonitor::new().await.expect("async operation should succeed in test");
        let mut new_thresholds = PortHealthThresholds::default();
        new_thresholds.utilization_warning = 70.0;
        new_thresholds.utilization_critical = 90.0;
        monitor.update_thresholds(new_thresholds).await.expect("async operation should succeed in test");
        let thresholds = monitor.alert_thresholds.read();
        assert_eq!(thresholds.utilization_warning, 70.0);
        assert_eq!(thresholds.utilization_critical, 90.0);
    }
}
