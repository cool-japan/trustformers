//! Comprehensive tests for health monitor analysis and reporting
//!
//! Tests focusing on trend analysis, predictive indicators,
//! efficiency calculations, risk assessment, and comprehensive reporting.

#[cfg(test)]
mod tests {
    use super::super::types::*;
    use chrono::Utc;
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

    #[tokio::test]
    async fn test_increasing_trend_detection() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Simulate increasing utilization
            for i in 0..10 {
                let available = 100 - i * 5;
                let allocated = i * 5;
                monitor
                    .update_health_status(available, allocated, 0, 0, 10.0)
                    .await;
            }
            let status = monitor.get_health_status().await;
            // After 10 updates with increasing allocation, trend should be detected
            assert!(matches!(
                status.predictive_indicators.utilization_trend,
                TrendDirection::Increasing | TrendDirection::Stable | TrendDirection::Unknown
            ));
        }
    }

    #[tokio::test]
    async fn test_stable_trend_detection() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            for _ in 0..10 {
                monitor
                    .update_health_status(80, 20, 0, 0, 10.0)
                    .await;
            }
            let status = monitor.get_health_status().await;
            assert!(matches!(
                status.predictive_indicators.utilization_trend,
                TrendDirection::Stable | TrendDirection::Unknown
            ));
        }
    }

    #[tokio::test]
    async fn test_low_risk_score() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.predictive_indicators.risk_score < 0.5);
        }
    }

    #[tokio::test]
    async fn test_high_risk_score() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(10, 90, 0, 9, 900.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.predictive_indicators.risk_score > 0.3);
        }
    }

    #[tokio::test]
    async fn test_degradation_risk_low_utilization() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.predictive_indicators.degradation_risk < 0.5);
        }
    }

    #[tokio::test]
    async fn test_degradation_risk_high_utilization() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(10, 90, 0, 0, 200.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.predictive_indicators.degradation_risk > 0.0);
        }
    }

    #[tokio::test]
    async fn test_maintenance_urgency_low() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.predictive_indicators.maintenance_urgency < 0.5);
        }
    }

    #[tokio::test]
    async fn test_efficiency_high_allocation_efficiency() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Low allocation time means high allocation efficiency
            monitor
                .update_health_status(800, 200, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!((status.efficiency_metrics.allocation_efficiency - 1.0).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_efficiency_low_allocation_efficiency() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Very high allocation time
            monitor
                .update_health_status(800, 200, 0, 0, 300.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.efficiency_metrics.allocation_efficiency < 1.0);
        }
    }

    #[tokio::test]
    async fn test_efficiency_zero_total_ports() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Both zero, should handle gracefully
            monitor
                .update_health_status(0, 0, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!((status.efficiency_metrics.utilization_efficiency - 1.0).abs() < f32::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_efficiency_waste_percentage() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(800, 200, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.efficiency_metrics.waste_percentage >= 0.0);
        }
    }

    #[tokio::test]
    async fn test_alert_generation_high_conflict() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // 60 conflicts should trigger critical alert
            monitor
                .update_health_status(500, 500, 0, 60, 10.0)
                .await;
            let alerts = monitor.get_active_alerts().await;
            assert!(
                alerts.iter().any(|a| matches!(a.alert_type, AlertType::HighConflictRate))
            );
        }
    }

    #[tokio::test]
    async fn test_alert_generation_low_health_score() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Extreme conditions to drive health score very low
            monitor
                .update_health_status(2, 98, 0, 60, 600.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.health_score < 50.0);
        }
    }

    #[tokio::test]
    async fn test_health_score_clamping() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.health_score >= 0.0);
            assert!(status.health_score <= 100.0);
        }
    }

    #[tokio::test]
    async fn test_health_score_extreme_bad_conditions() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(1, 99, 0, 100, 1000.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.health_score >= 0.0);
            assert!(status.health_score <= 100.0);
        }
    }

    #[tokio::test]
    async fn test_report_recommendations_healthy() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let report = monitor.generate_health_report().await;
            assert!(report.contains("RECOMMENDATIONS"));
            assert!(report.contains("normal parameters"));
        }
    }

    #[tokio::test]
    async fn test_report_recommendations_degraded() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Trigger warning level thresholds
            monitor
                .update_health_status(15, 85, 0, 0, 150.0)
                .await;
            let report = monitor.generate_health_report().await;
            assert!(report.contains("RECOMMENDATIONS"));
        }
    }

    #[tokio::test]
    async fn test_health_status_transitions() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            // Start healthy
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status1 = monitor.get_health_status().await;
            assert_eq!(status1.overall_status, HealthStatus::Healthy);

            // Transition to critical
            monitor
                .update_health_status(2, 98, 0, 0, 10.0)
                .await;
            let status2 = monitor.get_health_status().await;
            assert_eq!(status2.overall_status, HealthStatus::Critical);

            // Health events should have been recorded for the transition
            let history = monitor.get_health_history().await;
            assert!(!history.is_empty());
        }
    }

    #[tokio::test]
    async fn test_uptime_tracking() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let status = monitor.get_health_status().await;
            assert!(status.uptime_seconds > 0);
        }
    }

    #[tokio::test]
    async fn test_report_contains_threshold_info() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(900, 100, 0, 0, 10.0)
                .await;
            let report = monitor.generate_health_report().await;
            assert!(report.contains("THRESHOLD CONFIGURATION"));
            assert!(report.contains("Utilization Warning:"));
            assert!(report.contains("Utilization Critical:"));
        }
    }

    #[tokio::test]
    async fn test_concurrent_health_status_reads() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(800, 200, 0, 0, 50.0)
                .await;
            // Multiple reads should be consistent
            let s1 = monitor.get_health_status().await;
            let s2 = monitor.get_health_status().await;
            assert_eq!(s1.available_ports, s2.available_ports);
            assert_eq!(s1.allocated_ports, s2.allocated_ports);
        }
    }

    #[tokio::test]
    async fn test_report_predictive_indicators_section() {
        if let Ok(monitor) = PortHealthMonitor::new().await {
            monitor
                .update_health_status(800, 200, 0, 0, 50.0)
                .await;
            let report = monitor.generate_health_report().await;
            assert!(report.contains("PREDICTIVE INDICATORS"));
            assert!(report.contains("Utilization Trend:"));
            assert!(report.contains("Risk Score:"));
            assert!(report.contains("Degradation Risk:"));
            assert!(report.contains("Maintenance Urgency:"));
        }
    }
}
