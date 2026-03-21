//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::GpuAlertError;

/// Result type for alert operations
pub type GpuAlertResult<T> = Result<T, GpuAlertError>;
#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::resource_management::gpu_manager::types::{
        AlertSeverity, GpuAlertConfig, GpuAlertEventType, GpuAlertType, GpuClockSpeeds,
        GpuRealTimeMetrics,
    };
    use chrono::Utc;
    use std::time::Duration;
    /// Helper function to create test configuration
    fn create_test_alert_config() -> GpuAlertConfig {
        GpuAlertConfig::default()
    }
    /// Helper function to create test metrics
    fn create_test_metrics(
        device_id: usize,
        temp: f32,
        util: f32,
        memory_mb: u64,
    ) -> GpuRealTimeMetrics {
        GpuRealTimeMetrics {
            device_id,
            timestamp: Utc::now(),
            memory_usage_mb: memory_mb,
            utilization_percent: util,
            temperature_celsius: temp,
            power_consumption_watts: 250.0,
            clock_speeds: GpuClockSpeeds {
                core_clock_mhz: 1800,
                memory_clock_mhz: 7000,
                shader_clock_mhz: Some(1900),
            },
            fan_speeds: vec![50.0],
        }
    }
    #[tokio::test]
    async fn test_alert_system_creation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        assert!(!alert_system.is_running());
        let alerts = alert_system.get_active_alerts().await;
        assert!(alerts.is_empty());
    }
    #[tokio::test]
    async fn test_alert_system_start_stop() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        assert!(alert_system.is_running());
        alert_system.stop().await.expect("Alert system stop should succeed");
        assert!(!alert_system.is_running());
    }
    #[tokio::test]
    async fn test_temperature_alert_generation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let high_temp_metrics = create_test_metrics(0, 90.0, 50.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &high_temp_metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());
        let alert = alerts.values().next().expect("Should have alert");
        assert_eq!(alert.device_id, 0);
        assert_eq!(alert.alert_type, GpuAlertType::HighTemperature);
        assert_eq!(alert.severity, AlertSeverity::Critical);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_utilization_alert_generation() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let high_util_metrics = create_test_metrics(0, 60.0, 98.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &high_util_metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());
        let alert = alerts.values().next().expect("Should have alert");
        assert_eq!(alert.alert_type, GpuAlertType::HighUtilization);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let metrics = create_test_metrics(0, 90.0, 50.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let alerts = alert_system.get_active_alerts().await;
        assert!(!alerts.is_empty());
        let alert_id = alerts.keys().next().expect("Should have alert").clone();
        let alert = &alerts[&alert_id];
        assert!(!alert.acknowledged);
        alert_system
            .acknowledge_alert(&alert_id)
            .await
            .expect("Acknowledge alert should succeed");
        let alerts = alert_system.get_active_alerts().await;
        let alert = &alerts[&alert_id];
        assert!(alert.acknowledged);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_alert_history() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let metrics = create_test_metrics(0, 90.0, 50.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let history = alert_system.get_alert_history(None, None).await;
        assert!(!history.is_empty());
        let event = &history[0];
        assert_eq!(event.event_type, GpuAlertEventType::Triggered);
        assert_eq!(event.alert.device_id, 0);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_device_specific_alerts() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let metrics_0 = create_test_metrics(0, 90.0, 50.0, 8192);
        let metrics_1 = create_test_metrics(1, 70.0, 98.0, 8192);
        alert_system
            .check_metrics_for_alerts(0, &metrics_0)
            .await
            .expect("Check metrics should succeed");
        alert_system
            .check_metrics_for_alerts(1, &metrics_1)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let device_0_alerts = alert_system.get_device_alerts(0).await;
        let device_1_alerts = alert_system.get_device_alerts(1).await;
        assert!(!device_0_alerts.is_empty());
        assert!(!device_1_alerts.is_empty());
        assert_eq!(device_0_alerts[0].alert_type, GpuAlertType::HighTemperature);
        assert_eq!(device_1_alerts[0].alert_type, GpuAlertType::HighUtilization);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_alert_statistics() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let metrics = create_test_metrics(0, 90.0, 98.0, 20000);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let stats = alert_system.get_alert_statistics().await;
        assert!(stats.total_alerts_generated > 0);
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = create_test_alert_config();
        config.thresholds.temperature_warning = 90.0;
        config.thresholds.temperature_critical = 80.0;
        let result = GpuAlertSystem::new(config).await;
        assert!(result.is_err());
    }
    #[tokio::test]
    async fn test_bulk_acknowledgment() {
        let config = create_test_alert_config();
        let alert_system =
            GpuAlertSystem::new(config).await.expect("Alert system creation should succeed");
        alert_system.start().await.expect("Alert system start should succeed");
        let metrics = create_test_metrics(0, 90.0, 98.0, 20000);
        alert_system
            .check_metrics_for_alerts(0, &metrics)
            .await
            .expect("Check metrics should succeed");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let ack_count = alert_system
            .acknowledge_device_alerts(0)
            .await
            .expect("Acknowledge device alerts should succeed");
        assert!(ack_count > 0);
        let device_alerts = alert_system.get_device_alerts(0).await;
        for alert in device_alerts {
            assert!(alert.acknowledged);
        }
        alert_system.stop().await.expect("Alert system stop should succeed");
    }
}
