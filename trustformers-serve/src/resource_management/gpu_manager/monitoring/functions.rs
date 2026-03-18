//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::GpuMonitoringError;

/// Result type for GPU monitoring operations
pub type GpuMonitoringResult<T> = Result<T, GpuMonitoringError>;
#[cfg(test)]
mod tests {
    use super::super::*;

    use crate::resource_management::types::{
        GpuClockSpeeds, GpuMetricType, GpuMonitoringConfig, GpuRealTimeMetrics,
    };
    use chrono::Utc;
    use std::sync::Arc;
    use std::time::Duration;
    /// Helper function to create test monitoring configuration
    fn create_test_config() -> GpuMonitoringConfig {
        GpuMonitoringConfig::default()
    }
    /// Helper function to create test metrics
    fn create_test_metrics(device_id: usize) -> GpuRealTimeMetrics {
        GpuRealTimeMetrics {
            device_id,
            timestamp: Utc::now(),
            memory_usage_mb: 8192,
            utilization_percent: 75.0,
            temperature_celsius: 65.0,
            power_consumption_watts: 200.0,
            clock_speeds: GpuClockSpeeds {
                core_clock_mhz: 1800,
                memory_clock_mhz: 7000,
                shader_clock_mhz: Some(1900),
            },
            fan_speeds: vec![50.0],
        }
    }
    #[tokio::test]
    async fn test_monitoring_system_creation() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        assert!(!monitoring_system.is_monitoring_active());
        let metrics = monitoring_system.get_realtime_metrics().await;
        assert!(metrics.is_empty());
    }
    #[tokio::test]
    async fn test_start_stop_monitoring() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        assert!(monitoring_system.is_monitoring_active());
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
        assert!(!monitoring_system.is_monitoring_active());
    }
    #[tokio::test]
    async fn test_metrics_update_and_retrieval() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        let test_metrics = create_test_metrics(0);
        monitoring_system
            .update_metrics(0, test_metrics.clone())
            .await
            .expect("Update metrics should succeed");
        let realtime = monitoring_system.get_realtime_metrics().await;
        assert_eq!(realtime.len(), 1);
        assert!(realtime.contains_key(&0));
        let stored_metrics = &realtime[&0];
        assert_eq!(stored_metrics.device_id, 0);
        assert_eq!(stored_metrics.utilization_percent, 75.0);
        assert_eq!(stored_metrics.temperature_celsius, 65.0);
    }
    #[tokio::test]
    async fn test_historical_metrics() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        let test_metrics = create_test_metrics(0);
        for i in 0..5 {
            let mut metrics = test_metrics.clone();
            metrics.utilization_percent = 50.0 + (i as f32 * 10.0);
            metrics.timestamp = Utc::now();
            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        let historical = monitoring_system.get_historical_metrics(None, None, None).await;
        assert!(!historical.is_empty());
        let device_0_metrics = monitoring_system.get_historical_metrics(Some(0), None, None).await;
        assert!(!device_0_metrics.is_empty());
        assert!(device_0_metrics.iter().all(|m| m.device_id == 0));
        let utilization_metrics = monitoring_system
            .get_historical_metrics(None, Some(GpuMetricType::Utilization), None)
            .await;
        assert!(!utilization_metrics.is_empty());
        assert!(utilization_metrics
            .iter()
            .all(|m| matches!(m.metric_type, GpuMetricType::Utilization)));
    }
    #[tokio::test]
    async fn test_metrics_summary() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        for i in 0..10 {
            let mut metrics = create_test_metrics(0);
            metrics.utilization_percent = 50.0 + (i as f32 * 5.0);
            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let summary = monitoring_system
            .get_metrics_summary(0, GpuMetricType::Utilization, Duration::from_secs(60))
            .await;
        assert!(summary.is_some());
        let summary = summary.expect("Should get summary");
        assert_eq!(summary.device_id, 0);
        assert_eq!(summary.sample_count, 10);
        assert!(summary.average > 50.0);
        assert!(summary.maximum > summary.minimum);
    }
    #[tokio::test]
    async fn test_configuration_validation() {
        let mut config = create_test_config();
        config.monitoring_interval = Duration::from_secs(0);
        let result = GpuMonitoringSystem::new(config).await;
        assert!(result.is_err());
        let config = create_test_config();
        let result = GpuMonitoringSystem::new(config).await;
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn test_configuration_update() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        let mut new_config = create_test_config();
        new_config.monitoring_interval = Duration::from_secs(10);
        new_config.enable_performance_tracking = false;
        monitoring_system
            .update_config(new_config.clone())
            .await
            .expect("Update config should succeed");
        let current_config = monitoring_system.get_config().await;
        assert_eq!(current_config.monitoring_interval, Duration::from_secs(10));
        assert!(!current_config.enable_performance_tracking);
    }
    #[tokio::test]
    async fn test_monitoring_statistics() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        let metrics = create_test_metrics(0);
        monitoring_system
            .update_metrics(0, metrics)
            .await
            .expect("Update metrics should succeed");
        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(stats.is_active);
        assert_eq!(stats.monitored_devices, 1);
        assert!(stats.historical_entries > 0);
        assert!(stats.alerts_enabled);
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }
    #[tokio::test]
    async fn test_background_tasks() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(stats.background_tasks > 0);
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
        let stats = monitoring_system.get_monitoring_statistics().await;
        assert!(!stats.is_active);
    }
    #[tokio::test]
    async fn test_error_handling() {
        let config = create_test_config();
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        let mut metrics = create_test_metrics(0);
        metrics.device_id = 1;
        let result = monitoring_system.update_metrics(0, metrics).await;
        assert!(result.is_err());
        match result.unwrap_err() {
            GpuMonitoringError::MetricsCollectionError { .. } => {},
            other => panic!("Unexpected error type: {:?}", other),
        }
    }
    #[tokio::test]
    async fn test_historical_data_cleanup() {
        let mut config = create_test_config();
        config.monitoring_interval = Duration::from_millis(100);
        let monitoring_system = GpuMonitoringSystem::new(config)
            .await
            .expect("Monitoring system creation should succeed");
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        for i in 0..1000 {
            let mut metrics = create_test_metrics(0);
            metrics.timestamp = Utc::now();
            monitoring_system
                .update_metrics(0, metrics)
                .await
                .expect("Update metrics should succeed");
            if i % 100 == 0 {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        }
        tokio::time::sleep(Duration::from_millis(500)).await;
        let historical = monitoring_system.get_historical_metrics(None, None, None).await;
        let max_expected = GpuMonitoringSystem::calculate_max_historical_entries(
            &monitoring_system.get_config().await,
        );
        assert!(historical.len() <= max_expected * 2);
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }
    #[tokio::test]
    async fn test_concurrent_metrics_updates() {
        let config = create_test_config();
        let monitoring_system = Arc::new(
            GpuMonitoringSystem::new(config)
                .await
                .expect("Monitoring system creation should succeed"),
        );
        monitoring_system
            .start_monitoring()
            .await
            .expect("Start monitoring should succeed");
        let mut handles = Vec::new();
        for device_id in 0..3 {
            let ms = monitoring_system.clone();
            let handle = tokio::spawn(async move {
                for i in 0..10 {
                    let mut metrics = create_test_metrics(device_id);
                    metrics.utilization_percent = 50.0 + (i as f32 * 2.0);
                    ms.update_metrics(device_id, metrics)
                        .await
                        .expect("Update metrics should succeed");
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
            });
            handles.push(handle);
        }
        for handle in handles {
            handle.await.expect("Join handle should succeed");
        }
        let realtime = monitoring_system.get_realtime_metrics().await;
        assert_eq!(realtime.len(), 3);
        for device_id in 0..3 {
            assert!(realtime.contains_key(&device_id));
        }
        monitoring_system
            .stop_monitoring()
            .await
            .expect("Stop monitoring should succeed");
    }
}
