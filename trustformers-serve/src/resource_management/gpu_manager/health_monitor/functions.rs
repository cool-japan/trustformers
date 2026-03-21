//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use super::types::GpuHealthError;

/// Result type for GPU health operations
pub type GpuHealthResult<T> = Result<T, GpuHealthError>;
#[cfg(test)]
mod tests {
    use super::super::*;
    use crate::resource_management::gpu_manager::types::{GpuDeviceInfo, GpuDeviceStatus};
    use chrono::Duration as ChronoDuration;
    use chrono::Utc;
    use parking_lot::RwLock;
    use std::collections::{HashMap, VecDeque};
    use std::sync::Arc;
    use std::time::Duration;
    use tokio::sync::broadcast;
    fn create_test_device(device_id: usize) -> GpuDeviceInfo {
        GpuDeviceInfo {
            device_id,
            device_name: format!("Test GPU {}", device_id),
            total_memory_mb: 8192,
            available_memory_mb: 6144,
            utilization_percent: 50.0,
            capabilities: vec![],
            status: GpuDeviceStatus::Available,
            last_updated: Utc::now(),
        }
    }
    #[tokio::test]
    async fn test_health_monitor_creation() {
        let monitor = GpuHealthMonitor::new();
        assert!(!monitor.is_monitoring());
    }
    #[tokio::test]
    async fn test_health_check() {
        let device = create_test_device(0);
        let config = Arc::new(RwLock::new(GpuHealthConfig::default()));
        let health = GpuHealthMonitor::perform_comprehensive_health_check(&device, &config).await;
        assert_eq!(health.device_id, 0);
        assert!(health.health_score >= 0.0 && health.health_score <= 1.0);
        assert!(health.temperature_ok);
        assert!(health.memory_ok);
        assert!(health.performance_ok);
    }
    #[tokio::test]
    async fn test_health_monitoring_lifecycle() {
        let monitor = GpuHealthMonitor::new();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0));
        let devices = Arc::new(RwLock::new(devices));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        monitor
            .start_monitoring(devices, shutdown_rx)
            .await
            .expect("async operation should succeed in test");
        assert!(monitor.is_monitoring());
        tokio::time::sleep(Duration::from_millis(100)).await;
        let health_status = monitor.get_health_status().await;
        assert!(health_status.contains_key(&0));
        let _ = shutdown_tx.send(());
        monitor.stop_monitoring().await.expect("async operation should succeed in test");
        assert!(!monitor.is_monitoring());
    }
    #[tokio::test]
    async fn test_health_analytics() {
        let monitor = GpuHealthMonitor::new();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0));
        let devices = Arc::new(RwLock::new(devices));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        monitor
            .start_monitoring(devices, shutdown_rx)
            .await
            .expect("async operation should succeed in test");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let analytics = monitor.get_health_analytics().await;
        assert!(analytics.contains_key(&0));
        let device_analytics = &analytics[&0];
        assert!(!device_analytics.health_history.is_empty());
        let _ = shutdown_tx.send(());
        monitor.stop_monitoring().await.expect("async operation should succeed in test");
    }
    #[tokio::test]
    async fn test_health_summary() {
        let monitor = GpuHealthMonitor::new();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0));
        devices.insert(1, create_test_device(1));
        let devices = Arc::new(RwLock::new(devices));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        monitor
            .start_monitoring(devices, shutdown_rx)
            .await
            .expect("async operation should succeed in test");
        tokio::time::sleep(Duration::from_millis(100)).await;
        let summary = monitor.get_health_summary().await;
        assert_eq!(summary.total_devices, 2);
        assert!(summary.healthy_devices <= 2);
        assert!(summary.average_health_score >= 0.0);
        let _ = shutdown_tx.send(());
        monitor.stop_monitoring().await.expect("async operation should succeed in test");
    }
    #[tokio::test]
    async fn test_health_report_generation() {
        let monitor = GpuHealthMonitor::new();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0));
        let devices = Arc::new(RwLock::new(devices));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        monitor
            .start_monitoring(devices, shutdown_rx)
            .await
            .expect("async operation should succeed in test");
        tokio::time::sleep(Duration::from_millis(200)).await;
        let report = monitor.generate_health_report().await;
        assert!(report.contains("GPU Health Monitoring Report"));
        assert!(report.contains("OVERALL HEALTH SUMMARY"));
        assert!(report.contains("MONITORING STATISTICS"));
        assert!(report.contains("DEVICE HEALTH DETAILS"));
        assert!(report.contains("Device 0"));
        let _ = shutdown_tx.send(());
        monitor.stop_monitoring().await.expect("async operation should succeed in test");
    }
    #[tokio::test]
    async fn test_force_health_check() {
        let monitor = GpuHealthMonitor::new();
        let mut devices = HashMap::new();
        devices.insert(0, create_test_device(0));
        let devices = Arc::new(RwLock::new(devices));
        let (shutdown_tx, shutdown_rx) = broadcast::channel(1);
        monitor
            .start_monitoring(devices, shutdown_rx)
            .await
            .expect("async operation should succeed in test");
        let result = monitor.force_health_check().await;
        assert!(result.is_ok());
        let _ = shutdown_tx.send(());
        monitor.stop_monitoring().await.expect("async operation should succeed in test");
    }
    #[tokio::test]
    async fn test_config_update() {
        let monitor = GpuHealthMonitor::new();
        let mut new_config = GpuHealthConfig::default();
        new_config.temperature_threshold = 90.0;
        new_config.check_interval = Duration::from_secs(60);
        let result = monitor.update_config(new_config).await;
        assert!(result.is_ok());
    }
    #[tokio::test]
    async fn test_unhealthy_device_detection() {
        let mut device = create_test_device(0);
        device.utilization_percent = 99.0;
        let config = Arc::new(RwLock::new(GpuHealthConfig::default()));
        let health = GpuHealthMonitor::perform_comprehensive_health_check(&device, &config).await;
        assert!(!health.performance_ok);
        assert!(!health.issues.is_empty());
        assert!(health.health_score < 1.0);
    }
    #[tokio::test]
    async fn test_trend_analysis() {
        let mut analytics = GpuHealthAnalytics {
            device_id: 0,
            health_history: VecDeque::new(),
            temperature_history: VecDeque::new(),
            memory_history: VecDeque::new(),
            performance_history: VecDeque::new(),
            average_health_score: 0.8,
            health_trend_slope: -0.02,
            health_score_stddev: 0.1,
            trend_analysis: HealthTrendAnalysis {
                trend: HealthTrend::Unknown,
                confidence: 0.0,
                projected_24h: 0.0,
                projected_7d: 0.0,
                risk_level: HealthRiskLevel::Low,
                recommendations: Vec::new(),
            },
            prediction_model: None,
        };
        let now = Utc::now();
        for i in 0..10 {
            let timestamp = now - ChronoDuration::hours(i);
            let score = 1.0 - (i as f32 * 0.05);
            analytics.health_history.push_front((timestamp, score));
        }
        GpuHealthMonitor::compute_trend_analysis(&mut analytics);
        assert_eq!(analytics.trend_analysis.trend, HealthTrend::Declining);
        assert!(analytics.trend_analysis.projected_24h < analytics.average_health_score);
        assert!(!analytics.trend_analysis.recommendations.is_empty());
    }
}
