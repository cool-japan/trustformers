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
    /// Regression: `create_initial_health_status` used to fabricate a fully
    /// healthy record (`is_healthy: true`, `health_score: 1.0`, every `*_ok`
    /// flag `true`, plausible sensor readings) for a device that had never
    /// been probed, readable via `get_health_status` before any check ran.
    /// The initial state must instead read as unmeasured on every
    /// telemetry-dependent field, while the fields that genuinely are known
    /// at discovery time (memory, hardware status) are computed for real.
    #[test]
    fn test_create_initial_health_status_is_honest() {
        let device = create_test_device(0);
        let config = GpuHealthConfig::default();
        let status = GpuHealthMonitor::create_initial_health_status(&device, &config);
        assert!(!status.is_healthy, "a never-probed device is not healthy");
        assert_eq!(status.consecutive_healthy_checks, 0);
        assert_eq!(status.consecutive_unhealthy_checks, 0);
        assert!(!status.temperature_ok);
        assert!(!status.performance_ok);
        assert!(!status.power_ok);
        assert!(!status.driver_ok);
        assert!(
            status.current_temperature.is_nan(),
            "an unread sensor must not report a plausible number"
        );
        assert!(status.current_utilization.is_nan());
        assert!(status.current_power.is_nan());
        assert!(!status.issues.is_empty());
        assert!(status
            .issues
            .iter()
            .any(|issue| issue.contains("has not") || issue.contains("not been")));
        // Memory and hardware are known at discovery time (not from live
        // telemetry), so they are computed for real, not marked unknown: this
        // fixture has 25% memory in use and status `Available`.
        assert!(status.memory_ok);
        assert!(status.hardware_ok);
        assert!(
            status.health_score > 0.0 && status.health_score < 1.0,
            "expected a genuinely partial score, got {}",
            status.health_score
        );
    }
    /// Regression: `create_initial_analytics` used to seed `health_history`
    /// with one fabricated `(now, 1.0)` sample, feeding a fake perfect score
    /// straight into the least-squares trend fit as if it were a real
    /// measurement.
    #[test]
    fn test_create_initial_analytics_has_no_fabricated_history() {
        let analytics = GpuHealthMonitor::create_initial_analytics(0);
        assert!(analytics.health_history.is_empty());
        assert!(analytics.temperature_history.is_empty());
        assert!(analytics.memory_history.is_empty());
        assert!(analytics.performance_history.is_empty());
        assert_eq!(analytics.trend_r_squared, 0.0);
        assert_eq!(analytics.trend_analysis.trend, HealthTrend::Unknown);
        assert_eq!(analytics.trend_analysis.confidence, 0.0);
    }
    /// `update_analytics_metrics` must report a high `trend_r_squared` for a
    /// history that is genuinely close to a straight line.
    #[test]
    fn test_update_analytics_metrics_r_squared_high_for_linear_trend() {
        let mut analytics = GpuHealthMonitor::create_initial_analytics(0);
        let now = Utc::now();
        for i in 0..10 {
            let timestamp = now + ChronoDuration::hours(i);
            let score = 1.0 - (i as f32 * 0.05);
            analytics.health_history.push_back((timestamp, score));
        }
        GpuHealthMonitor::update_analytics_metrics(&mut analytics);
        assert!(
            analytics.trend_r_squared > 0.99,
            "a perfectly linear history should fit almost exactly, got {}",
            analytics.trend_r_squared
        );
        assert!(analytics.health_trend_slope < 0.0);
    }
    /// Regression target from the brief: a metric with zero recorded
    /// variance (nothing has ever moved it) must not be reported as a
    /// high-confidence trend fit. R^2 is undefined when there is no variance
    /// to explain, and the honest choice is zero confidence, not a
    /// fabricated 1.0 "perfect fit".
    #[test]
    fn test_update_analytics_metrics_r_squared_zero_for_constant_series() {
        let mut analytics = GpuHealthMonitor::create_initial_analytics(0);
        let now = Utc::now();
        for i in 0..10 {
            let timestamp = now + ChronoDuration::hours(i);
            analytics.health_history.push_back((timestamp, 0.9));
        }
        GpuHealthMonitor::update_analytics_metrics(&mut analytics);
        assert_eq!(
            analytics.trend_r_squared, 0.0,
            "a constant series has no variance for a linear trend to explain"
        );
        GpuHealthMonitor::compute_trend_analysis(&mut analytics);
        assert_eq!(analytics.trend_analysis.confidence, 0.0);
    }
    /// Regression: temperature and power used to be synthesized from the
    /// device's utilization (`45.0 + util * 0.5`), so a genuinely overheating GPU
    /// could never trip the check and `driver_ok` was hardcoded `true`.
    ///
    /// The values must now come from the driver, and a sensor the driver does
    /// not report must read as *not ok* / unknown rather than as comfortably
    /// within threshold.
    #[tokio::test]
    async fn test_health_check() {
        let device = create_test_device(0);
        let config = Arc::new(RwLock::new(GpuHealthConfig::default()));
        let health = GpuHealthMonitor::perform_comprehensive_health_check(&device, &config).await;

        assert_eq!(health.device_id, 0);
        assert!(health.health_score >= 0.0 && health.health_score <= 1.0);

        // Memory comes from the device record, which really carries it.
        assert!(health.memory_ok);

        // Utilization comes from the driver. 0.2.1: it fell back to
        // `GpuDeviceInfo::utilization_percent`, which discovery fixes at 0.0
        // and never updates -- so `performance_ok` was decided by a constant.
        // Absent telemetry must now read as *not ok*, with the reason stated.
        if health.driver_ok {
            assert!(
                health.current_utilization.is_finite(),
                "a driver that answered must report a real utilization"
            );
        } else {
            assert!(
                !health.performance_ok,
                "an unread utilization sensor must not read as within threshold"
            );
            assert!(health.current_utilization.is_nan());
            assert!(health
                .issues
                .iter()
                .any(|issue| issue.contains("Utilization sensor is unavailable")));
        }

        if health.driver_ok {
            // A driver answered, so the sensor values are real measurements.
            assert!(
                health.current_temperature.is_finite(),
                "a driver that answered must report a real temperature"
            );
        } else {
            // No driver answered for this synthetic device: nothing may be
            // reported as within threshold, and the reason must be stated.
            assert!(!health.temperature_ok);
            assert!(!health.power_ok);
            assert!(health.current_temperature.is_nan());
            assert!(health.current_power.is_nan());
            assert!(health
                .issues
                .iter()
                .any(|issue| issue.contains("Temperature sensor is unavailable")));
            assert!(health
                .issues
                .iter()
                .any(|issue| issue.contains("did not answer a telemetry query")));
        }

        // The old synthesized value must never reappear.
        let synthesized = 45.0 + device.utilization_percent * 0.5;
        assert!(
            health.current_temperature.is_nan()
                || (health.current_temperature - synthesized).abs() > f32::EPSILON,
            "temperature {} matches the removed synthetic formula",
            health.current_temperature
        );
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
    /// A device the driver will not talk about cannot be certified healthy.
    ///
    /// 0.2.1: this test set `device.utilization_percent = 99.0` and asserted
    /// `!performance_ok`, which passed only because the health check read that
    /// record field. It is a discovery-time constant no live path writes, so
    /// the assertion proved nothing about a real device. The real invariant is
    /// the one below.
    #[tokio::test]
    async fn test_unhealthy_device_detection() {
        let device = create_test_device(0);
        let config = Arc::new(RwLock::new(GpuHealthConfig::default()));
        let health = GpuHealthMonitor::perform_comprehensive_health_check(&device, &config).await;

        if !health.driver_ok {
            // This synthetic device has no driver behind it, so no sensor can
            // be read and nothing may be reported as within threshold.
            assert!(!health.performance_ok);
            assert!(!health.temperature_ok);
            assert!(!health.power_ok);
            assert!(
                !health.is_healthy,
                "a device with no readable sensor is not healthy"
            );
            assert!(!health.issues.is_empty());
            assert!(health.health_score < 1.0);
        }
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
            trend_r_squared: 0.75,
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
        // `confidence` now comes straight from `trend_r_squared` rather than a
        // sample-count ladder; this locks in that wiring.
        assert_eq!(analytics.trend_analysis.confidence, 0.75);
    }
}
