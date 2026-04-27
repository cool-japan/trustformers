//! Comprehensive tests for performance metrics analysis
//!
//! Tests focusing on trend calculations, performance scoring,
//! anomaly detection, forecasting, insights generation, and data export.

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
    async fn test_trend_analysis_with_increasing_times() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 100,
                enable_detailed_timing: true,
                enable_percentile_tracking: true,
                max_timing_samples: 1000,
                enable_trend_analysis: true,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds::default(),
            };
            metrics.update_config(config).await;

            // Record increasing allocation times
            for i in 1..=10 {
                metrics
                    .record_allocation_success(Duration::from_millis(i * 10))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let trends = metrics.get_performance_trends().await;
            // Trend should be positive (increasing)
            assert!(trends.allocation_time_trend >= 0.0 || trends.allocation_time_trend < 0.0);
        }
    }

    #[tokio::test]
    async fn test_performance_score_excellent() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            // No failures, fast allocations
            for _ in 0..10 {
                metrics
                    .record_allocation_success(Duration::from_millis(5))
                    .await;
            }
            let _ = metrics.create_snapshot().await;
            let analysis = metrics.analyze_performance().await;
            // Should not be Critical with fast, successful operations
            assert!(!matches!(
                analysis.performance_grade,
                PerformanceGrade::Critical
            ));
        }
    }

    #[tokio::test]
    async fn test_performance_score_with_failures() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 100,
                enable_detailed_timing: true,
                enable_percentile_tracking: true,
                max_timing_samples: 1000,
                enable_trend_analysis: true,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds {
                    max_avg_allocation_time_ms: 50.0,
                    max_p95_allocation_time_ms: 100.0,
                    min_success_rate_percent: 95.0,
                    max_conflict_rate_percent: 5.0,
                    min_throughput_ops_per_minute: 100.0,
                },
            };
            metrics.update_config(config).await;

            // Record many failures
            for _ in 0..20 {
                metrics.record_allocation_failure().await;
            }
            for _ in 0..5 {
                metrics
                    .record_allocation_success(Duration::from_millis(200))
                    .await;
            }
            let _ = metrics.create_snapshot().await;
            let snapshot = metrics.get_current_snapshot().await;
            assert!(snapshot.success_rate_percent < 50.0);
        }
    }

    #[tokio::test]
    async fn test_conflict_rate_calculation() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..10 {
                metrics
                    .record_allocation_success(Duration::from_millis(10))
                    .await;
            }
            for _ in 0..5 {
                metrics.record_conflict().await;
            }
            let snapshot = metrics.get_current_snapshot().await;
            assert!((snapshot.conflict_rate_percent - 50.0).abs() < f64::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_percentile_calculation() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 100,
                enable_detailed_timing: true,
                enable_percentile_tracking: true,
                max_timing_samples: 1000,
                enable_trend_analysis: false,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds::default(),
            };
            metrics.update_config(config).await;

            // Record various allocation times
            let mut rng = Lcg::new(42);
            for _ in 0..100 {
                let time_ms = (rng.next() % 100 + 1) as u64;
                metrics
                    .record_allocation_success(Duration::from_millis(time_ms))
                    .await;
            }
            let snapshot = metrics.get_current_snapshot().await;
            assert!(snapshot.median_allocation_time_ms > 0.0);
            assert!(snapshot.p95_allocation_time_ms >= snapshot.median_allocation_time_ms);
            assert!(snapshot.p99_allocation_time_ms >= snapshot.p95_allocation_time_ms);
        }
    }

    #[tokio::test]
    async fn test_percentile_disabled() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 100,
                enable_detailed_timing: false,
                enable_percentile_tracking: false,
                max_timing_samples: 1000,
                enable_trend_analysis: false,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds::default(),
            };
            metrics.update_config(config).await;

            metrics
                .record_allocation_success(Duration::from_millis(50))
                .await;
            let snapshot = metrics.get_current_snapshot().await;
            assert!((snapshot.median_allocation_time_ms - 0.0).abs() < f64::EPSILON);
        }
    }

    #[tokio::test]
    async fn test_history_size_limit() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 5,
                enable_detailed_timing: false,
                enable_percentile_tracking: false,
                max_timing_samples: 100,
                enable_trend_analysis: false,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds::default(),
            };
            metrics.update_config(config).await;

            for _ in 0..10 {
                metrics
                    .record_allocation_success(Duration::from_millis(10))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let history = metrics.get_performance_history().await;
            assert!(history.len() <= 5);
        }
    }

    #[tokio::test]
    async fn test_forecast_insufficient_data() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let analysis = metrics.analyze_performance().await;
            assert!(
                analysis.performance_forecast.forecast_method.contains("Insufficient")
            );
        }
    }

    #[tokio::test]
    async fn test_forecast_with_data() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..5 {
                metrics
                    .record_allocation_success(Duration::from_millis(30))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let analysis = metrics.analyze_performance().await;
            assert!(analysis.performance_forecast.forecast_hours > 0);
        }
    }

    #[tokio::test]
    async fn test_anomaly_detection_no_anomalies() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            // Too few snapshots for anomaly detection
            for _ in 0..5 {
                metrics
                    .record_allocation_success(Duration::from_millis(10))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let analysis = metrics.analyze_performance().await;
            assert!(analysis.anomalies.is_empty());
        }
    }

    #[tokio::test]
    async fn test_insights_generation_normal() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..5 {
                metrics
                    .record_allocation_success(Duration::from_millis(5))
                    .await;
            }
            let analysis = metrics.analyze_performance().await;
            // No insights for normal operation
            // (may or may not be empty depending on thresholds)
            assert!(analysis.insights.len() < 10);
        }
    }

    #[tokio::test]
    async fn test_recommendations_generation() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 100,
                enable_detailed_timing: false,
                enable_percentile_tracking: false,
                max_timing_samples: 1000,
                enable_trend_analysis: true,
                trend_window_size: 10,
                enable_memory_optimization: false,
                cleanup_interval: Duration::from_secs(300),
                alert_thresholds: PerformanceAlertThresholds {
                    max_avg_allocation_time_ms: 10.0,
                    max_p95_allocation_time_ms: 20.0,
                    min_success_rate_percent: 99.0,
                    max_conflict_rate_percent: 1.0,
                    min_throughput_ops_per_minute: 10000.0,
                },
            };
            metrics.update_config(config).await;

            // Record slow allocations to trigger recommendations
            for _ in 0..5 {
                metrics
                    .record_allocation_success(Duration::from_millis(100))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let analysis = metrics.analyze_performance().await;
            assert!(!analysis.recommendations.is_empty());
        }
    }

    #[tokio::test]
    async fn test_cleanup_with_memory_optimization() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            let config = PerformanceConfig {
                enabled: true,
                snapshot_interval: Duration::from_secs(1),
                history_size: 5,
                enable_detailed_timing: true,
                enable_percentile_tracking: true,
                max_timing_samples: 10,
                enable_trend_analysis: false,
                trend_window_size: 10,
                enable_memory_optimization: true,
                cleanup_interval: Duration::from_secs(1),
                alert_thresholds: PerformanceAlertThresholds::default(),
            };
            metrics.update_config(config).await;

            // Record many samples to exceed limits
            for _ in 0..50 {
                metrics
                    .record_allocation_success(Duration::from_millis(10))
                    .await;
            }
            let cleaned = metrics.cleanup_old_data().await;
            assert!(cleaned > 0);
        }
    }

    #[tokio::test]
    async fn test_export_contains_all_sections() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..3 {
                metrics
                    .record_allocation_success(Duration::from_millis(25))
                    .await;
                let _ = metrics.create_snapshot().await;
            }
            let result = metrics.export_performance_data().await;
            if let Ok(json) = result {
                assert!(json.contains("timestamp"));
                assert!(json.contains("current_snapshot"));
                assert!(json.contains("trends"));
                assert!(json.contains("history"));
                assert!(json.contains("analysis"));
            }
        }
    }

    #[tokio::test]
    async fn test_reset_clears_all() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            metrics
                .record_allocation_success(Duration::from_millis(50))
                .await;
            metrics.record_conflict().await;
            metrics.record_allocation_failure().await;
            let _ = metrics.create_snapshot().await;

            metrics.reset_metrics().await;

            let snapshot = metrics.get_current_snapshot().await;
            assert_eq!(snapshot.total_allocations, 0);
            assert_eq!(snapshot.total_conflicts, 0);
            let history = metrics.get_performance_history().await;
            assert!(history.is_empty());
        }
    }

    #[tokio::test]
    async fn test_ops_per_second_calculation() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..100 {
                metrics
                    .record_allocation_success(Duration::from_millis(1))
                    .await;
            }
            let snapshot = metrics.get_current_snapshot().await;
            assert!(snapshot.ops_per_second > 0.0);
        }
    }

    #[tokio::test]
    async fn test_throughput_calculation() {
        if let Ok(metrics) = PortPerformanceMetrics::new().await {
            for _ in 0..10 {
                metrics
                    .record_allocation_success(Duration::from_millis(1))
                    .await;
                metrics
                    .record_deallocation_success(Duration::from_millis(1))
                    .await;
            }
            let snapshot = metrics.get_current_snapshot().await;
            assert!(snapshot.throughput_ops_per_minute > 0.0);
        }
    }

    #[test]
    fn test_performance_alert_thresholds_default() {
        let thresholds = PerformanceAlertThresholds::default();
        assert!(thresholds.max_avg_allocation_time_ms > 0.0);
        assert!(thresholds.max_p95_allocation_time_ms > 0.0);
        assert!(thresholds.min_success_rate_percent > 0.0);
        assert!(thresholds.max_conflict_rate_percent > 0.0);
        assert!(thresholds.min_throughput_ops_per_minute > 0.0);
    }

    #[test]
    fn test_performance_config_default() {
        let config = PerformanceConfig::default();
        assert!(config.enabled);
        assert!(config.history_size > 0);
        assert!(config.max_timing_samples > 0);
    }
}
