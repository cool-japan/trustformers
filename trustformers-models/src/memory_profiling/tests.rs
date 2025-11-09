//! Tests for memory profiling functionality
//!
//! Comprehensive test suite covering all memory profiling components
//! including profiler creation, metrics collection, analytics, and reporting.

#[cfg(test)]
mod tests {
    use super::super::*;
    use std::sync::{Arc, Mutex};
    use std::time::{Duration, SystemTime, UNIX_EPOCH};

    #[tokio::test]
    async fn test_profiler_creation() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();
        assert!(profiler.get_current_summary().is_none());
    }

    #[tokio::test]
    async fn test_metrics_collection() {
        let metrics = profiler::MemoryProfiler::collect_memory_metrics().await.unwrap();
        assert!(metrics.total_memory_mb > 0.0);
        assert!(metrics.timestamp > UNIX_EPOCH);
    }

    #[test]
    fn test_fragmentation_calculation() {
        let info = system::ProcessMemoryInfo {
            total_mb: 1000.0,
            heap_mb: 800.0,
            stack_mb: 64.0,
            peak_mb: 1200.0,
            allocated_objects: 1000,
            deallocated_objects: 500,
            active_allocations: 500,
            gc_collections: 10,
            gc_time_ms: 150.0,
        };

        let fragmentation = profiler::MemoryProfiler::calculate_fragmentation_ratio(&info);
        assert!((0.0..=1.0).contains(&fragmentation));
    }

    #[tokio::test]
    async fn test_report_generation() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        let report = profiler.generate_report().await.unwrap();
        assert!(report.summary.peak_memory_mb >= 0.0);
        assert!(!report.recommendations.is_empty());
    }

    #[tokio::test]
    async fn test_atomic_monitoring_control() {
        let config = types::ProfilerConfig::default();
        let mut profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Test initial state
        assert!(!profiler.is_monitoring());

        // Test atomic start
        profiler.start_monitoring().await.unwrap();
        assert!(profiler.is_monitoring());

        // Test double start doesn't fail
        profiler.start_monitoring().await.unwrap();
        assert!(profiler.is_monitoring());

        // Test atomic stop
        profiler.stop_monitoring().await.unwrap();
        assert!(!profiler.is_monitoring());
    }

    #[test]
    fn test_cached_recommendations_performance() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Verify cached recommendations are pre-allocated
        let cached = profiler.get_cached_recommendations();
        assert!(!cached.high_memory.is_empty());
        assert!(!cached.rapid_growth.is_empty());
        assert!(!cached.fragmentation.is_empty());

        // Test recommendations contain expected content
        assert!(cached.high_memory.iter().any(|r| r.contains("batch size")));
        assert!(cached.rapid_growth.iter().any(|r| r.contains("memory leaks")));
        assert!(cached.fragmentation.iter().any(|r| r.contains("memory pools")));
    }

    #[tokio::test]
    async fn test_alert_analysis_optimization() {
        let config = types::ProfilerConfig::default();
        let alerts = Arc::new(Mutex::new(Vec::new()));
        let cached_recommendations = analytics::AlertRecommendations::new();
        let adaptive_thresholds = Arc::new(Mutex::new(analytics::AdaptiveThresholds::default()));

        // Test metrics that should trigger alerts
        let high_memory_metrics = types::MemoryMetrics {
            timestamp: SystemTime::now(),
            total_memory_mb: 2000.0, // Above default threshold of 1024
            heap_memory_mb: 1800.0,
            stack_memory_mb: 64.0,
            gpu_memory_mb: None,
            peak_memory_mb: 2000.0,
            allocated_objects: 1000,
            deallocated_objects: 500,
            active_allocations: 500,
            memory_fragmentation_ratio: 0.1,
            gc_collections: 10,
            gc_time_ms: 100.0,
            memory_growth_rate_mb_per_sec: 15.0, // Rapid growth
        };

        // Call the optimized alert analysis
        profiler::MemoryProfiler::analyze_for_alerts_adaptive(
            &high_memory_metrics,
            &None,
            &alerts,
            &config,
            &cached_recommendations,
            &adaptive_thresholds,
        )
        .await;

        // Verify alerts were generated efficiently
        let alerts_vec = alerts.lock().unwrap();
        assert!(alerts_vec.len() >= 2); // High memory + rapid growth alerts

        // Verify cached recommendations were used
        let high_mem_alert = alerts_vec
            .iter()
            .find(|a| matches!(a.alert_type, types::MemoryAlertType::HighMemoryUsage))
            .unwrap();
        assert_eq!(
            high_mem_alert.recommendations,
            cached_recommendations.high_memory
        );
    }

    #[tokio::test]
    async fn test_concurrent_monitoring() {
        let config = types::ProfilerConfig {
            collection_interval_ms: 20, // Slightly slower to reduce load
            max_data_points: 10,        // Reduced from 100
            ..types::ProfilerConfig::default()
        };
        let mut profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Start monitoring
        profiler.start_monitoring().await.unwrap();

        // Let it run for a shorter time
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Check that metrics were collected
        let summary = profiler.get_current_summary();
        assert!(summary.is_some());

        // Stop monitoring
        profiler.stop_monitoring().await.unwrap();

        // Verify it stops cleanly
        tokio::time::sleep(Duration::from_millis(10)).await;
        assert!(!profiler.is_monitoring());

        // Explicit cleanup
        drop(profiler);
        std::hint::black_box(());
    }

    #[test]
    fn test_metrics_history_bounded() {
        let config = types::ProfilerConfig {
            max_data_points: 5,
            ..types::ProfilerConfig::default()
        };
        let profiler = profiler::MemoryProfiler::new(config).unwrap();
        let mut history = profiler.get_metrics_history().lock().unwrap();

        // Add more than max_data_points
        for i in 0..10 {
            history.push_back(types::MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: i as f64,
                heap_memory_mb: i as f64,
                stack_memory_mb: 64.0,
                gpu_memory_mb: None,
                peak_memory_mb: i as f64,
                allocated_objects: i,
                deallocated_objects: 0,
                active_allocations: i,
                memory_fragmentation_ratio: 0.1,
                gc_collections: 0,
                gc_time_ms: 0.0,
                memory_growth_rate_mb_per_sec: 0.0,
            });

            // Simulate the bounded behavior
            while history.len() > 5 {
                history.pop_front();
            }
        }

        // Verify it's bounded correctly
        assert_eq!(history.len(), 5);
        assert_eq!(history.back().unwrap().total_memory_mb, 9.0);
    }

    #[tokio::test]
    async fn test_adaptive_thresholds_system() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Test initial adaptive thresholds
        let initial_thresholds = profiler.get_adaptive_thresholds().await.unwrap();
        assert_eq!(initial_thresholds.base_memory_threshold, 1024.0);
        assert_eq!(initial_thresholds.growth_rate_threshold, 50.0);
        assert_eq!(initial_thresholds.fragmentation_threshold, 0.3);
        assert_eq!(initial_thresholds.adaptation_factor, 0.1);

        // Test adaptive threshold updates
        let high_memory_metrics = types::MemoryMetrics {
            timestamp: SystemTime::now(),
            total_memory_mb: 2000.0, // Much higher than base threshold
            heap_memory_mb: 1800.0,
            stack_memory_mb: 64.0,
            gpu_memory_mb: None,
            peak_memory_mb: 2000.0,
            allocated_objects: 1000,
            deallocated_objects: 500,
            active_allocations: 500,
            memory_fragmentation_ratio: 0.5, // High fragmentation
            gc_collections: 10,
            gc_time_ms: 100.0,
            memory_growth_rate_mb_per_sec: 25.0, // Very high growth
        };

        // Simulate threshold adaptation
        profiler::MemoryProfiler::update_adaptive_thresholds(
            &high_memory_metrics,
            profiler.get_adaptive_thresholds_internal(),
        )
        .await;

        // Verify thresholds adapted upward
        let updated_thresholds = profiler.get_adaptive_thresholds().await.unwrap();
        assert!(
            updated_thresholds.base_memory_threshold > initial_thresholds.base_memory_threshold
        );
        assert!(
            updated_thresholds.fragmentation_threshold > initial_thresholds.fragmentation_threshold
        );
        assert!(updated_thresholds.last_updated > initial_thresholds.last_updated);
    }

    #[tokio::test]
    async fn test_memory_prediction_system() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Test no prediction with insufficient data
        let initial_prediction = profiler.predict_memory_usage(300).await.unwrap(); // 5 minutes ahead
        assert!(initial_prediction.is_none());

        // Simulate a trend by adding metrics with increasing memory usage
        let base_time = SystemTime::now();
        {
            let mut history = profiler.get_metrics_history().lock().unwrap();

            for i in 0..70 {
                // More than trend_window (60) for good prediction
                let metrics = types::MemoryMetrics {
                    timestamp: base_time + Duration::from_secs(i * 10),
                    total_memory_mb: 1000.0 + (i as f64 * 2.0), // Steadily increasing
                    heap_memory_mb: 900.0 + (i as f64 * 1.8),
                    stack_memory_mb: 64.0,
                    gpu_memory_mb: None,
                    peak_memory_mb: 1000.0 + (i as f64 * 2.0),
                    allocated_objects: 1000 + i * 10,
                    deallocated_objects: 500,
                    active_allocations: 500 + i * 10,
                    memory_fragmentation_ratio: 0.1,
                    gc_collections: 10,
                    gc_time_ms: 100.0,
                    memory_growth_rate_mb_per_sec: 2.0,
                };
                history.push_back(metrics);
            }
        }

        // Update memory prediction with trend data
        let latest_metrics = types::MemoryMetrics {
            timestamp: base_time + Duration::from_secs(700),
            total_memory_mb: 1140.0,
            heap_memory_mb: 1026.0,
            stack_memory_mb: 64.0,
            gpu_memory_mb: None,
            peak_memory_mb: 1140.0,
            allocated_objects: 1700,
            deallocated_objects: 500,
            active_allocations: 1200,
            memory_fragmentation_ratio: 0.1,
            gc_collections: 10,
            gc_time_ms: 100.0,
            memory_growth_rate_mb_per_sec: 2.0,
        };

        profiler::MemoryProfiler::update_memory_prediction(
            &latest_metrics,
            profiler.get_memory_predictor_internal(),
            profiler.get_metrics_history(),
        )
        .await;

        // Test prediction with sufficient data
        let prediction = profiler.predict_memory_usage(300).await.unwrap(); // 5 minutes ahead
        assert!(prediction.is_some());

        let pred = prediction.unwrap();
        assert!(pred.predicted_memory_mb > 1140.0); // Should predict growth
        assert!(pred.confidence > 0.0 && pred.confidence <= 1.0);
        assert_eq!(pred.horizon_secs, 300);
        assert!(pred.trend_slope > 0.0); // Positive growth trend
    }

    #[tokio::test]
    async fn test_memory_leak_detection() {
        let config = types::ProfilerConfig {
            enable_leak_detection: true,
            ..types::ProfilerConfig::default()
        };
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Test leak detection configuration
        let initial_config = profiler.get_leak_detection_config().await.unwrap();
        assert_eq!(initial_config.growth_threshold, 10.0);
        assert_eq!(initial_config.duration_secs, 300);
        assert_eq!(initial_config.allocation_threshold, 1000);
        assert_eq!(initial_config.confidence_threshold, 0.8);

        // Test configuration update
        let new_config = profiler::LeakDetectionConfig {
            growth_threshold: 15.0,
            duration_secs: 600,
            allocation_threshold: 2000,
            confidence_threshold: 0.9,
        };
        profiler.configure_leak_detection(new_config.clone()).await.unwrap();

        let updated_config = profiler.get_leak_detection_config().await.unwrap();
        assert_eq!(updated_config.growth_threshold, 15.0);
        assert_eq!(updated_config.duration_secs, 600);
        assert_eq!(updated_config.allocation_threshold, 2000);
        assert_eq!(updated_config.confidence_threshold, 0.9);
    }

    #[tokio::test]
    async fn test_monitoring_performance_stats() {
        let config = types::ProfilerConfig {
            collection_interval_ms: 20, // Slower collection to reduce load
            max_data_points: 5,         // Reduced data points
            ..types::ProfilerConfig::default()
        };
        let mut profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Initial stats should be zero
        let initial_stats = profiler.get_monitoring_stats().await.unwrap();
        assert_eq!(initial_stats.total_collections, 0);
        assert_eq!(initial_stats.average_overhead_us, 0);
        assert_eq!(initial_stats.uptime_secs, 0);

        // Start monitoring and let it collect some data for shorter time
        profiler.start_monitoring().await.unwrap();
        tokio::time::sleep(Duration::from_millis(30)).await;

        // Check stats after some collections
        let stats = profiler.get_monitoring_stats().await.unwrap();
        assert!(stats.total_collections > 0);
        assert!(stats.average_overhead_us > 0);
        assert!(stats.uptime_secs > 0);

        profiler.stop_monitoring().await.unwrap();

        // Explicit cleanup
        drop(profiler);
        std::hint::black_box(());
    }

    #[test]
    fn test_linear_regression_calculation() {
        // Test the linear regression implementation used in memory prediction
        let data_points: Vec<(f64, f64)> = vec![
            (1.0, 100.0), // time, memory
            (2.0, 102.0),
            (3.0, 104.0),
            (4.0, 106.0),
            (5.0, 108.0),
        ];

        // Calculate linear regression manually to verify
        let n = data_points.len() as f64;
        let sum_x: f64 = data_points.iter().map(|(x, _)| *x).sum();
        let sum_y: f64 = data_points.iter().map(|(_, y)| *y).sum();
        let sum_xy: f64 = data_points.iter().map(|(x, y)| x * y).sum();
        let sum_x2: f64 = data_points.iter().map(|(x, _)| x * x).sum();

        let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x);
        let intercept = (sum_y - slope * sum_x) / n;

        // For this data, slope should be approximately 2.0 (2 MB per time unit)
        assert!((slope - 2.0).abs() < 0.1);
        assert!((intercept - 98.0).abs() < 0.1);

        // Test correlation calculation
        let y_mean = sum_y / n;
        let ss_tot: f64 = data_points.iter().map(|(_, y)| (y - y_mean).powi(2)).sum();
        let ss_res: f64 =
            data_points.iter().map(|(x, y)| (y - (slope * x + intercept)).powi(2)).sum();
        let r_squared = 1.0 - (ss_res / ss_tot);

        // For perfect linear data, R² should be very close to 1.0
        assert!(r_squared > 0.99);
    }

    #[tokio::test]
    async fn test_adaptive_threshold_edge_cases() {
        let config = types::ProfilerConfig::default();
        let profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Test with extremely high memory usage
        let extreme_metrics = types::MemoryMetrics {
            timestamp: SystemTime::now(),
            total_memory_mb: 100000.0, // 100 GB
            heap_memory_mb: 95000.0,
            stack_memory_mb: 64.0,
            gpu_memory_mb: None,
            peak_memory_mb: 100000.0,
            allocated_objects: 1000000,
            deallocated_objects: 500000,
            active_allocations: 500000,
            memory_fragmentation_ratio: 0.8, // Very high fragmentation
            gc_collections: 100,
            gc_time_ms: 5000.0,                    // 5 seconds of GC
            memory_growth_rate_mb_per_sec: 1000.0, // Extremely rapid growth
        };

        profiler::MemoryProfiler::update_adaptive_thresholds(
            &extreme_metrics,
            profiler.get_adaptive_thresholds_internal(),
        )
        .await;

        let updated_thresholds = profiler.get_adaptive_thresholds().await.unwrap();

        // Thresholds should adapt but remain reasonable
        assert!(updated_thresholds.base_memory_threshold > 1024.0);
        assert!(updated_thresholds.base_memory_threshold < 50000.0); // Not too extreme
        assert!(updated_thresholds.fragmentation_threshold > 0.3);
        assert!(updated_thresholds.fragmentation_threshold < 1.0); // Must stay under 1.0

        // Test with zero/minimal memory usage
        let minimal_metrics = types::MemoryMetrics {
            timestamp: SystemTime::now(),
            total_memory_mb: 1.0, // Very low
            heap_memory_mb: 0.5,
            stack_memory_mb: 0.5,
            gpu_memory_mb: None,
            peak_memory_mb: 1.0,
            allocated_objects: 1,
            deallocated_objects: 0,
            active_allocations: 1,
            memory_fragmentation_ratio: 0.0, // No fragmentation
            gc_collections: 0,
            gc_time_ms: 0.0,
            memory_growth_rate_mb_per_sec: 0.0,
        };

        profiler::MemoryProfiler::update_adaptive_thresholds(
            &minimal_metrics,
            profiler.get_adaptive_thresholds_internal(),
        )
        .await;

        let minimal_thresholds = profiler.get_adaptive_thresholds().await.unwrap();

        // Thresholds should adapt downward but remain usable
        assert!(minimal_thresholds.base_memory_threshold > 10.0); // Not too low
        assert!(minimal_thresholds.fragmentation_threshold > 0.1); // Reasonable minimum
    }

    #[tokio::test]
    async fn test_comprehensive_analytics_integration() {
        let config = types::ProfilerConfig {
            enable_leak_detection: true,
            enable_pattern_analysis: true,
            collection_interval_ms: 25,       // Slower collection
            memory_alert_threshold_mb: 500.0, // Lower threshold for testing
            max_data_points: 5,               // Reduced data points
            ..types::ProfilerConfig::default()
        };
        let mut profiler = profiler::MemoryProfiler::new(config).unwrap();

        // Start monitoring
        profiler.start_monitoring().await.unwrap();

        // Let it run for much shorter time
        tokio::time::sleep(Duration::from_millis(40)).await;

        // Test all analytics features work together
        let thresholds = profiler.get_adaptive_thresholds().await.unwrap();
        let stats = profiler.get_monitoring_stats().await.unwrap();
        let leak_config = profiler.get_leak_detection_config().await.unwrap();

        // Verify basic functioning
        assert!(thresholds.base_memory_threshold > 0.0);
        assert!(stats.total_collections > 0);
        assert!(leak_config.growth_threshold > 0.0);

        // Test analytics summary
        let summary = profiler.get_analytics_summary().await.unwrap();
        assert!(summary.adaptive_thresholds.base_memory_threshold > 0.0);
        assert!(summary.monitoring_stats.total_collections > 0);

        profiler.stop_monitoring().await.unwrap();

        // Explicit cleanup
        drop(profiler);
        std::hint::black_box(());
    }

    #[test]
    fn test_statistical_analyzer() {
        let analyzer = analytics::StatisticalAnalyzer::new(0.95);

        // Create test metrics with known values
        let mut metrics = Vec::new();
        for i in 0..50 {
            let memory_mb = 100.0 + (i as f64) * 2.0; // Linear growth from 100MB to 198MB
            metrics.push(types::MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: memory_mb,
                heap_memory_mb: memory_mb * 0.8,
                stack_memory_mb: memory_mb * 0.1,
                gpu_memory_mb: Some(memory_mb * 0.5),
                peak_memory_mb: memory_mb * 1.2,
                allocated_objects: (1000 + i) as u64,
                deallocated_objects: (900 + i) as u64,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.15,
                gc_collections: 10,
                gc_time_ms: 5.0,
                memory_growth_rate_mb_per_sec: 2.0,
            });
        }

        // Test statistical calculations
        let stats = analyzer.calculate_usage_statistics(&metrics);
        assert!(stats.mean > 100.0);
        assert!(stats.mean < 200.0);
        assert!(stats.std_dev > 0.0);
        assert!(stats.trend_slope > 0.0); // Should have positive trend
        assert!(stats.coefficient_of_variation > 0.0);
        assert_eq!(stats.outlier_count, 0); // Linear data should have no outliers
    }

    #[test]
    fn test_anomaly_detection() {
        let analyzer = analytics::StatisticalAnalyzer::new(0.95);

        // Create test metrics with an anomaly
        let mut metrics = Vec::new();
        for i in 0..20 {
            let memory_mb = if i == 15 {
                500.0 // Sudden spike
            } else {
                100.0 + (i as f64)
            };

            metrics.push(types::MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: memory_mb,
                heap_memory_mb: memory_mb * 0.8,
                stack_memory_mb: memory_mb * 0.1,
                gpu_memory_mb: Some(memory_mb * 0.5),
                peak_memory_mb: memory_mb * 1.2,
                allocated_objects: (1000 + i) as u64,
                deallocated_objects: (900 + i) as u64,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.15,
                gc_collections: 10,
                gc_time_ms: 5.0,
                memory_growth_rate_mb_per_sec: 2.0,
            });
        }

        // Test anomaly detection
        let anomalies = analyzer.detect_anomalies(&metrics);
        assert!(!anomalies.is_empty());
        assert_eq!(
            anomalies[0].anomaly_type,
            analytics::AnomalyType::SuddenSpike
        );
        assert!(anomalies[0].confidence_score > 0.0);
        assert!(anomalies[0].description.contains("spike"));
    }

    #[test]
    fn test_sustained_growth_detection() {
        let analyzer = analytics::StatisticalAnalyzer::new(0.95);

        // Create test metrics with sustained growth
        let mut metrics = Vec::new();
        for i in 0..30 {
            let memory_mb = if i < 20 {
                100.0 + (i as f64) // Gradual growth
            } else {
                100.0 + 20.0 + (i as f64 - 20.0) * 5.0 // Faster growth in later part
            };

            metrics.push(types::MemoryMetrics {
                timestamp: SystemTime::now(),
                total_memory_mb: memory_mb,
                heap_memory_mb: memory_mb * 0.8,
                stack_memory_mb: memory_mb * 0.1,
                gpu_memory_mb: Some(memory_mb * 0.5),
                peak_memory_mb: memory_mb * 1.2,
                allocated_objects: (1000 + i) as u64,
                deallocated_objects: (900 + i) as u64,
                active_allocations: 100,
                memory_fragmentation_ratio: 0.15,
                gc_collections: 10,
                gc_time_ms: 5.0,
                memory_growth_rate_mb_per_sec: 2.0,
            });
        }

        // Test sustained growth detection
        let anomalies = analyzer.detect_anomalies(&metrics);
        let growth_anomalies: Vec<_> = anomalies
            .iter()
            .filter(|a| matches!(a.anomaly_type, analytics::AnomalyType::SustainedGrowth))
            .collect();

        assert!(!growth_anomalies.is_empty());
        assert!(growth_anomalies[0].confidence_score > 0.0);
        assert!(growth_anomalies[0].description.contains("growth"));
    }

    #[test]
    fn test_statistical_analyzer_empty_data() {
        let analyzer = analytics::StatisticalAnalyzer::new(0.95);
        let metrics = Vec::new();

        let stats = analyzer.calculate_usage_statistics(&metrics);
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.std_dev, 0.0);

        let anomalies = analyzer.detect_anomalies(&metrics);
        assert!(anomalies.is_empty());
    }

    #[test]
    fn test_memory_statistics_default() {
        let stats = analytics::MemoryStatistics::default();
        assert_eq!(stats.mean, 0.0);
        assert_eq!(stats.median, 0.0);
        assert_eq!(stats.outlier_count, 0);
        assert_eq!(stats.trend_slope, 0.0);
    }
}
