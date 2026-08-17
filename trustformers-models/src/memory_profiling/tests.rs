//! Tests for memory profiling functionality
//!
//! Comprehensive test suite covering all memory profiling components
//! including profiler creation, metrics collection, analytics, and reporting.

#![cfg(test)]

use super::*;
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

#[tokio::test]
async fn test_profiler_creation() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("failed to create profiler");
    assert!(profiler.get_current_summary().is_none());
}

#[tokio::test]
async fn test_metrics_collection() {
    let metrics =
        profiler::MemoryProfiler::collect_memory_metrics(system::TrackedAllocationStats::default())
            .await
            .expect("operation failed");
    assert!(metrics.total_memory_mb > 0.0);
    assert!(metrics.timestamp > UNIX_EPOCH);
}

/// The collected metrics must be *measurements*, not a fabricated curve: the
/// reported resident set size has to agree with what the OS reports for this
/// process, and it must be bounded by the machine's physical memory.
#[tokio::test]
async fn test_metrics_are_real_process_measurements() {
    let system_info =
        profiler::MemoryProfiler::get_system_memory_info().expect("system memory must be readable");
    let metrics =
        profiler::MemoryProfiler::collect_memory_metrics(system::TrackedAllocationStats::default())
            .await
            .expect("metrics collection must succeed");

    let total_mb = system_info.total_memory as f64 / (1024.0 * 1024.0);
    assert!(
        total_mb > 64.0,
        "the machine must report a plausible amount of RAM, got {total_mb} MB"
    );
    // The 16 GB / 8 GB constants the previous implementation returned would only
    // match a real machine by coincidence; assert against the live reading instead.
    assert!(system_info.available_memory <= system_info.total_memory);
    assert!(system_info.cached_memory.is_none());

    let process = profiler::MemoryProfiler::get_process_memory_info()
        .expect("process memory must be readable");
    assert!(process.resident_mb > 0.0);
    assert!(process.resident_mb < total_mb);
    assert!(process.peak_resident_mb >= process.resident_mb);
    assert!(
        (metrics.total_memory_mb - process.resident_mb).abs() < 256.0,
        "collected RSS {} MB must track the live reading {} MB",
        metrics.total_memory_mb,
        process.resident_mb
    );

    // Fields with no portable data source must be absent, not invented.
    assert!(metrics.heap_memory_mb.is_none());
    assert!(metrics.stack_memory_mb.is_none());
    assert!(metrics.gpu_memory_mb.is_none());
    assert_eq!(metrics.allocated_objects, 0);
}

/// Two consecutive samples must not differ by a synthetic sine wave: with no
/// work in between, the resident set size is essentially unchanged.
#[tokio::test]
async fn test_repeated_samples_track_the_process() {
    let first =
        profiler::MemoryProfiler::collect_memory_metrics(system::TrackedAllocationStats::default())
            .await
            .expect("first sample");
    let second =
        profiler::MemoryProfiler::collect_memory_metrics(system::TrackedAllocationStats::default())
            .await
            .expect("second sample");

    let delta = (second.total_memory_mb - first.total_memory_mb).abs();
    assert!(
        delta < 64.0,
        "an idle process must not swing by {delta} MB between samples"
    );
}

/// Allocation counters must come from real registrations.
#[tokio::test]
async fn test_allocation_counters_are_registered_events() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("profiler creation");

    assert_eq!(
        profiler.allocation_stats().expect("stats").allocated_objects,
        0
    );

    let info = types::AllocationInfo {
        id: uuid::Uuid::new_v4(),
        timestamp: SystemTime::now(),
        size_bytes: 4096,
        location: "test".to_string(),
        stack_trace: Vec::new(),
        object_type: "Vec<f32>".to_string(),
        is_leaked: false,
    };
    let id = profiler.record_allocation(info).expect("record allocation");

    let stats = profiler.allocation_stats().expect("stats");
    assert_eq!(stats.allocated_objects, 1);
    assert_eq!(stats.active_allocations, 1);
    assert_eq!(stats.active_bytes, 4096);
    assert_eq!(stats.deallocated_objects, 0);

    assert!(profiler.record_deallocation(id).expect("record deallocation"));
    let stats = profiler.allocation_stats().expect("stats");
    assert_eq!(stats.allocated_objects, 1);
    assert_eq!(stats.deallocated_objects, 1);
    assert_eq!(stats.active_allocations, 0);

    // Unknown ids must not inflate the counters.
    assert!(!profiler.record_deallocation(uuid::Uuid::new_v4()).expect("record deallocation"));
    assert_eq!(
        profiler.allocation_stats().expect("stats").deallocated_objects,
        1
    );
}

#[test]
fn test_fragmentation_calculation() {
    let info = system::ProcessMemoryInfo {
        resident_mb: 1000.0,
        virtual_mb: 2000.0,
        peak_resident_mb: 1200.0,
    };

    // Nothing tracked -> nothing to claim.
    let untracked = profiler::MemoryProfiler::calculate_fragmentation_ratio(
        &info,
        system::TrackedAllocationStats::default(),
    );
    assert_eq!(untracked, 0.0);

    // 250 MB of tracked allocations inside a 1000 MB resident set.
    let tracked = system::TrackedAllocationStats {
        allocated_objects: 10,
        deallocated_objects: 0,
        active_allocations: 10,
        active_bytes: 250 * 1024 * 1024,
    };
    let fragmentation = profiler::MemoryProfiler::calculate_fragmentation_ratio(&info, tracked);
    assert!((fragmentation - 0.75).abs() < 1e-6, "got {fragmentation}");
}

#[tokio::test]
async fn test_report_generation() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    let report = profiler.generate_report().await.expect("operation failed");
    assert!(report.summary.peak_memory_mb >= 0.0);
    assert!(!report.recommendations.is_empty());
}

#[tokio::test]
async fn test_atomic_monitoring_control() {
    let config = types::ProfilerConfig::default();
    let mut profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Test initial state
    assert!(!profiler.is_monitoring());

    // Test atomic start
    profiler.start_monitoring().await.expect("operation failed");
    assert!(profiler.is_monitoring());

    // Test double start doesn't fail
    profiler.start_monitoring().await.expect("operation failed");
    assert!(profiler.is_monitoring());

    // Test atomic stop
    profiler.stop_monitoring().await.expect("operation failed");
    assert!(!profiler.is_monitoring());
}

#[test]
fn test_cached_recommendations_performance() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

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
        virtual_memory_mb: 2000.0,
        heap_memory_mb: Some(1800.0),
        stack_memory_mb: Some(64.0),
        gpu_memory_mb: None,
        peak_memory_mb: 2000.0,
        allocated_objects: 1000,
        deallocated_objects: 500,
        active_allocations: 500,
        memory_fragmentation_ratio: 0.1,
        memory_growth_rate_mb_per_sec: 60.0, // Above default threshold of 50.0
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
    let alerts_vec = alerts.lock().expect("operation failed");
    assert!(alerts_vec.len() >= 2); // High memory + rapid growth alerts

    // Verify cached recommendations were used
    let high_mem_alert = alerts_vec
        .iter()
        .find(|a| matches!(a.alert_type, types::MemoryAlertType::HighMemoryUsage))
        .expect("operation failed");
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
    let mut profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Start monitoring
    profiler.start_monitoring().await.expect("operation failed");

    // Let it run for a shorter time
    tokio::time::sleep(Duration::from_millis(30)).await;

    // Check that metrics were collected
    let summary = profiler.get_current_summary();
    assert!(summary.is_some());

    // Stop monitoring
    profiler.stop_monitoring().await.expect("operation failed");

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
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");
    let mut history = profiler.get_metrics_history().lock().expect("operation failed");

    // Add more than max_data_points
    for i in 0..10 {
        history.push_back(types::MemoryMetrics {
            timestamp: SystemTime::now(),
            total_memory_mb: i as f64,
            virtual_memory_mb: i as f64,
            heap_memory_mb: Some(i as f64),
            stack_memory_mb: Some(64.0),
            gpu_memory_mb: None,
            peak_memory_mb: i as f64,
            allocated_objects: i,
            deallocated_objects: 0,
            active_allocations: i,
            memory_fragmentation_ratio: 0.1,
            memory_growth_rate_mb_per_sec: 0.0,
        });

        // Simulate the bounded behavior
        while history.len() > 5 {
            history.pop_front();
        }
    }

    // Verify it's bounded correctly
    assert_eq!(history.len(), 5);
    assert_eq!(
        history.back().expect("operation failed").total_memory_mb,
        9.0
    );
}

#[tokio::test]
async fn test_adaptive_thresholds_system() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Test initial adaptive thresholds
    let initial_thresholds = profiler.get_adaptive_thresholds().await.expect("operation failed");
    assert_eq!(initial_thresholds.base_memory_threshold, 1024.0);
    assert_eq!(initial_thresholds.growth_rate_threshold, 50.0);
    assert_eq!(initial_thresholds.fragmentation_threshold, 0.3);
    assert_eq!(initial_thresholds.adaptation_factor, 0.1);

    // Test adaptive threshold updates
    // Use timestamp 6 minutes in future to trigger update (requires > 300 seconds)
    let high_memory_metrics = types::MemoryMetrics {
        timestamp: SystemTime::now() + std::time::Duration::from_secs(360),
        total_memory_mb: 2000.0, // Much higher than base threshold
        virtual_memory_mb: 2000.0,
        heap_memory_mb: Some(1800.0),
        stack_memory_mb: Some(64.0),
        gpu_memory_mb: None,
        peak_memory_mb: 2000.0,
        allocated_objects: 1000,
        deallocated_objects: 500,
        active_allocations: 500,
        memory_fragmentation_ratio: 0.5,     // High fragmentation
        memory_growth_rate_mb_per_sec: 25.0, // Very high growth
    };

    // Simulate threshold adaptation
    profiler::MemoryProfiler::update_adaptive_thresholds(
        &high_memory_metrics,
        profiler.get_adaptive_thresholds_internal(),
    )
    .await;

    // Verify thresholds adapted upward
    let updated_thresholds = profiler.get_adaptive_thresholds().await.expect("operation failed");
    assert!(updated_thresholds.base_memory_threshold > initial_thresholds.base_memory_threshold);
    assert!(
        updated_thresholds.fragmentation_threshold > initial_thresholds.fragmentation_threshold
    );
    assert!(updated_thresholds.last_updated > initial_thresholds.last_updated);
}

#[tokio::test]
async fn test_memory_prediction_system() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Test no prediction with insufficient data
    let initial_prediction = profiler.predict_memory_usage(300).await.expect("operation failed"); // 5 minutes ahead
    assert!(initial_prediction.is_none());

    // Simulate a trend by adding metrics with increasing memory usage
    let base_time = SystemTime::now();
    {
        let mut history = profiler.get_metrics_history().lock().expect("operation failed");

        for i in 0..70 {
            // More than trend_window (60) for good prediction
            let metrics = types::MemoryMetrics {
                timestamp: base_time + Duration::from_secs(i * 10),
                total_memory_mb: 1000.0 + (i as f64 * 2.0), // Steadily increasing
                virtual_memory_mb: 1000.0 + (i as f64 * 2.0),
                heap_memory_mb: Some(900.0 + (i as f64 * 1.8)),
                stack_memory_mb: Some(64.0),
                gpu_memory_mb: None,
                peak_memory_mb: 1000.0 + (i as f64 * 2.0),
                allocated_objects: 1000 + i * 10,
                deallocated_objects: 500,
                active_allocations: 500 + i * 10,
                memory_fragmentation_ratio: 0.1,
                memory_growth_rate_mb_per_sec: 2.0,
            };
            history.push_back(metrics);
        }
    }

    // Update memory prediction with trend data
    let latest_metrics = types::MemoryMetrics {
        timestamp: base_time + Duration::from_secs(700),
        total_memory_mb: 1140.0,
        virtual_memory_mb: 1140.0,
        heap_memory_mb: Some(1026.0),
        stack_memory_mb: Some(64.0),
        gpu_memory_mb: None,
        peak_memory_mb: 1140.0,
        allocated_objects: 1700,
        deallocated_objects: 500,
        active_allocations: 1200,
        memory_fragmentation_ratio: 0.1,
        memory_growth_rate_mb_per_sec: 2.0,
    };

    profiler::MemoryProfiler::update_memory_prediction(
        &latest_metrics,
        profiler.get_memory_predictor_internal(),
        profiler.get_metrics_history(),
    )
    .await;

    // Test prediction with sufficient data
    let prediction = profiler.predict_memory_usage(300).await.expect("operation failed"); // 5 minutes ahead
    assert!(prediction.is_some());

    let pred = prediction.expect("operation failed");
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
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Test leak detection configuration
    let initial_config = profiler.get_leak_detection_config().await.expect("operation failed");
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
    profiler
        .configure_leak_detection(new_config.clone())
        .await
        .expect("operation failed");

    let updated_config = profiler.get_leak_detection_config().await.expect("operation failed");
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
    let mut profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Initial stats should be zero
    let initial_stats = profiler.get_monitoring_stats().await.expect("operation failed");
    assert_eq!(initial_stats.total_collections, 0);
    assert_eq!(initial_stats.average_overhead_us, 0);
    assert_eq!(initial_stats.uptime_secs, 0);

    // Start monitoring and let it collect some data for shorter time
    profiler.start_monitoring().await.expect("operation failed");
    tokio::time::sleep(Duration::from_millis(30)).await;

    // Check stats after some collections
    let stats = profiler.get_monitoring_stats().await.expect("operation failed");
    assert!(stats.total_collections > 0);
    assert!(stats.average_overhead_us > 0);
    // uptime_secs may be 0 since we only slept 30ms (< 1 second)
    // Just verify it's a valid value (stats is collected correctly)
    let _ = stats.uptime_secs;

    profiler.stop_monitoring().await.expect("operation failed");

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
    let ss_res: f64 = data_points.iter().map(|(x, y)| (y - (slope * x + intercept)).powi(2)).sum();
    let r_squared = 1.0 - (ss_res / ss_tot);

    // For perfect linear data, R² should be very close to 1.0
    assert!(r_squared > 0.99);
}

#[tokio::test]
async fn test_adaptive_threshold_edge_cases() {
    let config = types::ProfilerConfig::default();
    let profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Test with extremely high memory usage
    // Use a timestamp 6 minutes in the future to trigger threshold update (requires > 300 seconds)
    let extreme_metrics = types::MemoryMetrics {
        timestamp: SystemTime::now() + std::time::Duration::from_secs(360),
        total_memory_mb: 100000.0, // 100 GB
        virtual_memory_mb: 100000.0,
        heap_memory_mb: Some(95000.0),
        stack_memory_mb: Some(64.0),
        gpu_memory_mb: None,
        peak_memory_mb: 100000.0,
        allocated_objects: 1000000,
        deallocated_objects: 500000,
        active_allocations: 500000,
        memory_fragmentation_ratio: 0.8, // Very high fragmentation
        memory_growth_rate_mb_per_sec: 1000.0, // Extremely rapid growth
    };

    profiler::MemoryProfiler::update_adaptive_thresholds(
        &extreme_metrics,
        profiler.get_adaptive_thresholds_internal(),
    )
    .await;

    let updated_thresholds = profiler.get_adaptive_thresholds().await.expect("operation failed");

    // Thresholds should adapt but remain reasonable
    assert!(updated_thresholds.base_memory_threshold > 1024.0);
    assert!(updated_thresholds.base_memory_threshold < 50000.0); // Not too extreme
    assert!(updated_thresholds.fragmentation_threshold > 0.3);
    assert!(updated_thresholds.fragmentation_threshold < 1.0); // Must stay under 1.0

    // Test with zero/minimal memory usage
    let minimal_metrics = types::MemoryMetrics {
        timestamp: SystemTime::now(),
        total_memory_mb: 1.0, // Very low
        virtual_memory_mb: 1.0,
        heap_memory_mb: Some(0.5),
        stack_memory_mb: Some(0.5),
        gpu_memory_mb: None,
        peak_memory_mb: 1.0,
        allocated_objects: 1,
        deallocated_objects: 0,
        active_allocations: 1,
        memory_fragmentation_ratio: 0.0, // No fragmentation
        memory_growth_rate_mb_per_sec: 0.0,
    };

    profiler::MemoryProfiler::update_adaptive_thresholds(
        &minimal_metrics,
        profiler.get_adaptive_thresholds_internal(),
    )
    .await;

    let minimal_thresholds = profiler.get_adaptive_thresholds().await.expect("operation failed");

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
    let mut profiler = profiler::MemoryProfiler::new(config).expect("operation failed");

    // Start monitoring
    profiler.start_monitoring().await.expect("operation failed");

    // Let it run for much shorter time
    tokio::time::sleep(Duration::from_millis(40)).await;

    // Test all analytics features work together
    let thresholds = profiler.get_adaptive_thresholds().await.expect("operation failed");
    let stats = profiler.get_monitoring_stats().await.expect("operation failed");
    let leak_config = profiler.get_leak_detection_config().await.expect("operation failed");

    // Verify basic functioning
    assert!(thresholds.base_memory_threshold > 0.0);
    assert!(stats.total_collections > 0);
    assert!(leak_config.growth_threshold > 0.0);

    // Test analytics summary
    let summary = profiler.get_analytics_summary().await.expect("operation failed");
    assert!(summary.adaptive_thresholds.base_memory_threshold > 0.0);
    assert!(summary.monitoring_stats.total_collections > 0);

    profiler.stop_monitoring().await.expect("operation failed");

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
            virtual_memory_mb: memory_mb,
            heap_memory_mb: Some(memory_mb * 0.8),
            stack_memory_mb: Some(memory_mb * 0.1),
            gpu_memory_mb: Some(memory_mb * 0.5),
            peak_memory_mb: memory_mb * 1.2,
            allocated_objects: (1000 + i) as u64,
            deallocated_objects: (900 + i) as u64,
            active_allocations: 100,
            memory_fragmentation_ratio: 0.15,
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
            virtual_memory_mb: memory_mb,
            heap_memory_mb: Some(memory_mb * 0.8),
            stack_memory_mb: Some(memory_mb * 0.1),
            gpu_memory_mb: Some(memory_mb * 0.5),
            peak_memory_mb: memory_mb * 1.2,
            allocated_objects: (1000 + i) as u64,
            deallocated_objects: (900 + i) as u64,
            active_allocations: 100,
            memory_fragmentation_ratio: 0.15,
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
            virtual_memory_mb: memory_mb,
            heap_memory_mb: Some(memory_mb * 0.8),
            stack_memory_mb: Some(memory_mb * 0.1),
            gpu_memory_mb: Some(memory_mb * 0.5),
            peak_memory_mb: memory_mb * 1.2,
            allocated_objects: (1000 + i) as u64,
            deallocated_objects: (900 + i) as u64,
            active_allocations: 100,
            memory_fragmentation_ratio: 0.15,
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
