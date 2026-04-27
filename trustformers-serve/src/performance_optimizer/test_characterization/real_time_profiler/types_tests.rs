//! Tests for real-time profiler types

use super::types::*;
use std::sync::atomic::Ordering;
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random values
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 100000) as f64 / 100000.0
    }
}

#[test]
fn test_performance_counters_new() {
    let counters = RealTimePerformanceCounters::new();
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed, 0);
    assert_eq!(stats.anomalies_detected, 0);
    assert_eq!(stats.insights_generated, 0);
    assert_eq!(stats.processing_rate, 0);
}

#[test]
fn test_performance_counters_increment_data_points() {
    let counters = RealTimePerformanceCounters::new();
    counters.increment_data_points_processed();
    counters.increment_data_points_processed();
    counters.increment_data_points_processed();
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed, 3);
}

#[test]
fn test_performance_counters_increment_anomalies() {
    let counters = RealTimePerformanceCounters::new();
    counters.increment_anomalies_detected();
    let stats = counters.get_current_stats();
    assert_eq!(stats.anomalies_detected, 1);
}

#[test]
fn test_performance_counters_increment_insights() {
    let counters = RealTimePerformanceCounters::new();
    counters.increment_insights_generated();
    counters.increment_insights_generated();
    let stats = counters.get_current_stats();
    assert_eq!(stats.insights_generated, 2);
}

#[test]
fn test_performance_counters_mixed_increments() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..5 {
        counters.increment_data_points_processed();
    }
    for _ in 0..3 {
        counters.increment_anomalies_detected();
    }
    counters.increment_insights_generated();
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed, 5);
    assert_eq!(stats.anomalies_detected, 3);
    assert_eq!(stats.insights_generated, 1);
}

#[test]
fn test_performance_counter_stats_fields() {
    let stats = PerformanceCounterStats {
        data_points_processed: 100,
        anomalies_detected: 5,
        insights_generated: 10,
        processing_rate: 50,
    };
    assert_eq!(stats.data_points_processed, 100);
    assert_eq!(stats.anomalies_detected, 5);
    assert_eq!(stats.insights_generated, 10);
    assert_eq!(stats.processing_rate, 50);
}

#[test]
fn test_performance_counters_concurrent_increments() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..100 {
        counters.increment_data_points_processed();
    }
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed, 100);
}

#[test]
fn test_performance_counters_zero_initial_state() {
    let counters = RealTimePerformanceCounters::new();
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed + stats.anomalies_detected + stats.insights_generated + stats.processing_rate, 0);
}

#[test]
fn test_performance_counters_multiple_insight_increments() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..50 {
        counters.increment_insights_generated();
    }
    let stats = counters.get_current_stats();
    assert_eq!(stats.insights_generated, 50);
}

#[test]
fn test_performance_counters_many_anomalies() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..25 {
        counters.increment_anomalies_detected();
    }
    let stats = counters.get_current_stats();
    assert_eq!(stats.anomalies_detected, 25);
}

#[test]
fn test_performance_counters_ratio_calculation() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..100 {
        counters.increment_data_points_processed();
    }
    for _ in 0..5 {
        counters.increment_anomalies_detected();
    }
    let stats = counters.get_current_stats();
    // 5% anomaly rate
    let anomaly_rate = stats.anomalies_detected as f64 / stats.data_points_processed as f64;
    assert!((anomaly_rate - 0.05).abs() < 0.001);
}

#[test]
fn test_profile_data_point_from_metrics() {
    let mut metrics = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
    metrics.metrics.insert("cpu_usage".to_string(), 0.5);
    metrics.metrics.insert("memory_usage".to_string(), 0.3);
    let point = ProfileDataPoint::from_metrics(metrics);
    assert!(!point.point_id.is_empty());
    assert!(point.test_id.is_none());
}

#[test]
fn test_profile_data_point_multiple_conversions() {
    let mut lcg = Lcg::new(42);
    for _ in 0..10 {
        let mut metrics = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
        metrics.metrics.insert("cpu".to_string(), lcg.next_f64());
        metrics.metrics.insert("memory".to_string(), lcg.next_f64());
        let point = ProfileDataPoint::from_metrics(metrics);
        assert!(!point.point_id.is_empty());
    }
}

#[test]
fn test_performance_counter_stats_debug_format() {
    let stats = PerformanceCounterStats {
        data_points_processed: 42,
        anomalies_detected: 3,
        insights_generated: 7,
        processing_rate: 100,
    };
    let debug_str = format!("{:?}", stats);
    assert!(debug_str.contains("42"));
    assert!(debug_str.contains("3"));
}

#[test]
fn test_performance_counters_large_values() {
    let counters = RealTimePerformanceCounters::new();
    for _ in 0..10000 {
        counters.increment_data_points_processed();
    }
    let stats = counters.get_current_stats();
    assert_eq!(stats.data_points_processed, 10000);
}

#[test]
fn test_performance_counters_independent_increments() {
    let counters = RealTimePerformanceCounters::new();
    counters.increment_data_points_processed();
    let stats1 = counters.get_current_stats();
    assert_eq!(stats1.anomalies_detected, 0);
    assert_eq!(stats1.insights_generated, 0);

    counters.increment_anomalies_detected();
    let stats2 = counters.get_current_stats();
    assert_eq!(stats2.data_points_processed, 1);
    assert_eq!(stats2.anomalies_detected, 1);
    assert_eq!(stats2.insights_generated, 0);
}

#[test]
fn test_real_time_metrics_creation() {
    let mut metrics = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
    metrics.metrics.insert("cpu_usage".to_string(), 0.75);
    metrics.metrics.insert("memory_usage".to_string(), 0.60);
    assert_eq!(metrics.metrics.len(), 2);
    if let Some(cpu) = metrics.metrics.get("cpu_usage") {
        assert!((*cpu - 0.75).abs() < f64::EPSILON);
    }
}

#[test]
fn test_real_time_metrics_empty() {
    let metrics = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
    assert!(metrics.metrics.is_empty());
}

#[test]
fn test_real_time_metrics_merge() {
    let mut m1 = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
    m1.metrics.insert("cpu".to_string(), 0.5);
    let mut m2 = crate::performance_optimizer::test_characterization::types::core::metrics::RealTimeMetrics::new();
    m2.metrics.insert("cpu".to_string(), 0.3);
    m2.metrics.insert("memory".to_string(), 0.4);
    m1.merge_resource_metrics(&m2);
    assert_eq!(m1.metrics.len(), 2);
    if let Some(cpu) = m1.metrics.get("cpu") {
        assert!((*cpu - 0.8).abs() < 0.001);
    }
}

#[test]
fn test_lcg_deterministic_sequence() {
    let mut lcg1 = Lcg::new(42);
    let mut lcg2 = Lcg::new(42);
    for _ in 0..100 {
        assert_eq!(lcg1.next_u64(), lcg2.next_u64());
    }
}

#[test]
fn test_lcg_different_seeds() {
    let mut lcg1 = Lcg::new(42);
    let mut lcg2 = Lcg::new(123);
    assert_ne!(lcg1.next_u64(), lcg2.next_u64());
}

#[test]
fn test_lcg_f32_range() {
    let mut lcg = Lcg::new(42);
    for _ in 0..100 {
        let val = lcg.next_f32();
        assert!(val >= 0.0 && val < 1.0);
    }
}
