//! Tests for Advanced Statistics Computer
//!
//! Comprehensive tests for statistical analysis functions including
//! descriptive statistics, distribution analysis, correlation,
//! time series, percentiles, confidence intervals, and outlier detection.

use super::*;
use crate::performance_optimizer::types::{PerformanceDataPoint, SystemState, TestCharacteristics};
use std::time::Duration;

/// Simple LCG for deterministic pseudo-random generation
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

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn make_data_point(throughput: f64, latency_ms: u64, cpu: f32, mem: f32) -> PerformanceDataPoint {
    PerformanceDataPoint {
        parallelism: 1,
        throughput,
        latency: Duration::from_millis(latency_ms),
        cpu_utilization: cpu,
        memory_utilization: mem,
        resource_efficiency: 0.8,
        timestamp: chrono::Utc::now(),
        test_characteristics: TestCharacteristics::default(),
        system_state: SystemState::default(),
    }
}

fn generate_lcg_values(seed: u64, count: usize, scale: f64) -> Vec<f64> {
    let mut lcg = Lcg::new(seed);
    (0..count).map(|_| lcg.next_f64() * scale).collect()
}

// =============================================================================
// ADVANCED STATISTICS COMPUTER TESTS
// =============================================================================

#[test]
fn test_statistics_computer_new() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = computer.compute_basic_statistics(&values);
    assert!(result.is_ok());
}

#[test]
fn test_statistics_computer_with_config() {
    let config = StatisticsConfig {
        enable_advanced_metrics: true,
        window_size: 50,
        enable_correlation_analysis: true,
        enable_distribution_analysis: true,
        significance_threshold: 0.05,
    };
    let computer = AdvancedStatisticsComputer::with_config(config);
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = computer.compute_basic_statistics(&values);
    assert!(result.is_ok());
}

#[test]
fn test_statistics_computer_default() {
    let computer = AdvancedStatisticsComputer::default();
    let values = vec![10.0, 20.0, 30.0];
    let result = computer.compute_basic_statistics(&values);
    assert!(result.is_ok());
}

#[test]
fn test_basic_statistics_simple_sequence() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("basic stats should succeed");
    let epsilon = 1e-10;
    assert!((s.mean - 3.0).abs() < epsilon, "mean should be 3.0");
    assert!((s.median - 3.0).abs() < epsilon, "median should be 3.0");
    assert!((s.min - 1.0).abs() < epsilon, "min should be 1.0");
    assert!((s.max - 5.0).abs() < epsilon, "max should be 5.0");
    assert!((s.range - 4.0).abs() < epsilon, "range should be 4.0");
    assert!(s.std_dev > 0.0, "std_dev should be positive");
    assert!(s.variance > 0.0, "variance should be positive");
}

#[test]
fn test_basic_statistics_single_value() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![42.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("single value stats should succeed");
    assert!((s.mean - 42.0).abs() < 1e-10);
    assert!((s.median - 42.0).abs() < 1e-10);
    assert!((s.min - 42.0).abs() < 1e-10);
    assert!((s.max - 42.0).abs() < 1e-10);
    assert!((s.range).abs() < 1e-10);
    assert!((s.std_dev).abs() < 1e-10);
    assert!((s.variance).abs() < 1e-10);
}

#[test]
fn test_basic_statistics_empty_fails() {
    let computer = AdvancedStatisticsComputer::new();
    let values: Vec<f64> = vec![];
    let result = computer.compute_basic_statistics(&values);
    assert!(result.is_err());
}

#[test]
fn test_basic_statistics_even_count_median() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0, 4.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("even count stats should succeed");
    assert!(
        (s.median - 2.5).abs() < 1e-10,
        "median of [1,2,3,4] should be 2.5"
    );
}

#[test]
fn test_basic_statistics_negative_values() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![-5.0, -3.0, -1.0, 1.0, 3.0, 5.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("negative values stats should succeed");
    assert!(s.mean.abs() < 1e-10, "mean of symmetric range should be ~0");
    assert!((s.min - (-5.0)).abs() < 1e-10);
    assert!((s.max - 5.0).abs() < 1e-10);
    assert!((s.range - 10.0).abs() < 1e-10);
}

#[test]
fn test_basic_statistics_identical_values() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![7.0, 7.0, 7.0, 7.0, 7.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("identical values stats should succeed");
    assert!((s.mean - 7.0).abs() < 1e-10);
    assert!((s.std_dev).abs() < 1e-10);
    assert!((s.variance).abs() < 1e-10);
    assert!((s.range).abs() < 1e-10);
}

#[test]
fn test_basic_statistics_skewness_symmetric() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("skewness test should succeed");
    // Symmetric distribution should have skewness near zero
    assert!(
        s.skewness.abs() < 0.1,
        "skewness of symmetric data should be ~0"
    );
}

#[test]
fn test_basic_statistics_large_dataset() {
    let computer = AdvancedStatisticsComputer::new();
    let values = generate_lcg_values(12345, 1000, 100.0);
    let stats = computer.compute_basic_statistics(&values);
    assert!(stats.is_ok());
    let s = stats.expect("large dataset stats should succeed");
    assert!(s.mean > 0.0 && s.mean < 100.0);
    assert!(s.std_dev > 0.0);
    assert!(s.min >= 0.0);
    assert!(s.max <= 100.0);
}

#[test]
fn test_update_config() {
    let mut computer = AdvancedStatisticsComputer::new();
    let new_config = StatisticsConfig {
        enable_advanced_metrics: false,
        window_size: 200,
        enable_correlation_analysis: false,
        enable_distribution_analysis: false,
        significance_threshold: 0.01,
    };
    computer.update_config(new_config);
    // Verify it still works with new config
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = computer.compute_basic_statistics(&values);
    assert!(result.is_ok());
}

// =============================================================================
// DISTRIBUTION ANALYSIS TESTS
// =============================================================================

#[test]
fn test_distribution_analysis_normal_data() {
    let computer = AdvancedStatisticsComputer::new();
    // Generate roughly normal-like data using central limit theorem
    let mut lcg = Lcg::new(42);
    let values: Vec<f64> = (0..100)
        .map(|_| {
            let sum: f64 = (0..12).map(|_| lcg.next_f64()).sum();
            sum - 6.0 // Approximate normal(0,1)
        })
        .collect();
    let result = computer.analyze_distribution(&values);
    assert!(result.is_ok());
    let dist = result.expect("distribution analysis should succeed");
    assert!(dist.goodness_of_fit >= 0.0 && dist.goodness_of_fit <= 1.0);
}

#[test]
fn test_distribution_analysis_insufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0]; // Less than 10
    let result = computer.analyze_distribution(&values);
    assert!(result.is_err());
}

#[test]
fn test_distribution_analysis_uniform_data() {
    let computer = AdvancedStatisticsComputer::new();
    let values = generate_lcg_values(99, 50, 1.0);
    let result = computer.analyze_distribution(&values);
    assert!(result.is_ok());
    let dist = result.expect("uniform distribution analysis should succeed");
    assert!(dist.confidence_level > 0.0);
}

// =============================================================================
// CORRELATION ANALYSIS TESTS
// =============================================================================

#[test]
fn test_correlation_analysis_sufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let data_points: Vec<PerformanceDataPoint> = (0..10)
        .map(|i| make_data_point(100.0 + i as f64 * 10.0, 50 - i, 0.5, 0.5))
        .collect();
    let result = computer.analyze_correlations(&data_points);
    assert!(result.is_ok());
    let corr = result.expect("correlation analysis should succeed");
    assert!(corr.correlations.contains_key("throughput_latency"));
    assert!(corr.correlations.contains_key("throughput_time"));
    assert!(corr.correlations.contains_key("latency_time"));
    assert_eq!(corr.correlation_matrix.len(), 3);
}

#[test]
fn test_correlation_analysis_insufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let data_points: Vec<PerformanceDataPoint> =
        (0..2).map(|i| make_data_point(100.0 + i as f64, 50, 0.5, 0.5)).collect();
    let result = computer.analyze_correlations(&data_points);
    assert!(result.is_err());
}

// =============================================================================
// TIME SERIES ANALYSIS TESTS
// =============================================================================

#[test]
fn test_time_series_analysis_sufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let values: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 5.0).collect();
    let result = computer.analyze_time_series(&values);
    assert!(result.is_ok());
    let ts = result.expect("time series analysis should succeed");
    assert!(!ts.trend.is_empty());
    assert!(!ts.seasonal.is_empty());
    assert!(!ts.residual.is_empty());
}

#[test]
fn test_time_series_analysis_insufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let values = vec![1.0, 2.0, 3.0]; // Less than 4
    let result = computer.analyze_time_series(&values);
    assert!(result.is_err());
}

// =============================================================================
// STATISTICAL TESTS
// =============================================================================

#[test]
fn test_perform_statistical_tests_sufficient_data() {
    let computer = AdvancedStatisticsComputer::new();
    let throughputs: Vec<f64> = (0..20).map(|i| 100.0 + i as f64 * 5.0).collect();
    let latencies: Vec<f64> = (0..20).map(|i| 50.0 - i as f64 * 1.5).collect();
    let result = computer.perform_statistical_tests(&throughputs, &latencies);
    assert!(result.is_ok());
    let tests = result.expect("statistical tests should succeed");
    assert!(!tests.is_empty());
    for test in &tests {
        assert!(!test.test_name.is_empty());
    }
}

#[test]
fn test_perform_statistical_tests_small_dataset() {
    let computer = AdvancedStatisticsComputer::new();
    let throughputs = vec![1.0, 2.0];
    let latencies = vec![10.0, 20.0];
    let result = computer.perform_statistical_tests(&throughputs, &latencies);
    assert!(result.is_ok());
}

// =============================================================================
// COMPREHENSIVE STATISTICS TESTS
// =============================================================================

#[test]
fn test_comprehensive_statistics_with_enough_data() {
    let computer = AdvancedStatisticsComputer::with_config(StatisticsConfig {
        enable_advanced_metrics: true,
        window_size: 100,
        enable_correlation_analysis: true,
        enable_distribution_analysis: true,
        significance_threshold: 0.05,
    });
    let data_points: Vec<PerformanceDataPoint> = (0..20)
        .map(|i| make_data_point(100.0 + i as f64 * 10.0, 50 - (i % 40), 0.5, 0.5))
        .collect();
    let result = computer.compute_comprehensive_statistics(&data_points);
    assert!(result.is_ok());
    let stats = result.expect("comprehensive stats should succeed");
    assert!(stats.basic_stats.mean > 0.0);
}

#[test]
fn test_comprehensive_statistics_empty_fails() {
    let computer = AdvancedStatisticsComputer::new();
    let data_points: Vec<PerformanceDataPoint> = vec![];
    let result = computer.compute_comprehensive_statistics(&data_points);
    assert!(result.is_err());
}

#[test]
fn test_comprehensive_statistics_disabled_features() {
    let computer = AdvancedStatisticsComputer::with_config(StatisticsConfig {
        enable_advanced_metrics: false,
        window_size: 100,
        enable_correlation_analysis: false,
        enable_distribution_analysis: false,
        significance_threshold: 0.05,
    });
    let data_points: Vec<PerformanceDataPoint> =
        (0..5).map(|i| make_data_point(100.0 + i as f64, 50, 0.5, 0.5)).collect();
    let result = computer.compute_comprehensive_statistics(&data_points);
    assert!(result.is_ok());
}

// =============================================================================
// PUBLIC FUNCTION TESTS
// =============================================================================

#[test]
fn test_perform_descriptive_analysis_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = perform_descriptive_analysis(&values);
    assert!(result.is_ok());
    let analysis = result.expect("descriptive analysis should succeed");
    assert!(analysis.contains_key("mean"));
    assert!(analysis.contains_key("median"));
    assert!(analysis.contains_key("std_dev"));
    assert!(analysis.contains_key("variance"));
    assert!(analysis.contains_key("min"));
    assert!(analysis.contains_key("max"));
    assert!(analysis.contains_key("range"));
    assert!(analysis.contains_key("skewness"));
    assert!(analysis.contains_key("kurtosis"));
    if let Some(&mean) = analysis.get("mean") {
        assert!((mean - 3.0).abs() < 1e-10);
    }
}

#[test]
fn test_perform_descriptive_analysis_empty_fails() {
    let values: Vec<f64> = vec![];
    let result = perform_descriptive_analysis(&values);
    assert!(result.is_err());
}

#[test]
fn test_calculate_percentiles_basic() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let percentiles = vec![25.0, 50.0, 75.0];
    let result = calculate_percentiles(&values, &percentiles);
    assert!(result.is_ok());
    let p = result.expect("percentiles should succeed");
    assert_eq!(p.len(), 3);
}

#[test]
fn test_calculate_percentiles_empty_fails() {
    let values: Vec<f64> = vec![];
    let percentiles = vec![50.0];
    let result = calculate_percentiles(&values, &percentiles);
    assert!(result.is_err());
}

#[test]
fn test_calculate_percentiles_invalid_range() {
    let values = vec![1.0, 2.0, 3.0];
    let percentiles = vec![101.0]; // Invalid percentile
    let result = calculate_percentiles(&values, &percentiles);
    assert!(result.is_err());
}

#[test]
fn test_calculate_percentiles_zero_and_hundred() {
    let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
    let percentiles = vec![0.0, 100.0];
    let result = calculate_percentiles(&values, &percentiles);
    assert!(result.is_ok());
    let p = result.expect("boundary percentiles should succeed");
    assert_eq!(p.len(), 2);
}

#[test]
fn test_confidence_interval_basic() {
    let values = vec![10.0, 12.0, 11.0, 13.0, 9.0, 11.0, 10.0, 12.0];
    let result = calculate_confidence_interval(&values, 0.95);
    assert!(result.is_ok());
    let (lower, upper) = result.expect("confidence interval should succeed");
    assert!(lower < upper, "lower bound should be less than upper bound");
    let mean = values.iter().sum::<f64>() / values.len() as f64;
    assert!(
        lower < mean && mean < upper,
        "mean should be within interval"
    );
}

#[test]
fn test_confidence_interval_insufficient_data() {
    let values = vec![1.0]; // Need at least 2
    let result = calculate_confidence_interval(&values, 0.95);
    assert!(result.is_err());
}

#[test]
fn test_confidence_interval_large_sample() {
    let values = generate_lcg_values(777, 100, 10.0);
    let result = calculate_confidence_interval(&values, 0.99);
    assert!(result.is_ok());
    let (lower, upper) = result.expect("large sample CI should succeed");
    assert!(lower < upper);
}

#[test]
fn test_confidence_interval_different_levels() {
    let values = generate_lcg_values(111, 50, 10.0);
    let ci_90 = calculate_confidence_interval(&values, 0.90);
    let ci_95 = calculate_confidence_interval(&values, 0.95);
    let ci_99 = calculate_confidence_interval(&values, 0.99);
    assert!(ci_90.is_ok());
    assert!(ci_95.is_ok());
    assert!(ci_99.is_ok());
    let (l90, u90) = ci_90.expect("90% CI should succeed");
    let (l95, u95) = ci_95.expect("95% CI should succeed");
    let (l99, u99) = ci_99.expect("99% CI should succeed");
    // Higher confidence => wider interval
    assert!((u90 - l90) <= (u95 - l95) + 1e-10);
    assert!((u95 - l95) <= (u99 - l99) + 1e-10);
}

// =============================================================================
// OUTLIER DETECTION TESTS
// =============================================================================

#[test]
fn test_outlier_detection_iqr_with_outliers() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 100.0];
    let result = detect_outliers_iqr(&values, 1.5);
    assert!(result.is_ok());
    let outliers = result.expect("IQR outlier detection should succeed");
    assert!(!outliers.is_empty(), "should detect the outlier 100.0");
}

#[test]
fn test_outlier_detection_iqr_no_outliers() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let result = detect_outliers_iqr(&values, 1.5);
    assert!(result.is_ok());
    let outliers = result.expect("IQR no-outlier detection should succeed");
    assert!(outliers.is_empty(), "no outliers expected in tight data");
}

#[test]
fn test_outlier_detection_iqr_insufficient_data() {
    let values = vec![1.0, 2.0, 3.0]; // Less than 4
    let result = detect_outliers_iqr(&values, 1.5);
    assert!(result.is_ok());
    let outliers = result.expect("insufficient data should return empty");
    assert!(outliers.is_empty());
}

#[test]
fn test_outlier_detection_zscore_with_outliers() {
    let mut values: Vec<f64> = (0..50).map(|i| 10.0 + (i as f64 * 0.1)).collect();
    values.push(500.0); // Clear outlier
    let result = detect_outliers_zscore(&values, 2.0);
    assert!(result.is_ok());
    let outliers = result.expect("Z-score outlier detection should succeed");
    assert!(!outliers.is_empty(), "should detect 500.0 as outlier");
}

#[test]
fn test_outlier_detection_zscore_no_outliers() {
    let values = vec![10.0, 10.1, 9.9, 10.0, 10.05, 9.95];
    let result = detect_outliers_zscore(&values, 3.0);
    assert!(result.is_ok());
    let outliers = result.expect("Z-score no-outlier detection should succeed");
    assert!(outliers.is_empty());
}

#[test]
fn test_outlier_detection_zscore_insufficient_data() {
    let values = vec![1.0]; // Need at least 2
    let result = detect_outliers_zscore(&values, 2.0);
    assert!(result.is_ok());
    let outliers = result.expect("insufficient data should return empty");
    assert!(outliers.is_empty());
}

#[test]
fn test_outlier_detection_zscore_zero_stddev() {
    let values = vec![5.0, 5.0, 5.0, 5.0]; // Zero std dev
    let result = detect_outliers_zscore(&values, 2.0);
    assert!(result.is_ok());
    let outliers = result.expect("zero stddev should return empty");
    assert!(outliers.is_empty());
}

#[test]
fn test_outlier_detection_iqr_larger_multiplier() {
    let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 20.0];
    let result_strict = detect_outliers_iqr(&values, 1.0);
    let result_loose = detect_outliers_iqr(&values, 3.0);
    assert!(result_strict.is_ok());
    assert!(result_loose.is_ok());
    let strict_outliers = result_strict.expect("strict IQR should succeed");
    let loose_outliers = result_loose.expect("loose IQR should succeed");
    assert!(
        strict_outliers.len() >= loose_outliers.len(),
        "stricter multiplier should find more or equal outliers"
    );
}
