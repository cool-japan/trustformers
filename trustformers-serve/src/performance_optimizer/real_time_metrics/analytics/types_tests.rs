//! Tests for real_time_metrics analytics types
//!
//! Tests covering struct construction, enum variants, async analyzer creation,
//! statistical methods, and anomaly detection.

use super::*;
// `TrendDirection` lives in the shared real-time-metrics enum module; the
// `use super::*` glob above only reaches the analytics module's own items.
use crate::performance_optimizer::real_time_metrics::types::data_structures::TimestampedMetrics;
use crate::performance_optimizer::real_time_metrics::types::TrendDirection;
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

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn test_shape_assessment_construction() {
    let assessment = ShapeAssessment {
        is_unimodal: true,
        is_symmetric: false,
        has_heavy_tails: true,
        shape_description: "Right-skewed distribution".to_string(),
    };
    assert!(assessment.is_unimodal);
    assert!(!assessment.is_symmetric);
    assert!(assessment.has_heavy_tails);
    assert_eq!(assessment.shape_description, "Right-skewed distribution");
}

#[test]
fn test_shape_assessment_clone() {
    let assessment = ShapeAssessment {
        is_unimodal: false,
        is_symmetric: true,
        has_heavy_tails: false,
        shape_description: "Normal-like".to_string(),
    };
    let cloned = assessment.clone();
    assert_eq!(cloned.is_unimodal, assessment.is_unimodal);
    assert_eq!(cloned.is_symmetric, assessment.is_symmetric);
    assert_eq!(cloned.shape_description, assessment.shape_description);
}

#[test]
fn test_shape_assessment_debug() {
    let assessment = ShapeAssessment {
        is_unimodal: true,
        is_symmetric: true,
        has_heavy_tails: false,
        shape_description: "Symmetric unimodal".to_string(),
    };
    let debug_str = format!("{:?}", assessment);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_downtime_pattern_construction() {
    let pattern = DowntimePattern {
        pattern_type: "periodic".to_string(),
        frequency: 0.1,
        time_periods: vec!["monday_morning".to_string(), "friday_evening".to_string()],
        confidence: 0.85,
    };
    assert_eq!(pattern.pattern_type, "periodic");
    assert!((pattern.frequency - 0.1).abs() < f64::EPSILON);
    assert_eq!(pattern.time_periods.len(), 2);
    assert!((pattern.confidence - 0.85).abs() < f64::EPSILON);
}

#[test]
fn test_downtime_pattern_clone() {
    let pattern = DowntimePattern {
        pattern_type: "random".to_string(),
        frequency: 0.05,
        time_periods: Vec::new(),
        confidence: 0.6,
    };
    let cloned = pattern.clone();
    assert_eq!(cloned.pattern_type, pattern.pattern_type);
    assert!((cloned.frequency - pattern.frequency).abs() < f64::EPSILON);
    assert!((cloned.confidence - pattern.confidence).abs() < f64::EPSILON);
}

#[test]
fn test_trend_components_construction() {
    let components = TrendComponents {
        linear_trend: 0.02,
        quadratic_trend: -0.001,
        seasonal_component: 0.05,
        noise_component: 0.03,
        explained_variance: 0.75,
    };
    assert!((components.linear_trend - 0.02).abs() < f64::EPSILON);
    assert!((components.quadratic_trend - (-0.001)).abs() < f64::EPSILON);
    assert!((components.explained_variance - 0.75).abs() < f64::EPSILON);
}

#[test]
fn test_trend_components_clone() {
    let components = TrendComponents {
        linear_trend: 0.01,
        quadratic_trend: 0.0,
        seasonal_component: 0.0,
        noise_component: 0.1,
        explained_variance: 0.5,
    };
    let cloned = components.clone();
    assert!((cloned.linear_trend - components.linear_trend).abs() < f64::EPSILON);
    assert!((cloned.explained_variance - components.explained_variance).abs() < f64::EPSILON);
}

#[test]
fn test_causal_relationship_construction() {
    let mut test_stats = HashMap::new();
    test_stats.insert("p_value".to_string(), 0.01);
    test_stats.insert("f_statistic".to_string(), 12.5);
    let rel = CausalRelationship {
        cause: "cpu_usage".to_string(),
        effect: "latency".to_string(),
        strength: 0.7,
        confidence: 0.9,
        test_statistics: test_stats,
    };
    assert_eq!(rel.cause, "cpu_usage");
    assert_eq!(rel.effect, "latency");
    assert!((rel.strength - 0.7).abs() < f64::EPSILON);
    assert_eq!(rel.test_statistics.len(), 2);
}

#[test]
fn test_causal_relationship_clone() {
    let rel = CausalRelationship {
        cause: "memory".to_string(),
        effect: "throughput".to_string(),
        strength: 0.5,
        confidence: 0.8,
        test_statistics: HashMap::new(),
    };
    let cloned = rel.clone();
    assert_eq!(cloned.cause, rel.cause);
    assert_eq!(cloned.effect, rel.effect);
    assert!((cloned.strength - rel.strength).abs() < f64::EPSILON);
}

#[test]
fn test_goodness_of_fit_statistics_construction() {
    let gof = GoodnessOfFitStatistics {
        ks_statistic: 0.05,
        ks_p_value: 0.2,
        ad_statistic: 0.3,
        ad_p_value: Some(0.15),
        chi_square_statistic: 10.5,
        chi_square_p_value: 0.1,
        log_likelihood: -120.5,
        aic: 245.0,
        bic: 255.0,
    };
    assert!((gof.ks_statistic - 0.05).abs() < f64::EPSILON);
    assert!((gof.ks_p_value - 0.2).abs() < f64::EPSILON);
    assert!(gof.aic < gof.bic);
}

#[test]
fn test_goodness_of_fit_clone() {
    let gof = GoodnessOfFitStatistics {
        ks_statistic: 0.1,
        ks_p_value: 0.05,
        ad_statistic: 0.5,
        ad_p_value: Some(0.08),
        chi_square_statistic: 15.0,
        chi_square_p_value: 0.04,
        log_likelihood: -200.0,
        aic: 404.0,
        bic: 410.0,
    };
    let cloned = gof.clone();
    assert!((cloned.ks_statistic - gof.ks_statistic).abs() < f64::EPSILON);
    assert!((cloned.aic - gof.aic).abs() < f64::EPSILON);
}

#[test]
fn test_implementation_effort_variants() {
    let low = ImplementationEffort::Low;
    let medium = ImplementationEffort::Medium;
    let high = ImplementationEffort::High;
    let very_high = ImplementationEffort::VeryHigh;
    let _ = format!("{:?}", low);
    let _ = format!("{:?}", medium);
    let _ = format!("{:?}", high);
    let _ = format!("{:?}", very_high);
}

#[test]
fn test_implementation_effort_clone() {
    let effort = ImplementationEffort::Medium;
    let cloned = effort.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_relationship_type_variants() {
    let causal = RelationshipType::Causal;
    let temporal = RelationshipType::TemporalSequence;
    let co_occur = RelationshipType::CoOccurrence;
    let mutual_excl = RelationshipType::MutualExclusion;
    let hierarchical = RelationshipType::Hierarchical;
    let _ = format!("{:?}", causal);
    let _ = format!("{:?}", temporal);
    let _ = format!("{:?}", co_occur);
    let _ = format!("{:?}", mutual_excl);
    let _ = format!("{:?}", hierarchical);
}

#[test]
fn test_relationship_type_clone() {
    let rel_type = RelationshipType::Causal;
    let cloned = rel_type.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_change_direction_variants() {
    let increase = ChangeDirection::Increase;
    let decrease = ChangeDirection::Decrease;
    let level_shift = ChangeDirection::LevelShift;
    let variance_change = ChangeDirection::VarianceChange;
    let _ = format!("{:?}", increase);
    let _ = format!("{:?}", decrease);
    let _ = format!("{:?}", level_shift);
    let _ = format!("{:?}", variance_change);
}

#[test]
fn test_correlation_strength_variants() {
    let weak = CorrelationStrength::Weak;
    let moderate = CorrelationStrength::Moderate;
    let strong = CorrelationStrength::Strong;
    let very_strong = CorrelationStrength::VeryStrong;
    let _ = format!("{:?}", weak);
    let _ = format!("{:?}", moderate);
    let _ = format!("{:?}", strong);
    let _ = format!("{:?}", very_strong);
}

#[test]
fn test_correlation_strength_clone() {
    let strength = CorrelationStrength::Strong;
    let cloned = strength.clone();
    let _ = format!("{:?}", cloned);
}

#[test]
fn test_basic_statistics_construction() {
    let stats = BasicStatistics {
        count: 100,
        mean: 5.0,
        median: 4.9,
        mode: vec![4.5, 5.5],
        min: 1.0,
        max: 10.0,
        range: 9.0,
        sum: 500.0,
    };
    assert_eq!(stats.count, 100);
    assert!((stats.mean - 5.0).abs() < f64::EPSILON);
    assert!((stats.median - 4.9).abs() < f64::EPSILON);
    assert_eq!(stats.mode.len(), 2);
    assert!((stats.min - 1.0).abs() < f64::EPSILON);
    assert!((stats.max - 10.0).abs() < f64::EPSILON);
    assert!((stats.range - 9.0).abs() < f64::EPSILON);
}

#[test]
fn test_basic_statistics_clone() {
    let stats = BasicStatistics {
        count: 50,
        mean: 2.5,
        median: 2.3,
        mode: Vec::new(),
        min: 0.0,
        max: 5.0,
        range: 5.0,
        sum: 125.0,
    };
    let cloned = stats.clone();
    assert_eq!(cloned.count, stats.count);
    assert!((cloned.mean - stats.mean).abs() < f64::EPSILON);
}

#[test]
fn test_advanced_statistics_construction() {
    let stats = AdvancedStatistics {
        variance: 4.0,
        std_dev: 2.0,
        skewness: 0.3,
        kurtosis: -0.1,
        coefficient_of_variation: 0.4,
        standard_error: 0.2,
        geometric_mean: Some(4.5),
        harmonic_mean: Some(4.2),
    };
    assert!((stats.variance - 4.0).abs() < f64::EPSILON);
    assert!((stats.std_dev - 2.0).abs() < f64::EPSILON);
    assert!(stats.geometric_mean.is_some());
    assert!(stats.harmonic_mean.is_some());
}

#[test]
fn test_advanced_statistics_no_means() {
    let stats = AdvancedStatistics {
        variance: 1.0,
        std_dev: 1.0,
        skewness: 0.0,
        kurtosis: 0.0,
        coefficient_of_variation: 0.5,
        standard_error: 0.1,
        geometric_mean: None,
        harmonic_mean: None,
    };
    assert!(stats.geometric_mean.is_none());
    assert!(stats.harmonic_mean.is_none());
}

#[test]
fn test_advanced_statistics_clone() {
    let stats = AdvancedStatistics {
        variance: 2.25,
        std_dev: 1.5,
        skewness: -0.2,
        kurtosis: 0.5,
        coefficient_of_variation: 0.3,
        standard_error: 0.15,
        geometric_mean: Some(3.0),
        harmonic_mean: None,
    };
    let cloned = stats.clone();
    assert!((cloned.variance - stats.variance).abs() < f64::EPSILON);
    assert_eq!(cloned.geometric_mean, stats.geometric_mean);
}

#[test]
fn test_descriptive_statistics_construction() {
    let mut percentiles = HashMap::new();
    percentiles.insert(25u8, 2.5);
    percentiles.insert(50u8, 5.0);
    percentiles.insert(75u8, 7.5);
    let desc_stats = DescriptiveStatistics {
        percentiles,
        q1: 2.5,
        q2: 5.0,
        q3: 7.5,
        iqr: 5.0,
        lower_outlier_bound: -5.0,
        upper_outlier_bound: 15.0,
        outlier_count: 0,
    };
    assert!((desc_stats.q1 - 2.5).abs() < f64::EPSILON);
    assert!((desc_stats.q2 - 5.0).abs() < f64::EPSILON);
    assert!((desc_stats.q3 - 7.5).abs() < f64::EPSILON);
    assert!((desc_stats.iqr - 5.0).abs() < f64::EPSILON);
    assert_eq!(desc_stats.outlier_count, 0);
    assert_eq!(desc_stats.percentiles.len(), 3);
}

#[test]
fn test_latency_statistics_construction() {
    let stats = LatencyStatistics {
        mean: 10.0,
        median: 9.5,
        p95: 20.0,
        p99: 35.0,
        p999: 50.0,
        max: 100.0,
        std_dev: 5.0,
    };
    assert!(stats.mean <= stats.median || stats.mean >= stats.median); // always true
    assert!(stats.p95 <= stats.p99);
    assert!(stats.p99 <= stats.p999);
    assert!(stats.p999 <= stats.max);
}

#[test]
fn test_latency_statistics_clone() {
    let stats = LatencyStatistics {
        mean: 8.0,
        median: 7.5,
        p95: 15.0,
        p99: 25.0,
        p999: 40.0,
        max: 80.0,
        std_dev: 3.5,
    };
    let cloned = stats.clone();
    assert!((cloned.mean - stats.mean).abs() < f64::EPSILON);
    assert!((cloned.p99 - stats.p99).abs() < f64::EPSILON);
}

#[test]
fn test_correlation_measure_construction() {
    let measure = CorrelationMeasure {
        pearson: 0.75,
        spearman: 0.72,
        kendall_tau: 0.6,
        mutual_information: 0.5,
        distance_correlation: 0.65,
        p_value: 0.001,
        confidence_interval: (0.65, 0.85),
    };
    assert!((measure.pearson - 0.75).abs() < f64::EPSILON);
    assert!((measure.spearman - 0.72).abs() < f64::EPSILON);
    assert!(measure.p_value < 0.05);
    assert!(measure.confidence_interval.0 < measure.confidence_interval.1);
}

#[test]
fn test_correlation_measure_clone() {
    let measure = CorrelationMeasure {
        pearson: -0.5,
        spearman: -0.48,
        kendall_tau: -0.4,
        mutual_information: 0.3,
        distance_correlation: 0.5,
        p_value: 0.02,
        confidence_interval: (-0.65, -0.35),
    };
    let cloned = measure.clone();
    assert!((cloned.pearson - measure.pearson).abs() < f64::EPSILON);
    assert!((cloned.p_value - measure.p_value).abs() < f64::EPSILON);
}

#[test]
fn test_utilization_metrics_construction() {
    let metrics = UtilizationMetrics {
        current: 0.65,
        peak: 0.95,
        average: 0.7,
        variance: 0.02,
        saturation_points: Vec::new(),
    };
    assert!(metrics.current <= metrics.peak);
    assert!((metrics.variance - 0.02).abs() < f64::EPSILON);
    assert!(metrics.saturation_points.is_empty());
}

#[test]
fn test_utilization_metrics_clone() {
    let metrics = UtilizationMetrics {
        current: 0.5,
        peak: 0.8,
        average: 0.55,
        variance: 0.01,
        saturation_points: vec![Utc::now()],
    };
    let cloned = metrics.clone();
    assert!((cloned.current - metrics.current).abs() < f64::EPSILON);
    assert_eq!(
        cloned.saturation_points.len(),
        metrics.saturation_points.len()
    );
}

#[test]
fn test_trend_data_point_construction() {
    let point = TrendDataPoint {
        timestamp: Utc::now(),
        value: 42.5,
        predicted_value: 42.0,
        residual: 0.5,
        weight: 1.0,
    };
    assert!((point.value - 42.5).abs() < f64::EPSILON);
    assert!((point.predicted_value - 42.0).abs() < f64::EPSILON);
    assert!((point.residual - (point.value - point.predicted_value)).abs() < f64::EPSILON);
}

#[test]
fn test_trend_data_point_clone() {
    let point = TrendDataPoint {
        timestamp: Utc::now(),
        value: 10.0,
        predicted_value: 10.5,
        residual: -0.5,
        weight: 0.9,
    };
    let cloned = point.clone();
    assert!((cloned.value - point.value).abs() < f64::EPSILON);
    assert!((cloned.residual - point.residual).abs() < f64::EPSILON);
}

#[test]
fn test_pattern_relationship_construction() {
    let rel = PatternRelationship {
        source_pattern: "spike".to_string(),
        target_pattern: "recovery".to_string(),
        relationship_type: RelationshipType::TemporalSequence,
        strength: 0.8,
        temporal_offset: Some(Duration::from_secs(300)),
        confidence: 0.9,
    };
    assert_eq!(rel.source_pattern, "spike");
    assert_eq!(rel.target_pattern, "recovery");
    assert!((rel.strength - 0.8).abs() < f64::EPSILON);
    assert!(rel.temporal_offset.is_some());
}

#[test]
fn test_pattern_relationship_no_offset() {
    let rel = PatternRelationship {
        source_pattern: "pattern_a".to_string(),
        target_pattern: "pattern_b".to_string(),
        relationship_type: RelationshipType::CoOccurrence,
        strength: 0.6,
        temporal_offset: None,
        confidence: 0.75,
    };
    assert!(rel.temporal_offset.is_none());
    assert!((rel.confidence - 0.75).abs() < f64::EPSILON);
}

#[tokio::test]
async fn test_anomaly_detector_new() {
    let result = AnomalyDetector::new().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_anomaly_detector_refuses_an_empty_window() {
    // Before 0.2.1 `analyze` ignored its argument and returned a canned report
    // with `anomaly_rate: 0.02` and `detection_confidence: 0.9`, so this test
    // asserted `(analysis.anomaly_rate - 0.02).abs() < f64::EPSILON` for an
    // empty window -- it was locking in the fabrication. An empty window
    // supports no baseline and therefore no rate at all.
    let detector = AnomalyDetector::new().await.expect("construction never fails");
    let error = detector
        .analyze(&[])
        .await
        .expect_err("an empty window cannot establish a baseline");
    assert!(error.to_string().contains("at least 8"), "{error}");
}

#[tokio::test]
async fn test_anomaly_detector_shutdown() {
    let result = AnomalyDetector::new().await;
    assert!(result.is_ok());
    if let Ok(detector) = result {
        let shutdown_result = detector.shutdown().await;
        assert!(shutdown_result.is_ok());
    }
}

#[tokio::test]
async fn test_statistical_analyzer_new() {
    let result = StatisticalAnalyzer::new().await;
    assert!(result.is_ok());
}

#[tokio::test]
async fn test_statistical_analyzer_shutdown() {
    let result = StatisticalAnalyzer::new().await;
    assert!(result.is_ok());
    if let Ok(analyzer) = result {
        let shutdown_result = analyzer.shutdown().await;
        assert!(shutdown_result.is_ok());
    }
}

#[test]
fn test_lcg_values_in_range() {
    let mut rng = Lcg::new(12345);
    for _ in 0..50 {
        let v = rng.next_f64();
        assert!((0.0..1.0).contains(&v));
    }
}

#[test]
fn test_statistical_analyzer_config_default() {
    let config = StatisticalAnalyzerConfig::default();
    assert!((config.confidence_level - 0.95).abs() < f64::EPSILON);
    assert!(config.enable_robust_stats);
    assert!(config.bootstrap_iterations > 0);
}

#[test]
fn test_normality_test_result_construction() {
    let result = NormalityTestResult {
        statistic: 0.97,
        p_value: 0.15,
        is_normal: true,
        significance_level: 0.05,
    };
    assert!((result.statistic - 0.97).abs() < f64::EPSILON);
    assert!(result.p_value > result.significance_level);
    assert!(result.is_normal);
}

#[test]
fn test_normality_test_result_not_normal() {
    let result = NormalityTestResult {
        statistic: 0.7,
        p_value: 0.001,
        is_normal: false,
        significance_level: 0.05,
    };
    assert!(result.p_value < result.significance_level);
    assert!(!result.is_normal);
}

#[test]
fn test_efficiency_trend_construction() {
    let trend = EfficiencyTrend {
        component: "cpu".to_string(),
        direction: TrendDirection::Increasing,
        improvement_rate: 0.05,
        duration: Duration::from_secs(3600),
        confidence: 0.8,
    };
    assert_eq!(trend.component, "cpu");
    assert!((trend.improvement_rate - 0.05).abs() < f64::EPSILON);
    assert_eq!(trend.duration, Duration::from_secs(3600));
    assert!((trend.confidence - 0.8).abs() < f64::EPSILON);
}

#[test]
fn test_efficiency_trend_clone() {
    let trend = EfficiencyTrend {
        component: "memory".to_string(),
        direction: TrendDirection::Decreasing,
        improvement_rate: -0.02,
        duration: Duration::from_secs(1800),
        confidence: 0.7,
    };
    let cloned = trend.clone();
    assert_eq!(cloned.component, trend.component);
    assert!((cloned.improvement_rate - trend.improvement_rate).abs() < f64::EPSILON);
}

#[test]
fn test_distribution_fit_construction() {
    let mut parameters = HashMap::new();
    parameters.insert("mu".to_string(), 5.0);
    parameters.insert("sigma".to_string(), 1.5);
    let fit = DistributionFit {
        distribution_name: "Normal".to_string(),
        parameters,
        fit_statistics: GoodnessOfFitStatistics {
            ks_statistic: 0.05,
            ks_p_value: 0.2,
            ad_statistic: 0.3,
            ad_p_value: Some(0.15),
            chi_square_statistic: 8.0,
            chi_square_p_value: 0.1,
            log_likelihood: -100.0,
            aic: 204.0,
            bic: 210.0,
        },
        fit_score: 0.92,
        parameter_confidence_intervals: HashMap::new(),
    };
    assert_eq!(fit.distribution_name, "Normal");
    assert!((fit.fit_score - 0.92).abs() < f64::EPSILON);
    assert_eq!(fit.parameters.len(), 2);
}

#[test]
fn test_quality_recommendation_construction() {
    let rec = QualityRecommendation {
        recommendation_type: "data_completeness".to_string(),
        priority: 1,
        description: "Ensure all fields are populated".to_string(),
        expected_improvement: 0.15,
        implementation_effort: "low".to_string(),
        cost_benefit_ratio: Some(5.0),
    };
    assert_eq!(rec.priority, 1);
    assert!((rec.expected_improvement - 0.15).abs() < f64::EPSILON);
    assert!(rec.cost_benefit_ratio.is_some_and(|ratio| (ratio - 5.0).abs() < f64::EPSILON));
}

#[test]
fn test_lcg_different_seeds_give_different_results() {
    let mut rng1 = Lcg::new(111);
    let mut rng2 = Lcg::new(999);
    let v1 = rng1.next_f64();
    let v2 = rng2.next_f64();
    assert!((v1 - v2).abs() > 0.0);
}

// =============================================================================
// 0.2.1 REGRESSIONS: statistics that used to be constants
// =============================================================================

/// Builds `n` samples whose throughput series is `values`, everything else held
/// constant so only the planted series varies.
fn statistical_window(values: &[f64]) -> Vec<TimestampedMetrics> {
    values
        .iter()
        .enumerate()
        .map(|(index, value)| {
            let mut sample = TimestampedMetrics::default();
            sample.timestamp = Utc::now() + chrono::Duration::seconds(index as i64);
            sample.metrics.throughput = *value;
            sample.metrics.latency = Duration::from_millis(10);
            sample.metrics.resource_usage.cpu_usage = 0.25;
            sample
        })
        .collect()
}

/// Regression: `one_sample_t_test` set `p_value = if t.abs() > 2.0 { 0.05 }
/// else { 0.1 }`, then decided the verdict with `p_value < 0.05` -- which is
/// false for both branches, so the test could only ever say "fail to reject".
#[tokio::test]
async fn t_test_p_value_is_computed_from_the_statistic() {
    let analyzer = StatisticalAnalyzer::new().await.expect("analyzer constructs");
    // A series far from the hypothesised mean of 0.0, with tiny spread.
    let far = analyzer
        .analyze(&statistical_window(&[
            500.0, 500.1, 499.9, 500.2, 499.8, 500.05, 499.95, 500.15, 499.85, 500.0,
        ]))
        .await
        .expect("analysis runs");
    let far_test = far.statistical_tests.get("one_sample_t_test").expect("the t-test is reported");
    // `extract_numerical_values` interleaves throughput, latency and CPU, so
    // the analysed series is a wide mixture -- but its mean still sits far
    // enough from zero to reject at any conventional level.
    assert!(
        far_test.p_value < 0.01,
        "a series centred far from zero must have a small p-value, got {}",
        far_test.p_value
    );
    assert_eq!(far_test.result, "Reject null hypothesis");
    assert!(
        far_test.critical_value.is_none(),
        "no inverse-t quantile exists in tree, so no critical value is reported"
    );
    assert!(
        far_test.p_value != 0.05 && far_test.p_value != 0.1,
        "the old code could only ever answer 0.05 or 0.1"
    );
}

/// Regression: `shapiro_wilk_test`'s statistic reduced algebraically to a
/// constant 0.0 for every input, so `is_normal` was always false and the
/// p-value always 0.01. The replacement is a real KS test under its own name.
#[tokio::test]
async fn normality_testing_answers_differently_for_different_shapes() {
    let analyzer = StatisticalAnalyzer::new().await.expect("analyzer constructs");
    let mut rng = Lcg::new(20260824);
    // Roughly normal: the mean of several uniforms.
    let normalish: Vec<f64> =
        (0..200).map(|_| (0..8).map(|_| rng.next_f64()).sum::<f64>() / 8.0).collect();
    let normal_result =
        analyzer.analyze(&statistical_window(&normalish)).await.expect("analysis runs");
    let ks = normal_result
        .distribution_characteristics
        .normality_tests
        .get("kolmogorov_smirnov")
        .expect("the KS test is reported under its own name");
    assert!(
        !normal_result
            .distribution_characteristics
            .normality_tests
            .contains_key("shapiro_wilk"),
        "the test that was never Shapiro-Wilk no longer claims to be"
    );
    assert!(
        ks.statistic > 0.0,
        "a real KS statistic is positive, the old one was identically 0.0"
    );

    // A hard two-point series is nothing like a normal.
    let bimodal: Vec<f64> = (0..200).map(|i| if i % 2 == 0 { 0.0 } else { 100.0 }).collect();
    let bimodal_result =
        analyzer.analyze(&statistical_window(&bimodal)).await.expect("analysis runs");
    let bimodal_ks = bimodal_result
        .distribution_characteristics
        .normality_tests
        .get("kolmogorov_smirnov")
        .expect("the KS test is reported");
    assert!(
        bimodal_ks.statistic > ks.statistic,
        "a two-point series must depart from normal further than a bell-ish one: \
         {} vs {}",
        bimodal_ks.statistic,
        ks.statistic
    );
}

/// Regression: the Jarque-Bera p-value was one of three constants chosen by
/// bucketing the statistic at 6 and 10. It is now the chi-square(2) survival
/// function of the statistic.
#[tokio::test]
async fn jarque_bera_p_value_tracks_its_statistic() {
    let analyzer = StatisticalAnalyzer::new().await.expect("analyzer constructs");
    let result = analyzer
        .analyze(&statistical_window(
            &(0..200).map(|i| i as f64).collect::<Vec<_>>(),
        ))
        .await
        .expect("analysis runs");
    let jb = result
        .distribution_characteristics
        .normality_tests
        .get("jarque_bera")
        .expect("the JB test is reported");
    let expected = (-jb.statistic / 2.0).exp();
    assert!(
        (jb.p_value - expected).abs() < 1e-6,
        "chi-square(2) survival is exp(-x/2): expected {expected}, got {}",
        jb.p_value
    );
}

/// Regression: `calculate_confidence_intervals` copied one unlabelled series'
/// interval into the latency, CPU, memory, network, I/O, response-time and
/// error-rate fields, and reported the variance interval as the point estimate
/// times 0.8 and 1.2.
#[tokio::test]
async fn confidence_intervals_report_only_what_was_measured() {
    let analyzer = StatisticalAnalyzer::new().await.expect("analyzer constructs");
    let result = analyzer
        .analyze(&statistical_window(&[
            10.0, 12.0, 11.0, 13.0, 9.0, 10.5, 11.5, 12.5,
        ]))
        .await
        .expect("analysis runs");
    let intervals = &result.confidence_intervals;
    assert!(intervals.latency_interval.is_none());
    assert!(intervals.cpu_interval.is_none());
    assert!(intervals.memory_interval.is_none());
    assert!(intervals.network_interval.is_none());
    assert!(intervals.io_interval.is_none());
    assert!(intervals.response_time_interval.is_none());
    assert!(intervals.error_rate_interval.is_none());
    assert!(intervals.mean_lower < intervals.mean_upper);
    let (lower, upper) = (
        intervals.variance_lower.expect("variance interval is reported"),
        intervals.variance_upper.expect("variance interval is reported"),
    );
    assert!(lower < upper);
    // The old code produced exactly point*0.8 and point*1.2.
    let point = (lower + upper) / 2.0;
    assert!(
        (lower - point * 0.8).abs() > 1e-9 || (upper - point * 1.2).abs() > 1e-9,
        "the interval must not be the old fixed +/-20% band"
    );
}
