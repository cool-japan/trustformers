//! Tests for performance analytics types

use super::types::*;
use std::time::{Duration, SystemTime};

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
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() % 100000) as f64 / 100000.0
    }
}

#[test]
fn test_analytics_error_insufficient_data() {
    let err = AnalyticsError::InsufficientData {
        required: 10,
        available: 3,
    };
    let msg = format!("{:?}", err);
    assert!(msg.contains("InsufficientData"));
}

#[test]
fn test_analytics_error_statistical_analysis() {
    let err = AnalyticsError::StatisticalAnalysisError {
        reason: "division by zero".to_string(),
    };
    let msg = format!("{:?}", err);
    assert!(msg.contains("division by zero"));
}

#[test]
fn test_distribution_type_variants() {
    let types = [
        DistributionType::Normal,
        DistributionType::Skewed,
        DistributionType::Multimodal,
        DistributionType::Unknown,
    ];
    assert_eq!(types.len(), 4);
}

#[test]
fn test_data_quality_default() {
    let quality = DataQuality::default();
    assert!((quality.completeness - 0.0).abs() < f64::EPSILON);
    assert!((quality.accuracy - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_data_quality_creation() {
    let quality = DataQuality {
        completeness: 0.95,
        accuracy: 0.99,
        precision: 0.98,
        timeliness: 0.9,
        consistency: 0.97,
        overall_score: 0.96,
    };
    assert!(quality.overall_score > 0.0);
    assert!(quality.completeness <= 1.0);
}

#[test]
fn test_percentiles_creation() {
    let p = Percentiles {
        p1: 1.0,
        p5: 5.0,
        p10: 10.0,
        p25: 25.0,
        p50: 50.0,
        p75: 75.0,
        p90: 90.0,
        p95: 95.0,
        p99: 99.0,
    };
    assert!(p.p1 < p.p5);
    assert!(p.p5 < p.p10);
    assert!(p.p25 < p.p50);
    assert!(p.p50 < p.p75);
    assert!(p.p90 < p.p95);
    assert!(p.p95 < p.p99);
}

#[test]
fn test_statistical_analyzer_percentile_basic() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![StatisticalMethod::Mean],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let p50 = analyzer.percentile(&data, 0.5);
    assert!(p50 >= 4.0 && p50 <= 6.0);
}

#[test]
fn test_statistical_analyzer_percentile_edges() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![10.0, 20.0, 30.0];
    assert!((analyzer.percentile(&data, 0.0) - 10.0).abs() < f64::EPSILON);
    assert!((analyzer.percentile(&data, 1.0) - 30.0).abs() < f64::EPSILON);
}

#[test]
fn test_statistical_analyzer_percentile_empty() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data: Vec<f64> = vec![];
    assert!((analyzer.percentile(&data, 0.5) - 0.0).abs() < f64::EPSILON);
}

#[test]
fn test_statistical_analyzer_descriptive_stats() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![StatisticalMethod::Mean],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
    match analyzer.calculate_descriptive_statistics(&data) {
        Ok(summary) => {
            assert!((summary.mean - 5.0).abs() < 0.001);
            assert!(summary.standard_deviation > 0.0);
            assert!(summary.variance > 0.0);
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_statistical_analyzer_descriptive_stats_empty() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data: Vec<f64> = vec![];
    match analyzer.calculate_descriptive_statistics(&data) {
        Err(AnalyticsError::StatisticalAnalysisError { .. }) => {}
        other => panic!("expected StatisticalAnalysisError, got {:?}", other),
    }
}

#[test]
fn test_statistical_analyzer_outlier_detection() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![1.0, 2.0, 2.0, 3.0, 3.0, 3.0, 4.0, 4.0, 5.0, 100.0];
    match analyzer.detect_outliers(&data) {
        Ok(analysis) => {
            assert!(!analysis.outliers.is_empty(), "should detect 100.0 as outlier");
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_statistical_analyzer_outlier_detection_no_outliers() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![5.0, 5.1, 4.9, 5.0, 5.2, 4.8, 5.0];
    match analyzer.detect_outliers(&data) {
        Ok(analysis) => {
            assert!(analysis.outliers.is_empty(), "no outliers in tight data");
        }
        Err(e) => panic!("unexpected error: {:?}", e),
    }
}

#[test]
fn test_confidence_interval_creation() {
    let ci = ConfidenceInterval {
        confidence_level: 0.95,
        lower_bound: 4.5,
        upper_bound: 5.5,
        margin_of_error: 0.5,
    };
    assert!(ci.lower_bound < ci.upper_bound);
    assert!(ci.confidence_level > 0.0 && ci.confidence_level <= 1.0);
}

#[test]
fn test_data_point_creation() {
    let point = DataPoint {
        timestamp: SystemTime::now(),
        value: 42.0,
        quality: DataQuality::default(),
        annotations: vec![],
    };
    assert!((point.value - 42.0).abs() < f64::EPSILON);
}

#[test]
fn test_trend_point_creation() {
    let point = TrendPoint {
        timestamp: SystemTime::now(),
        value: 100.0,
        trend_direction: TrendDirection::Improving,
        trend_strength: 0.8,
        volatility: 0.1,
        confidence: 0.9,
    };
    assert!(point.trend_strength > 0.0);
    assert!(point.confidence > 0.0);
}

#[test]
fn test_analytics_config_default() {
    let config = AnalyticsConfig::default();
    let _ = format!("{:?}", config);
}

#[test]
fn test_statistical_summary_creation() {
    let summary = StatisticalSummary {
        mean: 50.0,
        median: 49.0,
        mode: Some(48.0),
        standard_deviation: 10.0,
        variance: 100.0,
        skewness: 0.1,
        kurtosis: 3.0,
        percentiles: Percentiles {
            p1: 20.0,
            p5: 25.0,
            p10: 30.0,
            p25: 40.0,
            p50: 49.0,
            p75: 60.0,
            p90: 70.0,
            p95: 75.0,
            p99: 80.0,
        },
        range: (20.0, 80.0),
        interquartile_range: 20.0,
    };
    assert!(summary.mean > summary.range.0);
    assert!(summary.mean < summary.range.1);
    assert!((summary.interquartile_range - 20.0).abs() < f64::EPSILON);
}

#[test]
fn test_multiple_data_points_with_lcg() {
    let mut lcg = Lcg::new(42);
    let points: Vec<DataPoint> = (0..20)
        .map(|_| DataPoint {
            timestamp: SystemTime::now(),
            value: lcg.next_f64() * 100.0,
            quality: DataQuality::default(),
            annotations: vec![],
        })
        .collect();
    assert_eq!(points.len(), 20);
    for p in &points {
        assert!(p.value >= 0.0 && p.value <= 100.0);
    }
}

#[test]
fn test_recommendation_summary_creation() {
    let rec = RecommendationSummary {
        recommendation_id: "rec_001".to_string(),
        recommendation_type: "optimization".to_string(),
        description: "Optimize database queries".to_string(),
        expected_benefit: 2.5,
        complexity: 0.6,
        priority: "high".to_string(),
        urgency: "medium".to_string(),
        required_resources: vec!["developer_time".to_string()],
        steps: vec!["Add indexes".to_string()],
        risk: 0.2,
        confidence: 0.9,
        expected_roi: 2.5,
    };
    assert!(rec.expected_roi > 0.0);
    assert!(rec.confidence > 0.0);
}

#[test]
fn test_performance_pattern_creation() {
    let pattern = PerformancePattern {
        pattern_id: "p_001".to_string(),
        pattern_name: "seasonal_load".to_string(),
        pattern_type: crate::performance_optimizer::test_characterization::types::PatternType::Performance,
        description: "Seasonal performance pattern".to_string(),
        characteristics: vec![],
        occurrence_frequency: 0.85,
        impact_severity: crate::performance_optimizer::test_characterization::pattern_engine::SeverityLevel::Medium,
        mitigation_strategies: vec!["auto-scaling".to_string()],
        detection_accuracy: 0.92,
    };
    assert!(pattern.occurrence_frequency > 0.0);
    assert!(pattern.detection_accuracy > 0.0);
}

#[test]
fn test_statistical_analyzer_single_element() {
    let analyzer = StatisticalAnalyzer {
        window_size: 100,
        confidence_level: 0.95,
        statistical_methods: vec![],
        outlier_detection_config: OutlierDetectionConfig {
            threshold: 1.5,
            window_size: Duration::from_secs(60),
            sensitivity: 1.0,
        },
    };
    let data = vec![42.0];
    let p50 = analyzer.percentile(&data, 0.5);
    assert!((p50 - 42.0).abs() < f64::EPSILON);
}
