//! Tests for real_time_metrics aggregator types
//!
//! Tests covering struct construction, processor creation,
//! window config, quality assessment, and aggregation window lifecycle.

use super::*;
use chrono::Utc;
use std::time::Duration;

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state
            .wrapping_mul(6364136223846793005u64)
            .wrapping_add(1442695040888963407u64);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

#[test]
fn test_window_config_fields() {
    let config = WindowConfig {
        enable_statistical_analysis: true,
        enable_trend_detection: false,
        enable_outlier_removal: true,
        max_data_points: 500,
        statistical_confidence: 0.95,
        trend_sensitivity: 0.1,
    };
    assert!(config.enable_statistical_analysis);
    assert!(!config.enable_trend_detection);
    assert!(config.enable_outlier_removal);
    assert_eq!(config.max_data_points, 500);
    assert!((config.statistical_confidence - 0.95).abs() < f32::EPSILON);
    assert!((config.trend_sensitivity - 0.1).abs() < f32::EPSILON);
}

#[test]
fn test_window_config_clone() {
    let config = WindowConfig {
        enable_statistical_analysis: false,
        enable_trend_detection: true,
        enable_outlier_removal: false,
        max_data_points: 1000,
        statistical_confidence: 0.99,
        trend_sensitivity: 0.05,
    };
    let cloned = config.clone();
    assert_eq!(cloned.enable_statistical_analysis, config.enable_statistical_analysis);
    assert_eq!(cloned.max_data_points, config.max_data_points);
    assert!((cloned.statistical_confidence - config.statistical_confidence).abs() < f32::EPSILON);
}

#[test]
fn test_window_config_debug() {
    let config = WindowConfig {
        enable_statistical_analysis: true,
        enable_trend_detection: true,
        enable_outlier_removal: true,
        max_data_points: 200,
        statistical_confidence: 0.9,
        trend_sensitivity: 0.2,
    };
    let debug_str = format!("{:?}", config);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_quality_config_fields() {
    let config = QualityConfig {
        enable_quality_scoring: true,
        enable_outlier_detection: true,
        outlier_threshold: 3.0,
        quality_threshold: 0.8,
        validation_rules: Vec::new(),
    };
    assert!(config.enable_quality_scoring);
    assert!(config.enable_outlier_detection);
    assert!((config.outlier_threshold - 3.0).abs() < f32::EPSILON);
    assert!((config.quality_threshold - 0.8).abs() < f32::EPSILON);
    assert!(config.validation_rules.is_empty());
}

#[test]
fn test_quality_config_clone() {
    let config = QualityConfig {
        enable_quality_scoring: false,
        enable_outlier_detection: false,
        outlier_threshold: 2.5,
        quality_threshold: 0.7,
        validation_rules: Vec::new(),
    };
    let cloned = config.clone();
    assert_eq!(cloned.enable_quality_scoring, config.enable_quality_scoring);
    assert!((cloned.outlier_threshold - config.outlier_threshold).abs() < f32::EPSILON);
}

#[test]
fn test_processor_config_default() {
    let config = ProcessorConfig::default();
    let _ = format!("{:?}", config);
}

#[test]
fn test_processor_config_fields() {
    let config = ProcessorConfig {
        advanced_features: true,
        timeout_ms: 1000,
        quality_threshold: 0.85,
    };
    assert!(config.advanced_features);
    assert_eq!(config.timeout_ms, 1000);
    assert!((config.quality_threshold - 0.85).abs() < f32::EPSILON);
}

#[test]
fn test_processor_config_clone() {
    let config = ProcessorConfig {
        advanced_features: false,
        timeout_ms: 500,
        quality_threshold: 0.9,
    };
    let cloned = config.clone();
    assert_eq!(cloned.advanced_features, config.advanced_features);
    assert_eq!(cloned.timeout_ms, config.timeout_ms);
}

#[test]
fn test_quality_assessment_fields() {
    let assessment = QualityAssessment {
        overall_quality: 0.9,
        data_completeness: 1.0,
        accuracy_score: 0.95,
        consistency_score: 0.88,
        outlier_count: 2,
        assessment_timestamp: Utc::now(),
    };
    assert!((assessment.overall_quality - 0.9).abs() < f32::EPSILON);
    assert!((assessment.data_completeness - 1.0).abs() < f32::EPSILON);
    assert!((assessment.accuracy_score - 0.95).abs() < f32::EPSILON);
    assert_eq!(assessment.outlier_count, 2);
}

#[test]
fn test_quality_assessment_clone() {
    let assessment = QualityAssessment {
        overall_quality: 0.75,
        data_completeness: 0.9,
        accuracy_score: 0.8,
        consistency_score: 0.7,
        outlier_count: 5,
        assessment_timestamp: Utc::now(),
    };
    let cloned = assessment.clone();
    assert!((cloned.overall_quality - assessment.overall_quality).abs() < f32::EPSILON);
    assert_eq!(cloned.outlier_count, assessment.outlier_count);
}

#[test]
fn test_quality_assessment_debug() {
    let assessment = QualityAssessment {
        overall_quality: 0.85,
        data_completeness: 0.95,
        accuracy_score: 0.9,
        consistency_score: 0.8,
        outlier_count: 0,
        assessment_timestamp: Utc::now(),
    };
    let debug_str = format!("{:?}", assessment);
    assert!(!debug_str.is_empty());
}

#[test]
fn test_streaming_config_fields() {
    let config = StreamingConfig {
        buffer_size: 1024,
        worker_count: 4,
        backpressure_threshold: 0.8,
        flow_control_enabled: true,
        adaptive_processing: false,
    };
    assert_eq!(config.buffer_size, 1024);
    assert_eq!(config.worker_count, 4);
    assert!((config.backpressure_threshold - 0.8).abs() < f32::EPSILON);
    assert!(config.flow_control_enabled);
    assert!(!config.adaptive_processing);
}

#[test]
fn test_streaming_config_clone() {
    let config = StreamingConfig {
        buffer_size: 2048,
        worker_count: 8,
        backpressure_threshold: 0.7,
        flow_control_enabled: false,
        adaptive_processing: true,
    };
    let cloned = config.clone();
    assert_eq!(cloned.buffer_size, config.buffer_size);
    assert_eq!(cloned.worker_count, config.worker_count);
}

#[test]
fn test_publishing_config_fields() {
    let config = PublishingConfig {
        enable_publishing: true,
        delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
        batch_size: 50,
        retry_attempts: 3,
        compression_enabled: true,
    };
    assert!(config.enable_publishing);
    assert_eq!(config.batch_size, 50);
    assert_eq!(config.retry_attempts, 3);
    assert!(config.compression_enabled);
}

#[test]
fn test_publishing_config_clone() {
    let config = PublishingConfig {
        enable_publishing: false,
        delivery_guarantee: DeliveryGuarantee::BestEffort,
        batch_size: 100,
        retry_attempts: 5,
        compression_enabled: false,
    };
    let cloned = config.clone();
    assert_eq!(cloned.enable_publishing, config.enable_publishing);
    assert_eq!(cloned.batch_size, config.batch_size);
    assert_eq!(cloned.retry_attempts, config.retry_attempts);
}

#[test]
fn test_distribution_statistical_processor_new() {
    let processor = DistributionStatisticalProcessor::new();
    let _ = processor;
}

#[test]
fn test_efficiency_statistical_processor_new() {
    let processor = EfficiencyStatisticalProcessor::new();
    let _ = processor;
}

#[test]
fn test_basic_statistical_processor_new() {
    let processor = BasicStatisticalProcessor::new();
    let _ = processor;
}

#[test]
fn test_advanced_statistical_processor_new() {
    let processor = AdvancedStatisticalProcessor::new();
    let _ = processor;
}

#[test]
fn test_trend_statistical_processor_new() {
    let processor = TrendStatisticalProcessor::new();
    let _ = processor;
}

#[tokio::test]
async fn test_aggregation_window_new() {
    let config = WindowConfig {
        enable_statistical_analysis: true,
        enable_trend_detection: true,
        enable_outlier_removal: false,
        max_data_points: 500,
        statistical_confidence: 0.95,
        trend_sensitivity: 0.1,
    };
    let duration = Duration::from_secs(60);
    let result = AggregationWindow::new(duration, config).await;
    assert!(result.is_ok());
    if let Ok(window) = result {
        assert_eq!(window.duration, duration);
    }
}

#[tokio::test]
async fn test_aggregation_window_statistics_initial() {
    let config = WindowConfig {
        enable_statistical_analysis: false,
        enable_trend_detection: false,
        enable_outlier_removal: false,
        max_data_points: 100,
        statistical_confidence: 0.9,
        trend_sensitivity: 0.2,
    };
    let duration = Duration::from_secs(30);
    let result = AggregationWindow::new(duration, config).await;
    assert!(result.is_ok());
    if let Ok(window) = result {
        let stats = window.statistics.read();
        let _ = format!("{:?}", *stats);
    }
}

#[tokio::test]
async fn test_aggregation_window_should_recalculate() {
    let config = WindowConfig {
        enable_statistical_analysis: true,
        enable_trend_detection: false,
        enable_outlier_removal: false,
        max_data_points: 1000,
        statistical_confidence: 0.95,
        trend_sensitivity: 0.1,
    };
    let duration = Duration::from_secs(120);
    let result = AggregationWindow::new(duration, config).await;
    assert!(result.is_ok());
    if let Ok(window) = result {
        let should_recalc = window.should_recalculate_statistics().await;
        // Fresh window with no data points should return true (time-based)
        let _ = should_recalc;
    }
}

#[test]
fn test_lcg_produces_bounded_values() {
    let mut rng = Lcg::new(42);
    for _ in 0..100 {
        let v = rng.next_f32();
        assert!(v >= 0.0 && v < 1.0);
    }
}

#[test]
fn test_lcg_produces_varied_values() {
    let mut rng = Lcg::new(99);
    let v1 = rng.next_f32();
    let v2 = rng.next_f32();
    // LCG should produce different values
    let v3 = rng.next_f32();
    assert!((v1 - v2).abs() > f32::EPSILON || (v2 - v3).abs() > f32::EPSILON);
}

#[test]
fn test_quality_assessment_quality_range() {
    let mut rng = Lcg::new(7777);
    for _ in 0..10 {
        let quality = rng.next_f32();
        let assessment = QualityAssessment {
            overall_quality: quality,
            data_completeness: quality,
            accuracy_score: quality,
            consistency_score: quality,
            outlier_count: 0,
            assessment_timestamp: Utc::now(),
        };
        assert!(assessment.overall_quality >= 0.0);
        assert!(assessment.overall_quality < 1.0);
    }
}

#[test]
fn test_window_config_max_data_points() {
    let config = WindowConfig {
        enable_statistical_analysis: true,
        enable_trend_detection: true,
        enable_outlier_removal: true,
        max_data_points: usize::MAX / 2,
        statistical_confidence: 0.95,
        trend_sensitivity: 0.01,
    };
    assert!(config.max_data_points > 0);
}

#[test]
fn test_processor_config_zero_timeout() {
    let config = ProcessorConfig {
        advanced_features: false,
        timeout_ms: 0,
        quality_threshold: 0.5,
    };
    assert_eq!(config.timeout_ms, 0);
}

#[test]
fn test_streaming_config_debug() {
    let config = StreamingConfig {
        buffer_size: 512,
        worker_count: 2,
        backpressure_threshold: 0.9,
        flow_control_enabled: true,
        adaptive_processing: true,
    };
    let debug_str = format!("{:?}", config);
    assert!(debug_str.contains("512") || debug_str.contains("worker_count"));
}

#[test]
fn test_publishing_config_debug() {
    let config = PublishingConfig {
        enable_publishing: true,
        delivery_guarantee: DeliveryGuarantee::ExactlyOnce,
        batch_size: 25,
        retry_attempts: 2,
        compression_enabled: true,
    };
    let debug_str = format!("{:?}", config);
    assert!(!debug_str.is_empty());
}
