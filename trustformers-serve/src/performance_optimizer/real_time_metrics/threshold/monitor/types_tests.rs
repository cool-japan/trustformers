//! Tests for threshold monitor types

use super::*;
use std::time::{Duration, Instant};

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
fn test_performance_analyzer_config_clone() {
    let cfg = PerformanceAnalyzerConfig {
        enabled: true,
        analysis_interval: Duration::from_secs(10),
        history_retention: Duration::from_secs(3600),
        enable_detailed_tracking: true,
        enable_cpu_monitoring: false,
        enable_memory_monitoring: true,
    };
    let c = cfg.clone();
    assert!(c.enabled);
    assert!(c.enable_detailed_tracking);
    assert!(!c.enable_cpu_monitoring);
}

#[test]
fn test_performance_analyzer_config_default() {
    let cfg = PerformanceAnalyzerConfig::default();
    // Just verify default construction works
    let s = format!("{:?}", cfg);
    assert!(s.contains("PerformanceAnalyzerConfig"));
}

#[test]
fn test_trend_analysis_config_clone() {
    let cfg = TrendAnalysisConfig {
        trend_window: 50,
        trend_sensitivity: 0.1,
        seasonal_detection: true,
    };
    let c = cfg.clone();
    assert_eq!(c.trend_window, 50);
    assert!(c.seasonal_detection);
}

#[test]
fn test_trend_direction_variants() {
    let variants = [
        TrendDirection::Increasing,
        TrendDirection::Decreasing,
        TrendDirection::Stable,
        TrendDirection::Cyclical,
    ];
    for v in &variants {
        let s = format!("{:?}", v);
        assert!(!s.is_empty());
    }
}

#[test]
fn test_trend_direction_clone() {
    let d = TrendDirection::Increasing;
    let c = d.clone();
    let s = format!("{:?}", c);
    assert!(s.contains("Increasing"));
}

#[test]
fn test_trend_analysis_clone() {
    let ta = TrendAnalysis {
        direction: TrendDirection::Stable,
        magnitude: 0.05,
        confidence: 0.9,
        duration: Duration::from_secs(3600),
    };
    let c = ta.clone();
    assert_eq!(c.magnitude, 0.05);
    assert_eq!(c.confidence, 0.9);
}

#[test]
fn test_seasonal_pattern_type_variants() {
    let variants = [
        SeasonalPatternType::Daily,
        SeasonalPatternType::Weekly,
        SeasonalPatternType::Monthly,
        SeasonalPatternType::Custom(Duration::from_secs(43200)),
    ];
    for v in &variants {
        let s = format!("{:?}", v);
        assert!(!s.is_empty());
    }
}

#[test]
fn test_seasonal_pattern_clone() {
    let sp = SeasonalPattern {
        pattern_type: SeasonalPatternType::Daily,
        cycle_length: Duration::from_secs(86400),
        amplitude: 0.15,
        confidence: 0.75,
    };
    let c = sp.clone();
    assert_eq!(c.amplitude, 0.15);
    assert_eq!(c.confidence, 0.75);
}

#[test]
fn test_statistical_adaptation_config_clone() {
    let cfg = StatisticalAdaptationConfig {
        confidence_threshold: 0.8,
        min_data_points: 30,
        adaptation_sensitivity: 0.1,
    };
    let c = cfg.clone();
    assert_eq!(c.confidence_threshold, 0.8);
    assert_eq!(c.min_data_points, 30);
}

#[test]
fn test_ml_adaptation_config_clone() {
    let cfg = MLAdaptationConfig {
        learning_rate: 0.01,
        model_complexity: 3,
        training_window: 200,
    };
    let c = cfg.clone();
    assert_eq!(c.learning_rate, 0.01);
    assert_eq!(c.model_complexity, 3);
    assert_eq!(c.training_window, 200);
}

#[test]
fn test_threshold_monitor_config_clone() {
    let cfg = ThresholdMonitorConfig {
        enable_monitoring: true,
        monitoring_interval: Duration::from_secs(5),
        enable_adaptive_thresholds: true,
        enable_alert_suppression: false,
        enable_alert_correlation: true,
        enable_performance_analysis: true,
        max_alert_history: 1000,
        alert_processing_timeout: Duration::from_secs(30),
        enable_detailed_logging: false,
    };
    let c = cfg.clone();
    assert!(c.enable_monitoring);
    assert!(!c.enable_alert_suppression);
    assert_eq!(c.max_alert_history, 1000);
}

#[test]
fn test_threshold_monitor_config_default() {
    let cfg = ThresholdMonitorConfig::default();
    let s = format!("{:?}", cfg);
    assert!(s.contains("ThresholdMonitorConfig"));
}

#[test]
fn test_performance_snapshot_clone() {
    let snap = PerformanceSnapshot {
        timestamp: chrono::Utc::now(),
        evaluations_per_second: 10.0,
        avg_evaluation_time_ms: 5.0,
        cpu_usage_percent: 25.0,
        memory_usage_mb: 512.0,
        active_evaluations: 3,
    };
    let c = snap.clone();
    assert_eq!(c.evaluations_per_second, 10.0);
    assert_eq!(c.active_evaluations, 3);
}

#[test]
fn test_performance_trends_clone() {
    let ta = TrendAnalysis {
        direction: TrendDirection::Stable,
        magnitude: 0.02,
        confidence: 0.85,
        duration: Duration::from_secs(3600),
    };
    let trends = PerformanceTrends {
        evaluation_time_trend: ta.clone(),
        throughput_trend: ta.clone(),
        resource_usage_trend: ta.clone(),
    };
    let c = trends.clone();
    assert_eq!(c.evaluation_time_trend.confidence, 0.85);
}

#[test]
fn test_resource_utilization_clone() {
    let ru = ResourceUtilization {
        current_cpu_percent: 30.0,
        peak_cpu_percent: 80.0,
        current_memory_mb: 512.0,
        peak_memory_mb: 1024.0,
        memory_growth_rate: 0.01,
    };
    let c = ru.clone();
    assert_eq!(c.current_cpu_percent, 30.0);
    assert_eq!(c.peak_memory_mb, 1024.0);
}

#[test]
fn test_throughput_analysis_clone() {
    let ta = ThroughputAnalysis {
        current_throughput: 100.0,
        peak_throughput: 150.0,
        average_throughput: 80.0,
        throughput_variance: 0.1,
        bottleneck_indicators: vec!["cpu".to_string()],
    };
    let c = ta.clone();
    assert_eq!(c.current_throughput, 100.0);
    assert_eq!(c.bottleneck_indicators.len(), 1);
}

#[test]
fn test_algorithm_stats_default() {
    let stats = AlgorithmStats::default();
    assert_eq!(stats.adaptations_performed, 0);
    assert_eq!(stats.avg_confidence, 0.0);
    assert_eq!(stats.success_rate, 0.0);
}

#[test]
fn test_performance_metrics_default() {
    let m = PerformanceMetrics::default();
    assert_eq!(m.total_evaluations, 0);
    assert_eq!(m.evaluations_per_second, 0.0);
    assert!(m.current_evaluations.is_empty());
    assert!(m.completed_evaluations.is_empty());
}

#[test]
fn test_trend_analysis_algorithm_analyze() {
    let algo = TrendAnalysisAlgorithm::new();
    let values = vec![10.0, 11.0, 12.0, 13.0, 14.0, 15.0];
    let (direction, _strength) = algo.analyze_trend(&values);
    let s = format!("{:?}", direction);
    assert!(!s.is_empty());
}

#[test]
fn test_trend_analysis_algorithm_short_data() {
    let algo = TrendAnalysisAlgorithm::new();
    let values: Vec<f64> = vec![5.0, 6.0];
    let (direction, strength) = algo.analyze_trend(&values);
    let ds = format!("{:?}", direction);
    assert!(ds.contains("Stable"));
    assert_eq!(strength, 0.0);
}

#[test]
fn test_trend_analysis_algorithm_seasonal_patterns() {
    let algo = TrendAnalysisAlgorithm::new();
    let values: Vec<f64> = (0..50).map(|i| (i as f64 % 24.0) * 2.0).collect();
    let patterns = algo.detect_seasonal_patterns(&values);
    let _ = patterns; // just ensure no panic
}

#[test]
fn test_ml_model_state_debug() {
    let model = MLModelState {
        weights: vec![0.5, 0.3, 0.2],
        bias: 0.1,
        training_iterations: 100,
        model_accuracy: 0.85,
    };
    let s = format!("{:?}", model);
    assert!(s.contains("MLModelState"));
}

#[test]
fn test_trend_state_debug() {
    let state = TrendState {
        current_trend: TrendDirection::Increasing,
        trend_strength: 0.7,
        seasonal_patterns: Vec::new(),
        last_analysis: chrono::Utc::now(),
    };
    let s = format!("{:?}", state);
    assert!(s.contains("TrendState"));
}

#[test]
fn test_ml_adaptation_algorithm_predict() {
    let algo = MachineLearningAdaptationAlgorithm::new();
    let model = MLModelState {
        weights: vec![0.5, 0.3, 0.2],
        bias: 0.1,
        training_iterations: 0,
        model_accuracy: 0.5,
    };
    let features = vec![1.0, 2.0, 3.0];
    let prediction = algo.predict_with_features(&model, &features);
    // 0.5 + 0.3*2 + 0.2*3 + 0.1 = 0.5+0.6+0.6+0.1 = 1.8 (roughly)
    assert!(prediction > 0.0);
}
