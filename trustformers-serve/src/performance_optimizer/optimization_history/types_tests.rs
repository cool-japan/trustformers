//! Tests for Optimization History Types
//!
//! Comprehensive tests for configuration types, Default implementations,
//! data structure construction, type aliases, and enum variants.

use super::*;
use std::collections::HashMap;
use std::time::Duration;

// =============================================================================
// CONFIGURATION DEFAULT TESTS
// =============================================================================

#[test]
fn test_history_retention_config_default() {
    let config = HistoryRetentionConfig::default();
    assert_eq!(config.max_events, 10000);
    assert!(config.auto_cleanup);
    assert!(config.max_age > Duration::from_secs(0));
    assert!(config.cleanup_interval > Duration::from_secs(0));
    assert!(config.compression_threshold > Duration::from_secs(0));
}

#[test]
fn test_history_retention_config_custom() {
    let config = HistoryRetentionConfig {
        max_events: 500,
        max_age: Duration::from_secs(3600),
        auto_cleanup: false,
        cleanup_interval: Duration::from_secs(60),
        compression_threshold: Duration::from_secs(120),
    };
    assert_eq!(config.max_events, 500);
    assert!(!config.auto_cleanup);
}

#[test]
fn test_trend_analysis_config_default() {
    let config = TrendAnalysisConfig::default();
    assert_eq!(config.min_data_points, 5);
    assert!(config.confidence_threshold > 0.0 && config.confidence_threshold <= 1.0);
    assert!(config.enable_prediction);
    assert!(config.enable_ml_models);
    assert!(config.cache_expiry > Duration::from_secs(0));
}

#[test]
fn test_trend_analysis_config_custom() {
    let config = TrendAnalysisConfig {
        min_data_points: 10,
        analysis_window: Duration::from_secs(7200),
        confidence_threshold: 0.9,
        enable_prediction: false,
        prediction_horizon: Duration::from_secs(600),
        cache_expiry: Duration::from_secs(120),
        enable_ml_models: false,
    };
    assert_eq!(config.min_data_points, 10);
    assert!(!config.enable_prediction);
    assert!(!config.enable_ml_models);
}

#[test]
fn test_pattern_recognition_config_default() {
    let config = PatternRecognitionConfig::default();
    assert_eq!(config.min_pattern_length, 3);
    assert!(config.similarity_threshold > 0.0 && config.similarity_threshold <= 1.0);
    assert!(config.enable_ml_learning);
    assert!(config.learning_rate > 0.0);
    assert!(config.cache_size > 0);
}

#[test]
fn test_anomaly_detection_config_default() {
    let config = AnomalyDetectionConfig::default();
    assert!(config.enable_detection);
    assert!(config.sensitivity > 0.0 && config.sensitivity <= 1.0);
    assert!(config.min_severity_threshold > 0.0);
    assert!(config.enable_ml_detection);
    assert!(config.learning_rate > 0.0);
}

#[test]
fn test_effectiveness_analysis_config_default() {
    let config = EffectivenessAnalysisConfig::default();
    assert!(config.enable_roi_calculation);
    assert!(config.enable_significance_testing);
    assert!(config.confidence_level > 0.0 && config.confidence_level <= 1.0);
    assert!(config.min_effectiveness_threshold > 0.0);
    if let CostCalculationMethod::ResourceBased = config.cost_calculation_method {
        // Expected default
    } else {
        panic!("Expected ResourceBased cost calculation method");
    }
}

#[test]
fn test_statistics_config_default() {
    let config = StatisticsConfig::default();
    assert!(config.enable_advanced_metrics);
    assert_eq!(config.window_size, 100);
    assert!(config.enable_correlation_analysis);
    assert!(config.enable_distribution_analysis);
    assert!(config.significance_threshold > 0.0 && config.significance_threshold < 1.0);
}

#[test]
fn test_predictive_analytics_config_default() {
    let config = PredictiveAnalyticsConfig::default();
    assert!(config.enable_prediction);
    assert!(!config.prediction_models.is_empty());
    assert!(config.prediction_horizon > Duration::from_secs(0));
    assert!(config.model_update_frequency > Duration::from_secs(0));
    assert!(config.min_data_points > 0);
}

// =============================================================================
// TREND ANALYSIS TYPES TESTS
// =============================================================================

#[test]
fn test_cached_trend_analysis_construction() {
    let trend = LegacyPerformanceTrend {
        direction: crate::test_performance_monitoring::TrendDirection::Improving,
        strength: 0.8,
        confidence: 0.9,
        period: Duration::from_secs(3600),
        data_points: Vec::new(),
    };
    let cached = CachedTrendAnalysis {
        trend,
        cached_at: chrono::Utc::now(),
        analysis_duration: Duration::from_millis(100),
        confidence: 0.9,
    };
    assert!(cached.confidence > 0.0);
    assert!(cached.analysis_duration > Duration::from_secs(0));
}

#[test]
fn test_trend_analysis_result_construction() {
    let result = TrendAnalysisResult {
        direction: crate::test_performance_monitoring::TrendDirection::Improving,
        strength: 0.75,
        confidence: 0.85,
        duration: Duration::from_secs(600),
        significance: 0.95,
        data_points: Vec::new(),
        method: "linear_regression".to_string(),
        metrics: HashMap::new(),
    };
    assert!(result.strength > 0.0 && result.strength <= 1.0);
    assert!(result.confidence > 0.0 && result.confidence <= 1.0);
    assert!(!result.method.is_empty());
}

#[test]
fn test_trend_prediction_construction() {
    let prediction = TrendPrediction {
        predicted_direction: crate::test_performance_monitoring::TrendDirection::Stable,
        confidence: 0.7,
        horizon: Duration::from_secs(1800),
        expected_values: Vec::new(),
        model: "moving_average".to_string(),
        uncertainty: 0.2,
    };
    assert!(prediction.confidence > 0.0);
    assert!(prediction.uncertainty >= 0.0);
}

// =============================================================================
// PATTERN RECOGNITION TYPES TESTS
// =============================================================================

#[test]
fn test_pattern_type_variants() {
    let cyclical = PatternType::Cyclical;
    let degradation = PatternType::Degradation;
    let improvement = PatternType::Improvement;
    let oscillation = PatternType::Oscillation;
    let threshold = PatternType::Threshold;
    let custom = PatternType::Custom("my_pattern".to_string());

    assert_eq!(cyclical, PatternType::Cyclical);
    assert_eq!(degradation, PatternType::Degradation);
    assert_eq!(improvement, PatternType::Improvement);
    assert_eq!(oscillation, PatternType::Oscillation);
    assert_eq!(threshold, PatternType::Threshold);
    assert_ne!(custom, PatternType::Cyclical);
}

#[test]
fn test_recognized_pattern_construction() {
    let pattern = RecognizedPattern {
        id: "pat_001".to_string(),
        pattern_type: PatternType::Cyclical,
        description: "Daily performance cycle".to_string(),
        frequency: 0.95,
        confidence: 0.88,
        events: Vec::new(),
        effectiveness: 0.72,
        first_observed: chrono::Utc::now(),
        last_observed: chrono::Utc::now(),
        characteristics: HashMap::new(),
    };
    assert!(!pattern.id.is_empty());
    assert!(pattern.frequency > 0.0);
    assert!(pattern.confidence > 0.0);
}

// =============================================================================
// ANOMALY DETECTION TYPES TESTS
// =============================================================================

#[test]
fn test_anomaly_type_variants() {
    let spike = AnomalyType::Spike;
    let drop_type = AnomalyType::Drop;
    let unusual = AnomalyType::UnusualPattern;
    let instability = AnomalyType::SystemInstability;
    let exhaustion = AnomalyType::ResourceExhaustion;
    let custom = AnomalyType::Custom("custom_anomaly".to_string());

    assert_eq!(spike, AnomalyType::Spike);
    assert_eq!(drop_type, AnomalyType::Drop);
    assert_ne!(unusual, instability);
    assert_ne!(exhaustion, custom);
}

#[test]
fn test_detected_anomaly_construction() {
    let anomaly = DetectedAnomaly {
        id: "anom_001".to_string(),
        anomaly_type: AnomalyType::Spike,
        description: "Performance spike detected".to_string(),
        severity: 0.9,
        confidence: 0.85,
        data_point: crate::performance_optimizer::types::PerformanceDataPoint {
            parallelism: 1,
            throughput: 500.0,
            latency: Duration::from_millis(10),
            cpu_utilization: 0.95,
            memory_utilization: 0.8,
            resource_efficiency: 0.7,
            timestamp: chrono::Utc::now(),
            test_characteristics: crate::performance_optimizer::types::TestCharacteristics::default(
            ),
            system_state: crate::performance_optimizer::types::SystemState::default(),
        },
        detected_at: chrono::Utc::now(),
        expected_range: (50.0, 200.0),
        deviation: 300.0,
        detection_method: "z_score".to_string(),
        metadata: HashMap::new(),
    };
    assert!(anomaly.severity > 0.0 && anomaly.severity <= 1.0);
    assert!(anomaly.deviation > anomaly.expected_range.1);
}

#[test]
fn test_performance_baseline_construction() {
    let baseline = PerformanceBaseline {
        baseline_throughput: 100.0,
        baseline_latency: Duration::from_millis(50),
        throughput_variance: 25.0,
        latency_variance: 10.0,
        baseline_timestamp: chrono::Utc::now(),
        sample_size: 100,
        confidence_interval: (90.0, 110.0),
    };
    assert!(baseline.baseline_throughput > 0.0);
    assert!(baseline.sample_size > 0);
    assert!(baseline.confidence_interval.0 < baseline.confidence_interval.1);
}

// =============================================================================
// EFFECTIVENESS ANALYSIS TYPES TESTS
// =============================================================================

#[test]
fn test_cost_benefit_analysis_construction() {
    let cba = CostBenefitAnalysis {
        implementation_cost: 100.0,
        operational_cost: 50.0,
        total_cost: 150.0,
        performance_benefit: 300.0,
        resource_savings: 75.0,
        total_benefit: 375.0,
        net_benefit: 225.0,
        payback_period: Duration::from_secs(3600),
    };
    assert!((cba.total_cost - (cba.implementation_cost + cba.operational_cost)).abs() < 1e-10);
    assert!(cba.net_benefit > 0.0);
}

#[test]
fn test_performance_improvement_construction() {
    let improvement = PerformanceImprovement {
        throughput_improvement: 0.25,
        latency_improvement: 0.15,
        resource_improvement: 0.10,
        overall_improvement: 0.20,
        improvement_duration: Duration::from_secs(1800),
    };
    assert!(improvement.throughput_improvement > 0.0);
    assert!(improvement.overall_improvement > 0.0);
}

#[test]
fn test_statistical_significance_construction() {
    let sig = StatisticalSignificance {
        test_statistic: 2.45,
        p_value: 0.014,
        confidence_level: 0.95,
        is_significant: true,
        test_method: "t_test".to_string(),
    };
    assert!(sig.is_significant);
    assert!(sig.p_value < 0.05);
}

// =============================================================================
// COST CALCULATION METHOD TESTS
// =============================================================================

#[test]
fn test_cost_calculation_method_variants() {
    let resource = CostCalculationMethod::ResourceBased;
    let time = CostCalculationMethod::TimeBased;
    let hybrid = CostCalculationMethod::Hybrid;
    let custom = CostCalculationMethod::Custom("custom_method".to_string());
    // Verify Debug implementation
    let _dbg_resource = format!("{:?}", resource);
    let _dbg_time = format!("{:?}", time);
    let _dbg_hybrid = format!("{:?}", hybrid);
    let _dbg_custom = format!("{:?}", custom);
}

// =============================================================================
// STATISTICS TYPES TESTS
// =============================================================================

#[test]
fn test_basic_statistics_construction() {
    let stats = BasicStatistics {
        mean: 50.0,
        median: 48.0,
        std_dev: 10.0,
        variance: 100.0,
        min: 20.0,
        max: 80.0,
        range: 60.0,
        skewness: 0.1,
        kurtosis: -0.5,
    };
    assert!((stats.variance - stats.std_dev * stats.std_dev).abs() < 1e-10);
    assert!((stats.range - (stats.max - stats.min)).abs() < 1e-10);
}

#[test]
fn test_distribution_type_default() {
    let dist_type = DistributionType::default();
    if let DistributionType::Normal = dist_type {
        // Expected
    } else {
        panic!("Expected Normal as default distribution type");
    }
}

#[test]
fn test_distribution_type_variants() {
    let normal = DistributionType::Normal;
    let exp = DistributionType::Exponential;
    let uniform = DistributionType::Uniform;
    let gamma = DistributionType::Gamma;
    let custom = DistributionType::Custom("weibull".to_string());
    let _dbg = format!(
        "{:?} {:?} {:?} {:?} {:?}",
        normal, exp, uniform, gamma, custom
    );
}

#[test]
fn test_prediction_model_type_variants() {
    let lr = PredictionModelType::LinearRegression;
    let ma = PredictionModelType::MovingAverage;
    let es = PredictionModelType::ExponentialSmoothing;
    let arima = PredictionModelType::ARIMA;
    let nn = PredictionModelType::NeuralNetwork;
    let custom = PredictionModelType::Custom("xgboost".to_string());
    assert_eq!(lr, PredictionModelType::LinearRegression);
    assert_ne!(ma, es);
    assert_ne!(arima, nn);
    let _dbg = format!("{:?}", custom);
}

// =============================================================================
// LEGACY TYPES TESTS
// =============================================================================

#[test]
fn test_legacy_optimization_history_default() {
    let history = LegacyOptimizationHistory::default();
    assert!(history.events.is_empty());
    assert!(history.trends.is_empty());
}

#[test]
fn test_legacy_optimization_effectiveness_default() {
    let eff = LegacyOptimizationEffectiveness::default();
    assert!((eff.overall_score).abs() < 1e-10);
    assert!(eff.by_type.is_empty());
}

#[test]
fn test_legacy_optimization_statistics_default() {
    let stats = LegacyOptimizationStatistics::default();
    assert_eq!(stats.total_optimizations, 0);
    assert_eq!(stats.successful_optimizations, 0);
}

#[test]
fn test_legacy_optimization_event_construction() {
    let event = LegacyOptimizationEvent {
        timestamp: chrono::Utc::now(),
        event_type: OptimizationEventType::ParallelismAdjustment,
        description: "Test optimization".to_string(),
        performance_before: None,
        performance_after: None,
        parameters: HashMap::new(),
        metadata: HashMap::new(),
    };
    assert!(!event.description.is_empty());
}

#[test]
fn test_model_performance_metrics_construction() {
    let metrics = ModelPerformanceMetrics {
        mae: 0.05,
        mse: 0.003,
        rmse: 0.055,
        r_squared: 0.92,
        mape: 0.04,
    };
    assert!(metrics.rmse >= 0.0);
    assert!(metrics.r_squared >= 0.0 && metrics.r_squared <= 1.0);
    assert!(metrics.mae >= 0.0);
}

#[test]
fn test_type_aliases_exist() {
    // Verify type aliases compile
    let _history: OptimizationHistory = LegacyOptimizationHistory::default();
    let _stats: OptimizationStatistics = LegacyOptimizationStatistics::default();
    let _eff: OptimizationEffectiveness = LegacyOptimizationEffectiveness::default();
}
