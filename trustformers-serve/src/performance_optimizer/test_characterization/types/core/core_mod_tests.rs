//! Tests for core/mod.rs types (structs defined in mod.rs itself)

#[allow(unused_imports)]
use std::collections::HashMap;
use std::time::Duration;

// When included via #[path = "core_mod_tests.rs"] from mod.rs,
// super::* brings in all types from the core module (mod.rs re-exports).
use super::*;
use crate::performance_optimizer::test_characterization::types::quality::SafetyValidationRule;

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

    fn next_f64(&mut self) -> f64 {
        (self.next() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[test]
fn test_algorithm_selector_default() {
    let selector = AlgorithmSelector::default();
    assert!(selector.strategies.is_empty());
    assert_eq!(selector.current_optimal, "default");
    assert!((selector.confidence_threshold - 0.7).abs() < 1e-9);
    assert!(selector.benchmarks.is_empty());
    assert!(selector.criteria_weights.is_empty());
}

#[test]
fn test_intensity_calculation_engine_default() {
    let engine = IntensityCalculationEngine::default();
    assert!(engine.algorithms.is_empty());
    assert_eq!(engine.default_algorithm, "default");
    assert!(engine.calculation_history.is_empty());
    assert!(engine.calibration_data.is_empty());
}

#[test]
fn test_live_insights_new() {
    let insights = LiveInsights::new();
    assert!(insights.insights.is_empty());
}

#[test]
fn test_live_insights_default() {
    let insights = LiveInsights::default();
    assert!(insights.insights.is_empty());
}

#[test]
fn test_live_insights_merge() {
    let mut a = LiveInsights {
        insights: vec!["insight_a".to_string()],
        timestamp: chrono::Utc::now(),
    };
    let b = LiveInsights {
        insights: vec!["insight_b".to_string(), "insight_c".to_string()],
        timestamp: chrono::Utc::now() + chrono::Duration::seconds(1),
    };
    a.merge(&b);
    assert_eq!(a.insights.len(), 3);
    assert!(a.insights.contains(&"insight_b".to_string()));
}

#[test]
fn test_test_characteristics_default() {
    let tc = TestCharacteristics::default();
    assert_eq!(tc.test_id, "");
    assert!((tc.quality_score - 0.0).abs() < 1e-9);
    assert!((tc.confidence_level - 0.0).abs() < 1e-9);
    assert!(tc.recommendations.is_empty());
    assert!(tc.detected_patterns.is_empty());
}

#[test]
fn test_test_characteristics_from_test_data() {
    let ri = ResourceIntensity::default();
    let cr = ConcurrencyRequirements::default();
    let sr = SynchronizationRequirements::default();
    let tc = TestCharacteristics::from_test_data(
        "test_001".to_string(),
        ri,
        cr,
        sr,
    );
    assert_eq!(tc.test_id, "test_001");
    assert!(tc.synchronization_dependencies.is_empty());
    assert!(tc.performance_patterns.is_empty());
}

#[test]
fn test_test_execution_data_default() {
    let data = TestExecutionData::default();
    assert_eq!(data.test_id, "");
    assert!(data.resource_access_patterns.is_empty());
    assert!(data.thread_interactions.is_empty());
    assert!(data.system_snapshots.is_empty());
}

#[test]
fn test_test_pattern_recognition_engine_new() {
    let engine = TestPatternRecognitionEngine::new();
    assert!(engine.enabled);
    assert!((engine.confidence_threshold - 0.8).abs() < 1e-9);
    assert_eq!(engine.algorithms.len(), 1);
    assert_eq!(engine.history.total_recognitions, 0);
}

#[test]
fn test_test_pattern_recognition_engine_default() {
    let engine = TestPatternRecognitionEngine::default();
    assert!(engine.enabled);
}

#[test]
fn test_estimation_result_fields() {
    let result = EstimationResult {
        algorithm: "conservative".to_string(),
        concurrency: 4,
        confidence: 0.9,
        duration: Duration::from_millis(50),
    };
    assert_eq!(result.algorithm, "conservative");
    assert_eq!(result.concurrency, 4);
    assert!((result.confidence - 0.9).abs() < 1e-9);
}

#[test]
fn test_conservative_estimation_algorithm_new() {
    let algo = ConservativeEstimationAlgorithm::new(0.2, true);
    assert!((algo.safety_margin - 0.2).abs() < 1e-9);
    assert!(algo.worst_case);
}

#[test]
fn test_conservative_estimation_fields() {
    // Test that the struct fields are accessible
    let algo = ConservativeEstimationAlgorithm::new(0.2, true);
    assert!((algo.safety_margin - 0.2).abs() < 1e-9);
    assert!(algo.worst_case);
    let algo2 = ConservativeEstimationAlgorithm::new(0.0, false);
    assert!(!algo2.worst_case);
}

#[test]
fn test_optimistic_estimation_algorithm_new() {
    let algo = OptimisticEstimationAlgorithm::new(true, 1.2);
    assert!(algo.best_case);
    assert!((algo.optimism_factor - 1.2).abs() < 1e-9);
}

#[test]
fn test_ml_based_estimation_algorithm_new() {
    let algo = MLBasedEstimationAlgorithm::new("linear_regression".to_string(), 0.85);
    assert_eq!(algo.model, "linear_regression");
    assert!((algo.confidence - 0.85).abs() < 1e-9);
}

#[test]
fn test_priority_level_ordering() {
    assert!(PriorityLevel::Critical > PriorityLevel::High);
    assert!(PriorityLevel::High > PriorityLevel::Medium);
    assert!(PriorityLevel::Medium > PriorityLevel::Low);
    assert!(PriorityLevel::Low > PriorityLevel::Lowest);
}

#[test]
fn test_priority_level_equality() {
    let p1 = PriorityLevel::High;
    let p2 = PriorityLevel::High;
    assert_eq!(p1, p2);
    assert_ne!(p1, PriorityLevel::Low);
}

#[test]
fn test_resolution_type_variants() {
    let variants = [
        ResolutionType::Avoidance,
        ResolutionType::Mitigation,
        ResolutionType::Isolation,
        ResolutionType::Scheduling,
        ResolutionType::ResourceAllocation,
        ResolutionType::Serialization,
        ResolutionType::Optimization,
    ];
    assert_eq!(variants.len(), 7);
    for v in &variants {
        let cloned = *v;
        assert_eq!(cloned, *v);
    }
}

#[test]
fn test_stream_status_variants() {
    let variants = [
        StreamStatus::Active,
        StreamStatus::Inactive,
        StreamStatus::Paused,
        StreamStatus::Stopped,
        StreamStatus::Error,
        StreamStatus::Initializing,
        StreamStatus::Buffering,
        StreamStatus::Draining,
    ];
    assert_eq!(variants.len(), 8);
}

#[test]
fn test_intensity_calculation_method_variants() {
    let m1 = IntensityCalculationMethod::MovingAverage;
    let m2 = IntensityCalculationMethod::ExponentialWeighted;
    let m3 = IntensityCalculationMethod::Peak;
    let m4 = IntensityCalculationMethod::Custom("my_algo".to_string());
    assert_ne!(m1, m2);
    assert_ne!(m1, m3);
    if let IntensityCalculationMethod::Custom(ref s) = m4 {
        assert_eq!(s, "my_algo");
    } else {
        panic!("expected Custom variant");
    }
}

#[test]
fn test_complexity_level_variants() {
    let variants = [
        ComplexityLevel::VerySimple,
        ComplexityLevel::Simple,
        ComplexityLevel::Medium,
        ComplexityLevel::Complex,
        ComplexityLevel::VeryComplex,
        ComplexityLevel::HighlyComplex,
        ComplexityLevel::ExtremelyComplex,
    ];
    assert_eq!(variants.len(), 7);
    for v in &variants {
        let cloned = *v;
        assert_eq!(cloned, *v);
    }
}

#[test]
fn test_operation_result_variants() {
    let r1 = OperationResult::Success;
    let r2 = OperationResult::Failure;
    let r3 = OperationResult::Timeout;
    assert_ne!(r1, r2);
    assert_ne!(r2, r3);
    let cloned = r1;
    assert_eq!(cloned, r1);
}

#[test]
fn test_potential_deadlock_construction() {
    let dl = PotentialDeadlock {
        locks: vec!["lock_a".to_string(), "lock_b".to_string()],
    };
    assert_eq!(dl.locks.len(), 2);
    let cloned = dl.clone();
    assert_eq!(cloned.locks.len(), 2);
}

#[test]
fn test_resolution_action_construction() {
    let action = ResolutionAction {
        action_id: "act_001".to_string(),
        action_type: "reschedule".to_string(),
        description: "Reschedule conflicting tasks".to_string(),
        priority: PriorityLevel::High,
        urgency: UrgencyLevel::Medium,
        estimated_duration: Duration::from_secs(10),
        estimated_time: Duration::from_secs(10),
        dependencies: vec![],
        success_criteria: vec!["no conflict".to_string()],
        rollback_procedure: None,
        parameters: HashMap::new(),
    };
    assert_eq!(action.action_id, "act_001");
    assert!(action.rollback_procedure.is_none());
    assert_eq!(action.success_criteria.len(), 1);
}

#[test]
fn test_prediction_model_predict() {
    let model = PredictionModel {
        model_type: "linear".to_string(),
        accuracy: 0.9,
        trained_at: chrono::Utc::now(),
    };
    let input = vec![1.0, 2.0, 3.0];
    let result = model.predict(&input);
    match result {
        Ok(output) => assert_eq!(output.len(), 3),
        Err(_) => panic!("predict should not fail"),
    }
}

#[test]
fn test_producer_consumer_detection_default() {
    let detection = ProducerConsumerDetection::default();
    assert!(!detection.detected);
    assert_eq!(detection.producer_count, 0);
    assert_eq!(detection.consumer_count, 0);
}

#[test]
fn test_master_worker_detection_default() {
    let detection = MasterWorkerDetection::default();
    assert!(!detection.detected);
    assert_eq!(detection.master_count, 0);
    assert_eq!(detection.worker_count, 0);
}

#[test]
fn test_pipeline_detection_default() {
    let detection = PipelineDetection::default();
    assert!(!detection.detected);
    assert_eq!(detection.stages, 0);
    assert!((detection.throughput - 0.0).abs() < 1e-9);
}

#[test]
fn test_fork_join_detection_default() {
    let detection = ForkJoinDetection::default();
    assert!(!detection.detected);
    assert_eq!(detection.fork_points, 0);
    assert_eq!(detection.join_points, 0);
}

#[test]
fn test_priority_calculator_fields() {
    let calc = PriorityCalculator {
        calculation_method: "weighted".to_string(),
        weights: {
            let mut m = HashMap::new();
            m.insert("cpu".to_string(), 0.4);
            m.insert("mem".to_string(), 0.6);
            m
        },
    };
    assert_eq!(calc.calculation_method, "weighted");
    assert_eq!(calc.weights.len(), 2);
}

#[test]
fn test_priority_ranking_fields() {
    let ranking = PriorityRanking {
        rank: 1,
        score: 0.95,
        item_id: "item_001".to_string(),
    };
    assert_eq!(ranking.rank, 1);
    assert!((ranking.score - 0.95).abs() < 1e-9);
}

#[test]
fn test_duration_statistics_default() {
    let stats = DurationStatistics::default();
    assert_eq!(stats.sample_count, 0);
    assert!((stats.variance - 0.0).abs() < 1e-9);
    assert_eq!(stats.min, Duration::from_secs(0));
}

#[test]
fn test_pipeline_config_default() {
    let config = PipelineConfig::default();
    assert_eq!(config.worker_threads, 0);
    assert!(!config.enable_parallel_execution);
    assert!(!config.enable_conflict_resolution);
}

#[test]
fn test_real_time_profiler_config_default() {
    let config = RealTimeProfilerConfig::default();
    assert_eq!(config.sampling_frequency, 0);
    assert!(!config.enable_streaming_analysis);
    assert!(!config.enable_adaptive_optimization);
    assert!(config.alert_thresholds.is_empty());
}

#[test]
fn test_streaming_analyzer_config_default() {
    let config = StreamingAnalyzerConfig::default();
    assert_eq!(config.window_size, 1000);
    assert!(config.enable_trend_analysis);
    assert!(config.anomaly_detection_enabled);
    assert_eq!(config.buffer_size, 10000);
}

#[test]
fn test_isolation_safety_rule_new() {
    let rule = IsolationSafetyRule::new();
    assert_eq!(rule.isolation_level, "default");
    assert!(rule.enforce_boundaries);
    assert!(rule.cross_contamination_check);
}

#[test]
fn test_isolation_safety_rule_validate() {
    let rule = IsolationSafetyRule::default();
    assert!(SafetyValidationRule::validate(&rule));

    let invalid = IsolationSafetyRule {
        isolation_level: "none".to_string(),
        enforce_boundaries: false,
        cross_contamination_check: true,
    };
    assert!(!SafetyValidationRule::validate(&invalid));
}

#[test]
fn test_threshold_anomaly_detector_default() {
    let detector = ThresholdAnomalyDetector::default();
    assert!((detector.upper_threshold - 100.0).abs() < 1e-9);
    assert!((detector.lower_threshold - 0.0).abs() < 1e-9);
    assert_eq!(detector.anomalies_detected, 0);
}

#[test]
fn test_buffer_size_optimizer_new() {
    let optimizer = BufferSizeOptimizer::new(1024, 4096);
    assert_eq!(optimizer.current_size, 1024);
    assert_eq!(optimizer.optimal_size, 4096);
}

#[test]
fn test_sampling_rate_optimizer_default() {
    let optimizer = SamplingRateOptimizer::default();
    assert!((optimizer.current_rate - 1.0).abs() < 1e-9);
    assert!((optimizer.target_rate - 1.0).abs() < 1e-9);
}

#[test]
fn test_real_time_dashboard_fields() {
    let dashboard = RealTimeDashboard {
        refresh_interval: Duration::from_secs(5),
        metrics: vec!["cpu".to_string(), "memory".to_string()],
    };
    assert_eq!(dashboard.refresh_interval, Duration::from_secs(5));
    assert_eq!(dashboard.metrics.len(), 2);
}
