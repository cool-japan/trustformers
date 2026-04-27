//! Tests for test_characterization/types/core/mod.rs
//!
//! Comprehensive tests for core types including algorithm selectors,
//! calculation engines, estimation algorithms, conflict detection,
//! and pattern detection algorithms.

use super::*;

// =============================================================================
// ALGORITHM SELECTOR TESTS
// =============================================================================

#[test]
fn test_algorithm_selector_default() {
    let selector = AlgorithmSelector::default();
    assert_eq!(selector.current_optimal, "default");
    assert!(selector.confidence_threshold > 0.0);
}

#[test]
fn test_algorithm_selector_default_has_empty_collections() {
    let selector = AlgorithmSelector::default();
    assert!(selector.performance_tracker.is_empty());
    assert!(selector.selection_history.is_empty());
    assert!(selector.benchmarks.is_empty());
}

// =============================================================================
// INTENSITY CALCULATION ENGINE TESTS
// =============================================================================

#[test]
fn test_intensity_calculation_engine_default() {
    let engine = IntensityCalculationEngine::default();
    let _dbg = format!("{:?}", engine);
}

// =============================================================================
// LIVE INSIGHTS TESTS
// =============================================================================

#[test]
fn test_live_insights_default() {
    let insights = LiveInsights::default();
    assert!(insights.insights.is_empty());
}

#[test]
fn test_live_insights_merge() {
    let mut insights_a = LiveInsights::new();
    insights_a.insights.push("insight_1".to_string());
    let mut insights_b = LiveInsights::new();
    insights_b.insights.push("insight_2".to_string());
    insights_a.merge(&insights_b);
    assert_eq!(insights_a.insights.len(), 2);
}

// =============================================================================
// TEST EXECUTION DATA TESTS
// =============================================================================

#[test]
fn test_test_execution_data_default() {
    let data = TestExecutionData::default();
    assert!(data.test_id.is_empty());
}

// =============================================================================
// TEST CHARACTERISTICS TESTS
// =============================================================================

#[test]
fn test_test_characteristics_default() {
    let chars = TestCharacteristics::default();
    let _dbg = format!("{:?}", chars);
}

// =============================================================================
// TEST PATTERN RECOGNITION ENGINE TESTS
// =============================================================================

#[test]
fn test_test_pattern_recognition_engine_default() {
    let engine = TestPatternRecognitionEngine::default();
    let _dbg = format!("{:?}", engine);
}

// =============================================================================
// ESTIMATION ALGORITHM TESTS
// =============================================================================

#[test]
fn test_conservative_estimation_algorithm_new() {
    let algo = ConservativeEstimationAlgorithm::new(0.2, true);
    assert_eq!(algo.name(), "ConservativeEstimation");
    assert!(algo.safety_margin > 0.0);
}

#[test]
fn test_conservative_estimation_parameters() {
    let algo = ConservativeEstimationAlgorithm::new(0.3, false);
    let params = algo.parameters();
    assert!(params.contains_key("safety_margin"));
}

#[test]
fn test_optimistic_estimation_algorithm_new() {
    let algo = OptimisticEstimationAlgorithm::new(true, 1.5);
    assert_eq!(algo.name(), "OptimisticEstimation");
    assert!(algo.optimism_factor > 1.0);
}

#[test]
fn test_optimistic_estimation_parameters() {
    let algo = OptimisticEstimationAlgorithm::new(false, 1.2);
    let params = algo.parameters();
    assert!(params.contains_key("optimism_factor"));
}

#[test]
fn test_ml_based_estimation_algorithm_new() {
    let algo = MLBasedEstimationAlgorithm::new("random_forest".to_string(), 0.85);
    assert_eq!(algo.name(), "MLBasedEstimation");
}

// =============================================================================
// CONFLICT DETECTION ALGORITHM TESTS
// =============================================================================

#[test]
fn test_static_conflict_detection_new() {
    let algo = StaticConflictDetectionAlgorithm::new(true, 3);
    assert_eq!(algo.name(), "StaticConflictDetection");
}

#[test]
fn test_static_conflict_detection_detect() {
    let algo = StaticConflictDetectionAlgorithm::new(true, 3);
    let result = algo.detect_conflicts(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_dynamic_conflict_detection_new() {
    let algo = DynamicConflictDetectionAlgorithm::new(true, 0.5);
    assert_eq!(algo.name(), "DynamicConflictDetection");
}

#[test]
fn test_dynamic_conflict_detection_detect() {
    let algo = DynamicConflictDetectionAlgorithm::new(true, 0.5);
    let result = algo.detect_conflicts(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_predictive_conflict_detection_new() {
    let algo = PredictiveConflictDetectionAlgorithm::new(10, 0.8);
    assert_eq!(algo.name(), "PredictiveConflictDetection");
}

#[test]
fn test_predictive_conflict_detection_detect() {
    let algo = PredictiveConflictDetectionAlgorithm::new(10, 0.8);
    let result = algo.detect_conflicts(&[]);
    assert!(result.is_ok());
}

#[test]
fn test_ml_conflict_detection_new() {
    let algo = MLConflictDetectionAlgorithm::new("svm".to_string(), 0.85);
    assert_eq!(algo.name(), "MLConflictDetection");
}

#[test]
fn test_ml_conflict_detection_detect() {
    let algo = MLConflictDetectionAlgorithm::new("svm".to_string(), 0.85);
    let result = algo.detect_conflicts(&[]);
    assert!(result.is_ok());
}

// =============================================================================
// PATTERN DETECTION ALGORITHM TESTS
// =============================================================================

#[test]
fn test_pipeline_detection_new() {
    let det = PipelineDetection::new();
    assert!(!det.detected);
    assert_eq!(det.stages, 0);
}

#[test]
fn test_pipeline_detection_default() {
    let det = PipelineDetection::default();
    assert!(!det.detected);
}

#[test]
fn test_pipeline_detection_detect_no_pattern() {
    let det = PipelineDetection::new();
    let result = det.detect();
    assert!(result.contains("No"));
}

#[test]
fn test_pipeline_detection_detect_with_pattern() {
    let det = PipelineDetection {
        detected: true,
        stages: 5,
        throughput: 100.0,
    };
    let result = det.detect();
    assert!(result.contains("Pipeline pattern detected"));
    assert!(result.contains("5"));
}

#[test]
fn test_pipeline_detection_name() {
    let det = PipelineDetection::new();
    assert_eq!(det.name(), "PipelineDetection");
}

#[test]
fn test_fork_join_detection_new() {
    let det = ForkJoinDetection::new();
    assert!(!det.detected);
    assert_eq!(det.fork_points, 0);
    assert_eq!(det.join_points, 0);
}

#[test]
fn test_fork_join_detection_default() {
    let det = ForkJoinDetection::default();
    assert!(!det.detected);
}

#[test]
fn test_fork_join_detection_detect_no_pattern() {
    let det = ForkJoinDetection::new();
    let result = det.detect();
    assert!(result.contains("No"));
}

#[test]
fn test_fork_join_detection_detect_with_pattern() {
    let det = ForkJoinDetection {
        detected: true,
        fork_points: 3,
        join_points: 2,
    };
    let result = det.detect();
    assert!(result.contains("Fork-Join pattern detected"));
    assert!(result.contains("3"));
}

#[test]
fn test_fork_join_detection_name() {
    let det = ForkJoinDetection::new();
    assert_eq!(det.name(), "ForkJoinDetection");
}

// =============================================================================
// CONTEXT FACTOR TYPE TESTS
// =============================================================================

#[test]
fn test_context_factor_type_construction() {
    let factor = ContextFactorType {
        factor_name: "cpu_load".to_string(),
        weight: 0.8,
    };
    assert!(!factor.factor_name.is_empty());
    assert!(factor.weight > 0.0 && factor.weight <= 1.0);
}

// =============================================================================
// ESTIMATION RESULT TESTS
// =============================================================================

#[test]
fn test_estimation_result_construction() {
    let result = EstimationResult {
        algorithm: "conservative".to_string(),
        concurrency: 8,
        confidence: 0.85,
        duration: std::time::Duration::from_millis(100),
    };
    assert!(result.concurrency > 0);
    assert!(result.confidence > 0.0 && result.confidence <= 1.0);
}

// =============================================================================
// SELECTION CONTEXT AND OUTCOME TESTS
// =============================================================================

#[test]
fn test_selection_context_default() {
    let context = SelectionContext::default();
    assert!(context.system_state.is_empty());
    assert!(context.resource_availability.is_empty());
}

#[test]
fn test_selection_context_construction() {
    let mut context = SelectionContext::default();
    context.risk_tolerance = 0.5;
    context.system_state.insert("cpu_load".to_string(), 0.7);
    assert!(context.risk_tolerance > 0.0);
    assert!(!context.system_state.is_empty());
}

#[test]
fn test_selection_outcome_construction() {
    let outcome = SelectionOutcome {
        success: true,
        selected_algorithm: "conservative".to_string(),
        performance_delta: 0.15,
        outcome_reason: "Best fit for workload".to_string(),
    };
    assert!(outcome.success);
    assert!(outcome.performance_delta > 0.0);
}

// =============================================================================
// RISK ASSESSMENT TESTS
// =============================================================================

#[test]
fn test_heuristic_risk_assessment_construction() {
    let assessment = HeuristicRiskAssessment {
        rules: Vec::new(),
        risk_level: "low".to_string(),
    };
    assert!(assessment.rules.is_empty());
}

#[test]
fn test_heuristic_risk_assessment_with_rules() {
    let assessment = HeuristicRiskAssessment::new(
        vec!["rule_1".to_string(), "rule_2".to_string()],
        "medium".to_string(),
    );
    assert_eq!(assessment.rules.len(), 2);
}

#[test]
fn test_probabilistic_risk_assessment_construction() {
    let assessment = ProbabilisticRiskAssessment {
        model: "bayesian".to_string(),
        probability: 0.15,
    };
    assert!(assessment.probability > 0.0);
}

#[test]
fn test_probabilistic_risk_assessment_new() {
    let assessment = ProbabilisticRiskAssessment::new("monte_carlo".to_string(), 0.25);
    assert_eq!(assessment.model, "monte_carlo");
}

#[test]
fn test_ml_risk_assessment_new() {
    let assessment = MachineLearningRiskAssessment::new("random_forest".to_string(), 0.9);
    let _dbg = format!("{:?}", assessment);
    assert!(assessment.confidence > 0.0);
}
