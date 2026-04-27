//! Tests for test grouping engine types

use super::types::*;
use std::collections::HashMap;
use std::time::Duration;

struct Lcg { state: u64 }
impl Lcg {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_f32(&mut self) -> f32 { (self.next_u64() % 10000) as f32 / 10000.0 }
}

#[test]
fn test_test_grouping_engine_new() {
    let engine = TestGroupingEngine::new();
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_groupings, 0);
}

#[test]
fn test_test_grouping_engine_with_config() {
    let config = GroupingEngineConfig::default();
    let engine = TestGroupingEngine::with_config(config);
    let metrics = engine.get_metrics();
    assert_eq!(metrics.total_groupings, 0);
}

#[test]
fn test_grouping_engine_config_default() {
    let config = GroupingEngineConfig::default();
    let _ = format!("{:?}", config);
}

#[test]
fn test_grouping_strategy_balanced() {
    let s = GroupingStrategyType::Balanced;
    let _ = format!("{:?}", s);
}

#[test]
fn test_grouping_strategy_resource_optimal() {
    let s = GroupingStrategyType::ResourceOptimal;
    let _ = format!("{:?}", s);
}

#[test]
fn test_grouping_strategy_time_optimal() {
    let s = GroupingStrategyType::TimeOptimal;
    let _ = format!("{:?}", s);
}

#[test]
fn test_duration_pattern_short_burst() {
    let p = DurationPattern::ShortBurst;
    let _ = format!("{:?}", p);
}

#[test]
fn test_duration_pattern_sustained() {
    let p = DurationPattern::Sustained;
    let _ = format!("{:?}", p);
}

#[test]
fn test_duration_pattern_periodic() {
    let p = DurationPattern::Periodic { interval: Duration::from_secs(10) };
    let _ = format!("{:?}", p);
}

#[test]
fn test_duration_pattern_custom() {
    let p = DurationPattern::Custom("my_pattern".to_string());
    let _ = format!("{:?}", p);
}

#[test]
fn test_optimization_target_minimize_time() {
    let t = OptimizationTarget::MinimizeExecutionTime;
    let _ = format!("{:?}", t);
}

#[test]
fn test_optimization_target_maximize_utilization() {
    let t = OptimizationTarget::MaximizeResourceUtilization;
    let _ = format!("{:?}", t);
}

#[test]
fn test_optimization_target_minimize_conflicts() {
    let t = OptimizationTarget::MinimizeConflicts;
    let _ = format!("{:?}", t);
}

#[test]
fn test_optimization_technique_greedy() {
    let t = OptimizationTechnique::GreedySearch;
    let _ = format!("{:?}", t);
}

#[test]
fn test_cooling_schedule_type_linear() {
    let t = CoolingScheduleType::Linear;
    let _ = format!("{:?}", t);
}

#[test]
fn test_cooling_schedule_type_exponential() {
    let t = CoolingScheduleType::Exponential;
    let _ = format!("{:?}", t);
}

#[test]
fn test_validation_severity_error() {
    let s = ValidationSeverity::Error;
    let _ = format!("{:?}", s);
}

#[test]
fn test_validation_severity_warning() {
    let s = ValidationSeverity::Warning;
    let _ = format!("{:?}", s);
}

#[test]
fn test_complexity_range_creation() {
    let range = ComplexityRange {
        min_complexity: 1.0,
        max_complexity: 10.0,
    };
    assert!(range.min_complexity <= range.max_complexity);
}

#[test]
fn test_grouping_outcome_creation() {
    let outcome = GroupingOutcome {
        description: "Improved test grouping".to_string(),
        expected_improvement: 0.25,
        confidence: 0.85,
        affected_metrics: vec!["execution_time".to_string()],
    };
    assert!(outcome.expected_improvement > 0.0);
    assert!(outcome.confidence > 0.0 && outcome.confidence <= 1.0);
}

#[test]
fn test_peak_timing_creation() {
    let timing = PeakTiming {
        relative_time: 0.5,
        peak_duration: Duration::from_secs(10),
        peak_intensity_multiplier: 2.0,
    };
    assert!(timing.relative_time >= 0.0 && timing.relative_time <= 1.0);
}

#[test]
fn test_temperature_schedule_creation() {
    let schedule = TemperatureSchedule {
        initial_temperature: 1.0,
        final_temperature: 0.01,
        cooling_rate: 0.95,
        schedule_type: CoolingScheduleType::Exponential,
    };
    assert!(schedule.initial_temperature > schedule.final_temperature);
}

#[test]
fn test_grouping_outcome_with_lcg() {
    let mut lcg = Lcg::new(42);
    let outcome = GroupingOutcome {
        description: "LCG-generated".to_string(),
        expected_improvement: lcg.next_f32().clamp(0.0, 1.0),
        confidence: lcg.next_f32().clamp(0.0, 1.0),
        affected_metrics: vec!["metric_a".to_string()],
    };
    assert!(outcome.expected_improvement >= 0.0);
}

#[test]
fn test_setup_operation_creation() {
    let op = SetupOperation {
        operation_id: "s1".to_string(),
        description: "Init DB".to_string(),
        operation_type: SetupOperationType::ResourceAllocation,
        expected_duration: Duration::from_secs(5),
        parameters: HashMap::new(),
    };
    assert!(!op.operation_id.is_empty());
}

#[test]
fn test_teardown_operation_creation() {
    let op = TeardownOperation {
        operation_id: "t1".to_string(),
        description: "Cleanup".to_string(),
        operation_type: TeardownOperationType::ResourceCleanup,
        expected_duration: Duration::from_secs(2),
        parameters: HashMap::new(),
    };
    assert!(!op.operation_id.is_empty());
}
