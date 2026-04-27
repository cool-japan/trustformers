//! Tests for conflict detector types

use super::types::*;
use std::collections::HashMap;
use std::time::Duration;

#[test]
fn test_conflict_detector_new() {
    let detector = ConflictDetector::new();
    let _ = format!("{:?}", detector);
}

#[test]
fn test_conflict_detector_with_default_config() {
    let config = ConflictDetectionConfig::default();
    let detector = ConflictDetector::with_config(config);
    let _ = format!("{:?}", detector);
}

#[test]
fn test_conflict_detection_config_default() {
    let config = ConflictDetectionConfig::default();
    assert!(config.confidence_threshold >= 0.0 && config.confidence_threshold <= 1.0);
}

#[test]
fn test_conflict_detection_statistics_default() {
    let stats = ConflictDetectionStatistics::default();
    let _ = format!("{:?}", stats);
}

#[test]
fn test_resource_conflict_thresholds_default() {
    let thresholds = ResourceConflictThresholds::default();
    let _ = format!("{:?}", thresholds);
}

#[test]
fn test_performance_impact_creation() {
    let impact = PerformanceImpact {
        cpu_degradation: 0.15,
        memory_degradation: 0.1,
        io_degradation: 0.05,
        network_degradation: 0.02,
        overall_degradation: 0.08,
    };
    assert!(impact.overall_degradation > 0.0);
    assert!(impact.cpu_degradation > impact.network_degradation);
}

#[test]
fn test_execution_time_impact_creation() {
    let impact = ExecutionTimeImpact {
        individual_test_time_increase: 0.15,
        total_suite_time_increase: 0.10,
        parallelization_efficiency_loss: 0.05,
    };
    assert!(impact.individual_test_time_increase > 0.0);
}

#[test]
fn test_resolution_cost_creation() {
    let cost = ResolutionCost {
        development_time: Duration::from_secs(3600),
        performance_cost: 0.02,
        resource_cost_multiplier: 1.1,
        maintenance_overhead: 0.05,
    };
    assert!(cost.performance_cost > 0.0);
    assert!(cost.resource_cost_multiplier >= 1.0);
}

#[test]
fn test_timing_pattern_creation() {
    let mut params = HashMap::new();
    params.insert("threshold".to_string(), 0.5);
    let pattern = TimingPattern {
        pattern_type: TimingPatternType::PeakUsage,
        parameters: params,
        strength: 0.85,
    };
    assert!(pattern.strength > 0.0);
    assert_eq!(pattern.parameters.len(), 1);
}

#[test]
fn test_conflict_condition_type_resource_usage() {
    let cond = ConflictConditionType::ResourceUsage("cpu".to_string());
    if let ConflictConditionType::ResourceUsage(resource) = &cond {
        assert_eq!(resource, "cpu");
    } else {
        panic!("expected ResourceUsage variant");
    }
}

#[test]
fn test_conflict_condition_type_custom() {
    let cond = ConflictConditionType::Custom("my_condition".to_string());
    let _ = format!("{:?}", cond);
}

#[test]
fn test_conflict_sensitivity_low() {
    let s = ConflictSensitivity::Conservative;
    let _ = format!("{:?}", s);
}

#[test]
fn test_conflict_sensitivity_high() {
    let s = ConflictSensitivity::Aggressive;
    let _ = format!("{:?}", s);
}

#[test]
fn test_resolution_complexity_trivial() {
    let c = ResolutionComplexity::Simple;
    let _ = format!("{:?}", c);
}

#[test]
fn test_conflict_detection_method_static() {
    let m = ConflictDetectionMethod::RuleBased;
    let _ = format!("{:?}", m);
}

#[test]
fn test_conflict_detection_method_pattern() {
    let m = ConflictDetectionMethod::PatternRecognition;
    let _ = format!("{:?}", m);
}

#[test]
fn test_conflict_resolution_type_isolation() {
    let t = ConflictResolutionType::Isolation;
    let _ = format!("{:?}", t);
}

#[test]
fn test_conflict_resolution_type_serialization() {
    let t = ConflictResolutionType::Sequential;
    let _ = format!("{:?}", t);
}

#[test]
fn test_network_conflict_type_port() {
    let t = NetworkConflictType::PortConflict;
    let _ = format!("{:?}", t);
}

#[test]
fn test_database_conflict_scope_table() {
    let s = DatabaseConflictScope::Table;
    let _ = format!("{:?}", s);
}

#[test]
fn test_conflict_detection_action_block() {
    let a = ConflictDetectionAction::Block;
    let _ = format!("{:?}", a);
}

#[test]
fn test_conflict_detection_action_warn() {
    let a = ConflictDetectionAction::Warn;
    let _ = format!("{:?}", a);
}

#[test]
fn test_side_effect_severity_negligible() {
    let s = SideEffectSeverity::Low;
    let _ = format!("{:?}", s);
}

#[test]
fn test_side_effect_severity_major() {
    let s = SideEffectSeverity::High;
    let _ = format!("{:?}", s);
}

#[test]
fn test_strategy_resource_requirements() {
    let reqs = StrategyResourceRequirements {
        cpu_overhead: 0.05,
        memory_overhead: 0.1,
        network_overhead: 0.02,
        time_overhead: Duration::from_secs(5),
        custom_overheads: HashMap::new(),
    };
    assert!(reqs.cpu_overhead > 0.0);
}

#[test]
fn test_conflict_impact_analysis_creation() {
    let analysis = ConflictImpactAnalysis {
        performance_impact: PerformanceImpact {
            cpu_degradation: 0.1,
            memory_degradation: 0.05,
            io_degradation: 0.02,
            network_degradation: 0.01,
            overall_degradation: 0.05,
        },
        reliability_impact: ReliabilityImpact { error_rate_increase: 0.0, timeout_probability_increase: 0.0, flakiness_increase: 0.0, reliability_decrease: 0.0 },
        efficiency_impact: EfficiencyImpact { utilization_efficiency_loss: 0.0, time_efficiency_loss: 0.0, cost_efficiency_impact: 0.0, overall_efficiency_loss: 0.0 },
        execution_time_impact: ExecutionTimeImpact {
            individual_test_time_increase: 0.1,
            total_suite_time_increase: 0.05,
            parallelization_efficiency_loss: 0.02,
        },
        overall_impact_score: 0.15,
    };
    assert!(analysis.overall_impact_score > 0.0);
}
