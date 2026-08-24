//! Unit tests for `types::optimization`.
//!
//! Split out of `optimization.rs` in 0.2.1 to keep that file under the
//! 2000-line limit; the tests are unchanged apart from the module move.

use super::optimization::*;
use super::performance::PerformanceMetrics;

use super::*;

struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
    fn next_usize(&mut self, bound: usize) -> usize {
        (self.next_u64() as usize) % bound.max(1)
    }
}

// ---- ConstraintType tests ----
#[test]
fn test_constraint_type_all_variants() {
    let variants = [
        ConstraintType::Order,
        ConstraintType::MutualExclusion,
        ConstraintType::Dependency,
        ConstraintType::Resource,
        ConstraintType::Timing,
    ];
    assert_eq!(variants.len(), 5);
}

#[test]
fn test_constraint_type_equality() {
    assert_eq!(ConstraintType::Order, ConstraintType::Order);
    assert_ne!(ConstraintType::Order, ConstraintType::Timing);
}

#[test]
fn test_constraint_type_hash() {
    use std::collections::HashSet;
    let mut set = HashSet::new();
    set.insert(ConstraintType::Order);
    set.insert(ConstraintType::Order);
    set.insert(ConstraintType::Resource);
    assert_eq!(set.len(), 2);
}

// ---- OptimizationEffort tests ----
#[test]
fn test_optimization_effort_all_variants() {
    let efforts = [
        OptimizationEffort::Minimal,
        OptimizationEffort::Low,
        OptimizationEffort::Medium,
        OptimizationEffort::High,
        OptimizationEffort::Maximum,
    ];
    assert_eq!(efforts.len(), 5);
}

#[test]
fn test_optimization_effort_equality() {
    assert_eq!(OptimizationEffort::Medium, OptimizationEffort::Medium);
}

// ---- OptimizationType tests ----
#[test]
fn test_optimization_type_all_variants() {
    let types = [
        OptimizationType::Parallelism,
        OptimizationType::Batching,
        OptimizationType::Caching,
        OptimizationType::ResourcePooling,
        OptimizationType::LoadBalancing,
        OptimizationType::ReduceOverhead,
        OptimizationType::SimplifyImplementation,
    ];
    assert_eq!(types.len(), 7);
}

// ---- AdaptiveOptimizerConfig default ----
#[test]
fn test_adaptive_optimizer_config_default() {
    let c = AdaptiveOptimizerConfig::default();
    assert!((c.adaptation_rate - 0.1).abs() < f64::EPSILON);
    assert_eq!(c.optimization_effort, OptimizationEffort::Medium);
    assert_eq!(c.max_iterations, 100);
    assert!((c.convergence_threshold - 0.001).abs() < f64::EPSILON);
    assert_eq!(c.optimization_interval, std::time::Duration::from_secs(10));
}

#[test]
fn test_adaptive_optimizer_config_default_empty_strings() {
    let c = AdaptiveOptimizerConfig::default();
    assert!(c.tracking_config.is_empty());
    assert!(c.analysis_config.is_empty());
}

// ---- AdaptiveMitigation ----
#[test]
fn test_adaptive_mitigation_new() {
    let m = AdaptiveMitigation::new();
    assert!(m.enabled);
    assert!((m.learning_rate - 0.1).abs() < f64::EPSILON);
}

#[test]
fn test_adaptive_mitigation_default() {
    let m = AdaptiveMitigation::default();
    assert!(m.enabled);
}

// ---- FlowControlManager ----
#[test]
fn test_flow_control_manager_new() {
    let f = FlowControlManager::new();
    assert!(f.control_enabled);
    assert!((f.flow_rate_limit - 100.0).abs() < f64::EPSILON);
    assert!(f.backpressure_enabled);
}

#[test]
fn test_flow_control_manager_default() {
    let f = FlowControlManager::default();
    assert!(f.control_enabled);
}

// ---- OptimizationPerformanceTracker ----
#[test]
fn test_optimization_performance_tracker_records_only_while_tracking() {
    let t = OptimizationPerformanceTracker::new();
    assert!(t.tracking_enabled);
    assert!(!t.is_tracking());
    // Nothing has been recorded, so there is no "current performance".
    assert!(t.get_current_performance().is_err());

    // A sample offered while stopped is ignored rather than back-dated in.
    t.record(PerformanceMetrics::default());
    assert_eq!(t.recorded_sample_count(), 0);

    t.start_tracking().expect("tracking is enabled");
    assert!(t.is_tracking());
    let mut sample = PerformanceMetrics::default();
    sample.cpu_usage_percent = 42.0;
    t.record(sample);
    assert_eq!(t.recorded_sample_count(), 1);
    let current = t.get_current_performance().expect("one sample recorded");
    assert!((current.cpu_usage_percent - 42.0).abs() < f64::EPSILON);

    t.stop_tracking().expect("stop always succeeds");
    assert!(!t.is_tracking());
}

#[test]
fn test_optimization_performance_tracker_default() {
    let t = OptimizationPerformanceTracker::default();
    assert!(t.tracking_enabled);
}

// ---- StrategyEffectivenessAnalyzer ----
#[test]
fn test_strategy_effectiveness_analyzer_records_only_while_analyzing() {
    let a = StrategyEffectivenessAnalyzer::new();
    assert!(a.analysis_enabled);
    assert!(!a.is_analyzing());
    assert!(a
        .analyze_current_effectiveness()
        .expect("an empty analyzer is a valid state")
        .is_empty());

    a.record_effectiveness("sampling", 0.9);
    assert!(
        a.analyze_current_effectiveness().expect("still empty").is_empty(),
        "a score offered while stopped must not be recorded"
    );

    a.start_analysis().expect("analysis is enabled");
    assert!(a.is_analyzing());
    a.record_effectiveness("sampling", 0.6);
    a.record_effectiveness("sampling", 0.8);
    let scores = a.analyze_current_effectiveness().expect("two scores recorded");
    let mean = scores.get("sampling").copied().expect("sampling was recorded");
    assert!((mean - 0.7).abs() < 1e-9, "mean of 0.6 and 0.8, got {mean}");

    a.stop_analysis().expect("stop always succeeds");
    assert!(!a.is_analyzing());
}

#[test]
fn test_strategy_effectiveness_analyzer_default() {
    let a = StrategyEffectivenessAnalyzer::default();
    assert!(a.analysis_enabled);
}

// ---- AdaptiveSamplingStrategy ----
#[test]
fn test_adaptive_sampling_strategy_new() {
    let s = AdaptiveSamplingStrategy::new();
    assert!((s.min_rate_hz - 1.0).abs() < f64::EPSILON);
    assert!((s.max_rate_hz - 1000.0).abs() < f64::EPSILON);
}

#[test]
fn test_adaptive_sampling_strategy_default() {
    let s = AdaptiveSamplingStrategy::default();
    assert!((s.adaptation_factor - 1.5).abs() < f64::EPSILON);
}

// ---- AdaptiveThresholdManager ----
#[test]
fn test_adaptive_threshold_manager_new() {
    let m = AdaptiveThresholdManager::new();
    assert!(m.adaptation_enabled);
    assert!(m.thresholds.is_empty());
}

#[test]
fn test_adaptive_threshold_manager_default() {
    let m = AdaptiveThresholdManager::default();
    assert!(m.adaptation_enabled);
}

// ---- StrategyPerformanceTracker ----
#[test]
fn test_strategy_performance_tracker_reports_the_best_observed_strategy() {
    let t = StrategyPerformanceTracker::new();
    assert!(!t.is_tracking());
    assert!(t.current_best_strategy().is_none(), "nothing observed yet");

    t.record("adaptive", 0.9);
    assert!(
        t.get_current_performance().expect("empty is valid").is_empty(),
        "a sample offered while stopped must not be recorded"
    );

    t.start_tracking().expect("start always succeeds");
    t.record("fixed", 0.2);
    t.record("adaptive", 0.8);
    t.record("adaptive", 0.6);
    let summary = t.get_current_performance().expect("three samples recorded");
    assert!((summary.get("fixed").copied().unwrap_or_default() - 0.2).abs() < 1e-9);
    assert!((summary.get("adaptive").copied().unwrap_or_default() - 0.7).abs() < 1e-9);
    assert_eq!(t.current_best_strategy().as_deref(), Some("adaptive"));
}

#[test]
fn test_strategy_performance_tracker_default() {
    let t = StrategyPerformanceTracker::default();
    assert!(t.get_current_performance().expect("empty is valid").is_empty());
}

// ---- CostBenefitAnalysis ----
#[test]
fn test_cost_benefit_analysis_construction() {
    let cba = CostBenefitAnalysis {
        total_cost: 100.0,
        total_benefit: 300.0,
        net_benefit: 200.0,
        benefit_cost_ratio: 3.0,
        payback_period: std::time::Duration::from_secs(60),
    };
    assert!((cba.benefit_cost_ratio - 3.0).abs() < f64::EPSILON);
    assert!((cba.net_benefit - 200.0).abs() < f64::EPSILON);
}

// ---- CostTargets ----
#[test]
fn test_cost_targets_construction() {
    let targets = CostTargets {
        target_cost: 50.0,
        max_acceptable_cost: 100.0,
        cost_reduction_goal: 30.0,
        target_timeframe: std::time::Duration::from_secs(3600),
    };
    assert!((targets.target_cost - 50.0).abs() < f64::EPSILON);
}

// ---- BackoffStrategy ----
#[test]
fn test_backoff_strategy_construction() {
    let bs = BackoffStrategy {
        initial_delay: std::time::Duration::from_millis(100),
        max_delay: std::time::Duration::from_secs(60),
        backoff_factor: 2.0,
        strategy_type: "exponential".to_string(),
    };
    assert!((bs.backoff_factor - 2.0).abs() < f64::EPSILON);
    assert_eq!(bs.strategy_type, "exponential");
}

// ---- BackpressureController ----
#[test]
fn test_backpressure_controller_construction() {
    let bp = BackpressureController {
        enabled: true,
        pressure_threshold: 0.8,
        control_actions: vec!["throttle".to_string()],
        current_pressure: 0.5,
    };
    assert!(bp.enabled);
    assert_eq!(bp.control_actions.len(), 1);
}

// ---- FlowController ----
#[test]
fn test_flow_controller_construction() {
    let fc = FlowController {
        max_flow_rate: 1000.0,
        current_flow_rate: 500.0,
        throttle_enabled: true,
        burst_capacity: 50,
    };
    assert!(fc.throttle_enabled);
    assert_eq!(fc.burst_capacity, 50);
}

// ---- LCG-driven tests ----
#[test]
fn test_lcg_selects_optimization_types() {
    let mut rng = Lcg::new(42);
    let types = [
        OptimizationType::Parallelism,
        OptimizationType::Batching,
        OptimizationType::Caching,
        OptimizationType::ResourcePooling,
    ];
    for _ in 0..20 {
        let idx = rng.next_usize(types.len());
        let formatted = format!("{:?}", types[idx]);
        assert!(!formatted.is_empty());
    }
}

#[test]
fn test_lcg_generates_cost_values() {
    let mut rng = Lcg::new(7777);
    for _ in 0..50 {
        let cost = rng.next_f64() * 1000.0;
        assert!((0.0..1000.0).contains(&cost));
    }
}
