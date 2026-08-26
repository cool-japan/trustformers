//! Regression tests for the conflict-detection algorithms.
//!
//! Every test here asserts something that is *derived from the input*. Until
//! 0.2.1 all three algorithms returned `Ok(Vec::new())` no matter what they
//! were handed, so each of these assertions would have failed against the old
//! code — which is the point: the previous tests only checked `is_ok()`, which
//! the fabrication satisfied.

use super::*;
use crate::performance_optimizer::test_characterization::types::locking::ContentionEvent;
use crate::performance_optimizer::test_characterization::types::patterns::SharingStrategy;
use std::time::{Duration, Instant};

fn capability(
    read_sharing: bool,
    write_sharing: bool,
    max_readers: Option<usize>,
    max_writers: Option<usize>,
) -> ResourceSharingCapabilities {
    ResourceSharingCapabilities {
        supports_read_sharing: read_sharing,
        supports_write_sharing: write_sharing,
        max_concurrent_readers: max_readers,
        max_concurrent_writers: max_writers,
        sharing_overhead: 0.0,
        consistency_guarantees: Vec::new(),
        isolation_requirements: Vec::new(),
        recommended_strategy: SharingStrategy::NoSharing,
        safety_assessment: 1.0,
        performance_tradeoffs: HashMap::new(),
        performance_overhead: 0.0,
        implementation_complexity: 0.0,
        sharing_mode: "test".to_string(),
    }
}

fn access(
    resource_id: &str,
    access_type: AccessType,
    sharing_capability: ResourceSharingCapabilities,
) -> ResourceAccessPattern {
    ResourceAccessPattern {
        resource_id: resource_id.to_string(),
        access_type,
        access_frequency: 1.0,
        average_duration: Duration::from_millis(10),
        timing_pattern: Vec::new(),
        contention_events: Vec::new(),
        predictability_score: 0.9,
        sharing_capability,
        performance_impact: 0.4,
        optimization_potential: 0.0,
    }
}

// ---------------------------------------------------------------- static ----

#[test]
fn static_detection_reports_two_writers_on_a_non_sharing_resource() {
    let algorithm = StaticConflictDetectionAlgorithm::new(true, 0.8);
    let patterns = vec![
        access(
            "db",
            AccessType::WriteOnly,
            capability(false, false, None, None),
        ),
        access(
            "db",
            AccessType::WriteOnly,
            capability(false, false, None, None),
        ),
    ];
    let conflicts = algorithm.detect_conflicts(&patterns).expect("analysis runs");
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    assert_eq!(conflicts[0].resource_id, "db");
    assert!(conflicts[0].conflict_id.contains("writer-overcommit"));
    // One writer of two is over the single admitted writer.
    assert!(
        (conflicts[0].probability - 0.5).abs() < 1e-9,
        "{conflicts:?}"
    );
    assert_eq!(conflicts[0].max_safe_concurrency, 1);
}

#[test]
fn static_detection_stays_quiet_when_the_resource_admits_the_readers() {
    let algorithm = StaticConflictDetectionAlgorithm::new(true, 0.8);
    let patterns = vec![
        access(
            "cache",
            AccessType::ReadOnly,
            capability(true, false, Some(8), None),
        ),
        access(
            "cache",
            AccessType::ReadOnly,
            capability(true, false, Some(8), None),
        ),
    ];
    let conflicts = algorithm.detect_conflicts(&patterns).expect("analysis runs");
    assert!(conflicts.is_empty(), "{conflicts:?}");
}

#[test]
fn static_detection_treats_an_exclusive_demand_as_certain() {
    let algorithm = StaticConflictDetectionAlgorithm::new(true, 0.8);
    let patterns = vec![
        access(
            "gpu",
            AccessType::Exclusive,
            capability(true, true, None, None),
        ),
        access(
            "gpu",
            AccessType::ReadOnly,
            capability(true, true, None, None),
        ),
    ];
    let conflicts = algorithm.detect_conflicts(&patterns).expect("analysis runs");
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    assert!((conflicts[0].probability - 1.0).abs() < 1e-9);
    assert_eq!(conflicts[0].severity, ConflictSeverity::Severe);
    assert!(conflicts[0].conflict_id.contains("exclusive-demand"));
}

#[test]
fn static_detection_reports_being_switched_off_instead_of_reporting_no_conflicts() {
    let algorithm = StaticConflictDetectionAlgorithm::new(false, 0.8);
    let error = algorithm
        .detect_conflicts(&[])
        .expect_err("a disabled analysis must not answer 'no conflicts'");
    assert!(error.to_string().contains("disabled"), "{error}");
}

#[test]
fn static_detection_measures_degradation_from_the_recorded_impact() {
    let algorithm = StaticConflictDetectionAlgorithm::new(true, 1.0);
    let mut first = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    first.performance_impact = 0.2;
    let mut second = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    second.performance_impact = 0.8;
    let conflicts = algorithm.detect_conflicts(&[first, second]).expect("analysis runs");
    assert_eq!(conflicts.len(), 1);
    let degradation = conflicts[0].performance_impact.performance_degradation;
    assert!((degradation - 0.5).abs() < 1e-9, "{degradation}");
    // Nothing here measures reliability or stability, so they stay unmeasured.
    assert!(conflicts[0].performance_impact.reliability_impact.is_none());
    assert!(conflicts[0].performance_impact.stability_impact.is_none());
}

#[test]
fn static_detection_on_no_accesses_is_a_genuine_empty_answer() {
    let algorithm = StaticConflictDetectionAlgorithm::new(true, 0.8);
    assert!(algorithm.detect_conflicts(&[]).expect("analysis runs").is_empty());
}

// --------------------------------------------------------------- dynamic ----

fn contention(impact: f64, threads: usize) -> ContentionEvent {
    ContentionEvent {
        resource_id: "db".to_string(),
        competing_threads: (0..threads as u64).collect(),
        performance_impact: impact,
        severity: ContentionSeverity::Minimal,
        ..ContentionEvent::default()
    }
}

#[test]
fn dynamic_detection_corrects_the_observed_contention_for_the_sample_rate() {
    let algorithm = DynamicConflictDetectionAlgorithm::new(true, 0.5);
    let mut pattern = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    pattern.contention_events = vec![contention(0.5, 3)];
    let conflicts = algorithm.detect_conflicts(&[pattern]).expect("analysis runs");
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    // 1 - (1 - 0.5)^(1/0.5) = 1 - 0.25 = 0.75
    assert!(
        (conflicts[0].probability - 0.75).abs() < 1e-9,
        "{conflicts:?}"
    );
    // Only half the run was sampled, and the finding says so.
    assert!((conflicts[0].confidence - 0.5).abs() < 1e-9);
    assert_eq!(conflicts[0].historical_count, 1);
    assert_eq!(conflicts[0].max_safe_concurrency, 2);
}

#[test]
fn dynamic_detection_scales_with_how_much_contention_was_observed() {
    let algorithm = DynamicConflictDetectionAlgorithm::new(true, 1.0);
    let mut light = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    light.contention_events = vec![contention(0.2, 2)];
    let mut heavy = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    heavy.contention_events = vec![contention(0.2, 2), contention(0.2, 2)];
    let light_probability = algorithm.detect_conflicts(&[light]).expect("runs")[0].probability;
    let heavy_probability = algorithm.detect_conflicts(&[heavy]).expect("runs")[0].probability;
    assert!(
        heavy_probability > light_probability,
        "{heavy_probability} vs {light_probability}"
    );
}

#[test]
fn dynamic_detection_stays_quiet_when_no_contention_was_recorded() {
    let algorithm = DynamicConflictDetectionAlgorithm::new(true, 0.5);
    let pattern = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    assert!(algorithm.detect_conflicts(&[pattern]).expect("runs").is_empty());
}

#[test]
fn dynamic_detection_refuses_when_nothing_was_monitored() {
    let algorithm = DynamicConflictDetectionAlgorithm::new(false, 0.5);
    let error = algorithm.detect_conflicts(&[]).expect_err("no monitoring, no answer");
    assert!(error.to_string().contains("runtime monitoring"), "{error}");
}

#[test]
fn dynamic_detection_refuses_a_zero_sample_rate() {
    let algorithm = DynamicConflictDetectionAlgorithm::new(true, 0.0);
    let error = algorithm.detect_conflicts(&[]).expect_err("nothing can be sampled at 0.0");
    assert!(error.to_string().contains("sample rate"), "{error}");
}

// ------------------------------------------------------------ predictive ----

fn timed(
    resource_id: &str,
    access_type: AccessType,
    base: Instant,
    offsets_ms: &[u64],
) -> ResourceAccessPattern {
    let mut pattern = access(
        resource_id,
        access_type,
        capability(false, false, None, None),
    );
    pattern.timing_pattern = offsets_ms
        .iter()
        .map(|offset| {
            (
                base + Duration::from_millis(*offset),
                Duration::from_millis(10),
            )
        })
        .collect();
    pattern
}

#[test]
fn predictive_detection_refuses_when_no_cadence_was_recorded() {
    let algorithm = PredictiveConflictDetectionAlgorithm::new(5, 0.8);
    let pattern = access(
        "db",
        AccessType::WriteOnly,
        capability(false, false, None, None),
    );
    let error = algorithm
        .detect_conflicts(&[pattern])
        .expect_err("one untimed accessor projects nothing");
    assert!(error.to_string().contains("cadence"), "{error}");
}

#[test]
fn predictive_detection_reports_projected_overlap_between_writers() {
    let base = Instant::now();
    let algorithm = PredictiveConflictDetectionAlgorithm::new(3, 0.8);
    let patterns = vec![
        timed("db", AccessType::WriteOnly, base, &[0, 100]),
        timed("db", AccessType::WriteOnly, base, &[0, 100]),
    ];
    let conflicts = algorithm.detect_conflicts(&patterns).expect("analysis runs");
    assert_eq!(conflicts.len(), 1, "{conflicts:?}");
    assert_eq!(conflicts[0].conflict_type, ConflictType::Timing);
    assert!(conflicts[0].conflict_id.contains("projected-overlap"));
    // Three projected windows each, three of the nine pairings coincide.
    assert!(
        (conflicts[0].probability - (3.0 / 9.0)).abs() < 1e-9,
        "{conflicts:?}"
    );
    assert!((conflicts[0].confidence - 0.9).abs() < 1e-9);
}

#[test]
fn predictive_detection_stays_quiet_when_the_projections_miss_each_other() {
    let base = Instant::now();
    let algorithm = PredictiveConflictDetectionAlgorithm::new(3, 0.8);
    let patterns = vec![
        timed("db", AccessType::WriteOnly, base, &[0, 100]),
        timed("db", AccessType::WriteOnly, base, &[50, 150]),
    ];
    assert!(algorithm.detect_conflicts(&patterns).expect("runs").is_empty());
}

#[test]
fn predictive_detection_does_not_call_two_readers_a_conflict() {
    let base = Instant::now();
    let algorithm = PredictiveConflictDetectionAlgorithm::new(3, 0.8);
    let patterns = vec![
        timed("db", AccessType::ReadOnly, base, &[0, 100]),
        timed("db", AccessType::ReadOnly, base, &[0, 100]),
    ];
    assert!(algorithm.detect_conflicts(&patterns).expect("runs").is_empty());
}

#[test]
fn predictive_detection_honours_its_accuracy_threshold() {
    let base = Instant::now();
    let algorithm = PredictiveConflictDetectionAlgorithm::new(3, 0.95);
    let patterns = vec![
        timed("db", AccessType::WriteOnly, base, &[0, 100]),
        timed("db", AccessType::WriteOnly, base, &[0, 100]),
    ];
    // The accessors' recorded predictability is 0.9, below the 0.95 bar.
    assert!(algorithm.detect_conflicts(&patterns).expect("runs").is_empty());
}

#[test]
fn predictive_detection_refuses_a_zero_length_horizon() {
    let base = Instant::now();
    let algorithm = PredictiveConflictDetectionAlgorithm::new(0, 0.8);
    let patterns = vec![timed("db", AccessType::WriteOnly, base, &[0, 100])];
    let error = algorithm.detect_conflicts(&patterns).expect_err("nothing is projected");
    assert!(error.to_string().contains("horizon"), "{error}");
}

// ------------------------------------------------------------ parameters ----

#[test]
fn update_parameters_moves_the_reported_sensitivity() {
    let mut algorithm = StaticConflictDetectionAlgorithm::new(true, 0.2);
    assert!((algorithm.sensitivity() - 0.2).abs() < 1e-9);
    let mut params = HashMap::new();
    params.insert("sensitivity".to_string(), 0.9);
    algorithm.update_parameters(params).expect("parameter accepted");
    assert!((algorithm.sensitivity() - 0.9).abs() < 1e-9);
}
