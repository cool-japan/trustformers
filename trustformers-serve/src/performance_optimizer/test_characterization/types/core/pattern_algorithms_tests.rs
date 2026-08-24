//! Regression tests for the concurrency-pattern detection algorithms.
//!
//! Until 0.2.1 every one of these detectors returned the fixed string
//! `"No <X> pattern detected"` regardless of input — it was handed no input at
//! all. Each test below asserts a value that comes out of the interaction graph
//! it supplies, so all of them would fail against the old code.

use super::*;
use crate::performance_optimizer::test_characterization::types::data_management::types::DataExchangePattern;

fn interaction(
    source_thread: u64,
    target_thread: u64,
    interaction_type: InteractionType,
    frequency: f64,
) -> ThreadInteraction {
    ThreadInteraction {
        source_thread,
        target_thread,
        interaction_type,
        frequency,
        analysis_duration: std::time::Duration::from_millis(1),
        data_patterns: Vec::<DataExchangePattern>::new(),
        sync_requirements: Vec::new(),
        performance_impact: 0.0,
        optimization_opportunities: Vec::new(),
        safety_considerations: Vec::new(),
        from_thread: source_thread,
        to_thread: target_thread,
        timestamp: chrono::Utc::now(),
        resource: String::new(),
        strength: 1.0,
    }
}

fn execution(interactions: Vec<ThreadInteraction>) -> TestExecutionData {
    TestExecutionData {
        test_id: "pattern_test".to_string(),
        thread_interactions: interactions,
        ..TestExecutionData::default()
    }
}

fn data_flow(source: u64, target: u64) -> ThreadInteraction {
    interaction(source, target, InteractionType::DataExchange, 5.0)
}

// -------------------------------------------------------- producer/consumer --

#[test]
fn producer_consumer_counts_the_threads_on_each_side() {
    let data = execution(vec![data_flow(1, 3), data_flow(2, 3)]);
    let pattern = ProducerConsumerDetection::new()
        .detect(&data)
        .expect("analysis runs")
        .expect("two producers feeding one consumer is the shape");
    assert_eq!(pattern.pattern_type, "ProducerConsumer");
    assert_eq!(pattern.thread_count, 3);
    assert!((pattern.confidence - 1.0).abs() < 1e-9);
    assert!((pattern.applicability - 1.0).abs() < 1e-9);
    assert!(
        pattern.characteristics.iter().any(|c| c.contains("2 producer thread(s)")),
        "{:?}",
        pattern.characteristics
    );
}

#[test]
fn producer_consumer_declines_a_chain_that_passes_through_a_middle_thread() {
    let data = execution(vec![data_flow(1, 2), data_flow(2, 3)]);
    let found = ProducerConsumerDetection::new().detect(&data).expect("analysis runs");
    assert!(
        found.is_none(),
        "a three-stage chain is a pipeline, not a hand-off"
    );
}

#[test]
fn producer_consumer_ignores_pure_synchronization_edges() {
    let data = execution(vec![interaction(
        1,
        2,
        InteractionType::Synchronization,
        1.0,
    )]);
    assert!(ProducerConsumerDetection::new().detect(&data).expect("analysis runs").is_none());
}

#[test]
fn every_detector_refuses_a_test_with_no_recorded_interactions() {
    let data = execution(Vec::new());
    for error in [
        ProducerConsumerDetection::new().detect(&data).expect_err("no data"),
        MasterWorkerDetection::new().detect(&data).expect_err("no data"),
        PipelineDetection::new().detect(&data).expect_err("no data"),
        ForkJoinDetection::new().detect(&data).expect_err("no data"),
    ] {
        assert!(
            error.to_string().contains("no thread interactions"),
            "{error}"
        );
    }
}

// ------------------------------------------------------------ master/worker --

#[test]
fn master_worker_finds_the_star_centre() {
    let data = execution(vec![data_flow(1, 2), data_flow(1, 3), data_flow(1, 4)]);
    let pattern = MasterWorkerDetection::new()
        .detect(&data)
        .expect("analysis runs")
        .expect("a three-spoke star is the shape");
    assert_eq!(pattern.pattern_type, "MasterWorker");
    assert_eq!(pattern.thread_count, 4);
    assert!((pattern.confidence - 1.0).abs() < 1e-9);
    assert!(
        pattern.characteristics.iter().any(|c| c.contains("master thread 1")),
        "{:?}",
        pattern.characteristics
    );
}

#[test]
fn master_worker_confidence_drops_when_workers_talk_to_each_other() {
    let star = execution(vec![data_flow(1, 2), data_flow(1, 3), data_flow(1, 4)]);
    let crosstalk = execution(vec![
        data_flow(1, 2),
        data_flow(1, 3),
        data_flow(1, 4),
        data_flow(2, 3),
    ]);
    let clean = MasterWorkerDetection::new()
        .detect(&star)
        .expect("runs")
        .expect("star detected")
        .confidence;
    let noisy = MasterWorkerDetection::new()
        .detect(&crosstalk)
        .expect("runs")
        .expect("star still detected")
        .confidence;
    assert!(noisy < clean, "{noisy} should be below {clean}");
}

#[test]
fn master_worker_declines_a_single_hand_off() {
    let data = execution(vec![data_flow(1, 2)]);
    assert!(MasterWorkerDetection::new().detect(&data).expect("runs").is_none());
}

// ----------------------------------------------------------------- pipeline --

#[test]
fn pipeline_counts_stages_and_reports_the_bottleneck_rate() {
    let mut first = data_flow(1, 2);
    first.frequency = 9.0;
    let mut second = data_flow(2, 3);
    second.frequency = 2.5;
    let mut third = data_flow(3, 4);
    third.frequency = 7.0;
    let data = execution(vec![first, second, third]);
    let pattern = PipelineDetection::new()
        .detect(&data)
        .expect("analysis runs")
        .expect("a four-stage chain is the shape");
    assert_eq!(pattern.pattern_type, "Pipeline");
    assert_eq!(pattern.thread_count, 4);
    assert!(
        pattern.characteristics.iter().any(|c| c.contains("2.500")),
        "the slowest stage sets the rate: {:?}",
        pattern.characteristics
    );
}

#[test]
fn pipeline_declines_a_two_thread_hand_off() {
    let data = execution(vec![data_flow(1, 2)]);
    assert!(PipelineDetection::new().detect(&data).expect("runs").is_none());
}

// ---------------------------------------------------------------- fork/join --

#[test]
fn fork_join_finds_the_scatter_and_the_gather() {
    let data = execution(vec![
        data_flow(1, 2),
        data_flow(1, 3),
        data_flow(2, 4),
        data_flow(3, 4),
    ]);
    let pattern = ForkJoinDetection::new()
        .detect(&data)
        .expect("analysis runs")
        .expect("fork at 1, join at 4");
    assert_eq!(pattern.pattern_type, "ForkJoin");
    assert_eq!(pattern.thread_count, 4);
    assert!(
        pattern.characteristics.iter().any(|c| c.contains("1 fork point(s)")),
        "{:?}",
        pattern.characteristics
    );
    assert!(
        pattern.characteristics.iter().any(|c| c.contains("1 join point(s)")),
        "{:?}",
        pattern.characteristics
    );
}

#[test]
fn fork_join_declines_a_fan_out_that_never_rejoins() {
    let data = execution(vec![data_flow(1, 2), data_flow(1, 3)]);
    assert!(ForkJoinDetection::new().detect(&data).expect("runs").is_none());
}

#[test]
fn applicability_is_the_share_of_interacting_threads_the_pattern_covers() {
    // Threads 1-4 form the fork/join; threads 7 and 8 interact separately.
    let data = execution(vec![
        data_flow(1, 2),
        data_flow(1, 3),
        data_flow(2, 4),
        data_flow(3, 4),
        data_flow(7, 8),
    ]);
    let pattern = ForkJoinDetection::new()
        .detect(&data)
        .expect("runs")
        .expect("fork/join still present");
    assert_eq!(pattern.thread_count, 4);
    assert!(
        (pattern.applicability - (4.0 / 6.0)).abs() < 1e-9,
        "{}",
        pattern.applicability
    );
}
