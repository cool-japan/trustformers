//! Tests for the advanced distributed features (auto-scaling, smart
//! checkpointing and the ML-driven performance optimizer).

use super::*;

fn scratch_dir(label: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "trustformers-ckpt-{label}-{}-{:?}",
        std::process::id(),
        std::thread::current().id()
    ));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn model_state(scale: f32) -> HashMap<String, Tensor> {
    let mut state = HashMap::new();
    state.insert(
        "encoder.weight".to_string(),
        Tensor::from_slice(
            &[0.11 * scale, -0.25 * scale, 0.5 * scale, -0.75 * scale],
            &[2, 2],
        )
        .expect("tensor must build in test"),
    );
    state.insert(
        "encoder.bias".to_string(),
        Tensor::from_slice(&[-0.03 * scale, 0.07 * scale], &[2])
            .expect("tensor must build in test"),
    );
    state
}

fn assert_states_equal(actual: &HashMap<String, Tensor>, expected: &HashMap<String, Tensor>) {
    assert_eq!(actual.len(), expected.len(), "parameter count");
    for (name, tensor) in expected {
        let restored = actual.get(name).unwrap_or_else(|| panic!("parameter `{name}` was lost"));
        assert_eq!(restored.shape(), tensor.shape(), "`{name}` shape");
        assert_eq!(
            restored.to_vec_f32().expect("read"),
            tensor.to_vec_f32().expect("read"),
            "`{name}` values must be bit-identical"
        );
    }
}

#[test]
fn checkpoint_save_load_round_trip_is_bit_identical() {
    let dir = scratch_dir("full");
    let config = CheckpointConfig {
        differential: false,
        compression: false,
        ..CheckpointConfig::default()
    };
    let mut manager =
        SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

    let state = model_state(1.0);
    let info = manager.create_checkpoint(10, &state).expect("checkpoint in test");
    assert!(info.validation_passed, "a real checkpoint must validate");
    assert!(!info.is_differential);

    let restored = manager.load_checkpoint(10).expect("load in test");
    assert_states_equal(&restored, &state);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn checkpoint_does_not_zero_sub_unit_weights() {
    // Regression: the previous serializer wrote `tensor.to_vec_u8()`,
    // which casts f32 values to u8 and mapped every weight in (-1, 1) to 0.
    let dir = scratch_dir("subunit");
    let config = CheckpointConfig {
        differential: false,
        compression: true,
        ..CheckpointConfig::default()
    };
    let mut manager =
        SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

    let mut state = HashMap::new();
    state.insert(
        "w".to_string(),
        Tensor::from_slice(&[0.004, -0.9, 0.5, 0.125], &[4]).expect("tensor must build in test"),
    );

    manager.create_checkpoint(1, &state).expect("checkpoint in test");
    let restored = manager.load_checkpoint(1).expect("load in test");
    let values = restored.get("w").expect("parameter must survive").to_vec_f32().expect("read");

    assert_eq!(values, vec![0.004f32, -0.9, 0.5, 0.125]);
    assert!(values.iter().all(|value| *value != 0.0));

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn differential_checkpoints_replay_onto_the_base() {
    let dir = scratch_dir("diff");
    let config = CheckpointConfig {
        differential: true,
        compression: true,
        retention_count: 10,
        ..CheckpointConfig::default()
    };
    let mut manager =
        SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

    let first = model_state(1.0);
    let second = model_state(2.0);
    let third = model_state(3.0);

    let full = manager.create_checkpoint(1, &first).expect("checkpoint in test");
    assert!(!full.is_differential);
    let diff_one = manager.create_checkpoint(2, &second).expect("checkpoint in test");
    assert!(diff_one.is_differential);
    let diff_two = manager.create_checkpoint(3, &third).expect("checkpoint in test");
    assert!(diff_two.is_differential);
    assert!(diff_two.validation_passed);

    assert_states_equal(&manager.load_checkpoint(1).expect("load in test"), &first);
    assert_states_equal(&manager.load_checkpoint(2).expect("load in test"), &second);
    assert_states_equal(&manager.load_checkpoint(3).expect("load in test"), &third);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn differential_checkpoint_is_smaller_than_a_full_one() {
    let dir = scratch_dir("size");
    let config = CheckpointConfig {
        differential: true,
        compression: false,
        retention_count: 10,
        ..CheckpointConfig::default()
    };
    let mut manager =
        SmartCheckpointManager::new(config, dir.clone()).expect("manager must build in test");

    let mut state = HashMap::new();
    state.insert(
        "w".to_string(),
        Tensor::from_slice(&[0.5f32; 256], &[256]).expect("tensor must build in test"),
    );
    let full = manager.create_checkpoint(1, &state).expect("checkpoint in test");

    // Change a single element.
    let mut changed: Vec<f32> = vec![0.5f32; 256];
    changed[7] = -1.25;
    state.insert(
        "w".to_string(),
        Tensor::from_slice(&changed, &[256]).expect("tensor must build in test"),
    );
    let diff = manager.create_checkpoint(2, &state).expect("checkpoint in test");

    assert!(
        diff.file_size < full.file_size,
        "a one-element delta must be smaller than the full state ({} vs {})",
        diff.file_size,
        full.file_size
    );
    assert_states_equal(&manager.load_checkpoint(2).expect("load in test"), &state);

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn loading_an_unknown_step_errors() {
    let dir = scratch_dir("missing");
    let manager = SmartCheckpointManager::new(CheckpointConfig::default(), dir.clone())
        .expect("manager must build in test");
    assert!(manager.load_checkpoint(42).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_auto_scaler_config() {
    let config = AutoScalerConfig {
        min_nodes: 2,
        max_nodes: 32,
        ..AutoScalerConfig::default()
    };

    // Validate the modified configuration
    assert_eq!(config.min_nodes, 2);
    assert_eq!(config.max_nodes, 32);
}

#[test]
fn test_auto_scaler_creation() {
    let config = AutoScalerConfig::default();
    let auto_scaler = AutoScaler::new(config)
        .with_min_nodes(2)
        .with_max_nodes(16)
        .with_scaling_strategy(ScalingStrategy::Performance);

    assert_eq!(auto_scaler.get_current_nodes(), 2);
    assert!(matches!(
        auto_scaler.config.strategy,
        ScalingStrategy::Performance
    ));
}

#[test]
fn test_workload_predictor() {
    let mut predictor = WorkloadPredictor::new();

    // Add some test data
    let metrics = PerformanceMetrics {
        throughput: 1000.0,
        gpu_utilization: vec![0.8, 0.7, 0.9],
        memory_usage: vec![0.6, 0.7, 0.5],
        communication_overhead: 0.2,
        compression_ratio: 0.1,
        bandwidth_utilization: 0.8,
        step_time: Duration::from_millis(100),
    };

    // Too little history: the predictor must say so rather than return an
    // invented "conservative 0.75".
    predictor.update_metrics(&metrics);
    assert!(!predictor.can_predict());
    assert!(predictor.predict_workload(Duration::from_secs(600)).is_err());

    for _ in 0..WorkloadPredictor::MIN_SAMPLES {
        predictor.update_metrics(&metrics);
    }
    assert!(predictor.can_predict());

    let prediction = predictor
        .predict_workload(Duration::from_secs(600))
        .expect("Operation failed in test");
    assert!((0.0..=1.0).contains(&prediction));
    // Every sample was 0.8; a prediction anywhere near 0.75 would mean the
    // hard-coded default is still in play.
    assert!(
        (prediction - 0.8).abs() < 0.05,
        "prediction {prediction} does not follow the observed utilization"
    );
}

#[test]
fn seasonal_analyzer_buckets_by_wall_clock_hour_and_refuses_to_guess() {
    let mut analyzer = SeasonalAnalyzer::new();
    assert!(
        analyzer.predict(Duration::from_secs(0)).is_err(),
        "an empty analyzer has nothing to predict from"
    );

    let now = SystemTime::now();
    let hour = SeasonalAnalyzer::hour_of_day(now);
    let other = SeasonalAnalyzer::hour_of_day(now + Duration::from_secs(3 * 3600));
    assert_ne!(
        hour, other,
        "three hours apart must land in different buckets"
    );

    analyzer.update(now, 0.2);
    analyzer.update(now + Duration::from_secs(3 * 3600), 0.9);

    // Predicting for "now" reads this hour's bucket, not a global average.
    // `predict_at` pins the instant so the bucket choice cannot race an
    // hour boundary.
    let immediate = analyzer.predict_at(now).expect("prediction must succeed in test");
    assert!((immediate - 0.2).abs() < 1e-5, "got {immediate}");

    let later = analyzer
        .predict_at(now + Duration::from_secs(3 * 3600))
        .expect("prediction must succeed in test");
    assert!((later - 0.9).abs() < 1e-5, "got {later}");

    // An hour with no samples falls back to the mean over all buckets,
    // which is still measured data.
    let unseen = analyzer
        .predict_at(now + Duration::from_secs(7 * 3600))
        .expect("prediction must succeed in test");
    assert!((unseen - 0.55).abs() < 1e-5, "got {unseen}");
}

#[test]
fn custom_scaling_strategy_reports_that_it_is_unimplemented() {
    let mut scaler = AutoScaler::new(AutoScalerConfig {
        strategy: ScalingStrategy::Custom("my-policy".to_string()),
        scaling_cooldown: Duration::from_secs(0),
        ..Default::default()
    });

    let sample = PerformanceMetrics {
        throughput: 100.0,
        gpu_utilization: vec![0.9],
        memory_usage: vec![0.5],
        communication_overhead: 0.2,
        compression_ratio: 1.0,
        bandwidth_utilization: 1000.0,
        step_time: Duration::from_millis(50),
    };

    let Err(error) = scaler.update_and_scale(&sample) else {
        panic!("an unimplemented custom strategy must not answer NoAction in test");
    };
    assert!(error.to_string().contains("my-policy"), "{error}");
}

#[test]
fn test_checkpoint_manager() {
    let config = CheckpointConfig::default();
    let temp_dir = std::env::temp_dir().join("test_checkpoints");

    if temp_dir.exists() {
        std::fs::remove_dir_all(&temp_dir).ok();
    }

    let manager = SmartCheckpointManager::new(config, temp_dir).expect("Construction failed");

    let metrics = PerformanceMetrics {
        throughput: 1000.0,
        gpu_utilization: vec![0.8],
        memory_usage: vec![0.6],
        communication_overhead: 0.2,
        compression_ratio: 0.1,
        bandwidth_utilization: 0.8,
        step_time: Duration::from_millis(100),
    };

    assert!(manager.should_checkpoint(1000, &metrics));
    assert!(!manager.should_checkpoint(999, &metrics));
}

#[test]
fn test_ml_optimizer() {
    let config = MLOptimizerConfig::default();
    let optimizer = PerformanceMLOptimizer::new(config)
        .with_prediction_horizon(50)
        .with_optimization_frequency(25);

    assert_eq!(optimizer.config.prediction_horizon, 50);
    assert_eq!(optimizer.config.optimization_frequency, 25);

    assert!(optimizer.should_optimize(25));
    assert!(!optimizer.should_optimize(24));
}

#[test]
fn test_trend_analyzer() {
    let mut analyzer = TrendAnalyzer::new();

    // Add increasing trend
    for i in 0..20 {
        analyzer.update(i as f32 * 0.1);
    }

    let prediction = analyzer.predict(Duration::from_secs(60)).expect("Operation failed in test");
    assert!(prediction > 1.0); // Should predict increasing trend
}

// ── ML optimizer: predictions must be derived, never constants ────────
//
// The previous implementation returned `performance_improvement: 0.15` for
// every compression change and `0.08` for every "communication" change —
// the latter without touching the configuration at all. These tests fail
// against that code because they vary the measured metrics and require the
// reported prediction to move with them.

fn metrics(communication_overhead: f32, bandwidth_mbps: f32) -> PerformanceMetrics {
    PerformanceMetrics {
        throughput: 100.0,
        gpu_utilization: vec![0.75],
        memory_usage: vec![0.6],
        communication_overhead,
        compression_ratio: 1.0,
        bandwidth_utilization: bandwidth_mbps,
        step_time: Duration::from_millis(100),
    }
}

#[test]
fn compression_prediction_tracks_the_measured_communication_overhead() {
    let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

    let mut light = DistributedConfig::new();
    light.compression.target_ratio = 0.5;
    let light_result = optimizer
        .optimize_compression(&metrics(0.4, 1000.0), &mut light)
        .expect("optimization must succeed in test")
        .expect("high communication overhead must trigger a change in test");

    let mut heavy = DistributedConfig::new();
    heavy.compression.target_ratio = 0.5;
    let heavy_result = optimizer
        .optimize_compression(&metrics(0.8, 1000.0), &mut heavy)
        .expect("optimization must succeed in test")
        .expect("high communication overhead must trigger a change in test");

    assert!(
        heavy_result.performance_improvement > light_result.performance_improvement,
        "a heavier communication phase must predict a larger saving: {} vs {}",
        heavy_result.performance_improvement,
        light_result.performance_improvement
    );

    // 20% fewer bytes out of a phase that is 80% of the step.
    assert!((heavy_result.performance_improvement - 0.8 * 0.2).abs() < 1e-5);

    // And the configuration really changed.
    assert!((heavy.compression.target_ratio - 0.4).abs() < 1e-6);
}

#[test]
fn compression_at_the_floor_reports_no_optimization() {
    let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());
    let mut config = DistributedConfig::new();
    config.compression.target_ratio = 0.05;

    assert!(optimizer
        .optimize_compression(&metrics(0.9, 1000.0), &mut config)
        .expect("optimization must succeed in test")
        .is_none());
    assert!((config.compression.target_ratio - 0.05).abs() < 1e-6);
}

#[test]
fn communication_optimization_applies_a_real_change_or_reports_none() {
    let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

    // Fast link: nothing to do.
    let mut fast = DistributedConfig::new();
    assert!(optimizer
        .optimize_communication(&metrics(0.5, 10_000.0), &mut fast)
        .expect("optimization must succeed in test")
        .is_none());
    assert!(!fast.compression.enabled);

    // Slow link with compression off: the knob is really turned.
    let mut slow = DistributedConfig::new();
    slow.compression.enabled = false;
    slow.compression.target_ratio = 0.25;
    let result = optimizer
        .optimize_communication(&metrics(0.5, 10.0), &mut slow)
        .expect("optimization must succeed in test")
        .expect("a slow interconnect must enable compression in test");
    assert!(slow.compression.enabled, "the config must actually change");
    assert!((result.performance_improvement - 0.5 * 0.75).abs() < 1e-5);

    // Already enabled: no further knob exists, so nothing is claimed.
    assert!(optimizer
        .optimize_communication(&metrics(0.5, 10.0), &mut slow)
        .expect("optimization must succeed in test")
        .is_none());
}

#[test]
fn shrinking_the_batch_never_reports_a_speedup() {
    let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

    // High utilization and high memory pressure => the model shrinks the
    // batch. The old code reported the negative size delta as an
    // "improvement".
    let mut config = DistributedConfig::new();
    config.dynamic_batching.initial_batch_size = 128;
    let pressured = PerformanceMetrics {
        gpu_utilization: vec![0.95],
        memory_usage: vec![0.95],
        ..metrics(0.5, 1000.0)
    };

    let result = optimizer
        .optimize_batch_sizes(&pressured, &mut config)
        .expect("optimization must succeed in test")
        .expect("a >10% size change must be reported in test");

    assert!(config.dynamic_batching.initial_batch_size < 128);
    assert_eq!(
        result.performance_improvement, 0.0,
        "a batch reduction taken for memory headroom predicts no speedup"
    );
}

#[test]
fn growing_the_batch_predicts_amortised_communication() {
    let optimizer = PerformanceMLOptimizer::new(MLOptimizerConfig::default());

    let mut config = DistributedConfig::new();
    config.dynamic_batching.initial_batch_size = 8;
    let idle = PerformanceMetrics {
        gpu_utilization: vec![0.5],
        memory_usage: vec![0.3],
        ..metrics(0.5, 1000.0)
    };

    let result = optimizer
        .optimize_batch_sizes(&idle, &mut config)
        .expect("optimization must succeed in test")
        .expect("an idle GPU must grow the batch in test");

    let new_batch = config.dynamic_batching.initial_batch_size as f32;
    assert!(new_batch > 8.0);
    let expected = 0.5 * (1.0 - 8.0 / new_batch);
    assert!(
        (result.performance_improvement - expected).abs() < 1e-4,
        "predicted {} but the derivation gives {expected}",
        result.performance_improvement
    );
}

// ---- Honest-contract tests for AutoScaler::execute_scale_up/execute_scale_down ----

fn scaling_metrics(avg_gpu_utilization: f32, node_count: usize) -> PerformanceMetrics {
    let samples = node_count.max(1);
    PerformanceMetrics {
        throughput: 100.0,
        gpu_utilization: vec![avg_gpu_utilization; samples],
        memory_usage: vec![0.5; samples],
        communication_overhead: 0.1,
        compression_ratio: 1.0,
        bandwidth_utilization: 100.0,
        step_time: Duration::from_millis(10),
    }
}

fn scaler_with_zero_cooldown(min_nodes: usize, max_nodes: usize) -> AutoScaler {
    AutoScaler::new(AutoScalerConfig {
        min_nodes,
        max_nodes,
        strategy: ScalingStrategy::Performance,
        scale_up_threshold: 0.85,
        scale_down_threshold: 0.6,
        scaling_cooldown: Duration::from_secs(0),
        predictive_scaling: false,
        cost_priority: 0.3,
    })
}

/// A [`NodeProvider`] test double that records every call it receives (so
/// tests can prove it was actually invoked, not just that no error
/// surfaced) and can be configured to under-provision/under-terminate.
#[derive(Default)]
struct CountingNodeProvider {
    provision_calls: Mutex<Vec<usize>>,
    terminate_calls: Mutex<Vec<usize>>,
    provision_shortfall: usize,
    terminate_shortfall: usize,
}

impl NodeProvider for CountingNodeProvider {
    fn provision_nodes(&self, count: usize) -> Result<usize> {
        self.provision_calls.lock().expect("lock should not be poisoned").push(count);
        Ok(count.saturating_sub(self.provision_shortfall))
    }

    fn terminate_nodes(&self, count: usize) -> Result<usize> {
        self.terminate_calls.lock().expect("lock should not be poisoned").push(count);
        Ok(count.saturating_sub(self.terminate_shortfall))
    }
}

#[test]
fn test_scale_up_without_node_provider_is_a_structured_error_not_fake_success() {
    let mut scaler = scaler_with_zero_cooldown(2, 16);

    let result = scaler.update_and_scale(&scaling_metrics(0.95, 2));

    let err = result.expect_err("scale-up with no NodeProvider must fail, not fabricate success");
    assert!(err.to_string().contains("NodeProvider"), "{err}");
    assert_eq!(
        scaler.get_current_nodes(),
        2,
        "current_nodes must not change: no node was actually provisioned"
    );
    assert!(
        scaler.get_scaling_history().is_empty(),
        "no scaling event actually happened, so none should be recorded"
    );
}

#[test]
fn test_scale_down_without_node_provider_is_a_structured_error_not_fake_success() {
    // `with_min_nodes` only ever RAISES `current_nodes` to meet a new
    // minimum, never lowers it (see its doc comment / implementation) --
    // so starting at min_nodes=5 and then lowering the configured minimum
    // to 2 leaves current_nodes at 5, strictly above the new min_nodes,
    // with no NodeProvider involved anywhere.
    let mut scaler = AutoScaler::new(AutoScalerConfig {
        min_nodes: 5,
        max_nodes: 16,
        strategy: ScalingStrategy::Performance,
        scale_up_threshold: 0.85,
        scale_down_threshold: 0.6,
        scaling_cooldown: Duration::from_secs(0),
        predictive_scaling: false,
        cost_priority: 0.3,
    })
    .with_min_nodes(2);
    assert_eq!(scaler.get_current_nodes(), 5);

    let result = scaler.update_and_scale(&scaling_metrics(0.1, 5));

    let err = result.expect_err("scale-down with no NodeProvider must fail, not fabricate success");
    assert!(err.to_string().contains("NodeProvider"), "{err}");
    assert_eq!(
        scaler.get_current_nodes(),
        5,
        "current_nodes must not change: no node was actually terminated"
    );
    assert!(
        scaler.get_scaling_history().is_empty(),
        "no scaling event actually happened, so none should be recorded"
    );
}

#[test]
fn test_scale_up_with_node_provider_succeeds_and_is_actually_invoked() {
    let provider = Arc::new(CountingNodeProvider::default());
    let mut scaler = scaler_with_zero_cooldown(2, 16).with_node_provider(provider.clone());

    let decision = scaler
        .update_and_scale(&scaling_metrics(0.95, 2))
        .expect("scale-up must succeed once a NodeProvider covers the request");

    assert!(
        matches!(decision, ScalingDecision::ScaleUp(1)),
        "{decision:?}"
    );
    assert_eq!(
        provider.provision_calls.lock().expect("lock should not be poisoned").as_slice(),
        [1],
        "the provider must actually be asked for the node, not bypassed"
    );
    assert_eq!(scaler.get_current_nodes(), 3);
    assert_eq!(scaler.get_scaling_history().len(), 1);
    assert_eq!(scaler.get_scaling_history()[0].nodes_changed, 1);
}

#[test]
fn test_scale_down_with_node_provider_succeeds_and_is_actually_invoked() {
    let provider = Arc::new(CountingNodeProvider::default());
    let mut scaler = scaler_with_zero_cooldown(2, 16).with_node_provider(provider.clone());

    scaler
        .update_and_scale(&scaling_metrics(0.95, 2))
        .expect("scale-up must succeed in test");
    assert_eq!(scaler.get_current_nodes(), 3);

    let decision = scaler
        .update_and_scale(&scaling_metrics(0.1, 3))
        .expect("scale-down must succeed once a NodeProvider covers the request");

    assert!(
        matches!(decision, ScalingDecision::ScaleDown(1)),
        "{decision:?}"
    );
    assert_eq!(
        provider.terminate_calls.lock().expect("lock should not be poisoned").as_slice(),
        [1],
        "the provider must actually be asked to terminate the node, not bypassed"
    );
    assert_eq!(scaler.get_current_nodes(), 2);
}

#[test]
fn test_scale_up_partial_provisioning_is_reported_as_failure_but_reflects_reality() {
    let provider = Arc::new(CountingNodeProvider {
        provision_shortfall: 1,
        ..Default::default()
    });
    let mut scaler = scaler_with_zero_cooldown(4, 16).with_node_provider(provider.clone());

    // avg_utilization 0.99 against a 0.85 threshold with current_nodes=4
    // computes a request for 2 nodes; the provider only provisions 1.
    let err = scaler
        .update_and_scale(&scaling_metrics(0.99, 4))
        .expect_err("provisioning fewer nodes than requested must be reported as an error");

    assert!(
        err.to_string().contains('2') && err.to_string().contains('1'),
        "{err}"
    );
    assert_eq!(
        provider.provision_calls.lock().expect("lock should not be poisoned").as_slice(),
        [2]
    );
    assert_eq!(
        scaler.get_current_nodes(),
        5,
        "current_nodes must reflect the 1 node actually provisioned, not the 2 requested"
    );
    assert_eq!(scaler.get_scaling_history().len(), 1);
    assert_eq!(scaler.get_scaling_history()[0].nodes_changed, 1);
}

/// Regression: `execute_scale_up`/`execute_scale_down` used to record a
/// hardcoded `ScalingEvent::reason` ("Performance threshold exceeded" /
/// "Low utilization detected") no matter which `ScalingStrategy` actually
/// fired -- false for three of the four strategies. Each strategy method
/// now returns the reason it genuinely computed, threaded through
/// unchanged into the event.
#[test]
fn test_scale_up_reason_names_the_queue_based_strategy_not_a_hardcoded_string() {
    let provider = Arc::new(CountingNodeProvider::default());
    let mut scaler = AutoScaler::new(AutoScalerConfig {
        min_nodes: 2,
        max_nodes: 16,
        strategy: ScalingStrategy::QueueBased,
        scale_up_threshold: 0.85,
        scale_down_threshold: 0.6,
        scaling_cooldown: Duration::from_secs(0),
        predictive_scaling: false,
        cost_priority: 0.3,
    })
    .with_node_provider(provider);

    // GPU utilization 0.5 is nowhere near the Performance strategy's own
    // 0.85 scale-up threshold, so this decision can only be explained by
    // `scaling_metrics`'s fixed low throughput (100.0, i.e. a 0.1 ratio
    // against queue_based_scaling's assumed 1000/sec baseline) -- the
    // QueueBased strategy actually configured, not GPU utilization.
    let decision = scaler
        .update_and_scale(&scaling_metrics(0.5, 2))
        .expect("QueueBased scale-up must succeed once a NodeProvider covers the request");
    assert!(
        matches!(decision, ScalingDecision::ScaleUp(1)),
        "{decision:?}"
    );

    let history = scaler.get_scaling_history();
    assert_eq!(history.len(), 1);
    let reason = &history[0].reason;
    assert!(
        reason.contains("Queue-based") && reason.contains("throughput"),
        "reason must name the strategy that actually fired: {reason}"
    );
    assert!(
        !reason.contains("Performance threshold exceeded") && !reason.contains("GPU utilization"),
        "reason must not be the old hardcoded Performance-strategy string: {reason}"
    );
}

#[test]
fn test_scale_down_reason_names_the_cost_optimized_strategy_not_a_hardcoded_string() {
    let provider = Arc::new(CountingNodeProvider::default());
    let mut scaler = AutoScaler::new(AutoScalerConfig {
        min_nodes: 1,
        max_nodes: 16,
        strategy: ScalingStrategy::CostOptimized,
        scale_up_threshold: 0.85,
        scale_down_threshold: 0.6,
        scaling_cooldown: Duration::from_secs(0),
        predictive_scaling: false,
        cost_priority: 0.3,
    })
    .with_node_provider(provider);

    // Seed a second node via a genuine cost-optimized scale-up so there is
    // room to scale back down below it.
    scaler
        .update_and_scale(&scaling_metrics(0.99, 1))
        .expect("seed scale-up failed");
    assert_eq!(scaler.get_current_nodes(), 2);

    let decision = scaler
        .update_and_scale(&scaling_metrics(0.1, 2))
        .expect("CostOptimized scale-down must succeed once a NodeProvider covers the request");
    assert!(
        matches!(decision, ScalingDecision::ScaleDown(1)),
        "{decision:?}"
    );

    let history = scaler.get_scaling_history();
    assert_eq!(history.len(), 2);
    let reason = &history[1].reason;
    assert!(
        reason.contains("Cost-optimized") && reason.contains("savings"),
        "reason must name the strategy that actually fired: {reason}"
    );
    assert!(
        !reason.contains("Low utilization detected"),
        "reason must not be the old hardcoded Performance-strategy string: {reason}"
    );
}

#[test]
fn test_simulated_node_provider_lets_update_and_scale_succeed_end_to_end() {
    let mut scaler =
        scaler_with_zero_cooldown(2, 16).with_node_provider(Arc::new(SimulatedNodeProvider::new()));

    let decision = scaler
        .update_and_scale(&scaling_metrics(0.95, 2))
        .expect("SimulatedNodeProvider must let scaling succeed end to end");

    assert!(
        matches!(decision, ScalingDecision::ScaleUp(1)),
        "{decision:?}"
    );
    assert_eq!(scaler.get_current_nodes(), 3);
}
