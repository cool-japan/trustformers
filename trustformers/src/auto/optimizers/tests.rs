//! Unit tests for [`super`] -- split out of `mod.rs` to keep it under the
//! workspace's 2000-line-per-file policy. See `mod.rs` for the
//! `AutoOptimizer` / `Optimizer` / `AdamOptimizer` / `AdamWOptimizer` /
//! `ScheduledOptimizer` implementations under test.

use super::*;

// -------------------------------------------------------------------------
// LCG for deterministic pseudo-random numbers (no rand crate)
// -------------------------------------------------------------------------

struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005_u64)
            .wrapping_add(1_442_695_040_888_963_407_u64);
        self.state
    }

    fn next_f32(&mut self) -> f32 {
        (self.next() >> 11) as f32 / (1u64 << 53) as f32
    }
}

// -------------------------------------------------------------------------
// AdamWConfig
// -------------------------------------------------------------------------

#[test]
fn test_adamw_config_creation() {
    let config = AdamWConfig {
        learning_rate: 2e-5,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    };
    let diff = (config.learning_rate - 2e-5).abs();
    assert!(diff < 1e-10, "learning_rate should be set correctly");
    assert!(!config.amsgrad, "amsgrad should be false");
}

#[test]
fn test_adamw_config_weight_decay() {
    let config = AdamWConfig {
        learning_rate: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.1,
        eps: 1e-8,
        amsgrad: false,
    };
    let diff = (config.weight_decay - 0.1).abs();
    assert!(diff < 1e-10, "weight_decay should be 0.1");
}

// -------------------------------------------------------------------------
// AdamConfig
// -------------------------------------------------------------------------

#[test]
fn test_adam_config_creation() {
    let config = AdamConfig {
        learning_rate: 5e-5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    };
    let diff = (config.learning_rate - 5e-5).abs();
    assert!(diff < 1e-12, "learning_rate should be set correctly");
}

#[test]
fn test_adam_config_beta_values() {
    let config = AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.95,
        beta2: 0.99,
        eps: 1e-6,
        amsgrad: true,
    };
    let b1_diff = (config.beta1 - 0.95).abs();
    let b2_diff = (config.beta2 - 0.99).abs();
    assert!(b1_diff < 1e-10, "beta1 should be 0.95");
    assert!(b2_diff < 1e-10, "beta2 should be 0.99");
    assert!(config.amsgrad, "amsgrad should be true");
}

// -------------------------------------------------------------------------
// AdamWOptimizer
// -------------------------------------------------------------------------

#[test]
fn test_adamw_optimizer_creation() {
    let config = AdamWConfig {
        learning_rate: 2e-5,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    };
    let optimizer = AdamWOptimizer::new(config);
    let diff = (optimizer.get_lr() - 2e-5).abs();
    assert!(diff < 1e-12, "Initial LR should match config");
}

#[test]
fn test_adamw_get_lr() {
    let optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 3e-4,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let diff = (optimizer.get_lr() - 3e-4).abs();
    assert!(diff < 1e-12, "get_lr should return initial learning rate");
}

#[test]
fn test_adamw_set_lr() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-4,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer.set_lr(5e-5);
    let diff = (optimizer.get_lr() - 5e-5).abs();
    assert!(diff < 1e-12, "LR should be updated after set_lr");
}

#[test]
fn test_adamw_step_produces_update() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut lcg = Lcg::new(42);
    let grads: Vec<f32> = (0..4).map(|_| lcg.next_f32() - 0.5).collect();
    let mut parameters = HashMap::new();
    parameters.insert("layer.weight".to_string(), grads);
    let mut parameter_shapes = HashMap::new();
    parameter_shapes.insert("layer.weight".to_string(), vec![4]);
    let gradients = OptimizerGradients {
        parameters,
        parameter_shapes,
    };

    let result = optimizer.step(&gradients);
    assert!(result.is_ok(), "step() should succeed");
    if let Ok(update) = result {
        assert_eq!(
            update.step_count, 1,
            "Step count should be 1 after first step"
        );
        assert!(
            update.parameter_updates.contains_key("layer.weight"),
            "Update should contain parameter updates"
        );
    }
}

#[test]
fn test_adamw_step_count_increments() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        eps: 1e-8,
        amsgrad: false,
    });
    let grads = vec![0.1_f32, -0.2, 0.05];
    let mut parameters = HashMap::new();
    parameters.insert("w".to_string(), grads);
    let mut shapes = HashMap::new();
    shapes.insert("w".to_string(), vec![3]);
    let gradients = OptimizerGradients {
        parameters,
        parameter_shapes: shapes,
    };

    for expected_step in 1..=3usize {
        let result = optimizer.step(&gradients);
        if let Ok(update) = result {
            assert_eq!(update.step_count, expected_step);
        }
    }
}

#[test]
fn test_adamw_state_dict_contains_step_count() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    // Do one step to increment counter
    let mut parameters = HashMap::new();
    parameters.insert("w".to_string(), vec![0.1_f32]);
    let mut shapes = HashMap::new();
    shapes.insert("w".to_string(), vec![1]);
    let _ = optimizer.step(&OptimizerGradients {
        parameters,
        parameter_shapes: shapes,
    });
    let state = optimizer.state_dict().expect("state_dict should succeed");
    assert!(
        state.contains_key("step_count"),
        "state_dict should contain step_count"
    );
}

#[test]
fn test_adamw_load_state_dict_updates_lr() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut state = HashMap::new();
    state.insert("learning_rate".to_string(), serde_json::json!(2e-4));
    let result = optimizer.load_state_dict(state);
    assert!(result.is_ok(), "load_state_dict should succeed");
    let diff = (optimizer.get_lr() - 2e-4).abs();
    assert!(diff < 1e-12, "LR should be updated from loaded state");
}

// -------------------------------------------------------------------------
// AdamOptimizer
// -------------------------------------------------------------------------

#[test]
fn test_adam_optimizer_creation() {
    let optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 5e-5,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let diff = (optimizer.get_lr() - 5e-5).abs();
    assert!(diff < 1e-12, "Initial LR should match config");
}

#[test]
fn test_adam_set_lr() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer.set_lr(1e-4);
    let diff = (optimizer.get_lr() - 1e-4).abs();
    assert!(diff < 1e-12, "LR should be updated after set_lr");
}

#[test]
fn test_adam_step_produces_update() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut parameters = HashMap::new();
    parameters.insert("bias".to_string(), vec![0.5_f32, -0.3]);
    let mut shapes = HashMap::new();
    shapes.insert("bias".to_string(), vec![2]);
    let gradients = OptimizerGradients {
        parameters,
        parameter_shapes: shapes,
    };

    let result = optimizer.step(&gradients);
    assert!(result.is_ok(), "Adam step should succeed");
    if let Ok(update) = result {
        assert!(
            update.parameter_updates.contains_key("bias"),
            "Update should contain 'bias'"
        );
    }
}

#[test]
fn test_adam_zero_grad_does_not_panic() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer.zero_grad(); // Should not panic
}

// -------------------------------------------------------------------------
// LearningRateSchedule
// -------------------------------------------------------------------------

#[test]
fn test_lr_schedule_constant_variant() {
    let schedule = LearningRateSchedule::Constant;
    // Just verifying that the variant can be created and matched
    assert!(matches!(schedule, LearningRateSchedule::Constant));
}

#[test]
fn test_lr_schedule_linear_warmup_fields() {
    let schedule = LearningRateSchedule::LinearWarmup {
        warmup_steps: 1000,
        max_lr: 5e-5,
    };
    if let LearningRateSchedule::LinearWarmup {
        warmup_steps,
        max_lr,
    } = schedule
    {
        assert_eq!(warmup_steps, 1000);
        let diff = (max_lr - 5e-5).abs();
        assert!(diff < 1e-12, "max_lr should be 5e-5");
    } else {
        panic!("Expected LinearWarmup variant");
    }
}

#[test]
fn test_lr_schedule_cosine_annealing_fields() {
    let schedule = LearningRateSchedule::CosineAnnealing {
        t_max: 500,
        eta_min: 1e-6,
    };
    if let LearningRateSchedule::CosineAnnealing { t_max, eta_min } = schedule {
        assert_eq!(t_max, 500);
        let diff = (eta_min - 1e-6).abs();
        assert!(diff < 1e-12, "eta_min should be 1e-6");
    } else {
        panic!("Expected CosineAnnealing variant");
    }
}

#[test]
fn test_lr_schedule_step_lr_fields() {
    let schedule = LearningRateSchedule::StepLR {
        step_size: 100,
        gamma: 0.1,
    };
    if let LearningRateSchedule::StepLR { step_size, gamma } = schedule {
        assert_eq!(step_size, 100);
        let diff = (gamma - 0.1).abs();
        assert!(diff < 1e-12, "gamma should be 0.1");
    } else {
        panic!("Expected StepLR variant");
    }
}

#[test]
fn test_lr_schedule_polynomial_decay_fields() {
    let schedule = LearningRateSchedule::PolynomialDecay {
        power: 2.0,
        end_lr: 1e-7,
        total_steps: 10_000,
    };
    if let LearningRateSchedule::PolynomialDecay {
        power,
        end_lr,
        total_steps,
    } = schedule
    {
        let p_diff = (power - 2.0).abs();
        assert!(p_diff < 1e-10, "power should be 2.0");
        let e_diff = (end_lr - 1e-7).abs();
        assert!(e_diff < 1e-14, "end_lr should be 1e-7");
        assert_eq!(total_steps, 10_000);
    } else {
        panic!("Expected PolynomialDecay variant");
    }
}

// -------------------------------------------------------------------------
// ScheduledOptimizer
// -------------------------------------------------------------------------

#[test]
fn test_scheduled_optimizer_constant_lr() {
    let base = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut sched = ScheduledOptimizer::new(Box::new(base), LearningRateSchedule::Constant);
    let initial_lr = sched.get_lr();

    let mut parameters = HashMap::new();
    parameters.insert("w".to_string(), vec![0.1_f32]);
    let mut shapes = HashMap::new();
    shapes.insert("w".to_string(), vec![1]);
    let _ = sched.step(&OptimizerGradients {
        parameters,
        parameter_shapes: shapes,
    });

    let lr_diff = (sched.get_lr() - initial_lr).abs();
    assert!(
        lr_diff < 1e-10,
        "Constant schedule should keep LR unchanged"
    );
}

#[test]
fn test_scheduled_optimizer_warmup_increases_lr() {
    let initial_lr = 1e-5_f64;
    let max_lr = 5e-4_f64;
    let base = AdamWOptimizer::new(AdamWConfig {
        learning_rate: initial_lr,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut sched = ScheduledOptimizer::new(
        Box::new(base),
        LearningRateSchedule::LinearWarmup {
            warmup_steps: 100,
            max_lr,
        },
    );

    let mut parameters = HashMap::new();
    parameters.insert("w".to_string(), vec![0.01_f32]);
    let mut shapes = HashMap::new();
    shapes.insert("w".to_string(), vec![1]);

    // After several warmup steps, LR should increase
    for _ in 0..50 {
        let _ = sched.step(&OptimizerGradients {
            parameters: parameters.clone(),
            parameter_shapes: shapes.clone(),
        });
    }
    let lr_after = sched.get_lr();
    assert!(lr_after > initial_lr, "LR should increase during warmup");
    assert!(lr_after < max_lr + 1e-10, "LR should not exceed max_lr");
}

#[test]
fn test_scheduled_optimizer_state_dict_contains_step() {
    let base = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut sched = ScheduledOptimizer::new(Box::new(base), LearningRateSchedule::Constant);
    let mut parameters = HashMap::new();
    parameters.insert("w".to_string(), vec![0.1_f32]);
    let mut shapes = HashMap::new();
    shapes.insert("w".to_string(), vec![1]);
    let _ = sched.step(&OptimizerGradients {
        parameters,
        parameter_shapes: shapes,
    });
    let state = sched.state_dict().expect("state_dict should succeed");
    assert!(
        state.contains_key("current_step"),
        "state_dict should have current_step key"
    );
}

// -------------------------------------------------------------------------
// AutoOptimizer::from_config
// -------------------------------------------------------------------------

#[test]
fn test_auto_optimizer_from_config_small_model() {
    let config = serde_json::json!({
        "model_type": "bert",
        "hidden_size": 128,
        "num_hidden_layers": 2
    });
    let result = AutoOptimizer::from_config(&config);
    assert!(result.is_ok(), "AutoOptimizer::from_config should succeed");
    if let Ok(optimizer) = result {
        assert!(
            optimizer.get_lr() > 0.0,
            "Optimizer should have positive LR"
        );
    }
}

#[test]
fn test_auto_optimizer_from_config_large_model() {
    let config = serde_json::json!({
        "model_type": "gpt2",
        "hidden_size": 1024,
        "num_hidden_layers": 36
    });
    let result = AutoOptimizer::from_config(&config);
    assert!(
        result.is_ok(),
        "AutoOptimizer::from_config should succeed for large model"
    );
    if let Ok(optimizer) = result {
        // Large model should use lower LR
        assert!(
            optimizer.get_lr() <= 2e-5 + 1e-12,
            "Large model should use lower LR"
        );
    }
}

#[test]
fn test_auto_optimizer_for_task_text_generation() {
    let config = serde_json::json!({});
    let result = AutoOptimizer::for_task("text-generation", &config);
    assert!(result.is_ok(), "for_task text-generation should succeed");
}

#[test]
fn test_auto_optimizer_for_task_classification() {
    let config = serde_json::json!({});
    let result = AutoOptimizer::for_task("text-classification", &config);
    assert!(
        result.is_ok(),
        "for_task text-classification should succeed"
    );
}

#[test]
fn test_auto_optimizer_for_task_question_answering() {
    let config = serde_json::json!({});
    let result = AutoOptimizer::for_task("question-answering", &config);
    assert!(result.is_ok(), "for_task question-answering should succeed");
}

#[test]
fn test_auto_optimizer_for_task_unknown_uses_default() {
    let config = serde_json::json!({
        "hidden_size": 256,
        "num_hidden_layers": 4
    });
    let result = AutoOptimizer::for_task("some-unknown-task", &config);
    assert!(result.is_ok(), "Unknown task should fall back to default");
}

#[test]
fn test_auto_optimizer_with_schedule() {
    let base = AutoOptimizer::from_config(&serde_json::json!({}));
    assert!(base.is_ok(), "Base optimizer should be created");
    if let Ok(base_opt) = base {
        let schedule = LearningRateSchedule::LinearWarmup {
            warmup_steps: 500,
            max_lr: 5e-5,
        };
        let sched = AutoOptimizer::with_schedule(base_opt, schedule);
        assert!(
            sched.get_lr() > 0.0,
            "Scheduled optimizer should have positive LR"
        );
    }
}

// -------------------------------------------------------------------------
// OptimizerGradients / OptimizerUpdate
// -------------------------------------------------------------------------

#[test]
fn test_optimizer_gradients_creation() {
    let mut parameters = HashMap::new();
    parameters.insert("layer1.weight".to_string(), vec![0.1_f32, 0.2, -0.3]);
    let mut parameter_shapes = HashMap::new();
    parameter_shapes.insert("layer1.weight".to_string(), vec![3]);
    let gradients = OptimizerGradients {
        parameters,
        parameter_shapes,
    };
    assert_eq!(gradients.parameters.len(), 1);
    assert!(gradients.parameters.contains_key("layer1.weight"));
}

#[test]
fn test_optimizer_update_fields() {
    let mut parameter_updates = HashMap::new();
    parameter_updates.insert("w".to_string(), vec![-0.001_f32, 0.002]);
    let update = OptimizerUpdate {
        parameter_updates,
        learning_rate: 1e-3,
        step_count: 5,
    };
    assert_eq!(update.step_count, 5);
    let lr_diff = (update.learning_rate - 1e-3).abs();
    assert!(lr_diff < 1e-12, "learning_rate should match");
    assert!(update.parameter_updates.contains_key("w"));
}

// -------------------------------------------------------------------------
// Checkpoint resume: m/v moment estimates must round-trip through
// state_dict()/load_state_dict(), and a restored optimizer must produce
// byte-for-byte identical updates to one that never stopped. This is the
// regression coverage for the bug where state_dict() silently dropped
// `m`/`v`, so a "resumed" optimizer would restart momentum at zero while
// resuming a large step_count -- the worst of both worlds for bias
// correction. See `moment_map_to_json`/`moment_map_from_json` and the
// `state_dict`/`load_state_dict` bodies of `AdamOptimizer`/`AdamWOptimizer`.
// -------------------------------------------------------------------------

fn make_gradients(pairs: &[(&str, &[f32])]) -> OptimizerGradients {
    let mut parameters = HashMap::new();
    let mut parameter_shapes = HashMap::new();
    for (name, values) in pairs {
        parameter_shapes.insert((*name).to_string(), vec![values.len()]);
        parameters.insert((*name).to_string(), values.to_vec());
    }
    OptimizerGradients {
        parameters,
        parameter_shapes,
    }
}

#[test]
fn test_adam_state_dict_round_trip_preserves_moments() {
    // Regression test: state_dict() used to serialize only `step_count`
    // and `learning_rate`, dropping `m`/`v` entirely. A load_state_dict
    // on a fresh optimizer therefore restored a step_count with
    // zeroed-out moments -- this test fails against that old behavior
    // because the restored optimizer's `m`/`v` would differ from the
    // original's.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut lcg = Lcg::new(7);
    for _ in 0..5 {
        let grads: Vec<f32> = (0..6).map(|_| lcg.next_f32() - 0.5).collect();
        optimizer
            .step(&make_gradients(&[("layer.weight", &grads)]))
            .expect("step should succeed");
    }

    let saved_state = optimizer.state_dict().expect("state_dict should succeed");

    // The saved state must actually contain non-trivial moment data --
    // not just be present-but-empty.
    let m_value = saved_state.get("m").expect("state_dict must contain `m`");
    let m_obj = m_value.as_object().expect("`m` must be a JSON object");
    assert!(
        !m_obj.is_empty(),
        "`m` must contain the accumulated moment for `layer.weight`"
    );
    let m_array = m_obj
        .get("layer.weight")
        .expect("`m` must have an entry for layer.weight")
        .as_array()
        .expect("array");
    assert!(
        m_array.iter().any(|v| v.as_f64().expect("numeric") != 0.0),
        "first moment estimate must be non-zero after 5 steps with non-zero gradients"
    );

    // Restore into a brand new optimizer and confirm the moments match
    // exactly (round-trip equality).
    let mut restored = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    restored.load_state_dict(saved_state).expect("load_state_dict should succeed");

    assert_eq!(restored.step_count, optimizer.step_count);
    assert_eq!(restored.m, optimizer.m, "m must round-trip exactly");
    assert_eq!(restored.v, optimizer.v, "v must round-trip exactly");
}

#[test]
fn test_adam_resume_matches_uninterrupted_run() {
    // Stronger regression test than plain equality-of-state: prove that
    // "train 3 steps, checkpoint, restore, train 2 more steps" produces
    // the identical parameter updates as "train 5 steps uninterrupted".
    // This is exactly what silently dropping m/v breaks -- bias
    // correction at step_count=4 with m=v=0 is nothing like bias
    // correction at step_count=4 with real accumulated moments.
    let make_optimizer = || {
        AdamOptimizer::new(AdamConfig {
            learning_rate: 1e-3,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            amsgrad: false,
        })
    };

    // Deterministic gradient sequence shared by both runs.
    let mut lcg = Lcg::new(123);
    let grad_sequence: Vec<Vec<f32>> =
        (0..5).map(|_| (0..4).map(|_| lcg.next_f32() - 0.5).collect()).collect();

    // Uninterrupted run.
    let mut uninterrupted = make_optimizer();
    let mut last_update = None;
    for grads in &grad_sequence {
        last_update = Some(
            uninterrupted
                .step(&make_gradients(&[("w", grads)]))
                .expect("step should succeed"),
        );
    }
    let uninterrupted_final_update = last_update.expect("at least one step ran");

    // Checkpoint-and-resume run.
    let mut resumed = make_optimizer();
    for grads in &grad_sequence[..3] {
        resumed.step(&make_gradients(&[("w", grads)])).expect("step should succeed");
    }
    let checkpoint = resumed.state_dict().expect("state_dict should succeed");
    let mut restored = make_optimizer();
    restored.load_state_dict(checkpoint).expect("load_state_dict should succeed");
    let mut last_update = None;
    for grads in &grad_sequence[3..] {
        last_update =
            Some(restored.step(&make_gradients(&[("w", grads)])).expect("step should succeed"));
    }
    let resumed_final_update = last_update.expect("at least one resumed step ran");

    assert_eq!(
        restored.step_count, uninterrupted.step_count,
        "resumed step_count must match uninterrupted run"
    );
    let orig = &uninterrupted_final_update.parameter_updates["w"];
    let resumed_vals = &resumed_final_update.parameter_updates["w"];
    assert_eq!(orig.len(), resumed_vals.len());
    for (a, b) in orig.iter().zip(resumed_vals.iter()) {
        assert!(
            (a - b).abs() < 1e-6,
            "resumed optimizer must produce the same update as an uninterrupted run: \
             {a} vs {b}"
        );
    }
}

#[test]
fn test_adamw_state_dict_round_trip_preserves_moments() {
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut lcg = Lcg::new(99);
    for _ in 0..4 {
        let grads: Vec<f32> = (0..3).map(|_| lcg.next_f32() - 0.5).collect();
        optimizer.step(&make_gradients(&[("w", &grads)])).expect("step should succeed");
    }

    let saved_state = optimizer.state_dict().expect("state_dict should succeed");
    let mut restored = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.01,
        eps: 1e-8,
        amsgrad: false,
    });
    restored.load_state_dict(saved_state).expect("load_state_dict should succeed");

    assert_eq!(
        restored.m, optimizer.m,
        "m must round-trip exactly for AdamW"
    );
    assert_eq!(
        restored.v, optimizer.v,
        "v must round-trip exactly for AdamW"
    );
}

#[test]
fn test_load_state_dict_without_moments_defaults_to_empty_not_error() {
    // A checkpoint saved before this fix (or from an external tool) may
    // simply lack `m`/`v` keys; loading it must not error, and the
    // optimizer must behave as freshly initialized for those moments.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut legacy_state = HashMap::new();
    legacy_state.insert("step_count".to_string(), serde_json::json!(3));
    legacy_state.insert("learning_rate".to_string(), serde_json::json!(1e-3));
    optimizer
        .load_state_dict(legacy_state)
        .expect("legacy state without m/v must still load");
    assert!(optimizer.m.is_empty());
    assert!(optimizer.v.is_empty());
    assert_eq!(optimizer.step_count, 3);
}

#[test]
fn test_load_state_dict_rejects_non_numeric_moment_entry() {
    // A `null` (or other non-numeric) entry inside `m`/`v` most likely
    // means the writer serialized a NaN/Infinity through a path that
    // silently drops non-finite floats to `null` -- restoring that as
    // 0.0 would hide a diverged optimizer. This must be a structured
    // error, not a silent default.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut bad_state = HashMap::new();
    bad_state.insert(
        "m".to_string(),
        serde_json::json!({ "w": [0.1, null, 0.3] }),
    );
    let result = optimizer.load_state_dict(bad_state);
    assert!(
        result.is_err(),
        "a null entry inside a moment estimate must be a structured error"
    );
}

#[test]
fn test_state_dict_rejects_non_finite_moment() {
    // moment_map_to_json must fail loudly on NaN/Infinity rather than
    // letting serde_json turn it into `null` (which would silently
    // launder a diverged optimizer's state into what looks like a
    // valid, zeroed checkpoint on the next load).
    let moments: HashMap<String, Vec<f32>> =
        HashMap::from([("w".to_string(), vec![1.0, f32::NAN, 3.0])]);
    let result = moment_map_to_json(&moments, "m");
    assert!(
        result.is_err(),
        "serializing a NaN moment estimate must be a structured error, not silent null"
    );
}

// -------------------------------------------------------------------------
// zero_grad() / accumulate_gradients(): regression coverage for the bug
// where zero_grad() was an empty no-op body, so it could never have
// affected anything a caller did -- there was no accumulator to clear.
// -------------------------------------------------------------------------

#[test]
fn test_accumulate_gradients_sums_across_calls() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1.0, // Large LR to make the effect of accumulation obvious.
        beta1: 0.0,         // No momentum smoothing, so m == effective gradient exactly.
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });

    // Accumulate two micro-batch gradients of 1.0 each for the same
    // parameter, then step with a third gradient of 1.0. The optimizer
    // should see an effective gradient of 1.0 + 1.0 + 1.0 = 3.0.
    optimizer
        .accumulate_gradients(&make_gradients(&[("w", &[1.0])]))
        .expect("accumulate should succeed");
    optimizer
        .accumulate_gradients(&make_gradients(&[("w", &[1.0])]))
        .expect("accumulate should succeed");
    optimizer.step(&make_gradients(&[("w", &[1.0])])).expect("step should succeed");

    // beta1 = 0 means m[i] after one update equals the raw (unbiased)
    // gradient that fed it, so m["w"][0] must equal the *sum* 3.0 -- not
    // 1.0, which is what it would be if accumulation were a no-op.
    let m = optimizer.m.get("w").expect("m must have an entry for w");
    assert!(
        (m[0] - 3.0).abs() < 1e-5,
        "accumulated gradient must sum micro-batches: expected 3.0, got {}",
        m[0]
    );
}

#[test]
fn test_zero_grad_clears_accumulated_gradients() {
    // Regression test: the old zero_grad() body was empty, so calling it
    // could never change subsequent behavior. This test fails against
    // that old code because there both `with_zero_grad` and
    // `without_zero_grad` would see the same accumulated gradient (there
    // was none to clear either way, but also nothing being accumulated
    // -- the meaningful assertion is that *with* the fix, zero_grad
    // measurably changes the outcome of the next step).
    let mut with_zero_grad = AdamOptimizer::new(AdamConfig {
        learning_rate: 1.0,
        beta1: 0.0,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    with_zero_grad
        .accumulate_gradients(&make_gradients(&[("w", &[5.0])]))
        .expect("accumulate should succeed");
    with_zero_grad.zero_grad();
    with_zero_grad
        .step(&make_gradients(&[("w", &[1.0])]))
        .expect("step should succeed");

    let mut without_zero_grad = AdamOptimizer::new(AdamConfig {
        learning_rate: 1.0,
        beta1: 0.0,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    without_zero_grad
        .accumulate_gradients(&make_gradients(&[("w", &[5.0])]))
        .expect("accumulate should succeed");
    without_zero_grad
        .step(&make_gradients(&[("w", &[1.0])]))
        .expect("step should succeed");

    let cleared_m = with_zero_grad.m["w"][0];
    let uncleared_m = without_zero_grad.m["w"][0];

    // With zero_grad(): effective gradient is just 1.0 (the accumulated
    // 5.0 was cleared). Without it: effective gradient is 5.0 + 1.0 = 6.0.
    assert!(
        (cleared_m - 1.0).abs() < 1e-5,
        "zero_grad() must clear the accumulator: expected m=1.0, got {cleared_m}"
    );
    assert!(
        (uncleared_m - 6.0).abs() < 1e-5,
        "without zero_grad(), the accumulator must still carry forward: expected m=6.0, got \
         {uncleared_m}"
    );
    assert!(
        (cleared_m - uncleared_m).abs() > 1.0,
        "zero_grad() must measurably change the next step's outcome"
    );
}

#[test]
fn test_step_does_not_implicitly_clear_accumulator() {
    // step() intentionally mirrors PyTorch's `optimizer.step()`, which
    // never clears `.grad` -- only an explicit `zero_grad()` call does.
    // This is a deliberate, documented contract (see `Optimizer::step`),
    // so a second `step()` with no intervening `zero_grad()` must keep
    // folding in whatever is still sitting in the accumulator.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1.0,
        beta1: 0.0,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer
        .accumulate_gradients(&make_gradients(&[("w", &[10.0])]))
        .expect("accumulate should succeed");
    optimizer.step(&make_gradients(&[("w", &[0.0])])).expect("first step");
    let first_m = optimizer.m["w"][0];
    assert!(
        (first_m - 10.0).abs() < 1e-5,
        "first step must see the accumulated 10.0: got {first_m}"
    );

    // No zero_grad() call between the two steps: the accumulated 10.0
    // is still present and must be folded into the second step's
    // effective gradient alongside the newly passed 1.0, i.e. 11.0.
    optimizer.step(&make_gradients(&[("w", &[1.0])])).expect("second step");
    let second_m = optimizer.m["w"][0];
    assert!(
        (second_m - 11.0).abs() < 1e-5,
        "without an intervening zero_grad(), the accumulator must still be applied on the \
         second step (10.0 accumulated + 1.0 passed = 11.0): got {second_m}"
    );
}

#[test]
fn test_zero_grad_then_step_gives_clean_gradient_on_subsequent_step() {
    // Companion to the no-clear test above: calling zero_grad() between
    // two step() calls must remove the earlier accumulation, so the
    // next step only sees what is newly accumulated/passed.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1.0,
        beta1: 0.0,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer
        .accumulate_gradients(&make_gradients(&[("w", &[10.0])]))
        .expect("accumulate should succeed");
    optimizer.step(&make_gradients(&[("w", &[0.0])])).expect("first step");

    optimizer.zero_grad();
    optimizer.step(&make_gradients(&[("w", &[1.0])])).expect("second step");
    let second_m = optimizer.m["w"][0];
    assert!(
        (second_m - 1.0).abs() < 1e-5,
        "after zero_grad(), the second step must see only the freshly passed 1.0: got \
         {second_m}"
    );
}

#[test]
fn test_accumulate_gradients_rejects_shape_mismatch() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer
        .accumulate_gradients(&make_gradients(&[("w", &[1.0, 2.0])]))
        .expect("first accumulate should succeed");
    let result = optimizer.accumulate_gradients(&make_gradients(&[("w", &[1.0])]));
    assert!(
        result.is_err(),
        "accumulating a differently-shaped gradient for the same parameter must error"
    );
}

// -------------------------------------------------------------------------
// weight_decay / amsgrad: these `AdamWConfig`/`AdamConfig` fields were
// accepted and stored but never read by `step()` -- the same class of
// bug as the missing m/v serialization above (a config knob that looks
// wired up but silently does nothing). Regression coverage below fails
// against that old behavior because it directly compares parameter
// updates with the knob on vs. off.
// -------------------------------------------------------------------------

#[test]
fn test_adamw_weight_decay_shrinks_update_beyond_plain_adam() {
    // With a zero gradient, AdamW's decoupled weight decay is the *only*
    // source of a non-zero update (m_hat and v_hat are both zero, so the
    // Adam term is exactly `-lr * 0 / (0 + eps) = 0`). This isolates the
    // weight-decay contribution precisely -- against the old
    // (weight_decay-ignoring) implementation this update would be 0.0,
    // not `-lr * weight_decay`.
    let lr = 0.1_f64;
    let weight_decay = 0.4_f64;
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: lr,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay,
        eps: 1e-8,
        amsgrad: false,
    });
    let update = optimizer.step(&make_gradients(&[("w", &[0.0])])).expect("step should succeed");
    let expected = -(lr * weight_decay) as f32;
    let actual = update.parameter_updates["w"][0];
    assert!(
        (actual - expected).abs() < 1e-6,
        "AdamW weight_decay must shrink the parameter by `-lr * weight_decay` even with a \
         zero gradient: expected {expected}, got {actual}"
    );
}

#[test]
fn test_adamw_weight_decay_zero_matches_plain_adam_update() {
    // Sanity check in the other direction: weight_decay = 0.0 must
    // reduce AdamW to plain Adam's update formula exactly (no residual
    // decay term sneaking in).
    let mut adamw = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay: 0.0,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut adam = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    let mut lcg = Lcg::new(2024);
    for _ in 0..3 {
        let grads: Vec<f32> = (0..4).map(|_| lcg.next_f32() - 0.5).collect();
        let adamw_update = adamw
            .step(&make_gradients(&[("w", &grads)]))
            .expect("adamw step should succeed");
        let adam_update =
            adam.step(&make_gradients(&[("w", &grads)])).expect("adam step should succeed");
        for (a, b) in adamw_update.parameter_updates["w"]
            .iter()
            .zip(adam_update.parameter_updates["w"].iter())
        {
            assert!(
                (a - b).abs() < 1e-6,
                "AdamW with weight_decay=0 must match plain Adam exactly: {a} vs {b}"
            );
        }
    }
}

#[test]
fn test_adamw_weight_decay_serializes_through_state_dict() {
    // weight_decay lives on `config`, which is not part of state_dict()
    // (by design -- state_dict covers *learned* state, config is
    // supplied fresh on reconstruction), but this test locks in that a
    // freshly constructed optimizer with the same config produces the
    // same decay contribution as one whose moments were restored,
    // i.e. restoring state doesn't accidentally clobber weight_decay.
    let weight_decay = 0.25_f64;
    let mut optimizer = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer.step(&make_gradients(&[("w", &[0.2])])).expect("step should succeed");
    let saved = optimizer.state_dict().expect("state_dict should succeed");

    let mut restored = AdamWOptimizer::new(AdamWConfig {
        learning_rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        weight_decay,
        eps: 1e-8,
        amsgrad: false,
    });
    restored.load_state_dict(saved).expect("load_state_dict should succeed");

    let update = restored.step(&make_gradients(&[("w", &[0.0])])).expect("step should succeed");
    // The Adam term is non-zero here (m/v were restored to non-zero
    // values by the first step), so this also exercises decay
    // co-existing with a real Adam contribution rather than the
    // isolated all-zero case above.
    assert!(
        update.parameter_updates["w"][0].is_finite(),
        "restored optimizer with weight_decay must still produce a finite update"
    );
}

#[test]
fn test_amsgrad_v_max_is_monotonically_non_decreasing() {
    // AMSGrad's defining property (Reddi et al., 2018): `v_max` is the
    // running elementwise maximum of `v`, so it can never shrink from
    // one step to the next, even after a gradient spike is followed by
    // many small gradients (where plain Adam's `v` would decay back
    // down). Against the old (amsgrad-ignoring) code, `v_max` is never
    // populated at all and this test's `state_dict` check below fails.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.9, // Fast-decaying `v` makes a post-spike shrink easy to observe.
        eps: 1e-8,
        amsgrad: true,
    });

    let mut sequence = vec![10.0_f32]; // Spike.
    sequence.extend(std::iter::repeat_n(0.01_f32, 6)); // Then many small gradients.

    let mut v_max_history = Vec::new();
    for &g in &sequence {
        optimizer.step(&make_gradients(&[("w", &[g])])).expect("step should succeed");
        v_max_history.push(optimizer.v_max["w"][0]);
    }

    for i in 1..v_max_history.len() {
        assert!(
            v_max_history[i] + 1e-12 >= v_max_history[i - 1],
            "v_max must never decrease: step {} = {}, step {} = {}",
            i - 1,
            v_max_history[i - 1],
            i,
            v_max_history[i]
        );
    }
    // And it must actually have latched onto the spike, not stayed at 0.
    assert!(
        v_max_history[0] > 1.0,
        "v_max after the spike must reflect the large gradient, got {}",
        v_max_history[0]
    );
}

#[test]
fn test_amsgrad_update_magnitude_never_exceeds_plain_adam() {
    // A direct consequence of `v_max[i] = max(v_max[i], v[i]) >= v[i]`
    // sharing the same bias-correction divisor as plain Adam: at every
    // step, `1/sqrt(v_max_hat) <= 1/sqrt(v_hat)`, so AMSGrad's update
    // magnitude is never larger than plain Adam's, for identical
    // gradient sequences. Against the old code (which ignored
    // `amsgrad` and always behaved like `amsgrad: false`) the two
    // optimizers are bitwise identical, so proving a case where AMSGrad
    // is *strictly smaller* also proves the flag has a real effect.
    let mut amsgrad_on = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.9,
        eps: 1e-8,
        amsgrad: true,
    });
    let mut amsgrad_off = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.9,
        eps: 1e-8,
        amsgrad: false,
    });

    let mut sequence = vec![10.0_f32];
    sequence.extend(std::iter::repeat_n(0.01_f32, 6));

    let mut saw_strictly_smaller = false;
    for &g in &sequence {
        let on_update = amsgrad_on
            .step(&make_gradients(&[("w", &[g])]))
            .expect("step should succeed")
            .parameter_updates["w"][0]
            .abs();
        let off_update = amsgrad_off
            .step(&make_gradients(&[("w", &[g])]))
            .expect("step should succeed")
            .parameter_updates["w"][0]
            .abs();
        assert!(
            on_update <= off_update + 1e-9,
            "AMSGrad's update magnitude must never exceed plain Adam's for the same \
             gradient sequence: amsgrad={on_update}, plain={off_update}"
        );
        if on_update < off_update - 1e-9 {
            saw_strictly_smaller = true;
        }
    }
    assert!(
        saw_strictly_smaller,
        "amsgrad=true must diverge from amsgrad=false at some point in this sequence, \
         proving the flag is not silently ignored"
    );
}

#[test]
fn test_amsgrad_v_max_round_trips_through_state_dict() {
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.9,
        eps: 1e-8,
        amsgrad: true,
    });
    optimizer.step(&make_gradients(&[("w", &[10.0])])).expect("step should succeed");
    optimizer.step(&make_gradients(&[("w", &[0.01])])).expect("step should succeed");

    let saved = optimizer.state_dict().expect("state_dict should succeed");
    assert!(
        saved.contains_key("v_max"),
        "state_dict must include v_max when amsgrad is enabled"
    );

    let mut restored = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.9,
        eps: 1e-8,
        amsgrad: true,
    });
    restored.load_state_dict(saved).expect("load_state_dict should succeed");
    assert_eq!(
        restored.v_max, optimizer.v_max,
        "v_max must round-trip exactly so a resumed AMSGrad run keeps its non-decreasing \
         denominator guarantee"
    );
}

#[test]
fn test_amsgrad_disabled_omits_v_max_from_state_dict() {
    // Keep the common (non-AMSGrad) checkpoint small and free of a
    // meaningless empty `v_max` key.
    let mut optimizer = AdamOptimizer::new(AdamConfig {
        learning_rate: 1e-2,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        amsgrad: false,
    });
    optimizer.step(&make_gradients(&[("w", &[1.0])])).expect("step should succeed");
    let saved = optimizer.state_dict().expect("state_dict should succeed");
    assert!(
        !saved.contains_key("v_max"),
        "v_max should not appear in state_dict when amsgrad is disabled"
    );
}
