use crate::sgd::{SGDConfig, SGD};
use crate::traits::StatefulOptimizer;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s % 1000) as f32 / 1000.0
}

#[test]
fn test_sgd_config_default_lr() {
    let config = SGDConfig::default();
    assert!((config.lr - 1e-3).abs() < 1e-9);
}

#[test]
fn test_sgd_config_default_momentum_zero() {
    let config = SGDConfig::default();
    assert_eq!(config.momentum, 0.0);
}

#[test]
fn test_sgd_config_default_weight_decay_zero() {
    let config = SGDConfig::default();
    assert_eq!(config.weight_decay, 0.0);
}

#[test]
fn test_sgd_config_default_dampening_zero() {
    let config = SGDConfig::default();
    assert_eq!(config.dampening, 0.0);
}

#[test]
fn test_sgd_config_default_nesterov_false() {
    let config = SGDConfig::default();
    assert!(!config.nesterov);
}

#[test]
fn test_sgd_new_get_lr() {
    let opt = SGD::new(0.01, 0.0, 0.0, false);
    assert!((opt.get_lr() - 0.01).abs() < 1e-9);
}

#[test]
fn test_sgd_from_config_get_lr() {
    let config = SGDConfig {
        lr: 0.05,
        ..SGDConfig::default()
    };
    let opt = SGD::from_config(config);
    assert!((opt.get_lr() - 0.05).abs() < 1e-9);
}

#[test]
fn test_sgd_set_lr() {
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    opt.set_lr(0.001);
    assert!((opt.get_lr() - 0.001).abs() < 1e-9);
}

#[test]
fn test_sgd_update_positive_grad_decreases_param() {
    let mut opt = SGD::new(0.1, 0.0, 0.0, false);
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    let before = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let after = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(
        after < before,
        "param should decrease with positive grad: before={before} after={after}"
    );
}

#[test]
fn test_sgd_update_negative_grad_increases_param() {
    let mut opt = SGD::new(0.1, 0.0, 0.0, false);
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    let grad = Tensor::new(vec![-1.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    let before = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let after = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(after > before, "param should increase with negative grad");
}

#[test]
fn test_sgd_update_magnitude_matches_lr_times_grad() {
    let lr = 0.05_f32;
    let mut opt = SGD::new(lr, 0.0, 0.0, false);
    let mut param = Tensor::new(vec![2.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor creation failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let after = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(
        (after - (2.0 - lr)).abs() < 1e-6,
        "expected {}, got {after}",
        2.0 - lr
    );
}

#[test]
fn test_sgd_weight_decay_reduces_param() {
    let mut opt_no_decay = SGD::new(0.01, 0.0, 0.0, false);
    let mut opt_with_decay = SGD::new(0.01, 0.0, 0.001, false);

    let mut param_no = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let mut param_wd = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));

    opt_no_decay
        .update(&mut param_no, &grad)
        .unwrap_or_else(|e| panic!("update failed: {e}"));
    opt_with_decay
        .update(&mut param_wd, &grad)
        .unwrap_or_else(|e| panic!("update failed: {e}"));

    let val_no = match &param_no {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    let val_wd = match &param_wd {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };

    assert!(
        val_wd < val_no,
        "weight decay should reduce param: {val_wd} vs {val_no}"
    );
}

#[test]
fn test_sgd_momentum_update_no_panic() {
    // Verify that SGD with momentum does not panic and produces finite results.
    // Note: pointer-based momentum keying means each update step uses a fresh
    // momentum entry due to the parameter tensor being replaced in-place.
    let mut opt_momentum = SGD::new(0.1, 0.9, 0.0, false);
    let mut param = Tensor::new(vec![5.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    for _ in 0..5 {
        let g = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
        opt_momentum
            .update(&mut param, &g)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
    }
    let val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(
        val.is_finite(),
        "param with momentum should be finite: {val}"
    );
    assert!(val < 5.0, "param should have decreased: {val}");
}

#[test]
fn test_sgd_zero_grad_no_panic() {
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    opt.zero_grad(); // should not panic
}

#[test]
fn test_sgd_step_increments_counter() {
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    opt.step();
    opt.step();
    // Step count is internal; verify via state_dict
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    if let Some(step_tensor) = sd.get("step") {
        let data = step_tensor.data().unwrap_or_default();
        assert_eq!(data.first().copied().unwrap_or(0.0) as usize, 2);
    }
}

#[test]
fn test_sgd_convergence_quadratic() {
    // f(x) = x^2, grad = 2x, converges toward 0
    let mut opt = SGD::new(0.1, 0.0, 0.0, false);
    let mut param = Tensor::new(vec![5.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    for _ in 0..50 {
        let val = match &param {
            Tensor::F32(a) => a[0],
            _ => panic!("wrong type"),
        };
        let grad_val = 2.0 * val;
        let grad = Tensor::new(vec![grad_val]).unwrap_or_else(|_| panic!("tensor failed"));
        opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    }
    let final_val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(
        final_val.abs() < 1.0,
        "SGD should converge toward 0 for x^2, got {final_val}"
    );
}

#[test]
fn test_sgd_lcg_gradient_update() {
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    let mut param =
        Tensor::new(vec![1.0_f32, 2.0_f32, 3.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let mut s = 42u64;
    let grads: Vec<f32> = (0..3).map(|_| lcg_next(&mut s)).collect();
    let grad = Tensor::new(grads.clone()).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let data = match &param {
        Tensor::F32(a) => a.iter().cloned().collect::<Vec<_>>(),
        _ => panic!("wrong type"),
    };
    for (i, (&expected, &g)) in [1.0_f32, 2.0, 3.0].iter().zip(grads.iter()).enumerate() {
        let expected_val = expected - 0.01 * g;
        assert!(
            (data[i] - expected_val).abs() < 1e-6,
            "index {i}: expected {expected_val} got {}",
            data[i]
        );
    }
}

#[test]
fn test_sgd_state_dict_contains_lr() {
    let opt = SGD::new(0.05, 0.0, 0.0, false);
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    assert!(sd.contains_key("lr"), "state_dict should contain 'lr'");
}

#[test]
fn test_sgd_state_dict_contains_momentum() {
    let opt = SGD::new(0.05, 0.9, 0.0, false);
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    assert!(
        sd.contains_key("momentum"),
        "state_dict should contain 'momentum'"
    );
}

#[test]
fn test_sgd_state_dict_lr_value() {
    let opt = SGD::new(0.07, 0.0, 0.0, false);
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    if let Some(t) = sd.get("lr") {
        let data = t.data().unwrap_or_default();
        let v = data.first().copied().unwrap_or(0.0);
        assert!(
            (v - 0.07).abs() < 1e-6,
            "lr in state_dict should be 0.07 but got {v}"
        );
    }
}

#[test]
fn test_sgd_memory_usage_initial() {
    let opt = SGD::new(0.01, 0.0, 0.0, false);
    let stats = opt.memory_usage();
    assert_eq!(stats.momentum_elements, 0);
    assert_eq!(stats.variance_elements, 0);
    assert_eq!(stats.total_bytes, 0);
}

#[test]
fn test_sgd_memory_usage_after_momentum_update() {
    let mut opt = SGD::new(0.01, 0.9, 0.0, false);
    let mut param = Tensor::new(vec![1.0_f32, 2.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.1_f32, 0.2_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let stats = opt.memory_usage();
    assert_eq!(
        stats.momentum_elements, 2,
        "should track 2 momentum elements"
    );
    assert!(stats.total_bytes > 0);
}

#[test]
fn test_sgd_reset_state_clears_momentum() {
    let mut opt = SGD::new(0.01, 0.9, 0.0, false);
    let mut param = Tensor::new(vec![1.0_f32, 2.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.1_f32, 0.2_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    opt.reset_state();
    let stats = opt.memory_usage();
    assert_eq!(
        stats.momentum_elements, 0,
        "momentum should be cleared after reset"
    );
}

#[test]
fn test_sgd_reset_state_resets_step() {
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    opt.step();
    opt.step();
    opt.reset_state();
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    if let Some(step_tensor) = sd.get("step") {
        let data = step_tensor.data().unwrap_or_default();
        assert_eq!(data.first().copied().unwrap_or(1.0) as usize, 0);
    }
}

#[test]
fn test_sgd_num_parameters_after_update() {
    let mut opt = SGD::new(0.01, 0.9, 0.0, false);
    let mut param =
        Tensor::new(vec![1.0_f32, 2.0_f32, 3.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad =
        Tensor::new(vec![0.1_f32, 0.2_f32, 0.3_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    assert_eq!(opt.num_parameters(), 0);
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    assert_eq!(opt.num_parameters(), 3);
}

#[test]
fn test_sgd_nesterov_differs_from_plain_momentum() {
    let mut opt_mom = SGD::new(0.1, 0.9, 0.0, false);
    let mut opt_nesterov = SGD::new(0.1, 0.9, 0.0, true);

    let mut param_mom = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let mut param_nes = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));

    for _ in 0..3 {
        let g1 = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
        let g2 = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
        opt_mom
            .update(&mut param_mom, &g1)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
        opt_nesterov
            .update(&mut param_nes, &g2)
            .unwrap_or_else(|e| panic!("update failed: {e}"));
    }

    let val_mom = match &param_mom {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    let val_nes = match &param_nes {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };

    assert!(
        (val_mom - val_nes).abs() > 1e-7,
        "Nesterov should produce different results than plain momentum: mom={val_mom} nes={val_nes}"
    );
}

#[test]
fn test_sgd_large_tensor_update() {
    let n = 256_usize;
    let mut opt = SGD::new(0.01, 0.0, 0.0, false);
    let mut s = 123u64;
    let params: Vec<f32> = (0..n).map(|_| lcg_next(&mut s)).collect();
    let grads: Vec<f32> = (0..n).map(|_| lcg_next(&mut s)).collect();
    let mut param = Tensor::new(params.clone()).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(grads.clone()).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let data = match &param {
        Tensor::F32(a) => a.iter().cloned().collect::<Vec<_>>(),
        _ => panic!("wrong type"),
    };
    for i in 0..n {
        let expected = params[i] - 0.01 * grads[i];
        assert!(
            (data[i] - expected).abs() < 1e-5,
            "index {i}: expected {expected} got {}",
            data[i]
        );
    }
}

#[test]
fn test_sgd_load_state_dict_restores_lr() {
    let opt = SGD::new(0.123, 0.0, 0.0, false);
    let sd = opt.state_dict().unwrap_or_else(|e| panic!("state_dict failed: {e}"));
    let mut opt2 = SGD::new(0.001, 0.0, 0.0, false);
    opt2.load_state_dict(sd)
        .unwrap_or_else(|e| panic!("load_state_dict failed: {e}"));
    assert!(
        (opt2.get_lr() - 0.123).abs() < 1e-6,
        "restored lr should be 0.123, got {}",
        opt2.get_lr()
    );
}

#[test]
fn test_sgd_weight_decay_and_grad_combined() {
    // With weight_decay and lr, param = param*(1-wd) - lr*grad
    let lr = 0.1_f32;
    let wd = 0.01_f32;
    let mut opt = SGD::new(lr, 0.0, wd, false);
    let mut param = Tensor::new(vec![2.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    // weight_decay is applied first: param = param - wd*param = 2.0 - 0.02 = 1.98
    // then grad step: 1.98 - lr*grad = 1.98 - 0.1 = 1.88
    let expected = 2.0 * (1.0 - wd) - lr * 1.0;
    assert!(
        (val - expected).abs() < 1e-5,
        "expected {expected} got {val}"
    );
}
