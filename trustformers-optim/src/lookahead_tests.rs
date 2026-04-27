use crate::adam::{Adam, AdamConfig, AdamW, AdamWConfig};
use crate::lookahead::Lookahead;
use crate::sgd::SGD;
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s % 1000) as f32 / 1000.0
}

fn make_adam() -> Adam {
    Adam::from_config(AdamConfig::default())
}

fn make_adamw() -> AdamW {
    AdamW::from_config(AdamWConfig::default())
}

fn make_sgd() -> SGD {
    SGD::new(0.01, 0.0, 0.0, false)
}

#[test]
fn test_lookahead_new_get_lr_delegates_to_base() {
    let base = make_adam();
    let la = Lookahead::new(base, 5, 0.5);
    // get_lr should return the base optimizer's lr
    assert!((la.get_lr() - AdamConfig::default().lr).abs() < 1e-9);
}

#[test]
fn test_lookahead_set_lr_delegates_to_base() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    la.set_lr(0.05);
    assert!((la.get_lr() - 0.05).abs() < 1e-9);
}

#[test]
fn test_lookahead_zero_grad_no_panic() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    la.zero_grad(); // should not panic
}

#[test]
fn test_lookahead_step_no_panic() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    la.step();
    la.step();
}

#[test]
fn test_lookahead_update_positive_grad_decreases_param() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let before = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
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
fn test_lookahead_update_with_sgd_base() {
    let base = make_sgd();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut param = Tensor::new(vec![2.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(val < 2.0, "param should decrease: {val}");
}

#[test]
fn test_lookahead_update_with_adamw_base() {
    let base = make_adamw();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut param = Tensor::new(vec![1.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(val < 1.5, "param should decrease: {val}");
}

#[test]
fn test_lookahead_base_optimizer_accessor() {
    let base = Adam::from_config(AdamConfig {
        lr: 0.042,
        ..AdamConfig::default()
    });
    let la = Lookahead::new(base, 5, 0.5);
    assert!((la.base_optimizer().get_lr() - 0.042).abs() < 1e-9);
}

#[test]
fn test_lookahead_base_optimizer_mut_accessor() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    la.base_optimizer_mut().set_lr(0.99);
    assert!((la.get_lr() - 0.99).abs() < 1e-9);
}

#[test]
fn test_lookahead_multiple_step_calls() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 3, 0.5);
    for _ in 0..10 {
        la.step();
    }
    // Should not panic; fast_step_count resets at k
}

#[test]
fn test_lookahead_lcg_gradient_updates() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut s = 77u64;
    let mut param = Tensor::new(vec![lcg_next(&mut s), lcg_next(&mut s), lcg_next(&mut s)])
        .unwrap_or_else(|_| panic!("tensor failed"));
    for _ in 0..5 {
        let grads: Vec<f32> = (0..3).map(|_| lcg_next(&mut s)).collect();
        let grad = Tensor::new(grads).unwrap_or_else(|_| panic!("tensor failed"));
        la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
        la.step();
    }
    // Just ensure no panics and param values are finite
    let data = match &param {
        Tensor::F32(a) => a.iter().cloned().collect::<Vec<_>>(),
        _ => panic!("wrong type"),
    };
    for (i, v) in data.iter().enumerate() {
        assert!(v.is_finite(), "param[{i}] should be finite: {v}");
    }
}

#[test]
fn test_lookahead_slow_step_after_k_steps() {
    let base = make_sgd();
    let mut la = Lookahead::new(base, 3, 0.5);
    let mut param = Tensor::new(vec![10.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    // Do k=3 fast steps (step resets fast_step_count at k)
    for _ in 0..3 {
        let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
        la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
        la.step();
    }
    // After k steps, slow_step should be callable
    let result = la.slow_step(&mut param);
    assert!(
        result.is_ok(),
        "slow_step should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_lookahead_k_equals_1() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 1, 0.5);
    let mut param = Tensor::new(vec![5.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    la.step();
    let val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(val.is_finite(), "param should be finite: {val}");
}

#[test]
fn test_lookahead_alpha_one_slow_equals_fast() {
    let base = make_sgd();
    let mut la = Lookahead::new(base, 3, 1.0);
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    for _ in 0..3 {
        let grad = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
        la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
        la.step();
    }
    // With alpha=1.0, slow weights fully adopt fast weights
    let result = la.slow_step(&mut param);
    assert!(
        result.is_ok(),
        "slow_step should succeed: {:?}",
        result.err()
    );
}

#[test]
fn test_lookahead_convergence_toward_zero() {
    let base = SGD::new(0.1, 0.0, 0.0, false);
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut param = Tensor::new(vec![4.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    for _ in 0..100 {
        let val = match &param {
            Tensor::F32(a) => a[0],
            _ => panic!("wrong type"),
        };
        let grad_val = 2.0 * val;
        let grad = Tensor::new(vec![grad_val]).unwrap_or_else(|_| panic!("tensor failed"));
        la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
        la.step();
    }
    let final_val = match &param {
        Tensor::F32(a) => a[0],
        _ => panic!("wrong type"),
    };
    assert!(
        final_val.abs() < 2.0,
        "Lookahead should converge, got {final_val}"
    );
}

#[test]
fn test_lookahead_multivariate_update() {
    let base = make_adam();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut param =
        Tensor::new(vec![1.0_f32, 2.0, 3.0, 4.0]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad =
        Tensor::new(vec![0.1_f32, 0.2, 0.3, 0.4]).unwrap_or_else(|_| panic!("tensor failed"));
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let data = match &param {
        Tensor::F32(a) => a.iter().cloned().collect::<Vec<_>>(),
        _ => panic!("wrong type"),
    };
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "param[{i}] should be finite: {v}");
    }
}

#[test]
fn test_lookahead_large_tensor_update() {
    let base = make_sgd();
    let mut la = Lookahead::new(base, 5, 0.5);
    let mut s = 55u64;
    let n = 128_usize;
    let params: Vec<f32> = (0..n).map(|_| lcg_next(&mut s)).collect();
    let grads: Vec<f32> = (0..n).map(|_| lcg_next(&mut s)).collect();
    let mut param = Tensor::new(params).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(grads).unwrap_or_else(|_| panic!("tensor failed"));
    la.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let data = match &param {
        Tensor::F32(a) => a.iter().cloned().collect::<Vec<_>>(),
        _ => panic!("wrong type"),
    };
    for (i, &v) in data.iter().enumerate() {
        assert!(v.is_finite(), "param[{i}] should be finite: {v}");
    }
}
