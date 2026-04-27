use crate::pde_aware::{PDEAwareConfig, PDEAwareOptimizer};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

fn lcg_next(s: &mut u64) -> f32 {
    *s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    (*s % 1000) as f32 / 1000.0
}

#[test]
fn test_pde_config_default_lr() {
    let config = PDEAwareConfig::default();
    assert!((config.learning_rate - 1e-3).abs() < 1e-9);
}

#[test]
fn test_pde_config_default_beta1() {
    let config = PDEAwareConfig::default();
    assert!((config.beta1 - 0.9).abs() < 1e-9);
}

#[test]
fn test_pde_config_default_beta2() {
    let config = PDEAwareConfig::default();
    assert!((config.beta2 - 0.999).abs() < 1e-9);
}

#[test]
fn test_pde_config_default_epsilon() {
    let config = PDEAwareConfig::default();
    assert!((config.epsilon - 1e-8).abs() < 1e-15);
}

#[test]
fn test_pde_config_default_weight_decay_zero() {
    let config = PDEAwareConfig::default();
    assert_eq!(config.weight_decay, 0.0);
}

#[test]
fn test_pde_config_default_residual_variance_weight() {
    let config = PDEAwareConfig::default();
    assert!((config.residual_variance_weight - 0.1).abs() < 1e-9);
}

#[test]
fn test_pde_config_default_smoothing_factor() {
    let config = PDEAwareConfig::default();
    assert!((config.smoothing_factor - 0.95).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_new_initial_step() {
    let opt = PDEAwareOptimizer::new();
    assert_eq!(opt.step, 0);
}

#[test]
fn test_pde_optimizer_new_empty_history() {
    let opt = PDEAwareOptimizer::new();
    assert!(opt.residual_variance_history.is_empty());
    assert!(opt.gradient_alignment_history.is_empty());
}

#[test]
fn test_pde_optimizer_get_lr() {
    let opt = PDEAwareOptimizer::new();
    assert!((opt.get_lr() - 1e-3).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_set_lr() {
    let mut opt = PDEAwareOptimizer::new();
    opt.set_lr(0.05);
    assert!((opt.get_lr() - 0.05).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_from_config() {
    let config = PDEAwareConfig {
        learning_rate: 0.005,
        beta1: 0.95,
        beta2: 0.995,
        epsilon: 1e-9,
        weight_decay: 1e-5,
        residual_variance_weight: 0.2,
        gradient_alignment_factor: 0.1,
        smoothing_factor: 0.98,
        sharp_gradient_threshold: 1.5,
    };
    let opt = PDEAwareOptimizer::from_config(config);
    assert!((opt.learning_rate - 0.005).abs() < 1e-9);
    assert!((opt.beta1 - 0.95).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_update_decreases_param() {
    let mut opt = PDEAwareOptimizer::new();
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
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
        "positive grad should decrease param: before={before} after={after}"
    );
}

#[test]
fn test_pde_optimizer_update_increments_step() {
    let mut opt = PDEAwareOptimizer::new();
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    assert_eq!(opt.step, 1);
}

#[test]
fn test_pde_optimizer_update_records_variance_history() {
    let mut opt = PDEAwareOptimizer::new();
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.5_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    assert!(
        !opt.residual_variance_history.is_empty(),
        "variance history should be non-empty after update"
    );
}

#[test]
fn test_pde_optimizer_zero_grad_no_panic() {
    let mut opt = PDEAwareOptimizer::new();
    opt.zero_grad(); // should not panic
}

#[test]
fn test_pde_optimizer_step_no_panic() {
    let mut opt = PDEAwareOptimizer::new();
    opt.step(); // no-op but should not panic
}

#[test]
fn test_pde_optimizer_get_pde_stats_initial() {
    let opt = PDEAwareOptimizer::new();
    let stats = opt.get_pde_stats();
    assert_eq!(stats.step, 0);
    assert_eq!(stats.average_residual_variance, 0.0);
    assert_eq!(stats.parameters_tracked, 0);
}

#[test]
fn test_pde_optimizer_get_pde_stats_after_update() {
    let mut opt = PDEAwareOptimizer::new();
    let mut param = Tensor::new(vec![1.0_f32, 2.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(vec![0.3_f32, 0.4_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    opt.update(&mut param, &grad).unwrap_or_else(|e| panic!("update failed: {e}"));
    let stats = opt.get_pde_stats();
    assert_eq!(stats.step, 1);
    assert_eq!(stats.parameters_tracked, 1);
}

#[test]
fn test_pde_optimizer_burgers_equation_lr() {
    let opt = PDEAwareOptimizer::for_burgers_equation();
    assert!((opt.learning_rate - 5e-4).abs() < 1e-10);
}

#[test]
fn test_pde_optimizer_burgers_equation_beta1() {
    let opt = PDEAwareOptimizer::for_burgers_equation();
    assert!((opt.beta1 - 0.95).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_burgers_equation_smoothing() {
    let opt = PDEAwareOptimizer::for_burgers_equation();
    assert!((opt.smoothing_factor - 0.98).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_allen_cahn_lr() {
    let opt = PDEAwareOptimizer::for_allen_cahn();
    assert!((opt.learning_rate - 1e-3).abs() < 1e-10);
}

#[test]
fn test_pde_optimizer_allen_cahn_residual_variance_weight() {
    let opt = PDEAwareOptimizer::for_allen_cahn();
    assert!((opt.residual_variance_weight - 0.2).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_kdv_equation_lr() {
    let opt = PDEAwareOptimizer::for_kdv_equation();
    assert!((opt.learning_rate - 2e-4).abs() < 1e-10);
}

#[test]
fn test_pde_optimizer_kdv_equation_weight_decay_zero() {
    let opt = PDEAwareOptimizer::for_kdv_equation();
    assert_eq!(opt.weight_decay, 0.0);
}

#[test]
fn test_pde_optimizer_kdv_residual_variance_weight() {
    let opt = PDEAwareOptimizer::for_kdv_equation();
    assert!((opt.residual_variance_weight - 0.25).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_sharp_gradients_lr() {
    let opt = PDEAwareOptimizer::for_sharp_gradients();
    assert!((opt.learning_rate - 1e-4).abs() < 1e-11);
}

#[test]
fn test_pde_optimizer_sharp_gradients_threshold() {
    let opt = PDEAwareOptimizer::for_sharp_gradients();
    assert!((opt.sharp_gradient_threshold - 0.3).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_adaptive_lr_high_variance() {
    let opt = PDEAwareOptimizer::new();
    let base_lr = 1e-3;
    let adaptive_lr = opt.adaptive_learning_rate(base_lr, 10.0, false);
    assert!(
        adaptive_lr < base_lr,
        "high variance should reduce lr: {adaptive_lr} vs {base_lr}"
    );
}

#[test]
fn test_pde_optimizer_adaptive_lr_sharp_region() {
    let opt = PDEAwareOptimizer::new();
    let base_lr = 1e-3;
    let adaptive_sharp = opt.adaptive_learning_rate(base_lr, 0.0, true);
    let adaptive_normal = opt.adaptive_learning_rate(base_lr, 0.0, false);
    assert!(
        adaptive_sharp < adaptive_normal,
        "sharp region should reduce lr further"
    );
}

#[test]
fn test_pde_optimizer_adaptive_lr_clamp_lower() {
    let opt = PDEAwareOptimizer::new();
    let base_lr = 1e-3;
    // Very high variance should be clamped at base_lr * 0.01
    let adaptive_lr = opt.adaptive_learning_rate(base_lr, 1e6, false);
    assert!(
        adaptive_lr >= base_lr * 0.01 - 1e-12,
        "lr should be clamped at base*0.01: {adaptive_lr}"
    );
}

#[test]
fn test_pde_optimizer_lcg_gradient_update() {
    let mut opt = PDEAwareOptimizer::new();
    let mut s = 99u64;
    let params: Vec<f32> = (0..4).map(|_| lcg_next(&mut s)).collect();
    let grads: Vec<f32> = (0..4).map(|_| lcg_next(&mut s)).collect();
    let mut param = Tensor::new(params.clone()).unwrap_or_else(|_| panic!("tensor failed"));
    let grad = Tensor::new(grads).unwrap_or_else(|_| panic!("tensor failed"));
    let result = opt.update(&mut param, &grad);
    assert!(result.is_ok(), "update should succeed: {:?}", result.err());
}

#[test]
fn test_pde_optimizer_multiple_updates_track_variance() {
    let mut opt = PDEAwareOptimizer::new();
    let mut param = Tensor::new(vec![1.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    for i in 0..5 {
        let g_val = (i + 1) as f32 * 0.1;
        let grad = Tensor::new(vec![g_val]).unwrap_or_else(|_| panic!("tensor failed"));
        opt.update(&mut param, &grad)
            .unwrap_or_else(|e| panic!("update failed at step {i}: {e}"));
    }
    assert_eq!(opt.residual_variance_history.len(), 5);
}

#[test]
fn test_pde_optimizer_default_trait() {
    let opt = PDEAwareOptimizer::default();
    assert_eq!(opt.step, 0);
    assert!((opt.learning_rate - 1e-3).abs() < 1e-9);
}

#[test]
fn test_pde_optimizer_convergence_quadratic() {
    // PDE optimizer uses adaptive LR that may reduce step size in sharp gradient regions.
    // Use higher base LR to ensure convergence progress is measurable.
    let config = PDEAwareConfig {
        learning_rate: 0.1,
        sharp_gradient_threshold: 100.0, // high threshold to avoid LR reduction
        ..PDEAwareConfig::default()
    };
    let mut opt = PDEAwareOptimizer::from_config(config);
    let mut param = Tensor::new(vec![3.0_f32]).unwrap_or_else(|_| panic!("tensor failed"));
    let initial_val = 3.0_f32;
    for _ in 0..200 {
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
        final_val.abs() < initial_val,
        "PDE optimizer should make progress toward 0 for x^2, initial={initial_val} final={final_val}"
    );
}
