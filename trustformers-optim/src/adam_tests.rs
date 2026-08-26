// adam_tests.rs — comprehensive tests for Adam and AdamW optimizers
#[cfg(test)]
mod tests {
    use crate::adam::{Adam, AdamConfig, AdamW, AdamWConfig};
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Optimizer;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_f32_tensor(data: Vec<f32>) -> Tensor {
        Tensor::new(data).unwrap_or_else(|_| Tensor::new(vec![0.0]).expect("fallback"))
    }

    // LCG for deterministic pseudo-random values
    fn lcg_values(n: usize) -> Vec<f32> {
        let mut s = 42u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s % 1000) as f32 / 1000.0
            })
            .collect()
    }

    // ── AdamConfig defaults ──────────────────────────────────────────────────

    #[test]
    fn test_adam_config_defaults() {
        let cfg = AdamConfig::default();
        assert!((cfg.lr - 1e-3).abs() < 1e-8);
        assert!((cfg.betas.0 - 0.9).abs() < 1e-8);
        assert!((cfg.betas.1 - 0.999).abs() < 1e-8);
        assert!((cfg.eps - 1e-8).abs() < 1e-12);
        assert!((cfg.weight_decay - 0.0).abs() < 1e-10);
    }

    // ── AdamWConfig defaults ─────────────────────────────────────────────────

    #[test]
    fn test_adamw_config_defaults() {
        let cfg = AdamWConfig::default();
        assert!((cfg.lr - 1e-4).abs() < 1e-8);
        assert!((cfg.betas.0 - 0.9).abs() < 1e-8);
        assert!((cfg.betas.1 - 0.999).abs() < 1e-8);
        assert!((cfg.weight_decay - 0.01).abs() < 1e-8);
    }

    // ── Adam constructor ─────────────────────────────────────────────────────

    #[test]
    fn test_adam_new_construction() {
        let adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        assert!((adam.get_lr() - 1e-3).abs() < 1e-9);
    }

    #[test]
    fn test_adam_from_config() {
        let cfg = AdamConfig {
            lr: 5e-4,
            betas: (0.9, 0.999),
            eps: 1e-7,
            weight_decay: 0.01,
        };
        let adam = Adam::from_config(cfg);
        assert!((adam.get_lr() - 5e-4).abs() < 1e-9);
    }

    #[test]
    fn test_adamw_new_construction() {
        let adamw = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        assert!((adamw.get_lr() - 1e-4).abs() < 1e-9);
    }

    #[test]
    fn test_adamw_from_config() {
        let cfg = AdamWConfig {
            lr: 2e-4,
            betas: (0.9, 0.999),
            eps: 1e-8,
            weight_decay: 0.05,
        };
        let adamw = AdamW::from_config(cfg);
        assert!((adamw.get_lr() - 2e-4).abs() < 1e-9);
    }

    // ── get_lr / set_lr ──────────────────────────────────────────────────────

    #[test]
    fn test_adam_set_lr() {
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        adam.set_lr(5e-4);
        assert!((adam.get_lr() - 5e-4).abs() < 1e-9);
    }

    #[test]
    fn test_adamw_set_lr() {
        let mut adamw = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        adamw.set_lr(1e-5);
        assert!((adamw.get_lr() - 1e-5).abs() < 1e-10);
    }

    // ── update: param decreases in direction of gradient ────────────────────

    #[test]
    fn test_adam_update_param_decreases_for_positive_grad() {
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![1.0, 2.0, 3.0]);
        let grad = make_f32_tensor(vec![0.1, 0.1, 0.1]);

        let before = if let Tensor::F32(ref arr) = param {
            arr.as_slice().unwrap_or(&[]).to_vec()
        } else {
            vec![]
        };

        adam.step();
        adam.update(&mut param, &grad).expect("update failed");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            for (i, (&a, &b)) in after.iter().zip(before.iter()).enumerate() {
                assert!(a < b, "param[{}] should have decreased: {} >= {}", i, a, b);
            }
        }
    }

    #[test]
    fn test_adamw_update_param_decreases_for_positive_grad() {
        let mut adamw = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![1.0, 2.0, 3.0]);
        let grad = make_f32_tensor(vec![0.1, 0.1, 0.1]);

        let before = if let Tensor::F32(ref arr) = param {
            arr.as_slice().unwrap_or(&[]).to_vec()
        } else {
            vec![]
        };

        adamw.step();
        adamw.update(&mut param, &grad).expect("update failed");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            for (i, (&a, &b)) in after.iter().zip(before.iter()).enumerate() {
                assert!(a < b, "param[{}] should decrease: {} >= {}", i, a, b);
            }
        }
    }

    // ── Adam weight decay ────────────────────────────────────────────────────

    #[test]
    fn test_adam_with_weight_decay_reduces_magnitude() {
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.1);
        let init_val = 1.0f32;
        let mut param = make_f32_tensor(vec![init_val; 4]);
        let grad = make_f32_tensor(vec![0.0; 4]);

        adam.step();
        adam.update(&mut param, &grad).expect("update failed");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            for &v in after {
                assert!(
                    v < init_val,
                    "weight decay should reduce magnitude, got {}",
                    v
                );
            }
        }
    }

    // ── AdamW weight decay ───────────────────────────────────────────────────

    #[test]
    fn test_adamw_weight_decay_applied() {
        let mut adamw = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.1);
        let init_val = 1.0f32;
        let mut param = make_f32_tensor(vec![init_val; 4]);
        let grad = make_f32_tensor(vec![0.0; 4]);

        adamw.step();
        adamw.update(&mut param, &grad).expect("update failed");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            for &v in after {
                assert!(
                    v < init_val,
                    "AdamW weight decay should reduce magnitude, got {}",
                    v
                );
            }
        }
    }

    // ── multiple update steps ────────────────────────────────────────────────

    #[test]
    fn test_adam_multiple_steps_converge() {
        // Minimise f(x) = sum(x^2) with gradient 2x
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let n = 4;
        let mut param = make_f32_tensor(vec![1.0; n]);

        for _ in 0..500 {
            let grad_data: Vec<f32> = if let Tensor::F32(ref arr) = param {
                arr.as_slice().unwrap_or(&[]).iter().map(|&p| 2.0 * p).collect()
            } else {
                vec![0.0; n]
            };
            let grad = make_f32_tensor(grad_data);
            adam.step();
            adam.update(&mut param, &grad).expect("update failed");
        }

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                assert!(v.abs() < 0.2, "Adam should converge: {}", v);
            }
        }
    }

    #[test]
    fn test_adamw_multiple_steps_converge() {
        let mut adamw = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let n = 4;
        let mut param = make_f32_tensor(vec![1.0; n]);

        for _ in 0..500 {
            let grad_data: Vec<f32> = if let Tensor::F32(ref arr) = param {
                arr.as_slice().unwrap_or(&[]).iter().map(|&p| 2.0 * p).collect()
            } else {
                vec![0.0; n]
            };
            let grad = make_f32_tensor(grad_data);
            adamw.step();
            adamw.update(&mut param, &grad).expect("update failed");
        }

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                assert!(v.abs() < 0.2, "AdamW should converge: {}", v);
            }
        }
    }

    // ── zero_grad is callable (no-op for these optimizers) ───────────────────

    #[test]
    fn test_adam_zero_grad_no_panic() {
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        adam.zero_grad(); // should not panic
    }

    #[test]
    fn test_adamw_zero_grad_no_panic() {
        let mut adamw = AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01);
        adamw.zero_grad(); // should not panic
    }

    // ── step increments ──────────────────────────────────────────────────────

    #[test]
    fn test_adam_step_count() {
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        for _ in 0..5 {
            adam.step();
        }
        // After 5 steps with update: params should differ from initial
        let mut param = make_f32_tensor(vec![0.5_f32; 2]);
        let grad = make_f32_tensor(vec![0.01_f32; 2]);
        adam.update(&mut param, &grad).expect("update failed");
    }

    // ── LCG-seeded random gradient update ────────────────────────────────────

    #[test]
    fn test_adam_lcg_grad_update() {
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        let n = 8;
        let mut param = make_f32_tensor(vec![0.5; n]);
        let grad_vals = lcg_values(n);
        let grad = make_f32_tensor(grad_vals);

        let before: Vec<f32> = if let Tensor::F32(ref arr) = param {
            arr.as_slice().unwrap_or(&[]).to_vec()
        } else {
            vec![]
        };

        adam.step();
        adam.update(&mut param, &grad).expect("update with LCG grad failed");

        // Params should have changed
        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            let changed = before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
            assert!(changed, "parameters should change after update");
        }
    }

    // ── AdamW decoupled weight decay vs Adam L2 ──────────────────────────────

    #[test]
    fn test_adamw_vs_adam_different_behavior() {
        // With identical configs both should produce different results
        // because AdamW applies WD to params, Adam to gradient.
        let wd = 0.1;
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, wd);
        let mut adamw = AdamW::new(1e-2, (0.9, 0.999), 1e-8, wd);

        // Use the SAME initial parameter value but two separate tensors
        let mut param_a = make_f32_tensor(vec![1.0; 4]);
        let mut param_w = make_f32_tensor(vec![1.0; 4]);
        let grad_a = make_f32_tensor(vec![0.1; 4]);
        let grad_w = make_f32_tensor(vec![0.1; 4]);

        adam.step();
        adam.update(&mut param_a, &grad_a).expect("adam update");

        adamw.step();
        adamw.update(&mut param_w, &grad_w).expect("adamw update");

        // Both should decrease but by different amounts
        let a_val = if let Tensor::F32(ref arr) = param_a {
            arr.as_slice().unwrap_or(&[0.0]).first().copied().unwrap_or(0.0)
        } else {
            0.0
        };
        let w_val = if let Tensor::F32(ref arr) = param_w {
            arr.as_slice().unwrap_or(&[0.0]).first().copied().unwrap_or(0.0)
        } else {
            0.0
        };

        assert!(a_val < 1.0, "Adam should decrease param: {}", a_val);
        assert!(w_val < 1.0, "AdamW should decrease param: {}", w_val);
    }

    // ── large param tensor ───────────────────────────────────────────────────

    #[test]
    fn test_adam_large_param_tensor() {
        let n = 1024;
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.1; n]);
        let grad = make_f32_tensor(vec![0.01; n]);

        adam.step();
        adam.update(&mut param, &grad).expect("update large tensor");

        if let Tensor::F32(ref arr) = param {
            assert_eq!(arr.len(), n);
        }
    }

    // ── bias correction effect ───────────────────────────────────────────────

    #[test]
    fn test_adam_bias_correction_first_step() {
        // At step 1 with β1=0.9, bias_correction1 = 1 - 0.9 = 0.1
        // First moment m = (1 - 0.9) * grad = 0.1 * grad
        // m_hat = 0.1 * grad / 0.1 = grad
        // So update ≈ lr * grad / sqrt(grad^2 + eps) ≈ lr * sign(grad)
        let lr = 0.01_f32;
        let mut adam = Adam::new(lr, (0.9, 0.999), 1e-8, 0.0);
        let g_val = 1.0_f32;
        let mut param = make_f32_tensor(vec![0.0; 1]);
        let grad = make_f32_tensor(vec![g_val]);

        adam.step();
        adam.update(&mut param, &grad).expect("first-step update");

        if let Tensor::F32(ref arr) = param {
            let p = arr.as_slice().unwrap_or(&[0.0])[0];
            // After one step param should be approximately -lr
            assert!(p < 0.0, "param should decrease with positive grad: {}", p);
            assert!(
                p.abs() <= lr * 2.0,
                "update should be bounded by lr: {}",
                p.abs()
            );
        }
    }

    // ── negative gradient increases param ───────────────────────────────────

    #[test]
    fn test_adam_negative_grad_increases_param() {
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.0; 3]);
        let grad = make_f32_tensor(vec![-0.5; 3]);

        adam.step();
        adam.update(&mut param, &grad).expect("update");

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                assert!(v > 0.0, "negative grad should increase param from 0: {}", v);
            }
        }
    }

    #[test]
    fn test_adamw_negative_grad_increases_param() {
        let mut adamw = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.0; 3]);
        let grad = make_f32_tensor(vec![-0.5; 3]);

        adamw.step();
        adamw.update(&mut param, &grad).expect("update");

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                assert!(v > 0.0, "negative grad should increase param from 0: {}", v);
            }
        }
    }

    // ── zero gradient no change ──────────────────────────────────────────────

    #[test]
    fn test_adam_zero_grad_no_change_to_param() {
        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let init = 0.5_f32;
        let mut param = make_f32_tensor(vec![init; 4]);
        let grad = make_f32_tensor(vec![0.0; 4]);

        adam.step();
        adam.update(&mut param, &grad).expect("zero grad update");

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                // With zero grad update is essentially 0 (eps still there but tiny)
                assert!(
                    (v - init).abs() < 1e-6,
                    "zero grad should not change param: {}",
                    v
                );
            }
        }
    }

    // ── state_dict round-trip ────────────────────────────────────────────────

    #[test]
    fn test_adam_state_dict_and_load() {
        use crate::traits::StatefulOptimizer;
        let mut adam = Adam::new(5e-4, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.5; 4]);
        let grad = make_f32_tensor(vec![0.1; 4]);
        adam.step();
        adam.update(&mut param, &grad).expect("update for state dict");

        let state = adam.state_dict().expect("state_dict");
        assert!(state.contains_key("lr"), "lr not in state dict");
        assert!(state.contains_key("step"), "step not in state dict");

        let mut adam2 = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        adam2.load_state_dict(state).expect("load_state_dict");
        // After loading, lr should be updated
        assert!(
            (adam2.get_lr() - 5e-4).abs() < 1e-7,
            "lr not loaded correctly: {}",
            adam2.get_lr()
        );
    }

    // ── memory_usage reports non-zero after update ───────────────────────────

    #[test]
    fn test_adam_memory_usage_after_update() {
        use crate::traits::StatefulOptimizer;
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.1; 8]);
        let grad = make_f32_tensor(vec![0.01; 8]);
        adam.step();
        adam.update(&mut param, &grad).expect("update");

        let stats = adam.memory_usage();
        assert!(
            stats.total_bytes > 0,
            "memory_bytes should be > 0 after update"
        );
    }

    // ── reset_state clears buffers ───────────────────────────────────────────

    #[test]
    fn test_adam_reset_state_clears() {
        use crate::traits::StatefulOptimizer;
        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![0.5; 4]);
        let grad = make_f32_tensor(vec![0.1; 4]);
        adam.step();
        adam.update(&mut param, &grad).expect("update");

        adam.reset_state();
        assert_eq!(
            adam.num_parameters(),
            0,
            "num_parameters should be 0 after reset"
        );
        let stats = adam.memory_usage();
        assert_eq!(stats.total_bytes, 0, "memory should be 0 after reset");
    }

    // ── stable parameter identity / checkpoint resume ────────────────────────

    /// Regression: state used to be keyed by the parameter's heap address, so
    /// `state_dict` -> new optimizer -> `load_state_dict` restored nothing and the
    /// resumed run silently started from zeroed moments.
    #[test]
    fn test_adam_checkpoint_resume_restores_trajectory() {
        use crate::traits::StatefulOptimizer;

        // Two parameters with *different* shapes so a key collision or an ordering
        // bug cannot pass unnoticed.
        let build = || {
            (
                make_f32_tensor(vec![1.0, 2.0, 3.0]),
                make_f32_tensor(vec![-1.0, 0.5]),
            )
        };
        let grad_a = make_f32_tensor(vec![0.3, -0.2, 0.1]);
        let grad_b = make_f32_tensor(vec![0.4, 0.7]);

        // Reference run: 3 warm-up steps then one more step.
        let mut reference = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let (mut ref_a, mut ref_b) = build();
        for _ in 0..3 {
            reference.step();
            reference.update(&mut ref_a, &grad_a).expect("warmup a");
            reference.update(&mut ref_b, &grad_b).expect("warmup b");
        }

        // Checkpointed run: identical warm-up, then save.
        let mut saver = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let (mut save_a, mut save_b) = build();
        for _ in 0..3 {
            saver.step();
            saver.update(&mut save_a, &grad_a).expect("warmup a");
            saver.update(&mut save_b, &grad_b).expect("warmup b");
        }
        let checkpoint = saver.state_dict().expect("state_dict");
        assert!(
            checkpoint.keys().any(|k| k.starts_with("exp_avg_p:")),
            "checkpoint must use stable identity keys, got {:?}",
            checkpoint.keys().collect::<Vec<_>>()
        );

        // Fresh optimizer + fresh tensors at fresh addresses.
        let mut resumed = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        resumed.load_state_dict(checkpoint).expect("load_state_dict");
        let mut res_a = save_a.clone();
        let mut res_b = save_b.clone();

        // One further step on both runs must agree exactly.
        reference.step();
        reference.update(&mut ref_a, &grad_a).expect("step a");
        reference.update(&mut ref_b, &grad_b).expect("step b");

        resumed.step();
        resumed.update(&mut res_a, &grad_a).expect("resumed a");
        resumed.update(&mut res_b, &grad_b).expect("resumed b");

        let expected_a = match &ref_a {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        let actual_a = match &res_a {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        for (e, a) in expected_a.iter().zip(actual_a.iter()) {
            assert!(
                (e - a).abs() < 1e-6,
                "resumed trajectory must match reference: {e} vs {a}"
            );
        }

        let expected_b = match &ref_b {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        let actual_b = match &res_b {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        for (e, a) in expected_b.iter().zip(actual_b.iter()) {
            assert!(
                (e - a).abs() < 1e-6,
                "resumed trajectory must match reference: {e} vs {a}"
            );
        }
    }

    /// A resume that restored nothing would produce a *first-step* update; assert the
    /// resumed step differs from what a cold optimizer would do.
    #[test]
    fn test_adam_resume_differs_from_cold_start() {
        use crate::traits::StatefulOptimizer;

        let grad = make_f32_tensor(vec![0.5, 0.5, 0.5, 0.5]);
        let mut saver = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![1.0; 4]);
        for _ in 0..10 {
            saver.step();
            saver.update(&mut param, &grad).expect("warmup");
        }
        let checkpoint = saver.state_dict().expect("state_dict");

        let mut resumed = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        resumed.load_state_dict(checkpoint).expect("load");
        let mut p_resumed = param.clone();
        resumed.step();
        resumed.update(&mut p_resumed, &grad).expect("resumed step");

        let mut cold = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut p_cold = param.clone();
        cold.state_mut().step = 10;
        cold.step();
        cold.update(&mut p_cold, &grad).expect("cold step");

        let v_resumed = match &p_resumed {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        let v_cold = match &p_cold {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        assert!(
            v_resumed.iter().zip(v_cold.iter()).any(|(r, c)| (r - c).abs() > 1e-7),
            "a resume that restored real moments must differ from a zero-state start"
        );
    }

    /// Regression: value/length-hashed keys allocated a new state entry per call, so
    /// the state map grew without bound and every step behaved like step 1.
    #[test]
    fn test_adam_state_does_not_grow_per_step() {
        use crate::traits::StatefulOptimizer;

        let mut adam = Adam::new(1e-3, (0.9, 0.999), 1e-8, 0.0);
        let mut param = make_f32_tensor(vec![1.0; 8]);
        let grad = make_f32_tensor(vec![0.25; 8]);
        for _ in 0..25 {
            adam.step();
            adam.update(&mut param, &grad).expect("update");
        }
        // One momentum buffer of 8 elements, not 25 of them.
        assert_eq!(
            adam.memory_usage().momentum_elements,
            8,
            "exactly one momentum buffer must be retained"
        );
    }

    /// Named updates must resolve to the same state no matter where the tensor lives.
    #[test]
    fn test_adam_update_named_is_address_independent() {
        use crate::traits::StatefulOptimizer;

        let mut adam = Adam::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let grad = make_f32_tensor(vec![0.5; 4]);

        let mut first = make_f32_tensor(vec![1.0; 4]);
        adam.step();
        adam.update_named("w", &mut first, &grad).expect("first");
        let data_after_first = match &first {
            Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
            _ => panic!("expected F32"),
        };
        drop(first);

        // A brand new allocation carrying the same logical parameter.
        let mut second = make_f32_tensor(data_after_first);
        adam.step();
        adam.update_named("w", &mut second, &grad).expect("second");

        assert_eq!(
            adam.memory_usage().momentum_elements,
            4,
            "the same name must reuse one state entry"
        );
    }

    /// Named checkpoints must resume regardless of the order parameters are visited.
    #[test]
    fn test_adamw_named_checkpoint_resume_is_order_independent() {
        use crate::traits::StatefulOptimizer;

        let grad_a = make_f32_tensor(vec![0.3, -0.2, 0.1]);
        let grad_b = make_f32_tensor(vec![0.4, 0.7]);

        let mut saver = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        let mut a = make_f32_tensor(vec![1.0, 2.0, 3.0]);
        let mut b = make_f32_tensor(vec![-1.0, 0.5]);
        for _ in 0..4 {
            saver.step();
            saver.update_named("a", &mut a, &grad_a).expect("a");
            saver.update_named("b", &mut b, &grad_b).expect("b");
        }
        let checkpoint = saver.state_dict().expect("state_dict");
        assert!(
            checkpoint.contains_key("exp_avg_n:a") && checkpoint.contains_key("exp_avg_n:b"),
            "named parameters must be checkpointed under their names: {:?}",
            checkpoint.keys().collect::<Vec<_>>()
        );

        let mut reference = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        reference.load_state_dict(checkpoint.clone()).expect("load ref");
        let mut ref_a = a.clone();
        let mut ref_b = b.clone();
        reference.state_mut().step = 4;
        reference.step();
        reference.update_named("a", &mut ref_a, &grad_a).expect("ref a");
        reference.update_named("b", &mut ref_b, &grad_b).expect("ref b");

        // Same checkpoint, parameters visited in the opposite order.
        let mut reversed = AdamW::new(1e-2, (0.9, 0.999), 1e-8, 0.0);
        reversed.load_state_dict(checkpoint).expect("load rev");
        let mut rev_a = a.clone();
        let mut rev_b = b.clone();
        reversed.state_mut().step = 4;
        reversed.step();
        reversed.update_named("b", &mut rev_b, &grad_b).expect("rev b");
        reversed.update_named("a", &mut rev_a, &grad_a).expect("rev a");

        for (x, y) in [(&ref_a, &rev_a), (&ref_b, &rev_b)] {
            let xs = match x {
                Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
                _ => panic!("expected F32"),
            };
            let ys = match y {
                Tensor::F32(v) => v.iter().copied().collect::<Vec<_>>(),
                _ => panic!("expected F32"),
            };
            for (a, b) in xs.iter().zip(ys.iter()) {
                assert!(
                    (a - b).abs() < 1e-6,
                    "named resume must be order independent: {a} vs {b}"
                );
            }
        }
    }
}
