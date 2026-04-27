// hardware_aware_tests.rs — comprehensive tests for hardware-aware optimizers
#[cfg(test)]
mod tests {
    use crate::adam::AdamW;
    use crate::hardware_aware::{
        create_edge_optimizer, create_gpu_adam, create_mobile_optimizer, create_tpu_optimizer,
        CompressionRatio, EdgeOptimizer, GPUAdam, HardwareAwareConfig, HardwareTarget,
        MobileOptimizer, TPUOptimizer, TPUVersion,
    };
    use crate::sgd::SGD;
    use trustformers_core::tensor::Tensor;
    use trustformers_core::traits::Optimizer;

    // ── helpers ───────────────────────────────────────────────────────────────

    fn make_tensor(data: Vec<f32>) -> Tensor {
        Tensor::new(data).unwrap_or_else(|_| Tensor::new(vec![0.0_f32]).expect("fallback"))
    }

    fn lcg_vals(n: usize) -> Vec<f32> {
        let mut s = 42u64;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (s % 1000) as f32 / 1000.0
            })
            .collect()
    }

    fn gpu_config(memory_gb: f32, cc: f32) -> HardwareAwareConfig {
        HardwareAwareConfig {
            target: HardwareTarget::GPU {
                memory_gb,
                compute_capability: cc,
                use_tensor_cores: cc >= 7.0,
            },
            base_learning_rate: 1e-3,
            enable_fusion: true,
            memory_efficient: false,
            use_mixed_precision: false,
            gradient_compression: None,
            custom_kernels: false,
        }
    }

    fn mobile_config(memory_mb: usize, latency_ms: f32) -> HardwareAwareConfig {
        HardwareAwareConfig {
            target: HardwareTarget::Mobile {
                memory_mb,
                cpu_cores: 4,
                target_latency_ms: latency_ms,
            },
            base_learning_rate: 1e-3,
            enable_fusion: false,
            memory_efficient: true,
            use_mixed_precision: false,
            gradient_compression: Some(CompressionRatio::Half),
            custom_kernels: false,
        }
    }

    fn edge_config(memory_mb: usize, power_mw: f32) -> HardwareAwareConfig {
        HardwareAwareConfig {
            target: HardwareTarget::Edge {
                memory_mb,
                power_budget_mw: power_mw,
                quantization_bits: 8,
            },
            base_learning_rate: 1e-3,
            enable_fusion: false,
            memory_efficient: true,
            use_mixed_precision: false,
            gradient_compression: Some(CompressionRatio::Eighth),
            custom_kernels: false,
        }
    }

    fn tpu_config(version: TPUVersion, num_cores: usize) -> HardwareAwareConfig {
        HardwareAwareConfig {
            target: HardwareTarget::TPU {
                version,
                num_cores,
                use_bfloat16: true,
            },
            base_learning_rate: 1e-4,
            enable_fusion: true,
            memory_efficient: true,
            use_mixed_precision: true,
            gradient_compression: None,
            custom_kernels: true,
        }
    }

    // ── HardwareTarget enum variants ─────────────────────────────────────────

    #[test]
    fn test_hardware_target_gpu_variant() {
        let t = HardwareTarget::GPU {
            memory_gb: 16.0,
            compute_capability: 8.6,
            use_tensor_cores: true,
        };
        match t {
            HardwareTarget::GPU {
                memory_gb,
                compute_capability,
                use_tensor_cores,
            } => {
                assert!((memory_gb - 16.0).abs() < 1e-6);
                assert!((compute_capability - 8.6).abs() < 1e-4);
                assert!(use_tensor_cores);
            },
            _ => panic!("expected GPU variant"),
        }
    }

    #[test]
    fn test_hardware_target_tpu_variant() {
        let t = HardwareTarget::TPU {
            version: TPUVersion::V4,
            num_cores: 8,
            use_bfloat16: true,
        };
        match t {
            HardwareTarget::TPU {
                version,
                num_cores,
                use_bfloat16,
            } => {
                assert_eq!(version, TPUVersion::V4);
                assert_eq!(num_cores, 8);
                assert!(use_bfloat16);
            },
            _ => panic!("expected TPU variant"),
        }
    }

    #[test]
    fn test_hardware_target_mobile_variant() {
        let t = HardwareTarget::Mobile {
            memory_mb: 512,
            cpu_cores: 6,
            target_latency_ms: 20.0,
        };
        match t {
            HardwareTarget::Mobile {
                memory_mb,
                cpu_cores,
                target_latency_ms,
            } => {
                assert_eq!(memory_mb, 512);
                assert_eq!(cpu_cores, 6);
                assert!((target_latency_ms - 20.0).abs() < 1e-5);
            },
            _ => panic!("expected Mobile variant"),
        }
    }

    #[test]
    fn test_hardware_target_edge_variant() {
        let t = HardwareTarget::Edge {
            memory_mb: 64,
            power_budget_mw: 100.0,
            quantization_bits: 4,
        };
        match t {
            HardwareTarget::Edge {
                memory_mb,
                power_budget_mw,
                quantization_bits,
            } => {
                assert_eq!(memory_mb, 64);
                assert!((power_budget_mw - 100.0).abs() < 1e-5);
                assert_eq!(quantization_bits, 4);
            },
            _ => panic!("expected Edge variant"),
        }
    }

    // ── TPUVersion equality ──────────────────────────────────────────────────

    #[test]
    fn test_tpu_version_equality() {
        assert_eq!(TPUVersion::V2, TPUVersion::V2);
        assert_eq!(TPUVersion::V3, TPUVersion::V3);
        assert_eq!(TPUVersion::V4, TPUVersion::V4);
        assert_eq!(TPUVersion::V5, TPUVersion::V5);
        assert_ne!(TPUVersion::V2, TPUVersion::V5);
    }

    // ── CompressionRatio enum ────────────────────────────────────────────────

    #[test]
    fn test_compression_ratio_variants() {
        let half = CompressionRatio::Half;
        let quarter = CompressionRatio::Quarter;
        let eighth = CompressionRatio::Eighth;
        // Just verify they can be created and pattern-matched
        let label = match half {
            CompressionRatio::Half => "half",
            CompressionRatio::Quarter => "quarter",
            CompressionRatio::Eighth => "eighth",
        };
        assert_eq!(label, "half");
        let _ = quarter;
        let _ = eighth;
    }

    // ── GPUAdam construction ─────────────────────────────────────────────────

    #[test]
    fn test_gpu_adam_new_success() {
        let cfg = gpu_config(8.0, 7.5);
        let result = GPUAdam::new(cfg);
        assert!(
            result.is_ok(),
            "GPUAdam::new should succeed with GPU target"
        );
    }

    #[test]
    fn test_gpu_adam_wrong_target_fails() {
        let cfg = mobile_config(512, 20.0);
        let result = GPUAdam::new(cfg);
        assert!(
            result.is_err(),
            "GPUAdam::new should fail with non-GPU target"
        );
    }

    #[test]
    fn test_gpu_adam_get_lr() {
        let cfg = gpu_config(8.0, 7.0);
        let gpu_adam = GPUAdam::new(cfg).expect("GPUAdam construction");
        let lr = gpu_adam.get_lr();
        assert!(lr > 0.0, "lr should be positive: {}", lr);
    }

    #[test]
    fn test_gpu_adam_set_lr() {
        let cfg = gpu_config(8.0, 7.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam");
        gpu_adam.set_lr(5e-5);
        assert!(
            (gpu_adam.get_lr() - 5e-5).abs() < 1e-10,
            "set_lr did not work"
        );
    }

    #[test]
    fn test_gpu_adam_update_basic() {
        let cfg = gpu_config(8.0, 7.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam");
        let mut param = make_tensor(vec![1.0, 2.0, 3.0]);
        let grad = make_tensor(vec![0.1, 0.1, 0.1]);
        gpu_adam.step();
        let result = gpu_adam.update(&mut param, &grad);
        assert!(result.is_ok(), "GPUAdam::update should succeed");
    }

    #[test]
    fn test_gpu_adam_zero_grad_no_panic() {
        let cfg = gpu_config(8.0, 8.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam");
        gpu_adam.zero_grad();
    }

    // ── TPUOptimizer ─────────────────────────────────────────────────────────

    #[test]
    fn test_tpu_optimizer_new_success() {
        let cfg = tpu_config(TPUVersion::V4, 8);
        let base = Box::new(AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01));
        let result = TPUOptimizer::new(base, cfg);
        assert!(result.is_ok(), "TPUOptimizer::new should succeed");
    }

    #[test]
    fn test_tpu_optimizer_wrong_target_fails() {
        let cfg = gpu_config(8.0, 7.0);
        let base = Box::new(AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01));
        let result = TPUOptimizer::new(base, cfg);
        assert!(
            result.is_err(),
            "TPUOptimizer::new should fail with non-TPU target"
        );
    }

    #[test]
    fn test_tpu_optimizer_update() {
        let cfg = tpu_config(TPUVersion::V3, 4);
        let base = Box::new(AdamW::new(1e-4, (0.9, 0.999), 1e-8, 0.01));
        let mut tpu_opt = TPUOptimizer::new(base, cfg).expect("TPUOptimizer");
        let mut param = make_tensor(vec![1.0; 4]);
        let grad = make_tensor(vec![0.05; 4]);
        tpu_opt.step();
        assert!(tpu_opt.update(&mut param, &grad).is_ok());
    }

    // ── MobileOptimizer ──────────────────────────────────────────────────────

    #[test]
    fn test_mobile_optimizer_new_success() {
        let cfg = mobile_config(512, 20.0);
        let base = Box::new(SGD::new(1e-3, 0.9, 0.0, false));
        let result = MobileOptimizer::new(base, cfg);
        assert!(result.is_ok(), "MobileOptimizer::new should succeed");
    }

    #[test]
    fn test_mobile_optimizer_wrong_target_fails() {
        let cfg = edge_config(64, 50.0);
        let base = Box::new(SGD::new(1e-3, 0.9, 0.0, false));
        let result = MobileOptimizer::new(base, cfg);
        assert!(
            result.is_err(),
            "MobileOptimizer::new should fail with non-Mobile target"
        );
    }

    #[test]
    fn test_mobile_optimizer_get_set_lr() {
        let cfg = mobile_config(256, 15.0);
        let base = Box::new(SGD::new(1e-3, 0.9, 0.0, false));
        let mut mobile_opt = MobileOptimizer::new(base, cfg).expect("MobileOptimizer");
        mobile_opt.set_lr(2e-3);
        assert!((mobile_opt.get_lr() - 2e-3).abs() < 1e-9);
    }

    #[test]
    fn test_mobile_optimizer_update_param_changes() {
        let cfg = mobile_config(512, 20.0);
        let base = Box::new(SGD::new(1e-2, 0.9, 0.0, false));
        let mut mobile_opt = MobileOptimizer::new(base, cfg).expect("MobileOptimizer");

        let mut param = make_tensor(vec![1.0, 2.0, 3.0]);
        let grad = make_tensor(vec![0.1, 0.1, 0.1]);

        let before: Vec<f32> = if let Tensor::F32(ref arr) = param {
            arr.as_slice().unwrap_or(&[]).to_vec()
        } else {
            vec![]
        };

        mobile_opt.step();
        mobile_opt.update(&mut param, &grad).expect("MobileOptimizer update");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            let changed = before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
            assert!(changed, "mobile optimizer should change params");
        }
    }

    // ── EdgeOptimizer ────────────────────────────────────────────────────────

    #[test]
    fn test_edge_optimizer_new_success() {
        let cfg = edge_config(128, 200.0);
        let base = Box::new(SGD::new(1e-3, 0.5, 0.0, false));
        let result = EdgeOptimizer::new(base, cfg);
        assert!(result.is_ok(), "EdgeOptimizer::new should succeed");
    }

    #[test]
    fn test_edge_optimizer_wrong_target_fails() {
        let cfg = mobile_config(512, 20.0);
        let base = Box::new(SGD::new(1e-3, 0.5, 0.0, false));
        let result = EdgeOptimizer::new(base, cfg);
        assert!(
            result.is_err(),
            "EdgeOptimizer::new should fail with non-Edge target"
        );
    }

    #[test]
    fn test_edge_optimizer_update() {
        let cfg = edge_config(64, 100.0);
        let base = Box::new(SGD::new(1e-3, 0.5, 0.0, false));
        let mut edge_opt = EdgeOptimizer::new(base, cfg).expect("EdgeOptimizer");
        let mut param = make_tensor(vec![0.5; 4]);
        let grad = make_tensor(vec![0.05; 4]);
        edge_opt.step();
        assert!(edge_opt.update(&mut param, &grad).is_ok());
    }

    #[test]
    fn test_edge_optimizer_zero_grad() {
        let cfg = edge_config(64, 100.0);
        let base = Box::new(SGD::new(1e-3, 0.5, 0.0, false));
        let mut edge_opt = EdgeOptimizer::new(base, cfg).expect("EdgeOptimizer");
        edge_opt.zero_grad();
    }

    // ── Factory functions ────────────────────────────────────────────────────

    #[test]
    fn test_create_gpu_adam_factory() {
        let result = create_gpu_adam(8.0, 8.6);
        assert!(result.is_ok(), "create_gpu_adam should succeed");
        let gpu_adam = result.expect("gpu_adam");
        assert!(gpu_adam.get_lr() > 0.0);
    }

    #[test]
    fn test_create_gpu_adam_tensor_cores_threshold() {
        // Compute capability < 7.0 → no tensor cores
        let result_old = create_gpu_adam(4.0, 6.5);
        assert!(result_old.is_ok());
        // Compute capability >= 7.0 → tensor cores enabled
        let result_new = create_gpu_adam(16.0, 8.0);
        assert!(result_new.is_ok());
    }

    #[test]
    fn test_create_tpu_optimizer_v2() {
        let result = create_tpu_optimizer(TPUVersion::V2, 8);
        assert!(result.is_ok(), "create_tpu_optimizer V2 should succeed");
    }

    #[test]
    fn test_create_tpu_optimizer_v5() {
        let result = create_tpu_optimizer(TPUVersion::V5, 16);
        assert!(result.is_ok(), "create_tpu_optimizer V5 should succeed");
    }

    #[test]
    fn test_create_mobile_optimizer_factory() {
        let result = create_mobile_optimizer(512, 20.0);
        assert!(result.is_ok(), "create_mobile_optimizer should succeed");
        let opt = result.expect("mobile_opt");
        assert!(opt.get_lr() > 0.0);
    }

    #[test]
    fn test_create_edge_optimizer_factory() {
        let result = create_edge_optimizer(128, 500.0);
        assert!(result.is_ok(), "create_edge_optimizer should succeed");
        let opt = result.expect("edge_opt");
        assert!(opt.get_lr() > 0.0);
    }

    // ── LCG-seeded updates ───────────────────────────────────────────────────

    #[test]
    fn test_gpu_adam_lcg_gradient_update() {
        let cfg = gpu_config(8.0, 7.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam");
        let n = 8;
        let mut param = make_tensor(vec![0.5; n]);
        let grad = make_tensor(lcg_vals(n));

        let before: Vec<f32> = if let Tensor::F32(ref arr) = param {
            arr.as_slice().unwrap_or(&[]).to_vec()
        } else {
            vec![]
        };

        gpu_adam.step();
        gpu_adam.update(&mut param, &grad).expect("update");

        if let Tensor::F32(ref arr) = param {
            let after = arr.as_slice().unwrap_or(&[]);
            let changed = before.iter().zip(after.iter()).any(|(a, b)| (a - b).abs() > 1e-8);
            assert!(changed, "LCG grad update should change params");
        }
    }

    #[test]
    fn test_mobile_optimizer_lcg_gradient_update() {
        let cfg = mobile_config(512, 20.0);
        let base = Box::new(SGD::new(1e-2, 0.9, 0.0, false));
        let mut mobile_opt = MobileOptimizer::new(base, cfg).expect("MobileOptimizer");
        let n = 16;
        let mut param = make_tensor(lcg_vals(n).iter().map(|x| x * 2.0 - 1.0).collect());
        let grad = make_tensor(lcg_vals(n));

        mobile_opt.step();
        assert!(
            mobile_opt.update(&mut param, &grad).is_ok(),
            "LCG mobile update"
        );
    }

    #[test]
    fn test_edge_optimizer_lcg_gradient_update() {
        let cfg = edge_config(64, 100.0);
        let base = Box::new(SGD::new(1e-3, 0.5, 0.0, false));
        let mut edge_opt = EdgeOptimizer::new(base, cfg).expect("EdgeOptimizer");
        let n = 12;
        let mut param = make_tensor(vec![0.1; n]);
        let grad = make_tensor(lcg_vals(n));

        edge_opt.step();
        assert!(
            edge_opt.update(&mut param, &grad).is_ok(),
            "LCG edge update"
        );
    }

    // ── HardwareAwareConfig fields ───────────────────────────────────────────

    #[test]
    fn test_hardware_aware_config_fields() {
        let cfg = HardwareAwareConfig {
            target: HardwareTarget::GPU {
                memory_gb: 24.0,
                compute_capability: 9.0,
                use_tensor_cores: true,
            },
            base_learning_rate: 3e-4,
            enable_fusion: true,
            memory_efficient: true,
            use_mixed_precision: true,
            gradient_compression: Some(CompressionRatio::Half),
            custom_kernels: true,
        };
        assert!((cfg.base_learning_rate - 3e-4).abs() < 1e-10);
        assert!(cfg.enable_fusion);
        assert!(cfg.memory_efficient);
        assert!(cfg.use_mixed_precision);
        assert!(cfg.custom_kernels);
        assert!(cfg.gradient_compression.is_some());
    }

    // ── Multiple step update ─────────────────────────────────────────────────

    #[test]
    fn test_gpu_adam_multiple_step_update() {
        let cfg = gpu_config(8.0, 8.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam");
        let mut param = make_tensor(vec![1.0; 4]);

        for _ in 0..10 {
            let grad_data: Vec<f32> = if let Tensor::F32(ref arr) = param {
                arr.as_slice().unwrap_or(&[]).iter().map(|&p| 2.0 * p).collect()
            } else {
                vec![0.1; 4]
            };
            let grad = make_tensor(grad_data);
            gpu_adam.step();
            gpu_adam.update(&mut param, &grad).expect("multi-step update");
        }

        if let Tensor::F32(ref arr) = param {
            for &v in arr.as_slice().unwrap_or(&[]) {
                assert!(v < 1.0, "param should have decreased over 10 steps: {}", v);
            }
        }
    }

    // ── Optimize for GPU arch ────────────────────────────────────────────────

    #[test]
    fn test_gpu_adam_optimize_for_old_arch() {
        let cfg = gpu_config(4.0, 6.0);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam old arch");
        assert!(gpu_adam.optimize_for_gpu(6.0).is_ok());
    }

    #[test]
    fn test_gpu_adam_optimize_for_turing() {
        let cfg = gpu_config(8.0, 7.5);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam turing");
        assert!(gpu_adam.optimize_for_gpu(7.5).is_ok());
    }

    #[test]
    fn test_gpu_adam_optimize_for_ampere() {
        let cfg = gpu_config(24.0, 8.6);
        let mut gpu_adam = GPUAdam::new(cfg).expect("GPUAdam ampere");
        assert!(gpu_adam.optimize_for_gpu(8.6).is_ok());
    }
}
