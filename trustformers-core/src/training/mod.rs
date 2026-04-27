//! Training utilities for TrustformeRS.
//!
//! This module provides building blocks for training transformer models:
//!
//! - [`gradient_accumulation`]: Simulate large batch sizes by accumulating
//!   gradients over multiple micro-batches with optional clipping and
//!   normalization.
//! - [`mixed_precision`]: AMP-style mixed precision utilities including
//!   BF16/FP16 weight casting, dynamic loss scaling, and FP32 master weights.

pub mod gradient_accumulation;
pub mod mixed_precision;

// Re-export commonly used items at the module level.
pub use gradient_accumulation::{
    clip_grad_norm_, global_grad_norm, GradAccumConfig, GradAccumStats, GradError,
    GradientAccumulator, GradientBuffer,
};
pub use mixed_precision::{
    cast_bf16_to_fp32, cast_fp16_to_fp32, cast_fp32_to_bf16, cast_fp32_to_fp16, AmpStats,
    BFloat16, LossScaler, MixedPrecisionContext, TrainingPrecisionMode,
};

// ---------------------------------------------------------------------------
// Module-level tests: integration across gradient accumulation and mixed
// precision, plus edge-case coverage for functions not tested in their own
// submodules.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // 1. Re-exported types are accessible at the crate training:: level
    // -----------------------------------------------------------------------
    #[test]
    fn test_reexports_are_accessible() {
        // If this compiles, the re-exports are working.
        let _cfg = GradAccumConfig::default();
        let _stats = GradAccumStats::default();
        let _amp = AmpStats::default();
        let _scaler = LossScaler::new();
        let _bf16 = BFloat16::from_f32(1.0);
        let _mode = TrainingPrecisionMode::Fp32;
    }

    // -----------------------------------------------------------------------
    // 2. global_grad_norm with a single zero vector returns 0.0
    // -----------------------------------------------------------------------
    #[test]
    fn test_global_grad_norm_zero() {
        let grads = vec![vec![0.0_f32; 4]];
        let norm = global_grad_norm(&grads);
        assert_eq!(norm, 0.0, "norm of zero gradients should be 0");
    }

    // -----------------------------------------------------------------------
    // 3. global_grad_norm: known value (3-4-5 right triangle)
    // -----------------------------------------------------------------------
    #[test]
    fn test_global_grad_norm_known_value() {
        // [3, 4] split across two parameters: sqrt(9 + 16) = 5
        let grads = vec![vec![3.0_f32], vec![4.0_f32]];
        let norm = global_grad_norm(&grads);
        assert!(
            (norm - 5.0).abs() < 1e-5,
            "expected 5.0, got {norm}"
        );
    }

    // -----------------------------------------------------------------------
    // 4. clip_grad_norm_ with max_norm == current_norm → no change
    // -----------------------------------------------------------------------
    #[test]
    fn test_clip_exact_max_norm() {
        let mut grads = vec![vec![3.0_f32], vec![4.0_f32]]; // norm = 5
        let original = grads.clone();
        clip_grad_norm_(&mut grads, 5.0);
        assert_eq!(grads, original, "grad should not change when norm == max_norm");
    }

    // -----------------------------------------------------------------------
    // 5. GradientBuffer: accumulate fewer than accumulation_steps → not ready
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_buffer_not_ready_before_steps() {
        let config = GradAccumConfig {
            accumulation_steps: 8,
            normalize_by_steps: false,
            clip_grad_norm: None,
            sync_on_last: true,
        };
        let mut buf = GradientBuffer::new(&[4_usize]);
        for _ in 0..7 {
            buf.accumulate(&[vec![1.0_f32; 4]], &config)
                .expect("accumulate");
        }
        assert!(!buf.is_ready, "buffer should not be ready after 7/8 steps");
    }

    // -----------------------------------------------------------------------
    // 6. GradientAccumulator: step() returns None until window is full
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_accumulator_step_returns_none_until_full() {
        let config = GradAccumConfig {
            accumulation_steps: 3,
            normalize_by_steps: true,
            clip_grad_norm: None,
            sync_on_last: true,
        };
        let mut acc = GradientAccumulator::new(config, &[2_usize]);
        let g = vec![vec![1.0_f32, 2.0_f32]];

        let r1 = acc.step(&g).expect("step 1");
        assert!(r1.is_none(), "step 1 should not yield gradients");
        let r2 = acc.step(&g).expect("step 2");
        assert!(r2.is_none(), "step 2 should not yield gradients");
        let r3 = acc.step(&g).expect("step 3");
        assert!(r3.is_some(), "step 3 should yield gradients");
        let grads = r3.expect("some");
        // Normalized by 3 → mean(1.0, 1.0, 1.0) = 1.0
        assert!((grads[0][0] - 1.0).abs() < 1e-5, "expected 1.0, got {}", grads[0][0]);
        assert!((grads[0][1] - 2.0).abs() < 1e-5, "expected 2.0, got {}", grads[0][1]);
    }

    // -----------------------------------------------------------------------
    // 7. GradAccumStats default is all-zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_grad_accum_stats_default_zero() {
        let stats = GradAccumStats::default();
        assert_eq!(stats.total_updates, 0);
        assert_eq!(stats.total_micro_batches, 0);
        assert_eq!(stats.clips_applied, 0);
        assert_eq!(stats.mean_grad_norm_before_clip, 0.0);
        assert_eq!(stats.max_grad_norm_seen, 0.0);
    }

    // -----------------------------------------------------------------------
    // 8. GradAccumStats: max_grad_norm_seen tracks maximum correctly
    // -----------------------------------------------------------------------
    #[test]
    fn test_grad_accum_stats_max_norm() {
        let mut stats = GradAccumStats::default();
        stats.record_update(3.0, false);
        stats.record_update(7.0, true);
        stats.record_update(2.0, false);
        assert!(
            (stats.max_grad_norm_seen - 7.0).abs() < 1e-5,
            "max grad norm should be 7.0, got {}",
            stats.max_grad_norm_seen
        );
    }

    // -----------------------------------------------------------------------
    // 9. GradAccumStats: mean_grad_norm_before_clip Welford running mean
    // -----------------------------------------------------------------------
    #[test]
    fn test_grad_accum_stats_welford_mean() {
        let mut stats = GradAccumStats::default();
        // three norms: 2.0, 4.0, 6.0 → mean = 4.0
        stats.record_update(2.0, false);
        stats.record_update(4.0, false);
        stats.record_update(6.0, false);
        assert!(
            (stats.mean_grad_norm_before_clip - 4.0).abs() < 1e-4,
            "expected mean 4.0, got {}",
            stats.mean_grad_norm_before_clip
        );
    }

    // -----------------------------------------------------------------------
    // 10. BFloat16: round-trip across the full batch conversion path
    // -----------------------------------------------------------------------
    #[test]
    fn test_bf16_batch_round_trip_wide_range() {
        let values: Vec<f32> = vec![
            0.0, 1.0, -1.0, 0.5, -0.5, 2048.0, -2048.0,
            f32::INFINITY, f32::NEG_INFINITY,
        ];
        let bf16_bits = cast_fp32_to_bf16(&values);
        let recovered = cast_bf16_to_fp32(&bf16_bits);

        // Infinity should survive exactly.
        assert!(recovered[7].is_infinite() && recovered[7].is_sign_positive());
        assert!(recovered[8].is_infinite() && recovered[8].is_sign_negative());

        // Finite values: relative error < 1% for representable numbers.
        for (&orig, &rec) in values[..7].iter().zip(recovered[..7].iter()) {
            if orig == 0.0 {
                assert!(rec.abs() < 1e-5, "zero round-trip: {rec}");
            } else {
                let rel = ((rec - orig) / orig).abs();
                assert!(rel < 0.01, "bf16 round-trip {orig} → {rec}, rel={rel}");
            }
        }
    }

    // -----------------------------------------------------------------------
    // 11. LossScaler: scale clamped to minimum 1.0 on excessive overflows
    // -----------------------------------------------------------------------
    #[test]
    fn test_loss_scaler_floor_at_one() {
        let mut scaler = LossScaler::new();
        // Drive scale to a very small value with many overflow steps.
        for _ in 0..100 {
            scaler.update(true);
        }
        assert!(
            scaler.current_scale() >= 1.0,
            "scale must not go below 1.0, got {}",
            scaler.current_scale()
        );
    }

    // -----------------------------------------------------------------------
    // 12. LossScaler: scale_loss × unscale round-trip preserves value
    // -----------------------------------------------------------------------
    #[test]
    fn test_loss_scaler_scale_unscale_round_trip() {
        let scaler = LossScaler::new();
        let loss = 0.25_f32;
        let scaled = scaler.scale_loss(loss);
        // Manually unscale
        let unscaled = scaled / scaler.current_scale();
        assert!(
            (unscaled - loss).abs() < 1e-5,
            "scale-unscale round-trip failed: expected {loss}, got {unscaled}"
        );
    }

    // -----------------------------------------------------------------------
    // 13. MixedPrecisionContext: FP16 mode creates a LossScaler
    // -----------------------------------------------------------------------
    #[test]
    fn test_mixed_precision_context_fp16_creates_scaler() {
        let ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Fp16, &[4_usize]);
        assert!(ctx.loss_scaler.is_some(), "FP16 mode must have a loss scaler");
    }

    // -----------------------------------------------------------------------
    // 14. MixedPrecisionContext: BF16 mode has no LossScaler
    // -----------------------------------------------------------------------
    #[test]
    fn test_mixed_precision_context_bf16_no_scaler() {
        let ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Bf16, &[4_usize]);
        assert!(ctx.loss_scaler.is_none(), "BF16 mode must not have a loss scaler");
    }

    // -----------------------------------------------------------------------
    // 15. MixedPrecisionContext: FP16 compute weights use FP16 encoding
    // -----------------------------------------------------------------------
    #[test]
    fn test_mixed_precision_context_fp16_weight_encoding() {
        let shapes = &[3_usize];
        let mut ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Fp16, shapes);
        ctx.master_weights[0] = vec![1.0_f32, 0.5, 2.0];
        let compute = ctx.get_compute_weights();
        assert_eq!(compute.len(), 1);
        assert_eq!(compute[0].len(), 3);
        // Recover via cast_fp16_to_fp32 and check round-trip.
        let recovered = cast_fp16_to_fp32(&compute[0]);
        for (orig, rec) in ctx.master_weights[0].iter().zip(recovered.iter()) {
            let rel = ((rec - orig) / orig).abs();
            assert!(rel < 0.002, "fp16 weight encoding: {orig} → {rec}");
        }
    }

    // -----------------------------------------------------------------------
    // 16. MixedPrecisionContext: BF16 compute weights use BF16 encoding
    // -----------------------------------------------------------------------
    #[test]
    fn test_mixed_precision_context_bf16_weight_encoding() {
        let shapes = &[3_usize];
        let mut ctx = MixedPrecisionContext::new(TrainingPrecisionMode::Bf16, shapes);
        ctx.master_weights[0] = vec![1.0_f32, 0.5, -3.0];
        let compute = ctx.get_compute_weights();
        let recovered = cast_bf16_to_fp32(&compute[0]);
        for (orig, rec) in ctx.master_weights[0].iter().zip(recovered.iter()) {
            let rel = if orig.abs() > 1e-6 {
                ((rec - orig) / orig).abs()
            } else {
                (rec - orig).abs()
            };
            assert!(rel < 0.01, "bf16 weight encoding: {orig} → {rec}");
        }
    }

    // -----------------------------------------------------------------------
    // 17. GradientAccumulator: total_micro_batches increments on every step
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_accumulator_micro_batch_counter() {
        let config = GradAccumConfig {
            accumulation_steps: 4,
            normalize_by_steps: false,
            clip_grad_norm: None,
            sync_on_last: true,
        };
        let mut acc = GradientAccumulator::new(config, &[2_usize]);
        let g = vec![vec![1.0_f32, 1.0_f32]];
        for i in 1..=6_u64 {
            acc.step(&g).expect("step");
            assert_eq!(acc.stats().total_micro_batches, i);
        }
    }

    // -----------------------------------------------------------------------
    // 18. GradientAccumulator: force_finalize on empty buffer returns None
    // -----------------------------------------------------------------------
    #[test]
    fn test_force_finalize_empty_returns_none() {
        let config = GradAccumConfig::default();
        let mut acc = GradientAccumulator::new(config, &[2_usize]);
        let result = acc.force_finalize().expect("force_finalize ok");
        assert!(result.is_none(), "empty buffer should yield None");
    }

    // -----------------------------------------------------------------------
    // 19. Integration: gradient accumulation + loss scaling interplay
    //
    // Simulate two optimizer steps: scale grads, check overflow, unscale,
    // accumulate, apply.  Verify that the stats are updated correctly.
    // -----------------------------------------------------------------------
    #[test]
    fn test_integration_amp_and_accumulation() {
        let mut scaler = LossScaler::new();
        scaler.growth_interval = 2; // fast growth for the test

        let accum_config = GradAccumConfig {
            accumulation_steps: 2,
            normalize_by_steps: true,
            clip_grad_norm: Some(1.0),
            sync_on_last: true,
        };
        let mut acc = GradientAccumulator::new(accum_config, &[2_usize]);
        let mut amp_stats = AmpStats::default();

        // Micro-batch 1
        let raw_grads_1: Vec<Vec<f32>> = vec![vec![0.5_f32, 0.5]];
        let overflow_1 = !scaler.check_gradients(&raw_grads_1);
        if !overflow_1 {
            let mut grads = raw_grads_1.clone();
            scaler.unscale_gradients(&mut grads);
            acc.step(&grads).expect("step 1");
        }
        scaler.update(overflow_1);
        amp_stats.update(overflow_1, scaler.current_scale());

        // Micro-batch 2 → triggers optimizer step
        let raw_grads_2: Vec<Vec<f32>> = vec![vec![0.3_f32, 0.7]];
        let overflow_2 = !scaler.check_gradients(&raw_grads_2);
        let update_result = if !overflow_2 {
            let mut grads = raw_grads_2.clone();
            scaler.unscale_gradients(&mut grads);
            acc.step(&grads).expect("step 2")
        } else {
            None
        };
        scaler.update(overflow_2);
        amp_stats.update(overflow_2, scaler.current_scale());

        assert_eq!(amp_stats.successful_steps, 2, "both steps should succeed");
        assert_eq!(amp_stats.overflow_count, 0, "no overflows expected");
        assert!(update_result.is_some(), "second step should yield gradients");
    }

    // -----------------------------------------------------------------------
    // 20. GradError: Display implementations are non-empty
    // -----------------------------------------------------------------------
    #[test]
    fn test_grad_error_display_messages() {
        use std::fmt::Write as _;
        let mut buf = String::new();

        let e1 = GradError::ShapeMismatch { expected: 2, got: 3 };
        write!(buf, "{e1}").expect("format");
        assert!(!buf.is_empty(), "ShapeMismatch display must be non-empty");
        buf.clear();

        let e2 = GradError::TensorLengthMismatch { param_idx: 0, expected: 4, got: 2 };
        write!(buf, "{e2}").expect("format");
        assert!(!buf.is_empty(), "TensorLengthMismatch display must be non-empty");
        buf.clear();

        let e3 = GradError::EmptyBuffer;
        write!(buf, "{e3}").expect("format");
        assert!(!buf.is_empty(), "EmptyBuffer display must be non-empty");
    }

    // -----------------------------------------------------------------------
    // 21. TrainingPrecisionMode: Display round-trip
    // -----------------------------------------------------------------------
    #[test]
    fn test_training_precision_mode_display() {
        assert_eq!(format!("{}", TrainingPrecisionMode::Fp32), "FP32");
        assert_eq!(format!("{}", TrainingPrecisionMode::Fp16), "FP16");
        assert_eq!(format!("{}", TrainingPrecisionMode::Bf16), "BF16");
        assert_eq!(format!("{}", TrainingPrecisionMode::Fp8E4M3), "FP8-E4M3");
    }

    // -----------------------------------------------------------------------
    // 22. GradientBuffer: current_step tracks correctly across cycles
    // -----------------------------------------------------------------------
    #[test]
    fn test_gradient_buffer_current_step_tracking() {
        let config = GradAccumConfig {
            accumulation_steps: 3,
            normalize_by_steps: false,
            clip_grad_norm: None,
            sync_on_last: true,
        };
        let mut buf = GradientBuffer::new(&[1_usize]);
        let g = vec![vec![1.0_f32]];

        assert_eq!(buf.current_step(), 0);
        buf.accumulate(&g, &config).expect("step 1");
        assert_eq!(buf.current_step(), 1);
        buf.accumulate(&g, &config).expect("step 2");
        assert_eq!(buf.current_step(), 2);
        buf.accumulate(&g, &config).expect("step 3");
        assert_eq!(buf.current_step(), 3);
        assert!(buf.is_ready);

        buf.finalize(&config).expect("finalize");
        assert_eq!(buf.current_step(), 0, "after finalize, step should reset");
    }

    // -----------------------------------------------------------------------
    // 23. cast_fp32_to_fp16 / cast_fp16_to_fp32: NaN and Infinity propagation
    // -----------------------------------------------------------------------
    #[test]
    fn test_fp16_special_values_propagation() {
        let special: Vec<f32> = vec![f32::INFINITY, f32::NEG_INFINITY, f32::NAN];
        let fp16 = cast_fp32_to_fp16(&special);
        let recovered = cast_fp16_to_fp32(&fp16);

        assert!(
            recovered[0].is_infinite() && recovered[0].is_sign_positive(),
            "+Inf should survive FP16 round-trip"
        );
        assert!(
            recovered[1].is_infinite() && recovered[1].is_sign_negative(),
            "-Inf should survive FP16 round-trip"
        );
        assert!(
            recovered[2].is_nan(),
            "NaN should survive FP16 round-trip"
        );
    }

    // -----------------------------------------------------------------------
    // 24. LossScaler: consecutive_overflows counter increments
    // -----------------------------------------------------------------------
    #[test]
    fn test_loss_scaler_consecutive_overflow_count() {
        let mut scaler = LossScaler::new();
        scaler.update(true);
        scaler.update(true);
        scaler.update(true);
        assert_eq!(scaler.consecutive_overflows, 3);
        // A success resets it.
        scaler.update(false);
        assert_eq!(scaler.consecutive_overflows, 0);
    }

    // -----------------------------------------------------------------------
    // 25. Large accumulation step count integration test
    // -----------------------------------------------------------------------
    #[test]
    fn test_large_accumulation_window() {
        let config = GradAccumConfig {
            accumulation_steps: 16,
            normalize_by_steps: true,
            clip_grad_norm: None,
            sync_on_last: true,
        };
        let num_params = 5;
        let param_shape: Vec<usize> = vec![10; num_params];
        let mut acc = GradientAccumulator::new(config, &param_shape);

        let grads: Vec<Vec<f32>> = (0..num_params)
            .map(|p| (0..10).map(|i| (p * 10 + i) as f32 * 0.1).collect())
            .collect();

        for step in 0..16 {
            let result = acc.step(&grads).expect("step");
            if step < 15 {
                assert!(result.is_none(), "should not yield until step 16");
            } else {
                let averaged = result.expect("step 16 should yield");
                // Each gradient value should equal the original (normalised mean of 16
                // identical micro-batches = original value).
                for (p, param_grads) in averaged.iter().enumerate() {
                    for (i, &val) in param_grads.iter().enumerate() {
                        let expected = (p * 10 + i) as f32 * 0.1;
                        assert!(
                            (val - expected).abs() < 1e-4,
                            "param[{p}][{i}]: expected {expected}, got {val}"
                        );
                    }
                }
            }
        }
        assert_eq!(acc.stats().total_updates, 1);
        assert_eq!(acc.stats().total_micro_batches, 16);
    }
}
