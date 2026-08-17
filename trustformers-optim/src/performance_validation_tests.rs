// performance_validation_tests.rs — validation-framework tests.
//
// Split out of `performance_validation.rs` to keep both files under the 2000-line
// limit; wired in with `#[path]` from the parent module.
#[cfg(test)]
mod tests {
    use crate::performance_validation::*;

    #[test]
    fn test_validation_config_creation() {
        let config = ValidationConfig::default();
        assert!(config.statistical_significance);
        assert!(config.memory_validation);
        assert_eq!(config.benchmark_iterations, 100);
        assert_eq!(config.confidence_level, 0.95);
    }

    #[test]
    fn test_performance_validator_creation() {
        let validator = PerformanceValidator::new()
            .with_statistical_significance(true)
            .with_memory_validation(true)
            .with_benchmark_iterations(50);

        assert!(validator.config.statistical_significance);
        assert!(validator.config.memory_validation);
        assert_eq!(validator.config.benchmark_iterations, 50);
    }

    #[test]
    fn test_mathematical_test_case_creation() {
        let test_cases = [MathematicalTestCase {
            name: "Test Case".to_string(),
            description: "Test Description".to_string(),
            parameters: HashMap::new(),
            gradients: HashMap::new(),
            expected_properties: vec![MathematicalProperty::Convergence],
            tolerance: 1e-6,
        }];

        assert_eq!(test_cases.len(), 1);
        assert_eq!(test_cases[0].name, "Test Case");
    }

    #[test]
    fn test_statistical_analyzer() {
        let analyzer = StatisticalAnalyzer::new();
        let step_times = vec![
            Duration::from_millis(10),
            Duration::from_millis(12),
            Duration::from_millis(11),
            Duration::from_millis(9),
            Duration::from_millis(13),
        ];

        let metrics = analyzer.analyze(&step_times, 0.95).expect("Operation failed in test");
        assert!(metrics.mean > Duration::from_millis(9));
        assert!(metrics.mean < Duration::from_millis(14));
    }

    #[test]
    fn test_test_data_creation() {
        let parameters = create_test_parameters(vec![10, 20]).expect("Operation failed in test");
        assert_eq!(parameters.len(), 2);

        let gradients = create_benchmark_gradients(&[10, 20], 5).expect("Operation failed in test");
        assert_eq!(gradients.len(), 2);
    }

    #[test]
    fn test_regression_detector() {
        let detector = RegressionDetector::new();

        let baseline = BenchmarkResult {
            avg_step_time: Duration::from_millis(10),
            throughput: 1000.0,
            memory_usage: 100.0,
        };

        let current = OptimizerBenchmarkResult {
            optimizer_name: "TestOptimizer".to_string(),
            avg_step_time: Duration::from_millis(12), // 20% slower
            min_step_time: Duration::from_millis(11),
            max_step_time: Duration::from_millis(13),
            throughput: 800.0,
            avg_memory_usage: 100.0,
            statistical_metrics: None,
        };

        let regression = detector
            .detect_regression(&baseline, &current, 5.0)
            .expect("Operation failed in test");
        assert!(regression.is_some());

        let regression_info = regression.expect("Operation failed in test");
        assert!(regression_info.regression_percentage > 5.0);
    }

    /// Regression: the memory-efficiency figure used to be `baseline * 0.25` with the
    /// comment "assume 75% reduction". It must now be a measurement of two optimizers
    /// that were actually stepped.
    #[test]
    fn memory_efficiency_is_measured_from_real_optimizer_state() {
        let validator = PerformanceValidator::new();
        let results = validator.test_memory_efficiency_claims().expect("memory claims");

        let adam_bytes = results.get("Adam.state_bytes").copied().expect("adam bytes");
        let quantized_bytes =
            results.get("Adam4bit.state_bytes").copied().expect("quantized bytes");
        assert!(adam_bytes > 0.0, "Adam must allocate state after a step");
        assert!(
            quantized_bytes > 0.0,
            "Adam4bit must allocate state after a step"
        );

        let reduction = results.get("Adam4bit").copied().expect("reduction");
        let expected = (adam_bytes - quantized_bytes) / adam_bytes * 100.0;
        assert!(
            (reduction - expected).abs() < 1e-6,
            "the reported reduction must follow from the measured byte counts"
        );
        // The old code always produced exactly 75.0 regardless of the optimizers.
        assert!(
            (reduction - 75.0).abs() > 1e-9,
            "a measured reduction must not coincide with the old hard-coded 75%"
        );
    }

    /// Regression: the compression table was a literal `[("TopK", 0.9), ...]`. Every
    /// figure must now come from a real compress/decompress round trip.
    #[test]
    fn compression_efficiency_is_measured_from_real_round_trips() {
        let validator = PerformanceValidator::new();
        let results = validator.test_gradient_compression_efficiency().expect("compression");

        assert!(results.contains_key("TopK"), "TopK must be measured");
        assert!(results.contains_key("SignSGD"), "SignSGD must be measured");

        // 1024-element gradient, dense payload 4096 bytes.
        // TopK k=102: 102 usize indices (8 B) + 102 f32 values (4 B) = 1224 B.
        let topk = results.get("TopK").copied().expect("TopK");
        let expected_topk = (4096.0 - 1224.0) / 4096.0 * 100.0;
        assert!(
            (topk - expected_topk).abs() < 1e-6,
            "TopK reduction {topk} must equal the measured {expected_topk}"
        );

        // 8-bit quantization: dense, one byte per value = 1024 B.
        let quantization = results.get("Quantization").copied().expect("Quantization");
        assert!(
            (quantization - 75.0).abs() < 1e-6,
            "8-bit quantization is exactly a 4x reduction: {quantization}"
        );

        // SignSGD: one bit per value = 128 B.
        let sign = results.get("SignSGD").copied().expect("SignSGD");
        assert!(
            (sign - 96.875).abs() < 1e-6,
            "SignSGD is exactly a 32x reduction: {sign}"
        );

        // The old literals were exactly 90.0 / 75.0 / 80.0 with no measurement behind
        // them; TopK in particular must no longer report 90.
        assert!(
            (topk - 90.0).abs() > 1e-9,
            "TopK must not report the old literal"
        );
    }

    /// Mixed precision must report the byte reduction an f16 tensor really gives.
    #[test]
    fn mixed_precision_reduction_is_exactly_half() {
        let validator = PerformanceValidator::new();
        let results = validator.test_memory_optimizations().expect("memory optimizations");

        let reduction = results.get("MixedPrecision").copied().expect("mixed precision");
        assert!(
            (reduction - 50.0).abs() < 1e-6,
            "f16 is exactly half of f32: {reduction}"
        );
        // Unmeasurable techniques must be absent rather than invented.
        assert!(!results.contains_key("GradientCheckpointing"));
        assert!(!results.contains_key("CPUOffloading"));
    }

    /// Regression: the loss curve used to be `initial * exp(-0.01·i) + sin(i)·0.1`,
    /// entirely independent of the optimizer being stepped. The measured curve must
    /// react to the learning rate.
    #[test]
    fn convergence_loss_comes_from_the_stepped_parameters() {
        let validator = PerformanceValidator::new();

        let adam = validator
            .test_optimizer_convergence("Adam", OptimizerType::Adam)
            .expect("adam convergence");
        let sgd = validator
            .test_optimizer_convergence("SGD", OptimizerType::SGD)
            .expect("sgd convergence");

        assert!(!adam.loss_history.is_empty());
        assert!(
            adam.final_loss < adam.loss_history[0],
            "the loss must fall: {} -> {}",
            adam.loss_history[0],
            adam.final_loss
        );
        // Two different optimizers on the same problem cannot produce identical curves
        // unless the curve is fabricated.
        assert!(
            (adam.final_loss - sgd.final_loss).abs() > 1e-9,
            "different optimizers must produce different trajectories"
        );
    }

    /// Speed and stability must be derived from the measured curves, not literals.
    #[test]
    fn convergence_analysis_is_derived_from_the_measured_curves() {
        use std::collections::HashMap;

        let validator = PerformanceValidator::new();
        let mut tests = HashMap::new();
        tests.insert(
            "Fast".to_string(),
            ConvergenceTestResult {
                converged: true,
                convergence_iteration: 1,
                convergence_rate: 1.0,
                final_loss: 0.001,
                loss_reduction: 0.999,
                loss_history: vec![1.0, 0.5, 0.001, 0.0005],
            },
        );
        tests.insert(
            "Bumpy".to_string(),
            ConvergenceTestResult {
                converged: false,
                convergence_iteration: 4,
                convergence_rate: 0.0,
                final_loss: 1.0,
                loss_reduction: 0.0,
                loss_history: vec![1.0, 2.0, 1.0, 2.0],
            },
        );

        let speed = validator.analyze_convergence_speed(&tests).expect("speed");
        let stability = validator.analyze_convergence_stability(&tests).expect("stability");

        // "Fast" reaches 1% of its initial loss at index 2 of 4 → 1 − 0.5 = 0.5.
        assert!((speed["Fast"] - 0.5).abs() < 1e-9, "{}", speed["Fast"]);
        assert_eq!(
            speed["Bumpy"], 0.0,
            "a run that never converged scores zero"
        );

        // "Fast" never increases; "Bumpy" increases on 2 of its 3 transitions.
        assert!((stability["Fast"] - 1.0).abs() < 1e-9);
        assert!(
            (stability["Bumpy"] - 1.0 / 3.0).abs() < 1e-6,
            "{}",
            stability["Bumpy"]
        );
        // The old code returned the literals 0.85/0.88/0.92/0.65 and 0.95/0.93/0.98/0.80.
        assert!(!speed.contains_key("Adam"));
    }

    /// ZeRO sharding memory reduction must be measured from real shard sizes.
    #[test]
    fn distributed_scaling_reports_measured_shard_reduction() {
        let validator = PerformanceValidator::new();
        let results = validator.test_distributed_scaling().expect("scaling");

        let one = results.get("1-rank-parameter-memory-reduction").copied().expect("world size 1");
        let four = results.get("4-rank-parameter-memory-reduction").copied().expect("world size 4");

        assert!(one.abs() < 1e-9, "a single rank saves nothing: {one}");
        assert!(
            (four - 0.75).abs() < 1e-6,
            "four ranks each hold a quarter: {four}"
        );
        // The old code reported a flat 0.85 "efficiency" at every world size.
        assert!(results.values().any(|v| (v - 0.85).abs() > 1e-9));
    }

    /// The shard → gather round trip must be exact, and the check must be real.
    #[test]
    fn communication_validation_round_trips_exactly() {
        let validator = PerformanceValidator::new();
        let results = validator.test_communication_efficiency().expect("communication");

        for (key, fraction) in &results {
            assert!(
                (fraction - 1.0).abs() < 1e-9,
                "{key} must round-trip every parameter exactly, got {fraction}"
            );
        }
    }

    /// Fault tolerance must claim only what it executed.
    #[test]
    fn fault_tolerance_reports_only_executed_checks() {
        let validator = PerformanceValidator::new();
        let results = validator.test_fault_tolerance().expect("fault tolerance");

        assert_eq!(
            results.get("CheckpointRecovery"),
            Some(&true),
            "the checkpoint round trip must succeed"
        );
        // Claims that need a real cluster must be absent, not asserted true.
        assert!(!results.contains_key("NodeFailureRecovery"));
        assert!(!results.contains_key("NetworkPartitionHandling"));
    }
}
