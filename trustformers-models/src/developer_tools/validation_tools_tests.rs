#[cfg(test)]
mod tests {
    use crate::developer_tools::validation_tools::*;
    use std::collections::HashMap;
    use trustformers_core::tensor::Tensor;

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
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }

        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    fn make_tensor(rng: &mut Lcg, shape: &[usize]) -> trustformers_core::errors::Result<Tensor> {
        let total: usize = shape.iter().product();
        let data: Vec<f32> = (0..total).map(|_| rng.next_f32() * 0.1).collect();
        Tensor::from_vec(data, shape)
    }

    // --- ValidationConfig tests ---

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!((config.numerical_tolerance - 1e-5).abs() < 1e-10);
        assert!(config.check_nan);
        assert!(config.check_infinite);
    }

    #[test]
    fn test_validation_config_custom() {
        let config = ValidationConfig {
            numerical_tolerance: 1e-3,
            check_nan: false,
            check_infinite: true,
            expected_shapes: HashMap::new(),
            value_ranges: HashMap::new(),
            performance_thresholds: PerformanceThresholds {
                max_forward_time_ms: 500.0,
                max_memory_mb: 1024.0,
                min_throughput: 10.0,
            },
        };
        assert!(!config.check_nan);
    }

    // --- ModelValidator tests ---

    #[test]
    fn test_model_validator_creation() {
        let config = ValidationConfig::default();
        let _validator = ModelValidator::new(config);
    }

    #[test]
    fn test_validate_tensor_valid_f32() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        let mut rng = Lcg::new(42);
        if let Ok(tensor) = make_tensor(&mut rng, &[2, 3]) {
            let result = validator.validate_tensor("test_output", &tensor);
            assert!(result.passed);
            assert!(result.errors.is_empty());
        }
    }

    #[test]
    fn test_validate_tensor_shape_check() {
        let mut expected = HashMap::new();
        expected.insert("output".to_string(), vec![2, 3]);
        let config = ValidationConfig {
            expected_shapes: expected,
            ..ValidationConfig::default()
        };
        let validator = ModelValidator::new(config);
        let mut rng = Lcg::new(42);
        if let Ok(tensor) = make_tensor(&mut rng, &[2, 3]) {
            let result = validator.validate_tensor("output", &tensor);
            assert!(result.passed);
        }
    }

    #[test]
    fn test_validate_tensor_shape_mismatch() {
        let mut expected = HashMap::new();
        expected.insert("output".to_string(), vec![4, 5]);
        let config = ValidationConfig {
            expected_shapes: expected,
            ..ValidationConfig::default()
        };
        let validator = ModelValidator::new(config);
        let mut rng = Lcg::new(42);
        if let Ok(tensor) = make_tensor(&mut rng, &[2, 3]) {
            let result = validator.validate_tensor("output", &tensor);
            assert!(!result.passed);
            assert!(!result.errors.is_empty());
        }
    }

    #[test]
    fn test_validate_config_basic() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        let test_config = PerformanceThresholds {
            max_forward_time_ms: 100.0,
            max_memory_mb: 512.0,
            min_throughput: 50.0,
        };
        let result = validator.validate_config(&test_config);
        assert!(result.passed);
    }

    #[test]
    fn test_validate_performance_basic() {
        let config = ValidationConfig {
            performance_thresholds: PerformanceThresholds {
                max_forward_time_ms: 10000.0,
                max_memory_mb: 4096.0,
                min_throughput: 0.001,
            },
            ..ValidationConfig::default()
        };
        let validator = ModelValidator::new(config);
        let result = validator
            .validate_performance(|| Tensor::zeros(&[2, 3]).map_err(|e| anyhow::anyhow!("{}", e)));
        assert!(result.passed);
        assert!(result.performance_metrics.is_some());
    }

    #[test]
    fn test_validate_performance_failure() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        let result = validator.validate_performance(|| Err(anyhow::anyhow!("forced failure")));
        assert!(!result.passed);
    }

    // --- Compare tensors tests ---

    #[test]
    fn test_compare_tensors_identical() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        let mut rng = Lcg::new(42);
        if let Ok(t1) = make_tensor(&mut rng, &[3, 4]) {
            let t2 = t1.clone();
            let result = validator.compare_tensors("test", &t1, &t2);
            assert!(result.passed);
        }
    }

    #[test]
    fn test_compare_tensors_shape_mismatch() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        let mut rng = Lcg::new(42);
        if let Ok(t1) = make_tensor(&mut rng, &[3, 4]) {
            if let Ok(t2) = make_tensor(&mut rng, &[4, 3]) {
                let result = validator.compare_tensors("test", &t1, &t2);
                assert!(!result.passed);
            }
        }
    }

    #[test]
    fn test_compare_tensors_different_values() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        if let Ok(t1) = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]) {
            if let Ok(t2) = Tensor::from_vec(vec![10.0_f32, 20.0, 30.0], &[3]) {
                let result = validator.compare_tensors("test", &t1, &t2);
                assert!(!result.passed);
            }
        }
    }

    #[test]
    fn test_compare_tensors_f32_identical() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        if let Ok(t1) = Tensor::from_vec(vec![1.0_f32, 2.0, 3.0], &[3]) {
            let t2 = t1.clone();
            let result = validator.compare_tensors("test", &t1, &t2);
            assert!(result.passed);
        }
    }

    // --- ValidationUtils tests ---

    #[test]
    fn test_strict_config() {
        let config = ValidationUtils::strict_config();
        assert!(config.numerical_tolerance < 1e-5);
        assert!(config.performance_thresholds.max_forward_time_ms < 1000.0);
    }

    #[test]
    fn test_lenient_config() {
        let config = ValidationUtils::lenient_config();
        assert!(config.numerical_tolerance > 1e-5);
        assert!(config.performance_thresholds.max_forward_time_ms > 1000.0);
    }

    // --- ValidationResult tests ---

    #[test]
    fn test_validation_result_fields() {
        let result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: vec!["minor warning".to_string()],
            performance_metrics: None,
        };
        assert!(result.passed);
        assert!(result.errors.is_empty());
        assert_eq!(result.warnings.len(), 1);
    }

    // --- PerformanceMetrics tests ---

    #[test]
    fn test_performance_metrics_fields() {
        let metrics = PerformanceMetrics {
            forward_time_ms: 5.0,
            memory_usage_mb: 128.0,
            throughput: 200.0,
            parameter_count: 1000000,
        };
        assert!((metrics.forward_time_ms - 5.0).abs() < f64::EPSILON);
        assert_eq!(metrics.parameter_count, 1000000);
    }

    // --- Validate against reference tests ---

    #[test]
    fn test_validate_against_reference_matching() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        if let Ok(input) = Tensor::zeros(&[2, 3]) {
            let test_inputs = vec![input];
            let result = validator.validate_against_reference(
                |_t| Tensor::zeros(&[2, 3]).map_err(|e| anyhow::anyhow!("{}", e)),
                |_t| Tensor::zeros(&[2, 3]).map_err(|e| anyhow::anyhow!("{}", e)),
                test_inputs,
            );
            assert!(result.passed);
        }
    }

    #[test]
    fn test_validate_against_reference_model_error() {
        let config = ValidationConfig::default();
        let validator = ModelValidator::new(config);
        if let Ok(input) = Tensor::zeros(&[2, 3]) {
            let test_inputs = vec![input];
            let result = validator.validate_against_reference(
                |_t| -> anyhow::Result<Tensor> { Err(anyhow::anyhow!("model error")) },
                |_t| Tensor::zeros(&[2, 3]).map_err(|e| anyhow::anyhow!("{}", e)),
                test_inputs,
            );
            assert!(!result.passed);
        }
    }
}
