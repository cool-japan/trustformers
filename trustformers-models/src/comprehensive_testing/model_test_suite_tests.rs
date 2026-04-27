#[cfg(test)]
mod tests {
    use crate::comprehensive_testing::config::{TestDataType, TestInputConfig, ValidationConfig};
    use crate::comprehensive_testing::model_test_suite::*;
    use crate::comprehensive_testing::types::*;
    use std::time::Duration;

    // --- ModelTestSuite tests ---

    #[test]
    fn test_model_test_suite_new() {
        let _suite = ModelTestSuite::new("test-model");
        // Verify it was created successfully
    }

    #[test]
    fn test_model_test_suite_with_config() {
        let config = ValidationConfig {
            numerical_tolerance: 1e-3,
            ..ValidationConfig::default()
        };
        let _suite = ModelTestSuite::with_config("custom-model", config);
        // Suite created with custom config
    }

    #[test]
    fn test_model_test_suite_with_default_config() {
        let _suite = ModelTestSuite::new("test-model");
        // Suite created with default configuration
    }

    // --- ValidationConfig tests ---

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert!((config.numerical_tolerance - 1e-4).abs() < 1e-8);
        assert!(config.run_performance_tests);
        assert!(!config.compare_with_reference);
    }

    #[test]
    fn test_validation_config_test_inputs() {
        let config = ValidationConfig::default();
        assert_eq!(config.test_inputs.len(), 3);
        assert_eq!(config.test_inputs[0].name, "small_batch");
        assert_eq!(config.test_inputs[1].name, "medium_batch");
        assert_eq!(config.test_inputs[2].name, "large_batch");
    }

    #[test]
    fn test_validation_config_required_inputs() {
        let config = ValidationConfig::default();
        let required_count = config.test_inputs.iter().filter(|i| i.required).count();
        assert_eq!(required_count, 2); // small_batch and medium_batch
    }

    #[test]
    fn test_validation_config_data_types() {
        let config = ValidationConfig::default();
        assert!(config.test_data_types.contains(&TestDataType::F32));
        assert!(config.test_data_types.contains(&TestDataType::F16));
    }

    #[test]
    fn test_validation_config_max_inference_time() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_inference_time_ms, 10000);
    }

    #[test]
    fn test_validation_config_max_memory() {
        let config = ValidationConfig::default();
        assert_eq!(config.max_memory_usage_mb, 16384);
    }

    // --- TestInputConfig tests ---

    #[test]
    fn test_input_config_creation() {
        let input = TestInputConfig {
            name: "custom_input".to_string(),
            dimensions: vec![1, 64],
            data_type: TestDataType::F32,
            required: true,
        };
        assert_eq!(input.name, "custom_input");
        assert_eq!(input.dimensions, vec![1, 64]);
        assert!(input.required);
    }

    #[test]
    fn test_input_config_clone() {
        let input = TestInputConfig {
            name: "original".to_string(),
            dimensions: vec![2, 128],
            data_type: TestDataType::I64,
            required: false,
        };
        let cloned = input.clone();
        assert_eq!(cloned.name, "original");
        assert_eq!(cloned.dimensions, vec![2, 128]);
    }

    // --- TestDataType tests ---

    #[test]
    fn test_data_type_equality() {
        assert_eq!(TestDataType::F32, TestDataType::F32);
        assert_eq!(TestDataType::F16, TestDataType::F16);
        assert_eq!(TestDataType::I32, TestDataType::I32);
        assert_eq!(TestDataType::I64, TestDataType::I64);
    }

    #[test]
    fn test_data_type_inequality() {
        assert_ne!(TestDataType::F32, TestDataType::F16);
        assert_ne!(TestDataType::I32, TestDataType::I64);
    }

    // --- NumericalParityResults tests ---

    #[test]
    fn test_numerical_parity_results_all_passed() {
        let results = NumericalParityResults {
            all_passed: true,
            test_results: vec![TestResult {
                name: "forward_pass".to_string(),
                passed: true,
                error_message: None,
                numerical_differences: None,
                execution_time: Duration::from_millis(10),
            }],
            statistics: TestStatistics {
                total_tests: 1,
                passed_tests: 1,
                failed_tests: 0,
                pass_rate: 100.0,
            },
            timing: TimingInfo {
                total_time: Duration::from_millis(10),
                average_time: Duration::from_millis(10),
                fastest_time: Duration::from_millis(10),
                slowest_time: Duration::from_millis(10),
            },
        };
        assert!(results.all_passed);
        assert_eq!(results.statistics.total_tests, 1);
    }

    // --- TestResult tests ---

    #[test]
    fn test_result_passed() {
        let result = TestResult {
            name: "stability_test".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: Duration::from_millis(5),
        };
        assert!(result.passed);
        assert!(result.error_message.is_none());
    }

    #[test]
    fn test_result_failed() {
        let result = TestResult {
            name: "precision_test".to_string(),
            passed: false,
            error_message: Some("NaN detected".to_string()),
            numerical_differences: Some(NumericalDifferences {
                max_abs_diff: 1.5,
                mean_abs_diff: 0.3,
                rms_diff: 0.5,
                within_tolerance_percent: 80.0,
            }),
            execution_time: Duration::from_millis(15),
        };
        assert!(!result.passed);
        assert!(result.numerical_differences.is_some());
    }

    // --- NumericalDifferences tests ---

    #[test]
    fn test_numerical_differences_fields() {
        let diffs = NumericalDifferences {
            max_abs_diff: 0.001,
            mean_abs_diff: 0.0005,
            rms_diff: 0.0007,
            within_tolerance_percent: 99.9,
        };
        assert!(diffs.max_abs_diff > diffs.mean_abs_diff);
        assert!(diffs.within_tolerance_percent > 99.0);
    }

    // --- TestStatistics tests ---

    #[test]
    fn test_statistics_pass_rate() {
        let stats = TestStatistics {
            total_tests: 10,
            passed_tests: 8,
            failed_tests: 2,
            pass_rate: 80.0,
        };
        assert_eq!(stats.total_tests, stats.passed_tests + stats.failed_tests);
        assert!((stats.pass_rate - 80.0).abs() < f32::EPSILON);
    }

    // --- TimingInfo tests ---

    #[test]
    fn test_timing_info() {
        let timing = TimingInfo {
            total_time: Duration::from_millis(100),
            average_time: Duration::from_millis(20),
            fastest_time: Duration::from_millis(5),
            slowest_time: Duration::from_millis(50),
        };
        assert!(timing.fastest_time <= timing.average_time);
        assert!(timing.average_time <= timing.slowest_time);
    }
}
