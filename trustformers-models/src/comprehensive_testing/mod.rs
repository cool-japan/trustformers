//! # Comprehensive Model Testing and Validation Framework
//!
//! This module provides an extensive testing infrastructure for validating model implementations
//! against reference implementations, ensuring numerical parity, and providing performance
//! benchmarking capabilities.
//!
//! ## Features
//!
//! - **Cross-Framework Validation**: Compare outputs with HuggingFace, PyTorch, and other frameworks
//! - **Numerical Parity Testing**: Ensure mathematical correctness of implementations
//! - **Performance Profiling**: Layer-wise latency and memory usage analysis
//! - **Gradient Flow Verification**: Ensure proper gradient propagation
//! - **Automated Benchmarking**: Generate comprehensive benchmark reports
//! - **Reference Value Comparison**: Compare against known good outputs
//! - **Architecture Unit Tests**: Test individual components in isolation
//! - **Fairness Assessment**: Comprehensive bias detection and mitigation analysis
//!
//! ## Usage
//!
//! ```rust,no_run
//! use trustformers_models::comprehensive_testing::{ModelTestSuite, PerformanceProfiler};
//! use trustformers_core::{
//!     traits::{Config, Model},
//!     tensor::Tensor,
//!     Result,
//! };
//! use serde::{Deserialize, Serialize};
//!
//! # #[derive(Debug, Clone, Serialize, Deserialize)]
//! # struct DocConfig;
//! # impl Config for DocConfig {
//! #     fn architecture(&self) -> &'static str { "doc" }
//! # }
//! # struct DocModel;
//! # impl Model for DocModel {
//! #     type Config = DocConfig;
//! #     type Input = Tensor;
//! #     type Output = Tensor;
//! #     fn forward(&self, input: Tensor) -> Result<Tensor> { Ok(input) }
//! #     fn load_pretrained(&mut self, _r: &mut dyn std::io::Read) -> Result<()> { Ok(()) }
//! #     fn get_config(&self) -> &DocConfig { &DocConfig }
//! #     fn num_parameters(&self) -> usize { 0 }
//! # }
//!
//! # fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
//! // Create a test suite for a model
//! let mut test_suite = ModelTestSuite::new("llama-7b");
//! # let model = DocModel;
//! let _parity_results = test_suite.run_numerical_parity_tests(&model)?;
//!
//! // Profile model performance
//! let profiler = PerformanceProfiler::new();
//! let _perf_results = profiler.profile_model(&model)?;
//! # Ok(())
//! # }
//! ```

pub mod config;
pub mod fairness;
#[cfg(test)]
mod fairness_tests;
pub mod model_test_suite;
#[cfg(test)]
mod model_test_suite_tests;
pub mod performance;
pub mod reference_comparison;
pub mod reporting;
pub mod stats;
#[cfg(test)]
mod stats_tests;
pub mod types;

// Re-export all public types and functions for backward compatibility
pub use config::{TestDataType, TestInputConfig, ValidationConfig};
pub use fairness::{
    BiasMetric, BiasmitigationStrategy, FairnessAssessment, FairnessConfig, FairnessMetricType,
    FairnessResult, FairnessTestData, FairnessViolation, GroupData, StatisticalTest,
};
pub use model_test_suite::ModelTestSuite;
pub use performance::PerformanceProfiler;
pub use reference_comparison::ReferenceComparator;
pub use reporting::{generate_test_report, save_report_to_file};
pub use stats::{
    chi_square_p_value, chi_square_test, expected_calibration_error, normal_cdf, normal_quantile,
    two_proportion_z_test, ChiSquareResult, TwoProportionTest,
};
pub use types::{
    LayerPerformance, MemoryAnalysis, NumericalDifferences, NumericalParityResults,
    OverallPerformance, PerformanceResults, TestResult, TestStatistics, ThroughputMeasurements,
    TimingInfo,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validation_config_default() {
        let config = ValidationConfig::default();
        assert_eq!(config.numerical_tolerance, 1e-4);
        assert!(config.run_performance_tests);
        assert!(!config.compare_with_reference);
        assert_eq!(config.test_inputs.len(), 3);
    }

    #[test]
    fn test_model_test_suite_creation() {
        let _test_suite = ModelTestSuite::new("test-model");
        // We can't access private fields directly, so just test construction succeeds
    }

    #[test]
    fn test_performance_profiler_creation() {
        let _profiler = PerformanceProfiler::new();
    }

    #[test]
    fn test_reference_comparator() {
        let comparator = ReferenceComparator::new(1e-3);
        // Verify comparator is usable
        let _ = comparator.tolerance();
    }

    #[test]
    fn test_numerical_differences_validation() {
        let comparator = ReferenceComparator::new(1e-3);

        let good_diffs = NumericalDifferences {
            max_abs_diff: 1e-4,
            mean_abs_diff: 1e-5,
            rms_diff: 1e-5,
            within_tolerance_percent: 99.0,
        };
        assert!(comparator.validate_differences(&good_diffs));

        let bad_diffs = NumericalDifferences {
            max_abs_diff: 1e-2,
            mean_abs_diff: 1e-3,
            rms_diff: 1e-3,
            within_tolerance_percent: 90.0,
        };
        assert!(!comparator.validate_differences(&bad_diffs));
    }

    #[test]
    fn test_test_statistics_calculation() {
        let stats = TestStatistics {
            total_tests: 10,
            passed_tests: 8,
            failed_tests: 2,
            pass_rate: 80.0,
        };
        assert_eq!(stats.pass_rate, 80.0);
        assert_eq!(stats.total_tests, stats.passed_tests + stats.failed_tests);
    }
}
