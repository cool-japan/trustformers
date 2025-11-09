//! Configuration types for comprehensive testing framework

use serde::{Deserialize, Serialize};

/// Configuration for model validation tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Numerical tolerance for comparisons
    pub numerical_tolerance: f32,
    /// Whether to run performance benchmarks
    pub run_performance_tests: bool,
    /// Whether to compare against reference implementations
    pub compare_with_reference: bool,
    /// Maximum acceptable inference time (milliseconds)
    pub max_inference_time_ms: u64,
    /// Maximum acceptable memory usage (MB)
    pub max_memory_usage_mb: u64,
    /// Test input configurations
    pub test_inputs: Vec<TestInputConfig>,
    /// Supported data types for testing
    pub test_data_types: Vec<TestDataType>,
}

/// Test input configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestInputConfig {
    /// Input name/description
    pub name: String,
    /// Input dimensions
    pub dimensions: Vec<usize>,
    /// Input data type
    pub data_type: TestDataType,
    /// Whether this is a required test
    pub required: bool,
}

/// Supported data types for testing
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum TestDataType {
    F32,
    F16,
    I32,
    I64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-4,
            run_performance_tests: true,
            compare_with_reference: false,
            max_inference_time_ms: 10000,
            max_memory_usage_mb: 16384,
            test_inputs: vec![
                TestInputConfig {
                    name: "small_batch".to_string(),
                    dimensions: vec![1, 128],
                    data_type: TestDataType::I32,
                    required: true,
                },
                TestInputConfig {
                    name: "medium_batch".to_string(),
                    dimensions: vec![4, 256],
                    data_type: TestDataType::I32,
                    required: true,
                },
                TestInputConfig {
                    name: "large_batch".to_string(),
                    dimensions: vec![16, 512],
                    data_type: TestDataType::I32,
                    required: false,
                },
            ],
            test_data_types: vec![TestDataType::F32, TestDataType::F16],
        }
    }
}
