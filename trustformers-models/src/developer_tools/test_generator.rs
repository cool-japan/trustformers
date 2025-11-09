//! Test Generator
//!
//! Automatic generation of comprehensive test suites for model implementations.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Test suite configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TestGeneratorConfig {
    /// Model name to generate tests for
    pub model_name: String,
    /// Test categories to include
    pub test_categories: Vec<TestCategory>,
    /// Model configuration parameters
    pub model_config: HashMap<String, String>,
    /// Expected output shapes for validation
    pub output_shapes: HashMap<String, Vec<usize>>,
}

/// Test category
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TestCategory {
    Basic,
    Numerical,
    Performance,
    EdgeCases,
    Integration,
    PropertyBased,
}

/// Test generator
pub struct TestGenerator {
    config: TestGeneratorConfig,
}

impl TestGenerator {
    /// Create a new test generator
    pub fn new(config: TestGeneratorConfig) -> Self {
        Self { config }
    }

    /// Generate comprehensive test suite
    pub fn generate_tests(&self, output_path: &Path) -> Result<()> {
        let mut test_content = self.generate_header();

        for category in &self.config.test_categories {
            match category {
                TestCategory::Basic => test_content.push_str(&self.generate_basic_tests()),
                TestCategory::Numerical => test_content.push_str(&self.generate_numerical_tests()),
                TestCategory::Performance => {
                    test_content.push_str(&self.generate_performance_tests())
                },
                TestCategory::EdgeCases => test_content.push_str(&self.generate_edge_case_tests()),
                TestCategory::Integration => {
                    test_content.push_str(&self.generate_integration_tests())
                },
                TestCategory::PropertyBased => {
                    test_content.push_str(&self.generate_property_tests())
                },
            }
        }

        std::fs::write(output_path, test_content)?;
        Ok(())
    }

    /// Generate test file header
    fn generate_header(&self) -> String {
        format!(
            "//! Comprehensive Test Suite for {}\n//!\n//! This file is auto-generated. Do not edit manually.\n\nuse super::{{{}Config, {}Model}};\nuse trustformers_core::tensor::Tensor;\nuse trustformers_core::errors::Result;\nuse approx::assert_abs_diff_eq;\nuse std::time::Instant;\n\n",
            self.config.model_name,
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate basic functionality tests
    fn generate_basic_tests(&self) -> String {
        format!(
            "// ========== Basic Tests ==========\n\n#[test]\nfn test_{}_creation() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    // Model creation should succeed\n}}\n\n#[test]\nfn test_{}_config_validation() {{\n    let config = {}Config::default();\n    // Validate configuration parameters\n    assert!(config.hidden_size > 0);\n    assert!(config.vocab_size > 0);\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase(),
            self.config.model_name
        )
    }

    /// Generate numerical stability tests
    fn generate_numerical_tests(&self) -> String {
        format!(
            "// ========== Numerical Tests ==========\n\n#[test]\nfn test_{}_numerical_stability() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    // Test with various input sizes\n    let input_sizes = vec![1, 4, 16, 32];\n    \n    for batch_size in input_sizes {{\n        let input = Tensor::zeros(&[batch_size, 512]);\n        let output = model.forward(&input).expect(\"Forward pass failed\");\n        \n        // Check for NaN or infinite values\n        match &output {{\n            Tensor::F32(arr) => {{\n                for &val in arr.iter() {{\n                    assert!(val.is_finite(), \"Output contains non-finite values\");\n                }}\n            }}\n            _ => panic!(\"Expected F32 tensor\"),\n        }}\n    }}\n}}\n\n#[test]\nfn test_{}_output_ranges() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    let input = Tensor::randn(&[4, 512]);\n    let output = model.forward(&input).expect(\"Forward pass failed\");\n    \n    // Verify output is within reasonable ranges\n    match &output {{\n        Tensor::F32(arr) => {{\n            let min_val = arr.iter().cloned().fold(f32::INFINITY, f32::min);\n            let max_val = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max);\n            \n            assert!(min_val > -100.0, \"Output values too negative: {{}}\", min_val);\n            assert!(max_val < 100.0, \"Output values too positive: {{}}\", max_val);\n        }}\n        _ => panic!(\"Expected F32 tensor\"),\n    }}\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate performance tests
    fn generate_performance_tests(&self) -> String {
        format!(
            "// ========== Performance Tests ==========\n\n#[test]\nfn test_{}_performance_baseline() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    let input = Tensor::randn(&[8, 512]);\n    \n    // Warm-up run\n    let _ = model.forward(&input).expect(\"Warm-up failed\");\n    \n    // Benchmark run\n    let start = Instant::now();\n    let _output = model.forward(&input).expect(\"Benchmark run failed\");\n    let duration = start.elapsed();\n    \n    // Performance should be reasonable (adjust threshold as needed)\n    assert!(duration.as_millis() < 1000, \"Forward pass too slow: {{:?}}\", duration);\n}}\n\n#[test]\nfn test_{}_memory_usage() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    // Test with different batch sizes\n    let batch_sizes = vec![1, 4, 8, 16];\n    \n    for batch_size in batch_sizes {{\n        let input = Tensor::randn(&[batch_size, 512]);\n        let output = model.forward(&input).expect(\"Forward pass failed\");\n        \n        // Verify output shapes scale correctly\n        match &output {{\n            Tensor::F32(arr) => {{\n                assert_eq!(arr.shape()[0], batch_size, \"Batch size mismatch\");\n            }}\n            _ => panic!(\"Expected F32 tensor\"),\n        }}\n    }}\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate edge case tests
    fn generate_edge_case_tests(&self) -> String {
        format!(
            "// ========== Edge Case Tests ==========\n\n#[test]\nfn test_{}_zero_input() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    let input = Tensor::zeros(&[4, 512]);\n    let output = model.forward(&input).expect(\"Forward pass with zeros failed\");\n    \n    // Model should handle zero input gracefully\n    match &output {{\n        Tensor::F32(arr) => {{\n            assert!(!arr.iter().any(|&x| x.is_nan()), \"Zero input produced NaN\");\n        }}\n        _ => panic!(\"Expected F32 tensor\"),\n    }}\n}}\n\n#[test]\nfn test_{}_single_batch() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    let input = Tensor::randn(&[1, 512]);\n    let output = model.forward(&input).expect(\"Single batch forward pass failed\");\n    \n    // Single batch should work correctly\n    match &output {{\n        Tensor::F32(arr) => {{\n            assert_eq!(arr.shape()[0], 1, \"Single batch output shape incorrect\");\n        }}\n        _ => panic!(\"Expected F32 tensor\"),\n    }}\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate integration tests
    fn generate_integration_tests(&self) -> String {
        format!(
            "// ========== Integration Tests ==========\n\n#[test]\nfn test_{}_reproducibility() {{\n    let config = {}Config::default();\n    let model1 = {}Model::new(config.clone()).expect(\"Failed to create model 1\");\n    let model2 = {}Model::new(config).expect(\"Failed to create model 2\");\n    \n    let input = Tensor::ones(&[4, 512]);\n    \n    let output1 = model1.forward(&input).expect(\"Model 1 forward pass failed\");\n    let output2 = model2.forward(&input).expect(\"Model 2 forward pass failed\");\n    \n    // Outputs should be identical for same configuration\n    match (&output1, &output2) {{\n        (Tensor::F32(arr1), Tensor::F32(arr2)) => {{\n            for (a, b) in arr1.iter().zip(arr2.iter()) {{\n                assert_abs_diff_eq!(a, b, epsilon = 1e-6);\n            }}\n        }}\n        _ => panic!(\"Expected F32 tensors\"),\n    }}\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name,
            self.config.model_name
        )
    }

    /// Generate property-based tests
    fn generate_property_tests(&self) -> String {
        format!(
            "// ========== Property-Based Tests ==========\n\n#[test]\nfn test_{}_shape_consistency() {{\n    let config = {}Config::default();\n    let model = {}Model::new(config).expect(\"Failed to create model\");\n    \n    // Test various input shapes\n    let test_shapes = vec![\n        vec![1, 512],\n        vec![4, 512],\n        vec![8, 256],\n        vec![16, 128],\n    ];\n    \n    for shape in test_shapes {{\n        let input = Tensor::randn(&shape);\n        let output = model.forward(&input).expect(\"Forward pass failed\");\n        \n        // Output batch dimension should match input\n        match &output {{\n            Tensor::F32(arr) => {{\n                assert_eq!(arr.shape()[0], shape[0], \"Batch size mismatch for shape {{:?}}\", shape);\n            }}\n            _ => panic!(\"Expected F32 tensor\"),\n        }}\n    }}\n}}\n\n",
            self.config.model_name.to_lowercase(),
            self.config.model_name,
            self.config.model_name
        )
    }
}

/// Predefined test configurations
pub struct TestTemplates;

impl TestTemplates {
    /// Get comprehensive test configuration
    pub fn comprehensive(model_name: String) -> TestGeneratorConfig {
        TestGeneratorConfig {
            model_name,
            test_categories: vec![
                TestCategory::Basic,
                TestCategory::Numerical,
                TestCategory::Performance,
                TestCategory::EdgeCases,
                TestCategory::Integration,
                TestCategory::PropertyBased,
            ],
            model_config: HashMap::new(),
            output_shapes: HashMap::new(),
        }
    }

    /// Get basic test configuration
    pub fn basic(model_name: String) -> TestGeneratorConfig {
        TestGeneratorConfig {
            model_name,
            test_categories: vec![TestCategory::Basic, TestCategory::Numerical],
            model_config: HashMap::new(),
            output_shapes: HashMap::new(),
        }
    }

    /// Get performance-focused test configuration
    pub fn performance(model_name: String) -> TestGeneratorConfig {
        TestGeneratorConfig {
            model_name,
            test_categories: vec![TestCategory::Basic, TestCategory::Performance],
            model_config: HashMap::new(),
            output_shapes: HashMap::new(),
        }
    }
}
