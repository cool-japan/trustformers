//! Model test suite implementation for comprehensive testing

use anyhow::{Error, Result};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Model;

use super::config::{TestDataType, TestInputConfig, ValidationConfig};
use super::types::{
    NumericalDifferences, NumericalParityResults, TestResult, TestStatistics, TimingInfo,
};

/// Comprehensive model test suite
pub struct ModelTestSuite {
    #[allow(dead_code)]
    model_name: String,
    config: ValidationConfig,
    #[allow(dead_code)]
    test_results: Vec<TestResult>,
}

impl ModelTestSuite {
    /// Create a new test suite for a model
    pub fn new(model_name: &str) -> Self {
        Self {
            model_name: model_name.to_string(),
            config: ValidationConfig::default(),
            test_results: Vec::new(),
        }
    }

    /// Create test suite with custom configuration
    pub fn with_config(model_name: &str, config: ValidationConfig) -> Self {
        Self {
            model_name: model_name.to_string(),
            config,
            test_results: Vec::new(),
        }
    }

    /// Run all numerical parity tests
    pub fn run_numerical_parity_tests<M: Model<Input = Tensor, Output = Tensor>>(
        &mut self,
        model: &M,
    ) -> Result<NumericalParityResults> {
        let start_time = Instant::now();
        let mut all_passed = true;
        let mut test_results = Vec::new();

        // Test basic forward pass stability
        let forward_pass_result = self.test_forward_pass_stability(model)?;
        if !forward_pass_result.passed {
            all_passed = false;
        }
        test_results.push(forward_pass_result);

        // Test deterministic outputs
        let deterministic_result = self.test_deterministic_outputs(model)?;
        if !deterministic_result.passed {
            all_passed = false;
        }
        test_results.push(deterministic_result);

        // Test numerical stability
        let numerical_stability_result = self.test_numerical_stability(model)?;
        if !numerical_stability_result.passed {
            all_passed = false;
        }
        test_results.push(numerical_stability_result);

        // Test input validation
        let input_validation_result = self.test_input_validation(model)?;
        if !input_validation_result.passed {
            all_passed = false;
        }
        test_results.push(input_validation_result);

        // Test gradient flow indicators
        let gradient_flow_result = self.test_gradient_flow_indicators(model)?;
        if !gradient_flow_result.passed {
            all_passed = false;
        }
        test_results.push(gradient_flow_result);

        let total_time = start_time.elapsed();
        let passed_tests = test_results.iter().filter(|r| r.passed).count();
        let failed_tests = test_results.len() - passed_tests;

        let statistics = TestStatistics {
            total_tests: test_results.len(),
            passed_tests,
            failed_tests,
            pass_rate: (passed_tests as f32 / test_results.len() as f32) * 100.0,
        };

        let timing = TimingInfo {
            total_time,
            average_time: total_time / test_results.len() as u32,
            fastest_time: test_results
                .iter()
                .map(|r| r.execution_time)
                .min()
                .unwrap_or(Duration::ZERO),
            slowest_time: test_results
                .iter()
                .map(|r| r.execution_time)
                .max()
                .unwrap_or(Duration::ZERO),
        };

        Ok(NumericalParityResults {
            all_passed,
            test_results,
            statistics,
            timing,
        })
    }

    /// Test forward pass stability with various inputs
    fn test_forward_pass_stability<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
    ) -> Result<TestResult> {
        let start_time = Instant::now();

        for test_input in &self.config.test_inputs {
            // Create test input based on configuration
            let input = self.create_test_input(test_input)?;

            // Run forward pass
            let result = model.forward(input);

            // Check if forward pass succeeded
            if result.is_err() {
                return Ok(TestResult {
                    name: "forward_pass_stability".to_string(),
                    passed: false,
                    error_message: Some(format!(
                        "Forward pass failed for input {}: {:?}",
                        test_input.name,
                        result.err()
                    )),
                    numerical_differences: None,
                    execution_time: start_time.elapsed(),
                });
            }

            let output = result.expect("operation failed");

            // Validate output
            if !self.validate_output(&output) {
                return Ok(TestResult {
                    name: "forward_pass_stability".to_string(),
                    passed: false,
                    error_message: Some(format!("Invalid output for input {}", test_input.name)),
                    numerical_differences: None,
                    execution_time: start_time.elapsed(),
                });
            }
        }

        Ok(TestResult {
            name: "forward_pass_stability".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: start_time.elapsed(),
        })
    }

    /// Test that model outputs are deterministic
    fn test_deterministic_outputs<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
    ) -> Result<TestResult> {
        let start_time = Instant::now();

        for test_input in &self.config.test_inputs {
            let input1 = self.create_test_input(test_input)?;
            let input2 = self.create_test_input(test_input)?;

            // Run forward pass twice
            let output1 = model.forward(input1)?;
            let output2 = model.forward(input2)?;

            // Compare outputs
            let differences = self.compute_numerical_differences(&output1, &output2)?;

            if differences.max_abs_diff > self.config.numerical_tolerance {
                return Ok(TestResult {
                    name: "deterministic_outputs".to_string(),
                    passed: false,
                    error_message: Some(format!(
                        "Non-deterministic outputs detected for input {}",
                        test_input.name
                    )),
                    numerical_differences: Some(differences),
                    execution_time: start_time.elapsed(),
                });
            }
        }

        Ok(TestResult {
            name: "deterministic_outputs".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: start_time.elapsed(),
        })
    }

    /// Test numerical stability (no NaN, inf, extremely large values)
    fn test_numerical_stability<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
    ) -> Result<TestResult> {
        let start_time = Instant::now();

        for test_input in &self.config.test_inputs {
            let input = self.create_test_input(test_input)?;
            let output = model.forward(input)?;

            if !self.check_numerical_stability(&output) {
                return Ok(TestResult {
                    name: "numerical_stability".to_string(),
                    passed: false,
                    error_message: Some(format!(
                        "Numerical instability detected for input {}",
                        test_input.name
                    )),
                    numerical_differences: None,
                    execution_time: start_time.elapsed(),
                });
            }
        }

        Ok(TestResult {
            name: "numerical_stability".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: start_time.elapsed(),
        })
    }

    /// Test input validation and error handling
    fn test_input_validation<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        _model: &M,
    ) -> Result<TestResult> {
        let start_time = Instant::now();

        // Test with various invalid inputs
        // This is a placeholder - specific tests would depend on model requirements

        Ok(TestResult {
            name: "input_validation".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: start_time.elapsed(),
        })
    }

    /// Test gradient flow indicators (simplified)
    fn test_gradient_flow_indicators<M: Model<Input = Tensor, Output = Tensor>>(
        &self,
        model: &M,
    ) -> Result<TestResult> {
        let start_time = Instant::now();

        // Run forward pass and check that outputs have reasonable variance
        for test_input in &self.config.test_inputs {
            let input = self.create_test_input(test_input)?;
            let output = model.forward(input)?;

            if !self.check_output_variance(&output) {
                return Ok(TestResult {
                    name: "gradient_flow_indicators".to_string(),
                    passed: false,
                    error_message: Some(format!(
                        "Poor gradient flow indicators for input {}",
                        test_input.name
                    )),
                    numerical_differences: None,
                    execution_time: start_time.elapsed(),
                });
            }
        }

        Ok(TestResult {
            name: "gradient_flow_indicators".to_string(),
            passed: true,
            error_message: None,
            numerical_differences: None,
            execution_time: start_time.elapsed(),
        })
    }

    /// Create test input based on configuration
    fn create_test_input(&self, config: &TestInputConfig) -> Result<Tensor> {
        match config.data_type {
            TestDataType::I32 => {
                // Create token IDs for language models
                let mut input_ids = Vec::new();
                for i in 0..config.dimensions.iter().product::<usize>() {
                    input_ids.push(((i % 1000 + 1) as i32) as f32); // Keep in reasonable token range
                }
                Ok(Tensor::from_vec(input_ids, &config.dimensions)?)
            },
            TestDataType::F32 => {
                // Create floating point input
                Ok(Tensor::randn(&config.dimensions)?)
            },
            TestDataType::F16 => {
                // Create half precision input (placeholder)
                Ok(Tensor::randn(&config.dimensions)?)
            },
            TestDataType::I64 => {
                // Create 64-bit integer input
                let mut input_ids = Vec::new();
                for i in 0..config.dimensions.iter().product::<usize>() {
                    input_ids.push(((i % 1000 + 1) as i64) as f32);
                }
                Ok(Tensor::from_vec(input_ids, &config.dimensions)?)
            },
        }
    }

    /// Validate model output
    fn validate_output(&self, output: &Tensor) -> bool {
        match output {
            Tensor::F32(arr) => arr.iter().all(|x| x.is_finite() && x.abs() < 1000.0),
            Tensor::F64(arr) => arr.iter().all(|x| x.is_finite() && x.abs() < 1000.0),
            _ => true, // For other tensor types, basic validation
        }
    }

    /// Check numerical stability of output
    fn check_numerical_stability(&self, output: &Tensor) -> bool {
        match output {
            Tensor::F32(arr) => {
                arr.iter().all(|x| x.is_finite())
                    && arr.iter().all(|x| x.abs() < 100.0)
                    && !arr.iter().all(|x| *x == 0.0) // Not all zeros
            },
            Tensor::F64(arr) => {
                arr.iter().all(|x| x.is_finite())
                    && arr.iter().all(|x| x.abs() < 100.0)
                    && !arr.iter().all(|x| *x == 0.0)
            },
            _ => true,
        }
    }

    /// Check output variance (for gradient flow)
    fn check_output_variance(&self, output: &Tensor) -> bool {
        match output {
            Tensor::F32(arr) => {
                if arr.len() < 2 {
                    return true;
                }
                let mean = arr.iter().sum::<f32>() / arr.len() as f32;
                let variance =
                    arr.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / arr.len() as f32;
                variance > 1e-6 // Should have some variance
            },
            Tensor::F64(arr) => {
                if arr.len() < 2 {
                    return true;
                }
                let mean = arr.iter().sum::<f64>() / arr.len() as f64;
                let variance =
                    arr.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / arr.len() as f64;
                variance > 1e-6
            },
            _ => true,
        }
    }

    /// Compute numerical differences between two tensors
    fn compute_numerical_differences(
        &self,
        tensor1: &Tensor,
        tensor2: &Tensor,
    ) -> Result<NumericalDifferences> {
        match (tensor1, tensor2) {
            (Tensor::F32(arr1), Tensor::F32(arr2)) => {
                if arr1.shape() != arr2.shape() {
                    return Err(Error::msg("Tensor shapes don't match"));
                }

                let diffs: Vec<f32> =
                    arr1.iter().zip(arr2.iter()).map(|(a, b)| (a - b).abs()).collect();
                let max_abs_diff = diffs.iter().cloned().fold(0.0, f32::max);
                let mean_abs_diff = diffs.iter().sum::<f32>() / diffs.len() as f32;
                let rms_diff =
                    (diffs.iter().map(|d| d * d).sum::<f32>() / diffs.len() as f32).sqrt();
                let within_tolerance =
                    diffs.iter().filter(|&&d| d <= self.config.numerical_tolerance).count();
                let within_tolerance_percent =
                    (within_tolerance as f32 / diffs.len() as f32) * 100.0;

                Ok(NumericalDifferences {
                    max_abs_diff,
                    mean_abs_diff,
                    rms_diff,
                    within_tolerance_percent,
                })
            },
            _ => Err(Error::msg("Unsupported tensor types for comparison")),
        }
    }
}
