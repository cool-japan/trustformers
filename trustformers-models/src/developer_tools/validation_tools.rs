//! Validation Tools
//!
//! Tools for validating model implementations, configurations, and outputs.

use anyhow::Result;
use scirs2_core::ndarray::ArrayD; // SciRS2 Integration Policy
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Numerical tolerance for comparisons
    pub numerical_tolerance: f64,
    /// Whether to check for NaN values
    pub check_nan: bool,
    /// Whether to check for infinite values
    pub check_infinite: bool,
    /// Expected output shapes
    pub expected_shapes: HashMap<String, Vec<usize>>,
    /// Value range constraints
    pub value_ranges: HashMap<String, (f64, f64)>,
    /// Performance thresholds
    pub performance_thresholds: PerformanceThresholds,
}

/// Performance validation thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThresholds {
    /// Maximum forward pass time in milliseconds
    pub max_forward_time_ms: f64,
    /// Maximum memory usage in MB
    pub max_memory_mb: f64,
    /// Minimum throughput (samples/second)
    pub min_throughput: f64,
}

/// Validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Whether validation passed
    pub passed: bool,
    /// Validation errors
    pub errors: Vec<String>,
    /// Validation warnings
    pub warnings: Vec<String>,
    /// Performance metrics
    pub performance_metrics: Option<PerformanceMetrics>,
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Forward pass time in milliseconds
    pub forward_time_ms: f64,
    /// Memory usage in MB
    pub memory_usage_mb: f64,
    /// Throughput in samples/second
    pub throughput: f64,
    /// Parameter count
    pub parameter_count: usize,
}

/// Model validator
pub struct ModelValidator {
    config: ValidationConfig,
}

impl ModelValidator {
    /// Create a new model validator
    pub fn new(config: ValidationConfig) -> Self {
        Self { config }
    }

    /// Validate tensor output
    pub fn validate_tensor(&self, name: &str, tensor: &Tensor) -> ValidationResult {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: None,
        };

        // Validate tensor shape
        if let Some(expected_shape) = self.config.expected_shapes.get(name) {
            let actual_shape = match tensor {
                Tensor::F32(arr) => arr.shape().to_vec(),
                Tensor::F64(arr) => arr.shape().to_vec(),
                Tensor::I64(arr) => arr.shape().to_vec(),
                _ => {
                    result.errors.push("Unsupported tensor type for shape validation".to_string());
                    result.passed = false;
                    return result;
                },
            };

            if actual_shape != *expected_shape {
                result.errors.push(format!(
                    "Shape mismatch for '{}': expected {:?}, got {:?}",
                    name, expected_shape, actual_shape
                ));
                result.passed = false;
            }
        }

        // Validate numerical values
        match tensor {
            Tensor::F32(arr) => {
                self.validate_f32_values(name, arr, &mut result);
            },
            Tensor::F64(arr) => {
                self.validate_f64_values(name, arr, &mut result);
            },
            Tensor::I64(_) => {
                // Integer tensors are generally safe from NaN/Inf issues
            },
            _ => {
                // Other tensor types (F16, BF16, Complex) not supported for validation yet
            },
        }

        result
    }

    /// Validate F32 tensor values
    fn validate_f32_values(&self, name: &str, arr: &ArrayD<f32>, result: &mut ValidationResult) {
        // Check for NaN values
        if self.config.check_nan {
            let nan_count = arr.iter().filter(|&&x| x.is_nan()).count();
            if nan_count > 0 {
                result.errors.push(format!(
                    "Found {} NaN values in tensor '{}'",
                    nan_count, name
                ));
                result.passed = false;
            }
        }

        // Check for infinite values
        if self.config.check_infinite {
            let inf_count = arr.iter().filter(|&&x| x.is_infinite()).count();
            if inf_count > 0 {
                result.errors.push(format!(
                    "Found {} infinite values in tensor '{}'",
                    inf_count, name
                ));
                result.passed = false;
            }
        }

        // Check value ranges
        if let Some((min_val, max_val)) = self.config.value_ranges.get(name) {
            let actual_min = arr.iter().cloned().fold(f32::INFINITY, f32::min) as f64;
            let actual_max = arr.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;

            if actual_min < *min_val {
                result.warnings.push(format!(
                    "Values in '{}' below expected minimum: {} < {}",
                    name, actual_min, min_val
                ));
            }

            if actual_max > *max_val {
                result.warnings.push(format!(
                    "Values in '{}' above expected maximum: {} > {}",
                    name, actual_max, max_val
                ));
            }
        }
    }

    /// Validate F64 tensor values
    fn validate_f64_values(&self, name: &str, arr: &ArrayD<f64>, result: &mut ValidationResult) {
        // Check for NaN values
        if self.config.check_nan {
            let nan_count = arr.iter().filter(|&&x| x.is_nan()).count();
            if nan_count > 0 {
                result.errors.push(format!(
                    "Found {} NaN values in tensor '{}'",
                    nan_count, name
                ));
                result.passed = false;
            }
        }

        // Check for infinite values
        if self.config.check_infinite {
            let inf_count = arr.iter().filter(|&&x| x.is_infinite()).count();
            if inf_count > 0 {
                result.errors.push(format!(
                    "Found {} infinite values in tensor '{}'",
                    inf_count, name
                ));
                result.passed = false;
            }
        }

        // Check value ranges
        if let Some((min_val, max_val)) = self.config.value_ranges.get(name) {
            let actual_min = arr.iter().cloned().fold(f64::INFINITY, f64::min);
            let actual_max = arr.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

            if actual_min < *min_val {
                result.warnings.push(format!(
                    "Values in '{}' below expected minimum: {} < {}",
                    name, actual_min, min_val
                ));
            }

            if actual_max > *max_val {
                result.warnings.push(format!(
                    "Values in '{}' above expected maximum: {} > {}",
                    name, actual_max, max_val
                ));
            }
        }
    }

    /// Validate model configuration
    pub fn validate_config<T>(&self, _config: &T) -> ValidationResult
    where
        T: std::fmt::Debug,
    {
        // This would typically use reflection or custom traits to validate config
        // For now, we provide a basic framework

        ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: None,
        }
    }

    /// Validate model performance
    pub fn validate_performance<F>(&self, mut model_fn: F) -> ValidationResult
    where
        F: FnMut() -> Result<Tensor>,
    {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: None,
        };

        // Warmup runs
        for _ in 0..3 {
            if let Err(e) = model_fn() {
                result.errors.push(format!("Warmup run failed: {}", e));
                result.passed = false;
                return result;
            }
        }

        // Timed runs
        let mut durations = Vec::new();
        let num_runs = 10;

        for _ in 0..num_runs {
            let start = std::time::Instant::now();
            if let Err(e) = model_fn() {
                result.errors.push(format!("Timed run failed: {}", e));
                result.passed = false;
                return result;
            }
            durations.push(start.elapsed().as_millis() as f64);
        }

        // Calculate performance metrics
        let avg_time_ms = durations.iter().sum::<f64>() / durations.len() as f64;
        let throughput = 1000.0 / avg_time_ms; // samples per second

        let performance_metrics = PerformanceMetrics {
            forward_time_ms: avg_time_ms,
            memory_usage_mb: 0.0, // Would need platform-specific memory measurement
            throughput,
            parameter_count: 0, // Would need model introspection
        };

        // Check against thresholds
        if avg_time_ms > self.config.performance_thresholds.max_forward_time_ms {
            result.errors.push(format!(
                "Forward pass too slow: {:.2}ms > {:.2}ms",
                avg_time_ms, self.config.performance_thresholds.max_forward_time_ms
            ));
            result.passed = false;
        }

        if throughput < self.config.performance_thresholds.min_throughput {
            result.warnings.push(format!(
                "Throughput below threshold: {:.2} < {:.2} samples/sec",
                throughput, self.config.performance_thresholds.min_throughput
            ));
        }

        result.performance_metrics = Some(performance_metrics);
        result
    }

    /// Compare two tensors for numerical equality
    pub fn compare_tensors(
        &self,
        name: &str,
        expected: &Tensor,
        actual: &Tensor,
    ) -> ValidationResult {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: None,
        };

        match (expected, actual) {
            (Tensor::F32(exp_arr), Tensor::F32(act_arr)) => {
                if exp_arr.shape() != act_arr.shape() {
                    result.errors.push(format!(
                        "Shape mismatch for '{}': expected {:?}, got {:?}",
                        name,
                        exp_arr.shape(),
                        act_arr.shape()
                    ));
                    result.passed = false;
                    return result;
                }

                let mut max_diff = 0.0f64;
                let mut total_diff = 0.0f64;
                let mut count = 0;

                for (exp_val, act_val) in exp_arr.iter().zip(act_arr.iter()) {
                    let diff = (*exp_val - *act_val).abs() as f64;
                    max_diff = max_diff.max(diff);
                    total_diff += diff;
                    count += 1;

                    if diff > self.config.numerical_tolerance {
                        result.errors.push(format!(
                            "Value difference exceeds tolerance for '{}': {} vs {} (diff: {})",
                            name, exp_val, act_val, diff
                        ));
                        result.passed = false;
                    }
                }

                let avg_diff = total_diff / count as f64;
                if avg_diff > self.config.numerical_tolerance * 0.1 {
                    result.warnings.push(format!(
                        "Average difference for '{}' is high: {:.6}",
                        name, avg_diff
                    ));
                }
            },
            (Tensor::I64(exp_arr), Tensor::I64(act_arr)) => {
                if exp_arr.shape() != act_arr.shape() {
                    result.errors.push(format!(
                        "Shape mismatch for '{}': expected {:?}, got {:?}",
                        name,
                        exp_arr.shape(),
                        act_arr.shape()
                    ));
                    result.passed = false;
                    return result;
                }

                for (exp_val, act_val) in exp_arr.iter().zip(act_arr.iter()) {
                    if exp_val != act_val {
                        result.errors.push(format!(
                            "Integer value mismatch for '{}': {} vs {}",
                            name, exp_val, act_val
                        ));
                        result.passed = false;
                    }
                }
            },
            _ => {
                result.errors.push(format!(
                    "Tensor type mismatch for '{}': different tensor types",
                    name
                ));
                result.passed = false;
            },
        }

        result
    }

    /// Validate model against reference implementation
    pub fn validate_against_reference<M, R>(
        &self,
        model: M,
        reference: R,
        test_inputs: Vec<Tensor>,
    ) -> ValidationResult
    where
        M: Fn(&Tensor) -> Result<Tensor>,
        R: Fn(&Tensor) -> Result<Tensor>,
    {
        let mut result = ValidationResult {
            passed: true,
            errors: Vec::new(),
            warnings: Vec::new(),
            performance_metrics: None,
        };

        for (i, input) in test_inputs.iter().enumerate() {
            let model_output = match model(input) {
                Ok(output) => output,
                Err(e) => {
                    result.errors.push(format!("Model failed on input {}: {}", i, e));
                    result.passed = false;
                    continue;
                },
            };

            let reference_output = match reference(input) {
                Ok(output) => output,
                Err(e) => {
                    result.errors.push(format!("Reference failed on input {}: {}", i, e));
                    result.passed = false;
                    continue;
                },
            };

            let comparison_result =
                self.compare_tensors(&format!("output_{}", i), &reference_output, &model_output);

            if !comparison_result.passed {
                result.errors.extend(comparison_result.errors);
                result.passed = false;
            }
            result.warnings.extend(comparison_result.warnings);
        }

        result
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            numerical_tolerance: 1e-5,
            check_nan: true,
            check_infinite: true,
            expected_shapes: HashMap::new(),
            value_ranges: HashMap::new(),
            performance_thresholds: PerformanceThresholds {
                max_forward_time_ms: 1000.0,
                max_memory_mb: 1024.0,
                min_throughput: 10.0,
            },
        }
    }
}

/// Validation utilities
pub struct ValidationUtils;

impl ValidationUtils {
    /// Create a strict validation config
    pub fn strict_config() -> ValidationConfig {
        ValidationConfig {
            numerical_tolerance: 1e-6,
            check_nan: true,
            check_infinite: true,
            expected_shapes: HashMap::new(),
            value_ranges: HashMap::new(),
            performance_thresholds: PerformanceThresholds {
                max_forward_time_ms: 500.0,
                max_memory_mb: 512.0,
                min_throughput: 50.0,
            },
        }
    }

    /// Create a lenient validation config
    pub fn lenient_config() -> ValidationConfig {
        ValidationConfig {
            numerical_tolerance: 1e-3,
            check_nan: true,
            check_infinite: true,
            expected_shapes: HashMap::new(),
            value_ranges: HashMap::new(),
            performance_thresholds: PerformanceThresholds {
                max_forward_time_ms: 5000.0,
                max_memory_mb: 4096.0,
                min_throughput: 1.0,
            },
        }
    }

    /// Create test tensors for validation
    pub fn create_test_tensors() -> Result<Vec<Tensor>> {
        Ok(vec![
            Tensor::zeros(&[1, 10])?,
            Tensor::ones(&[2, 20])?,
            Tensor::randn(&[4, 32])?,
            Tensor::randn(&[8, 64])?, // Using randn instead of rand
        ])
    }

    /// Generate validation report
    pub fn generate_report(results: &[ValidationResult]) -> String {
        let mut report = String::new();
        report.push_str("# Validation Report\n\n");

        let total_tests = results.len();
        let passed_tests = results.iter().filter(|r| r.passed).count();
        let failed_tests = total_tests - passed_tests;

        report.push_str("## Summary\n\n");
        report.push_str(&format!("- Total tests: {}\n", total_tests));
        report.push_str(&format!("- Passed: {}\n", passed_tests));
        report.push_str(&format!("- Failed: {}\n\n", failed_tests));

        if failed_tests > 0 {
            report.push_str("## Errors\n\n");
            for (i, result) in results.iter().enumerate() {
                if !result.passed {
                    report.push_str(&format!("### Test {}\n\n", i + 1));
                    for error in &result.errors {
                        report.push_str(&format!("- ❌ {}\n", error));
                    }
                    report.push('\n');
                }
            }
        }

        let total_warnings: usize = results.iter().map(|r| r.warnings.len()).sum();
        if total_warnings > 0 {
            report.push_str("## Warnings\n\n");
            for (i, result) in results.iter().enumerate() {
                if !result.warnings.is_empty() {
                    report.push_str(&format!("### Test {}\n\n", i + 1));
                    for warning in &result.warnings {
                        report.push_str(&format!("- ⚠️ {}\n", warning));
                    }
                    report.push('\n');
                }
            }
        }

        // Performance metrics
        let performance_results: Vec<_> =
            results.iter().filter_map(|r| r.performance_metrics.as_ref()).collect();

        if !performance_results.is_empty() {
            report.push_str("## Performance Metrics\n\n");
            let avg_time: f64 = performance_results.iter().map(|p| p.forward_time_ms).sum::<f64>()
                / performance_results.len() as f64;
            let avg_throughput: f64 = performance_results.iter().map(|p| p.throughput).sum::<f64>()
                / performance_results.len() as f64;

            report.push_str(&format!("- Average forward time: {:.2} ms\n", avg_time));
            report.push_str(&format!(
                "- Average throughput: {:.2} samples/sec\n",
                avg_throughput
            ));
        }

        report
    }
}

/// Batch validator for running multiple validations
pub struct BatchValidator {
    validators: Vec<ModelValidator>,
}

impl BatchValidator {
    /// Create a new batch validator
    pub fn new() -> Self {
        Self {
            validators: Vec::new(),
        }
    }

    /// Add a validator to the batch
    pub fn add_validator(&mut self, validator: ModelValidator) {
        self.validators.push(validator);
    }

    /// Run all validators on a tensor
    pub fn validate_tensor_batch(&self, name: &str, tensor: &Tensor) -> Vec<ValidationResult> {
        self.validators
            .iter()
            .map(|validator| validator.validate_tensor(name, tensor))
            .collect()
    }

    /// Generate combined report
    pub fn generate_batch_report(&self, results: &[Vec<ValidationResult>]) -> String {
        let mut report = String::new();
        report.push_str("# Batch Validation Report\n\n");

        for (validator_idx, validator_results) in results.iter().enumerate() {
            report.push_str(&format!("## Validator {}\n\n", validator_idx + 1));
            report.push_str(&ValidationUtils::generate_report(validator_results));
            report.push('\n');
        }

        report
    }
}

impl Default for BatchValidator {
    fn default() -> Self {
        Self::new()
    }
}
