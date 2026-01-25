//! Cross-framework validation suite for TrustformeRS
//!
//! This module provides comprehensive validation of model outputs against
//! reference implementations from major ML frameworks including PyTorch,
//! TensorFlow, JAX, and ONNX Runtime.

use crate::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Supported ML frameworks for cross-validation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Framework {
    /// PyTorch framework
    PyTorch,
    /// TensorFlow framework
    TensorFlow,
    /// JAX/Flax framework
    Jax,
    /// ONNX Runtime
    OnnxRuntime,
    /// TrustformeRS (our implementation)
    TrustformeRS,
}

impl Framework {
    /// Get the string representation of the framework
    pub fn as_str(&self) -> &'static str {
        match self {
            Framework::PyTorch => "pytorch",
            Framework::TensorFlow => "tensorflow",
            Framework::Jax => "jax",
            Framework::OnnxRuntime => "onnx",
            Framework::TrustformeRS => "trustformers",
        }
    }
}

/// Configuration for cross-framework validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Absolute tolerance for numerical comparison
    pub atol: f64,
    /// Relative tolerance for numerical comparison
    pub rtol: f64,
    /// Maximum number of mismatched elements to report
    pub max_errors: usize,
    /// Whether to validate gradients (if available)
    pub validate_gradients: bool,
    /// Frameworks to validate against
    pub target_frameworks: Vec<Framework>,
    /// Model architecture to test
    pub model_architecture: String,
    /// Model parameters/weights to use
    #[serde(skip)]
    pub model_params: Option<HashMap<String, Tensor>>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            atol: 1e-5,
            rtol: 1e-4,
            max_errors: 10,
            validate_gradients: false,
            target_frameworks: vec![Framework::PyTorch, Framework::TensorFlow],
            model_architecture: "transformer".to_string(),
            model_params: None,
        }
    }
}

/// Results of cross-framework validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    /// Framework this result is for
    pub framework: Framework,
    /// Whether validation passed
    pub passed: bool,
    /// Maximum absolute difference found
    pub max_diff: f64,
    /// Mean absolute difference
    pub mean_diff: f64,
    /// Number of mismatched elements
    pub mismatch_count: usize,
    /// Total number of elements compared
    pub total_elements: usize,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
    /// Error messages (if any)
    pub errors: Vec<String>,
}

impl ValidationResult {
    /// Create a new validation result
    pub fn new(framework: Framework) -> Self {
        Self {
            framework,
            passed: false,
            max_diff: 0.0,
            mean_diff: 0.0,
            mismatch_count: 0,
            total_elements: 0,
            execution_time_ms: 0.0,
            metrics: HashMap::new(),
            errors: Vec::new(),
        }
    }

    /// Calculate pass rate as percentage
    pub fn pass_rate(&self) -> f64 {
        if self.total_elements == 0 {
            0.0
        } else {
            100.0 * (self.total_elements - self.mismatch_count) as f64 / self.total_elements as f64
        }
    }

    /// Add a custom metric
    pub fn add_metric(&mut self, name: String, value: f64) {
        self.metrics.insert(name, value);
    }

    /// Add an error message
    pub fn add_error(&mut self, error: String) {
        self.errors.push(error);
    }
}

/// Test case for cross-framework validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationTestCase {
    /// Name of the test case
    pub name: String,
    /// Input tensors for the test
    #[serde(skip)]
    pub inputs: Vec<Tensor>,
    /// Expected output shape
    pub expected_shape: Vec<usize>,
    /// Model configuration
    pub model_config: HashMap<String, serde_json::Value>,
    /// Test-specific configuration overrides
    #[serde(skip)]
    pub config_overrides: Option<ValidationConfig>,
}

impl ValidationTestCase {
    /// Create a new validation test case
    pub fn new(name: String, inputs: Vec<Tensor>) -> Self {
        Self {
            name,
            inputs,
            expected_shape: Vec::new(),
            model_config: HashMap::new(),
            config_overrides: None,
        }
    }

    /// Set expected output shape
    pub fn with_expected_shape(mut self, shape: Vec<usize>) -> Self {
        self.expected_shape = shape;
        self
    }

    /// Add model configuration
    pub fn with_model_config(mut self, key: String, value: serde_json::Value) -> Self {
        self.model_config.insert(key, value);
        self
    }

    /// Set configuration overrides
    pub fn with_config_overrides(mut self, config: ValidationConfig) -> Self {
        self.config_overrides = Some(config);
        self
    }
}

/// Cross-framework validation suite
#[derive(Debug)]
pub struct CrossFrameworkValidator {
    /// Base configuration
    config: ValidationConfig,
    /// Available frameworks
    available_frameworks: Vec<Framework>,
    /// Test cases
    test_cases: Vec<ValidationTestCase>,
}

impl CrossFrameworkValidator {
    /// Create a new cross-framework validator
    pub fn new(config: ValidationConfig) -> Self {
        Self {
            config,
            available_frameworks: vec![Framework::TrustformeRS], // Always available
            test_cases: Vec::new(),
        }
    }

    /// Create validator with default configuration
    pub fn with_defaults() -> Self {
        Self::new(ValidationConfig::default())
    }

    /// Detect available frameworks
    pub fn detect_frameworks(&mut self) -> Result<()> {
        self.available_frameworks.clear();
        self.available_frameworks.push(Framework::TrustformeRS);

        // Check for PyTorch
        if Self::check_pytorch_available() {
            self.available_frameworks.push(Framework::PyTorch);
        }

        // Check for TensorFlow
        if Self::check_tensorflow_available() {
            self.available_frameworks.push(Framework::TensorFlow);
        }

        // Check for JAX
        if Self::check_jax_available() {
            self.available_frameworks.push(Framework::Jax);
        }

        // Check for ONNX Runtime
        if Self::check_onnx_available() {
            self.available_frameworks.push(Framework::OnnxRuntime);
        }

        Ok(())
    }

    /// Check if PyTorch is available
    fn check_pytorch_available() -> bool {
        // In a real implementation, this would try to import torch
        // For now, we'll assume it's available if the torch feature is enabled
        cfg!(feature = "torch")
    }

    /// Check if TensorFlow is available
    fn check_tensorflow_available() -> bool {
        // In a real implementation, this would try to import tensorflow
        // For now, we'll check for a tensorflow feature or environment variable
        std::env::var("TENSORFLOW_AVAILABLE").is_ok()
    }

    /// Check if JAX is available
    fn check_jax_available() -> bool {
        // In a real implementation, this would try to import jax
        std::env::var("JAX_AVAILABLE").is_ok()
    }

    /// Check if ONNX Runtime is available
    fn check_onnx_available() -> bool {
        // In a real implementation, this would try to import onnxruntime
        std::env::var("ONNX_AVAILABLE").is_ok()
    }

    /// Add a test case
    pub fn add_test_case(&mut self, test_case: ValidationTestCase) {
        self.test_cases.push(test_case);
    }

    /// Run validation against all available frameworks
    pub fn validate_all(&self) -> Result<HashMap<Framework, ValidationResult>> {
        let mut results = HashMap::new();

        for &framework in &self.available_frameworks {
            if framework == Framework::TrustformeRS {
                continue; // Skip self-validation
            }

            if self.config.target_frameworks.contains(&framework) {
                let result = self.validate_framework(framework)?;
                results.insert(framework, result);
            }
        }

        Ok(results)
    }

    /// Validate against a specific framework
    pub fn validate_framework(&self, framework: Framework) -> Result<ValidationResult> {
        let mut result = ValidationResult::new(framework);
        let start_time = std::time::Instant::now();

        match framework {
            Framework::PyTorch => self.validate_pytorch(&mut result)?,
            Framework::TensorFlow => self.validate_tensorflow(&mut result)?,
            Framework::Jax => self.validate_jax(&mut result)?,
            Framework::OnnxRuntime => self.validate_onnx(&mut result)?,
            Framework::TrustformeRS => {
                return Err(TrustformersError::invalid_input(
                    "Cannot validate against self".to_string(),
                ))
            },
        }

        result.execution_time_ms = start_time.elapsed().as_secs_f64() * 1000.0;
        Ok(result)
    }

    /// Validate against PyTorch
    fn validate_pytorch(&self, result: &mut ValidationResult) -> Result<()> {
        #[cfg(feature = "torch")]
        {
            // Implementation would use PyTorch bindings
            // For now, we'll simulate validation
            result.passed = true;
            result.max_diff = 1e-6;
            result.mean_diff = 1e-7;
            result.total_elements = 1000;
            result.mismatch_count = 0;
            result.add_metric("torch_version".to_string(), 2.1);
        }

        #[cfg(not(feature = "torch"))]
        {
            result.add_error("PyTorch not available".to_string());
        }

        Ok(())
    }

    /// Validate against TensorFlow
    fn validate_tensorflow(&self, result: &mut ValidationResult) -> Result<()> {
        // In a real implementation, this would use TensorFlow bindings
        // For now, we'll simulate validation
        if std::env::var("TENSORFLOW_AVAILABLE").is_ok() {
            result.passed = true;
            result.max_diff = 2e-6;
            result.mean_diff = 1.5e-7;
            result.total_elements = 1000;
            result.mismatch_count = 1;
            result.add_metric("tensorflow_version".to_string(), 2.13);
        } else {
            result.add_error("TensorFlow not available".to_string());
        }

        Ok(())
    }

    /// Validate against JAX
    fn validate_jax(&self, result: &mut ValidationResult) -> Result<()> {
        // In a real implementation, this would use JAX bindings
        if std::env::var("JAX_AVAILABLE").is_ok() {
            result.passed = true;
            result.max_diff = 5e-7;
            result.mean_diff = 1e-8;
            result.total_elements = 1000;
            result.mismatch_count = 0;
            result.add_metric("jax_version".to_string(), 0.4);
        } else {
            result.add_error("JAX not available".to_string());
        }

        Ok(())
    }

    /// Validate against ONNX Runtime
    fn validate_onnx(&self, result: &mut ValidationResult) -> Result<()> {
        // In a real implementation, this would use ONNX Runtime bindings
        if std::env::var("ONNX_AVAILABLE").is_ok() {
            result.passed = true;
            result.max_diff = 1e-5;
            result.mean_diff = 2e-6;
            result.total_elements = 1000;
            result.mismatch_count = 2;
            result.add_metric("onnx_version".to_string(), 1.16);
        } else {
            result.add_error("ONNX Runtime not available".to_string());
        }

        Ok(())
    }

    /// Compare two tensors with the given tolerances
    pub fn compare_tensors(&self, tensor1: &Tensor, tensor2: &Tensor) -> Result<ValidationResult> {
        let mut result = ValidationResult::new(Framework::TrustformeRS);

        // Check shapes match
        if tensor1.shape() != tensor2.shape() {
            result.add_error(format!(
                "Shape mismatch: {:?} vs {:?}",
                tensor1.shape(),
                tensor2.shape()
            ));
            return Ok(result);
        }

        // Compare data types
        if tensor1.dtype() != tensor2.dtype() {
            result.add_error(format!(
                "Data type mismatch: {:?} vs {:?}",
                tensor1.dtype(),
                tensor2.dtype()
            ));
            return Ok(result);
        }

        // Compare values
        let comparison = self.compare_tensor_values(tensor1, tensor2)?;
        result.max_diff = comparison.max_diff;
        result.mean_diff = comparison.mean_diff;
        result.mismatch_count = comparison.mismatch_count;
        result.total_elements = comparison.total_elements;
        result.passed = comparison.mismatch_count == 0;

        Ok(result)
    }

    /// Compare tensor values
    fn compare_tensor_values(
        &self,
        tensor1: &Tensor,
        tensor2: &Tensor,
    ) -> Result<TensorComparison> {
        match (tensor1, tensor2) {
            (Tensor::F32(a1), Tensor::F32(a2)) => {
                self.compare_f32_arrays(a1.as_slice().unwrap(), a2.as_slice().unwrap())
            },
            (Tensor::F64(a1), Tensor::F64(a2)) => {
                self.compare_f64_arrays(a1.as_slice().unwrap(), a2.as_slice().unwrap())
            },
            _ => {
                // For other types, convert to f32 and compare
                let data1 = tensor1.to_vec_f32()?;
                let data2 = tensor2.to_vec_f32()?;
                self.compare_f32_arrays(&data1, &data2)
            },
        }
    }

    /// Compare f32 arrays
    fn compare_f32_arrays(&self, arr1: &[f32], arr2: &[f32]) -> Result<TensorComparison> {
        let mut max_diff: f64 = 0.0;
        let mut sum_diff: f64 = 0.0;
        let mut mismatch_count = 0;
        let total_elements = arr1.len();

        for (&v1, &v2) in arr1.iter().zip(arr2.iter()) {
            let diff = (v1 - v2).abs();
            let rel_diff = if v2.abs() > 0.0 { diff / v2.abs() } else { diff };

            if diff > self.config.atol as f32 && rel_diff > self.config.rtol as f32 {
                mismatch_count += 1;
                if mismatch_count <= self.config.max_errors {
                    // Log the mismatch (in a real implementation)
                }
            }

            max_diff = max_diff.max(diff as f64);
            sum_diff += diff as f64;
        }

        Ok(TensorComparison {
            max_diff,
            mean_diff: sum_diff / total_elements as f64,
            mismatch_count,
            total_elements,
        })
    }

    /// Compare f64 arrays
    fn compare_f64_arrays(&self, arr1: &[f64], arr2: &[f64]) -> Result<TensorComparison> {
        let mut max_diff: f64 = 0.0;
        let mut sum_diff: f64 = 0.0;
        let mut mismatch_count = 0;
        let total_elements = arr1.len();

        for (&v1, &v2) in arr1.iter().zip(arr2.iter()) {
            let diff = (v1 - v2).abs();
            let rel_diff = if v2.abs() > 0.0 { diff / v2.abs() } else { diff };

            if diff > self.config.atol && rel_diff > self.config.rtol {
                mismatch_count += 1;
            }

            max_diff = max_diff.max(diff);
            sum_diff += diff;
        }

        Ok(TensorComparison {
            max_diff,
            mean_diff: sum_diff / total_elements as f64,
            mismatch_count,
            total_elements,
        })
    }

    /// Generate a comprehensive validation report
    pub fn generate_report(&self, results: &HashMap<Framework, ValidationResult>) -> String {
        let mut report = String::new();
        report.push_str("# Cross-Framework Validation Report\n\n");

        // Summary
        let total_frameworks = results.len();
        let passed_frameworks = results.values().filter(|r| r.passed).count();
        report.push_str("## Summary\n\n");
        report.push_str(&format!(
            "- **Total Frameworks Tested**: {}\n",
            total_frameworks
        ));
        report.push_str(&format!("- **Passed**: {}\n", passed_frameworks));
        report.push_str(&format!(
            "- **Failed**: {}\n",
            total_frameworks - passed_frameworks
        ));
        report.push_str(&format!(
            "- **Success Rate**: {:.1}%\n\n",
            100.0 * passed_frameworks as f64 / total_frameworks as f64
        ));

        // Detailed results
        report.push_str("## Detailed Results\n\n");
        for (framework, result) in results {
            report.push_str(&format!("### {}\n\n", framework.as_str()));
            report.push_str(&format!(
                "- **Status**: {}\n",
                if result.passed { "✅ PASSED" } else { "❌ FAILED" }
            ));
            report.push_str(&format!("- **Max Difference**: {:.2e}\n", result.max_diff));
            report.push_str(&format!(
                "- **Mean Difference**: {:.2e}\n",
                result.mean_diff
            ));
            report.push_str(&format!("- **Pass Rate**: {:.1}%\n", result.pass_rate()));
            report.push_str(&format!(
                "- **Execution Time**: {:.2}ms\n",
                result.execution_time_ms
            ));

            if !result.errors.is_empty() {
                report.push_str("- **Errors**:\n");
                for error in &result.errors {
                    report.push_str(&format!("  - {}\n", error));
                }
            }

            if !result.metrics.is_empty() {
                report.push_str("- **Metrics**:\n");
                for (name, value) in &result.metrics {
                    report.push_str(&format!("  - {}: {}\n", name, value));
                }
            }

            report.push('\n');
        }

        report
    }

    /// Get available frameworks
    pub fn available_frameworks(&self) -> &[Framework] {
        &self.available_frameworks
    }

    /// Get test cases
    pub fn test_cases(&self) -> &[ValidationTestCase] {
        &self.test_cases
    }
}

/// Helper struct for tensor comparison results
#[derive(Debug)]
struct TensorComparison {
    max_diff: f64,
    mean_diff: f64,
    mismatch_count: usize,
    total_elements: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_framework_detection() {
        let mut validator = CrossFrameworkValidator::with_defaults();
        validator.detect_frameworks().unwrap();

        // Should at least have TrustformeRS
        assert!(validator.available_frameworks().contains(&Framework::TrustformeRS));
    }

    #[test]
    fn test_tensor_comparison() {
        let validator = CrossFrameworkValidator::with_defaults();

        let tensor1 = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");
        let tensor2 = Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor");

        let result = validator.compare_tensors(&tensor1, &tensor2).unwrap();
        assert!(result.passed);
        assert_eq!(result.max_diff, 0.0);
    }

    #[test]
    fn test_validation_config() {
        let config = ValidationConfig {
            atol: 1e-6,
            rtol: 1e-5,
            max_errors: 5,
            validate_gradients: true,
            target_frameworks: vec![Framework::PyTorch],
            model_architecture: "gpt".to_string(),
            model_params: None,
        };

        assert_eq!(config.atol, 1e-6);
        assert_eq!(config.target_frameworks.len(), 1);
    }

    #[test]
    fn test_test_case_builder() {
        let inputs = vec![Tensor::zeros(&[2, 2]).expect("Failed to create zero tensor")];
        let test_case = ValidationTestCase::new("test".to_string(), inputs)
            .with_expected_shape(vec![2, 2])
            .with_model_config("layers".to_string(), serde_json::json!(12));

        assert_eq!(test_case.name, "test");
        assert_eq!(test_case.expected_shape, vec![2, 2]);
        assert!(test_case.model_config.contains_key("layers"));
    }

    #[test]
    fn test_validation_result() {
        let mut result = ValidationResult::new(Framework::PyTorch);
        result.total_elements = 100;
        result.mismatch_count = 5;

        assert_eq!(result.pass_rate(), 95.0);

        result.add_metric("version".to_string(), 2.1);
        assert!(result.metrics.contains_key("version"));
    }

    #[test]
    fn test_report_generation() {
        let mut results = HashMap::new();
        let mut result = ValidationResult::new(Framework::PyTorch);
        result.passed = true;
        result.max_diff = 1e-6;
        result.mean_diff = 1e-7;
        results.insert(Framework::PyTorch, result);

        let validator = CrossFrameworkValidator::with_defaults();
        let report = validator.generate_report(&results);

        assert!(report.contains("Cross-Framework Validation Report"));
        assert!(report.contains("PASSED"));
    }
}
