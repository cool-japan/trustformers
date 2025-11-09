//! Numerical gradient checking for automatic differentiation validation.
//!
//! This module provides utilities to validate automatic differentiation implementations
//! by comparing analytical gradients computed by the autodiff engine with numerical
//! gradients computed using finite differences.

#![allow(unused_variables)] // Gradient checker with test parameters

use crate::autodiff::{AutodiffEngine, Variable};
use crate::errors::{Result, TrustformersError};
use crate::tensor::Tensor;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Configuration for gradient checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCheckConfig {
    /// Step size for finite differences (epsilon)
    pub epsilon: f64,
    /// Relative tolerance for gradient comparison
    pub relative_tolerance: f64,
    /// Absolute tolerance for gradient comparison
    pub absolute_tolerance: f64,
    /// Whether to use centered finite differences (more accurate but 2x cost)
    pub use_centered_differences: bool,
    /// Maximum number of elements to check (for large tensors)
    pub max_elements_to_check: Option<usize>,
    /// Whether to check gradients element-wise or only the norm
    pub check_elementwise: bool,
    /// Whether to print detailed comparison results
    pub verbose: bool,
}

impl Default for GradientCheckConfig {
    fn default() -> Self {
        Self {
            epsilon: 1e-5,
            relative_tolerance: 1e-3,
            absolute_tolerance: 1e-5,
            use_centered_differences: true,
            max_elements_to_check: Some(1000),
            check_elementwise: true,
            verbose: false,
        }
    }
}

/// Result of gradient checking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCheckResult {
    /// Whether the gradient check passed
    pub passed: bool,
    /// Maximum relative error found
    pub max_relative_error: f64,
    /// Maximum absolute error found
    pub max_absolute_error: f64,
    /// Number of elements checked
    pub elements_checked: usize,
    /// Number of elements that failed the check
    pub failed_elements: usize,
    /// Detailed error information per element (if verbose)
    pub element_errors: Option<Vec<ElementError>>,
    /// Summary statistics
    pub statistics: GradientCheckStats,
}

/// Detailed error information for a specific element
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElementError {
    /// Index of the element
    pub index: Vec<usize>,
    /// Analytical gradient value
    pub analytical: f64,
    /// Numerical gradient value
    pub numerical: f64,
    /// Relative error
    pub relative_error: f64,
    /// Absolute error
    pub absolute_error: f64,
}

/// Statistics about the gradient check
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GradientCheckStats {
    /// Mean relative error
    pub mean_relative_error: f64,
    /// Standard deviation of relative errors
    pub std_relative_error: f64,
    /// Mean absolute error
    pub mean_absolute_error: f64,
    /// Standard deviation of absolute errors
    pub std_absolute_error: f64,
    /// Percentage of elements that passed
    pub pass_percentage: f64,
}

/// Numerical gradient checker
pub struct GradientChecker {
    config: GradientCheckConfig,
    engine: Arc<AutodiffEngine>,
}

impl GradientChecker {
    /// Create a new gradient checker with default configuration
    pub fn new(engine: Arc<AutodiffEngine>) -> Self {
        Self {
            config: GradientCheckConfig::default(),
            engine,
        }
    }

    /// Create a new gradient checker with custom configuration
    pub fn with_config(engine: Arc<AutodiffEngine>, config: GradientCheckConfig) -> Self {
        Self { config, engine }
    }

    /// Check gradients for a scalar function
    ///
    /// # Arguments
    ///
    /// * `f` - Function that takes input variables and returns a scalar loss
    /// * `inputs` - Input variables to compute gradients for
    ///
    /// # Returns
    ///
    /// A map from variable names to gradient check results
    pub fn check_gradients<F>(
        &self,
        f: F,
        inputs: &HashMap<String, Variable>,
    ) -> Result<HashMap<String, GradientCheckResult>>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable> + Clone,
    {
        let mut results = HashMap::new();

        for (name, input_var) in inputs {
            if self.config.verbose {
                println!("Checking gradients for variable: {}", name);
            }

            let result = self.check_single_variable_gradient(f.clone(), inputs, name)?;
            results.insert(name.clone(), result);

            if self.config.verbose {
                let result = &results[name];
                println!(
                    "  Passed: {}, Max Rel Error: {:.2e}, Max Abs Error: {:.2e}",
                    result.passed, result.max_relative_error, result.max_absolute_error
                );
            }
        }

        Ok(results)
    }

    /// Check gradients for a single variable
    fn check_single_variable_gradient<F>(
        &self,
        f: F,
        inputs: &HashMap<String, Variable>,
        target_var_name: &str,
    ) -> Result<GradientCheckResult>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable> + Clone,
    {
        let target_var = inputs.get(target_var_name).ok_or_else(|| {
            TrustformersError::autodiff_error(format!(
                "Variable '{}' not found in inputs",
                target_var_name
            ))
        })?;

        // Compute analytical gradient
        let analytical_grad =
            self.compute_analytical_gradient(f.clone(), inputs, target_var_name)?;

        // Compute numerical gradient
        let numerical_grad = self.compute_numerical_gradient(f, inputs, target_var_name)?;

        // Compare gradients
        self.compare_gradients(&analytical_grad, &numerical_grad, target_var_name)
    }

    /// Compute analytical gradient using autodiff
    fn compute_analytical_gradient<F>(
        &self,
        f: F,
        inputs: &HashMap<String, Variable>,
        target_var_name: &str,
    ) -> Result<Tensor>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable>,
    {
        // Set up gradient computation
        let mut gradient_inputs = inputs.clone();
        for (_, var) in gradient_inputs.iter_mut() {
            var.set_requires_grad(true);
        }

        // Forward pass
        let output = f(&gradient_inputs)?;

        // Backward pass
        self.engine.backward(&output, None)?;

        // Extract gradient for target variable
        let target_var = &gradient_inputs[target_var_name];
        let grad = target_var.grad()?.ok_or_else(|| {
            TrustformersError::autodiff_error(format!(
                "No gradient computed for variable '{}'",
                target_var_name
            ))
        })?;

        Ok(grad)
    }

    /// Compute numerical gradient using finite differences
    fn compute_numerical_gradient<F>(
        &self,
        f: F,
        inputs: &HashMap<String, Variable>,
        target_var_name: &str,
    ) -> Result<Tensor>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable>,
    {
        let target_var = &inputs[target_var_name];
        let original_data = target_var.data()?.clone();
        let shape = original_data.shape();

        // Create output tensor for numerical gradients
        let mut numerical_grad_data = Vec::new();

        // Determine indices to check
        let total_elements = original_data.len();
        let indices_to_check: Vec<usize> =
            if let Some(max_elements) = self.config.max_elements_to_check {
                if total_elements <= max_elements {
                    (0..total_elements).collect()
                } else {
                    // Sample evenly distributed indices
                    let step = total_elements / max_elements;
                    (0..total_elements).step_by(step).take(max_elements).collect()
                }
            } else {
                (0..total_elements).collect()
            };

        // Compute numerical gradients for each element
        for flat_idx in 0..total_elements {
            let numerical_grad = if indices_to_check.contains(&flat_idx) {
                self.compute_numerical_gradient_single_element(
                    &f,
                    inputs,
                    target_var_name,
                    &original_data,
                    flat_idx,
                )?
            } else {
                0.0 // Skip elements not being checked
            };
            numerical_grad_data.push(numerical_grad);
        }

        Tensor::from_vec(numerical_grad_data, &shape)
    }

    /// Compute numerical gradient for a single element using finite differences
    fn compute_numerical_gradient_single_element<F>(
        &self,
        f: &F,
        inputs: &HashMap<String, Variable>,
        target_var_name: &str,
        original_data: &Tensor,
        element_index: usize,
    ) -> Result<f32>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable>,
    {
        let epsilon = self.config.epsilon as f32;

        if self.config.use_centered_differences {
            // Centered differences: (f(x+h) - f(x-h)) / (2*h)
            let f_plus = self.evaluate_with_perturbed_element(
                f,
                inputs,
                target_var_name,
                original_data,
                element_index,
                epsilon,
            )?;
            let f_minus = self.evaluate_with_perturbed_element(
                f,
                inputs,
                target_var_name,
                original_data,
                element_index,
                -epsilon,
            )?;
            Ok((f_plus - f_minus) / (2.0 * epsilon))
        } else {
            // Forward differences: (f(x+h) - f(x)) / h
            let f_original = self.evaluate_function(f, inputs)?;
            let f_plus = self.evaluate_with_perturbed_element(
                f,
                inputs,
                target_var_name,
                original_data,
                element_index,
                epsilon,
            )?;
            Ok((f_plus - f_original) / epsilon)
        }
    }

    /// Evaluate function with a perturbed element
    fn evaluate_with_perturbed_element<F>(
        &self,
        f: &F,
        inputs: &HashMap<String, Variable>,
        target_var_name: &str,
        original_data: &Tensor,
        element_index: usize,
        perturbation: f32,
    ) -> Result<f32>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable>,
    {
        // Create perturbed tensor
        let mut perturbed_data = original_data.clone();
        let flat_indices = self.flat_index_to_multi_index(element_index, &original_data.shape())?;

        // Add perturbation to the specific element
        match &mut perturbed_data {
            Tensor::F32(arr) => {
                let current_val = arr[flat_indices.as_slice()];
                arr[flat_indices.as_slice()] = current_val + perturbation;
            },
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Numerical gradient checking only supports F32 tensors",
                    "gradient_check",
                ))
            },
        }

        // Create new inputs with perturbed variable
        let mut perturbed_inputs = inputs.clone();
        let perturbed_var = Variable::from_tensor(perturbed_data);
        perturbed_inputs.insert(target_var_name.to_string(), perturbed_var);

        // Evaluate function
        self.evaluate_function(f, &perturbed_inputs)
    }

    /// Evaluate function and return scalar output
    fn evaluate_function<F>(&self, f: &F, inputs: &HashMap<String, Variable>) -> Result<f32>
    where
        F: Fn(&HashMap<String, Variable>) -> Result<Variable>,
    {
        let output = f(inputs)?;
        let output_data = output.data();

        // Extract scalar value
        match output_data {
            Ok(Tensor::F32(arr)) => {
                if arr.len() != 1 {
                    return Err(TrustformersError::autodiff_error(
                        "Function must return a scalar (single element tensor)".into(),
                    ));
                }
                Ok(*arr.iter().next().unwrap())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Function output must be F32 tensor",
                "evaluate_function",
            )),
        }
    }

    /// Convert flat index to multi-dimensional index
    fn flat_index_to_multi_index(&self, flat_index: usize, shape: &[usize]) -> Result<Vec<usize>> {
        let mut indices = Vec::new();
        let mut remaining = flat_index;

        for &dim_size in shape.iter().rev() {
            indices.push(remaining % dim_size);
            remaining /= dim_size;
        }

        indices.reverse();
        Ok(indices)
    }

    /// Compare analytical and numerical gradients
    fn compare_gradients(
        &self,
        analytical: &Tensor,
        numerical: &Tensor,
        var_name: &str,
    ) -> Result<GradientCheckResult> {
        if analytical.shape() != numerical.shape() {
            return Err(TrustformersError::autodiff_error(format!(
                "Gradient shape mismatch for variable '{}': {:?} vs {:?}",
                var_name,
                analytical.shape(),
                numerical.shape()
            )));
        }

        let (analytical_data, numerical_data) = match (analytical, numerical) {
            (Tensor::F32(a), Tensor::F32(n)) => (a, n),
            _ => {
                return Err(TrustformersError::tensor_op_error(
                    "Gradient comparison only supports F32 tensors",
                    "compare_gradients",
                ))
            },
        };

        let mut relative_errors = Vec::new();
        let mut absolute_errors = Vec::new();
        let mut element_errors = if self.config.verbose { Some(Vec::new()) } else { None };
        let mut failed_count = 0;
        let mut max_rel_error: f32 = 0.0;
        let mut max_abs_error: f32 = 0.0;

        // Compare element-wise
        let shape = analytical.shape();
        let indices_to_check = self.get_indices_to_check(analytical.len());

        for flat_idx in &indices_to_check {
            let multi_idx = self.flat_index_to_multi_index(*flat_idx, &shape)?;
            let analytical_val = analytical_data[multi_idx.as_slice()] as f64;
            let numerical_val = numerical_data[multi_idx.as_slice()] as f64;

            let abs_error = (analytical_val - numerical_val).abs();
            let rel_error = if numerical_val.abs() > 1e-10 {
                abs_error / numerical_val.abs()
            } else if analytical_val.abs() > 1e-10 {
                abs_error / analytical_val.abs()
            } else {
                0.0
            };

            relative_errors.push(rel_error);
            absolute_errors.push(abs_error);

            max_rel_error = max_rel_error.max(rel_error as f32);
            max_abs_error = max_abs_error.max(abs_error as f32);

            let element_passed = rel_error <= self.config.relative_tolerance
                || abs_error <= self.config.absolute_tolerance;

            if !element_passed {
                failed_count += 1;
            }

            if let Some(ref mut errors) = element_errors {
                if !element_passed || self.config.verbose {
                    errors.push(ElementError {
                        index: multi_idx,
                        analytical: analytical_val,
                        numerical: numerical_val,
                        relative_error: rel_error,
                        absolute_error: abs_error,
                    });
                }
            }
        }

        let elements_checked = indices_to_check.len();
        let passed = failed_count == 0;

        // Compute statistics
        let mean_rel_error = relative_errors.iter().sum::<f64>() / relative_errors.len() as f64;
        let mean_abs_error = absolute_errors.iter().sum::<f64>() / absolute_errors.len() as f64;

        let variance_rel =
            relative_errors.iter().map(|x| (x - mean_rel_error).powi(2)).sum::<f64>()
                / relative_errors.len() as f64;
        let std_rel_error = variance_rel.sqrt();

        let variance_abs =
            absolute_errors.iter().map(|x| (x - mean_abs_error).powi(2)).sum::<f64>()
                / absolute_errors.len() as f64;
        let std_abs_error = variance_abs.sqrt();

        let pass_percentage =
            (elements_checked - failed_count) as f64 / elements_checked as f64 * 100.0;

        Ok(GradientCheckResult {
            passed,
            max_relative_error: max_rel_error as f64,
            max_absolute_error: max_abs_error as f64,
            elements_checked,
            failed_elements: failed_count,
            element_errors,
            statistics: GradientCheckStats {
                mean_relative_error: mean_rel_error,
                std_relative_error: std_rel_error,
                mean_absolute_error: mean_abs_error,
                std_absolute_error: std_abs_error,
                pass_percentage,
            },
        })
    }

    /// Get indices to check based on configuration
    fn get_indices_to_check(&self, total_elements: usize) -> Vec<usize> {
        if let Some(max_elements) = self.config.max_elements_to_check {
            if total_elements <= max_elements {
                (0..total_elements).collect()
            } else {
                let step = total_elements / max_elements;
                (0..total_elements).step_by(step).take(max_elements).collect()
            }
        } else {
            (0..total_elements).collect()
        }
    }

    /// Print detailed gradient check report
    pub fn print_report(&self, results: &HashMap<String, GradientCheckResult>) {
        println!("\n=== Gradient Check Report ===");
        println!("Configuration:");
        println!("  Epsilon: {:.2e}", self.config.epsilon);
        println!(
            "  Relative Tolerance: {:.2e}",
            self.config.relative_tolerance
        );
        println!(
            "  Absolute Tolerance: {:.2e}",
            self.config.absolute_tolerance
        );
        println!(
            "  Centered Differences: {}",
            self.config.use_centered_differences
        );
        println!();

        let mut all_passed = true;
        for (var_name, result) in results {
            all_passed &= result.passed;

            println!("Variable: {}", var_name);
            println!("  Status: {}", if result.passed { "PASS" } else { "FAIL" });
            println!("  Elements Checked: {}", result.elements_checked);
            println!("  Failed Elements: {}", result.failed_elements);
            println!(
                "  Pass Percentage: {:.1}%",
                result.statistics.pass_percentage
            );
            println!("  Max Relative Error: {:.2e}", result.max_relative_error);
            println!("  Max Absolute Error: {:.2e}", result.max_absolute_error);
            println!(
                "  Mean Relative Error: {:.2e} ± {:.2e}",
                result.statistics.mean_relative_error, result.statistics.std_relative_error
            );
            println!(
                "  Mean Absolute Error: {:.2e} ± {:.2e}",
                result.statistics.mean_absolute_error, result.statistics.std_absolute_error
            );
            println!();

            if !result.passed && self.config.verbose {
                self.print_element_errors(&result.element_errors);
            }
        }

        println!(
            "Overall Status: {}",
            if all_passed { "PASS" } else { "FAIL" }
        );
        println!("==============================\n");
    }

    /// Helper method to print element errors for failed gradient checks
    fn print_element_errors(&self, element_errors: &Option<Vec<ElementError>>) {
        if let Some(ref errors) = element_errors {
            println!("  Failed Elements (first 10):");
            for (i, error) in errors.iter().take(10).enumerate() {
                println!("    [{:?}]: analytical={:.6e}, numerical={:.6e}, rel_err={:.2e}, abs_err={:.2e}",
                    error.index, error.analytical, error.numerical,
                    error.relative_error, error.absolute_error);
            }
            println!();
        }
    }
}

/// Convenience function to create and run a gradient check
pub fn check_gradients<F>(
    engine: Arc<AutodiffEngine>,
    f: F,
    inputs: &HashMap<String, Variable>,
    config: Option<GradientCheckConfig>,
) -> Result<HashMap<String, GradientCheckResult>>
where
    F: Fn(&HashMap<String, Variable>) -> Result<Variable> + Clone,
{
    let checker = if let Some(config) = config {
        GradientChecker::with_config(engine, config)
    } else {
        GradientChecker::new(engine)
    };

    checker.check_gradients(f, inputs)
}

/// Convenience function to run gradient check with default config and print report
pub fn check_and_report_gradients<F>(
    engine: Arc<AutodiffEngine>,
    f: F,
    inputs: &HashMap<String, Variable>,
) -> Result<bool>
where
    F: Fn(&HashMap<String, Variable>) -> Result<Variable> + Clone,
{
    let config = GradientCheckConfig {
        verbose: true,
        ..Default::default()
    };

    let checker = GradientChecker::with_config(engine, config);
    let results = checker.check_gradients(f, inputs)?;

    checker.print_report(&results);

    Ok(results.values().all(|r| r.passed))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::Tensor;
    use std::collections::HashMap;

    fn create_test_engine() -> Arc<AutodiffEngine> {
        Arc::new(AutodiffEngine::default())
    }

    #[test]
    fn test_gradient_checker_creation() {
        let engine = create_test_engine();
        let checker = GradientChecker::new(engine);

        assert_eq!(checker.config.epsilon, 1e-5);
        assert_eq!(checker.config.relative_tolerance, 1e-3);
        assert!(checker.config.use_centered_differences);
    }

    #[test]
    fn test_simple_linear_function() -> Result<()> {
        let engine = create_test_engine();
        let config = GradientCheckConfig {
            relative_tolerance: 1e-2, // More realistic tolerance for numerical gradients
            ..Default::default()
        };
        let checker = GradientChecker::with_config(engine.clone(), config);

        // Test f(x) = x + x (equivalent to 2*x), gradient should be 2
        let x_data = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?;
        let x = engine.variable(x_data, true);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);

        let f = |inputs: &HashMap<String, Variable>| -> Result<Variable> {
            let x = &inputs["x"];
            // Use x + x = 2*x, then sum to get a scalar
            let doubled = x.add(x)?;
            doubled.sum(None)
        };

        let results = checker.check_gradients(f, &inputs)?;
        let x_result = &results["x"];

        assert!(
            x_result.passed,
            "Linear function gradient check should pass"
        );
        assert!(
            x_result.max_relative_error < 1e-2,
            "Relative error too high: {:.2e}",
            x_result.max_relative_error
        );

        Ok(())
    }

    #[test]
    fn test_quadratic_function() -> Result<()> {
        let engine = create_test_engine();
        let config = GradientCheckConfig {
            relative_tolerance: 1e-2, // More realistic tolerance for numerical gradients
            ..Default::default()
        };
        let checker = GradientChecker::with_config(engine.clone(), config);

        // Test f(x) = x^2, gradient should be 2*x
        let x_data = Tensor::from_vec(vec![2.0], &[1])?;
        let x = engine.variable(x_data, true);

        let mut inputs = HashMap::new();
        inputs.insert("x".to_string(), x);

        let f = |inputs: &HashMap<String, Variable>| -> Result<Variable> {
            let x = &inputs["x"];
            x.mul(x) // x^2 (already scalar since input is [2.0])
        };

        let results = checker.check_gradients(f, &inputs)?;
        let x_result = &results["x"];

        assert!(
            x_result.passed,
            "Quadratic function gradient check should pass"
        );

        Ok(())
    }

    #[test]
    fn test_config_serialization() {
        let config = GradientCheckConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: GradientCheckConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(config.epsilon, deserialized.epsilon);
        assert_eq!(config.relative_tolerance, deserialized.relative_tolerance);
        assert_eq!(
            config.use_centered_differences,
            deserialized.use_centered_differences
        );
    }

    #[test]
    fn test_flat_index_conversion() {
        let engine = create_test_engine();
        let checker = GradientChecker::new(engine);

        let shape = &[2, 3, 4];
        let flat_idx = 10; // Should be [0, 2, 2] in multi-index

        let multi_idx = checker.flat_index_to_multi_index(flat_idx, shape).unwrap();
        assert_eq!(multi_idx, vec![0, 2, 2]);
    }

    #[test]
    fn test_gradient_check_result_statistics() {
        let result = GradientCheckResult {
            passed: true,
            max_relative_error: 1e-6,
            max_absolute_error: 1e-8,
            elements_checked: 100,
            failed_elements: 0,
            element_errors: None,
            statistics: GradientCheckStats {
                mean_relative_error: 1e-7,
                std_relative_error: 1e-8,
                mean_absolute_error: 1e-9,
                std_absolute_error: 1e-10,
                pass_percentage: 100.0,
            },
        };

        assert!(result.passed);
        assert_eq!(result.statistics.pass_percentage, 100.0);
        assert!(result.max_relative_error < 1e-5);
    }
}
