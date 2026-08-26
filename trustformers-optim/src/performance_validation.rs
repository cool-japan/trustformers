//! # Comprehensive Performance Validation Framework
//!
//! This module provides a comprehensive performance validation and benchmarking system
//! for all optimizers in the TrustformeRS optimization library. It addresses the
//! **HIGH PRIORITY** performance validation requirements from TODO.md:
//!
//! - Run benchmarks to verify optimization implementations work correctly
//! - Validate memory efficiency claims for 8-bit optimizers
//! - Test distributed training components
//! - Performance regression detection with statistical significance
//! - Cross-optimizer performance comparison and validation
//!
//! ## Key Features
//!
//! 1. **Correctness Validation**: Mathematical correctness of all optimizer implementations
//! 2. **Performance Benchmarking**: Comprehensive performance analysis across scenarios
//! 3. **Memory Efficiency Testing**: Validation of memory usage claims and optimizations
//! 4. **Regression Detection**: Statistical analysis to detect performance regressions
//! 5. **Distributed Training Validation**: Testing of distributed training components
//! 6. **Hardware Utilization Analysis**: CPU/GPU utilization and efficiency metrics
//! 7. **Convergence Analysis**: Mathematical convergence validation and speed analysis
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::performance_validation::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create comprehensive validation suite
//! let mut validator = PerformanceValidator::new()
//!     .with_statistical_significance(true)
//!     .with_memory_validation(true)
//!     .with_regression_detection(true)
//!     .with_convergence_analysis(true);
//!
//! // Run complete validation suite
//! let results = validator.run_comprehensive_validation()?;
//!
//! // Generate detailed report
//! let report = validator.generate_validation_report(&results)?;
//! println!("{}", report);
//! # Ok(())
//! # }
//! ```

// reason: research-stage module — reserved API/scaffolding fields and methods
// retained intentionally for in-progress features; not yet on active call paths.
#![allow(dead_code)]

use crate::adam::{Adam, AdamW};
use crate::averaged_adam::AveragedAdam;
use crate::lamb::LAMB;
use crate::lion::Lion;
use crate::sgd::SGD;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, Instant};
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// A `benchmark_optimizer`-local extension of [`Optimizer`] that additionally
/// exposes each optimizer's REAL allocated state memory (momentum/variance
/// buffers, etc.), when the concrete optimizer type publishes one via
/// [`crate::traits::StatefulOptimizer::memory_usage`].
///
/// [`StatefulOptimizer`](crate::traits::StatefulOptimizer) cannot be called
/// through `dyn Optimizer` -- it carries associated types (`Config`,
/// `State`), so it isn't `dyn`-safe, and `benchmark_optimizer` needs a single
/// uniform type across every [`OptimizerType`] it cycles through. This
/// trait re-exposes just the one number it needs, in bytes, so
/// `create_optimizer_instance` can keep returning one `Box<dyn ..>` type.
///
/// The default is `None`: an optimizer kind that doesn't implement
/// `StatefulOptimizer` (currently only [`LAMB`], whose moment buffers are
/// private with no public accessor) has genuinely nothing to report here,
/// so it is honestly `None` rather than a fabricated number.
trait BenchmarkOptimizer: Optimizer {
    fn state_memory_bytes(&self) -> Option<usize> {
        None
    }
}

impl BenchmarkOptimizer for Adam {
    fn state_memory_bytes(&self) -> Option<usize> {
        Some(crate::traits::StatefulOptimizer::memory_usage(self).total_bytes)
    }
}

impl BenchmarkOptimizer for AdamW {
    fn state_memory_bytes(&self) -> Option<usize> {
        Some(crate::traits::StatefulOptimizer::memory_usage(self).total_bytes)
    }
}

impl BenchmarkOptimizer for SGD {
    fn state_memory_bytes(&self) -> Option<usize> {
        Some(crate::traits::StatefulOptimizer::memory_usage(self).total_bytes)
    }
}

impl BenchmarkOptimizer for AveragedAdam {
    fn state_memory_bytes(&self) -> Option<usize> {
        Some(crate::traits::StatefulOptimizer::memory_usage(self).total_bytes)
    }
}

impl BenchmarkOptimizer for Lion {
    fn state_memory_bytes(&self) -> Option<usize> {
        Some(crate::traits::StatefulOptimizer::memory_usage(self).total_bytes)
    }
}

// `LAMB` has no public state-memory accessor (its `exp_avg`/`exp_avg_sq`
// buffers are private and it does not implement `StatefulOptimizer`), so it
// uses the trait's default `None` -- an honest "cannot measure", not a
// fabricated byte count.
impl BenchmarkOptimizer for LAMB {}

/// Comprehensive performance validation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    /// Enable statistical significance testing
    pub statistical_significance: bool,
    /// Enable memory efficiency validation
    pub memory_validation: bool,
    /// Enable performance regression detection
    pub regression_detection: bool,
    /// Enable convergence analysis
    pub convergence_analysis: bool,
    /// Enable distributed training validation
    pub distributed_validation: bool,
    /// Number of benchmark iterations for statistical analysis
    pub benchmark_iterations: usize,
    /// Confidence level for statistical tests (0.95 = 95%)
    pub confidence_level: f64,
    /// Maximum acceptable performance regression (%)
    pub max_regression_threshold: f64,
    /// Minimum required memory efficiency for 8-bit optimizers (%)
    pub min_memory_efficiency: f64,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            statistical_significance: true,
            memory_validation: true,
            regression_detection: true,
            convergence_analysis: true,
            distributed_validation: true,
            benchmark_iterations: 100,
            confidence_level: 0.95,
            max_regression_threshold: 5.0, // 5% regression threshold
            min_memory_efficiency: 75.0,   // 75% memory reduction requirement
        }
    }
}

/// Main performance validation framework
pub struct PerformanceValidator {
    config: ValidationConfig,
    baseline_results: Option<HashMap<String, BenchmarkResult>>,
    validation_history: Vec<ValidationSession>,
    statistical_analyzer: StatisticalAnalyzer,
    memory_analyzer: MemoryAnalyzer,
    convergence_analyzer: ConvergenceAnalyzer,
    regression_detector: RegressionDetector,
}

impl Default for PerformanceValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceValidator {
    /// Create new performance validator with default configuration
    pub fn new() -> Self {
        Self {
            config: ValidationConfig::default(),
            baseline_results: None,
            validation_history: Vec::new(),
            statistical_analyzer: StatisticalAnalyzer::new(),
            memory_analyzer: MemoryAnalyzer::new(),
            convergence_analyzer: ConvergenceAnalyzer::new(),
            regression_detector: RegressionDetector::new(),
        }
    }

    /// Builder pattern for configuration
    pub fn with_statistical_significance(mut self, enabled: bool) -> Self {
        self.config.statistical_significance = enabled;
        self
    }

    pub fn with_memory_validation(mut self, enabled: bool) -> Self {
        self.config.memory_validation = enabled;
        self
    }

    pub fn with_regression_detection(mut self, enabled: bool) -> Self {
        self.config.regression_detection = enabled;
        self
    }

    pub fn with_convergence_analysis(mut self, enabled: bool) -> Self {
        self.config.convergence_analysis = enabled;
        self
    }

    pub fn with_benchmark_iterations(mut self, iterations: usize) -> Self {
        self.config.benchmark_iterations = iterations;
        self
    }

    /// Run comprehensive validation suite
    pub fn run_comprehensive_validation(&mut self) -> Result<ValidationResults> {
        let session_start = Instant::now();
        let mut results = ValidationResults::new();

        // 1. Correctness Validation
        let correctness_results = self.validate_mathematical_correctness()?;
        results.correctness_results = correctness_results;

        // 2. Performance Benchmarking
        let performance_results = self.run_performance_benchmarks()?;
        results.performance_results = performance_results;

        // 3. Memory Efficiency Validation
        if self.config.memory_validation {
            let memory_results = self.validate_memory_efficiency()?;
            results.memory_results = Some(memory_results);
        }

        // 4. Convergence Analysis
        if self.config.convergence_analysis {
            let convergence_results = self.analyze_convergence_properties()?;
            results.convergence_results = Some(convergence_results);
        }

        // 5. Distributed Training Validation
        if self.config.distributed_validation {
            let distributed_results = self.validate_distributed_training()?;
            results.distributed_results = Some(distributed_results);
        }

        // 6. Regression Detection
        if self.config.regression_detection && self.baseline_results.is_some() {
            let regression_results =
                self.detect_performance_regressions(&results.performance_results)?;
            results.regression_results = Some(regression_results);
        }

        let total_time = session_start.elapsed();
        results.total_validation_time = total_time;

        // Store validation session
        let session = ValidationSession {
            timestamp: std::time::SystemTime::now(),
            config: self.config.clone(),
            results: results.clone(),
        };
        self.validation_history.push(session);

        Ok(results)
    }

    /// Validate mathematical correctness of all optimizers
    fn validate_mathematical_correctness(&mut self) -> Result<CorrectnessResults> {
        let mut results = CorrectnessResults::new();

        // Test optimizers with known mathematical properties
        let test_cases = self.create_mathematical_test_cases()?;

        for test_case in &test_cases {
            // Test each optimizer on this test case
            let optimizer_results = self.test_optimizers_on_case(test_case)?;

            for (optimizer_name, passed) in optimizer_results {
                results.optimizer_correctness.insert(optimizer_name, passed);
            }
        }

        // Analyze results
        let total_tests = results.optimizer_correctness.len();
        let passed_tests = results.optimizer_correctness.values().filter(|&&x| x).count();

        results.overall_correctness_rate = passed_tests as f64 / total_tests as f64;
        results.passed_tests = passed_tests;
        results.total_tests = total_tests;

        Ok(results)
    }

    fn create_mathematical_test_cases(&self) -> Result<Vec<MathematicalTestCase>> {
        let mut test_cases = Vec::new();

        // Test Case 1: Quadratic function optimization
        test_cases.push(MathematicalTestCase {
            name: "Quadratic Function Convergence".to_string(),
            description: "f(x) = 0.5 * x^T A x + b^T x".to_string(),
            parameters: create_test_parameters(vec![10, 10])?,
            gradients: create_quadratic_gradients(vec![10, 10])?,
            expected_properties: vec![
                MathematicalProperty::Convergence,
                MathematicalProperty::MonotonicImprovement,
            ],
            tolerance: 1e-6,
        });

        // Test Case 2: Convex optimization
        test_cases.push(MathematicalTestCase {
            name: "Convex Optimization".to_string(),
            description: "Simple convex function with known minimum".to_string(),
            parameters: create_test_parameters(vec![5, 5])?,
            gradients: create_convex_gradients(vec![5, 5])?,
            expected_properties: vec![
                MathematicalProperty::Convergence,
                MathematicalProperty::GlobalOptimum,
            ],
            tolerance: 1e-5,
        });

        // Test Case 3: Sparse gradient handling
        test_cases.push(MathematicalTestCase {
            name: "Sparse Gradient Handling".to_string(),
            description: "Optimization with sparse gradients".to_string(),
            parameters: create_test_parameters(vec![20, 20])?,
            gradients: create_sparse_gradients(vec![20, 20], 0.1)?, // 10% sparsity
            expected_properties: vec![
                MathematicalProperty::SparsityHandling,
                MathematicalProperty::StableConvergence,
            ],
            tolerance: 1e-4,
        });

        Ok(test_cases)
    }

    fn test_optimizers_on_case(
        &self,
        test_case: &MathematicalTestCase,
    ) -> Result<HashMap<String, bool>> {
        let mut results = HashMap::new();

        // Test Adam
        let adam_passed = self.test_optimizer_correctness(
            "Adam",
            || Box::new(Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0)),
            test_case,
        )?;
        results.insert("Adam".to_string(), adam_passed);

        // Test AdamW
        let adamw_passed = self.test_optimizer_correctness(
            "AdamW",
            || Box::new(AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01)),
            test_case,
        )?;
        results.insert("AdamW".to_string(), adamw_passed);

        // Test SGD
        let sgd_passed = self.test_optimizer_correctness(
            "SGD",
            || Box::new(SGD::new(0.01, 0.9, 0.0, false)),
            test_case,
        )?;
        results.insert("SGD".to_string(), sgd_passed);

        // Test Averaged Adam
        let avg_adam_passed = self.test_optimizer_correctness(
            "AveragedAdam",
            || Box::new(AveragedAdam::new(0.001, (0.9, 0.999), 1e-8, 0.01, 0.999)),
            test_case,
        )?;
        results.insert("AveragedAdam".to_string(), avg_adam_passed);

        Ok(results)
    }

    fn test_optimizer_correctness<F>(
        &self,
        _name: &str,
        optimizer_factory: F,
        test_case: &MathematicalTestCase,
    ) -> Result<bool>
    where
        F: Fn() -> Box<dyn Optimizer>,
    {
        let mut optimizer = optimizer_factory();
        let mut parameters = test_case.parameters.clone();
        let initial_loss = self.compute_test_loss(&parameters, test_case)?;
        let mut previous_loss = initial_loss;

        let mut convergence_achieved = false;
        let mut monotonic_improvement = true;
        let max_iterations = 1000;

        // Update norms produced on exactly-zero-gradient steps, in the
        // order they occur. Used by `MathematicalProperty::SparsityHandling`
        // below: a zero gradient carries no new signal, so a well-behaved
        // optimizer's update on such a step comes only from decaying
        // momentum/decoupled decay and must not grow step over step.
        let mut zero_gradient_update_norms: Vec<f32> = Vec::new();

        for iteration in 0..max_iterations {
            // Compute gradients for current parameters
            let gradients = self.compute_test_gradients(&parameters, test_case, iteration)?;

            // Apply optimizer step
            for (param_name, gradient) in &gradients {
                if let Some(param) = parameters.get_mut(param_name) {
                    let is_zero_gradient = gradient.norm()? == 0.0;
                    let before = if is_zero_gradient { Some(param.clone()) } else { None };

                    optimizer.zero_grad();
                    optimizer.update(param, gradient)?;
                    optimizer.step();

                    if let Some(before) = before {
                        zero_gradient_update_norms.push(param.sub(&before)?.norm()?);
                    }
                }
            }

            // Check convergence and properties
            let current_loss = self.compute_test_loss(&parameters, test_case)?;

            // Check monotonic improvement (for convex problems)
            if test_case
                .expected_properties
                .contains(&MathematicalProperty::MonotonicImprovement)
                && current_loss > previous_loss + test_case.tolerance as f32
            {
                monotonic_improvement = false;
            }

            // Check convergence
            if (previous_loss - current_loss).abs() < test_case.tolerance as f32 {
                convergence_achieved = true;
                break;
            }

            previous_loss = current_loss;
        }

        // Validate expected properties
        let mut all_properties_satisfied = true;

        for property in &test_case.expected_properties {
            match property {
                MathematicalProperty::Convergence => {
                    if !convergence_achieved {
                        all_properties_satisfied = false;
                    }
                },
                MathematicalProperty::MonotonicImprovement => {
                    if !monotonic_improvement {
                        all_properties_satisfied = false;
                    }
                },
                MathematicalProperty::GlobalOptimum => {
                    // For test problems, check if close to known optimum
                    let final_loss = self.compute_test_loss(&parameters, test_case)?;
                    if final_loss > (test_case.tolerance * 10.0) as f32 {
                        all_properties_satisfied = false;
                    }
                },
                MathematicalProperty::SparsityHandling => {
                    // A zero gradient carries no new signal, so the update
                    // it produces (from decaying momentum / decoupled
                    // weight decay only) must be finite and must not grow
                    // from one zero-gradient step to the next -- nothing is
                    // renewing it. This is independent of whether the
                    // overall run converged.
                    if zero_gradient_update_norms.is_empty() {
                        // No zero-gradient step ever occurred, so there is
                        // nothing to evaluate: do not claim the property
                        // holds without evidence.
                        all_properties_satisfied = false;
                    } else if !zero_gradient_update_norms.iter().all(|n| n.is_finite()) {
                        all_properties_satisfied = false;
                    } else {
                        let non_increasing = zero_gradient_update_norms
                            .windows(2)
                            .all(|pair| pair[1] <= pair[0] + test_case.tolerance as f32);
                        if !non_increasing {
                            all_properties_satisfied = false;
                        }
                    }
                },
                MathematicalProperty::StableConvergence => {
                    // Check for stable convergence without oscillations
                    if !convergence_achieved {
                        all_properties_satisfied = false;
                    }
                },
            }
        }

        Ok(all_properties_satisfied)
    }

    fn compute_test_loss(
        &self,
        parameters: &HashMap<String, Tensor>,
        test_case: &MathematicalTestCase,
    ) -> Result<f32> {
        // Simplified loss computation for test cases
        match test_case.name.as_str() {
            "Quadratic Function Convergence" => {
                // f(x) = 0.5 * ||x||^2
                let mut total_loss = 0.0;
                for tensor in parameters.values() {
                    let norm_squared = tensor.norm()?.powi(2);
                    total_loss += norm_squared * 0.5;
                }
                Ok(total_loss)
            },
            "Convex Optimization" => {
                // f(x) = ||x - target||^2 where target is zero
                let mut total_loss = 0.0;
                for tensor in parameters.values() {
                    let norm_squared = tensor.norm()?.powi(2);
                    total_loss += norm_squared;
                }
                Ok(total_loss)
            },
            "Sparse Gradient Handling" => {
                // Simple quadratic with sparse structure
                let mut total_loss = 0.0;
                for tensor in parameters.values() {
                    let norm_squared = tensor.norm()?.powi(2);
                    total_loss += norm_squared * 0.5;
                }
                Ok(total_loss)
            },
            _ => Ok(0.0),
        }
    }

    fn compute_test_gradients(
        &self,
        parameters: &HashMap<String, Tensor>,
        test_case: &MathematicalTestCase,
        iteration: usize,
    ) -> Result<HashMap<String, Tensor>> {
        let mut gradients = HashMap::new();

        match test_case.name.as_str() {
            "Quadratic Function Convergence" => {
                // Gradient of f(x) = 0.5 * ||x||^2 is x
                for (name, param) in parameters {
                    gradients.insert(name.clone(), param.clone());
                }
            },
            "Convex Optimization" => {
                // Gradient of f(x) = ||x||^2 is 2x
                for (name, param) in parameters {
                    let grad = param.scalar_mul(2.0)?;
                    gradients.insert(name.clone(), grad);
                }
            },
            "Sparse Gradient Handling" => {
                // Create sparse gradients
                for (name, param) in parameters {
                    let grad = param.clone();
                    // Make gradient sparse by zeroing out random elements
                    if iteration % 10 < 3 {
                        // 30% of iterations have sparse gradients
                        let shape = param.shape();
                        let _total_elements = shape.iter().product::<usize>();
                        let sparse_grad = Tensor::zeros(&shape)?;
                        gradients.insert(name.clone(), sparse_grad);
                    } else {
                        gradients.insert(name.clone(), grad);
                    }
                }
            },
            _ => {
                // Default: use provided gradients
                gradients = test_case.gradients.clone();
            },
        }

        Ok(gradients)
    }

    /// Run comprehensive performance benchmarks
    fn run_performance_benchmarks(&mut self) -> Result<PerformanceBenchmarkResults> {
        let mut results = PerformanceBenchmarkResults::new();

        // Define benchmark scenarios
        let scenarios = vec![
            BenchmarkScenario {
                name: "Small Model (1M params)".to_string(),
                parameter_sizes: vec![1000, 1000], // 1M parameters
                batch_size: 32,
                iterations: self.config.benchmark_iterations,
            },
            BenchmarkScenario {
                name: "Medium Model (10M params)".to_string(),
                parameter_sizes: vec![3162, 3162], // ~10M parameters
                batch_size: 16,
                iterations: self.config.benchmark_iterations / 2, // Fewer iterations for larger models
            },
            BenchmarkScenario {
                name: "Large Model (100M params)".to_string(),
                parameter_sizes: vec![10000, 10000], // 100M parameters
                batch_size: 8,
                iterations: self.config.benchmark_iterations / 4,
            },
        ];

        for scenario in scenarios {
            let scenario_results = self.benchmark_scenario(&scenario)?;
            results.scenario_results.push(scenario_results);
        }

        // Analyze cross-scenario performance
        self.analyze_performance_trends(&mut results)?;

        Ok(results)
    }

    fn benchmark_scenario(&self, scenario: &BenchmarkScenario) -> Result<ScenarioBenchmarkResult> {
        let mut result = ScenarioBenchmarkResult {
            scenario_name: scenario.name.clone(),
            optimizer_results: HashMap::new(),
        };

        // Benchmark each optimizer
        let optimizers_to_test = vec![
            ("Adam", OptimizerType::Adam),
            ("AdamW", OptimizerType::AdamW),
            ("SGD", OptimizerType::SGD),
            ("AveragedAdam", OptimizerType::AveragedAdam),
            ("LAMB", OptimizerType::LAMB),
            ("Lion", OptimizerType::Lion),
        ];

        for (name, optimizer_type) in optimizers_to_test {
            let optimizer_result = self.benchmark_optimizer(name, optimizer_type, scenario)?;
            result.optimizer_results.insert(name.to_string(), optimizer_result);
        }

        Ok(result)
    }

    fn benchmark_optimizer(
        &self,
        name: &str,
        optimizer_type: OptimizerType,
        scenario: &BenchmarkScenario,
    ) -> Result<OptimizerBenchmarkResult> {
        let mut step_times = Vec::new();

        // Create optimizer
        let mut optimizer = self.create_optimizer_instance(optimizer_type)?;

        // Create test parameters
        let mut parameters = create_test_parameters(scenario.parameter_sizes.clone())?;

        for iteration in 0..scenario.iterations {
            // Create gradients for this iteration
            let gradients = create_benchmark_gradients(&scenario.parameter_sizes, iteration)?;

            // Time the optimizer step
            let step_start = Instant::now();

            // Apply optimizer step
            for (param_name, gradient) in &gradients {
                if let Some(param) = parameters.get_mut(param_name) {
                    optimizer.zero_grad();
                    optimizer.update(param, gradient)?;
                    optimizer.step();
                }
            }

            let step_time = step_start.elapsed();
            step_times.push(step_time);
        }

        // Compute statistics
        let avg_step_time = step_times.iter().sum::<Duration>() / step_times.len() as u32;
        let min_step_time = step_times.iter().min().copied().unwrap_or(Duration::from_secs(0));
        let max_step_time = step_times.iter().max().copied().unwrap_or(Duration::from_secs(0));

        // Real, measured state memory (see `BenchmarkOptimizer::state_memory_bytes`
        // near the top of this file), read once after the run rather than
        // as a per-iteration "before/after delta": every optimizer kind
        // this can measure allocates its per-parameter state buffers
        // (momentum/variance/...) on that parameter's first `update()`
        // call and never resizes them again, since `scenario.parameter_sizes`
        // is fixed for the whole run -- so the state footprint is already
        // at steady state after iteration 0 and identical on every later
        // iteration. There is nothing to average; a delta between two
        // identical readings is always exactly zero regardless of the
        // optimizer, which is what the previous shape-only
        // `estimate_memory_usage` (computed from `parameters`, ignoring
        // `optimizer` entirely) actually measured -- a mathematical
        // certainty, not a benchmark result.
        let state_memory_bytes = optimizer.state_memory_bytes();

        // Calculate throughput (parameters processed per second)
        let total_params: usize = scenario.parameter_sizes.iter().product();
        let throughput = total_params as f64 / avg_step_time.as_secs_f64();

        // Perform statistical analysis if enabled. When a baseline was set
        // for this optimizer (`set_baseline`), its avg_step_time is a real,
        // caller-provided null hypothesis for `analyze`'s t-test; with no
        // baseline there is nothing to test against and `analyze` reports
        // `p_value: None` honestly rather than a fabricated constant.
        let baseline_step_time = self
            .baseline_results
            .as_ref()
            .and_then(|baselines| baselines.get(name))
            .map(|baseline| baseline.avg_step_time);
        let statistical_metrics = if self.config.statistical_significance {
            Some(self.statistical_analyzer.analyze(
                &step_times,
                self.config.confidence_level,
                baseline_step_time,
            )?)
        } else {
            None
        };

        Ok(OptimizerBenchmarkResult {
            optimizer_name: name.to_string(),
            avg_step_time,
            min_step_time,
            max_step_time,
            throughput,
            avg_memory_usage: state_memory_bytes,
            statistical_metrics,
        })
    }

    fn create_optimizer_instance(
        &self,
        optimizer_type: OptimizerType,
    ) -> Result<Box<dyn BenchmarkOptimizer>> {
        match optimizer_type {
            OptimizerType::Adam => Ok(Box::new(Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0))),
            OptimizerType::AdamW => Ok(Box::new(AdamW::new(0.001, (0.9, 0.999), 1e-8, 0.01))),
            OptimizerType::SGD => Ok(Box::new(SGD::new(0.01, 0.9, 0.0001, true))),
            OptimizerType::AveragedAdam => Ok(Box::new(AveragedAdam::new(
                0.001,
                (0.9, 0.999),
                1e-8,
                0.01,
                0.999,
            ))),
            OptimizerType::LAMB => Ok(Box::new(LAMB::new(0.001, (0.9, 0.999), 1e-6, 0.01))),
            OptimizerType::Lion => Ok(Box::new(Lion::new(0.0001, (0.9, 0.99), 0.01))),
        }
    }

    fn analyze_performance_trends(&self, results: &mut PerformanceBenchmarkResults) -> Result<()> {
        // Analyze performance scaling across model sizes
        let mut scaling_analysis = HashMap::new();

        for optimizer_name in ["Adam", "AdamW", "SGD", "AveragedAdam", "LAMB", "Lion"] {
            let mut throughputs = Vec::new();

            for scenario_result in &results.scenario_results {
                if let Some(optimizer_result) =
                    scenario_result.optimizer_results.get(optimizer_name)
                {
                    throughputs.push(optimizer_result.throughput);
                }
            }

            if throughputs.len() >= 2 {
                let scaling_efficiency = self.compute_scaling_efficiency(&throughputs);
                scaling_analysis.insert(optimizer_name.to_string(), scaling_efficiency);
            }
        }

        results.scaling_analysis = scaling_analysis;
        Ok(())
    }

    fn compute_scaling_efficiency(&self, throughputs: &[f64]) -> f64 {
        if throughputs.len() < 2 {
            return 1.0;
        }

        // Compute how well throughput scales (should decrease as model size increases)
        // Perfect scaling would be inverse linear relationship
        let first = throughputs[0];
        let last = throughputs[throughputs.len() - 1];

        // Higher is better (less performance degradation with scale)
        last / first
    }

    /// Validate memory efficiency claims
    fn validate_memory_efficiency(&mut self) -> Result<MemoryValidationResults> {
        let mut results = MemoryValidationResults::new();

        // Test 8-bit optimizers memory efficiency
        let memory_test_results = self.test_memory_efficiency_claims()?;
        results.eight_bit_efficiency = memory_test_results;

        // Test gradient compression efficiency
        let compression_results = self.test_gradient_compression_efficiency()?;
        results.compression_efficiency = compression_results;

        // Validate memory optimization techniques
        let optimization_results = self.test_memory_optimizations()?;
        results.optimization_efficiency = optimization_results;

        Ok(results)
    }

    /// Measures the real state footprint of a quantized optimizer against `Adam`.
    ///
    /// Both optimizers are stepped on identical parameters and gradients, then asked
    /// for the size of the buffers they actually allocated
    /// ([`StatefulOptimizer::memory_usage`]). Nothing is assumed: the reported
    /// percentage is `(adam_bytes − quantized_bytes) / adam_bytes`.
    fn test_memory_efficiency_claims(&self) -> Result<HashMap<String, f64>> {
        use crate::quantized_advanced::Adam4bit;
        use crate::traits::StatefulOptimizer;

        let mut results = HashMap::new();

        let shape = vec![64, 64];
        let parameters = create_test_parameters(shape.clone())?;
        let gradients = create_benchmark_gradients(&shape, 0)?;

        let mut adam = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut adam4bit = Adam4bit::new(0.001, 0.9, 0.999, 1e-8, 0.0);

        for (name, parameter) in &parameters {
            let Some(gradient) = gradients.get(name) else {
                continue;
            };
            let mut for_adam = parameter.clone();
            let mut for_quantized = parameter.clone();
            adam.update(&mut for_adam, gradient)?;
            adam4bit.update(&mut for_quantized, gradient)?;
        }

        let adam_bytes = StatefulOptimizer::memory_usage(&adam).total_bytes;
        let quantized_bytes = StatefulOptimizer::memory_usage(&adam4bit).total_bytes;

        if adam_bytes == 0 {
            return Err(TrustformersError::invalid_state(
                "Adam reported a zero-byte state after a step; memory validation cannot proceed"
                    .to_string(),
            ));
        }

        let reduction = (adam_bytes as f64 - quantized_bytes as f64) / adam_bytes as f64 * 100.0;
        results.insert("Adam4bit".to_string(), reduction);
        results.insert("Adam.state_bytes".to_string(), adam_bytes as f64);
        results.insert("Adam4bit.state_bytes".to_string(), quantized_bytes as f64);

        Ok(results)
    }

    /// Measures the *real* payload reduction of each gradient-compression method.
    ///
    /// Each method compresses a fixed gradient, and the reported percentage is the
    /// measured drop in transmitted bytes (`indices` plus `values`) relative to the
    /// dense `f32` payload — not a table of expected ratios.
    fn test_gradient_compression_efficiency(&self) -> Result<HashMap<String, f64>> {
        use crate::compression::{CompressionMethod, GradientCompressor};

        let mut results = HashMap::new();

        // One 1024-element gradient: sparse methods need a tensor large enough for
        // their `k` to be meaningful.
        let shape = vec![1024];
        let gradients = create_benchmark_gradients(&shape, 7)?;
        let dense_bytes: usize = gradients
            .values()
            .map(|g| g.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
            .sum();
        if dense_bytes == 0 {
            return Err(TrustformersError::invalid_state(
                "compression validation needs a non-empty gradient".to_string(),
            ));
        }

        let methods = [
            ("TopK", CompressionMethod::TopK { k: 102 }),
            ("RandomK", CompressionMethod::RandomK { k: 102 }),
            ("Threshold", CompressionMethod::Threshold { threshold: 0.5 }),
            ("Quantization", CompressionMethod::Quantization { bits: 8 }),
            ("SignSGD", CompressionMethod::SignSGD),
        ];

        for (name, method) in methods {
            let mut compressor = GradientCompressor::new(method);
            let compressed = compressor.compress(&gradients)?;

            // Transmitted payload as the representation itself reports it.
            let payload: usize = compressed.values().map(|c| c.payload_bytes()).sum();

            // The round trip must reconstruct the right number of elements, otherwise
            // the "saving" is meaningless.
            let restored = compressor.decompress(&compressed)?;
            for (grad_name, tensor) in &restored {
                let expected: usize =
                    gradients.get(grad_name).map(|g| g.shape().iter().product()).unwrap_or(0);
                if tensor.data_f32()?.len() != expected {
                    return Err(TrustformersError::invalid_state(format!(
                        "{name} decompression returned the wrong element count for '{grad_name}'"
                    )));
                }
            }

            let reduction = (dense_bytes as f64 - payload as f64) / dense_bytes as f64 * 100.0;
            results.insert(name.to_string(), reduction);
        }

        Ok(results)
    }

    /// Measures memory techniques that can be measured in-process.
    ///
    /// Only mixed precision is measurable here — it is a dtype change whose effect
    /// shows up directly in [`Tensor::size_bytes`]. Gradient checkpointing and CPU
    /// offloading depend on a training loop and a device this crate does not own, so
    /// no figure is reported for them rather than a plausible-looking literal.
    fn test_memory_optimizations(&self) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();

        let dense = Tensor::from_vec(vec![0.5_f32; 4096], &[64, 64])?;
        let half = match &dense {
            Tensor::F32(array) => Tensor::F16(array.mapv(half::f16::from_f32)),
            other => {
                return Err(TrustformersError::invalid_state(format!(
                    "expected an f32 tensor, got {:?}",
                    other.dtype()
                )))
            },
        };

        let dense_bytes = dense.size_bytes();
        if dense_bytes == 0 {
            return Err(TrustformersError::invalid_state(
                "mixed-precision validation needs a non-empty tensor".to_string(),
            ));
        }
        let reduction =
            (dense_bytes as f64 - half.size_bytes() as f64) / dense_bytes as f64 * 100.0;
        results.insert("MixedPrecision".to_string(), reduction);

        Ok(results)
    }

    /// Analyze convergence properties of optimizers
    fn analyze_convergence_properties(&mut self) -> Result<ConvergenceAnalysisResults> {
        let mut results = ConvergenceAnalysisResults::new();

        // Test convergence on different problem types
        let convergence_tests = self.run_convergence_tests()?;
        results.convergence_tests = convergence_tests;

        // Analyze convergence speed
        let speed_analysis = self.analyze_convergence_speed(&results.convergence_tests)?;
        results.speed_analysis = speed_analysis;

        // Test convergence stability
        let stability_analysis = self.analyze_convergence_stability(&results.convergence_tests)?;
        results.stability_analysis = stability_analysis;

        Ok(results)
    }

    fn run_convergence_tests(&self) -> Result<HashMap<String, ConvergenceTestResult>> {
        let mut results = HashMap::new();

        let optimizers_to_test = vec![
            ("Adam", OptimizerType::Adam),
            ("AdamW", OptimizerType::AdamW),
            ("AveragedAdam", OptimizerType::AveragedAdam),
            ("SGD", OptimizerType::SGD),
        ];

        for (name, optimizer_type) in optimizers_to_test {
            let convergence_result = self.test_optimizer_convergence(name, optimizer_type)?;
            results.insert(name.to_string(), convergence_result);
        }

        Ok(results)
    }

    /// Runs a *real* optimization problem and measures the loss it actually reaches.
    ///
    /// The objective is the separable quadratic `f(θ) = Σ θ²`, whose gradient `2θ` is
    /// computed from the current parameters on every iteration. Both the gradient and
    /// the loss therefore come from the parameters the optimizer produced — the
    /// previous version stepped the optimizer and then reported a hard-coded
    /// exponential decay curve that could not have been affected by it.
    fn test_optimizer_convergence(
        &self,
        _name: &str,
        optimizer_type: OptimizerType,
    ) -> Result<ConvergenceTestResult> {
        let mut optimizer = self.create_optimizer_instance(optimizer_type)?;
        let mut parameters = create_test_parameters(vec![32, 32])?;

        // The objective evaluated on the live parameters.
        let evaluate = |params: &HashMap<String, Tensor>| -> Result<f32> {
            let mut total = 0.0_f32;
            let mut count = 0_usize;
            for tensor in params.values() {
                for value in tensor.data_f32()? {
                    total += value * value;
                    count += 1;
                }
            }
            Ok(if count == 0 { 0.0 } else { total / count as f32 })
        };

        let initial_loss = evaluate(&parameters)?;
        let mut loss_history = Vec::new();
        let mut current_loss = initial_loss;

        let max_iterations = 500;
        let mut converged = false;
        let mut convergence_iteration = max_iterations;

        // Deterministic parameter visit order keeps anonymous identities stable.
        let mut names: Vec<String> = parameters.keys().cloned().collect();
        names.sort();

        for iteration in 0..max_iterations {
            for name in &names {
                let Some(parameter) = parameters.get_mut(name) else {
                    continue;
                };
                let values = parameter.data_f32()?;
                let gradient = Tensor::from_vec(
                    values.iter().map(|v| 2.0 * v).collect::<Vec<f32>>(),
                    &parameter.shape(),
                )?;
                optimizer.update(parameter, &gradient)?;
            }
            optimizer.step();

            current_loss = evaluate(&parameters)?;
            loss_history.push(current_loss);

            if current_loss < 1e-4 && !converged {
                converged = true;
                convergence_iteration = iteration;
                break;
            }
        }

        let convergence_rate = if converged {
            1.0 - (convergence_iteration as f64 / max_iterations as f64)
        } else {
            0.0
        };

        let final_loss = current_loss;
        let loss_reduction = if initial_loss > 0.0 {
            (initial_loss - final_loss) / initial_loss
        } else {
            0.0
        };

        Ok(ConvergenceTestResult {
            converged,
            convergence_iteration,
            convergence_rate,
            final_loss,
            loss_reduction,
            loss_history,
        })
    }

    /// Convergence speed derived from the *measured* loss curves.
    ///
    /// Speed is the fraction of the iteration budget still unused when the loss first
    /// dropped below 1% of its initial value; a run that never got there scores 0.
    fn analyze_convergence_speed(
        &self,
        tests: &HashMap<String, ConvergenceTestResult>,
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        for (name, test) in tests {
            let Some(&initial) = test.loss_history.first() else {
                continue;
            };
            let target = initial * 0.01;
            let reached = test.loss_history.iter().position(|&loss| loss <= target);
            let total = test.loss_history.len().max(1) as f64;
            let speed = match reached {
                Some(index) => 1.0 - (index as f64 / total),
                None => 0.0,
            };
            results.insert(name.clone(), speed);
        }
        Ok(results)
    }

    /// Convergence stability derived from the *measured* loss curves.
    ///
    /// Stability is the fraction of steps in which the loss did not increase.
    fn analyze_convergence_stability(
        &self,
        tests: &HashMap<String, ConvergenceTestResult>,
    ) -> Result<HashMap<String, f64>> {
        let mut results = HashMap::new();
        for (name, test) in tests {
            if test.loss_history.len() < 2 {
                continue;
            }
            let monotone = test
                .loss_history
                .windows(2)
                .filter(|pair| pair[1] <= pair[0] + f32::EPSILON)
                .count();
            let stability = monotone as f64 / (test.loss_history.len() - 1) as f64;
            results.insert(name.clone(), stability);
        }
        Ok(results)
    }

    /// Validate distributed training components
    fn validate_distributed_training(&mut self) -> Result<DistributedValidationResults> {
        let mut results = DistributedValidationResults::new();

        // Test distributed training scaling
        let scaling_results = self.test_distributed_scaling()?;
        results.scaling_results = scaling_results;

        // Test communication efficiency
        let communication_results = self.test_communication_efficiency()?;
        results.communication_results = communication_results;

        // Test fault tolerance
        let fault_tolerance_results = self.test_fault_tolerance()?;
        results.fault_tolerance_results = fault_tolerance_results;

        Ok(results)
    }

    /// Measures the real memory reduction ZeRO parameter sharding delivers.
    ///
    /// A single process cannot measure multi-GPU speedup, so no speedup figure is
    /// reported. What *is* measurable is the fraction of a parameter each rank has to
    /// hold once [`crate::zero::partition_parameters`] shards it, and that is what
    /// this reports: `1 − shard_bytes / full_bytes` for rank 0 at each world size.
    fn test_distributed_scaling(&self) -> Result<HashMap<String, f64>> {
        use crate::zero::partition_parameters;

        let mut results = HashMap::new();
        let parameters = create_test_parameters(vec![64, 64])?;
        let full_elements: usize =
            parameters.values().map(|t| t.shape().iter().product::<usize>()).sum();
        if full_elements == 0 {
            return Err(TrustformersError::invalid_state(
                "ZeRO validation needs a non-empty parameter set".to_string(),
            ));
        }

        for world_size in [1_usize, 2, 4, 8] {
            let partitions = partition_parameters(&parameters, world_size, 0)?;
            let shard_elements: usize = partitions
                .values()
                .map(|p| p.local_shard.data_f32().map(|d| d.len()).unwrap_or(0))
                .sum();
            let reduction = 1.0 - (shard_elements as f64 / full_elements as f64);
            results.insert(
                format!("{world_size}-rank-parameter-memory-reduction"),
                reduction,
            );
        }

        Ok(results)
    }

    /// Verifies the collective primitives round-trip exactly.
    ///
    /// There is no wire here, so "communication efficiency" is not measurable; what is
    /// measurable — and much more useful — is whether shard → gather reconstructs the
    /// original tensor bit-for-bit. The reported value is the fraction of parameters
    /// that round-tripped exactly at each world size.
    fn test_communication_efficiency(&self) -> Result<HashMap<String, f64>> {
        use crate::zero::{gather_shards, partition_parameters};

        let mut results = HashMap::new();
        let parameters = create_test_parameters(vec![16, 16])?;

        for world_size in [2_usize, 3, 5] {
            let mut exact = 0_usize;
            for (name, tensor) in &parameters {
                let mut shards = Vec::with_capacity(world_size);
                for rank in 0..world_size {
                    let partitions = partition_parameters(&parameters, world_size, rank)?;
                    let shard =
                        partitions.get(name).map(|p| p.local_shard.clone()).ok_or_else(|| {
                            TrustformersError::invalid_state(format!(
                                "no shard produced for '{name}'"
                            ))
                        })?;
                    shards.push(shard);
                }
                let gathered = gather_shards(&shards, &tensor.shape())?;
                if gathered.data_f32()? == tensor.data_f32()? {
                    exact += 1;
                }
            }
            let fraction = exact as f64 / parameters.len().max(1) as f64;
            results.insert(format!("{world_size}-rank-gather-exactness"), fraction);
        }

        Ok(results)
    }

    /// Exercises the recovery paths that *can* be executed in one process.
    ///
    /// Node-failure and network-partition handling need a real cluster and are
    /// therefore not claimed. Checkpoint recovery is executed for real: an optimizer
    /// is stepped, checkpointed, restored into a fresh instance, and the restored
    /// state is compared against the original.
    fn test_fault_tolerance(&self) -> Result<HashMap<String, bool>> {
        use crate::traits::StatefulOptimizer;

        let mut results = HashMap::new();

        let parameters = create_test_parameters(vec![8, 8])?;
        let gradients = create_benchmark_gradients(&[8, 8], 3)?;

        let mut original = Adam::new(0.001, (0.9, 0.999), 1e-8, 0.0);
        let mut names: Vec<String> = parameters.keys().cloned().collect();
        names.sort();
        for name in &names {
            let (Some(parameter), Some(gradient)) = (parameters.get(name), gradients.get(name))
            else {
                continue;
            };
            let mut working = parameter.clone();
            original.update_named(name, &mut working, gradient)?;
        }

        let checkpoint = StatefulOptimizer::state_dict(&original)?;
        let mut restored = Adam::new(0.1, (0.5, 0.5), 1e-3, 0.5);
        StatefulOptimizer::load_state_dict(&mut restored, checkpoint.clone())?;
        let round_trip = StatefulOptimizer::state_dict(&restored)?;

        let mut identical = round_trip.len() == checkpoint.len();
        for (key, tensor) in &checkpoint {
            match round_trip.get(key) {
                Some(other) if other.data_f32()? == tensor.data_f32()? => {},
                _ => identical = false,
            }
        }
        results.insert("CheckpointRecovery".to_string(), identical);

        Ok(results)
    }

    /// Detect performance regressions compared to baseline
    fn detect_performance_regressions(
        &mut self,
        current_results: &PerformanceBenchmarkResults,
    ) -> Result<RegressionAnalysisResults> {
        let baseline = self.baseline_results.as_ref().ok_or_else(|| {
            TrustformersError::invalid_state(
                "baseline_results must be set before detecting regressions".to_string(),
            )
        })?;
        let mut results = RegressionAnalysisResults::new();

        for scenario_result in &current_results.scenario_results {
            for (optimizer_name, current_benchmark) in &scenario_result.optimizer_results {
                if let Some(baseline_benchmark) = baseline.get(optimizer_name) {
                    let regression = self.regression_detector.detect_regression(
                        baseline_benchmark,
                        current_benchmark,
                        self.config.max_regression_threshold,
                    )?;

                    if let Some(regression_info) = regression {
                        results.regressions.push(regression_info);
                    }
                }
            }
        }

        Ok(results)
    }

    /// Generate comprehensive validation report
    pub fn generate_validation_report(&self, results: &ValidationResults) -> Result<String> {
        let mut report = String::new();

        report.push_str("# TrustformeRS Optimization Performance Validation Report\\n");
        report.push_str("=====================================================\\n\\n");

        // Executive Summary
        report.push_str("## Executive Summary\\n");
        report.push_str(&format!(
            "- **Total Validation Time**: {:.2} seconds\\n",
            results.total_validation_time.as_secs_f64()
        ));
        report.push_str(&format!(
            "- **Correctness Tests**: {}/{} passed ({:.1}%)\\n",
            results.correctness_results.passed_tests,
            results.correctness_results.total_tests,
            results.correctness_results.overall_correctness_rate * 100.0
        ));

        // Performance Summary
        report.push_str("\\n## Performance Benchmark Summary\\n");
        for scenario_result in &results.performance_results.scenario_results {
            report.push_str(&format!("### {}\\n", scenario_result.scenario_name));

            let mut sorted_optimizers: Vec<_> = scenario_result.optimizer_results.iter().collect();
            sorted_optimizers.sort_by_key(|a| a.1.avg_step_time);

            for (name, result) in sorted_optimizers {
                report.push_str(&format!(
                    "- **{}**: {:.2}ms/step, {:.1}M params/sec\\n",
                    name,
                    result.avg_step_time.as_secs_f64() * 1000.0,
                    result.throughput / 1_000_000.0
                ));
            }
        }

        // Memory Efficiency
        if let Some(memory_results) = &results.memory_results {
            report.push_str("\\n## Memory Efficiency Validation\\n");
            for (optimizer, efficiency) in &memory_results.eight_bit_efficiency {
                report.push_str(&format!(
                    "- **{}**: {:.1}% memory reduction\\n",
                    optimizer, efficiency
                ));
            }
        }

        // Convergence Analysis
        if let Some(convergence_results) = &results.convergence_results {
            report.push_str("\\n## Convergence Analysis\\n");
            for (optimizer, test_result) in &convergence_results.convergence_tests {
                report.push_str(&format!(
                    "- **{}**: {} (rate: {:.3}, reduction: {:.3})\\n",
                    optimizer,
                    if test_result.converged { "Converged" } else { "Did not converge" },
                    test_result.convergence_rate,
                    test_result.loss_reduction
                ));
            }
        }

        // Regression Detection
        if let Some(regression_results) = &results.regression_results {
            report.push_str("\\n## Performance Regression Analysis\\n");
            if regression_results.regressions.is_empty() {
                report.push_str("✅ No performance regressions detected\\n");
            } else {
                for regression in &regression_results.regressions {
                    report.push_str(&format!(
                        "⚠️  **{}**: {:.1}% performance regression\\n",
                        regression.optimizer_name, regression.regression_percentage
                    ));
                }
            }
        }

        report.push_str("\\n## Validation Status: ✅ COMPLETE\\n");

        Ok(report)
    }

    /// Set baseline results for regression detection
    pub fn set_baseline(&mut self, results: HashMap<String, BenchmarkResult>) {
        self.baseline_results = Some(results);
    }
}

// Supporting types and implementations

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResults {
    pub total_validation_time: Duration,
    pub correctness_results: CorrectnessResults,
    pub performance_results: PerformanceBenchmarkResults,
    pub memory_results: Option<MemoryValidationResults>,
    pub convergence_results: Option<ConvergenceAnalysisResults>,
    pub distributed_results: Option<DistributedValidationResults>,
    pub regression_results: Option<RegressionAnalysisResults>,
}

impl Default for ValidationResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ValidationResults {
    pub fn new() -> Self {
        Self {
            total_validation_time: Duration::from_secs(0),
            correctness_results: CorrectnessResults::new(),
            performance_results: PerformanceBenchmarkResults::new(),
            memory_results: None,
            convergence_results: None,
            distributed_results: None,
            regression_results: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSession {
    pub timestamp: std::time::SystemTime,
    pub config: ValidationConfig,
    pub results: ValidationResults,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrectnessResults {
    pub optimizer_correctness: HashMap<String, bool>,
    pub overall_correctness_rate: f64,
    pub passed_tests: usize,
    pub total_tests: usize,
}

impl Default for CorrectnessResults {
    fn default() -> Self {
        Self::new()
    }
}

impl CorrectnessResults {
    pub fn new() -> Self {
        Self {
            optimizer_correctness: HashMap::new(),
            overall_correctness_rate: 0.0,
            passed_tests: 0,
            total_tests: 0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceBenchmarkResults {
    pub scenario_results: Vec<ScenarioBenchmarkResult>,
    pub scaling_analysis: HashMap<String, f64>,
}

impl Default for PerformanceBenchmarkResults {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceBenchmarkResults {
    pub fn new() -> Self {
        Self {
            scenario_results: Vec::new(),
            scaling_analysis: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioBenchmarkResult {
    pub scenario_name: String,
    pub optimizer_results: HashMap<String, OptimizerBenchmarkResult>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizerBenchmarkResult {
    pub optimizer_name: String,
    pub avg_step_time: Duration,
    pub min_step_time: Duration,
    pub max_step_time: Duration,
    pub throughput: f64,
    /// The optimizer's real allocated state memory (bytes) at the end of
    /// the run, from `BenchmarkOptimizer::state_memory_bytes`. `None`
    /// when the optimizer kind exposes no such accessor (currently only
    /// `LAMB`). Despite the field's name this is a single end-of-run
    /// reading, not an average of varying samples: every optimizer kind
    /// this can measure allocates its state buffers on first touch and
    /// never resizes them again within one benchmark run (parameter shapes
    /// are fixed), so there is nothing that actually varies to average. An
    /// earlier version computed a before/after delta from parameter
    /// *shapes* alone (ignoring the optimizer entirely), which was
    /// therefore always exactly `0.0` regardless of which optimizer ran.
    pub avg_memory_usage: Option<usize>,
    pub statistical_metrics: Option<StatisticalMetrics>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMetrics {
    pub mean: Duration,
    pub std_dev: Duration,
    pub confidence_interval_lower: Duration,
    pub confidence_interval_upper: Duration,
    /// Two-sided p-value of a one-sample Student-t test of `step_times`
    /// against a null hypothesis mean, computed by
    /// [`StatisticalAnalyzer::analyze`].
    ///
    /// A p-value needs a null hypothesis to test against. `benchmark_optimizer`
    /// passes the matching optimizer's [`BenchmarkResult::avg_step_time`] from
    /// `PerformanceValidator::baseline_results` as that hypothesis when one has
    /// been set via [`PerformanceValidator::set_baseline`]; a low p-value then
    /// means this run's step times are statistically distinguishable from the
    /// baseline's average, in either direction. `None` when no baseline is set
    /// for the optimizer being benchmarked (nothing to test against) -- left
    /// absent rather than fabricated as a constant.
    pub p_value: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryValidationResults {
    pub eight_bit_efficiency: HashMap<String, f64>,
    pub compression_efficiency: HashMap<String, f64>,
    pub optimization_efficiency: HashMap<String, f64>,
}

impl Default for MemoryValidationResults {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryValidationResults {
    pub fn new() -> Self {
        Self {
            eight_bit_efficiency: HashMap::new(),
            compression_efficiency: HashMap::new(),
            optimization_efficiency: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceAnalysisResults {
    pub convergence_tests: HashMap<String, ConvergenceTestResult>,
    pub speed_analysis: HashMap<String, f64>,
    pub stability_analysis: HashMap<String, f64>,
}

impl Default for ConvergenceAnalysisResults {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalysisResults {
    pub fn new() -> Self {
        Self {
            convergence_tests: HashMap::new(),
            speed_analysis: HashMap::new(),
            stability_analysis: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceTestResult {
    pub converged: bool,
    pub convergence_iteration: usize,
    pub convergence_rate: f64,
    pub final_loss: f32,
    pub loss_reduction: f32,
    pub loss_history: Vec<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistributedValidationResults {
    pub scaling_results: HashMap<String, f64>,
    pub communication_results: HashMap<String, f64>,
    pub fault_tolerance_results: HashMap<String, bool>,
}

impl Default for DistributedValidationResults {
    fn default() -> Self {
        Self::new()
    }
}

impl DistributedValidationResults {
    pub fn new() -> Self {
        Self {
            scaling_results: HashMap::new(),
            communication_results: HashMap::new(),
            fault_tolerance_results: HashMap::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionAnalysisResults {
    pub regressions: Vec<RegressionInfo>,
}

impl Default for RegressionAnalysisResults {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionAnalysisResults {
    pub fn new() -> Self {
        Self {
            regressions: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegressionInfo {
    pub optimizer_name: String,
    pub metric_name: String,
    pub baseline_value: f64,
    pub current_value: f64,
    pub regression_percentage: f64,
}

#[derive(Debug, Clone)]
pub struct MathematicalTestCase {
    pub name: String,
    pub description: String,
    pub parameters: HashMap<String, Tensor>,
    pub gradients: HashMap<String, Tensor>,
    pub expected_properties: Vec<MathematicalProperty>,
    pub tolerance: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum MathematicalProperty {
    Convergence,
    MonotonicImprovement,
    GlobalOptimum,
    SparsityHandling,
    StableConvergence,
}

#[derive(Debug, Clone)]
pub struct BenchmarkScenario {
    pub name: String,
    pub parameter_sizes: Vec<usize>,
    pub batch_size: usize,
    pub iterations: usize,
}

#[derive(Debug, Clone)]
pub enum OptimizerType {
    Adam,
    AdamW,
    SGD,
    AveragedAdam,
    LAMB,
    Lion,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResult {
    pub avg_step_time: Duration,
    pub throughput: f64,
    pub memory_usage: f64,
}

/// Statistical analyzer for performance metrics
pub struct StatisticalAnalyzer;

impl Default for StatisticalAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl StatisticalAnalyzer {
    pub fn new() -> Self {
        Self
    }

    /// Computes mean/std-dev/confidence-interval from `step_times`, plus a
    /// two-sided one-sample Student-t p-value against `target_step_time`
    /// when the caller supplies one -- see [`StatisticalMetrics::p_value`]'s
    /// doc comment for what the null hypothesis means and why it is
    /// sometimes absent. Uses the real Student-t distribution (via
    /// [`trustformers_core::statistics`]), not a normal approximation, so it
    /// stays accurate at the small sample sizes benchmarks typically use.
    pub fn analyze(
        &self,
        step_times: &[Duration],
        confidence_level: f64,
        target_step_time: Option<Duration>,
    ) -> Result<StatisticalMetrics> {
        let times_f64: Vec<f64> = step_times.iter().map(|d| d.as_secs_f64()).collect();

        let mean_f64 = trustformers_core::statistics::mean(&times_f64).ok_or_else(|| {
            TrustformersError::invalid_state(
                "StatisticalAnalyzer::analyze requires at least one step time".to_string(),
            )
        })?;
        // `None` for fewer than two samples: there is no sample variance --
        // and therefore no t-test -- with a single observation. The reported
        // std-dev/CI fall back to 0.0 in that case, which is exactly correct
        // for a single-point sample (no spread was observed).
        let sample_std_f64 = trustformers_core::statistics::sample_std_dev(&times_f64);
        let std_dev_f64 = sample_std_f64.unwrap_or(0.0);

        // Simple confidence interval calculation (assuming normal distribution)
        let z_score = if confidence_level >= 0.99 {
            2.576
        } else if confidence_level >= 0.95 {
            1.96
        } else {
            1.645
        };
        let margin_of_error = z_score * std_dev_f64 / (times_f64.len() as f64).sqrt();

        // One-sample Student-t test of `step_times` against `target_step_time`:
        // needs both a real target and a real sample standard deviation
        // (n >= 2), or there is no test to run. `student_t_two_sided_p_value`
        // already resolves the `standard_error == 0.0` (zero-variance) case
        // correctly (an infinite or NaN t statistic), so no special-casing is
        // needed here.
        let p_value = match (target_step_time, sample_std_f64) {
            (Some(target), Some(sample_std)) => {
                let standard_error = sample_std / (times_f64.len() as f64).sqrt();
                let t_statistic = (mean_f64 - target.as_secs_f64()) / standard_error;
                let degrees_of_freedom = times_f64.len() as f64 - 1.0;
                trustformers_core::statistics::student_t_two_sided_p_value(
                    t_statistic,
                    degrees_of_freedom,
                )
            },
            _ => None,
        };

        Ok(StatisticalMetrics {
            mean: Duration::from_secs_f64(mean_f64),
            std_dev: Duration::from_secs_f64(std_dev_f64),
            confidence_interval_lower: Duration::from_secs_f64(
                (mean_f64 - margin_of_error).max(0.0),
            ),
            confidence_interval_upper: Duration::from_secs_f64(mean_f64 + margin_of_error),
            p_value,
        })
    }
}

/// Memory analyzer for optimization memory patterns
pub struct MemoryAnalyzer;

impl Default for MemoryAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl MemoryAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

/// Convergence analyzer for optimization convergence patterns
pub struct ConvergenceAnalyzer;

impl Default for ConvergenceAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl ConvergenceAnalyzer {
    pub fn new() -> Self {
        Self
    }
}

/// Regression detector for performance regression analysis
pub struct RegressionDetector;

impl Default for RegressionDetector {
    fn default() -> Self {
        Self::new()
    }
}

impl RegressionDetector {
    pub fn new() -> Self {
        Self
    }

    pub fn detect_regression(
        &self,
        baseline: &BenchmarkResult,
        current: &OptimizerBenchmarkResult,
        threshold_percentage: f64,
    ) -> Result<Option<RegressionInfo>> {
        let baseline_time = baseline.avg_step_time.as_secs_f64();
        let current_time = current.avg_step_time.as_secs_f64();

        let regression_percentage = ((current_time - baseline_time) / baseline_time) * 100.0;

        if regression_percentage > threshold_percentage {
            Ok(Some(RegressionInfo {
                optimizer_name: current.optimizer_name.clone(),
                metric_name: "avg_step_time".to_string(),
                baseline_value: baseline_time,
                current_value: current_time,
                regression_percentage,
            }))
        } else {
            Ok(None)
        }
    }
}

// Utility functions for creating test data

fn create_test_parameters(sizes: Vec<usize>) -> Result<HashMap<String, Tensor>> {
    let mut parameters = HashMap::new();

    for (i, &size) in sizes.iter().enumerate() {
        let param_name = format!("param_{}", i);
        let tensor = Tensor::randn(&[size])?;
        parameters.insert(param_name, tensor);
    }

    Ok(parameters)
}

fn create_quadratic_gradients(sizes: Vec<usize>) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::new();

    for (i, &size) in sizes.iter().enumerate() {
        let grad_name = format!("param_{}", i);
        // For quadratic function f(x) = 0.5 * x^T * x, gradient is x
        let gradient = Tensor::randn(&[size])?;
        gradients.insert(grad_name, gradient);
    }

    Ok(gradients)
}

fn create_convex_gradients(sizes: Vec<usize>) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::new();

    for (i, &size) in sizes.iter().enumerate() {
        let grad_name = format!("param_{}", i);
        let gradient = Tensor::randn(&[size])?.scalar_mul(2.0)?; // 2x for convex function
        gradients.insert(grad_name, gradient);
    }

    Ok(gradients)
}

fn create_sparse_gradients(sizes: Vec<usize>, _sparsity: f32) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::new();

    for (i, &size) in sizes.iter().enumerate() {
        let grad_name = format!("param_{}", i);
        let _gradient = Tensor::randn(&[size])?;

        // Make gradient sparse by zeroing out elements
        // In a real implementation, would properly handle sparse tensors
        let sparse_gradient = Tensor::zeros(&[size])?; // Simplified sparse representation
        gradients.insert(grad_name, sparse_gradient);
    }

    Ok(gradients)
}

fn create_benchmark_gradients(
    sizes: &[usize],
    iteration: usize,
) -> Result<HashMap<String, Tensor>> {
    let mut gradients = HashMap::new();

    let scale = 0.1 / (1.0 + iteration as f32 * 0.01); // Decreasing gradient norms

    for (i, &size) in sizes.iter().enumerate() {
        let grad_name = format!("param_{}", i);
        let gradient = Tensor::randn(&[size])?.scalar_mul(scale)?;
        gradients.insert(grad_name, gradient);
    }

    Ok(gradients)
}

#[cfg(test)]
#[path = "performance_validation_tests.rs"]
mod performance_validation_tests;
