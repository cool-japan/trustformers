//! # AdaFactor Optimizer
//!
//! This module implements the AdaFactor optimizer, a memory-efficient variant of Adam
//! that factors the second moment estimation matrix to reduce memory usage while
//! maintaining performance.
//!
//! ## AdaFactor
//!
//! AdaFactor reduces memory usage by factorizing the second moment matrix:
//! - For 2D tensors: Uses row-wise and column-wise moving averages
//! - For 1D tensors: Uses standard moving averages like Adam
//! - Automatically scales learning rate based on parameter scale
//! - Uses dynamic beta2 decay for improved convergence
//!
//! Reference: "Adafactor: Adaptive Learning Rates with Sublinear Memory Cost"
//! by Shazeer & Stern (2018)
//!
//! ## Key Features
//!
//! - **Memory Efficiency**: Uses O(sqrt(n)) memory instead of O(n) for 2D tensors
//! - **Automatic Scaling**: Dynamic learning rate scaling based on parameter RMS
//! - **Factorized Second Moments**: Row/column factorization for large matrices
//! - **Optional First Moments**: Can disable first moment for further memory savings
//! - **Gradient Clipping**: Built-in gradient clipping with configurable threshold
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::AdaFactor;
//! use trustformers_core::traits::Optimizer;
//!
//! // Create AdaFactor optimizer with default settings
//! let mut optimizer = AdaFactor::new();
//!
//! // Or create with custom configuration
//! let mut custom_optimizer = AdaFactor::with_config(
//!     true,       // scale_parameter: Enable automatic LR scaling
//!     true,       // relative_step_size: Use relative step size
//!     false,      // warmup_init: Disable warmup initialization
//!     Some(0.0),  // beta1: First moment coefficient (None disables)
//!     1.0,        // clip_threshold: Gradient clipping threshold
//!     0.8,        // decay_rate: Beta2 decay rate
//!     1e-30,      // epsilon: Numerical stability
//!     true,       // factorize: Enable factorization for 2D tensors
//! );
//! ```
//!
//! ## Memory Usage
//!
//! - Standard Adam: O(2n) memory for parameters of size n
//! - AdaFactor 1D: O(n) memory (no first moment) or O(2n) (with first moment)
//! - AdaFactor 2D: O(sqrt(n)) memory for n = rows × cols matrix
//!
//! This makes AdaFactor particularly useful for large transformer models where
//! parameter matrices can be very large.

use crate::common::{OptimizerState, StateMemoryStats};
use crate::traits::StatefulOptimizer;
use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Optimizer;

/// Configuration for AdaFactor optimizer.
#[derive(Debug, Clone)]
pub struct AdaFactorConfig {
    /// Base learning rate (used when scale_parameter is false)
    pub lr: f32,
    /// First moment coefficient (None disables first moment estimation)
    pub beta1: Option<f32>,
    /// Second moment decay rate coefficient
    pub beta2: f32,
    /// Term added for numerical stability
    pub eps: f32,
    /// Gradient clipping threshold
    pub clip_threshold: f32,
    /// Decay rate for beta2 (negative for AdaFactor behavior)
    pub decay_rate_sqrt: f32,
    /// Whether to decay beta1 over time
    pub beta1_decay: bool,
    /// Whether to scale learning rate by parameter RMS
    pub scale_parameter: bool,
    /// Whether to use relative step size
    pub relative_step_size: bool,
    /// Whether to use warmup initialization
    pub warmup_init: bool,
    /// Whether to use factorized second moments for 2D tensors
    pub factorize: bool,
}

impl Default for AdaFactorConfig {
    fn default() -> Self {
        Self {
            lr: 1e-2,
            beta1: None, // Memory efficient default
            beta2: -0.8,
            eps: 1e-30,
            clip_threshold: 1.0,
            decay_rate_sqrt: 0.8,
            beta1_decay: true,
            scale_parameter: true,
            relative_step_size: true,
            warmup_init: false,
            factorize: true,
        }
    }
}

/// AdaFactor optimizer with factorized second moment estimation.
///
/// Implements the AdaFactor algorithm from "Adafactor: Adaptive Learning Rates
/// with Sublinear Memory Cost" by Shazeer & Stern (2018). This optimizer reduces
/// memory usage by factorizing the second moment matrix while maintaining
/// competitive performance.
#[derive(Debug)]
pub struct AdaFactor {
    /// Configuration for this optimizer
    config: AdaFactorConfig,
    /// Optimizer state tracking steps
    state: OptimizerState,
    /// First moment estimates (m_t) - optional for memory efficiency
    exp_avg: HashMap<String, Vec<f32>>,
    /// Factorized second moment estimates for rows
    exp_avg_sq_row: HashMap<String, Vec<f32>>,
    /// Factorized second moment estimates for columns
    exp_avg_sq_col: HashMap<String, Vec<f32>>,
    /// Full second moment estimates for 1D tensors
    exp_avg_sq: HashMap<String, Vec<f32>>,
    /// Parameter shapes for determining factorization strategy
    param_shapes: HashMap<String, Vec<usize>>,
}

impl AdaFactor {
    /// Creates a new AdaFactor optimizer with default settings.
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::AdaFactor;
    /// let optimizer = AdaFactor::new();
    /// ```
    pub fn new() -> Self {
        Self::from_config(AdaFactorConfig::default())
    }

    /// Creates a new AdaFactor optimizer with custom settings.
    ///
    /// # Arguments
    ///
    /// * `scale_parameter` - Whether to scale learning rate by parameter RMS
    /// * `relative_step_size` - Whether to use relative step size
    /// * `warmup_init` - Whether to use warmup initialization
    /// * `beta1` - First moment coefficient (None for memory efficiency)
    /// * `clip_threshold` - Gradient clipping threshold
    /// * `decay_rate` - Beta2 decay rate
    /// * `epsilon` - Numerical stability term
    /// * `factorize` - Whether to factorize 2D tensors
    ///
    /// # Example
    ///
    /// ```
    /// use trustformers_optim::AdaFactor;
    /// let optimizer = AdaFactor::with_config(
    ///     true,          // scale_parameter
    ///     true,          // relative_step_size
    ///     false,         // warmup_init
    ///     Some(0.0),     // beta1
    ///     1.0,           // clip_threshold
    ///     0.8,           // decay_rate
    ///     1e-30,         // epsilon
    ///     true,          // factorize
    /// );
    /// ```
    pub fn with_config(
        scale_parameter: bool,
        relative_step_size: bool,
        warmup_init: bool,
        beta1: Option<f32>,
        clip_threshold: f32,
        decay_rate: f32,
        epsilon: f32,
        factorize: bool,
    ) -> Self {
        let config = AdaFactorConfig {
            lr: if scale_parameter && relative_step_size { 1.0 } else { 1e-2 },
            beta1,
            beta2: -decay_rate,
            eps: epsilon,
            clip_threshold,
            decay_rate_sqrt: decay_rate,
            beta1_decay: true,
            scale_parameter,
            relative_step_size,
            warmup_init,
            factorize,
        };
        Self::from_config(config)
    }

    /// Creates a new AdaFactor optimizer from configuration.
    pub fn from_config(config: AdaFactorConfig) -> Self {
        Self {
            config,
            state: OptimizerState::new(),
            exp_avg: HashMap::new(),
            exp_avg_sq_row: HashMap::new(),
            exp_avg_sq_col: HashMap::new(),
            exp_avg_sq: HashMap::new(),
            param_shapes: HashMap::new(),
        }
    }

    /// Gets the current effective learning rate.
    fn get_lr(&self) -> f32 {
        if self.config.relative_step_size {
            let min_step = if self.config.warmup_init { 1e-6 } else { 1e-2 };
            let rel_step_sz = ((self.state.step + 1) as f32).powf(-0.5).min(min_step);
            if self.config.scale_parameter {
                rel_step_sz * self.config.lr.sqrt()
            } else {
                rel_step_sz
            }
        } else {
            self.config.lr
        }
    }

    /// Gets the current effective beta1.
    fn get_beta1(&self) -> f32 {
        if let Some(beta1) = self.config.beta1 {
            if self.config.beta1_decay {
                beta1 * (1.0 - ((self.state.step + 1) as f32).powf(-self.config.decay_rate_sqrt))
            } else {
                beta1
            }
        } else {
            0.0
        }
    }

    /// Gets the current effective beta2.
    fn get_beta2(&self) -> f32 {
        1.0 - ((self.state.step + 1) as f32).powf(self.config.beta2)
    }

    /// Determines whether to use factorized second moment for given shape.
    fn should_use_factored_second_moment(&self, shape: &[usize]) -> bool {
        self.config.factorize && shape.len() >= 2
    }

    /// Computes approximate squared gradient from factorized moments.
    fn approximate_sq_grad(
        &self,
        exp_avg_sq_row: &[f32],
        exp_avg_sq_col: &[f32],
        shape: &[usize],
    ) -> Vec<f32> {
        if shape.len() < 2 {
            return Vec::new();
        }

        let (rows, cols) = (shape[0], shape[1]);
        let mut result = vec![0.0; rows * cols];

        // Compute outer product approximation
        for i in 0..rows {
            for j in 0..cols {
                result[i * cols + j] = exp_avg_sq_row[i] * exp_avg_sq_col[j];
            }
        }

        // Normalize by geometric mean to maintain scale
        let r_factor: f32 = exp_avg_sq_row.iter().sum::<f32>() / rows as f32;
        let c_factor: f32 = exp_avg_sq_col.iter().sum::<f32>() / cols as f32;
        let norm_factor = (r_factor * c_factor).sqrt();

        if norm_factor > 0.0 {
            for val in result.iter_mut() {
                *val /= norm_factor;
            }
        }

        result
    }
}

impl Optimizer for AdaFactor {
    fn update(&mut self, parameter: &mut Tensor, grad: &Tensor) -> Result<()> {
        match (parameter, grad) {
            (Tensor::F32(param), Tensor::F32(grad_arr)) => {
                let param_id = format!("{:p}", param.as_ptr());
                let size = grad_arr.len();

                // Determine parameter shape for factorization strategy
                let shape = self.param_shapes.get(&param_id).cloned().unwrap_or_else(|| {
                    // For 1D tensors, use the length as shape
                    // For larger tensors, assume square matrix if possible
                    if size == 1 {
                        vec![1]
                    } else {
                        let sqrt_size = (size as f32).sqrt() as usize;
                        if sqrt_size * sqrt_size == size {
                            vec![sqrt_size, sqrt_size]
                        } else {
                            vec![size]
                        }
                    }
                });

                // Store shape for next time
                self.param_shapes.insert(param_id.clone(), shape.clone());

                let factored = self.should_use_factored_second_moment(&shape);
                let lr = self.get_lr();
                let beta1 = self.get_beta1();
                let beta2 = self.get_beta2();

                // Apply gradient clipping
                let grad_norm_sq: f32 = grad_arr.iter().map(|g| g * g).sum();
                let grad_norm = grad_norm_sq.sqrt();
                let clip_coeff = (self.config.clip_threshold / grad_norm.max(1e-8)).min(1.0);

                // Initialize states if needed
                let use_first_moment = self.config.beta1.is_some();
                if use_first_moment && !self.exp_avg.contains_key(&param_id) {
                    self.exp_avg.insert(param_id.clone(), vec![0.0; size]);
                }

                if factored && shape.len() >= 2 {
                    // Factorized second moment for 2D tensors
                    let (rows, cols) = (shape[0], shape[1]);
                    if !self.exp_avg_sq_row.contains_key(&param_id) {
                        self.exp_avg_sq_row.insert(param_id.clone(), vec![0.0; rows]);
                        self.exp_avg_sq_col.insert(param_id.clone(), vec![0.0; cols]);
                    }
                } else {
                    // Full second moment for 1D tensors
                    if !self.exp_avg_sq.contains_key(&param_id) {
                        self.exp_avg_sq.insert(param_id.clone(), vec![0.0; size]);
                    }
                }

                // Update first moment if enabled
                let clipped_grad: Vec<f32> = grad_arr.iter().map(|g| g * clip_coeff).collect();
                if use_first_moment {
                    if let Some(exp_avg) = self.exp_avg.get_mut(&param_id) {
                        for (m, &g) in exp_avg.iter_mut().zip(clipped_grad.iter()) {
                            *m = beta1 * *m + (1.0 - beta1) * g;
                        }
                    }
                }

                // Update second moment
                if factored && shape.len() >= 2 {
                    // Factorized update for 2D tensors
                    let (rows, cols) = (shape[0], shape[1]);

                    if let (Some(exp_avg_sq_row), Some(exp_avg_sq_col)) = (
                        self.exp_avg_sq_row.get_mut(&param_id),
                        self.exp_avg_sq_col.get_mut(&param_id),
                    ) {
                        // Compute row-wise and column-wise squared gradient means
                        for i in 0..rows {
                            let mut row_sq_sum = 0.0;
                            for j in 0..cols {
                                let g = clipped_grad[i * cols + j];
                                row_sq_sum += g * g;
                            }
                            let row_sq_mean = row_sq_sum / cols as f32;
                            exp_avg_sq_row[i] =
                                beta2 * exp_avg_sq_row[i] + (1.0 - beta2) * row_sq_mean;
                        }

                        for j in 0..cols {
                            let mut col_sq_sum = 0.0;
                            for i in 0..rows {
                                let g = clipped_grad[i * cols + j];
                                col_sq_sum += g * g;
                            }
                            let col_sq_mean = col_sq_sum / rows as f32;
                            exp_avg_sq_col[j] =
                                beta2 * exp_avg_sq_col[j] + (1.0 - beta2) * col_sq_mean;
                        }
                    }
                } else {
                    // Full second moment for 1D tensors
                    if let Some(exp_avg_sq) = self.exp_avg_sq.get_mut(&param_id) {
                        for (v, &g) in exp_avg_sq.iter_mut().zip(clipped_grad.iter()) {
                            *v = beta2 * *v + (1.0 - beta2) * g * g;
                        }
                    }
                }

                // Compute parameter update
                if factored && shape.len() >= 2 {
                    // Use factorized approximation
                    if let (Some(exp_avg_sq_row), Some(exp_avg_sq_col)) = (
                        self.exp_avg_sq_row.get(&param_id),
                        self.exp_avg_sq_col.get(&param_id),
                    ) {
                        let exp_avg_sq_approx =
                            self.approximate_sq_grad(exp_avg_sq_row, exp_avg_sq_col, &shape);

                        let update_vals: Vec<f32> = if use_first_moment {
                            if let Some(exp_avg) = self.exp_avg.get(&param_id) {
                                exp_avg
                                    .iter()
                                    .zip(exp_avg_sq_approx.iter())
                                    .map(|(m, v)| m / (v.sqrt() + self.config.eps))
                                    .collect()
                            } else {
                                clipped_grad
                                    .iter()
                                    .zip(exp_avg_sq_approx.iter())
                                    .map(|(g, v)| g / (v.sqrt() + self.config.eps))
                                    .collect()
                            }
                        } else {
                            clipped_grad
                                .iter()
                                .zip(exp_avg_sq_approx.iter())
                                .map(|(g, v)| g / (v.sqrt() + self.config.eps))
                                .collect()
                        };

                        // Apply update with learning rate scaling
                        let effective_lr = if self.config.scale_parameter {
                            let param_rms: f32 =
                                (param.iter().map(|p| p * p).sum::<f32>() / size as f32).sqrt();
                            lr * param_rms.max(self.config.eps)
                        } else {
                            lr
                        };

                        for (p, &update) in param.iter_mut().zip(update_vals.iter()) {
                            *p -= effective_lr * update;
                        }
                    }
                } else {
                    // Use full second moment
                    if let Some(exp_avg_sq) = self.exp_avg_sq.get(&param_id) {
                        let update_vals: Vec<f32> = if use_first_moment {
                            if let Some(exp_avg) = self.exp_avg.get(&param_id) {
                                exp_avg
                                    .iter()
                                    .zip(exp_avg_sq.iter())
                                    .map(|(m, v)| m / (v.sqrt() + self.config.eps))
                                    .collect()
                            } else {
                                clipped_grad
                                    .iter()
                                    .zip(exp_avg_sq.iter())
                                    .map(|(g, v)| g / (v.sqrt() + self.config.eps))
                                    .collect()
                            }
                        } else {
                            clipped_grad
                                .iter()
                                .zip(exp_avg_sq.iter())
                                .map(|(g, v)| g / (v.sqrt() + self.config.eps))
                                .collect()
                        };

                        // Apply update with learning rate scaling
                        let effective_lr = if self.config.scale_parameter {
                            let param_rms: f32 =
                                (param.iter().map(|p| p * p).sum::<f32>() / size as f32).sqrt();
                            lr * param_rms.max(self.config.eps)
                        } else {
                            lr
                        };

                        for (p, &update) in param.iter_mut().zip(update_vals.iter()) {
                            *p -= effective_lr * update;
                        }
                    }
                }

                Ok(())
            },
            _ => Err(TrustformersError::tensor_op_error(
                "Unsupported tensor types for AdaFactor",
                "AdaFactor::update",
            )),
        }
    }

    fn zero_grad(&mut self) {}

    fn step(&mut self) {
        self.state.step += 1;
    }

    fn get_lr(&self) -> f32 {
        self.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.lr = lr;
    }
}

impl StatefulOptimizer for AdaFactor {
    type Config = AdaFactorConfig;
    type State = OptimizerState;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.state
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();

        // Save configuration
        state_dict.insert("lr".to_string(), Tensor::new(vec![self.config.lr])?);
        if let Some(beta1) = self.config.beta1 {
            state_dict.insert("beta1".to_string(), Tensor::new(vec![beta1])?);
        }
        state_dict.insert("beta2".to_string(), Tensor::new(vec![self.config.beta2])?);
        state_dict.insert("eps".to_string(), Tensor::new(vec![self.config.eps])?);
        state_dict.insert(
            "clip_threshold".to_string(),
            Tensor::new(vec![self.config.clip_threshold])?,
        );
        state_dict.insert(
            "step".to_string(),
            Tensor::new(vec![self.state.step as f32])?,
        );

        // Save moment buffers
        for (param_id, exp_avg) in &self.exp_avg {
            state_dict.insert(
                format!("exp_avg_{}", param_id),
                Tensor::new(exp_avg.clone())?,
            );
        }

        for (param_id, exp_avg_sq) in &self.exp_avg_sq {
            state_dict.insert(
                format!("exp_avg_sq_{}", param_id),
                Tensor::new(exp_avg_sq.clone())?,
            );
        }

        for (param_id, exp_avg_sq_row) in &self.exp_avg_sq_row {
            state_dict.insert(
                format!("exp_avg_sq_row_{}", param_id),
                Tensor::new(exp_avg_sq_row.clone())?,
            );
        }

        for (param_id, exp_avg_sq_col) in &self.exp_avg_sq_col {
            state_dict.insert(
                format!("exp_avg_sq_col_{}", param_id),
                Tensor::new(exp_avg_sq_col.clone())?,
            );
        }

        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        // Load configuration
        if let Some(lr_tensor) = state.get("lr") {
            if let Ok(lr_vec) = lr_tensor.data() {
                if !lr_vec.is_empty() {
                    self.config.lr = lr_vec[0];
                }
            }
        }

        // Load moment buffers
        for (key, tensor) in state.iter() {
            if key.starts_with("exp_avg_") && !key.contains("sq") {
                let param_id = key.trim_start_matches("exp_avg_");
                if let Ok(exp_avg) = tensor.data() {
                    self.exp_avg.insert(param_id.to_string(), exp_avg.clone());
                }
            } else if key.starts_with("exp_avg_sq_row_") {
                let param_id = key.trim_start_matches("exp_avg_sq_row_");
                if let Ok(exp_avg_sq_row) = tensor.data() {
                    self.exp_avg_sq_row.insert(param_id.to_string(), exp_avg_sq_row.clone());
                }
            } else if key.starts_with("exp_avg_sq_col_") {
                let param_id = key.trim_start_matches("exp_avg_sq_col_");
                if let Ok(exp_avg_sq_col) = tensor.data() {
                    self.exp_avg_sq_col.insert(param_id.to_string(), exp_avg_sq_col.clone());
                }
            } else if key.starts_with("exp_avg_sq_") {
                let param_id = key.trim_start_matches("exp_avg_sq_");
                if let Ok(exp_avg_sq) = tensor.data() {
                    self.exp_avg_sq.insert(param_id.to_string(), exp_avg_sq.clone());
                }
            }
        }

        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        let mut momentum_elements = 0;
        let mut variance_elements = 0;

        for exp_avg in self.exp_avg.values() {
            momentum_elements += exp_avg.len();
        }

        for exp_avg_sq in self.exp_avg_sq.values() {
            variance_elements += exp_avg_sq.len();
        }

        for exp_avg_sq_row in self.exp_avg_sq_row.values() {
            variance_elements += exp_avg_sq_row.len();
        }

        for exp_avg_sq_col in self.exp_avg_sq_col.values() {
            variance_elements += exp_avg_sq_col.len();
        }

        let total_elements = momentum_elements + variance_elements;
        let total_bytes = total_elements * std::mem::size_of::<f32>();

        StateMemoryStats {
            momentum_elements,
            variance_elements,
            third_moment_elements: 0,
            total_bytes,
            num_parameters: momentum_elements,
        }
    }

    fn reset_state(&mut self) {
        self.state.step = 0;
        self.exp_avg.clear();
        self.exp_avg_sq.clear();
        self.exp_avg_sq_row.clear();
        self.exp_avg_sq_col.clear();
        self.param_shapes.clear();
    }

    fn num_parameters(&self) -> usize {
        self.exp_avg.values().map(|v| v.len()).sum::<usize>()
            + self.exp_avg_sq.values().map(|v| v.len()).sum::<usize>()
    }
}

impl Default for AdaFactor {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adafactor_creation() {
        let optimizer = AdaFactor::new();
        assert!(optimizer.config.scale_parameter);
        assert!(optimizer.config.relative_step_size);
        assert!(optimizer.config.factorize);
    }

    #[test]
    fn test_adafactor_with_config() {
        let optimizer = AdaFactor::with_config(
            true,      // scale_parameter
            true,      // relative_step_size
            false,     // warmup_init
            Some(0.9), // beta1
            1.0,       // clip_threshold
            0.8,       // decay_rate
            1e-30,     // epsilon
            true,      // factorize
        );
        assert_eq!(optimizer.config.beta1, Some(0.9));
        assert!(optimizer.config.scale_parameter);
    }

    #[test]
    fn test_factorized_second_moment_check() {
        let config = AdaFactorConfig::default();
        let optimizer = AdaFactor::from_config(config);

        assert!(!optimizer.should_use_factored_second_moment(&[100])); // 1D
        assert!(optimizer.should_use_factored_second_moment(&[10, 20])); // 2D
        assert!(optimizer.should_use_factored_second_moment(&[5, 10, 15])); // 3D
    }

    #[test]
    fn test_learning_rate_scaling() {
        let mut optimizer = AdaFactor::with_config(true, true, false, None, 1.0, 0.8, 1e-30, true);
        optimizer.state.step = 100;

        let lr = optimizer.get_lr();
        assert!(lr > 0.0);
        assert!(lr < 1.0); // Should be scaled down due to step^(-0.5)
    }

    #[test]
    fn test_beta_decay() {
        let mut optimizer =
            AdaFactor::with_config(true, true, false, Some(0.9), 1.0, 0.8, 1e-30, true);

        optimizer.state.step = 0;
        let beta1_0 = optimizer.get_beta1();

        optimizer.state.step = 100;
        let beta1_100 = optimizer.get_beta1();

        assert!(beta1_100 > beta1_0); // Beta1 should increase with decay formula
    }

    #[test]
    fn test_approximate_sq_grad() {
        let optimizer = AdaFactor::new();
        let exp_avg_sq_row = vec![0.1, 0.2];
        let exp_avg_sq_col = vec![0.3, 0.4];
        let shape = vec![2, 2];

        let result = optimizer.approximate_sq_grad(&exp_avg_sq_row, &exp_avg_sq_col, &shape);
        assert_eq!(result.len(), 4);
        assert!(result.iter().all(|&x| x >= 0.0));
    }

    #[test]
    fn test_memory_efficiency() {
        let optimizer = AdaFactor::new();
        let stats = optimizer.memory_usage();

        // Initially should have no memory usage
        assert_eq!(stats.total_bytes, 0);
        assert_eq!(stats.momentum_elements, 0);
        assert_eq!(stats.variance_elements, 0);
    }
}
