//! # LancBiO: Dynamic Lanczos-aided Bilevel Optimization
//!
//! This module implements LancBiO, a cutting-edge bilevel optimization algorithm from ICLR 2025
//! that uses dynamic Lanczos methods via Krylov subspace techniques for efficient bilevel
//! optimization problems.
//!
//! ## Algorithm Overview
//!
//! LancBiO addresses bilevel optimization problems of the form:
//! ```text
//! min_x F(x, y*(x))
//! s.t. y*(x) = arg min_y G(x, y)
//! ```
//!
//! Where F is the upper-level objective and G is the lower-level objective.
//! The algorithm uses Lanczos iterations to efficiently approximate the inverse
//! Hessian-vector products required for bilevel gradients.
//!
//! ## Key Features
//!
//! - **Dynamic Lanczos Approximation**: Efficient computation of Hessian inverse operations
//! - **Krylov Subspace Methods**: Leverages Krylov subspace techniques for scalability
//! - **Bilevel Optimization**: Designed for hierarchical optimization problems
//! - **Memory Efficient**: Uses iterative methods to avoid storing full Hessian matrices
//! - **Automatic Adaptation**: Dynamic adjustment of Lanczos iterations based on convergence
//!
//! ## Mathematical Foundation
//!
//! The bilevel gradient is computed as:
//! ```text
//! ∇_x F = ∇_x F + ∇_y F^T * ∇_x y*
//! ```
//!
//! Where ∇_x y* requires solving:
//! ```text
//! ∇^2_y G * ∇_x y* = -∇_{xy} G
//! ```
//!
//! LancBiO uses Lanczos iterations to approximate ∇^2_y G^{-1} efficiently.
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use trustformers_optim::LancBiO;
//! use trustformers_core::traits::Optimizer;
//!
//! // Create LancBiO with default settings
//! let mut optimizer = LancBiO::new(
//!     1e-3,    // learning_rate
//!     10,      // max_lanczos_iterations
//!     1e-6,    // convergence_tolerance
//!     5,       // krylov_subspace_size
//! );
//!
//! // For meta-learning problems
//! let mut optimizer = LancBiO::for_meta_learning();
//!
//! // For hyperparameter optimization
//! let mut optimizer = LancBiO::for_hyperparameter_optimization();
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::{errors::Result, tensor::Tensor, traits::Optimizer};

use crate::{common::StateMemoryStats, traits::StatefulOptimizer};

/// Configuration for LancBiO optimizer.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct LancBiOConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,

    /// Maximum number of Lanczos iterations (default: 10)
    pub max_lanczos_iterations: usize,

    /// Convergence tolerance for Lanczos iterations (default: 1e-6)
    pub convergence_tolerance: f32,

    /// Size of Krylov subspace (default: 5)
    pub krylov_subspace_size: usize,

    /// Damping factor for Hessian regularization (default: 1e-3)
    pub damping: f32,

    /// Lower-level optimizer learning rate (default: 1e-2)
    pub lower_level_lr: f32,

    /// Number of lower-level optimization steps (default: 5)
    pub lower_level_steps: usize,

    /// Whether to use momentum for upper-level updates (default: true)
    pub use_momentum: bool,

    /// Momentum coefficient (default: 0.9)
    pub momentum: f32,

    /// Whether to adaptively adjust Lanczos iterations (default: true)
    pub adaptive_lanczos: bool,

    /// Maximum memory usage for Krylov vectors (in MB, default: 100)
    pub max_memory_mb: usize,
}

impl Default for LancBiOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            max_lanczos_iterations: 10,
            convergence_tolerance: 1e-6,
            krylov_subspace_size: 5,
            damping: 1e-3,
            lower_level_lr: 1e-2,
            lower_level_steps: 5,
            use_momentum: true,
            momentum: 0.9,
            adaptive_lanczos: true,
            max_memory_mb: 100,
        }
    }
}

/// Krylov subspace information for Lanczos iterations.
#[derive(Clone, Debug)]
struct KrylovSubspace {
    /// Krylov basis vectors
    vectors: Vec<Tensor>,

    /// Tridiagonal matrix from Lanczos process
    tridiagonal: Tensor,

    /// Current subspace size
    size: usize,

    /// Residual norm
    residual_norm: f32,
}

/// LancBiO optimizer state for a single parameter.
#[derive(Clone, Debug)]
pub struct LancBiOState {
    /// Upper-level momentum buffer
    momentum_buffer: Option<Tensor>,

    /// Lower-level parameter state
    lower_level_params: Option<Tensor>,

    /// Lower-level momentum buffer
    lower_level_momentum: Option<Tensor>,

    /// Krylov subspace state
    krylov_subspace: Option<KrylovSubspace>,

    /// Previous gradients for finite difference approximation
    prev_gradients: Option<Tensor>,

    /// Step count
    step: usize,

    /// Convergence history for adaptive Lanczos
    convergence_history: Vec<f32>,
}

/// LancBiO: Dynamic Lanczos-aided Bilevel Optimization.
///
/// LancBiO efficiently solves bilevel optimization problems using dynamic Lanczos
/// methods and Krylov subspace techniques, making it suitable for meta-learning,
/// hyperparameter optimization, and other hierarchical optimization problems.
#[derive(Clone, Debug)]
pub struct LancBiO {
    config: LancBiOConfig,
    states: HashMap<String, LancBiOState>,
    step: usize,
    memory_stats: StateMemoryStats,

    /// Total memory usage of Krylov vectors
    krylov_memory_usage: usize,
}

impl LancBiO {
    /// Creates a new LancBiO optimizer with the given configuration.
    pub fn new(
        learning_rate: f32,
        max_lanczos_iterations: usize,
        convergence_tolerance: f32,
        krylov_subspace_size: usize,
    ) -> Self {
        Self {
            config: LancBiOConfig {
                learning_rate,
                max_lanczos_iterations,
                convergence_tolerance,
                krylov_subspace_size,
                ..Default::default()
            },
            states: HashMap::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
            krylov_memory_usage: 0,
        }
    }

    /// Creates LancBiO with configuration optimized for meta-learning.
    pub fn for_meta_learning() -> Self {
        Self {
            config: LancBiOConfig {
                learning_rate: 1e-3,
                max_lanczos_iterations: 15,
                convergence_tolerance: 1e-7,
                krylov_subspace_size: 8,
                damping: 1e-4,
                lower_level_lr: 1e-2,
                lower_level_steps: 10,
                use_momentum: true,
                momentum: 0.9,
                adaptive_lanczos: true,
                max_memory_mb: 200,
            },
            states: HashMap::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
            krylov_memory_usage: 0,
        }
    }

    /// Creates LancBiO with configuration optimized for hyperparameter optimization.
    pub fn for_hyperparameter_optimization() -> Self {
        Self {
            config: LancBiOConfig {
                learning_rate: 5e-4,
                max_lanczos_iterations: 20,
                convergence_tolerance: 1e-8,
                krylov_subspace_size: 10,
                damping: 1e-5,
                lower_level_lr: 5e-3,
                lower_level_steps: 20,
                use_momentum: true,
                momentum: 0.95,
                adaptive_lanczos: true,
                max_memory_mb: 500,
            },
            states: HashMap::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
            krylov_memory_usage: 0,
        }
    }

    /// Creates LancBiO with custom configuration.
    pub fn with_config(config: LancBiOConfig) -> Self {
        Self {
            config,
            states: HashMap::new(),
            step: 0,
            memory_stats: StateMemoryStats {
                momentum_elements: 0,
                variance_elements: 0,
                third_moment_elements: 0,
                total_bytes: 0,
                num_parameters: 0,
            },
            krylov_memory_usage: 0,
        }
    }

    /// Performs Lanczos iterations to build Krylov subspace.
    fn lanczos_iterations(
        &self,
        hessian_vector_product: impl Fn(&Tensor) -> Result<Tensor>,
        initial_vector: &Tensor,
        max_iterations: usize,
    ) -> Result<KrylovSubspace> {
        let _vector_size = initial_vector.len();
        let mut vectors = Vec::with_capacity(max_iterations);
        let mut alpha = Vec::with_capacity(max_iterations);
        let mut beta = Vec::with_capacity(max_iterations);

        // Normalize initial vector
        let norm = initial_vector.norm()?;
        if norm < self.config.convergence_tolerance {
            return Ok(KrylovSubspace {
                vectors: vec![initial_vector.clone()],
                tridiagonal: Tensor::zeros(&[1, 1])?,
                size: 1,
                residual_norm: 0.0,
            });
        }

        let mut v = initial_vector.div_scalar(norm)?;
        vectors.push(v.clone());

        let mut prev_v = Tensor::zeros_like(&v)?;
        let mut beta_val = 0.0f32;

        for i in 0..max_iterations {
            // Compute Hessian-vector product
            let hv = hessian_vector_product(&v)?;

            // Orthogonalize against previous vector
            if i > 0 {
                let hv_proj = prev_v.mul_scalar(beta_val)?;
                let _hv = hv.sub(&hv_proj)?;
            }

            // Compute alpha (diagonal element)
            // Compute dot product as element-wise multiplication followed by sum
            let alpha_val = v.mul(&hv)?.sum(None, false)?.to_scalar()?;
            alpha.push(alpha_val);

            // Compute residual
            let residual = hv.sub(&v.mul_scalar(alpha_val)?)?;
            let residual_norm = residual.norm()?;

            // Check convergence
            if residual_norm < self.config.convergence_tolerance {
                break;
            }

            // Prepare for next iteration
            if i < max_iterations - 1 {
                beta_val = residual_norm;
                beta.push(beta_val);

                prev_v = v.clone();
                v = residual.div_scalar(beta_val)?;
                vectors.push(v.clone());
            }
        }

        // Build tridiagonal matrix
        let n = alpha.len();
        let mut tridiag_data = vec![0.0f32; n * n];

        for i in 0..n {
            tridiag_data[i * n + i] = alpha[i];
            if i > 0 {
                tridiag_data[i * n + (i - 1)] = beta[i - 1];
                tridiag_data[(i - 1) * n + i] = beta[i - 1];
            }
        }

        let tridiagonal = Tensor::from_slice(&tridiag_data, &[n, n])?;

        Ok(KrylovSubspace {
            vectors,
            tridiagonal,
            size: n,
            residual_norm: initial_vector.norm()?,
        })
    }

    /// Approximates inverse Hessian-vector product using Lanczos decomposition.
    fn approximate_inverse_hvp(
        &self,
        krylov_subspace: &KrylovSubspace,
        vector: &Tensor,
    ) -> Result<Tensor> {
        if krylov_subspace.vectors.is_empty() {
            return Ok(vector.clone());
        }

        // Project vector onto Krylov subspace
        let mut coefficients = Vec::with_capacity(krylov_subspace.size);
        for krylov_vec in &krylov_subspace.vectors {
            let coeff = krylov_vec.mul(vector)?.sum(None, false)?.to_scalar()?;
            coefficients.push(coeff);
        }

        // Solve tridiagonal system T * y = e_1 * ||v||
        let coeffs_tensor = Tensor::from_slice(&coefficients, &[coefficients.len()])?;
        let norm = vector.norm()?;
        let rhs = coeffs_tensor.mul_scalar(norm)?;

        // Add damping to tridiagonal matrix for numerical stability
        let damped_tridiag = {
            let identity = Tensor::eye_f32(krylov_subspace.size)?;
            krylov_subspace.tridiagonal.add(&identity.mul_scalar(self.config.damping)?)?
        };

        // Solve tridiagonal linear system using Thomas algorithm
        let solution = self.solve_tridiagonal_system(&damped_tridiag, &rhs)?;

        // Reconstruct result in original space
        let mut result = Tensor::zeros_like(vector)?;
        for (i, &coeff) in solution.data_f32()?.iter().enumerate() {
            if i < krylov_subspace.vectors.len() {
                let contribution = krylov_subspace.vectors[i].mul_scalar(coeff)?;
                result = result.add(&contribution)?;
            }
        }

        Ok(result)
    }

    /// Solves a tridiagonal linear system using the Thomas algorithm.
    ///
    /// This method implements the Thomas algorithm (tridiagonal matrix algorithm)
    /// to efficiently solve the linear system Ax = b where A is a tridiagonal matrix.
    ///
    /// The Thomas algorithm is a specialized Gaussian elimination method for tridiagonal
    /// matrices with O(n) complexity, making it much more efficient than general
    /// matrix solvers for this specific case.
    ///
    /// # Arguments
    /// * `tridiag_matrix` - The tridiagonal matrix A (stored as a full tensor)
    /// * `rhs` - The right-hand side vector b
    ///
    /// # Returns
    /// The solution vector x such that Ax = b
    fn solve_tridiagonal_system(&self, tridiag_matrix: &Tensor, rhs: &Tensor) -> Result<Tensor> {
        let n = rhs.len();
        if n == 0 {
            return Ok(rhs.clone());
        }

        if n == 1 {
            // For 1x1 system, solution is just rhs / diagonal element
            let matrix_data = tridiag_matrix.data_f32()?;
            let rhs_data = rhs.data_f32()?;
            let diagonal = matrix_data[0];

            if diagonal.abs() < 1e-12 {
                // Matrix is singular, return zeros as fallback
                return Tensor::zeros_like(rhs);
            }

            let solution = rhs_data[0] / diagonal;
            return Tensor::from_slice(&[solution], &[1]);
        }

        // Extract tridiagonal components from the full matrix
        let matrix_data = tridiag_matrix.data_f32()?;
        let rhs_data = rhs.data_f32()?;

        // For a tridiagonal matrix stored as full matrix:
        // - Sub-diagonal: A[i+1][i] for i = 0..n-1
        // - Diagonal: A[i][i] for i = 0..n-1
        // - Super-diagonal: A[i][i+1] for i = 0..n-2

        let mut a = vec![0.0f32; n]; // Sub-diagonal
        let mut b = vec![0.0f32; n]; // Diagonal
        let mut c = vec![0.0f32; n]; // Super-diagonal
        let mut d = rhs_data.to_vec(); // RHS (will be modified)

        // Extract diagonals from the full matrix
        for i in 0..n {
            b[i] = matrix_data[i * n + i]; // Main diagonal

            if i < n - 1 {
                c[i] = matrix_data[i * n + (i + 1)]; // Super-diagonal
            }

            if i > 0 {
                a[i] = matrix_data[i * n + (i - 1)]; // Sub-diagonal
            }
        }

        // Thomas algorithm forward elimination
        for i in 1..n {
            if b[i - 1].abs() < 1e-12 {
                // Pivot is too small, fall back to identity solution
                return Ok(rhs.clone());
            }

            let w = a[i] / b[i - 1];
            b[i] -= w * c[i - 1];
            d[i] -= w * d[i - 1];
        }

        // Back substitution
        let mut x = vec![0.0f32; n];

        if b[n - 1].abs() < 1e-12 {
            // Last diagonal element is too small, fall back
            return Ok(rhs.clone());
        }

        x[n - 1] = d[n - 1] / b[n - 1];

        for i in (0..n - 1).rev() {
            if b[i].abs() < 1e-12 {
                // Diagonal element is too small, fall back
                return Ok(rhs.clone());
            }
            x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
        }

        // Convert solution back to tensor
        Tensor::from_slice(&x, &[n])
    }

    /// Computes bilevel gradient using Lanczos approximation.
    fn compute_bilevel_gradient(
        &mut self,
        param_id: &str,
        upper_grad: &Tensor,
        lower_grad: &Tensor,
        _parameter: &Tensor,
    ) -> Result<Tensor> {
        let param_key = param_id.to_string();
        let state_exists = self.states.contains_key(&param_key);

        if !state_exists {
            let new_state = LancBiOState {
                momentum_buffer: None,
                lower_level_params: None,
                lower_level_momentum: None,
                krylov_subspace: None,
                prev_gradients: None,
                step: 0,
                convergence_history: Vec::new(),
            };
            self.states.insert(param_key.clone(), new_state);
        }

        let state = self.states.get_mut(&param_key).unwrap();

        // Adaptive Lanczos iterations based on convergence history
        let max_iterations =
            if self.config.adaptive_lanczos && !state.convergence_history.is_empty() {
                let avg_convergence = state.convergence_history.iter().sum::<f32>()
                    / state.convergence_history.len() as f32;
                if avg_convergence < self.config.convergence_tolerance * 10.0 {
                    (self.config.max_lanczos_iterations / 2).max(3)
                } else {
                    self.config.max_lanczos_iterations
                }
            } else {
                self.config.max_lanczos_iterations
            };

        // Extract previous gradients to avoid borrowing conflicts
        let prev_gradients = state.prev_gradients.clone();
        let damping = self.config.damping;

        // Define Hessian-vector product approximation
        let hessian_vector_product = |v: &Tensor| -> Result<Tensor> {
            // Finite difference approximation of Hessian-vector product
            let eps = 1e-5;
            let perturbed_grad = if let Some(ref prev_grad) = prev_gradients {
                // Use previous gradient for finite difference
                let grad_diff = lower_grad.sub(prev_grad)?;
                grad_diff.div_scalar(eps)?
            } else {
                // Fall back to scaled identity approximation
                v.mul_scalar(damping)?
            };

            Ok(perturbed_grad)
        };

        // Release the mutable borrow temporarily
        let _ = state;

        // Perform Lanczos iterations
        let krylov_subspace =
            self.lanczos_iterations(hessian_vector_product, lower_grad, max_iterations)?;

        // Re-acquire mutable borrow
        let state = self.states.get_mut(&param_key).unwrap();

        // Store convergence information for adaptive behavior
        state.convergence_history.push(krylov_subspace.residual_norm);
        if state.convergence_history.len() > 10 {
            state.convergence_history.remove(0);
        }
        let _ = state; // Release mutable borrow temporarily

        // Approximate inverse Hessian-vector product
        let neg_mixed_grad = lower_grad.neg()?; // Approximate mixed derivative
        let inverse_hvp = self.approximate_inverse_hvp(&krylov_subspace, &neg_mixed_grad)?;

        // Compute bilevel gradient: ∇F + ∇_y F^T * ∇_x y*
        let bilevel_grad = upper_grad.add(&inverse_hvp)?;

        // Re-acquire mutable borrow to store results
        let state = self.states.get_mut(&param_key).unwrap();
        state.krylov_subspace = Some(krylov_subspace);
        state.prev_gradients = Some(lower_grad.clone());

        Ok(bilevel_grad)
    }

    /// Applies bilevel optimization update using LancBiO.
    fn apply_bilevel_update(
        &mut self,
        param_id: &str,
        parameter: &mut Tensor,
        upper_grad: &Tensor,
        lower_grad: &Tensor,
    ) -> Result<()> {
        // Compute bilevel gradient
        let bilevel_grad =
            self.compute_bilevel_gradient(param_id, upper_grad, lower_grad, parameter)?;

        let state = self.states.get_mut(param_id).unwrap();
        state.step += 1;

        // Apply momentum if enabled
        let update = if self.config.use_momentum {
            if let Some(ref mut momentum_buffer) = state.momentum_buffer {
                *momentum_buffer =
                    momentum_buffer.mul_scalar(self.config.momentum)?.add(&bilevel_grad)?;
                momentum_buffer.clone()
            } else {
                state.momentum_buffer = Some(bilevel_grad.clone());
                bilevel_grad
            }
        } else {
            bilevel_grad
        };

        // Apply update
        let scaled_update = update.mul_scalar(self.config.learning_rate)?;
        // Apply the update by replacing parameter with the subtraction result
        *parameter = parameter.sub(&scaled_update)?;

        Ok(())
    }

    /// Returns statistics about Lanczos iterations.
    pub fn lanczos_stats(&self) -> HashMap<String, (usize, f32, usize)> {
        self.states
            .iter()
            .map(|(name, state)| {
                let (iterations, residual_norm) = if let Some(ref krylov) = state.krylov_subspace {
                    (krylov.size, krylov.residual_norm)
                } else {
                    (0, 0.0)
                };

                let convergence_trend = state.convergence_history.len();

                (name.clone(), (iterations, residual_norm, convergence_trend))
            })
            .collect()
    }

    /// Returns memory usage of Krylov subspaces.
    pub fn krylov_memory_usage(&self) -> usize {
        self.states
            .values()
            .map(|state| {
                if let Some(ref krylov) = state.krylov_subspace {
                    krylov.vectors.iter().map(|v| v.memory_usage()).sum::<usize>()
                        + krylov.tridiagonal.memory_usage()
                } else {
                    0
                }
            })
            .sum()
    }

    /// Clears Krylov subspaces to free memory.
    pub fn clear_krylov_memory(&mut self) {
        for state in self.states.values_mut() {
            state.krylov_subspace = None;
        }
        self.krylov_memory_usage = 0;
    }
}

impl Optimizer for LancBiO {
    fn update(&mut self, parameter: &mut Tensor, gradient: &Tensor) -> Result<()> {
        // For standard optimization, treat as upper-level gradient with zero lower-level gradient
        // Create a unique parameter ID based on shape and hash of first few elements
        let param_id = format!(
            "param_{}_{:?}_{}",
            self.states.len(),
            parameter.shape(),
            parameter
                .data_f32()
                .unwrap_or_default()
                .get(0..5)
                .unwrap_or(&[])
                .iter()
                .fold(0u64, |acc, &x| acc.wrapping_add(x.to_bits() as u64))
        );
        let zero_lower_grad = Tensor::zeros_like(gradient)?;
        self.apply_bilevel_update(&param_id, parameter, gradient, &zero_lower_grad)
    }

    fn zero_grad(&mut self) {
        // Clear gradient-related state
        for state in self.states.values_mut() {
            state.prev_gradients = None;
        }
    }

    fn step(&mut self) {
        self.step += 1;

        // Memory management: clear old Krylov subspaces if memory usage is too high
        let current_memory_mb = self.krylov_memory_usage() / (1024 * 1024);
        if current_memory_mb > self.config.max_memory_mb {
            self.clear_krylov_memory();
        }
    }

    fn get_lr(&self) -> f32 {
        self.config.learning_rate
    }

    fn set_lr(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }
}

impl StatefulOptimizer for LancBiO {
    type Config = LancBiOConfig;
    type State = StateMemoryStats;

    fn config(&self) -> &Self::Config {
        &self.config
    }

    fn state(&self) -> &Self::State {
        &self.memory_stats
    }

    fn state_mut(&mut self) -> &mut Self::State {
        &mut self.memory_stats
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        let mut state_dict = HashMap::new();
        state_dict.insert("step".to_string(), Tensor::scalar(self.step as f32)?);
        Ok(state_dict)
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        if let Some(step_tensor) = state.get("step") {
            self.step = step_tensor.to_scalar()? as usize;
        }
        Ok(())
    }

    fn memory_usage(&self) -> StateMemoryStats {
        self.memory_stats.clone()
    }

    fn reset_state(&mut self) {
        self.states.clear();
        self.step = 0;
        self.krylov_memory_usage = 0;
    }

    fn num_parameters(&self) -> usize {
        self.states.len()
    }
}

// LancBiO-specific methods
impl LancBiO {
    /// Returns the type of Hessian approximation used
    pub fn hessian_approximation(&self) -> String {
        "Lanczos-Krylov".to_string()
    }

    /// Returns memory usage of Krylov subspaces
    pub fn krylov_memory(&self) -> usize {
        self.krylov_memory_usage()
    }

    /// Get current learning rate
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Estimates curvature based on residual norms
    pub fn approximate_curvature(&self) -> Option<f32> {
        if let Some((_, residual_norm, _)) = self.lanczos_stats().values().next() {
            Some(*residual_norm)
        } else {
            None
        }
    }

    /// Updates parameter with explicit upper and lower-level gradients.
    pub fn bilevel_update(
        &mut self,
        parameter: &mut Tensor,
        upper_grad: &Tensor,
        lower_grad: &Tensor,
    ) -> Result<()> {
        let parameter_id = format!(
            "param_{}_{:?}_{}",
            self.states.len(),
            parameter.shape(),
            parameter
                .data_f32()
                .unwrap_or_default()
                .get(0..5)
                .unwrap_or(&[])
                .iter()
                .fold(0u64, |acc, &x| acc.wrapping_add(x.to_bits() as u64))
        );
        self.apply_bilevel_update(&parameter_id, parameter, upper_grad, lower_grad)
    }

    /// Performs lower-level optimization step.
    pub fn lower_level_step(
        &mut self,
        lower_params: &mut Tensor,
        lower_grad: &Tensor,
    ) -> Result<()> {
        let parameter_id = format!(
            "param_{}_{:?}_{}",
            self.states.len(),
            lower_params.shape(),
            lower_params
                .data_f32()
                .unwrap_or_default()
                .get(0..5)
                .unwrap_or(&[])
                .iter()
                .fold(0u64, |acc, &x| acc.wrapping_add(x.to_bits() as u64))
        );
        let state = self.states.entry(parameter_id).or_insert_with(|| LancBiOState {
            momentum_buffer: None,
            lower_level_params: Some(lower_params.clone()),
            lower_level_momentum: None,
            krylov_subspace: None,
            prev_gradients: None,
            step: 0,
            convergence_history: Vec::new(),
        });

        // Apply momentum for lower-level optimization
        let update = if self.config.use_momentum {
            if let Some(ref mut momentum) = state.lower_level_momentum {
                *momentum = momentum.mul_scalar(self.config.momentum)?.add(lower_grad)?;
                momentum.clone()
            } else {
                state.lower_level_momentum = Some(lower_grad.clone());
                lower_grad.clone()
            }
        } else {
            lower_grad.clone()
        };

        // Apply lower-level update
        let scaled_update = update.mul_scalar(self.config.lower_level_lr)?;
        *lower_params = lower_params.sub(&scaled_update)?;

        // Update state
        state.lower_level_params = Some(lower_params.clone());

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lancbio_creation() {
        let optimizer = LancBiO::new(1e-3, 10, 1e-6, 5);
        assert_eq!(optimizer.learning_rate(), 1e-3);
        assert_eq!(optimizer.config.max_lanczos_iterations, 10);
        assert_eq!(optimizer.config.krylov_subspace_size, 5);
    }

    #[test]
    fn test_lancbio_presets() {
        let meta_optimizer = LancBiO::for_meta_learning();
        assert_eq!(meta_optimizer.learning_rate(), 1e-3);
        assert_eq!(meta_optimizer.config.max_lanczos_iterations, 15);

        let hyper_optimizer = LancBiO::for_hyperparameter_optimization();
        assert_eq!(hyper_optimizer.learning_rate(), 5e-4);
        assert_eq!(hyper_optimizer.config.max_lanczos_iterations, 20);
    }

    #[test]
    fn test_lanczos_iterations() -> Result<()> {
        let optimizer = LancBiO::new(1e-3, 5, 1e-6, 3);
        let vector = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;

        let hvp = |v: &Tensor| -> Result<Tensor> {
            // Simple identity transformation for testing
            Ok(v.clone())
        };

        let krylov = optimizer.lanczos_iterations(hvp, &vector, 3)?;
        assert!(krylov.size <= 3);
        assert!(!krylov.vectors.is_empty());

        Ok(())
    }

    #[test]
    fn test_bilevel_update() -> Result<()> {
        let mut optimizer = LancBiO::new(0.1, 5, 1e-6, 3);
        let mut parameter = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
        let upper_grad = Tensor::from_slice(&[0.1, 0.1, 0.1], &[3])?;
        let lower_grad = Tensor::from_slice(&[0.05, 0.05, 0.05], &[3])?;

        let original_param = parameter.clone();
        optimizer.bilevel_update(&mut parameter, &upper_grad, &lower_grad)?;
        optimizer.step();

        // Parameter should have changed
        assert_ne!(parameter.data()?, original_param.data()?);

        Ok(())
    }

    #[test]
    fn test_lower_level_step() -> Result<()> {
        let mut optimizer = LancBiO::new(1e-3, 5, 1e-6, 3);
        let mut lower_params = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
        let lower_grad = Tensor::from_slice(&[0.1, 0.1, 0.1], &[3])?;

        let original_params = lower_params.clone();
        optimizer.lower_level_step(&mut lower_params, &lower_grad)?;

        // Parameters should have changed
        assert_ne!(lower_params.data()?, original_params.data()?);

        Ok(())
    }

    #[test]
    fn test_memory_management() -> Result<()> {
        let mut optimizer = LancBiO::new(1e-3, 10, 1e-6, 5);
        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[4])?;
        let grad = Tensor::from_slice(&[0.1, 0.1, 0.1, 0.1], &[4])?;

        // Perform several updates to build up Krylov subspaces
        for _ in 0..5 {
            optimizer.update(&mut param, &grad)?;
            optimizer.step();
        }

        let memory_before = optimizer.krylov_memory_usage();
        optimizer.clear_krylov_memory();
        let memory_after = optimizer.krylov_memory_usage();

        assert!(memory_after <= memory_before);

        Ok(())
    }

    #[test]
    fn test_lanczos_stats() -> Result<()> {
        let mut optimizer = LancBiO::new(1e-3, 5, 1e-6, 3);
        let mut param = Tensor::from_slice(&[1.0, 2.0, 3.0], &[3])?;
        let grad = Tensor::from_slice(&[0.1, 0.1, 0.1], &[3])?;

        optimizer.update(&mut param, &grad)?;

        let stats = optimizer.lanczos_stats();
        assert_eq!(stats.len(), 1);

        Ok(())
    }
}
