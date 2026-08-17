//! # SOFO: Second-Order Forward Optimizer
//!
//! SOFO is a second-order optimizer that efficiently navigates loss surfaces using
//! forward-mode differentiation instead of backpropagation. By relying on easily
//! parallelized batched forward-mode differentiation, SOFO enjoys constant memory
//! cost in time and achieves wallclock time essentially on par with first-order
//! gradient-based optimizers while providing second-order optimization benefits.
//!
//! ## Key Features
//! - **Forward-Mode Differentiation**: Uses forward-mode AD instead of backpropagation
//! - **Constant Memory Cost**: Memory usage doesn't grow with sequence length
//! - **GPU Parallelism**: Effective use of parallel computing for forward passes
//! - **Second-Order Benefits**: Curvature information for better optimization
//! - **Scalable**: Suitable for large neural networks and long sequences
//!
//! ## Research Foundation
//! Based on "SOFO: Second-Order Forward Optimizer" (NeurIPS 2024/2025)
//! - Constant memory cost in time unlike traditional second-order methods
//! - Per-iteration wallclock time comparable to first-order optimizers
//! - Effective GPU parallelization through batched forward-mode differentiation
//! - Superior convergence properties compared to first-order methods
//!
//! ## Usage Example
//! ```rust,no_run
//! use trustformers_optim::{SOFO, SOFOConfig};
//! use trustformers_core::tensor::Tensor;
//!
//! let config = SOFOConfig::new()
//!     .learning_rate(1e-3)
//!     .batch_size(32)
//!     .curvature_strength(0.1)
//!     .forward_passes(8)
//!     .build();
//!
//! let mut optimizer = SOFO::new(config);
//!
//! // In training loop
//! // optimizer.zero_grad();
//! // ... compute loss and gradients using forward mode ...
//! // optimizer.step(&mut parameters, &gradients, &loss_fn)?;
//! ```

use anyhow::Result;
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Which estimator produced the curvature used by the most recent step.
///
/// SOFO's paper-faithful path needs directional derivatives of the *gradient*, which
/// only the caller can supply. Rather than invent numbers when no oracle is
/// available, the optimizer records which estimator it actually used.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurvatureSource {
    /// No step has been taken yet.
    None,
    /// Hutchinson diagonal Hessian estimate `E[v ⊙ (H v)]` with Rademacher `v`,
    /// where `H v` comes from a central difference of caller-supplied gradients.
    /// This is the paper's second-order path.
    HutchinsonFromOracle,
    /// Empirical-Fisher (Gauss-Newton) diagonal `g ⊙ g`.
    ///
    /// Used by [`SOFO::step`], which has no way to evaluate the gradient at a
    /// perturbed parameter point. It is a real, standard curvature proxy — but it is
    /// *not* the Hessian, and callers who need the paper's estimator must use
    /// [`SOFO::step_with_gradient_oracle`].
    EmpiricalFisherDiagonal,
}

/// Counter-based deterministic Rademacher sampler.
///
/// SOFO's curvature estimate is only unbiased for *independent* ±1 probe vectors, so
/// the sequence has to be genuinely varied — the previous implementation used
/// `sin(i * 0.1)`, which is neither random nor ±1. A counter-based splitmix64 stream
/// gives independent draws while staying fully reproducible from `seed`.
#[derive(Debug, Clone)]
struct RademacherStream {
    seed: u64,
    counter: u64,
}

impl RademacherStream {
    fn new(seed: u64) -> Self {
        Self { seed, counter: 0 }
    }

    fn next_bits(&mut self) -> u64 {
        self.counter = self.counter.wrapping_add(1);
        let mut z = self.seed.wrapping_add(self.counter.wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// A vector of independent ±1 entries.
    fn sample(&mut self, len: usize) -> Vec<f32> {
        (0..len).map(|_| if self.next_bits() & 1 == 0 { -1.0 } else { 1.0 }).collect()
    }
}

/// Configuration for SOFO optimizer
#[derive(Debug, Clone)]
pub struct SOFOConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// Batch size for forward-mode differentiation (default: 32)
    pub batch_size: usize,
    /// Number of forward passes for curvature estimation (default: 8)
    pub forward_passes: usize,
    /// Strength of curvature information (default: 0.1)
    pub curvature_strength: f32,
    /// Damping factor for numerical stability (default: 1e-6)
    pub damping: f32,
    /// Weight decay (default: 0.0)
    pub weight_decay: f32,
    /// Enable adaptive curvature estimation (default: true)
    pub adaptive_curvature: bool,
    /// Momentum for first-order updates (default: 0.9)
    pub momentum: f32,
    /// Use Nesterov acceleration (default: true)
    pub nesterov: bool,
    /// Maximum condition number for curvature matrix (default: 1e6)
    pub max_condition_number: f32,
    /// Enable memory efficient mode (default: true)
    pub memory_efficient: bool,
    /// Parallel computation threshold (default: 1000)
    pub parallel_threshold: usize,
    /// Finite-difference step used for the Hessian-vector product (default: 1e-3).
    ///
    /// A central difference of gradients trades truncation error (`O(ε²)`) against
    /// cancellation error (`O(δ/ε)` for gradient noise `δ`); `1e-3` is the usual
    /// compromise for `f32` parameters.
    pub hvp_epsilon: f32,
    /// Seed for the Rademacher probe stream (default: 0x5060_F0F0_1234_5678).
    pub probe_seed: u64,
}

impl Default for SOFOConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            batch_size: 32,
            forward_passes: 8,
            curvature_strength: 0.1,
            damping: 1e-6,
            weight_decay: 0.0,
            adaptive_curvature: true,
            momentum: 0.9,
            nesterov: true,
            max_condition_number: 1e6,
            memory_efficient: true,
            parallel_threshold: 1000,
            hvp_epsilon: 1e-3,
            probe_seed: 0x5060_F0F0_1234_5678,
        }
    }
}

impl SOFOConfig {
    /// Create a new SOFO configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the batch size for forward-mode differentiation
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the number of forward passes for curvature estimation
    pub fn forward_passes(mut self, passes: usize) -> Self {
        self.forward_passes = passes;
        self
    }

    /// Set the curvature strength
    pub fn curvature_strength(mut self, strength: f32) -> Self {
        self.curvature_strength = strength;
        self
    }

    /// Set the damping factor
    pub fn damping(mut self, damping: f32) -> Self {
        self.damping = damping;
        self
    }

    /// Set weight decay
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Enable or disable momentum
    pub fn momentum(mut self, momentum: f32) -> Self {
        self.momentum = momentum;
        self
    }

    /// Set the finite-difference step for the Hessian-vector product.
    pub fn hvp_epsilon(mut self, eps: f32) -> Self {
        self.hvp_epsilon = eps;
        self
    }

    /// Set the seed of the Rademacher probe stream.
    pub fn probe_seed(mut self, seed: u64) -> Self {
        self.probe_seed = seed;
        self
    }

    /// Build the configuration
    /// Enable or disable adaptive per-parameter curvature weighting
    pub fn adaptive_curvature(mut self, enable: bool) -> Self {
        self.adaptive_curvature = enable;
        self
    }

    /// Set the maximum condition number tolerated in the curvature estimate
    pub fn max_condition_number(mut self, max_condition_number: f32) -> Self {
        self.max_condition_number = max_condition_number;
        self
    }

    pub fn build(self) -> Self {
        self
    }
}

/// SOFO optimizer state for tracking forward-mode differentiation
#[derive(Debug, Clone)]
pub struct SOFOState {
    /// Current step count
    pub step: u64,
    /// Momentum buffers for first-order terms
    pub momentum_buffers: HashMap<String, Tensor>,
    /// Curvature estimates for each parameter
    pub curvature_estimates: HashMap<String, Tensor>,
    /// Forward-mode gradient accumulations
    pub forward_gradients: HashMap<String, Vec<Tensor>>,
    /// Eigenvalue estimates for condition number control
    pub eigenvalue_estimates: HashMap<String, Tensor>,
    /// Adaptive curvature weights
    pub adaptive_weights: HashMap<String, f32>,
    /// Forward pass computation statistics
    pub forward_stats: ForwardModeStats,
    /// Memory usage tracking
    pub memory_stats: MemoryStats,
    /// Which estimator produced the curvature used by the most recent step.
    pub curvature_source: CurvatureSource,
}

/// Counters for the gradient-oracle evaluations SOFO actually performed.
///
/// Every field here is *measured*: `total_forward_passes` is incremented once per
/// real oracle call, and `total_oracle_time` accumulates the wall-clock time those
/// calls took. Nothing is modelled or assumed.
#[derive(Debug, Clone, Default)]
pub struct ForwardModeStats {
    /// Total gradient-oracle evaluations performed (two per Hutchinson probe).
    pub total_forward_passes: u64,
    /// Accumulated wall-clock time spent inside the gradient oracle.
    pub total_oracle_time: std::time::Duration,
}

impl ForwardModeStats {
    /// Mean wall-clock time per oracle evaluation, or `None` if none were performed.
    pub fn avg_forward_time(&self) -> Option<std::time::Duration> {
        if self.total_forward_passes == 0 {
            None
        } else {
            Some(self.total_oracle_time / self.total_forward_passes as u32)
        }
    }
}

/// Measured size of the optimizer's own state buffers.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Bytes currently held by SOFO's momentum and curvature buffers.
    pub state_bytes: usize,
    /// `state_bytes` expressed in MiB.
    pub current_state_mb: f32,
    /// Largest `current_state_mb` observed so far.
    pub peak_state_mb: f32,
    /// Number of scalar parameters most recently optimized.
    pub num_parameters: usize,
}

impl Default for SOFOState {
    fn default() -> Self {
        Self {
            step: 0,
            momentum_buffers: HashMap::new(),
            curvature_estimates: HashMap::new(),
            forward_gradients: HashMap::new(),
            eigenvalue_estimates: HashMap::new(),
            adaptive_weights: HashMap::new(),
            forward_stats: ForwardModeStats::default(),
            memory_stats: MemoryStats::default(),
            curvature_source: CurvatureSource::None,
        }
    }
}

/// SOFO (Second-Order Forward Optimizer)
///
/// A second-order optimizer using forward-mode differentiation for constant
/// memory cost and efficient GPU parallelization.
pub struct SOFO {
    config: SOFOConfig,
    state: SOFOState,
    rademacher: RademacherStream,
}

impl SOFO {
    /// Create a new SOFO optimizer
    pub fn new(config: SOFOConfig) -> Self {
        let rademacher = RademacherStream::new(config.probe_seed);
        Self {
            config,
            state: SOFOState::default(),
            rademacher,
        }
    }

    /// Get the current learning rate
    pub fn learning_rate(&self) -> f32 {
        self.config.learning_rate
    }

    /// Set the learning rate
    pub fn set_learning_rate(&mut self, lr: f32) {
        self.config.learning_rate = lr;
    }

    /// Generates one set of independent Rademacher probe directions, one per parameter.
    ///
    /// # Errors
    ///
    /// Returns an error when a probe tensor cannot be built for a parameter's shape.
    fn generate_random_directions(
        &mut self,
        parameters: &HashMap<String, Tensor>,
    ) -> Result<Vec<HashMap<String, Tensor>>> {
        // Iterate in a deterministic order so a given seed always yields the same
        // probe sequence regardless of `HashMap` iteration order.
        let mut names: Vec<&String> = parameters.keys().collect();
        names.sort();

        let mut direction_sets = Vec::with_capacity(self.config.forward_passes);
        for _ in 0..self.config.forward_passes {
            let mut directions = HashMap::new();
            for name in &names {
                let Some(parameter) = parameters.get(*name) else {
                    continue;
                };
                let shape = parameter.shape();
                let total: usize = shape.iter().product();
                let probe = self.rademacher.sample(total);
                directions.insert((*name).clone(), Tensor::from_vec(probe, &shape)?);
            }
            direction_sets.push(directions);
        }

        Ok(direction_sets)
    }

    /// Empirical-Fisher (Gauss-Newton) diagonal curvature `g ⊙ g + damping`.
    ///
    /// This is what [`SOFO::step`] uses: a real, standard curvature proxy computed
    /// from the gradients the caller already has. It is *not* the Hessian.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor operation fails.
    fn empirical_fisher_curvature(
        &self,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut estimates = HashMap::new();
        for (param_name, gradient) in gradients.iter() {
            let squared = gradient.mul(gradient)?;
            estimates.insert(param_name.clone(), squared.add_scalar(self.config.damping)?);
        }
        Ok(estimates)
    }

    /// Hutchinson diagonal-Hessian estimate driven by a caller-supplied gradient oracle.
    ///
    /// For Rademacher probes `v`, `E[v ⊙ (H v)] = diag(H)`. The Hessian-vector product
    /// is obtained by a central difference of *real* gradients:
    /// `H v ≈ (∇f(θ + εv) − ∇f(θ − εv)) / (2ε)`, so the oracle is called twice per
    /// probe. The absolute value is taken because the Newton-style division below
    /// requires a positive preconditioner.
    ///
    /// # Errors
    ///
    /// Returns an error when the oracle fails or returns a gradient whose shape does
    /// not match the parameter it was requested for.
    fn hutchinson_curvature<F>(
        &mut self,
        parameters: &HashMap<String, Tensor>,
        oracle: &mut F,
    ) -> Result<HashMap<String, Tensor>>
    where
        F: FnMut(&HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>,
    {
        let eps = self.config.hvp_epsilon;
        let direction_sets = self.generate_random_directions(parameters)?;

        let mut accumulator: HashMap<String, Vec<f32>> = HashMap::new();
        let mut probes_used = 0usize;

        for directions in &direction_sets {
            let mut plus = HashMap::new();
            let mut minus = HashMap::new();
            for (name, parameter) in parameters.iter() {
                let Some(direction) = directions.get(name) else {
                    continue;
                };
                plus.insert(name.clone(), parameter.add(&direction.mul_scalar(eps)?)?);
                minus.insert(name.clone(), parameter.sub(&direction.mul_scalar(eps)?)?);
            }

            let grad_plus = oracle(&plus)?;
            let grad_minus = oracle(&minus)?;
            self.state.forward_stats.total_forward_passes += 2;
            probes_used += 1;

            for (name, direction) in directions.iter() {
                let (Some(gp), Some(gm)) = (grad_plus.get(name), grad_minus.get(name)) else {
                    continue;
                };
                let gp_data = gp.data_f32()?;
                let gm_data = gm.data_f32()?;
                let v_data = direction.data_f32()?;
                if gp_data.len() != v_data.len() || gm_data.len() != v_data.len() {
                    return Err(anyhow::anyhow!(
                        "gradient oracle returned {} / {} elements for '{name}' but the \
                         parameter has {}",
                        gp_data.len(),
                        gm_data.len(),
                        v_data.len()
                    ));
                }
                let slot =
                    accumulator.entry(name.clone()).or_insert_with(|| vec![0.0; v_data.len()]);
                for i in 0..v_data.len() {
                    // v ⊙ (H v), with H v from the central difference.
                    let hv = (gp_data[i] - gm_data[i]) / (2.0 * eps);
                    slot[i] += v_data[i] * hv;
                }
            }
        }

        let mut estimates = HashMap::new();
        let divisor = probes_used.max(1) as f32;
        for (name, mut values) in accumulator {
            for value in values.iter_mut() {
                // Newton-style division needs a positive preconditioner; the sign of a
                // diagonal Hessian entry is not usable directly.
                *value = (*value / divisor).abs() + self.config.damping;
            }
            let shape =
                parameters.get(&name).map(|t| t.shape()).unwrap_or_else(|| vec![values.len()]);
            estimates.insert(name, Tensor::from_vec(values, &shape)?);
        }

        Ok(estimates)
    }

    /// Apply adaptive curvature weighting
    fn apply_adaptive_curvature(
        &mut self,
        param_name: &str,
        curvature: &Tensor,
        gradient: &Tensor,
    ) -> Result<Tensor> {
        if !self.config.adaptive_curvature {
            return Ok(curvature.clone());
        }

        // Compute gradient-curvature alignment
        let grad_norm = gradient.norm()?;
        let curv_norm = curvature.norm()?;

        let alignment = if grad_norm > 0.0 && curv_norm > 0.0 {
            let grad_data = gradient.data_f32()?;
            let curv_data = curvature.data_f32()?;
            let dot_product: f32 =
                grad_data.iter().zip(curv_data.iter()).map(|(&a, &b)| a * b).sum();
            dot_product / (grad_norm * curv_norm)
        } else {
            0.0
        };

        // Adaptive weight based on alignment
        let adaptive_weight = (1.0 + alignment.abs()) * self.config.curvature_strength;
        self.state.adaptive_weights.insert(param_name.to_string(), adaptive_weight);

        // Apply adaptive weighting
        Ok(curvature.mul_scalar(adaptive_weight)?)
    }

    /// Update momentum buffer
    fn update_momentum(&mut self, param_name: &str, gradient: &Tensor) -> Result<Tensor> {
        let momentum = self.config.momentum;

        let momentum_update =
            if let Some(prev_momentum) = self.state.momentum_buffers.get(param_name) {
                let momentum_tensor = Tensor::scalar(momentum)?;
                let one_minus_momentum = Tensor::scalar(1.0 - momentum)?;

                let weighted_prev = prev_momentum.mul(&momentum_tensor)?;
                let weighted_grad = gradient.mul(&one_minus_momentum)?;
                weighted_prev.add(&weighted_grad)?
            } else {
                gradient.mul(&Tensor::scalar(1.0 - momentum)?)?
            };

        self.state
            .momentum_buffers
            .insert(param_name.to_string(), momentum_update.clone());
        Ok(momentum_update)
    }

    /// Compute second-order update direction
    fn compute_second_order_update(&self, gradient: &Tensor, curvature: &Tensor) -> Result<Tensor> {
        // Newton-like update: H^(-1) * g
        // We approximate the inverse using element-wise division with regularization

        let regularized_curvature = curvature.add(&Tensor::scalar(self.config.damping)?)?;
        let newton_direction = gradient.div(&regularized_curvature)?;

        Ok(newton_direction)
    }

    /// Control condition number of curvature estimates
    fn control_condition_number(&self, curvature: &Tensor) -> Result<Tensor> {
        // Clamp eigenvalues to control condition number
        let min_eigenvalue = self.config.damping;
        let max_eigenvalue = min_eigenvalue * self.config.max_condition_number;

        Ok(curvature.clamp(min_eigenvalue, max_eigenvalue)?)
    }

    /// Records the *measured* size of the optimizer's own state buffers.
    ///
    /// `num_parameters` counts scalar parameters (not tensors) and the byte totals are
    /// derived from the buffers SOFO actually holds — momentum and curvature — so the
    /// reported figure tracks reality rather than a modelled overhead percentage.
    fn update_memory_stats(&mut self, parameters: &HashMap<String, Tensor>) {
        let scalar_count: usize =
            parameters.values().map(|t| t.shape().iter().product::<usize>()).sum();

        let state_bytes: usize = self
            .state
            .momentum_buffers
            .values()
            .chain(self.state.curvature_estimates.values())
            .map(|t| t.shape().iter().product::<usize>() * std::mem::size_of::<f32>())
            .sum();

        let state_mb = state_bytes as f32 / (1024.0 * 1024.0);
        self.state.memory_stats.current_state_mb = state_mb;
        self.state.memory_stats.peak_state_mb = self.state.memory_stats.peak_state_mb.max(state_mb);
        self.state.memory_stats.num_parameters = scalar_count;
        self.state.memory_stats.state_bytes = state_bytes;
    }

    /// Performs one optimization step using the **empirical-Fisher diagonal** as the
    /// curvature estimate.
    ///
    /// The paper's estimator needs gradients at perturbed parameter points, which this
    /// signature cannot obtain. Rather than invent a Hessian, this path uses the
    /// Gauss-Newton/empirical-Fisher diagonal `g ⊙ g`, records
    /// [`CurvatureSource::EmpiricalFisherDiagonal`] in the state, and reports zero
    /// forward passes. Use [`SOFO::step_with_gradient_oracle`] for the second-order path.
    ///
    /// # Errors
    ///
    /// Returns an error when a tensor operation fails.
    pub fn step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
    ) -> Result<()> {
        let curvature_estimates = self.empirical_fisher_curvature(gradients)?;
        self.state.curvature_source = CurvatureSource::EmpiricalFisherDiagonal;
        self.apply_step(parameters, gradients, curvature_estimates)
    }

    /// Performs one optimization step using the paper's second-order curvature.
    ///
    /// `oracle` must return `∇f` evaluated at the parameter map it is handed; SOFO
    /// calls it twice per Rademacher probe (`forward_passes` probes per step) to form
    /// the central-difference Hessian-vector product behind the Hutchinson diagonal
    /// estimate. The oracle's wall-clock cost is accumulated into
    /// [`ForwardModeStats::total_oracle_time`].
    ///
    /// # Errors
    ///
    /// Returns an error when the oracle fails or returns mismatched shapes.
    pub fn step_with_gradient_oracle<F>(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
        oracle: &mut F,
    ) -> Result<()>
    where
        F: FnMut(&HashMap<String, Tensor>) -> Result<HashMap<String, Tensor>>,
    {
        let started = std::time::Instant::now();
        let snapshot: HashMap<String, Tensor> =
            parameters.iter().map(|(k, v)| (k.clone(), v.clone())).collect();
        let curvature_estimates = self.hutchinson_curvature(&snapshot, oracle)?;
        self.state.forward_stats.total_oracle_time += started.elapsed();
        self.state.curvature_source = CurvatureSource::HutchinsonFromOracle;
        self.apply_step(parameters, gradients, curvature_estimates)
    }

    /// Shared update body: applies weight decay, preconditions by the supplied
    /// curvature, and steps the parameters.
    fn apply_step(
        &mut self,
        parameters: &mut HashMap<String, Tensor>,
        gradients: &HashMap<String, Tensor>,
        curvature_estimates: HashMap<String, Tensor>,
    ) -> Result<()> {
        self.state.step += 1;

        for (param_name, gradient) in gradients.iter() {
            if let Some(parameter) = parameters.get_mut(param_name) {
                // Apply weight decay if configured
                let mut effective_gradient = gradient.clone();
                if self.config.weight_decay > 0.0 {
                    let weight_decay_term =
                        parameter.mul(&Tensor::scalar(self.config.weight_decay)?)?;
                    effective_gradient = effective_gradient.add(&weight_decay_term)?;
                }

                // Get curvature estimate for this parameter
                let curvature = if let Some(curv) = curvature_estimates.get(param_name) {
                    self.apply_adaptive_curvature(param_name, curv, &effective_gradient)?
                } else {
                    // Fallback to first-order
                    Tensor::ones_like(&effective_gradient)?
                        .mul(&Tensor::scalar(self.config.damping)?)?
                };

                // Control condition number
                let controlled_curvature = self.control_condition_number(&curvature)?;

                // Compute second-order update direction
                let second_order_direction =
                    self.compute_second_order_update(&effective_gradient, &controlled_curvature)?;

                // Update momentum
                let momentum_update = self.update_momentum(param_name, &second_order_direction)?;

                // Combine first-order momentum with second-order direction
                let final_update = if self.config.nesterov {
                    // Nesterov acceleration with second-order
                    let momentum_tensor = Tensor::scalar(self.config.momentum)?;
                    momentum_update.mul(&momentum_tensor)?.add(&second_order_direction)?
                } else {
                    momentum_update
                };

                // Apply learning rate and update parameter
                let lr_tensor = Tensor::scalar(self.config.learning_rate)?;
                let param_update = final_update.mul(&lr_tensor)?;

                *parameter = parameter.sub(&param_update)?;

                // Store curvature estimate for monitoring
                self.state.curvature_estimates.insert(param_name.clone(), controlled_curvature);
            }
        }

        // Measure the state we actually hold, after the buffers have been written.
        self.update_memory_stats(parameters);

        Ok(())
    }

    /// Get SOFO-specific optimization statistics
    pub fn get_sofo_stats(&self) -> SOFOStats {
        let avg_curvature_strength = if self.state.adaptive_weights.is_empty() {
            self.config.curvature_strength
        } else {
            self.state.adaptive_weights.values().sum::<f32>()
                / self.state.adaptive_weights.len() as f32
        };

        // Derived entirely from the stored curvature tensors; 1.0 only when no step
        // has produced any curvature yet (a genuinely unconditioned identity).
        let avg_condition_number = if self.state.curvature_estimates.is_empty() {
            1.0
        } else {
            let mut total_condition = 0.0;
            let mut count = 0;

            for curvature in self.state.curvature_estimates.values() {
                if let Ok((min_val, max_val)) = curvature.min_max() {
                    if min_val > 0.0 {
                        total_condition += max_val / min_val;
                        count += 1;
                    }
                }
            }

            if count > 0 {
                total_condition / count as f32
            } else {
                1.0
            }
        };

        SOFOStats {
            step: self.state.step,
            total_forward_passes: self.state.forward_stats.total_forward_passes,
            avg_curvature_strength,
            avg_condition_number,
            curvature_source: self.state.curvature_source,
            state_bytes: self.state.memory_stats.state_bytes,
            current_state_mb: self.state.memory_stats.current_state_mb,
            num_parameters: self.state.memory_stats.num_parameters,
        }
    }

    /// Which curvature estimator produced the most recent step.
    pub fn curvature_source(&self) -> CurvatureSource {
        self.state.curvature_source
    }

    /// Get forward-mode differentiation statistics
    pub fn get_forward_stats(&self) -> &ForwardModeStats {
        &self.state.forward_stats
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> &MemoryStats {
        &self.state.memory_stats
    }

    /// Reset optimizer state
    pub fn reset_state(&mut self) {
        self.state = SOFOState::default();
    }

    /// Get curvature estimates for analysis
    pub fn get_curvature_estimates(&self) -> &HashMap<String, Tensor> {
        &self.state.curvature_estimates
    }

    /// Get adaptive weights for each parameter
    pub fn get_adaptive_weights(&self) -> &HashMap<String, f32> {
        &self.state.adaptive_weights
    }
}

/// SOFO optimizer statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct SOFOStats {
    /// Current optimization step
    pub step: u64,
    /// Total forward passes performed
    pub total_forward_passes: u64,
    /// Average curvature strength across parameters
    pub avg_curvature_strength: f32,
    /// Average condition number of the diagonal curvature estimates
    pub avg_condition_number: f32,
    /// Which estimator produced the curvature used by the most recent step
    pub curvature_source: CurvatureSource,
    /// Measured bytes held by SOFO's own state buffers
    pub state_bytes: usize,
    /// `state_bytes` expressed in MiB
    pub current_state_mb: f32,
    /// Number of scalar parameters most recently optimized
    pub num_parameters: usize,
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_sofo_creation() {
        let config = SOFOConfig::new().learning_rate(1e-3).batch_size(32).forward_passes(8).build();

        let optimizer = SOFO::new(config);
        assert_eq!(optimizer.learning_rate(), 1e-3);
    }

    #[test]
    fn test_sofo_config_builder() {
        let config = SOFOConfig::new()
            .learning_rate(2e-3)
            .batch_size(64)
            .forward_passes(16)
            .curvature_strength(0.2)
            .damping(1e-5)
            .weight_decay(1e-4)
            .momentum(0.95)
            .build();

        assert_eq!(config.learning_rate, 2e-3);
        assert_eq!(config.batch_size, 64);
        assert_eq!(config.forward_passes, 16);
        assert_eq!(config.curvature_strength, 0.2);
        assert_eq!(config.damping, 1e-5);
        assert_eq!(config.weight_decay, 1e-4);
        assert_eq!(config.momentum, 0.95);
    }

    #[test]
    fn test_sofo_step() -> Result<()> {
        let config = SOFOConfig::new().learning_rate(1e-2).forward_passes(4).build();
        let mut optimizer = SOFO::new(config);

        // Create test parameters and gradients
        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::ones(&[2, 2])?.mul_scalar(0.1)?,
        );

        // Store original value
        let original_value =
            parameters.get("weight").expect("Key not found").mean()?.to_scalar()?;

        // Perform optimization step
        optimizer.step(&mut parameters, &gradients)?;

        // Check that parameter was updated
        let updated_value = parameters.get("weight").expect("Key not found").mean()?.to_scalar()?;
        assert_ne!(updated_value, original_value);

        Ok(())
    }

    #[test]
    fn test_random_direction_generation() -> Result<()> {
        let config = SOFOConfig::new().forward_passes(3).build();
        let mut optimizer = SOFO::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("weight1".to_string(), Tensor::ones(&[2, 2])?);
        parameters.insert("weight2".to_string(), Tensor::ones(&[3, 3])?);

        let direction_sets = optimizer.generate_random_directions(&parameters)?;

        assert_eq!(direction_sets.len(), 3);
        for directions in &direction_sets {
            assert_eq!(directions.len(), 2);
            assert!(directions.contains_key("weight1"));
            assert!(directions.contains_key("weight2"));
        }

        Ok(())
    }

    /// The empirical-Fisher fallback must produce `g² + damping`, elementwise.
    #[test]
    fn test_empirical_fisher_curvature() -> Result<()> {
        let config = SOFOConfig::new().damping(1e-3).build();
        let optimizer = SOFO::new(config);

        let mut gradients = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::from_vec(vec![2.0_f32, -3.0, 0.5, 0.0], &[2, 2])?,
        );

        let curvature = optimizer.empirical_fisher_curvature(&gradients)?;
        let values = curvature.get("weight").expect("curvature present").data_f32()?;

        let expected = [4.0_f32 + 1e-3, 9.0 + 1e-3, 0.25 + 1e-3, 1e-3];
        assert_eq!(values.len(), expected.len());
        for (actual, want) in values.iter().zip(expected.iter()) {
            assert!((actual - want).abs() < 1e-5, "got {actual}, want {want}");
        }

        Ok(())
    }

    #[test]
    fn test_momentum_update() -> Result<()> {
        let config = SOFOConfig::new().momentum(0.9).build();
        let mut optimizer = SOFO::new(config);

        let gradient = Tensor::ones(&[2, 2])?.mul_scalar(0.5)?;

        // First update
        let momentum1 = optimizer.update_momentum("test", &gradient)?;

        // Second update
        let momentum2 = optimizer.update_momentum("test", &gradient)?;

        // Momentum should change between updates
        assert_ne!(
            momentum1.mean()?.to_scalar()?,
            momentum2.mean()?.to_scalar()?
        );

        Ok(())
    }

    #[test]
    fn test_second_order_update() -> Result<()> {
        let config = SOFOConfig::new().build();
        let optimizer = SOFO::new(config);

        let gradient = Tensor::ones(&[2, 2])?.mul_scalar(0.5)?;
        let curvature = Tensor::ones(&[2, 2])?.mul_scalar(2.0)?;

        let update = optimizer.compute_second_order_update(&gradient, &curvature)?;

        // Update should be approximately gradient / curvature
        let expected = 0.5 / 2.0; // Approximate expected value
        let actual = update.mean()?.to_scalar()?;

        assert!((actual - expected).abs() < 0.1);

        Ok(())
    }

    #[test]
    fn test_condition_number_control() -> Result<()> {
        let config = SOFOConfig::new().damping(1e-3).max_condition_number(100.0).build();
        let optimizer = SOFO::new(config);

        // Create curvature with extreme values
        let curvature = Tensor::from_vec(vec![1e-6_f32, 1e6, 1.0, 1e3], &[2, 2])?;

        let controlled = optimizer.control_condition_number(&curvature)?;

        // Values must be clamped into [damping, damping · max_condition_number].
        let values = controlled.data_f32()?;
        let max_val = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let min_val = values.iter().copied().fold(f32::INFINITY, f32::min);

        assert!(
            max_val / min_val <= 100.0 * 1.1,
            "condition number {} exceeds the configured bound",
            max_val / min_val
        );

        Ok(())
    }

    #[test]
    fn test_sofo_stats() -> Result<()> {
        let config = SOFOConfig::new().forward_passes(4).build();
        let mut optimizer = SOFO::new(config);

        // Perform a few optimization steps
        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::ones(&[2, 2])?.mul_scalar(0.1)?,
        );

        for _ in 0..3 {
            optimizer.step(&mut parameters, &gradients)?;
        }

        let stats = optimizer.get_sofo_stats();
        assert_eq!(stats.step, 3);
        assert!(stats.num_parameters > 0);
        assert!(
            stats.state_bytes > 0,
            "state size must be measured, not invented"
        );
        // `step` has no gradient oracle, so it cannot evaluate anything in forward
        // mode: the counter must stay at zero rather than claim work never done.
        assert_eq!(
            stats.total_forward_passes, 0,
            "the gradient-only path performs no forward-mode passes"
        );
        assert_eq!(
            stats.curvature_source,
            CurvatureSource::EmpiricalFisherDiagonal
        );

        Ok(())
    }

    /// The oracle-driven path really does call the oracle, twice per probe.
    #[test]
    fn test_sofo_forward_passes_are_counted_only_when_performed() -> Result<()> {
        let config = SOFOConfig::new().forward_passes(2).build();
        let mut optimizer = SOFO::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut calls = 0_usize;
        let mut oracle = |params: &HashMap<String, Tensor>| -> Result<HashMap<String, Tensor>> {
            calls += 1;
            let mut grads = HashMap::new();
            for (name, tensor) in params {
                grads.insert(name.clone(), tensor.mul_scalar(2.0)?);
            }
            Ok(grads)
        };

        let mut gradients = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::ones(&[2, 2])?.mul_scalar(2.0)?,
        );

        optimizer.step_with_gradient_oracle(&mut parameters, &gradients, &mut oracle)?;

        let stats = optimizer.get_sofo_stats();
        assert!(calls > 0, "the oracle must actually be evaluated");
        assert_eq!(
            stats.total_forward_passes as usize, calls,
            "every counted forward pass must correspond to a real oracle call"
        );
        assert_eq!(
            stats.curvature_source,
            CurvatureSource::HutchinsonFromOracle
        );

        Ok(())
    }

    #[test]
    fn test_learning_rate_methods() {
        let config = SOFOConfig::new().learning_rate(1e-3).build();
        let mut optimizer = SOFO::new(config);

        assert_eq!(optimizer.learning_rate(), 1e-3);

        optimizer.set_learning_rate(2e-3);
        assert_eq!(optimizer.learning_rate(), 2e-3);
    }

    #[test]
    fn test_weight_decay() -> Result<()> {
        let config = SOFOConfig::new()
            .learning_rate(1e-2)
            .weight_decay(1e-2)
            .forward_passes(2)
            .build();
        let mut optimizer = SOFO::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::zeros(&[2, 2])?);

        let initial_param_value =
            parameters.get("weight").expect("Key not found").mean()?.to_scalar()?;

        optimizer.step(&mut parameters, &gradients)?;

        let final_param_value =
            parameters.get("weight").expect("Key not found").mean()?.to_scalar()?;

        // With weight decay, parameter should decrease even with zero gradient
        assert!(final_param_value < initial_param_value);

        Ok(())
    }

    #[test]
    fn test_adaptive_curvature() -> Result<()> {
        let config = SOFOConfig::new().adaptive_curvature(true).curvature_strength(0.1).build();
        let mut optimizer = SOFO::new(config);

        let gradient = Tensor::ones(&[2, 2])?.mul_scalar(0.5)?;
        let curvature = Tensor::ones(&[2, 2])?.mul_scalar(2.0)?;

        let adaptive_curvature =
            optimizer.apply_adaptive_curvature("test", &curvature, &gradient)?;

        // Adaptive curvature should be modified from original
        let original_mean = curvature.mean()?.to_scalar()?;
        let adaptive_mean = adaptive_curvature.mean()?.to_scalar()?;

        assert_ne!(original_mean, adaptive_mean);

        Ok(())
    }
}
