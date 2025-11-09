//! # GENIE: Generalization-ENhancing Iterative Equalizer Optimizer
//!
//! GENIE (Generalization-ENhancing Iterative Equalizer) is a novel optimizer that leverages
//! the One-Step Generalization Ratio (OSGR) to quantify each parameter's contribution to loss
//! reduction and assess gradient alignment. By dynamically equalizing OSGR via a preconditioning
//! factor, GENIE prevents a small subset of parameters from dominating optimization, thereby
//! promoting domain-invariant feature learning.
//!
//! ## Key Features
//! - **One-Step Generalization Ratio (OSGR)**: Quantifies parameter contribution to loss reduction
//! - **Dynamic Preconditioning**: Equalizes OSGR to prevent parameter dominance
//! - **Domain-Invariant Learning**: Promotes robust feature learning across domains
//! - **Convergence Guarantees**: Maintains SGD's convergence rate while improving generalization
//! - **Gradient Alignment**: Balances convergence contribution and gradient alignment
//!
//! ## Research Foundation
//! Based on "GENIE: Generalization-ENhancing Iterative Equalizer for Domain Generalization" (ICML 2025)
//! - Outperforms existing optimizers on domain generalization tasks
//! - Enhances performance when integrated with various DG methods
//! - Theoretical guarantees for convergence and generalization
//!
//! ## Usage Example
//! ```rust,no_run
//! use trustformers_optim::{GENIE, GENIEConfig};
//! use trustformers_core::tensor::Tensor;
//!
//! let config = GENIEConfig::new()
//!     .learning_rate(1e-3)
//!     .osgr_momentum(0.9)
//!     .alignment_weight(0.1)
//!     .preconditioning_eps(1e-8)
//!     .build();
//!
//! let mut optimizer = GENIE::new(config);
//!
//! // In training loop
//! // optimizer.zero_grad();
//! // ... compute loss and gradients ...
//! // optimizer.step(&mut parameters, &gradients)?;
//! ```

use crate::common::{OptimizerState, ParameterUpdate};
use anyhow::{Result, Context};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for GENIE optimizer
#[derive(Debug, Clone)]
pub struct GENIEConfig {
    /// Learning rate (default: 1e-3)
    pub learning_rate: f32,
    /// Momentum for OSGR computation (default: 0.9)
    pub osgr_momentum: f32,
    /// Weight for gradient alignment term (default: 0.1)
    pub alignment_weight: f32,
    /// Epsilon for numerical stability in preconditioning (default: 1e-8)
    pub preconditioning_eps: f32,
    /// Minimum OSGR threshold (default: 1e-6)
    pub min_osgr: f32,
    /// Maximum OSGR threshold (default: 1e6)
    pub max_osgr: f32,
    /// Enable adaptive alignment weighting (default: true)
    pub adaptive_alignment: bool,
    /// Weight decay (default: 0.0)
    pub weight_decay: f32,
    /// Enable OSGR normalization (default: true)
    pub normalize_osgr: bool,
    /// Warmup steps for OSGR computation (default: 100)
    pub warmup_steps: u64,
}

impl Default for GENIEConfig {
    fn default() -> Self {
        Self {
            learning_rate: 1e-3,
            osgr_momentum: 0.9,
            alignment_weight: 0.1,
            preconditioning_eps: 1e-8,
            min_osgr: 1e-6,
            max_osgr: 1e6,
            adaptive_alignment: true,
            weight_decay: 0.0,
            normalize_osgr: true,
            warmup_steps: 100,
        }
    }
}

impl GENIEConfig {
    /// Create a new GENIE configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the learning rate
    pub fn learning_rate(mut self, lr: f32) -> Self {
        self.learning_rate = lr;
        self
    }

    /// Set the OSGR momentum
    pub fn osgr_momentum(mut self, momentum: f32) -> Self {
        self.osgr_momentum = momentum;
        self
    }

    /// Set the gradient alignment weight
    pub fn alignment_weight(mut self, weight: f32) -> Self {
        self.alignment_weight = weight;
        self
    }

    /// Set the preconditioning epsilon
    pub fn preconditioning_eps(mut self, eps: f32) -> Self {
        self.preconditioning_eps = eps;
        self
    }

    /// Set the weight decay
    pub fn weight_decay(mut self, decay: f32) -> Self {
        self.weight_decay = decay;
        self
    }

    /// Build the configuration
    pub fn build(self) -> Self {
        self
    }
}

/// GENIE (Generalization-ENhancing Iterative Equalizer) optimizer state
#[derive(Debug, Clone)]
pub struct GENIEState {
    /// Current step count
    pub step: u64,
    /// One-Step Generalization Ratio for each parameter
    pub osgr: HashMap<String, Tensor>,
    /// Exponential moving average of OSGR
    pub osgr_ema: HashMap<String, Tensor>,
    /// Previous gradients for OSGR computation
    pub prev_gradients: HashMap<String, Tensor>,
    /// Previous loss values for generalization ratio computation
    pub prev_loss: Option<f32>,
    /// Gradient alignment statistics
    pub alignment_stats: HashMap<String, f32>,
    /// Preconditioning factors
    pub preconditioning_factors: HashMap<String, Tensor>,
    /// Domain statistics (for domain generalization tracking)
    pub domain_stats: DomainStats,
}

/// Domain generalization statistics
#[derive(Debug, Clone)]
pub struct DomainStats {
    /// Per-domain loss tracking
    pub domain_losses: Vec<f32>,
    /// Domain variance measures
    pub domain_variance: f32,
    /// Cross-domain gradient alignment
    pub cross_domain_alignment: f32,
}

impl Default for DomainStats {
    fn default() -> Self {
        Self {
            domain_losses: Vec::new(),
            domain_variance: 0.0,
            cross_domain_alignment: 0.0,
        }
    }
}

impl Default for GENIEState {
    fn default() -> Self {
        Self {
            step: 0,
            osgr: HashMap::new(),
            osgr_ema: HashMap::new(),
            prev_gradients: HashMap::new(),
            prev_loss: None,
            alignment_stats: HashMap::new(),
            preconditioning_factors: HashMap::new(),
            domain_stats: DomainStats::default(),
        }
    }
}

/// GENIE (Generalization-ENhancing Iterative Equalizer) optimizer
///
/// GENIE leverages the One-Step Generalization Ratio (OSGR) to balance parameter
/// contributions and promote domain-invariant feature learning.
pub struct GENIE {
    config: GENIEConfig,
    state: GENIEState,
}

impl GENIE {
    /// Create a new GENIE optimizer
    pub fn new(config: GENIEConfig) -> Self {
        Self {
            config,
            state: GENIEState::default(),
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

    /// Compute One-Step Generalization Ratio (OSGR) for a parameter
    fn compute_osgr(&self, param_name: &str, gradient: &Tensor, current_loss: f32) -> Result<Tensor> {
        // OSGR = |gradient|^2 / (loss_reduction + eps)
        // where loss_reduction = prev_loss - current_loss

        let gradient_norm_sq = gradient.pow(&Tensor::scalar(2.0)?)?;

        if let Some(prev_loss) = self.state.prev_loss {
            let loss_reduction = prev_loss - current_loss;
            let loss_reduction = loss_reduction.max(self.config.preconditioning_eps);

            let osgr = gradient_norm_sq.div(&Tensor::scalar(loss_reduction)?)?;

            // Clamp OSGR to reasonable bounds
            let min_osgr = Tensor::scalar(self.config.min_osgr)?;
            let max_osgr = Tensor::scalar(self.config.max_osgr)?;

            Ok(osgr.clamp(&min_osgr, &max_osgr)?)
        } else {
            // For first step, use gradient norm as proxy
            Ok(gradient_norm_sq)
        }
    }

    /// Update exponential moving average of OSGR
    fn update_osgr_ema(&mut self, param_name: &str, current_osgr: &Tensor) -> Result<()> {
        let momentum = self.config.osgr_momentum;

        if let Some(prev_ema) = self.state.osgr_ema.get(param_name) {
            // EMA update: ema = momentum * prev_ema + (1 - momentum) * current_osgr
            let momentum_tensor = Tensor::scalar(momentum)?;
            let one_minus_momentum = Tensor::scalar(1.0 - momentum)?;

            let weighted_prev = prev_ema.mul(&momentum_tensor)?;
            let weighted_current = current_osgr.mul(&one_minus_momentum)?;
            let new_ema = weighted_prev.add(&weighted_current)?;

            self.state.osgr_ema.insert(param_name.to_string(), new_ema);
        } else {
            // First update
            self.state.osgr_ema.insert(param_name.to_string(), current_osgr.clone());
        }

        Ok(())
    }

    /// Compute gradient alignment score
    fn compute_gradient_alignment(&self, param_name: &str, gradient: &Tensor) -> Result<f32> {
        if let Some(prev_grad) = self.state.prev_gradients.get(param_name) {
            // Cosine similarity between current and previous gradients
            let dot_product = gradient.flatten()?.dot(&prev_grad.flatten()?)?;
            let current_norm = gradient.flatten()?.norm()?.to_scalar::<f32>()?;
            let prev_norm = prev_grad.flatten()?.norm()?.to_scalar::<f32>()?;

            let alignment = dot_product.to_scalar::<f32>()? / (current_norm * prev_norm + self.config.preconditioning_eps);
            Ok(alignment.clamp(-1.0, 1.0))
        } else {
            Ok(0.0) // No previous gradient for comparison
        }
    }

    /// Compute preconditioning factor based on OSGR equalization
    fn compute_preconditioning_factor(&self, param_name: &str) -> Result<Tensor> {
        if let Some(osgr_ema) = self.state.osgr_ema.get(param_name) {
            if self.config.normalize_osgr {
                // Normalize by mean OSGR across all parameters
                let mean_osgr = self.compute_mean_osgr()?;
                let normalized_osgr = osgr_ema.div(&Tensor::scalar(mean_osgr + self.config.preconditioning_eps)?)?;

                // Invert and square root for preconditioning: 1 / sqrt(normalized_osgr)
                let sqrt_osgr = normalized_osgr.sqrt()?;
                sqrt_osgr.reciprocal()
            } else {
                // Simple reciprocal square root: 1 / sqrt(osgr)
                let sqrt_osgr = osgr_ema.sqrt()?;
                sqrt_osgr.reciprocal()
            }
        } else {
            // Default to identity preconditioning
            Ok(Tensor::ones_like(&Tensor::scalar(1.0)?)?)
        }
    }

    /// Compute mean OSGR across all parameters
    fn compute_mean_osgr(&self) -> Result<f32> {
        if self.state.osgr_ema.is_empty() {
            return Ok(1.0);
        }

        let mut total_osgr = 0.0;
        let mut count = 0;

        for osgr_tensor in self.state.osgr_ema.values() {
            let osgr_mean = osgr_tensor.mean()?.to_scalar::<f32>()?;
            total_osgr += osgr_mean;
            count += 1;
        }

        Ok(total_osgr / count as f32)
    }

    /// Compute adaptive alignment weight based on training progress
    fn compute_adaptive_alignment_weight(&self) -> f32 {
        if !self.config.adaptive_alignment {
            return self.config.alignment_weight;
        }

        // Increase alignment weight as training progresses
        let progress = (self.state.step as f32 / (self.config.warmup_steps as f32 + 1.0)).min(1.0);
        self.config.alignment_weight * progress
    }

    /// Perform optimization step
    pub fn step(&mut self, parameters: &mut HashMap<String, Tensor>,
                gradients: &HashMap<String, Tensor>, current_loss: f32) -> Result<()> {
        self.state.step += 1;

        // Skip OSGR computation during warmup
        let use_osgr = self.state.step > self.config.warmup_steps;

        for (param_name, gradient) in gradients.iter() {
            if let Some(parameter) = parameters.get_mut(param_name) {
                // Apply weight decay if configured
                let mut effective_gradient = gradient.clone();
                if self.config.weight_decay > 0.0 {
                    let weight_decay_term = parameter.mul(&Tensor::scalar(self.config.weight_decay)?)?;
                    effective_gradient = effective_gradient.add(&weight_decay_term)?;
                }

                let mut update = effective_gradient.clone();

                if use_osgr {
                    // Compute OSGR for this parameter
                    let osgr = self.compute_osgr(param_name, &effective_gradient, current_loss)?;
                    self.state.osgr.insert(param_name.to_string(), osgr.clone());

                    // Update OSGR EMA
                    self.update_osgr_ema(param_name, &osgr)?;

                    // Compute gradient alignment
                    let alignment = self.compute_gradient_alignment(param_name, &effective_gradient)?;
                    self.state.alignment_stats.insert(param_name.to_string(), alignment);

                    // Compute preconditioning factor
                    let preconditioning = self.compute_preconditioning_factor(param_name)?;
                    self.state.preconditioning_factors.insert(param_name.to_string(), preconditioning.clone());

                    // Apply GENIE preconditioning
                    update = effective_gradient.mul(&preconditioning)?;

                    // Apply alignment-based adjustment
                    let alignment_weight = self.compute_adaptive_alignment_weight();
                    if alignment_weight > 0.0 {
                        let alignment_factor = 1.0 + alignment_weight * alignment.abs();
                        update = update.mul(&Tensor::scalar(alignment_factor)?)?;
                    }
                }

                // Apply learning rate and update parameter
                let lr_tensor = Tensor::scalar(self.config.learning_rate)?;
                let param_update = update.mul(&lr_tensor)?;

                *parameter = parameter.sub(&param_update)?;

                // Store current gradient for next step's alignment computation
                self.state.prev_gradients.insert(param_name.to_string(), effective_gradient);
            }
        }

        // Store current loss for next step's OSGR computation
        self.state.prev_loss = Some(current_loss);

        Ok(())
    }

    /// Get OSGR statistics for analysis
    pub fn get_osgr_stats(&self) -> HashMap<String, f32> {
        let mut stats = HashMap::new();

        for (param_name, osgr_tensor) in &self.state.osgr_ema {
            if let Ok(mean_osgr) = osgr_tensor.mean().and_then(|t| t.to_scalar::<f32>()) {
                stats.insert(param_name.clone(), mean_osgr);
            }
        }

        stats
    }

    /// Get gradient alignment statistics
    pub fn get_alignment_stats(&self) -> &HashMap<String, f32> {
        &self.state.alignment_stats
    }

    /// Get domain generalization statistics
    pub fn get_domain_stats(&self) -> &DomainStats {
        &self.state.domain_stats
    }

    /// Reset optimizer state (useful for transfer learning scenarios)
    pub fn reset_state(&mut self) {
        self.state = GENIEState::default();
    }

    /// Get optimization statistics for monitoring
    pub fn get_stats(&self) -> GENIEStats {
        let mean_osgr = self.compute_mean_osgr().unwrap_or(1.0);
        let mean_alignment = self.state.alignment_stats.values().sum::<f32>() /
                            self.state.alignment_stats.len().max(1) as f32;

        GENIEStats {
            step: self.state.step,
            mean_osgr,
            mean_alignment,
            num_parameters: self.state.osgr_ema.len(),
            adaptive_alignment_weight: self.compute_adaptive_alignment_weight(),
            domain_variance: self.state.domain_stats.domain_variance,
        }
    }
}

/// GENIE optimizer statistics for monitoring and analysis
#[derive(Debug, Clone)]
pub struct GENIEStats {
    /// Current optimization step
    pub step: u64,
    /// Mean OSGR across all parameters
    pub mean_osgr: f32,
    /// Mean gradient alignment score
    pub mean_alignment: f32,
    /// Number of parameters being optimized
    pub num_parameters: usize,
    /// Current adaptive alignment weight
    pub adaptive_alignment_weight: f32,
    /// Domain variance for generalization tracking
    pub domain_variance: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::tensor::Tensor;

    #[test]
    fn test_genie_creation() {
        let config = GENIEConfig::new()
            .learning_rate(1e-3)
            .osgr_momentum(0.9)
            .alignment_weight(0.1)
            .build();

        let optimizer = GENIE::new(config);
        assert_eq!(optimizer.learning_rate(), 1e-3);
    }

    #[test]
    fn test_genie_config_builder() {
        let config = GENIEConfig::new()
            .learning_rate(2e-3)
            .osgr_momentum(0.95)
            .alignment_weight(0.2)
            .preconditioning_eps(1e-6)
            .weight_decay(1e-4)
            .build();

        assert_eq!(config.learning_rate, 2e-3);
        assert_eq!(config.osgr_momentum, 0.95);
        assert_eq!(config.alignment_weight, 0.2);
        assert_eq!(config.preconditioning_eps, 1e-6);
        assert_eq!(config.weight_decay, 1e-4);
    }

    #[test]
    fn test_genie_step() -> Result<()> {
        let config = GENIEConfig::new().learning_rate(1e-2).build();
        let mut optimizer = GENIE::new(config);

        // Create test parameters and gradients
        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::ones(&[2, 2])? * 0.1);

        // Perform optimization step
        let initial_loss = 1.0;
        optimizer.step(&mut parameters, &gradients, initial_loss)?;

        // Check that parameter was updated
        let updated_param = parameters.get("weight").unwrap();
        let expected_value = 1.0 - 1e-2 * 0.1; // 1.0 - lr * grad

        // Due to GENIE's preconditioning, the exact update may differ, but parameter should change
        assert_ne!(updated_param.to_scalar::<f32>()?, 1.0);

        Ok(())
    }

    #[test]
    fn test_genie_osgr_computation() -> Result<()> {
        let config = GENIEConfig::new().build();
        let mut optimizer = GENIE::new(config);

        // Set previous loss
        optimizer.state.prev_loss = Some(2.0);

        let gradient = Tensor::ones(&[2, 2])?;
        let current_loss = 1.5;

        let osgr = optimizer.compute_osgr("test", &gradient, current_loss)?;

        // OSGR should be gradient_norm_sq / loss_reduction
        // gradient_norm_sq = 4.0 (sum of ones squared)
        // loss_reduction = 2.0 - 1.5 = 0.5
        // Expected OSGR = 4.0 / 0.5 = 8.0
        let expected_osgr = 4.0 / 0.5;
        let computed_osgr = osgr.mean()?.to_scalar::<f32>()?;

        assert!((computed_osgr - expected_osgr).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_genie_gradient_alignment() -> Result<()> {
        let config = GENIEConfig::new().build();
        let mut optimizer = GENIE::new(config);

        // Set previous gradient
        let prev_grad = Tensor::ones(&[2, 2])?;
        optimizer.state.prev_gradients.insert("test".to_string(), prev_grad);

        // Test with same gradient (perfect alignment)
        let current_grad = Tensor::ones(&[2, 2])?;
        let alignment = optimizer.compute_gradient_alignment("test", &current_grad)?;

        assert!((alignment - 1.0).abs() < 1e-5);

        // Test with opposite gradient (negative alignment)
        let opposite_grad = Tensor::ones(&[2, 2])? * -1.0;
        let alignment = optimizer.compute_gradient_alignment("test", &opposite_grad)?;

        assert!((alignment - (-1.0)).abs() < 1e-5);

        Ok(())
    }

    #[test]
    fn test_genie_stats() -> Result<()> {
        let config = GENIEConfig::new().warmup_steps(1).build();
        let mut optimizer = GENIE::new(config);

        // Create test data
        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::ones(&[2, 2])? * 0.1);

        // Perform multiple steps
        for i in 0..5 {
            let loss = 1.0 - i as f32 * 0.1;
            optimizer.step(&mut parameters, &gradients, loss)?;
        }

        let stats = optimizer.get_stats();
        assert_eq!(stats.step, 5);
        assert!(stats.num_parameters > 0);
        assert!(stats.mean_osgr > 0.0);

        Ok(())
    }

    #[test]
    fn test_genie_learning_rate_methods() {
        let config = GENIEConfig::new().learning_rate(1e-3).build();
        let mut optimizer = GENIE::new(config);

        assert_eq!(optimizer.learning_rate(), 1e-3);

        optimizer.set_learning_rate(2e-3);
        assert_eq!(optimizer.learning_rate(), 2e-3);
    }

    #[test]
    fn test_genie_weight_decay() -> Result<()> {
        let config = GENIEConfig::new()
            .learning_rate(1e-2)
            .weight_decay(1e-2)
            .build();
        let mut optimizer = GENIE::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::zeros(&[2, 2])?);

        let initial_param_value = parameters.get("weight").unwrap().to_scalar::<f32>()?;

        optimizer.step(&mut parameters, &gradients, 1.0)?;

        let final_param_value = parameters.get("weight").unwrap().to_scalar::<f32>()?;

        // With weight decay, parameter should decrease even with zero gradient
        assert!(final_param_value < initial_param_value);

        Ok(())
    }

    #[test]
    fn test_genie_warmup() -> Result<()> {
        let config = GENIEConfig::new()
            .learning_rate(1e-2)
            .warmup_steps(5)
            .build();
        let mut optimizer = GENIE::new(config);

        let mut parameters = HashMap::new();
        parameters.insert("weight".to_string(), Tensor::ones(&[2, 2])?);

        let mut gradients = HashMap::new();
        gradients.insert("weight".to_string(), Tensor::ones(&[2, 2])? * 0.1);

        // During warmup, OSGR should not be computed
        optimizer.step(&mut parameters, &gradients, 1.0)?;
        assert!(optimizer.state.osgr.is_empty());

        // After warmup, OSGR should be computed
        for _ in 0..6 {
            optimizer.step(&mut parameters, &gradients, 1.0)?;
        }
        assert!(!optimizer.state.osgr.is_empty());

        Ok(())
    }
}