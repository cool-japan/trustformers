//! # Gradient Processing Enhancements
//!
//! This module provides advanced gradient processing techniques that can improve
//! training stability, convergence speed, and final model performance.
//!
//! ## Available Techniques
//!
//! - **Gradient Centralization**: Removes the mean of gradients to improve convergence
//! - **Gradient Standardization**: Normalizes gradients to unit variance
//! - **Adaptive Gradient Clipping**: Dynamically adjusts clipping based on gradient history
//! - **Gradient Noise Injection**: Adds controlled noise to escape local minima
//! - **Gradient Smoothing**: Applies exponential moving average to gradients
//! - **Hessian-based Preconditioning**: Uses second-order information to precondition gradients

use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::tensor::Tensor;

/// Configuration for gradient processing techniques.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GradientProcessingConfig {
    /// Enable gradient centralization
    pub enable_centralization: bool,
    /// Enable gradient standardization
    pub enable_standardization: bool,
    /// Enable adaptive gradient clipping
    pub enable_adaptive_clipping: bool,
    /// Enable gradient noise injection
    pub enable_noise_injection: bool,
    /// Enable gradient smoothing
    pub enable_smoothing: bool,
    /// Enable Hessian-based preconditioning
    pub enable_hessian_preconditioning: bool,
    /// Adaptive clipping parameters
    pub adaptive_clipping: AdaptiveClippingConfig,
    /// Noise injection parameters
    pub noise_injection: NoiseInjectionConfig,
    /// Smoothing parameters
    pub smoothing: SmoothingConfig,
    /// Hessian preconditioning parameters
    pub hessian_preconditioning: HessianPreconditioningConfig,
}

/// Configuration for adaptive gradient clipping.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveClippingConfig {
    /// Initial clipping threshold
    pub initial_clip_norm: f32,
    /// Minimum clipping threshold
    pub min_clip_norm: f32,
    /// Maximum clipping threshold
    pub max_clip_norm: f32,
    /// Adaptation rate
    pub adaptation_rate: f32,
    /// Target gradient norm percentile
    pub target_percentile: f32,
    /// History window size for computing statistics
    pub history_window: usize,
}

impl Default for AdaptiveClippingConfig {
    fn default() -> Self {
        Self {
            initial_clip_norm: 1.0,
            min_clip_norm: 0.1,
            max_clip_norm: 10.0,
            adaptation_rate: 0.01,
            target_percentile: 0.9,
            history_window: 100,
        }
    }
}

/// Configuration for gradient noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseInjectionConfig {
    /// Initial noise scale
    pub initial_noise_scale: f32,
    /// Noise decay rate per step
    pub decay_rate: f32,
    /// Minimum noise scale
    pub min_noise_scale: f32,
    /// Noise type
    pub noise_type: NoiseType,
}

impl Default for NoiseInjectionConfig {
    fn default() -> Self {
        Self {
            initial_noise_scale: 0.1,
            decay_rate: 0.999,
            min_noise_scale: 1e-6,
            noise_type: NoiseType::Gaussian,
        }
    }
}

/// Configuration for gradient smoothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SmoothingConfig {
    /// Exponential moving average decay rate
    pub decay: f32,
    /// Whether to debias the moving average
    pub debias: bool,
}

impl Default for SmoothingConfig {
    fn default() -> Self {
        Self {
            decay: 0.9,
            debias: true,
        }
    }
}

/// Configuration for Hessian-based preconditioning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HessianPreconditioningConfig {
    /// Type of Hessian approximation to use
    pub approximation_type: HessianApproximationType,
    /// Damping factor for numerical stability
    pub damping: f32,
    /// Update frequency for Hessian approximation (every N steps)
    pub update_frequency: usize,
    /// History window for maintaining Hessian approximation
    pub history_window: usize,
    /// Minimum eigenvalue threshold for conditioning
    pub min_eigenvalue: f32,
    /// Maximum condition number allowed
    pub max_condition_number: f32,
}

impl Default for HessianPreconditioningConfig {
    fn default() -> Self {
        Self {
            approximation_type: HessianApproximationType::Diagonal,
            damping: 1e-4,
            update_frequency: 10,
            history_window: 20,
            min_eigenvalue: 1e-8,
            max_condition_number: 1e6,
        }
    }
}

/// Types of noise for gradient noise injection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NoiseType {
    Gaussian,
    Uniform,
    Laplace,
}

/// Types of Hessian approximation methods.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HessianApproximationType {
    /// Use only the diagonal of the Hessian (most efficient)
    Diagonal,
    /// Use Gauss-Newton approximation (J^T J)
    GaussNewton,
    /// Use Fisher Information Matrix approximation
    FisherInformation,
    /// Use quasi-Newton L-BFGS-style approximation
    QuasiNewton,
}

/// Gradient processor that applies various enhancement techniques.
#[derive(Debug)]
pub struct GradientProcessor {
    config: GradientProcessingConfig,
    current_step: usize,

    // Adaptive clipping state
    gradient_norm_history: Vec<f32>,
    current_clip_norm: f32,

    // Noise injection state
    current_noise_scale: f32,

    // Smoothing state
    smoothed_gradients: HashMap<usize, Tensor>,
    smoothing_bias_correction: f32,

    // Hessian preconditioning state
    hessian_diagonal: HashMap<usize, Tensor>,
    hessian_inverse: HashMap<usize, Tensor>,
    last_hessian_update: usize,
    gradient_history: Vec<Vec<Tensor>>,
}

impl GradientProcessor {
    /// Create a new gradient processor with the given configuration.
    pub fn new(config: GradientProcessingConfig) -> Self {
        Self {
            current_clip_norm: config.adaptive_clipping.initial_clip_norm,
            current_noise_scale: config.noise_injection.initial_noise_scale,
            config,
            current_step: 0,
            gradient_norm_history: Vec::new(),
            smoothed_gradients: HashMap::new(),
            smoothing_bias_correction: 1.0,
            hessian_diagonal: HashMap::new(),
            hessian_inverse: HashMap::new(),
            last_hessian_update: 0,
            gradient_history: Vec::new(),
        }
    }

    /// Create a gradient processor with default configuration.
    pub fn default() -> Self {
        Self::new(GradientProcessingConfig::default())
    }

    /// Process gradients with enabled techniques.
    pub fn process_gradients(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        self.current_step += 1;

        // Apply gradient centralization
        if self.config.enable_centralization {
            self.apply_centralization(gradients)?;
        }

        // Apply gradient standardization
        if self.config.enable_standardization {
            self.apply_standardization(gradients)?;
        }

        // Apply gradient smoothing
        if self.config.enable_smoothing {
            self.apply_smoothing(gradients)?;
        }

        // Apply Hessian-based preconditioning
        if self.config.enable_hessian_preconditioning {
            self.apply_hessian_preconditioning(gradients)?;
        }

        // Apply adaptive gradient clipping
        if self.config.enable_adaptive_clipping {
            self.apply_adaptive_clipping(gradients)?;
        }

        // Apply gradient noise injection
        if self.config.enable_noise_injection {
            self.apply_noise_injection(gradients)?;
        }

        Ok(())
    }

    /// Apply gradient centralization (remove mean).
    fn apply_centralization(&self, gradients: &mut [Tensor]) -> Result<()> {
        for gradient in gradients.iter_mut() {
            // Compute mean across all dimensions
            let mean = gradient.mean()?;
            *gradient = gradient.sub(&mean)?;
        }
        Ok(())
    }

    /// Apply gradient standardization (normalize to unit variance).
    fn apply_standardization(&self, gradients: &mut [Tensor]) -> Result<()> {
        for gradient in gradients.iter_mut() {
            // Compute standard deviation manually
            let mean = gradient.mean()?;
            let centered = gradient.sub(&mean)?;
            let squared = centered.mul(&centered)?;
            let variance = squared.mean()?;
            let std_dev = variance.sqrt()?;

            // Add small epsilon to prevent division by zero
            let epsilon = Tensor::scalar(1e-8)?;
            let std_dev_safe = std_dev.add(&epsilon)?;

            // Normalize
            *gradient = gradient.div(&std_dev_safe)?;
        }
        Ok(())
    }

    /// Apply adaptive gradient clipping.
    fn apply_adaptive_clipping(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // Compute total gradient norm
        let mut total_norm_sq = 0.0;
        for gradient in gradients.iter() {
            let norm_sq = gradient.norm_squared()?.to_scalar()?;
            total_norm_sq += norm_sq;
        }
        let total_norm = total_norm_sq.sqrt();

        // Update gradient norm history
        self.gradient_norm_history.push(total_norm);
        if self.gradient_norm_history.len() > self.config.adaptive_clipping.history_window {
            self.gradient_norm_history.remove(0);
        }

        // Update adaptive clipping threshold
        if self.gradient_norm_history.len() >= 10 {
            // Compute target percentile of gradient norms
            let mut sorted_norms = self.gradient_norm_history.clone();
            sorted_norms.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let percentile_idx = (sorted_norms.len() as f32
                * self.config.adaptive_clipping.target_percentile)
                as usize;
            let target_norm = sorted_norms[percentile_idx.min(sorted_norms.len() - 1)];

            // Adapt clipping threshold towards target
            let adaptation = self.config.adaptive_clipping.adaptation_rate
                * (target_norm - self.current_clip_norm);
            self.current_clip_norm += adaptation;

            // Clamp to bounds
            self.current_clip_norm = self
                .current_clip_norm
                .max(self.config.adaptive_clipping.min_clip_norm)
                .min(self.config.adaptive_clipping.max_clip_norm);
        }

        // Apply clipping if needed
        if total_norm > self.current_clip_norm {
            let clip_factor = self.current_clip_norm / total_norm;
            for gradient in gradients.iter_mut() {
                *gradient = gradient.mul_scalar(clip_factor)?;
            }
        }

        Ok(())
    }

    /// Apply gradient noise injection.
    fn apply_noise_injection(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // Decay noise scale
        self.current_noise_scale *= self.config.noise_injection.decay_rate;
        self.current_noise_scale =
            self.current_noise_scale.max(self.config.noise_injection.min_noise_scale);

        for gradient in gradients.iter_mut() {
            let noise = match self.config.noise_injection.noise_type {
                NoiseType::Gaussian => {
                    let noise_tensor = Tensor::randn(&gradient.shape())?;
                    noise_tensor.mul_scalar(self.current_noise_scale)?;
                    noise_tensor
                },
                NoiseType::Uniform => {
                    let bound = self.current_noise_scale * 3.0_f32.sqrt(); // Match variance with Gaussian
                    let noise_tensor = Tensor::randn(&gradient.shape())?;
                    noise_tensor.mul_scalar(bound)?;
                    noise_tensor
                },
                NoiseType::Laplace => {
                    // Approximate Laplace with scaled Gaussian (simplified)
                    let noise_tensor = Tensor::randn(&gradient.shape())?;
                    noise_tensor.mul_scalar(self.current_noise_scale * 2.0_f32.sqrt())?;
                    noise_tensor
                },
            };

            *gradient = gradient.add(&noise)?;
        }

        Ok(())
    }

    /// Apply gradient smoothing with exponential moving average.
    fn apply_smoothing(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        let decay = self.config.smoothing.decay;

        for (i, gradient) in gradients.iter_mut().enumerate() {
            if let Some(smoothed) = self.smoothed_gradients.get(&i) {
                // Update smoothed gradient: smoothed = decay * smoothed + (1 - decay) * gradient
                let new_smoothed =
                    smoothed.mul_scalar(decay)?.add(&gradient.mul_scalar(1.0 - decay)?)?;
                self.smoothed_gradients.insert(i, new_smoothed.clone());

                // Apply bias correction if enabled
                if self.config.smoothing.debias {
                    self.smoothing_bias_correction *= decay;
                    let bias_corrected =
                        new_smoothed.div_scalar(1.0 - self.smoothing_bias_correction)?;
                    *gradient = bias_corrected;
                } else {
                    *gradient = new_smoothed;
                }
            } else {
                // First time seeing this gradient
                self.smoothed_gradients.insert(i, gradient.clone());
            }
        }

        Ok(())
    }

    /// Apply Hessian-based preconditioning to gradients.
    fn apply_hessian_preconditioning(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // Store gradient history for Hessian approximation
        self.gradient_history.push(gradients.to_vec());
        if self.gradient_history.len() > self.config.hessian_preconditioning.history_window {
            self.gradient_history.remove(0);
        }

        // Update Hessian approximation if needed
        if self.current_step - self.last_hessian_update
            >= self.config.hessian_preconditioning.update_frequency
        {
            self.update_hessian_approximation(gradients)?;
            self.last_hessian_update = self.current_step;
        }

        // Apply preconditioning based on approximation type
        match self.config.hessian_preconditioning.approximation_type {
            HessianApproximationType::Diagonal => {
                self.apply_diagonal_preconditioning(gradients)?;
            },
            HessianApproximationType::GaussNewton => {
                self.apply_gauss_newton_preconditioning(gradients)?;
            },
            HessianApproximationType::FisherInformation => {
                self.apply_fisher_information_preconditioning(gradients)?;
            },
            HessianApproximationType::QuasiNewton => {
                self.apply_quasi_newton_preconditioning(gradients)?;
            },
        }

        Ok(())
    }

    /// Update Hessian approximation based on gradient history.
    fn update_hessian_approximation(&mut self, gradients: &[Tensor]) -> Result<()> {
        match self.config.hessian_preconditioning.approximation_type {
            HessianApproximationType::Diagonal => {
                self.update_diagonal_hessian(gradients)?;
            },
            HessianApproximationType::GaussNewton => {
                self.update_gauss_newton_hessian(gradients)?;
            },
            HessianApproximationType::FisherInformation => {
                self.update_fisher_information_hessian(gradients)?;
            },
            HessianApproximationType::QuasiNewton => {
                self.update_quasi_newton_hessian(gradients)?;
            },
        }
        Ok(())
    }

    /// Update diagonal Hessian approximation using gradient variance.
    fn update_diagonal_hessian(&mut self, gradients: &[Tensor]) -> Result<()> {
        for (i, gradient) in gradients.iter().enumerate() {
            // Approximate diagonal Hessian using gradient variance over history
            if self.gradient_history.len() > 1 {
                let mut variance = Tensor::zeros(&gradient.shape())?;
                let mut mean = Tensor::zeros(&gradient.shape())?;

                // Compute mean
                for grad_vec in &self.gradient_history {
                    if let Some(hist_grad) = grad_vec.get(i) {
                        mean = mean.add(hist_grad)?;
                    }
                }
                mean = mean.div_scalar(self.gradient_history.len() as f32)?;

                // Compute variance (approximation of diagonal Hessian)
                for grad_vec in &self.gradient_history {
                    if let Some(hist_grad) = grad_vec.get(i) {
                        let diff = hist_grad.sub(&mean)?;
                        variance = variance.add(&diff.mul(&diff)?)?;
                    }
                }
                variance = variance.div_scalar(self.gradient_history.len() as f32)?;

                // Add damping for numerical stability
                let damping = Tensor::ones(&gradient.shape())?
                    .mul_scalar(self.config.hessian_preconditioning.damping)?;
                variance = variance.add(&damping)?;

                self.hessian_diagonal.insert(i, variance);
            }
        }
        Ok(())
    }

    /// Update Gauss-Newton Hessian approximation (simplified).
    fn update_gauss_newton_hessian(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Simplified Gauss-Newton approximation using gradient outer product
        for (i, gradient) in gradients.iter().enumerate() {
            // Approximate with gradient outer product (simplified)
            let outer_product = gradient.mul(gradient)?;

            // Add damping
            let damping = Tensor::ones(&gradient.shape())?
                .mul_scalar(self.config.hessian_preconditioning.damping)?;
            let hessian_approx = outer_product.add(&damping)?;

            self.hessian_diagonal.insert(i, hessian_approx);
        }
        Ok(())
    }

    /// Update Fisher Information Matrix approximation.
    fn update_fisher_information_hessian(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Fisher Information Matrix approximation (similar to Gauss-Newton for this context)
        for (i, gradient) in gradients.iter().enumerate() {
            // Approximate Fisher Information using gradient squared
            let fisher_approx = gradient.mul(gradient)?;

            // Add damping
            let damping = Tensor::ones(&gradient.shape())?
                .mul_scalar(self.config.hessian_preconditioning.damping)?;
            let hessian_approx = fisher_approx.add(&damping)?;

            self.hessian_diagonal.insert(i, hessian_approx);
        }
        Ok(())
    }

    /// Update quasi-Newton Hessian approximation using L-BFGS-style update.
    fn update_quasi_newton_hessian(&mut self, gradients: &[Tensor]) -> Result<()> {
        // Simplified quasi-Newton approximation using gradient differences
        if self.gradient_history.len() > 1 {
            for (i, gradient) in gradients.iter().enumerate() {
                // Get previous gradient
                if let Some(prev_grad_vec) =
                    self.gradient_history.get(self.gradient_history.len() - 2)
                {
                    if let Some(prev_grad) = prev_grad_vec.get(i) {
                        // Compute gradient difference
                        let grad_diff = gradient.sub(prev_grad)?;

                        // Approximate Hessian using gradient difference magnitude
                        let hessian_approx = grad_diff.abs()?;

                        // Add damping
                        let damping = Tensor::ones(&gradient.shape())?
                            .mul_scalar(self.config.hessian_preconditioning.damping)?;
                        let final_hessian = hessian_approx.add(&damping)?;

                        self.hessian_diagonal.insert(i, final_hessian);
                    }
                }
            }
        }
        Ok(())
    }

    /// Apply diagonal preconditioning to gradients.
    fn apply_diagonal_preconditioning(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        for (i, gradient) in gradients.iter_mut().enumerate() {
            if let Some(hessian_diag) = self.hessian_diagonal.get(&i) {
                // Compute preconditioned gradient: H^{-1} * g
                // For diagonal H, this is element-wise division
                let min_val = Tensor::scalar(self.config.hessian_preconditioning.min_eigenvalue)?;
                let clamped_hessian = hessian_diag.max(&min_val)?;

                *gradient = gradient.div(&clamped_hessian)?;
            }
        }
        Ok(())
    }

    /// Apply Gauss-Newton preconditioning to gradients.
    fn apply_gauss_newton_preconditioning(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // For simplicity, use diagonal approximation
        self.apply_diagonal_preconditioning(gradients)
    }

    /// Apply Fisher Information preconditioning to gradients.
    fn apply_fisher_information_preconditioning(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // For simplicity, use diagonal approximation
        self.apply_diagonal_preconditioning(gradients)
    }

    /// Apply quasi-Newton preconditioning to gradients.
    fn apply_quasi_newton_preconditioning(&mut self, gradients: &mut [Tensor]) -> Result<()> {
        // For simplicity, use diagonal approximation
        self.apply_diagonal_preconditioning(gradients)
    }

    /// Get current adaptive clipping threshold.
    pub fn get_current_clip_norm(&self) -> f32 {
        self.current_clip_norm
    }

    /// Get current noise scale.
    pub fn get_current_noise_scale(&self) -> f32 {
        self.current_noise_scale
    }

    /// Get gradient norm statistics.
    pub fn get_gradient_norm_stats(&self) -> Option<(f32, f32, f32)> {
        if self.gradient_norm_history.is_empty() {
            return None;
        }

        let sum: f32 = self.gradient_norm_history.iter().sum();
        let mean = sum / self.gradient_norm_history.len() as f32;

        let variance = self.gradient_norm_history.iter().map(|x| (x - mean).powi(2)).sum::<f32>()
            / self.gradient_norm_history.len() as f32;
        let std_dev = variance.sqrt();

        let max_norm = self.gradient_norm_history.iter().fold(0.0f32, |acc, &x| acc.max(x));

        Some((mean, std_dev, max_norm))
    }

    /// Reset internal state.
    pub fn reset(&mut self) {
        self.current_step = 0;
        self.gradient_norm_history.clear();
        self.smoothed_gradients.clear();
        self.current_clip_norm = self.config.adaptive_clipping.initial_clip_norm;
        self.current_noise_scale = self.config.noise_injection.initial_noise_scale;
        self.smoothing_bias_correction = 1.0;
        self.hessian_diagonal.clear();
        self.hessian_inverse.clear();
        self.last_hessian_update = 0;
        self.gradient_history.clear();
    }

    /// Update configuration.
    pub fn set_config(&mut self, config: GradientProcessingConfig) {
        self.config = config;
        self.reset();
    }

    /// Get current configuration.
    pub fn get_config(&self) -> &GradientProcessingConfig {
        &self.config
    }
}

/// Wrapper for optimizers that automatically applies gradient processing.
pub struct GradientProcessedOptimizer<T> {
    base_optimizer: T,
    gradient_processor: GradientProcessor,
}

impl<T> GradientProcessedOptimizer<T> {
    /// Create a new gradient-processed optimizer.
    pub fn new(base_optimizer: T, config: GradientProcessingConfig) -> Self {
        Self {
            base_optimizer,
            gradient_processor: GradientProcessor::new(config),
        }
    }

    /// Create with default gradient processing configuration.
    pub fn with_default_processing(base_optimizer: T) -> Self {
        Self::new(base_optimizer, GradientProcessingConfig::default())
    }

    /// Get reference to the gradient processor.
    pub fn gradient_processor(&self) -> &GradientProcessor {
        &self.gradient_processor
    }

    /// Get mutable reference to the gradient processor.
    pub fn gradient_processor_mut(&mut self) -> &mut GradientProcessor {
        &mut self.gradient_processor
    }

    /// Get reference to the base optimizer.
    pub fn base_optimizer(&self) -> &T {
        &self.base_optimizer
    }

    /// Get mutable reference to the base optimizer.
    pub fn base_optimizer_mut(&mut self) -> &mut T {
        &mut self.base_optimizer
    }
}

impl<T: crate::optimizer::OptimizerState> crate::optimizer::OptimizerState
    for GradientProcessedOptimizer<T>
{
    fn zero_grad(&mut self) -> Result<()> {
        self.base_optimizer.zero_grad()
    }

    fn step(&mut self, parameters: &mut [Tensor]) -> Result<()> {
        // Extract gradients from parameters
        let mut gradients = Vec::new();
        for param in parameters.iter() {
            if let Ok(grad) = param.grad() {
                gradients.push(grad);
            } else {
                return Err(anyhow!("Parameter missing gradient"));
            }
        }

        // Process gradients
        self.gradient_processor.process_gradients(&mut gradients)?;

        // Update parameter gradients with processed versions
        for (param, processed_grad) in parameters.iter_mut().zip(gradients.iter()) {
            param.set_grad(processed_grad.clone())?;
        }

        // Perform optimization step
        self.base_optimizer.step(parameters)
    }

    fn get_lr(&self) -> f32 {
        self.base_optimizer.get_lr()
    }

    fn set_lr(&mut self, lr: f32) {
        self.base_optimizer.set_lr(lr);
    }

    fn state_dict(&self) -> Result<HashMap<String, Tensor>> {
        // For simplicity, we'll only save the base optimizer state
        // In a full implementation, we'd also save gradient processor state
        self.base_optimizer.state_dict()
    }

    fn load_state_dict(&mut self, state: HashMap<String, Tensor>) -> Result<()> {
        self.base_optimizer.load_state_dict(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gradient_processing_config_default() {
        let config = GradientProcessingConfig::default();
        assert!(!config.enable_centralization);
        assert!(!config.enable_standardization);
        assert!(!config.enable_adaptive_clipping);
        assert!(!config.enable_noise_injection);
        assert!(!config.enable_smoothing);
    }

    #[test]
    fn test_adaptive_clipping_config_default() {
        let config = AdaptiveClippingConfig::default();
        assert_eq!(config.initial_clip_norm, 1.0);
        assert_eq!(config.min_clip_norm, 0.1);
        assert_eq!(config.max_clip_norm, 10.0);
        assert_eq!(config.adaptation_rate, 0.01);
        assert_eq!(config.target_percentile, 0.9);
        assert_eq!(config.history_window, 100);
    }

    #[test]
    fn test_gradient_processor_creation() {
        let processor = GradientProcessor::default();
        assert_eq!(processor.current_step, 0);
        assert_eq!(processor.gradient_norm_history.len(), 0);
    }

    #[test]
    fn test_gradient_norm_stats_empty() {
        let processor = GradientProcessor::default();
        assert!(processor.get_gradient_norm_stats().is_none());
    }

    #[test]
    fn test_gradient_processor_reset() {
        let mut processor = GradientProcessor::default();
        processor.current_step = 10;
        processor.gradient_norm_history.push(1.0);

        processor.reset();

        assert_eq!(processor.current_step, 0);
        assert_eq!(processor.gradient_norm_history.len(), 0);
        assert_eq!(processor.hessian_diagonal.len(), 0);
        assert_eq!(processor.gradient_history.len(), 0);
    }

    #[test]
    fn test_hessian_preconditioning_config_default() {
        let config = HessianPreconditioningConfig::default();
        assert!(matches!(
            config.approximation_type,
            HessianApproximationType::Diagonal
        ));
        assert_eq!(config.damping, 1e-4);
        assert_eq!(config.update_frequency, 10);
        assert_eq!(config.history_window, 20);
        assert_eq!(config.min_eigenvalue, 1e-8);
        assert_eq!(config.max_condition_number, 1e6);
    }

    #[test]
    fn test_hessian_preconditioning_enabled() {
        let mut config = GradientProcessingConfig::default();
        config.enable_hessian_preconditioning = true;

        let processor = GradientProcessor::new(config);
        assert!(processor.config.enable_hessian_preconditioning);
    }

    #[test]
    fn test_hessian_approximation_types() {
        let mut config = GradientProcessingConfig::default();
        config.enable_hessian_preconditioning = true;

        // Test different approximation types
        config.hessian_preconditioning.approximation_type = HessianApproximationType::Diagonal;
        let processor = GradientProcessor::new(config.clone());
        assert!(matches!(
            processor.config.hessian_preconditioning.approximation_type,
            HessianApproximationType::Diagonal
        ));

        config.hessian_preconditioning.approximation_type = HessianApproximationType::GaussNewton;
        let processor = GradientProcessor::new(config.clone());
        assert!(matches!(
            processor.config.hessian_preconditioning.approximation_type,
            HessianApproximationType::GaussNewton
        ));

        config.hessian_preconditioning.approximation_type =
            HessianApproximationType::FisherInformation;
        let processor = GradientProcessor::new(config.clone());
        assert!(matches!(
            processor.config.hessian_preconditioning.approximation_type,
            HessianApproximationType::FisherInformation
        ));

        config.hessian_preconditioning.approximation_type = HessianApproximationType::QuasiNewton;
        let processor = GradientProcessor::new(config.clone());
        assert!(matches!(
            processor.config.hessian_preconditioning.approximation_type,
            HessianApproximationType::QuasiNewton
        ));
    }
}
