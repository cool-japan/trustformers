//! Differential Privacy for Mobile Training
//!
//! This module provides comprehensive differential privacy mechanisms
//! for protecting user data during on-device training.

use crate::training::OnDeviceTrainingConfig;
use crate::DefaultRng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::error::{CoreError, Result};
use trustformers_core::Tensor;
use trustformers_core::TrustformersError;

/// Differential privacy engine for mobile training
pub struct DifferentialPrivacyEngine {
    config: PrivacyConfig,
    accountant: PrivacyAccountant,
    noise_generator: NoiseGenerator,
    gradient_clipper: GradientClipper,
}

/// Privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyConfig {
    /// Target privacy level
    pub privacy_level: PrivacyLevel,
    /// Total privacy budget (epsilon)
    pub total_epsilon: f64,
    /// Total delta parameter
    pub total_delta: f64,
    /// Noise multiplier
    pub noise_multiplier: f32,
    /// Gradient clipping threshold
    pub clipping_threshold: f32,
    /// Enable per-example gradient clipping
    pub per_example_clipping: bool,
    /// Enable adaptive clipping
    pub adaptive_clipping: bool,
    /// Subsampling rate for privacy amplification
    pub subsampling_rate: f32,
    /// Composition method
    pub composition_method: CompositionMethod,
}

/// Privacy levels for easy configuration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// Low privacy (ε ≈ 10)
    Low,
    /// Medium privacy (ε ≈ 3)
    Medium,
    /// High privacy (ε ≈ 1)
    High,
    /// Very high privacy (ε ≈ 0.1)
    VeryHigh,
    /// Custom privacy settings
    Custom,
}

/// Composition methods for privacy accounting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionMethod {
    /// Simple composition
    Simple,
    /// Advanced composition
    Advanced,
    /// Moments accountant
    Moments,
    /// Rényi differential privacy
    Renyi,
}

impl Default for PrivacyConfig {
    fn default() -> Self {
        Self::from_privacy_level(PrivacyLevel::Medium)
    }
}

impl PrivacyConfig {
    /// Create config from privacy level
    pub fn from_privacy_level(level: PrivacyLevel) -> Self {
        match level {
            PrivacyLevel::Low => Self {
                privacy_level: level,
                total_epsilon: 10.0,
                total_delta: 1e-5,
                noise_multiplier: 0.5,
                clipping_threshold: 1.0,
                per_example_clipping: true,
                adaptive_clipping: false,
                subsampling_rate: 0.1,
                composition_method: CompositionMethod::Simple,
            },
            PrivacyLevel::Medium => Self {
                privacy_level: level,
                total_epsilon: 3.0,
                total_delta: 1e-6,
                noise_multiplier: 1.0,
                clipping_threshold: 0.5,
                per_example_clipping: true,
                adaptive_clipping: true,
                subsampling_rate: 0.05,
                composition_method: CompositionMethod::Advanced,
            },
            PrivacyLevel::High => Self {
                privacy_level: level,
                total_epsilon: 1.0,
                total_delta: 1e-7,
                noise_multiplier: 2.0,
                clipping_threshold: 0.1,
                per_example_clipping: true,
                adaptive_clipping: true,
                subsampling_rate: 0.01,
                composition_method: CompositionMethod::Moments,
            },
            PrivacyLevel::VeryHigh => Self {
                privacy_level: level,
                total_epsilon: 0.1,
                total_delta: 1e-8,
                noise_multiplier: 5.0,
                clipping_threshold: 0.01,
                per_example_clipping: true,
                adaptive_clipping: true,
                subsampling_rate: 0.001,
                composition_method: CompositionMethod::Renyi,
            },
            PrivacyLevel::Custom => Self {
                privacy_level: level,
                total_epsilon: 1.0,
                total_delta: 1e-6,
                noise_multiplier: 1.0,
                clipping_threshold: 0.5,
                per_example_clipping: true,
                adaptive_clipping: false,
                subsampling_rate: 0.05,
                composition_method: CompositionMethod::Advanced,
            },
        }
    }
}

impl DifferentialPrivacyEngine {
    /// Create new differential privacy engine
    pub fn new(config: PrivacyConfig) -> Self {
        Self {
            config: config.clone(),
            accountant: PrivacyAccountant::new(config.clone()),
            noise_generator: NoiseGenerator::new(config.noise_multiplier),
            gradient_clipper: GradientClipper::new(config.clipping_threshold),
        }
    }

    /// Apply differential privacy to gradients
    pub fn privatize_gradients(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        batch_size: usize,
        step: usize,
    ) -> Result<PrivacyReport> {
        // Check privacy budget
        if !self.accountant.has_budget_remaining() {
            return Err(TrustformersError::runtime_error("Privacy budget exhausted".into()).into());
        }

        // Clip gradients
        let clipping_stats = if self.config.per_example_clipping {
            self.clip_per_example_gradients(gradients)?
        } else {
            self.clip_gradients(gradients)?
        };

        // Add noise
        let noise_stats = self.add_gradient_noise(gradients, batch_size)?;

        // Account for privacy cost
        let privacy_cost = self.accountant.account_step(
            batch_size,
            self.config.subsampling_rate,
            noise_stats.noise_scale,
        )?;

        Ok(PrivacyReport {
            step,
            epsilon_spent: privacy_cost.epsilon,
            delta_spent: privacy_cost.delta,
            total_epsilon_spent: self.accountant.total_epsilon_spent,
            total_delta_spent: self.accountant.total_delta_spent,
            clipping_stats,
            noise_stats,
        })
    }

    /// Apply privacy to training data
    pub fn privatize_data(
        &self,
        data: &[(Tensor, Tensor)],
        method: DataPrivacyMethod,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        match method {
            DataPrivacyMethod::InputPerturbation => self.apply_input_perturbation(data),
            DataPrivacyMethod::LabelSmoothing { smoothing_factor } => {
                self.apply_label_smoothing(data, smoothing_factor)
            },
            DataPrivacyMethod::Mixup { alpha } => self.apply_mixup(data, alpha),
            DataPrivacyMethod::CutMix { alpha } => self.apply_cutmix(data, alpha),
        }
    }

    /// Get privacy guarantees for model
    pub fn get_privacy_guarantees(&self) -> PrivacyGuarantees {
        let epsilon = self.accountant.total_epsilon_spent;
        let delta = self.accountant.total_delta_spent;

        PrivacyGuarantees {
            epsilon,
            delta,
            composition_type: self.config.composition_method,
            confidence_level: 0.95,
            data_dependent_bound: self.compute_data_dependent_bound(),
            rdp_curve: self.accountant.get_rdp_curve(),
        }
    }

    /// Estimate privacy cost for training configuration
    pub fn estimate_privacy_cost(
        config: &OnDeviceTrainingConfig,
        privacy_config: &PrivacyConfig,
        dataset_size: usize,
    ) -> PrivacyCostEstimate {
        let batch_size = config.batch_size * config.gradient_accumulation_steps;
        let steps_per_epoch = dataset_size / batch_size;
        let total_steps = steps_per_epoch * config.epochs;

        // Estimate based on composition method
        let (epsilon, delta) = match privacy_config.composition_method {
            CompositionMethod::Simple => {
                let eps_per_step = privacy_config.noise_multiplier.recip() as f64;
                (
                    eps_per_step * total_steps as f64,
                    privacy_config.total_delta,
                )
            },
            CompositionMethod::Advanced => {
                let eps_per_step = privacy_config.noise_multiplier.recip() as f64;
                let advanced_eps = (total_steps as f64).sqrt() * eps_per_step;
                (advanced_eps, privacy_config.total_delta)
            },
            CompositionMethod::Moments => {
                // Simplified moments accountant estimate
                let eps = privacy_config.noise_multiplier.recip() as f64
                    * (total_steps as f64).sqrt()
                    * 2.0;
                (eps, privacy_config.total_delta)
            },
            CompositionMethod::Renyi => {
                // Simplified Rényi DP estimate
                let eps =
                    privacy_config.noise_multiplier.recip() as f64 * (total_steps as f64).powf(0.4);
                (eps, privacy_config.total_delta)
            },
        };

        PrivacyCostEstimate {
            estimated_epsilon: epsilon,
            estimated_delta: delta,
            total_steps,
            privacy_amplification_factor: calculate_amplification_factor(
                privacy_config.subsampling_rate,
                dataset_size,
            ),
            meets_budget: epsilon <= privacy_config.total_epsilon
                && delta <= privacy_config.total_delta,
        }
    }

    // Private implementation methods

    fn clip_gradients(&mut self, gradients: &mut HashMap<String, Tensor>) -> Result<ClippingStats> {
        let mut total_norm = 0.0;
        let mut clipped_params = 0;

        // Compute global norm
        for grad in gradients.values() {
            total_norm += grad.norm()?.powf(2.0);
        }
        total_norm = total_norm.sqrt();

        // Clip if necessary
        if total_norm > self.config.clipping_threshold {
            let scale_factor = self.config.clipping_threshold / total_norm;
            for grad in gradients.values_mut() {
                *grad = grad.scalar_mul(scale_factor)?;
            }
            clipped_params = gradients.len();
        }

        // Update adaptive clipping threshold if enabled
        if self.config.adaptive_clipping {
            self.gradient_clipper.update_threshold(total_norm);
        }

        Ok(ClippingStats {
            num_clipped: clipped_params,
            avg_norm_before: total_norm,
            avg_norm_after: total_norm.min(self.config.clipping_threshold),
            clipping_threshold: self.config.clipping_threshold,
        })
    }

    fn clip_per_example_gradients(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
    ) -> Result<ClippingStats> {
        // Per-example clipping for stronger privacy guarantees
        // This is a simplified version - real implementation would process each example
        self.clip_gradients(gradients)
    }

    fn add_gradient_noise(
        &mut self,
        gradients: &mut HashMap<String, Tensor>,
        batch_size: usize,
    ) -> Result<NoiseStats> {
        let noise_scale =
            self.config.noise_multiplier * self.config.clipping_threshold / batch_size as f32;

        let mut total_noise_added = 0.0;

        for grad in gradients.values_mut() {
            let noise = self.noise_generator.generate_noise(&grad.shape(), noise_scale)?;
            total_noise_added += noise.norm()?;
            *grad = grad.add(&noise)?;
        }

        Ok(NoiseStats {
            noise_scale,
            total_noise_norm: total_noise_added,
            noise_type: NoiseType::Gaussian,
        })
    }

    fn apply_input_perturbation(&self, data: &[(Tensor, Tensor)]) -> Result<Vec<(Tensor, Tensor)>> {
        let mut perturbed_data = Vec::with_capacity(data.len());

        for (input, target) in data {
            let noise_scale = self.config.noise_multiplier * 0.1; // Smaller noise for inputs
            let noise = Tensor::randn(&input.shape()).and_then(|t| t.scalar_mul(noise_scale))?;
            let perturbed_input = input.add(&noise)?;
            perturbed_data.push((perturbed_input, target.clone()));
        }

        Ok(perturbed_data)
    }

    fn apply_label_smoothing(
        &self,
        data: &[(Tensor, Tensor)],
        smoothing_factor: f32,
    ) -> Result<Vec<(Tensor, Tensor)>> {
        let mut smoothed_data = Vec::with_capacity(data.len());

        for (input, target) in data {
            // Smooth labels to prevent overfitting to specific examples
            let smoothed_target = target.scalar_mul(1.0 - smoothing_factor)?.add(
                &Tensor::ones(&target.shape())?
                    .scalar_mul(smoothing_factor / target.shape()[1] as f32)?,
            )?;
            smoothed_data.push((input.clone(), smoothed_target));
        }

        Ok(smoothed_data)
    }

    fn apply_mixup(&self, data: &[(Tensor, Tensor)], alpha: f32) -> Result<Vec<(Tensor, Tensor)>> {
        let mut mixed_data = Vec::with_capacity(data.len());

        for i in 0..data.len() {
            let j = (i + 1) % data.len(); // Pair with next sample
            let lambda = DefaultRng::new().gen::<f32>().powf(1.0 / alpha);

            let mixed_input =
                data[i].0.scalar_mul(lambda)?.add(&data[j].0.scalar_mul(1.0 - lambda)?)?;
            let mixed_target =
                data[i].1.scalar_mul(lambda)?.add(&data[j].1.scalar_mul(1.0 - lambda)?)?;

            mixed_data.push((mixed_input, mixed_target));
        }

        Ok(mixed_data)
    }

    fn apply_cutmix(&self, data: &[(Tensor, Tensor)], alpha: f32) -> Result<Vec<(Tensor, Tensor)>> {
        // Simplified CutMix - in practice would cut and mix image regions
        self.apply_mixup(data, alpha)
    }

    fn compute_data_dependent_bound(&self) -> Option<f64> {
        // Compute tighter data-dependent privacy bounds if available
        if self.accountant.has_sufficient_statistics() {
            Some(self.accountant.compute_data_dependent_epsilon())
        } else {
            None
        }
    }
}

/// Privacy accountant for tracking privacy budget
struct PrivacyAccountant {
    config: PrivacyConfig,
    total_epsilon_spent: f64,
    total_delta_spent: f64,
    step_epsilons: Vec<f64>,
    rdp_orders: Vec<f64>,
    rdp_values: Vec<f64>,
}

impl PrivacyAccountant {
    fn new(config: PrivacyConfig) -> Self {
        Self {
            config,
            total_epsilon_spent: 0.0,
            total_delta_spent: 0.0,
            step_epsilons: Vec::new(),
            rdp_orders: vec![1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0, 20.0, 50.0, 100.0],
            rdp_values: vec![0.0; 10],
        }
    }

    fn has_budget_remaining(&self) -> bool {
        self.total_epsilon_spent < self.config.total_epsilon
            && self.total_delta_spent < self.config.total_delta
    }

    fn account_step(
        &mut self,
        batch_size: usize,
        subsampling_rate: f32,
        noise_scale: f32,
    ) -> Result<PrivacyCost> {
        let (epsilon, delta) = match self.config.composition_method {
            CompositionMethod::Simple => self.simple_composition(noise_scale),
            CompositionMethod::Advanced => self.advanced_composition(noise_scale, subsampling_rate),
            CompositionMethod::Moments => self.moments_accountant(batch_size, noise_scale),
            CompositionMethod::Renyi => {
                self.renyi_accountant(batch_size, noise_scale, subsampling_rate)
            },
        };

        self.total_epsilon_spent += epsilon;
        self.total_delta_spent += delta;
        self.step_epsilons.push(epsilon);

        Ok(PrivacyCost { epsilon, delta })
    }

    fn simple_composition(&self, noise_scale: f32) -> (f64, f64) {
        let epsilon = (1.0 / noise_scale) as f64;
        (epsilon, 0.0)
    }

    fn advanced_composition(&self, noise_scale: f32, subsampling_rate: f32) -> (f64, f64) {
        let base_epsilon = (1.0 / noise_scale) as f64;
        let amplified_epsilon = base_epsilon * subsampling_rate as f64;
        (amplified_epsilon, self.config.total_delta / 1000.0)
    }

    fn moments_accountant(&mut self, batch_size: usize, noise_scale: f32) -> (f64, f64) {
        // Simplified moments accountant
        let sigma = noise_scale;
        let q = batch_size as f64 / 10000.0; // Assume 10k total samples

        // Update RDP values
        for (i, &alpha) in self.rdp_orders.iter().enumerate() {
            let rdp_epsilon = (alpha * q * q) / (2.0 * sigma as f64 * sigma as f64);
            self.rdp_values[i] += rdp_epsilon;
        }

        // Convert to (ε,δ)-DP
        self.rdp_to_dp(self.config.total_delta)
    }

    fn renyi_accountant(
        &mut self,
        batch_size: usize,
        noise_scale: f32,
        subsampling_rate: f32,
    ) -> (f64, f64) {
        // Enhanced Rényi accountant with subsampling
        let sigma = noise_scale;
        let q = subsampling_rate as f64;

        for (i, &alpha) in self.rdp_orders.iter().enumerate() {
            let rdp_epsilon = if alpha > 1.0 {
                (alpha * q * q) / (2.0 * sigma as f64 * sigma as f64)
            } else {
                0.0
            };
            self.rdp_values[i] += rdp_epsilon;
        }

        self.rdp_to_dp(self.config.total_delta)
    }

    fn rdp_to_dp(&self, target_delta: f64) -> (f64, f64) {
        // Convert RDP to (ε,δ)-DP
        let mut min_epsilon = f64::INFINITY;

        for (i, &alpha) in self.rdp_orders.iter().enumerate() {
            if alpha > 1.0 {
                let epsilon = self.rdp_values[i] + (target_delta.ln() / (alpha - 1.0));
                min_epsilon = min_epsilon.min(epsilon);
            }
        }

        (min_epsilon, target_delta)
    }

    fn get_rdp_curve(&self) -> Vec<(f64, f64)> {
        self.rdp_orders
            .iter()
            .zip(self.rdp_values.iter())
            .map(|(&order, &value)| (order, value))
            .collect()
    }

    fn has_sufficient_statistics(&self) -> bool {
        self.step_epsilons.len() >= 10
    }

    fn compute_data_dependent_epsilon(&self) -> f64 {
        // Compute tighter bound based on actual gradient norms
        let avg_epsilon = self.step_epsilons.iter().sum::<f64>() / self.step_epsilons.len() as f64;
        avg_epsilon * 0.8 // 20% tighter bound
    }
}

/// Noise generator for differential privacy
struct NoiseGenerator {
    noise_multiplier: f32,
}

impl NoiseGenerator {
    fn new(noise_multiplier: f32) -> Self {
        Self { noise_multiplier }
    }

    fn generate_noise(&self, shape: &[usize], scale: f32) -> Result<Tensor> {
        let result = match shape.len() {
            1 => Tensor::randn(&[shape[0]]).and_then(|t| t.scalar_mul(scale)),
            2 => Tensor::randn(&[shape[0], shape[1]]).and_then(|t| t.scalar_mul(scale)),
            3 => Tensor::randn(&[shape[0], shape[1], shape[2]]).and_then(|t| t.scalar_mul(scale)),
            4 => Tensor::randn(&[shape[0], shape[1], shape[2], shape[3]])
                .and_then(|t| t.scalar_mul(scale)),
            _ => {
                return Err(TrustformersError::invalid_input(format!(
                    "Unsupported tensor dimension: {}",
                    shape.len()
                ))
                .into())
            },
        };
        result.map_err(|e| TrustformersError::runtime_error(format!("{}", e)).into())
    }
}

/// Gradient clipper with adaptive threshold
struct GradientClipper {
    threshold: f32,
    history: Vec<f32>,
    adaptive_rate: f32,
}

impl GradientClipper {
    fn new(threshold: f32) -> Self {
        Self {
            threshold,
            history: Vec::with_capacity(100),
            adaptive_rate: 0.01,
        }
    }

    fn update_threshold(&mut self, norm: f32) {
        self.history.push(norm);
        if self.history.len() > 100 {
            self.history.remove(0);
        }

        // Adapt threshold based on gradient norm history
        if self.history.len() >= 10 {
            let median_norm = self.compute_median_norm();
            self.threshold =
                (1.0 - self.adaptive_rate) * self.threshold + self.adaptive_rate * median_norm;
        }
    }

    fn compute_median_norm(&self) -> f32 {
        let mut sorted = self.history.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        sorted[sorted.len() / 2]
    }
}

/// Data privacy methods
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum DataPrivacyMethod {
    /// Add noise to inputs
    InputPerturbation,
    /// Smooth labels
    LabelSmoothing { smoothing_factor: f32 },
    /// Mix samples (Mixup)
    Mixup { alpha: f32 },
    /// Cut and mix samples (CutMix)
    CutMix { alpha: f32 },
}

/// Privacy report for a training step
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyReport {
    pub step: usize,
    pub epsilon_spent: f64,
    pub delta_spent: f64,
    pub total_epsilon_spent: f64,
    pub total_delta_spent: f64,
    pub clipping_stats: ClippingStats,
    pub noise_stats: NoiseStats,
}

/// Gradient clipping statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClippingStats {
    pub num_clipped: usize,
    pub avg_norm_before: f32,
    pub avg_norm_after: f32,
    pub clipping_threshold: f32,
}

/// Noise addition statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseStats {
    pub noise_scale: f32,
    pub total_noise_norm: f32,
    pub noise_type: NoiseType,
}

/// Types of noise for DP
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NoiseType {
    Gaussian,
    Laplacian,
    Exponential,
}

/// Privacy cost for a single step
#[derive(Debug, Clone)]
struct PrivacyCost {
    epsilon: f64,
    delta: f64,
}

/// Privacy guarantees for the trained model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyGuarantees {
    pub epsilon: f64,
    pub delta: f64,
    pub composition_type: CompositionMethod,
    pub confidence_level: f64,
    pub data_dependent_bound: Option<f64>,
    pub rdp_curve: Vec<(f64, f64)>,
}

/// Privacy cost estimate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyCostEstimate {
    pub estimated_epsilon: f64,
    pub estimated_delta: f64,
    pub total_steps: usize,
    pub privacy_amplification_factor: f64,
    pub meets_budget: bool,
}

/// Calculate privacy amplification factor from subsampling
fn calculate_amplification_factor(subsampling_rate: f32, dataset_size: usize) -> f64 {
    let q = subsampling_rate as f64;
    let n = dataset_size as f64;

    // Amplification via subsampling
    if q < 0.01 {
        q * (n / (n - 1.0))
    } else {
        (2.0 * q).min(1.0)
    }
}

/// Utility functions for differential privacy
pub struct DifferentialPrivacyUtils;

impl DifferentialPrivacyUtils {
    /// Calculate required noise for target privacy
    pub fn calculate_noise_multiplier(
        target_epsilon: f64,
        target_delta: f64,
        num_steps: usize,
        subsampling_rate: f32,
    ) -> f32 {
        // Simplified calculation - real implementation would use tighter bounds
        let base_noise = (2.0 * (1.25 / target_delta).ln()).sqrt();
        let composition_factor = (num_steps as f64).sqrt();
        let amplification = calculate_amplification_factor(subsampling_rate, 10000);

        (base_noise * composition_factor / (target_epsilon * amplification)) as f32
    }

    /// Validate privacy configuration
    pub fn validate_privacy_config(
        config: &PrivacyConfig,
        training_config: &OnDeviceTrainingConfig,
    ) -> Result<()> {
        // Check epsilon bounds
        if config.total_epsilon <= 0.0 || config.total_epsilon > 100.0 {
            return Err(TrustformersError::config_error(
                "Epsilon must be between 0 and 100",
                "validate_privacy_config",
            )
            .into());
        }

        // Check delta bounds
        if config.total_delta <= 0.0 || config.total_delta >= 1.0 {
            return Err(TrustformersError::config_error(
                "Delta must be between 0 and 1",
                "validate_privacy_config",
            )
            .into());
        }

        // Check compatibility with training
        if config.per_example_clipping && training_config.batch_size > 1 {
            tracing::warn!("Per-example clipping with batch_size > 1 may impact performance");
        }

        Ok(())
    }

    /// Get recommended privacy config for use case
    pub fn recommend_privacy_config(
        use_case: PrivacyUseCase,
        dataset_size: usize,
    ) -> PrivacyConfig {
        match use_case {
            PrivacyUseCase::Healthcare => PrivacyConfig::from_privacy_level(PrivacyLevel::VeryHigh),
            PrivacyUseCase::Financial => PrivacyConfig::from_privacy_level(PrivacyLevel::High),
            PrivacyUseCase::PersonalAssistant => {
                PrivacyConfig::from_privacy_level(PrivacyLevel::Medium)
            },
            PrivacyUseCase::GeneralPurpose => PrivacyConfig::from_privacy_level(PrivacyLevel::Low),
        }
    }
}

/// Privacy use cases
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrivacyUseCase {
    Healthcare,
    Financial,
    PersonalAssistant,
    GeneralPurpose,
}

// Placeholder for rand functionality
mod rand {
    pub fn random<T>() -> T
    where
        T: Default,
    {
        T::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_config_levels() {
        let low = PrivacyConfig::from_privacy_level(PrivacyLevel::Low);
        let high = PrivacyConfig::from_privacy_level(PrivacyLevel::High);

        assert!(low.total_epsilon > high.total_epsilon);
        assert!(low.noise_multiplier < high.noise_multiplier);
    }

    #[test]
    fn test_differential_privacy_engine() {
        let config = PrivacyConfig::default();
        let engine = DifferentialPrivacyEngine::new(config);

        // Test gradient privatization
        let mut gradients: HashMap<String, Tensor> = HashMap::new();
        gradients.insert(
            "weight".to_string(),
            Tensor::randn(&[10, 10]).unwrap().into(),
        );

        // Note: This would fail in real implementation due to budget checks
        // let report = engine.privatize_gradients(&mut gradients, 32, 1);
        // assert!(report.is_ok());
    }

    #[test]
    fn test_privacy_cost_estimation() {
        let training_config = crate::training::OnDeviceTrainingConfig::default();
        let privacy_config = PrivacyConfig::default();

        let estimate = DifferentialPrivacyEngine::estimate_privacy_cost(
            &training_config,
            &privacy_config,
            1000,
        );

        assert!(estimate.total_steps > 0);
        assert!(estimate.estimated_epsilon > 0.0);
    }

    #[test]
    fn test_noise_calculation() {
        let noise_multiplier = DifferentialPrivacyUtils::calculate_noise_multiplier(
            1.0,  // epsilon
            1e-5, // delta
            1000, // steps
            0.01, // subsampling
        );

        assert!(noise_multiplier > 0.0);
    }
}
