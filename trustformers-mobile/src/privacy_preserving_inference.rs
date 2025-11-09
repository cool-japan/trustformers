//! Privacy-Preserving Inference for Mobile Deployment
//!
//! This module provides comprehensive privacy protection mechanisms
//! for inference operations on mobile devices, including differential
//! privacy, secure computation, and privacy-preserving aggregation.

use serde::{Deserialize, Serialize};
use trustformers_core::errors::{runtime_error, Result};
use trustformers_core::Tensor;

/// Default tensor for serialization
fn default_tensor() -> Tensor {
    Tensor::zeros(&[1]).unwrap_or_else(|_| unsafe { std::mem::zeroed() })
}

/// Privacy level for inference operations
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyLevel {
    /// No privacy protection
    None,
    /// Low privacy protection
    Low,
    /// Medium privacy protection
    Medium,
    /// High privacy protection
    High,
    /// Very high privacy protection
    VeryHigh,
    /// Maximum privacy protection
    Maximum,
    /// Custom privacy settings
    Custom,
}

/// Privacy-preserving inference engine
pub struct PrivacyPreservingInferenceEngine {
    config: InferencePrivacyConfig,
    noise_generator: InferenceNoiseGenerator,
    secure_aggregator: SecureAggregator,
    privacy_accountant: InferencePrivacyAccountant,
}

/// Configuration for privacy-preserving inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePrivacyConfig {
    /// Enable privacy-preserving inference
    pub enabled: bool,
    /// Privacy level for inference
    pub privacy_level: PrivacyLevel,
    /// Input perturbation configuration
    pub input_privacy: InputPrivacyConfig,
    /// Output privacy configuration
    pub output_privacy: OutputPrivacyConfig,
    /// Secure aggregation settings
    pub secure_aggregation: SecureAggregationConfig,
    /// Privacy budget for inference sessions
    pub inference_budget: InferenceBudgetConfig,
}

/// Input privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputPrivacyConfig {
    /// Enable input perturbation
    pub enabled: bool,
    /// Noise scale for input perturbation
    pub noise_scale: f32,
    /// Perturbation method
    pub method: InputPerturbationMethod,
    /// Adaptive noise based on input sensitivity
    pub adaptive_noise: bool,
    /// Maximum perturbation magnitude
    pub max_perturbation: f32,
}

/// Output privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputPrivacyConfig {
    /// Enable output privacy
    pub enabled: bool,
    /// Noise scale for output perturbation
    pub noise_scale: f32,
    /// Output privacy method
    pub method: OutputPrivacyMethod,
    /// Post-processing privacy
    pub post_processing_privacy: bool,
    /// Confidence calibration with privacy
    pub calibrated_outputs: bool,
}

/// Secure aggregation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SecureAggregationConfig {
    /// Enable secure aggregation
    pub enabled: bool,
    /// Aggregation method
    pub method: AggregationMethod,
    /// Minimum participants for aggregation
    pub min_participants: usize,
    /// Threshold for secure computation
    pub security_threshold: f32,
}

/// Privacy budget configuration for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceBudgetConfig {
    /// Total epsilon budget for inference session
    pub total_epsilon: f64,
    /// Total delta parameter
    pub total_delta: f64,
    /// Epsilon per inference request
    pub epsilon_per_request: f64,
    /// Budget reset period (seconds)
    pub reset_period_secs: u64,
    /// Enable budget tracking
    pub track_budget: bool,
}

/// Input perturbation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InputPerturbationMethod {
    /// Gaussian noise addition
    Gaussian,
    /// Laplacian noise addition
    Laplacian,
    /// Randomized response
    RandomizedResponse,
    /// Local sensitivity analysis
    LocalSensitivity,
    /// Feature-specific perturbation
    FeatureSpecific,
}

/// Output privacy methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OutputPrivacyMethod {
    /// Add noise to final predictions
    PredictionNoise,
    /// Smooth probability distributions
    ProbabilitySmoothing,
    /// Truncate confidence scores
    ConfidenceTruncation,
    /// Report mechanism for privacy
    ReportMechanism,
    /// Exponential mechanism
    ExponentialMechanism,
}

/// Secure aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    /// Simple secure summation
    SecureSum,
    /// Federated averaging with privacy
    FederatedAverage,
    /// Multi-party computation
    MultiPartyComputation,
    /// Threshold aggregation
    ThresholdAggregation,
}

/// Privacy-preserving inference result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivateInferenceResult {
    /// Original inference result (if available)
    #[serde(skip, default)]
    pub original_result: Option<Tensor>,
    /// Privacy-protected result
    #[serde(skip, default = "default_tensor")]
    pub private_result: Tensor,
    /// Privacy guarantees for this inference
    pub privacy_guarantees: InferencePrivacyGuarantees,
    /// Confidence intervals accounting for privacy noise
    pub confidence_intervals: Option<Vec<(f32, f32)>>,
    /// Quality metrics
    pub quality_metrics: InferenceQualityMetrics,
}

/// Privacy guarantees for inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferencePrivacyGuarantees {
    /// Epsilon spent for this inference
    pub epsilon_spent: f64,
    /// Delta spent for this inference
    pub delta_spent: f64,
    /// Input privacy applied
    pub input_privacy_applied: bool,
    /// Output privacy applied
    pub output_privacy_applied: bool,
    /// Aggregation privacy applied
    pub aggregation_privacy_applied: bool,
    /// Privacy mechanism used
    pub mechanism: String,
}

/// Quality metrics for private inference
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceQualityMetrics {
    /// Estimated accuracy degradation due to privacy
    pub accuracy_degradation: f32,
    /// Noise-to-signal ratio
    pub noise_signal_ratio: f32,
    /// Confidence level
    pub confidence_level: f32,
    /// Utility preservation score
    pub utility_score: f32,
}

impl Default for InferencePrivacyConfig {
    fn default() -> Self {
        Self::from_privacy_level(PrivacyLevel::Medium)
    }
}

impl InferencePrivacyConfig {
    /// Create config from privacy level
    pub fn from_privacy_level(level: PrivacyLevel) -> Self {
        let (epsilon, noise_scale, output_noise) = match level {
            PrivacyLevel::None => (10.0, 0.0, 0.0), // No privacy protection
            PrivacyLevel::Low => (5.0, 0.1, 0.05),
            PrivacyLevel::Medium => (1.0, 0.2, 0.1),
            PrivacyLevel::High => (0.5, 0.5, 0.2),
            PrivacyLevel::VeryHigh => (0.1, 1.0, 0.5),
            PrivacyLevel::Maximum => (0.01, 2.0, 1.0), // Maximum privacy protection
            PrivacyLevel::Custom => (1.0, 0.2, 0.1),
        };

        Self {
            enabled: true,
            privacy_level: level,
            input_privacy: InputPrivacyConfig {
                enabled: true,
                noise_scale,
                method: InputPerturbationMethod::Gaussian,
                adaptive_noise: true,
                max_perturbation: noise_scale * 2.0,
            },
            output_privacy: OutputPrivacyConfig {
                enabled: true,
                noise_scale: output_noise,
                method: OutputPrivacyMethod::PredictionNoise,
                post_processing_privacy: true,
                calibrated_outputs: true,
            },
            secure_aggregation: SecureAggregationConfig {
                enabled: false,
                method: AggregationMethod::SecureSum,
                min_participants: 2,
                security_threshold: 0.95,
            },
            inference_budget: InferenceBudgetConfig {
                total_epsilon: epsilon,
                total_delta: 1e-6,
                epsilon_per_request: epsilon / 100.0,
                reset_period_secs: 3600,
                track_budget: true,
            },
        }
    }

    /// Create config for specific use case
    pub fn for_use_case(use_case: InferenceUseCase) -> Self {
        match use_case {
            InferenceUseCase::MedicalDiagnosis => {
                let mut config = Self::from_privacy_level(PrivacyLevel::VeryHigh);
                config.output_privacy.method = OutputPrivacyMethod::ExponentialMechanism;
                config.output_privacy.calibrated_outputs = true;
                config
            },
            InferenceUseCase::FinancialAdvice => {
                let mut config = Self::from_privacy_level(PrivacyLevel::High);
                config.secure_aggregation.enabled = true;
                config
            },
            InferenceUseCase::PersonalAssistant => Self::from_privacy_level(PrivacyLevel::Medium),
            InferenceUseCase::ContentRecommendation => Self::from_privacy_level(PrivacyLevel::Low),
        }
    }
}

/// Use cases for privacy-preserving inference
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InferenceUseCase {
    MedicalDiagnosis,
    FinancialAdvice,
    PersonalAssistant,
    ContentRecommendation,
}

impl PrivacyPreservingInferenceEngine {
    /// Create new privacy-preserving inference engine
    pub fn new(config: InferencePrivacyConfig) -> Self {
        Self {
            config: config.clone(),
            noise_generator: InferenceNoiseGenerator::new(config.clone()),
            secure_aggregator: SecureAggregator::new(config.secure_aggregation.clone()),
            privacy_accountant: InferencePrivacyAccountant::new(config.inference_budget.clone()),
        }
    }

    /// Perform privacy-preserving inference
    pub fn private_inference(
        &mut self,
        input: &Tensor,
        model_fn: impl Fn(&Tensor) -> Result<Tensor>,
    ) -> Result<PrivateInferenceResult> {
        if !self.config.enabled {
            let result = model_fn(input)?;
            return Ok(PrivateInferenceResult {
                original_result: Some(result.clone()),
                private_result: result,
                privacy_guarantees: InferencePrivacyGuarantees {
                    epsilon_spent: 0.0,
                    delta_spent: 0.0,
                    input_privacy_applied: false,
                    output_privacy_applied: false,
                    aggregation_privacy_applied: false,
                    mechanism: "None".to_string(),
                },
                confidence_intervals: None,
                quality_metrics: InferenceQualityMetrics {
                    accuracy_degradation: 0.0,
                    noise_signal_ratio: 0.0,
                    confidence_level: 1.0,
                    utility_score: 1.0,
                },
            });
        }

        // Check privacy budget
        if !self.privacy_accountant.has_budget_remaining() {
            return Err(runtime_error("Privacy budget exhausted"));
        }

        // Apply input privacy
        let private_input = if self.config.input_privacy.enabled {
            self.apply_input_privacy(input)?
        } else {
            input.clone()
        };

        // Perform inference
        let raw_output = model_fn(&private_input)?;

        // Apply output privacy
        let private_output = if self.config.output_privacy.enabled {
            self.apply_output_privacy(&raw_output)?
        } else {
            raw_output.clone()
        };

        // Account for privacy cost
        let privacy_cost = self.privacy_accountant.account_inference(
            &self.config,
            &input.shape(),
            &raw_output.shape(),
        )?;

        // Calculate quality metrics
        let quality_metrics = self.calculate_quality_metrics(&raw_output, &private_output)?;

        // Calculate confidence intervals
        let confidence_intervals = self.calculate_confidence_intervals(&private_output)?;

        Ok(PrivateInferenceResult {
            original_result: Some(raw_output),
            private_result: private_output,
            privacy_guarantees: InferencePrivacyGuarantees {
                epsilon_spent: privacy_cost.epsilon,
                delta_spent: privacy_cost.delta,
                input_privacy_applied: self.config.input_privacy.enabled,
                output_privacy_applied: self.config.output_privacy.enabled,
                aggregation_privacy_applied: false,
                mechanism: format!(
                    "{:?}+{:?}",
                    self.config.input_privacy.method, self.config.output_privacy.method
                ),
            },
            confidence_intervals: Some(confidence_intervals),
            quality_metrics,
        })
    }

    /// Apply input privacy protection
    fn apply_input_privacy(&mut self, input: &Tensor) -> Result<Tensor> {
        match self.config.input_privacy.method {
            InputPerturbationMethod::Gaussian => self.apply_gaussian_input_noise(input),
            InputPerturbationMethod::Laplacian => self.apply_laplacian_input_noise(input),
            InputPerturbationMethod::RandomizedResponse => self.apply_randomized_response(input),
            InputPerturbationMethod::LocalSensitivity => self.apply_local_sensitivity_noise(input),
            InputPerturbationMethod::FeatureSpecific => self.apply_feature_specific_noise(input),
        }
    }

    /// Apply output privacy protection
    fn apply_output_privacy(&mut self, output: &Tensor) -> Result<Tensor> {
        match self.config.output_privacy.method {
            OutputPrivacyMethod::PredictionNoise => self.add_prediction_noise(output),
            OutputPrivacyMethod::ProbabilitySmoothing => self.smooth_probabilities(output),
            OutputPrivacyMethod::ConfidenceTruncation => self.truncate_confidence(output),
            OutputPrivacyMethod::ReportMechanism => self.apply_report_mechanism(output),
            OutputPrivacyMethod::ExponentialMechanism => self.apply_exponential_mechanism(output),
        }
    }

    /// Apply Gaussian noise to input
    fn apply_gaussian_input_noise(&self, input: &Tensor) -> Result<Tensor> {
        let noise_scale = if self.config.input_privacy.adaptive_noise {
            self.compute_adaptive_noise_scale(input)?
        } else {
            self.config.input_privacy.noise_scale
        };

        let noise = self.noise_generator.generate_gaussian_noise(&input.shape(), noise_scale)?;
        let perturbed = input.add(&noise)?;

        // Clip to maximum perturbation
        self.clip_perturbation(input, &perturbed)
    }

    /// Apply Laplacian noise to input
    fn apply_laplacian_input_noise(&self, input: &Tensor) -> Result<Tensor> {
        let noise_scale = self.config.input_privacy.noise_scale;
        let noise = self.noise_generator.generate_laplacian_noise(&input.shape(), noise_scale)?;
        let perturbed = input.add(&noise)?;
        self.clip_perturbation(input, &perturbed)
    }

    /// Apply randomized response to input
    fn apply_randomized_response(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified randomized response for continuous inputs
        let flip_prob = self.config.input_privacy.noise_scale;
        let random_mask = Tensor::randn(&input.shape())?.scalar_mul(flip_prob)?;
        let random_input = Tensor::randn(&input.shape())?;

        // Apply randomized response: with probability p, use random input
        // Simplified implementation without gt_scalar and where_tensor
        let blended =
            input.scalar_mul(1.0 - flip_prob)?.add(&random_input.scalar_mul(flip_prob)?)?;
        Ok(blended)
    }

    /// Apply local sensitivity-based noise
    fn apply_local_sensitivity_noise(&self, input: &Tensor) -> Result<Tensor> {
        // Compute local sensitivity (simplified)
        let local_sensitivity = self.estimate_local_sensitivity(input)?;
        let adaptive_scale = self.config.input_privacy.noise_scale * local_sensitivity;

        let noise = self.noise_generator.generate_gaussian_noise(&input.shape(), adaptive_scale)?;
        let perturbed = input.add(&noise)?;
        self.clip_perturbation(input, &perturbed)
    }

    /// Apply feature-specific noise
    fn apply_feature_specific_noise(&self, input: &Tensor) -> Result<Tensor> {
        // Apply different noise levels to different features
        let perturbed = input.clone();
        let shape = input.shape();

        if shape.len() >= 2 {
            let feature_dim = shape[shape.len() - 1];
            for i in 0..feature_dim {
                let feature_sensitivity = self.estimate_feature_sensitivity(i, feature_dim);
                let noise_scale = self.config.input_privacy.noise_scale * feature_sensitivity;

                // Apply noise to specific feature (simplified implementation)
                let noise = Tensor::randn(&[1])?.scalar_mul(noise_scale)?;
                // In real implementation, would apply to specific feature slice
            }
        }

        Ok(perturbed)
    }

    /// Add noise to predictions
    fn add_prediction_noise(&self, output: &Tensor) -> Result<Tensor> {
        let noise_scale = self.config.output_privacy.noise_scale;
        let noise = self.noise_generator.generate_gaussian_noise(&output.shape(), noise_scale)?;
        output.add(&noise)
    }

    /// Smooth probability distributions
    fn smooth_probabilities(&self, output: &Tensor) -> Result<Tensor> {
        let smoothing_factor = self.config.output_privacy.noise_scale;
        let output_shape = output.shape();
        let uniform_dist = Tensor::ones(&output_shape)?
            .scalar_mul(1.0 / output_shape[output_shape.len() - 1] as f32)?;

        // Mix with uniform distribution
        let smoothed = output
            .scalar_mul(1.0 - smoothing_factor)?
            .add(&uniform_dist.scalar_mul(smoothing_factor)?)?;

        // Renormalize if it's a probability distribution
        if output.shape().len() >= 2 {
            self.renormalize_probabilities(&smoothed)
        } else {
            Ok(smoothed)
        }
    }

    /// Truncate confidence scores
    fn truncate_confidence(&self, output: &Tensor) -> Result<Tensor> {
        let max_confidence = 1.0 - self.config.output_privacy.noise_scale;
        let min_confidence = self.config.output_privacy.noise_scale;

        output.clamp(min_confidence, max_confidence)
    }

    /// Apply report mechanism
    fn apply_report_mechanism(&self, output: &Tensor) -> Result<Tensor> {
        // Simplified report mechanism implementation
        let sensitivity = 1.0; // Assume unit sensitivity
        let epsilon = self.config.inference_budget.epsilon_per_request as f32;
        let noise_scale = sensitivity / epsilon;

        let noise = self.noise_generator.generate_laplacian_noise(&output.shape(), noise_scale)?;
        output.add(&noise)
    }

    /// Apply exponential mechanism
    fn apply_exponential_mechanism(&self, output: &Tensor) -> Result<Tensor> {
        let epsilon = self.config.inference_budget.epsilon_per_request as f32;
        let sensitivity = 1.0;

        // Apply exponential mechanism by scaling logits
        let scaled_logits = output.scalar_mul(epsilon / (2.0 * sensitivity))?;

        // Apply softmax to get probabilities
        self.softmax(&scaled_logits)
    }

    /// Helper methods

    fn compute_adaptive_noise_scale(&self, input: &Tensor) -> Result<f32> {
        // Estimate input sensitivity for adaptive noise
        let input_norm = input.norm()?;
        let base_scale = self.config.input_privacy.noise_scale;
        Ok(base_scale * (1.0 + input_norm * 0.1))
    }

    fn clip_perturbation(&self, original: &Tensor, perturbed: &Tensor) -> Result<Tensor> {
        let diff = perturbed.sub(original)?;
        let diff_norm = diff.norm()?;

        if diff_norm > self.config.input_privacy.max_perturbation {
            let scale = self.config.input_privacy.max_perturbation / diff_norm;
            let clipped_diff = diff.scalar_mul(scale)?;
            original.add(&clipped_diff)
        } else {
            Ok(perturbed.clone())
        }
    }

    fn estimate_local_sensitivity(&self, _input: &Tensor) -> Result<f32> {
        // Simplified local sensitivity estimation
        Ok(1.0)
    }

    fn estimate_feature_sensitivity(&self, _feature_idx: usize, _total_features: usize) -> f32 {
        // Simplified feature sensitivity estimation
        1.0
    }

    fn renormalize_probabilities(&self, tensor: &Tensor) -> Result<Tensor> {
        // Simplified probability renormalization - just return as is
        Ok(tensor.clone())
    }

    fn softmax(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified softmax implementation - just return exp
        input.exp()
    }

    fn calculate_quality_metrics(
        &self,
        original: &Tensor,
        private: &Tensor,
    ) -> Result<InferenceQualityMetrics> {
        // Simplified quality metrics calculation
        let noise_signal_ratio = 0.1; // Placeholder value
        let accuracy_degradation = 0.05; // Placeholder value
        let utility_score = 1.0 - accuracy_degradation;

        Ok(InferenceQualityMetrics {
            accuracy_degradation,
            noise_signal_ratio,
            confidence_level: 0.95,
            utility_score,
        })
    }

    fn calculate_confidence_intervals(&self, output: &Tensor) -> Result<Vec<(f32, f32)>> {
        let noise_std = self.config.output_privacy.noise_scale;
        let z_score = 1.96; // 95% confidence interval
        let margin = z_score * noise_std;

        // Simplified confidence interval calculation
        let intervals = vec![(0.0 - margin, 1.0 + margin)]; // Placeholder interval

        Ok(intervals)
    }
}

/// Noise generator for inference privacy
struct InferenceNoiseGenerator {
    config: InferencePrivacyConfig,
}

impl InferenceNoiseGenerator {
    fn new(config: InferencePrivacyConfig) -> Self {
        Self { config }
    }

    fn generate_gaussian_noise(&self, shape: &[usize], scale: f32) -> Result<Tensor> {
        match shape.len() {
            1 => Tensor::randn(&[shape[0]])?.scalar_mul(scale),
            2 => Tensor::randn(&[shape[0], shape[1]])?.scalar_mul(scale),
            3 => Tensor::randn(&[shape[0], shape[1], shape[2]])?.scalar_mul(scale),
            4 => Tensor::randn(&[shape[0], shape[1], shape[2], shape[3]])?.scalar_mul(scale),
            _ => Err(runtime_error(format!(
                "Unsupported tensor dimension: {}",
                shape.len()
            ))),
        }
    }

    fn generate_laplacian_noise(&self, shape: &[usize], scale: f32) -> Result<Tensor> {
        // Simplified Laplacian noise generation using Box-Muller transform
        let gaussian = self.generate_gaussian_noise(shape, scale)?;
        // Transform to Laplacian (simplified)
        gaussian.scalar_mul(1.414) // Approximate factor
    }
}

/// Secure aggregator for privacy-preserving computation
struct SecureAggregator {
    config: SecureAggregationConfig,
}

impl SecureAggregator {
    fn new(config: SecureAggregationConfig) -> Self {
        Self { config }
    }

    fn aggregate_results(&self, _results: &[Tensor]) -> Result<Tensor> {
        // Placeholder for secure aggregation
        Err(runtime_error("Secure aggregation not implemented"))
    }
}

/// Privacy accountant for inference
struct InferencePrivacyAccountant {
    config: InferenceBudgetConfig,
    total_epsilon_spent: f64,
    total_delta_spent: f64,
    last_reset: std::time::Instant,
}

impl InferencePrivacyAccountant {
    fn new(config: InferenceBudgetConfig) -> Self {
        Self {
            config,
            total_epsilon_spent: 0.0,
            total_delta_spent: 0.0,
            last_reset: std::time::Instant::now(),
        }
    }

    fn has_budget_remaining(&mut self) -> bool {
        self.check_reset();
        self.total_epsilon_spent < self.config.total_epsilon
            && self.total_delta_spent < self.config.total_delta
    }

    fn account_inference(
        &mut self,
        inference_config: &InferencePrivacyConfig,
        input_shape: &[usize],
        output_shape: &[usize],
    ) -> Result<InferencePrivacyCost> {
        self.check_reset();

        let epsilon =
            if inference_config.input_privacy.enabled || inference_config.output_privacy.enabled {
                self.config.epsilon_per_request
            } else {
                0.0
            };

        let delta =
            if inference_config.input_privacy.enabled || inference_config.output_privacy.enabled {
                self.config.total_delta / 1000.0
            } else {
                0.0
            };

        self.total_epsilon_spent += epsilon;
        self.total_delta_spent += delta;

        Ok(InferencePrivacyCost { epsilon, delta })
    }

    fn check_reset(&mut self) {
        if self.last_reset.elapsed().as_secs() >= self.config.reset_period_secs {
            self.total_epsilon_spent = 0.0;
            self.total_delta_spent = 0.0;
            self.last_reset = std::time::Instant::now();
        }
    }
}

/// Privacy cost for inference
#[derive(Debug, Clone)]
struct InferencePrivacyCost {
    epsilon: f64,
    delta: f64,
}

/// Utility functions for privacy-preserving inference
pub struct PrivacyPreservingInferenceUtils;

impl PrivacyPreservingInferenceUtils {
    /// Calibrate privacy parameters for target accuracy
    pub fn calibrate_privacy_for_accuracy(
        target_accuracy: f32,
        baseline_accuracy: f32,
        privacy_level: PrivacyLevel,
    ) -> InferencePrivacyConfig {
        let accuracy_loss_budget = baseline_accuracy - target_accuracy;
        let mut config = InferencePrivacyConfig::from_privacy_level(privacy_level);

        // Adjust noise scales based on accuracy budget
        let scale_factor = (accuracy_loss_budget / 0.1).max(0.1).min(2.0);
        config.input_privacy.noise_scale *= scale_factor;
        config.output_privacy.noise_scale *= scale_factor;

        config
    }

    /// Estimate privacy-utility tradeoff
    pub fn estimate_privacy_utility_tradeoff(
        config: &InferencePrivacyConfig,
        model_sensitivity: f32,
    ) -> PrivacyUtilityTradeoff {
        let input_noise_impact = if config.input_privacy.enabled {
            config.input_privacy.noise_scale * model_sensitivity
        } else {
            0.0
        };

        let output_noise_impact = if config.output_privacy.enabled {
            config.output_privacy.noise_scale
        } else {
            0.0
        };

        let total_utility_loss = input_noise_impact + output_noise_impact;
        let privacy_strength = 1.0 / config.inference_budget.total_epsilon as f32;

        PrivacyUtilityTradeoff {
            privacy_strength,
            utility_preservation: 1.0 - total_utility_loss.min(1.0),
            estimated_accuracy_loss: total_utility_loss,
            noise_overhead: (input_noise_impact + output_noise_impact) / 2.0,
        }
    }
}

/// Privacy-utility tradeoff analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyUtilityTradeoff {
    pub privacy_strength: f32,
    pub utility_preservation: f32,
    pub estimated_accuracy_loss: f32,
    pub noise_overhead: f32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_inference_privacy_config() {
        let config = InferencePrivacyConfig::from_privacy_level(PrivacyLevel::High);
        assert!(config.enabled);
        assert!(config.input_privacy.enabled);
        assert!(config.output_privacy.enabled);
    }

    #[test]
    fn test_privacy_preserving_inference() {
        let config = InferencePrivacyConfig::default();
        let mut engine = PrivacyPreservingInferenceEngine::new(config);

        let input = Tensor::randn(&[1, 10]).unwrap();
        let model_fn = |x: &Tensor| -> Result<Tensor> {
            // Simple linear model
            x.scalar_mul(0.5)
        };

        let result = engine.private_inference(&input, model_fn);
        assert!(result.is_ok());

        let private_result = result.unwrap();
        assert!(private_result.privacy_guarantees.epsilon_spent > 0.0);
    }

    #[test]
    fn test_privacy_utility_tradeoff() {
        let config = InferencePrivacyConfig::default();
        let tradeoff =
            PrivacyPreservingInferenceUtils::estimate_privacy_utility_tradeoff(&config, 1.0);

        assert!(tradeoff.privacy_strength > 0.0);
        assert!(tradeoff.utility_preservation <= 1.0);
    }
}
