//! Advanced Differential Privacy Mechanisms
//!
//! This module implements state-of-the-art differential privacy mechanisms
//! for federated learning, including various noise distributions, composition
//! methods, and privacy amplification techniques.

use serde::{Deserialize, Serialize};

/// Advanced differential privacy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdvancedPrivacyConfig {
    /// Privacy budget (epsilon)
    pub epsilon: f64,
    /// Privacy budget (delta)
    pub delta: f64,
    /// Privacy mechanism type
    pub mechanism: PrivacyMechanism,
    /// Local vs central differential privacy
    pub privacy_model: PrivacyModel,
    /// Advanced composition method
    pub composition_method: CompositionMethod,
    /// Adaptive privacy budgeting
    pub adaptive_budgeting: bool,
    /// Privacy amplification settings
    pub amplification_config: PrivacyAmplificationConfig,
    /// Noise distribution parameters
    pub noise_config: NoiseDistributionConfig,
}

/// Privacy mechanisms for differential privacy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyMechanism {
    /// Gaussian mechanism
    Gaussian,
    /// Laplace mechanism
    Laplace,
    /// Exponential mechanism
    Exponential,
    /// Above threshold mechanism
    AboveThreshold,
    /// Sparse vector technique
    SparseVector,
    /// Private aggregation of teacher ensembles (PATE)
    PATE,
    /// Renyi differential privacy
    RenyiDP,
    /// Zero-concentrated differential privacy
    ZCDP,
}

/// Privacy models
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyModel {
    /// Local differential privacy
    LocalDP,
    /// Central differential privacy
    CentralDP,
    /// Shuffled model
    ShuffledModel,
    /// Hybrid model
    HybridModel,
}

/// Advanced composition methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum CompositionMethod {
    /// Basic composition
    Basic,
    /// Advanced composition
    Advanced,
    /// Optimal composition
    Optimal,
    /// Renyi differential privacy composition
    RenyiComposition,
    /// Zero-concentrated differential privacy composition
    ZCDPComposition,
    /// Privacy loss distribution tracking
    PLDTracking,
}

/// Privacy amplification configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAmplificationConfig {
    /// Subsampling ratio
    pub subsampling_ratio: f64,
    /// Shuffling enabled
    pub shuffling_enabled: bool,
    /// Secure aggregation amplification
    pub secure_aggregation_amplification: bool,
    /// Random sampling method
    pub sampling_method: SamplingMethod,
}

/// Sampling methods for privacy amplification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SamplingMethod {
    /// Uniform random sampling
    UniformRandom,
    /// Poisson sampling
    Poisson,
    /// Systematic sampling
    Systematic,
    /// Stratified sampling
    Stratified,
}

/// Noise distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoiseDistributionConfig {
    /// Base noise multiplier
    pub noise_multiplier: f64,
    /// Clipping norm for gradient clipping
    pub clipping_norm: f64,
    /// Adaptive clipping enabled
    pub adaptive_clipping: bool,
    /// Per-layer noise scaling
    pub per_layer_scaling: bool,
    /// Correlated noise for efficiency
    pub correlated_noise: bool,
}

/// Privacy accounting configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrivacyAccountingConfig {
    /// Privacy accounting method
    pub accounting_method: PrivacyAccountingMethod,
    /// Maximum epsilon allowed
    pub max_epsilon: f64,
    /// Maximum delta allowed
    pub max_delta: f64,
    /// Privacy budget tracking granularity
    pub tracking_granularity: TrackingGranularity,
    /// Composition tracking enabled
    pub composition_tracking: bool,
    /// Privacy loss distribution parameters
    pub pld_params: PLDParameters,
}

/// Privacy accounting methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PrivacyAccountingMethod {
    /// Basic composition accounting
    BasicComposition,
    /// Moments accountant
    MomentsAccountant,
    /// Renyi differential privacy accountant
    RenyiAccountant,
    /// Privacy loss distribution
    PrivacyLossDistribution,
    /// Zero-concentrated DP accountant
    ZCDPAccountant,
}

/// Tracking granularity for privacy accounting
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TrackingGranularity {
    /// Per-round tracking
    PerRound,
    /// Per-participant tracking
    PerParticipant,
    /// Per-query tracking
    PerQuery,
    /// Global tracking
    Global,
}

/// Privacy Loss Distribution parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PLDParameters {
    /// Discretization interval
    pub discretization_interval: f64,
    /// Truncation bound for PLD
    pub truncation_bound: f64,
    /// Pessimistic estimate factor
    pub pessimistic_estimate: bool,
    /// Tail bound approximation
    pub tail_bound_approximation: bool,
}

impl Default for AdvancedPrivacyConfig {
    fn default() -> Self {
        Self {
            epsilon: 1.0,
            delta: 1e-5,
            mechanism: PrivacyMechanism::Gaussian,
            privacy_model: PrivacyModel::CentralDP,
            composition_method: CompositionMethod::Advanced,
            adaptive_budgeting: true,
            amplification_config: PrivacyAmplificationConfig::default(),
            noise_config: NoiseDistributionConfig::default(),
        }
    }
}

impl Default for PrivacyAmplificationConfig {
    fn default() -> Self {
        Self {
            subsampling_ratio: 0.01,
            shuffling_enabled: true,
            secure_aggregation_amplification: true,
            sampling_method: SamplingMethod::UniformRandom,
        }
    }
}

impl Default for NoiseDistributionConfig {
    fn default() -> Self {
        Self {
            noise_multiplier: 1.0,
            clipping_norm: 1.0,
            adaptive_clipping: true,
            per_layer_scaling: false,
            correlated_noise: false,
        }
    }
}

impl Default for PrivacyAccountingConfig {
    fn default() -> Self {
        Self {
            accounting_method: PrivacyAccountingMethod::MomentsAccountant,
            max_epsilon: 10.0,
            max_delta: 1e-5,
            tracking_granularity: TrackingGranularity::PerRound,
            composition_tracking: true,
            pld_params: PLDParameters::default(),
        }
    }
}

impl Default for PLDParameters {
    fn default() -> Self {
        Self {
            discretization_interval: 1e-3,
            truncation_bound: 1e2,
            pessimistic_estimate: false,
            tail_bound_approximation: true,
        }
    }
}

/// Privacy budget tracker for monitoring epsilon/delta consumption
#[derive(Debug, Clone)]
pub struct PrivacyBudgetTracker {
    /// Current epsilon consumed
    pub epsilon_consumed: f64,
    /// Current delta consumed
    pub delta_consumed: f64,
    /// Maximum allowed epsilon
    pub max_epsilon: f64,
    /// Maximum allowed delta
    pub max_delta: f64,
    /// Privacy accounting method
    pub accounting_method: PrivacyAccountingMethod,
    /// Round-by-round epsilon tracking
    pub epsilon_history: Vec<f64>,
    /// Round-by-round delta tracking
    pub delta_history: Vec<f64>,
}

impl PrivacyBudgetTracker {
    /// Create new privacy budget tracker
    pub fn new(max_epsilon: f64, max_delta: f64, accounting_method: PrivacyAccountingMethod) -> Self {
        Self {
            epsilon_consumed: 0.0,
            delta_consumed: 0.0,
            max_epsilon,
            max_delta,
            accounting_method,
            epsilon_history: Vec::new(),
            delta_history: Vec::new(),
        }
    }

    /// Check if privacy budget allows for additional consumption
    pub fn can_consume(&self, epsilon: f64, delta: f64) -> bool {
        self.epsilon_consumed + epsilon <= self.max_epsilon &&
        self.delta_consumed + delta <= self.max_delta
    }

    /// Consume privacy budget
    pub fn consume(&mut self, epsilon: f64, delta: f64) -> Result<(), String> {
        if !self.can_consume(epsilon, delta) {
            return Err("Privacy budget exceeded".to_string());
        }

        self.epsilon_consumed += epsilon;
        self.delta_consumed += delta;
        self.epsilon_history.push(epsilon);
        self.delta_history.push(delta);

        Ok(())
    }

    /// Get remaining privacy budget
    pub fn remaining_budget(&self) -> (f64, f64) {
        (
            self.max_epsilon - self.epsilon_consumed,
            self.max_delta - self.delta_consumed,
        )
    }

    /// Reset privacy budget
    pub fn reset(&mut self) {
        self.epsilon_consumed = 0.0;
        self.delta_consumed = 0.0;
        self.epsilon_history.clear();
        self.delta_history.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_privacy_config_default() {
        let config = AdvancedPrivacyConfig::default();
        assert_eq!(config.epsilon, 1.0);
        assert_eq!(config.delta, 1e-5);
        assert_eq!(config.mechanism, PrivacyMechanism::Gaussian);
        assert_eq!(config.privacy_model, PrivacyModel::CentralDP);
    }

    #[test]
    fn test_privacy_budget_tracker() {
        let mut tracker = PrivacyBudgetTracker::new(2.0, 1e-5, PrivacyAccountingMethod::BasicComposition);

        assert!(tracker.can_consume(1.0, 5e-6));
        assert!(tracker.consume(1.0, 5e-6).is_ok());

        assert_eq!(tracker.epsilon_consumed, 1.0);
        assert_eq!(tracker.delta_consumed, 5e-6);

        // Should not be able to consume more than budget
        assert!(!tracker.can_consume(1.5, 1e-5));
        assert!(tracker.consume(1.5, 1e-5).is_err());

        // Should be able to consume within remaining budget
        assert!(tracker.can_consume(1.0, 5e-6));
        assert!(tracker.consume(1.0, 5e-6).is_ok());

        let (remaining_eps, remaining_delta) = tracker.remaining_budget();
        assert_eq!(remaining_eps, 0.0);
        assert_eq!(remaining_delta, 0.0);
    }

    #[test]
    fn test_privacy_mechanism_serialization() {
        let mechanism = PrivacyMechanism::Gaussian;
        let serialized = serde_json::to_string(&mechanism).expect("Operation failed");
        let deserialized: PrivacyMechanism = serde_json::from_str(&serialized).expect("Operation failed");
        assert_eq!(mechanism, deserialized);
    }

    #[test]
    fn test_privacy_amplification_config() {
        let config = PrivacyAmplificationConfig::default();
        assert_eq!(config.subsampling_ratio, 0.01);
        assert!(config.shuffling_enabled);
        assert!(config.secure_aggregation_amplification);
        assert_eq!(config.sampling_method, SamplingMethod::UniformRandom);
    }

    #[test]
    fn test_composition_methods() {
        let methods = [
            CompositionMethod::Basic,
            CompositionMethod::Advanced,
            CompositionMethod::Optimal,
            CompositionMethod::RenyiComposition,
            CompositionMethod::ZCDPComposition,
            CompositionMethod::PLDTracking,
        ];

        for method in &methods {
            let serialized = serde_json::to_string(method).expect("Operation failed");
            let deserialized: CompositionMethod = serde_json::from_str(&serialized).expect("Operation failed");
            assert_eq!(*method, deserialized);
        }
    }
}