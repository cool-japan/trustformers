//! Core Types and Configuration for Simulation Tools
//!
//! This module provides the fundamental types, enums, and configuration structures
//! used throughout the simulation tools system for what-if analysis, perturbation testing,
//! adversarial probing, and edge case discovery.

use serde::{Deserialize, Serialize};

/// Configuration for simulation tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Enable what-if analysis
    pub enable_what_if_analysis: bool,
    /// Enable perturbation testing
    pub enable_perturbation_testing: bool,
    /// Enable adversarial probing
    pub enable_adversarial_probing: bool,
    /// Enable robustness testing
    pub enable_robustness_testing: bool,
    /// Enable edge case discovery
    pub enable_edge_case_discovery: bool,
    /// Number of what-if scenarios to generate
    pub num_what_if_scenarios: usize,
    /// Number of perturbation samples
    pub num_perturbation_samples: usize,
    /// Number of adversarial examples to generate
    pub num_adversarial_examples: usize,
    /// Robustness test iterations
    pub robustness_test_iterations: usize,
    /// Edge case search depth
    pub edge_case_search_depth: usize,
    /// Perturbation intensity levels
    pub perturbation_intensities: Vec<f64>,
    /// Adversarial attack methods
    pub adversarial_methods: Vec<AdversarialMethod>,
    /// Robustness test types
    pub robustness_test_types: Vec<RobustnessTestType>,
}

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            enable_what_if_analysis: true,
            enable_perturbation_testing: true,
            enable_adversarial_probing: true,
            enable_robustness_testing: true,
            enable_edge_case_discovery: true,
            num_what_if_scenarios: 50,
            num_perturbation_samples: 1000,
            num_adversarial_examples: 100,
            robustness_test_iterations: 500,
            edge_case_search_depth: 10,
            perturbation_intensities: vec![0.01, 0.05, 0.1, 0.2, 0.5],
            adversarial_methods: vec![
                AdversarialMethod::FGSM,
                AdversarialMethod::PGD,
                AdversarialMethod::CW,
                AdversarialMethod::DeepFool,
            ],
            robustness_test_types: vec![
                RobustnessTestType::NoiseRobustness,
                RobustnessTestType::InputVariation,
                RobustnessTestType::FeatureDropout,
                RobustnessTestType::DistributionShift,
            ],
        }
    }
}

/// Adversarial attack methods
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum AdversarialMethod {
    /// Fast Gradient Sign Method
    FGSM,
    /// Projected Gradient Descent
    PGD,
    /// Carlini & Wagner attack
    CW,
    /// DeepFool attack
    DeepFool,
    /// Universal Adversarial Perturbations
    UAP,
    /// Boundary attack
    Boundary,
}

/// Types of robustness tests
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustnessTestType {
    /// Gaussian noise robustness
    NoiseRobustness,
    /// Input variation robustness
    InputVariation,
    /// Feature dropout robustness
    FeatureDropout,
    /// Distribution shift robustness
    DistributionShift,
    /// Occlusion robustness
    Occlusion,
    /// Transformation robustness
    Transformation,
}

/// Direction of feature change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDirection {
    Increase,
    Decrease,
    Categorical,
}

/// Type of feature change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeType {
    /// Small incremental change
    Incremental,
    /// Large significant change
    Significant,
    /// Complete value replacement
    Replacement,
    /// Outlier value
    Outlier,
}

/// Type of feature interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Synergistic interaction
    Synergistic,
    /// Antagonistic interaction
    Antagonistic,
    /// Conditional interaction
    Conditional,
    /// Independent
    Independent,
}

/// Feasibility of implementing changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationFeasibility {
    /// Easily implementable
    Easy,
    /// Moderately difficult
    Moderate,
    /// Difficult to implement
    Difficult,
    /// Practically impossible
    Impossible,
}

/// Classification of boundary complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComplexityClass {
    Linear,
    Quadratic,
    Polynomial,
    HighlyNonlinear,
    Chaotic,
}

/// Classification of model robustness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RobustnessClass {
    /// Very robust to perturbations
    VeryRobust,
    /// Moderately robust
    Robust,
    /// Somewhat robust
    SomewhatRobust,
    /// Sensitive to perturbations
    Sensitive,
    /// Very sensitive/fragile
    Fragile,
}

/// Type of sensitivity hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HotspotType {
    /// Local sensitivity around a point
    Local,
    /// Global sensitivity across regions
    Global,
    /// Feature-specific sensitivity
    FeatureSpecific,
    /// Boundary sensitivity
    Boundary,
}

/// Type of triggering condition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConditionType {
    /// Feature value exceeds threshold
    Exceeds,
    /// Feature value below threshold
    Below,
    /// Feature value in range
    InRange,
    /// Feature value equals specific value
    Equals,
    /// Combination of features
    Combination,
}

/// Severity of failure
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum FailureSeverity {
    /// Minor prediction error
    Minor,
    /// Moderate prediction error
    Moderate,
    /// Major prediction error
    Major,
    /// Critical system failure
    Critical,
}

/// Cost of implementing mitigation strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationCost {
    Low,
    Medium,
    High,
    VeryHigh,
}

/// Level of attack sophistication required
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SophisticationLevel {
    Basic,
    Intermediate,
    Advanced,
    Expert,
}

/// Type of robustness guarantee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GuaranteeType {
    /// L-infinity norm guarantee
    LInfinity,
    /// L2 norm guarantee
    L2,
    /// Feature-wise guarantee
    FeatureWise,
    /// Semantic guarantee
    Semantic,
}

/// Complexity of defense implementation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DefenseComplexity {
    Simple,
    Moderate,
    Complex,
    VeryComplex,
}

/// Impact on model performance
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PerformanceImpact {
    Negligible,
    Low,
    Medium,
    High,
}

/// Type of edge case
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeCaseType {
    /// Input at distribution boundaries
    DistributionBoundary,
    /// Outlier input
    Outlier,
    /// Corner case in feature space
    CornerCase,
    /// Adversarial-like input
    AdversarialLike,
    /// Systematic bias case
    SystematicBias,
    /// Model confusion case
    ModelConfusion,
}

/// Severity of edge case
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum EdgeCaseSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Impact of systematic issue
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SystematicIssueImpact {
    Localized,
    Regional,
    Global,
    Critical,
}
