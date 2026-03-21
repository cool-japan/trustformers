//! Adversarial Analysis and Attack Resistance Testing
//!
//! This module provides comprehensive adversarial analysis capabilities including
//! attack generation, robustness assessment, vulnerability analysis, and defense recommendations.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Adversarial probing result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialProbingResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Base input
    pub base_input: HashMap<String, f64>,
    /// Adversarial examples by method
    pub adversarial_examples: HashMap<AdversarialMethod, Vec<AdversarialExample>>,
    /// Attack success analysis
    pub attack_success_analysis: AttackSuccessAnalysis,
    /// Adversarial robustness assessment
    pub robustness_assessment: AdversarialRobustnessAssessment,
    /// Defense recommendations
    pub defense_recommendations: Vec<DefenseRecommendation>,
}

/// Individual adversarial example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialExample {
    /// Example ID
    pub id: String,
    /// Attack method used
    pub attack_method: AdversarialMethod,
    /// Original input
    pub original_input: HashMap<String, f64>,
    /// Adversarial input
    pub adversarial_input: HashMap<String, f64>,
    /// Original prediction
    pub original_prediction: f64,
    /// Adversarial prediction
    pub adversarial_prediction: f64,
    /// Perturbation vector
    pub perturbation: HashMap<String, f64>,
    /// Perturbation norm
    pub perturbation_norm: f64,
    /// Attack success
    pub is_successful: bool,
    /// Attack confidence
    pub confidence: f64,
}

/// Analysis of attack success rates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackSuccessAnalysis {
    /// Success rate by attack method
    pub success_rate_by_method: HashMap<AdversarialMethod, f64>,
    /// Overall success rate
    pub overall_success_rate: f64,
    /// Average perturbation magnitude
    pub avg_perturbation_magnitude: f64,
    /// Most effective attack methods
    pub most_effective_methods: Vec<AdversarialMethod>,
    /// Attack difficulty analysis
    pub attack_difficulty: AttackDifficultyAnalysis,
}

/// Analysis of attack difficulty
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttackDifficultyAnalysis {
    /// Easy targets (low perturbation needed)
    pub easy_targets: Vec<String>,
    /// Hard targets (high perturbation needed)
    pub hard_targets: Vec<String>,
    /// Average perturbation needed by feature
    pub perturbation_by_feature: HashMap<String, f64>,
    /// Attack complexity assessment
    pub complexity_assessment: ComplexityAssessment,
}

/// Assessment of attack complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComplexityAssessment {
    /// Complexity score
    pub complexity_score: f64,
    /// Number of features required for attack
    pub features_required: usize,
    /// Minimum perturbation magnitude
    pub min_perturbation: f64,
    /// Attack sophistication level
    pub sophistication_level: SophisticationLevel,
}

/// Assessment of adversarial robustness
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdversarialRobustnessAssessment {
    /// Overall robustness score
    pub robustness_score: f64,
    /// Robustness by attack type
    pub robustness_by_attack: HashMap<AdversarialMethod, f64>,
    /// Vulnerability hotspots
    pub vulnerability_hotspots: Vec<VulnerabilityHotspot>,
    /// Certified robustness analysis
    pub certified_robustness: CertifiedRobustnessAnalysis,
}

/// Vulnerability hotspot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VulnerabilityHotspot {
    /// Hotspot location
    pub location: HashMap<String, f64>,
    /// Vulnerability score
    pub vulnerability_score: f64,
    /// Susceptible attack methods
    pub susceptible_attacks: Vec<AdversarialMethod>,
    /// Hotspot radius
    pub radius: f64,
}

/// Certified robustness analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CertifiedRobustnessAnalysis {
    /// Certified radius
    pub certified_radius: f64,
    /// Certification confidence
    pub certification_confidence: f64,
    /// Certification method used
    pub certification_method: String,
    /// Robustness guarantees
    pub robustness_guarantees: Vec<RobustnessGuarantee>,
}

/// Robustness guarantee
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RobustnessGuarantee {
    /// Guarantee type
    pub guarantee_type: GuaranteeType,
    /// Guarantee strength
    pub strength: f64,
    /// Applicable conditions
    pub conditions: Vec<String>,
    /// Confidence level
    pub confidence: f64,
}

/// Defense recommendation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DefenseRecommendation {
    /// Defense name
    pub name: String,
    /// Defense description
    pub description: String,
    /// Target vulnerabilities
    pub target_vulnerabilities: Vec<String>,
    /// Expected effectiveness
    pub effectiveness: f64,
    /// Implementation complexity
    pub complexity: DefenseComplexity,
    /// Performance impact
    pub performance_impact: PerformanceImpact,
}
