//! Counterfactual generation and analysis
//!
//! This module provides counterfactual explanation capabilities, helping to understand
//! what changes would be needed to achieve different model predictions.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Counterfactual generation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Generated counterfactuals
    pub counterfactuals: Vec<Counterfactual>,
    /// Counterfactual quality metrics
    pub quality_metrics: CounterfactualQualityMetrics,
    /// Feature sensitivity analysis
    pub feature_sensitivity: FeatureSensitivityAnalysis,
    /// Decision boundary analysis
    pub decision_boundary: DecisionBoundaryAnalysis,
    /// Actionable insights
    pub actionable_insights: Vec<ActionableInsight>,
}

/// Individual counterfactual example
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Counterfactual {
    /// Counterfactual ID
    pub id: String,
    /// Original instance
    pub original_instance: HashMap<String, f64>,
    /// Counterfactual instance
    pub counterfactual_instance: HashMap<String, f64>,
    /// Changed features
    pub changed_features: Vec<FeatureChange>,
    /// Original prediction
    pub original_prediction: f64,
    /// Counterfactual prediction
    pub counterfactual_prediction: f64,
    /// Distance from original
    pub distance: f64,
    /// Plausibility score
    pub plausibility: f64,
    /// Actionability score
    pub actionability: f64,
}

/// Change made to a feature in counterfactual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureChange {
    /// Feature name
    pub feature_name: String,
    /// Original value
    pub original_value: f64,
    /// New value
    pub new_value: f64,
    /// Change magnitude
    pub change_magnitude: f64,
    /// Change direction
    pub change_direction: ChangeDirection,
    /// Change difficulty/cost
    pub change_cost: f64,
}

/// Direction of feature change
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ChangeDirection {
    Increase,
    Decrease,
    Categorical,
}

/// Quality metrics for counterfactuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualQualityMetrics {
    /// Average distance to original instances
    pub avg_distance: f64,
    /// Average plausibility score
    pub avg_plausibility: f64,
    /// Average actionability score
    pub avg_actionability: f64,
    /// Diversity among counterfactuals
    pub diversity: f64,
    /// Coverage of feature space
    pub coverage: f64,
    /// Sparsity (average number of changed features)
    pub sparsity: f64,
}

/// Feature sensitivity analysis from counterfactuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSensitivityAnalysis {
    /// Sensitivity scores for each feature
    pub feature_sensitivities: HashMap<String, f64>,
    /// Most sensitive features
    pub most_sensitive: Vec<String>,
    /// Least sensitive features
    pub least_sensitive: Vec<String>,
    /// Feature interaction effects
    pub interaction_effects: Vec<InteractionEffect>,
    /// Threshold analysis
    pub threshold_analysis: ThresholdAnalysis,
}

/// Effect of feature interactions on sensitivity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InteractionEffect {
    /// Features involved in interaction
    pub features: Vec<String>,
    /// Interaction effect magnitude
    pub effect_magnitude: f64,
    /// Effect type
    pub effect_type: InteractionEffectType,
}

/// Type of interaction effect
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionEffectType {
    /// Synergistic (combined effect > sum of individual effects)
    Synergistic,
    /// Antagonistic (combined effect < sum of individual effects)
    Antagonistic,
    /// Independent (combined effect = sum of individual effects)
    Independent,
}

/// Analysis of decision thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThresholdAnalysis {
    /// Feature thresholds for decision changes
    pub feature_thresholds: HashMap<String, f64>,
    /// Threshold confidence intervals
    pub threshold_confidence: HashMap<String, (f64, f64)>,
    /// Critical features (small changes cause big effects)
    pub critical_features: Vec<String>,
    /// Robust features (large changes needed for effects)
    pub robust_features: Vec<String>,
}

/// Decision boundary analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBoundaryAnalysis {
    /// Local boundary curvature
    pub boundary_curvature: f64,
    /// Boundary complexity
    pub boundary_complexity: f64,
    /// Distance to boundary
    pub distance_to_boundary: f64,
    /// Boundary crossing points
    pub crossing_points: Vec<BoundaryCrossingPoint>,
    /// Local linearity assessment
    pub local_linearity: f64,
}

/// Point where instance crosses decision boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCrossingPoint {
    /// Point coordinates in feature space
    pub coordinates: HashMap<String, f64>,
    /// Distance from original instance
    pub distance: f64,
    /// Prediction at crossing point
    pub boundary_prediction: f64,
    /// Crossing direction
    pub crossing_direction: Vec<f64>,
}

/// Actionable insight from counterfactual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ActionableInsight {
    /// Insight description
    pub description: String,
    /// Required feature changes
    pub required_changes: Vec<FeatureChange>,
    /// Expected outcome
    pub expected_outcome: f64,
    /// Confidence in insight
    pub confidence: f64,
    /// Implementation difficulty
    pub difficulty: ImplementationDifficulty,
    /// Time horizon for implementation
    pub time_horizon: TimeHorizon,
}

/// Difficulty of implementing suggested changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImplementationDifficulty {
    Easy,
    Moderate,
    Hard,
    VeryHard,
    Impossible,
}

/// Time horizon for implementing changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeHorizon {
    Immediate,
    ShortTerm,
    MediumTerm,
    LongTerm,
}