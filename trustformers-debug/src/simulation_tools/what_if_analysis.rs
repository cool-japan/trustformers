//! What-If Analysis for Model Behavior Exploration
//!
//! This module provides comprehensive what-if analysis capabilities including
//! scenario generation, feature sensitivity analysis, counterfactual insights,
//! and decision boundary exploration.

use super::types::*;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// What-if analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhatIfAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Base scenario (original input)
    pub base_scenario: Scenario,
    /// Generated what-if scenarios
    pub scenarios: Vec<Scenario>,
    /// Scenario impact analysis
    pub impact_analysis: ScenarioImpactAnalysis,
    /// Feature sensitivity analysis
    pub sensitivity_analysis: FeatureSensitivityAnalysis,
    /// Counterfactual insights
    pub counterfactual_insights: Vec<CounterfactualInsight>,
    /// Decision boundary exploration
    pub decision_boundary_exploration: DecisionBoundaryExploration,
}

/// Individual scenario for what-if analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    /// Scenario ID
    pub id: String,
    /// Scenario description
    pub description: String,
    /// Input features for this scenario
    pub features: HashMap<String, f64>,
    /// Model prediction for this scenario
    pub prediction: f64,
    /// Prediction confidence
    pub confidence: f64,
    /// Changed features from base scenario
    pub changed_features: Vec<FeatureChange>,
    /// Distance from base scenario
    pub distance_from_base: f64,
    /// Scenario plausibility
    pub plausibility: f64,
}

/// Change made to a feature in a scenario
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
    /// Change type
    pub change_type: ChangeType,
}

/// Analysis of scenario impacts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScenarioImpactAnalysis {
    /// Most impactful scenarios
    pub high_impact_scenarios: Vec<String>,
    /// Scenarios with prediction flips
    pub prediction_flip_scenarios: Vec<String>,
    /// Average prediction change
    pub avg_prediction_change: f64,
    /// Maximum prediction change
    pub max_prediction_change: f64,
    /// Prediction stability analysis
    pub stability_analysis: PredictionStabilityAnalysis,
    /// Feature importance ranking from scenarios
    pub feature_importance_ranking: Vec<FeatureImportanceRank>,
}

/// Analysis of prediction stability across scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionStabilityAnalysis {
    /// Stability score (0-1, higher is more stable)
    pub stability_score: f64,
    /// Prediction variance across scenarios
    pub prediction_variance: f64,
    /// Number of prediction flips
    pub prediction_flips: usize,
    /// Stability by feature change magnitude
    pub stability_by_magnitude: HashMap<String, f64>,
}

/// Feature importance rank from scenario analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportanceRank {
    /// Feature name
    pub feature_name: String,
    /// Importance score
    pub importance_score: f64,
    /// Rank (1 = most important)
    pub rank: usize,
    /// Average impact when changed
    pub avg_impact: f64,
    /// Number of scenarios where this feature was changed
    pub change_frequency: usize,
}

/// Feature sensitivity analysis from what-if scenarios
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureSensitivityAnalysis {
    /// Sensitivity scores for each feature
    pub feature_sensitivities: HashMap<String, f64>,
    /// Most sensitive features
    pub most_sensitive_features: Vec<String>,
    /// Least sensitive features
    pub least_sensitive_features: Vec<String>,
    /// Non-linear sensitivity detection
    pub non_linear_features: Vec<String>,
    /// Feature interaction sensitivities
    pub interaction_sensitivities: Vec<FeatureInteractionSensitivity>,
}

/// Sensitivity of feature interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInteractionSensitivity {
    /// First feature
    pub feature1: String,
    /// Second feature
    pub feature2: String,
    /// Interaction sensitivity score
    pub sensitivity_score: f64,
    /// Interaction type
    pub interaction_type: InteractionType,
}

/// Counterfactual insight from what-if analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CounterfactualInsight {
    /// Insight description
    pub description: String,
    /// Required feature changes for desired outcome
    pub required_changes: Vec<FeatureChange>,
    /// Predicted outcome
    pub predicted_outcome: f64,
    /// Confidence in prediction
    pub confidence: f64,
    /// Implementation feasibility
    pub feasibility: ImplementationFeasibility,
}

/// Decision boundary exploration from what-if analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionBoundaryExploration {
    /// Points near decision boundary
    pub boundary_points: Vec<BoundaryPoint>,
    /// Boundary complexity assessment
    pub boundary_complexity: BoundaryComplexity,
    /// Local linearity analysis
    pub local_linearity: LocalLinearityAnalysis,
    /// Boundary crossing analysis
    pub crossing_analysis: BoundaryCrossingAnalysis,
}

/// Point near decision boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryPoint {
    /// Point coordinates
    pub coordinates: HashMap<String, f64>,
    /// Distance to boundary
    pub distance_to_boundary: f64,
    /// Prediction at this point
    pub prediction: f64,
    /// Gradient direction
    pub gradient_direction: HashMap<String, f64>,
}

/// Assessment of decision boundary complexity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryComplexity {
    /// Complexity score
    pub complexity_score: f64,
    /// Boundary curvature
    pub curvature: f64,
    /// Number of inflection points
    pub inflection_points: usize,
    /// Complexity classification
    pub complexity_class: ComplexityClass,
}

/// Local linearity analysis of decision boundary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalLinearityAnalysis {
    /// Average local linearity score
    pub avg_linearity: f64,
    /// Linearity by region
    pub linearity_by_region: HashMap<String, f64>,
    /// Most linear regions
    pub most_linear_regions: Vec<String>,
    /// Most non-linear regions
    pub most_nonlinear_regions: Vec<String>,
}

/// Analysis of decision boundary crossings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryCrossingAnalysis {
    /// Number of boundary crossings found
    pub crossing_count: usize,
    /// Average crossing distance
    pub avg_crossing_distance: f64,
    /// Crossing directions
    pub crossing_directions: Vec<CrossingDirection>,
    /// Most common crossing features
    pub common_crossing_features: Vec<String>,
}

/// Direction of boundary crossing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrossingDirection {
    /// Direction vector
    pub direction: HashMap<String, f64>,
    /// Direction magnitude
    pub magnitude: f64,
    /// Frequency of this direction
    pub frequency: usize,
}
