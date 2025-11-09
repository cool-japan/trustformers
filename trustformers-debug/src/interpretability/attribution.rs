//! Feature attribution analysis
//!
//! This module provides feature attribution capabilities using various methods
//! like Integrated Gradients, GradCAM, SHAP, and others.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::AttributionMethod;

/// Feature attribution analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAttributionResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Attribution results by method
    pub attribution_by_method: HashMap<AttributionMethod, AttributionMethodResult>,
    /// Consensus attribution (averaged across methods)
    pub consensus_attribution: Vec<FeatureAttribution>,
    /// Attribution agreement analysis
    pub method_agreement: MethodAgreementAnalysis,
    /// Top attributed features
    pub top_features: Vec<TopFeature>,
    /// Attribution visualization data
    pub visualization_data: AttributionVisualizationData,
}

/// Attribution result for a specific method
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionMethodResult {
    /// Attribution method used
    pub method: AttributionMethod,
    /// Feature attributions
    pub attributions: Vec<FeatureAttribution>,
    /// Method-specific parameters
    pub method_parameters: HashMap<String, f64>,
    /// Method reliability score
    pub reliability_score: f64,
    /// Computation time (milliseconds)
    pub computation_time_ms: f64,
}

/// Individual feature attribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureAttribution {
    /// Feature identifier
    pub feature_id: String,
    /// Feature name
    pub feature_name: String,
    /// Attribution value
    pub attribution_value: f64,
    /// Attribution confidence
    pub confidence: f64,
    /// Feature value
    pub feature_value: f64,
    /// Normalized attribution (0-1)
    pub normalized_attribution: f64,
}

/// Analysis of agreement between attribution methods
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodAgreementAnalysis {
    /// Pairwise correlation between methods
    pub method_correlations: HashMap<(AttributionMethod, AttributionMethod), f64>,
    /// Overall agreement score
    pub overall_agreement: f64,
    /// Most consistent features across methods
    pub consistent_features: Vec<String>,
    /// Most divergent features across methods
    pub divergent_features: Vec<String>,
    /// Method reliability ranking
    pub method_reliability: Vec<(AttributionMethod, f64)>,
}

/// Top attributed feature with additional information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopFeature {
    /// Feature information
    pub feature: FeatureAttribution,
    /// Rank across all methods
    pub overall_rank: usize,
    /// Method-specific ranks
    pub method_ranks: HashMap<AttributionMethod, usize>,
    /// Feature stability score
    pub stability: f64,
    /// Interpretation guidance
    pub interpretation: String,
}

/// Data for visualizing attributions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionVisualizationData {
    /// Attribution heatmap data
    pub heatmap_data: Vec<Vec<f64>>,
    /// Feature names for heatmap
    pub feature_names: Vec<String>,
    /// Method names for heatmap
    pub method_names: Vec<String>,
    /// Attribution timeline (if applicable)
    pub timeline_data: Option<Vec<TimelinePoint>>,
    /// Feature interaction data
    pub interaction_data: Vec<FeatureInteraction>,
}

/// Point in attribution timeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimelinePoint {
    /// Time point (sequence position, time step, etc.)
    pub time_point: usize,
    /// Feature attributions at this time point
    pub attributions: Vec<FeatureAttribution>,
}

/// Feature interaction information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureInteraction {
    /// First feature
    pub feature1: String,
    /// Second feature
    pub feature2: String,
    /// Interaction strength
    pub interaction_strength: f64,
    /// Interaction type
    pub interaction_type: InteractionType,
}

/// Type of feature interaction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    /// Positive interaction (features reinforce each other)
    Positive,
    /// Negative interaction (features oppose each other)
    Negative,
    /// Conditional interaction (effect depends on context)
    Conditional,
    /// Independent (no significant interaction)
    Independent,
}