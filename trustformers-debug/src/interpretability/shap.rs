//! SHAP (SHapley Additive exPlanations) analysis
//!
//! This module implements SHAP analysis for model interpretability, providing
//! feature importance and contribution analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SHAP (SHapley Additive exPlanations) analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// SHAP values for each feature
    pub shap_values: HashMap<String, f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Base value (expected model output)
    pub base_value: f64,
    /// Model prediction
    pub prediction: f64,
    /// Feature contributions (sorted by importance)
    pub feature_contributions: Vec<FeatureContribution>,
    /// Top positive features
    pub top_positive_features: Vec<FeatureContribution>,
    /// Top negative features
    pub top_negative_features: Vec<FeatureContribution>,
    /// SHAP interaction values (if computed)
    pub interaction_values: Option<HashMap<(String, String), f64>>,
    /// Summary statistics
    pub summary: ShapSummary,
}

/// Individual feature contribution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureContribution {
    /// Feature name
    pub feature_name: String,
    /// SHAP value (contribution to prediction)
    pub shap_value: f64,
    /// Feature value
    pub feature_value: f64,
    /// Absolute importance rank
    pub importance_rank: usize,
    /// Contribution percentage
    pub contribution_percentage: f64,
}

/// SHAP analysis summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ShapSummary {
    /// Total positive contribution
    pub total_positive_contribution: f64,
    /// Total negative contribution
    pub total_negative_contribution: f64,
    /// Number of important features (above threshold)
    pub num_important_features: usize,
    /// Feature importance distribution
    pub importance_distribution: HashMap<String, f64>,
    /// Model explanation completeness (0.0 to 1.0)
    pub explanation_completeness: f64,
}