//! LIME (Local Interpretable Model-agnostic Explanations) analysis
//!
//! This module implements LIME analysis for local model interpretability,
//! providing local explanations of model predictions through perturbation analysis.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// LIME (Local Interpretable Model-agnostic Explanations) analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LimeAnalysisResult {
    /// Analysis timestamp
    pub timestamp: DateTime<Utc>,
    /// Local model coefficients
    pub local_coefficients: HashMap<String, f64>,
    /// Feature names
    pub feature_names: Vec<String>,
    /// Local model R-squared
    pub local_r_squared: f64,
    /// Local model intercept
    pub intercept: f64,
    /// Feature importance scores
    pub feature_importance: Vec<FeatureImportance>,
    /// Perturbation analysis
    pub perturbation_analysis: PerturbationAnalysis,
    /// Local neighborhood statistics
    pub neighborhood_stats: NeighborhoodStats,
    /// Model fidelity in local region
    pub local_fidelity: f64,
}

/// Feature importance from LIME
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FeatureImportance {
    /// Feature name
    pub feature_name: String,
    /// Importance score
    pub importance_score: f64,
    /// Confidence interval
    pub confidence_interval: (f64, f64),
    /// Statistical significance
    pub p_value: f64,
    /// Feature stability across perturbations
    pub stability: f64,
}

/// Perturbation analysis details
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationAnalysis {
    /// Number of perturbations generated
    pub num_perturbations: usize,
    /// Perturbation strategy used
    pub strategy: String,
    /// Average prediction variance
    pub prediction_variance: f64,
    /// Neighborhood coverage
    pub neighborhood_coverage: f64,
    /// Most influential perturbations
    pub influential_perturbations: Vec<PerturbationResult>,
}

/// Individual perturbation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerturbationResult {
    /// Perturbation ID
    pub id: String,
    /// Features that were perturbed
    pub perturbed_features: Vec<String>,
    /// Original prediction
    pub original_prediction: f64,
    /// Perturbed prediction
    pub perturbed_prediction: f64,
    /// Prediction change
    pub prediction_change: f64,
    /// Distance from original instance
    pub distance: f64,
}

/// Local neighborhood statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeighborhoodStats {
    /// Mean prediction in neighborhood
    pub mean_prediction: f64,
    /// Standard deviation of predictions
    pub std_prediction: f64,
    /// Neighborhood density
    pub density: f64,
    /// Feature correlation matrix in neighborhood
    pub correlation_matrix: HashMap<(String, String), f64>,
}