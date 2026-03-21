//! Configuration types for interpretability analysis
//!
//! This module contains the main configuration structures and enums used across
//! all interpretability analysis methods.

use serde::{Deserialize, Serialize};

/// Configuration for interpretability tools
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityConfig {
    /// Enable SHAP (SHapley Additive exPlanations) analysis
    pub enable_shap: bool,
    /// Enable LIME (Local Interpretable Model-agnostic Explanations) analysis
    pub enable_lime: bool,
    /// Enable attention analysis for transformer models
    pub enable_attention_analysis: bool,
    /// Enable feature attribution analysis
    pub enable_feature_attribution: bool,
    /// Enable counterfactual generation
    pub enable_counterfactual_generation: bool,
    /// Number of SHAP samples for approximation
    pub shap_samples: usize,
    /// Number of LIME perturbations
    pub lime_perturbations: usize,
    /// Maximum sequence length for attention analysis
    pub max_attention_seq_length: usize,
    /// Number of counterfactuals to generate
    pub num_counterfactuals: usize,
    /// Feature attribution methods to use
    pub attribution_methods: Vec<AttributionMethod>,
    /// Background dataset size for SHAP
    pub shap_background_size: usize,
}

impl Default for InterpretabilityConfig {
    fn default() -> Self {
        Self {
            enable_shap: true,
            enable_lime: true,
            enable_attention_analysis: true,
            enable_feature_attribution: true,
            enable_counterfactual_generation: true,
            shap_samples: 1000,
            lime_perturbations: 5000,
            max_attention_seq_length: 512,
            num_counterfactuals: 10,
            attribution_methods: vec![
                AttributionMethod::IntegratedGradients,
                AttributionMethod::GradientShap,
                AttributionMethod::DeepLift,
            ],
            shap_background_size: 100,
        }
    }
}

/// Attribution methods for feature importance
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub enum AttributionMethod {
    /// Integrated Gradients
    IntegratedGradients,
    /// Gradient × Input
    GradientInput,
    /// SmoothGrad
    SmoothGrad,
    /// Gradient SHAP
    GradientShap,
    /// DeepLIFT
    DeepLift,
    /// Layer-wise Relevance Propagation
    LRP,
    /// Guided Backpropagation
    GuidedBackprop,
    /// Grad-CAM (Gradient-weighted Class Activation Mapping)
    GradCAM,
    /// Grad-CAM++
    GradCAMPlusPlus,
    /// Score-CAM
    ScoreCAM,
    /// Expected Gradients
    ExpectedGradients,
    /// Attention Rollout
    AttentionRollout,
    /// Path Integrated Gradients
    PathIntegratedGradients,
}