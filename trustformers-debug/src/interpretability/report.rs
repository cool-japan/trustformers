//! Interpretability reporting functionality
//!
//! This module provides comprehensive reporting capabilities for interpretability analyses.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::config::InterpretabilityConfig;
use super::shap::ShapAnalysisResult;
use super::lime::LimeAnalysisResult;
use super::attention::AttentionAnalysisResult;
use super::attribution::FeatureAttributionResult;
use super::counterfactual::CounterfactualResult;

/// Comprehensive interpretability report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterpretabilityReport {
    pub timestamp: DateTime<Utc>,
    pub config: InterpretabilityConfig,
    pub shap_analyses_count: usize,
    pub lime_analyses_count: usize,
    pub attention_analyses_count: usize,
    pub attribution_analyses_count: usize,
    pub counterfactual_analyses_count: usize,
    pub recent_shap_results: Vec<ShapAnalysisResult>,
    pub recent_lime_results: Vec<LimeAnalysisResult>,
    pub recent_attention_results: Vec<AttentionAnalysisResult>,
    pub recent_attribution_results: Vec<FeatureAttributionResult>,
    pub recent_counterfactual_results: Vec<CounterfactualResult>,
    pub interpretability_summary: HashMap<String, String>,
}