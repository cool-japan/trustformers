//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;
use crate::interpretability::attention::AttentionAnalysisResult;
use crate::interpretability::attribution::FeatureAttributionResult;
use crate::interpretability::config::InterpretabilityConfig;
use crate::interpretability::counterfactual::CounterfactualResult;
use crate::interpretability::lime::{
    FeatureImportance, LimeAnalysisResult, NeighborhoodStats, PerturbationAnalysis,
    PerturbationResult,
};
use crate::interpretability::report::InterpretabilityReport;
use crate::interpretability::shap::{FeatureContribution, ShapAnalysisResult, ShapSummary};
use anyhow::Result;
use chrono::Utc;
use scirs2_core::random::*;
use std::collections::HashMap;
