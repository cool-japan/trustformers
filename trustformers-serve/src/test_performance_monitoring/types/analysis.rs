//! Analysis Type Definitions

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// Import types from sibling modules
use super::utilities::ImprovementOpportunity;

#[derive(Debug, Serialize, Deserialize)]
pub struct FlapDetection {
    pub enabled: bool,
    pub threshold: u32,
    pub window: Duration,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub success: bool,
    pub space_saved_bytes: u64,
    pub optimization_time_ms: f64,
    pub compression_savings: u64,
    pub storage_optimization: u64,
    pub retention_cleanup: u64,
    pub total_space_saved: u64,
    pub optimization_time: Duration,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrendAnalysis {
    pub trend_direction: String,
    pub trend_strength: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IoBottleneck {
    pub detected: bool,
    pub bottleneck_type: String,
    pub severity: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DriftDetection {
    pub enabled: bool,
    pub threshold: f64,
    pub window_size: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutlierAnalysis {
    pub outliers: Vec<Outlier>,
    pub method: String,
    pub threshold: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct Outlier {
    pub value: f64,
    pub score: f64,
    pub timestamp: DateTime<Utc>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RegressionAnalysis {
    pub coefficients: Vec<f64>,
    pub r_squared: f64,
    pub predictions: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TrendComponents {
    pub trend: Vec<f64>,
    pub seasonal: Vec<f64>,
    pub residual: Vec<f64>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BottleneckAnalysisResult {
    pub bottlenecks: Vec<String>,
    pub severity_scores: Vec<f64>,
    pub recommendations: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ResourceEfficiencyAnalysis {
    pub efficiency_score: f64,
    pub resource_usage: std::collections::HashMap<String, f64>,
    pub optimization_opportunities: Vec<ImprovementOpportunity>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CostBenefitAnalysis {
    pub benefits: f64,
    pub costs: f64,
    pub roi: f64,
    pub payback_period: Duration,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OptimizationRiskAssessment {
    pub risk_level: String,
    pub risk_factors: Vec<String>,
    pub mitigation_strategies: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RegressionAnalysisResult {
    pub analysis: RegressionAnalysis,
    pub model_quality: f64,
    pub metadata: std::collections::HashMap<String, String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ImprovementAnalysisResult {
    pub baseline: f64,
    pub current: f64,
    pub improvement_percentage: f64,
    pub opportunities: Vec<ImprovementOpportunity>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImpactAnalysis {
    pub impact_score: f64,
    pub affected_areas: Vec<String>,
    pub severity: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CostOptimization {
    pub optimization_type: String,
    pub target_cost: f64,
    pub enabled: bool,
}

impl Default for CostOptimization {
    fn default() -> Self {
        Self {
            optimization_type: String::new(),
            target_cost: 0.0,
            enabled: false,
        }
    }
}
