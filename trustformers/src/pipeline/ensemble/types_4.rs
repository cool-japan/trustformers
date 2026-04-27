//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, Mutex};

use super::functions::{GatingNetwork, Router};
use super::types::{BootstrapStats, ModelSelectionInfo, ModelSelectionStrategy, ModelWeight};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnsembleStrategy {
    /// Simple averaging of predictions
    Average,
    /// Weighted average with custom weights
    WeightedAverage(Vec<f32>),
    /// Majority voting for classification tasks
    MajorityVote,
    /// Take maximum prediction across models
    Maximum,
    /// Take minimum prediction across models
    Minimum,
    /// Stacked ensemble with a meta-learner
    Stacking,
    /// Boosting-style ensemble
    Boosting,
    /// Bagging-style ensemble
    Bagging,
    /// Dynamic weighting based on confidence
    DynamicWeighting,
    /// Rank-based ensemble
    RankFusion,
    /// Mixture of experts
    MixtureOfExperts,
    /// Cascade pipeline with early exit
    CascadePipeline,
    /// Dynamic routing based on input characteristics
    DynamicRouting,
    /// Quality-latency trade-off optimization
    QualityLatencyOptimized,
    /// Resource-aware execution
    ResourceAware,
    /// Uncertainty-based ensemble
    UncertaintyBased,
    /// Adaptive voting with learned weights
    AdaptiveVoting,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnsemblePrediction {
    pub final_prediction: PipelineOutput,
    pub individual_predictions: Vec<PipelineOutput>,
    pub model_weights: Vec<ModelWeight>,
    pub confidence_score: f32,
    pub consensus_score: f32,
    pub diversity_score: f32,
    pub explanation: Option<String>,
    pub processing_time_ms: u64,
    pub models_used: Vec<String>,
    pub uncertainty_score: f32,
    pub resource_usage_mb: u64,
    pub quality_latency_score: f32,
    pub early_exit_triggered: bool,
    pub routing_decision: Option<String>,
    pub model_selection_info: Option<ModelSelectionInfo>,
    /// Bootstrap resampling statistics when Bagging strategy is used
    #[serde(skip)]
    pub bootstrap_stats: Option<BootstrapStats>,
}
#[derive()]
pub struct EnsembleModel {
    pub model_id: String,
    pub pipeline: Box<dyn Pipeline<Input = String, Output = PipelineOutput>>,
    pub weight: ModelWeight,
    pub performance_history: Vec<f32>,
    pub last_prediction_time_ms: u64,
    pub total_predictions: u64,
    pub successful_predictions: u64,
}
impl EnsembleModel {
    pub fn new(
        model_id: String,
        pipeline: Box<dyn Pipeline<Input = String, Output = PipelineOutput>>,
        initial_weight: f32,
    ) -> Self {
        Self {
            model_id: model_id.clone(),
            pipeline,
            weight: ModelWeight::new(model_id, initial_weight),
            performance_history: Vec::new(),
            last_prediction_time_ms: 0,
            total_predictions: 0,
            successful_predictions: 0,
        }
    }
    pub fn accuracy(&self) -> f32 {
        if self.total_predictions == 0 {
            1.0
        } else {
            self.successful_predictions as f32 / self.total_predictions as f32
        }
    }
    pub fn average_performance(&self) -> f32 {
        if self.performance_history.is_empty() {
            0.5
        } else {
            self.performance_history.iter().sum::<f32>() / self.performance_history.len() as f32
        }
    }
    pub fn update_performance(&mut self, score: f32, prediction_time_ms: u64) {
        self.performance_history.push(score);
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }
        self.last_prediction_time_ms = prediction_time_ms;
        self.total_predictions += 1;
        if score > 0.5 {
            self.successful_predictions += 1;
        }
    }
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InputCharacteristics {
    pub length: usize,
    pub complexity_score: f32,
    pub estimated_processing_time: u64,
    pub required_resource_mb: u64,
    pub domain: Option<String>,
    pub language: Option<String>,
}
#[derive(Clone, Serialize, Deserialize)]
pub struct EnsembleConfig {
    pub strategy: EnsembleStrategy,
    pub confidence_threshold: f32,
    pub require_consensus: bool,
    pub consensus_threshold: f32,
    pub enable_diversity_boost: bool,
    pub diversity_weight: f32,
    pub enable_calibration: bool,
    pub calibration_samples: usize,
    pub enable_explanation: bool,
    pub parallel_execution: bool,
    pub max_concurrent_models: usize,
    pub fallback_strategy: Option<EnsembleStrategy>,
    pub timeout_ms: u64,
    pub cascade_early_exit_threshold: f32,
    pub cascade_max_models: usize,
    pub quality_latency_weight: f32,
    pub resource_budget_mb: u64,
    pub uncertainty_sampling_rate: f32,
    pub adaptive_learning_rate: f32,
    pub routing_features: Vec<String>,
    pub enable_model_selection: bool,
    pub model_selection_strategy: ModelSelectionStrategy,
    pub boosting_learning_rate: f32,
    pub random_seed: u64,
    pub moe_top_k: usize,
    #[serde(skip)]
    pub gating_network: Option<Arc<dyn GatingNetwork>>,
    #[serde(skip)]
    pub router: Option<Arc<dyn Router>>,
}
/// Cosine-similarity gating gate backed by pre-computed expert embeddings.
pub struct SoftmaxEmbeddingGate {
    pub(super) expert_embeddings: Vec<Vec<f32>>,
    pub(super) input_embedding_fn: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
}
impl SoftmaxEmbeddingGate {
    pub fn new(
        expert_embeddings: Vec<Vec<f32>>,
        input_embedding_fn: Arc<dyn Fn(&str) -> Vec<f32> + Send + Sync>,
    ) -> Self {
        Self {
            expert_embeddings,
            input_embedding_fn,
        }
    }
    pub(super) fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)).clamp(-1.0, 1.0)
        }
    }
}
/// Load-balance statistics emitted after each MoE call
#[derive(Debug, Clone)]
pub struct LoadBalanceStats {
    /// Fraction of routing probability mass assigned to each expert
    pub expert_loads: Vec<f32>,
    /// Max minus min expert load
    pub load_imbalance: f32,
    /// Gini coefficient over expert loads
    pub gini_coefficient: f32,
}
