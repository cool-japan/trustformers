//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};

use super::types::ModelSelectionStrategy;
use super::types_3::EnsemblePipeline;
use super::types_4::{EnsembleConfig, EnsembleStrategy};

/// Gating network trait: given input text and the number of experts, returns
/// a probability distribution (softmax scores) over experts.
pub trait GatingNetwork: Send + Sync {
    fn gate(&self, input_text: &str, num_experts: usize) -> Result<Vec<f32>>;
}
/// Model routing trait: given input text and available model IDs, returns
/// (model_index, weight) pairs sorted descending by weight.
pub trait Router: Send + Sync {
    fn route(&self, input_text: &str, model_ids: &[String]) -> Vec<(usize, f32)>;
}
pub fn create_ensemble_pipeline(config: EnsembleConfig) -> EnsemblePipeline {
    EnsemblePipeline::new(config)
}
pub fn create_classification_ensemble(
    model_names: &[&str],
    weights: Option<Vec<f32>>,
) -> Result<EnsemblePipeline> {
    let default_weights = vec![1.0 / model_names.len() as f32; model_names.len()];
    let final_weights = weights.as_ref().unwrap_or(&default_weights);
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::WeightedAverage(final_weights.clone());
    let mut ensemble = EnsemblePipeline::new(config);
    for (i, &model_name) in model_names.iter().enumerate() {
        let weight = final_weights[i];
        ensemble.add_model_from_pretrained(model_name, "text-classification", weight, None)?;
    }
    Ok(ensemble)
}
pub fn create_qa_ensemble(
    model_names: &[&str],
    strategy: EnsembleStrategy,
) -> Result<EnsemblePipeline> {
    let mut config = EnsembleConfig::default();
    config.strategy = strategy;
    let mut ensemble = EnsemblePipeline::new(config);
    for &model_name in model_names {
        let weight = 1.0 / model_names.len() as f32;
        ensemble.add_model_from_pretrained(model_name, "question-answering", weight, None)?;
    }
    Ok(ensemble)
}
pub fn create_generation_ensemble(
    model_names: &[&str],
    strategy: EnsembleStrategy,
) -> Result<EnsemblePipeline> {
    let mut config = EnsembleConfig::default();
    config.strategy = strategy;
    let mut ensemble = EnsemblePipeline::new(config);
    for &model_name in model_names {
        let weight = 1.0 / model_names.len() as f32;
        ensemble.add_model_from_pretrained(model_name, "text-generation", weight, None)?;
    }
    Ok(ensemble)
}
pub fn create_dynamic_ensemble() -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::DynamicWeighting;
    config.enable_diversity_boost = true;
    config.enable_calibration = true;
    config.enable_explanation = true;
    EnsemblePipeline::new(config)
}
pub fn create_consensus_ensemble(consensus_threshold: f32) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::MajorityVote;
    config.require_consensus = true;
    config.consensus_threshold = consensus_threshold;
    EnsemblePipeline::new(config)
}
pub fn create_cascade_ensemble(early_exit_threshold: f32, max_models: usize) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::CascadePipeline;
    config.cascade_early_exit_threshold = early_exit_threshold;
    config.cascade_max_models = max_models;
    config.enable_explanation = true;
    EnsemblePipeline::new(config)
}
pub fn create_dynamic_routing_ensemble(routing_features: Vec<String>) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::DynamicRouting;
    config.routing_features = routing_features;
    config.enable_model_selection = true;
    config.model_selection_strategy = ModelSelectionStrategy::Dynamic;
    EnsemblePipeline::new(config)
}
pub fn create_quality_latency_ensemble(quality_weight: f32) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::QualityLatencyOptimized;
    config.quality_latency_weight = quality_weight;
    config.enable_explanation = true;
    EnsemblePipeline::new(config)
}
pub fn create_resource_aware_ensemble(budget_mb: u64) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::ResourceAware;
    config.resource_budget_mb = budget_mb;
    config.enable_model_selection = true;
    config.model_selection_strategy = ModelSelectionStrategy::ResourceConstrained;
    EnsemblePipeline::new(config)
}
pub fn create_uncertainty_ensemble(sampling_rate: f32) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::UncertaintyBased;
    config.uncertainty_sampling_rate = sampling_rate;
    config.enable_calibration = true;
    EnsemblePipeline::new(config)
}
pub fn create_adaptive_voting_ensemble(learning_rate: f32) -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::AdaptiveVoting;
    config.adaptive_learning_rate = learning_rate;
    config.enable_explanation = true;
    EnsemblePipeline::new(config)
}
pub fn create_high_performance_ensemble() -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::AdaptiveVoting;
    config.parallel_execution = true;
    config.max_concurrent_models = 8;
    config.enable_diversity_boost = true;
    config.enable_calibration = true;
    config.enable_explanation = true;
    config.enable_model_selection = true;
    config.model_selection_strategy = ModelSelectionStrategy::TopK(5);
    config.adaptive_learning_rate = 0.02;
    EnsemblePipeline::new(config)
}
pub fn create_efficient_ensemble() -> EnsemblePipeline {
    let mut config = EnsembleConfig::default();
    config.strategy = EnsembleStrategy::CascadePipeline;
    config.cascade_early_exit_threshold = 0.85;
    config.cascade_max_models = 2;
    config.resource_budget_mb = 1024;
    config.quality_latency_weight = 0.3;
    config.enable_model_selection = true;
    config.model_selection_strategy = ModelSelectionStrategy::ResourceConstrained;
    EnsemblePipeline::new(config)
}
