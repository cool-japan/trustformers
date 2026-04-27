//! # EmbeddingCosineRouter - Trait Implementations
//!
//! This module contains trait implementations for `EmbeddingCosineRouter`.
//!
//! ## Implemented Traits
//!
//! - `Router`
//! - `Debug`
//! - `Default`
//! - `Pipeline`
//! - `GatingNetwork`
//! - `Router`
//! - `GatingNetwork`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};

use super::functions::{GatingNetwork, Router};
use super::types::{
    BootstrapStats, EmbeddingCosineRouter, HashRoutingGate, KeywordRouter, ModelSelectionInfo,
    ModelSelectionStrategy, ModelWeight,
};
use super::types_3::EnsemblePipeline;
use super::types_4::{EnsembleConfig, EnsemblePrediction, EnsembleStrategy, SoftmaxEmbeddingGate};

impl Router for EmbeddingCosineRouter {
    fn route(&self, input_text: &str, model_ids: &[String]) -> Vec<(usize, f32)> {
        if model_ids.is_empty() {
            return Vec::new();
        }
        let input_emb = (self.input_embedding_fn)(input_text);
        let mut scores: Vec<(usize, f32)> = model_ids
            .iter()
            .enumerate()
            .filter_map(|(i, model_id)| {
                self.model_embeddings
                    .iter()
                    .find(|(name, _)| name == model_id)
                    .map(|(_, emb)| (i, Self::cosine_similarity(&input_emb, emb)))
            })
            .collect();
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores
    }
}

impl std::fmt::Debug for EnsembleConfig {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EnsembleConfig")
            .field("strategy", &self.strategy)
            .field("confidence_threshold", &self.confidence_threshold)
            .field("require_consensus", &self.require_consensus)
            .field("consensus_threshold", &self.consensus_threshold)
            .field("enable_diversity_boost", &self.enable_diversity_boost)
            .field("diversity_weight", &self.diversity_weight)
            .field("enable_calibration", &self.enable_calibration)
            .field("calibration_samples", &self.calibration_samples)
            .field("enable_explanation", &self.enable_explanation)
            .field("parallel_execution", &self.parallel_execution)
            .field("max_concurrent_models", &self.max_concurrent_models)
            .field("fallback_strategy", &self.fallback_strategy)
            .field("timeout_ms", &self.timeout_ms)
            .field(
                "cascade_early_exit_threshold",
                &self.cascade_early_exit_threshold,
            )
            .field("cascade_max_models", &self.cascade_max_models)
            .field("quality_latency_weight", &self.quality_latency_weight)
            .field("resource_budget_mb", &self.resource_budget_mb)
            .field("uncertainty_sampling_rate", &self.uncertainty_sampling_rate)
            .field("adaptive_learning_rate", &self.adaptive_learning_rate)
            .field("routing_features", &self.routing_features)
            .field("enable_model_selection", &self.enable_model_selection)
            .field("model_selection_strategy", &self.model_selection_strategy)
            .field("boosting_learning_rate", &self.boosting_learning_rate)
            .field("random_seed", &self.random_seed)
            .field("moe_top_k", &self.moe_top_k)
            .field(
                "gating_network",
                &self.gating_network.as_ref().map(|_| "<gating_network>"),
            )
            .field("router", &self.router.as_ref().map(|_| "<router>"))
            .finish()
    }
}

impl Default for EnsembleConfig {
    fn default() -> Self {
        Self {
            strategy: EnsembleStrategy::Average,
            confidence_threshold: 0.5,
            require_consensus: false,
            consensus_threshold: 0.7,
            enable_diversity_boost: false,
            diversity_weight: 0.2,
            enable_calibration: false,
            calibration_samples: 1000,
            enable_explanation: false,
            parallel_execution: true,
            max_concurrent_models: 4,
            fallback_strategy: Some(EnsembleStrategy::MajorityVote),
            timeout_ms: 30000,
            cascade_early_exit_threshold: 0.8,
            cascade_max_models: 3,
            quality_latency_weight: 0.5,
            resource_budget_mb: 2048,
            uncertainty_sampling_rate: 0.1,
            adaptive_learning_rate: 0.01,
            routing_features: vec!["input_length".to_string(), "complexity".to_string()],
            enable_model_selection: false,
            model_selection_strategy: ModelSelectionStrategy::All,
            boosting_learning_rate: 1.0,
            random_seed: 42,
            moe_top_k: 0,
            gating_network: None,
            router: None,
        }
    }
}

impl Pipeline for EnsemblePipeline {
    type Input = String;
    type Output = EnsemblePrediction;
    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        if self.models.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No models in ensemble".to_string(),
            ));
        }
        let start_time = std::time::Instant::now();
        let input_characteristics = self.analyze_input_characteristics(&input);
        let predictions = self.predict_individual_models(&input)?;
        let weights = match &self.config.strategy {
            EnsembleStrategy::DynamicWeighting => {
                let confidence_weights: Vec<f32> = predictions
                    .iter()
                    .map(|(_, output, _)| self.extract_confidence(output))
                    .collect();
                let sum: f32 = confidence_weights.iter().sum();
                if sum > 0.0 {
                    confidence_weights.iter().map(|w| w / sum).collect()
                } else {
                    vec![1.0 / predictions.len() as f32; predictions.len()]
                }
            },
            EnsembleStrategy::WeightedAverage(custom_weights) => {
                if custom_weights.len() == predictions.len() {
                    custom_weights.clone()
                } else {
                    vec![1.0 / predictions.len() as f32; predictions.len()]
                }
            },
            _ => vec![1.0 / predictions.len() as f32; predictions.len()],
        };
        let final_prediction =
            self.apply_ensemble_strategy(&predictions, &weights, &input_characteristics)?;
        let confidence_score = self.extract_confidence(&final_prediction);
        let consensus_score = self.calculate_consensus_score(&predictions);
        let diversity_score = self.calculate_diversity_score(&predictions);
        let uncertainty_score = self.calculate_uncertainty_score(&predictions);
        let quality_latency_score = self.calculate_quality_latency_score(&predictions, &weights);
        let resource_usage_mb = self.estimate_resource_usage(&predictions, &input_characteristics);
        let model_weights: Vec<ModelWeight> = self
            .models
            .iter()
            .enumerate()
            .map(|(i, model)| {
                let mut weight = model.weight.clone();
                weight.dynamic_weight = *weights.get(i).unwrap_or(&1.0);
                weight
            })
            .collect();
        let processing_time = start_time.elapsed().as_millis() as u64;
        let early_exit_triggered =
            matches!(self.config.strategy, EnsembleStrategy::CascadePipeline)
                && predictions.len() < self.models.len();
        let routing_decision = match &self.config.strategy {
            EnsembleStrategy::DynamicRouting => Some(format!(
                "Routed to {} models based on input characteristics",
                predictions.len()
            )),
            EnsembleStrategy::ResourceAware => Some(format!(
                "Selected {} models within resource budget",
                predictions.len()
            )),
            _ => None,
        };
        let model_selection_info = if self.config.enable_model_selection {
            Some(ModelSelectionInfo {
                selected_models: predictions.iter().map(|(id, _, _)| id.clone()).collect(),
                selection_reason: format!(
                    "Selected using {:?} strategy",
                    self.config.model_selection_strategy
                ),
                selection_confidence: confidence_score,
                alternative_models: self
                    .models
                    .iter()
                    .filter(|m| !predictions.iter().any(|(id, _, _)| id == &m.model_id))
                    .map(|m| m.model_id.clone())
                    .collect(),
            })
        } else {
            None
        };
        // Capture bootstrap stats when Bagging strategy was active
        let bootstrap_stats: Option<BootstrapStats> =
            if matches!(self.config.strategy, EnsembleStrategy::Bagging) {
                self.last_bagging_stats()
            } else {
                None
            };
        let mut ensemble_prediction = EnsemblePrediction {
            final_prediction,
            individual_predictions: predictions
                .iter()
                .map(|(_, output, _)| output.clone())
                .collect(),
            model_weights,
            confidence_score,
            consensus_score,
            diversity_score,
            explanation: None,
            processing_time_ms: processing_time,
            models_used: predictions.iter().map(|(model_id, _, _)| model_id.clone()).collect(),
            uncertainty_score,
            resource_usage_mb,
            quality_latency_score,
            early_exit_triggered,
            routing_decision,
            model_selection_info,
            bootstrap_stats,
        };
        if self.config.enable_explanation {
            ensemble_prediction.explanation = Some(self.generate_explanation(&ensemble_prediction));
        }
        Ok(ensemble_prediction)
    }
    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter().map(|input| self.__call__(input)).collect()
    }
}

impl GatingNetwork for HashRoutingGate {
    fn gate(&self, input_text: &str, num_experts: usize) -> Result<Vec<f32>> {
        if num_experts == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "num_experts must be > 0".to_string(),
            ));
        }
        let mut hash: u64 = 14_695_981_039_346_656_037;
        for byte in input_text.bytes() {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(1_099_511_628_211);
        }
        let primary = (hash % num_experts as u64) as usize;
        let mut scores: Vec<f32> = (0..num_experts)
            .map(|i| {
                let dist = (i as isize - primary as isize).unsigned_abs() as f32;
                (-(dist * self.temperature)).exp()
            })
            .collect();
        let sum: f32 = scores.iter().sum();
        if sum > 0.0 {
            scores.iter_mut().for_each(|s| *s /= sum);
        }
        Ok(scores)
    }
}

impl Router for KeywordRouter {
    fn route(&self, input_text: &str, model_ids: &[String]) -> Vec<(usize, f32)> {
        if model_ids.is_empty() {
            return Vec::new();
        }
        let length = input_text.len();
        let mut results: Vec<(usize, f32)> = model_ids
            .iter()
            .enumerate()
            .filter_map(|(i, model_id)| {
                let selected = match length {
                    0..=100 => model_id.contains("small") || model_id.contains("fast"),
                    101..=500 => !model_id.contains("large"),
                    _ => true,
                };
                if selected {
                    Some((i, 1.0))
                } else {
                    None
                }
            })
            .collect();
        if results.is_empty() {
            results = model_ids.iter().enumerate().map(|(i, _)| (i, 1.0)).collect();
        }
        let n = results.len() as f32;
        results.iter_mut().for_each(|(_, w)| *w = 1.0 / n);
        results
    }
}

impl GatingNetwork for SoftmaxEmbeddingGate {
    fn gate(&self, input_text: &str, num_experts: usize) -> Result<Vec<f32>> {
        if num_experts == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "num_experts must be > 0".to_string(),
            ));
        }
        let input_emb = (self.input_embedding_fn)(input_text);
        let n = num_experts.min(self.expert_embeddings.len());
        let mut logits: Vec<f32> = (0..n)
            .map(|i| Self::cosine_similarity(&input_emb, &self.expert_embeddings[i]))
            .collect();
        logits.resize(num_experts, 0.0);
        let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|l| (l - max_logit).exp()).collect();
        let sum: f32 = exps.iter().sum();
        let scores: Vec<f32> = exps.iter().map(|e| e / sum).collect();
        Ok(scores)
    }
}
