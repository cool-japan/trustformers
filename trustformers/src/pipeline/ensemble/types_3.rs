//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use super::functions::{GatingNetwork, Router};
use super::types::{BootstrapStats, KeywordRouter, ModelWeight};
use super::types_4::{
    EnsembleConfig, EnsembleModel, EnsemblePrediction, EnsembleStrategy, InputCharacteristics,
    LoadBalanceStats,
};

pub struct EnsemblePipeline {
    pub(super) config: EnsembleConfig,
    pub(super) models: Vec<EnsembleModel>,
    meta_learner: Option<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>>,
    calibration_data: Vec<(String, PipelineOutput)>,
    performance_tracker: HashMap<String, Vec<f32>>,
    /// Ordered list of model IDs (mirrors self.models[i].model_id), used by the router
    model_ids: Vec<String>,
    /// Last MoE load balance stats; updated on each MoE call
    load_balance_stats: Arc<Mutex<Option<LoadBalanceStats>>>,
    /// Bootstrap stats from the most recent Bagging call
    bagging_stats: Arc<Mutex<Option<BootstrapStats>>>,
}
impl EnsemblePipeline {
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
            meta_learner: None,
            calibration_data: Vec::new(),
            performance_tracker: HashMap::new(),
            model_ids: Vec::new(),
            load_balance_stats: Arc::new(Mutex::new(None)),
            bagging_stats: Arc::new(Mutex::new(None)),
        }
    }
    /// Returns the last MoE load balance stats (None if MoE was never called)
    pub fn last_load_balance_stats(&self) -> Option<LoadBalanceStats> {
        self.load_balance_stats.lock().ok().and_then(|g| g.clone())
    }
    /// Returns the bootstrap resampling stats from the most recent Bagging call (None otherwise)
    pub fn last_bagging_stats(&self) -> Option<BootstrapStats> {
        self.bagging_stats.lock().ok().and_then(|g| g.clone())
    }
    pub fn add_model(
        &mut self,
        model_id: String,
        pipeline: Box<dyn Pipeline<Input = String, Output = PipelineOutput>>,
        weight: f32,
    ) -> Result<()> {
        let ensemble_model = EnsembleModel::new(model_id.clone(), pipeline, weight);
        self.models.push(ensemble_model);
        self.performance_tracker.insert(model_id.clone(), Vec::new());
        self.model_ids.push(model_id);
        Ok(())
    }
    pub fn add_model_from_pretrained(
        &mut self,
        model_name: &str,
        task: &str,
        weight: f32,
        options: Option<PipelineOptions>,
    ) -> Result<()> {
        let pipeline = crate::pipeline::pipeline(task, Some(model_name), options)?;
        self.add_model(model_name.to_string(), pipeline, weight)
    }
    pub fn set_meta_learner(
        &mut self,
        meta_learner: Box<dyn Pipeline<Input = String, Output = PipelineOutput>>,
    ) {
        self.meta_learner = Some(meta_learner);
    }
    pub fn remove_model(&mut self, model_id: &str) -> bool {
        if let Some(pos) = self.models.iter().position(|m| m.model_id == *model_id) {
            self.models.remove(pos);
            self.performance_tracker.remove(model_id);
            self.model_ids.retain(|id| id != model_id);
            true
        } else {
            false
        }
    }
    pub fn update_model_weight(&mut self, model_id: &str, new_weight: f32) -> bool {
        if let Some(model) = self.models.iter_mut().find(|m| m.model_id == *model_id) {
            model.weight.weight = new_weight;
            true
        } else {
            false
        }
    }
    pub fn get_model_weights(&self) -> Vec<ModelWeight> {
        self.models.iter().map(|m| m.weight.clone()).collect()
    }
    pub(crate) fn predict_individual_models(
        &self,
        input: &str,
    ) -> Result<Vec<(String, PipelineOutput, u64)>> {
        if self.config.parallel_execution && self.models.len() > 1 {
            let n = self.models.len();
            let mut slot_results: Vec<Option<Result<(String, PipelineOutput, u64)>>> =
                (0..n).map(|_| None).collect();
            std::thread::scope(|s| {
                let mut handles = Vec::with_capacity(n);
                for model in &self.models {
                    let input_owned = input.to_string();
                    let handle = s.spawn(move || {
                        let start_time = std::time::Instant::now();
                        let prediction = model.pipeline.__call__(input_owned)?;
                        let duration = start_time.elapsed().as_millis() as u64;
                        Ok((model.model_id.clone(), prediction, duration))
                    });
                    handles.push(handle);
                }
                for (i, handle) in handles.into_iter().enumerate() {
                    slot_results[i] = Some(handle.join().unwrap_or_else(|_| {
                        Err(TrustformersError::invalid_input_simple(
                            "parallel model thread panicked".to_string(),
                        ))
                    }));
                }
            });
            slot_results
                .into_iter()
                .map(|opt| {
                    opt.unwrap_or_else(|| {
                        Err(TrustformersError::invalid_input_simple(
                            "thread result missing".to_string(),
                        ))
                    })
                })
                .collect()
        } else {
            let mut predictions = Vec::with_capacity(self.models.len());
            for model in &self.models {
                let start_time = std::time::Instant::now();
                let prediction = model.pipeline.__call__(input.to_string())?;
                let duration = start_time.elapsed().as_millis() as u64;
                predictions.push((model.model_id.clone(), prediction, duration));
            }
            Ok(predictions)
        }
    }
    fn calculate_dynamic_weights(
        &mut self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Vec<f32> {
        let mut weights = Vec::new();
        for (model_id, output, _) in predictions {
            let confidence = self.extract_confidence(output);
            if let Some(model) = self.models.iter_mut().find(|m| m.model_id == *model_id) {
                model.weight.confidence_weight = confidence;
                model.weight.accuracy_weight = model.accuracy();
                let recent_performance = model.average_performance();
                model.weight.dynamic_weight = (recent_performance + confidence) / 2.0;
            }
            let model_weight = self
                .models
                .iter()
                .find(|m| m.model_id == *model_id)
                .map(|m| m.weight.total_weight())
                .unwrap_or(1.0);
            weights.push(model_weight);
        }
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }
        weights
    }
    pub(super) fn extract_confidence(&self, output: &PipelineOutput) -> f32 {
        match output {
            PipelineOutput::Classification(results) => {
                results.iter().map(|r| r.score).fold(0.0f32, f32::max)
            },
            PipelineOutput::QuestionAnswering(result) => result.score,
            PipelineOutput::FillMask(results) => {
                results.iter().map(|r| r.score).fold(0.0f32, f32::max)
            },
            _ => 0.8,
        }
    }
    pub(super) fn apply_ensemble_strategy(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
        input_characteristics: &InputCharacteristics,
    ) -> Result<PipelineOutput> {
        let weights_f64: Vec<f64> = weights.iter().map(|w| *w as f64).collect();
        match &self.config.strategy {
            EnsembleStrategy::Average => self.average_predictions(predictions, weights),
            EnsembleStrategy::WeightedAverage(custom_weights) => {
                self.weighted_average_predictions(predictions, custom_weights)
            },
            EnsembleStrategy::MajorityVote => self.majority_vote_predictions(predictions),
            EnsembleStrategy::Maximum => self.maximum_predictions(predictions),
            EnsembleStrategy::Minimum => self.minimum_predictions(predictions),
            EnsembleStrategy::Stacking => self.stacking_predictions(predictions),
            EnsembleStrategy::Boosting => self.boosting_predictions(predictions, &weights_f64),
            EnsembleStrategy::Bagging => self.bagging_predictions(predictions, &weights_f64),
            EnsembleStrategy::DynamicWeighting => self.average_predictions(predictions, weights),
            EnsembleStrategy::RankFusion => self.rank_fusion_predictions(predictions),
            EnsembleStrategy::MixtureOfExperts => {
                self.mixture_of_experts_predictions(predictions, weights, input_characteristics)
            },
            EnsembleStrategy::CascadePipeline => self.cascade_predictions(predictions),
            EnsembleStrategy::DynamicRouting => {
                self.dynamic_routing_predictions(predictions, input_characteristics)
            },
            EnsembleStrategy::QualityLatencyOptimized => {
                self.quality_latency_optimized_predictions(predictions, weights)
            },
            EnsembleStrategy::ResourceAware => {
                self.resource_aware_predictions(predictions, weights, input_characteristics)
            },
            EnsembleStrategy::UncertaintyBased => {
                self.uncertainty_based_predictions(predictions, weights)
            },
            EnsembleStrategy::AdaptiveVoting => {
                self.adaptive_voting_predictions(predictions, weights)
            },
        }
    }
    pub(crate) fn average_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        if predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No predictions to ensemble".to_string(),
            ));
        }
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                self.average_classification_predictions(predictions, weights)
            },
            PipelineOutput::Generation(_) => self.select_best_generation(predictions, weights),
            PipelineOutput::QuestionAnswering(_) => {
                self.average_qa_predictions(predictions, weights)
            },
            _ => {
                let best_index = weights
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(i, _)| i)
                    .unwrap_or(0);
                Ok(predictions[best_index].1.clone())
            },
        }
    }
    fn average_classification_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let mut label_scores: HashMap<String, f32> = HashMap::new();
        for (i, (_, output, _)) in predictions.iter().enumerate() {
            if let PipelineOutput::Classification(results) = output {
                let weight = weights.get(i).unwrap_or(&1.0);
                for result in results {
                    *label_scores.entry(result.label.clone()).or_insert(0.0) +=
                        result.score * weight;
                }
            }
        }
        let mut averaged_results: Vec<crate::pipeline::ClassificationOutput> = label_scores
            .into_iter()
            .map(|(label, score)| crate::pipeline::ClassificationOutput { label, score })
            .collect();
        averaged_results
            .sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(PipelineOutput::Classification(averaged_results))
    }
    fn average_qa_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let mut best_score = 0.0;
        let mut best_answer = String::new();
        let mut best_start = 0;
        let mut best_end = 0;
        for (i, (_, output, _)) in predictions.iter().enumerate() {
            if let PipelineOutput::QuestionAnswering(result) = output {
                let weight = weights.get(i).unwrap_or(&1.0);
                let weighted_score = result.score * weight;
                if weighted_score > best_score {
                    best_score = weighted_score;
                    best_answer = result.answer.clone();
                    best_start = result.start;
                    best_end = result.end;
                }
            }
        }
        Ok(PipelineOutput::QuestionAnswering(
            crate::pipeline::QuestionAnsweringOutput {
                answer: best_answer,
                score: best_score,
                start: best_start,
                end: best_end,
            },
        ))
    }
    fn select_best_generation(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let best_index = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0);
        Ok(predictions[best_index].1.clone())
    }
    fn weighted_average_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        custom_weights: &[f32],
    ) -> Result<PipelineOutput> {
        let weights = if custom_weights.len() == predictions.len() {
            custom_weights.to_vec()
        } else {
            vec![1.0 / predictions.len() as f32; predictions.len()]
        };
        self.average_predictions(predictions, &weights)
    }
    pub(crate) fn majority_vote_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut label_votes: HashMap<String, u32> = HashMap::new();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top_result) = results.first() {
                            *label_votes.entry(top_result.label.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let (winning_label, vote_count) = label_votes
                    .into_iter()
                    .max_by_key(|(_, count)| *count)
                    .unwrap_or(("unknown".to_string(), 0));
                let confidence = vote_count as f32 / predictions.len() as f32;
                Ok(PipelineOutput::Classification(vec![
                    crate::pipeline::ClassificationOutput {
                        label: winning_label,
                        score: confidence,
                    },
                ]))
            },
            _ => {
                let weights = vec![1.0 / predictions.len() as f32; predictions.len()];
                self.average_predictions(predictions, &weights)
            },
        }
    }
    fn maximum_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut best_score = 0.0;
                let mut best_output = predictions[0].1.clone();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top_result) = results.first() {
                            if top_result.score > best_score {
                                best_score = top_result.score;
                                best_output = output.clone();
                            }
                        }
                    }
                }
                Ok(best_output)
            },
            _ => Ok(predictions[0].1.clone()),
        }
    }
    fn minimum_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut min_score = f32::INFINITY;
                let mut best_output = predictions[0].1.clone();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top_result) = results.first() {
                            if top_result.score < min_score {
                                min_score = top_result.score;
                                best_output = output.clone();
                            }
                        }
                    }
                }
                Ok(best_output)
            },
            _ => Ok(predictions[0].1.clone()),
        }
    }
    fn stacking_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        if let Some(meta_learner) = &self.meta_learner {
            let features = self.create_stacking_features(predictions)?;
            meta_learner.__call__(features)
        } else {
            let weights = vec![1.0 / predictions.len() as f32; predictions.len()];
            self.average_predictions(predictions, &weights)
        }
    }
    fn rank_fusion_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut label_rank_scores: HashMap<String, f32> = HashMap::new();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        for (rank, result) in results.iter().enumerate() {
                            let rank_score = 1.0 / (rank + 1) as f32;
                            *label_rank_scores.entry(result.label.clone()).or_insert(0.0) +=
                                rank_score;
                        }
                    }
                }
                let mut rank_results: Vec<crate::pipeline::ClassificationOutput> =
                    label_rank_scores
                        .into_iter()
                        .map(|(label, score)| crate::pipeline::ClassificationOutput {
                            label,
                            score,
                        })
                        .collect();
                rank_results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(PipelineOutput::Classification(rank_results))
            },
            _ => {
                let weights = vec![1.0 / predictions.len() as f32; predictions.len()];
                self.average_predictions(predictions, &weights)
            },
        }
    }
    pub(crate) fn mixture_of_experts_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
        input_characteristics: &InputCharacteristics,
    ) -> Result<PipelineOutput> {
        let n = predictions.len();
        if n == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "No predictions for MoE".to_string(),
            ));
        }
        if let Some(gate) = &self.config.gating_network {
            let input_text = &input_characteristics.domain.as_deref().unwrap_or("");
            let gate_scores = gate.gate(input_text, n)?;
            let k = if self.config.moe_top_k == 0 || self.config.moe_top_k >= n {
                n
            } else {
                self.config.moe_top_k
            };
            let mut indexed_scores: Vec<(usize, f32)> =
                gate_scores.iter().copied().enumerate().collect();
            indexed_scores
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let top_k: Vec<(usize, f32)> = indexed_scores.into_iter().take(k).collect();
            let sum: f32 = top_k.iter().map(|(_, s)| s).sum();
            let final_scores: Vec<f32> = top_k.iter().map(|(_, s)| s / sum.max(1e-9)).collect();
            let load_imbalance = gate_scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max)
                - gate_scores.iter().cloned().fold(f32::INFINITY, f32::min);
            let gini = Self::gini_coefficient(&gate_scores);
            let stats = LoadBalanceStats {
                expert_loads: gate_scores,
                load_imbalance,
                gini_coefficient: gini,
            };
            if let Ok(mut guard) = self.load_balance_stats.lock() {
                *guard = Some(stats);
            }
            let selected_predictions: Vec<(String, PipelineOutput, u64)> =
                top_k.iter().map(|(i, _)| predictions[*i].clone()).collect();
            self.average_predictions(&selected_predictions, &final_scores)
        } else {
            self.average_predictions(predictions, weights)
        }
    }
    fn gini_coefficient(loads: &[f32]) -> f32 {
        let n = loads.len();
        if n == 0 {
            return 0.0;
        }
        let mut sorted = loads.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let sum: f32 = sorted.iter().sum();
        if sum == 0.0 {
            return 0.0;
        }
        let n_i64 = n as i64;
        let numerator: f32 = sorted
            .iter()
            .enumerate()
            .map(|(i, &v)| (2 * (i as i64 + 1) - n_i64 - 1) as f32 * v)
            .sum();
        numerator / (n as f32 * sum)
    }
    pub(crate) fn boosting_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f64],
    ) -> Result<PipelineOutput> {
        if predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No predictions for Boosting".to_string(),
            ));
        }
        let lr = self.config.boosting_learning_rate as f64;
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut label_scores: HashMap<String, f64> = HashMap::new();
                for (i, (_, output, _)) in predictions.iter().enumerate() {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top) = results.first() {
                            let alpha = weights.get(i).copied().unwrap_or(1.0) * lr;
                            *label_scores.entry(top.label.clone()).or_insert(0.0) += alpha;
                        }
                    }
                }
                if label_scores.is_empty() {
                    return Err(TrustformersError::invalid_input_simple(
                        "Boosting: no classification labels found".to_string(),
                    ));
                }
                let max_score = label_scores.values().cloned().fold(f64::NEG_INFINITY, f64::max);
                let exps: Vec<(String, f64)> = label_scores
                    .iter()
                    .map(|(label, &score)| (label.clone(), (score - max_score).exp()))
                    .collect();
                let sum: f64 = exps.iter().map(|(_, e)| e).sum();
                let mut results: Vec<crate::pipeline::ClassificationOutput> = exps
                    .into_iter()
                    .map(|(label, e)| crate::pipeline::ClassificationOutput {
                        label,
                        score: (e / sum) as f32,
                    })
                    .collect();
                results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(PipelineOutput::Classification(results))
            },
            _ => {
                let weights_f32: Vec<f32> = weights.iter().map(|w| *w as f32).collect();
                self.weighted_average_predictions(predictions, &weights_f32)
            },
        }
    }
    pub(crate) fn bagging_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        _weights: &[f64],
    ) -> Result<PipelineOutput> {
        if predictions.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "No predictions for Bagging".to_string(),
            ));
        }
        let n = predictions.len();
        let seed = self.config.random_seed;
        let bootstrap_indices = Self::lcg_bootstrap(seed, n);

        // Collect per-sample confidence scores for bootstrap statistics
        let sample_scores: Vec<f64> = bootstrap_indices
            .iter()
            .map(|&idx| self.extract_confidence(&predictions[idx].1) as f64)
            .collect();
        let n_samples = sample_scores.len();
        let mean: f64 = if n_samples > 0 {
            sample_scores.iter().sum::<f64>() / n_samples as f64
        } else {
            0.0
        };
        let variance: f64 = if n_samples > 1 {
            sample_scores.iter().map(|s| (s - mean).powi(2)).sum::<f64>() / (n_samples - 1) as f64
        } else {
            0.0
        };
        let stats = BootstrapStats {
            mean,
            variance,
            n_samples,
        };
        if let Ok(mut guard) = self.bagging_stats.lock() {
            *guard = Some(stats);
        }

        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut label_votes: HashMap<String, u32> = HashMap::new();
                for &idx in &bootstrap_indices {
                    if let PipelineOutput::Classification(results) = &predictions[idx].1 {
                        if let Some(top) = results.first() {
                            *label_votes.entry(top.label.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let total = bootstrap_indices.len() as f32;
                let mut results: Vec<crate::pipeline::ClassificationOutput> = label_votes
                    .into_iter()
                    .map(|(label, count)| crate::pipeline::ClassificationOutput {
                        label,
                        score: count as f32 / total,
                    })
                    .collect();
                results.sort_by(|a, b| {
                    b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal)
                });
                Ok(PipelineOutput::Classification(results))
            },
            _ => {
                let weights_uniform: Vec<f32> =
                    bootstrap_indices.iter().map(|_| 1.0 / n as f32).collect();
                let bootstrap_preds: Vec<(String, PipelineOutput, u64)> =
                    bootstrap_indices.iter().map(|&i| predictions[i].clone()).collect();
                self.average_predictions(&bootstrap_preds, &weights_uniform)
            },
        }
    }
    /// Linear congruential generator producing N bootstrap indices in [0, n)
    fn lcg_bootstrap(seed: u64, n: usize) -> Vec<usize> {
        const A: u64 = 6_364_136_223_846_793_005;
        const C: u64 = 1_442_695_040_888_963_407;
        let mut state = seed;
        let mut indices = Vec::with_capacity(n);
        for _ in 0..n {
            state = state.wrapping_mul(A).wrapping_add(C);
            indices.push((state >> 33) as usize % n);
        }
        indices
    }
    fn create_stacking_features(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<String> {
        let mut features = Vec::new();
        for (model_id, output, duration) in predictions {
            match output {
                PipelineOutput::Classification(results) => {
                    for result in results {
                        features.push(format!("{}:{}:{}", model_id, result.label, result.score));
                    }
                },
                PipelineOutput::QuestionAnswering(result) => {
                    features.push(format!("{}:{}:{}", model_id, result.answer, result.score));
                },
                _ => {
                    features.push(format!("{}:unknown:0.5", model_id));
                },
            }
            features.push(format!("{}:time:{}", model_id, duration));
        }
        Ok(features.join("|"))
    }
    fn cascade_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        let mut cumulative_confidence = 0.0;
        let mut used_predictions = Vec::new();
        let threshold = self.config.cascade_early_exit_threshold;
        let max_models = std::cmp::min(self.config.cascade_max_models, predictions.len());
        for (i, (model_id, output, _)) in predictions.iter().take(max_models).enumerate() {
            let confidence = self.extract_confidence(output);
            cumulative_confidence =
                (cumulative_confidence * i as f32 + confidence) / (i + 1) as f32;
            used_predictions.push((model_id.clone(), output.clone()));
            if cumulative_confidence >= threshold {
                break;
            }
        }
        let weights = vec![1.0 / used_predictions.len() as f32; used_predictions.len()];
        let used_predictions_with_time: Vec<(String, PipelineOutput, u64)> =
            used_predictions.into_iter().map(|(id, output)| (id, output, 0)).collect();
        self.average_predictions(&used_predictions_with_time, &weights)
    }
    fn dynamic_routing_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        input_characteristics: &InputCharacteristics,
    ) -> Result<PipelineOutput> {
        let selected_models = self.select_models_for_input(input_characteristics);
        let filtered_predictions: Vec<(String, PipelineOutput, u64)> = predictions
            .iter()
            .filter(|(model_id, _, _)| selected_models.contains(&model_id.as_str()))
            .cloned()
            .collect();
        if filtered_predictions.is_empty() {
            let weights = vec![1.0 / predictions.len() as f32; predictions.len()];
            return self.average_predictions(predictions, &weights);
        }
        let weights = vec![1.0 / filtered_predictions.len() as f32; filtered_predictions.len()];
        self.average_predictions(&filtered_predictions, &weights)
    }
    fn quality_latency_optimized_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let quality_weight = self.config.quality_latency_weight;
        let latency_weight = 1.0 - quality_weight;
        let mut optimized_weights = Vec::new();
        for (i, (_, output, duration)) in predictions.iter().enumerate() {
            let quality_score = self.extract_confidence(output);
            let latency_score = 1.0 / (1.0 + *duration as f32 / 1000.0);
            let combined_score = quality_score * quality_weight + latency_score * latency_weight;
            let base_weight = weights.get(i).unwrap_or(&1.0);
            optimized_weights.push(base_weight * combined_score);
        }
        let sum: f32 = optimized_weights.iter().sum();
        if sum > 0.0 {
            optimized_weights.iter_mut().for_each(|w| *w /= sum);
        }
        self.average_predictions(predictions, &optimized_weights)
    }
    fn resource_aware_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
        input_characteristics: &InputCharacteristics,
    ) -> Result<PipelineOutput> {
        let budget_mb = self.config.resource_budget_mb;
        let required_mb = input_characteristics.required_resource_mb;
        if required_mb <= budget_mb {
            self.average_predictions(predictions, weights)
        } else {
            let mut model_efficiency: Vec<(usize, f32)> = predictions
                .iter()
                .enumerate()
                .map(|(i, (_, output, duration))| {
                    let quality = self.extract_confidence(output);
                    let efficiency = quality / (*duration as f32 + 1.0);
                    (i, efficiency)
                })
                .collect();
            model_efficiency
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            let mut selected_indices = Vec::new();
            let mut used_resources = 0u64;
            for (idx, _) in &model_efficiency {
                let estimated_usage = required_mb / predictions.len() as u64;
                if used_resources + estimated_usage <= budget_mb {
                    selected_indices.push(*idx);
                    used_resources += estimated_usage;
                }
            }
            if selected_indices.is_empty() {
                selected_indices.push(model_efficiency[0].0);
            }
            let filtered_predictions: Vec<(String, PipelineOutput, u64)> =
                selected_indices.iter().map(|&i| predictions[i].clone()).collect();
            let filtered_weights: Vec<f32> =
                selected_indices.iter().map(|&i| *weights.get(i).unwrap_or(&1.0)).collect();
            self.average_predictions(&filtered_predictions, &filtered_weights)
        }
    }
    fn uncertainty_based_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let mut uncertainty_weights = Vec::new();
        for (i, (_, output, _)) in predictions.iter().enumerate() {
            let confidence = self.extract_confidence(output);
            let uncertainty = 1.0 - confidence;
            let base_weight = weights.get(i).unwrap_or(&1.0);
            let adjusted_weight =
                base_weight * (1.0 - uncertainty * self.config.uncertainty_sampling_rate);
            uncertainty_weights.push(adjusted_weight.max(0.01));
        }
        let sum: f32 = uncertainty_weights.iter().sum();
        if sum > 0.0 {
            uncertainty_weights.iter_mut().for_each(|w| *w /= sum);
        }
        self.average_predictions(predictions, &uncertainty_weights)
    }
    fn adaptive_voting_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        let mut adaptive_weights = Vec::new();
        let learning_rate = self.config.adaptive_learning_rate;
        for (i, (model_id, output, _)) in predictions.iter().enumerate() {
            let confidence = self.extract_confidence(output);
            let base_weight = weights.get(i).unwrap_or(&1.0);
            let recent_performance = self
                .models
                .iter()
                .find(|m| m.model_id == *model_id)
                .map(|m| m.average_performance())
                .unwrap_or(0.5);
            let performance_adjustment = recent_performance * learning_rate;
            let confidence_adjustment = confidence * learning_rate;
            let adaptive_weight =
                base_weight * (1.0 + performance_adjustment + confidence_adjustment);
            adaptive_weights.push(adaptive_weight.max(0.01));
        }
        let sum: f32 = adaptive_weights.iter().sum();
        if sum > 0.0 {
            adaptive_weights.iter_mut().for_each(|w| *w /= sum);
        }
        self.average_predictions(predictions, &adaptive_weights)
    }
    pub(super) fn select_models_for_input(
        &self,
        characteristics: &InputCharacteristics,
    ) -> Vec<&str> {
        if let Some(router) = &self.config.router {
            let routed = router.route(
                characteristics.domain.as_deref().unwrap_or(""),
                &self.model_ids,
            );
            routed
                .into_iter()
                .filter_map(|(i, _)| self.models.get(i).map(|m| m.model_id.as_str()))
                .collect()
        } else {
            // Build a representative input string from input characteristics so that
            // KeywordRouter.route() applies its length-based heuristics correctly.
            // A string of length == characteristics.length triggers the right branch.
            let proxy_input = format!(
                "{}{}",
                " ".repeat(characteristics.length.min(501)),
                characteristics.domain.as_deref().unwrap_or("")
            );
            let keyword_router = KeywordRouter;
            let routed = keyword_router.route(&proxy_input, &self.model_ids);
            let mut selected: Vec<&str> = routed
                .into_iter()
                .filter_map(|(i, _)| self.models.get(i).map(|m| m.model_id.as_str()))
                .collect();
            if selected.is_empty() {
                selected = self.models.iter().map(|m| m.model_id.as_str()).collect();
            }
            selected
        }
    }
    pub(super) fn analyze_input_characteristics(&self, input: &str) -> InputCharacteristics {
        let length = input.len();
        let complexity_score = self.estimate_complexity(input);
        let estimated_time = (length as f32 * 0.1 + complexity_score * 100.0) as u64;
        let required_memory = (length as f32 * 0.001 + complexity_score * 10.0) as u64;
        InputCharacteristics {
            length,
            complexity_score,
            estimated_processing_time: estimated_time,
            required_resource_mb: required_memory,
            domain: self.detect_domain(input),
            language: self.detect_language(input),
        }
    }
    pub(crate) fn estimate_complexity(&self, input: &str) -> f32 {
        let words = input.split_whitespace().count();
        let unique_words = input.split_whitespace().collect::<std::collections::HashSet<_>>().len();
        let avg_word_length =
            if words > 0 { input.chars().count() as f32 / words as f32 } else { 0.0 };
        let lexical_diversity = if words > 0 { unique_words as f32 / words as f32 } else { 0.0 };
        (avg_word_length / 20.0 + lexical_diversity).min(1.0)
    }
    pub(crate) fn detect_domain(&self, input: &str) -> Option<String> {
        KeywordRouter::detect_domain(input)
    }
    pub(crate) fn detect_language(&self, input: &str) -> Option<String> {
        KeywordRouter::detect_language(input)
    }
    pub(super) fn calculate_consensus_score(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> f32 {
        if predictions.len() < 2 {
            return 1.0;
        }
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut label_counts: HashMap<String, u32> = HashMap::new();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top_result) = results.first() {
                            *label_counts.entry(top_result.label.clone()).or_insert(0) += 1;
                        }
                    }
                }
                let max_votes = label_counts.values().max().unwrap_or(&0);
                *max_votes as f32 / predictions.len() as f32
            },
            _ => 0.7,
        }
    }
    pub(super) fn calculate_diversity_score(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> f32 {
        if predictions.len() < 2 {
            return 0.0;
        }
        match &predictions[0].1 {
            PipelineOutput::Classification(_) => {
                let mut unique_labels = std::collections::HashSet::new();
                for (_, output, _) in predictions {
                    if let PipelineOutput::Classification(results) = output {
                        if let Some(top_result) = results.first() {
                            unique_labels.insert(top_result.label.clone());
                        }
                    }
                }
                unique_labels.len() as f32 / predictions.len() as f32
            },
            _ => 0.5,
        }
    }
    pub(super) fn generate_explanation(&self, prediction: &EnsemblePrediction) -> String {
        let mut explanation = String::new();
        explanation.push_str(&format!(
            "Ensemble prediction using {} strategy with {} models. ",
            format!("{:?}", self.config.strategy),
            prediction.models_used.len()
        ));
        explanation.push_str(&format!(
            "Confidence: {:.2}, Consensus: {:.2}, Diversity: {:.2}. ",
            prediction.confidence_score, prediction.consensus_score, prediction.diversity_score
        ));
        explanation.push_str("Model weights: ");
        for weight in &prediction.model_weights {
            explanation.push_str(&format!(
                "{}:{:.2} ",
                weight.model_id,
                weight.total_weight()
            ));
        }
        explanation
    }
    pub(super) fn calculate_uncertainty_score(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let confidences: Vec<f32> = predictions
            .iter()
            .map(|(_, output, _)| self.extract_confidence(output))
            .collect();
        let mean_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter().map(|c| (c - mean_confidence).powi(2)).sum::<f32>()
            / confidences.len() as f32;
        variance.sqrt()
    }
    pub(super) fn calculate_quality_latency_score(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }
        let quality_weight = self.config.quality_latency_weight;
        let latency_weight = 1.0 - quality_weight;
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        for (i, (_, output, duration)) in predictions.iter().enumerate() {
            let quality_score = self.extract_confidence(output);
            let latency_score = 1.0 / (1.0 + *duration as f32 / 1000.0);
            let combined_score = quality_score * quality_weight + latency_score * latency_weight;
            let weight = weights.get(i).unwrap_or(&1.0);
            total_score += combined_score * weight;
            total_weight += weight;
        }
        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }
    pub(super) fn estimate_resource_usage(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        characteristics: &InputCharacteristics,
    ) -> u64 {
        let base_memory = characteristics.required_resource_mb;
        let model_overhead = predictions.len() as u64 * 50;
        let processing_overhead = characteristics.length as u64 / 1000;
        base_memory + model_overhead + processing_overhead
    }
}
