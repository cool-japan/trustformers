use crate::error::{Result, TrustformersError};
use crate::pipeline::{Pipeline, PipelineOptions, PipelineOutput};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelSelectionStrategy {
    /// Use all models in ensemble
    All,
    /// Select top-k performing models
    TopK(usize),
    /// Select models based on confidence threshold
    ConfidenceBased,
    /// Select models based on resource constraints
    ResourceConstrained,
    /// Dynamic selection based on input characteristics
    Dynamic,
    /// Random selection for diversity
    Random(usize),
    /// Select models based on prediction agreement
    AgreementBased,
}

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
    // New advanced configuration options
    pub cascade_early_exit_threshold: f32,
    pub cascade_max_models: usize,
    pub quality_latency_weight: f32,
    pub resource_budget_mb: u64,
    pub uncertainty_sampling_rate: f32,
    pub adaptive_learning_rate: f32,
    pub routing_features: Vec<String>,
    pub enable_model_selection: bool,
    pub model_selection_strategy: ModelSelectionStrategy,
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
            // New advanced defaults
            cascade_early_exit_threshold: 0.8,
            cascade_max_models: 3,
            quality_latency_weight: 0.5,
            resource_budget_mb: 2048,
            uncertainty_sampling_rate: 0.1,
            adaptive_learning_rate: 0.01,
            routing_features: vec!["input_length".to_string(), "complexity".to_string()],
            enable_model_selection: false,
            model_selection_strategy: ModelSelectionStrategy::All,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelWeight {
    pub model_id: String,
    pub weight: f32,
    pub confidence_weight: f32,
    pub accuracy_weight: f32,
    pub dynamic_weight: f32,
}

impl ModelWeight {
    pub fn new(model_id: String, weight: f32) -> Self {
        Self {
            model_id,
            weight,
            confidence_weight: 1.0,
            accuracy_weight: 1.0,
            dynamic_weight: 1.0,
        }
    }

    pub fn total_weight(&self) -> f32 {
        self.weight * self.confidence_weight * self.accuracy_weight * self.dynamic_weight
    }
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
    // New advanced prediction fields
    pub uncertainty_score: f32,
    pub resource_usage_mb: u64,
    pub quality_latency_score: f32,
    pub early_exit_triggered: bool,
    pub routing_decision: Option<String>,
    pub model_selection_info: Option<ModelSelectionInfo>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelSelectionInfo {
    pub selected_models: Vec<String>,
    pub selection_reason: String,
    pub selection_confidence: f32,
    pub alternative_models: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeResult {
    pub predictions: Vec<PipelineOutput>,
    pub exit_at_model: usize,
    pub cumulative_confidence: f32,
    pub processing_times: Vec<u64>,
    pub resource_usage: Vec<u64>,
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

pub struct EnsemblePipeline {
    config: EnsembleConfig,
    models: Vec<EnsembleModel>,
    meta_learner: Option<Box<dyn Pipeline<Input = String, Output = PipelineOutput>>>,
    calibration_data: Vec<(String, PipelineOutput)>,
    performance_tracker: HashMap<String, Vec<f32>>,
}

impl EnsemblePipeline {
    pub fn new(config: EnsembleConfig) -> Self {
        Self {
            config,
            models: Vec::new(),
            meta_learner: None,
            calibration_data: Vec::new(),
            performance_tracker: HashMap::new(),
        }
    }

    pub fn add_model(
        &mut self,
        model_id: String,
        pipeline: Box<dyn Pipeline<Input = String, Output = PipelineOutput>>,
        weight: f32,
    ) -> Result<()> {
        let ensemble_model = EnsembleModel::new(model_id.clone(), pipeline, weight);
        self.models.push(ensemble_model);
        self.performance_tracker.insert(model_id, Vec::new());
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

    fn predict_individual_models(&self, input: &str) -> Result<Vec<(String, PipelineOutput, u64)>> {
        let mut predictions = Vec::new();

        if self.config.parallel_execution {
            // Parallel execution (simulated for now)
            for model in &self.models {
                let start_time = std::time::Instant::now();
                let prediction = model.pipeline.__call__(input.to_string())?;
                let duration = start_time.elapsed().as_millis() as u64;
                predictions.push((model.model_id.clone(), prediction, duration));
            }
        } else {
            // Sequential execution
            for model in &self.models {
                let start_time = std::time::Instant::now();
                let prediction = model.pipeline.__call__(input.to_string())?;
                let duration = start_time.elapsed().as_millis() as u64;
                predictions.push((model.model_id.clone(), prediction, duration));
            }
        }

        Ok(predictions)
    }

    fn calculate_dynamic_weights(
        &mut self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Vec<f32> {
        let mut weights = Vec::new();

        for (model_id, output, _) in predictions {
            let confidence = self.extract_confidence(output);

            // Update model performance
            if let Some(model) = self.models.iter_mut().find(|m| m.model_id == *model_id) {
                model.weight.confidence_weight = confidence;
                model.weight.accuracy_weight = model.accuracy();

                // Dynamic weight based on recent performance
                let recent_performance = model.average_performance();
                model.weight.dynamic_weight = (recent_performance + confidence) / 2.0;
            }

            // Calculate final weight
            let model_weight = self
                .models
                .iter()
                .find(|m| m.model_id == *model_id)
                .map(|m| m.weight.total_weight())
                .unwrap_or(1.0);

            weights.push(model_weight);
        }

        // Normalize weights
        let sum: f32 = weights.iter().sum();
        if sum > 0.0 {
            weights.iter_mut().for_each(|w| *w /= sum);
        }

        weights
    }

    fn extract_confidence(&self, output: &PipelineOutput) -> f32 {
        match output {
            PipelineOutput::Classification(results) => {
                results.iter().map(|r| r.score).fold(0.0f32, f32::max)
            },
            PipelineOutput::QuestionAnswering(result) => result.score,
            PipelineOutput::FillMask(results) => {
                results.iter().map(|r| r.score).fold(0.0f32, f32::max)
            },
            _ => 0.8, // Default confidence for outputs without scores
        }
    }

    fn apply_ensemble_strategy(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
        input_characteristics: &InputCharacteristics,
    ) -> Result<PipelineOutput> {
        match &self.config.strategy {
            EnsembleStrategy::Average => self.average_predictions(predictions, weights),
            EnsembleStrategy::WeightedAverage(custom_weights) => {
                self.weighted_average_predictions(predictions, custom_weights)
            },
            EnsembleStrategy::MajorityVote => self.majority_vote_predictions(predictions),
            EnsembleStrategy::Maximum => self.maximum_predictions(predictions),
            EnsembleStrategy::Minimum => self.minimum_predictions(predictions),
            EnsembleStrategy::Stacking => self.stacking_predictions(predictions),
            EnsembleStrategy::DynamicWeighting => self.average_predictions(predictions, weights),
            EnsembleStrategy::RankFusion => self.rank_fusion_predictions(predictions),
            EnsembleStrategy::MixtureOfExperts => {
                self.mixture_of_experts_predictions(predictions, weights)
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
            _ => self.average_predictions(predictions, weights), // Fallback
        }
    }

    fn average_predictions(
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
                // For other types, return the highest weighted prediction
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

        // Sort by score descending
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

    fn majority_vote_predictions(
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
                // For non-classification tasks, fall back to average
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
            _ => {
                // For other types, return first prediction
                Ok(predictions[0].1.clone())
            },
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
            _ => {
                // For other types, return first prediction
                Ok(predictions[0].1.clone())
            },
        }
    }

    fn stacking_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
    ) -> Result<PipelineOutput> {
        if let Some(meta_learner) = &self.meta_learner {
            // Create a feature vector from all predictions
            let features = self.create_stacking_features(predictions)?;
            meta_learner.__call__(features)
        } else {
            // Fall back to average if no meta-learner
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
                            let rank_score = 1.0 / (rank + 1) as f32; // Reciprocal rank
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
                // Fall back to average for non-classification
                let weights = vec![1.0 / predictions.len() as f32; predictions.len()];
                self.average_predictions(predictions, &weights)
            },
        }
    }

    fn mixture_of_experts_predictions(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        weights: &[f32],
    ) -> Result<PipelineOutput> {
        // For now, implement as a weighted average with expert selection
        // In a full implementation, this would include gating networks
        self.average_predictions(predictions, weights)
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

            // Add timing feature
            features.push(format!("{}:time:{}", model_id, duration));
        }

        Ok(features.join("|"))
    }

    // New advanced ensemble strategy implementations
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

            // Early exit if confidence threshold is met
            if cumulative_confidence >= threshold {
                break;
            }
        }

        // Return the average of used predictions
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
        // Route based on input characteristics
        let selected_models = self.select_models_for_input(input_characteristics);

        let filtered_predictions: Vec<(String, PipelineOutput, u64)> = predictions
            .iter()
            .filter(|(model_id, _, _)| selected_models.contains(&model_id.as_str()))
            .cloned()
            .collect();

        if filtered_predictions.is_empty() {
            // Fallback to all models if no matches
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
            let latency_score = 1.0 / (1.0 + *duration as f32 / 1000.0); // Normalize latency

            let combined_score = quality_score * quality_weight + latency_score * latency_weight;
            let base_weight = weights.get(i).unwrap_or(&1.0);

            optimized_weights.push(base_weight * combined_score);
        }

        // Normalize weights
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
            // Use all models if within budget
            self.average_predictions(predictions, weights)
        } else {
            // Select subset of models based on resource efficiency
            let mut model_efficiency: Vec<(usize, f32)> = predictions
                .iter()
                .enumerate()
                .map(|(i, (_, output, duration))| {
                    let quality = self.extract_confidence(output);
                    let efficiency = quality / (*duration as f32 + 1.0);
                    (i, efficiency)
                })
                .collect();

            // Sort by efficiency descending
            model_efficiency
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Select top models within budget
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
                selected_indices.push(model_efficiency[0].0); // At least use the most efficient
            }

            // Create filtered predictions and weights
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
        // Calculate uncertainty for each prediction
        let mut uncertainty_weights = Vec::new();

        for (i, (_, output, _)) in predictions.iter().enumerate() {
            let confidence = self.extract_confidence(output);
            let uncertainty = 1.0 - confidence;
            let base_weight = weights.get(i).unwrap_or(&1.0);

            // Weight models with higher uncertainty less
            let adjusted_weight =
                base_weight * (1.0 - uncertainty * self.config.uncertainty_sampling_rate);
            uncertainty_weights.push(adjusted_weight.max(0.01)); // Minimum weight
        }

        // Normalize weights
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
        // Adapt weights based on recent performance and confidence
        let mut adaptive_weights = Vec::new();
        let learning_rate = self.config.adaptive_learning_rate;

        for (i, (model_id, output, _)) in predictions.iter().enumerate() {
            let confidence = self.extract_confidence(output);
            let base_weight = weights.get(i).unwrap_or(&1.0);

            // Get model's recent performance
            let recent_performance = self
                .models
                .iter()
                .find(|m| m.model_id == *model_id)
                .map(|m| m.average_performance())
                .unwrap_or(0.5);

            // Adaptive weight update
            let performance_adjustment = recent_performance * learning_rate;
            let confidence_adjustment = confidence * learning_rate;

            let adaptive_weight =
                base_weight * (1.0 + performance_adjustment + confidence_adjustment);
            adaptive_weights.push(adaptive_weight.max(0.01));
        }

        // Normalize weights
        let sum: f32 = adaptive_weights.iter().sum();
        if sum > 0.0 {
            adaptive_weights.iter_mut().for_each(|w| *w /= sum);
        }

        self.average_predictions(predictions, &adaptive_weights)
    }

    // Helper methods for new strategies
    fn select_models_for_input(&self, characteristics: &InputCharacteristics) -> Vec<&str> {
        // Simple routing logic based on input characteristics
        let mut selected_models = Vec::new();

        for model in &self.models {
            let should_select = match characteristics.length {
                0..=100 => model.model_id.contains("small") || model.model_id.contains("fast"),
                101..=500 => !model.model_id.contains("large"),
                _ => true, // Use all models for long inputs
            };

            if should_select {
                selected_models.push(model.model_id.as_str());
            }
        }

        if selected_models.is_empty() {
            // Fallback to all models
            self.models.iter().map(|m| m.model_id.as_str()).collect()
        } else {
            selected_models
        }
    }

    fn analyze_input_characteristics(&self, input: &str) -> InputCharacteristics {
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

    fn estimate_complexity(&self, input: &str) -> f32 {
        // Simple complexity estimation based on linguistic features
        let words = input.split_whitespace().count();
        let unique_words = input.split_whitespace().collect::<std::collections::HashSet<_>>().len();
        let avg_word_length =
            if words > 0 { input.chars().count() as f32 / words as f32 } else { 0.0 };

        let lexical_diversity = if words > 0 { unique_words as f32 / words as f32 } else { 0.0 };

        // Complexity score between 0 and 1
        (avg_word_length / 20.0 + lexical_diversity).min(1.0)
    }

    fn detect_domain(&self, input: &str) -> Option<String> {
        let input_lower = input.to_lowercase();

        if input_lower.contains("medical")
            || input_lower.contains("patient")
            || input_lower.contains("diagnosis")
        {
            Some("medical".to_string())
        } else if input_lower.contains("legal")
            || input_lower.contains("contract")
            || input_lower.contains("law")
        {
            Some("legal".to_string())
        } else if input_lower.contains("science")
            || input_lower.contains("research")
            || input_lower.contains("experiment")
        {
            Some("scientific".to_string())
        } else if input_lower.contains("code")
            || input_lower.contains("programming")
            || input_lower.contains("function")
        {
            Some("technical".to_string())
        } else {
            None
        }
    }

    fn detect_language(&self, input: &str) -> Option<String> {
        // Simple language detection based on character patterns
        let has_chinese = input.chars().any(|c| ('\u{4e00}'..='\u{9fff}').contains(&c));
        let has_arabic = input.chars().any(|c| ('\u{0600}'..='\u{06ff}').contains(&c));
        let has_cyrillic = input.chars().any(|c| ('\u{0400}'..='\u{04ff}').contains(&c));

        if has_chinese {
            Some("zh".to_string())
        } else if has_arabic {
            Some("ar".to_string())
        } else if has_cyrillic {
            Some("ru".to_string())
        } else {
            Some("en".to_string()) // Default to English
        }
    }

    fn calculate_consensus_score(&self, predictions: &[(String, PipelineOutput, u64)]) -> f32 {
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
            _ => {
                // For other types, assume moderate consensus
                0.7
            },
        }
    }

    fn calculate_diversity_score(&self, predictions: &[(String, PipelineOutput, u64)]) -> f32 {
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
            _ => {
                // For other types, assume moderate diversity
                0.5
            },
        }
    }

    fn generate_explanation(&self, prediction: &EnsemblePrediction) -> String {
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

    // Additional helper methods for advanced features
    fn calculate_uncertainty_score(&self, predictions: &[(String, PipelineOutput, u64)]) -> f32 {
        if predictions.is_empty() {
            return 0.0;
        }

        let confidences: Vec<f32> = predictions
            .iter()
            .map(|(_, output, _)| self.extract_confidence(output))
            .collect();

        // Calculate uncertainty as variance in confidence scores
        let mean_confidence = confidences.iter().sum::<f32>() / confidences.len() as f32;
        let variance = confidences.iter().map(|c| (c - mean_confidence).powi(2)).sum::<f32>()
            / confidences.len() as f32;

        variance.sqrt() // Standard deviation as uncertainty measure
    }

    fn calculate_quality_latency_score(
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
            let latency_score = 1.0 / (1.0 + *duration as f32 / 1000.0); // Normalize latency

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

    fn estimate_resource_usage(
        &self,
        predictions: &[(String, PipelineOutput, u64)],
        characteristics: &InputCharacteristics,
    ) -> u64 {
        // Simple resource usage estimation
        let base_memory = characteristics.required_resource_mb;
        let model_overhead = predictions.len() as u64 * 50; // 50MB per model overhead
        let processing_overhead = characteristics.length as u64 / 1000; // 1MB per 1000 characters

        base_memory + model_overhead + processing_overhead
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

        // Analyze input characteristics for advanced strategies
        let input_characteristics = self.analyze_input_characteristics(&input);

        // Get predictions from all models
        let predictions = self.predict_individual_models(&input)?;

        // Calculate dynamic weights
        let weights = match &self.config.strategy {
            EnsembleStrategy::DynamicWeighting => {
                // For dynamic weighting, we need mutable access, but we're in an immutable method
                // In a real implementation, this would use interior mutability (RefCell, etc.)
                let confidence_weights: Vec<f32> = predictions
                    .iter()
                    .map(|(_, output, _)| self.extract_confidence(output))
                    .collect();

                // Normalize confidence weights
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

        // Apply ensemble strategy with input characteristics
        let final_prediction =
            self.apply_ensemble_strategy(&predictions, &weights, &input_characteristics)?;

        // Calculate scores
        let confidence_score = self.extract_confidence(&final_prediction);
        let consensus_score = self.calculate_consensus_score(&predictions);
        let diversity_score = self.calculate_diversity_score(&predictions);
        let uncertainty_score = self.calculate_uncertainty_score(&predictions);
        let quality_latency_score = self.calculate_quality_latency_score(&predictions, &weights);

        // Calculate resource usage
        let resource_usage_mb = self.estimate_resource_usage(&predictions, &input_characteristics);

        // Create model weights for output
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

        // Determine if early exit was triggered (for cascade strategy)
        let early_exit_triggered =
            matches!(self.config.strategy, EnsembleStrategy::CascadePipeline)
                && predictions.len() < self.models.len();

        // Create routing decision info
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

        // Create model selection info if enabled
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
            // New advanced fields
            uncertainty_score,
            resource_usage_mb,
            quality_latency_score,
            early_exit_triggered,
            routing_decision,
            model_selection_info,
        };

        // Generate explanation if requested
        if self.config.enable_explanation {
            ensemble_prediction.explanation = Some(self.generate_explanation(&ensemble_prediction));
        }

        Ok(ensemble_prediction)
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        inputs.into_iter().map(|input| self.__call__(input)).collect()
    }
}

// Factory functions for creating ensemble pipelines
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

// New advanced factory functions
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
    config.quality_latency_weight = 0.3; // Favor latency over quality
    config.enable_model_selection = true;
    config.model_selection_strategy = ModelSelectionStrategy::ResourceConstrained;

    EnsemblePipeline::new(config)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ensemble_config_default() {
        let config = EnsembleConfig::default();
        assert_eq!(config.confidence_threshold, 0.5);
        assert_eq!(config.consensus_threshold, 0.7);
    }

    #[test]
    fn test_model_weight_calculation() {
        let weight = ModelWeight {
            model_id: "test_model".to_string(),
            weight: 0.5,
            confidence_weight: 0.8,
            accuracy_weight: 0.9,
            dynamic_weight: 1.1,
        };

        assert!((weight.total_weight() - 0.396).abs() < 0.001);
    }

    #[test]
    fn test_ensemble_pipeline_creation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);
        assert_eq!(ensemble.models.len(), 0);
    }

    #[test]
    fn test_advanced_ensemble_config() {
        let config = EnsembleConfig::default();
        assert_eq!(config.cascade_early_exit_threshold, 0.8);
        assert_eq!(config.cascade_max_models, 3);
        assert_eq!(config.resource_budget_mb, 2048);
        assert_eq!(config.uncertainty_sampling_rate, 0.1);
    }

    #[test]
    fn test_model_selection_strategy() {
        let strategy = ModelSelectionStrategy::TopK(3);
        match strategy {
            ModelSelectionStrategy::TopK(k) => assert_eq!(k, 3),
            _ => panic!("Unexpected strategy type"),
        }
    }

    #[test]
    fn test_input_characteristics_analysis() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);

        let input = "This is a test input for complexity analysis";
        let characteristics = ensemble.analyze_input_characteristics(input);

        assert!(characteristics.length > 0);
        assert!(characteristics.complexity_score >= 0.0);
        assert!(characteristics.complexity_score <= 1.0);
        assert!(characteristics.estimated_processing_time > 0);
    }

    #[test]
    fn test_domain_detection() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);

        assert_eq!(
            ensemble.detect_domain("medical diagnosis of patient"),
            Some("medical".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("legal contract review"),
            Some("legal".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("science research experiment"),
            Some("scientific".to_string())
        );
        assert_eq!(
            ensemble.detect_domain("programming code function"),
            Some("technical".to_string())
        );
        assert_eq!(ensemble.detect_domain("general conversation"), None);
    }

    #[test]
    fn test_language_detection() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);

        assert_eq!(
            ensemble.detect_language("Hello world"),
            Some("en".to_string())
        );
        assert_eq!(ensemble.detect_language("你好世界"), Some("zh".to_string()));
        assert_eq!(
            ensemble.detect_language("مرحبا بالعالم"),
            Some("ar".to_string())
        );
        assert_eq!(
            ensemble.detect_language("Привет мир"),
            Some("ru".to_string())
        );
    }

    #[test]
    fn test_cascade_ensemble_creation() {
        let ensemble = create_cascade_ensemble(0.9, 2);
        match ensemble.config.strategy {
            EnsembleStrategy::CascadePipeline => {},
            _ => panic!("Expected CascadePipeline strategy"),
        }
        assert_eq!(ensemble.config.cascade_early_exit_threshold, 0.9);
        assert_eq!(ensemble.config.cascade_max_models, 2);
    }

    #[test]
    fn test_dynamic_routing_ensemble_creation() {
        let features = vec!["length".to_string(), "complexity".to_string()];
        let ensemble = create_dynamic_routing_ensemble(features.clone());

        match ensemble.config.strategy {
            EnsembleStrategy::DynamicRouting => {},
            _ => panic!("Expected DynamicRouting strategy"),
        }
        assert_eq!(ensemble.config.routing_features, features);
        assert!(ensemble.config.enable_model_selection);
    }

    #[test]
    fn test_quality_latency_ensemble_creation() {
        let ensemble = create_quality_latency_ensemble(0.7);

        match ensemble.config.strategy {
            EnsembleStrategy::QualityLatencyOptimized => {},
            _ => panic!("Expected QualityLatencyOptimized strategy"),
        }
        assert_eq!(ensemble.config.quality_latency_weight, 0.7);
    }

    #[test]
    fn test_resource_aware_ensemble_creation() {
        let ensemble = create_resource_aware_ensemble(1024);

        match ensemble.config.strategy {
            EnsembleStrategy::ResourceAware => {},
            _ => panic!("Expected ResourceAware strategy"),
        }
        assert_eq!(ensemble.config.resource_budget_mb, 1024);
    }

    #[test]
    fn test_uncertainty_ensemble_creation() {
        let ensemble = create_uncertainty_ensemble(0.2);

        match ensemble.config.strategy {
            EnsembleStrategy::UncertaintyBased => {},
            _ => panic!("Expected UncertaintyBased strategy"),
        }
        assert_eq!(ensemble.config.uncertainty_sampling_rate, 0.2);
    }

    #[test]
    fn test_adaptive_voting_ensemble_creation() {
        let ensemble = create_adaptive_voting_ensemble(0.05);

        match ensemble.config.strategy {
            EnsembleStrategy::AdaptiveVoting => {},
            _ => panic!("Expected AdaptiveVoting strategy"),
        }
        assert_eq!(ensemble.config.adaptive_learning_rate, 0.05);
    }

    #[test]
    fn test_high_performance_ensemble_creation() {
        let ensemble = create_high_performance_ensemble();

        assert!(ensemble.config.parallel_execution);
        assert_eq!(ensemble.config.max_concurrent_models, 8);
        assert!(ensemble.config.enable_diversity_boost);
        assert!(ensemble.config.enable_calibration);
        assert!(ensemble.config.enable_explanation);
        assert!(ensemble.config.enable_model_selection);
    }

    #[test]
    fn test_efficient_ensemble_creation() {
        let ensemble = create_efficient_ensemble();

        match ensemble.config.strategy {
            EnsembleStrategy::CascadePipeline => {},
            _ => panic!("Expected CascadePipeline strategy"),
        }
        assert_eq!(ensemble.config.cascade_early_exit_threshold, 0.85);
        assert_eq!(ensemble.config.cascade_max_models, 2);
        assert_eq!(ensemble.config.resource_budget_mb, 1024);
        assert_eq!(ensemble.config.quality_latency_weight, 0.3);
    }

    #[test]
    fn test_model_selection_info() {
        let info = ModelSelectionInfo {
            selected_models: vec!["model1".to_string(), "model2".to_string()],
            selection_reason: "Top performers".to_string(),
            selection_confidence: 0.85,
            alternative_models: vec!["model3".to_string()],
        };

        assert_eq!(info.selected_models.len(), 2);
        assert_eq!(info.alternative_models.len(), 1);
        assert!(info.selection_confidence > 0.8);
    }

    #[test]
    fn test_input_characteristics() {
        let characteristics = InputCharacteristics {
            length: 100,
            complexity_score: 0.5,
            estimated_processing_time: 50,
            required_resource_mb: 256,
            domain: Some("technical".to_string()),
            language: Some("en".to_string()),
        };

        assert_eq!(characteristics.length, 100);
        assert_eq!(characteristics.complexity_score, 0.5);
        assert_eq!(characteristics.required_resource_mb, 256);
        assert_eq!(characteristics.domain, Some("technical".to_string()));
    }

    #[test]
    fn test_complexity_estimation() {
        let config = EnsembleConfig::default();
        let ensemble = EnsemblePipeline::new(config);

        let simple_text = "Hello world";
        let complex_text = "The comprehensive implementation of advanced machine learning algorithms requires sophisticated understanding of mathematical foundations and computational complexity theory";

        let simple_complexity = ensemble.estimate_complexity(simple_text);
        let complex_complexity = ensemble.estimate_complexity(complex_text);

        assert!(simple_complexity >= 0.0 && simple_complexity <= 1.0);
        assert!(complex_complexity >= 0.0 && complex_complexity <= 1.0);
        // Complex text should generally have higher complexity
        assert!(complex_complexity >= simple_complexity);
    }
}
