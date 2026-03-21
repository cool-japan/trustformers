//! Enhanced Optimization Components
//!
//! Supporting components for optimization engine including strategy selection,
//! impact assessment, and adaptive learning

use anyhow::Result;
use chrono::{DateTime, Utc};
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicBool, Ordering},
        Arc,
    },
    time::Duration,
};
use tracing::info;

use super::super::types::*;
use super::support::TrainingExample;

// =============================================================================

/// Impact assessor for optimization recommendations
///
/// Evaluates and predicts the impact of optimization recommendations
/// before implementation to ensure positive outcomes.
pub struct ImpactAssessor {
    assessment_history: Arc<Mutex<VecDeque<ImpactAssessmentRecord>>>,
    ml_model: Arc<ImpactPredictionModel>,
    config: Arc<RwLock<ImpactAssessorConfig>>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct ImpactAssessmentRecord {
    recommendation_id: String,
    timestamp: DateTime<Utc>,
    predicted_impact: ImpactAssessment,
    actual_impact: Option<ImpactAssessment>,
    accuracy_score: Option<f32>,
}

/// Impact prediction model using machine learning
pub struct ImpactPredictionModel {
    feature_weights: HashMap<String, f32>,
    historical_accuracy: f32,
    model_version: String,
}

#[derive(Debug, Clone)]
pub struct ImpactAssessorConfig {
    pub prediction_horizon: Duration,
    pub confidence_threshold: f32,
    pub max_history_size: usize,
    pub model_update_interval: Duration,
}

impl Default for ImpactAssessorConfig {
    fn default() -> Self {
        Self {
            prediction_horizon: Duration::from_secs(3600), // 1 hour
            confidence_threshold: 0.7,
            max_history_size: 10000,
            model_update_interval: Duration::from_secs(86400), // 24 hours
        }
    }
}

impl ImpactAssessor {
    pub async fn new() -> Result<Self> {
        let model = ImpactPredictionModel::new().await?;

        Ok(Self {
            assessment_history: Arc::new(Mutex::new(VecDeque::new())),
            ml_model: Arc::new(model),
            config: Arc::new(RwLock::new(ImpactAssessorConfig::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting impact assessor");
        Ok(())
    }

    /// Assess the impact of an optimization recommendation
    pub async fn assess_impact(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<ImpactAssessment> {
        let features = self.extract_features(recommendation).await?;
        let predicted_impact = self.ml_model.predict_impact(&features).await?;

        // Store assessment record
        let record = ImpactAssessmentRecord {
            recommendation_id: recommendation.id.clone(),
            timestamp: Utc::now(),
            predicted_impact: predicted_impact.clone(),
            actual_impact: None,
            accuracy_score: None,
        };

        let mut history = self.assessment_history.lock();
        history.push_back(record);

        // Limit history size
        let config = self.config.read();
        if history.len() > config.max_history_size {
            history.pop_front();
        }

        Ok(predicted_impact)
    }

    /// Update assessment with actual outcome
    pub async fn update_with_outcome(
        &self,
        recommendation_id: &str,
        actual_impact: ImpactAssessment,
    ) -> Result<()> {
        let mut history = self.assessment_history.lock();

        if let Some(record) = history.iter_mut().find(|r| r.recommendation_id == recommendation_id)
        {
            let actual_impact_clone = actual_impact.clone();
            record.actual_impact = Some(actual_impact);
            record.accuracy_score =
                Some(self.calculate_accuracy_score(&record.predicted_impact, &actual_impact_clone));
        }

        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("Impact assessor shutdown complete");
        Ok(())
    }

    async fn extract_features(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Result<HashMap<String, f32>> {
        let mut features = HashMap::new();

        // Basic recommendation features
        features.insert("confidence".to_string(), recommendation.confidence);
        features.insert("priority".to_string(), recommendation.priority as f32);
        features.insert(
            "action_count".to_string(),
            recommendation.actions.len() as f32,
        );
        features.insert("risk_count".to_string(), recommendation.risks.len() as f32);

        // Impact features
        features.insert(
            "expected_performance_impact".to_string(),
            recommendation.expected_impact.performance_impact,
        );
        features.insert(
            "expected_resource_impact".to_string(),
            recommendation.expected_impact.resource_impact,
        );
        features.insert(
            "complexity".to_string(),
            recommendation.expected_impact.complexity,
        );
        features.insert(
            "risk_level".to_string(),
            recommendation.expected_impact.risk_level,
        );

        // Action type features
        let action_type_weights = self.get_action_type_weights();
        for action in &recommendation.actions {
            if let Some(&weight) = action_type_weights.get(&format!("{:?}", action.action_type)) {
                features.insert(format!("action_{:?}", action.action_type), weight);
            }
        }

        Ok(features)
    }

    fn get_action_type_weights(&self) -> HashMap<String, f32> {
        let mut weights = HashMap::new();
        weights.insert("AdjustParallelism".to_string(), 0.8);
        weights.insert("AdjustResources".to_string(), 0.7);
        weights.insert("OptimizeMemoryUsage".to_string(), 0.6);
        weights.insert("AdjustBatchSize".to_string(), 0.5);
        weights.insert("TuneParameters".to_string(), 0.4);
        weights
    }

    fn calculate_accuracy_score(
        &self,
        predicted: &ImpactAssessment,
        actual: &ImpactAssessment,
    ) -> f32 {
        let performance_accuracy =
            1.0 - (predicted.performance_impact - actual.performance_impact).abs();
        let resource_accuracy = 1.0 - (predicted.resource_impact - actual.resource_impact).abs();
        let complexity_accuracy = 1.0 - (predicted.complexity - actual.complexity).abs();
        let risk_accuracy = 1.0 - (predicted.risk_level - actual.risk_level).abs();

        (performance_accuracy + resource_accuracy + complexity_accuracy + risk_accuracy) / 4.0
    }
}

impl ImpactPredictionModel {
    pub async fn new() -> Result<Self> {
        let mut feature_weights = HashMap::new();

        // Initialize with reasonable defaults
        feature_weights.insert("confidence".to_string(), 0.3);
        feature_weights.insert("priority".to_string(), 0.2);
        feature_weights.insert("complexity".to_string(), -0.1);
        feature_weights.insert("risk_level".to_string(), -0.2);
        feature_weights.insert("action_count".to_string(), 0.1);

        Ok(Self {
            feature_weights,
            historical_accuracy: 0.75,
            model_version: "v1.0".to_string(),
        })
    }

    pub async fn predict_impact(
        &self,
        features: &HashMap<String, f32>,
    ) -> Result<ImpactAssessment> {
        let mut performance_impact = 0.0;
        let mut resource_impact = 0.0;
        let mut complexity = 0.5;
        let mut risk_level = 0.3;

        // Simple linear model prediction
        for (feature, &value) in features {
            if let Some(&weight) = self.feature_weights.get(feature) {
                performance_impact += value * weight * 0.5;
                resource_impact += value * weight * 0.3;

                match feature.as_str() {
                    "complexity" => complexity = value,
                    "risk_level" => risk_level = value,
                    _ => {},
                }
            }
        }

        // Normalize values
        performance_impact = performance_impact.clamp(-1.0, 1.0);
        resource_impact = resource_impact.clamp(-1.0, 1.0);
        complexity = complexity.clamp(0.0, 1.0);
        risk_level = risk_level.clamp(0.0, 1.0);

        let estimated_benefit = (performance_impact + 1.0) / 2.0; // Convert to 0-1 range

        Ok(ImpactAssessment {
            performance_impact,
            resource_impact,
            complexity,
            risk_level,
            estimated_benefit,
            implementation_time: Duration::from_secs(120), // Default implementation time
        })
    }
}

/// Strategy selector for choosing optimal optimization algorithms
///
/// Intelligently selects the most appropriate optimization algorithms
/// based on system characteristics and historical performance.
pub struct StrategySelector {
    algorithm_performance: Arc<Mutex<HashMap<String, StrategyPerformance>>>,
    selection_history: Arc<Mutex<VecDeque<SelectionRecord>>>,
    config: Arc<RwLock<StrategySelectorConfig>>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct StrategyPerformance {
    algorithm_name: String,
    success_rate: f32,
    average_impact: f32,
    usage_count: u64,
    last_used: DateTime<Utc>,
    effectiveness_score: f32,
}

#[derive(Debug, Clone)]
struct SelectionRecord {
    timestamp: DateTime<Utc>,
    context_hash: String,
    selected_algorithms: Vec<String>,
    outcome_success: Option<bool>,
}

#[derive(Debug, Clone)]
pub struct StrategySelectorConfig {
    pub max_algorithms_per_selection: usize,
    pub min_success_rate_threshold: f32,
    pub performance_weight: f32,
    pub recency_weight: f32,
    pub diversity_bonus: f32,
}

impl Default for StrategySelectorConfig {
    fn default() -> Self {
        Self {
            max_algorithms_per_selection: 4,
            min_success_rate_threshold: 0.6,
            performance_weight: 0.4,
            recency_weight: 0.3,
            diversity_bonus: 0.1,
        }
    }
}

impl StrategySelector {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            algorithm_performance: Arc::new(Mutex::new(HashMap::new())),
            selection_history: Arc::new(Mutex::new(VecDeque::new())),
            config: Arc::new(RwLock::new(StrategySelectorConfig::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting strategy selector");
        self.initialize_algorithm_performance().await?;
        Ok(())
    }

    /// Select optimal algorithms for given optimization context
    pub async fn select_algorithms(&self, context: &OptimizationContext) -> Result<Vec<String>> {
        let context_hash = self.calculate_context_hash(context);
        let performance_map = self.algorithm_performance.lock();
        let config = self.config.read();

        let mut algorithm_scores: Vec<(String, f32)> = performance_map
            .iter()
            .map(|(name, perf)| {
                let score = self.calculate_algorithm_score(perf, context);
                (name.clone(), score)
            })
            .collect();

        // Sort by score descending
        algorithm_scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Select top algorithms
        let selected: Vec<String> = algorithm_scores
            .into_iter()
            .filter(|(_, score)| *score >= config.min_success_rate_threshold)
            .take(config.max_algorithms_per_selection)
            .map(|(name, _)| name)
            .collect();

        // Record selection
        let record = SelectionRecord {
            timestamp: Utc::now(),
            context_hash,
            selected_algorithms: selected.clone(),
            outcome_success: None,
        };

        let mut history = self.selection_history.lock();
        history.push_back(record);

        info!("Selected {} algorithms for optimization", selected.len());
        Ok(selected)
    }

    /// Update strategy weights based on performance feedback
    pub async fn update_strategy_weights(&self) -> Result<()> {
        let mut performance_map = self.algorithm_performance.lock();

        for (_, performance) in performance_map.iter_mut() {
            // Update effectiveness score based on recent performance
            performance.effectiveness_score = self.calculate_effectiveness_score(performance);
        }

        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("Strategy selector shutdown complete");
        Ok(())
    }

    async fn initialize_algorithm_performance(&self) -> Result<()> {
        let mut performance_map = self.algorithm_performance.lock();

        let algorithms = vec![
            "parallelism_optimization",
            "resource_optimization",
            "batching_optimization",
            "performance_tuning",
            "memory_optimization",
            "io_optimization",
            "network_optimization",
            "threadpool_optimization",
        ];

        for algorithm in algorithms {
            performance_map.insert(
                algorithm.to_string(),
                StrategyPerformance {
                    algorithm_name: algorithm.to_string(),
                    success_rate: 0.75,  // Default success rate
                    average_impact: 0.5, // Default impact
                    usage_count: 0,
                    last_used: Utc::now() - chrono::Duration::days(1),
                    effectiveness_score: 0.75,
                },
            );
        }

        Ok(())
    }

    fn calculate_context_hash(&self, context: &OptimizationContext) -> String {
        // Simple hash based on context characteristics
        format!(
            "cores_{}_objectives_{}_constraints_{}",
            context.system_state.available_cores,
            context.objectives.len(),
            context.constraints.len()
        )
    }

    fn calculate_algorithm_score(
        &self,
        performance: &StrategyPerformance,
        context: &OptimizationContext,
    ) -> f32 {
        let config = self.config.read();

        // Base score from success rate and impact
        let performance_score = performance.success_rate * config.performance_weight
            + performance.average_impact * (1.0 - config.performance_weight);

        // Recency bonus (more recent usage gets higher score)
        let now_timestamp = Utc::now().timestamp();
        let last_used_timestamp = performance.last_used.timestamp();
        let hours_since_last_use = (now_timestamp - last_used_timestamp) as f32 / 3600.0;
        let recency_score = (1.0 / (1.0 + hours_since_last_use / 24.0)) * config.recency_weight;

        // Context-specific bonuses
        let context_bonus = self.calculate_context_bonus(&performance.algorithm_name, context);

        performance_score + recency_score + context_bonus
    }

    fn calculate_context_bonus(&self, algorithm_name: &str, context: &OptimizationContext) -> f32 {
        match algorithm_name {
            "parallelism_optimization" if context.system_state.available_cores > 4 => 0.2,
            "memory_optimization" if context.constraints.contains_key("memory_pressure") => 0.3,
            "network_optimization" if context.constraints.contains_key("network_latency") => 0.25,
            "io_optimization" if context.constraints.contains_key("io_intensive") => 0.3,
            _ => 0.0,
        }
    }

    fn calculate_effectiveness_score(&self, performance: &StrategyPerformance) -> f32 {
        // Combine success rate, impact, and usage frequency
        let usage_factor = (performance.usage_count as f32 / 100.0).min(1.0); // Normalize to 0-1

        performance.success_rate * 0.5 + performance.average_impact * 0.3 + usage_factor * 0.2
    }
}

/// Adaptive learner for continuous optimization improvement
///
/// Uses machine learning techniques to continuously improve optimization
/// recommendations based on historical outcomes and system behavior.
pub struct AdaptiveLearner {
    learning_model: Arc<Mutex<LearningModel>>,
    training_data: Arc<Mutex<VecDeque<TrainingExample>>>,
    model_performance: Arc<Mutex<ModelPerformanceMetrics>>,
    config: Arc<RwLock<AdaptiveLearnerConfig>>,
    shutdown: Arc<AtomicBool>,
}

#[derive(Debug, Clone)]
struct LearningModel {
    feature_weights: HashMap<String, f32>,
    bias_terms: HashMap<String, f32>,
    learning_rate: f32,
    model_accuracy: f32,
    training_iterations: u64,
}

#[derive(Debug, Clone)]
struct ModelPerformanceMetrics {
    accuracy: f32,
    precision: f32,
    recall: f32,
    f1_score: f32,
    last_updated: DateTime<Utc>,
}

#[derive(Debug, Clone)]
pub struct AdaptiveLearnerConfig {
    pub learning_rate: f32,
    pub batch_size: usize,
    pub max_training_data: usize,
    pub retraining_interval: Duration,
    pub minimum_accuracy_threshold: f32,
}

impl Default for AdaptiveLearnerConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            batch_size: 32,
            max_training_data: 10000,
            retraining_interval: Duration::from_secs(3600), // 1 hour
            minimum_accuracy_threshold: 0.7,
        }
    }
}

impl AdaptiveLearner {
    pub async fn new() -> Result<Self> {
        let learning_model = LearningModel {
            feature_weights: HashMap::new(),
            bias_terms: HashMap::new(),
            learning_rate: 0.01,
            model_accuracy: 0.5,
            training_iterations: 0,
        };

        let performance_metrics = ModelPerformanceMetrics {
            accuracy: 0.5,
            precision: 0.5,
            recall: 0.5,
            f1_score: 0.5,
            last_updated: Utc::now(),
        };

        Ok(Self {
            learning_model: Arc::new(Mutex::new(learning_model)),
            training_data: Arc::new(Mutex::new(VecDeque::new())),
            model_performance: Arc::new(Mutex::new(performance_metrics)),
            config: Arc::new(RwLock::new(AdaptiveLearnerConfig::default())),
            shutdown: Arc::new(AtomicBool::new(false)),
        })
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting adaptive learner");
        self.initialize_model().await?;
        Ok(())
    }

    /// Update model with new recommendations and outcomes
    pub async fn update_with_recommendations(
        &self,
        recommendations: &[OptimizationRecommendation],
    ) -> Result<()> {
        let mut training_data = self.training_data.lock();

        for recommendation in recommendations {
            let features = self.extract_recommendation_features(recommendation);
            let example = TrainingExample {
                input_features: features,
                output_success: true, // Will be updated when actual outcome is known
                recommendation_type: format!(
                    "{:?}",
                    recommendation
                        .actions
                        .first()
                        .map(|a| &a.action_type)
                        .unwrap_or(&ActionType::TuneParameters)
                ),
                timestamp: Utc::now(),
            };

            training_data.push_back(example);
        }

        // Limit training data size
        let config = self.config.read();
        while training_data.len() > config.max_training_data {
            training_data.pop_front();
        }

        Ok(())
    }

    /// Perform background learning and model updates
    pub async fn perform_background_learning(&self) -> Result<()> {
        let should_retrain = {
            let config = self.config.read();
            let retraining_interval = config.retraining_interval;
            drop(config);

            let performance = self.model_performance.lock();
            let now_timestamp = Utc::now().timestamp();
            let last_updated_timestamp = performance.last_updated.timestamp();
            let seconds_since_update = now_timestamp - last_updated_timestamp;
            if seconds_since_update < 0 {
                false
            } else {
                Duration::from_secs(seconds_since_update as u64) > retraining_interval
            }
        };

        if should_retrain {
            self.retrain_model().await?;
        }

        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown.store(true, Ordering::Relaxed);
        info!("Adaptive learner shutdown complete");
        Ok(())
    }

    async fn initialize_model(&self) -> Result<()> {
        let mut model = self.learning_model.lock();

        // Initialize feature weights with reasonable defaults
        let features = vec![
            "confidence",
            "priority",
            "complexity",
            "risk_level",
            "performance_impact",
            "resource_impact",
            "action_count",
        ];

        for feature in features {
            model.feature_weights.insert(feature.to_string(), 0.1);
            model.bias_terms.insert(feature.to_string(), 0.0);
        }

        Ok(())
    }

    fn extract_recommendation_features(
        &self,
        recommendation: &OptimizationRecommendation,
    ) -> Vec<f32> {
        vec![
            recommendation.confidence,
            recommendation.priority as f32 / 10.0, // Normalize
            recommendation.expected_impact.complexity,
            recommendation.expected_impact.risk_level,
            recommendation.expected_impact.performance_impact,
            recommendation.expected_impact.resource_impact,
            recommendation.actions.len() as f32 / 10.0, // Normalize
        ]
    }

    async fn retrain_model(&self) -> Result<()> {
        let training_data = self.training_data.lock();
        let config = self.config.read();

        if training_data.len() < config.batch_size {
            return Ok(()); // Not enough data to train
        }

        let mut model = self.learning_model.lock();

        // Simple gradient descent training
        let batch_size = config.batch_size.min(training_data.len());
        let batch: Vec<_> = training_data.iter().rev().take(batch_size).collect();

        for example in batch {
            let predicted = self.predict_success(&example.input_features, &model);
            let actual = if example.output_success { 1.0 } else { 0.0 };
            let error = actual - predicted;

            // Update weights using gradient descent
            for (i, &feature_value) in example.input_features.iter().enumerate() {
                let feature_name = format!("feature_{}", i);
                let current_weight =
                    model.feature_weights.get(&feature_name).copied().unwrap_or(0.0);
                let new_weight = current_weight + model.learning_rate * error * feature_value;
                model.feature_weights.insert(feature_name, new_weight);
            }
        }

        model.training_iterations += 1;

        // Update model performance metrics
        let accuracy = self.calculate_model_accuracy(&training_data, &model);
        model.model_accuracy = accuracy;

        let mut performance = self.model_performance.lock();
        performance.accuracy = accuracy;
        performance.last_updated = Utc::now();

        info!("Model retrained with accuracy: {:.3}", accuracy);
        Ok(())
    }

    fn predict_success(&self, features: &[f32], model: &LearningModel) -> f32 {
        let mut prediction = 0.0;

        for (i, &feature_value) in features.iter().enumerate() {
            let feature_name = format!("feature_{}", i);
            let weight = model.feature_weights.get(&feature_name).copied().unwrap_or(0.0);
            prediction += weight * feature_value;
        }

        // Apply sigmoid activation
        1.0 / (1.0 + (-prediction).exp())
    }

    fn calculate_model_accuracy(
        &self,
        training_data: &VecDeque<TrainingExample>,
        model: &LearningModel,
    ) -> f32 {
        if training_data.is_empty() {
            return 0.5;
        }

        let correct_predictions = training_data
            .iter()
            .map(|example| {
                let predicted = self.predict_success(&example.input_features, model);
                let predicted_class = predicted > 0.5;
                let actual_class = example.output_success;
                if predicted_class == actual_class {
                    1.0
                } else {
                    0.0
                }
            })
            .sum::<f32>();

        correct_predictions / training_data.len() as f32
    }
}
