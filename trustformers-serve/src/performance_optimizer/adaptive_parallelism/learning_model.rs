//! Adaptive Learning Model for Parallelism Optimization
//!
//! This module provides machine learning capabilities for adaptive parallelism
//! optimization including online learning, model validation, and performance
//! prediction. It includes various learning algorithms and model management.

use anyhow::Result;
use chrono::Utc;
use parking_lot::{Mutex, RwLock};
use std::{collections::HashMap, sync::Arc};

use super::{feedback_system::PerformanceFeedbackSystem, validation::ModelValidation};
use crate::performance_optimizer::types::*;

// =============================================================================
// LEARNING ALGORITHM TRAIT (Local extension trait)
// =============================================================================

/// Extension trait for learning algorithms (local helper methods)
/// Note: The main LearningAlgorithm trait is defined in types.rs
pub trait LearningAlgorithmExt {
    /// Update model with single training example (helper method)
    fn update_single(&mut self, example: &TrainingExample) -> Result<()>;

    /// Get model parameters
    fn get_parameters(&self) -> HashMap<String, f64>;

    /// Set model parameters
    fn set_parameters(&mut self, parameters: HashMap<String, f64>) -> Result<()>;
}

// =============================================================================
// ADAPTIVE LEARNING MODEL IMPLEMENTATION
// =============================================================================

impl AdaptiveLearningModel {
    /// Create a new adaptive learning model
    pub async fn new() -> Result<Self> {
        Ok(Self {
            model_state: Arc::new(RwLock::new(ModelState {
                parameters: HashMap::new(),
                weights: Vec::new(),
                bias: 0.0,
                version: 1,
                last_training: Utc::now(),
                performance_metrics: ModelPerformanceMetrics {
                    training_accuracy: 0.0,
                    validation_accuracy: 0.0,
                    test_accuracy: 0.0,
                    loss: 0.0,
                    convergence_status: ConvergenceStatus::NotConverged,
                    accuracy: 0.0,
                    precision: 0.0,
                    recall: 0.0,
                    f1_score: 0.0,
                    training_examples: 0,
                    last_updated: Utc::now(),
                },
                learning_rate: 0.01,
                accuracy: 0.5,
                last_updated: Utc::now(),
                training_examples_count: 0,
            })),
            learning_algorithm: Arc::new(Mutex::new(Box::new(AdaptiveLinearRegression::new()))),
            learning_history: Arc::new(Mutex::new(LearningHistory {
                training_epochs: Vec::new(),
                model_updates: Vec::new(),
                performance_evolution: Vec::new(),
                learning_rate_history: Vec::new(),
                timestamp: Utc::now(),
                event_type: "initialization".to_string(),
                parameters_before: HashMap::new(),
                parameters_after: HashMap::new(),
                performance_impact: 0.0,
            })),
            model_validation: Arc::new(ModelValidation::new().await?),
            training_data: Arc::new(Mutex::new(TrainingDataset {
                examples: Vec::new(),
                split_ratios: DatasetSplitRatios {
                    training: 0.7,
                    validation: 0.2,
                    test: 0.1,
                },
                statistics: DatasetStatistics {
                    example_count: 0,
                    feature_stats: Vec::new(),
                    target_stats: TargetStatistics {
                        mean: 0.0,
                        std_dev: 0.0,
                        min: 0.0,
                        max: 0.0,
                        distribution: TargetDistribution {
                            distribution_type: DistributionType::Normal,
                            parameters: HashMap::new(),
                            goodness_of_fit: 0.0,
                        },
                    },
                    quality_metrics: DataQualityMetrics {
                        completeness: 1.0,
                        consistency: 1.0,
                        accuracy: 1.0,
                        validity: 1.0,
                        outlier_percentage: 0.0,
                        timeliness: 1.0,
                    },
                    total_examples: 0,
                    feature_statistics: Vec::new(),
                    target_statistics: TargetStatistics {
                        mean: 0.0,
                        std_dev: 0.0,
                        min: 0.0,
                        max: 0.0,
                        distribution: TargetDistribution {
                            distribution_type: DistributionType::Normal,
                            parameters: HashMap::new(),
                            goodness_of_fit: 0.0,
                        },
                    },
                },
                version: 1,
                last_updated: Utc::now(),
                validation_split: 0.2,
            })),
            training_dataset: Arc::new(Mutex::new(TrainingDataset {
                examples: Vec::new(),
                split_ratios: DatasetSplitRatios {
                    training: 0.7,
                    validation: 0.2,
                    test: 0.1,
                },
                statistics: DatasetStatistics {
                    example_count: 0,
                    feature_stats: Vec::new(),
                    target_stats: TargetStatistics {
                        mean: 0.0,
                        std_dev: 0.0,
                        min: 0.0,
                        max: 0.0,
                        distribution: TargetDistribution {
                            distribution_type: DistributionType::Normal,
                            parameters: HashMap::new(),
                            goodness_of_fit: 0.0,
                        },
                    },
                    quality_metrics: DataQualityMetrics {
                        completeness: 1.0,
                        consistency: 1.0,
                        accuracy: 1.0,
                        validity: 1.0,
                        outlier_percentage: 0.0,
                        timeliness: 1.0,
                    },
                    total_examples: 0,
                    feature_statistics: Vec::new(),
                    target_statistics: TargetStatistics {
                        mean: 0.0,
                        std_dev: 0.0,
                        min: 0.0,
                        max: 0.0,
                        distribution: TargetDistribution {
                            distribution_type: DistributionType::Normal,
                            parameters: HashMap::new(),
                            goodness_of_fit: 0.0,
                        },
                    },
                },
                version: 1,
                last_updated: Utc::now(),
                validation_split: 0.2,
            })),
        })
    }

    /// Adjust parallelism estimate based on learned patterns
    pub async fn adjust_estimate(
        &self,
        initial_estimate: &ParallelismEstimate,
        characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<ParallelismEstimate> {
        // Prepare features for prediction
        let features = self.extract_features(characteristics, system_state)?;

        // Get prediction from learning algorithm
        let algorithm = self.learning_algorithm.lock();
        let predicted_adjustment = algorithm.predict(&features)?;

        // Apply adjustment to initial estimate
        let adjusted_parallelism = (initial_estimate.optimal_parallelism as f64
            * (1.0 + predicted_adjustment))
            .round() as usize;

        // Adjust confidence based on model accuracy
        let model_state = self.model_state.read();
        let adjusted_confidence = initial_estimate.confidence * (model_state.accuracy as f32);

        let mut adjusted_estimate = initial_estimate.clone();
        adjusted_estimate.optimal_parallelism = adjusted_parallelism.max(1);
        adjusted_estimate.confidence = adjusted_confidence;
        adjusted_estimate.method = format!("{}_ml_adjusted", initial_estimate.method);

        Ok(adjusted_estimate)
    }

    /// Update model from feedback system
    pub async fn update_from_feedback(
        &self,
        feedback_system: &PerformanceFeedbackSystem,
    ) -> Result<()> {
        let recent_feedback = feedback_system.get_recent_aggregated_feedback().await?;

        for feedback in &recent_feedback {
            // Convert feedback to training example
            let training_example = self.convert_feedback_to_training_example(&feedback)?;

            // Update learning algorithm
            {
                let mut algorithm = self.learning_algorithm.lock();
                algorithm.update(&[training_example.clone()])?;
            }

            // Add to training dataset
            {
                let mut dataset = self.training_dataset.lock();
                dataset.examples.push(training_example);
                dataset.statistics.total_examples += 1;
            }

            // Record learning event
            {
                let mut history = self.learning_history.lock();
                history.model_updates.push(ModelUpdate {
                    timestamp: Utc::now(),
                    update_type: ModelUpdateType::Incremental,
                    previous_version: 0,
                    new_version: 1,
                    reason: "feedback_update".to_string(),
                    performance_impact: Some(feedback.confidence),
                });
            }
        }

        // Update model state
        {
            let mut state = self.model_state.write();
            state.last_updated = Utc::now();
            state.training_examples_count += recent_feedback.len();
        }

        Ok(())
    }

    /// Extract features from test characteristics and system state
    fn extract_features(
        &self,
        characteristics: &TestCharacteristics,
        system_state: &SystemState,
    ) -> Result<Vec<f64>> {
        let features = vec![
            characteristics.average_duration.as_millis() as f64,
            characteristics.resource_intensity.cpu_intensity as f64,
            characteristics.resource_intensity.memory_intensity as f64,
            characteristics.resource_intensity.io_intensity as f64,
            characteristics.dependency_complexity as f64,
            system_state.available_cores as f64,
            system_state.load_average as f64,
            system_state.available_memory_mb as f64 / 1024.0, // Convert to GB
        ];

        Ok(features)
    }

    /// Convert aggregated feedback to training example
    fn convert_feedback_to_training_example(
        &self,
        feedback: &AggregatedFeedback,
    ) -> Result<TrainingExample> {
        // Simplified feature extraction for example
        let features = vec![
            feedback.aggregated_value as f64,
            feedback.confidence as f64,
            feedback.contributing_feedback_count as f64,
        ];

        let target = feedback.aggregated_value as f64; // Simplified target

        Ok(TrainingExample {
            features,
            target,
            weight: feedback.confidence as f64,
            timestamp: feedback.timestamp,
            metadata: HashMap::new(),
        })
    }

    /// Get model performance metrics
    pub fn get_performance_metrics(&self) -> ModelPerformanceMetrics {
        let state = self.model_state.read();
        ModelPerformanceMetrics {
            training_accuracy: state.accuracy as f32,
            validation_accuracy: (state.accuracy * 0.95) as f32, // Placeholder
            test_accuracy: (state.accuracy * 0.9) as f32,        // Placeholder
            loss: 0.1,                                           // Placeholder
            convergence_status: ConvergenceStatus::NotConverged,
            accuracy: state.accuracy as f32,
            precision: 0.8, // Placeholder
            recall: 0.75,   // Placeholder
            f1_score: 0.77, // Placeholder
            training_examples: state.training_examples_count,
            last_updated: state.last_updated,
        }
    }
}

// =============================================================================
// ADAPTIVE LINEAR REGRESSION ALGORITHM
// =============================================================================

/// Adaptive linear regression learning algorithm
pub struct AdaptiveLinearRegression {
    name: String,
    weights: Vec<f64>,
    bias: f64,
    learning_rate: f64,
    momentum: Vec<f64>,
    momentum_factor: f64,
}

impl AdaptiveLinearRegression {
    pub fn new() -> Self {
        Self {
            name: "adaptive_linear_regression".to_string(),
            weights: Vec::new(),
            bias: 0.0,
            learning_rate: 0.01,
            momentum: Vec::new(),
            momentum_factor: 0.9,
        }
    }

    /// Initialize weights if needed
    fn ensure_weights_initialized(&mut self, feature_count: usize) {
        if self.weights.len() != feature_count {
            self.weights = vec![0.1; feature_count]; // Small random initialization
            self.momentum = vec![0.0; feature_count];
        }
    }
}

// Implement the main LearningAlgorithm trait from types.rs
impl LearningAlgorithm for AdaptiveLinearRegression {
    fn train(&mut self, training_data: &TrainingDataset) -> Result<ModelState> {
        // Train on all examples in the dataset
        for example in &training_data.examples {
            self.update_single(example)?;
        }

        // Return current model state
        Ok(ModelState {
            parameters: self.get_parameters(),
            weights: self.weights.clone(),
            bias: self.bias,
            version: 1,
            last_training: Utc::now(),
            performance_metrics: ModelPerformanceMetrics::default(),
            learning_rate: self.learning_rate,
            accuracy: 0.8, // Placeholder - should be calculated
            last_updated: Utc::now(),
            training_examples_count: training_data.examples.len(),
        })
    }

    fn predict(&self, input: &[f64]) -> Result<f64> {
        // If model hasn't been trained yet (no weights), return a default prediction
        if self.weights.is_empty() {
            // Return a neutral prediction (e.g., average of expected range)
            return Ok(0.5);
        }

        if self.weights.len() != input.len() {
            return Err(anyhow::anyhow!(
                "Feature count mismatch: expected {}, got {}",
                self.weights.len(),
                input.len()
            ));
        }

        let prediction =
            input.iter().zip(self.weights.iter()).map(|(f, w)| f * w).sum::<f64>() + self.bias;

        Ok(prediction)
    }

    fn update(&mut self, new_data: &[TrainingExample]) -> Result<ModelState> {
        // Update model with multiple training examples
        for example in new_data {
            self.update_single(example)?;
        }

        // Return updated model state
        Ok(ModelState {
            parameters: self.get_parameters(),
            weights: self.weights.clone(),
            bias: self.bias,
            version: 1,
            last_training: Utc::now(),
            performance_metrics: ModelPerformanceMetrics::default(),
            learning_rate: self.learning_rate,
            accuracy: 0.8, // Placeholder - should be calculated
            last_updated: Utc::now(),
            training_examples_count: new_data.len(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }
}

// Implement the extension trait for helper methods
impl LearningAlgorithmExt for AdaptiveLinearRegression {
    fn update_single(&mut self, example: &TrainingExample) -> Result<()> {
        self.ensure_weights_initialized(example.features.len());

        // Calculate prediction
        let prediction = self.predict(&example.features)?;

        // Calculate error
        let error = example.target - prediction;

        // Update weights using gradient descent with momentum
        for i in 0..self.weights.len() {
            let gradient = error * example.features[i] * example.weight;
            self.momentum[i] =
                self.momentum_factor * self.momentum[i] + self.learning_rate * gradient;
            self.weights[i] += self.momentum[i];
        }

        // Update bias
        self.bias += self.learning_rate * error * example.weight;

        Ok(())
    }

    fn get_parameters(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        for (i, weight) in self.weights.iter().enumerate() {
            params.insert(format!("weight_{}", i), *weight);
        }
        params.insert("bias".to_string(), self.bias);
        params.insert("learning_rate".to_string(), self.learning_rate);
        params
    }

    fn set_parameters(&mut self, parameters: HashMap<String, f64>) -> Result<()> {
        if let Some(bias) = parameters.get("bias") {
            self.bias = *bias;
        }
        if let Some(lr) = parameters.get("learning_rate") {
            self.learning_rate = *lr;
        }

        // Set weights
        let mut weight_count = 0;
        for (key, _value) in parameters.iter() {
            if key.starts_with("weight_") {
                weight_count += 1;
            }
        }

        if weight_count > 0 {
            self.weights = vec![0.0; weight_count];
            for i in 0..weight_count {
                if let Some(weight) = parameters.get(&format!("weight_{}", i)) {
                    self.weights[i] = *weight;
                }
            }
        }

        Ok(())
    }
}
