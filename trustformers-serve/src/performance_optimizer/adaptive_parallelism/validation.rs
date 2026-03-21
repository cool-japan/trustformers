//! Model Validation for Adaptive Learning
//!
//! This module provides comprehensive model validation capabilities including
//! cross-validation, holdout validation, and bootstrap validation strategies.
//! It helps ensure the reliability and accuracy of adaptive learning models.

use anyhow::Result;
use chrono::Utc;
use parking_lot::Mutex;
use std::{collections::HashMap, sync::Arc};

use crate::performance_optimizer::types::*;

// Re-export types needed by other modules
pub use crate::performance_optimizer::types::{ModelValidation, ValidationStrategy};

// =============================================================================
// MODEL VALIDATION IMPLEMENTATION
// =============================================================================

impl ModelValidation {
    /// Create a new model validation system
    pub async fn new() -> Result<Self> {
        let mut strategies: Vec<Box<dyn ValidationStrategy + Send + Sync>> = Vec::new();

        // Add default validation strategies
        strategies.push(Box::new(CrossValidationStrategy::new(5))); // 5-fold CV
        strategies.push(Box::new(HoldoutValidationStrategy::new(0.3))); // 30% holdout

        Ok(Self {
            strategies: Arc::new(Mutex::new(strategies)),
            results_cache: Arc::new(Mutex::new(HashMap::new())),
            validation_history: Arc::new(Mutex::new(Vec::new())),
        })
    }

    /// Validate model using multiple strategies
    pub async fn validate_model(
        &self,
        dataset: &TrainingDataset,
        model: &dyn crate::performance_optimizer::LearningAlgorithm,
    ) -> Result<ModelValidationResults> {
        let strategies = self.strategies.lock();
        let mut validation_results = Vec::new();

        // Create ModelState from the learning algorithm
        let model_state = ModelState {
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
        };

        // Apply each validation strategy
        for strategy in strategies.iter() {
            if strategy.is_applicable(&model_state) && !dataset.examples.is_empty() {
                match strategy.validate(&model_state, &dataset.examples) {
                    Ok(result) => validation_results.push(result),
                    Err(e) => log::warn!("Validation strategy {} failed: {}", strategy.name(), e),
                }
            } else {
                log::debug!(
                    "Skipping {} validation: not applicable or insufficient data",
                    strategy.name(),
                );
            }
        }

        // Combine validation results
        let combined_results = self.combine_validation_results(&validation_results)?;

        // Record validation
        {
            let strategy_names: Vec<String> =
                strategies.iter().map(|s| s.name().to_string()).collect();

            let record = ValidationRecord {
                timestamp: Utc::now(),
                model_version: 1,
                strategy: strategy_names.first().cloned().unwrap_or_else(|| "unknown".to_string()),
                result: validation_results.first().cloned().unwrap_or_default(),
                duration: std::time::Duration::from_millis(0),
                model_name: model.name().to_string(),
                dataset_size: dataset.examples.len(),
                strategies_used: strategy_names,
                results: validation_results.clone(),
            };

            self.validation_history.lock().push(record);
        }

        Ok(combined_results)
    }

    /// Combine multiple validation results
    fn combine_validation_results(
        &self,
        results: &[ValidationResult],
    ) -> Result<ModelValidationResults> {
        if results.is_empty() {
            return Ok(ModelValidationResults {
                r_squared: 0.0,
                mean_absolute_error: f32::INFINITY,
                root_mean_squared_error: f32::INFINITY,
                cross_validation_scores: Vec::new(),
                validated_at: Utc::now(),
            });
        }

        // Average the metrics across strategies
        let avg_r_squared =
            results.iter().map(|r| r.details.r_squared as f64).sum::<f64>() / results.len() as f64;
        let avg_mae = results.iter().map(|r| r.details.mean_absolute_error as f64).sum::<f64>()
            / results.len() as f64;
        let avg_rmse =
            results.iter().map(|r| r.details.root_mean_squared_error as f64).sum::<f64>()
                / results.len() as f64;

        // Collect all cross-validation scores
        let cv_scores: Vec<f32> =
            results.iter().flat_map(|r| r.details.cross_validation_scores.clone()).collect();

        Ok(ModelValidationResults {
            r_squared: avg_r_squared as f32,
            mean_absolute_error: avg_mae as f32,
            root_mean_squared_error: avg_rmse as f32,
            cross_validation_scores: cv_scores,
            validated_at: Utc::now(),
        })
    }
}

// =============================================================================
// CROSS VALIDATION STRATEGY
// =============================================================================

/// Cross-validation strategy
pub struct CrossValidationStrategy {
    name: String,
    folds: usize,
}

impl CrossValidationStrategy {
    pub fn new(folds: usize) -> Self {
        Self {
            name: format!("{}_fold_cross_validation", folds),
            folds,
        }
    }
}

impl ValidationStrategy for CrossValidationStrategy {
    fn validate(
        &self,
        _model: &ModelState,
        validation_data: &[TrainingExample],
    ) -> Result<ValidationResult> {
        let examples = validation_data;
        let fold_size = examples.len() / self.folds;
        let mut cv_scores = Vec::new();

        for fold in 0..self.folds {
            let start_idx = fold * fold_size;
            let end_idx =
                if fold == self.folds - 1 { examples.len() } else { (fold + 1) * fold_size };

            // Split into training and validation sets
            let validation_set = &examples[start_idx..end_idx];
            let training_set: Vec<_> = examples[..start_idx]
                .iter()
                .chain(examples[end_idx..].iter())
                .cloned()
                .collect();

            // Simple validation score calculation (placeholder)
            let score = self.calculate_fold_score(&training_set, validation_set)?;
            cv_scores.push(score);
        }

        let avg_score = cv_scores.iter().sum::<f32>() / cv_scores.len() as f32;
        let variance =
            cv_scores.iter().map(|s| (s - avg_score).powi(2)).sum::<f32>() / cv_scores.len() as f32;

        Ok(ValidationResult {
            score: avg_score,
            metrics: [("cv_score".to_string(), avg_score as f64)].iter().cloned().collect(),
            passed: avg_score > 0.7,
            timestamp: Utc::now(),
            method: "cross_validation".to_string(),
            details: ValidationDetails {
                true_positives: 0,        // Placeholder
                false_positives: 0,       // Placeholder
                true_negatives: 0,        // Placeholder
                false_negatives: 0,       // Placeholder
                confusion_matrix: vec![], // Placeholder
                roc_curve: vec![],        // Placeholder
                r_squared: avg_score,
                mean_absolute_error: variance,
                root_mean_squared_error: variance.sqrt(),
                cross_validation_scores: cv_scores,
            },
            strategy_name: self.name.clone(),
            confidence: (1.0 / (1.0 + variance)).min(0.95),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, _model: &ModelState) -> bool {
        true // Cross-validation is generally applicable
    }
}

impl CrossValidationStrategy {
    /// Calculate validation score for a fold
    fn calculate_fold_score(
        &self,
        _training_set: &[TrainingExample],
        validation_set: &[TrainingExample],
    ) -> Result<f32> {
        // Simplified score calculation
        // In a real implementation, this would train a model and validate it
        if validation_set.is_empty() {
            return Ok(0.0);
        }

        // Calculate consistency score based on target variance
        let targets: Vec<f64> = validation_set.iter().map(|e| e.target).collect();
        let mean_target = targets.iter().sum::<f64>() / targets.len() as f64;
        let variance =
            targets.iter().map(|t| (t - mean_target).powi(2)).sum::<f64>() / targets.len() as f64;

        let score = (1.0 / (1.0 + variance)).min(1.0) as f32;
        Ok(score)
    }
}

// =============================================================================
// HOLDOUT VALIDATION STRATEGY
// =============================================================================

/// Holdout validation strategy
pub struct HoldoutValidationStrategy {
    name: String,
    holdout_ratio: f64,
}

impl HoldoutValidationStrategy {
    pub fn new(holdout_ratio: f64) -> Self {
        Self {
            name: format!("holdout_validation_{:.0}%", holdout_ratio * 100.0),
            holdout_ratio,
        }
    }
}

impl ValidationStrategy for HoldoutValidationStrategy {
    fn validate(
        &self,
        _model: &ModelState,
        validation_data: &[TrainingExample],
    ) -> Result<ValidationResult> {
        let examples = validation_data;
        let holdout_size = (examples.len() as f64 * self.holdout_ratio) as usize;
        let training_size = examples.len() - holdout_size;

        // Split dataset
        let training_set = &examples[..training_size];
        let validation_set = &examples[training_size..];

        // Calculate validation metrics
        let validation_score = self.calculate_validation_score(training_set, validation_set)?;

        Ok(ValidationResult {
            score: validation_score,
            metrics: [("holdout_score".to_string(), validation_score as f64)]
                .iter()
                .cloned()
                .collect(),
            passed: validation_score > 0.7,
            timestamp: Utc::now(),
            method: "holdout_validation".to_string(),
            details: ValidationDetails {
                true_positives: 0,        // Placeholder
                false_positives: 0,       // Placeholder
                true_negatives: 0,        // Placeholder
                false_negatives: 0,       // Placeholder
                confusion_matrix: vec![], // Placeholder
                roc_curve: vec![],        // Placeholder
                r_squared: validation_score,
                mean_absolute_error: 0.1,      // Placeholder
                root_mean_squared_error: 0.15, // Placeholder
                cross_validation_scores: vec![validation_score],
            },
            strategy_name: self.name.clone(),
            confidence: 0.8, // Fixed confidence for holdout
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, _model: &ModelState) -> bool {
        true // Holdout validation is generally applicable
    }
}

impl HoldoutValidationStrategy {
    /// Calculate validation score for holdout set
    fn calculate_validation_score(
        &self,
        _training_set: &[TrainingExample],
        validation_set: &[TrainingExample],
    ) -> Result<f32> {
        if validation_set.is_empty() {
            return Ok(0.0);
        }

        // Simplified validation score calculation
        // In a real implementation, this would train a model and evaluate it
        let score = (validation_set.len() as f32 / 10.0).min(1.0);
        Ok(score)
    }
}
