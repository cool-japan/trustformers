//! Model Validation for Adaptive Learning
//!
//! This module provides comprehensive model validation capabilities including
//! cross-validation, holdout validation, and bootstrap validation strategies.
//! It helps ensure the reliability and accuracy of adaptive learning models.

use anyhow::{anyhow, bail, Result};
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
        strategies.push(Box::new(FoldedEvaluationStrategy::new(5)));
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

        // Apply each validation strategy
        for strategy in strategies.iter() {
            if strategy.is_applicable(model) && !dataset.examples.is_empty() {
                match strategy.validate(model, &dataset.examples) {
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
            results.iter().flat_map(|r| r.details.fold_scores.clone()).collect();

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
// REGRESSION METRICS
// =============================================================================

/// Regression accuracy of `model` over `examples`, computed from real
/// predictions.
///
/// Returns `None` for an empty slice — there is no accuracy to report — and
/// propagates any prediction failure rather than substituting a default.
///
/// `r_squared` is the coefficient of determination against the examples' own
/// mean. It is left unclamped: a model that predicts worse than the mean gets a
/// negative score, which is the honest reading, and callers that need a
/// bounded score clamp it themselves.
fn regression_metrics(
    model: &dyn LearningAlgorithm,
    examples: &[TrainingExample],
) -> Result<Option<RegressionMetrics>> {
    if examples.is_empty() {
        return Ok(None);
    }

    let mut absolute_error_sum = 0.0_f64;
    let mut squared_error_sum = 0.0_f64;
    let mut residuals = Vec::with_capacity(examples.len());

    for example in examples {
        let prediction = model.predict(&example.features)?;
        let residual = prediction - example.target;
        absolute_error_sum += residual.abs();
        squared_error_sum += residual * residual;
        residuals.push(residual);
    }

    let count = examples.len() as f64;
    let mean_target = examples.iter().map(|e| e.target).sum::<f64>() / count;
    let total_sum_of_squares =
        examples.iter().map(|e| (e.target - mean_target).powi(2)).sum::<f64>();

    // With zero variance in the targets, R² is undefined; report a perfect fit
    // only when the residuals are actually zero.
    let r_squared = if total_sum_of_squares <= f64::EPSILON {
        if squared_error_sum <= f64::EPSILON {
            1.0
        } else {
            0.0
        }
    } else {
        1.0 - squared_error_sum / total_sum_of_squares
    };

    Ok(Some(RegressionMetrics {
        r_squared: r_squared as f32,
        mean_absolute_error: (absolute_error_sum / count) as f32,
        root_mean_squared_error: (squared_error_sum / count).sqrt() as f32,
        sample_count: examples.len(),
    }))
}

/// Accuracy of a regression model over one evaluated split.
#[derive(Debug, Clone, Copy)]
pub struct RegressionMetrics {
    /// Coefficient of determination.
    pub r_squared: f32,
    /// Mean absolute error.
    pub mean_absolute_error: f32,
    /// Root mean squared error.
    pub root_mean_squared_error: f32,
    /// Number of examples the metrics were computed over.
    pub sample_count: usize,
}

// =============================================================================
// FOLDED EVALUATION STRATEGY
// =============================================================================

/// Evaluates an already-trained model over `folds` disjoint partitions of the
/// data and reports the spread of its accuracy.
///
/// This is **not** k-fold cross-validation, and it deliberately does not claim
/// to be. Cross-validation retrains the model once per fold, which needs a way
/// to build a fresh model — [`ValidationStrategy`] receives `&dyn
/// LearningAlgorithm`, whose `train` takes `&mut self`, so no such handle
/// exists here. What this measures is real and useful on its own terms: how
/// stable the trained model's error is across independent slices of the data.
///
/// Before 0.2.1 this type was named `CrossValidationStrategy` and its per-fold
/// score was `1 / (1 + variance(targets))` — a function of the *labels* alone.
/// It never called the model, so a model that predicted a constant scored
/// identically to one that fit the data.
pub struct FoldedEvaluationStrategy {
    name: String,
    folds: usize,
}

impl FoldedEvaluationStrategy {
    /// Evaluate over `folds` partitions. `folds` is clamped to at least one.
    pub fn new(folds: usize) -> Self {
        let folds = folds.max(1);
        Self {
            name: format!("{}_fold_evaluation", folds),
            folds,
        }
    }

    /// Number of partitions this strategy evaluates over.
    pub fn folds(&self) -> usize {
        self.folds
    }
}

impl ValidationStrategy for FoldedEvaluationStrategy {
    fn validate(
        &self,
        model: &dyn LearningAlgorithm,
        validation_data: &[TrainingExample],
    ) -> Result<ValidationResult> {
        if validation_data.is_empty() {
            bail!(
                "{} cannot evaluate a model against an empty validation set",
                self.name
            );
        }

        // Never produce more folds than there are examples, so every fold holds
        // at least one example and no fold score is computed over nothing.
        let folds = self.folds.min(validation_data.len());
        let mut fold_metrics = Vec::with_capacity(folds);

        for fold in 0..folds {
            let start_idx = fold * validation_data.len() / folds;
            let end_idx = (fold + 1) * validation_data.len() / folds;
            let Some(metrics) = regression_metrics(model, &validation_data[start_idx..end_idx])?
            else {
                continue;
            };
            fold_metrics.push(metrics);
        }

        if fold_metrics.is_empty() {
            bail!("{} produced no evaluable folds", self.name);
        }

        let fold_scores: Vec<f32> = fold_metrics.iter().map(|m| m.r_squared).collect();
        let overall = regression_metrics(model, validation_data)?
            .ok_or_else(|| anyhow!("validation set became empty during evaluation"))?;

        let mean_score = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;
        let variance = fold_scores.iter().map(|s| (s - mean_score).powi(2)).sum::<f32>()
            / fold_scores.len() as f32;

        Ok(ValidationResult {
            score: overall.r_squared,
            metrics: [
                ("r_squared".to_string(), overall.r_squared as f64),
                (
                    "mean_absolute_error".to_string(),
                    overall.mean_absolute_error as f64,
                ),
                (
                    "root_mean_squared_error".to_string(),
                    overall.root_mean_squared_error as f64,
                ),
                ("fold_score_variance".to_string(), variance as f64),
            ]
            .iter()
            .cloned()
            .collect(),
            passed: overall.r_squared > 0.7,
            timestamp: Utc::now(),
            method: "folded_evaluation".to_string(),
            details: ValidationDetails {
                classification: None,
                r_squared: overall.r_squared,
                mean_absolute_error: overall.mean_absolute_error,
                root_mean_squared_error: overall.root_mean_squared_error,
                fold_scores,
            },
            strategy_name: self.name.clone(),
            // Confidence falls as the model's accuracy varies between folds.
            confidence: (1.0 / (1.0 + variance)).min(0.95),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, _model: &dyn LearningAlgorithm) -> bool {
        // Any algorithm that can predict can be evaluated this way.
        true
    }
}

// =============================================================================
// HOLDOUT VALIDATION STRATEGY
// =============================================================================

/// Evaluates a trained model on the tail of the dataset, holding out
/// `holdout_ratio` of the examples.
///
/// The reported error is measured: every held-out example is predicted and
/// compared against its recorded target. Before 0.2.1 the score was
/// `min(len(validation_set) / 10, 1.0)` — a function of the *sample count* —
/// and the errors were the literals `0.1` and `0.15`.
pub struct HoldoutValidationStrategy {
    name: String,
    holdout_ratio: f64,
}

impl HoldoutValidationStrategy {
    /// Hold out `holdout_ratio` of the dataset. The ratio is clamped into
    /// `(0, 1)` so that both parts of the split are non-empty.
    pub fn new(holdout_ratio: f64) -> Self {
        let holdout_ratio = holdout_ratio.clamp(0.01, 0.99);
        Self {
            name: format!("holdout_validation_{:.0}%", holdout_ratio * 100.0),
            holdout_ratio,
        }
    }

    /// Fraction of the dataset held out for evaluation.
    pub fn holdout_ratio(&self) -> f64 {
        self.holdout_ratio
    }
}

impl ValidationStrategy for HoldoutValidationStrategy {
    fn validate(
        &self,
        model: &dyn LearningAlgorithm,
        validation_data: &[TrainingExample],
    ) -> Result<ValidationResult> {
        if validation_data.is_empty() {
            bail!(
                "{} cannot evaluate a model against an empty validation set",
                self.name
            );
        }

        // Always keep at least one example on the held-out side.
        let holdout_size = ((validation_data.len() as f64 * self.holdout_ratio).round() as usize)
            .clamp(1, validation_data.len());
        let split_at = validation_data.len() - holdout_size;
        let holdout_set = &validation_data[split_at..];

        let metrics = regression_metrics(model, holdout_set)?
            .ok_or_else(|| anyhow!("holdout split produced no examples"))?;

        Ok(ValidationResult {
            score: metrics.r_squared,
            metrics: [
                ("r_squared".to_string(), metrics.r_squared as f64),
                (
                    "mean_absolute_error".to_string(),
                    metrics.mean_absolute_error as f64,
                ),
                (
                    "root_mean_squared_error".to_string(),
                    metrics.root_mean_squared_error as f64,
                ),
                ("holdout_examples".to_string(), metrics.sample_count as f64),
            ]
            .iter()
            .cloned()
            .collect(),
            passed: metrics.r_squared > 0.7,
            timestamp: Utc::now(),
            method: "holdout_validation".to_string(),
            details: ValidationDetails {
                classification: None,
                r_squared: metrics.r_squared,
                mean_absolute_error: metrics.mean_absolute_error,
                root_mean_squared_error: metrics.root_mean_squared_error,
                fold_scores: vec![metrics.r_squared],
            },
            strategy_name: self.name.clone(),
            // A single split gives no spread to reason about, so confidence
            // tracks how much data backed the measurement instead of being a
            // fixed literal.
            confidence: (metrics.sample_count as f32 / (metrics.sample_count as f32 + 10.0))
                .min(0.95),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn is_applicable(&self, _model: &dyn LearningAlgorithm) -> bool {
        true
    }
}
