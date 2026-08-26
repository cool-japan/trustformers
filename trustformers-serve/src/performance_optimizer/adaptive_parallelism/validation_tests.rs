//! Tests for [`super::validation`].
//!
//! The point of every assertion here is that the reported score changes when
//! the *model* changes. The strategies these replaced scored a model without
//! ever calling it: the holdout score was `min(len / 10, 1.0)` and the
//! per-fold score was `1 / (1 + variance(targets))`, so a model that predicted
//! a constant scored exactly the same as one that fitted the data perfectly.

use super::learning_model::AdaptiveLinearRegression;
use super::validation::{FoldedEvaluationStrategy, HoldoutValidationStrategy};
use crate::performance_optimizer::types::{LearningAlgorithm, TrainingExample, ValidationStrategy};
use chrono::Utc;
use std::collections::HashMap;

/// Model with fixed coefficients, so the expected residuals can be computed by
/// hand.
#[derive(Debug)]
struct FixedLinearModel {
    name: String,
    weights: Vec<f64>,
    bias: f64,
}

impl FixedLinearModel {
    fn new(weights: Vec<f64>, bias: f64) -> Self {
        Self {
            name: "fixed_linear".to_string(),
            weights,
            bias,
        }
    }
}

impl LearningAlgorithm for FixedLinearModel {
    fn train(
        &mut self,
        _training_data: &crate::performance_optimizer::types::TrainingDataset,
    ) -> anyhow::Result<crate::performance_optimizer::types::ModelState> {
        anyhow::bail!("FixedLinearModel does not train")
    }

    fn predict(&self, input: &[f64]) -> anyhow::Result<f64> {
        if input.len() != self.weights.len() {
            anyhow::bail!("feature count mismatch");
        }
        Ok(input.iter().zip(self.weights.iter()).map(|(x, w)| x * w).sum::<f64>() + self.bias)
    }

    fn update(
        &mut self,
        _new_data: &[TrainingExample],
    ) -> anyhow::Result<crate::performance_optimizer::types::ModelState> {
        anyhow::bail!("FixedLinearModel does not update")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

/// Model whose prediction ignores its input entirely.
#[derive(Debug)]
struct ConstantModel {
    name: String,
    value: f64,
}

impl LearningAlgorithm for ConstantModel {
    fn train(
        &mut self,
        _training_data: &crate::performance_optimizer::types::TrainingDataset,
    ) -> anyhow::Result<crate::performance_optimizer::types::ModelState> {
        anyhow::bail!("ConstantModel does not train")
    }

    fn predict(&self, _input: &[f64]) -> anyhow::Result<f64> {
        Ok(self.value)
    }

    fn update(
        &mut self,
        _new_data: &[TrainingExample],
    ) -> anyhow::Result<crate::performance_optimizer::types::ModelState> {
        anyhow::bail!("ConstantModel does not update")
    }

    fn name(&self) -> &str {
        &self.name
    }
}

fn example(x: f64, target: f64) -> TrainingExample {
    TrainingExample {
        features: vec![x],
        target,
        weight: 1.0,
        timestamp: Utc::now(),
        metadata: HashMap::new(),
    }
}

/// `y = 2x + 1` over x = 0..=9.
fn linear_dataset() -> Vec<TrainingExample> {
    (0..10).map(|i| example(i as f64, 2.0 * i as f64 + 1.0)).collect()
}

#[test]
fn holdout_scores_a_perfect_model_perfectly() {
    let model = FixedLinearModel::new(vec![2.0], 1.0);
    let strategy = HoldoutValidationStrategy::new(0.3);

    let result = strategy.validate(&model, &linear_dataset()).expect("validation should succeed");

    assert!(
        (result.details.r_squared - 1.0).abs() < 1e-5,
        "a model that reproduces the targets exactly must score 1.0, got {}",
        result.details.r_squared
    );
    assert!(result.details.mean_absolute_error.abs() < 1e-5);
    assert!(result.details.root_mean_squared_error.abs() < 1e-5);
    assert!(result.passed);
    assert!(result.details.classification.is_none());
}

#[test]
fn holdout_penalises_a_model_that_ignores_its_input() {
    let data = linear_dataset();
    let good = FixedLinearModel::new(vec![2.0], 1.0);
    let bad = ConstantModel {
        name: "constant".to_string(),
        value: 0.0,
    };
    let strategy = HoldoutValidationStrategy::new(0.3);

    let good_result = strategy.validate(&good, &data).expect("validate good");
    let bad_result = strategy.validate(&bad, &data).expect("validate bad");

    assert!(
        good_result.details.r_squared > bad_result.details.r_squared,
        "the fitted model ({}) must outscore the constant one ({})",
        good_result.details.r_squared,
        bad_result.details.r_squared
    );
    assert!(
        bad_result.details.mean_absolute_error > 0.0,
        "a model that predicts 0 for every target must have non-zero error"
    );
    assert!(!bad_result.passed);
}

#[test]
fn holdout_error_reflects_the_actual_residuals() {
    // A model biased by exactly +3 on every point: MAE and RMSE are both 3.
    let model = FixedLinearModel::new(vec![2.0], 4.0);
    let strategy = HoldoutValidationStrategy::new(0.5);

    let result = strategy.validate(&model, &linear_dataset()).expect("validate");

    assert!(
        (result.details.mean_absolute_error - 3.0).abs() < 1e-5,
        "MAE must be the measured residual, got {}",
        result.details.mean_absolute_error
    );
    assert!(
        (result.details.root_mean_squared_error - 3.0).abs() < 1e-5,
        "RMSE must be the measured residual, got {}",
        result.details.root_mean_squared_error
    );
}

#[test]
fn holdout_confidence_grows_with_the_size_of_the_holdout_set() {
    let model = FixedLinearModel::new(vec![2.0], 1.0);
    let small = HoldoutValidationStrategy::new(0.1);
    let large = HoldoutValidationStrategy::new(0.9);

    let data: Vec<TrainingExample> =
        (0..100).map(|i| example(i as f64, 2.0 * i as f64 + 1.0)).collect();

    let small_result = small.validate(&model, &data).expect("validate small");
    let large_result = large.validate(&model, &data).expect("validate large");

    assert!(
        large_result.confidence > small_result.confidence,
        "confidence must follow the amount of evidence: {} vs {}",
        large_result.confidence,
        small_result.confidence
    );
}

#[test]
fn folded_evaluation_reports_one_score_per_fold() {
    let model = FixedLinearModel::new(vec![2.0], 1.0);
    let strategy = FoldedEvaluationStrategy::new(5);

    let result = strategy.validate(&model, &linear_dataset()).expect("validate");

    assert_eq!(result.details.fold_scores.len(), 5);
    assert_eq!(result.method, "folded_evaluation");
    assert!((result.details.r_squared - 1.0).abs() < 1e-5);
}

#[test]
fn folded_evaluation_never_makes_more_folds_than_examples() {
    let model = FixedLinearModel::new(vec![2.0], 1.0);
    let strategy = FoldedEvaluationStrategy::new(16);

    let result = strategy
        .validate(&model, &[example(1.0, 3.0), example(2.0, 5.0)])
        .expect("validate");

    assert_eq!(
        result.details.fold_scores.len(),
        2,
        "an empty fold has no score to report"
    );
}

#[test]
fn validation_refuses_an_empty_dataset() {
    let model = FixedLinearModel::new(vec![2.0], 1.0);

    let folded = FoldedEvaluationStrategy::new(3)
        .validate(&model, &[])
        .expect_err("an empty validation set cannot produce a score");
    assert!(folded.to_string().contains("empty validation set"));

    let holdout = HoldoutValidationStrategy::new(0.3)
        .validate(&model, &[])
        .expect_err("an empty validation set cannot produce a score");
    assert!(holdout.to_string().contains("empty validation set"));
}

#[test]
fn validation_propagates_a_prediction_failure() {
    // Two features expected, one supplied: the model errors, and the strategy
    // must surface that instead of scoring around it.
    let model = FixedLinearModel::new(vec![2.0, 3.0], 1.0);

    let error = HoldoutValidationStrategy::new(0.5)
        .validate(&model, &linear_dataset())
        .expect_err("a failing prediction must not be scored");
    assert!(
        error.to_string().contains("feature count mismatch"),
        "unexpected error: {error}"
    );
}

#[test]
fn untrained_linear_regression_refuses_to_predict() {
    let model = AdaptiveLinearRegression::new();

    let error = model.predict(&[1.0]).expect_err("an untrained model has no prediction to give");
    assert!(
        error.to_string().contains("has not been trained"),
        "unexpected error: {error}"
    );
}

#[test]
fn trained_linear_regression_reports_measured_accuracy() {
    use crate::performance_optimizer::types::{
        DataQualityMetrics, DatasetSplitRatios, DatasetStatistics, DistributionType,
        TargetDistribution, TargetStatistics, TrainingDataset,
    };

    let examples = linear_dataset();
    let target_stats = TargetStatistics {
        mean: 0.0,
        std_dev: 0.0,
        min: 0.0,
        max: 0.0,
        distribution: TargetDistribution {
            distribution_type: DistributionType::Normal,
            parameters: HashMap::new(),
            goodness_of_fit: 0.0,
        },
    };
    let dataset = TrainingDataset {
        examples: examples.clone(),
        split_ratios: DatasetSplitRatios {
            training: 0.7,
            validation: 0.2,
            test: 0.1,
        },
        statistics: DatasetStatistics {
            example_count: examples.len(),
            feature_stats: Vec::new(),
            target_stats: target_stats.clone(),
            quality_metrics: DataQualityMetrics {
                completeness: 1.0,
                consistency: 1.0,
                accuracy: 1.0,
                validity: 1.0,
                outlier_percentage: 0.0,
                timeliness: 1.0,
            },
            total_examples: examples.len(),
            feature_statistics: Vec::new(),
            target_statistics: target_stats,
        },
        version: 1,
        last_updated: Utc::now(),
        validation_split: 0.2,
    };

    let mut model = AdaptiveLinearRegression::new();
    let state = model.train(&dataset).expect("training should succeed");

    assert_ne!(
        state.accuracy, 0.8,
        "accuracy must be measured, not the literal it used to be"
    );
    assert!(
        state.accuracy.is_finite(),
        "accuracy must be a real number, got {}",
        state.accuracy
    );
    assert_eq!(state.training_examples_count, examples.len());
}
