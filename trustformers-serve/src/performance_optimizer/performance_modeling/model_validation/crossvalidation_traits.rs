//! # CrossValidation - Trait Implementations
//!
//! This module contains trait implementations for `CrossValidation`.
//!
//! ## Implemented Traits
//!
//! - `ValidationStrategy`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;
use tracing;

use crate::performance_optimizer::performance_modeling::types::{
    DistributionInfo, DistributionType, PerformancePredictor, PredictionRequest, ResidualAnalysis,
    TestDataStatistics, ValidationConfig, ValidationDetails, ValidationMetric, ValidationResult,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::functions::{MetricCalculator, ValidationStrategy};
use super::types::{CrossValidation, MAECalculator, RMSECalculator, RSquaredCalculator};

#[async_trait]
impl ValidationStrategy for CrossValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let folds = self.create_folds(data);
        let mut fold_scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();
        for (fold_idx, (_train_data, test_data)) in folds.iter().enumerate() {
            tracing::debug!("Processing fold {}/{}", fold_idx + 1, self.folds);
            let mut fold_predictions = Vec::new();
            let mut fold_actuals = Vec::new();
            for data_point in test_data {
                let prediction_request = PredictionRequest {
                    parallelism_levels: vec![data_point.parallelism],
                    test_characteristics: data_point.test_characteristics.clone(),
                    system_state: data_point.system_state.clone(),
                    prediction_horizon: None,
                    confidence_level: 0.8,
                    include_uncertainty: false,
                };
                if let Ok(prediction) = model.predict(&prediction_request) {
                    fold_predictions.push(prediction.throughput);
                    fold_actuals.push(data_point.throughput);
                }
            }
            if !fold_predictions.is_empty() {
                let mae = fold_predictions
                    .iter()
                    .zip(fold_actuals.iter())
                    .map(|(p, a)| (p - a).abs())
                    .sum::<f64>() as f32
                    / fold_predictions.len() as f32;
                fold_scores.push(1.0 - mae / 100.0);
                all_predictions.extend(fold_predictions);
                all_actuals.extend(fold_actuals);
            }
        }
        if fold_scores.is_empty() {
            return Err(anyhow!("Cross-validation produced no valid scores"));
        }
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };
            let value = calculator.calculate(&all_predictions, &all_actuals)?;
            metrics.insert(*metric, value);
        }
        let average_score = fold_scores.iter().sum::<f32>() / fold_scores.len() as f32;
        Ok(ValidationResult {
            metrics,
            cv_scores: fold_scores,
            confidence: average_score.clamp(0.0, 1.0),
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32,
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: all_predictions
                    .iter()
                    .zip(all_actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.05,
                    heteroscedasticity_p_value: 0.6,
                    normality_p_value: 0.7,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }
    fn name(&self) -> &str {
        "CrossValidation"
    }
    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= self.folds * 2
    }
}
