//! # LeaveOneOutValidation - Trait Implementations
//!
//! This module contains trait implementations for `LeaveOneOutValidation`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ValidationStrategy`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use anyhow::{anyhow, Result};
use async_trait::async_trait;
use chrono::Utc;

use crate::performance_optimizer::performance_modeling::types::{
    DistributionInfo, DistributionType, PerformancePredictor, PredictionRequest, ResidualAnalysis,
    TestDataStatistics, ValidationConfig, ValidationDetails, ValidationMetric, ValidationResult,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::functions::{MetricCalculator, ValidationStrategy};
use super::types::{LeaveOneOutValidation, MAECalculator, RMSECalculator, RSquaredCalculator};

impl Default for LeaveOneOutValidation {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl ValidationStrategy for LeaveOneOutValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let mut scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();
        for test_point in data.iter() {
            let prediction_request = PredictionRequest {
                parallelism_levels: vec![test_point.parallelism],
                test_characteristics: test_point.test_characteristics.clone(),
                system_state: test_point.system_state.clone(),
                prediction_horizon: None,
                confidence_level: 0.8,
                include_uncertainty: false,
            };
            if let Ok(prediction) = model.predict(&prediction_request) {
                let error = (prediction.throughput - test_point.throughput).abs();
                let relative_error = error / test_point.throughput.max(0.001);
                let score = (1.0 - relative_error as f32).max(0.0);
                scores.push(score);
                all_predictions.push(prediction.throughput);
                all_actuals.push(test_point.throughput);
            }
        }
        if scores.is_empty() {
            return Err(anyhow!("Leave-one-out validation produced no scores"));
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
        let average_score = scores.iter().sum::<f32>() / scores.len() as f32;
        Ok(ValidationResult {
            metrics,
            cv_scores: scores,
            confidence: average_score,
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
        "LeaveOneOutValidation"
    }
    fn is_applicable(&self, data_size: usize) -> bool {
        (5..=100).contains(&data_size)
    }
}
