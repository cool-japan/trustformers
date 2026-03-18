//! # HoldOutValidation - Trait Implementations
//!
//! This module contains trait implementations for `HoldOutValidation`.
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

use crate::performance_optimizer::performance_modeling::types::{
    DistributionInfo, DistributionType, PerformancePredictor, PredictionRequest, ResidualAnalysis,
    TestDataStatistics, ValidationConfig, ValidationDetails, ValidationMetric, ValidationResult,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::functions::{MetricCalculator, ValidationStrategy};
use super::types::{HoldOutValidation, MAECalculator, RMSECalculator, RSquaredCalculator};

#[async_trait]
impl ValidationStrategy for HoldOutValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let test_count = (data.len() as f32 * self.test_size) as usize;
        if test_count == 0 {
            return Err(anyhow!("Test set would be empty"));
        }
        let (_train_data, test_data) = data.split_at(data.len() - test_count);
        let mut predictions = Vec::new();
        let mut actuals = Vec::new();
        for data_point in test_data {
            let prediction_request = PredictionRequest {
                parallelism_levels: vec![data_point.parallelism],
                test_characteristics: data_point.test_characteristics.clone(),
                system_state: data_point.system_state.clone(),
                prediction_horizon: None,
                confidence_level: 0.8,
                include_uncertainty: false,
            };
            match model.predict(&prediction_request) {
                Ok(prediction) => {
                    predictions.push(prediction.throughput);
                    actuals.push(data_point.throughput);
                },
                Err(e) => {
                    tracing::warn!("Prediction failed during validation: {}", e);
                    continue;
                },
            }
        }
        if predictions.is_empty() {
            return Err(anyhow!("No valid predictions generated"));
        }
        let mut metrics = HashMap::new();
        for metric in &config.metrics {
            let calculator = match metric {
                ValidationMetric::MeanAbsoluteError => &MAECalculator as &dyn MetricCalculator,
                ValidationMetric::RootMeanSquaredError => &RMSECalculator,
                ValidationMetric::RSquared => &RSquaredCalculator,
                _ => continue,
            };
            let value = calculator.calculate(&predictions, &actuals)?;
            metrics.insert(*metric, value);
        }
        let confidence = self.calculate_confidence(&metrics);
        Ok(ValidationResult {
            metrics,
            cv_scores: vec![confidence],
            confidence,
            details: ValidationDetails {
                test_samples: predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: actuals.iter().sum::<f64>() as f32 / actuals.len() as f32,
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: predictions
                    .iter()
                    .zip(actuals.iter())
                    .map(|(p, a)| (p - a) as f32)
                    .collect(),
                residual_analysis: ResidualAnalysis {
                    autocorrelation: 0.1,
                    heteroscedasticity_p_value: 0.5,
                    normality_p_value: 0.5,
                    outliers: Vec::new(),
                },
            },
            validated_at: Utc::now(),
        })
    }
    fn name(&self) -> &str {
        "HoldOutValidation"
    }
    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= 10
    }
}
