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
    PerformancePredictor, PredictionRequest, ValidationConfig, ValidationMetric, ValidationResult,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::functions::{MetricCalculator, ValidationStrategy};
use super::residuals::measured_details;
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
            details: measured_details(&predictions, &actuals),
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
