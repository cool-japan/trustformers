//! # BootstrapValidation - Trait Implementations
//!
//! This module contains trait implementations for `BootstrapValidation`.
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
use super::types::{BootstrapValidation, MAECalculator, RMSECalculator, RSquaredCalculator};

#[async_trait]
impl ValidationStrategy for BootstrapValidation {
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult> {
        let mut bootstrap_scores = Vec::new();
        let mut all_predictions = Vec::new();
        let mut all_actuals = Vec::new();
        for bootstrap_iter in 0..self.n_bootstrap {
            let bootstrap_sample = self.create_bootstrap_sample(data);
            let mut iter_predictions = Vec::new();
            let mut iter_actuals = Vec::new();
            for data_point in &bootstrap_sample {
                let prediction_request = PredictionRequest {
                    parallelism_levels: vec![data_point.parallelism],
                    test_characteristics: data_point.test_characteristics.clone(),
                    system_state: data_point.system_state.clone(),
                    prediction_horizon: None,
                    confidence_level: 0.8,
                    include_uncertainty: false,
                };
                if let Ok(prediction) = model.predict(&prediction_request) {
                    iter_predictions.push(prediction.throughput);
                    iter_actuals.push(data_point.throughput);
                }
            }
            if !iter_predictions.is_empty() {
                let mae = iter_predictions
                    .iter()
                    .zip(iter_actuals.iter())
                    .map(|(p, a)| (p - a).abs())
                    .sum::<f64>() as f32
                    / iter_predictions.len() as f32;
                bootstrap_scores.push(1.0 - mae / 100.0);
                if bootstrap_iter < 10 {
                    all_predictions.extend(iter_predictions);
                    all_actuals.extend(iter_actuals);
                }
            }
        }
        if bootstrap_scores.is_empty() {
            return Err(anyhow!("Bootstrap validation produced no scores"));
        }
        let mut metrics = HashMap::new();
        if !all_predictions.is_empty() {
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
        }
        let average_score = bootstrap_scores.iter().sum::<f32>() / bootstrap_scores.len() as f32;
        Ok(ValidationResult {
            metrics,
            cv_scores: bootstrap_scores,
            confidence: average_score.clamp(0.0, 1.0),
            details: ValidationDetails {
                test_samples: all_predictions.len(),
                test_statistics: TestDataStatistics {
                    mean_target: if !all_actuals.is_empty() {
                        all_actuals.iter().sum::<f64>() as f32 / all_actuals.len() as f32
                    } else {
                        0.0
                    },
                    target_std: 1.0,
                    feature_correlations: HashMap::new(),
                    distribution_info: DistributionInfo {
                        distribution_type: DistributionType::Normal,
                        parameters: HashMap::new(),
                        normality_p_value: 0.5,
                    },
                },
                prediction_errors: if !all_predictions.is_empty() {
                    all_predictions
                        .iter()
                        .zip(all_actuals.iter())
                        .map(|(p, a)| (p - a) as f32)
                        .collect()
                } else {
                    Vec::new()
                },
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
        "BootstrapValidation"
    }
    fn is_applicable(&self, data_size: usize) -> bool {
        data_size >= 10
    }
}
