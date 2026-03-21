//! # ExplainedVarianceCalculator - Trait Implementations
//!
//! This module contains trait implementations for `ExplainedVarianceCalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::ExplainedVarianceCalculator;

impl MetricCalculator for ExplainedVarianceCalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
        let var_actual =
            actuals.iter().map(|a| (a - mean_actual).powi(2)).sum::<f64>() / actuals.len() as f64;
        let var_residual = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (a - p).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        let explained_variance =
            if var_actual > 1e-12 { 1.0 - var_residual / var_actual } else { 0.0 };
        Ok(explained_variance as f32)
    }
    fn name(&self) -> &str {
        "ExplainedVariance"
    }
}
