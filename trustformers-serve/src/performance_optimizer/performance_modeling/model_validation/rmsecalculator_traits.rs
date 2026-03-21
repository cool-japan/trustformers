//! # RMSECalculator - Trait Implementations
//!
//! This module contains trait implementations for `RMSECalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::RMSECalculator;

impl MetricCalculator for RMSECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        let mse = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        Ok(mse.sqrt() as f32)
    }
    fn name(&self) -> &str {
        "RMSE"
    }
}
