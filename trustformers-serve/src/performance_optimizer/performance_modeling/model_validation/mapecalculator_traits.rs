//! # MAPECalculator - Trait Implementations
//!
//! This module contains trait implementations for `MAPECalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::MAPECalculator;

impl MetricCalculator for MAPECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        let mape = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| if a.abs() > 1e-12 { ((p - a) / a).abs() } else { 0.0 })
            .sum::<f64>()
            / predictions.len() as f64;
        Ok((mape * 100.0) as f32)
    }
    fn name(&self) -> &str {
        "MAPE"
    }
}
