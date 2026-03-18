//! # MAECalculator - Trait Implementations
//!
//! This module contains trait implementations for `MAECalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::MAECalculator;

impl MetricCalculator for MAECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        let mae = predictions.iter().zip(actuals.iter()).map(|(p, a)| (p - a).abs()).sum::<f64>()
            / predictions.len() as f64;
        Ok(mae as f32)
    }
    fn name(&self) -> &str {
        "MAE"
    }
}
