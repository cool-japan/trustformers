//! # MSLECalculator - Trait Implementations
//!
//! This module contains trait implementations for `MSLECalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::MSLECalculator;

impl MetricCalculator for MSLECalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        if predictions.iter().any(|&p| p < 0.0) || actuals.iter().any(|&a| a < 0.0) {
            return Err(anyhow!("MSLE requires non-negative values"));
        }
        let msle = predictions
            .iter()
            .zip(actuals.iter())
            .map(|(p, a)| ((1.0 + p).ln() - (1.0 + a).ln()).powi(2))
            .sum::<f64>()
            / predictions.len() as f64;
        Ok(msle as f32)
    }
    fn name(&self) -> &str {
        "MSLE"
    }
}
