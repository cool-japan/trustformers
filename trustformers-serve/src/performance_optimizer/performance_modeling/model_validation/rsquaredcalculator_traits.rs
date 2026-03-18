//! # RSquaredCalculator - Trait Implementations
//!
//! This module contains trait implementations for `RSquaredCalculator`.
//!
//! ## Implemented Traits
//!
//! - `MetricCalculator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use super::functions::MetricCalculator;
use super::types::RSquaredCalculator;

impl MetricCalculator for RSquaredCalculator {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32> {
        if predictions.len() != actuals.len() || predictions.is_empty() {
            return Err(anyhow!("Predictions and actuals length mismatch or empty"));
        }
        let mean_actual = actuals.iter().sum::<f64>() / actuals.len() as f64;
        let ss_tot: f64 = actuals.iter().map(|a| (a - mean_actual).powi(2)).sum();
        let ss_res: f64 =
            predictions.iter().zip(actuals.iter()).map(|(p, a)| (a - p).powi(2)).sum();
        let r_squared = if ss_tot > 1e-12 { 1.0 - ss_res / ss_tot } else { 0.0 };
        Ok(r_squared as f32)
    }
    fn name(&self) -> &str {
        "R²"
    }
}
