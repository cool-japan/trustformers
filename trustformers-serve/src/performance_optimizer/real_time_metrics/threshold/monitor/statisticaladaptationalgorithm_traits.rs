//! # StatisticalAdaptationAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `StatisticalAdaptationAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ThresholdAdaptationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::super::super::types::{AlertEvent, TimestampedMetrics};
use super::super::adaptive_controller::ThresholdAdaptationAlgorithm;
use super::super::error::{Result, ThresholdError};

use super::types::StatisticalAdaptationAlgorithm;

impl Default for StatisticalAdaptationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdAdaptationAlgorithm for StatisticalAdaptationAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        _alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < self.config.min_data_points {
            return Ok(current_threshold);
        }
        let mut values = Vec::new();
        for metrics in metrics_history.iter().rev().take(100) {
            if let Some(value) = metrics.metrics.values().first() {
                values.push(*value);
            }
        }
        if values.is_empty() {
            return Ok(current_threshold);
        }
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();
        let adapted_threshold = mean + 2.0 * std_dev;
        let adjustment =
            (adapted_threshold - current_threshold) * self.config.adaptation_sensitivity as f64;
        let new_threshold = current_threshold + adjustment;
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.adaptations_performed += 1;
        Ok(new_threshold.max(0.0))
    }
    fn name(&self) -> &str {
        "statistical_adaptation"
    }
    fn confidence(&self, data_quality: f32) -> f32 {
        if data_quality >= self.config.confidence_threshold {
            0.9
        } else {
            data_quality * 0.8
        }
    }
}
