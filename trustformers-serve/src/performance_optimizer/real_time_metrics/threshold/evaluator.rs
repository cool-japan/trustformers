//! Threshold evaluator trait definition.

use super::super::types::ThresholdConfig;
use super::error::Result;
use super::types::ThresholdEvaluation;
use std::fmt::Debug;

/// Threshold evaluator trait for threshold monitoring
///
/// Interface for threshold evaluators that assess metric values against
/// configured thresholds and generate alerts when violations occur.
pub trait ThresholdEvaluator: Debug + Send + Sync {
    /// Evaluate threshold against current value
    fn evaluate(&self, config: &ThresholdConfig, value: f64) -> Result<ThresholdEvaluation>;

    /// Get evaluator name
    fn name(&self) -> &str;

    /// Check if evaluator supports threshold type
    fn supports_threshold(&self, threshold_type: &str) -> bool;

    /// Get evaluation confidence for given data quality
    fn confidence(&self, data_quality: f32) -> f32 {
        data_quality.clamp(0.0, 1.0)
    }

    /// Perform initialization if needed
    fn initialize(&mut self) -> Result<()> {
        Ok(())
    }

    /// Clean up resources
    fn cleanup(&mut self) -> Result<()> {
        Ok(())
    }
}
