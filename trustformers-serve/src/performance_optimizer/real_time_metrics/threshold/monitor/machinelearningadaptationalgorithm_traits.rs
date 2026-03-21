//! # MachineLearningAdaptationAlgorithm - Trait Implementations
//!
//! This module contains trait implementations for `MachineLearningAdaptationAlgorithm`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `ThresholdAdaptationAlgorithm`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use tracing::warn;

use super::super::super::types::{AlertEvent, TimestampedMetrics};
use super::super::adaptive_controller::ThresholdAdaptationAlgorithm;
use super::super::error::Result;

use super::types::MachineLearningAdaptationAlgorithm;

impl Default for MachineLearningAdaptationAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

impl ThresholdAdaptationAlgorithm for MachineLearningAdaptationAlgorithm {
    fn adapt_threshold(
        &self,
        current_threshold: f64,
        metrics_history: &[TimestampedMetrics],
        alert_history: &[AlertEvent],
    ) -> Result<f64> {
        if metrics_history.len() < 10 {
            return Ok(current_threshold);
        }
        let features = self.extract_features(metrics_history);
        if alert_history.len() >= 5 {
            let training_features: Vec<Vec<f64>> = vec![features.clone()];
            let training_targets: Vec<f64> = vec![current_threshold];
            if let Err(e) = self.train_model(&training_features, &training_targets) {
                warn!("ML model training failed: {}", e);
                return Ok(current_threshold);
            }
        }
        let model = self.model_state.lock().expect("Model state lock poisoned");
        let predicted_threshold = self.predict_with_features(&model, &features);
        let bounded_threshold =
            predicted_threshold.clamp(current_threshold * 0.5, current_threshold * 2.0);
        Ok(bounded_threshold)
    }
    fn name(&self) -> &str {
        "machine_learning_adaptation"
    }
    fn confidence(&self, data_quality: f32) -> f32 {
        let model = self.model_state.lock().expect("Model state lock poisoned");
        model.model_accuracy * data_quality
    }
}
