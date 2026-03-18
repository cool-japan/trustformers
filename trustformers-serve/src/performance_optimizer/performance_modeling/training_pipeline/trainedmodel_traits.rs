//! # TrainedModel - Trait Implementations
//!
//! This module contains trait implementations for `TrainedModel`.
//!
//! ## Implemented Traits
//!
//! - `PerformancePredictor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;

use crate::performance_optimizer::performance_modeling::types::{
    ModelAccuracyMetrics, PerformancePrediction, PerformancePredictor, PredictionRequest,
};

use super::types::TrainedModel;

impl PerformancePredictor for TrainedModel {
    fn predict(&self, request: &PredictionRequest) -> Result<PerformancePrediction> {
        self.model.predict(request)
    }
    fn get_accuracy(&self) -> ModelAccuracyMetrics {
        self.model.get_accuracy()
    }
    fn name(&self) -> &str {
        self.model.name()
    }
    fn supports_online_learning(&self) -> bool {
        self.model.supports_online_learning()
    }
}
