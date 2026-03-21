//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::performance_modeling::types::{
    PerformancePredictor, ValidationConfig, ValidationResult,
};
use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;
use async_trait::async_trait;

/// Trait for validation strategies
#[async_trait]
pub trait ValidationStrategy: std::fmt::Debug + Send + Sync {
    /// Perform validation
    async fn validate(
        &self,
        model: &dyn PerformancePredictor,
        data: &[PerformanceDataPoint],
        config: &ValidationConfig,
    ) -> Result<ValidationResult>;
    /// Get strategy name
    fn name(&self) -> &str;
    /// Check if strategy is applicable
    fn is_applicable(&self, data_size: usize) -> bool;
}
/// Trait for metric calculators
pub trait MetricCalculator: std::fmt::Debug + Send + Sync {
    fn calculate(&self, predictions: &[f64], actuals: &[f64]) -> Result<f32>;
    fn name(&self) -> &str;
}
