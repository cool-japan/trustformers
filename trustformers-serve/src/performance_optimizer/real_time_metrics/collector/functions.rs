//! Auto-generated module
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::Result;
use std::collections::HashMap;

use super::types::{
    CollectionError, ErrorRecoveryAction, ErrorType, PublishError, PublishErrorType,
    PublishRecoveryAction, RecoveryStrategy, RetryStrategy,
};

/// Trait for sample rate adjustment algorithms
pub trait SampleRateAlgorithm {
    /// Calculate optimal sample rate
    fn calculate_rate(
        &self,
        current_load: f32,
        target_accuracy: f32,
        resource_availability: f32,
    ) -> Result<f32>;
    /// Get algorithm name
    fn name(&self) -> &str;
    /// Get algorithm parameters
    fn parameters(&self) -> HashMap<String, f32>;
    /// Update algorithm configuration
    fn update_config(&mut self, config: HashMap<String, f32>) -> Result<()>;
}
/// Trait for collection error handling
pub trait CollectionErrorHandler {
    /// Handle collection error
    fn handle_error(&self, error: CollectionError) -> Result<ErrorRecoveryAction>;
    /// Get error recovery strategy
    fn recovery_strategy(&self, error_type: ErrorType) -> RecoveryStrategy;
    /// Report error for analysis
    fn report_error(&self, error: CollectionError) -> Result<()>;
}
/// Trait for publish error handling
pub trait PublishErrorHandler {
    /// Handle publish error
    fn handle_error(&self, error: PublishError) -> Result<PublishRecoveryAction>;
    /// Get retry strategy
    fn retry_strategy(&self, error_type: PublishErrorType) -> RetryStrategy;
    /// Report publish failure
    fn report_failure(&self, error: PublishError) -> Result<()>;
}
