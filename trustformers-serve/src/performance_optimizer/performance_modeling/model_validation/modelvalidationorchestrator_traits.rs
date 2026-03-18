//! # ModelValidationOrchestrator - Trait Implementations
//!
//! This module contains trait implementations for `ModelValidationOrchestrator`.
//!
//! ## Implemented Traits
//!
//! - `ModelValidator`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use anyhow::{anyhow, Result};

use crate::performance_optimizer::performance_modeling::types::{
    ModelValidator, PerformancePredictor, ValidationConfig, ValidationResult,
    ValidationStrategyType,
};
use crate::performance_optimizer::types::PerformanceDataPoint;

use super::types::ModelValidationOrchestrator;

impl ModelValidator for ModelValidationOrchestrator {
    fn validate(
        &self,
        model: &dyn PerformancePredictor,
        test_data: &[PerformanceDataPoint],
    ) -> Result<ValidationResult> {
        let config = self.config.read();
        let strategy = self
            .strategies
            .get(&config.strategy)
            .ok_or_else(|| anyhow!("Validation strategy not found: {:?}", config.strategy))?;
        let rt = tokio::runtime::Runtime::new()?;
        rt.block_on(strategy.validate(model, test_data, &*config))
    }
    fn strategy(&self) -> ValidationStrategyType {
        self.config.read().strategy
    }
    fn config(&self) -> &ValidationConfig {
        unsafe { &*(&*self.config.read() as *const ValidationConfig) }
    }
}
