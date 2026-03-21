//! # HyperparameterTuningConfig - Trait Implementations
//!
//! This module contains trait implementations for `HyperparameterTuningConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use crate::performance_optimizer::performance_modeling::types::TuningAlgorithm;

use super::types::HyperparameterTuningConfig;

impl Default for HyperparameterTuningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            algorithm: TuningAlgorithm::RandomSearch,
            max_iterations: 50,
            timeout: Duration::from_secs(1800),
            cv_folds: 5,
            search_space: HashMap::new(),
        }
    }
}
