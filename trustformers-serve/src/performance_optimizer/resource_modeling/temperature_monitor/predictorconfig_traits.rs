//! # PredictorConfig - Trait Implementations
//!
//! This module contains trait implementations for `PredictorConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::PredictorConfig;

impl Default for PredictorConfig {
    fn default() -> Self {
        Self {
            max_prediction_history: 1000,
        }
    }
}
