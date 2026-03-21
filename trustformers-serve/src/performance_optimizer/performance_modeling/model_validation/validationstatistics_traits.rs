//! # ValidationStatistics - Trait Implementations
//!
//! This module contains trait implementations for `ValidationStatistics`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::ValidationStatistics;

impl Default for ValidationStatistics {
    fn default() -> Self {
        Self {
            total_validations: 0,
            average_confidence: 0.0,
            best_confidence: 0.0,
            validation_frequency: 0.0,
            last_validation: None,
        }
    }
}
