//! # ScoringWeights - Trait Implementations
//!
//! This module contains trait implementations for `ScoringWeights`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for ScoringWeights {
    fn default() -> Self {
        Self {
            recency_weight: 0.3,
            severity_weight: 0.4,
            frequency_weight: 0.1,
            user_attention_weight: 0.1,
            system_impact_weight: 0.1,
        }
    }
}
