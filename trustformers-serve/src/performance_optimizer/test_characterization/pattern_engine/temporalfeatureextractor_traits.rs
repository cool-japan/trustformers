//! # TemporalFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `TemporalFeatureExtractor`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::TestExecutionData;
use anyhow::Result;
use std::collections::HashMap;

use super::functions::FeatureExtractor;
use super::types::TemporalFeatureExtractor;

impl Default for TemporalFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for TemporalFeatureExtractor {
    fn extract_features(&self, _data: &TestExecutionData) -> Result<Vec<f64>> {
        Ok(vec![0.0; 6])
    }
    fn feature_names(&self) -> Vec<String> {
        vec!["duration".to_string(), "frequency".to_string()]
    }
    fn feature_importance(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
