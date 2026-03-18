//! # ResourceFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `ResourceFeatureExtractor`.
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
use super::types::ResourceFeatureExtractor;

impl Default for ResourceFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for ResourceFeatureExtractor {
    fn extract_features(&self, _data: &TestExecutionData) -> Result<Vec<f64>> {
        Ok(vec![0.0; 10])
    }
    fn feature_names(&self) -> Vec<String> {
        vec!["cpu_usage".to_string(), "memory_usage".to_string()]
    }
    fn feature_importance(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
