//! # PerformanceFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceFeatureExtractor`.
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
use super::types::PerformanceFeatureExtractor;

impl Default for PerformanceFeatureExtractor {
    fn default() -> Self {
        Self::new()
    }
}

impl FeatureExtractor for PerformanceFeatureExtractor {
    fn extract_features(&self, _data: &TestExecutionData) -> Result<Vec<f64>> {
        Ok(vec![0.0; 8])
    }
    fn feature_names(&self) -> Vec<String> {
        vec!["execution_time".to_string(), "throughput".to_string()]
    }
    fn feature_importance(&self) -> HashMap<String, f64> {
        HashMap::new()
    }
}
