//! # BasicFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `BasicFeatureExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::functions::FeatureExtractor;
use super::types::{BasicFeatureExtractor, ExtractedFeatures};

impl FeatureExtractor for BasicFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "parallelism".to_string(),
            "parallelism_sqrt".to_string(),
            "parallelism_log".to_string(),
            "parallelism_squared".to_string(),
        ];
        for data_point in data_points {
            let parallelism = data_point.parallelism as f64;
            let point_features = vec![
                parallelism,
                parallelism.sqrt(),
                parallelism.ln(),
                parallelism * parallelism,
            ];
            features.push(point_features);
        }
        Ok(ExtractedFeatures { features, names })
    }
    fn name(&self) -> &str {
        "BasicFeatureExtractor"
    }
    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Raw parallelism level".to_string(),
            "Square root of parallelism".to_string(),
            "Natural logarithm of parallelism".to_string(),
            "Parallelism squared".to_string(),
        ]
    }
}
