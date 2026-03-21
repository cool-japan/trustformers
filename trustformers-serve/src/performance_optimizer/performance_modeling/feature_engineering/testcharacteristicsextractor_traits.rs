//! # TestCharacteristicsExtractor - Trait Implementations
//!
//! This module contains trait implementations for `TestCharacteristicsExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::functions::FeatureExtractor;
use super::types::{ExtractedFeatures, TestCharacteristicsExtractor};

impl FeatureExtractor for TestCharacteristicsExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "avg_duration_secs".to_string(),
            "cpu_intensity".to_string(),
            "memory_intensity".to_string(),
            "io_intensity".to_string(),
            "dependency_complexity".to_string(),
            "parallel_capable".to_string(),
            "max_safe_concurrency".to_string(),
            "resource_intensity_score".to_string(),
        ];
        for data_point in data_points {
            let test_chars = &data_point.test_characteristics;
            let point_features = vec![
                test_chars.average_duration.as_secs_f64(),
                test_chars.resource_intensity.cpu_intensity as f64,
                test_chars.resource_intensity.memory_intensity as f64,
                test_chars.resource_intensity.io_intensity as f64,
                test_chars.dependency_complexity as f64,
                if test_chars.concurrency_requirements.parallel_capable { 1.0 } else { 0.0 },
                test_chars.concurrency_requirements.max_safe_concurrency as f64,
                (test_chars.resource_intensity.cpu_intensity
                    + test_chars.resource_intensity.memory_intensity
                    + test_chars.resource_intensity.io_intensity) as f64
                    / 3.0,
            ];
            features.push(point_features);
        }
        Ok(ExtractedFeatures { features, names })
    }
    fn name(&self) -> &str {
        "TestCharacteristicsExtractor"
    }
    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Average test duration in seconds".to_string(),
            "CPU intensity score".to_string(),
            "Memory intensity score".to_string(),
            "I/O intensity score".to_string(),
            "Test dependency complexity".to_string(),
            "Whether test is parallel-capable".to_string(),
            "Maximum safe concurrency level".to_string(),
            "Overall resource intensity score".to_string(),
        ]
    }
}
