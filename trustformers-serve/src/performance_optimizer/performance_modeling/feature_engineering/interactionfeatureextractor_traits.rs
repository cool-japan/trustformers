//! # InteractionFeatureExtractor - Trait Implementations
//!
//! This module contains trait implementations for `InteractionFeatureExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::functions::FeatureExtractor;
use super::types::{ExtractedFeatures, InteractionFeatureExtractor};

impl FeatureExtractor for InteractionFeatureExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "parallelism_x_cores".to_string(),
            "parallelism_x_load".to_string(),
            "cores_x_memory".to_string(),
            "cpu_intensity_x_parallelism".to_string(),
            "memory_intensity_x_cores".to_string(),
            "load_x_io_wait".to_string(),
            "duration_x_complexity".to_string(),
        ];
        for data_point in data_points {
            let parallelism = data_point.parallelism as f64;
            let system = &data_point.system_state;
            let test_chars = &data_point.test_characteristics;
            let point_features = vec![
                parallelism * (system.available_cores as f64),
                parallelism * (system.load_average as f64),
                (system.available_cores as f64) * (system.available_memory_mb as f64),
                (test_chars.resource_intensity.cpu_intensity as f64) * parallelism,
                (test_chars.resource_intensity.memory_intensity as f64)
                    * (system.available_cores as f64),
                (system.load_average as f64) * (system.io_wait_percent as f64),
                test_chars.average_duration.as_secs_f64()
                    * (test_chars.dependency_complexity as f64),
            ];
            features.push(point_features);
        }
        Ok(ExtractedFeatures { features, names })
    }
    fn name(&self) -> &str {
        "InteractionFeatureExtractor"
    }
    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Parallelism × Available cores".to_string(),
            "Parallelism × System load".to_string(),
            "Available cores × Memory".to_string(),
            "CPU intensity × Parallelism".to_string(),
            "Memory intensity × Cores".to_string(),
            "System load × I/O wait".to_string(),
            "Duration × Dependency complexity".to_string(),
        ]
    }
}
