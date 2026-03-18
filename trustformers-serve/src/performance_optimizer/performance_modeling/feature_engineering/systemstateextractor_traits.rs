//! # SystemStateExtractor - Trait Implementations
//!
//! This module contains trait implementations for `SystemStateExtractor`.
//!
//! ## Implemented Traits
//!
//! - `FeatureExtractor`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::types::PerformanceDataPoint;
use anyhow::Result;

use super::functions::FeatureExtractor;
use super::types::{ExtractedFeatures, SystemStateExtractor};

impl FeatureExtractor for SystemStateExtractor {
    fn extract_features(&self, data_points: &[PerformanceDataPoint]) -> Result<ExtractedFeatures> {
        let mut features = Vec::new();
        let names = vec![
            "available_cores".to_string(),
            "available_memory_mb".to_string(),
            "load_average".to_string(),
            "active_processes".to_string(),
            "io_wait_percent".to_string(),
            "network_utilization".to_string(),
            "memory_pressure".to_string(),
            "cpu_pressure".to_string(),
        ];
        for data_point in data_points {
            let system = &data_point.system_state;
            let point_features = vec![
                system.available_cores as f64,
                system.available_memory_mb as f64,
                system.load_average as f64,
                system.active_processes as f64,
                system.io_wait_percent as f64,
                system.network_utilization as f64,
                (system.available_memory_mb as f64) / (system.available_cores as f64).max(1.0),
                system.load_average as f64 / (system.available_cores as f64).max(1.0),
            ];
            features.push(point_features);
        }
        Ok(ExtractedFeatures { features, names })
    }
    fn name(&self) -> &str {
        "SystemStateExtractor"
    }
    fn feature_descriptions(&self) -> Vec<String> {
        vec![
            "Available CPU cores".to_string(),
            "Available memory in MB".to_string(),
            "System load average".to_string(),
            "Number of active processes".to_string(),
            "I/O wait percentage".to_string(),
            "Network utilization".to_string(),
            "Memory pressure (memory per core)".to_string(),
            "CPU pressure (load per core)".to_string(),
        ]
    }
}
