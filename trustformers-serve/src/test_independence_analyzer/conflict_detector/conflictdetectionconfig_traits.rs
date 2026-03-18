//! # ConflictDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `ConflictDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ConflictDetectionConfig, ConflictSensitivity, ResourceConflictThresholds};

impl Default for ConflictDetectionConfig {
    fn default() -> Self {
        Self {
            aggressive_detection: false,
            sensitivity_level: ConflictSensitivity::Moderate,
            enable_ml_patterns: true,
            confidence_threshold: 0.7,
            predictive_analysis: true,
            max_analysis_time: Duration::from_millis(500),
            detailed_logging: false,
            resource_thresholds: ResourceConflictThresholds::default(),
        }
    }
}
