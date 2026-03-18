//! # AnomalyDetectionConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnomalyDetectionConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::AnomalyDetectionConfig;

impl Default for AnomalyDetectionConfig {
    fn default() -> Self {
        Self {
            enabled_algorithms: vec![
                "statistical".to_string(),
                "threshold".to_string(),
                "pattern".to_string(),
            ],
            sensitivity: 0.8,
            confidence_threshold: 0.7,
            history_window: Duration::from_secs(3600),
            pattern_recognition: true,
            ml_enabled: false,
            ml_update_interval: Duration::from_secs(86400),
        }
    }
}
