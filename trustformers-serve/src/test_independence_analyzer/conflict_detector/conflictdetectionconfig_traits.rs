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

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_aggressive_detection_disabled() {
        let config = ConflictDetectionConfig::default();
        assert!(!config.aggressive_detection);
    }

    #[test]
    fn test_default_sensitivity_moderate() {
        let config = ConflictDetectionConfig::default();
        let debug_str = format!("{:?}", config.sensitivity_level);
        assert!(debug_str.contains("Moderate"));
    }

    #[test]
    fn test_default_ml_patterns_enabled() {
        let config = ConflictDetectionConfig::default();
        assert!(config.enable_ml_patterns);
    }

    #[test]
    fn test_default_confidence_threshold() {
        let config = ConflictDetectionConfig::default();
        assert!((config.confidence_threshold - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_default_predictive_analysis_enabled() {
        let config = ConflictDetectionConfig::default();
        assert!(config.predictive_analysis);
    }

    #[test]
    fn test_default_max_analysis_time() {
        let config = ConflictDetectionConfig::default();
        assert_eq!(config.max_analysis_time, Duration::from_millis(500));
    }

    #[test]
    fn test_default_detailed_logging_disabled() {
        let config = ConflictDetectionConfig::default();
        assert!(!config.detailed_logging);
    }

    #[test]
    fn test_confidence_threshold_in_range() {
        let config = ConflictDetectionConfig::default();
        assert!(config.confidence_threshold >= 0.0 && config.confidence_threshold <= 1.0);
    }

    #[test]
    fn test_max_analysis_time_positive() {
        let config = ConflictDetectionConfig::default();
        assert!(config.max_analysis_time > Duration::from_millis(0));
    }

    #[test]
    fn test_enable_aggressive_detection() {
        let mut config = ConflictDetectionConfig::default();
        config.aggressive_detection = true;
        assert!(config.aggressive_detection);
    }

    #[test]
    fn test_enable_detailed_logging() {
        let mut config = ConflictDetectionConfig::default();
        config.detailed_logging = true;
        assert!(config.detailed_logging);
    }

    #[test]
    fn test_disable_ml_patterns() {
        let mut config = ConflictDetectionConfig::default();
        config.enable_ml_patterns = false;
        assert!(!config.enable_ml_patterns);
    }

    #[test]
    fn test_update_confidence_threshold() {
        let mut config = ConflictDetectionConfig::default();
        config.confidence_threshold = 0.95;
        assert!((config.confidence_threshold - 0.95).abs() < f32::EPSILON);
    }

    #[test]
    fn test_resource_thresholds_accessible() {
        let config = ConflictDetectionConfig::default();
        let _ = &config.resource_thresholds;
    }

    #[test]
    fn test_default_is_repeatable() {
        let c1 = ConflictDetectionConfig::default();
        let c2 = ConflictDetectionConfig::default();
        assert_eq!(c1.aggressive_detection, c2.aggressive_detection);
        assert_eq!(c1.enable_ml_patterns, c2.enable_ml_patterns);
        assert_eq!(c1.predictive_analysis, c2.predictive_analysis);
        assert_eq!(c1.detailed_logging, c2.detailed_logging);
        assert_eq!(c1.max_analysis_time, c2.max_analysis_time);
    }

    #[test]
    fn test_high_performance_config() {
        let mut config = ConflictDetectionConfig::default();
        config.enable_ml_patterns = false;
        config.predictive_analysis = false;
        config.max_analysis_time = Duration::from_millis(100);
        config.confidence_threshold = 0.5;
        assert!(!config.enable_ml_patterns);
        assert!(config.max_analysis_time < Duration::from_millis(500));
    }

    #[test]
    fn test_max_sensitivity_config() {
        let mut config = ConflictDetectionConfig::default();
        config.aggressive_detection = true;
        config.detailed_logging = true;
        config.confidence_threshold = 0.95;
        config.max_analysis_time = Duration::from_secs(2);
        assert!(config.aggressive_detection);
        assert!(config.detailed_logging);
        assert!(config.confidence_threshold > 0.7);
    }

    #[test]
    fn test_sensitivity_moderate_by_default() {
        let config = ConflictDetectionConfig::default();
        match config.sensitivity_level {
            ConflictSensitivity::Moderate => {},
            _ => panic!("expected Moderate sensitivity by default"),
        }
    }

    #[test]
    fn test_analysis_time_sub_second() {
        let config = ConflictDetectionConfig::default();
        assert!(config.max_analysis_time < Duration::from_secs(1));
    }

    #[test]
    fn test_ml_and_predictive_enabled_together() {
        let config = ConflictDetectionConfig::default();
        assert!(config.enable_ml_patterns && config.predictive_analysis);
    }
}
