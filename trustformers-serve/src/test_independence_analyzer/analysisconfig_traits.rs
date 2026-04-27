//! # AnalysisConfig - Trait Implementations
//!
//! This module contains trait implementations for `AnalysisConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

pub use super::analysis_cache::{AnalysisCache, CacheConfig};
pub use super::conflict_detector::{
    ConflictDetectionConfig, ConflictDetectionDetails, ConflictDetectionStatistics,
    ConflictDetector, ConflictImpactAnalysis, ConflictResolutionOption, ConflictSensitivity,
    DetectedConflict,
};
pub use super::resource_database::{
    CleanupResult, DatabaseConfig, ResourceAllocationEvent, ResourceTypeDefinition,
    ResourceUsageDatabase, ResourceUsageRecord, TestUsageSummary, UsageReport,
};
pub use super::test_grouping_engine::{
    GroupCharacteristics, GroupRequirements, GroupingEngineConfig, GroupingMetrics,
    GroupingStrategy, GroupingStrategyType, TestGroup, TestGroupingEngine,
};
pub use super::types::*;

use std::time::Duration;

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            enable_advanced_dependency_analysis: true,
            enable_ml_conflict_prediction: false,
            enable_adaptive_grouping: true,
            max_analysis_time: Duration::from_secs(60),
            cache_config: CacheConfig::default(),
            conflict_detection_config: ConflictDetectionConfig::default(),
            grouping_config: GroupingEngineConfig::default(),
            database_config: DatabaseConfig::default(),
            enable_performance_metrics: true,
            quality_thresholds: AnalysisQualityThresholds::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn test_default_enables_advanced_dependency_analysis() {
        let config = AnalysisConfig::default();
        assert!(config.enable_advanced_dependency_analysis);
    }

    #[test]
    fn test_default_disables_ml_conflict_prediction() {
        let config = AnalysisConfig::default();
        assert!(!config.enable_ml_conflict_prediction);
    }

    #[test]
    fn test_default_enables_adaptive_grouping() {
        let config = AnalysisConfig::default();
        assert!(config.enable_adaptive_grouping);
    }

    #[test]
    fn test_default_max_analysis_time() {
        let config = AnalysisConfig::default();
        assert_eq!(config.max_analysis_time, Duration::from_secs(60));
    }

    #[test]
    fn test_default_enables_performance_metrics() {
        let config = AnalysisConfig::default();
        assert!(config.enable_performance_metrics);
    }

    #[test]
    fn test_max_analysis_time_positive() {
        let config = AnalysisConfig::default();
        assert!(config.max_analysis_time > Duration::from_secs(0));
    }

    #[test]
    fn test_quality_thresholds_has_defaults() {
        let config = AnalysisConfig::default();
        assert!(config.quality_thresholds.min_dependency_accuracy > 0.0);
        assert!(config.quality_thresholds.min_conflict_accuracy > 0.0);
    }

    #[test]
    fn test_enable_ml_prediction_toggle() {
        let mut config = AnalysisConfig::default();
        assert!(!config.enable_ml_conflict_prediction);
        config.enable_ml_conflict_prediction = true;
        assert!(config.enable_ml_conflict_prediction);
    }

    #[test]
    fn test_disable_advanced_dependency_analysis() {
        let mut config = AnalysisConfig::default();
        assert!(config.enable_advanced_dependency_analysis);
        config.enable_advanced_dependency_analysis = false;
        assert!(!config.enable_advanced_dependency_analysis);
    }

    #[test]
    fn test_max_analysis_time_override() {
        let mut config = AnalysisConfig::default();
        config.max_analysis_time = Duration::from_secs(300);
        assert_eq!(config.max_analysis_time, Duration::from_secs(300));
    }

    #[test]
    fn test_config_default_is_repeatable() {
        let c1 = AnalysisConfig::default();
        let c2 = AnalysisConfig::default();
        assert_eq!(
            c1.enable_ml_conflict_prediction,
            c2.enable_ml_conflict_prediction
        );
        assert_eq!(c1.max_analysis_time, c2.max_analysis_time);
        assert_eq!(c1.enable_performance_metrics, c2.enable_performance_metrics);
    }

    #[test]
    fn test_quality_threshold_dependency_in_range() {
        let config = AnalysisConfig::default();
        let t = &config.quality_thresholds;
        assert!(t.min_dependency_accuracy >= 0.0 && t.min_dependency_accuracy <= 1.0);
    }

    #[test]
    fn test_quality_threshold_conflict_in_range() {
        let config = AnalysisConfig::default();
        let t = &config.quality_thresholds;
        assert!(t.min_conflict_accuracy >= 0.0 && t.min_conflict_accuracy <= 1.0);
    }

    #[test]
    fn test_quality_threshold_grouping_in_range() {
        let config = AnalysisConfig::default();
        let t = &config.quality_thresholds;
        assert!(t.min_grouping_quality >= 0.0 && t.min_grouping_quality <= 1.0);
    }

    #[test]
    fn test_analysis_time_less_than_quality_threshold_time() {
        let config = AnalysisConfig::default();
        // The config analysis time should be <= quality threshold's max time
        assert!(config.max_analysis_time <= config.quality_thresholds.max_analysis_time);
    }

    #[test]
    fn test_all_boolean_flags_accessible() {
        let config = AnalysisConfig::default();
        let _ = config.enable_advanced_dependency_analysis;
        let _ = config.enable_ml_conflict_prediction;
        let _ = config.enable_adaptive_grouping;
        let _ = config.enable_performance_metrics;
    }

    #[test]
    fn test_cache_config_accessible() {
        let config = AnalysisConfig::default();
        // Just verify the field exists and is accessible
        let _ = &config.cache_config;
    }

    #[test]
    fn test_grouping_config_accessible() {
        let config = AnalysisConfig::default();
        let _ = &config.grouping_config;
    }

    #[test]
    fn test_database_config_accessible() {
        let config = AnalysisConfig::default();
        let _ = &config.database_config;
    }

    #[test]
    fn test_conflict_detection_config_accessible() {
        let config = AnalysisConfig::default();
        let _ = &config.conflict_detection_config;
    }

    #[test]
    fn test_ml_disabled_in_basic_mode() {
        // In default mode, ML is disabled for performance
        let config = AnalysisConfig::default();
        assert!(
            !config.enable_ml_conflict_prediction,
            "ML should be disabled by default to ensure baseline performance"
        );
    }
}
