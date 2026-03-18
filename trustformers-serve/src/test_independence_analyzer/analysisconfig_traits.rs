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
