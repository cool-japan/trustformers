//! # GroupingEngineConfig - Trait Implementations
//!
//! This module contains trait implementations for `GroupingEngineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{
    GroupingBalancingWeights, GroupingEngineConfig, GroupingQualityThresholds, GroupingStrategyType,
};

impl Default for GroupingEngineConfig {
    fn default() -> Self {
        Self {
            default_strategy: GroupingStrategyType::Balanced,
            max_tests_per_group: 10,
            min_tests_per_group: 2,
            target_resource_utilization: 0.7,
            max_resource_utilization: 0.9,
            adaptive_grouping: true,
            ml_optimization: false,
            grouping_timeout: Duration::from_secs(30),
            detailed_logging: false,
            balancing_weights: GroupingBalancingWeights::default(),
            quality_thresholds: GroupingQualityThresholds::default(),
        }
    }
}
