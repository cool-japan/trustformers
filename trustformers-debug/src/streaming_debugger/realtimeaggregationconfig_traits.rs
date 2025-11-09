//! # RealTimeAggregationConfig - Trait Implementations
//!
//! This module contains trait implementations for `RealTimeAggregationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

impl Default for RealTimeAggregationConfig {
    fn default() -> Self {
        Self {
            enable_aggregation: true,
            window_size_seconds: 5,
            aggregation_functions: vec![
                AggregationFunction::Mean,
                AggregationFunction::Max,
                AggregationFunction::Min,
                AggregationFunction::Count,
            ],
            enable_sliding_window: true,
            max_windows: 100,
            enable_custom_rules: false,
        }
    }
}
