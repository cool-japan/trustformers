//! # ProfilingRequest - Trait Implementations
//!
//! This module contains trait implementations for `ProfilingRequest`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use crate::performance_optimizer::test_characterization::types::{
    ProfilingOptions, TestExecutionData,
};
use chrono::Utc;
use std::collections::HashMap;

use super::types::{ProfilingPriority, ProfilingRequest, ProfilingStageType};

impl Default for ProfilingRequest {
    fn default() -> Self {
        Self {
            test_id: String::new(),
            test_data: TestExecutionData::default(),
            profiling_options: ProfilingOptions::default(),
            priority: ProfilingPriority::Normal,
            context: HashMap::new(),
            timestamp: Utc::now(),
            stages: vec![
                ProfilingStageType::PreExecution,
                ProfilingStageType::ResourceAnalysis,
                ProfilingStageType::ConcurrencyAnalysis,
                ProfilingStageType::PostExecution,
                ProfilingStageType::Validation,
                ProfilingStageType::Aggregation,
            ],
        }
    }
}
