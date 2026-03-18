//! # RequestProfilingConfig - Trait Implementations
//!
//! This module contains trait implementations for `RequestProfilingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{ProfileExportFormat, RequestProfilingConfig};

impl Default for RequestProfilingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_profiles_in_memory: 10000,
            enable_detailed_timing: true,
            enable_resource_tracking: true,
            enable_call_stack_tracking: true,
            enable_memory_profiling: true,
            enable_cpu_profiling: true,
            enable_io_profiling: true,
            sampling_rate: 0.1,
            min_duration_to_profile_ms: 10,
            enable_performance_recommendations: true,
            enable_profile_aggregation: true,
            aggregation_window_secs: 300,
            enable_profile_export: true,
            profile_export_format: ProfileExportFormat::JSON,
            enable_flame_graphs: true,
            enable_bottleneck_detection: true,
            bottleneck_threshold_percent: 20.0,
        }
    }
}
