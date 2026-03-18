//! # ProfilingPipelineConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProfilingPipelineConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{ProfilingPipelineConfig, ResourceLimits, ValidationStrictnessLevel};

impl Default for ProfilingPipelineConfig {
    fn default() -> Self {
        Self {
            max_concurrent_sessions: 10,
            stage_timeout: Duration::from_secs(120),
            pipeline_timeout: Duration::from_secs(600),
            enable_caching: true,
            cache_size_limit: 1000,
            cache_ttl: Duration::from_secs(3600),
            enable_validation: true,
            validation_strictness: ValidationStrictnessLevel::Standard,
            enable_parallel_stages: true,
            max_parallel_stages: 4,
            data_collection_interval: Duration::from_millis(100),
            quality_threshold: 0.8,
            enable_comprehensive_reporting: true,
            max_retry_attempts: 3,
            enable_metrics_collection: true,
            resource_limits: ResourceLimits {
                max_cpu_utilization: 0.8,
                max_memory_bytes: 2 * 1024 * 1024 * 1024,
                max_disk_io_rate: 100 * 1024 * 1024,
                max_network_io_rate: 50 * 1024 * 1024,
            },
        }
    }
}
