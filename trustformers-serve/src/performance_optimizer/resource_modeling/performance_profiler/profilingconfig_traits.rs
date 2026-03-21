//! # ProfilingConfig - Trait Implementations
//!
//! This module contains trait implementations for `ProfilingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::*;
use std::time::Duration;

use super::types::ProfilingConfig;

impl Default for ProfilingConfig {
    fn default() -> Self {
        Self {
            cpu_config: CpuProfilingConfig::default(),
            memory_config: MemoryProfilingConfig::default(),
            io_config: IoProfilingConfig::default(),
            network_config: NetworkProfilingConfig::default(),
            gpu_config: GpuProfilingConfig::default(),
            cache_config: CacheAnalysisConfig::default(),
            benchmark_config: BenchmarkConfig::default(),
            processing_config: ResultsProcessingConfig::default(),
            validation_config: ValidationConfig::default(),
            enable_gpu_profiling: true,
            cache_results: true,
            profiling_timeout: Duration::from_secs(300),
            enable_concurrent_profiling: true,
            max_concurrent_profilers: 6,
        }
    }
}
