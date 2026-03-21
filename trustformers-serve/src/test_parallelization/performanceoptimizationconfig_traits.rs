//! # PerformanceOptimizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `PerformanceOptimizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    CpuScalingConfig, MemoryOptimizationConfig, ParallelPerformanceMonitoringConfig,
    PerformanceOptimizationConfig, TestBatchingConfig, WarmupOptimizationConfig,
};

impl Default for PerformanceOptimizationConfig {
    fn default() -> Self {
        Self {
            adaptive_parallelism: true,
            cpu_scaling: CpuScalingConfig::default(),
            memory_optimization: MemoryOptimizationConfig::default(),
            warmup_optimization: WarmupOptimizationConfig::default(),
            test_batching: TestBatchingConfig::default(),
            performance_monitoring: ParallelPerformanceMonitoringConfig::default(),
        }
    }
}
