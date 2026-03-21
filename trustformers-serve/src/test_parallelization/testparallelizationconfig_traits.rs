//! # TestParallelizationConfig - Trait Implementations
//!
//! This module contains trait implementations for `TestParallelizationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;

use super::types::{
    IndependenceAnalysisConfig, PerformanceOptimizationConfig, ResourceLimits,
    ResourceManagementConfig, SchedulingConfig, TestParallelizationConfig,
    TestSuiteOrganizationConfig,
};

impl Default for TestParallelizationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_concurrent_tests: num_cpus::get().max(4),
            resource_limits: ResourceLimits::default(),
            independence_analysis: IndependenceAnalysisConfig::default(),
            scheduling: SchedulingConfig::default(),
            performance_optimization: PerformanceOptimizationConfig::default(),
            resource_management: ResourceManagementConfig::default(),
            test_suite_organization: TestSuiteOrganizationConfig::default(),
            environment_overrides: HashMap::new(),
        }
    }
}
