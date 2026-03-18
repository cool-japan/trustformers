//! # ResourceManagementConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceManagementConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ConflictResolutionConfig, ResourceCleanupConfig, ResourceManagementConfig,
    ResourceMonitoringConfig, ResourcePoolConfig,
};

impl Default for ResourceManagementConfig {
    fn default() -> Self {
        Self {
            resource_pools: ResourcePoolConfig::default(),
            conflict_resolution: ConflictResolutionConfig::default(),
            resource_monitoring: ResourceMonitoringConfig::default(),
            resource_cleanup: ResourceCleanupConfig::default(),
        }
    }
}
