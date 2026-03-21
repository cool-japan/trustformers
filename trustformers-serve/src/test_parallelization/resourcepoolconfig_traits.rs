//! # ResourcePoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourcePoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DatabasePoolConfig, GpuPoolConfig, PortPoolConfig, ResourcePoolConfig, TempDirPoolConfig,
};

impl Default for ResourcePoolConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            network_port_pool: PortPoolConfig::default(),
            temp_directory_pool: TempDirPoolConfig::default(),
            gpu_device_pool: GpuPoolConfig::default(),
            database_pool: DatabasePoolConfig::default(),
        }
    }
}
