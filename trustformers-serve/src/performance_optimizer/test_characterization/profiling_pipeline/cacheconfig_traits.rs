//! # CacheConfig - Trait Implementations
//!
//! This module contains trait implementations for `CacheConfig`.
//!
//! ## Implemented Traits
//!
//! - `From`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{CacheConfig, ProfilingPipelineConfig};

impl From<&ProfilingPipelineConfig> for CacheConfig {
    fn from(config: &ProfilingPipelineConfig) -> Self {
        Self {
            enable_caching: config.enable_caching,
            cache_size_limit: config.cache_size_limit,
            cache_ttl: config.cache_ttl,
        }
    }
}
