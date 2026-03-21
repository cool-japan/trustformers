//! # LoadBalancingConfig - Trait Implementations
//!
//! This module contains trait implementations for `LoadBalancingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{LoadBalancingConfig, LoadBalancingStrategy};

impl Default for LoadBalancingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            strategy: LoadBalancingStrategy::ResourceBased,
            rebalancing_interval: Duration::from_secs(10),
            load_threshold: 0.8,
            work_stealing: true,
        }
    }
}
