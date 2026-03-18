//! # ThreadPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `ThreadPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::*;

use super::types::{LoadBalancingAlgorithm, ThreadPoolConfig};

impl Default for ThreadPoolConfig {
    fn default() -> Self {
        Self {
            min_threads: 2,
            max_threads: num_cpus::get() * 2,
            scaling_threshold: 0.8,
            scale_up_delay: Duration::from_secs(30),
            scale_down_delay: Duration::from_secs(300),
            load_balancing: LoadBalancingAlgorithm::LoadBased,
        }
    }
}
