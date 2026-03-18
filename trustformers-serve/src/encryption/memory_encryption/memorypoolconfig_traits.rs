//! # MemoryPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `MemoryPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{GrowthStrategy, MemoryPoolConfig, ProtectionLevel};

impl Default for MemoryPoolConfig {
    fn default() -> Self {
        Self {
            name: "default".to_string(),
            block_size: 4096,
            initial_size: 100,
            max_size: 1000,
            growth_strategy: GrowthStrategy::Linear {
                increment: 50,
            },
            encrypted: true,
            protection_level: ProtectionLevel::Basic,
        }
    }
}

