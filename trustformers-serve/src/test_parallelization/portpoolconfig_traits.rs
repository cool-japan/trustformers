//! # PortPoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `PortPoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::PortPoolConfig;

impl Default for PortPoolConfig {
    fn default() -> Self {
        Self {
            start_port: 10000,
            end_port: 20000,
            reserved_ports: vec![],
            allocation_timeout: Duration::from_secs(5),
        }
    }
}
