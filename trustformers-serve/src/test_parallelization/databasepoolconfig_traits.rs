//! # DatabasePoolConfig - Trait Implementations
//!
//! This module contains trait implementations for `DatabasePoolConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{DatabaseCleanupStrategy, DatabaseIsolationStrategy, DatabasePoolConfig};

impl Default for DatabasePoolConfig {
    fn default() -> Self {
        Self {
            max_connections: 10,
            connection_timeout: Duration::from_secs(30),
            isolation_strategy: DatabaseIsolationStrategy::PerSchema,
            cleanup_strategy: DatabaseCleanupStrategy::Truncate,
        }
    }
}
