//! # AuditTrailConfig - Trait Implementations
//!
//! This module contains trait implementations for `AuditTrailConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::AuditTrailConfig;

impl Default for AuditTrailConfig {
    fn default() -> Self {
        Self {
            max_size: 1_000_000,
            retention_period: Duration::from_secs(365 * 24 * 3600),
            encryption_enabled: true,
            tamper_protection: true,
            auto_archival: true,
        }
    }
}

