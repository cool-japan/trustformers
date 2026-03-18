//! # SiteConfig - Trait Implementations
//!
//! This module contains trait implementations for `SiteConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{HealthCheckConfig, SiteCapacity, SiteConfig, SiteStatus, SiteType};

impl Default for SiteConfig {
    fn default() -> Self {
        Self {
            site_id: "primary-site".to_string(),
            name: "Primary Site".to_string(),
            location: "us-east-1".to_string(),
            site_type: SiteType::Primary,
            endpoints: vec!["https://api.example.com".to_string()],
            priority: 0,
            capacity: SiteCapacity {
                max_requests_per_second: 10000,
                max_concurrent_users: 50000,
                storage_capacity_gb: 50000,
                compute_capacity: 1.0,
            },
            health_check: HealthCheckConfig {
                enabled: true,
                interval_seconds: 10,
                timeout_seconds: 5,
                failure_threshold: 3,
                success_threshold: 2,
            },
            status: SiteStatus::Active,
        }
    }
}
