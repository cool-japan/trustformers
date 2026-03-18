//! # DisasterRecoveryConfig - Trait Implementations
//!
//! This module contains trait implementations for `DisasterRecoveryConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    BackupConfig, DRMonitoringConfig, DRTestingConfig, DisasterRecoveryConfig, FailoverConfig,
    HealthCheckConfig, NotificationConfig, ReplicationConfig, SiteCapacity, SiteConfig, SiteStatus,
    SiteType,
};

impl Default for DisasterRecoveryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rto_seconds: 300,
            rpo_seconds: 60,
            primary_site: SiteConfig::default(),
            dr_sites: vec![SiteConfig {
                site_id: "dr-site-1".to_string(),
                name: "Disaster Recovery Site 1".to_string(),
                location: "us-west-2".to_string(),
                site_type: SiteType::DisasterRecovery,
                endpoints: vec!["https://dr1.example.com".to_string()],
                priority: 1,
                capacity: SiteCapacity {
                    max_requests_per_second: 5000,
                    max_concurrent_users: 10000,
                    storage_capacity_gb: 10000,
                    compute_capacity: 0.8,
                },
                health_check: HealthCheckConfig {
                    enabled: true,
                    interval_seconds: 30,
                    timeout_seconds: 10,
                    failure_threshold: 3,
                    success_threshold: 2,
                },
                status: SiteStatus::Standby,
            }],
            failover: FailoverConfig::default(),
            replication: ReplicationConfig::default(),
            backup: BackupConfig::default(),
            monitoring: DRMonitoringConfig::default(),
            testing: DRTestingConfig::default(),
            notifications: NotificationConfig::default(),
        }
    }
}
