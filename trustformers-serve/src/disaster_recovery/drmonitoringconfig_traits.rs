//! # DRMonitoringConfig - Trait Implementations
//!
//! This module contains trait implementations for `DRMonitoringConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{DRAlertThresholds, DRMonitoringConfig, MetricsConfig};

impl Default for DRMonitoringConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            monitoring_interval_seconds: 30,
            metrics: MetricsConfig {
                rto_tracking: true,
                rpo_tracking: true,
                replication_lag_tracking: true,
                site_health_tracking: true,
                backup_status_tracking: true,
            },
            alert_thresholds: DRAlertThresholds {
                rto_threshold_seconds: 600,
                rpo_threshold_seconds: 300,
                replication_lag_threshold_seconds: 120,
                backup_failure_threshold: 2,
                site_unavailable_threshold_seconds: 60,
            },
        }
    }
}
