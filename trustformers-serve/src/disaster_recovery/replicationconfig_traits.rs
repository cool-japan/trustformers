//! # ReplicationConfig - Trait Implementations
//!
//! This module contains trait implementations for `ReplicationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    ConflictResolution, ConsistencyLevel, ReplicationConfig, ReplicationMode,
    ReplicationMonitoring, ReplicationTarget, ReplicationType,
};

impl Default for ReplicationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            mode: ReplicationMode::Asynchronous,
            targets: vec![ReplicationTarget {
                target_id: "dr-site-1".to_string(),
                endpoint: "https://dr1.example.com/replication".to_string(),
                replication_type: ReplicationType::FullReplica,
                lag_tolerance_seconds: 30,
                priority: 1,
            }],
            consistency_level: ConsistencyLevel::EventualConsistency,
            conflict_resolution: ConflictResolution::LastWriterWins,
            monitoring: ReplicationMonitoring {
                lag_alert_threshold_seconds: 60,
                failure_alert_threshold: 3,
                health_check_interval_seconds: 30,
            },
        }
    }
}
