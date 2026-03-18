//! # FailoverConfig - Trait Implementations
//!
//! This module contains trait implementations for `FailoverConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    FailoverConfig, FailoverStrategy, FailoverTrigger, RollbackCondition, RollbackConfig,
    TrafficSplittingConfig, TrafficStage,
};

impl Default for FailoverConfig {
    fn default() -> Self {
        Self {
            auto_failover_enabled: true,
            trigger_conditions: vec![
                FailoverTrigger::SiteUnavailable {
                    site_id: "primary-site".to_string(),
                },
                FailoverTrigger::HighErrorRate {
                    threshold: 0.1,
                    duration_seconds: 300,
                },
                FailoverTrigger::HighLatency {
                    threshold_ms: 5000,
                    duration_seconds: 300,
                },
            ],
            strategy: FailoverStrategy::HighestPriority,
            max_failover_time_seconds: 300,
            traffic_splitting: TrafficSplittingConfig {
                enabled: true,
                gradual_failover: true,
                failover_stages: vec![
                    TrafficStage {
                        percentage: 10,
                        duration_seconds: 60,
                    },
                    TrafficStage {
                        percentage: 50,
                        duration_seconds: 120,
                    },
                    TrafficStage {
                        percentage: 100,
                        duration_seconds: 0,
                    },
                ],
            },
            rollback: RollbackConfig {
                auto_rollback_enabled: true,
                rollback_conditions: vec![
                    RollbackCondition::PrimarySiteRecovered,
                    RollbackCondition::DRSiteUnhealthy,
                ],
                rollback_delay_seconds: 300,
            },
        }
    }
}
