//! # ResourceAlertConfig - Trait Implementations
//!
//! This module contains trait implementations for `ResourceAlertConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::time::Duration;

use super::types::{AlertAction, AlertLevel, ResourceAlertConfig};

impl Default for ResourceAlertConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            cooldown_period: Duration::from_secs(60),
            escalation_levels: vec![
                AlertLevel {
                    level: "warning".to_string(),
                    threshold: 0.7,
                    action: AlertAction::Log,
                },
                AlertLevel {
                    level: "critical".to_string(),
                    threshold: 0.9,
                    action: AlertAction::Throttle,
                },
            ],
        }
    }
}
