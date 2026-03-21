//! # NotificationConfig - Trait Implementations
//!
//! This module contains trait implementations for `NotificationConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use super::types::{
    DREventType, NotificationChannel, NotificationConfig, NotificationRule, NotificationSeverity,
};

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            channels: vec![
                NotificationChannel::Email {
                    addresses: vec!["admin@example.com".to_string()],
                },
                NotificationChannel::Slack {
                    webhook_url: "https://hooks.slack.com/services/example".to_string(),
                    channel: "#incidents".to_string(),
                },
            ],
            rules: vec![NotificationRule {
                event_type: DREventType::FailoverTriggered,
                severity: NotificationSeverity::Critical,
                channels: vec!["email".to_string(), "slack".to_string()],
                cooldown_seconds: 300,
            }],
        }
    }
}
