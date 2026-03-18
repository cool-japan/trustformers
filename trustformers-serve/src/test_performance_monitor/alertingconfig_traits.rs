//! # AlertingConfig - Trait Implementations
//!
//! This module contains trait implementations for `AlertingConfig`.
//!
//! ## Implemented Traits
//!
//! - `Default`
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::collections::HashMap;
use std::time::Duration;

use super::types::{
    AlertCondition, AlertRule, AlertSeverity, AlertThrottlingConfig, AlertingConfig,
    EscalationConfig, NotificationChannel, NotificationChannelConfig, NotificationChannelType,
};

impl Default for AlertingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            rules: vec![
                AlertRule {
                    name: "High Timeout Rate".to_string(),
                    condition: AlertCondition::TimeoutFailureRate {
                        threshold: 0.1,
                        window: Duration::from_secs(300),
                    },
                    severity: AlertSeverity::Warning,
                    notification_channels: vec!["console".to_string()],
                    enabled: true,
                },
                AlertRule {
                    name: "Performance Regression".to_string(),
                    condition: AlertCondition::PerformanceRegression {
                        threshold: 0.2,
                        category: None,
                    },
                    severity: AlertSeverity::Critical,
                    notification_channels: vec!["console".to_string()],
                    enabled: true,
                },
            ],
            notification_channels: vec![NotificationChannel {
                name: "console".to_string(),
                channel_type: NotificationChannelType::Console,
                config: NotificationChannelConfig {
                    settings: HashMap::new(),
                },
                enabled: true,
            }],
            throttling: AlertThrottlingConfig {
                min_interval: Duration::from_secs(300),
                max_alerts_per_hour: 10,
                escalation: EscalationConfig {
                    enabled: false,
                    levels: Vec::new(),
                },
            },
        }
    }
}
