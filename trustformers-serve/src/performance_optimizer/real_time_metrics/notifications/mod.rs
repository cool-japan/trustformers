//! # Comprehensive Notifications Module for Real-Time Metrics System
//!
//! This module provides a robust, feature-rich notification system for processing alerts
//! from the threshold monitoring system and delivering them through multiple channels with
//! configurable reliability guarantees, intelligent routing, and comprehensive tracking.
//!
//! ## Key Components
//!
//! - **NotificationManager**: Central notification orchestration and routing system
//! - **Alert Processors**: Multiple alert processing strategies with enhanced capabilities
//! - **Notification Channels**: Pluggable multi-channel architecture with health monitoring
//! - **Message Formatting**: Intelligent templating and formatting for different channels
//! - **Delivery Guarantees**: Configurable reliability levels (BestEffort, AtLeastOnce, ExactlyOnce)
//! - **Rate Limiting**: Advanced throttling and spam prevention mechanisms
//! - **Retry Engine**: Robust retry logic with exponential backoff and circuit breakers
//! - **Health Monitor**: Real-time channel health monitoring and automatic failover
//! - **Audit System**: Comprehensive notification tracking and historical analysis
//! - **Escalation Engine**: Advanced escalation workflows and policy management
//!
//! ## Features
//!
//! - **Multi-Channel Support**: Log, Email, Webhook, Slack, SMS, PagerDuty with extensible architecture
//! - **Intelligent Routing**: Dynamic channel selection based on severity, content, and availability
//! - **Template Engine**: Advanced message templating with context-aware formatting
//! - **Delivery Guarantees**: Configurable reliability with persistence and acknowledgment tracking
//! - **Rate Limiting**: Intelligent throttling with burst allowances and adaptive limits
//! - **Circuit Breakers**: Automatic failover and recovery for unreliable channels
//! - **Retry Logic**: Exponential backoff with jitter and maximum retry limits
//! - **Health Monitoring**: Real-time channel health assessment and diagnostics
//! - **Audit Trail**: Complete notification history with delivery confirmation tracking
//! - **Performance Optimization**: High-throughput processing with minimal latency overhead
//! - **Thread Safety**: Concurrent notification processing with lock-free optimizations
//! - **Configuration**: Extensive runtime configuration with hot-reloading support
//!
//! ## Architecture
//!
//! The notification system follows a pluggable architecture with clear separation of concerns:
//!
//! ```text
//! ┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
//! │ Alert Generator │───▶│ NotificationMgr  │───▶│ Channel Router  │
//! └─────────────────┘    └──────────────────┘    └─────────────────┘
//!                                 │                        │
//!                        ┌────────▼────────┐              │
//!                        │ Alert Processor │              │
//!                        └─────────────────┘              │
//!                                 │                        │
//!                        ┌────────▼────────┐              │
//!                        │ Message Format  │              │
//!                        └─────────────────┘              │
//!                                 │                        │
//!                                 ▼                        ▼
//!        ┌─────────────────────────────────────────────────────────────┐
//!        │                 Notification Channels                      │
//!        ├─────────┬─────────┬─────────┬─────────┬─────────┬─────────┤
//!        │   Log   │  Email  │Webhook  │ Slack   │   SMS   │PagerDuty│
//!        └─────────┴─────────┴─────────┴─────────┴─────────┴─────────┘
//!                                 │
//!                        ┌────────▼────────┐
//!                        │ Delivery Engine │
//!                        └─────────────────┘
//!                                 │
//!                        ┌────────▼────────┐
//!                        │  Audit System   │
//!                        └─────────────────┘
//! ```
//!
//! ## Usage Example
//!
//! ```rust,ignore
//! use crate::performance_optimizer::real_time_metrics::notifications::{
//!     NotificationManager, NotificationConfig, DeliveryGuarantee
//! };
//!
//! #[tokio::main]
//! async fn main() -> Result<()> {
//!     // Create notification manager with configuration
//!     let config = NotificationConfig {
//!         delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
//!         retry_attempts: 3,
//!         rate_limit_per_minute: 100,
//!         enable_health_monitoring: true,
//!         ..Default::default()
//!     };
//!
//!     let notification_manager = NotificationManager::new(config).await?;
//!
//!     // Start the notification system
//!     notification_manager.start().await?;
//!
//!     // Process alerts (typically called by threshold monitor)
//!     notification_manager.process_alert(alert_event).await?;
//!
//!     Ok(())
//! }
//! ```

// Module declarations
pub mod audit;
pub mod channels;
pub mod delivery;
pub mod escalation;
pub mod formatters;
pub mod health_monitor;
pub mod manager;
pub mod processors;
pub mod rate_limiting;
pub mod retry;
pub mod types;

// Re-export commonly used types and structures
pub use types::{
    AtomicF32, ChannelConfig, DeliveryGuarantee, DeliveryResult, HealthCheckConfig, Notification,
    NotificationConfig, NotificationPriority, NotificationStats, ProcessedNotification,
    ProcessingStatus, RetryPolicy,
};

pub use manager::NotificationManager;

pub use processors::{
    AlertProcessor, CriticalAlertProcessor, DefaultAlertProcessor, PerformanceAlertProcessor,
    ResourceAlertProcessor,
};

pub use channels::{
    EmailNotificationChannel, LogNotificationChannel, NotificationChannel,
    PagerDutyNotificationChannel, SlackNotificationChannel, SmsNotificationChannel,
    WebhookNotificationChannel,
};

pub use formatters::{
    ChannelFormatter, CompiledTemplate, FormattingStats, MessageFormatter, TemplateEngine,
    TemplateFunction,
};

pub use delivery::{
    DeliveryEngine, DeliveryRecord, DeliveryStats, PendingDelivery, RetryItem, RetryScheduler,
    RetryStats,
};

pub use rate_limiting::{
    AdaptiveRateConfig, AdaptiveRateController, ChannelRateLimitStats, ChannelRateLimiter,
    GlobalRateLimitStats, GlobalRateLimiter, LoadMetrics, PriorityQueue, RateAdjustment,
    RateLimiter, RateLimitingStats, ThrottledNotification, TokenBucket,
};

pub use health_monitor::{ChannelHealth, ChannelHealthMonitor};

pub use audit::AuditSystem;

pub use escalation::EscalationEngine;

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::performance_optimizer::real_time_metrics::types::*;
    use chrono::Utc;
    use std::collections::HashMap;
    use std::time::Duration;

    #[tokio::test]
    async fn test_notification_manager_creation() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_default_alert_processor() {
        let processor = DefaultAlertProcessor::new().await.expect("Failed to create processor");

        let alert = AlertEvent {
            timestamp: Utc::now(),
            alert_id: "test_alert".to_string(),
            correlation_id: Some("test_correlation".to_string()),
            threshold: ThresholdConfig {
                name: "test_threshold".to_string(),
                metric: "cpu_utilization".to_string(),
                warning_threshold: 0.8,
                critical_threshold: 0.9,
                direction: ThresholdDirection::Above,
                adaptive: false,
                evaluation_window: Duration::from_secs(60),
                min_trigger_count: 1,
                cooldown_period: Duration::from_secs(300),
                escalation_policy: String::new(),
            },
            current_value: 0.85,
            threshold_value: 0.8,
            severity: SeverityLevel::High,
            message: "CPU utilization is high".to_string(),
            context: HashMap::new(),
            actions: Vec::new(),
            suppression_info: None,
            metadata: HashMap::new(),
        };

        let notifications = processor.process_alert(&alert).await.expect("Failed to process alert");
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].alert_id, "test_alert");
    }

    #[tokio::test]
    async fn test_log_notification_channel() {
        let channel = LogNotificationChannel::new().await.expect("Failed to create channel");

        let notification = Notification {
            id: "test_notification".to_string(),
            alert_id: "test_alert".to_string(),
            channels: vec!["log".to_string()],
            recipients: vec!["test@example.com".to_string()],
            subject: "Test Notification".to_string(),
            content: "This is a test notification".to_string(),
            priority: NotificationPriority::Normal,
            severity: SeverityLevel::Medium,
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            created_at: Utc::now(),
            deadline: None,
            template: None,
            template_vars: HashMap::new(),
            metadata: HashMap::new(),
            tags: Vec::new(),
            escalation_policy: None,
            correlation_id: None,
        };

        let result = channel
            .send_notification(&notification)
            .await
            .expect("Failed to send notification");
        assert!(result.success);
        assert!(result.delivered_at.is_some());
    }
}
