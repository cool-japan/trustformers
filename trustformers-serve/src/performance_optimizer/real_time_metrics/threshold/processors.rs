//! Alert processor implementations.

use super::super::types::*;
use super::alert_manager::{
    AlertProcessor, Notification, NotificationChannel, NotificationPriority, ProcessedAlert,
};
use super::error::{Result, ThresholdError};
use chrono::Utc;
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

// =============================================================================
// ALERT PROCESSOR IMPLEMENTATIONS
// =============================================================================

/// Default alert processor for general alerts
pub struct DefaultAlertProcessor {
    /// Processor configuration
    config: DefaultProcessorConfig,
    /// Processing statistics
    stats: Arc<Mutex<ProcessorStats>>,
}

/// Configuration for default alert processor
#[derive(Debug, Clone)]
pub struct DefaultProcessorConfig {
    /// Priority level
    pub priority: u8,
    /// Enable detailed logging
    pub detailed_logging: bool,
    /// Include context in notifications
    pub include_context: bool,
}

/// Statistics for alert processors
#[derive(Debug, Clone)]
pub struct ProcessorStats {
    /// Alerts processed
    pub alerts_processed: u64,
    /// Processing errors
    pub processing_errors: u64,
    /// Average processing time
    pub avg_processing_time: Duration,
    /// Last processing time
    pub last_processing_time: Instant,
}

impl Default for ProcessorStats {
    fn default() -> Self {
        Self {
            alerts_processed: 0,
            processing_errors: 0,
            avg_processing_time: Duration::default(),
            last_processing_time: Instant::now(),
        }
    }
}

impl Default for DefaultAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl DefaultAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: DefaultProcessorConfig {
                priority: 50,
                detailed_logging: true,
                include_context: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn with_config(config: DefaultProcessorConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }

    pub fn get_stats(&self) -> ProcessorStats {
        self.stats.lock().expect("Stats lock poisoned").clone()
    }
}

impl AlertProcessor for DefaultAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = vec![Notification {
            id: format!("notif_{}", alert.alert_id),
            channel: "log".to_string(),
            recipients: vec!["system".to_string()],
            subject: format!("Alert: {}", alert.threshold.name),
            content: if self.config.include_context {
                format!("{}\n\nContext: {:?}", alert.message, alert.context)
            } else {
                alert.message.clone()
            },
            priority: match alert.severity {
                SeverityLevel::Critical => NotificationPriority::Critical,
                SeverityLevel::High => NotificationPriority::High,
                SeverityLevel::Medium => NotificationPriority::Normal,
                SeverityLevel::Low => NotificationPriority::Low,
                _ => NotificationPriority::Normal,
            },
            metadata: HashMap::new(),
        }];

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "default".to_string());
        results.insert(
            "notifications_created".to_string(),
            notifications.len().to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "default_alert_processor"
    }

    fn supports(&self, _alert: &AlertEvent) -> bool {
        true // Default processor supports all alerts
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Performance-focused alert processor
pub struct PerformanceAlertProcessor {
    config: PerformanceProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct PerformanceProcessorConfig {
    pub priority: u8,
    pub performance_threshold: f64,
    pub enable_auto_scaling_recommendations: bool,
}

impl Default for PerformanceAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl PerformanceAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: PerformanceProcessorConfig {
                priority: 80,
                performance_threshold: 0.8,
                enable_auto_scaling_recommendations: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for PerformanceAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = if matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms"
        ) {
            let mut notifs = vec![
                Notification {
                    id: format!("perf_notif_{}", alert.alert_id),
                    channel: "email".to_string(),
                    recipients: vec!["performance-team@company.com".to_string()],
                    subject: format!("Performance Alert: {}", alert.threshold.name),
                    content: format!(
                        "Performance issue detected: {}\n\nCurrent value: {}\nThreshold: {}\nSeverity: {:?}",
                        alert.message, alert.current_value, alert.threshold_value, alert.severity
                    ),
                    priority: NotificationPriority::High,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("metric_type".to_string(), alert.threshold.metric.clone());
                        metadata.insert("performance_issue".to_string(), "true".to_string());
                        metadata
                    },
                }
            ];

            // Add auto-scaling recommendation if enabled
            if self.config.enable_auto_scaling_recommendations
                && alert.current_value > self.config.performance_threshold
            {
                notifs.push(Notification {
                    id: format!("scaling_rec_{}", alert.alert_id),
                    channel: "webhook".to_string(),
                    recipients: vec!["auto-scaler".to_string()],
                    subject: "Auto-scaling Recommendation".to_string(),
                    content: format!(
                        "Consider scaling up due to performance alert: {}",
                        alert.alert_id
                    ),
                    priority: NotificationPriority::High,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("action".to_string(), "scale_up".to_string());
                        metadata.insert("trigger_alert".to_string(), alert.alert_id.clone());
                        metadata
                    },
                });
            }

            notifs
        } else {
            Vec::new()
        };

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "performance".to_string());
        results.insert("metric_type".to_string(), alert.threshold.metric.clone());
        results.insert(
            "auto_scaling_recommended".to_string(),
            (self.config.enable_auto_scaling_recommendations
                && alert.current_value > self.config.performance_threshold)
                .to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "performance_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms" | "response_time" | "cpu_utilization"
        )
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Resource-focused alert processor
pub struct ResourceAlertProcessor {
    config: ResourceProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct ResourceProcessorConfig {
    pub priority: u8,
    pub critical_resource_threshold: f64,
    pub enable_resource_optimization: bool,
}

impl Default for ResourceAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ResourceAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: ResourceProcessorConfig {
                priority: 70,
                critical_resource_threshold: 0.95,
                enable_resource_optimization: true,
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for ResourceAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let notifications = if matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization"
        ) {
            vec![Notification {
                id: format!("resource_notif_{}", alert.alert_id),
                channel: "webhook".to_string(),
                recipients: vec!["ops-team".to_string()],
                subject: format!("Resource Alert: {}", alert.threshold.name),
                content: format!(
                    "Resource issue detected: {}\n\nCurrent utilization: {:.2}%\nThreshold: {:.2}%",
                    alert.message,
                    alert.current_value * 100.0,
                    alert.threshold_value * 100.0
                ),
                priority: if alert.current_value > self.config.critical_resource_threshold {
                    NotificationPriority::Critical
                } else {
                    NotificationPriority::High
                },
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("resource_type".to_string(), alert.threshold.metric.clone());
                    metadata.insert(
                        "utilization".to_string(),
                        format!("{:.2}", alert.current_value),
                    );
                    if alert.current_value > self.config.critical_resource_threshold {
                        metadata.insert("critical_resource".to_string(), "true".to_string());
                    }
                    metadata
                },
            }]
        } else {
            Vec::new()
        };

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "resource".to_string());
        results.insert("resource_type".to_string(), alert.threshold.metric.clone());
        results.insert(
            "utilization_percent".to_string(),
            format!("{:.2}", alert.current_value * 100.0),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        "resource_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization" | "disk_usage" | "network_utilization"
        )
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

/// Critical alert processor for high-priority alerts
pub struct CriticalAlertProcessor {
    config: CriticalProcessorConfig,
    stats: Arc<Mutex<ProcessorStats>>,
}

#[derive(Debug, Clone)]
pub struct CriticalProcessorConfig {
    pub priority: u8,
    pub enable_immediate_escalation: bool,
    pub escalation_channels: Vec<String>,
}

impl Default for CriticalAlertProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl CriticalAlertProcessor {
    pub fn new() -> Self {
        Self {
            config: CriticalProcessorConfig {
                priority: 100, // Highest priority
                enable_immediate_escalation: true,
                escalation_channels: vec![
                    "email".to_string(),
                    "slack".to_string(),
                    "sms".to_string(),
                ],
            },
            stats: Arc::new(Mutex::new(ProcessorStats::default())),
        }
    }
}

impl AlertProcessor for CriticalAlertProcessor {
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert> {
        let start_time = Instant::now();

        let mut notifications = Vec::new();

        if matches!(alert.severity, SeverityLevel::Critical) {
            // Create multiple notifications for critical alerts
            for channel in &self.config.escalation_channels {
                notifications.push(Notification {
                    id: format!("critical_{}_{}", channel, alert.alert_id),
                    channel: channel.clone(),
                    recipients: match channel.as_str() {
                        "email" => vec!["oncall@company.com".to_string(), "manager@company.com".to_string()],
                        "slack" => vec!["#critical-alerts".to_string()],
                        "sms" => vec!["+1234567890".to_string()],
                        _ => vec!["system".to_string()],
                    },
                    subject: format!("🚨 CRITICAL ALERT: {}", alert.threshold.name),
                    content: format!(
                        "CRITICAL ALERT DETECTED\n\nAlert ID: {}\nMetric: {}\nCurrent Value: {}\nThreshold: {}\nSeverity: Critical\n\nImmediate attention required!\n\nMessage: {}",
                        alert.alert_id, alert.threshold.metric, alert.current_value, alert.threshold_value, alert.message
                    ),
                    priority: NotificationPriority::Emergency,
                    metadata: {
                        let mut metadata = HashMap::new();
                        metadata.insert("alert_type".to_string(), "critical".to_string());
                        metadata.insert("escalation_level".to_string(), "immediate".to_string());
                        metadata.insert("requires_acknowledgment".to_string(), "true".to_string());
                        metadata
                    },
                });
            }
        }

        let mut results = HashMap::new();
        results.insert("processor".to_string(), "critical".to_string());
        results.insert("severity".to_string(), format!("{:?}", alert.severity));
        results.insert(
            "immediate_escalation".to_string(),
            self.config.enable_immediate_escalation.to_string(),
        );
        results.insert(
            "notifications_sent".to_string(),
            notifications.len().to_string(),
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.alerts_processed += 1;
        stats.avg_processing_time = (stats.avg_processing_time + start_time.elapsed()) / 2;
        stats.last_processing_time = Instant::now();

        Ok(ProcessedAlert {
            alert: alert.clone(),
            processed_at: Utc::now(),
            results,
            notifications,
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("critical_processing".to_string(), "true".to_string());
                metadata.insert("escalation_triggered".to_string(), "true".to_string());
                metadata
            },
        })
    }

    fn name(&self) -> &str {
        "critical_alert_processor"
    }

    fn supports(&self, alert: &AlertEvent) -> bool {
        matches!(alert.severity, SeverityLevel::Critical)
    }

    fn priority(&self) -> u8 {
        self.config.priority
    }
}

// =============================================================================
// NOTIFICATION CHANNEL IMPLEMENTATIONS
// =============================================================================

/// Log notification channel
pub struct LogNotificationChannel {
    config: LogChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct LogChannelConfig {
    pub log_level: String,
    pub include_metadata: bool,
    pub format_json: bool,
}

#[derive(Debug, Default, Clone)]
pub struct ChannelStats {
    pub notifications_sent: u64,
    pub send_failures: u64,
    pub avg_send_time: Duration,
    pub last_send_time: Option<Instant>,
}

impl Default for LogNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl LogNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: LogChannelConfig {
                log_level: "info".to_string(),
                include_metadata: true,
                format_json: false,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn get_stats(&self) -> ChannelStats {
        self.stats.lock().expect("Stats lock poisoned").clone()
    }
}

impl NotificationChannel for LogNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        let log_message = if self.config.format_json {
            serde_json::json!({
                "notification_id": notification.id,
                "subject": notification.subject,
                "content": notification.content,
                "priority": format!("{:?}", notification.priority),
                "recipients": notification.recipients,
                "metadata": if self.config.include_metadata { Some(&notification.metadata) } else { None }
            }).to_string()
        } else {
            format!(
                "Alert notification: {} - {} (Priority: {:?}){}",
                notification.subject,
                notification.content,
                notification.priority,
                if self.config.include_metadata {
                    format!(" | Metadata: {:?}", notification.metadata)
                } else {
                    String::new()
                }
            )
        };

        match self.config.log_level.as_str() {
            "error" => error!("{}", log_message),
            "warn" => warn!("{}", log_message),
            "debug" => debug!("{}", log_message),
            "trace" => tracing::trace!("{}", log_message),
            _ => info!("{}", log_message),
        }

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "log"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "log"
    }

    fn max_message_size(&self) -> usize {
        100000 // 100KB for log messages
    }

    fn is_available(&self) -> bool {
        true // Log channel is always available
    }
}

/// Email notification channel
pub struct EmailNotificationChannel {
    config: EmailChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct EmailChannelConfig {
    pub smtp_server: String,
    pub smtp_port: u16,
    pub username: String,
    pub from_address: String,
    pub use_tls: bool,
    pub max_recipients: usize,
}

impl Default for EmailNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl EmailNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: EmailChannelConfig {
                smtp_server: "localhost".to_string(),
                smtp_port: 587,
                username: "alerts".to_string(),
                from_address: "alerts@company.com".to_string(),
                use_tls: true,
                max_recipients: 50,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn with_config(config: EmailChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }
}

impl NotificationChannel for EmailNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // Validate recipient count
        if notification.recipients.len() > self.config.max_recipients {
            return Err(ThresholdError::NotificationError(format!(
                "Too many recipients: {} (max: {})",
                notification.recipients.len(),
                self.config.max_recipients
            )));
        }

        // Placeholder for email sending implementation
        // In a real implementation, this would use an SMTP library like lettre
        info!(
            "Email notification sent to {:?}: {} - {} (Server: {}:{})",
            notification.recipients,
            notification.subject,
            notification.content,
            self.config.smtp_server,
            self.config.smtp_port
        );

        // Simulate email sending delay
        std::thread::sleep(Duration::from_millis(100));

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "email"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "email"
    }

    fn max_message_size(&self) -> usize {
        1000000 // 1MB for email messages
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check SMTP server connectivity
        true
    }
}

/// Webhook notification channel
pub struct WebhookNotificationChannel {
    config: WebhookChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
    client: reqwest::Client,
}

#[derive(Debug, Clone)]
pub struct WebhookChannelConfig {
    pub webhook_url: String,
    pub timeout_seconds: u64,
    pub retry_attempts: u8,
    pub auth_header: Option<String>,
    pub custom_headers: HashMap<String, String>,
}

impl Default for WebhookNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl WebhookNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: WebhookChannelConfig {
                webhook_url: "http://localhost:8080/webhook".to_string(),
                timeout_seconds: 30,
                retry_attempts: 3,
                auth_header: None,
                custom_headers: HashMap::new(),
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
            client: reqwest::Client::new(),
        }
    }

    pub fn with_config(config: WebhookChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
            client: reqwest::Client::new(),
        }
    }
}

impl NotificationChannel for WebhookNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // For now, just log the webhook notification
        // In a real implementation, this would make an HTTP request
        info!(
            "Webhook notification sent: {} - {} (URL: {})",
            notification.subject, notification.content, self.config.webhook_url
        );

        // Simulate webhook sending
        std::thread::sleep(Duration::from_millis(50));

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "webhook"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "webhook"
    }

    fn max_message_size(&self) -> usize {
        500000 // 500KB for webhook payloads
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check webhook URL accessibility
        true
    }
}

/// Slack notification channel
pub struct SlackNotificationChannel {
    config: SlackChannelConfig,
    stats: Arc<Mutex<ChannelStats>>,
}

#[derive(Debug, Clone)]
pub struct SlackChannelConfig {
    pub webhook_url: String,
    pub default_channel: String,
    pub username: String,
    pub icon_emoji: String,
    pub timeout_seconds: u64,
}

impl Default for SlackNotificationChannel {
    fn default() -> Self {
        Self::new()
    }
}

impl SlackNotificationChannel {
    pub fn new() -> Self {
        Self {
            config: SlackChannelConfig {
                webhook_url: "https://hooks.slack.com/services/...".to_string(),
                default_channel: "#alerts".to_string(),
                username: "AlertBot".to_string(),
                icon_emoji: ":warning:".to_string(),
                timeout_seconds: 30,
            },
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }

    pub fn with_config(config: SlackChannelConfig) -> Self {
        Self {
            config,
            stats: Arc::new(Mutex::new(ChannelStats::default())),
        }
    }
}

impl NotificationChannel for SlackNotificationChannel {
    fn send_notification(&self, notification: &Notification) -> Result<()> {
        let start_time = Instant::now();

        // Format message for Slack
        let slack_message = format!(
            "{} {}\n{}",
            match notification.priority {
                NotificationPriority::Emergency => "🚨",
                NotificationPriority::Critical => "❗",
                NotificationPriority::High => "⚠️",
                NotificationPriority::Normal => "ℹ️",
                NotificationPriority::Low => "💡",
            },
            notification.subject,
            notification.content
        );

        // Placeholder for Slack webhook implementation
        info!(
            "Slack notification sent to {}: {}",
            self.config.default_channel, slack_message
        );

        // Update statistics
        let mut stats = self
            .stats
            .lock()
            .map_err(|_| ThresholdError::InternalError("Lock poisoned".to_string()))?;
        stats.notifications_sent += 1;
        stats.avg_send_time = (stats.avg_send_time + start_time.elapsed()) / 2;
        stats.last_send_time = Some(Instant::now());

        Ok(())
    }

    fn name(&self) -> &str {
        "slack"
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "slack"
    }

    fn max_message_size(&self) -> usize {
        3000 // Slack message limit
    }

    fn is_available(&self) -> bool {
        // In a real implementation, this would check Slack webhook accessibility
        true
    }
}
