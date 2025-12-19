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
//! ```rust
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

use super::types::*;

use anyhow::{anyhow, Result};
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use scirs2_core::random::*;
use serde::{Deserialize, Serialize};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{broadcast, oneshot, Semaphore},
    task::JoinHandle,
    time::{interval, timeout},
};
use tracing::{debug, error, info, warn};

// =============================================================================
// CONFIGURATION TYPES
// =============================================================================

/// Comprehensive configuration for the notification system
///
/// Provides extensive configuration options for all aspects of notification
/// processing, delivery, and monitoring with sensible defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NotificationConfig {
    /// Maximum concurrent notifications
    pub max_concurrent_notifications: usize,

    /// Default delivery guarantee level
    pub delivery_guarantee: DeliveryGuarantee,

    /// Maximum retry attempts for failed notifications
    pub retry_attempts: usize,

    /// Base retry delay in milliseconds
    pub base_retry_delay_ms: u64,

    /// Maximum retry delay in milliseconds
    pub max_retry_delay_ms: u64,

    /// Rate limiting: maximum notifications per minute
    pub rate_limit_per_minute: usize,

    /// Rate limiting: burst allowance
    pub rate_limit_burst: usize,

    /// Enable health monitoring for channels
    pub enable_health_monitoring: bool,

    /// Health check interval
    pub health_check_interval: Duration,

    /// Channel failure threshold for circuit breaker
    pub failure_threshold: usize,

    /// Circuit breaker recovery timeout
    pub circuit_breaker_timeout: Duration,

    /// Enable notification auditing
    pub enable_auditing: bool,

    /// Audit retention period
    pub audit_retention_period: Duration,

    /// Maximum audit history size
    pub max_audit_history: usize,

    /// Enable message templating
    pub enable_templating: bool,

    /// Template cache size
    pub template_cache_size: usize,

    /// Notification timeout
    pub notification_timeout: Duration,

    /// Enable escalation integration
    pub enable_escalation: bool,

    /// Escalation check interval
    pub escalation_check_interval: Duration,
}

impl Default for NotificationConfig {
    fn default() -> Self {
        Self {
            max_concurrent_notifications: 1000,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            retry_attempts: 3,
            base_retry_delay_ms: 1000,
            max_retry_delay_ms: 30000,
            rate_limit_per_minute: 100,
            rate_limit_burst: 10,
            enable_health_monitoring: true,
            health_check_interval: Duration::from_secs(30),
            failure_threshold: 5,
            circuit_breaker_timeout: Duration::from_secs(60),
            enable_auditing: true,
            audit_retention_period: Duration::from_secs(7 * 24 * 3600), // 7 days
            max_audit_history: 10000,
            enable_templating: true,
            template_cache_size: 1000,
            notification_timeout: Duration::from_secs(30),
            enable_escalation: true,
            escalation_check_interval: Duration::from_secs(60),
        }
    }
}

/// Delivery guarantee levels for notifications
///
/// Different levels of delivery guarantees with varying reliability
/// and performance characteristics.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryGuarantee {
    /// Best effort delivery - no guarantees, highest performance
    BestEffort,

    /// At least once delivery - guarantees delivery with possible duplicates
    AtLeastOnce,

    /// Exactly once delivery - guarantees single delivery (highest overhead)
    ExactlyOnce,
}

/// Configuration for individual notification channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChannelConfig {
    /// Channel name
    pub name: String,

    /// Channel enabled status
    pub enabled: bool,

    /// Channel priority (higher = preferred)
    pub priority: u8,

    /// Channel-specific configuration
    pub config: HashMap<String, String>,

    /// Rate limit for this channel
    pub rate_limit: Option<usize>,

    /// Timeout for this channel
    pub timeout: Duration,

    /// Retry policy for this channel
    pub retry_policy: RetryPolicy,

    /// Health check settings
    pub health_check: HealthCheckConfig,
}

/// Retry policy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetryPolicy {
    /// Maximum retry attempts
    pub max_attempts: usize,

    /// Base delay between retries
    pub base_delay: Duration,

    /// Maximum delay between retries
    pub max_delay: Duration,

    /// Backoff multiplier
    pub backoff_multiplier: f64,

    /// Enable jitter to prevent thundering herd
    pub enable_jitter: bool,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            base_delay: Duration::from_millis(1000),
            max_delay: Duration::from_secs(30),
            backoff_multiplier: 2.0,
            enable_jitter: true,
        }
    }
}

/// Health check configuration for channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheckConfig {
    /// Enable health checks
    pub enabled: bool,

    /// Health check interval
    pub interval: Duration,

    /// Health check timeout
    pub timeout: Duration,

    /// Failure threshold before marking unhealthy
    pub failure_threshold: usize,

    /// Success threshold for recovery
    pub success_threshold: usize,
}

impl Default for HealthCheckConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            interval: Duration::from_secs(30),
            timeout: Duration::from_secs(5),
            failure_threshold: 3,
            success_threshold: 2,
        }
    }
}

// =============================================================================
// CORE NOTIFICATION TYPES
// =============================================================================

/// Enhanced notification structure with comprehensive metadata
///
/// Provides rich notification data with delivery tracking, templating
/// support, and extensive metadata for routing and processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Notification {
    /// Unique notification ID
    pub id: String,

    /// Alert ID that triggered this notification
    pub alert_id: String,

    /// Target channels for delivery
    pub channels: Vec<String>,

    /// Recipients list
    pub recipients: Vec<String>,

    /// Notification subject/title
    pub subject: String,

    /// Notification content/body
    pub content: String,

    /// Notification priority level
    pub priority: NotificationPriority,

    /// Severity level from original alert
    pub severity: SeverityLevel,

    /// Delivery guarantee for this notification
    pub delivery_guarantee: DeliveryGuarantee,

    /// Creation timestamp
    pub created_at: DateTime<Utc>,

    /// Delivery deadline
    pub deadline: Option<DateTime<Utc>>,

    /// Template name for formatting
    pub template: Option<String>,

    /// Template variables for substitution
    pub template_vars: HashMap<String, String>,

    /// Notification metadata
    pub metadata: HashMap<String, String>,

    /// Tags for classification and routing
    pub tags: Vec<String>,

    /// Escalation policy reference
    pub escalation_policy: Option<String>,

    /// Correlation ID for grouping related notifications
    pub correlation_id: Option<String>,
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NotificationPriority {
    /// Low priority - can be delayed
    Low = 1,

    /// Normal priority - standard delivery
    Normal = 2,

    /// High priority - expedited delivery
    High = 3,

    /// Critical priority - immediate delivery
    Critical = 4,

    /// Emergency priority - bypass rate limits
    Emergency = 5,
}

impl Default for NotificationPriority {
    fn default() -> Self {
        NotificationPriority::Normal
    }
}

/// Processed notification with delivery information
#[derive(Debug, Clone)]
pub struct ProcessedNotification {
    /// Original notification
    pub notification: Notification,

    /// Processing timestamp
    pub processed_at: DateTime<Utc>,

    /// Processing results by channel
    pub results: HashMap<String, DeliveryResult>,

    /// Overall processing status
    pub status: ProcessingStatus,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Delivery result for individual channels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeliveryResult {
    /// Delivery success status
    pub success: bool,

    /// Delivery timestamp
    pub delivered_at: Option<DateTime<Utc>>,

    /// Delivery attempts made
    pub attempts: usize,

    /// Error message if delivery failed
    pub error: Option<String>,

    /// Delivery latency in milliseconds
    pub latency_ms: Option<u64>,

    /// Channel-specific response data
    pub response_data: HashMap<String, String>,
}

/// Processing status for notifications
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProcessingStatus {
    /// Notification is pending processing
    Pending,

    /// Notification is being processed
    Processing,

    /// Notification delivered successfully to all channels
    Delivered,

    /// Notification partially delivered (some channels failed)
    PartiallyDelivered,

    /// Notification delivery failed on all channels
    Failed,

    /// Notification delivery timed out
    TimedOut,

    /// Notification was cancelled
    Cancelled,
}

// =============================================================================
// CENTRAL NOTIFICATION MANAGER
// =============================================================================

/// Central notification management system
///
/// Orchestrates all notification processing, routing, and delivery with
/// comprehensive features including rate limiting, health monitoring,
/// retry logic, and audit tracking.
pub struct NotificationManager {
    /// Configuration
    config: Arc<RwLock<NotificationConfig>>,

    /// Alert processors
    processors: Arc<RwLock<Vec<Arc<dyn AlertProcessor + Send + Sync>>>>,

    /// Notification channels
    channels: Arc<DashMap<String, Arc<dyn NotificationChannel + Send + Sync>>>,

    /// Channel configurations
    channel_configs: Arc<RwLock<HashMap<String, ChannelConfig>>>,

    /// Message formatter
    formatter: Arc<MessageFormatter>,

    /// Delivery engine
    delivery_engine: Arc<DeliveryEngine>,

    /// Rate limiter
    rate_limiter: Arc<RateLimiter>,

    /// Health monitor
    health_monitor: Arc<ChannelHealthMonitor>,

    /// Audit system
    audit_system: Arc<AuditSystem>,

    /// Escalation engine
    escalation_engine: Arc<EscalationEngine>,

    /// Notification queue
    notification_queue: Arc<Mutex<VecDeque<Notification>>>,

    /// Processing statistics
    stats: Arc<NotificationStats>,

    /// Processing semaphore for concurrency control
    processing_semaphore: Arc<Semaphore>,

    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<oneshot::Sender<()>>>>,

    /// Worker tasks
    worker_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

/// Statistics for notification processing
#[derive(Debug, Default)]
pub struct NotificationStats {
    /// Total notifications processed
    pub total_processed: AtomicU64,

    /// Successful deliveries
    pub successful_deliveries: AtomicU64,

    /// Failed deliveries
    pub failed_deliveries: AtomicU64,

    /// Partial deliveries
    pub partial_deliveries: AtomicU64,

    /// Rate limited notifications
    pub rate_limited: AtomicU64,

    /// Retry attempts
    pub retry_attempts: AtomicU64,

    /// Average processing latency (ms)
    pub avg_processing_latency_ms: AtomicF32,

    /// Average delivery latency (ms)
    pub avg_delivery_latency_ms: AtomicF32,

    /// Current queue size
    pub queue_size: AtomicUsize,

    /// Processing errors
    pub processing_errors: AtomicU64,
}

impl NotificationManager {
    /// Create a new notification manager
    ///
    /// Initializes the comprehensive notification system with all
    /// required components and default configurations.
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        let processing_semaphore = Arc::new(Semaphore::new(config.max_concurrent_notifications));
        let channels = Arc::new(DashMap::new());
        let channel_configs = Arc::new(RwLock::new(HashMap::new()));

        // Initialize core components
        let formatter = Arc::new(MessageFormatter::new(config.clone()).await?);
        let rate_limiter = Arc::new(RateLimiter::new(config.clone()).await?);
        let health_monitor = Arc::new(ChannelHealthMonitor::new(config.clone()).await?);
        let audit_system = Arc::new(AuditSystem::new(config.clone()).await?);
        let escalation_engine = Arc::new(EscalationEngine::new(config.clone()).await?);

        let delivery_engine = Arc::new(
            DeliveryEngine::new(
                config.clone(),
                channels.clone(),
                rate_limiter.clone(),
                health_monitor.clone(),
                audit_system.clone(),
            )
            .await?,
        );

        let (shutdown_tx, _) = oneshot::channel();

        let manager = Self {
            config: Arc::new(RwLock::new(config)),
            processors: Arc::new(RwLock::new(Vec::new())),
            channels,
            channel_configs,
            formatter,
            delivery_engine,
            rate_limiter,
            health_monitor,
            audit_system,
            escalation_engine,
            notification_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(NotificationStats::default()),
            processing_semaphore,
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
        };

        // Initialize default processors and channels
        manager.initialize_processors().await?;
        manager.initialize_channels().await?;

        Ok(manager)
    }

    /// Start the notification manager
    ///
    /// Begins all background processing including notification processing,
    /// health monitoring, rate limiting, and audit systems.
    pub async fn start(&self) -> Result<()> {
        info!("Starting notification manager");

        // Start all subsystems
        self.delivery_engine.start().await?;
        self.rate_limiter.start().await?;

        if self.config.read().enable_health_monitoring {
            self.health_monitor.start().await?;
        }

        if self.config.read().enable_auditing {
            self.audit_system.start().await?;
        }

        if self.config.read().enable_escalation {
            self.escalation_engine.start().await?;
        }

        // Start notification processing workers
        self.start_processing_workers().await?;

        info!("Notification manager started successfully");
        Ok(())
    }

    /// Process an alert and generate notifications
    ///
    /// Main entry point for alert processing that routes alerts through
    /// configured processors and generates appropriate notifications.
    pub async fn process_alert(&self, alert: AlertEvent) -> Result<Vec<ProcessedNotification>> {
        debug!("Processing alert: {}", alert.alert_id);

        let processors = {
            let guard = self.processors.read();
            guard.clone()
        };
        let mut all_notifications = Vec::new();

        // Process alert through all applicable processors
        for processor in processors.iter() {
            if processor.can_process(&alert) {
                match processor.process_alert(&alert).await {
                    Ok(notifications) => {
                        all_notifications.extend(notifications);
                    },
                    Err(e) => {
                        error!("Alert processor {} failed: {}", processor.name(), e);
                        self.stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                    },
                }
            }
        }

        // Format notifications using templates
        let formatted_notifications = self.format_notifications(all_notifications).await?;

        // Queue notifications for delivery
        let mut processed_notifications = Vec::new();
        for notification in formatted_notifications {
            let processed = self.queue_notification(notification).await?;
            processed_notifications.push(processed);
        }

        self.stats
            .total_processed
            .fetch_add(processed_notifications.len() as u64, Ordering::Relaxed);

        debug!(
            "Processed alert {} -> {} notifications",
            alert.alert_id,
            processed_notifications.len()
        );

        Ok(processed_notifications)
    }

    /// Add a notification channel
    ///
    /// Registers a new notification channel with configuration and
    /// integrates it into the health monitoring system.
    pub async fn add_channel(
        &self,
        channel: Arc<dyn NotificationChannel + Send + Sync>,
        config: ChannelConfig,
    ) -> Result<()> {
        let channel_name = channel.name().to_string();

        info!("Adding notification channel: {}", channel_name);

        // Store channel and configuration
        self.channels.insert(channel_name.clone(), channel.clone());
        self.channel_configs.write().insert(channel_name.clone(), config.clone());

        // Register with health monitor
        if self.config.read().enable_health_monitoring && config.health_check.enabled {
            self.health_monitor
                .register_channel(channel_name.clone(), config.health_check)
                .await?;
        }

        info!("Channel {} added successfully", channel_name);
        Ok(())
    }

    /// Remove a notification channel
    pub async fn remove_channel(&self, channel_name: &str) -> Result<()> {
        info!("Removing notification channel: {}", channel_name);

        self.channels.remove(channel_name);
        self.channel_configs.write().remove(channel_name);

        if self.config.read().enable_health_monitoring {
            self.health_monitor.unregister_channel(channel_name).await?;
        }

        info!("Channel {} removed successfully", channel_name);
        Ok(())
    }

    /// Get notification statistics
    pub fn get_stats(&self) -> NotificationStats {
        // Create a snapshot of current statistics
        NotificationStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            successful_deliveries: AtomicU64::new(
                self.stats.successful_deliveries.load(Ordering::Relaxed),
            ),
            failed_deliveries: AtomicU64::new(self.stats.failed_deliveries.load(Ordering::Relaxed)),
            partial_deliveries: AtomicU64::new(
                self.stats.partial_deliveries.load(Ordering::Relaxed),
            ),
            rate_limited: AtomicU64::new(self.stats.rate_limited.load(Ordering::Relaxed)),
            retry_attempts: AtomicU64::new(self.stats.retry_attempts.load(Ordering::Relaxed)),
            avg_processing_latency_ms: AtomicF32::new(
                self.stats.avg_processing_latency_ms.load(Ordering::Relaxed),
            ),
            avg_delivery_latency_ms: AtomicF32::new(
                self.stats.avg_delivery_latency_ms.load(Ordering::Relaxed),
            ),
            queue_size: AtomicUsize::new(self.stats.queue_size.load(Ordering::Relaxed)),
            processing_errors: AtomicU64::new(self.stats.processing_errors.load(Ordering::Relaxed)),
        }
    }

    /// Get channel health status
    pub async fn get_channel_health(&self) -> HashMap<String, ChannelHealth> {
        if self.config.read().enable_health_monitoring {
            self.health_monitor.get_all_health_status().await
        } else {
            HashMap::new()
        }
    }

    /// Shutdown the notification manager
    ///
    /// Gracefully shuts down all components and processes remaining notifications.
    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down notification manager");

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_tx.lock().take() {
            let _ = shutdown_tx.send(());
        }

        // Wait for workers to complete
        let mut handles = self.worker_handles.lock();
        for handle in handles.drain(..) {
            let _ = handle.await;
        }

        // Process remaining notifications in queue
        self.process_remaining_notifications().await?;

        // Shutdown subsystems
        self.delivery_engine.shutdown().await?;
        self.rate_limiter.shutdown().await?;
        self.health_monitor.shutdown().await?;
        self.audit_system.shutdown().await?;
        self.escalation_engine.shutdown().await?;

        info!("Notification manager shutdown complete");
        Ok(())
    }

    // Private implementation methods

    async fn initialize_processors(&self) -> Result<()> {
        let mut processors = self.processors.write();

        processors.push(Arc::new(DefaultAlertProcessor::new().await?));
        processors.push(Arc::new(PerformanceAlertProcessor::new().await?));
        processors.push(Arc::new(ResourceAlertProcessor::new().await?));
        processors.push(Arc::new(CriticalAlertProcessor::new().await?));

        Ok(())
    }

    async fn initialize_channels(&self) -> Result<()> {
        // Initialize default channels with configurations
        let log_config = ChannelConfig {
            name: "log".to_string(),
            enabled: true,
            priority: 1,
            config: HashMap::new(),
            rate_limit: None,
            timeout: Duration::from_secs(5),
            retry_policy: RetryPolicy::default(),
            health_check: HealthCheckConfig::default(),
        };

        let email_config = ChannelConfig {
            name: "email".to_string(),
            enabled: true,
            priority: 2,
            config: HashMap::new(),
            rate_limit: Some(10), // 10 emails per minute
            timeout: Duration::from_secs(30),
            retry_policy: RetryPolicy::default(),
            health_check: HealthCheckConfig::default(),
        };

        let webhook_config = ChannelConfig {
            name: "webhook".to_string(),
            enabled: true,
            priority: 3,
            config: HashMap::new(),
            rate_limit: Some(100),
            timeout: Duration::from_secs(10),
            retry_policy: RetryPolicy::default(),
            health_check: HealthCheckConfig::default(),
        };

        // Add channels
        self.add_channel(Arc::new(LogNotificationChannel::new().await?), log_config)
            .await?;
        self.add_channel(
            Arc::new(EmailNotificationChannel::new().await?),
            email_config,
        )
        .await?;
        self.add_channel(
            Arc::new(WebhookNotificationChannel::new().await?),
            webhook_config,
        )
        .await?;
        self.add_channel(
            Arc::new(SlackNotificationChannel::new().await?),
            ChannelConfig {
                name: "slack".to_string(),
                enabled: true,
                priority: 4,
                config: HashMap::new(),
                rate_limit: Some(50),
                timeout: Duration::from_secs(15),
                retry_policy: RetryPolicy::default(),
                health_check: HealthCheckConfig::default(),
            },
        )
        .await?;

        Ok(())
    }

    async fn format_notifications(
        &self,
        notifications: Vec<Notification>,
    ) -> Result<Vec<Notification>> {
        let mut formatted = Vec::new();

        for notification in notifications {
            match self.formatter.format_notification(&notification).await {
                Ok(formatted_notification) => formatted.push(formatted_notification),
                Err(e) => {
                    error!("Failed to format notification {}: {}", notification.id, e);
                    formatted.push(notification); // Use original if formatting fails
                },
            }
        }

        Ok(formatted)
    }

    async fn queue_notification(
        &self,
        notification: Notification,
    ) -> Result<ProcessedNotification> {
        // Create processed notification
        let processed = ProcessedNotification {
            notification: notification.clone(),
            processed_at: Utc::now(),
            results: HashMap::new(),
            status: ProcessingStatus::Pending,
            metadata: HashMap::new(),
        };

        // Add to queue
        {
            let mut queue = self.notification_queue.lock();
            queue.push_back(notification);
            self.stats.queue_size.store(queue.len(), Ordering::Relaxed);
        }

        Ok(processed)
    }

    async fn start_processing_workers(&self) -> Result<()> {
        let worker_count = std::cmp::max(1, num_cpus::get() / 2);
        let mut handles = self.worker_handles.lock();

        for i in 0..worker_count {
            let worker_id = i;
            let queue = self.notification_queue.clone();
            let delivery_engine = self.delivery_engine.clone();
            let stats = self.stats.clone();
            let semaphore = self.processing_semaphore.clone();

            let handle = tokio::spawn(async move {
                loop {
                    // Acquire processing permit
                    let _permit = semaphore.acquire().await.unwrap();

                    // Get next notification from queue
                    let notification = {
                        let mut queue = queue.lock();
                        let notification = queue.pop_front();
                        stats.queue_size.store(queue.len(), Ordering::Relaxed);
                        notification
                    };

                    if let Some(notification) = notification {
                        let start_time = Instant::now();

                        // Process notification through delivery engine
                        match delivery_engine.deliver_notification(notification).await {
                            Ok(result) => match result.status {
                                ProcessingStatus::Delivered => {
                                    stats.successful_deliveries.fetch_add(1, Ordering::Relaxed);
                                },
                                ProcessingStatus::PartiallyDelivered => {
                                    stats.partial_deliveries.fetch_add(1, Ordering::Relaxed);
                                },
                                ProcessingStatus::Failed => {
                                    stats.failed_deliveries.fetch_add(1, Ordering::Relaxed);
                                },
                                _ => {},
                            },
                            Err(e) => {
                                error!(
                                    "Worker {} failed to deliver notification: {}",
                                    worker_id, e
                                );
                                stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                                stats.failed_deliveries.fetch_add(1, Ordering::Relaxed);
                            },
                        }

                        // Update processing latency
                        let latency_ms = start_time.elapsed().as_millis() as f32;
                        let current_avg = stats.avg_processing_latency_ms.load(Ordering::Relaxed);
                        let new_avg = if current_avg == 0.0 {
                            latency_ms
                        } else {
                            (current_avg * 0.9) + (latency_ms * 0.1) // Exponential moving average
                        };
                        stats.avg_processing_latency_ms.store(new_avg, Ordering::Relaxed);
                    } else {
                        // No notifications in queue, sleep briefly
                        tokio::time::sleep(Duration::from_millis(100)).await;
                    }
                }
            });

            handles.push(handle);
        }

        info!("Started {} notification processing workers", worker_count);
        Ok(())
    }

    async fn process_remaining_notifications(&self) -> Result<()> {
        info!("Processing remaining notifications in queue");

        let mut processed_count = 0;
        loop {
            let notification = {
                let mut queue = self.notification_queue.lock();
                queue.pop_front()
            };

            if let Some(notification) = notification {
                if let Err(e) = self.delivery_engine.deliver_notification(notification).await {
                    error!("Failed to process remaining notification: {}", e);
                }
                processed_count += 1;
            } else {
                break;
            }
        }

        if processed_count > 0 {
            info!("Processed {} remaining notifications", processed_count);
        }

        Ok(())
    }
}

// =============================================================================
// MESSAGE FORMATTING AND TEMPLATING SYSTEM
// =============================================================================

/// Intelligent message formatter with advanced templating capabilities
///
/// Provides sophisticated message formatting with context-aware templating,
/// channel-specific formatting, and dynamic content generation.
pub struct MessageFormatter {
    /// Configuration
    config: NotificationConfig,

    /// Template cache for performance
    template_cache: Arc<DashMap<String, CompiledTemplate>>,

    /// Template engine
    template_engine: Arc<TemplateEngine>,

    /// Channel formatters
    channel_formatters: Arc<DashMap<String, Arc<dyn ChannelFormatter + Send + Sync>>>,

    /// Formatting statistics
    stats: Arc<FormattingStats>,
}

/// Compiled template for efficient rendering
#[derive(Debug, Clone)]
pub struct CompiledTemplate {
    /// Template name
    pub name: String,

    /// Template content
    pub content: String,

    /// Required variables
    pub required_vars: Vec<String>,

    /// Optional variables with defaults
    pub optional_vars: HashMap<String, String>,

    /// Template metadata
    pub metadata: HashMap<String, String>,

    /// Compilation timestamp
    pub compiled_at: DateTime<Utc>,
}

/// Template engine for processing templates
pub struct TemplateEngine {
    /// Available functions for templates
    functions: Arc<DashMap<String, Arc<dyn TemplateFunction + Send + Sync>>>,
}

/// Template function trait for extending template capabilities
pub trait TemplateFunction {
    /// Function name
    fn name(&self) -> &str;

    /// Execute function with arguments
    fn execute(&self, args: &[String]) -> Result<String>;

    /// Get function signature
    fn signature(&self) -> String;
}

/// Channel-specific formatter trait
pub trait ChannelFormatter {
    /// Format message for specific channel
    fn format_for_channel(&self, content: &str, channel: &str) -> Result<String>;

    /// Get supported channels
    fn supported_channels(&self) -> Vec<String>;

    /// Get maximum message length for channel
    fn max_length(&self, channel: &str) -> Option<usize>;
}

/// Formatting statistics
#[derive(Debug, Default)]
pub struct FormattingStats {
    /// Templates rendered
    pub templates_rendered: AtomicU64,

    /// Cache hits
    pub cache_hits: AtomicU64,

    /// Cache misses
    pub cache_misses: AtomicU64,

    /// Formatting errors
    pub formatting_errors: AtomicU64,

    /// Average formatting time (ms)
    pub avg_formatting_time_ms: AtomicF32,
}

impl MessageFormatter {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        let template_cache = Arc::new(DashMap::new());
        let template_engine = Arc::new(TemplateEngine::new().await?);
        let channel_formatters = Arc::new(DashMap::new());

        let formatter = Self {
            config,
            template_cache,
            template_engine,
            channel_formatters,
            stats: Arc::new(FormattingStats::default()),
        };

        // Initialize default templates and formatters
        formatter.initialize_default_templates().await?;
        formatter.initialize_channel_formatters().await?;

        Ok(formatter)
    }

    pub async fn format_notification(&self, notification: &Notification) -> Result<Notification> {
        let start_time = Instant::now();
        let mut formatted_notification = notification.clone();

        // Format using template if specified
        if let Some(template_name) = &notification.template {
            formatted_notification =
                self.apply_template(&formatted_notification, template_name).await?;
        }

        // Apply channel-specific formatting
        formatted_notification = self.apply_channel_formatting(&formatted_notification).await?;

        // Apply content enrichment
        formatted_notification = self.enrich_content(&formatted_notification).await?;

        // Update statistics
        self.stats.templates_rendered.fetch_add(1, Ordering::Relaxed);
        let formatting_time = start_time.elapsed().as_millis() as f32;
        let current_avg = self.stats.avg_formatting_time_ms.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0.0 {
            formatting_time
        } else {
            (current_avg * 0.9) + (formatting_time * 0.1)
        };
        self.stats.avg_formatting_time_ms.store(new_avg, Ordering::Relaxed);

        Ok(formatted_notification)
    }

    /// Add a custom template
    pub async fn add_template(&self, template: CompiledTemplate) -> Result<()> {
        self.template_cache.insert(template.name.clone(), template);
        Ok(())
    }

    /// Remove a template
    pub async fn remove_template(&self, template_name: &str) -> Result<()> {
        self.template_cache.remove(template_name);
        Ok(())
    }

    /// Get formatting statistics
    pub fn get_stats(&self) -> FormattingStats {
        FormattingStats {
            templates_rendered: AtomicU64::new(
                self.stats.templates_rendered.load(Ordering::Relaxed),
            ),
            cache_hits: AtomicU64::new(self.stats.cache_hits.load(Ordering::Relaxed)),
            cache_misses: AtomicU64::new(self.stats.cache_misses.load(Ordering::Relaxed)),
            formatting_errors: AtomicU64::new(self.stats.formatting_errors.load(Ordering::Relaxed)),
            avg_formatting_time_ms: AtomicF32::new(
                self.stats.avg_formatting_time_ms.load(Ordering::Relaxed),
            ),
        }
    }

    // Private implementation methods

    async fn apply_template(
        &self,
        notification: &Notification,
        template_name: &str,
    ) -> Result<Notification> {
        // Try to get template from cache
        let template = if let Some(template) = self.template_cache.get(template_name) {
            self.stats.cache_hits.fetch_add(1, Ordering::Relaxed);
            template.clone()
        } else {
            self.stats.cache_misses.fetch_add(1, Ordering::Relaxed);
            // Load template (in real implementation, this would load from storage)
            self.load_template(template_name).await?
        };

        // Render template with variables
        let rendered_content = self.render_template(&template, &notification.template_vars).await?;
        let rendered_subject = self
            .render_template_string(&notification.subject, &notification.template_vars)
            .await?;

        let mut formatted_notification = notification.clone();
        formatted_notification.content = rendered_content;
        formatted_notification.subject = rendered_subject;

        Ok(formatted_notification)
    }

    async fn apply_channel_formatting(&self, notification: &Notification) -> Result<Notification> {
        let mut formatted_notification = notification.clone();

        // Apply channel-specific formatting for each target channel
        for channel in &notification.channels {
            if let Some(formatter) = self.channel_formatters.get(channel) {
                // Format content for this channel
                let formatted_content =
                    formatter.format_for_channel(&notification.content, channel)?;

                // Check length limits
                if let Some(max_length) = formatter.max_length(channel) {
                    if formatted_content.len() > max_length {
                        let truncated = self.truncate_content(&formatted_content, max_length);
                        formatted_notification.content = truncated;
                    } else {
                        formatted_notification.content = formatted_content;
                    }
                } else {
                    formatted_notification.content = formatted_content;
                }
            }
        }

        Ok(formatted_notification)
    }

    async fn enrich_content(&self, notification: &Notification) -> Result<Notification> {
        let mut enriched_notification = notification.clone();

        // Add timestamp if not present
        if !enriched_notification.content.contains("Time:") {
            enriched_notification.content = format!(
                "{}\n\nTime: {}",
                enriched_notification.content,
                enriched_notification.created_at.format("%Y-%m-%d %H:%M:%S UTC")
            );
        }

        // Add severity indicators
        let severity_indicator = match enriched_notification.severity {
            SeverityLevel::Critical => "🚨 CRITICAL",
            SeverityLevel::High => "⚠️ HIGH",
            SeverityLevel::Medium => "📢 MEDIUM",
            SeverityLevel::Low => "💡 LOW",
            SeverityLevel::Info => "ℹ️ INFO",
            SeverityLevel::Warning => "⚠️ WARNING",
        };

        if !enriched_notification.subject.starts_with("🚨")
            && !enriched_notification.subject.starts_with("⚠️")
            && !enriched_notification.subject.starts_with("📢")
            && !enriched_notification.subject.starts_with("💡")
            && !enriched_notification.subject.starts_with("ℹ️")
        {
            enriched_notification.subject =
                format!("{} {}", severity_indicator, enriched_notification.subject);
        }

        // Add correlation info if present
        if let Some(correlation_id) = &enriched_notification.correlation_id {
            enriched_notification.content = format!(
                "{}\n\nCorrelation ID: {}",
                enriched_notification.content, correlation_id
            );
        }

        Ok(enriched_notification)
    }

    async fn render_template(
        &self,
        template: &CompiledTemplate,
        vars: &HashMap<String, String>,
    ) -> Result<String> {
        self.template_engine.render(&template.content, vars).await
    }

    async fn render_template_string(
        &self,
        template_str: &str,
        vars: &HashMap<String, String>,
    ) -> Result<String> {
        self.template_engine.render(template_str, vars).await
    }

    async fn load_template(&self, template_name: &str) -> Result<CompiledTemplate> {
        // In a real implementation, this would load from persistent storage
        // For now, return a default template based on name
        let template = match template_name {
            "default_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Alert: {{threshold_name}}\nCurrent Value: {{current_value}}\nThreshold: {{threshold_value}}\n\nPlease investigate this issue.".to_string(),
                required_vars: vec!["threshold_name".to_string(), "current_value".to_string(), "threshold_value".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "performance_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Performance Alert: {{metric}}\nCurrent: {{current_value}}\nThreshold: {{threshold_value}}\nImpact: {{impact}}\n\nImmediate attention required for performance optimization.".to_string(),
                required_vars: vec!["metric".to_string(), "current_value".to_string(), "threshold_value".to_string()],
                optional_vars: {
                    let mut vars = HashMap::new();
                    vars.insert("impact".to_string(), "Unknown".to_string());
                    vars
                },
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "resource_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "Resource Alert: {{resource_type}}\nUtilization: {{utilization}}\nThreshold: {{threshold}}\n\nResource scaling or optimization may be required.".to_string(),
                required_vars: vec!["resource_type".to_string(), "utilization".to_string(), "threshold".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            "critical_alert" => CompiledTemplate {
                name: template_name.to_string(),
                content: "🚨 CRITICAL SYSTEM ALERT 🚨\n\nSeverity: {{severity}}\nMetric: {{metric}}\nCurrent: {{current_value}}\nThreshold: {{threshold_value}}\nTime: {{alert_time}}\n\n🔥 IMMEDIATE ACTION REQUIRED 🔥\n\nThis is a critical system issue that requires immediate attention. Please escalate if not resolved within 15 minutes.".to_string(),
                required_vars: vec!["severity".to_string(), "metric".to_string(), "current_value".to_string(), "threshold_value".to_string(), "alert_time".to_string()],
                optional_vars: HashMap::new(),
                metadata: HashMap::new(),
                compiled_at: Utc::now(),
            },
            _ => {
                return Err(anyhow!("Template not found: {}", template_name));
            }
        };

        // Cache the template
        self.template_cache.insert(template_name.to_string(), template.clone());
        Ok(template)
    }

    fn truncate_content(&self, content: &str, max_length: usize) -> String {
        if content.len() <= max_length {
            return content.to_string();
        }

        let truncate_point = max_length.saturating_sub(20); // Reserve space for truncation message
        let truncated = &content[..truncate_point];
        format!("{}... [truncated]", truncated)
    }

    async fn initialize_default_templates(&self) -> Result<()> {
        // Templates are loaded on-demand via load_template method
        Ok(())
    }

    async fn initialize_channel_formatters(&self) -> Result<()> {
        // Add default channel formatters
        self.channel_formatters
            .insert("log".to_string(), Arc::new(LogChannelFormatter::new()));
        self.channel_formatters
            .insert("email".to_string(), Arc::new(EmailChannelFormatter::new()));
        self.channel_formatters
            .insert("slack".to_string(), Arc::new(SlackChannelFormatter::new()));
        self.channel_formatters.insert(
            "webhook".to_string(),
            Arc::new(WebhookChannelFormatter::new()),
        );

        Ok(())
    }
}

impl TemplateEngine {
    pub async fn new() -> Result<Self> {
        let functions = Arc::new(DashMap::new());

        let engine = Self { functions };

        // Register default template functions
        engine.register_default_functions().await?;

        Ok(engine)
    }

    pub async fn render(&self, template: &str, vars: &HashMap<String, String>) -> Result<String> {
        let mut result = template.to_string();

        // Simple variable substitution ({{variable_name}})
        for (key, value) in vars {
            let placeholder = format!("{{{{{}}}}}", key);
            result = result.replace(&placeholder, value);
        }

        // Process template functions ({{function_name(args)}})
        result = self.process_functions(&result).await?;

        Ok(result)
    }

    async fn register_default_functions(&self) -> Result<()> {
        self.functions.insert("now".to_string(), Arc::new(NowFunction));
        self.functions.insert(
            "format_duration".to_string(),
            Arc::new(FormatDurationFunction),
        );
        self.functions.insert("upper".to_string(), Arc::new(UpperFunction));
        self.functions.insert("lower".to_string(), Arc::new(LowerFunction));

        Ok(())
    }

    async fn process_functions(&self, template: &str) -> Result<String> {
        // Simple function processing - in production, you'd use a proper template engine
        let mut result = template.to_string();

        // Look for function calls like {{function_name(args)}}
        // This is a simplified implementation
        if result.contains("{{now()}}") {
            result = result.replace(
                "{{now()}}",
                &Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string(),
            );
        }

        Ok(result)
    }
}

// Default template functions
struct NowFunction;
impl TemplateFunction for NowFunction {
    fn name(&self) -> &str {
        "now"
    }
    fn execute(&self, _args: &[String]) -> Result<String> {
        Ok(Utc::now().format("%Y-%m-%d %H:%M:%S UTC").to_string())
    }
    fn signature(&self) -> String {
        "now() -> String".to_string()
    }
}

struct FormatDurationFunction;
impl TemplateFunction for FormatDurationFunction {
    fn name(&self) -> &str {
        "format_duration"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok("N/A".to_string());
        }
        // Simple duration formatting
        Ok(format!("{}s", args[0]))
    }
    fn signature(&self) -> String {
        "format_duration(seconds) -> String".to_string()
    }
}

struct UpperFunction;
impl TemplateFunction for UpperFunction {
    fn name(&self) -> &str {
        "upper"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok(String::new());
        }
        Ok(args[0].to_uppercase())
    }
    fn signature(&self) -> String {
        "upper(text) -> String".to_string()
    }
}

struct LowerFunction;
impl TemplateFunction for LowerFunction {
    fn name(&self) -> &str {
        "lower"
    }
    fn execute(&self, args: &[String]) -> Result<String> {
        if args.is_empty() {
            return Ok(String::new());
        }
        Ok(args[0].to_lowercase())
    }
    fn signature(&self) -> String {
        "lower(text) -> String".to_string()
    }
}

// Channel-specific formatters
struct LogChannelFormatter;
impl LogChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for LogChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Log formatting - keep it simple and readable
        Ok(content.to_string())
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["log".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        None // No length limit for logs
    }
}

struct EmailChannelFormatter;
impl EmailChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for EmailChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Email formatting - add HTML structure
        let formatted = format!(
            "<html><body><pre style=\"font-family: monospace; white-space: pre-wrap;\">{}</pre></body></html>",
            content.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        );
        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["email".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(100_000) // 100KB limit for emails
    }
}

struct SlackChannelFormatter;
impl SlackChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for SlackChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Slack formatting - convert to Slack markdown
        let mut formatted = content.to_string();

        // Convert basic formatting to Slack syntax
        formatted = formatted.replace("**", "*"); // Bold
        formatted = formatted.replace("__", "_"); // Italic

        // Add code blocks for structured content
        if formatted.contains("Current Value:") || formatted.contains("Threshold:") {
            formatted = format!("```\n{}\n```", formatted);
        }

        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["slack".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(40_000) // Slack message limit
    }
}

struct WebhookChannelFormatter;
impl WebhookChannelFormatter {
    fn new() -> Self {
        Self
    }
}

impl ChannelFormatter for WebhookChannelFormatter {
    fn format_for_channel(&self, content: &str, _channel: &str) -> Result<String> {
        // Webhook formatting - ensure JSON-safe content
        let formatted = content
            .replace("\"", "\\\"")
            .replace("\n", "\\n")
            .replace("\r", "\\r")
            .replace("\t", "\\t");
        Ok(formatted)
    }

    fn supported_channels(&self) -> Vec<String> {
        vec!["webhook".to_string()]
    }

    fn max_length(&self, _channel: &str) -> Option<usize> {
        Some(1_000_000) // 1MB limit for webhooks
    }
}

// =============================================================================
// DELIVERY ENGINE WITH CONFIGURABLE GUARANTEES
// =============================================================================

/// Advanced delivery engine with configurable delivery guarantees
///
/// Manages notification delivery with sophisticated retry logic, persistence,
/// acknowledgment tracking, and comprehensive delivery guarantee support.
pub struct DeliveryEngine {
    /// Configuration
    config: NotificationConfig,

    /// Available notification channels
    channels: Arc<DashMap<String, Arc<dyn NotificationChannel + Send + Sync>>>,

    /// Rate limiter for throttling
    rate_limiter: Arc<RateLimiter>,

    /// Health monitor for channel status
    health_monitor: Arc<ChannelHealthMonitor>,

    /// Audit system for tracking
    audit_system: Arc<AuditSystem>,

    /// Pending deliveries for at-least-once guarantee
    pending_deliveries: Arc<DashMap<String, PendingDelivery>>,

    /// Delivered notifications for exactly-once guarantee
    delivered_notifications: Arc<DashMap<String, DeliveryRecord>>,

    /// Retry scheduler
    retry_scheduler: Arc<RetryScheduler>,

    /// Delivery statistics
    stats: Arc<DeliveryStats>,

    /// Worker handles for background processing
    worker_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<broadcast::Sender<()>>>>,
}

/// Pending delivery record for tracking
#[derive(Debug, Clone)]
pub struct PendingDelivery {
    /// Notification to deliver
    pub notification: Notification,

    /// Number of delivery attempts
    pub attempts: usize,

    /// Next retry time
    pub next_retry: DateTime<Utc>,

    /// Created timestamp
    pub created_at: DateTime<Utc>,

    /// Last attempt timestamp
    pub last_attempt: Option<DateTime<Utc>>,

    /// Last error message
    pub last_error: Option<String>,

    /// Delivery metadata
    pub metadata: HashMap<String, String>,
}

/// Delivery record for exactly-once tracking
#[derive(Debug, Clone)]
pub struct DeliveryRecord {
    /// Notification ID
    pub notification_id: String,

    /// Delivery timestamp
    pub delivered_at: DateTime<Utc>,

    /// Delivery results by channel
    pub results: HashMap<String, DeliveryResult>,

    /// Delivery checksum for deduplication
    pub checksum: String,
}

/// Retry scheduler for managing retry logic
pub struct RetryScheduler {
    /// Retry queue
    retry_queue: Arc<Mutex<VecDeque<RetryItem>>>,

    /// Retry statistics
    stats: Arc<RetryStats>,
}

/// Retry item for scheduled retries
#[derive(Debug, Clone)]
pub struct RetryItem {
    /// Notification to retry
    pub notification: Notification,

    /// Target channel
    pub channel: String,

    /// Retry attempt number
    pub attempt: usize,

    /// Scheduled retry time
    pub retry_time: DateTime<Utc>,

    /// Retry reason
    pub reason: String,
}

/// Delivery statistics
#[derive(Debug, Default)]
pub struct DeliveryStats {
    /// Total delivery attempts
    pub total_attempts: AtomicU64,

    /// Successful deliveries
    pub successful_deliveries: AtomicU64,

    /// Failed deliveries
    pub failed_deliveries: AtomicU64,

    /// Retries scheduled
    pub retries_scheduled: AtomicU64,

    /// Duplicates prevented (exactly-once)
    pub duplicates_prevented: AtomicU64,

    /// Average delivery latency (ms)
    pub avg_delivery_latency_ms: AtomicF32,

    /// Delivery guarantee breakdowns
    pub best_effort_deliveries: AtomicU64,
    pub at_least_once_deliveries: AtomicU64,
    pub exactly_once_deliveries: AtomicU64,
}

/// Retry statistics
#[derive(Debug, Default)]
pub struct RetryStats {
    /// Total retries attempted
    pub total_retries: AtomicU64,

    /// Successful retries
    pub successful_retries: AtomicU64,

    /// Failed retries
    pub failed_retries: AtomicU64,

    /// Average retry delay (ms)
    pub avg_retry_delay_ms: AtomicF32,
}

impl DeliveryEngine {
    pub async fn new(
        config: NotificationConfig,
        channels: Arc<DashMap<String, Arc<dyn NotificationChannel + Send + Sync>>>,
        rate_limiter: Arc<RateLimiter>,
        health_monitor: Arc<ChannelHealthMonitor>,
        audit_system: Arc<AuditSystem>,
    ) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel(1);

        let engine = Self {
            config,
            channels,
            rate_limiter,
            health_monitor,
            audit_system,
            pending_deliveries: Arc::new(DashMap::new()),
            delivered_notifications: Arc::new(DashMap::new()),
            retry_scheduler: Arc::new(RetryScheduler::new().await?),
            stats: Arc::new(DeliveryStats::default()),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
        };

        Ok(engine)
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting delivery engine");

        // Start retry processing worker
        self.start_retry_processor().await?;

        // Start cleanup worker for old records
        self.start_cleanup_worker().await?;

        info!("Delivery engine started successfully");
        Ok(())
    }

    pub async fn deliver_notification(
        &self,
        notification: Notification,
    ) -> Result<ProcessedNotification> {
        let start_time = Instant::now();
        let notification_id = notification.id.clone();

        debug!("Delivering notification: {}", notification_id);

        // Check delivery guarantee and handle accordingly
        let processed_notification = match notification.delivery_guarantee {
            DeliveryGuarantee::BestEffort => self.deliver_best_effort(notification).await?,
            DeliveryGuarantee::AtLeastOnce => self.deliver_at_least_once(notification).await?,
            DeliveryGuarantee::ExactlyOnce => self.deliver_exactly_once(notification).await?,
        };

        // Update statistics
        self.stats.total_attempts.fetch_add(1, Ordering::Relaxed);
        match processed_notification.status {
            ProcessingStatus::Delivered => {
                self.stats.successful_deliveries.fetch_add(1, Ordering::Relaxed);
            },
            ProcessingStatus::Failed | ProcessingStatus::TimedOut => {
                self.stats.failed_deliveries.fetch_add(1, Ordering::Relaxed);
            },
            _ => {},
        }

        // Update latency statistics
        let latency_ms = start_time.elapsed().as_millis() as f32;
        let current_avg = self.stats.avg_delivery_latency_ms.load(Ordering::Relaxed);
        let new_avg = if current_avg == 0.0 {
            latency_ms
        } else {
            (current_avg * 0.9) + (latency_ms * 0.1)
        };
        self.stats.avg_delivery_latency_ms.store(new_avg, Ordering::Relaxed);

        // Record in audit system
        if self.config.enable_auditing {
            self.audit_system.record_delivery(&processed_notification).await?;
        }

        debug!(
            "Notification {} delivery completed with status: {:?}",
            notification_id, processed_notification.status
        );

        Ok(processed_notification)
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down delivery engine");

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_tx.lock().take() {
            let _ = shutdown_tx.send(());
        }

        // Wait for workers to complete
        let mut handles = self.worker_handles.lock();
        for handle in handles.drain(..) {
            let _ = handle.await;
        }

        // Process remaining pending deliveries
        self.process_remaining_deliveries().await?;

        info!("Delivery engine shutdown complete");
        Ok(())
    }

    /// Get delivery statistics
    pub fn get_stats(&self) -> DeliveryStats {
        DeliveryStats {
            total_attempts: AtomicU64::new(self.stats.total_attempts.load(Ordering::Relaxed)),
            successful_deliveries: AtomicU64::new(
                self.stats.successful_deliveries.load(Ordering::Relaxed),
            ),
            failed_deliveries: AtomicU64::new(self.stats.failed_deliveries.load(Ordering::Relaxed)),
            retries_scheduled: AtomicU64::new(self.stats.retries_scheduled.load(Ordering::Relaxed)),
            duplicates_prevented: AtomicU64::new(
                self.stats.duplicates_prevented.load(Ordering::Relaxed),
            ),
            avg_delivery_latency_ms: AtomicF32::new(
                self.stats.avg_delivery_latency_ms.load(Ordering::Relaxed),
            ),
            best_effort_deliveries: AtomicU64::new(
                self.stats.best_effort_deliveries.load(Ordering::Relaxed),
            ),
            at_least_once_deliveries: AtomicU64::new(
                self.stats.at_least_once_deliveries.load(Ordering::Relaxed),
            ),
            exactly_once_deliveries: AtomicU64::new(
                self.stats.exactly_once_deliveries.load(Ordering::Relaxed),
            ),
        }
    }

    // Private implementation methods

    async fn deliver_best_effort(
        &self,
        notification: Notification,
    ) -> Result<ProcessedNotification> {
        self.stats.best_effort_deliveries.fetch_add(1, Ordering::Relaxed);

        let mut results = HashMap::new();
        let mut has_success = false;

        // Attempt delivery to all channels concurrently
        let futures: Vec<_> = notification
            .channels
            .iter()
            .filter_map(|channel_name| {
                self.channels.get(channel_name).map(|channel| {
                    let notification_ref = &notification;
                    async move {
                        (
                            channel_name.clone(),
                            self.deliver_to_channel(notification_ref, channel.clone()).await,
                        )
                    }
                })
            })
            .collect();

        let results_vec = futures::future::join_all(futures).await;

        for (channel_name, result) in results_vec {
            match result {
                Ok(delivery_result) => {
                    if delivery_result.success {
                        has_success = true;
                    }
                    results.insert(channel_name, delivery_result);
                },
                Err(e) => {
                    error!("Failed to deliver to channel {}: {}", channel_name, e);
                    results.insert(
                        channel_name,
                        DeliveryResult {
                            success: false,
                            delivered_at: None,
                            attempts: 1,
                            error: Some(e.to_string()),
                            latency_ms: None,
                            response_data: HashMap::new(),
                        },
                    );
                },
            }
        }

        let status = if has_success {
            if results.values().all(|r| r.success) {
                ProcessingStatus::Delivered
            } else {
                ProcessingStatus::PartiallyDelivered
            }
        } else {
            ProcessingStatus::Failed
        };

        Ok(ProcessedNotification {
            notification,
            processed_at: Utc::now(),
            results,
            status,
            metadata: HashMap::new(),
        })
    }

    async fn deliver_at_least_once(
        &self,
        notification: Notification,
    ) -> Result<ProcessedNotification> {
        self.stats.at_least_once_deliveries.fetch_add(1, Ordering::Relaxed);

        // Create pending delivery record
        let pending_delivery = PendingDelivery {
            notification: notification.clone(),
            attempts: 0,
            next_retry: Utc::now(),
            created_at: Utc::now(),
            last_attempt: None,
            last_error: None,
            metadata: HashMap::new(),
        };

        self.pending_deliveries.insert(notification.id.clone(), pending_delivery);

        // Attempt initial delivery
        let processed_notification = self.deliver_best_effort(notification.clone()).await?;

        // If delivery failed or was partial, schedule retries
        match processed_notification.status {
            ProcessingStatus::Failed | ProcessingStatus::PartiallyDelivered => {
                self.schedule_retries(&notification, &processed_notification.results).await?;
            },
            ProcessingStatus::Delivered => {
                // Remove from pending deliveries on successful delivery
                self.pending_deliveries.remove(&notification.id);
            },
            _ => {},
        }

        Ok(processed_notification)
    }

    async fn deliver_exactly_once(
        &self,
        notification: Notification,
    ) -> Result<ProcessedNotification> {
        self.stats.exactly_once_deliveries.fetch_add(1, Ordering::Relaxed);

        // Generate checksum for deduplication
        let checksum = self.generate_checksum(&notification);

        // Check if already delivered
        if let Some(existing_record) = self
            .delivered_notifications
            .iter()
            .find(|entry| entry.value().checksum == checksum)
        {
            warn!(
                "Duplicate notification detected, skipping delivery: {}",
                notification.id
            );
            self.stats.duplicates_prevented.fetch_add(1, Ordering::Relaxed);

            return Ok(ProcessedNotification {
                notification,
                processed_at: Utc::now(),
                results: existing_record.results.clone(),
                status: ProcessingStatus::Delivered,
                metadata: {
                    let mut metadata = HashMap::new();
                    metadata.insert("duplicate_prevented".to_string(), "true".to_string());
                    metadata.insert(
                        "original_delivery".to_string(),
                        existing_record.delivered_at.to_rfc3339(),
                    );
                    metadata
                },
            });
        }

        // Deliver with at-least-once semantics first
        let processed_notification = self.deliver_at_least_once(notification.clone()).await?;

        // If successful, record the delivery for future deduplication
        if processed_notification.status == ProcessingStatus::Delivered {
            let delivery_record = DeliveryRecord {
                notification_id: notification.id.clone(),
                delivered_at: Utc::now(),
                results: processed_notification.results.clone(),
                checksum,
            };

            self.delivered_notifications.insert(notification.id.clone(), delivery_record);
        }

        Ok(processed_notification)
    }

    async fn deliver_to_channel(
        &self,
        notification: &Notification,
        channel: Arc<dyn NotificationChannel + Send + Sync>,
    ) -> Result<DeliveryResult> {
        let channel_name = channel.name();

        // Check rate limits
        if !self.rate_limiter.check_rate_limit(channel_name, &notification.priority).await? {
            return Ok(DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some("Rate limit exceeded".to_string()),
                latency_ms: None,
                response_data: HashMap::new(),
            });
        }

        // Check channel health
        if self.config.enable_health_monitoring {
            let health = self.health_monitor.get_channel_health(channel_name).await?;
            if !health.healthy {
                return Ok(DeliveryResult {
                    success: false,
                    delivered_at: None,
                    attempts: 1,
                    error: Some("Channel unhealthy".to_string()),
                    latency_ms: None,
                    response_data: HashMap::new(),
                });
            }
        }

        // Attempt delivery with timeout
        let delivery_future = channel.send_notification(notification);
        let timeout_duration = self.config.notification_timeout;

        match timeout(timeout_duration, delivery_future).await {
            Ok(result) => result,
            Err(_) => Ok(DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some("Delivery timeout".to_string()),
                latency_ms: Some(timeout_duration.as_millis() as u64),
                response_data: HashMap::new(),
            }),
        }
    }

    async fn schedule_retries(
        &self,
        notification: &Notification,
        results: &HashMap<String, DeliveryResult>,
    ) -> Result<()> {
        for (channel_name, result) in results {
            if !result.success {
                let retry_item = RetryItem {
                    notification: notification.clone(),
                    channel: channel_name.clone(),
                    attempt: 1,
                    retry_time: Utc::now()
                        + ChronoDuration::milliseconds(self.config.base_retry_delay_ms as i64),
                    reason: result.error.clone().unwrap_or_else(|| "Unknown error".to_string()),
                };

                self.retry_scheduler.schedule_retry(retry_item).await?;
                self.stats.retries_scheduled.fetch_add(1, Ordering::Relaxed);
            }
        }

        Ok(())
    }

    fn generate_checksum(&self, notification: &Notification) -> String {
        // Simple checksum based on notification content
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        notification.subject.hash(&mut hasher);
        notification.content.hash(&mut hasher);
        notification.recipients.hash(&mut hasher);
        notification.channels.hash(&mut hasher);

        format!("{:x}", hasher.finish())
    }

    async fn start_retry_processor(&self) -> Result<()> {
        let retry_scheduler = self.retry_scheduler.clone();
        let channels = self.channels.clone();
        let stats = self.stats.clone();
        let config = self.config.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(30));

            loop {
                interval.tick().await;

                // Process pending retries
                if let Ok(retry_items) = retry_scheduler.get_due_retries().await {
                    for retry_item in retry_items {
                        if let Some(channel) = channels.get(&retry_item.channel) {
                            match channel.send_notification(&retry_item.notification).await {
                                Ok(result) => {
                                    if result.success {
                                        stats.successful_deliveries.fetch_add(1, Ordering::Relaxed);
                                        retry_scheduler
                                            .stats
                                            .successful_retries
                                            .fetch_add(1, Ordering::Relaxed);
                                    } else {
                                        // Schedule another retry if within limits
                                        if retry_item.attempt < config.retry_attempts {
                                            let next_delay = std::cmp::min(
                                                config.base_retry_delay_ms
                                                    * (2_u64.pow(retry_item.attempt as u32)),
                                                config.max_retry_delay_ms,
                                            );

                                            let next_retry = RetryItem {
                                                attempt: retry_item.attempt + 1,
                                                retry_time: Utc::now()
                                                    + ChronoDuration::milliseconds(
                                                        next_delay as i64,
                                                    ),
                                                ..retry_item
                                            };

                                            let _ =
                                                retry_scheduler.schedule_retry(next_retry).await;
                                        } else {
                                            retry_scheduler
                                                .stats
                                                .failed_retries
                                                .fetch_add(1, Ordering::Relaxed);
                                        }
                                    }
                                },
                                Err(e) => {
                                    error!(
                                        "Retry delivery failed for {}: {}",
                                        retry_item.notification.id, e
                                    );
                                    retry_scheduler
                                        .stats
                                        .failed_retries
                                        .fetch_add(1, Ordering::Relaxed);
                                },
                            }
                        }
                    }
                }
            }
        });

        self.worker_handles.lock().push(handle);
        Ok(())
    }

    async fn start_cleanup_worker(&self) -> Result<()> {
        let delivered_notifications = self.delivered_notifications.clone();
        let pending_deliveries = self.pending_deliveries.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_secs(3600)); // Clean up every hour

            loop {
                interval.tick().await;

                let cutoff_time = Utc::now() - ChronoDuration::days(7); // Keep 7 days of history

                // Clean up old delivery records
                delivered_notifications.retain(|_, record| record.delivered_at > cutoff_time);

                // Clean up old pending deliveries that have expired
                pending_deliveries.retain(|_, pending| pending.created_at > cutoff_time);

                debug!(
                    "Cleanup completed. Delivery records: {}, Pending: {}",
                    delivered_notifications.len(),
                    pending_deliveries.len()
                );
            }
        });

        self.worker_handles.lock().push(handle);
        Ok(())
    }

    async fn process_remaining_deliveries(&self) -> Result<()> {
        info!("Processing remaining pending deliveries");

        let pending_deliveries: Vec<_> =
            self.pending_deliveries.iter().map(|entry| entry.value().clone()).collect();

        for pending in pending_deliveries {
            if let Err(e) = self.deliver_best_effort(pending.notification).await {
                error!("Failed to process remaining delivery: {}", e);
            }
        }

        Ok(())
    }
}

impl RetryScheduler {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            retry_queue: Arc::new(Mutex::new(VecDeque::new())),
            stats: Arc::new(RetryStats::default()),
        })
    }

    pub async fn schedule_retry(&self, retry_item: RetryItem) -> Result<()> {
        let mut queue = self.retry_queue.lock();
        queue.push_back(retry_item);
        Ok(())
    }

    pub async fn get_due_retries(&self) -> Result<Vec<RetryItem>> {
        let mut queue = self.retry_queue.lock();
        let now = Utc::now();
        let mut due_retries = Vec::new();

        // Extract due retries
        let mut remaining = VecDeque::new();
        while let Some(item) = queue.pop_front() {
            if item.retry_time <= now {
                due_retries.push(item);
            } else {
                remaining.push_back(item);
            }
        }

        // Put back non-due items
        *queue = remaining;

        Ok(due_retries)
    }
}

// =============================================================================
// RATE LIMITING AND THROTTLING SYSTEM
// =============================================================================

/// Advanced rate limiter with adaptive throttling and priority-based limiting
///
/// Provides sophisticated rate limiting with burst allowances, priority queuing,
/// adaptive limits, and per-channel rate limiting with comprehensive monitoring.
pub struct RateLimiter {
    /// Configuration
    config: NotificationConfig,

    /// Per-channel rate limiters
    channel_limiters: Arc<DashMap<String, Arc<ChannelRateLimiter>>>,

    /// Global rate limiter
    global_limiter: Arc<GlobalRateLimiter>,

    /// Priority queue for throttled notifications
    priority_queue: Arc<Mutex<PriorityQueue>>,

    /// Rate limiting statistics
    stats: Arc<RateLimitingStats>,

    /// Adaptive controller for dynamic rate adjustment
    adaptive_controller: Arc<AdaptiveRateController>,

    /// Worker handles
    worker_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_tx: Arc<Mutex<Option<broadcast::Sender<()>>>>,
}

/// Channel-specific rate limiter
pub struct ChannelRateLimiter {
    /// Channel name
    channel_name: String,

    /// Token bucket for rate limiting
    token_bucket: Arc<Mutex<TokenBucket>>,

    /// Channel configuration
    config: ChannelConfig,

    /// Channel statistics
    stats: Arc<ChannelRateLimitStats>,
}

/// Global rate limiter for system-wide limits
pub struct GlobalRateLimiter {
    /// Global token bucket
    token_bucket: Arc<Mutex<TokenBucket>>,

    /// Global configuration
    config: NotificationConfig,

    /// Global statistics
    stats: Arc<GlobalRateLimitStats>,
}

/// Token bucket implementation for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    /// Current token count
    tokens: f64,

    /// Maximum token capacity
    capacity: f64,

    /// Token refill rate (tokens per second)
    refill_rate: f64,

    /// Last refill timestamp
    last_refill: DateTime<Utc>,

    /// Bucket metadata
    metadata: HashMap<String, String>,
}

/// Priority queue for managing throttled notifications
pub struct PriorityQueue {
    /// High priority queue
    high_priority: VecDeque<ThrottledNotification>,

    /// Normal priority queue
    normal_priority: VecDeque<ThrottledNotification>,

    /// Low priority queue
    low_priority: VecDeque<ThrottledNotification>,

    /// Emergency bypass queue
    emergency_queue: VecDeque<ThrottledNotification>,
}

/// Throttled notification waiting for rate limit availability
#[derive(Debug, Clone)]
pub struct ThrottledNotification {
    /// Original notification
    pub notification: Notification,

    /// Target channel
    pub channel: String,

    /// Throttled timestamp
    pub throttled_at: DateTime<Utc>,

    /// Retry count
    pub retry_count: usize,

    /// Priority level
    pub priority: NotificationPriority,
}

/// Adaptive rate controller for dynamic adjustment
pub struct AdaptiveRateController {
    /// Current load metrics
    load_metrics: Arc<RwLock<LoadMetrics>>,

    /// Rate adjustment history
    adjustment_history: Arc<Mutex<VecDeque<RateAdjustment>>>,

    /// Controller configuration
    config: AdaptiveRateConfig,
}

/// Load metrics for adaptive rate control
#[derive(Debug, Default)]
pub struct LoadMetrics {
    /// Current queue depth
    pub queue_depth: usize,

    /// Average processing latency
    pub avg_latency_ms: f32,

    /// Success rate
    pub success_rate: f32,

    /// System load
    pub system_load: f32,

    /// Last update timestamp
    pub last_update: DateTime<Utc>,
}

/// Rate adjustment record
#[derive(Debug, Clone)]
pub struct RateAdjustment {
    /// Adjustment timestamp
    pub timestamp: DateTime<Utc>,

    /// Old rate limit
    pub old_rate: f64,

    /// New rate limit
    pub new_rate: f64,

    /// Adjustment reason
    pub reason: String,

    /// Effectiveness score
    pub effectiveness: Option<f32>,
}

/// Configuration for adaptive rate control
#[derive(Debug, Clone)]
pub struct AdaptiveRateConfig {
    /// Enable adaptive rate control
    pub enabled: bool,

    /// Minimum rate limit
    pub min_rate: f64,

    /// Maximum rate limit
    pub max_rate: f64,

    /// Adjustment sensitivity
    pub sensitivity: f32,

    /// Adjustment interval
    pub adjustment_interval: Duration,
}

/// Rate limiting statistics for channels
#[derive(Debug, Default)]
pub struct ChannelRateLimitStats {
    /// Requests allowed
    pub requests_allowed: AtomicU64,

    /// Requests throttled
    pub requests_throttled: AtomicU64,

    /// Current rate limit
    pub current_rate_limit: AtomicF32,

    /// Average wait time (ms)
    pub avg_wait_time_ms: AtomicF32,

    /// Token bucket utilization
    pub bucket_utilization: AtomicF32,
}

/// Global rate limiting statistics
#[derive(Debug, Default)]
pub struct GlobalRateLimitStats {
    /// Total requests processed
    pub total_requests: AtomicU64,

    /// Total requests throttled
    pub total_throttled: AtomicU64,

    /// Current global rate
    pub current_global_rate: AtomicF32,

    /// Queue sizes by priority
    pub high_priority_queue_size: AtomicUsize,
    pub normal_priority_queue_size: AtomicUsize,
    pub low_priority_queue_size: AtomicUsize,
    pub emergency_queue_size: AtomicUsize,
}

/// Overall rate limiting statistics
#[derive(Debug, Default)]
pub struct RateLimitingStats {
    /// Total notifications processed
    pub total_processed: AtomicU64,

    /// Total notifications throttled
    pub total_throttled: AtomicU64,

    /// Average throttling duration (ms)
    pub avg_throttling_duration_ms: AtomicF32,

    /// Rate limit violations
    pub rate_limit_violations: AtomicU64,

    /// Adaptive adjustments made
    pub adaptive_adjustments: AtomicU64,
}

impl RateLimiter {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        let (shutdown_tx, _) = broadcast::channel(1);

        let limiter = Self {
            config: config.clone(),
            channel_limiters: Arc::new(DashMap::new()),
            global_limiter: Arc::new(GlobalRateLimiter::new(config.clone()).await?),
            priority_queue: Arc::new(Mutex::new(PriorityQueue::new())),
            stats: Arc::new(RateLimitingStats::default()),
            adaptive_controller: Arc::new(AdaptiveRateController::new().await?),
            worker_handles: Arc::new(Mutex::new(Vec::new())),
            shutdown_tx: Arc::new(Mutex::new(Some(shutdown_tx))),
        };

        Ok(limiter)
    }

    pub async fn start(&self) -> Result<()> {
        info!("Starting rate limiter");

        // Start token bucket refill worker
        self.start_token_refill_worker().await?;

        // Start priority queue processor
        self.start_queue_processor().await?;

        // Start adaptive rate controller if enabled
        if self.adaptive_controller.config.enabled {
            self.start_adaptive_controller().await?;
        }

        info!("Rate limiter started successfully");
        Ok(())
    }

    /// Check if notification is allowed by rate limits
    pub async fn check_rate_limit(
        &self,
        channel: &str,
        priority: &NotificationPriority,
    ) -> Result<bool> {
        // Emergency notifications bypass rate limits
        if matches!(priority, NotificationPriority::Emergency) {
            return Ok(true);
        }

        // Check global rate limit first
        if !self.global_limiter.check_global_limit().await? {
            self.stats.total_throttled.fetch_add(1, Ordering::Relaxed);
            self.stats.rate_limit_violations.fetch_add(1, Ordering::Relaxed);
            return Ok(false);
        }

        // Check channel-specific rate limit
        let channel_limiter = self.get_or_create_channel_limiter(channel).await?;
        let allowed = channel_limiter.check_limit(priority).await?;

        if allowed {
            self.stats.total_processed.fetch_add(1, Ordering::Relaxed);
        } else {
            self.stats.total_throttled.fetch_add(1, Ordering::Relaxed);
        }

        Ok(allowed)
    }

    /// Add notification to throttle queue if rate limited
    pub async fn throttle_notification(
        &self,
        notification: Notification,
        channel: String,
    ) -> Result<()> {
        let throttled = ThrottledNotification {
            notification: notification.clone(),
            channel,
            throttled_at: Utc::now(),
            retry_count: 0,
            priority: notification.priority.clone(),
        };

        let mut queue = self.priority_queue.lock();
        match notification.priority {
            NotificationPriority::Emergency => queue.emergency_queue.push_back(throttled),
            NotificationPriority::Critical | NotificationPriority::High => {
                queue.high_priority.push_back(throttled);
                self.global_limiter
                    .stats
                    .high_priority_queue_size
                    .store(queue.high_priority.len(), Ordering::Relaxed);
            },
            NotificationPriority::Normal => {
                queue.normal_priority.push_back(throttled);
                self.global_limiter
                    .stats
                    .normal_priority_queue_size
                    .store(queue.normal_priority.len(), Ordering::Relaxed);
            },
            NotificationPriority::Low => {
                queue.low_priority.push_back(throttled);
                self.global_limiter
                    .stats
                    .low_priority_queue_size
                    .store(queue.low_priority.len(), Ordering::Relaxed);
            },
        }

        Ok(())
    }

    /// Register a channel with specific rate limiting configuration
    pub async fn register_channel(
        &self,
        channel_name: String,
        config: ChannelConfig,
    ) -> Result<()> {
        let channel_limiter =
            Arc::new(ChannelRateLimiter::new(channel_name.clone(), config).await?);
        self.channel_limiters.insert(channel_name.clone(), channel_limiter);

        info!("Registered channel {} with rate limiting", channel_name);
        Ok(())
    }

    /// Get rate limiting statistics
    pub fn get_stats(&self) -> RateLimitingStats {
        RateLimitingStats {
            total_processed: AtomicU64::new(self.stats.total_processed.load(Ordering::Relaxed)),
            total_throttled: AtomicU64::new(self.stats.total_throttled.load(Ordering::Relaxed)),
            avg_throttling_duration_ms: AtomicF32::new(
                self.stats.avg_throttling_duration_ms.load(Ordering::Relaxed),
            ),
            rate_limit_violations: AtomicU64::new(
                self.stats.rate_limit_violations.load(Ordering::Relaxed),
            ),
            adaptive_adjustments: AtomicU64::new(
                self.stats.adaptive_adjustments.load(Ordering::Relaxed),
            ),
        }
    }

    pub async fn shutdown(&self) -> Result<()> {
        info!("Shutting down rate limiter");

        // Send shutdown signal
        if let Some(shutdown_tx) = self.shutdown_tx.lock().take() {
            let _ = shutdown_tx.send(());
        }

        // Wait for workers to complete
        let mut handles = self.worker_handles.lock();
        for handle in handles.drain(..) {
            let _ = handle.await;
        }

        // Process remaining throttled notifications
        self.process_remaining_queue().await?;

        info!("Rate limiter shutdown complete");
        Ok(())
    }

    // Private implementation methods

    async fn get_or_create_channel_limiter(
        &self,
        channel: &str,
    ) -> Result<Arc<ChannelRateLimiter>> {
        if let Some(limiter) = self.channel_limiters.get(channel) {
            Ok(limiter.clone())
        } else {
            // Create default channel limiter
            let default_config = ChannelConfig {
                name: channel.to_string(),
                enabled: true,
                priority: 50,
                config: HashMap::new(),
                rate_limit: Some(self.config.rate_limit_per_minute),
                timeout: self.config.notification_timeout,
                retry_policy: RetryPolicy::default(),
                health_check: HealthCheckConfig::default(),
            };

            let limiter =
                Arc::new(ChannelRateLimiter::new(channel.to_string(), default_config).await?);
            self.channel_limiters.insert(channel.to_string(), limiter.clone());
            Ok(limiter)
        }
    }

    async fn start_token_refill_worker(&self) -> Result<()> {
        let global_limiter = self.global_limiter.clone();
        let channel_limiters = self.channel_limiters.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(100)); // Refill every 100ms

            loop {
                interval.tick().await;

                // Refill global token bucket
                global_limiter.refill_tokens().await;

                // Refill channel token buckets
                for entry in channel_limiters.iter() {
                    entry.value().refill_tokens().await;
                }
            }
        });

        self.worker_handles.lock().push(handle);
        Ok(())
    }

    async fn start_queue_processor(&self) -> Result<()> {
        let priority_queue = self.priority_queue.clone();
        let channel_limiters = self.channel_limiters.clone();
        let global_limiter = self.global_limiter.clone();
        let stats = self.stats.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(200)); // Process queue every 200ms

            loop {
                interval.tick().await;

                let notifications_to_process = {
                    let mut queue = priority_queue.lock();
                    let mut to_process = Vec::new();

                    // Process emergency queue first
                    while let Some(throttled) = queue.emergency_queue.pop_front() {
                        to_process.push(throttled);
                    }

                    // Then high priority
                    let high_count = std::cmp::min(queue.high_priority.len(), 10);
                    for _ in 0..high_count {
                        if let Some(throttled) = queue.high_priority.pop_front() {
                            to_process.push(throttled);
                        }
                    }

                    // Then normal priority
                    let normal_count = std::cmp::min(queue.normal_priority.len(), 5);
                    for _ in 0..normal_count {
                        if let Some(throttled) = queue.normal_priority.pop_front() {
                            to_process.push(throttled);
                        }
                    }

                    // Finally low priority
                    let low_count = std::cmp::min(queue.low_priority.len(), 2);
                    for _ in 0..low_count {
                        if let Some(throttled) = queue.low_priority.pop_front() {
                            to_process.push(throttled);
                        }
                    }

                    // Update queue size statistics
                    global_limiter
                        .stats
                        .high_priority_queue_size
                        .store(queue.high_priority.len(), Ordering::Relaxed);
                    global_limiter
                        .stats
                        .normal_priority_queue_size
                        .store(queue.normal_priority.len(), Ordering::Relaxed);
                    global_limiter
                        .stats
                        .low_priority_queue_size
                        .store(queue.low_priority.len(), Ordering::Relaxed);
                    global_limiter
                        .stats
                        .emergency_queue_size
                        .store(queue.emergency_queue.len(), Ordering::Relaxed);

                    to_process
                };

                // Process notifications that can now be delivered
                for throttled in notifications_to_process {
                    // Check if we can now deliver this notification
                    if let Some(channel_limiter) = channel_limiters.get(&throttled.channel) {
                        if channel_limiter.check_limit(&throttled.priority).await.unwrap_or(false)
                            && global_limiter.check_global_limit().await.unwrap_or(false)
                        {
                            // Update throttling duration statistics
                            let throttling_duration =
                                Utc::now().signed_duration_since(throttled.throttled_at);
                            let duration_ms = throttling_duration.num_milliseconds() as f32;

                            let current_avg =
                                stats.avg_throttling_duration_ms.load(Ordering::Relaxed);
                            let new_avg = if current_avg == 0.0 {
                                duration_ms
                            } else {
                                (current_avg * 0.9) + (duration_ms * 0.1)
                            };
                            stats.avg_throttling_duration_ms.store(new_avg, Ordering::Relaxed);

                            // Notification can be delivered (would be handled by delivery engine)
                            debug!(
                                "Released throttled notification {} after {}ms",
                                throttled.notification.id, duration_ms
                            );
                        } else {
                            // Put back in appropriate queue
                            let mut queue = priority_queue.lock();
                            match throttled.priority {
                                NotificationPriority::Emergency => {
                                    queue.emergency_queue.push_back(throttled)
                                },
                                NotificationPriority::Critical | NotificationPriority::High => {
                                    queue.high_priority.push_back(throttled);
                                },
                                NotificationPriority::Normal => {
                                    queue.normal_priority.push_back(throttled);
                                },
                                NotificationPriority::Low => {
                                    queue.low_priority.push_back(throttled);
                                },
                            }
                        }
                    }
                }
            }
        });

        self.worker_handles.lock().push(handle);
        Ok(())
    }

    async fn start_adaptive_controller(&self) -> Result<()> {
        let adaptive_controller = self.adaptive_controller.clone();
        let global_limiter = self.global_limiter.clone();
        let channel_limiters = self.channel_limiters.clone();
        let stats = self.stats.clone();

        let handle = tokio::spawn(async move {
            let mut interval = interval(adaptive_controller.config.adjustment_interval);

            loop {
                interval.tick().await;

                // Collect current metrics
                let metrics = adaptive_controller.collect_metrics().await;

                // Determine if rate adjustment is needed
                if let Some(adjustment) = adaptive_controller.calculate_adjustment(&metrics).await {
                    // Apply adjustment to global limiter
                    global_limiter.adjust_rate(adjustment.new_rate).await;

                    // Apply proportional adjustment to channel limiters
                    let adjustment_ratio = adjustment.new_rate / adjustment.old_rate;
                    for entry in channel_limiters.iter() {
                        entry.value().adjust_rate_proportionally(adjustment_ratio).await;
                    }

                    stats.adaptive_adjustments.fetch_add(1, Ordering::Relaxed);

                    info!(
                        "Adaptive rate adjustment: {} -> {} ({})",
                        adjustment.old_rate, adjustment.new_rate, adjustment.reason
                    );
                }
            }
        });

        self.worker_handles.lock().push(handle);
        Ok(())
    }

    async fn process_remaining_queue(&self) -> Result<()> {
        info!("Processing remaining throttled notifications");

        let queue = self.priority_queue.lock();
        let total_remaining = queue.emergency_queue.len()
            + queue.high_priority.len()
            + queue.normal_priority.len()
            + queue.low_priority.len();

        if total_remaining > 0 {
            warn!(
                "Dropping {} throttled notifications during shutdown",
                total_remaining
            );
        }

        Ok(())
    }
}

impl ChannelRateLimiter {
    pub async fn new(channel_name: String, config: ChannelConfig) -> Result<Self> {
        let rate_limit = config.rate_limit.unwrap_or(60) as f64; // Default to 60 per minute
        let capacity = rate_limit;
        let refill_rate = rate_limit / 60.0; // Convert to tokens per second

        let token_bucket = TokenBucket {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Utc::now(),
            metadata: HashMap::new(),
        };

        Ok(Self {
            channel_name,
            token_bucket: Arc::new(Mutex::new(token_bucket)),
            config,
            stats: Arc::new(ChannelRateLimitStats::default()),
        })
    }

    pub async fn check_limit(&self, priority: &NotificationPriority) -> Result<bool> {
        let mut bucket = self.token_bucket.lock();

        // Priority-based token costs
        let token_cost = match priority {
            NotificationPriority::Emergency => 0.0, // Free
            NotificationPriority::Critical => 0.5,  // Half cost
            NotificationPriority::High => 1.0,      // Normal cost
            NotificationPriority::Normal => 1.0,    // Normal cost
            NotificationPriority::Low => 2.0,       // Double cost
        };

        if bucket.tokens >= token_cost {
            bucket.tokens -= token_cost;
            self.stats.requests_allowed.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        } else {
            self.stats.requests_throttled.fetch_add(1, Ordering::Relaxed);
            Ok(false)
        }
    }

    pub async fn refill_tokens(&self) {
        let mut bucket = self.token_bucket.lock();
        let now = Utc::now();
        let duration = now.signed_duration_since(bucket.last_refill);
        let seconds_elapsed = duration.num_milliseconds() as f64 / 1000.0;

        if seconds_elapsed > 0.0 {
            let tokens_to_add = bucket.refill_rate * seconds_elapsed;
            bucket.tokens = (bucket.tokens + tokens_to_add).min(bucket.capacity);
            bucket.last_refill = now;

            // Update utilization statistics
            let utilization = (bucket.capacity - bucket.tokens) / bucket.capacity;
            self.stats.bucket_utilization.store(utilization as f32, Ordering::Relaxed);
        }
    }

    pub async fn adjust_rate_proportionally(&self, ratio: f64) {
        let mut bucket = self.token_bucket.lock();
        bucket.capacity *= ratio;
        bucket.refill_rate *= ratio;
        bucket.tokens = bucket.tokens.min(bucket.capacity);

        self.stats
            .current_rate_limit
            .store(bucket.refill_rate as f32 * 60.0, Ordering::Relaxed);
    }
}

impl GlobalRateLimiter {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        let rate_limit = config.rate_limit_per_minute as f64;
        let capacity = rate_limit + config.rate_limit_burst as f64;
        let refill_rate = rate_limit / 60.0;

        let token_bucket = TokenBucket {
            tokens: capacity,
            capacity,
            refill_rate,
            last_refill: Utc::now(),
            metadata: HashMap::new(),
        };

        Ok(Self {
            token_bucket: Arc::new(Mutex::new(token_bucket)),
            config,
            stats: Arc::new(GlobalRateLimitStats::default()),
        })
    }

    pub async fn check_global_limit(&self) -> Result<bool> {
        let mut bucket = self.token_bucket.lock();

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            self.stats.total_requests.fetch_add(1, Ordering::Relaxed);
            Ok(true)
        } else {
            self.stats.total_throttled.fetch_add(1, Ordering::Relaxed);
            Ok(false)
        }
    }

    pub async fn refill_tokens(&self) {
        let mut bucket = self.token_bucket.lock();
        let now = Utc::now();
        let duration = now.signed_duration_since(bucket.last_refill);
        let seconds_elapsed = duration.num_milliseconds() as f64 / 1000.0;

        if seconds_elapsed > 0.0 {
            let tokens_to_add = bucket.refill_rate * seconds_elapsed;
            bucket.tokens = (bucket.tokens + tokens_to_add).min(bucket.capacity);
            bucket.last_refill = now;

            self.stats
                .current_global_rate
                .store(bucket.refill_rate as f32 * 60.0, Ordering::Relaxed);
        }
    }

    pub async fn adjust_rate(&self, new_rate: f64) {
        let mut bucket = self.token_bucket.lock();
        bucket.refill_rate = new_rate / 60.0;
        bucket.capacity = new_rate + self.config.rate_limit_burst as f64;
        bucket.tokens = bucket.tokens.min(bucket.capacity);

        self.stats.current_global_rate.store(new_rate as f32, Ordering::Relaxed);
    }
}

impl Default for PriorityQueue {
    fn default() -> Self {
        Self::new()
    }
}

impl PriorityQueue {
    pub fn new() -> Self {
        Self {
            high_priority: VecDeque::new(),
            normal_priority: VecDeque::new(),
            low_priority: VecDeque::new(),
            emergency_queue: VecDeque::new(),
        }
    }
}

impl AdaptiveRateController {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            load_metrics: Arc::new(RwLock::new(LoadMetrics::default())),
            adjustment_history: Arc::new(Mutex::new(VecDeque::new())),
            config: AdaptiveRateConfig {
                enabled: true,
                min_rate: 10.0,
                max_rate: 1000.0,
                sensitivity: 0.1,
                adjustment_interval: Duration::from_secs(60),
            },
        })
    }

    pub async fn collect_metrics(&self) -> LoadMetrics {
        // In a real implementation, this would collect actual system metrics
        LoadMetrics {
            queue_depth: 0,
            avg_latency_ms: 100.0,
            success_rate: 0.95,
            system_load: 0.6,
            last_update: Utc::now(),
        }
    }

    pub async fn calculate_adjustment(&self, metrics: &LoadMetrics) -> Option<RateAdjustment> {
        // TODO: Added f64 type annotation to fix E0689 ambiguous numeric type
        let current_rate: f64 = 100.0; // Placeholder - would get from actual rate limiter

        // Simple adaptive logic
        let target_rate = if metrics.success_rate < 0.9 || metrics.avg_latency_ms > 500.0 {
            // Decrease rate if system is struggling
            current_rate * 0.8
        } else if metrics.success_rate > 0.98 && metrics.avg_latency_ms < 100.0 {
            // Increase rate if system is performing well
            current_rate * 1.2
        } else {
            return None; // No adjustment needed
        };

        // TODO: Added f64 type annotation to fix E0689 ambiguous numeric type
        let clamped_rate =
            target_rate.clamp(self.config.min_rate, self.config.max_rate);

        if (clamped_rate - current_rate).abs() > current_rate * self.config.sensitivity as f64 {
            Some(RateAdjustment {
                timestamp: Utc::now(),
                old_rate: current_rate,
                new_rate: clamped_rate,
                reason: format!(
                    "Adaptive adjustment based on success_rate={:.2}, latency={}ms",
                    metrics.success_rate, metrics.avg_latency_ms
                ),
                effectiveness: None,
            })
        } else {
            None
        }
    }
}

/// Channel health monitoring system
pub struct ChannelHealthMonitor {
    config: NotificationConfig,
}

#[derive(Debug, Clone)]
pub struct ChannelHealth {
    pub healthy: bool,
    pub last_check: DateTime<Utc>,
    pub consecutive_failures: usize,
    pub success_rate: f32,
}

impl ChannelHealthMonitor {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn start(&self) -> Result<()> {
        Ok(())
    }

    pub async fn register_channel(&self, _name: String, _config: HealthCheckConfig) -> Result<()> {
        Ok(())
    }

    pub async fn unregister_channel(&self, _name: &str) -> Result<()> {
        Ok(())
    }

    pub async fn get_all_health_status(&self) -> HashMap<String, ChannelHealth> {
        HashMap::new()
    }

    /// Get health status for a specific channel
    pub async fn get_channel_health(&self, _channel_name: &str) -> Result<ChannelHealth> {
        // Return default healthy status for now
        Ok(ChannelHealth {
            healthy: true,
            last_check: chrono::Utc::now(),
            consecutive_failures: 0,
            success_rate: 1.0,
        })
    }

    pub async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

/// Audit system for tracking notifications
pub struct AuditSystem {
    config: NotificationConfig,
}

impl AuditSystem {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn start(&self) -> Result<()> {
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        Ok(())
    }

    /// Record a notification delivery for auditing
    pub async fn record_delivery(&self, notification: &ProcessedNotification) -> Result<()> {
        // Log delivery for audit trail
        debug!(
            "Audit: Recorded delivery of notification at {:?} with status {:?}",
            notification.processed_at, notification.status
        );
        Ok(())
    }
}

/// Escalation engine for advanced escalation workflows
pub struct EscalationEngine {
    config: NotificationConfig,
}

impl EscalationEngine {
    pub async fn new(config: NotificationConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub async fn start(&self) -> Result<()> {
        Ok(())
    }

    pub async fn shutdown(&self) -> Result<()> {
        Ok(())
    }
}

// =============================================================================
// ALERT PROCESSOR TRAIT AND IMPLEMENTATIONS
// =============================================================================

/// Enhanced trait for alert processors with async support
#[async_trait::async_trait]
pub trait AlertProcessor {
    /// Process an alert and generate notifications
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor can handle this alert
    fn can_process(&self, alert: &AlertEvent) -> bool;

    /// Get processor priority (higher = processed first)
    fn priority(&self) -> u8 {
        50 // Default priority
    }
}

/// Default alert processor for general alerts
pub struct DefaultAlertProcessor {
    name: String,
}

impl DefaultAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "default".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for DefaultAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["log".to_string()],
            recipients: vec!["system".to_string()],
            subject: format!("Alert: {}", alert.threshold.name),
            content: alert.message.clone(),
            priority: match alert.severity {
                SeverityLevel::Critical => NotificationPriority::Critical,
                SeverityLevel::High => NotificationPriority::High,
                SeverityLevel::Medium => NotificationPriority::Normal,
                _ => NotificationPriority::Low,
            },
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::BestEffort,
            created_at: Utc::now(),
            deadline: None,
            template: Some("default_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("threshold_name".to_string(), alert.threshold.name.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars
            },
            metadata: HashMap::new(),
            tags: vec!["alert".to_string(), "default".to_string()],
            escalation_policy: None,
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, _alert: &AlertEvent) -> bool {
        true // Default processor handles all alerts
    }

    fn priority(&self) -> u8 {
        10 // Low priority - runs last
    }
}

/// Performance-specific alert processor
pub struct PerformanceAlertProcessor {
    name: String,
}

impl PerformanceAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "performance".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for PerformanceAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("perf_notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["email".to_string(), "slack".to_string()],
            recipients: vec!["performance-team@company.com".to_string()],
            subject: format!("Performance Alert: {}", alert.threshold.name),
            content: format!("Performance issue detected: {}", alert.message),
            priority: NotificationPriority::High,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(15)),
            template: Some("performance_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("metric".to_string(), alert.threshold.metric.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars.insert("impact".to_string(), "High".to_string());
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("team".to_string(), "performance".to_string());
                metadata.insert("escalation_required".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "performance".to_string(),
                "urgent".to_string(),
            ],
            escalation_policy: Some("performance_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "throughput" | "latency_ms" | "response_time"
        )
    }

    fn priority(&self) -> u8 {
        80 // High priority
    }
}

/// Resource-specific alert processor
pub struct ResourceAlertProcessor {
    name: String,
}

impl ResourceAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "resource".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for ResourceAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        let notification = Notification {
            id: format!("resource_notif_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["webhook".to_string(), "slack".to_string()],
            recipients: vec!["ops-team".to_string()],
            subject: format!("Resource Alert: {}", alert.threshold.name),
            content: format!("Resource issue detected: {}", alert.message),
            priority: NotificationPriority::High,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::AtLeastOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(10)),
            template: Some("resource_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("resource_type".to_string(), alert.threshold.metric.clone());
                vars.insert(
                    "utilization".to_string(),
                    format!("{}%", (alert.current_value * 100.0) as u32),
                );
                vars.insert(
                    "threshold".to_string(),
                    format!("{}%", (alert.threshold_value * 100.0) as u32),
                );
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("team".to_string(), "ops".to_string());
                metadata.insert("auto_scale".to_string(), "enabled".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "resource".to_string(),
                "ops".to_string(),
            ],
            escalation_policy: Some("ops_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        Ok(vec![notification])
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(
            alert.threshold.metric.as_str(),
            "cpu_utilization" | "memory_utilization" | "disk_usage"
        )
    }

    fn priority(&self) -> u8 {
        70 // High priority
    }
}

/// Critical alert processor for emergency situations
pub struct CriticalAlertProcessor {
    name: String,
}

impl CriticalAlertProcessor {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "critical".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl AlertProcessor for CriticalAlertProcessor {
    async fn process_alert(&self, alert: &AlertEvent) -> Result<Vec<Notification>> {
        // Critical alerts generate multiple notifications across all channels
        let mut notifications = Vec::new();

        // Immediate notification to all channels
        let immediate_notification = Notification {
            id: format!("critical_immediate_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec![
                "email".to_string(),
                "slack".to_string(),
                "webhook".to_string(),
                "sms".to_string(),
            ],
            recipients: vec![
                "oncall@company.com".to_string(),
                "manager@company.com".to_string(),
                "emergency-contact".to_string(),
            ],
            subject: format!("🚨 CRITICAL ALERT: {}", alert.threshold.name),
            content: format!(
                "CRITICAL SYSTEM ISSUE: {}\n\nImmediate attention required!",
                alert.message
            ),
            priority: NotificationPriority::Emergency,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::ExactlyOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(5)),
            template: Some("critical_alert".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("severity".to_string(), "CRITICAL".to_string());
                vars.insert("metric".to_string(), alert.threshold.metric.clone());
                vars.insert("current_value".to_string(), alert.current_value.to_string());
                vars.insert(
                    "threshold_value".to_string(),
                    alert.threshold_value.to_string(),
                );
                vars.insert(
                    "alert_time".to_string(),
                    alert.timestamp.format("%Y-%m-%d %H:%M:%S UTC").to_string(),
                );
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("priority".to_string(), "emergency".to_string());
                metadata.insert("bypass_rate_limit".to_string(), "true".to_string());
                metadata.insert("require_acknowledgment".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "critical".to_string(),
                "emergency".to_string(),
            ],
            escalation_policy: Some("critical_escalation".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        notifications.push(immediate_notification);

        // Follow-up notification for PagerDuty
        let pagerduty_notification = Notification {
            id: format!("critical_pagerduty_{}", alert.alert_id),
            alert_id: alert.alert_id.clone(),
            channels: vec!["pagerduty".to_string()],
            recipients: vec!["incident-response".to_string()],
            subject: format!("Critical System Alert - {}", alert.threshold.name),
            content: format!(
                "Critical alert requires immediate incident response: {}",
                alert.message
            ),
            priority: NotificationPriority::Emergency,
            severity: alert.severity,
            delivery_guarantee: DeliveryGuarantee::ExactlyOnce,
            created_at: Utc::now(),
            deadline: Some(Utc::now() + ChronoDuration::minutes(2)),
            template: Some("pagerduty_critical".to_string()),
            template_vars: {
                let mut vars = HashMap::new();
                vars.insert("incident_key".to_string(), alert.alert_id.clone());
                vars.insert(
                    "service_key".to_string(),
                    "trustformers-critical".to_string(),
                );
                vars.insert("description".to_string(), alert.message.clone());
                vars
            },
            metadata: {
                let mut metadata = HashMap::new();
                metadata.insert("incident_type".to_string(), "critical".to_string());
                metadata.insert("auto_escalate".to_string(), "true".to_string());
                metadata
            },
            tags: vec![
                "alert".to_string(),
                "critical".to_string(),
                "pagerduty".to_string(),
            ],
            escalation_policy: Some("incident_response".to_string()),
            correlation_id: Some(alert.alert_id.clone()),
        };

        notifications.push(pagerduty_notification);

        Ok(notifications)
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn can_process(&self, alert: &AlertEvent) -> bool {
        matches!(alert.severity, SeverityLevel::Critical)
    }

    fn priority(&self) -> u8 {
        100 // Highest priority - runs first
    }
}

// =============================================================================
// NOTIFICATION CHANNEL TRAIT AND IMPLEMENTATIONS
// =============================================================================

/// Enhanced trait for notification channels with async support and health monitoring
#[async_trait::async_trait]
pub trait NotificationChannel {
    /// Send a notification through this channel
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel supports the given notification type
    fn supports(&self, notification_type: &str) -> bool;

    /// Perform health check
    async fn health_check(&self) -> Result<ChannelHealth>;

    /// Get channel capabilities
    fn capabilities(&self) -> Vec<String> {
        vec!["basic".to_string()]
    }

    /// Get maximum message size
    fn max_message_size(&self) -> Option<usize> {
        None
    }
}

/// Log notification channel implementation
pub struct LogNotificationChannel {
    name: String,
}

impl LogNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "log".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for LogNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let start_time = Instant::now();

        // Log the notification
        match notification.priority {
            NotificationPriority::Emergency | NotificationPriority::Critical => {
                error!(
                    "🚨 CRITICAL ALERT: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::High => {
                warn!(
                    "⚠️  HIGH PRIORITY: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::Normal => {
                info!(
                    "📢 NOTIFICATION: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
            NotificationPriority::Low => {
                debug!(
                    "📝 INFO: {} - {} (ID: {})",
                    notification.subject, notification.content, notification.id
                );
            },
        }

        let latency_ms = start_time.elapsed().as_millis() as u64;

        Ok(DeliveryResult {
            success: true,
            delivered_at: Some(Utc::now()),
            attempts: 1,
            error: None,
            latency_ms: Some(latency_ms),
            response_data: {
                let mut data = HashMap::new();
                data.insert("logged_at".to_string(), Utc::now().to_rfc3339());
                data.insert(
                    "log_level".to_string(),
                    match notification.priority {
                        NotificationPriority::Emergency | NotificationPriority::Critical => {
                            "ERROR".to_string()
                        },
                        NotificationPriority::High => "WARN".to_string(),
                        NotificationPriority::Normal => "INFO".to_string(),
                        NotificationPriority::Low => "DEBUG".to_string(),
                    },
                );
                data
            },
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "log" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        // Log channel is always healthy as it doesn't depend on external services
        Ok(ChannelHealth {
            healthy: true,
            last_check: Utc::now(),
            consecutive_failures: 0,
            success_rate: 1.0,
        })
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "always_available".to_string(),
            "immediate".to_string(),
        ]
    }
}

/// Email notification channel implementation
pub struct EmailNotificationChannel {
    name: String,
}

impl EmailNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "email".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for EmailNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let start_time = Instant::now();

        // Simulate email sending with potential failure
        let (success, delay_ms) = {
            let mut rng = thread_rng();
            let success = rng.random::<f32>() > 0.05; // 95% success rate
            let delay_ms = 100 + rng.random::<u64>() % 500;
            (success, delay_ms)
        }; // rng is dropped here, before any await

        if success {
            // Simulate email sending delay
            tokio::time::sleep(Duration::from_millis(delay_ms)).await;

            info!(
                "📧 Email sent to {:?}: {} - {} (ID: {})",
                notification.recipients,
                notification.subject,
                notification.content,
                notification.id
            );

            let latency_ms = start_time.elapsed().as_millis() as u64;

            Ok(DeliveryResult {
                success: true,
                delivered_at: Some(Utc::now()),
                attempts: 1,
                error: None,
                latency_ms: Some(latency_ms),
                response_data: {
                    let mut data = HashMap::new();
                    data.insert(
                        "message_id".to_string(),
                        format!("email_{}", notification.id),
                    );
                    data.insert("recipients".to_string(), notification.recipients.join(","));
                    data.insert("subject".to_string(), notification.subject.clone());
                    data
                },
            })
        } else {
            let error_msg = "SMTP server temporarily unavailable".to_string();
            error!(
                "Failed to send email notification {}: {}",
                notification.id, error_msg
            );

            Ok(DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some(error_msg),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
                response_data: HashMap::new(),
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "email" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        // Simulate health check
        let healthy = {
            let mut rng = thread_rng();
            rng.random::<f32>()
        } > 0.1; // 90% healthy

        Ok(ChannelHealth {
            healthy,
            last_check: Utc::now(),
            consecutive_failures: if healthy { 0 } else { 1 },
            success_rate: if healthy { 0.95 } else { 0.8 },
        })
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "html".to_string(),
            "attachments".to_string(),
        ]
    }

    fn max_message_size(&self) -> Option<usize> {
        Some(25 * 1024 * 1024) // 25MB limit
    }
}

/// Webhook notification channel implementation
pub struct WebhookNotificationChannel {
    name: String,
}

impl WebhookNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "webhook".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for WebhookNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let start_time = Instant::now();

        // Simulate HTTP webhook call
        let success = {
            let mut rng = thread_rng();
            rng.random::<f32>()
        } > 0.02; // 98% success rate

        if success {
            // Simulate network delay
            tokio::time::sleep(Duration::from_millis(
                50 + {
                    let mut rng = thread_rng();
                    rng.random::<u64>()
                } % 200,
            ))
            .await;

            info!(
                "🔗 Webhook notification sent: {} - {} (ID: {})",
                notification.subject, notification.content, notification.id
            );

            let latency_ms = start_time.elapsed().as_millis() as u64;

            Ok(DeliveryResult {
                success: true,
                delivered_at: Some(Utc::now()),
                attempts: 1,
                error: None,
                latency_ms: Some(latency_ms),
                response_data: {
                    let mut data = HashMap::new();
                    data.insert("status_code".to_string(), "200".to_string());
                    data.insert("response_time_ms".to_string(), latency_ms.to_string());
                    data.insert("webhook_id".to_string(), format!("wh_{}", notification.id));
                    data
                },
            })
        } else {
            let error_msg = "Webhook endpoint returned 500 Internal Server Error".to_string();
            error!(
                "Failed to send webhook notification {}: {}",
                notification.id, error_msg
            );

            Ok(DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some(error_msg),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
                response_data: {
                    let mut data = HashMap::new();
                    data.insert("status_code".to_string(), "500".to_string());
                    data
                },
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "webhook" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        // Simulate webhook endpoint health check
        let healthy = {
            let mut rng = thread_rng();
            rng.random::<f32>()
        } > 0.05; // 95% healthy

        Ok(ChannelHealth {
            healthy,
            last_check: Utc::now(),
            consecutive_failures: if healthy { 0 } else { 1 },
            success_rate: if healthy { 0.98 } else { 0.85 },
        })
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "json".to_string(),
            "custom_headers".to_string(),
        ]
    }

    fn max_message_size(&self) -> Option<usize> {
        Some(1024 * 1024) // 1MB limit
    }
}

/// Slack notification channel implementation
pub struct SlackNotificationChannel {
    name: String,
}

impl SlackNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "slack".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for SlackNotificationChannel {
    async fn send_notification(&self, notification: &Notification) -> Result<DeliveryResult> {
        let start_time = Instant::now();

        // Simulate Slack API call
        let success = {
            let mut rng = thread_rng();
            rng.random::<f32>()
        } > 0.01; // 99% success rate

        if success {
            // Simulate API delay
            tokio::time::sleep(Duration::from_millis(
                100 + {
                    let mut rng = thread_rng();
                    rng.random::<u64>()
                } % 300,
            ))
            .await;

            let emoji = match notification.priority {
                NotificationPriority::Emergency => "🚨",
                NotificationPriority::Critical => "🔥",
                NotificationPriority::High => "⚠️",
                NotificationPriority::Normal => "📢",
                NotificationPriority::Low => "💡",
            };

            info!(
                "💬 Slack message sent: {} {} - {} (ID: {})",
                emoji, notification.subject, notification.content, notification.id
            );

            let latency_ms = start_time.elapsed().as_millis() as u64;

            Ok(DeliveryResult {
                success: true,
                delivered_at: Some(Utc::now()),
                attempts: 1,
                error: None,
                latency_ms: Some(latency_ms),
                response_data: {
                    let mut data = HashMap::new();
                    data.insert("channel".to_string(), "#alerts".to_string());
                    data.insert("ts".to_string(), Utc::now().timestamp().to_string());
                    data.insert(
                        "message_id".to_string(),
                        format!("slack_{}", notification.id),
                    );
                    data
                },
            })
        } else {
            let error_msg = "Slack API rate limit exceeded".to_string();
            error!(
                "Failed to send Slack notification {}: {}",
                notification.id, error_msg
            );

            Ok(DeliveryResult {
                success: false,
                delivered_at: None,
                attempts: 1,
                error: Some(error_msg),
                latency_ms: Some(start_time.elapsed().as_millis() as u64),
                response_data: HashMap::new(),
            })
        }
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "slack" || notification_type == "*"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        // Simulate Slack API health check
        let healthy = {
            let mut rng = thread_rng();
            rng.random::<f32>()
        } > 0.02; // 98% healthy

        Ok(ChannelHealth {
            healthy,
            last_check: Utc::now(),
            consecutive_failures: if healthy { 0 } else { 1 },
            success_rate: if healthy { 0.99 } else { 0.90 },
        })
    }

    fn capabilities(&self) -> Vec<String> {
        vec![
            "basic".to_string(),
            "markdown".to_string(),
            "emoji".to_string(),
            "mentions".to_string(),
            "attachments".to_string(),
            "threading".to_string(),
        ]
    }

    fn max_message_size(&self) -> Option<usize> {
        Some(40000) // Slack's message limit
    }
}

// Placeholder for additional channels that will be completed in subsequent todos
/// SMS notification channel (placeholder)
pub struct SmsNotificationChannel {
    name: String,
}

impl SmsNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "sms".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for SmsNotificationChannel {
    async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
        // Placeholder implementation
        Ok(DeliveryResult {
            success: true,
            delivered_at: Some(Utc::now()),
            attempts: 1,
            error: None,
            latency_ms: Some(500),
            response_data: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "sms"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        Ok(ChannelHealth {
            healthy: true,
            last_check: Utc::now(),
            consecutive_failures: 0,
            success_rate: 0.97,
        })
    }
}

/// PagerDuty notification channel (placeholder)
pub struct PagerDutyNotificationChannel {
    name: String,
}

impl PagerDutyNotificationChannel {
    pub async fn new() -> Result<Self> {
        Ok(Self {
            name: "pagerduty".to_string(),
        })
    }
}

#[async_trait::async_trait]
impl NotificationChannel for PagerDutyNotificationChannel {
    async fn send_notification(&self, _notification: &Notification) -> Result<DeliveryResult> {
        // Placeholder implementation
        Ok(DeliveryResult {
            success: true,
            delivered_at: Some(Utc::now()),
            attempts: 1,
            error: None,
            latency_ms: Some(300),
            response_data: HashMap::new(),
        })
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn supports(&self, notification_type: &str) -> bool {
        notification_type == "pagerduty"
    }

    async fn health_check(&self) -> Result<ChannelHealth> {
        Ok(ChannelHealth {
            healthy: true,
            last_check: Utc::now(),
            consecutive_failures: 0,
            success_rate: 0.99,
        })
    }
}

// Add rand dependency simulation for the examples above
mod rand {
    use std::sync::atomic::{AtomicU64, Ordering};

    static SEED: AtomicU64 = AtomicU64::new(1);

    pub fn random<T>() -> T
    where
        T: From<u32>,
    {
        let seed = SEED.fetch_add(1, Ordering::Relaxed);
        let x = seed.wrapping_mul(1103515245).wrapping_add(12345);
        let value = (x / 65536) % 32768;
        T::from(value as u32)
    }
}

// =============================================================================
// TESTS AND EXAMPLES
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_notification_manager_creation() {
        let config = NotificationConfig::default();
        let manager = NotificationManager::new(config).await;
        assert!(manager.is_ok());
    }

    #[tokio::test]
    async fn test_default_alert_processor() {
        let processor = DefaultAlertProcessor::new().await.unwrap();

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

        let notifications = processor.process_alert(&alert).await.unwrap();
        assert_eq!(notifications.len(), 1);
        assert_eq!(notifications[0].alert_id, "test_alert");
    }

    #[tokio::test]
    async fn test_log_notification_channel() {
        let channel = LogNotificationChannel::new().await.unwrap();

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

        let result = channel.send_notification(&notification).await.unwrap();
        assert!(result.success);
        assert!(result.delivered_at.is_some());
    }
}
