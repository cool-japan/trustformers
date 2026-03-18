//! # Central Notification Manager
//!
//! Orchestrates all notification processing, routing, and delivery.

use super::audit::AuditSystem;
use super::channels::{
    EmailNotificationChannel, LogNotificationChannel, NotificationChannel,
    SlackNotificationChannel, WebhookNotificationChannel,
};
use super::delivery::DeliveryEngine;
use super::escalation::EscalationEngine;
use super::formatters::MessageFormatter;
use super::health_monitor::{ChannelHealth, ChannelHealthMonitor};
use super::processors::{
    AlertProcessor, CriticalAlertProcessor, DefaultAlertProcessor, PerformanceAlertProcessor,
    ResourceAlertProcessor,
};
use super::rate_limiting::RateLimiter;
use super::types::*;
use crate::performance_optimizer::real_time_metrics::types::*;
use anyhow::Result;
use chrono::Utc;
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::{oneshot, Semaphore},
    task::JoinHandle,
};
use tracing::{debug, error, info};

#[derive(Debug)]

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
                    let _permit = match semaphore.acquire().await {
                        Ok(permit) => permit,
                        Err(_) => {
                            tracing::warn!(
                                "Notification processing semaphore closed, shutting down worker"
                            );
                            break;
                        },
                    };

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
