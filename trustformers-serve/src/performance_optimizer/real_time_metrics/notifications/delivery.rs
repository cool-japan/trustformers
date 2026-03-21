//! # Delivery Engine for Notification Delivery
//!
//! Handles reliable notification delivery with different guarantee levels.

use super::audit::AuditSystem;
use super::channels::NotificationChannel;
use super::health_monitor::ChannelHealthMonitor;
use super::rate_limiting::RateLimiter;
use super::types::*;
use anyhow::Result;
use chrono::{DateTime, Duration as ChronoDuration, Utc};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use tokio::{
    sync::broadcast,
    task::JoinHandle,
    time::{interval, timeout},
};
use tracing::{debug, error, info, warn};

#[derive(Debug)]

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
#[derive(Debug)]

pub struct RetryScheduler {
    /// Retry queue
    pub retry_queue: Arc<Mutex<VecDeque<RetryItem>>>,

    /// Retry statistics
    pub stats: Arc<RetryStats>,
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
