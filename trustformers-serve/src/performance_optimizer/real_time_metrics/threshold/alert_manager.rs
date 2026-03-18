//! Alert management system for processing and dispatching alerts.

use super::super::types::*;
use super::correlator::AlertCorrelator;
use super::error::{Result, ThresholdError};
use super::processors::{
    CriticalAlertProcessor, DefaultAlertProcessor, EmailNotificationChannel,
    LogNotificationChannel, PerformanceAlertProcessor, ResourceAlertProcessor,
    SlackNotificationChannel, WebhookNotificationChannel,
};
use super::suppressor::AlertSuppressor;
use chrono::{DateTime, Utc};
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use tokio::sync::Mutex as TokioMutex;
use tokio::time::interval;
use tracing::{debug, error, info, warn};

// =============================================================================
// ALERT MANAGEMENT SYSTEM
// =============================================================================

/// Alert manager for handling alert processing and notifications
///
/// Comprehensive alert manager that processes alerts, manages notifications,
/// handles suppression and correlation, and provides detailed analytics.
pub struct AlertManager {
    /// Alert processors
    processors: Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,

    /// Notification channels
    channels: Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,

    /// Alert queue for processing
    alert_queue: Arc<TokioMutex<VecDeque<AlertEvent>>>,

    /// Processing statistics
    stats: Arc<AlertManagerStats>,

    /// Alert suppressor
    suppressor: Arc<AlertSuppressor>,

    /// Alert correlator
    correlator: Arc<AlertCorrelator>,

    /// Processing configuration
    config: Arc<RwLock<AlertManagerConfig>>,

    /// Processing task handle
    processing_handle: Arc<TokioMutex<Option<tokio::task::JoinHandle<()>>>>,

    /// Shutdown signal
    shutdown_signal: Arc<AtomicBool>,
}

/// Statistics for alert manager
#[derive(Debug, Default)]
pub struct AlertManagerStats {
    /// Alerts processed
    pub alerts_processed: AtomicU64,

    /// Notifications sent
    pub notifications_sent: AtomicU64,

    /// Processing errors
    pub processing_errors: AtomicU64,

    /// Alerts suppressed
    pub alerts_suppressed: AtomicU64,

    /// Alerts correlated
    pub alerts_correlated: AtomicU64,

    /// Average processing time
    pub avg_processing_time: Arc<Mutex<Duration>>,

    /// Queue size
    pub queue_size: AtomicU64,

    /// Peak queue size
    pub peak_queue_size: AtomicU64,
}

/// Configuration for alert manager
#[derive(Debug, Clone)]
pub struct AlertManagerConfig {
    /// Maximum queue size
    pub max_queue_size: usize,

    /// Processing batch size
    pub batch_size: usize,

    /// Processing interval
    pub processing_interval: Duration,

    /// Enable alert suppression
    pub enable_suppression: bool,

    /// Enable alert correlation
    pub enable_correlation: bool,

    /// Maximum processing time per alert
    pub max_processing_time: Duration,

    /// Number of worker threads
    pub worker_threads: usize,
}

/// Alert processor trait for processing alerts
pub trait AlertProcessor: Send + Sync {
    /// Process an alert
    fn process_alert(&self, alert: &AlertEvent) -> Result<ProcessedAlert>;

    /// Get processor name
    fn name(&self) -> &str;

    /// Check if processor supports alert type
    fn supports(&self, alert: &AlertEvent) -> bool;

    /// Get processing priority
    fn priority(&self) -> u8 {
        50 // Default priority
    }
}

/// Processed alert result
#[derive(Debug, Clone)]
pub struct ProcessedAlert {
    /// Original alert
    pub alert: AlertEvent,

    /// Processing timestamp
    pub processed_at: DateTime<Utc>,

    /// Processing results
    pub results: HashMap<String, String>,

    /// Notifications to send
    pub notifications: Vec<Notification>,

    /// Processing metadata
    pub metadata: HashMap<String, String>,
}

/// Notification to be sent
#[derive(Debug, Clone)]
pub struct Notification {
    /// Notification ID
    pub id: String,

    /// Channel to use
    pub channel: String,

    /// Recipients
    pub recipients: Vec<String>,

    /// Subject/title
    pub subject: String,

    /// Content/body
    pub content: String,

    /// Priority level
    pub priority: NotificationPriority,

    /// Metadata
    pub metadata: HashMap<String, String>,
}

/// Notification priority levels
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum NotificationPriority {
    /// Low priority
    Low,

    /// Normal priority
    Normal,

    /// High priority
    High,

    /// Critical priority
    Critical,

    /// Emergency priority
    Emergency,
}

/// Trait for notification channels
pub trait NotificationChannel: Send + Sync {
    /// Send notification
    fn send_notification(&self, notification: &Notification) -> Result<()>;

    /// Get channel name
    fn name(&self) -> &str;

    /// Check if channel supports notification type
    fn supports(&self, notification_type: &str) -> bool;

    /// Get maximum message size
    fn max_message_size(&self) -> usize {
        10000 // Default 10KB
    }

    /// Check if channel is available
    fn is_available(&self) -> bool {
        true // Default available
    }
}

impl AlertManager {
    /// Create a new alert manager
    pub async fn new() -> Result<Self> {
        let manager = Self {
            processors: Arc::new(Mutex::new(Vec::new())),
            channels: Arc::new(Mutex::new(Vec::new())),
            alert_queue: Arc::new(TokioMutex::new(VecDeque::new())),
            stats: Arc::new(AlertManagerStats::default()),
            suppressor: Arc::new(AlertSuppressor::new()),
            correlator: Arc::new(AlertCorrelator::new()),
            config: Arc::new(RwLock::new(AlertManagerConfig::default())),
            processing_handle: Arc::new(TokioMutex::new(None)),
            shutdown_signal: Arc::new(AtomicBool::new(false)),
        };

        manager.initialize_processors().await?;
        manager.initialize_channels().await?;
        Ok(manager)
    }

    /// Start alert manager
    pub async fn start(&self) -> Result<()> {
        let mut handle = self.processing_handle.lock().await;

        if handle.is_some() {
            return Err(ThresholdError::InternalError(
                "Alert manager already started".to_string(),
            ));
        }

        let queue = Arc::clone(&self.alert_queue);
        let processors = Arc::clone(&self.processors);
        let channels = Arc::clone(&self.channels);
        let stats = Arc::clone(&self.stats);
        let suppressor = Arc::clone(&self.suppressor);
        let correlator = Arc::clone(&self.correlator);
        let config = Arc::clone(&self.config);
        let shutdown_signal = Arc::clone(&self.shutdown_signal);

        let processing_task = tokio::spawn(async move {
            Self::processing_loop(
                queue,
                processors,
                channels,
                stats,
                suppressor,
                correlator,
                config,
                shutdown_signal,
            )
            .await;
        });

        *handle = Some(processing_task);
        info!("Alert manager started");
        Ok(())
    }

    /// Stop alert manager
    pub async fn shutdown(&self) -> Result<()> {
        self.shutdown_signal.store(true, Ordering::Relaxed);

        let mut handle = self.processing_handle.lock().await;
        if let Some(task) = handle.take() {
            task.abort();
            info!("Alert manager stopped");
        }

        Ok(())
    }

    /// Process an alert
    pub async fn process_alert(&self, alert: AlertEvent) -> Result<()> {
        let config = self.config.read().expect("Config RwLock poisoned");
        let mut queue = self.alert_queue.lock().await;

        // Check queue size limits
        if queue.len() >= config.max_queue_size {
            return Err(ThresholdError::AlertProcessingError(
                "Alert queue is full".to_string(),
            ));
        }

        queue.push_back(alert);

        // Update statistics
        let queue_size = queue.len() as u64;
        self.stats.queue_size.store(queue_size, Ordering::Relaxed);

        let peak_size = self.stats.peak_queue_size.load(Ordering::Relaxed);
        if queue_size > peak_size {
            self.stats.peak_queue_size.store(queue_size, Ordering::Relaxed);
        }

        Ok(())
    }

    /// Get alert manager statistics
    pub fn get_stats(&self) -> AlertManagerStats {
        AlertManagerStats {
            alerts_processed: AtomicU64::new(self.stats.alerts_processed.load(Ordering::Relaxed)),
            notifications_sent: AtomicU64::new(
                self.stats.notifications_sent.load(Ordering::Relaxed),
            ),
            processing_errors: AtomicU64::new(self.stats.processing_errors.load(Ordering::Relaxed)),
            alerts_suppressed: AtomicU64::new(self.stats.alerts_suppressed.load(Ordering::Relaxed)),
            alerts_correlated: AtomicU64::new(self.stats.alerts_correlated.load(Ordering::Relaxed)),
            avg_processing_time: Arc::new(Mutex::new(
                *self
                    .stats
                    .avg_processing_time
                    .lock()
                    .expect("Avg processing time lock poisoned"),
            )),
            queue_size: AtomicU64::new(self.stats.queue_size.load(Ordering::Relaxed)),
            peak_queue_size: AtomicU64::new(self.stats.peak_queue_size.load(Ordering::Relaxed)),
        }
    }

    /// Main processing loop
    async fn processing_loop(
        queue: Arc<TokioMutex<VecDeque<AlertEvent>>>,
        processors: Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,
        channels: Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
        stats: Arc<AlertManagerStats>,
        suppressor: Arc<AlertSuppressor>,
        correlator: Arc<AlertCorrelator>,
        config: Arc<RwLock<AlertManagerConfig>>,
        shutdown_signal: Arc<AtomicBool>,
    ) {
        let mut interval = interval(Duration::from_millis(100)); // 100ms processing interval

        while !shutdown_signal.load(Ordering::Relaxed) {
            interval.tick().await;

            let (batch_size, enable_suppression, enable_correlation) = {
                let config_read = config.read().expect("Config RwLock poisoned");
                (
                    config_read.batch_size,
                    config_read.enable_suppression,
                    config_read.enable_correlation,
                )
            };

            // Process alerts in batches
            let alerts = {
                let mut queue_guard = queue.lock().await;
                let mut batch = Vec::new();

                for _ in 0..batch_size {
                    if let Some(alert) = queue_guard.pop_front() {
                        batch.push(alert);
                    } else {
                        break;
                    }
                }

                stats.queue_size.store(queue_guard.len() as u64, Ordering::Relaxed);
                batch
            };

            if alerts.is_empty() {
                continue;
            }

            // Process each alert
            for mut alert in alerts {
                let start_time = Instant::now();

                // Apply suppression
                if enable_suppression && suppressor.should_suppress(&alert).await {
                    suppressor.suppress_alert(&mut alert).await;
                    stats.alerts_suppressed.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // Apply correlation
                if enable_correlation {
                    correlator.correlate_alert(&mut alert).await;
                    stats.alerts_correlated.fetch_add(1, Ordering::Relaxed);
                }

                // Process alert
                match Self::process_single_alert(&alert, &processors, &channels).await {
                    Ok(_) => {
                        stats.alerts_processed.fetch_add(1, Ordering::Relaxed);
                    },
                    Err(e) => {
                        error!("Failed to process alert {}: {}", alert.alert_id, e);
                        stats.processing_errors.fetch_add(1, Ordering::Relaxed);
                    },
                }

                // Update processing time statistics
                let processing_time = start_time.elapsed();
                let mut avg_time =
                    stats.avg_processing_time.lock().expect("Avg time lock poisoned");
                *avg_time = (*avg_time + processing_time) / 2; // Simple moving average
            }
        }
    }

    /// Process a single alert
    async fn process_single_alert(
        alert: &AlertEvent,
        processors: &Arc<Mutex<Vec<Box<dyn AlertProcessor + Send + Sync>>>>,
        channels: &Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
    ) -> Result<()> {
        let all_notifications = {
            let processors_guard = processors.lock().expect("Processors lock poisoned");
            let mut notifications = Vec::new();

            // Find and run appropriate processors
            for processor in processors_guard.iter() {
                if processor.supports(alert) {
                    match processor.process_alert(alert) {
                        Ok(processed) => {
                            notifications.extend(processed.notifications);
                        },
                        Err(e) => {
                            warn!(
                                "Processor {} failed for alert {}: {}",
                                processor.name(),
                                alert.alert_id,
                                e
                            );
                        },
                    }
                }
            }

            notifications
        };

        // Send notifications
        if !all_notifications.is_empty() {
            Self::send_notifications(&all_notifications, channels).await?;
        }

        Ok(())
    }

    /// Send notifications through available channels
    async fn send_notifications(
        notifications: &[Notification],
        channels: &Arc<Mutex<Vec<Box<dyn NotificationChannel + Send + Sync>>>>,
    ) -> Result<()> {
        let channels_guard = channels.lock().expect("Channels lock poisoned");

        for notification in notifications {
            for channel in channels_guard.iter() {
                if channel.supports(&notification.channel) && channel.is_available() {
                    match channel.send_notification(notification) {
                        Ok(()) => {
                            debug!(
                                "Sent notification {} via {}",
                                notification.id,
                                channel.name()
                            );
                        },
                        Err(e) => {
                            warn!(
                                "Failed to send notification {} via {}: {}",
                                notification.id,
                                channel.name(),
                                e
                            );
                        },
                    }
                }
            }
        }

        Ok(())
    }

    /// Initialize alert processors
    async fn initialize_processors(&self) -> Result<()> {
        let mut processors = self.processors.lock().expect("Processors lock poisoned");
        processors.push(Box::new(DefaultAlertProcessor::new()));
        processors.push(Box::new(PerformanceAlertProcessor::new()));
        processors.push(Box::new(ResourceAlertProcessor::new()));
        processors.push(Box::new(CriticalAlertProcessor::new()));
        Ok(())
    }

    /// Initialize notification channels
    async fn initialize_channels(&self) -> Result<()> {
        let mut channels = self.channels.lock().expect("Channels lock poisoned");
        channels.push(Box::new(LogNotificationChannel::new()));
        channels.push(Box::new(EmailNotificationChannel::new()));
        channels.push(Box::new(WebhookNotificationChannel::new()));
        channels.push(Box::new(SlackNotificationChannel::new()));
        Ok(())
    }
}

impl Default for AlertManagerConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            batch_size: 50,
            processing_interval: Duration::from_millis(100),
            enable_suppression: true,
            enable_correlation: true,
            max_processing_time: Duration::from_secs(30),
            worker_threads: 4,
        }
    }
}
