//! # Rate Limiting and Throttling System
//!
//! Advanced rate limiting with adaptive throttling and priority-based limiting.

use super::types::*;
use anyhow::Result;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use parking_lot::{Mutex, RwLock};
use std::{
    collections::{HashMap, VecDeque},
    sync::{
        atomic::{AtomicU64, AtomicUsize, Ordering},
        Arc,
    },
    time::Duration,
};
use tokio::{sync::broadcast, task::JoinHandle, time::interval};
use tracing::{debug, info, warn};

#[derive(Debug)]

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
#[derive(Debug)]

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
#[derive(Debug)]

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
#[derive(Debug)]

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
#[derive(Debug)]

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
        let clamped_rate = target_rate.clamp(self.config.min_rate, self.config.max_rate);

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
