//! Batch Scheduler for Dynamic Batching

use crate::batching::{
    aggregator::{RequestBatch, RequestId},
    config::{BatchingConfig, Priority},
};
use anyhow::Result;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Batch scheduler that manages batch execution order
pub struct BatchScheduler {
    config: BatchingConfig,
    policy: SchedulingPolicy,
    queue: Arc<Mutex<PriorityQueue<ScheduledBatch>>>,
    stats: Arc<Mutex<SchedulerStats>>,
}

impl BatchScheduler {
    pub fn new(config: BatchingConfig) -> Self {
        let policy = match config.optimization_target {
            crate::batching::config::OptimizationTarget::Throughput => SchedulingPolicy::Throughput,
            crate::batching::config::OptimizationTarget::Latency => SchedulingPolicy::Latency,
            crate::batching::config::OptimizationTarget::Balanced => SchedulingPolicy::Balanced,
            crate::batching::config::OptimizationTarget::Cost => SchedulingPolicy::Cost,
        };

        Self {
            config,
            policy,
            queue: Arc::new(Mutex::new(PriorityQueue::new())),
            stats: Arc::new(Mutex::new(SchedulerStats::default())),
        }
    }

    /// Schedule a batch for execution
    pub async fn schedule_batch(&self, batch: RequestBatch) -> Result<()> {
        let scheduled = ScheduledBatch::new(batch, &self.policy);

        let mut queue = self.queue.lock().await;
        queue.push(scheduled);

        self.stats.lock().await.record_scheduled();

        Ok(())
    }

    /// Get the next batch to process
    pub async fn get_next_batch(&self) -> Option<RequestBatch> {
        let mut queue = self.queue.lock().await;

        if let Some(scheduled) = queue.pop() {
            self.stats.lock().await.record_dispatched();
            Some(scheduled.batch)
        } else {
            None
        }
    }

    /// Reorder queue based on new policy
    pub async fn reorder_queue(&self, policy: SchedulingPolicy) -> Result<()> {
        let mut queue = self.queue.lock().await;

        // Extract all batches
        let mut batches = Vec::new();
        while let Some(scheduled) = queue.pop() {
            batches.push(scheduled.batch);
        }

        // Re-insert with new policy
        for batch in batches {
            queue.push(ScheduledBatch::new(batch, &policy));
        }

        Ok(())
    }

    /// Update scheduler configuration
    pub fn update_config(&self, _config: BatchingConfig) -> Result<()> {
        // In practice, would update internal config
        Ok(())
    }

    /// Get scheduler statistics
    pub fn get_stats(&self) -> SchedulerStats {
        // Would need async access
        SchedulerStats::default()
    }

    /// Get queue depth
    pub async fn queue_depth(&self) -> usize {
        self.queue.lock().await.len()
    }
}

/// Scheduled batch with priority
#[derive(Debug)]
struct ScheduledBatch {
    batch: RequestBatch,
    priority_score: f64,
    scheduled_at: Instant,
}

impl ScheduledBatch {
    fn new(batch: RequestBatch, policy: &SchedulingPolicy) -> Self {
        let priority_score = policy.calculate_priority(&batch);

        Self {
            batch,
            priority_score,
            scheduled_at: Instant::now(),
        }
    }
}

impl PartialEq for ScheduledBatch {
    fn eq(&self, other: &Self) -> bool {
        self.priority_score == other.priority_score
    }
}

impl Eq for ScheduledBatch {}

impl Ord for ScheduledBatch {
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority_score
            .partial_cmp(&other.priority_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for ScheduledBatch {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Scheduling policy
#[derive(Debug, Clone, Copy)]
pub enum SchedulingPolicy {
    /// Optimize for throughput
    Throughput,
    /// Optimize for latency
    Latency,
    /// Balance throughput and latency
    Balanced,
    /// Optimize for cost
    Cost,
    /// Custom policy
    Custom,
}

impl SchedulingPolicy {
    /// Calculate priority score for a batch
    fn calculate_priority(&self, batch: &RequestBatch) -> f64 {
        match self {
            Self::Throughput => {
                // Larger batches get higher priority
                batch.requests.len() as f64
            },
            Self::Latency => {
                // Older batches get higher priority
                let age = batch.created_at.elapsed().as_secs_f64();
                let priority_boost = batch.priority as u8 as f64;
                age * (1.0 + priority_boost)
            },
            Self::Balanced => {
                // Balance between batch size and age
                let size_score = batch.requests.len() as f64 / 32.0; // Normalize
                let age_score = batch.created_at.elapsed().as_secs_f64();
                (size_score + age_score) / 2.0
            },
            Self::Cost => {
                // Optimize for efficient resource usage
                let efficiency = batch.requests.len() as f64 / batch.total_memory as f64;
                efficiency * 1000.0 // Scale up
            },
            Self::Custom => {
                // Default to balanced
                Self::Balanced.calculate_priority(batch)
            },
        }
    }
}

/// Priority queue for scheduled batches
pub struct PriorityQueue<T> {
    heap: BinaryHeap<T>,
}

impl<T: Ord> Default for PriorityQueue<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Ord> PriorityQueue<T> {
    pub fn new() -> Self {
        Self {
            heap: BinaryHeap::new(),
        }
    }

    pub fn push(&mut self, item: T) {
        self.heap.push(item);
    }

    pub fn pop(&mut self) -> Option<T> {
        self.heap.pop()
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }
}

/// Timeout policy for batch scheduling
#[derive(Debug, Clone)]
pub struct TimeoutPolicy {
    base_timeout: Duration,
    priority_multipliers: HashMap<Priority, f64>,
    max_timeout: Duration,
}

impl Default for TimeoutPolicy {
    fn default() -> Self {
        let mut priority_multipliers = HashMap::new();
        priority_multipliers.insert(Priority::Low, 2.0);
        priority_multipliers.insert(Priority::Normal, 1.0);
        priority_multipliers.insert(Priority::High, 0.5);
        priority_multipliers.insert(Priority::Critical, 0.25);

        Self {
            base_timeout: Duration::from_millis(50),
            priority_multipliers,
            max_timeout: Duration::from_millis(200),
        }
    }
}

impl TimeoutPolicy {
    /// Calculate timeout for a request based on priority
    pub fn calculate_timeout(&self, priority: Priority) -> Duration {
        let multiplier = self.priority_multipliers.get(&priority).copied().unwrap_or(1.0);

        let timeout = self.base_timeout.mul_f64(multiplier);
        timeout.min(self.max_timeout)
    }
}

/// Advanced scheduler with multiple queues
pub struct MultiQueueScheduler {
    queues: HashMap<Priority, Arc<Mutex<VecDeque<RequestBatch>>>>,
    round_robin_index: Arc<Mutex<usize>>,
}

impl Default for MultiQueueScheduler {
    fn default() -> Self {
        Self::new()
    }
}

impl MultiQueueScheduler {
    pub fn new() -> Self {
        let mut queues = HashMap::new();
        queues.insert(Priority::Critical, Arc::new(Mutex::new(VecDeque::new())));
        queues.insert(Priority::High, Arc::new(Mutex::new(VecDeque::new())));
        queues.insert(Priority::Normal, Arc::new(Mutex::new(VecDeque::new())));
        queues.insert(Priority::Low, Arc::new(Mutex::new(VecDeque::new())));

        Self {
            queues,
            round_robin_index: Arc::new(Mutex::new(0)),
        }
    }

    /// Add batch to appropriate queue
    pub async fn add_batch(&self, batch: RequestBatch) -> Result<()> {
        let priority = batch.priority;

        if let Some(queue) = self.queues.get(&priority) {
            queue.lock().await.push_back(batch);
        }

        Ok(())
    }

    /// Get next batch using weighted round-robin
    pub async fn get_next_batch(&self) -> Option<RequestBatch> {
        // Check critical queue first
        if let Some(queue) = self.queues.get(&Priority::Critical) {
            if let Some(batch) = queue.lock().await.pop_front() {
                return Some(batch);
            }
        }

        // Then use weighted round-robin for other priorities
        let priorities = [
            Priority::High,
            Priority::High, // High gets 2x weight
            Priority::Normal,
            Priority::Low,
        ];

        let mut index = self.round_robin_index.lock().await;

        for _ in 0..priorities.len() {
            let priority = &priorities[*index % priorities.len()];
            *index += 1;

            if let Some(queue) = self.queues.get(priority) {
                if let Some(batch) = queue.lock().await.pop_front() {
                    return Some(batch);
                }
            }
        }

        None
    }
}

/// Scheduler statistics
#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct SchedulerStats {
    pub total_scheduled: usize,
    pub total_dispatched: usize,
    pub avg_queue_time_ms: f64,
    #[serde(skip)]
    pub queue_depth_history: Vec<(Instant, usize)>,
}

impl SchedulerStats {
    fn record_scheduled(&mut self) {
        self.total_scheduled += 1;
    }

    fn record_dispatched(&mut self) {
        self.total_dispatched += 1;
    }
}

use std::collections::VecDeque;

/// Advanced Request Queuing and Prioritization System
///
/// This system provides comprehensive request queuing with intelligent prioritization,
/// quality of service guarantees, and adaptive scheduling algorithms.

/// Advanced priority levels with dynamic adjustment capabilities
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AdvancedPriority {
    /// Emergency requests (system critical)
    Emergency = 100,
    /// Real-time requests with strict latency requirements
    RealTime = 90,
    /// Premium tier requests with SLA guarantees
    Premium = 80,
    /// Business critical requests
    Critical = 70,
    /// High priority user requests
    High = 60,
    /// Standard priority requests
    Normal = 50,
    /// Background processing requests
    Background = 40,
    /// Batch processing requests
    Batch = 30,
    /// Low priority analytical requests
    Analytics = 20,
    /// System maintenance requests
    Maintenance = 10,
}

impl AdvancedPriority {
    /// Get timeout multiplier for this priority level
    pub fn timeout_multiplier(&self) -> f64 {
        match self {
            Self::Emergency => 0.1,
            Self::RealTime => 0.2,
            Self::Premium => 0.3,
            Self::Critical => 0.5,
            Self::High => 0.7,
            Self::Normal => 1.0,
            Self::Background => 2.0,
            Self::Batch => 5.0,
            Self::Analytics => 10.0,
            Self::Maintenance => 20.0,
        }
    }

    /// Get resource allocation weight
    pub fn resource_weight(&self) -> f64 {
        *self as u8 as f64 / 100.0
    }
}

/// Quality of Service configuration for request handling
#[derive(Debug, Clone)]
pub struct QoSConfig {
    /// Maximum allowed latency by priority level
    pub max_latency_ms: HashMap<AdvancedPriority, u64>,
    /// Resource reservation by priority
    pub resource_reservations: HashMap<AdvancedPriority, f64>,
    /// SLA requirements
    pub sla_requirements: HashMap<String, SLARequirement>,
    /// Rate limiting by client/priority
    pub rate_limits: HashMap<String, RateLimit>,
}

/// Service Level Agreement requirement
#[derive(Debug, Clone)]
pub struct SLARequirement {
    pub max_response_time_ms: u64,
    pub availability_percent: f64,
    pub throughput_rps: f64,
    pub priority_boost: i8,
}

/// Rate limiting configuration
#[derive(Debug, Clone)]
pub struct RateLimit {
    pub requests_per_second: u32,
    pub burst_capacity: u32,
    pub window_ms: u64,
}

/// Advanced request with comprehensive metadata
#[derive(Debug, Clone)]
pub struct AdvancedRequest {
    pub id: RequestId,
    pub priority: AdvancedPriority,
    pub client_id: String,
    pub service_tier: String,
    pub sla_class: Option<String>,
    pub deadline: Option<Instant>,
    pub retry_count: u32,
    pub original_priority: AdvancedPriority,
    pub created_at: Instant,
    pub size_estimate: usize,
    pub complexity_score: f64,
    pub resource_requirements: ResourceRequirements,
    pub timeout_ms: u64,
    pub callback_url: Option<String>,
    pub metadata: HashMap<String, String>,
}

/// Resource requirements for a request
#[derive(Debug, Clone)]
pub struct ResourceRequirements {
    pub cpu_cores: f64,
    pub memory_mb: u64,
    pub gpu_memory_mb: u64,
    pub disk_io_mb: u64,
    pub network_bandwidth_mbps: f64,
}

/// Comprehensive Queue Management System
#[derive(Debug)]
pub struct AdvancedQueueManager {
    /// Priority-based queues
    priority_queues: HashMap<AdvancedPriority, Arc<Mutex<VecDeque<AdvancedRequest>>>>,
    /// Deadline-aware queue for time-sensitive requests
    deadline_queue: Arc<Mutex<BinaryHeap<DeadlineRequest>>>,
    /// Fair-share queues by client
    client_queues: HashMap<String, Arc<Mutex<VecDeque<AdvancedRequest>>>>,
    /// QoS configuration
    qos_config: QoSConfig,
    /// Rate limiters by client
    rate_limiters: HashMap<String, Arc<Mutex<TokenBucket>>>,
    /// Queue statistics
    stats: Arc<Mutex<AdvancedQueueStats>>,
    /// Configuration
    config: AdvancedQueueConfig,
    /// Load monitor
    load_monitor: Arc<Mutex<SystemLoadMonitor>>,
}

/// Request with deadline priority for deadline queue
#[derive(Debug, Clone)]
struct DeadlineRequest {
    request: AdvancedRequest,
    deadline_score: f64,
}

impl PartialEq for DeadlineRequest {
    fn eq(&self, other: &Self) -> bool {
        self.deadline_score == other.deadline_score
    }
}

impl Eq for DeadlineRequest {}

impl Ord for DeadlineRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse order for min-heap behavior (earliest deadline first)
        other
            .deadline_score
            .partial_cmp(&self.deadline_score)
            .unwrap_or(Ordering::Equal)
    }
}

impl PartialOrd for DeadlineRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Token bucket for rate limiting
#[derive(Debug)]
pub struct TokenBucket {
    capacity: u32,
    tokens: u32,
    refill_rate: f64,
    last_refill: Instant,
}

impl TokenBucket {
    pub fn new(capacity: u32, refill_rate: f64) -> Self {
        Self {
            capacity,
            tokens: capacity,
            refill_rate,
            last_refill: Instant::now(),
        }
    }

    /// Try to consume tokens for a request
    pub fn try_consume(&mut self, tokens: u32) -> bool {
        self.refill();

        if self.tokens >= tokens {
            self.tokens -= tokens;
            true
        } else {
            false
        }
    }

    fn refill(&mut self) {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();

        let new_tokens = (elapsed * self.refill_rate) as u32;
        self.tokens = (self.tokens + new_tokens).min(self.capacity);
        self.last_refill = now;
    }
}

/// System load monitoring for adaptive scheduling
#[derive(Debug, Clone)]
pub struct SystemLoadMonitor {
    cpu_usage: f64,
    memory_usage: f64,
    gpu_usage: f64,
    queue_depths: HashMap<AdvancedPriority, usize>,
    throughput_history: VecDeque<(Instant, f64)>,
    last_update: Instant,
}

impl Default for SystemLoadMonitor {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemLoadMonitor {
    pub fn new() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_usage: 0.0,
            gpu_usage: 0.0,
            queue_depths: HashMap::new(),
            throughput_history: VecDeque::new(),
            last_update: Instant::now(),
        }
    }

    /// Update system metrics
    pub fn update_metrics(&mut self, cpu: f64, memory: f64, gpu: f64) {
        self.cpu_usage = cpu;
        self.memory_usage = memory;
        self.gpu_usage = gpu;
        self.last_update = Instant::now();
    }

    /// Get overall system load score (0.0 = idle, 1.0 = fully loaded)
    pub fn get_load_score(&self) -> f64 {
        let cpu_weight = 0.4;
        let memory_weight = 0.3;
        let gpu_weight = 0.3;

        (self.cpu_usage * cpu_weight
            + self.memory_usage * memory_weight
            + self.gpu_usage * gpu_weight)
            .min(1.0)
    }

    /// Recommend scheduling strategy based on current load
    pub fn recommend_strategy(&self) -> SchedulingStrategy {
        let load = self.get_load_score();

        if load > 0.9 {
            SchedulingStrategy::ConservativeHighPriority
        } else if load > 0.7 {
            SchedulingStrategy::BalancedAdaptive
        } else if load > 0.4 {
            SchedulingStrategy::ThroughputOptimized
        } else {
            SchedulingStrategy::FairShare
        }
    }
}

/// Advanced scheduling strategies
#[derive(Debug, Clone, Copy)]
pub enum SchedulingStrategy {
    /// Strict priority-based scheduling
    StrictPriority,
    /// Fair share among clients
    FairShare,
    /// Deadline-aware scheduling
    DeadlineAware,
    /// Throughput optimized
    ThroughputOptimized,
    /// Balanced adaptive scheduling
    BalancedAdaptive,
    /// Conservative high-priority only during high load
    ConservativeHighPriority,
    /// Quality of Service aware
    QoSAware,
}

/// Configuration for advanced queue manager
#[derive(Debug, Clone)]
pub struct AdvancedQueueConfig {
    pub max_queue_size: usize,
    pub default_timeout_ms: u64,
    pub enable_deadline_scheduling: bool,
    pub enable_fair_share: bool,
    pub enable_rate_limiting: bool,
    pub enable_priority_inheritance: bool,
    pub enable_adaptive_scheduling: bool,
    pub load_balancing_strategy: SchedulingStrategy,
    pub priority_aging_enabled: bool,
    pub priority_aging_factor: f64,
}

impl Default for AdvancedQueueConfig {
    fn default() -> Self {
        Self {
            max_queue_size: 10000,
            default_timeout_ms: 30000,
            enable_deadline_scheduling: true,
            enable_fair_share: true,
            enable_rate_limiting: true,
            enable_priority_inheritance: true,
            enable_adaptive_scheduling: true,
            load_balancing_strategy: SchedulingStrategy::BalancedAdaptive,
            priority_aging_enabled: true,
            priority_aging_factor: 1.1,
        }
    }
}

/// Comprehensive queue statistics
#[derive(Debug, Clone, Default)]
pub struct AdvancedQueueStats {
    pub total_enqueued: usize,
    pub total_dequeued: usize,
    pub total_dropped: usize,
    pub total_timeouts: usize,
    pub avg_wait_time_ms: f64,
    pub avg_processing_time_ms: f64,
    pub throughput_rps: f64,
    pub queue_depths: HashMap<AdvancedPriority, usize>,
    pub client_stats: HashMap<String, ClientStats>,
    pub sla_violations: usize,
    pub priority_promotions: usize,
}

/// Per-client statistics
#[derive(Debug, Clone, Default)]
pub struct ClientStats {
    pub requests_served: usize,
    pub avg_latency_ms: f64,
    pub rate_limit_violations: usize,
    pub sla_violations: usize,
    pub last_request_time: Option<Instant>,
}

impl AdvancedQueueManager {
    /// Create a new advanced queue manager
    pub fn new(config: AdvancedQueueConfig, qos_config: QoSConfig) -> Self {
        let mut priority_queues = HashMap::new();
        for priority in [
            AdvancedPriority::Emergency,
            AdvancedPriority::RealTime,
            AdvancedPriority::Premium,
            AdvancedPriority::Critical,
            AdvancedPriority::High,
            AdvancedPriority::Normal,
            AdvancedPriority::Background,
            AdvancedPriority::Batch,
            AdvancedPriority::Analytics,
            AdvancedPriority::Maintenance,
        ] {
            priority_queues.insert(priority, Arc::new(Mutex::new(VecDeque::new())));
        }

        Self {
            priority_queues,
            deadline_queue: Arc::new(Mutex::new(BinaryHeap::new())),
            client_queues: HashMap::new(),
            qos_config,
            rate_limiters: HashMap::new(),
            stats: Arc::new(Mutex::new(AdvancedQueueStats::default())),
            config,
            load_monitor: Arc::new(Mutex::new(SystemLoadMonitor::new())),
        }
    }

    /// Enqueue a request with comprehensive validation and prioritization
    pub async fn enqueue_request(&mut self, mut request: AdvancedRequest) -> Result<()> {
        // Rate limiting check
        if self.config.enable_rate_limiting && !self.check_rate_limit(&request.client_id).await {
            self.stats
                .lock()
                .await
                .client_stats
                .entry(request.client_id.clone())
                .or_default()
                .rate_limit_violations += 1;

            return Err(anyhow::anyhow!(
                "Rate limit exceeded for client: {}",
                request.client_id
            ));
        }

        // Apply SLA-based priority adjustments
        if let Some(sla_class) = &request.sla_class {
            if let Some(sla) = self.qos_config.sla_requirements.get(sla_class) {
                request.priority = self.adjust_priority_for_sla(request.priority, sla);
            }
        }

        // Apply priority aging if enabled
        if self.config.priority_aging_enabled {
            request.priority = self.apply_priority_aging(request.priority, request.created_at);
        }

        // Check queue capacity
        let total_queued = self.get_total_queue_depth().await;
        if total_queued >= self.config.max_queue_size {
            // Drop lowest priority request to make room
            if !self.make_room_for_request(&request).await {
                self.stats.lock().await.total_dropped += 1;
                return Err(anyhow::anyhow!(
                    "Queue full, could not make room for request"
                ));
            }
        }

        // Add to appropriate queues
        self.add_to_priority_queue(request.clone()).await?;

        if self.config.enable_deadline_scheduling && request.deadline.is_some() {
            self.add_to_deadline_queue(request.clone()).await?;
        }

        if self.config.enable_fair_share {
            self.add_to_client_queue(request.clone()).await?;
        }

        // Update statistics
        self.stats.lock().await.total_enqueued += 1;
        self.update_queue_depth_stats().await;

        Ok(())
    }

    /// Dequeue next request using intelligent scheduling
    pub async fn dequeue_request(&mut self) -> Option<AdvancedRequest> {
        let load_monitor = self.load_monitor.lock().await;
        let strategy = if self.config.enable_adaptive_scheduling {
            load_monitor.recommend_strategy()
        } else {
            self.config.load_balancing_strategy
        };
        drop(load_monitor);

        let request = match strategy {
            SchedulingStrategy::StrictPriority => self.dequeue_by_priority().await,
            SchedulingStrategy::FairShare => self.dequeue_fair_share().await,
            SchedulingStrategy::DeadlineAware => self.dequeue_deadline_aware().await,
            SchedulingStrategy::ThroughputOptimized => self.dequeue_throughput_optimized().await,
            SchedulingStrategy::BalancedAdaptive => self.dequeue_balanced_adaptive().await,
            SchedulingStrategy::ConservativeHighPriority => {
                self.dequeue_conservative_high_priority().await
            },
            SchedulingStrategy::QoSAware => self.dequeue_qos_aware().await,
        };

        if request.is_some() {
            self.stats.lock().await.total_dequeued += 1;
            self.update_queue_depth_stats().await;
        }

        request
    }

    /// Check if client is within rate limits
    async fn check_rate_limit(&mut self, client_id: &str) -> bool {
        if let Some(rate_limit) = self.qos_config.rate_limits.get(client_id) {
            let limiter = self.rate_limiters.entry(client_id.to_string()).or_insert_with(|| {
                Arc::new(Mutex::new(TokenBucket::new(
                    rate_limit.burst_capacity,
                    rate_limit.requests_per_second as f64,
                )))
            });

            limiter.lock().await.try_consume(1)
        } else {
            true // No rate limit configured
        }
    }

    /// Adjust priority based on SLA requirements
    fn adjust_priority_for_sla(
        &self,
        base_priority: AdvancedPriority,
        sla: &SLARequirement,
    ) -> AdvancedPriority {
        let base_value = base_priority as u8;
        let adjusted_value = (base_value as i16 + sla.priority_boost as i16)
            .max(AdvancedPriority::Maintenance as i16)
            .min(AdvancedPriority::Emergency as i16) as u8;

        match adjusted_value {
            100..=255 => AdvancedPriority::Emergency,
            90..=99 => AdvancedPriority::RealTime,
            80..=89 => AdvancedPriority::Premium,
            70..=79 => AdvancedPriority::Critical,
            60..=69 => AdvancedPriority::High,
            50..=59 => AdvancedPriority::Normal,
            40..=49 => AdvancedPriority::Background,
            30..=39 => AdvancedPriority::Batch,
            20..=29 => AdvancedPriority::Analytics,
            _ => AdvancedPriority::Maintenance,
        }
    }

    /// Apply priority aging to increase priority over time
    fn apply_priority_aging(
        &self,
        base_priority: AdvancedPriority,
        created_at: Instant,
    ) -> AdvancedPriority {
        let age_seconds = created_at.elapsed().as_secs() as f64;
        let aging_boost = (age_seconds / 60.0 * self.config.priority_aging_factor) as u8;

        let current_value = base_priority as u8;
        let aged_value = current_value.saturating_add(aging_boost).min(100);

        match aged_value {
            90..=100 => AdvancedPriority::Emergency,
            80..=89 => AdvancedPriority::RealTime,
            70..=79 => AdvancedPriority::Premium,
            60..=69 => AdvancedPriority::Critical,
            50..=59 => AdvancedPriority::High,
            40..=49 => AdvancedPriority::Normal,
            30..=39 => AdvancedPriority::Background,
            20..=29 => AdvancedPriority::Batch,
            _ => base_priority,
        }
    }

    /// Make room for a new request by dropping lower priority ones
    async fn make_room_for_request(&mut self, request: &AdvancedRequest) -> bool {
        // Try to drop from lowest priority queues first
        let priorities_to_check = [
            AdvancedPriority::Maintenance,
            AdvancedPriority::Analytics,
            AdvancedPriority::Batch,
            AdvancedPriority::Background,
        ];

        for priority in priorities_to_check {
            if priority < request.priority {
                if let Some(queue) = self.priority_queues.get(&priority) {
                    let mut q = queue.lock().await;
                    if !q.is_empty() {
                        q.pop_back(); // Drop oldest lower priority request
                        return true;
                    }
                }
            }
        }

        false
    }

    async fn add_to_priority_queue(&self, request: AdvancedRequest) -> Result<()> {
        if let Some(queue) = self.priority_queues.get(&request.priority) {
            queue.lock().await.push_back(request);
        }
        Ok(())
    }

    async fn add_to_deadline_queue(&self, request: AdvancedRequest) -> Result<()> {
        if let Some(deadline) = request.deadline {
            let deadline_score = deadline.duration_since(Instant::now()).as_secs_f64();
            let deadline_request = DeadlineRequest {
                request,
                deadline_score,
            };
            self.deadline_queue.lock().await.push(deadline_request);
        }
        Ok(())
    }

    async fn add_to_client_queue(&mut self, request: AdvancedRequest) -> Result<()> {
        let client_queue = self
            .client_queues
            .entry(request.client_id.clone())
            .or_insert_with(|| Arc::new(Mutex::new(VecDeque::new())));

        client_queue.lock().await.push_back(request);
        Ok(())
    }

    async fn dequeue_by_priority(&self) -> Option<AdvancedRequest> {
        for priority in [
            AdvancedPriority::Emergency,
            AdvancedPriority::RealTime,
            AdvancedPriority::Premium,
            AdvancedPriority::Critical,
            AdvancedPriority::High,
            AdvancedPriority::Normal,
            AdvancedPriority::Background,
            AdvancedPriority::Batch,
            AdvancedPriority::Analytics,
            AdvancedPriority::Maintenance,
        ] {
            if let Some(queue) = self.priority_queues.get(&priority) {
                if let Some(request) = queue.lock().await.pop_front() {
                    return Some(request);
                }
            }
        }
        None
    }

    async fn dequeue_deadline_aware(&self) -> Option<AdvancedRequest> {
        // Check deadline queue first
        if let Some(deadline_request) = self.deadline_queue.lock().await.pop() {
            return Some(deadline_request.request);
        }

        // Fallback to priority-based
        self.dequeue_by_priority().await
    }

    async fn dequeue_fair_share(&self) -> Option<AdvancedRequest> {
        // Simple round-robin among clients
        for queue in self.client_queues.values() {
            if let Some(request) = queue.lock().await.pop_front() {
                return Some(request);
            }
        }
        None
    }

    async fn dequeue_throughput_optimized(&self) -> Option<AdvancedRequest> {
        // Prefer larger batches and higher priority
        // For now, fall back to priority-based
        self.dequeue_by_priority().await
    }

    async fn dequeue_balanced_adaptive(&self) -> Option<AdvancedRequest> {
        // Mix of deadline-aware and priority-based
        if let Some(deadline_request) = self.deadline_queue.lock().await.peek() {
            if deadline_request.deadline_score < 10.0 {
                // Less than 10 seconds to deadline
                return self.deadline_queue.lock().await.pop().map(|dr| dr.request);
            }
        }

        self.dequeue_by_priority().await
    }

    async fn dequeue_conservative_high_priority(&self) -> Option<AdvancedRequest> {
        // Only serve high priority requests during high load
        for priority in [
            AdvancedPriority::Emergency,
            AdvancedPriority::RealTime,
            AdvancedPriority::Premium,
            AdvancedPriority::Critical,
            AdvancedPriority::High,
        ] {
            if let Some(queue) = self.priority_queues.get(&priority) {
                if let Some(request) = queue.lock().await.pop_front() {
                    return Some(request);
                }
            }
        }
        None
    }

    async fn dequeue_qos_aware(&self) -> Option<AdvancedRequest> {
        // Consider SLA violations and adjust scheduling
        // For now, fall back to balanced adaptive
        self.dequeue_balanced_adaptive().await
    }

    async fn get_total_queue_depth(&self) -> usize {
        let mut total = 0;
        for queue in self.priority_queues.values() {
            total += queue.lock().await.len();
        }
        total
    }

    async fn update_queue_depth_stats(&self) {
        let mut stats = self.stats.lock().await;
        for (priority, queue) in &self.priority_queues {
            let depth = queue.lock().await.len();
            stats.queue_depths.insert(*priority, depth);
        }
    }

    /// Get comprehensive queue statistics
    pub async fn get_stats(&self) -> AdvancedQueueStats {
        self.stats.lock().await.clone()
    }

    /// Update system load metrics
    pub async fn update_load_metrics(&self, cpu: f64, memory: f64, gpu: f64) {
        self.load_monitor.lock().await.update_metrics(cpu, memory, gpu);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scheduling_policy_priority() {
        let batch = RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests: vec![],
            created_at: Instant::now(),
            total_memory: 1000,
            max_sequence_length: 100,
            priority: Priority::High,
        };

        let throughput_score = SchedulingPolicy::Throughput.calculate_priority(&batch);
        let latency_score = SchedulingPolicy::Latency.calculate_priority(&batch);

        assert!(throughput_score >= 0.0);
        assert!(latency_score >= 0.0);
    }

    #[test]
    fn test_timeout_policy() {
        let policy = TimeoutPolicy::default();

        let critical_timeout = policy.calculate_timeout(Priority::Critical);
        let normal_timeout = policy.calculate_timeout(Priority::Normal);

        assert!(critical_timeout < normal_timeout);
    }

    #[tokio::test]
    async fn test_multi_queue_scheduler() {
        let scheduler = MultiQueueScheduler::new();

        let batch = RequestBatch {
            id: uuid::Uuid::new_v4(),
            requests: vec![],
            created_at: Instant::now(),
            total_memory: 1000,
            max_sequence_length: 100,
            priority: Priority::High,
        };

        scheduler.add_batch(batch).await.unwrap();

        let retrieved = scheduler.get_next_batch().await;
        assert!(retrieved.is_some());
    }
}
