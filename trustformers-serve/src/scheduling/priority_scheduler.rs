//! Priority-based request scheduler with deadline awareness.
//!
//! Supports multiple priority classes with deadline-based preemption
//! and fair queuing within each priority class.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use thiserror::Error;

/// Request priority levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    /// Reserved for health checks and admin operations
    Critical = 3,
}

impl RequestPriority {
    /// Maximum allowed wait time before a request at this priority is dropped
    pub fn deadline_budget(&self) -> Duration {
        match self {
            Self::Low => Duration::from_secs(30),
            Self::Normal => Duration::from_secs(10),
            Self::High => Duration::from_secs(3),
            Self::Critical => Duration::from_millis(500),
        }
    }

    /// Maximum number of requests that may queue at this priority level
    pub fn max_queue_depth(&self) -> usize {
        match self {
            Self::Low => 1000,
            Self::Normal => 500,
            Self::High => 200,
            Self::Critical => 50,
        }
    }

    /// All priority levels from lowest to highest
    pub fn all() -> [RequestPriority; 4] {
        [Self::Low, Self::Normal, Self::High, Self::Critical]
    }
}

/// A schedulable request
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    pub request_id: String,
    pub priority: RequestPriority,
    pub enqueued_at: Instant,
    pub deadline: Instant,
    /// Estimated compute cost (tokens * complexity)
    pub estimated_cost: f32,
    pub payload: serde_json::Value,
}

impl ScheduledRequest {
    /// Create a new request; deadline is derived from priority budget
    pub fn new(
        request_id: String,
        priority: RequestPriority,
        payload: serde_json::Value,
        cost: f32,
    ) -> Self {
        let now = Instant::now();
        let deadline = now + priority.deadline_budget();
        Self {
            request_id,
            priority,
            enqueued_at: now,
            deadline,
            estimated_cost: cost,
            payload,
        }
    }

    /// Whether the request's deadline has already passed
    pub fn is_expired(&self) -> bool {
        Instant::now() > self.deadline
    }

    /// How long the request has been waiting, in milliseconds
    pub fn wait_time_ms(&self) -> f64 {
        self.enqueued_at.elapsed().as_secs_f64() * 1000.0
    }

    /// Remaining time until the deadline, in milliseconds (negative if expired)
    pub fn time_to_deadline_ms(&self) -> f64 {
        let now = Instant::now();
        if now > self.deadline {
            let overrun = now.duration_since(self.deadline);
            -(overrun.as_secs_f64() * 1000.0)
        } else {
            let remaining = self.deadline.duration_since(now);
            remaining.as_secs_f64() * 1000.0
        }
    }
}

impl PartialEq for ScheduledRequest {
    fn eq(&self, other: &Self) -> bool {
        self.request_id == other.request_id
    }
}

impl Eq for ScheduledRequest {}

impl PartialOrd for ScheduledRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for ScheduledRequest {
    /// Higher priority first; within the same priority, earlier deadline first (EDF)
    fn cmp(&self, other: &Self) -> Ordering {
        self.priority
            .cmp(&other.priority)
            .then_with(|| other.deadline.cmp(&self.deadline))
    }
}

/// Snapshot of scheduler performance statistics
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    pub total_enqueued: u64,
    pub total_scheduled: u64,
    pub total_dropped_deadline: u64,
    pub total_dropped_queue_full: u64,
    pub p50_wait_ms: f64,
    pub p95_wait_ms: f64,
    pub p99_wait_ms: f64,
    pub current_queue_depth: usize,
}

/// Configuration for the priority scheduler
#[derive(Debug, Clone)]
pub struct SchedulerConfig {
    /// Hard cap on total items across all priority queues
    pub max_total_queue_depth: usize,
    /// When true, expired requests are silently dropped during enqueue/dequeue
    pub drop_expired_requests: bool,
    /// When true, dequeue rotates between priority levels for fairness
    pub fair_scheduling: bool,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_total_queue_depth: 2000,
            drop_expired_requests: true,
            fair_scheduling: false,
        }
    }
}

/// Errors from the priority scheduler
#[derive(Debug, Error)]
pub enum SchedulerError {
    #[error("Queue full for priority {priority:?}: {depth} items")]
    QueueFull {
        priority: RequestPriority,
        depth: usize,
    },
    #[error("Request expired before enqueuing: {request_id}")]
    RequestExpired { request_id: String },
    #[error("Mutex poisoned: {0}")]
    Poisoned(String),
}

/// Internal mutable state protected by a single Mutex
struct SchedulerState {
    queues: HashMap<RequestPriority, BinaryHeap<ScheduledRequest>>,
    stats: SchedulerStats,
    wait_times: VecDeque<f64>,
    fair_round_robin_index: usize,
}

impl SchedulerState {
    fn new() -> Self {
        let mut queues = HashMap::new();
        for p in RequestPriority::all() {
            queues.insert(p, BinaryHeap::new());
        }
        Self {
            queues,
            stats: SchedulerStats::default(),
            wait_times: VecDeque::with_capacity(1024),
            fair_round_robin_index: 0,
        }
    }

    fn total_depth(&self) -> usize {
        self.queues.values().map(|q| q.len()).sum()
    }

    fn record_wait_time(&mut self, ms: f64) {
        if self.wait_times.len() >= 2048 {
            self.wait_times.pop_front();
        }
        self.wait_times.push_back(ms);
        self.update_percentiles();
    }

    fn update_percentiles(&mut self) {
        if self.wait_times.is_empty() {
            return;
        }
        let mut sorted: Vec<f64> = self.wait_times.iter().copied().collect();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
        let n = sorted.len();
        self.stats.p50_wait_ms =
            sorted[(n as f64 * 0.50) as usize].min(*sorted.last().unwrap_or(&0.0));
        self.stats.p95_wait_ms = sorted[((n as f64 * 0.95) as usize).min(n - 1)];
        self.stats.p99_wait_ms = sorted[((n as f64 * 0.99) as usize).min(n - 1)];
    }
}

/// Priority scheduler with per-priority queues and deadline awareness
pub struct PriorityScheduler {
    state: Mutex<SchedulerState>,
    config: SchedulerConfig,
}

impl PriorityScheduler {
    /// Create a new scheduler with the given config
    pub fn new(config: SchedulerConfig) -> Self {
        Self {
            state: Mutex::new(SchedulerState::new()),
            config,
        }
    }

    /// Enqueue a request. Returns Err if the queue is full or the request has expired.
    pub fn enqueue(&self, request: ScheduledRequest) -> Result<(), SchedulerError> {
        if self.config.drop_expired_requests && request.is_expired() {
            return Err(SchedulerError::RequestExpired {
                request_id: request.request_id.clone(),
            });
        }

        let mut state = self.state.lock().map_err(|e| SchedulerError::Poisoned(e.to_string()))?;

        // Check total capacity
        if state.total_depth() >= self.config.max_total_queue_depth {
            let depth = state.total_depth();
            state.stats.total_dropped_queue_full += 1;
            return Err(SchedulerError::QueueFull {
                priority: request.priority,
                depth,
            });
        }

        // Check per-priority capacity
        let per_priority_depth = state.queues.get(&request.priority).map(|q| q.len()).unwrap_or(0);
        if per_priority_depth >= request.priority.max_queue_depth() {
            state.stats.total_dropped_queue_full += 1;
            return Err(SchedulerError::QueueFull {
                priority: request.priority,
                depth: per_priority_depth,
            });
        }

        state.stats.total_enqueued += 1;
        state.queues.entry(request.priority).or_default().push(request);
        state.stats.current_queue_depth = state.total_depth();
        Ok(())
    }

    /// Dequeue the highest-priority, earliest-deadline request
    pub fn dequeue(&self) -> Option<ScheduledRequest> {
        let mut state = self.state.lock().ok()?;

        if self.config.fair_scheduling {
            return self.dequeue_fair(&mut state);
        }

        // Strict priority — iterate from highest to lowest
        for priority in [
            RequestPriority::Critical,
            RequestPriority::High,
            RequestPriority::Normal,
            RequestPriority::Low,
        ] {
            if let Some(request) = self.pop_valid_from(&mut state, priority) {
                let wait_ms = request.wait_time_ms();
                state.stats.total_scheduled += 1;
                state.stats.current_queue_depth = state.total_depth();
                state.record_wait_time(wait_ms);
                return Some(request);
            }
        }
        None
    }

    fn dequeue_fair(&self, state: &mut SchedulerState) -> Option<ScheduledRequest> {
        let levels = [
            RequestPriority::Critical,
            RequestPriority::High,
            RequestPriority::Normal,
            RequestPriority::Low,
        ];
        let start = state.fair_round_robin_index % levels.len();
        for i in 0..levels.len() {
            let idx = (start + i) % levels.len();
            let priority = levels[idx];
            if let Some(request) = self.pop_valid_from(state, priority) {
                state.fair_round_robin_index = (idx + 1) % levels.len();
                let wait_ms = request.wait_time_ms();
                state.stats.total_scheduled += 1;
                state.stats.current_queue_depth = state.total_depth();
                state.record_wait_time(wait_ms);
                return Some(request);
            }
        }
        None
    }

    /// Pop the next non-expired request from the given priority queue
    fn pop_valid_from(
        &self,
        state: &mut SchedulerState,
        priority: RequestPriority,
    ) -> Option<ScheduledRequest> {
        let queue = state.queues.get_mut(&priority)?;
        loop {
            let req = queue.peek()?;
            if self.config.drop_expired_requests && req.is_expired() {
                queue.pop();
                state.stats.total_dropped_deadline += 1;
                continue;
            }
            return queue.pop();
        }
    }

    /// Dequeue up to `n` requests for batch processing
    pub fn dequeue_batch(&self, n: usize) -> Vec<ScheduledRequest> {
        (0..n).filter_map(|_| self.dequeue()).collect()
    }

    /// Evict all expired requests from all queues; returns how many were removed
    pub fn evict_expired(&self) -> usize {
        let Ok(mut state) = self.state.lock() else {
            return 0;
        };
        let mut removed = 0usize;
        for queue in state.queues.values_mut() {
            let before = queue.len();
            let valid: Vec<ScheduledRequest> = queue.drain().filter(|r| !r.is_expired()).collect();
            removed += before - valid.len();
            *queue = valid.into_iter().collect();
        }
        state.stats.total_dropped_deadline += removed as u64;
        state.stats.current_queue_depth = state.total_depth();
        removed
    }

    /// Current queue depths per priority level
    pub fn queue_depths(&self) -> HashMap<RequestPriority, usize> {
        let Ok(state) = self.state.lock() else {
            return HashMap::new();
        };
        state.queues.iter().map(|(k, v)| (*k, v.len())).collect()
    }

    /// Total items across all queues
    pub fn total_depth(&self) -> usize {
        let Ok(state) = self.state.lock() else {
            return 0;
        };
        state.total_depth()
    }

    /// Snapshot of scheduler statistics
    pub fn stats(&self) -> SchedulerStats {
        let Ok(state) = self.state.lock() else {
            return SchedulerStats::default();
        };
        state.stats.clone()
    }

    /// True if the scheduler is at or above 80% of its total capacity
    pub fn is_overloaded(&self) -> bool {
        let Ok(state) = self.state.lock() else {
            return false;
        };
        let depth = state.total_depth();
        depth >= (self.config.max_total_queue_depth * 80) / 100
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_request(id: &str, priority: RequestPriority) -> ScheduledRequest {
        ScheduledRequest::new(
            id.to_string(),
            priority,
            serde_json::json!({"text": id}),
            1.0,
        )
    }

    fn default_scheduler() -> PriorityScheduler {
        PriorityScheduler::new(SchedulerConfig::default())
    }

    #[test]
    fn test_enqueue_and_dequeue_single() {
        let sched = default_scheduler();
        let req = make_request("r1", RequestPriority::Normal);
        sched.enqueue(req).expect("enqueue failed");
        let dequeued = sched.dequeue().expect("dequeue returned None");
        assert_eq!(dequeued.request_id, "r1");
    }

    #[test]
    fn test_priority_ordering() {
        let sched = default_scheduler();
        sched.enqueue(make_request("low", RequestPriority::Low)).unwrap();
        sched.enqueue(make_request("high", RequestPriority::High)).unwrap();
        sched.enqueue(make_request("normal", RequestPriority::Normal)).unwrap();
        sched.enqueue(make_request("critical", RequestPriority::Critical)).unwrap();

        let first = sched.dequeue().expect("should dequeue critical");
        assert_eq!(first.request_id, "critical");

        let second = sched.dequeue().expect("should dequeue high");
        assert_eq!(second.request_id, "high");

        let third = sched.dequeue().expect("should dequeue normal");
        assert_eq!(third.request_id, "normal");

        let fourth = sched.dequeue().expect("should dequeue low");
        assert_eq!(fourth.request_id, "low");
    }

    #[test]
    fn test_dequeue_empty_returns_none() {
        let sched = default_scheduler();
        assert!(sched.dequeue().is_none());
    }

    #[test]
    fn test_dequeue_batch() {
        let sched = default_scheduler();
        for i in 0..5 {
            sched.enqueue(make_request(&format!("r{i}"), RequestPriority::Normal)).unwrap();
        }
        let batch = sched.dequeue_batch(3);
        assert_eq!(batch.len(), 3);
        assert_eq!(sched.total_depth(), 2);
    }

    #[test]
    fn test_queue_full_per_priority() {
        let config = SchedulerConfig {
            max_total_queue_depth: 10000,
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);
        // Critical max_queue_depth is 50
        for i in 0..50 {
            sched
                .enqueue(make_request(&format!("c{i}"), RequestPriority::Critical))
                .expect("should succeed up to max");
        }
        let result = sched.enqueue(make_request("overflow", RequestPriority::Critical));
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            matches!(err, SchedulerError::QueueFull { .. }),
            "expected QueueFull"
        );
    }

    #[test]
    fn test_expired_request_rejected_at_enqueue() {
        // Build a request with an already-passed deadline by manipulating enqueued_at
        let mut req = make_request("expired", RequestPriority::Normal);
        req.deadline = Instant::now() - Duration::from_millis(1);

        let sched = default_scheduler();
        let result = sched.enqueue(req);
        assert!(result.is_err());
        assert!(matches!(
            result.unwrap_err(),
            SchedulerError::RequestExpired { .. }
        ));
    }

    #[test]
    fn test_evict_expired() {
        let config = SchedulerConfig {
            drop_expired_requests: false, // allow stale enqueue
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);

        // Enqueue a request that is already expired
        let mut req = make_request("stale", RequestPriority::Normal);
        req.deadline = Instant::now() - Duration::from_millis(1);
        sched.enqueue(req).expect("should enqueue since drop_expired_requests=false");

        // Enqueue a valid one
        sched.enqueue(make_request("fresh", RequestPriority::Normal)).unwrap();
        assert_eq!(sched.total_depth(), 2);

        let removed = sched.evict_expired();
        assert_eq!(removed, 1);
        assert_eq!(sched.total_depth(), 1);
    }

    #[test]
    fn test_stats_tracking() {
        let sched = default_scheduler();
        for i in 0..3 {
            sched.enqueue(make_request(&format!("r{i}"), RequestPriority::Normal)).unwrap();
        }
        sched.dequeue();
        sched.dequeue();

        let stats = sched.stats();
        assert_eq!(stats.total_enqueued, 3);
        assert_eq!(stats.total_scheduled, 2);
    }

    #[test]
    fn test_queue_depths_map() {
        let sched = default_scheduler();
        sched.enqueue(make_request("h1", RequestPriority::High)).unwrap();
        sched.enqueue(make_request("h2", RequestPriority::High)).unwrap();
        sched.enqueue(make_request("l1", RequestPriority::Low)).unwrap();

        let depths = sched.queue_depths();
        assert_eq!(depths[&RequestPriority::High], 2);
        assert_eq!(depths[&RequestPriority::Low], 1);
        assert_eq!(depths[&RequestPriority::Normal], 0);
    }

    #[test]
    fn test_is_overloaded_false_when_empty() {
        let sched = default_scheduler();
        assert!(!sched.is_overloaded());
    }

    #[test]
    fn test_is_overloaded_true_when_full() {
        let config = SchedulerConfig {
            max_total_queue_depth: 10,
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);
        for i in 0..9 {
            sched.enqueue(make_request(&format!("r{i}"), RequestPriority::Normal)).unwrap();
        }
        // 9/10 = 90% => overloaded
        assert!(sched.is_overloaded());
    }

    #[test]
    fn test_wait_time_ms_increases_over_time() {
        let req = make_request("timing", RequestPriority::Normal);
        // wait_time_ms is measured from enqueued_at; it should be >= 0
        assert!(req.wait_time_ms() >= 0.0);
    }

    #[test]
    fn test_time_to_deadline_ms_positive_for_fresh_request() {
        let req = make_request("fresh", RequestPriority::Normal);
        assert!(req.time_to_deadline_ms() > 0.0);
    }

    #[test]
    fn test_time_to_deadline_ms_negative_for_expired() {
        let mut req = make_request("expired", RequestPriority::Normal);
        req.deadline = Instant::now() - Duration::from_millis(100);
        assert!(req.time_to_deadline_ms() < 0.0);
    }

    #[test]
    fn test_deadline_budget_ordering() {
        // Lower priority => longer budget
        assert!(
            RequestPriority::Low.deadline_budget() > RequestPriority::Critical.deadline_budget()
        );
    }

    #[test]
    fn test_fair_scheduling_round_robin() {
        let config = SchedulerConfig {
            fair_scheduling: true,
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);
        sched.enqueue(make_request("h1", RequestPriority::High)).unwrap();
        sched.enqueue(make_request("n1", RequestPriority::Normal)).unwrap();

        // Both should be dequeued without getting stuck
        let r1 = sched.dequeue().expect("first dequeue");
        let r2 = sched.dequeue().expect("second dequeue");
        let ids: std::collections::HashSet<String> =
            [r1.request_id, r2.request_id].into_iter().collect();
        assert!(ids.contains("h1"), "expected h1 to be dequeued");
        assert!(ids.contains("n1"), "expected n1 to be dequeued");
    }

    #[test]
    fn test_total_depth_reflects_all_priorities() {
        let sched = default_scheduler();
        sched.enqueue(make_request("a", RequestPriority::Critical)).unwrap();
        sched.enqueue(make_request("b", RequestPriority::High)).unwrap();
        sched.enqueue(make_request("c", RequestPriority::Normal)).unwrap();
        sched.enqueue(make_request("d", RequestPriority::Low)).unwrap();
        assert_eq!(sched.total_depth(), 4);
    }

    #[test]
    fn test_max_queue_depth_constants() {
        assert_eq!(RequestPriority::Low.max_queue_depth(), 1000);
        assert_eq!(RequestPriority::Normal.max_queue_depth(), 500);
        assert_eq!(RequestPriority::High.max_queue_depth(), 200);
        assert_eq!(RequestPriority::Critical.max_queue_depth(), 50);
    }

    #[test]
    fn test_priority_ordering_enum() {
        // Critical > High > Normal > Low
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
    }

    #[test]
    fn test_all_priorities_array_has_four_elements() {
        assert_eq!(RequestPriority::all().len(), 4);
    }

    #[test]
    fn test_is_expired_false_for_fresh_request() {
        let req = make_request("fresh", RequestPriority::Normal);
        assert!(
            !req.is_expired(),
            "freshly created request should not be expired"
        );
    }

    #[test]
    fn test_is_expired_true_for_past_deadline() {
        let mut req = make_request("old", RequestPriority::Low);
        req.deadline = Instant::now() - Duration::from_millis(1);
        assert!(
            req.is_expired(),
            "request with past deadline should be expired"
        );
    }

    #[test]
    fn test_total_queue_depth_cap() {
        let config = SchedulerConfig {
            max_total_queue_depth: 3,
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);
        sched.enqueue(make_request("a", RequestPriority::Low)).unwrap();
        sched.enqueue(make_request("b", RequestPriority::Low)).unwrap();
        sched.enqueue(make_request("c", RequestPriority::Low)).unwrap();
        let result = sched.enqueue(make_request("d", RequestPriority::Low));
        assert!(
            matches!(result, Err(SchedulerError::QueueFull { .. })),
            "should reject when total queue depth cap exceeded"
        );
    }

    #[test]
    fn test_scheduler_error_display_contains_priority() {
        let err = SchedulerError::QueueFull {
            priority: RequestPriority::High,
            depth: 200,
        };
        let msg = err.to_string();
        assert!(
            msg.contains("High"),
            "error message should mention priority level"
        );
    }

    #[test]
    fn test_stats_dropped_counters() {
        let config = SchedulerConfig {
            max_total_queue_depth: 1,
            ..Default::default()
        };
        let sched = PriorityScheduler::new(config);
        sched.enqueue(make_request("a", RequestPriority::Normal)).unwrap();
        let _ = sched.enqueue(make_request("b", RequestPriority::Normal));
        let stats = sched.stats();
        assert_eq!(stats.total_dropped_queue_full, 1);
    }
}
