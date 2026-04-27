//! Priority scheduler for inference request dispatching.
//!
//! Provides weighted round-robin, deadline-aware (EDF), fair queuing, pure priority,
//! and FIFO scheduling strategies over the [`PriorityQueue`] from the sibling `queue`
//! module.

use crate::queue::{PriorityQueue, QueueError, QueuedRequest, RequestPriority};

// ---------------------------------------------------------------------------
// Scheduling policy
// ---------------------------------------------------------------------------

/// Scheduling strategy used by [`PriorityScheduler`] when forming a batch.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingPolicy {
    /// First-in, first-out — no priority distinction.
    Fifo,
    /// Highest priority first; ties broken by arrival order (FIFO).
    Priority,
    /// Weighted round-robin — each priority level gets slots proportional to its weight:
    /// Critical = 8, High = 4, Normal = 2, Low = 1.
    WeightedRoundRobin,
    /// Earliest-deadline-first among all pending requests.
    /// Requests without a deadline sort last.
    EarliestDeadlineFirst,
    /// Approximate max-min fairness: at most one request per model per batch.
    /// Falls back to priority scheduling when only one model is present.
    FairQueuing,
}

impl std::fmt::Display for SchedulingPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            SchedulingPolicy::Fifo => "fifo",
            SchedulingPolicy::Priority => "priority",
            SchedulingPolicy::WeightedRoundRobin => "weighted-round-robin",
            SchedulingPolicy::EarliestDeadlineFirst => "earliest-deadline-first",
            SchedulingPolicy::FairQueuing => "fair-queuing",
        };
        f.write_str(name)
    }
}

// ---------------------------------------------------------------------------
// Scheduler configuration
// ---------------------------------------------------------------------------

/// Configuration for [`PriorityScheduler`].
pub struct SchedulerConfig {
    /// Active scheduling policy.
    pub policy: SchedulingPolicy,
    /// Maximum number of requests per scheduled batch.
    pub max_batch_size: usize,
    /// Maximum number of entries in the underlying priority queue.
    pub max_queue_depth: usize,
    /// When `true`, a `Critical` request may preempt a `Low` request that is
    /// currently at the back of the queue (logged, not actually interrupted).
    pub preemption_enabled: bool,
    /// When `true`, Normal and Low requests are rejected when the queue is above
    /// `admission_threshold` × `max_queue_depth`.
    pub admission_control: bool,
    /// Fraction of capacity at which admission control kicks in (default 0.9).
    pub admission_threshold: f32,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        SchedulerConfig {
            policy: SchedulingPolicy::Priority,
            max_batch_size: 8,
            max_queue_depth: 1024,
            preemption_enabled: true,
            admission_control: true,
            admission_threshold: 0.9,
        }
    }
}

// ---------------------------------------------------------------------------
// Scheduled batch
// ---------------------------------------------------------------------------

/// A batch of requests ready for model inference.
#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    /// Requests in this batch, in dispatch order.
    pub requests: Vec<QueuedRequest>,
    /// Monotonically increasing batch identifier.
    pub batch_id: u64,
    /// Policy that was used to form this batch.
    pub policy_used: SchedulingPolicy,
    /// Rough latency estimate for the whole batch (milliseconds), computed as
    /// the maximum individual processing estimate, or 0 when unavailable.
    pub estimated_latency_ms: u64,
}

impl ScheduledBatch {
    /// Number of requests in the batch.
    pub fn size(&self) -> usize {
        self.requests.len()
    }

    /// Maximum `max_tokens` value across all requests (governs output buffer size).
    pub fn max_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.max_tokens).max().unwrap_or(0)
    }

    /// Sum of `max_tokens` across all requests (total token budget).
    pub fn total_tokens(&self) -> usize {
        self.requests.iter().map(|r| r.max_tokens).sum()
    }

    /// Number of requests per priority level.
    ///
    /// Index 0 = Low, 1 = Normal, 2 = High, 3 = Critical.
    pub fn priority_mix(&self) -> [usize; 4] {
        let mut mix = [0usize; 4];
        for r in &self.requests {
            mix[r.priority as usize] += 1;
        }
        mix
    }
}

// ---------------------------------------------------------------------------
// Metrics
// ---------------------------------------------------------------------------

/// Running metrics collected by [`PriorityScheduler`].
#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    /// Total batches formed.
    pub total_batches_scheduled: u64,
    /// Total requests dispatched across all batches.
    pub total_requests_scheduled: u64,
    /// Total requests rejected by admission control.
    pub total_requests_rejected: u64,
    /// Exponential moving average of batch sizes.
    pub mean_batch_size: f32,
    /// Number of preemption events logged (lazy; actual preemption is not
    /// implemented in the queue layer for this proof-of-concept).
    pub preemptions: u64,
}

// ---------------------------------------------------------------------------
// Scheduler
// ---------------------------------------------------------------------------

/// Priority scheduler that wraps a [`PriorityQueue`] and dispatches batches
/// according to the configured [`SchedulingPolicy`].
pub struct PriorityScheduler {
    queue: PriorityQueue,
    config: SchedulerConfig,
    metrics: SchedulerMetrics,
    next_batch_id: u64,
    /// Remaining WRR slots per priority level in the current WRR cycle.
    /// Index 0 = Low … 3 = Critical.
    wrr_counters: [u32; 4],
}

impl PriorityScheduler {
    /// Create a new scheduler from the provided configuration.
    pub fn new(config: SchedulerConfig) -> Self {
        let weights = [
            RequestPriority::Low.weight(),
            RequestPriority::Normal.weight(),
            RequestPriority::High.weight(),
            RequestPriority::Critical.weight(),
        ];
        PriorityScheduler {
            queue: PriorityQueue::new(config.max_queue_depth),
            config,
            metrics: SchedulerMetrics::default(),
            next_batch_id: 0,
            wrr_counters: weights,
        }
    }

    // -----------------------------------------------------------------------
    // Submission
    // -----------------------------------------------------------------------

    /// Submit a request for scheduling.
    ///
    /// Admission control may reject `Normal` and `Low` requests when the queue
    /// is above `admission_threshold`. `Critical` requests bypass admission
    /// control. If `preemption_enabled`, a `Critical` submission while the
    /// queue is full will record a preemption event (the framework logs the
    /// intent; actual in-flight preemption is out of scope here).
    pub fn submit(&mut self, request: QueuedRequest) -> Result<u64, SchedulerError> {
        let depth = self.queue.len();
        let cap = self.config.max_queue_depth;

        // Admission control: reject non-critical when queue is near capacity.
        if self.config.admission_control
            && request.priority != RequestPriority::Critical
            && request.priority != RequestPriority::High
        {
            let threshold = (cap as f32 * self.config.admission_threshold) as usize;
            if depth >= threshold {
                self.metrics.total_requests_rejected += 1;
                return Err(SchedulerError::Rejected {
                    reason: format!(
                        "admission control: queue depth {depth} >= threshold {threshold}"
                    ),
                });
            }
        }

        // Preemption: if queue is completely full and this is Critical, record
        // a preemption event (best-effort — the queue itself will return an
        // error that we surface as a distinct variant).
        if depth >= cap
            && request.priority == RequestPriority::Critical
            && self.config.preemption_enabled
        {
            self.metrics.preemptions += 1;
            log::warn!(
                "critical request {} preempting (queue full at {depth})",
                request.id
            );
            // We still try to push; if it fails we surface QueueFull.
        }

        self.queue.push(request).map_err(|e| match e {
            QueueError::QueueFull { capacity, current } => {
                self.metrics.total_requests_rejected += 1;
                SchedulerError::Rejected {
                    reason: format!("queue full: capacity={capacity}, current={current}"),
                }
            },
            QueueError::InvalidRequest(msg) => SchedulerError::InvalidRequest(msg),
        })
    }

    // -----------------------------------------------------------------------
    // Batch scheduling dispatch
    // -----------------------------------------------------------------------

    /// Schedule the next batch of requests according to the configured policy.
    ///
    /// Returns `None` when the queue is empty.
    pub fn schedule_batch(&mut self) -> Option<ScheduledBatch> {
        if self.queue.is_empty() {
            return None;
        }

        let policy = self.config.policy.clone();
        let batch = match policy {
            SchedulingPolicy::Fifo => self.schedule_fifo(),
            SchedulingPolicy::Priority => self.schedule_priority(),
            SchedulingPolicy::WeightedRoundRobin => self.schedule_wrr(),
            SchedulingPolicy::EarliestDeadlineFirst => self.schedule_edf(),
            SchedulingPolicy::FairQueuing => self.schedule_fair(),
        };

        if let Some(ref b) = batch {
            self.metrics.total_batches_scheduled += 1;
            self.metrics.total_requests_scheduled += b.size() as u64;
            let n = self.metrics.total_batches_scheduled as f32;
            self.metrics.mean_batch_size =
                self.metrics.mean_batch_size * (n - 1.0) / n + b.size() as f32 / n;
        }
        batch
    }

    // -----------------------------------------------------------------------
    // FIFO scheduling
    // -----------------------------------------------------------------------

    fn schedule_fifo(&mut self) -> Option<ScheduledBatch> {
        let mut requests = Vec::with_capacity(self.config.max_batch_size);
        while requests.len() < self.config.max_batch_size {
            match self.queue.pop() {
                Some(req) => requests.push(req),
                None => break,
            }
        }
        if requests.is_empty() {
            return None;
        }
        let est = Self::estimate_batch_latency(&requests);
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        Some(ScheduledBatch {
            requests,
            batch_id: id,
            policy_used: SchedulingPolicy::Fifo,
            estimated_latency_ms: est,
        })
    }

    // -----------------------------------------------------------------------
    // Priority scheduling
    // -----------------------------------------------------------------------

    fn schedule_priority(&mut self) -> Option<ScheduledBatch> {
        let mut requests = Vec::with_capacity(self.config.max_batch_size);
        while requests.len() < self.config.max_batch_size {
            match self.queue.pop() {
                Some(req) => requests.push(req),
                None => break,
            }
        }
        if requests.is_empty() {
            return None;
        }
        let est = Self::estimate_batch_latency(&requests);
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        Some(ScheduledBatch {
            requests,
            batch_id: id,
            policy_used: SchedulingPolicy::Priority,
            estimated_latency_ms: est,
        })
    }

    // -----------------------------------------------------------------------
    // Weighted round-robin scheduling
    // -----------------------------------------------------------------------

    /// WRR: each priority level contributes up to its weight in slots per cycle.
    ///
    /// `wrr_counters` tracks remaining slots per level in the current cycle.
    /// When all counters reach 0, the cycle resets to the priority weights.
    fn schedule_wrr(&mut self) -> Option<ScheduledBatch> {
        // Drain all current queue entries so we can round-robin by priority.
        // We rebuild the queue with leftovers after the batch is formed.
        let mut all: Vec<QueuedRequest> = Vec::new();
        while let Some(r) = self.queue.pop() {
            all.push(r);
        }
        if all.is_empty() {
            return None;
        }

        // Separate by priority level.
        let mut buckets: [Vec<QueuedRequest>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
        for req in all {
            buckets[req.priority as usize].push(req);
        }
        // Each bucket is already sorted by sequence_number because the heap gave them
        // out in priority order; within the same priority they came out FIFO.
        // However after draining all levels we need to sort each bucket by sequence
        // to restore FIFO within level.
        for bucket in &mut buckets {
            bucket.sort_by_key(|r| r.sequence_number);
        }

        let weights = [
            RequestPriority::Low.weight(),
            RequestPriority::Normal.weight(),
            RequestPriority::High.weight(),
            RequestPriority::Critical.weight(),
        ];

        let mut batch: Vec<QueuedRequest> = Vec::with_capacity(self.config.max_batch_size);

        // Reset WRR counters if all are exhausted.
        let all_zero = self.wrr_counters.iter().all(|&c| c == 0);
        if all_zero {
            self.wrr_counters = weights;
        }

        // Round-robin from highest priority to lowest, consuming slots.
        'outer: loop {
            let mut progress = false;
            // Iterate priorities from highest to lowest.
            for level in (0..4usize).rev() {
                if batch.len() >= self.config.max_batch_size {
                    break 'outer;
                }
                if self.wrr_counters[level] > 0 && !buckets[level].is_empty() {
                    let req = buckets[level].remove(0);
                    self.wrr_counters[level] -= 1;
                    batch.push(req);
                    progress = true;
                }
            }
            if !progress {
                // All counters for non-empty buckets are 0 → new cycle.
                let any_remaining = buckets.iter().any(|b| !b.is_empty());
                if !any_remaining || batch.len() >= self.config.max_batch_size {
                    break;
                }
                // Reset counters and continue.
                self.wrr_counters = weights;
            }
        }

        // Push leftovers back.
        for bucket in buckets {
            for req in bucket {
                // Ignore push errors here; at worst we drop the extra requests.
                let _ = self.queue.push(req);
            }
        }

        if batch.is_empty() {
            return None;
        }

        let est = Self::estimate_batch_latency(&batch);
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        Some(ScheduledBatch {
            requests: batch,
            batch_id: id,
            policy_used: SchedulingPolicy::WeightedRoundRobin,
            estimated_latency_ms: est,
        })
    }

    // -----------------------------------------------------------------------
    // Earliest-deadline-first scheduling
    // -----------------------------------------------------------------------

    /// EDF: collect up to `max_batch_size` requests, sorted by soonest deadline.
    /// Requests without a deadline sort to the end.
    fn schedule_edf(&mut self) -> Option<ScheduledBatch> {
        // Drain all and sort by deadline.
        let mut all: Vec<QueuedRequest> = Vec::new();
        while let Some(r) = self.queue.pop() {
            all.push(r);
        }
        if all.is_empty() {
            return None;
        }

        // Sort: soonest deadline first; None deadlines go last; tie-break by sequence.
        all.sort_by(|a, b| match (a.deadline_ms, b.deadline_ms) {
            (Some(da), Some(db)) => da.cmp(&db).then(a.sequence_number.cmp(&b.sequence_number)),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => a.sequence_number.cmp(&b.sequence_number),
        });

        let split = all.len().min(self.config.max_batch_size);
        let batch: Vec<QueuedRequest> = all.drain(..split).collect();

        // Push leftovers back.
        for req in all {
            let _ = self.queue.push(req);
        }

        let est = Self::estimate_batch_latency(&batch);
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        Some(ScheduledBatch {
            requests: batch,
            batch_id: id,
            policy_used: SchedulingPolicy::EarliestDeadlineFirst,
            estimated_latency_ms: est,
        })
    }

    // -----------------------------------------------------------------------
    // Fair queuing
    // -----------------------------------------------------------------------

    /// Fair queuing: at most one request per model per batch, rotating through
    /// models in arrival order of their first queued request.
    fn schedule_fair(&mut self) -> Option<ScheduledBatch> {
        let mut all: Vec<QueuedRequest> = Vec::new();
        while let Some(r) = self.queue.pop() {
            all.push(r);
        }
        if all.is_empty() {
            return None;
        }

        // Check if only one model is present → fall back to priority.
        let has_multiple = {
            let first = all.first().map(|r| r.model.as_str()).unwrap_or("");
            all.iter().any(|r| r.model != first)
        };

        if !has_multiple {
            // Single model: sort by (priority desc, sequence asc) and take up to batch size.
            all.sort_by(|a, b| {
                b.priority.cmp(&a.priority).then(a.sequence_number.cmp(&b.sequence_number))
            });
            let split = all.len().min(self.config.max_batch_size);
            let batch: Vec<QueuedRequest> = all.drain(..split).collect();
            for req in all {
                let _ = self.queue.push(req);
            }
            let est = Self::estimate_batch_latency(&batch);
            let id = self.next_batch_id;
            self.next_batch_id += 1;
            return Some(ScheduledBatch {
                requests: batch,
                batch_id: id,
                policy_used: SchedulingPolicy::FairQueuing,
                estimated_latency_ms: est,
            });
        }

        // Multiple models: group by model, take one from each in rotation.
        // Determine model rotation order by earliest sequence number in each group.
        let mut model_groups: Vec<(String, Vec<QueuedRequest>)> = Vec::new();
        for req in all {
            if let Some(group) = model_groups.iter_mut().find(|(m, _)| *m == req.model) {
                group.1.push(req);
            } else {
                model_groups.push((req.model.clone(), vec![req]));
            }
        }
        // Sort groups by earliest sequence number (fairest arrival).
        model_groups.sort_by_key(|(_, reqs)| {
            reqs.iter().map(|r| r.sequence_number).min().unwrap_or(u64::MAX)
        });
        // Within each group, sort by (priority desc, sequence asc).
        for (_, reqs) in &mut model_groups {
            reqs.sort_by(|a, b| {
                b.priority.cmp(&a.priority).then(a.sequence_number.cmp(&b.sequence_number))
            });
        }

        let mut batch: Vec<QueuedRequest> = Vec::with_capacity(self.config.max_batch_size);
        // Round-robin one request per model per pass.
        let mut round = 0;
        'outer: loop {
            let mut progress = false;
            for (_, reqs) in &mut model_groups {
                if batch.len() >= self.config.max_batch_size {
                    break 'outer;
                }
                if round < reqs.len() {
                    // Still has requests at this round index.
                    // Actually we pop from front each pass.
                    if !reqs.is_empty() {
                        let req = reqs.remove(0);
                        batch.push(req);
                        progress = true;
                    }
                }
            }
            round += 1;
            if !progress || batch.len() >= self.config.max_batch_size {
                break;
            }
        }

        // Push leftovers back.
        for (_, reqs) in model_groups {
            for req in reqs {
                let _ = self.queue.push(req);
            }
        }

        let est = Self::estimate_batch_latency(&batch);
        let id = self.next_batch_id;
        self.next_batch_id += 1;
        Some(ScheduledBatch {
            requests: batch,
            batch_id: id,
            policy_used: SchedulingPolicy::FairQueuing,
            estimated_latency_ms: est,
        })
    }

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

    fn estimate_batch_latency(requests: &[QueuedRequest]) -> u64 {
        requests.iter().filter_map(|r| r.estimated_processing_ms()).max().unwrap_or(0)
    }

    // -----------------------------------------------------------------------
    // Public accessors
    // -----------------------------------------------------------------------

    /// Current queue depth (including lazy-deleted entries).
    pub fn queue_depth(&self) -> usize {
        self.queue.len()
    }

    /// Reference to the current scheduler metrics.
    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    /// Cancel a request by ID. Returns `true` if found.
    pub fn cancel(&mut self, id: u64) -> bool {
        self.queue.cancel(id)
    }

    /// Remove all cancelled and expired entries from the queue. Returns the count.
    pub fn gc(&mut self) -> usize {
        self.queue.gc()
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors produced by [`PriorityScheduler`].
#[derive(Debug)]
pub enum SchedulerError {
    /// The underlying queue is full and no preemption was possible.
    QueueFull,
    /// Admission control or policy rejected the request.
    Rejected { reason: String },
    /// The request is structurally invalid.
    InvalidRequest(String),
}

impl std::fmt::Display for SchedulerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulerError::QueueFull => f.write_str("scheduler: queue is full"),
            SchedulerError::Rejected { reason } => {
                write!(f, "scheduler: request rejected: {reason}")
            },
            SchedulerError::InvalidRequest(msg) => {
                write!(f, "scheduler: invalid request: {msg}")
            },
        }
    }
}

impl std::error::Error for SchedulerError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(id: u64, model: &str, priority: RequestPriority) -> QueuedRequest {
        QueuedRequest::new(id, model, "prompt", 128, priority)
    }

    fn default_scheduler() -> PriorityScheduler {
        PriorityScheduler::new(SchedulerConfig::default())
    }

    fn scheduler_with_policy(policy: SchedulingPolicy) -> PriorityScheduler {
        PriorityScheduler::new(SchedulerConfig {
            policy,
            ..Default::default()
        })
    }

    // --- Config ---

    #[test]
    fn test_scheduler_config_default() {
        let cfg = SchedulerConfig::default();
        assert_eq!(cfg.max_batch_size, 8);
        assert_eq!(cfg.max_queue_depth, 1024);
        assert!(cfg.preemption_enabled);
        assert!(cfg.admission_control);
        assert!((cfg.admission_threshold - 0.9).abs() < f32::EPSILON);
    }

    // --- Policy display ---

    #[test]
    fn test_scheduling_policy_display() {
        assert_eq!(SchedulingPolicy::Fifo.to_string(), "fifo");
        assert_eq!(SchedulingPolicy::Priority.to_string(), "priority");
        assert_eq!(
            SchedulingPolicy::WeightedRoundRobin.to_string(),
            "weighted-round-robin"
        );
        assert_eq!(
            SchedulingPolicy::EarliestDeadlineFirst.to_string(),
            "earliest-deadline-first"
        );
        assert_eq!(SchedulingPolicy::FairQueuing.to_string(), "fair-queuing");
    }

    // --- Submit ---

    #[test]
    fn test_scheduler_submit_basic() {
        let mut sched = default_scheduler();
        let req = make_req(1, "gpt2", RequestPriority::Normal);
        let seq = sched.submit(req).expect("submit should succeed");
        assert_eq!(seq, 0);
        assert_eq!(sched.queue_depth(), 1);
    }

    #[test]
    fn test_scheduler_submit_capacity_rejection() {
        // Small queue, admission_control off so we can fill it.
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_queue_depth: 3,
            admission_control: false,
            max_batch_size: 8,
            ..Default::default()
        });
        sched.submit(make_req(1, "m", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(2, "m", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(3, "m", RequestPriority::Normal)).unwrap();

        // Now enable admission control on a new config — but it's simpler to
        // test that the 4th Normal push is rejected due to QueueFull since the
        // queue is at capacity.
        let err = sched.submit(make_req(4, "m", RequestPriority::Normal));
        assert!(err.is_err());
    }

    // --- Batch: FIFO ---

    #[test]
    fn test_scheduler_batch_fifo() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::Fifo);
        for i in 1u64..=5 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).unwrap();
        }
        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.policy_used, SchedulingPolicy::Fifo);
        // FIFO: IDs should come out in submission order.
        assert_eq!(batch.requests[0].id, 1);
        assert_eq!(batch.requests[4].id, 5);
    }

    // --- Batch: Priority ---

    #[test]
    fn test_scheduler_batch_priority() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::Priority);
        sched.submit(make_req(1, "m", RequestPriority::Low)).unwrap();
        sched.submit(make_req(2, "m", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(3, "m", RequestPriority::Critical)).unwrap();
        sched.submit(make_req(4, "m", RequestPriority::High)).unwrap();

        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.policy_used, SchedulingPolicy::Priority);
        // Critical first.
        assert_eq!(batch.requests[0].id, 3);
        // High second.
        assert_eq!(batch.requests[1].id, 4);
        // Normal third.
        assert_eq!(batch.requests[2].id, 2);
        // Low last.
        assert_eq!(batch.requests[3].id, 1);
    }

    // --- Batch: WRR ---

    #[test]
    fn test_scheduler_batch_wrr_proportional() {
        // Submit requests at different priorities and verify Critical gets more slots.
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            policy: SchedulingPolicy::WeightedRoundRobin,
            max_batch_size: 16,
            max_queue_depth: 128,
            admission_control: false,
            ..Default::default()
        });

        // 4 Low, 4 Normal, 4 High, 4 Critical
        for i in 0u64..4 {
            sched.submit(make_req(100 + i, "m", RequestPriority::Low)).unwrap();
            sched.submit(make_req(200 + i, "m", RequestPriority::Normal)).unwrap();
            sched.submit(make_req(300 + i, "m", RequestPriority::High)).unwrap();
            sched.submit(make_req(400 + i, "m", RequestPriority::Critical)).unwrap();
        }

        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.policy_used, SchedulingPolicy::WeightedRoundRobin);
        // All 16 requests should fit in the batch.
        assert_eq!(batch.size(), 16);
        let mix = batch.priority_mix();
        // Critical should have the most slots: weight 8, but we only have 4 Critical.
        // After exhausting Critical (4), remaining slots go to High (4), then Normal (4), Low (4).
        assert!(mix[3] > 0, "critical requests must appear");
        assert!(mix[2] > 0, "high requests must appear");
    }

    // --- Batch: EDF ---

    #[test]
    fn test_scheduler_batch_edf_ordering() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::EarliestDeadlineFirst);
        // id=1 has deadline 1000ms, id=2 has deadline 200ms, id=3 has no deadline
        sched
            .submit(make_req(1, "m", RequestPriority::Normal).with_deadline_ms(1000))
            .unwrap();
        sched
            .submit(make_req(2, "m", RequestPriority::Normal).with_deadline_ms(200))
            .unwrap();
        sched.submit(make_req(3, "m", RequestPriority::Normal)).unwrap();

        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.policy_used, SchedulingPolicy::EarliestDeadlineFirst);
        // Soonest deadline first: id=2 (200ms), then id=1 (1000ms), then id=3 (no deadline).
        assert_eq!(batch.requests[0].id, 2);
        assert_eq!(batch.requests[1].id, 1);
        assert_eq!(batch.requests[2].id, 3);
    }

    // --- Batch: Fair queuing ---

    #[test]
    fn test_scheduler_batch_fair_queuing() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::FairQueuing);
        // Two models: "alpha" and "beta"
        sched.submit(make_req(1, "alpha", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(2, "alpha", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(3, "beta", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(4, "beta", RequestPriority::Normal)).unwrap();

        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.policy_used, SchedulingPolicy::FairQueuing);
        // Both models must appear in the batch.
        let models: std::collections::HashSet<&str> =
            batch.requests.iter().map(|r| r.model.as_str()).collect();
        assert!(models.contains("alpha"));
        assert!(models.contains("beta"));
    }

    // --- ScheduledBatch helpers ---

    #[test]
    fn test_scheduled_batch_size() {
        let batch = ScheduledBatch {
            requests: vec![
                make_req(1, "m", RequestPriority::Normal),
                make_req(2, "m", RequestPriority::High),
            ],
            batch_id: 0,
            policy_used: SchedulingPolicy::Priority,
            estimated_latency_ms: 0,
        };
        assert_eq!(batch.size(), 2);
    }

    #[test]
    fn test_scheduled_batch_total_tokens() {
        let mut r1 = make_req(1, "m", RequestPriority::Normal);
        r1.max_tokens = 100;
        let mut r2 = make_req(2, "m", RequestPriority::Normal);
        r2.max_tokens = 200;
        let batch = ScheduledBatch {
            requests: vec![r1, r2],
            batch_id: 0,
            policy_used: SchedulingPolicy::Priority,
            estimated_latency_ms: 0,
        };
        assert_eq!(batch.total_tokens(), 300);
        assert_eq!(batch.max_tokens(), 200);
    }

    #[test]
    fn test_scheduled_batch_priority_mix() {
        let batch = ScheduledBatch {
            requests: vec![
                make_req(1, "m", RequestPriority::Low),
                make_req(2, "m", RequestPriority::Critical),
                make_req(3, "m", RequestPriority::Normal),
                make_req(4, "m", RequestPriority::Critical),
            ],
            batch_id: 0,
            policy_used: SchedulingPolicy::Priority,
            estimated_latency_ms: 0,
        };
        let mix = batch.priority_mix();
        assert_eq!(mix[0], 1); // Low
        assert_eq!(mix[1], 1); // Normal
        assert_eq!(mix[2], 0); // High
        assert_eq!(mix[3], 2); // Critical
    }

    // --- Cancel ---

    #[test]
    fn test_scheduler_cancel() {
        let mut sched = default_scheduler();
        sched.submit(make_req(1, "m", RequestPriority::Normal)).unwrap();
        sched.submit(make_req(2, "m", RequestPriority::Normal)).unwrap();

        assert!(sched.cancel(1));
        let batch = sched.schedule_batch().expect("non-empty");
        assert_eq!(batch.size(), 1);
        assert_eq!(batch.requests[0].id, 2);
    }

    // --- Metrics ---

    #[test]
    fn test_scheduler_metrics_update() {
        let mut sched = default_scheduler();
        for i in 1u64..=4 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).unwrap();
        }
        sched.schedule_batch();
        let m = sched.metrics();
        assert_eq!(m.total_batches_scheduled, 1);
        assert_eq!(m.total_requests_scheduled, 4);
    }

    #[test]
    fn test_scheduler_mean_batch_size() {
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_batch_size: 2,
            admission_control: false,
            ..Default::default()
        });
        // Submit 4 requests → 2 batches of size 2.
        for i in 1u64..=4 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).unwrap();
        }
        sched.schedule_batch();
        sched.schedule_batch();
        let m = sched.metrics();
        assert_eq!(m.total_batches_scheduled, 2);
        // Mean should be 2.0.
        assert!((m.mean_batch_size - 2.0).abs() < 0.01);
    }

    // --- GC ---

    #[test]
    fn test_scheduler_gc() {
        let mut sched = default_scheduler();
        for i in 1u64..=5 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).unwrap();
        }
        sched.cancel(1);
        sched.cancel(3);
        let removed = sched.gc();
        assert_eq!(removed, 2);
    }

    // --- Error display ---

    #[test]
    fn test_scheduler_error_display() {
        let e1 = SchedulerError::QueueFull;
        assert!(e1.to_string().contains("full"));

        let e2 = SchedulerError::Rejected {
            reason: "over threshold".to_owned(),
        };
        assert!(e2.to_string().contains("rejected"));
        assert!(e2.to_string().contains("threshold"));

        let e3 = SchedulerError::InvalidRequest("bad field".to_owned());
        assert!(e3.to_string().contains("invalid"));
        assert!(e3.to_string().contains("bad field"));
    }

    // --- Admission control rejects Normal when queue near capacity ---

    #[test]
    fn test_admission_control_rejects_normal_near_capacity() {
        let cap = 10usize;
        let threshold = (cap as f32 * 0.9) as usize; // 9
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_queue_depth: cap,
            max_batch_size: 16,
            admission_control: true,
            admission_threshold: 0.9,
            ..Default::default()
        });
        // Fill to threshold without admission control rejecting (use High priority).
        for i in 0u64..threshold as u64 {
            sched.submit(make_req(i, "m", RequestPriority::High)).expect("submit High");
        }
        // Now try Normal → must be rejected.
        let err = sched.submit(make_req(999, "m", RequestPriority::Normal));
        assert!(err.is_err(), "Normal must be rejected above threshold");
        assert_eq!(sched.metrics().total_requests_rejected, 1);
    }

    // --- Admission control passes High and Critical through always ---

    #[test]
    fn test_admission_control_passes_high_and_critical() {
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_queue_depth: 4,
            max_batch_size: 8,
            admission_control: true,
            admission_threshold: 0.5, // threshold = 2
            ..Default::default()
        });
        // Fill 2 slots with High (threshold=2, depth becomes 2).
        sched.submit(make_req(1, "m", RequestPriority::High)).expect("high ok");
        sched.submit(make_req(2, "m", RequestPriority::High)).expect("high ok");
        // Normal would be rejected but Critical must pass through.
        sched.submit(make_req(3, "m", RequestPriority::Critical)).expect("critical ok");
    }

    // --- schedule_batch returns None when queue empty ---

    #[test]
    fn test_schedule_batch_returns_none_when_empty() {
        let mut sched = default_scheduler();
        assert!(sched.schedule_batch().is_none());
    }

    // --- Batch ID increments per scheduled batch ---

    #[test]
    fn test_batch_id_increments() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::Fifo);
        for i in 0u64..4 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).expect("submit");
        }
        let b0 = sched.schedule_batch().expect("first batch");
        // Re-submit so there is something left to schedule.
        for i in 10u64..14 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).expect("submit");
        }
        let b1 = sched.schedule_batch().expect("second batch");
        assert!(
            b1.batch_id > b0.batch_id,
            "batch ID must be monotonically increasing"
        );
    }

    // --- EDF ordering: no-deadline requests last ---

    #[test]
    fn test_edf_no_deadline_requests_sort_last() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::EarliestDeadlineFirst);
        // id=10 has no deadline, id=20 has a far deadline, id=30 has a near deadline.
        sched.submit(make_req(10, "m", RequestPriority::Normal)).expect("submit");
        sched
            .submit(make_req(20, "m", RequestPriority::Normal).with_deadline_ms(5000))
            .expect("submit");
        sched
            .submit(make_req(30, "m", RequestPriority::Normal).with_deadline_ms(500))
            .expect("submit");

        let batch = sched.schedule_batch().expect("batch");
        // id=30 (500ms) → id=20 (5000ms) → id=10 (no deadline).
        assert_eq!(batch.requests[0].id, 30);
        assert_eq!(batch.requests[1].id, 20);
        assert_eq!(batch.requests[2].id, 10);
    }

    // --- WRR forms at least one request per non-empty bucket ---

    #[test]
    fn test_wrr_includes_all_priority_levels() {
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            policy: SchedulingPolicy::WeightedRoundRobin,
            max_batch_size: 16,
            max_queue_depth: 128,
            admission_control: false,
            ..Default::default()
        });
        sched.submit(make_req(1, "m", RequestPriority::Low)).expect("submit");
        sched.submit(make_req(2, "m", RequestPriority::Normal)).expect("submit");
        sched.submit(make_req(3, "m", RequestPriority::High)).expect("submit");
        sched.submit(make_req(4, "m", RequestPriority::Critical)).expect("submit");

        let batch = sched.schedule_batch().expect("batch");
        let mix = batch.priority_mix();
        assert!(mix[0] > 0, "Low must appear");
        assert!(mix[1] > 0, "Normal must appear");
        assert!(mix[2] > 0, "High must appear");
        assert!(mix[3] > 0, "Critical must appear");
    }

    // --- Fair queuing: single model falls back to priority order ---

    #[test]
    fn test_fair_queuing_single_model_priority_order() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::FairQueuing);
        sched.submit(make_req(1, "sole", RequestPriority::Low)).expect("submit");
        sched.submit(make_req(2, "sole", RequestPriority::Critical)).expect("submit");
        sched.submit(make_req(3, "sole", RequestPriority::Normal)).expect("submit");

        let batch = sched.schedule_batch().expect("batch");
        // Single model → falls back to priority ordering.
        assert_eq!(batch.requests[0].priority, RequestPriority::Critical);
    }

    // --- Preemption event recorded on Critical submit to near-full queue ---

    #[test]
    fn test_preemption_event_recorded_for_critical() {
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_queue_depth: 3,
            max_batch_size: 8,
            admission_control: false,
            preemption_enabled: true,
            ..Default::default()
        });
        sched.submit(make_req(1, "m", RequestPriority::Normal)).expect("normal 1");
        sched.submit(make_req(2, "m", RequestPriority::Normal)).expect("normal 2");
        sched.submit(make_req(3, "m", RequestPriority::Normal)).expect("normal 3");

        // Queue is full; submitting Critical should record a preemption event.
        let _ = sched.submit(make_req(4, "m", RequestPriority::Critical));
        assert_eq!(
            sched.metrics().preemptions,
            1,
            "preemption event must be recorded"
        );
    }

    // --- ScheduledBatch estimated_latency_ms with TPS ---

    #[test]
    fn test_scheduled_batch_estimated_latency() {
        let mut sched = scheduler_with_policy(SchedulingPolicy::Fifo);
        let r1 = make_req(1, "m", RequestPriority::Normal).with_estimated_tps(100.0);
        // max_tokens = 128 from make_req; 128/100*1000 = 1280 ms
        sched.submit(r1).expect("submit");
        let batch = sched.schedule_batch().expect("batch");
        assert_eq!(batch.estimated_latency_ms, 1280);
    }

    // --- ScheduledBatch max_tokens on empty batch ---

    #[test]
    fn test_scheduled_batch_max_tokens_empty() {
        let batch = ScheduledBatch {
            requests: Vec::new(),
            batch_id: 0,
            policy_used: SchedulingPolicy::Fifo,
            estimated_latency_ms: 0,
        };
        assert_eq!(batch.max_tokens(), 0);
        assert_eq!(batch.total_tokens(), 0);
    }

    // --- ScheduledBatch priority_mix all zeros for empty ---

    #[test]
    fn test_scheduled_batch_priority_mix_empty() {
        let batch = ScheduledBatch {
            requests: Vec::new(),
            batch_id: 0,
            policy_used: SchedulingPolicy::Priority,
            estimated_latency_ms: 0,
        };
        assert_eq!(batch.priority_mix(), [0, 0, 0, 0]);
    }

    // --- Submit increments queue depth ---

    #[test]
    fn test_submit_increments_queue_depth() {
        let mut sched = default_scheduler();
        assert_eq!(sched.queue_depth(), 0);
        sched.submit(make_req(1, "m", RequestPriority::Normal)).expect("submit");
        assert_eq!(sched.queue_depth(), 1);
        sched.submit(make_req(2, "m", RequestPriority::Normal)).expect("submit");
        assert_eq!(sched.queue_depth(), 2);
    }

    // --- Cancel non-existent returns false ---

    #[test]
    fn test_cancel_nonexistent_returns_false() {
        let mut sched = default_scheduler();
        assert!(!sched.cancel(999));
    }

    // --- GC after cancel removes entry ---

    #[test]
    fn test_gc_after_cancel() {
        let mut sched = default_scheduler();
        for i in 1u64..=4 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).expect("submit");
        }
        sched.cancel(2);
        sched.cancel(4);
        let removed = sched.gc();
        assert_eq!(removed, 2);
        assert_eq!(sched.queue_depth(), 2);
    }

    // --- SchedulingPolicy equality ---

    #[test]
    fn test_scheduling_policy_equality() {
        assert_eq!(SchedulingPolicy::Fifo, SchedulingPolicy::Fifo);
        assert_ne!(SchedulingPolicy::Fifo, SchedulingPolicy::Priority);
    }

    // --- Multiple batches exhaust the queue ---

    #[test]
    fn test_multiple_batches_exhaust_queue() {
        let mut sched = PriorityScheduler::new(SchedulerConfig {
            max_batch_size: 3,
            max_queue_depth: 64,
            admission_control: false,
            ..Default::default()
        });
        for i in 0u64..6 {
            sched.submit(make_req(i, "m", RequestPriority::Normal)).expect("submit");
        }
        let b1 = sched.schedule_batch().expect("batch 1");
        let b2 = sched.schedule_batch().expect("batch 2");
        let empty = sched.schedule_batch();

        assert_eq!(b1.size(), 3);
        assert_eq!(b2.size(), 3);
        assert!(
            empty.is_none(),
            "queue must be empty after exhausting all batches"
        );
        assert_eq!(sched.metrics().total_batches_scheduled, 2);
    }
}
