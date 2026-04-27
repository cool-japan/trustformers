//! Priority request queue for inference serving
//!
//! Provides a priority-based request queue with deadline awareness and cancellation support.
//! Uses a max-heap (BinaryHeap) ordered by priority level, with FIFO tie-breaking via
//! monotonic sequence numbers.

mod queue_extra_tests;

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Priority levels for inference requests.
///
/// Higher numeric value == higher priority.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RequestPriority {
    Low = 0,
    Normal = 1,
    High = 2,
    Critical = 3,
}

impl RequestPriority {
    /// Human-readable name for the priority level.
    pub fn name(&self) -> &'static str {
        match self {
            RequestPriority::Low => "low",
            RequestPriority::Normal => "normal",
            RequestPriority::High => "high",
            RequestPriority::Critical => "critical",
        }
    }

    /// Queue weight for weighted scheduling.
    ///
    /// Critical requests receive 8× more slots than Low requests.
    pub fn weight(&self) -> u32 {
        match self {
            RequestPriority::Low => 1,
            RequestPriority::Normal => 2,
            RequestPriority::High => 4,
            RequestPriority::Critical => 8,
        }
    }
}

impl std::fmt::Display for RequestPriority {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.name())
    }
}

/// A queued inference request, carrying all metadata needed for scheduling.
#[derive(Debug, Clone)]
pub struct QueuedRequest {
    /// Unique request identifier.
    pub id: u64,
    /// Target model name.
    pub model: String,
    /// Input prompt text.
    pub prompt: String,
    /// Maximum number of tokens to generate.
    pub max_tokens: usize,
    /// Scheduling priority.
    pub priority: RequestPriority,
    /// Relative deadline expressed in milliseconds from enqueue time.
    /// Uses sequence-number arithmetic as a time proxy (1 unit ≈ 1 ms).
    pub deadline_ms: Option<u64>,
    /// Monotonic sequence number assigned at enqueue time. Doubles as a time proxy.
    pub sequence_number: u64,
    /// When `true` this request should be skipped during dequeue.
    pub cancelled: bool,
    /// Measured or estimated tokens-per-second throughput for this request's model.
    pub estimated_tps: Option<f32>,
}

impl QueuedRequest {
    /// Construct a new request with sensible defaults.
    pub fn new(
        id: u64,
        model: &str,
        prompt: &str,
        max_tokens: usize,
        priority: RequestPriority,
    ) -> Self {
        QueuedRequest {
            id,
            model: model.to_owned(),
            prompt: prompt.to_owned(),
            max_tokens,
            priority,
            deadline_ms: None,
            sequence_number: 0,
            cancelled: false,
            estimated_tps: None,
        }
    }

    /// Attach a relative deadline (milliseconds from enqueue).
    pub fn with_deadline_ms(mut self, ms: u64) -> Self {
        self.deadline_ms = Some(ms);
        self
    }

    /// Attach an estimated tokens-per-second rate.
    pub fn with_estimated_tps(mut self, tps: f32) -> Self {
        self.estimated_tps = Some(tps);
        self
    }

    /// Mark this request as cancelled. The queue will skip it during dequeue (lazy deletion).
    pub fn cancel(&mut self) {
        self.cancelled = true;
    }

    /// Estimated processing time in milliseconds, derived from `estimated_tps` and `max_tokens`.
    ///
    /// Returns `None` when no TPS estimate is available.
    pub fn estimated_processing_ms(&self) -> Option<u64> {
        self.estimated_tps.map(|tps| {
            if tps <= 0.0 {
                return u64::MAX;
            }
            (self.max_tokens as f64 / tps as f64 * 1000.0) as u64
        })
    }

    /// Whether this request has exceeded its deadline.
    ///
    /// Uses sequence-number arithmetic as a monotonic time proxy where
    /// 1 sequence unit ≈ 1 millisecond.
    pub fn is_expired(&self, current_sequence: u64) -> bool {
        match self.deadline_ms {
            Some(d) => current_sequence.saturating_sub(self.sequence_number) > d,
            None => false,
        }
    }
}

// ---------------------------------------------------------------------------
// Heap ordering wrapper
// ---------------------------------------------------------------------------

/// Newtype wrapper that implements `Ord` so `BinaryHeap` becomes a max-heap
/// ordered by priority (highest first), with FIFO tie-breaking (lowest
/// sequence number wins).
struct PriorityRequest(QueuedRequest);

impl PartialEq for PriorityRequest {
    fn eq(&self, other: &Self) -> bool {
        self.0.priority == other.0.priority && self.0.sequence_number == other.0.sequence_number
    }
}

impl Eq for PriorityRequest {}

impl PartialOrd for PriorityRequest {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PriorityRequest {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher priority → greater in heap.
        // Tie-break: lower sequence_number (earlier arrival) → greater in heap (FIFO).
        self.0
            .priority
            .cmp(&other.0.priority)
            .then(other.0.sequence_number.cmp(&self.0.sequence_number))
    }
}

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// Snapshot statistics for the priority queue.
#[derive(Debug, Clone, Default)]
pub struct QueueStats {
    /// Total number of requests ever submitted to the queue.
    pub total_enqueued: u64,
    /// Total number of requests successfully dequeued for processing.
    pub total_dequeued: u64,
    /// Total number of requests cancelled before processing.
    pub total_cancelled: u64,
    /// Total number of requests dropped due to deadline expiry.
    pub total_expired: u64,
    /// Current logical queue depth (heap size, including lazy-deleted entries).
    pub current_depth: usize,
    /// Per-priority enqueue counts: index 0 = Low … index 3 = Critical.
    pub by_priority: [u64; 4],
}

impl QueueStats {
    /// Fraction of enqueued requests that were dequeued (0.0–1.0).
    pub fn throughput_ratio(&self) -> f32 {
        if self.total_enqueued == 0 {
            return 0.0;
        }
        self.total_dequeued as f32 / self.total_enqueued as f32
    }
}

// ---------------------------------------------------------------------------
// Priority queue
// ---------------------------------------------------------------------------

/// A bounded, priority-ordered request queue with lazy cancellation and
/// deadline-expiry support.
///
/// # Cancellation
/// Cancelled requests are marked in-place and skipped during `pop()`. Call
/// `gc()` periodically to reclaim heap memory.
///
/// # Deadline expiry
/// Expiry is also handled lazily: `pop()` skips entries whose deadline has
/// passed according to `is_expired(current_sequence)`.
pub struct PriorityQueue {
    heap: BinaryHeap<PriorityRequest>,
    stats: QueueStats,
    capacity: usize,
    next_sequence: u64,
}

impl PriorityQueue {
    /// Create a new queue with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        PriorityQueue {
            heap: BinaryHeap::new(),
            stats: QueueStats::default(),
            capacity,
            next_sequence: 0,
        }
    }

    /// Enqueue a request.
    ///
    /// Assigns a monotonic sequence number and updates statistics.
    ///
    /// # Errors
    /// Returns [`QueueError::QueueFull`] when the heap has reached `capacity`.
    pub fn push(&mut self, mut request: QueuedRequest) -> Result<u64, QueueError> {
        if self.heap.len() >= self.capacity {
            return Err(QueueError::QueueFull {
                capacity: self.capacity,
                current: self.heap.len(),
            });
        }

        let seq = self.next_sequence;
        self.next_sequence += 1;
        request.sequence_number = seq;

        self.stats.total_enqueued += 1;
        self.stats.by_priority[request.priority as usize] += 1;
        self.stats.current_depth = self.heap.len() + 1;

        self.heap.push(PriorityRequest(request));
        Ok(seq)
    }

    /// Dequeue the highest-priority, non-cancelled, non-expired request.
    ///
    /// Lazily skips cancelled and expired entries, updating statistics for
    /// each skipped entry.
    pub fn pop(&mut self) -> Option<QueuedRequest> {
        let current_seq = self.next_sequence;

        loop {
            match self.heap.pop() {
                None => {
                    self.stats.current_depth = 0;
                    return None;
                },
                Some(PriorityRequest(req)) => {
                    if req.cancelled {
                        self.stats.total_cancelled += 1;
                        continue;
                    }
                    if req.is_expired(current_seq) {
                        self.stats.total_expired += 1;
                        continue;
                    }
                    self.stats.total_dequeued += 1;
                    self.stats.current_depth = self.heap.len();
                    return Some(req);
                },
            }
        }
    }

    /// Peek at the highest-priority request without removing it.
    ///
    /// Note: the peeked request may be cancelled or expired.
    pub fn peek(&self) -> Option<&QueuedRequest> {
        self.heap.peek().map(|pr| &pr.0)
    }

    /// Cancel a request by ID using lazy deletion.
    ///
    /// Returns `true` if the request was found and marked cancelled.
    /// Uses an `O(n)` scan because `BinaryHeap` does not support random access.
    pub fn cancel(&mut self, id: u64) -> bool {
        // BinaryHeap does not expose mutable iterators; drain and rebuild.
        let mut found = false;
        let entries: Vec<PriorityRequest> = self.heap.drain().collect();
        for mut pr in entries {
            if pr.0.id == id && !pr.0.cancelled {
                pr.0.cancelled = true;
                found = true;
            }
            self.heap.push(pr);
        }
        found
    }

    /// Current heap size, including cancelled and expired (lazy-deleted) entries.
    pub fn len(&self) -> usize {
        self.heap.len()
    }

    /// Whether the heap contains no entries at all (including lazy-deleted ones).
    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Reference to the current queue statistics snapshot.
    pub fn stats(&self) -> &QueueStats {
        &self.stats
    }

    /// Remove all cancelled and expired entries from the heap.
    ///
    /// Returns the number of entries removed.
    pub fn gc(&mut self) -> usize {
        let current_seq = self.next_sequence;
        let before = self.heap.len();
        let entries: Vec<PriorityRequest> = self.heap.drain().collect();
        let mut removed = 0usize;
        for pr in entries {
            if pr.0.cancelled || pr.0.is_expired(current_seq) {
                removed += 1;
            } else {
                self.heap.push(pr);
            }
        }
        let _ = before; // suppress unused warning
        self.stats.current_depth = self.heap.len();
        removed
    }

    /// Per-priority count of non-cancelled pending requests.
    ///
    /// Index 0 = `Low`, index 1 = `Normal`, index 2 = `High`, index 3 = `Critical`.
    pub fn depth_by_priority(&self) -> [usize; 4] {
        let mut counts = [0usize; 4];
        for item in &self.heap {
            if !item.0.cancelled {
                counts[item.0.priority as usize] += 1;
            }
        }
        counts
    }
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors that can occur when interacting with the priority queue.
#[derive(Debug)]
pub enum QueueError {
    /// The queue has reached its capacity limit.
    QueueFull { capacity: usize, current: usize },
    /// The request itself is malformed.
    InvalidRequest(String),
}

impl std::fmt::Display for QueueError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QueueError::QueueFull { capacity, current } => {
                write!(f, "queue is full: capacity={capacity}, current={current}")
            },
            QueueError::InvalidRequest(msg) => {
                write!(f, "invalid request: {msg}")
            },
        }
    }
}

impl std::error::Error for QueueError {}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_req(id: u64, priority: RequestPriority) -> QueuedRequest {
        QueuedRequest::new(id, "model-a", "hello", 100, priority)
    }

    // --- RequestPriority ---

    #[test]
    fn test_request_priority_ordering() {
        assert!(RequestPriority::Critical > RequestPriority::High);
        assert!(RequestPriority::High > RequestPriority::Normal);
        assert!(RequestPriority::Normal > RequestPriority::Low);
        assert_eq!(RequestPriority::Normal, RequestPriority::Normal);
    }

    #[test]
    fn test_request_priority_weight() {
        assert_eq!(RequestPriority::Low.weight(), 1);
        assert_eq!(RequestPriority::Normal.weight(), 2);
        assert_eq!(RequestPriority::High.weight(), 4);
        assert_eq!(RequestPriority::Critical.weight(), 8);
    }

    // --- QueuedRequest ---

    #[test]
    fn test_queued_request_new() {
        let req = QueuedRequest::new(42, "gpt2", "prompt", 256, RequestPriority::High);
        assert_eq!(req.id, 42);
        assert_eq!(req.model, "gpt2");
        assert_eq!(req.prompt, "prompt");
        assert_eq!(req.max_tokens, 256);
        assert_eq!(req.priority, RequestPriority::High);
        assert!(!req.cancelled);
        assert!(req.deadline_ms.is_none());
        assert!(req.estimated_tps.is_none());
    }

    #[test]
    fn test_queued_request_deadline() {
        let req =
            QueuedRequest::new(1, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(500);
        assert_eq!(req.deadline_ms, Some(500));
    }

    #[test]
    fn test_queued_request_expired() {
        let mut req =
            QueuedRequest::new(1, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(100);
        req.sequence_number = 0;
        // current_sequence - sequence_number = 200 > 100 → expired
        assert!(req.is_expired(200));
    }

    #[test]
    fn test_queued_request_not_expired() {
        let mut req =
            QueuedRequest::new(1, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(500);
        req.sequence_number = 0;
        // current_sequence - sequence_number = 100 ≤ 500 → not expired
        assert!(!req.is_expired(100));
    }

    #[test]
    fn test_queued_request_no_deadline_never_expired() {
        let mut req = QueuedRequest::new(1, "m", "p", 10, RequestPriority::Normal);
        req.sequence_number = 0;
        assert!(!req.is_expired(u64::MAX));
    }

    #[test]
    fn test_queued_request_processing_estimate() {
        let req = QueuedRequest::new(1, "m", "p", 1000, RequestPriority::Normal)
            .with_estimated_tps(100.0);
        // 1000 tokens / 100 tps * 1000 ms/s = 10_000 ms
        assert_eq!(req.estimated_processing_ms(), Some(10_000));
    }

    #[test]
    fn test_queued_request_processing_estimate_none() {
        let req = QueuedRequest::new(1, "m", "p", 1000, RequestPriority::Normal);
        assert!(req.estimated_processing_ms().is_none());
    }

    // --- PriorityQueue: basic push/pop ---

    #[test]
    fn test_priority_queue_push_pop_fifo_same_priority() {
        let mut q = PriorityQueue::new(16);
        q.push(make_req(1, RequestPriority::Normal)).unwrap();
        q.push(make_req(2, RequestPriority::Normal)).unwrap();
        q.push(make_req(3, RequestPriority::Normal)).unwrap();

        // Same priority → FIFO (id 1 first)
        assert_eq!(q.pop().unwrap().id, 1);
        assert_eq!(q.pop().unwrap().id, 2);
        assert_eq!(q.pop().unwrap().id, 3);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_priority_queue_push_pop_priority_order() {
        let mut q = PriorityQueue::new(16);
        q.push(make_req(10, RequestPriority::Low)).unwrap();
        q.push(make_req(20, RequestPriority::Normal)).unwrap();
        q.push(make_req(30, RequestPriority::Critical)).unwrap();
        q.push(make_req(40, RequestPriority::High)).unwrap();

        assert_eq!(q.pop().unwrap().id, 30); // Critical first
        assert_eq!(q.pop().unwrap().id, 40); // High
        assert_eq!(q.pop().unwrap().id, 20); // Normal
        assert_eq!(q.pop().unwrap().id, 10); // Low
    }

    // --- Cancel ---

    #[test]
    fn test_priority_queue_cancel() {
        let mut q = PriorityQueue::new(8);
        q.push(make_req(1, RequestPriority::Normal)).unwrap();
        q.push(make_req(2, RequestPriority::Normal)).unwrap();

        assert!(q.cancel(1));
        // id=1 is skipped; id=2 comes out
        assert_eq!(q.pop().unwrap().id, 2);
        assert!(q.pop().is_none());
    }

    #[test]
    fn test_priority_queue_skip_cancelled() {
        let mut q = PriorityQueue::new(8);
        for i in 1u64..=5 {
            q.push(make_req(i, RequestPriority::Normal)).unwrap();
        }
        // Cancel 2, 3, 4
        q.cancel(2);
        q.cancel(3);
        q.cancel(4);

        assert_eq!(q.pop().unwrap().id, 1);
        assert_eq!(q.pop().unwrap().id, 5);
        assert!(q.pop().is_none());
    }

    // --- Capacity ---

    #[test]
    fn test_priority_queue_capacity_limit() {
        let mut q = PriorityQueue::new(3);
        q.push(make_req(1, RequestPriority::Normal)).unwrap();
        q.push(make_req(2, RequestPriority::Normal)).unwrap();
        q.push(make_req(3, RequestPriority::Normal)).unwrap();

        let err = q.push(make_req(4, RequestPriority::Normal)).unwrap_err();
        match err {
            QueueError::QueueFull { capacity, current } => {
                assert_eq!(capacity, 3);
                assert_eq!(current, 3);
            },
            _ => panic!("expected QueueFull"),
        }
    }

    // --- GC ---

    #[test]
    fn test_priority_queue_gc() {
        let mut q = PriorityQueue::new(16);
        for i in 1u64..=6 {
            q.push(make_req(i, RequestPriority::Normal)).unwrap();
        }
        q.cancel(1);
        q.cancel(3);
        q.cancel(5);

        let removed = q.gc();
        assert_eq!(removed, 3);
        assert_eq!(q.len(), 3);
    }

    // --- Depth by priority ---

    #[test]
    fn test_priority_queue_depth_by_priority() {
        let mut q = PriorityQueue::new(32);
        q.push(make_req(1, RequestPriority::Low)).unwrap();
        q.push(make_req(2, RequestPriority::Low)).unwrap();
        q.push(make_req(3, RequestPriority::Normal)).unwrap();
        q.push(make_req(4, RequestPriority::High)).unwrap();
        q.push(make_req(5, RequestPriority::Critical)).unwrap();

        let d = q.depth_by_priority();
        assert_eq!(d[0], 2); // Low
        assert_eq!(d[1], 1); // Normal
        assert_eq!(d[2], 1); // High
        assert_eq!(d[3], 1); // Critical

        // Cancelled entries must not be counted
        q.cancel(1);
        let d2 = q.depth_by_priority();
        assert_eq!(d2[0], 1);
    }

    // --- Stats ---

    #[test]
    fn test_queue_stats_throughput() {
        let mut q = PriorityQueue::new(16);
        q.push(make_req(1, RequestPriority::Normal)).unwrap();
        q.push(make_req(2, RequestPriority::Normal)).unwrap();
        q.push(make_req(3, RequestPriority::Normal)).unwrap();
        q.push(make_req(4, RequestPriority::Normal)).unwrap();

        q.pop();
        q.pop();

        let s = q.stats();
        assert_eq!(s.total_enqueued, 4);
        assert_eq!(s.total_dequeued, 2);
        let ratio = s.throughput_ratio();
        assert!((ratio - 0.5).abs() < f32::EPSILON);
    }

    #[test]
    fn test_queue_stats_throughput_zero_enqueued() {
        let q = PriorityQueue::new(16);
        assert_eq!(q.stats().throughput_ratio(), 0.0);
    }

    // --- Peek ---

    #[test]
    fn test_priority_queue_peek() {
        let mut q = PriorityQueue::new(8);
        q.push(make_req(1, RequestPriority::Low)).unwrap();
        q.push(make_req(2, RequestPriority::High)).unwrap();

        // Peek should show the highest-priority item
        let top = q.peek().expect("non-empty");
        assert_eq!(top.priority, RequestPriority::High);
        // Queue size unchanged
        assert_eq!(q.len(), 2);
    }

    // --- Error display ---

    #[test]
    fn test_queue_error_display() {
        let full = QueueError::QueueFull {
            capacity: 10,
            current: 10,
        };
        let s = full.to_string();
        assert!(s.contains("full"));
        assert!(s.contains("10"));

        let inv = QueueError::InvalidRequest("bad id".to_owned());
        let s2 = inv.to_string();
        assert!(s2.contains("invalid"));
        assert!(s2.contains("bad id"));
    }

    // --- RequestPriority display ---

    #[test]
    fn test_request_priority_display() {
        assert_eq!(RequestPriority::Low.to_string(), "low");
        assert_eq!(RequestPriority::Normal.to_string(), "normal");
        assert_eq!(RequestPriority::High.to_string(), "high");
        assert_eq!(RequestPriority::Critical.to_string(), "critical");
    }

    // --- RequestPriority name ---

    #[test]
    fn test_request_priority_name() {
        assert_eq!(RequestPriority::Low.name(), "low");
        assert_eq!(RequestPriority::Normal.name(), "normal");
        assert_eq!(RequestPriority::High.name(), "high");
        assert_eq!(RequestPriority::Critical.name(), "critical");
    }

    // --- QueuedRequest cancel ---

    #[test]
    fn test_queued_request_cancel() {
        let mut req = make_req(1, RequestPriority::Normal);
        assert!(!req.cancelled);
        req.cancel();
        assert!(req.cancelled);
    }

    // --- estimated_processing_ms with zero tps ---

    #[test]
    fn test_estimated_processing_ms_zero_tps() {
        let req =
            QueuedRequest::new(1, "m", "p", 100, RequestPriority::Normal).with_estimated_tps(0.0);
        assert_eq!(req.estimated_processing_ms(), Some(u64::MAX));
    }

    // --- estimated_processing_ms is proportional to tokens ---

    #[test]
    fn test_estimated_processing_ms_proportional() {
        let r1 =
            QueuedRequest::new(1, "m", "p", 100, RequestPriority::Normal).with_estimated_tps(100.0);
        let r2 =
            QueuedRequest::new(2, "m", "p", 200, RequestPriority::Normal).with_estimated_tps(100.0);
        // r2 has double the tokens → double the estimated time.
        assert_eq!(
            r2.estimated_processing_ms().unwrap(),
            2 * r1.estimated_processing_ms().unwrap()
        );
    }

    // --- is_expired: exactly at boundary ---

    #[test]
    fn test_is_expired_exactly_at_boundary() {
        let mut req =
            QueuedRequest::new(1, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(100);
        req.sequence_number = 0;
        // current - seq = 100; deadline = 100; 100 > 100 is false → not expired.
        assert!(!req.is_expired(100));
        // current - seq = 101 > 100 → expired.
        assert!(req.is_expired(101));
    }

    // --- Queue empty pop returns None ---

    #[test]
    fn test_queue_empty_pop_returns_none() {
        let mut q = PriorityQueue::new(16);
        assert!(q.pop().is_none());
        assert!(q.is_empty());
    }

    // --- Sequence number assignment ---

    #[test]
    fn test_sequence_number_assigned_monotonically() {
        let mut q = PriorityQueue::new(8);
        let s0 = q.push(make_req(1, RequestPriority::Normal)).expect("push ok");
        let s1 = q.push(make_req(2, RequestPriority::Normal)).expect("push ok");
        let s2 = q.push(make_req(3, RequestPriority::Normal)).expect("push ok");
        assert!(s1 > s0);
        assert!(s2 > s1);
    }

    // --- Queue len tracks insertions ---

    #[test]
    fn test_queue_len_tracks_insertions() {
        let mut q = PriorityQueue::new(16);
        assert_eq!(q.len(), 0);
        q.push(make_req(1, RequestPriority::Normal)).expect("push");
        assert_eq!(q.len(), 1);
        q.push(make_req(2, RequestPriority::Normal)).expect("push");
        assert_eq!(q.len(), 2);
    }

    // --- Stats by_priority tracks counts ---

    #[test]
    fn test_stats_by_priority_tracks_counts() {
        let mut q = PriorityQueue::new(32);
        for _ in 0..3 {
            q.push(make_req(0, RequestPriority::Critical)).expect("push");
        }
        q.push(make_req(0, RequestPriority::Low)).expect("push");
        let s = q.stats();
        assert_eq!(s.by_priority[3], 3, "three critical requests");
        assert_eq!(s.by_priority[0], 1, "one low request");
    }

    // --- All expired requests skipped ---

    #[test]
    fn test_all_expired_requests_skipped() {
        let mut q = PriorityQueue::new(16);
        // deadline = 1ms; push at seq 0, 1, 2; then pop when current_seq is high.
        for id in 0u64..3 {
            let req =
                QueuedRequest::new(id, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(1);
            q.push(req).expect("push");
        }
        // Advance next_sequence by pushing enough dummy entries to exhaust deadline.
        for id in 100u64..200 {
            let _ = q.push(make_req(id, RequestPriority::Low));
        }
        // Pop all; the 3 with deadline=1 should be expired.
        // (They may or may not appear depending on what survived, but at least
        // expired count must increase as we drain.)
        while q.pop().is_some() {}
        assert!(
            q.stats().total_expired > 0,
            "some requests must have been expired"
        );
    }

    // --- GC removes expired as well as cancelled ---

    #[test]
    fn test_gc_removes_expired_entries() {
        let mut q = PriorityQueue::new(64);
        // Push 3 requests with tiny deadlines.
        for id in 0u64..3 {
            let req =
                QueuedRequest::new(id, "m", "p", 10, RequestPriority::Normal).with_deadline_ms(0);
            q.push(req).expect("push");
        }
        // Push 50 more so that next_sequence advances well past deadline.
        for id in 10u64..60 {
            q.push(make_req(id, RequestPriority::Normal)).expect("push");
        }
        let removed = q.gc();
        // All 3 deadline=0 entries should be garbage collected.
        assert!(
            removed >= 3,
            "gc must remove at least the 3 expired entries"
        );
    }

    // --- Cancel non-existent ID returns false ---

    #[test]
    fn test_cancel_nonexistent_id_returns_false() {
        let mut q = PriorityQueue::new(8);
        q.push(make_req(1, RequestPriority::Normal)).expect("push");
        let found = q.cancel(999);
        assert!(!found, "cancelling unknown ID must return false");
    }

    // --- Cancel same request twice: second call returns false ---

    #[test]
    fn test_cancel_same_request_twice() {
        let mut q = PriorityQueue::new(8);
        q.push(make_req(42, RequestPriority::Normal)).expect("push");
        assert!(q.cancel(42));
        // Second cancel: already cancelled → returns false.
        assert!(!q.cancel(42));
    }

    // --- stats current_depth updated after push ---

    #[test]
    fn test_stats_current_depth_after_push() {
        let mut q = PriorityQueue::new(16);
        q.push(make_req(1, RequestPriority::Normal)).expect("push");
        q.push(make_req(2, RequestPriority::Normal)).expect("push");
        assert_eq!(q.stats().current_depth, 2);
    }

    // --- peek returns None on empty queue ---

    #[test]
    fn test_peek_returns_none_on_empty_queue() {
        let q = PriorityQueue::new(8);
        assert!(q.peek().is_none());
    }

    // --- Critical vs Normal ordering through heap ---

    #[test]
    fn test_critical_always_before_normal() {
        let mut q = PriorityQueue::new(16);
        // Push a mix of Normal and Critical in alternating order.
        q.push(make_req(1, RequestPriority::Normal)).expect("push");
        q.push(make_req(2, RequestPriority::Critical)).expect("push");
        q.push(make_req(3, RequestPriority::Normal)).expect("push");
        q.push(make_req(4, RequestPriority::Critical)).expect("push");

        // Both critical requests must come out before any normal request.
        let first = q.pop().expect("first");
        let second = q.pop().expect("second");
        assert_eq!(first.priority, RequestPriority::Critical);
        assert_eq!(second.priority, RequestPriority::Critical);
    }

    // --- FIFO within same priority is preserved after cancel + pop ---

    #[test]
    fn test_fifo_preserved_after_cancel() {
        let mut q = PriorityQueue::new(16);
        for id in 1u64..=5 {
            q.push(make_req(id, RequestPriority::High)).expect("push");
        }
        q.cancel(1); // Remove head.
                     // Next pop must be id=2 (next in FIFO order).
        let req = q.pop().expect("pop");
        assert_eq!(req.id, 2);
    }
}
