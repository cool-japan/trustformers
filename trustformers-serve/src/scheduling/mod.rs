//! Advanced multi-model scheduling strategies for TrustformeRS serving.
//!
//! This module provides two complementary schedulers that operate at different
//! abstraction levels:
//!
//! * [`WorkConservingScheduler`] — a work-conserving, priority-aware scheduler
//!   with optional preemption and SLO-deadline tracking. It maximises GPU
//!   utilisation by ensuring that no compute capacity sits idle while runnable
//!   requests exist.
//!
//! * [`IterationLevelScheduler`] — a vLLM-style, per-iteration scheduler that
//!   operates at the granularity of a single autoregressive decoding step.
//!   It tracks KV-cache block allocation and separates *prefill* from *decode*
//!   groups, enabling continuous batching.
//!
//! Neither scheduler duplicates the simpler [`super::scheduler`] module, which
//! only handles intra-queue ordering (FIFO, WRR, EDF, FairQueuing). These
//! advanced schedulers build on top of that foundation.

pub mod priority_scheduler;

mod scheduling_extra_tests;

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Public error type
// ---------------------------------------------------------------------------

/// Errors produced by the advanced schedulers.
#[derive(Debug, Clone, PartialEq)]
pub enum SchedulingError {
    /// The pending queue has reached its configured capacity.
    QueueFull,
    /// The referenced request_id does not exist in the running set.
    RequestNotFound,
    /// The scheduler cannot satisfy the token or sequence budget.
    CapacityExceeded,
    /// The request is structurally invalid.
    InvalidRequest(String),
}

impl std::fmt::Display for SchedulingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SchedulingError::QueueFull => write!(f, "scheduling: pending queue is full"),
            SchedulingError::RequestNotFound => {
                write!(f, "scheduling: request_id not found in running set")
            },
            SchedulingError::CapacityExceeded => {
                write!(f, "scheduling: capacity (tokens or sequences) exceeded")
            },
            SchedulingError::InvalidRequest(msg) => {
                write!(f, "scheduling: invalid request — {msg}")
            },
        }
    }
}

impl std::error::Error for SchedulingError {}

// ---------------------------------------------------------------------------
// Preemption policy
// ---------------------------------------------------------------------------

/// Policy that governs when the [`WorkConservingScheduler`] may preempt an
/// in-flight request to admit a higher-priority one.
#[derive(Debug, Clone, PartialEq)]
pub enum PreemptionPolicy {
    /// Never preempt a running request; new arrivals wait.
    NoPreemption,
    /// Preempt the lowest-priority running request when the pending queue is
    /// full and a higher-priority request cannot be admitted otherwise.
    PriorityBased,
    /// Preempt to avoid missing SLO deadlines; the `max_deadline_miss_rate`
    /// (fraction in `[0.0, 1.0]`) caps the tolerable fraction of late
    /// completions before preemption is triggered.
    DeadlineBased { max_deadline_miss_rate: f32 },
}

// ---------------------------------------------------------------------------
// Work-conserving scheduler — data structures
// ---------------------------------------------------------------------------

/// Configuration for [`WorkConservingScheduler`].
#[derive(Debug, Clone)]
pub struct WorkConservingConfig {
    /// Maximum simultaneous in-flight requests.
    pub max_running_requests: usize,
    /// Maximum total tokens permitted across all running requests
    /// (input_tokens + max_output_tokens per request).
    pub max_tokens_in_flight: usize,
    /// Preemption policy.
    pub preemption_policy: PreemptionPolicy,
    /// Factor by which effective priority grows per second of wait time.
    /// A value of `1.0` means the effective priority increases by 1.0 per
    /// second (preventing starvation).
    pub priority_aging_factor: f32,
    /// Maximum pending queue depth before new enqueues are rejected.
    pub max_pending_depth: usize,
}

impl Default for WorkConservingConfig {
    fn default() -> Self {
        WorkConservingConfig {
            max_running_requests: 32,
            max_tokens_in_flight: 65536,
            preemption_policy: PreemptionPolicy::PriorityBased,
            priority_aging_factor: 0.5,
            max_pending_depth: 1024,
        }
    }
}

/// A request enqueued but not yet dispatched to hardware.
#[derive(Debug, Clone)]
pub struct ScheduledRequest {
    /// Unique identifier for this request.
    pub request_id: String,
    /// The model this request targets.
    pub model_id: String,
    /// Number of input prompt tokens.
    pub input_tokens: usize,
    /// Maximum number of tokens to generate.
    pub max_output_tokens: usize,
    /// Static priority in `[0, 255]`; higher values indicate more urgent requests.
    pub priority: u8,
    /// Optional wall-clock deadline; a miss is recorded in [`SchedulerStats`].
    pub deadline: Option<Instant>,
    /// Moment the request entered the pending queue.
    pub enqueued_at: Instant,
}

impl ScheduledRequest {
    /// Create a new pending request with `enqueued_at = Instant::now()`.
    pub fn new(
        request_id: impl Into<String>,
        model_id: impl Into<String>,
        input_tokens: usize,
        max_output_tokens: usize,
        priority: u8,
    ) -> Self {
        ScheduledRequest {
            request_id: request_id.into(),
            model_id: model_id.into(),
            input_tokens,
            max_output_tokens,
            priority,
            deadline: None,
            enqueued_at: Instant::now(),
        }
    }

    /// Attach an SLO deadline to this request.
    pub fn with_deadline(mut self, deadline: Instant) -> Self {
        self.deadline = Some(deadline);
        self
    }

    /// Total token budget = input + output capacity.
    pub fn token_budget(&self) -> usize {
        self.input_tokens.saturating_add(self.max_output_tokens)
    }
}

/// A request that has been dispatched and is currently running.
#[derive(Debug, Clone)]
pub struct RunningRequest {
    /// The underlying pending request metadata.
    pub request: ScheduledRequest,
    /// Moment inference began.
    pub started_at: Instant,
    /// Number of tokens generated so far (updated by the caller).
    pub tokens_generated: usize,
}

impl RunningRequest {
    fn new(request: ScheduledRequest) -> Self {
        RunningRequest {
            request,
            started_at: Instant::now(),
            tokens_generated: 0,
        }
    }

    /// Elapsed milliseconds since this request started.
    fn elapsed_ms(&self) -> f64 {
        self.started_at.elapsed().as_secs_f64() * 1_000.0
    }
}

/// Cumulative statistics collected by [`WorkConservingScheduler`].
#[derive(Debug, Clone, Default)]
pub struct SchedulerStats {
    /// Total requests that transitioned from pending → running.
    pub total_scheduled: u64,
    /// Total requests preempted (moved back to pending).
    pub total_preempted: u64,
    /// Total requests completed (by the caller calling
    /// [`WorkConservingScheduler::complete_request`]).
    pub total_completed: u64,
    /// Running Exponential-Moving-Average of queue wait time in ms.
    pub mean_queue_time_ms: f64,
    /// Running EMA of execution time (start → complete) in ms.
    pub mean_execution_time_ms: f64,
    /// Number of times a completed request missed its SLO deadline.
    pub slo_violation_count: u64,
}

// ---------------------------------------------------------------------------
// WorkConservingScheduler
// ---------------------------------------------------------------------------

/// A work-conserving GPU-utilisation scheduler.
///
/// Design invariants
/// -----------------
/// * At most `config.max_running_requests` requests run simultaneously.
/// * Total tokens in flight never exceeds `config.max_tokens_in_flight`.
/// * If the pending queue is full, enqueue returns [`SchedulingError::QueueFull`].
/// * Priority aging (`aged_priority`) prevents starvation: long-waiting requests
///   have their effective priority boosted continuously.
/// * `check_preemption` is called externally (e.g. on each scheduler tick) and
///   may preempt the lowest-priority running request to make room.
pub struct WorkConservingScheduler {
    config: WorkConservingConfig,
    pending_requests: VecDeque<ScheduledRequest>,
    running_requests: HashMap<String, RunningRequest>,
    stats: SchedulerStats,
    tokens_in_flight: usize,
}

impl WorkConservingScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: WorkConservingConfig) -> Self {
        WorkConservingScheduler {
            config,
            pending_requests: VecDeque::new(),
            running_requests: HashMap::new(),
            stats: SchedulerStats::default(),
            tokens_in_flight: 0,
        }
    }

    // -----------------------------------------------------------------------
    // Public interface
    // -----------------------------------------------------------------------

    /// Add `request` to the pending queue.
    ///
    /// Returns [`SchedulingError::QueueFull`] if the pending queue is at
    /// capacity; [`SchedulingError::InvalidRequest`] if the request has no
    /// tokens.
    pub fn enqueue(&mut self, request: ScheduledRequest) -> Result<(), SchedulingError> {
        if request.input_tokens == 0 && request.max_output_tokens == 0 {
            return Err(SchedulingError::InvalidRequest(
                "request must have at least one token".to_owned(),
            ));
        }
        if self.pending_requests.len() >= self.config.max_pending_depth {
            return Err(SchedulingError::QueueFull);
        }
        self.pending_requests.push_back(request);
        Ok(())
    }

    /// Pop the highest aged-priority pending request and move it to running.
    ///
    /// Returns `None` if no pending request can be admitted (either the queue
    /// is empty or capacity is exhausted).
    pub fn next_request(&mut self) -> Option<ScheduledRequest> {
        // Find the index of the pending request with the highest aged priority
        // that also fits within the capacity limits.
        let best_index = self.find_best_pending_index()?;

        let request = self.pending_requests.remove(best_index)?;

        // Admit to running.
        let queue_wait_ms = request.enqueued_at.elapsed().as_secs_f64() * 1_000.0;
        Self::update_mean_ema(&mut self.stats.mean_queue_time_ms, queue_wait_ms);

        let budget = request.token_budget();
        let running = RunningRequest::new(request.clone());
        self.running_requests.insert(request.request_id.clone(), running);
        self.tokens_in_flight = self.tokens_in_flight.saturating_add(budget);
        self.stats.total_scheduled += 1;

        Some(request)
    }

    /// Mark the request identified by `request_id` as completed.
    ///
    /// Updates statistics (including SLO violation tracking) and releases its
    /// token budget from `tokens_in_flight`.
    pub fn complete_request(
        &mut self,
        request_id: &str,
        tokens_generated: usize,
    ) -> Result<(), SchedulingError> {
        let mut running = self
            .running_requests
            .remove(request_id)
            .ok_or(SchedulingError::RequestNotFound)?;

        running.tokens_generated = tokens_generated;
        let exec_ms = running.elapsed_ms();
        Self::update_mean_ema(&mut self.stats.mean_execution_time_ms, exec_ms);
        self.stats.total_completed += 1;

        // SLO violation: check if request finished after its deadline.
        if let Some(deadline) = running.request.deadline {
            if Instant::now() > deadline {
                self.stats.slo_violation_count += 1;
            }
        }

        // Release token budget.
        let budget = running.request.token_budget();
        self.tokens_in_flight = self.tokens_in_flight.saturating_sub(budget);

        Ok(())
    }

    /// Evaluate the preemption policy and return any requests that should be
    /// moved back to the pending queue (preempted).
    ///
    /// The caller is responsible for stopping inference for each returned
    /// request and re-enqueueing it via [`Self::enqueue`] when appropriate.
    pub fn check_preemption(&mut self) -> Vec<ScheduledRequest> {
        match &self.config.preemption_policy.clone() {
            PreemptionPolicy::NoPreemption => vec![],
            PreemptionPolicy::PriorityBased => self.preempt_priority_based(),
            PreemptionPolicy::DeadlineBased {
                max_deadline_miss_rate,
            } => self.preempt_deadline_based(*max_deadline_miss_rate),
        }
    }

    /// Compute the effective aged priority of a pending request.
    ///
    /// `aged_priority = base_priority + elapsed_seconds * priority_aging_factor`
    ///
    /// This monotonically increases with wait time, preventing indefinite
    /// starvation of low-priority requests.
    pub fn aged_priority(&self, request: &ScheduledRequest) -> f32 {
        let elapsed_secs = request.enqueued_at.elapsed().as_secs_f64() as f32;
        request.priority as f32 + elapsed_secs * self.config.priority_aging_factor
    }

    /// Read-only view of cumulative statistics.
    pub fn stats(&self) -> &SchedulerStats {
        &self.stats
    }

    /// Number of pending (not yet admitted) requests.
    pub fn pending_count(&self) -> usize {
        self.pending_requests.len()
    }

    /// Number of currently running requests.
    pub fn running_count(&self) -> usize {
        self.running_requests.len()
    }

    /// Current token budget in use across all running requests.
    pub fn tokens_in_flight(&self) -> usize {
        self.tokens_in_flight
    }

    // -----------------------------------------------------------------------
    // Private helpers
    // -----------------------------------------------------------------------

    /// Find the index of the pending request with the highest aged priority
    /// that fits within both the running-request count limit and the token
    /// budget limit.
    fn find_best_pending_index(&self) -> Option<usize> {
        if self.running_requests.len() >= self.config.max_running_requests {
            return None;
        }

        let mut best_index: Option<usize> = None;
        let mut best_score = f32::NEG_INFINITY;

        for (i, req) in self.pending_requests.iter().enumerate() {
            // Token capacity check.
            let prospective_total = self.tokens_in_flight.saturating_add(req.token_budget());
            if prospective_total > self.config.max_tokens_in_flight {
                continue;
            }
            let score = self.aged_priority(req);
            if score > best_score {
                best_score = score;
                best_index = Some(i);
            }
        }

        best_index
    }

    /// Priority-based preemption: if the pending queue is at capacity and there
    /// exist running requests with lower effective priority than the
    /// highest-priority pending request, preempt the lowest-running one.
    fn preempt_priority_based(&mut self) -> Vec<ScheduledRequest> {
        if self.pending_requests.len() < self.config.max_pending_depth {
            return vec![];
        }

        // Highest aged-priority among pending.
        let max_pending_priority = self
            .pending_requests
            .iter()
            .map(|r| self.aged_priority(r))
            .fold(f32::NEG_INFINITY, f32::max);

        // Find running request with the lowest priority.
        let lowest_running_id = self
            .running_requests
            .iter()
            .min_by(|(_, a), (_, b)| {
                let pa = a.request.priority as f32;
                let pb = b.request.priority as f32;
                pa.partial_cmp(&pb).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(id, r)| (id.clone(), r.request.priority as f32));

        match lowest_running_id {
            Some((id, running_priority)) if running_priority < max_pending_priority => {
                // Preempt it.
                if let Some(running) = self.running_requests.remove(&id) {
                    self.tokens_in_flight =
                        self.tokens_in_flight.saturating_sub(running.request.token_budget());
                    self.stats.total_preempted += 1;
                    vec![running.request]
                } else {
                    vec![]
                }
            },
            _ => vec![],
        }
    }

    /// Deadline-based preemption: if the current SLO violation rate (estimated
    /// as `slo_violations / total_completed`) exceeds `max_miss_rate`, preempt
    /// the running request whose deadline is furthest in the future (least
    /// urgent), to free capacity for deadline-critical requests.
    fn preempt_deadline_based(&mut self, max_miss_rate: f32) -> Vec<ScheduledRequest> {
        let total = self.stats.total_completed;
        if total == 0 {
            return vec![];
        }
        let current_rate = self.stats.slo_violation_count as f32 / total as f32;
        if current_rate <= max_miss_rate {
            return vec![];
        }

        // Find running request with deadline furthest away (most expendable).
        let victim_id = self
            .running_requests
            .iter()
            .filter_map(|(id, r)| r.request.deadline.map(|d| (id.clone(), d)))
            .max_by_key(|(_, d)| *d)
            .map(|(id, _)| id);

        match victim_id {
            Some(id) => {
                if let Some(running) = self.running_requests.remove(&id) {
                    self.tokens_in_flight =
                        self.tokens_in_flight.saturating_sub(running.request.token_budget());
                    self.stats.total_preempted += 1;
                    vec![running.request]
                } else {
                    vec![]
                }
            },
            None => vec![],
        }
    }

    /// Exponential moving-average update (α = 0.1).
    fn update_mean_ema(mean: &mut f64, new_sample: f64) {
        const ALPHA: f64 = 0.1;
        if *mean == 0.0 {
            *mean = new_sample;
        } else {
            *mean = *mean * (1.0 - ALPHA) + new_sample * ALPHA;
        }
    }
}

// ---------------------------------------------------------------------------
// Iteration-level scheduler (vLLM style)
// ---------------------------------------------------------------------------

/// Status of a single sequence within a [`SequenceGroup`].
#[derive(Debug, Clone, PartialEq)]
pub enum SeqStatus {
    /// Waiting in the FIFO queue; prompt not yet prefilled.
    Waiting,
    /// Active in the current iteration's batch.
    Running,
    /// Swapped to CPU/disk because of KV-cache pressure.
    Swapped,
    /// Finished generating tokens.
    Finished,
}

/// Sampling hyper-parameters for a sequence group.
#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
}

impl Default for SamplingParams {
    fn default() -> Self {
        SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            max_tokens: 256,
        }
    }
}

/// An individual autoregressive sequence.
#[derive(Debug, Clone)]
pub struct Sequence {
    /// Unique sequence identifier.
    pub seq_id: u64,
    /// Token ids from the prompt.
    pub prompt_tokens: Vec<u32>,
    /// Token ids generated so far.
    pub output_tokens: Vec<u32>,
    /// Current lifecycle status.
    pub status: SeqStatus,
    /// Number of KV-cache blocks currently allocated for this sequence.
    pub num_kv_blocks: usize,
}

impl Sequence {
    /// Create a new sequence in the Waiting state.
    pub fn new(seq_id: u64, prompt_tokens: Vec<u32>) -> Self {
        Sequence {
            seq_id,
            prompt_tokens,
            output_tokens: Vec::new(),
            status: SeqStatus::Waiting,
            num_kv_blocks: 0,
        }
    }

    /// Total length (prompt + output).
    pub fn total_len(&self) -> usize {
        self.prompt_tokens.len() + self.output_tokens.len()
    }

    /// Number of KV-cache blocks required given a block size.
    pub fn required_kv_blocks(&self, block_size: usize) -> usize {
        if block_size == 0 {
            return 0;
        }
        self.total_len().div_ceil(block_size)
    }

    /// Whether this sequence has reached its maximum token budget.
    pub fn is_finished(&self) -> bool {
        self.status == SeqStatus::Finished
    }
}

/// A group of parallel sequences sharing the same prompt and sampling params
/// (e.g. beam search candidates, best-of-N samples).
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Unique group identifier.
    pub group_id: String,
    /// All sequences in this group.
    pub sequences: Vec<Sequence>,
    /// Shared sampling parameters.
    pub sampling_params: SamplingParams,
    /// When this group was created (used for FIFO ordering).
    pub created_at: Instant,
}

impl SequenceGroup {
    /// Create a new sequence group, setting created_at to now.
    pub fn new(
        group_id: impl Into<String>,
        sequences: Vec<Sequence>,
        sampling_params: SamplingParams,
    ) -> Self {
        SequenceGroup {
            group_id: group_id.into(),
            sequences,
            sampling_params,
            created_at: Instant::now(),
        }
    }

    /// Whether every sequence in the group has finished.
    pub fn is_all_finished(&self) -> bool {
        self.sequences.iter().all(|s| s.is_finished())
    }

    /// Number of sequences in Waiting status.
    pub fn waiting_count(&self) -> usize {
        self.sequences.iter().filter(|s| s.status == SeqStatus::Waiting).count()
    }

    /// Number of sequences in Running status.
    pub fn running_count(&self) -> usize {
        self.sequences.iter().filter(|s| s.status == SeqStatus::Running).count()
    }

    /// Total token count across all sequences (prompt + output).
    pub fn total_tokens(&self) -> usize {
        self.sequences.iter().map(|s| s.total_len()).sum()
    }

    /// True if the group is in the prefill phase (no output tokens yet for any
    /// running sequence).
    pub fn is_prefill(&self) -> bool {
        self.sequences
            .iter()
            .filter(|s| s.status == SeqStatus::Running || s.status == SeqStatus::Waiting)
            .all(|s| s.output_tokens.is_empty())
    }
}

/// Configuration for [`IterationLevelScheduler`].
#[derive(Debug, Clone)]
pub struct IterationConfig {
    /// Maximum simultaneous sequences across all groups.
    pub max_num_seqs: usize,
    /// Maximum total token count (prompt + output) per scheduling iteration.
    pub max_num_batched_tokens: usize,
    /// KV-cache block size in tokens (e.g. 16 or 32).
    pub block_size: usize,
    /// Total available KV-cache blocks.
    pub total_kv_blocks: usize,
}

impl Default for IterationConfig {
    fn default() -> Self {
        IterationConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 8192,
            block_size: 16,
            total_kv_blocks: 4096,
        }
    }
}

/// Output of one scheduling iteration: which groups to execute and how.
#[derive(Debug, Clone, Default)]
pub struct ScheduleOutput {
    /// `group_id`s selected for this iteration, in dispatch order.
    pub scheduled_groups: Vec<String>,
    /// Total tokens across all scheduled groups.
    pub total_tokens: usize,
    /// Groups in prefill phase (first pass over the prompt).
    pub prefill_groups: Vec<String>,
    /// Groups in decode phase (generating output tokens).
    pub decode_groups: Vec<String>,
}

// ---------------------------------------------------------------------------
// IterationLevelScheduler
// ---------------------------------------------------------------------------

/// A vLLM-style, per-iteration continuous-batching scheduler.
///
/// On each call to [`Self::schedule_iteration`] the scheduler:
/// 1. Selects *running* (already prefilled) groups for decode — they take
///    priority because they hold allocated KV-cache blocks.
/// 2. Admits *waiting* groups for prefill up to the remaining token budget.
/// 3. Returns a [`ScheduleOutput`] describing the batch.
pub struct IterationLevelScheduler {
    config: IterationConfig,
    sequence_groups: Vec<SequenceGroup>,
    free_kv_blocks: usize,
}

impl IterationLevelScheduler {
    /// Create a new iteration-level scheduler.
    pub fn new(config: IterationConfig) -> Self {
        let free = config.total_kv_blocks;
        IterationLevelScheduler {
            config,
            sequence_groups: Vec::new(),
            free_kv_blocks: free,
        }
    }

    // -----------------------------------------------------------------------
    // Public interface
    // -----------------------------------------------------------------------

    /// Admit a new sequence group (initially all sequences are Waiting).
    pub fn add_sequence_group(&mut self, group: SequenceGroup) -> Result<(), SchedulingError> {
        if group.sequences.is_empty() {
            return Err(SchedulingError::InvalidRequest(
                "sequence group must contain at least one sequence".to_owned(),
            ));
        }
        let running_seqs: usize = self.sequence_groups.iter().map(|g| g.running_count()).sum();
        if running_seqs + group.sequences.len() > self.config.max_num_seqs {
            return Err(SchedulingError::CapacityExceeded);
        }
        self.sequence_groups.push(group);
        Ok(())
    }

    /// Schedule one autoregressive iteration.
    ///
    /// Returns the [`ScheduleOutput`] describing which groups to execute.
    pub fn schedule_iteration(&mut self) -> ScheduleOutput {
        let mut output = ScheduleOutput::default();
        let mut remaining_tokens = self.config.max_num_batched_tokens;
        let mut remaining_seqs = self.config.max_num_seqs;

        // Phase 1 – promote decode (Running) groups first.
        for group in self.sequence_groups.iter_mut() {
            if group.is_all_finished() {
                continue;
            }
            let running = group.running_count();
            if running == 0 {
                continue; // no running sequences; skip to prefill phase.
            }
            let group_tokens = group.total_tokens();
            if group_tokens > remaining_tokens || running > remaining_seqs {
                // Not enough budget; try to swap (mark as Swapped).
                for seq in group.sequences.iter_mut() {
                    if seq.status == SeqStatus::Running {
                        seq.status = SeqStatus::Swapped;
                    }
                }
                continue;
            }
            remaining_tokens = remaining_tokens.saturating_sub(group_tokens);
            remaining_seqs = remaining_seqs.saturating_sub(running);
            output.scheduled_groups.push(group.group_id.clone());
            output.total_tokens += group_tokens;
            if group.is_prefill() {
                output.prefill_groups.push(group.group_id.clone());
            } else {
                output.decode_groups.push(group.group_id.clone());
            }
        }

        // Phase 2 – admit Waiting groups for prefill, FIFO order.
        // Sort waiting groups by creation time so oldest are admitted first.
        let mut waiting_indices: Vec<usize> = self
            .sequence_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| g.waiting_count() > 0 && !g.is_all_finished())
            .map(|(i, _)| i)
            .collect();
        waiting_indices.sort_by_key(|&i| self.sequence_groups[i].created_at);

        for idx in waiting_indices {
            let group = &mut self.sequence_groups[idx];
            let waiting = group.waiting_count();
            if waiting == 0 {
                continue;
            }
            // Each waiting sequence contributes its prompt token count.
            let prompt_tokens: usize = group
                .sequences
                .iter()
                .filter(|s| s.status == SeqStatus::Waiting)
                .map(|s| s.prompt_tokens.len())
                .sum();

            // KV-cache block requirement.
            let needed_blocks: usize = group
                .sequences
                .iter()
                .filter(|s| s.status == SeqStatus::Waiting)
                .map(|s| s.required_kv_blocks(self.config.block_size))
                .sum();

            if prompt_tokens > remaining_tokens
                || waiting > remaining_seqs
                || needed_blocks > self.free_kv_blocks
            {
                continue;
            }

            // Admit.
            for seq in group.sequences.iter_mut() {
                if seq.status == SeqStatus::Waiting {
                    seq.status = SeqStatus::Running;
                    seq.num_kv_blocks = seq.required_kv_blocks(self.config.block_size);
                }
            }
            self.free_kv_blocks = self.free_kv_blocks.saturating_sub(needed_blocks);
            remaining_tokens = remaining_tokens.saturating_sub(prompt_tokens);
            remaining_seqs = remaining_seqs.saturating_sub(waiting);
            output.scheduled_groups.push(group.group_id.clone());
            output.total_tokens += prompt_tokens;
            output.prefill_groups.push(group.group_id.clone());
        }

        output
    }

    /// Remove all fully-finished sequence groups and reclaim their KV blocks.
    ///
    /// Returns the number of groups freed.
    pub fn free_finished_sequences(&mut self) -> usize {
        let mut freed = 0usize;
        self.sequence_groups.retain(|group| {
            if group.is_all_finished() {
                // Reclaim KV-cache blocks.
                let blocks: usize = group.sequences.iter().map(|s| s.num_kv_blocks).sum();
                self.free_kv_blocks = self.free_kv_blocks.saturating_add(blocks);
                freed += 1;
                false
            } else {
                true
            }
        });
        freed
    }

    /// Number of sequence groups currently tracked.
    pub fn group_count(&self) -> usize {
        self.sequence_groups.len()
    }

    /// Number of free KV-cache blocks.
    pub fn free_kv_blocks(&self) -> usize {
        self.free_kv_blocks
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    // ── helpers ────────────────────────────────────────────────────────────

    fn make_request(id: &str, priority: u8, input: usize, output: usize) -> ScheduledRequest {
        ScheduledRequest::new(id, "model_a", input, output, priority)
    }

    fn default_wc_scheduler() -> WorkConservingScheduler {
        WorkConservingScheduler::new(WorkConservingConfig::default())
    }

    fn make_seq_group(id: &str, prompt_len: usize, num_seqs: u64) -> SequenceGroup {
        let seqs = (0..num_seqs).map(|i| Sequence::new(i, vec![1u32; prompt_len])).collect();
        SequenceGroup::new(id, seqs, SamplingParams::default())
    }

    // ── WorkConservingScheduler: basic lifecycle ───────────────────────────

    #[test]
    fn test_wc_enqueue_and_count() {
        let mut sched = default_wc_scheduler();
        sched.enqueue(make_request("r1", 128, 100, 50)).unwrap();
        sched.enqueue(make_request("r2", 64, 200, 100)).unwrap();
        assert_eq!(sched.pending_count(), 2);
        assert_eq!(sched.running_count(), 0);
    }

    #[test]
    fn test_wc_enqueue_queue_full() {
        let mut sched = WorkConservingScheduler::new(WorkConservingConfig {
            max_pending_depth: 2,
            ..Default::default()
        });
        sched.enqueue(make_request("r1", 10, 10, 10)).unwrap();
        sched.enqueue(make_request("r2", 10, 10, 10)).unwrap();
        let err = sched.enqueue(make_request("r3", 10, 10, 10));
        assert_eq!(err, Err(SchedulingError::QueueFull));
    }

    #[test]
    fn test_wc_enqueue_invalid_empty_tokens() {
        let mut sched = default_wc_scheduler();
        let err = sched.enqueue(make_request("r0", 10, 0, 0));
        assert_eq!(
            err,
            Err(SchedulingError::InvalidRequest(
                "request must have at least one token".to_owned()
            ))
        );
    }

    #[test]
    fn test_wc_next_request_highest_priority() {
        let mut sched = default_wc_scheduler();
        sched.enqueue(make_request("low", 10, 50, 50)).unwrap();
        sched.enqueue(make_request("high", 200, 50, 50)).unwrap();
        sched.enqueue(make_request("mid", 100, 50, 50)).unwrap();

        let first = sched.next_request().expect("should schedule");
        assert_eq!(first.request_id, "high");
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.pending_count(), 2);
    }

    #[test]
    fn test_wc_next_request_respects_running_limit() {
        let mut sched = WorkConservingScheduler::new(WorkConservingConfig {
            max_running_requests: 1,
            ..Default::default()
        });
        sched.enqueue(make_request("r1", 10, 10, 10)).unwrap();
        sched.enqueue(make_request("r2", 10, 10, 10)).unwrap();
        sched.next_request();
        assert_eq!(sched.running_count(), 1);
        // Second call must return None (limit hit).
        assert!(sched.next_request().is_none());
    }

    #[test]
    fn test_wc_next_request_respects_token_budget() {
        let mut sched = WorkConservingScheduler::new(WorkConservingConfig {
            max_tokens_in_flight: 100,
            ..Default::default()
        });
        // First request occupies 90 tokens.
        sched.enqueue(make_request("r1", 10, 50, 40)).unwrap();
        // Second request needs 80 tokens — won't fit.
        sched.enqueue(make_request("r2", 10, 40, 40)).unwrap();
        sched.next_request(); // admits r1
                              // r2 cannot be admitted (90+80 = 170 > 100).
        assert!(sched.next_request().is_none());
    }

    #[test]
    fn test_wc_complete_request_updates_stats() {
        let mut sched = default_wc_scheduler();
        sched.enqueue(make_request("r1", 10, 50, 50)).unwrap();
        sched.next_request();
        sched.complete_request("r1", 42).unwrap();
        assert_eq!(sched.stats().total_completed, 1);
        assert_eq!(sched.running_count(), 0);
    }

    #[test]
    fn test_wc_complete_request_not_found() {
        let mut sched = default_wc_scheduler();
        let err = sched.complete_request("nonexistent", 0);
        assert_eq!(err, Err(SchedulingError::RequestNotFound));
    }

    #[test]
    fn test_wc_complete_request_slo_violation() {
        let mut sched = default_wc_scheduler();
        // Deadline in the past.
        let mut req = make_request("r1", 10, 10, 10);
        req.deadline = Some(Instant::now() - Duration::from_secs(1));
        sched.enqueue(req).unwrap();
        sched.next_request();
        sched.complete_request("r1", 5).unwrap();
        assert_eq!(sched.stats().slo_violation_count, 1);
    }

    #[test]
    fn test_wc_complete_request_no_slo_violation_future_deadline() {
        let mut sched = default_wc_scheduler();
        let mut req = make_request("r1", 10, 10, 10);
        req.deadline = Some(Instant::now() + Duration::from_secs(60));
        sched.enqueue(req).unwrap();
        sched.next_request();
        sched.complete_request("r1", 5).unwrap();
        assert_eq!(sched.stats().slo_violation_count, 0);
    }

    #[test]
    fn test_wc_aged_priority_increases_with_wait() {
        let sched = WorkConservingScheduler::new(WorkConservingConfig {
            priority_aging_factor: 10.0,
            ..Default::default()
        });
        let req = ScheduledRequest {
            request_id: "r".into(),
            model_id: "m".into(),
            input_tokens: 10,
            max_output_tokens: 10,
            priority: 0,
            deadline: None,
            // simulate a request that has been waiting 1 second
            enqueued_at: Instant::now() - Duration::from_secs(1),
        };
        let ap = sched.aged_priority(&req);
        // 0 + 1.0 * 10.0 = ~10.0
        assert!(
            ap >= 9.0,
            "aged priority should reflect elapsed time, got {ap}"
        );
    }

    #[test]
    fn test_wc_check_preemption_no_preemption_policy() {
        let mut sched = WorkConservingScheduler::new(WorkConservingConfig {
            preemption_policy: PreemptionPolicy::NoPreemption,
            ..Default::default()
        });
        sched.enqueue(make_request("r1", 10, 10, 10)).unwrap();
        sched.next_request();
        let preempted = sched.check_preemption();
        assert!(preempted.is_empty());
    }

    #[test]
    fn test_wc_check_preemption_priority_based_triggers_when_queue_full() {
        let mut sched = WorkConservingScheduler::new(WorkConservingConfig {
            max_pending_depth: 2,
            preemption_policy: PreemptionPolicy::PriorityBased,
            ..Default::default()
        });
        // Run a low-priority request.
        sched.enqueue(make_request("low", 1, 10, 10)).unwrap();
        sched.next_request();
        // Fill the pending queue with high-priority requests.
        sched.enqueue(make_request("hp1", 255, 10, 10)).unwrap();
        sched.enqueue(make_request("hp2", 255, 10, 10)).unwrap();
        let preempted = sched.check_preemption();
        assert_eq!(preempted.len(), 1);
        assert_eq!(preempted[0].request_id, "low");
        assert_eq!(sched.stats().total_preempted, 1);
    }

    #[test]
    fn test_wc_tokens_released_after_complete() {
        let mut sched = default_wc_scheduler();
        sched.enqueue(make_request("r1", 10, 40, 60)).unwrap();
        sched.next_request();
        assert_eq!(sched.tokens_in_flight(), 100);
        sched.complete_request("r1", 60).unwrap();
        assert_eq!(sched.tokens_in_flight(), 0);
    }

    // ── SchedulingError display ─────────────────────────────────────────────

    #[test]
    fn test_scheduling_error_display() {
        assert!(SchedulingError::QueueFull.to_string().contains("full"));
        assert!(SchedulingError::RequestNotFound.to_string().contains("not found"));
        assert!(SchedulingError::CapacityExceeded.to_string().contains("capacity"));
        assert!(SchedulingError::InvalidRequest("bad".into()).to_string().contains("bad"));
    }

    // ── Sequence & SequenceGroup ────────────────────────────────────────────

    #[test]
    fn test_sequence_new_and_total_len() {
        let seq = Sequence::new(1, vec![10u32, 20u32, 30u32]);
        assert_eq!(seq.total_len(), 3);
        assert_eq!(seq.status, SeqStatus::Waiting);
    }

    #[test]
    fn test_sequence_required_kv_blocks() {
        let mut seq = Sequence::new(1, vec![0u32; 17]);
        seq.output_tokens = vec![0u32; 3]; // total 20 tokens
                                           // block_size=16 → ceil(20/16) = 2
        assert_eq!(seq.required_kv_blocks(16), 2);
        // block_size=32 → ceil(20/32) = 1
        assert_eq!(seq.required_kv_blocks(32), 1);
    }

    #[test]
    fn test_sequence_group_is_prefill() {
        let mut g = make_seq_group("g1", 10, 1);
        g.sequences[0].status = SeqStatus::Running;
        assert!(g.is_prefill()); // no output tokens yet
        g.sequences[0].output_tokens.push(99);
        assert!(!g.is_prefill());
    }

    #[test]
    fn test_sequence_group_all_finished() {
        let mut g = make_seq_group("g1", 5, 2);
        assert!(!g.is_all_finished());
        for s in g.sequences.iter_mut() {
            s.status = SeqStatus::Finished;
        }
        assert!(g.is_all_finished());
    }

    // ── IterationLevelScheduler ─────────────────────────────────────────────

    #[test]
    fn test_ils_add_sequence_group() {
        let mut sched = IterationLevelScheduler::new(IterationConfig::default());
        let g = make_seq_group("g1", 10, 2);
        sched.add_sequence_group(g).unwrap();
        assert_eq!(sched.group_count(), 1);
    }

    #[test]
    fn test_ils_add_empty_group_fails() {
        let mut sched = IterationLevelScheduler::new(IterationConfig::default());
        let g = SequenceGroup::new("empty", vec![], SamplingParams::default());
        let err = sched.add_sequence_group(g);
        assert!(matches!(err, Err(SchedulingError::InvalidRequest(_))));
    }

    #[test]
    fn test_ils_schedule_iteration_admits_waiting_group() {
        let mut sched = IterationLevelScheduler::new(IterationConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 1024,
            block_size: 16,
            total_kv_blocks: 256,
        });
        let g = make_seq_group("g1", 10, 1);
        sched.add_sequence_group(g).unwrap();
        let out = sched.schedule_iteration();
        assert!(out.scheduled_groups.contains(&"g1".to_owned()));
        assert!(out.prefill_groups.contains(&"g1".to_owned()));
    }

    #[test]
    fn test_ils_schedule_iteration_token_budget_respected() {
        let mut sched = IterationLevelScheduler::new(IterationConfig {
            max_num_seqs: 256,
            max_num_batched_tokens: 5, // tiny budget
            block_size: 16,
            total_kv_blocks: 4096,
        });
        // Group with 10 prompt tokens — won't fit.
        let g = make_seq_group("big", 10, 1);
        sched.add_sequence_group(g).unwrap();
        let out = sched.schedule_iteration();
        assert!(!out.scheduled_groups.contains(&"big".to_owned()));
    }

    #[test]
    fn test_ils_free_finished_sequences_reclaims_blocks() {
        let mut sched = IterationLevelScheduler::new(IterationConfig {
            max_num_seqs: 8,
            max_num_batched_tokens: 1024,
            block_size: 16,
            total_kv_blocks: 64,
        });
        let g = make_seq_group("g1", 16, 1);
        sched.add_sequence_group(g).unwrap();
        sched.schedule_iteration(); // admit

        // Mark all sequences as finished.
        for group in sched.sequence_groups.iter_mut() {
            for seq in group.sequences.iter_mut() {
                seq.status = SeqStatus::Finished;
            }
        }
        let before = sched.free_kv_blocks();
        let freed_count = sched.free_finished_sequences();
        assert_eq!(freed_count, 1);
        assert_eq!(sched.group_count(), 0);
        assert!(sched.free_kv_blocks() >= before);
    }

    #[test]
    fn test_ils_schedule_separates_prefill_decode() {
        let mut sched = IterationLevelScheduler::new(IterationConfig {
            max_num_seqs: 16,
            max_num_batched_tokens: 2048,
            block_size: 16,
            total_kv_blocks: 512,
        });
        // Group 1: already has output tokens → decode.
        let mut g1 = make_seq_group("decode_group", 8, 1);
        g1.sequences[0].status = SeqStatus::Running;
        g1.sequences[0].output_tokens = vec![1, 2, 3];
        sched.sequence_groups.push(g1);

        // Group 2: no output tokens → prefill.
        let g2 = make_seq_group("prefill_group", 8, 1);
        sched.add_sequence_group(g2).unwrap();

        let out = sched.schedule_iteration();
        assert!(out.decode_groups.contains(&"decode_group".to_owned()));
        assert!(out.prefill_groups.contains(&"prefill_group".to_owned()));
    }

    #[test]
    fn test_ils_schedule_output_total_tokens_correct() {
        let mut sched = IterationLevelScheduler::new(IterationConfig::default());
        let g = make_seq_group("g1", 20, 1);
        sched.add_sequence_group(g).unwrap();
        let out = sched.schedule_iteration();
        // Group has 20 prompt tokens; total_tokens should be 20.
        assert_eq!(out.total_tokens, 20);
    }
}
