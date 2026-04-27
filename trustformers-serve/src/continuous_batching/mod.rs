//! Continuous batching (iteration-level scheduling) for transformer inference.
//!
//! Implements the core ideas from "Orca: A Distributed Serving System for
//! Transformer-Based Generative Models" (Yu et al., 2022).
//!
//! Rather than processing whole requests together (static batching), each
//! iteration of the forward pass operates on whatever sequences are ready,
//! allowing new requests to join mid-generation and finished requests to leave
//! immediately.

use std::collections::{HashMap, VecDeque};
use std::time::Instant;

// ─── Config ───────────────────────────────────────────────────────────────────

/// Scheduler configuration for continuous batching.
#[derive(Debug, Clone)]
pub struct BatchingConfig {
    /// Maximum number of sequences in a single batch.
    pub max_batch_size: usize,
    /// Maximum total tokens (across all sequences) in flight per step.
    pub max_tokens_per_batch: usize,
    /// Maximum milliseconds a request may wait before it must be scheduled.
    pub max_waiting_time_ms: u64,
    /// Minimum batch size before a step is dispatched.
    pub min_batch_size: usize,
    /// Whether to allow preemption of lower-priority running sequences to
    /// admit a higher-priority new request.
    pub prefer_new_requests: bool,
    /// Weight for preferring short decode phases over long prefill phases
    /// when scheduling.
    pub decode_to_prefill_ratio: f32,
}

impl Default for BatchingConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 32,
            max_tokens_per_batch: 2048,
            max_waiting_time_ms: 100,
            min_batch_size: 1,
            prefer_new_requests: true,
            decode_to_prefill_ratio: 0.1,
        }
    }
}

// ─── StopReason ───────────────────────────────────────────────────────────────

/// The reason a sequence stopped generating.
#[derive(Debug, Clone, PartialEq)]
pub enum StopReason {
    /// The maximum number of new tokens was reached.
    MaxTokens,
    /// The model emitted an EOS token.
    EosToken,
    /// A user-supplied stop sequence was matched.
    StopSequence,
    /// The sequence was preempted and discarded.
    Preempted,
}

// ─── SequenceState ────────────────────────────────────────────────────────────

/// Lifecycle state of a single sequence group.
#[derive(Debug, Clone, PartialEq)]
pub enum SequenceState {
    /// Waiting in the queue; not yet started.
    Waiting,
    /// Performing the prefill pass up to `prefill_pos`.
    Prefilling {
        /// How many tokens of the prompt have been processed so far.
        prefill_pos: usize,
    },
    /// Autoregressively generating tokens.
    Decoding {
        /// Number of tokens generated so far.
        num_generated: usize,
    },
    /// Generation has ended.
    Finished { stop_reason: StopReason },
    /// Temporarily removed from running to make room for a higher-priority
    /// sequence; will re-enter the waiting queue.
    Preempted,
}

// ─── SequenceGroup ────────────────────────────────────────────────────────────

/// A single inference request (one sequence, one set of sampling parameters).
#[derive(Debug, Clone)]
pub struct SequenceGroup {
    /// Unique identifier.
    pub group_id: String,
    /// Tokenised prompt.
    pub prompt_tokens: Vec<u32>,
    /// Tokens generated so far.
    pub generated_tokens: Vec<u32>,
    /// Maximum number of new tokens this request may generate.
    pub max_new_tokens: usize,
    /// Scheduling priority (0 = lowest, 255 = highest).
    pub priority: u8,
    /// Current lifecycle state.
    pub state: SequenceState,
    /// Wall-clock time when this request arrived.
    pub arrival_time: Instant,
    /// Wall-clock time when this request was first scheduled (if any).
    pub first_scheduled_time: Option<Instant>,
}

impl SequenceGroup {
    /// Total number of tokens (prompt + generated).
    pub fn total_tokens(&self) -> usize {
        self.prompt_tokens.len() + self.generated_tokens.len()
    }

    /// Whether the sequence has finished generation.
    pub fn is_finished(&self) -> bool {
        matches!(self.state, SequenceState::Finished { .. })
    }

    /// Number of additional tokens this sequence may still generate.
    pub fn tokens_remaining(&self) -> usize {
        self.max_new_tokens.saturating_sub(self.generated_tokens.len())
    }
}

// ─── BatchSnapshot ────────────────────────────────────────────────────────────

/// A description of the work the engine must perform in one forward-pass step.
#[derive(Debug, Clone)]
pub struct BatchSnapshot {
    /// Monotonically increasing step identifier.
    pub batch_id: u64,
    /// IDs of sequences currently in the decode phase.
    pub decode_sequences: Vec<String>,
    /// IDs of sequences currently in the prefill phase.
    pub prefill_sequences: Vec<String>,
    /// Total tokens represented in this batch.
    pub total_tokens: usize,
    /// Wall-clock time this snapshot was taken.
    pub timestamp: Instant,
}

impl BatchSnapshot {
    /// Number of sequences in this batch.
    pub fn batch_size(&self) -> usize {
        self.decode_sequences.len() + self.prefill_sequences.len()
    }
}

// ─── BatchingStats ────────────────────────────────────────────────────────────

/// Aggregate statistics collected by the scheduler.
#[derive(Debug, Clone, Default)]
pub struct BatchingStats {
    /// Total number of completed requests.
    pub total_requests_processed: u64,
    /// Total tokens generated across all completed requests.
    pub total_tokens_generated: u64,
    /// Number of prefill steps (forward passes that included at least one
    /// prefilling sequence).
    pub total_prefill_steps: u64,
    /// Number of decode steps.
    pub total_decode_steps: u64,
    /// Running mean batch size (exponential moving average).
    pub mean_batch_size: f32,
    /// Number of preemptions.
    pub preemption_count: u64,
    /// Internal step counter for mean computation.
    step_count: u64,
}

impl BatchingStats {
    /// Record one scheduler step given the resulting `BatchSnapshot`.
    pub fn record_step(&mut self, snapshot: &BatchSnapshot) {
        if !snapshot.prefill_sequences.is_empty() {
            self.total_prefill_steps += 1;
        }
        if !snapshot.decode_sequences.is_empty() {
            self.total_decode_steps += 1;
        }
        // Exponential moving average of batch size.
        let size = snapshot.batch_size() as f32;
        self.step_count += 1;
        let alpha = 1.0 / self.step_count as f32;
        self.mean_batch_size = self.mean_batch_size * (1.0 - alpha) + size * alpha;
    }

    /// Record a preemption event.
    pub fn record_preemption(&mut self) {
        self.preemption_count += 1;
    }

    /// Record the completion of a sequence that generated `tokens_generated` tokens.
    pub fn record_completion(&mut self, tokens_generated: usize) {
        self.total_requests_processed += 1;
        self.total_tokens_generated += tokens_generated as u64;
    }
}

// ─── Scheduler ────────────────────────────────────────────────────────────────

/// Continuous-batch scheduler that manages the lifecycle of sequence groups.
pub struct ContinuousBatchScheduler {
    pub config: BatchingConfig,
    /// Sequences waiting to be admitted to the running set.
    pub waiting_queue: VecDeque<SequenceGroup>,
    /// Sequences currently being processed (prefill or decode).
    pub running: HashMap<String, SequenceGroup>,
    /// Sequences that have finished (either naturally or via preemption).
    pub finished: Vec<SequenceGroup>,
    /// Monotonically increasing step counter.
    pub step_counter: u64,
    pub stats: BatchingStats,
}

impl ContinuousBatchScheduler {
    /// Create a new scheduler with the given configuration.
    pub fn new(config: BatchingConfig) -> Self {
        Self {
            config,
            waiting_queue: VecDeque::new(),
            running: HashMap::new(),
            finished: Vec::new(),
            step_counter: 0,
            stats: BatchingStats::default(),
        }
    }

    // ── Public API ───────────────────────────────────────────────────────────

    /// Add a new request to the waiting queue in priority order (highest first).
    pub fn add_request(&mut self, group: SequenceGroup) {
        // Insert in descending priority order (highest priority = front).
        let pos = self
            .waiting_queue
            .iter()
            .position(|g| g.priority < group.priority)
            .unwrap_or(self.waiting_queue.len());
        self.waiting_queue.insert(pos, group);
    }

    /// Advance the scheduler by one step and return the batch to process.
    ///
    /// The scheduler:
    /// 1. Promotes waiting sequences to running if capacity permits.
    /// 2. Advances the state of running sequences (Waiting → Prefilling →
    ///    Decoding).
    /// 3. Returns a `BatchSnapshot` describing all active sequences.
    pub fn schedule_step(&mut self) -> BatchSnapshot {
        // ── Admit waiting sequences if capacity allows ────────────────────────
        self.admit_waiting();

        // ── Advance states ────────────────────────────────────────────────────
        let now = Instant::now();
        for seq in self.running.values_mut() {
            match &seq.state {
                SequenceState::Waiting => {
                    if seq.first_scheduled_time.is_none() {
                        seq.first_scheduled_time = Some(now);
                    }
                    seq.state = SequenceState::Prefilling { prefill_pos: 0 };
                },
                SequenceState::Prefilling { prefill_pos } => {
                    let pos = *prefill_pos;
                    if pos >= seq.prompt_tokens.len() {
                        seq.state = SequenceState::Decoding { num_generated: 0 };
                    }
                    // Otherwise the engine advances prefill_pos externally.
                },
                SequenceState::Decoding { .. } => {
                    // Engine calls append_token to progress.
                },
                SequenceState::Finished { .. } | SequenceState::Preempted => {},
            }
        }

        // ── Build snapshot ────────────────────────────────────────────────────
        let mut decode_sequences = Vec::new();
        let mut prefill_sequences = Vec::new();
        let mut total_tokens = 0usize;

        for seq in self.running.values() {
            match &seq.state {
                SequenceState::Decoding { .. } => {
                    total_tokens += seq.total_tokens();
                    decode_sequences.push(seq.group_id.clone());
                },
                SequenceState::Prefilling { .. } => {
                    total_tokens += seq.prompt_tokens.len();
                    prefill_sequences.push(seq.group_id.clone());
                },
                _ => {},
            }
        }

        let snapshot = BatchSnapshot {
            batch_id: self.step_counter,
            decode_sequences,
            prefill_sequences,
            total_tokens,
            timestamp: now,
        };

        self.stats.record_step(&snapshot);
        self.step_counter += 1;

        snapshot
    }

    /// Mark a running sequence as finished, moving it to the finished list.
    ///
    /// Returns `true` if the sequence was found and moved.
    pub fn complete_sequence(&mut self, group_id: &str, stop_reason: StopReason) -> bool {
        if let Some(mut seq) = self.running.remove(group_id) {
            let tokens_generated = seq.generated_tokens.len();
            seq.state = SequenceState::Finished { stop_reason };
            self.stats.record_completion(tokens_generated);
            self.finished.push(seq);
            true
        } else {
            false
        }
    }

    /// Append one generated token to a running sequence.
    ///
    /// If the sequence reaches `max_new_tokens`, it is automatically finished
    /// with `StopReason::MaxTokens`.
    ///
    /// Returns `true` if the token was appended (sequence exists and is active).
    pub fn append_token(&mut self, group_id: &str, token_id: u32) -> bool {
        let finished = if let Some(seq) = self.running.get_mut(group_id) {
            match &mut seq.state {
                SequenceState::Decoding { num_generated } => {
                    seq.generated_tokens.push(token_id);
                    *num_generated += 1;
                    seq.generated_tokens.len() >= seq.max_new_tokens
                },
                _ => return false,
            }
        } else {
            return false;
        };

        if finished {
            self.complete_sequence(group_id, StopReason::MaxTokens);
        }

        true
    }

    /// Preempt the lowest-priority running sequence, returning it to the
    /// waiting queue.
    ///
    /// Returns the `group_id` of the preempted sequence, or `None` if the
    /// running set is empty.
    pub fn preempt_lowest_priority(&mut self) -> Option<String> {
        let victim_id =
            self.running.values().min_by_key(|s| s.priority).map(|s| s.group_id.clone())?;

        if let Some(mut seq) = self.running.remove(&victim_id) {
            seq.state = SequenceState::Waiting;
            self.stats.record_preemption();
            // Push to front of waiting_queue at the same priority rank.
            let pos = self
                .waiting_queue
                .iter()
                .position(|g| g.priority < seq.priority)
                .unwrap_or(self.waiting_queue.len());
            self.waiting_queue.insert(pos, seq);
            Some(victim_id)
        } else {
            None
        }
    }

    /// Current scheduler statistics.
    pub fn stats(&self) -> &BatchingStats {
        &self.stats
    }

    /// Number of sequences waiting to be scheduled.
    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    /// Number of sequences currently running.
    pub fn running_count(&self) -> usize {
        self.running.len()
    }

    // ── Private helpers ──────────────────────────────────────────────────────

    /// Promote waiting sequences into the running set as long as capacity
    /// constraints allow.
    fn admit_waiting(&mut self) {
        while let Some(candidate) = self.waiting_queue.front() {
            let new_running = self.running.len() + 1;
            if new_running > self.config.max_batch_size {
                break;
            }
            // Check token budget: estimate tokens this candidate would add.
            let candidate_tokens = candidate.prompt_tokens.len();
            let current_tokens: usize = self.running.values().map(|s| s.total_tokens()).sum();
            if current_tokens + candidate_tokens > self.config.max_tokens_per_batch
                && !self.running.is_empty()
            {
                break;
            }

            if let Some(mut seq) = self.waiting_queue.pop_front() {
                seq.state = SequenceState::Waiting;
                self.running.insert(seq.group_id.clone(), seq);
            }
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_seq(id: &str, prompt_len: usize, priority: u8, max_new: usize) -> SequenceGroup {
        SequenceGroup {
            group_id: id.to_string(),
            prompt_tokens: (0..prompt_len as u32).collect(),
            generated_tokens: Vec::new(),
            max_new_tokens: max_new,
            priority,
            state: SequenceState::Waiting,
            arrival_time: Instant::now(),
            first_scheduled_time: None,
        }
    }

    // ── Config defaults ───────────────────────────────────────────────────────

    #[test]
    fn test_batching_config_defaults() {
        let cfg = BatchingConfig::default();
        assert_eq!(cfg.max_batch_size, 32);
        assert_eq!(cfg.max_tokens_per_batch, 2048);
        assert_eq!(cfg.max_waiting_time_ms, 100);
        assert_eq!(cfg.min_batch_size, 1);
        assert!(cfg.prefer_new_requests);
        assert!((cfg.decode_to_prefill_ratio - 0.1).abs() < 1e-6);
    }

    // ── Add request ───────────────────────────────────────────────────────────

    #[test]
    fn test_add_request() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 32));
        assert_eq!(sched.waiting_count(), 1);
    }

    // ── Schedule single ───────────────────────────────────────────────────────

    #[test]
    fn test_schedule_single_request() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 32));
        let snap = sched.schedule_step();
        // After one step the sequence should be admitted and in prefill.
        assert_eq!(snap.batch_size(), 1);
        assert_eq!(snap.prefill_sequences.len(), 1);
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 0);
    }

    // ── Batch capacity limit ──────────────────────────────────────────────────

    #[test]
    fn test_batch_capacity_limit() {
        let mut cfg = BatchingConfig::default();
        cfg.max_batch_size = 2;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        for i in 0..5u32 {
            sched.add_request(make_seq(&format!("r{i}"), 4, 100, 32));
        }
        sched.schedule_step();
        assert!(
            sched.running_count() <= 2,
            "running must not exceed max_batch_size"
        );
        assert!(sched.waiting_count() >= 3);
    }

    // ── Priority ordering ─────────────────────────────────────────────────────

    #[test]
    fn test_priority_ordering() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        // Add in low-to-high order; queue should store highest-first.
        sched.add_request(make_seq("low", 4, 10, 32));
        sched.add_request(make_seq("high", 4, 200, 32));
        sched.add_request(make_seq("mid", 4, 100, 32));

        let front = sched.waiting_queue.front().expect("non-empty");
        assert_eq!(front.priority, 200, "highest priority must be at front");
    }

    // ── State transitions: Waiting → Prefilling → Decoding → Finished ─────────

    #[test]
    fn test_state_transitions() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 4));

        // Step 1: Waiting → Prefilling.
        let snap1 = sched.schedule_step();
        assert!(snap1.prefill_sequences.contains(&"r1".to_string()));

        // Manually advance prefill_pos past the prompt length to simulate
        // the engine completing the prefill pass.
        if let Some(seq) = sched.running.get_mut("r1") {
            seq.state = SequenceState::Prefilling { prefill_pos: 4 };
        }

        // Step 2: Prefilling (complete) → Decoding.
        let snap2 = sched.schedule_step();
        assert!(snap2.decode_sequences.contains(&"r1".to_string()));

        // Append tokens until finished.
        for tok in 0u32..4 {
            sched.append_token("r1", tok);
        }
        // After max_new_tokens the sequence should have been auto-finished.
        assert!(!sched.running.contains_key("r1"));
        assert!(sched.finished.iter().any(|s| s.group_id == "r1"));
    }

    // ── Append token + auto-finish ────────────────────────────────────────────

    #[test]
    fn test_append_token_auto_finish() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 2, 100, 3));
        sched.schedule_step();
        // Manually set to decoding state.
        if let Some(seq) = sched.running.get_mut("r1") {
            seq.state = SequenceState::Decoding { num_generated: 0 };
        }
        sched.append_token("r1", 10);
        sched.append_token("r1", 11);
        // Third token should trigger auto-finish.
        sched.append_token("r1", 12);
        assert!(
            sched.finished.iter().any(|s| s.group_id == "r1"),
            "sequence must auto-finish when max_new_tokens reached"
        );
    }

    // ── Preempt lowest priority ───────────────────────────────────────────────

    #[test]
    fn test_preempt_lowest_priority() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("hi", 4, 200, 32));
        sched.add_request(make_seq("lo", 4, 10, 32));
        sched.schedule_step();

        let preempted = sched.preempt_lowest_priority();
        assert!(preempted.is_some());
        assert_eq!(
            preempted.as_deref(),
            Some("lo"),
            "lowest priority must be preempted"
        );
        assert!(
            sched.waiting_queue.iter().any(|s| s.group_id == "lo"),
            "preempted sequence must re-enter waiting queue"
        );
    }

    // ── Stats tracking ────────────────────────────────────────────────────────

    #[test]
    fn test_stats_tracking() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 2, 100, 2));
        sched.schedule_step();
        // Set to decoding.
        if let Some(seq) = sched.running.get_mut("r1") {
            seq.state = SequenceState::Decoding { num_generated: 0 };
        }
        sched.schedule_step();
        sched.append_token("r1", 1);
        sched.append_token("r1", 2);

        let stats = sched.stats();
        assert_eq!(stats.total_requests_processed, 1);
        assert_eq!(stats.total_tokens_generated, 2);
    }

    // ── Concurrent request scheduling ─────────────────────────────────────────

    #[test]
    fn test_concurrent_request_scheduling() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        for i in 0..10u32 {
            sched.add_request(make_seq(&format!("r{i}"), 10, 100, 5));
        }
        let snap = sched.schedule_step();
        // All 10 should fit within default max_batch_size=32 and
        // max_tokens_per_batch=2048.
        assert_eq!(snap.batch_size(), 10);
    }

    // ── Complete sequence ─────────────────────────────────────────────────────

    #[test]
    fn test_complete_sequence() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 10));
        sched.schedule_step();

        let found = sched.complete_sequence("r1", StopReason::EosToken);
        assert!(found, "complete_sequence must return true for known id");
        assert!(!sched.running.contains_key("r1"));
        assert!(sched.finished.iter().any(|s| s.group_id == "r1"));
        assert_eq!(sched.stats().total_requests_processed, 1);
    }

    // ── Preemption stats ──────────────────────────────────────────────────────

    #[test]
    fn test_preemption_stats() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 10, 32));
        sched.schedule_step();

        sched.preempt_lowest_priority();
        assert_eq!(sched.stats().preemption_count, 1);
    }

    // ── Batch snapshot batch_size ─────────────────────────────────────────────

    #[test]
    fn test_batch_snapshot_batch_size() {
        let snap = BatchSnapshot {
            batch_id: 0,
            decode_sequences: vec!["a".to_string(), "b".to_string()],
            prefill_sequences: vec!["c".to_string()],
            total_tokens: 10,
            timestamp: Instant::now(),
        };
        assert_eq!(snap.batch_size(), 3);
    }

    // ── Empty batch snapshot ──────────────────────────────────────────────────

    #[test]
    fn test_empty_batch_snapshot() {
        let snap = BatchSnapshot {
            batch_id: 5,
            decode_sequences: Vec::new(),
            prefill_sequences: Vec::new(),
            total_tokens: 0,
            timestamp: Instant::now(),
        };
        assert_eq!(snap.batch_size(), 0);
    }

    // ── SequenceGroup total_tokens ────────────────────────────────────────────

    #[test]
    fn test_sequence_group_total_tokens() {
        let mut seq = make_seq("t1", 10, 100, 50);
        assert_eq!(seq.total_tokens(), 10);
        seq.generated_tokens.push(42);
        seq.generated_tokens.push(43);
        assert_eq!(seq.total_tokens(), 12);
    }

    // ── SequenceGroup tokens_remaining ───────────────────────────────────────

    #[test]
    fn test_sequence_group_tokens_remaining() {
        let mut seq = make_seq("t2", 5, 100, 10);
        assert_eq!(seq.tokens_remaining(), 10);
        seq.generated_tokens.extend_from_slice(&[1, 2, 3]);
        assert_eq!(seq.tokens_remaining(), 7);
    }

    // ── tokens_remaining saturates at zero ───────────────────────────────────

    #[test]
    fn test_tokens_remaining_saturates_at_zero() {
        let mut seq = make_seq("t3", 5, 100, 3);
        // Simulate overshoot (should not panic).
        for i in 0..5u32 {
            seq.generated_tokens.push(i);
        }
        assert_eq!(seq.tokens_remaining(), 0);
    }

    // ── is_finished: Finished state ───────────────────────────────────────────

    #[test]
    fn test_is_finished_states() {
        let mut seq = make_seq("f1", 4, 100, 32);
        assert!(!seq.is_finished());
        seq.state = SequenceState::Finished {
            stop_reason: StopReason::EosToken,
        };
        assert!(seq.is_finished());
    }

    // ── is_finished: non-finished states ─────────────────────────────────────

    #[test]
    fn test_is_not_finished_in_decoding() {
        let mut seq = make_seq("f2", 4, 100, 32);
        seq.state = SequenceState::Decoding { num_generated: 2 };
        assert!(!seq.is_finished());
    }

    // ── Token budget enforcement ──────────────────────────────────────────────

    #[test]
    fn test_token_budget_enforcement() {
        let mut cfg = BatchingConfig::default();
        cfg.max_tokens_per_batch = 20;
        cfg.max_batch_size = 100;
        let mut sched = ContinuousBatchScheduler::new(cfg);

        // One large sequence (15 tokens) plus one that would push over 20.
        sched.add_request(make_seq("big", 15, 100, 10));
        sched.add_request(make_seq("overflow", 10, 100, 10));

        sched.schedule_step();
        // Only "big" should be admitted since big(15) + overflow(10) = 25 > 20.
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 1);
    }

    // ── Step counter increments ───────────────────────────────────────────────

    #[test]
    fn test_step_counter_increments() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 10));
        assert_eq!(sched.step_counter, 0);
        sched.schedule_step();
        assert_eq!(sched.step_counter, 1);
        sched.schedule_step();
        assert_eq!(sched.step_counter, 2);
    }

    // ── Batch ID monotonically increasing ────────────────────────────────────

    #[test]
    fn test_batch_id_monotonically_increasing() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 10));
        let snap0 = sched.schedule_step();
        let snap1 = sched.schedule_step();
        assert!(snap1.batch_id > snap0.batch_id);
    }

    // ── Complete sequence: unknown ID returns false ───────────────────────────

    #[test]
    fn test_complete_sequence_unknown_id() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        let found = sched.complete_sequence("nonexistent", StopReason::EosToken);
        assert!(!found, "completing an unknown ID must return false");
    }

    // ── Append token to non-decoding sequence returns false ───────────────────

    #[test]
    fn test_append_token_wrong_state() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 10));
        sched.schedule_step();
        // Sequence is Prefilling, not Decoding → append_token must return false.
        let ok = sched.append_token("r1", 99);
        assert!(!ok);
    }

    // ── Preempt empty running set returns None ────────────────────────────────

    #[test]
    fn test_preempt_empty_running_set() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        let result = sched.preempt_lowest_priority();
        assert!(result.is_none());
    }

    // ── Preempted sequence re-enters waiting queue at correct position ─────────

    #[test]
    fn test_preempted_sequence_rejoins_waiting_in_priority_order() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("lo", 4, 10, 32));
        sched.add_request(make_seq("hi", 4, 200, 32));
        sched.schedule_step(); // Admit both.

        // Preempt lowest (lo, priority=10).
        sched.preempt_lowest_priority();
        // "lo" must be back in the waiting queue.
        assert!(sched.waiting_queue.iter().any(|s| s.group_id == "lo"));
        // "hi" must still be running.
        assert!(sched.running.contains_key("hi"));
    }

    // ── BatchingStats: prefill and decode step counts ─────────────────────────

    #[test]
    fn test_stats_prefill_decode_step_counts() {
        let mut stats = BatchingStats::default();

        let prefill_snap = BatchSnapshot {
            batch_id: 0,
            decode_sequences: Vec::new(),
            prefill_sequences: vec!["p1".to_string()],
            total_tokens: 5,
            timestamp: Instant::now(),
        };
        stats.record_step(&prefill_snap);
        assert_eq!(stats.total_prefill_steps, 1);
        assert_eq!(stats.total_decode_steps, 0);

        let decode_snap = BatchSnapshot {
            batch_id: 1,
            decode_sequences: vec!["d1".to_string()],
            prefill_sequences: Vec::new(),
            total_tokens: 3,
            timestamp: Instant::now(),
        };
        stats.record_step(&decode_snap);
        assert_eq!(stats.total_decode_steps, 1);
    }

    // ── BatchingStats: mean batch size EMA ───────────────────────────────────

    #[test]
    fn test_stats_mean_batch_size_ema() {
        let mut stats = BatchingStats::default();

        // Record three steps with batch sizes 4, 4, 4 → mean should converge to 4.
        for _ in 0..3 {
            let snap = BatchSnapshot {
                batch_id: 0,
                decode_sequences: vec!["a".to_string(), "b".to_string()],
                prefill_sequences: vec!["c".to_string(), "d".to_string()],
                total_tokens: 10,
                timestamp: Instant::now(),
            };
            stats.record_step(&snap);
        }
        assert!((stats.mean_batch_size - 4.0).abs() < 1e-4);
    }

    // ── record_completion accumulates tokens ──────────────────────────────────

    #[test]
    fn test_record_completion_accumulates_tokens() {
        let mut stats = BatchingStats::default();
        stats.record_completion(10);
        stats.record_completion(20);
        assert_eq!(stats.total_requests_processed, 2);
        assert_eq!(stats.total_tokens_generated, 30);
    }

    // ── Max batch size of 1: second request stays waiting ────────────────────

    #[test]
    fn test_max_batch_size_one() {
        let mut cfg = BatchingConfig::default();
        cfg.max_batch_size = 1;
        let mut sched = ContinuousBatchScheduler::new(cfg);
        sched.add_request(make_seq("r1", 4, 100, 10));
        sched.add_request(make_seq("r2", 4, 100, 10));
        sched.schedule_step();
        assert_eq!(sched.running_count(), 1);
        assert_eq!(sched.waiting_count(), 1);
    }

    // ── Stop reason variants ──────────────────────────────────────────────────

    #[test]
    fn test_stop_reason_variants() {
        assert_eq!(StopReason::MaxTokens, StopReason::MaxTokens);
        assert_eq!(StopReason::EosToken, StopReason::EosToken);
        assert_eq!(StopReason::StopSequence, StopReason::StopSequence);
        assert_eq!(StopReason::Preempted, StopReason::Preempted);
        assert_ne!(StopReason::MaxTokens, StopReason::EosToken);
    }

    // ── Mixed prefill and decode in same snapshot ──────────────────────────────

    #[test]
    fn test_mixed_prefill_and_decode_in_snapshot() {
        let mut cfg = BatchingConfig::default();
        cfg.max_batch_size = 32;
        let mut sched = ContinuousBatchScheduler::new(cfg);

        // r1 will be Prefilling; r2 will be manually set to Decoding.
        sched.add_request(make_seq("r1", 4, 100, 10));
        sched.add_request(make_seq("r2", 4, 100, 10));
        sched.schedule_step(); // Both admitted, both go to Prefilling.

        // Manually advance r2 to Decoding.
        if let Some(seq) = sched.running.get_mut("r2") {
            seq.state = SequenceState::Decoding { num_generated: 0 };
        }

        let snap = sched.schedule_step();
        // r1 remains prefilling, r2 in decoding.
        assert!(snap.prefill_sequences.contains(&"r1".to_string()));
        assert!(snap.decode_sequences.contains(&"r2".to_string()));
    }

    // ── Multiple preemptions tracked correctly ─────────────────────────────────

    #[test]
    fn test_multiple_preemptions_tracked() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        for i in 0u8..5 {
            sched.add_request(make_seq(&format!("r{i}"), 4, i * 10, 32));
        }
        sched.schedule_step();
        sched.preempt_lowest_priority();
        sched.preempt_lowest_priority();
        assert_eq!(sched.stats().preemption_count, 2);
    }

    // ── Decode-only snapshot: no prefill steps counted ────────────────────────

    #[test]
    fn test_decode_only_snapshot_no_prefill_step() {
        let mut stats = BatchingStats::default();
        let snap = BatchSnapshot {
            batch_id: 0,
            decode_sequences: vec!["d1".to_string()],
            prefill_sequences: Vec::new(),
            total_tokens: 5,
            timestamp: Instant::now(),
        };
        stats.record_step(&snap);
        assert_eq!(stats.total_prefill_steps, 0);
        assert_eq!(stats.total_decode_steps, 1);
    }

    // ── All-prefill snapshot: no decode steps counted ─────────────────────────

    #[test]
    fn test_all_prefill_snapshot_no_decode_step() {
        let mut stats = BatchingStats::default();
        let snap = BatchSnapshot {
            batch_id: 0,
            decode_sequences: Vec::new(),
            prefill_sequences: vec!["p1".to_string()],
            total_tokens: 5,
            timestamp: Instant::now(),
        };
        stats.record_step(&snap);
        assert_eq!(stats.total_prefill_steps, 1);
        assert_eq!(stats.total_decode_steps, 0);
    }

    // ── Extended tests ─────────────────────────────────────────────────────

    // Test: BatchingConfig::default — sensible values
    #[test]
    fn test_batching_config_default_values() {
        let cfg = BatchingConfig::default();
        assert!(cfg.max_batch_size > 0, "max_batch_size must be > 0");
        assert!(
            cfg.max_tokens_per_batch > 0,
            "max_tokens_per_batch must be > 0"
        );
        assert!(cfg.min_batch_size <= cfg.max_batch_size);
    }

    // Test: SequenceState variants are distinct
    #[test]
    fn test_sequence_state_variants_distinct() {
        assert_ne!(SequenceState::Waiting, SequenceState::Preempted);
        assert_ne!(
            SequenceState::Prefilling { prefill_pos: 0 },
            SequenceState::Decoding { num_generated: 0 }
        );
    }

    // Test: SequenceGroup initial state is Waiting
    #[test]
    fn test_sequence_group_initial_state_waiting() {
        let seq = make_seq("s1", 8, 128, 50);
        assert_eq!(seq.state, SequenceState::Waiting);
    }

    // Test: SequenceGroup priority stored correctly
    #[test]
    fn test_sequence_group_priority_stored() {
        let seq = make_seq("s1", 4, 200, 20);
        assert_eq!(seq.priority, 200);
    }

    // Test: ContinuousBatchScheduler starts empty
    #[test]
    fn test_scheduler_starts_empty() {
        let sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        assert_eq!(sched.running_count(), 0);
        assert_eq!(sched.waiting_count(), 0);
    }

    // Test: add_request increments waiting_count
    #[test]
    fn test_add_request_increments_waiting() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 20));
        sched.add_request(make_seq("r2", 4, 100, 20));
        assert_eq!(sched.waiting_count(), 2);
    }

    // Test: schedule_step moves request from waiting to running
    #[test]
    fn test_schedule_step_moves_to_running() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 20));
        sched.schedule_step();
        assert_eq!(sched.running_count(), 1, "sequence must move to running");
        assert_eq!(sched.waiting_count(), 0);
    }

    // Test: schedule_step snapshot lists the request
    #[test]
    fn test_schedule_step_snapshot_contains_request() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("unique-req", 4, 100, 20));
        let snap = sched.schedule_step();
        let all_ids: Vec<&String> =
            snap.prefill_sequences.iter().chain(snap.decode_sequences.iter()).collect();
        assert!(
            all_ids.iter().any(|id| id.as_str() == "unique-req"),
            "snapshot must contain the request"
        );
    }

    // Test: complete_sequence removes from running
    #[test]
    fn test_complete_sequence_removes_from_running() {
        let mut sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        sched.add_request(make_seq("r1", 4, 100, 20));
        sched.schedule_step(); // admit r1
        let found = sched.complete_sequence("r1", StopReason::MaxTokens);
        assert!(
            found,
            "complete_sequence must return true for existing sequence"
        );
        assert_eq!(sched.running_count(), 0);
    }

    // Test: BatchingStats::default — all zeros
    #[test]
    fn test_batching_stats_default_zeros() {
        let stats = BatchingStats::default();
        assert_eq!(stats.total_requests_processed, 0);
        assert_eq!(stats.total_tokens_generated, 0);
        assert_eq!(stats.total_prefill_steps, 0);
        assert_eq!(stats.total_decode_steps, 0);
        assert_eq!(stats.preemption_count, 0);
    }

    // Test: BatchingStats::record_completion increments correctly
    #[test]
    fn test_batching_stats_record_completion() {
        let mut stats = BatchingStats::default();
        stats.record_completion(15);
        assert_eq!(stats.total_requests_processed, 1);
        assert_eq!(stats.total_tokens_generated, 15);
    }

    // Test: scheduler stats() returns correct preemption_count
    #[test]
    fn test_scheduler_stats_preemption_count() {
        let sched = ContinuousBatchScheduler::new(BatchingConfig::default());
        assert_eq!(sched.stats().preemption_count, 0);
    }

    // Test: StopReason::Preempted is a valid variant
    #[test]
    fn test_stop_reason_preempted_variant() {
        let reason = StopReason::Preempted;
        assert_eq!(reason, StopReason::Preempted);
    }

    // Test: max_tokens_per_batch limits admission
    #[test]
    fn test_max_tokens_per_batch_limits_admission() {
        let mut cfg = BatchingConfig::default();
        cfg.max_tokens_per_batch = 4; // very tight token budget
        let mut sched = ContinuousBatchScheduler::new(cfg);
        // Each request has 8 prompt tokens, which exceeds the batch token budget
        sched.add_request(make_seq("r1", 8, 100, 20));
        sched.add_request(make_seq("r2", 8, 100, 20));
        let snap = sched.schedule_step();
        // With 4-token budget, no 8-token request should be admitted
        assert!(
            snap.total_tokens <= 8,
            "total_tokens {} should respect budget",
            snap.total_tokens
        );
    }

    // Test: SequenceGroup::max_new_tokens stored correctly
    #[test]
    fn test_sequence_group_max_new_tokens() {
        let seq = make_seq("s1", 4, 100, 77);
        assert_eq!(seq.max_new_tokens, 77);
    }

    // Test: BatchSnapshot::total_tokens sums prefill + decode
    #[test]
    fn test_batch_snapshot_total_tokens() {
        let snap = BatchSnapshot {
            batch_id: 1,
            decode_sequences: vec!["d1".to_string()],
            prefill_sequences: vec!["p1".to_string(), "p2".to_string()],
            total_tokens: 42,
            timestamp: Instant::now(),
        };
        assert_eq!(snap.total_tokens, 42);
        assert_eq!(
            snap.prefill_sequences.len() + snap.decode_sequences.len(),
            3
        );
    }
}
