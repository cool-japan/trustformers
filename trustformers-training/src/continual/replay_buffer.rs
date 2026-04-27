//! Experience replay buffer and continual learning metrics.
//!
//! Provides:
//!
//! - [`ReplayBuffer`]: Ring / random / reservoir / prioritized replay memory.
//! - [`LateralAdapter`]: PNN lateral connection layer.
//! - [`compute_bwt`], [`compute_fwt`], [`compute_intransigence`]: Transfer metrics.

use std::collections::{HashMap, VecDeque};

// ---------------------------------------------------------------------------
// Replay sample
// ---------------------------------------------------------------------------

/// A single sample stored in the replay buffer.
#[derive(Debug, Clone)]
pub struct ReplaySample {
    /// Token-id input sequence.
    pub input_ids: Vec<u32>,
    /// Ground-truth label token ids.
    pub labels: Vec<u32>,
    /// Which task this sample came from.
    pub task_id: usize,
    /// Importance weight (used by prioritized replay).
    pub importance: f32,
}

// ---------------------------------------------------------------------------
// Replay strategy
// ---------------------------------------------------------------------------

/// Strategy that governs which samples are kept / evicted when the buffer
/// is full.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReplayStrategy {
    /// Uniformly random eviction.
    Random,
    /// Reservoir sampling — maintains a uniform distribution over all samples
    /// seen so far (implemented via the stream-based reservoir algorithm).
    Reservoir,
    /// Keep the highest-importance samples; evict the lowest when full.
    Prioritized,
    /// First-in-first-out ring buffer.
    RingBuffer,
}

// ---------------------------------------------------------------------------
// Minimal LCG random number generator (no rand crate)
// ---------------------------------------------------------------------------

/// A simple LCG pseudo-random number generator seeded with a `u64`.
struct Lcg {
    state: u64,
}

impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    /// Generate the next `u64`.
    fn next_u64(&mut self) -> u64 {
        // Knuth multiplicative hash LCG constants.
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    /// Generate a `usize` in `[0, n)`.
    fn next_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            return 0;
        }
        (self.next_u64() % n as u64) as usize
    }
}

// ---------------------------------------------------------------------------
// Experience replay buffer
// ---------------------------------------------------------------------------

/// Experience replay buffer for continual learning.
pub struct ReplayBuffer {
    capacity: usize,
    samples: VecDeque<ReplaySample>,
    strategy: ReplayStrategy,
    /// Monotonically increasing counter of total samples ever seen (needed for
    /// reservoir sampling).
    total_seen: usize,
}

impl ReplayBuffer {
    /// Create a new replay buffer.
    ///
    /// `capacity` must be > 0.
    pub fn new(capacity: usize, strategy: ReplayStrategy) -> Self {
        assert!(capacity > 0, "ReplayBuffer capacity must be > 0");
        Self {
            capacity,
            samples: VecDeque::with_capacity(capacity),
            strategy,
            total_seen: 0,
        }
    }

    /// Add a sample to the buffer, applying the eviction policy when full.
    pub fn add(&mut self, sample: ReplaySample) {
        self.total_seen += 1;

        match &self.strategy {
            ReplayStrategy::RingBuffer => {
                if self.samples.len() >= self.capacity {
                    self.samples.pop_front();
                }
                self.samples.push_back(sample);
            },
            ReplayStrategy::Random => {
                if self.samples.len() < self.capacity {
                    self.samples.push_back(sample);
                } else {
                    // Evict a random existing sample.
                    let mut lcg = Lcg::new(self.total_seen as u64);
                    let idx = lcg.next_usize(self.capacity);
                    self.samples[idx] = sample;
                }
            },
            ReplayStrategy::Reservoir => {
                if self.samples.len() < self.capacity {
                    self.samples.push_back(sample);
                } else {
                    // Standard reservoir sampling: replace slot j with probability
                    // capacity / total_seen.
                    let mut lcg = Lcg::new(self.total_seen as u64 ^ 0xDEAD_BEEF_CAFE_1234);
                    let j = lcg.next_usize(self.total_seen);
                    if j < self.capacity {
                        self.samples[j] = sample;
                    }
                }
            },
            ReplayStrategy::Prioritized => {
                if self.samples.len() < self.capacity {
                    self.samples.push_back(sample);
                } else {
                    // Evict the sample with the lowest importance.
                    let min_idx = self
                        .samples
                        .iter()
                        .enumerate()
                        .min_by(|(_, a), (_, b)| {
                            a.importance
                                .partial_cmp(&b.importance)
                                .unwrap_or(std::cmp::Ordering::Equal)
                        })
                        .map(|(i, _)| i);

                    if let Some(idx) = min_idx {
                        if self.samples[idx].importance < sample.importance {
                            self.samples[idx] = sample;
                        }
                    }
                }
            },
        }
    }

    /// Sample a batch of `batch_size` items (with replacement) using the given
    /// LCG seed.
    ///
    /// Returns fewer items if the buffer holds fewer than `batch_size` samples.
    pub fn sample_batch(&self, batch_size: usize, seed: u64) -> Vec<&ReplaySample> {
        if self.samples.is_empty() {
            return Vec::new();
        }
        let n = self.samples.len();
        let actual = batch_size.min(n);
        let mut lcg = Lcg::new(seed);
        let mut result = Vec::with_capacity(actual);
        for _ in 0..actual {
            let idx = lcg.next_usize(n);
            result.push(&self.samples[idx]);
        }
        result
    }

    /// Count the number of samples per task.
    pub fn task_sample_counts(&self) -> HashMap<usize, usize> {
        let mut counts: HashMap<usize, usize> = HashMap::new();
        for s in &self.samples {
            *counts.entry(s.task_id).or_insert(0) += 1;
        }
        counts
    }

    /// Current number of samples in the buffer.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    /// Whether the buffer has reached its capacity.
    pub fn is_full(&self) -> bool {
        self.samples.len() >= self.capacity
    }

    /// Whether the buffer is empty.
    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Current capacity.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

// ---------------------------------------------------------------------------
// Progressive Neural Networks — lateral adapter
// ---------------------------------------------------------------------------

/// Configuration for Progressive Neural Networks.
#[derive(Debug, Clone)]
pub struct PnnConfig {
    /// Total number of tasks.
    pub num_tasks: usize,
    /// Hidden size of each column.
    pub hidden_size: usize,
    /// Size of lateral connection adapters between columns.
    pub lateral_connection_size: usize,
}

impl PnnConfig {
    pub fn new(num_tasks: usize, hidden_size: usize, lateral_connection_size: usize) -> Self {
        Self {
            num_tasks,
            hidden_size,
            lateral_connection_size,
        }
    }
}

/// A lateral connection that adapts activations from a previous PNN column
/// into the input space of the current column.
///
/// The adapter implements a linear projection:
/// output = W * prev_activation    (W ∈ ℝ^{current_hidden × prev_hidden})
pub struct LateralAdapter {
    pub prev_hidden: usize,
    pub current_hidden: usize,
    /// Weight matrix stored row-major: row i is weights for output neuron i.
    pub adapter_weights: Vec<f32>,
}

impl LateralAdapter {
    /// Create a new lateral adapter with Xavier-style uniform initialization.
    ///
    /// Weights are initialised in [-limit, +limit] where
    /// limit = sqrt(6 / (prev_hidden + current_hidden)).
    pub fn new(prev_hidden: usize, current_hidden: usize) -> Self {
        let size = current_hidden * prev_hidden;
        let limit = (6.0_f32 / (prev_hidden + current_hidden) as f32).sqrt();
        let mut weights = Vec::with_capacity(size);

        // Use an LCG seeded with the layer sizes for reproducibility.
        let seed = (prev_hidden as u64).wrapping_mul(31)
            ^ (current_hidden as u64).wrapping_mul(17)
            ^ 0xABCD_EF01_2345_6789;
        let mut lcg = Lcg::new(seed);

        for _ in 0..size {
            // Map u64 to [-1, 1], then scale by limit.
            let raw = (lcg.next_u64() as i64) as f32 / i64::MAX as f32;
            weights.push(raw * limit);
        }

        Self {
            prev_hidden,
            current_hidden,
            adapter_weights: weights,
        }
    }

    /// Forward pass: project `prev_activation` (length `prev_hidden`) into a
    /// vector of length `current_hidden`.
    pub fn forward(&self, prev_activation: &[f32]) -> Vec<f32> {
        assert_eq!(
            prev_activation.len(),
            self.prev_hidden,
            "prev_activation length mismatch: expected {}, got {}",
            self.prev_hidden,
            prev_activation.len()
        );

        let mut output = vec![0.0_f32; self.current_hidden];
        for i in 0..self.current_hidden {
            let row_offset = i * self.prev_hidden;
            let mut sum = 0.0_f32;
            for j in 0..self.prev_hidden {
                sum += self.adapter_weights[row_offset + j] * prev_activation[j];
            }
            output[i] = sum;
        }
        output
    }
}

// ---------------------------------------------------------------------------
// Continual learning evaluation metrics
// ---------------------------------------------------------------------------

/// Compute Backward Transfer (BWT).
///
/// BWT measures how much learning new tasks affects performance on old ones.
/// A negative BWT indicates catastrophic forgetting.
///
/// `accuracy_matrix[i][j]` = accuracy on task j after training on tasks 0..=i.
///
/// `BWT = 1/(T-1) * Σ_{i=0}^{T-2} (A\[T-1\]\[i\] - A\[i\]\[i\])`
///
/// where T is the number of tasks.
///
/// Returns 0.0 if fewer than 2 tasks are present.
pub fn compute_bwt(accuracy_matrix: &[Vec<f32>]) -> f32 {
    let t = accuracy_matrix.len();
    if t < 2 {
        return 0.0;
    }
    let last_row = &accuracy_matrix[t - 1];
    let mut sum = 0.0_f32;
    for i in 0..t - 1 {
        let a_final_i = if i < last_row.len() { last_row[i] } else { 0.0 };
        let a_i_i = if i < accuracy_matrix[i].len() { accuracy_matrix[i][i] } else { 0.0 };
        sum += a_final_i - a_i_i;
    }
    sum / (t - 1) as f32
}

/// Compute Forward Transfer (FWT).
///
/// FWT measures how much knowing previous tasks helps learning new ones
/// compared to a random baseline.
///
/// `FWT = 1/(T-1) * Σ_{i=1}^{T-1} (A\[i-1\]\[i\] - random_performance)`
///
/// where `A[i-1][i]` is the accuracy on task i *before* training on it
/// (zero-shot transfer from the previous tasks).
///
/// Returns 0.0 if fewer than 2 tasks.
pub fn compute_fwt(accuracy_matrix: &[Vec<f32>], random_performance: f32) -> f32 {
    let t = accuracy_matrix.len();
    if t < 2 {
        return 0.0;
    }
    let mut sum = 0.0_f32;
    for i in 1..t {
        // A[i-1][i] = accuracy on task i as evaluated after training on task i-1.
        let a_prev_i =
            if i < accuracy_matrix[i - 1].len() { accuracy_matrix[i - 1][i] } else { 0.0 };
        sum += a_prev_i - random_performance;
    }
    sum / (t - 1) as f32
}

/// Compute Intransigence.
///
/// Intransigence measures how much performance on the *current* task drops
/// compared to an upper bound (joint training on all tasks so far).
///
/// Here we approximate it as how much the diagonal drops from step to step:
/// `Intransigence = 1/T * Σ_{i} (A\[i\]\[i\] - A\[T-1\]\[i\])`
///
/// A positive value indicates forgetting of previous tasks.
///
/// Returns 0.0 for fewer than 2 tasks.
pub fn compute_intransigence(accuracy_matrix: &[Vec<f32>]) -> f32 {
    let t = accuracy_matrix.len();
    if t < 2 {
        return 0.0;
    }
    let last_row = &accuracy_matrix[t - 1];
    let mut sum = 0.0_f32;
    for i in 0..t {
        let a_i_i = if i < accuracy_matrix[i].len() { accuracy_matrix[i][i] } else { 0.0 };
        let a_final_i = if i < last_row.len() { last_row[i] } else { 0.0 };
        sum += a_i_i - a_final_i;
    }
    sum / t as f32
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // --- ReplayBuffer — RingBuffer ------------------------------------------

    #[test]
    fn test_ring_buffer_fifo_eviction() {
        let mut buf = ReplayBuffer::new(3, ReplayStrategy::RingBuffer);
        for i in 0_u32..5 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![i],
                task_id: 0,
                importance: 1.0,
            });
        }
        assert_eq!(buf.len(), 3);
        // Ring buffer should hold the last 3: [2, 3, 4].
        let ids: Vec<u32> = buf.samples.iter().map(|s| s.input_ids[0]).collect();
        assert_eq!(ids, vec![2, 3, 4]);
    }

    #[test]
    fn test_ring_buffer_is_full() {
        let mut buf = ReplayBuffer::new(2, ReplayStrategy::RingBuffer);
        assert!(!buf.is_full());
        buf.add(ReplaySample {
            input_ids: vec![0],
            labels: vec![],
            task_id: 0,
            importance: 1.0,
        });
        buf.add(ReplaySample {
            input_ids: vec![1],
            labels: vec![],
            task_id: 0,
            importance: 1.0,
        });
        assert!(buf.is_full());
    }

    // --- ReplayBuffer — Random ----------------------------------------------

    #[test]
    fn test_random_buffer_capacity_respected() {
        let mut buf = ReplayBuffer::new(5, ReplayStrategy::Random);
        for i in 0..20 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: i as usize % 3,
                importance: i as f32,
            });
        }
        assert_eq!(buf.len(), 5);
    }

    // --- ReplayBuffer — Reservoir -------------------------------------------

    #[test]
    fn test_reservoir_capacity_respected() {
        let mut buf = ReplayBuffer::new(10, ReplayStrategy::Reservoir);
        for i in 0..100 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: 0,
                importance: 1.0,
            });
        }
        assert_eq!(buf.len(), 10);
    }

    // --- ReplayBuffer — Prioritized -----------------------------------------

    #[test]
    fn test_prioritized_keeps_high_importance() {
        let mut buf = ReplayBuffer::new(2, ReplayStrategy::Prioritized);
        buf.add(ReplaySample {
            input_ids: vec![0],
            labels: vec![],
            task_id: 0,
            importance: 0.1,
        });
        buf.add(ReplaySample {
            input_ids: vec![1],
            labels: vec![],
            task_id: 0,
            importance: 0.2,
        });
        // Buffer is full. Adding a high-importance sample should evict the lowest.
        buf.add(ReplaySample {
            input_ids: vec![2],
            labels: vec![],
            task_id: 0,
            importance: 0.9,
        });
        assert_eq!(buf.len(), 2);
        let importances: Vec<f32> = buf.samples.iter().map(|s| s.importance).collect();
        assert!(
            importances.iter().all(|&imp| imp >= 0.2),
            "Low-importance sample (0.1) should have been evicted, got {:?}",
            importances
        );
    }

    // --- sample_batch -------------------------------------------------------

    #[test]
    fn test_sample_batch_size() {
        let mut buf = ReplayBuffer::new(10, ReplayStrategy::RingBuffer);
        for i in 0..10 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: 0,
                importance: 1.0,
            });
        }
        let batch = buf.sample_batch(5, 42);
        assert_eq!(batch.len(), 5);
    }

    #[test]
    fn test_sample_batch_empty_buffer() {
        let buf = ReplayBuffer::new(10, ReplayStrategy::RingBuffer);
        let batch = buf.sample_batch(5, 42);
        assert!(batch.is_empty());
    }

    // --- task_sample_counts -------------------------------------------------

    #[test]
    fn test_task_sample_counts() {
        let mut buf = ReplayBuffer::new(10, ReplayStrategy::RingBuffer);
        for i in 0..6 {
            buf.add(ReplaySample {
                input_ids: vec![i as u32],
                labels: vec![],
                task_id: i % 2,
                importance: 1.0,
            });
        }
        let counts = buf.task_sample_counts();
        assert_eq!(counts.get(&0).copied().unwrap_or(0), 3);
        assert_eq!(counts.get(&1).copied().unwrap_or(0), 3);
    }

    // --- LateralAdapter -----------------------------------------------------

    #[test]
    fn test_lateral_adapter_output_shape() {
        let adapter = LateralAdapter::new(4, 8);
        let prev = vec![1.0_f32; 4];
        let out = adapter.forward(&prev);
        assert_eq!(out.len(), 8);
    }

    #[test]
    fn test_lateral_adapter_zero_input() {
        let adapter = LateralAdapter::new(4, 4);
        let prev = vec![0.0_f32; 4];
        let out = adapter.forward(&prev);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_lateral_adapter_identity_like() {
        // With a single input and single output neuron, set weight manually.
        let mut adapter = LateralAdapter::new(1, 1);
        adapter.adapter_weights[0] = 2.0;
        let out = adapter.forward(&[3.0]);
        assert!((out[0] - 6.0).abs() < 1e-5, "Expected 6.0, got {}", out[0]);
    }

    // --- compute_bwt --------------------------------------------------------

    #[test]
    fn test_bwt_no_forgetting() {
        // Perfect retention: accuracy at end equals diagonal.
        let matrix = vec![vec![0.9_f32, 0.0], vec![0.9_f32, 0.8]];
        let bwt = compute_bwt(&matrix);
        // (0.9 - 0.9) / 1 = 0.0
        assert!((bwt - 0.0).abs() < 1e-5, "BWT should be 0.0, got {}", bwt);
    }

    #[test]
    fn test_bwt_catastrophic_forgetting() {
        let matrix = vec![
            vec![0.9_f32, 0.0, 0.0],
            vec![0.5_f32, 0.8, 0.0],
            vec![0.1_f32, 0.4, 0.7],
        ];
        // BWT = ((0.1 - 0.9) + (0.4 - 0.8)) / 2 = (-0.8 + -0.4) / 2 = -0.6
        let bwt = compute_bwt(&matrix);
        assert!((bwt - (-0.6)).abs() < 1e-5, "Expected -0.6, got {}", bwt);
    }

    #[test]
    fn test_bwt_single_task() {
        let matrix = vec![vec![0.9_f32]];
        assert_eq!(compute_bwt(&matrix), 0.0);
    }

    // --- compute_fwt --------------------------------------------------------

    #[test]
    fn test_fwt_positive_transfer() {
        // Task 1 zero-shot accuracy before training = 0.6; random = 0.1.
        let matrix = vec![
            vec![0.9_f32, 0.6], // after task 0, task 1 zero-shot = 0.6
            vec![0.9_f32, 0.8],
        ];
        // FWT = (0.6 - 0.1) / 1 = 0.5
        let fwt = compute_fwt(&matrix, 0.1);
        assert!((fwt - 0.5).abs() < 1e-5, "Expected 0.5, got {}", fwt);
    }

    #[test]
    fn test_fwt_single_task() {
        let matrix = vec![vec![0.9_f32]];
        assert_eq!(compute_fwt(&matrix, 0.1), 0.0);
    }

    // --- compute_intransigence ----------------------------------------------

    #[test]
    fn test_intransigence_no_forgetting() {
        let matrix = vec![vec![0.9_f32, 0.0], vec![0.9_f32, 0.8]];
        // Diagonal values equal final-row values → intransigence = 0.
        let intr = compute_intransigence(&matrix);
        assert!((intr - 0.0).abs() < 1e-5, "Expected 0.0, got {}", intr);
    }

    #[test]
    fn test_intransigence_positive() {
        let matrix = vec![vec![0.9_f32, 0.0], vec![0.5_f32, 0.8]];
        // Task 0: A[0][0]=0.9, A[1][0]=0.5 → drop of 0.4
        // Task 1: A[1][1]=0.8, A[1][1]=0.8 → no drop
        // Intransigence = (0.4 + 0.0) / 2 = 0.2
        let intr = compute_intransigence(&matrix);
        assert!((intr - 0.2).abs() < 1e-5, "Expected 0.2, got {}", intr);
    }

    #[test]
    fn test_intransigence_single_task() {
        let matrix = vec![vec![0.9_f32]];
        assert_eq!(compute_intransigence(&matrix), 0.0);
    }

    // --- ReplayBuffer — Reservoir: total_seen counter ----------------------

    #[test]
    fn test_reservoir_total_seen_increments() {
        let mut buf = ReplayBuffer::new(5, ReplayStrategy::Reservoir);
        for i in 0..15u32 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: 0,
                importance: 1.0,
            });
        }
        assert_eq!(
            buf.total_seen, 15,
            "total_seen should reflect all samples ever added"
        );
        assert_eq!(buf.len(), 5, "buffer len should not exceed capacity");
    }

    // --- ReplayBuffer — Prioritized: low-importance not evicted by higher-low --

    /// Adding a sample with lower importance than the current minimum should
    /// NOT replace anything in the buffer.
    #[test]
    fn test_prioritized_low_importance_not_added() {
        let mut buf = ReplayBuffer::new(2, ReplayStrategy::Prioritized);
        buf.add(ReplaySample {
            input_ids: vec![0],
            labels: vec![],
            task_id: 0,
            importance: 0.5,
        });
        buf.add(ReplaySample {
            input_ids: vec![1],
            labels: vec![],
            task_id: 0,
            importance: 0.6,
        });
        // Full; new sample has lower importance than minimum (0.5) → should be dropped.
        buf.add(ReplaySample {
            input_ids: vec![2],
            labels: vec![],
            task_id: 0,
            importance: 0.3,
        });
        assert_eq!(buf.len(), 2, "buffer should still have 2 samples");
        let ids: Vec<u32> = buf.samples.iter().map(|s| s.input_ids[0]).collect();
        assert!(
            !ids.contains(&2),
            "low-importance sample should not be in buffer"
        );
    }

    // --- ReplayBuffer — sample_batch: reproducible with same seed -----------

    #[test]
    fn test_sample_batch_reproducible() {
        let mut buf = ReplayBuffer::new(10, ReplayStrategy::RingBuffer);
        for i in 0..10u32 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: 0,
                importance: 1.0,
            });
        }
        let batch1: Vec<u32> = buf.sample_batch(4, 12345).iter().map(|s| s.input_ids[0]).collect();
        let batch2: Vec<u32> = buf.sample_batch(4, 12345).iter().map(|s| s.input_ids[0]).collect();
        assert_eq!(batch1, batch2, "same seed should produce identical batches");
    }

    // --- ReplayBuffer — task_sample_counts: empty buffer -------------------

    #[test]
    fn test_task_sample_counts_empty() {
        let buf = ReplayBuffer::new(5, ReplayStrategy::RingBuffer);
        assert!(
            buf.task_sample_counts().is_empty(),
            "empty buffer should have no task counts"
        );
    }

    // --- LateralAdapter: weight count matches sizes -------------------------

    #[test]
    fn test_lateral_adapter_weight_count() {
        let adapter = LateralAdapter::new(6, 4);
        assert_eq!(
            adapter.adapter_weights.len(),
            6 * 4,
            "weight count should be prev_hidden * current_hidden"
        );
    }

    // --- compute_bwt: positive BWT (backward facilitation) ------------------

    #[test]
    fn test_bwt_positive_backward_transfer() {
        // Task 0 accuracy improved after training on task 1.
        let matrix = vec![
            vec![0.5_f32, 0.0],
            vec![0.8_f32, 0.9], // A[1][0] > A[0][0]
        ];
        // BWT = (0.8 - 0.5) / 1 = 0.3
        let bwt = compute_bwt(&matrix);
        assert!((bwt - 0.3).abs() < 1e-5, "Expected BWT=0.3, got {bwt}");
    }

    // --- compute_fwt: negative FWT (no forward transfer) -------------------

    #[test]
    fn test_fwt_no_forward_transfer() {
        // Zero-shot accuracy on task 1 equals random performance.
        let matrix = vec![
            vec![0.9_f32, 0.1], // zero-shot on task 1 = 0.1 = random
            vec![0.9_f32, 0.8],
        ];
        let fwt = compute_fwt(&matrix, 0.1);
        assert!(
            (fwt - 0.0).abs() < 1e-5,
            "FWT should be 0 when zero-shot equals random, got {fwt}"
        );
    }

    // --- compute_intransigence: three tasks ---------------------------------

    #[test]
    fn test_intransigence_three_tasks() {
        // Tasks: 0 forgets, 1 forgets, 2 no change.
        let matrix = vec![
            vec![0.9_f32, 0.0, 0.0],
            vec![0.7_f32, 0.8, 0.0],
            vec![0.5_f32, 0.6, 0.7],
        ];
        // Intransigence = ((0.9-0.5) + (0.8-0.6) + (0.7-0.7)) / 3
        //               = (0.4 + 0.2 + 0.0) / 3 ≈ 0.2
        let intr = compute_intransigence(&matrix);
        assert!(
            (intr - 0.2).abs() < 1e-5,
            "Expected intransigence ≈ 0.2, got {intr}"
        );
    }

    // --- ReplayBuffer is_empty --------------------------------------------

    #[test]
    fn test_is_empty_initially() {
        let buf = ReplayBuffer::new(10, ReplayStrategy::RingBuffer);
        assert!(buf.is_empty(), "new buffer should be empty");
    }

    // --- ReplayBuffer capacity accessor -----------------------------------

    #[test]
    fn test_capacity_accessor() {
        let buf = ReplayBuffer::new(7, ReplayStrategy::Random);
        assert_eq!(
            buf.capacity(),
            7,
            "capacity accessor should return configured capacity"
        );
    }

    // --- Reservoir: all samples fill buffer before overflow ---------------

    #[test]
    fn test_reservoir_fills_before_overflow() {
        let mut buf = ReplayBuffer::new(5, ReplayStrategy::Reservoir);
        for i in 0..5u32 {
            buf.add(ReplaySample {
                input_ids: vec![i],
                labels: vec![],
                task_id: 0,
                importance: 1.0,
            });
        }
        assert!(
            buf.is_full(),
            "buffer should be full after adding capacity samples"
        );
        assert_eq!(buf.len(), 5, "length should equal capacity");
    }

    // --- BWT: empty matrix -----------------------------------------------

    #[test]
    fn test_bwt_empty_matrix() {
        assert_eq!(compute_bwt(&[]), 0.0, "BWT of empty matrix should be 0");
    }

    // --- FWT: empty matrix -----------------------------------------------

    #[test]
    fn test_fwt_empty_matrix() {
        assert_eq!(
            compute_fwt(&[], 0.1),
            0.0,
            "FWT of empty matrix should be 0"
        );
    }

    // --- Intransigence: empty matrix ----------------------------------------

    #[test]
    fn test_intransigence_empty_matrix() {
        assert_eq!(
            compute_intransigence(&[]),
            0.0,
            "intransigence of empty matrix should be 0"
        );
    }
}
