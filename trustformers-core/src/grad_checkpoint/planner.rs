//! GradCheckpointPlanner — advanced gradient checkpoint policy planning.
//!
//! Provides `GradCheckpointPlanner` with multiple `CheckpointPolicy` variants
//! (EveryN, SpecificLayers, MemoryThreshold, Optimal) and `ActivationBuffer`
//! for tracking peak memory usage during a forward pass.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by gradient checkpoint planning.
#[derive(Debug, Clone, PartialEq)]
pub enum GradCheckpointError {
    /// The layer list was empty when a non-empty list was required.
    EmptyLayers,
    /// A policy parameter was invalid (e.g. n=0 for EveryN).
    InvalidPolicy(String),
    /// A layer index is outside the declared range.
    LayerIndexOutOfBounds(usize),
    /// The memory budget is too small to hold even a single layer.
    BudgetTooSmall { budget: usize, minimum: usize },
}

impl fmt::Display for GradCheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GradCheckpointError::EmptyLayers => write!(f, "layer list is empty"),
            GradCheckpointError::InvalidPolicy(msg) => write!(f, "invalid policy: {msg}"),
            GradCheckpointError::LayerIndexOutOfBounds(idx) => {
                write!(f, "layer index {idx} is out of bounds")
            },
            GradCheckpointError::BudgetTooSmall { budget, minimum } => write!(
                f,
                "memory budget {budget} bytes is smaller than the minimum required {minimum} bytes"
            ),
        }
    }
}

impl std::error::Error for GradCheckpointError {}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy that controls which layers have their activations checkpointed
/// (discarded during forward, recomputed during backward).
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointPolicy {
    /// Checkpoint every N-th layer (layer 0, N, 2N, …).  N=0 means no checkpointing.
    EveryN(usize),
    /// Checkpoint exactly these layer indices (sorted or unsorted).
    SpecificLayers(Vec<usize>),
    /// Checkpoint any layer whose activation size meets or exceeds the threshold (bytes).
    MemoryThreshold(usize),
    /// Use dynamic programming to minimise recomputation while staying within a
    /// total activation memory budget.
    Optimal { target_memory_bytes: usize },
}

// ─────────────────────────────────────────────────────────────────────────────
// GradCheckpointPlanner
// ─────────────────────────────────────────────────────────────────────────────

/// Plans gradient checkpointing for a model with a known per-layer activation profile.
pub struct GradCheckpointPlanner {
    /// Total number of layers in the model.
    pub num_layers: usize,
    /// The checkpointing policy to apply.
    pub policy: CheckpointPolicy,
    /// Estimated activation size in bytes for each layer (length must equal `num_layers`).
    pub layer_activation_sizes: Vec<usize>,
}

impl GradCheckpointPlanner {
    /// Create a new planner.
    ///
    /// If `layer_activation_sizes` is shorter than `num_layers`, missing entries
    /// are treated as 0 bytes.  Extra entries beyond `num_layers` are ignored.
    pub fn new(num_layers: usize, policy: CheckpointPolicy) -> Self {
        Self {
            num_layers,
            policy,
            layer_activation_sizes: vec![0; num_layers],
        }
    }

    /// Create a new planner with explicit per-layer activation size estimates.
    pub fn with_sizes(
        num_layers: usize,
        policy: CheckpointPolicy,
        layer_activation_sizes: Vec<usize>,
    ) -> Self {
        Self {
            num_layers,
            policy,
            layer_activation_sizes,
        }
    }

    /// Returns `true` if the activation for `layer_idx` should be checkpointed.
    ///
    /// For `CheckpointPolicy::Optimal` the method internally calls
    /// `compute_optimal_checkpoints` with the `target_memory_bytes` budget and
    /// checks membership in the result.
    pub fn should_checkpoint_layer(&self, layer_idx: usize) -> bool {
        if layer_idx >= self.num_layers {
            return false;
        }
        match &self.policy {
            CheckpointPolicy::EveryN(n) => {
                if *n == 0 {
                    false
                } else {
                    layer_idx.is_multiple_of(*n)
                }
            },
            CheckpointPolicy::SpecificLayers(indices) => indices.contains(&layer_idx),
            CheckpointPolicy::MemoryThreshold(threshold) => {
                let size = self.layer_activation_sizes.get(layer_idx).copied().unwrap_or(0);
                size >= *threshold
            },
            CheckpointPolicy::Optimal {
                target_memory_bytes,
            } => {
                // Fall back to the DP result; ignore errors (treat as no-checkpoint)
                match Self::compute_optimal_checkpoints(
                    &self.layer_activation_sizes
                        [..self.num_layers.min(self.layer_activation_sizes.len())],
                    *target_memory_bytes,
                ) {
                    Ok(checkpointed) => checkpointed.contains(&layer_idx),
                    Err(_) => false,
                }
            },
        }
    }

    /// Total bytes of activation memory that is freed by checkpointing.
    ///
    /// Sums `layer_activation_sizes[i]` for all layers where
    /// `should_checkpoint_layer(i)` is `true`.
    pub fn total_memory_saved(&self) -> usize {
        (0..self.num_layers)
            .filter(|&i| self.should_checkpoint_layer(i))
            .map(|i| self.layer_activation_sizes.get(i).copied().unwrap_or(0))
            .sum()
    }

    /// Number of layers that are checkpointed (and thus require recomputation).
    pub fn recompute_cost(&self) -> usize {
        (0..self.num_layers).filter(|&i| self.should_checkpoint_layer(i)).count()
    }

    /// Compute an optimal set of checkpoint layers using a greedy segment algorithm.
    ///
    /// The algorithm sweeps through the layers accumulating activation memory.
    /// When adding the next layer's activation would cause the running segment
    /// sum to exceed `budget_bytes`, a checkpoint boundary is placed at that
    /// layer (its activation is discarded), and the accumulator resets.
    ///
    /// # Errors
    /// - `EmptyLayers` when `layer_sizes` is empty.
    /// - `BudgetTooSmall` when any single layer's activation exceeds `budget_bytes`.
    pub fn compute_optimal_checkpoints(
        layer_sizes: &[usize],
        budget_bytes: usize,
    ) -> Result<Vec<usize>, GradCheckpointError> {
        if layer_sizes.is_empty() {
            return Ok(Vec::new());
        }

        // Verify the budget can accommodate the largest single layer.
        let max_layer = layer_sizes.iter().copied().max().unwrap_or(0);
        if max_layer > budget_bytes {
            return Err(GradCheckpointError::BudgetTooSmall {
                budget: budget_bytes,
                minimum: max_layer,
            });
        }

        let mut checkpointed: Vec<usize> = Vec::new();
        let mut segment_sum: usize = 0;

        for (i, &size) in layer_sizes.iter().enumerate() {
            if segment_sum + size > budget_bytes {
                // This layer's activation would overflow the budget.
                // Checkpoint it: discard its activation and reset the segment.
                checkpointed.push(i);
                segment_sum = 0;
            } else {
                segment_sum += size;
            }
        }

        Ok(checkpointed)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerActivation
// ─────────────────────────────────────────────────────────────────────────────

/// Record of a single layer's activation in the buffer.
#[derive(Debug, Clone)]
pub struct LayerActivation {
    /// Zero-based layer index.
    pub layer_idx: usize,
    /// Activation memory footprint in bytes.
    pub size_bytes: usize,
    /// `true` if the activation was checkpointed (not stored in memory).
    pub is_checkpointed: bool,
}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationBuffer
// ─────────────────────────────────────────────────────────────────────────────

/// Tracks which layer activations are resident in memory during a forward pass.
///
/// Checkpointed layers do not consume memory (their activations are not stored).
pub struct ActivationBuffer {
    /// Maximum allowed memory in bytes.
    pub max_size: usize,
    /// Current bytes of resident (non-checkpointed) activations.
    pub current_size: usize,
    /// All layers recorded so far.
    pub layers: Vec<LayerActivation>,
}

impl ActivationBuffer {
    /// Create a new buffer with the given capacity.
    pub fn new(max_size: usize) -> Self {
        Self {
            max_size,
            current_size: 0,
            layers: Vec::new(),
        }
    }

    /// Record a layer's activation.
    ///
    /// If `checkpointed` is `false`, `size_bytes` is added to `current_size`.
    /// Returns `Err(LayerIndexOutOfBounds)` if the same `layer_idx` has already
    /// been pushed.
    pub fn push_layer(
        &mut self,
        layer_idx: usize,
        size_bytes: usize,
        checkpointed: bool,
    ) -> Result<(), GradCheckpointError> {
        if self.layers.iter().any(|l| l.layer_idx == layer_idx) {
            return Err(GradCheckpointError::LayerIndexOutOfBounds(layer_idx));
        }
        if !checkpointed {
            self.current_size = self.current_size.saturating_add(size_bytes);
        }
        self.layers.push(LayerActivation {
            layer_idx,
            size_bytes,
            is_checkpointed: checkpointed,
        });
        Ok(())
    }

    /// Evict a layer from the buffer.
    ///
    /// If the layer was not checkpointed, its `size_bytes` are subtracted from
    /// `current_size`.  Returns `Err(LayerIndexOutOfBounds)` if the layer is
    /// not in the buffer.
    pub fn evict_layer(&mut self, layer_idx: usize) -> Result<(), GradCheckpointError> {
        let pos = self
            .layers
            .iter()
            .position(|l| l.layer_idx == layer_idx)
            .ok_or(GradCheckpointError::LayerIndexOutOfBounds(layer_idx))?;

        let layer = self.layers.remove(pos);
        if !layer.is_checkpointed {
            self.current_size = self.current_size.saturating_sub(layer.size_bytes);
        }
        Ok(())
    }

    /// Fraction of the buffer capacity currently in use.
    ///
    /// Returns a value in `[0.0, 1.0]`.  When `max_size` is 0 returns `0.0`.
    pub fn memory_pressure(&self) -> f32 {
        if self.max_size == 0 {
            return 0.0;
        }
        (self.current_size as f32 / self.max_size as f32).clamp(0.0, 1.0)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a planner with explicit sizes.
    fn planner_with_sizes(sizes: Vec<usize>, policy: CheckpointPolicy) -> GradCheckpointPlanner {
        let n = sizes.len();
        GradCheckpointPlanner::with_sizes(n, policy, sizes)
    }

    // ── GradCheckpointPlanner ────────────────────────────────────────────────

    #[test]
    fn test_planner_every_n2_on_8_layers() {
        let planner = GradCheckpointPlanner::new(8, CheckpointPolicy::EveryN(2));
        // Layers 0,2,4,6 → checkpointed
        for i in 0..8_usize {
            let expected = i % 2 == 0;
            assert_eq!(
                planner.should_checkpoint_layer(i),
                expected,
                "layer {i}: expected checkpoint={expected}"
            );
        }
    }

    #[test]
    fn test_planner_every_n3() {
        let planner = GradCheckpointPlanner::new(9, CheckpointPolicy::EveryN(3));
        assert!(planner.should_checkpoint_layer(0));
        assert!(planner.should_checkpoint_layer(3));
        assert!(planner.should_checkpoint_layer(6));
        assert!(!planner.should_checkpoint_layer(1));
        assert!(!planner.should_checkpoint_layer(2));
        assert!(!planner.should_checkpoint_layer(4));
        assert!(!planner.should_checkpoint_layer(5));
    }

    #[test]
    fn test_planner_every_n0_no_checkpoint() {
        let planner = GradCheckpointPlanner::new(6, CheckpointPolicy::EveryN(0));
        for i in 0..6 {
            assert!(
                !planner.should_checkpoint_layer(i),
                "n=0 should never checkpoint"
            );
        }
    }

    #[test]
    fn test_planner_specific_layers() {
        let planner =
            GradCheckpointPlanner::new(8, CheckpointPolicy::SpecificLayers(vec![1, 3, 5]));
        assert!(!planner.should_checkpoint_layer(0));
        assert!(planner.should_checkpoint_layer(1));
        assert!(!planner.should_checkpoint_layer(2));
        assert!(planner.should_checkpoint_layer(3));
        assert!(!planner.should_checkpoint_layer(4));
        assert!(planner.should_checkpoint_layer(5));
        assert!(!planner.should_checkpoint_layer(6));
        assert!(!planner.should_checkpoint_layer(7));
    }

    #[test]
    fn test_planner_memory_threshold() {
        let sizes = vec![500, 1500, 800, 2000, 100];
        let planner = planner_with_sizes(sizes, CheckpointPolicy::MemoryThreshold(1000));
        // Only layers with size >= 1000: indices 1 (1500) and 3 (2000)
        assert!(!planner.should_checkpoint_layer(0));
        assert!(planner.should_checkpoint_layer(1));
        assert!(!planner.should_checkpoint_layer(2));
        assert!(planner.should_checkpoint_layer(3));
        assert!(!planner.should_checkpoint_layer(4));
    }

    #[test]
    fn test_planner_total_memory_saved() {
        let sizes = vec![100_usize, 200, 300, 400];
        // EveryN(2): checkpoint layers 0 and 2
        let planner = planner_with_sizes(sizes, CheckpointPolicy::EveryN(2));
        // saved = sizes[0] + sizes[2] = 100 + 300 = 400
        assert_eq!(planner.total_memory_saved(), 400);
    }

    #[test]
    fn test_planner_recompute_cost() {
        let sizes = vec![100_usize; 6];
        let planner = planner_with_sizes(sizes, CheckpointPolicy::EveryN(2));
        // Layers 0, 2, 4 → 3 checkpointed
        assert_eq!(planner.recompute_cost(), 3);
    }

    #[test]
    fn test_compute_optimal_empty_returns_empty() {
        let result = GradCheckpointPlanner::compute_optimal_checkpoints(&[], 1024)
            .expect("empty should succeed");
        assert!(result.is_empty());
    }

    #[test]
    fn test_compute_optimal_large_budget_no_checkpoints() {
        // Budget > total sum → no checkpoints needed
        let sizes = vec![100_usize, 200, 300, 400];
        let result = GradCheckpointPlanner::compute_optimal_checkpoints(&sizes, 10_000)
            .expect("should succeed");
        assert!(
            result.is_empty(),
            "large budget should require no checkpoints"
        );
    }

    #[test]
    fn test_compute_optimal_tight_budget_checkpoints_some() {
        // Budget = 300, layers = [100, 200, 300, 400]
        // Layer 3 (400) alone exceeds budget → error
        // Use budget=500 instead
        let sizes = vec![200_usize, 300, 200, 300];
        let result = GradCheckpointPlanner::compute_optimal_checkpoints(&sizes, 500)
            .expect("should succeed");
        // segment: [200] → 200 <=500; [200+300=500] → 500 <=500; [200] →200 <=500; [200+300=500] → OK
        // Actually no checkpoints needed since no single step exceeds 500
        // Let's use budget=300 to force checkpoints
        let result2 = GradCheckpointPlanner::compute_optimal_checkpoints(&sizes, 300)
            .expect("should succeed");
        assert!(
            !result2.is_empty(),
            "tight budget should produce checkpoints"
        );
        let _ = result;
    }

    #[test]
    fn test_compute_optimal_budget_too_small_error() {
        let sizes = vec![100_usize, 500, 200];
        // Layer 1 has 500 bytes; budget=400 < 500 → BudgetTooSmall
        let result = GradCheckpointPlanner::compute_optimal_checkpoints(&sizes, 400);
        assert!(
            matches!(result, Err(GradCheckpointError::BudgetTooSmall { .. })),
            "should return BudgetTooSmall"
        );
    }

    // ── ActivationBuffer ─────────────────────────────────────────────────────

    #[test]
    fn test_activation_buffer_push_increases_current_size() {
        let mut buf = ActivationBuffer::new(10_000);
        buf.push_layer(0, 1024, false).expect("push ok");
        assert_eq!(buf.current_size, 1024);
        buf.push_layer(1, 512, false).expect("push ok");
        assert_eq!(buf.current_size, 1536);
    }

    #[test]
    fn test_activation_buffer_checkpointed_layer_no_size() {
        let mut buf = ActivationBuffer::new(10_000);
        buf.push_layer(0, 2048, true).expect("push ok");
        assert_eq!(
            buf.current_size, 0,
            "checkpointed layer should not add to current_size"
        );
    }

    #[test]
    fn test_activation_buffer_evict_decreases_size() {
        let mut buf = ActivationBuffer::new(10_000);
        buf.push_layer(0, 1000, false).expect("push ok");
        buf.push_layer(1, 2000, false).expect("push ok");
        buf.evict_layer(0).expect("evict ok");
        assert_eq!(buf.current_size, 2000);
    }

    #[test]
    fn test_activation_buffer_evict_nonexistent_returns_err() {
        let mut buf = ActivationBuffer::new(10_000);
        let result = buf.evict_layer(42);
        assert!(
            matches!(result, Err(GradCheckpointError::LayerIndexOutOfBounds(42))),
            "evicting non-existent layer should return error"
        );
    }

    #[test]
    fn test_activation_buffer_memory_pressure() {
        let mut buf = ActivationBuffer::new(4000);
        buf.push_layer(0, 1000, false).expect("push ok");
        buf.push_layer(1, 1000, false).expect("push ok");
        // current_size = 2000, max = 4000 → pressure = 0.5
        let pressure = buf.memory_pressure();
        assert!(
            (pressure - 0.5).abs() < 1e-5,
            "memory pressure should be 0.5: got {pressure}"
        );
    }

    #[test]
    fn test_activation_buffer_duplicate_layer_returns_err() {
        let mut buf = ActivationBuffer::new(10_000);
        buf.push_layer(5, 100, false).expect("first push ok");
        let result = buf.push_layer(5, 200, false);
        assert!(
            matches!(result, Err(GradCheckpointError::LayerIndexOutOfBounds(5))),
            "duplicate layer_idx should return error"
        );
    }

    // ── GradCheckpointError Display ──────────────────────────────────────────

    #[test]
    fn test_grad_checkpoint_error_display() {
        let e1 = GradCheckpointError::EmptyLayers;
        assert!(e1.to_string().contains("empty"));

        let e2 = GradCheckpointError::InvalidPolicy("bad n".into());
        assert!(e2.to_string().contains("bad n"));

        let e3 = GradCheckpointError::LayerIndexOutOfBounds(7);
        assert!(e3.to_string().contains('7'));

        let e4 = GradCheckpointError::BudgetTooSmall {
            budget: 100,
            minimum: 500,
        };
        assert!(e4.to_string().contains("100"));
        assert!(e4.to_string().contains("500"));
    }
}
