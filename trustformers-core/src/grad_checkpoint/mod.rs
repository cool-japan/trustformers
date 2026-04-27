//! Gradient checkpointing — recompute activations during backward pass to save memory
//!
//! Gradient checkpointing trades computation for memory by not storing intermediate
//! activations during the forward pass, instead recomputing them when needed during
//! the backward pass. This module provides primitives to manage this tradeoff.

pub mod planner;
pub mod selective;

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during checkpoint operations.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointError {
    /// Layer name was not found in the snapshot store.
    LayerNotFound(String),
    /// Recomputation of a checkpointed activation failed.
    RecomputationFailed(String),
    /// Shape mismatch between expected and received tensor.
    InvalidShape {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Segment index is out of range.
    InvalidSegment(usize),
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CheckpointError::LayerNotFound(name) => {
                write!(f, "checkpoint layer not found: '{name}'")
            },
            CheckpointError::RecomputationFailed(msg) => {
                write!(f, "recomputation failed: {msg}")
            },
            CheckpointError::InvalidShape { expected, got } => {
                write!(f, "shape mismatch: expected {expected:?}, got {got:?}")
            },
            CheckpointError::InvalidSegment(idx) => {
                write!(f, "invalid segment index: {idx}")
            },
        }
    }
}

impl std::error::Error for CheckpointError {}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationSnapshot
// ─────────────────────────────────────────────────────────────────────────────

/// A recorded activation snapshot from a forward pass layer.
///
/// When `is_checkpointed` is `true`, the `tensor_data` is empty and the
/// activation must be recomputed from inputs during the backward pass.
#[derive(Debug, Clone)]
pub struct ActivationSnapshot {
    /// Identifier for the layer that produced this activation.
    pub layer_name: String,
    /// Raw tensor data (empty when checkpointed).
    pub tensor_data: Vec<f32>,
    /// Logical shape of the tensor.
    pub shape: Vec<usize>,
    /// `false` = stored normally, `true` = will be recomputed.
    pub is_checkpointed: bool,
    /// Estimated FLOP cost to recompute this activation (used for scheduling).
    pub computation_cost: f32,
}

impl ActivationSnapshot {
    /// Create a fully-stored activation snapshot.
    pub fn new(layer_name: &str, data: Vec<f32>, shape: Vec<usize>) -> Self {
        Self {
            layer_name: layer_name.to_owned(),
            tensor_data: data,
            shape,
            is_checkpointed: false,
            computation_cost: 0.0,
        }
    }

    /// Create a checkpointed snapshot: shape and cost are stored, data is not.
    pub fn checkpointed(layer_name: &str, shape: Vec<usize>, cost: f32) -> Self {
        Self {
            layer_name: layer_name.to_owned(),
            tensor_data: Vec::new(),
            shape,
            is_checkpointed: true,
            computation_cost: cost,
        }
    }

    /// Total number of elements in the tensor.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }

    /// Memory used by stored data in bytes (f32 = 4 bytes each).
    pub fn memory_bytes(&self) -> usize {
        self.numel() * 4
    }

    /// Returns `true` if the tensor data is actually stored (not checkpointed).
    pub fn is_stored(&self) -> bool {
        !self.tensor_data.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointPolicy
// ─────────────────────────────────────────────────────────────────────────────

/// Policy that determines which layers should be checkpointed.
#[derive(Debug, Clone, PartialEq)]
pub enum CheckpointPolicy {
    /// Checkpoint all layers (maximum memory savings, maximum recompute cost).
    All,
    /// Checkpoint every N-th layer (layer indices 0, N, 2N, …).
    EveryN(usize),
    /// Checkpoint layers whose name starts with one of the given prefixes.
    ByName(Vec<String>),
    /// No checkpointing — store every activation.
    None,
    /// Checkpoint layers that would exceed a memory budget.
    GreedyMemory {
        /// Maximum total bytes to allow before triggering checkpointing.
        budget_bytes: usize,
    },
}

impl CheckpointPolicy {
    /// Decide whether a layer should be checkpointed.
    ///
    /// # Arguments
    /// * `layer_name`  - Human-readable name of the layer.
    /// * `layer_idx`   - Zero-based index of the layer in the model.
    /// * `memory_bytes`- Bytes that would be stored if not checkpointed.
    pub fn should_checkpoint(
        &self,
        layer_name: &str,
        layer_idx: usize,
        memory_bytes: usize,
    ) -> bool {
        match self {
            CheckpointPolicy::All => true,
            CheckpointPolicy::EveryN(n) => {
                if *n == 0 {
                    false
                } else {
                    layer_idx.is_multiple_of(*n)
                }
            },
            CheckpointPolicy::ByName(patterns) => {
                patterns.iter().any(|p| layer_name.starts_with(p.as_str()))
            },
            CheckpointPolicy::None => false,
            CheckpointPolicy::GreedyMemory { .. } => memory_bytes > 0,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointMemoryStats
// ─────────────────────────────────────────────────────────────────────────────

/// Memory statistics collected by [`CheckpointManager`].
#[derive(Debug, Clone, Default)]
pub struct CheckpointMemoryStats {
    /// Bytes currently occupied by stored (non-checkpointed) activations.
    pub stored_bytes: usize,
    /// Bytes that *would* have been stored but were checkpointed instead.
    pub checkpointed_bytes: usize,
    /// Cumulative bytes recomputed during backward passes.
    pub recomputed_bytes: usize,
    /// Number of layers whose activations are stored.
    pub num_stored_layers: usize,
    /// Number of layers that are checkpointed (will be recomputed).
    pub num_checkpointed_layers: usize,
    /// Fraction of bytes saved via checkpointing:
    /// `checkpointed_bytes / (stored_bytes + checkpointed_bytes)`.
    pub memory_savings_ratio: f32,
}

impl CheckpointMemoryStats {
    /// Recompute the `memory_savings_ratio` from the current byte counts.
    fn refresh_ratio(&mut self) {
        let total = self.stored_bytes + self.checkpointed_bytes;
        self.memory_savings_ratio =
            if total == 0 { 0.0 } else { self.checkpointed_bytes as f32 / total as f32 };
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointManager
// ─────────────────────────────────────────────────────────────────────────────

/// Manages storage and on-demand recomputation of layer activations.
///
/// During the forward pass, call [`CheckpointManager::record_activation`] for
/// each layer.  During the backward pass, call
/// [`CheckpointManager::get_activation`] to retrieve (or recompute) them.
pub struct CheckpointManager {
    /// Policy governing which layers are checkpointed.
    pub policy: CheckpointPolicy,
    snapshots: Vec<ActivationSnapshot>,
    recompute_log: Vec<String>,
    stats: CheckpointMemoryStats,
}

impl CheckpointManager {
    /// Create a new manager with the given checkpointing policy.
    pub fn new(policy: CheckpointPolicy) -> Self {
        Self {
            policy,
            snapshots: Vec::new(),
            recompute_log: Vec::new(),
            stats: CheckpointMemoryStats::default(),
        }
    }

    /// Record a forward activation.
    ///
    /// Based on the current policy the activation data is either stored
    /// in full or discarded (only shape and cost metadata are kept).
    pub fn record_activation(
        &mut self,
        layer_name: &str,
        layer_idx: usize,
        data: Vec<f32>,
        shape: Vec<usize>,
        computation_cost: f32,
    ) -> Result<(), CheckpointError> {
        let numel: usize = shape.iter().product();
        let bytes = numel * 4;

        let should_ckpt = self.policy.should_checkpoint(layer_name, layer_idx, bytes);

        if should_ckpt {
            let snap = ActivationSnapshot::checkpointed(layer_name, shape, computation_cost);
            self.stats.checkpointed_bytes += bytes;
            self.stats.num_checkpointed_layers += 1;
            self.snapshots.push(snap);
        } else {
            let snap = ActivationSnapshot::new(layer_name, data, shape);
            self.stats.stored_bytes += bytes;
            self.stats.num_stored_layers += 1;
            self.snapshots.push(snap);
        }

        self.stats.refresh_ratio();
        Ok(())
    }

    /// Retrieve the activation for a layer.
    ///
    /// If the activation is stored it is returned directly.  If it was
    /// checkpointed, `recompute_fn` is called to regenerate the data.
    pub fn get_activation(
        &mut self,
        layer_name: &str,
        recompute_fn: impl Fn() -> Result<Vec<f32>, CheckpointError>,
    ) -> Result<Vec<f32>, CheckpointError> {
        let pos = self.snapshots.iter().position(|s| s.layer_name == layer_name);

        match pos {
            None => Err(CheckpointError::LayerNotFound(layer_name.to_owned())),
            Some(idx) => {
                let snap = &self.snapshots[idx];
                if snap.is_stored() {
                    Ok(snap.tensor_data.clone())
                } else {
                    // Recompute
                    let data = recompute_fn()
                        .map_err(|e| CheckpointError::RecomputationFailed(e.to_string()))?;
                    let bytes = data.len() * 4;
                    self.stats.recomputed_bytes += bytes;
                    self.recompute_log.push(layer_name.to_owned());
                    Ok(data)
                }
            },
        }
    }

    /// Return a slice of all recorded snapshots.
    pub fn snapshots(&self) -> &[ActivationSnapshot] {
        &self.snapshots
    }

    /// Return the current memory statistics.
    pub fn memory_stats(&self) -> &CheckpointMemoryStats {
        &self.stats
    }

    /// Return the names of layers that were recomputed (in order).
    pub fn recompute_log(&self) -> &[String] {
        &self.recompute_log
    }

    /// Drop all snapshots and reset statistics.  Call this after the backward pass.
    pub fn clear(&mut self) {
        self.snapshots.clear();
        self.recompute_log.clear();
        self.stats = CheckpointMemoryStats::default();
    }

    /// Total bytes currently occupied by stored (not checkpointed) activations.
    pub fn current_memory_bytes(&self) -> usize {
        self.snapshots.iter().filter(|s| s.is_stored()).map(|s| s.memory_bytes()).sum()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SegmentCheckpointer
// ─────────────────────────────────────────────────────────────────────────────

/// Segment-based checkpointing analogous to PyTorch `checkpoint_sequential`.
///
/// The model is divided into `num_segments` equal segments.  Only the outputs
/// at segment boundaries are retained; all intermediate activations within a
/// segment must be recomputed during the backward pass.
pub struct SegmentCheckpointer {
    /// Number of segments.
    pub num_segments: usize,
    /// Stored outputs at each segment boundary.
    pub segment_outputs: Vec<Option<Vec<f32>>>,
    total_layers: usize,
}

impl SegmentCheckpointer {
    /// Create a new segment checkpointer.
    ///
    /// `num_segments` must be ≥ 1 and ≤ `total_layers`.
    pub fn new(num_segments: usize, total_layers: usize) -> Self {
        let cap = num_segments.max(1);
        Self {
            num_segments: cap,
            segment_outputs: vec![None; cap],
            total_layers,
        }
    }

    /// Return `true` if `layer_idx` is a segment boundary.
    pub fn is_boundary(&self, layer_idx: usize) -> bool {
        if self.total_layers == 0 {
            return false;
        }
        let segment_size = self.total_layers.div_ceil(self.num_segments);
        if segment_size == 0 {
            return false;
        }
        layer_idx.is_multiple_of(segment_size) || layer_idx == self.total_layers.saturating_sub(1)
    }

    /// Store the output for a segment.
    pub fn record_segment(
        &mut self,
        segment_idx: usize,
        output: Vec<f32>,
    ) -> Result<(), CheckpointError> {
        if segment_idx >= self.num_segments {
            return Err(CheckpointError::InvalidSegment(segment_idx));
        }
        self.segment_outputs[segment_idx] = Some(output);
        Ok(())
    }

    /// Retrieve a previously recorded segment output.
    pub fn get_segment_output(&self, segment_idx: usize) -> Option<&[f32]> {
        self.segment_outputs.get(segment_idx).and_then(|opt| opt.as_deref())
    }

    /// Fraction of layers that are segment boundaries (memory savings factor).
    ///
    /// A value of `0.25` means only 25 % of the layer outputs need to be kept.
    pub fn memory_savings_factor(&self) -> f32 {
        if self.total_layers == 0 {
            return 1.0;
        }
        self.num_segments as f32 / self.total_layers as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── ActivationSnapshot ──────────────────────────────────────────────────

    #[test]
    fn test_activation_snapshot_new() {
        let data = vec![1.0_f32, 2.0, 3.0, 4.0];
        let shape = vec![2, 2];
        let snap = ActivationSnapshot::new("layer0", data.clone(), shape.clone());
        assert_eq!(snap.layer_name, "layer0");
        assert_eq!(snap.tensor_data, data);
        assert_eq!(snap.shape, shape);
        assert!(!snap.is_checkpointed);
        assert!(snap.is_stored());
    }

    #[test]
    fn test_activation_snapshot_checkpointed() {
        let shape = vec![4, 8];
        let snap = ActivationSnapshot::checkpointed("attn", shape.clone(), 1024.0);
        assert_eq!(snap.layer_name, "attn");
        assert!(snap.tensor_data.is_empty());
        assert_eq!(snap.shape, shape);
        assert!(snap.is_checkpointed);
        assert!(!snap.is_stored());
        assert_eq!(snap.computation_cost, 1024.0);
    }

    #[test]
    fn test_activation_snapshot_memory_bytes() {
        let shape = vec![3, 4]; // 12 elements
        let snap = ActivationSnapshot::new("fc", vec![0.0; 12], shape);
        assert_eq!(snap.numel(), 12);
        assert_eq!(snap.memory_bytes(), 48); // 12 * 4 bytes
    }

    // ── CheckpointPolicy ────────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_policy_all() {
        let policy = CheckpointPolicy::All;
        assert!(policy.should_checkpoint("any", 0, 100));
        assert!(policy.should_checkpoint("layer99", 99, 0));
    }

    #[test]
    fn test_checkpoint_policy_every_n() {
        let policy = CheckpointPolicy::EveryN(3);
        assert!(policy.should_checkpoint("layer0", 0, 100));
        assert!(!policy.should_checkpoint("layer1", 1, 100));
        assert!(!policy.should_checkpoint("layer2", 2, 100));
        assert!(policy.should_checkpoint("layer3", 3, 100));
        assert!(policy.should_checkpoint("layer6", 6, 100));
        assert!(!policy.should_checkpoint("layer7", 7, 100));
    }

    #[test]
    fn test_checkpoint_policy_by_name() {
        let policy = CheckpointPolicy::ByName(vec!["attention".to_owned(), "ffn".to_owned()]);
        assert!(policy.should_checkpoint("attention_0", 0, 100));
        assert!(policy.should_checkpoint("ffn_layer", 1, 100));
        assert!(!policy.should_checkpoint("norm", 2, 100));
        assert!(!policy.should_checkpoint("embed", 3, 100));
    }

    #[test]
    fn test_checkpoint_policy_none() {
        let policy = CheckpointPolicy::None;
        assert!(!policy.should_checkpoint("any", 0, 999));
        assert!(!policy.should_checkpoint("layer", 5, 0));
    }

    // ── CheckpointManager ───────────────────────────────────────────────────

    #[test]
    fn test_checkpoint_manager_record_stored() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::None);
        let data = vec![1.0_f32; 16];
        mgr.record_activation("layer0", 0, data.clone(), vec![4, 4], 0.0)
            .expect("record should succeed");

        let snaps = mgr.snapshots();
        assert_eq!(snaps.len(), 1);
        assert!(snaps[0].is_stored());
        assert_eq!(snaps[0].tensor_data, data);
    }

    #[test]
    fn test_checkpoint_manager_record_checkpointed() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::All);
        mgr.record_activation("layer0", 0, vec![1.0; 8], vec![2, 4], 50.0)
            .expect("record should succeed");

        let snaps = mgr.snapshots();
        assert_eq!(snaps.len(), 1);
        assert!(!snaps[0].is_stored());
        assert!(snaps[0].is_checkpointed);
    }

    #[test]
    fn test_checkpoint_manager_get_activation_stored() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::None);
        let data = vec![3.0_f32, 1.0, 4.0, 1.0];
        mgr.record_activation("fc", 0, data.clone(), vec![4], 0.0)
            .expect("record should succeed");

        let retrieved = mgr
            .get_activation("fc", || {
                Err(CheckpointError::RecomputationFailed(
                    "should not call".into(),
                ))
            })
            .expect("get should succeed");

        assert_eq!(retrieved, data);
        assert_eq!(mgr.recompute_log().len(), 0);
    }

    #[test]
    fn test_checkpoint_manager_get_activation_recompute() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::All);
        mgr.record_activation("attn", 0, vec![0.0; 4], vec![2, 2], 100.0)
            .expect("record should succeed");

        let expected = vec![9.0_f32, 8.0, 7.0, 6.0];
        let exp_clone = expected.clone();
        let retrieved = mgr
            .get_activation("attn", move || Ok(exp_clone.clone()))
            .expect("recompute should succeed");

        assert_eq!(retrieved, expected);
        assert_eq!(mgr.recompute_log(), &["attn"]);
    }

    #[test]
    fn test_checkpoint_manager_memory_stats() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::EveryN(2));
        // layer 0 → checkpointed (0 % 2 == 0)
        mgr.record_activation("l0", 0, vec![0.0; 4], vec![4], 10.0).expect("ok");
        // layer 1 → stored (1 % 2 != 0)
        mgr.record_activation("l1", 1, vec![1.0; 4], vec![4], 10.0).expect("ok");

        let stats = mgr.memory_stats();
        assert_eq!(stats.num_checkpointed_layers, 1);
        assert_eq!(stats.num_stored_layers, 1);
        assert_eq!(stats.checkpointed_bytes, 16);
        assert_eq!(stats.stored_bytes, 16);
        assert!((stats.memory_savings_ratio - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_checkpoint_manager_clear() {
        let mut mgr = CheckpointManager::new(CheckpointPolicy::None);
        mgr.record_activation("l0", 0, vec![1.0; 8], vec![8], 0.0).expect("ok");
        assert_eq!(mgr.snapshots().len(), 1);
        mgr.clear();
        assert_eq!(mgr.snapshots().len(), 0);
        assert_eq!(mgr.memory_stats().stored_bytes, 0);
    }

    // ── SegmentCheckpointer ─────────────────────────────────────────────────

    #[test]
    fn test_segment_checkpointer_boundaries() {
        // 12 layers, 3 segments → segment size = 4
        let sc = SegmentCheckpointer::new(3, 12);
        assert!(sc.is_boundary(0));
        assert!(!sc.is_boundary(1));
        assert!(!sc.is_boundary(2));
        assert!(!sc.is_boundary(3));
        assert!(sc.is_boundary(4));
        assert!(sc.is_boundary(8));
        assert!(sc.is_boundary(11)); // last layer is always a boundary
    }

    #[test]
    fn test_segment_checkpointer_record_get() {
        let mut sc = SegmentCheckpointer::new(4, 16);
        let out = vec![1.0_f32, 2.0, 3.0];
        sc.record_segment(1, out.clone()).expect("record ok");
        let got = sc.get_segment_output(1).expect("should exist");
        assert_eq!(got, out.as_slice());
        assert!(sc.get_segment_output(2).is_none());
    }

    #[test]
    fn test_segment_checkpointer_memory_savings() {
        // 4 segments / 16 layers = 0.25
        let sc = SegmentCheckpointer::new(4, 16);
        let factor = sc.memory_savings_factor();
        assert!((factor - 0.25).abs() < 1e-6);
    }

    // ── CheckpointError Display ─────────────────────────────────────────────

    #[test]
    fn test_checkpoint_error_display() {
        let e1 = CheckpointError::LayerNotFound("block3".into());
        assert!(e1.to_string().contains("block3"));

        let e2 = CheckpointError::RecomputationFailed("oom".into());
        assert!(e2.to_string().contains("oom"));

        let e3 = CheckpointError::InvalidShape {
            expected: vec![4, 8],
            got: vec![4, 4],
        };
        assert!(e3.to_string().contains("mismatch"));

        let e4 = CheckpointError::InvalidSegment(7);
        assert!(e4.to_string().contains('7'));
    }
}
