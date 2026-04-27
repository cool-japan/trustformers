//! Selective activation checkpointing strategies.
//! Balance memory savings vs recomputation cost.
//!
//! This module provides advanced strategies for deciding *which* layers to
//! checkpoint (discard activations and recompute them on the backward pass)
//! versus which to store in full.  Different strategies expose different
//! tradeoffs between peak memory usage and total recomputation FLOPs.

use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by checkpoint planning functions.
#[derive(Debug, Clone, PartialEq)]
pub enum CkptError {
    /// The layer profile list was empty when a non-empty list was required.
    EmptyProfiles,
    /// A strategy parameter was out of range (e.g. ratio > 1.0).
    InvalidParameter(String),
    /// A layer index referenced in the plan does not exist in the profile list.
    LayerIndexOutOfBounds(usize),
}

impl fmt::Display for CkptError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CkptError::EmptyProfiles => write!(f, "layer profile list is empty"),
            CkptError::InvalidParameter(msg) => write!(f, "invalid parameter: {msg}"),
            CkptError::LayerIndexOutOfBounds(idx) => {
                write!(f, "layer index {idx} is out of bounds")
            },
        }
    }
}

impl std::error::Error for CkptError {}

// ─────────────────────────────────────────────────────────────────────────────
// LayerTypeHint
// ─────────────────────────────────────────────────────────────────────────────

/// Coarse classification of a transformer layer's role.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerTypeHint {
    /// Multi-head (self/cross) attention layer.
    Attention,
    /// Feed-forward network (MLP) sublayer.
    Ffn,
    /// Token or positional embedding layer.
    Embedding,
    /// Layer normalisation (RMSNorm, LayerNorm, …).
    Norm,
    /// Any other layer type.
    Other,
}

impl fmt::Display for LayerTypeHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerTypeHint::Attention => write!(f, "Attention"),
            LayerTypeHint::Ffn => write!(f, "FFN"),
            LayerTypeHint::Embedding => write!(f, "Embedding"),
            LayerTypeHint::Norm => write!(f, "Norm"),
            LayerTypeHint::Other => write!(f, "Other"),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerMemoryProfile
// ─────────────────────────────────────────────────────────────────────────────

/// Memory and compute profile for a single model layer.
///
/// Used by [`build_checkpoint_plan`] to decide whether to checkpoint the layer.
#[derive(Debug, Clone)]
pub struct LayerMemoryProfile {
    /// Zero-based layer index within the model.
    pub layer_idx: usize,
    /// Broad category of the layer.
    pub layer_type: LayerTypeHint,
    /// Bytes occupied by the layer's activations if they are stored in full.
    pub activation_size_bytes: usize,
    /// Estimated floating-point operations required to recompute this layer.
    pub recompute_flops: u64,
}

impl LayerMemoryProfile {
    /// Create a new profile.
    pub fn new(
        layer_idx: usize,
        layer_type: LayerTypeHint,
        activation_size_bytes: usize,
        recompute_flops: u64,
    ) -> Self {
        Self {
            layer_idx,
            layer_type,
            activation_size_bytes,
            recompute_flops,
        }
    }

    /// Ratio of recomputation FLOPs to stored activation bytes.
    ///
    /// A *higher* ratio means this layer is more expensive to recompute
    /// relative to the memory saved, so it is a *worse* candidate for
    /// checkpointing.
    pub fn recompute_cost_ratio(&self) -> f32 {
        if self.activation_size_bytes == 0 {
            return 0.0;
        }
        self.recompute_flops as f32 / self.activation_size_bytes as f32
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SelectiveCheckpointStrategy
// ─────────────────────────────────────────────────────────────────────────────

/// Strategy that determines which layers are checkpointed.
#[derive(Debug, Clone, PartialEq)]
pub enum SelectiveCheckpointStrategy {
    /// No checkpointing — all layer activations are stored (maximum memory,
    /// zero recomputation overhead).
    None,

    /// Checkpoint every layer (maximum memory savings, maximum recomputation).
    Full,

    /// Checkpoint approximately `checkpoint_ratio` of layers, chosen evenly
    /// across the depth of the network.
    ///
    /// `checkpoint_ratio` must be in `[0.0, 1.0]`.
    Selective { checkpoint_ratio: f32 },

    /// Checkpoint only attention layers; FFN activations are kept in memory.
    AttentionOnly,

    /// Checkpoint only FFN (feed-forward) layers; attention activations are kept.
    FfnOnly,

    /// Checkpoint every N-th layer (layers where `layer_idx % n == 0`).
    ///
    /// `n == 0` is treated as "no checkpointing".
    EveryNthLayer { n: usize },

    /// Checkpoint the `top_k` layers with the largest activation footprint.
    LargestLayers { top_k: usize },

    /// Greedily checkpoint layers until the total saved activation memory falls
    /// within the given budget.
    ///
    /// Layers are added to the checkpoint set in order of "best savings per
    /// recompute cost" (largest `activation_size / recompute_flops`).
    Budget { max_memory_mb: f32 },
}

// ─────────────────────────────────────────────────────────────────────────────
// CheckpointPlan
// ─────────────────────────────────────────────────────────────────────────────

/// Output of [`build_checkpoint_plan`]: which layers to checkpoint and which to
/// keep stored, together with aggregate statistics.
#[derive(Debug, Clone)]
pub struct CheckpointPlan {
    /// Indices of layers whose activations will be *discarded* (recomputed
    /// during the backward pass).
    pub checkpointed_layers: Vec<usize>,

    /// Indices of layers whose activations will be *stored* in full.
    pub saved_layers: Vec<usize>,

    /// Total bytes stored for checkpointed-layer activations.
    ///
    /// Under pure checkpointing this is zero because the data is discarded;
    /// the field represents the "peak" per-segment cost if recomputation is
    /// serialised.  For simplicity the planner sets this to the size of the
    /// single largest checkpointed activation.
    pub total_activation_memory_bytes: usize,

    /// Total FLOPs needed to recompute all checkpointed layers.
    pub total_recompute_flops: u64,
}

impl CheckpointPlan {
    /// Fraction of layers that are checkpointed (`0.0` … `1.0`).
    pub fn checkpoint_ratio(&self) -> f32 {
        let total = self.checkpointed_layers.len() + self.saved_layers.len();
        if total == 0 {
            return 0.0;
        }
        self.checkpointed_layers.len() as f32 / total as f32
    }

    /// Ratio by which memory is reduced relative to storing all activations.
    ///
    /// A value of `0.5` means peak memory is roughly halved.
    pub fn estimated_memory_reduction_ratio(&self) -> f32 {
        let total = self.checkpointed_layers.len() + self.saved_layers.len();
        if total == 0 {
            return 0.0;
        }
        // Approximation: fraction of layers that are checkpointed ≈ fraction
        // of activation memory that is freed.
        self.checkpoint_ratio()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// ActivationStats
// ─────────────────────────────────────────────────────────────────────────────

/// Summary statistics computed from a [`CheckpointPlan`] and its profiles.
#[derive(Debug, Clone)]
pub struct ActivationStats {
    /// Peak memory (bytes) that must be resident at any point during training.
    pub peak_memory_bytes: usize,
    /// Average activation size across all layers.
    pub mean_activation_size_bytes: f32,
    /// Total number of layers in the model.
    pub total_layers: usize,
    /// Number of layers that are checkpointed.
    pub checkpointed_layers: usize,
}

impl ActivationStats {
    /// Percentage reduction in activation memory compared to storing everything.
    pub fn memory_reduction_percent(&self) -> f32 {
        if self.total_layers == 0 {
            return 0.0;
        }
        (self.checkpointed_layers as f32 / self.total_layers as f32) * 100.0
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// build_checkpoint_plan
// ─────────────────────────────────────────────────────────────────────────────

/// Build a [`CheckpointPlan`] for the given layer profiles using the selected
/// strategy.
///
/// # Errors
///
/// Returns [`CkptError::InvalidParameter`] if a strategy parameter is invalid
/// (e.g. `checkpoint_ratio` outside `[0.0, 1.0]`, `n == 0` for
/// [`SelectiveCheckpointStrategy::EveryNthLayer`], or `max_memory_mb < 0`).
pub fn build_checkpoint_plan(
    profiles: &[LayerMemoryProfile],
    strategy: &SelectiveCheckpointStrategy,
) -> Result<CheckpointPlan, CkptError> {
    match strategy {
        SelectiveCheckpointStrategy::None => plan_none(profiles),
        SelectiveCheckpointStrategy::Full => plan_full(profiles),
        SelectiveCheckpointStrategy::Selective { checkpoint_ratio } => {
            plan_selective(profiles, *checkpoint_ratio)
        },
        SelectiveCheckpointStrategy::AttentionOnly => {
            plan_by_type(profiles, LayerTypeHint::Attention)
        },
        SelectiveCheckpointStrategy::FfnOnly => plan_by_type(profiles, LayerTypeHint::Ffn),
        SelectiveCheckpointStrategy::EveryNthLayer { n } => plan_every_nth(profiles, *n),
        SelectiveCheckpointStrategy::LargestLayers { top_k } => plan_largest(profiles, *top_k),
        SelectiveCheckpointStrategy::Budget { max_memory_mb } => {
            plan_budget(profiles, *max_memory_mb)
        },
    }
}

// ── None ────────────────────────────────────────────────────────────────────

fn plan_none(profiles: &[LayerMemoryProfile]) -> Result<CheckpointPlan, CkptError> {
    let saved_layers: Vec<usize> = profiles.iter().map(|p| p.layer_idx).collect();
    Ok(CheckpointPlan {
        checkpointed_layers: Vec::new(),
        saved_layers,
        total_activation_memory_bytes: 0,
        total_recompute_flops: 0,
    })
}

// ── Full ────────────────────────────────────────────────────────────────────

fn plan_full(profiles: &[LayerMemoryProfile]) -> Result<CheckpointPlan, CkptError> {
    let checkpointed_layers: Vec<usize> = profiles.iter().map(|p| p.layer_idx).collect();
    let peak = profiles.iter().map(|p| p.activation_size_bytes).max().unwrap_or(0);
    let total_recompute_flops = profiles.iter().map(|p| p.recompute_flops).sum();
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers: Vec::new(),
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ── Selective ───────────────────────────────────────────────────────────────

fn plan_selective(
    profiles: &[LayerMemoryProfile],
    checkpoint_ratio: f32,
) -> Result<CheckpointPlan, CkptError> {
    if !(0.0..=1.0).contains(&checkpoint_ratio) {
        return Err(CkptError::InvalidParameter(format!(
            "checkpoint_ratio {checkpoint_ratio} is not in [0.0, 1.0]"
        )));
    }
    let n_layers = profiles.len();
    if n_layers == 0 {
        return Ok(CheckpointPlan {
            checkpointed_layers: Vec::new(),
            saved_layers: Vec::new(),
            total_activation_memory_bytes: 0,
            total_recompute_flops: 0,
        });
    }

    // Checkpoint `round(ratio * n)` layers, spaced evenly.
    let n_to_checkpoint = (checkpoint_ratio * n_layers as f32).round() as usize;
    let step = if n_to_checkpoint == 0 {
        usize::MAX
    } else {
        // Distribute evenly: checkpoint layer at positions 0, step, 2*step, …
        // Use a spacing of 1/ratio.
        (n_layers as f32 / n_to_checkpoint as f32).round().max(1.0) as usize
    };

    let mut checkpointed_layers = Vec::new();
    let mut saved_layers = Vec::new();

    for (i, profile) in profiles.iter().enumerate() {
        if step != usize::MAX && i % step == 0 && checkpointed_layers.len() < n_to_checkpoint {
            checkpointed_layers.push(profile.layer_idx);
        } else {
            saved_layers.push(profile.layer_idx);
        }
    }

    let peak = checkpointed_peak(profiles, &checkpointed_layers);
    let total_recompute_flops = recompute_flops_for(profiles, &checkpointed_layers);
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers,
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ── By type ─────────────────────────────────────────────────────────────────

fn plan_by_type(
    profiles: &[LayerMemoryProfile],
    target_type: LayerTypeHint,
) -> Result<CheckpointPlan, CkptError> {
    let mut checkpointed_layers = Vec::new();
    let mut saved_layers = Vec::new();

    for profile in profiles {
        if profile.layer_type == target_type {
            checkpointed_layers.push(profile.layer_idx);
        } else {
            saved_layers.push(profile.layer_idx);
        }
    }

    let peak = checkpointed_peak(profiles, &checkpointed_layers);
    let total_recompute_flops = recompute_flops_for(profiles, &checkpointed_layers);
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers,
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ── Every Nth ────────────────────────────────────────────────────────────────

fn plan_every_nth(profiles: &[LayerMemoryProfile], n: usize) -> Result<CheckpointPlan, CkptError> {
    if n == 0 {
        return Err(CkptError::InvalidParameter(
            "n must be > 0 for EveryNthLayer strategy".to_owned(),
        ));
    }
    let mut checkpointed_layers = Vec::new();
    let mut saved_layers = Vec::new();

    for (i, profile) in profiles.iter().enumerate() {
        if i % n == 0 {
            checkpointed_layers.push(profile.layer_idx);
        } else {
            saved_layers.push(profile.layer_idx);
        }
    }

    let peak = checkpointed_peak(profiles, &checkpointed_layers);
    let total_recompute_flops = recompute_flops_for(profiles, &checkpointed_layers);
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers,
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ── Largest layers ───────────────────────────────────────────────────────────

fn plan_largest(
    profiles: &[LayerMemoryProfile],
    top_k: usize,
) -> Result<CheckpointPlan, CkptError> {
    // Sort indices by activation size descending.
    let mut order: Vec<usize> = (0..profiles.len()).collect();
    order.sort_by(|&a, &b| {
        profiles[b].activation_size_bytes.cmp(&profiles[a].activation_size_bytes)
    });

    let checkpoint_set: std::collections::HashSet<usize> =
        order.iter().take(top_k).map(|&i| profiles[i].layer_idx).collect();

    let mut checkpointed_layers: Vec<usize> = Vec::new();
    let mut saved_layers: Vec<usize> = Vec::new();

    for profile in profiles {
        if checkpoint_set.contains(&profile.layer_idx) {
            checkpointed_layers.push(profile.layer_idx);
        } else {
            saved_layers.push(profile.layer_idx);
        }
    }

    let peak = checkpointed_peak(profiles, &checkpointed_layers);
    let total_recompute_flops = recompute_flops_for(profiles, &checkpointed_layers);
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers,
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ── Budget ───────────────────────────────────────────────────────────────────

fn plan_budget(
    profiles: &[LayerMemoryProfile],
    max_memory_mb: f32,
) -> Result<CheckpointPlan, CkptError> {
    if max_memory_mb < 0.0 {
        return Err(CkptError::InvalidParameter(format!(
            "max_memory_mb must be >= 0, got {max_memory_mb}"
        )));
    }
    let budget_bytes = (max_memory_mb * 1024.0 * 1024.0) as usize;

    // Total memory if we stored everything.
    let total_bytes: usize = profiles.iter().map(|p| p.activation_size_bytes).sum();

    if total_bytes <= budget_bytes {
        // Already within budget — no checkpointing needed.
        return plan_none(profiles);
    }

    // Sort by "memory saved per recompute cost" — best candidates first.
    // Score = activation_size / (recompute_flops + 1) to avoid division by zero.
    let mut order: Vec<usize> = (0..profiles.len()).collect();
    order.sort_by(|&a, &b| {
        let score_a =
            profiles[a].activation_size_bytes as f64 / (profiles[a].recompute_flops as f64 + 1.0);
        let score_b =
            profiles[b].activation_size_bytes as f64 / (profiles[b].recompute_flops as f64 + 1.0);
        score_b.partial_cmp(&score_a).unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut remaining = total_bytes;
    let mut to_checkpoint = std::collections::HashSet::new();

    for &i in &order {
        if remaining <= budget_bytes {
            break;
        }
        to_checkpoint.insert(profiles[i].layer_idx);
        remaining = remaining.saturating_sub(profiles[i].activation_size_bytes);
    }

    let mut checkpointed_layers: Vec<usize> = Vec::new();
    let mut saved_layers: Vec<usize> = Vec::new();

    for profile in profiles {
        if to_checkpoint.contains(&profile.layer_idx) {
            checkpointed_layers.push(profile.layer_idx);
        } else {
            saved_layers.push(profile.layer_idx);
        }
    }

    let peak = checkpointed_peak(profiles, &checkpointed_layers);
    let total_recompute_flops = recompute_flops_for(profiles, &checkpointed_layers);
    Ok(CheckpointPlan {
        checkpointed_layers,
        saved_layers,
        total_activation_memory_bytes: peak,
        total_recompute_flops,
    })
}

// ─────────────────────────────────────────────────────────────────────────────
// Helper functions
// ─────────────────────────────────────────────────────────────────────────────

/// Peak activation size (bytes) for a set of checkpointed layers.
/// Under serialised recomputation the peak is the size of the largest single
/// checkpointed activation.
fn checkpointed_peak(profiles: &[LayerMemoryProfile], checkpointed: &[usize]) -> usize {
    let idx_set: std::collections::HashSet<usize> = checkpointed.iter().copied().collect();
    profiles
        .iter()
        .filter(|p| idx_set.contains(&p.layer_idx))
        .map(|p| p.activation_size_bytes)
        .max()
        .unwrap_or(0)
}

/// Sum FLOPs for the given set of checkpointed layer indices.
fn recompute_flops_for(profiles: &[LayerMemoryProfile], checkpointed: &[usize]) -> u64 {
    let idx_set: std::collections::HashSet<usize> = checkpointed.iter().copied().collect();
    profiles
        .iter()
        .filter(|p| idx_set.contains(&p.layer_idx))
        .map(|p| p.recompute_flops)
        .sum()
}

// ─────────────────────────────────────────────────────────────────────────────
// Public utility functions
// ─────────────────────────────────────────────────────────────────────────────

/// Estimate the total memory (bytes) required given a checkpoint plan.
///
/// Memory = sum(activation_size for saved_layers) + peak activation size across
/// checkpointed layers (only one is resident during recomputation at a time).
pub fn estimate_memory_with_plan(profiles: &[LayerMemoryProfile], plan: &CheckpointPlan) -> usize {
    let saved_set: std::collections::HashSet<usize> = plan.saved_layers.iter().copied().collect();
    let saved_bytes: usize = profiles
        .iter()
        .filter(|p| saved_set.contains(&p.layer_idx))
        .map(|p| p.activation_size_bytes)
        .sum();
    saved_bytes + plan.total_activation_memory_bytes
}

/// Estimate the total recomputation FLOPs for a checkpoint plan.
pub fn estimate_total_recompute_flops(
    profiles: &[LayerMemoryProfile],
    plan: &CheckpointPlan,
) -> u64 {
    recompute_flops_for(profiles, &plan.checkpointed_layers)
}

/// Build aggregate activation statistics for a given plan.
pub fn activation_stats(profiles: &[LayerMemoryProfile], plan: &CheckpointPlan) -> ActivationStats {
    let total_layers = profiles.len();
    let peak_memory_bytes = estimate_memory_with_plan(profiles, plan);
    let mean_activation_size_bytes = if total_layers == 0 {
        0.0
    } else {
        profiles.iter().map(|p| p.activation_size_bytes as f32).sum::<f32>() / total_layers as f32
    };
    ActivationStats {
        peak_memory_bytes,
        mean_activation_size_bytes,
        total_layers,
        checkpointed_layers: plan.checkpointed_layers.len(),
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn attn(idx: usize, size: usize, flops: u64) -> LayerMemoryProfile {
        LayerMemoryProfile::new(idx, LayerTypeHint::Attention, size, flops)
    }

    fn ffn(idx: usize, size: usize, flops: u64) -> LayerMemoryProfile {
        LayerMemoryProfile::new(idx, LayerTypeHint::Ffn, size, flops)
    }

    fn other(idx: usize, size: usize, flops: u64) -> LayerMemoryProfile {
        LayerMemoryProfile::new(idx, LayerTypeHint::Other, size, flops)
    }

    fn profiles_6() -> Vec<LayerMemoryProfile> {
        vec![
            attn(0, 1024, 2_000_000),
            ffn(1, 512, 1_000_000),
            attn(2, 1024, 2_000_000),
            ffn(3, 512, 1_000_000),
            attn(4, 1024, 2_000_000),
            ffn(5, 512, 1_000_000),
        ]
    }

    // ── Test 1: None strategy saves all layers ────────────────────────────

    #[test]
    fn test_none_strategy_saves_all() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::None).expect("ok");
        assert!(plan.checkpointed_layers.is_empty());
        assert_eq!(plan.saved_layers.len(), 6);
        assert_eq!(plan.total_recompute_flops, 0);
        assert_eq!(plan.total_activation_memory_bytes, 0);
    }

    // ── Test 2: Full checkpoints all ─────────────────────────────────────

    #[test]
    fn test_full_checkpoints_all() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::Full).expect("ok");
        assert_eq!(plan.checkpointed_layers.len(), 6);
        assert!(plan.saved_layers.is_empty());
        // Peak = max activation size = 1024
        assert_eq!(plan.total_activation_memory_bytes, 1024);
        assert_eq!(plan.total_recompute_flops, 9_000_000);
    }

    // ── Test 3: EveryNthLayer ─────────────────────────────────────────────

    #[test]
    fn test_every_nth_layer() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::EveryNthLayer { n: 2 },
        )
        .expect("ok");
        // Positions 0, 2, 4 are checkpointed
        assert_eq!(plan.checkpointed_layers, vec![0, 2, 4]);
        assert_eq!(plan.saved_layers, vec![1, 3, 5]);
    }

    // ── Test 4: EveryNthLayer n=0 returns error ───────────────────────────

    #[test]
    fn test_every_nth_layer_zero_is_error() {
        let profiles = profiles_6();
        let result = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::EveryNthLayer { n: 0 },
        );
        assert!(result.is_err());
        match result.unwrap_err() {
            CkptError::InvalidParameter(_) => {},
            other => panic!("expected InvalidParameter, got {other:?}"),
        }
    }

    // ── Test 5: AttentionOnly filters correctly ───────────────────────────

    #[test]
    fn test_attention_only_filters() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::AttentionOnly)
            .expect("ok");
        assert_eq!(plan.checkpointed_layers, vec![0, 2, 4]);
        assert_eq!(plan.saved_layers, vec![1, 3, 5]);
    }

    // ── Test 6: FfnOnly filters correctly ────────────────────────────────

    #[test]
    fn test_ffn_only_filters() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::FfnOnly).expect("ok");
        assert_eq!(plan.checkpointed_layers, vec![1, 3, 5]);
        assert_eq!(plan.saved_layers, vec![0, 2, 4]);
    }

    // ── Test 7: LargestLayers top-k ──────────────────────────────────────

    #[test]
    fn test_largest_layers_top_k() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::LargestLayers { top_k: 2 },
        )
        .expect("ok");
        // The 3 attention layers are size 1024; top 2 are any 2 of them.
        assert_eq!(plan.checkpointed_layers.len(), 2);
        assert_eq!(plan.saved_layers.len(), 4);
        for &idx in &plan.checkpointed_layers {
            assert!(profiles[idx].activation_size_bytes == 1024);
        }
    }

    // ── Test 8: Selective ratio ~0.5 ─────────────────────────────────────

    #[test]
    fn test_selective_ratio_half() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::Selective {
                checkpoint_ratio: 0.5,
            },
        )
        .expect("ok");
        // 0.5 * 6 = 3 checkpointed
        assert_eq!(plan.checkpointed_layers.len(), 3);
        assert_eq!(plan.saved_layers.len(), 3);
    }

    // ── Test 9: Selective ratio out of range is error ─────────────────────

    #[test]
    fn test_selective_ratio_invalid() {
        let profiles = profiles_6();
        let result = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::Selective {
                checkpoint_ratio: 1.5,
            },
        );
        assert!(result.is_err());
    }

    // ── Test 10: Budget respects memory limit ─────────────────────────────

    #[test]
    fn test_budget_respects_limit() {
        // 6 layers each 1 MB of activations (1_048_576 bytes)
        let profiles: Vec<LayerMemoryProfile> = (0..6)
            .map(|i| LayerMemoryProfile::new(i, LayerTypeHint::Other, 1_048_576, 1_000_000))
            .collect();
        // Budget: 3 MB. Total is 6 MB, so we need to checkpoint ~3 layers.
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::Budget { max_memory_mb: 3.0 },
        )
        .expect("ok");
        // After checkpointing, saved memory ≤ budget
        let saved_bytes: usize =
            plan.saved_layers.iter().map(|&i| profiles[i].activation_size_bytes).sum();
        assert!(
            saved_bytes <= 3 * 1_048_576,
            "saved_bytes {saved_bytes} exceeded budget"
        );
    }

    // ── Test 11: Memory estimate ──────────────────────────────────────────

    #[test]
    fn test_memory_estimate() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::None).expect("ok");
        let mem = estimate_memory_with_plan(&profiles, &plan);
        // None strategy: all layers saved, no peak from checkpointed layers
        let expected: usize = profiles.iter().map(|p| p.activation_size_bytes).sum();
        assert_eq!(mem, expected);
    }

    // ── Test 12: Recompute flops estimate ────────────────────────────────

    #[test]
    fn test_recompute_flops_estimate() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::Full).expect("ok");
        let flops = estimate_total_recompute_flops(&profiles, &plan);
        assert_eq!(flops, 9_000_000);
    }

    // ── Test 13: checkpoint_ratio calculation ────────────────────────────

    #[test]
    fn test_checkpoint_ratio_calculation() {
        let profiles = profiles_6();
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::Full).expect("ok");
        let ratio = plan.checkpoint_ratio();
        assert!((ratio - 1.0).abs() < 1e-6, "expected 1.0, got {ratio}");

        let plan_none =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::None).expect("ok");
        let ratio_none = plan_none.checkpoint_ratio();
        assert!((ratio_none - 0.0).abs() < 1e-6);
    }

    // ── Test 14: Memory reduction ratio ─────────────────────────────────

    #[test]
    fn test_memory_reduction_ratio() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::Selective {
                checkpoint_ratio: 0.5,
            },
        )
        .expect("ok");
        let ratio = plan.estimated_memory_reduction_ratio();
        // 3 / 6 = 0.5
        assert!((ratio - 0.5).abs() < 0.1, "unexpected ratio {ratio}");
    }

    // ── Test 15: Empty profile ────────────────────────────────────────────

    #[test]
    fn test_empty_profile() {
        let profiles: Vec<LayerMemoryProfile> = vec![];
        let plan =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::Full).expect("ok");
        assert!(plan.checkpointed_layers.is_empty());
        assert!(plan.saved_layers.is_empty());
        assert_eq!(plan.total_activation_memory_bytes, 0);
        assert_eq!(plan.total_recompute_flops, 0);

        let plan_none =
            build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::None).expect("ok");
        assert!(plan_none.checkpointed_layers.is_empty());

        // checkpoint_ratio on empty plan should be 0.0
        assert_eq!(plan.checkpoint_ratio(), 0.0);
    }

    // ── Test 16: ActivationStats memory_reduction_percent ─────────────────

    #[test]
    fn test_activation_stats_memory_reduction_percent() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::AttentionOnly)
            .expect("ok");
        let stats = activation_stats(&profiles, &plan);
        assert_eq!(stats.total_layers, 6);
        assert_eq!(stats.checkpointed_layers, 3);
        // 3/6 * 100 = 50%
        let pct = stats.memory_reduction_percent();
        assert!((pct - 50.0).abs() < 1e-4, "got {pct}");
    }

    // ── Test 17: recompute_cost_ratio ─────────────────────────────────────

    #[test]
    fn test_recompute_cost_ratio() {
        let p = attn(0, 1024, 2_048_000);
        let ratio = p.recompute_cost_ratio();
        assert!((ratio - 2000.0).abs() < 1.0, "got {ratio}");

        // Zero activation size -> ratio is 0
        let p2 = attn(1, 0, 100);
        assert_eq!(p2.recompute_cost_ratio(), 0.0);
    }

    // ── Test 18: LargestLayers with top_k >= num_layers checkpoints all ───

    #[test]
    fn test_largest_layers_all() {
        let profiles = profiles_6();
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::LargestLayers { top_k: 10 },
        )
        .expect("ok");
        assert_eq!(plan.checkpointed_layers.len(), 6);
        assert!(plan.saved_layers.is_empty());
    }

    // ── Test 19: Budget no-op when already within limit ───────────────────

    #[test]
    fn test_budget_noop_when_within_limit() {
        let profiles: Vec<LayerMemoryProfile> = (0..3)
            .map(|i| LayerMemoryProfile::new(i, LayerTypeHint::Other, 1024, 10_000))
            .collect();
        // Total = 3072 bytes ≈ 0.003 MB. Budget 100 MB is plenty.
        let plan = build_checkpoint_plan(
            &profiles,
            &SelectiveCheckpointStrategy::Budget {
                max_memory_mb: 100.0,
            },
        )
        .expect("ok");
        assert!(plan.checkpointed_layers.is_empty());
        assert_eq!(plan.saved_layers.len(), 3);
    }

    // ── Test 20: Other layer type correctly routed ────────────────────────

    #[test]
    fn test_other_layer_type_attention_only() {
        let profiles = vec![
            other(0, 512, 500_000),
            attn(1, 1024, 2_000_000),
            other(2, 512, 500_000),
        ];
        let plan = build_checkpoint_plan(&profiles, &SelectiveCheckpointStrategy::AttentionOnly)
            .expect("ok");
        assert_eq!(plan.checkpointed_layers, vec![1]);
        assert_eq!(plan.saved_layers, vec![0, 2]);
    }
}
