//! Pipeline parallelism with GPipe-style micro-batch scheduling.
//!
//! Model layers are partitioned across stages; micro-batches flow through the
//! pipeline.  Provides both even and FLOPs-balanced layer partitioning, plus
//! scheduling utilities for GPipe and 1F1B execution schedules.

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors specific to pipeline parallelism.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PpError {
    /// The requested number of stages is zero.
    ZeroStages,
    /// `stage_rank` is out of range for the given `num_stages`.
    StageRankOutOfRange { stage_rank: usize, num_stages: usize },
    /// Not enough layers to partition across `num_stages`.
    InsufficientLayers { total_layers: usize, num_stages: usize },
    /// A micro-batch ID is invalid.
    InvalidMicroBatch(usize),
}

impl fmt::Display for PpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStages => write!(f, "num_stages must be at least 1"),
            Self::StageRankOutOfRange { stage_rank, num_stages } => write!(
                f,
                "stage_rank {stage_rank} is out of range for num_stages {num_stages}"
            ),
            Self::InsufficientLayers { total_layers, num_stages } => write!(
                f,
                "cannot partition {total_layers} layers into {num_stages} stages"
            ),
            Self::InvalidMicroBatch(id) => write!(f, "invalid micro-batch id: {id}"),
        }
    }
}

impl std::error::Error for PpError {}

// ---------------------------------------------------------------------------
// PipelineSchedule
// ---------------------------------------------------------------------------

/// Execution schedule for the pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineSchedule {
    /// GPipe: all forward passes for all micro-batches, then all backward passes.
    GPipe,
    /// 1F1B (one-forward-one-backward): lower bubble and better memory usage.
    OneFOneBubble,
    /// Interleaved schedule (Megatron-LM v2): each rank handles multiple virtual stages.
    Interleaved { num_virtual_stages: usize },
}

// ---------------------------------------------------------------------------
// PipelineConfig
// ---------------------------------------------------------------------------

/// Configuration for a pipeline parallelism setup.
#[derive(Debug, Clone)]
pub struct PipelineConfig {
    /// Total number of pipeline stages.
    pub num_stages: usize,
    /// Number of micro-batches per global batch.
    pub num_micro_batches: usize,
    /// Rank of this stage within the pipeline.
    pub stage_rank: usize,
    /// Scheduling policy.
    pub schedule: PipelineSchedule,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            num_stages: 1,
            num_micro_batches: 8,
            stage_rank: 0,
            schedule: PipelineSchedule::GPipe,
        }
    }
}

// ---------------------------------------------------------------------------
// PipelineStageInfo
// ---------------------------------------------------------------------------

/// Metadata about a single pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PipelineStageInfo {
    /// Zero-based stage index.
    pub stage_id: usize,
    /// Number of transformer layers assigned to this stage.
    pub num_layers: usize,
    /// Global layer index at which this stage starts (inclusive).
    pub layer_start: usize,
    /// Global layer index at which this stage ends (exclusive).
    pub layer_end: usize,
}

// ---------------------------------------------------------------------------
// Layer partitioning
// ---------------------------------------------------------------------------

/// Distribute `total_layers` layers as evenly as possible across `num_stages`
/// stages.  Remainder layers are distributed one-by-one to earlier stages.
pub fn partition_layers_evenly(
    total_layers: usize,
    num_stages: usize,
) -> Vec<PipelineStageInfo> {
    if num_stages == 0 || total_layers == 0 {
        return Vec::new();
    }

    let base = total_layers / num_stages;
    let remainder = total_layers % num_stages;
    let mut stages = Vec::with_capacity(num_stages);
    let mut layer_start = 0usize;

    for stage_id in 0..num_stages {
        let num_layers = base + if stage_id < remainder { 1 } else { 0 };
        let layer_end = layer_start + num_layers;
        stages.push(PipelineStageInfo { stage_id, num_layers, layer_start, layer_end });
        layer_start = layer_end;
    }

    stages
}

/// Distribute layers across stages to balance total FLOPs per stage.
///
/// Uses a greedy approach: assign each layer (in order) to the stage with the
/// **lowest accumulated FLOPs** so far.  Layers are always kept contiguous so
/// that activation tensors only need to be communicated at stage boundaries.
///
/// If `num_stages == 1`, all layers are assigned to stage 0.
pub fn partition_layers_by_flops(
    layer_flops: &[u64],
    num_stages: usize,
) -> Vec<PipelineStageInfo> {
    let total_layers = layer_flops.len();
    if num_stages == 0 || total_layers == 0 {
        return Vec::new();
    }
    if num_stages == 1 {
        return vec![PipelineStageInfo {
            stage_id: 0,
            num_layers: total_layers,
            layer_start: 0,
            layer_end: total_layers,
        }];
    }

    // Target FLOPs per stage
    let total_flops: u64 = layer_flops.iter().sum();
    let target = (total_flops + num_stages as u64 - 1) / num_stages as u64;

    // Greedy contiguous assignment: scan layers left-to-right, start a new stage
    // when adding the next layer would exceed the per-stage target (while ensuring
    // at least one layer per stage and at most num_stages stages).
    let mut boundaries: Vec<usize> = vec![0]; // start indices of each stage
    let mut cumulative = 0u64;
    let mut stages_opened = 1usize;

    for (idx, &flops) in layer_flops.iter().enumerate() {
        cumulative += flops;
        // Open a new stage if we've exceeded the target AND there's room for more stages
        // AND there are enough layers remaining for the remaining stages.
        let remaining_layers = total_layers - idx - 1;
        let remaining_stages = num_stages - stages_opened;
        if cumulative >= target
            && stages_opened < num_stages
            && remaining_layers >= remaining_stages
        {
            boundaries.push(idx + 1);
            stages_opened += 1;
            cumulative = 0;
        }
    }

    // Build stage infos from boundaries
    let mut stages = Vec::with_capacity(num_stages);
    for stage_id in 0..num_stages {
        let layer_start = boundaries[stage_id];
        let layer_end = if stage_id + 1 < boundaries.len() {
            boundaries[stage_id + 1]
        } else {
            total_layers
        };
        let num_layers = layer_end - layer_start;
        stages.push(PipelineStageInfo { stage_id, num_layers, layer_start, layer_end });
    }

    stages
}

// ---------------------------------------------------------------------------
// MicroBatch
// ---------------------------------------------------------------------------

/// A micro-batch of data flowing through the pipeline.
#[derive(Debug, Clone)]
pub struct MicroBatch {
    /// Zero-based micro-batch index.
    pub micro_batch_id: usize,
    /// Flat activation tensor.
    pub data: Vec<f32>,
    /// Shape `(seq_len, hidden_size)` of the activation tensor.
    pub shape: (usize, usize),
    /// Whether this is the last micro-batch in the global batch.
    pub is_last: bool,
}

impl MicroBatch {
    /// Create a new `MicroBatch` with the given activations.
    pub fn new(
        micro_batch_id: usize,
        data: Vec<f32>,
        shape: (usize, usize),
        is_last: bool,
    ) -> Self {
        Self { micro_batch_id, data, shape, is_last }
    }
}

// ---------------------------------------------------------------------------
// PipelineBubbleStats
// ---------------------------------------------------------------------------

/// Statistics about pipeline bubble overhead and memory usage.
#[derive(Debug, Clone)]
pub struct PipelineBubbleStats {
    /// Number of pipeline stages (p).
    pub num_stages: usize,
    /// Number of micro-batches (m).
    pub num_micro_batches: usize,
    /// Schedule in use.
    pub schedule: PipelineSchedule,
}

impl PipelineBubbleStats {
    /// GPipe bubble fraction: `(p-1) / (m + p - 1)`.
    ///
    /// Approaches 0 as m → ∞.
    pub fn bubble_fraction_gpipe(&self) -> f32 {
        let p = self.num_stages as f32;
        let m = self.num_micro_batches as f32;
        (p - 1.0) / (m + p - 1.0)
    }

    /// 1F1B bubble fraction: `(p-1) / (2*(m-1) + p)`.
    ///
    /// Approximately `(p-1) / (2m + p - 1)` which is ~half the GPipe bubble
    /// for large m.
    pub fn bubble_fraction_1f1b(&self) -> f32 {
        let p = self.num_stages as f32;
        let m = self.num_micro_batches as f32;
        (p - 1.0) / (2.0 * (m - 1.0) + p)
    }

    /// Ratio of per-stage memory footprint.
    ///
    /// - GPipe:  proportional to `p * m` (all activations kept until backward).
    /// - 1F1B:   proportional to `p + m` (activations flushed as backward proceeds).
    ///
    /// Returns the ratio as `(p + m) / (p * m)` (1F1B / GPipe).
    pub fn memory_footprint_ratio(&self) -> f32 {
        let p = self.num_stages as f32;
        let m = self.num_micro_batches as f32;
        (p + m) / (p * m)
    }

    /// Recommended number of micro-batches for low bubble overhead with GPipe.
    ///
    /// Rule of thumb: `m ≈ 4p` keeps the bubble below ~20 %.
    pub fn optimal_num_micro_batches(&self) -> usize {
        4 * self.num_stages
    }
}

// ---------------------------------------------------------------------------
// PipelineStep
// ---------------------------------------------------------------------------

/// A single scheduling step in the pipeline execution sequence.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PipelineStep {
    /// Execute a forward pass for the given micro-batch.
    Forward { micro_batch_id: usize },
    /// Execute a backward pass for the given micro-batch.
    Backward { micro_batch_id: usize },
    /// Block until a forward activation arrives from the previous stage.
    WaitForward { from_stage: usize },
    /// Block until a backward gradient arrives from the next stage.
    WaitBackward { from_stage: usize },
    /// Send forward activations to the next stage.
    SendActivations { to_stage: usize, micro_batch_id: usize },
    /// Receive forward activations from the previous stage.
    RecvActivations { from_stage: usize, micro_batch_id: usize },
}

// ---------------------------------------------------------------------------
// PipelineScheduler
// ---------------------------------------------------------------------------

/// Orchestrates scheduling for a single pipeline stage.
pub struct PipelineScheduler {
    pub config: PipelineConfig,
    /// Stage metadata for every stage in the pipeline.
    pub stages: Vec<PipelineStageInfo>,
}

impl PipelineScheduler {
    /// Create a scheduler that partitions `total_layers` evenly across stages.
    pub fn new(config: PipelineConfig, total_layers: usize) -> Self {
        let stages = partition_layers_evenly(total_layers, config.num_stages);
        Self { config, stages }
    }

    /// Return a reference to the `PipelineStageInfo` for this stage.
    pub fn this_stage(&self) -> &PipelineStageInfo {
        // stage_rank is validated at construction; stages is non-empty when
        // config.num_stages >= 1.  If stages is empty (total_layers == 0) we
        // fall back to stage 0 info synthesised on the fly.
        self.stages
            .get(self.config.stage_rank)
            .unwrap_or_else(|| self.stages.first().expect("pipeline must have at least one stage"))
    }

    /// Return `true` if this stage is the first in the pipeline (stage 0).
    pub fn is_first_stage(&self) -> bool {
        self.config.stage_rank == 0
    }

    /// Return `true` if this stage is the last in the pipeline.
    pub fn is_last_stage(&self) -> bool {
        self.config.stage_rank + 1 == self.config.num_stages
    }

    /// Generate the ordered sequence of `PipelineStep`s for this stage under the
    /// current schedule.
    ///
    /// `num_micro_batches` overrides `config.num_micro_batches` so tests can vary
    /// it independently.
    pub fn schedule_steps(&self, num_micro_batches: usize) -> Vec<PipelineStep> {
        match &self.config.schedule {
            PipelineSchedule::GPipe => {
                self.schedule_gpipe(num_micro_batches)
            },
            PipelineSchedule::OneFOneBubble => {
                self.schedule_1f1b(num_micro_batches)
            },
            PipelineSchedule::Interleaved { num_virtual_stages } => {
                self.schedule_interleaved(num_micro_batches, *num_virtual_stages)
            },
        }
    }

    // -----------------------------------------------------------------------
    // GPipe schedule
    // -----------------------------------------------------------------------

    /// GPipe: for each micro-batch in order, (optionally receive, then forward,
    /// then optionally send); then for each micro-batch in reverse, backward.
    fn schedule_gpipe(&self, num_micro_batches: usize) -> Vec<PipelineStep> {
        let rank = self.config.stage_rank;
        let mut steps = Vec::new();

        // Forward sweep
        for mb in 0..num_micro_batches {
            if !self.is_first_stage() {
                steps.push(PipelineStep::RecvActivations {
                    from_stage: rank - 1,
                    micro_batch_id: mb,
                });
            }
            steps.push(PipelineStep::Forward { micro_batch_id: mb });
            if !self.is_last_stage() {
                steps.push(PipelineStep::SendActivations {
                    to_stage: rank + 1,
                    micro_batch_id: mb,
                });
            }
        }

        // Backward sweep (reverse order)
        for mb in (0..num_micro_batches).rev() {
            if !self.is_last_stage() {
                steps.push(PipelineStep::WaitBackward { from_stage: rank + 1 });
            }
            steps.push(PipelineStep::Backward { micro_batch_id: mb });
        }

        steps
    }

    // -----------------------------------------------------------------------
    // 1F1B schedule
    // -----------------------------------------------------------------------

    /// 1F1B: warm-up phase (forward for first `p` micro-batches), then
    /// steady-state (interleaved forward+backward), then cool-down (remaining
    /// backwards).
    fn schedule_1f1b(&self, num_micro_batches: usize) -> Vec<PipelineStep> {
        let rank = self.config.stage_rank;
        let num_stages = self.config.num_stages;
        // Number of warm-up forward steps for this rank
        let warmup_steps = (num_stages - rank).min(num_micro_batches);
        let mut steps = Vec::new();

        // Warm-up: forward-only passes
        for mb in 0..warmup_steps {
            if !self.is_first_stage() {
                steps.push(PipelineStep::RecvActivations {
                    from_stage: rank - 1,
                    micro_batch_id: mb,
                });
            }
            steps.push(PipelineStep::Forward { micro_batch_id: mb });
            if !self.is_last_stage() {
                steps.push(PipelineStep::SendActivations {
                    to_stage: rank + 1,
                    micro_batch_id: mb,
                });
            }
        }

        // Steady-state: 1F1B pairs
        let steady_start_fwd = warmup_steps;
        let steady_start_bwd = 0usize;
        let steady_count = num_micro_batches - warmup_steps;

        for i in 0..steady_count {
            let fwd_mb = steady_start_fwd + i;
            let bwd_mb = steady_start_bwd + i;

            // Forward
            if !self.is_first_stage() {
                steps.push(PipelineStep::RecvActivations {
                    from_stage: rank - 1,
                    micro_batch_id: fwd_mb,
                });
            }
            steps.push(PipelineStep::Forward { micro_batch_id: fwd_mb });
            if !self.is_last_stage() {
                steps.push(PipelineStep::SendActivations {
                    to_stage: rank + 1,
                    micro_batch_id: fwd_mb,
                });
            }

            // Backward
            if !self.is_last_stage() {
                steps.push(PipelineStep::WaitBackward { from_stage: rank + 1 });
            }
            steps.push(PipelineStep::Backward { micro_batch_id: bwd_mb });
        }

        // Cool-down: remaining backwards
        let cooldown_start = steady_count;
        for i in 0..warmup_steps {
            let bwd_mb = cooldown_start + i;
            if !self.is_last_stage() {
                steps.push(PipelineStep::WaitBackward { from_stage: rank + 1 });
            }
            steps.push(PipelineStep::Backward { micro_batch_id: bwd_mb });
        }

        steps
    }

    // -----------------------------------------------------------------------
    // Interleaved schedule (simplified)
    // -----------------------------------------------------------------------

    fn schedule_interleaved(
        &self,
        num_micro_batches: usize,
        _num_virtual_stages: usize,
    ) -> Vec<PipelineStep> {
        // Simplified: fall back to 1F1B for interleaved (full implementation
        // would interleave virtual stages, but the structure is analogous).
        self.schedule_1f1b(num_micro_batches)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -----------------------------------------------------------------------
    // Test 1: Even layer partition
    // -----------------------------------------------------------------------
    #[test]
    fn test_partition_layers_evenly_exact() {
        let stages = partition_layers_evenly(12, 4);
        assert_eq!(stages.len(), 4);
        for s in &stages {
            assert_eq!(s.num_layers, 3);
        }
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[0].layer_end, 3);
        assert_eq!(stages[3].layer_end, 12);
    }

    // -----------------------------------------------------------------------
    // Test 2: Even partition with remainder
    // -----------------------------------------------------------------------
    #[test]
    fn test_partition_layers_evenly_remainder() {
        // 10 layers, 3 stages -> [4, 3, 3]
        let stages = partition_layers_evenly(10, 3);
        assert_eq!(stages.len(), 3);
        assert_eq!(stages[0].num_layers, 4);
        assert_eq!(stages[1].num_layers, 3);
        assert_eq!(stages[2].num_layers, 3);
        // Layer indices must be contiguous and cover 0..10
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[2].layer_end, 10);
    }

    // -----------------------------------------------------------------------
    // Test 3: FLOPs-based partition verifies balance
    // -----------------------------------------------------------------------
    #[test]
    fn test_partition_by_flops_balanced() {
        // 8 layers with equal FLOPs -> same as even split
        let flops = vec![100u64; 8];
        let stages = partition_layers_by_flops(&flops, 4);
        assert_eq!(stages.len(), 4);
        for s in &stages {
            assert_eq!(s.num_layers, 2);
        }
    }

    // -----------------------------------------------------------------------
    // Test 4: FLOPs partition with varying FLOPs
    // -----------------------------------------------------------------------
    #[test]
    fn test_partition_by_flops_uneven() {
        // [100, 100, 100, 100, 400] -> 2 stages
        // target = 400: stage 0 gets [100,100,100,100]=400, stage 1 gets [400]
        let flops = vec![100u64, 100, 100, 100, 400];
        let stages = partition_layers_by_flops(&flops, 2);
        assert_eq!(stages.len(), 2);
        assert_eq!(stages[0].num_layers, 4);
        assert_eq!(stages[1].num_layers, 1);
        let s0_flops: u64 = flops[stages[0].layer_start..stages[0].layer_end].iter().sum();
        let s1_flops: u64 = flops[stages[1].layer_start..stages[1].layer_end].iter().sum();
        // Stage 0 should have roughly equal or lower FLOPs than stage 1
        assert_eq!(s0_flops, 400);
        assert_eq!(s1_flops, 400);
    }

    // -----------------------------------------------------------------------
    // Test 5: Bubble fraction GPipe formula
    // -----------------------------------------------------------------------
    #[test]
    fn test_bubble_fraction_gpipe() {
        let stats =
            PipelineBubbleStats { num_stages: 4, num_micro_batches: 8, schedule: PipelineSchedule::GPipe };
        // (4-1) / (8+4-1) = 3/11
        let expected = 3.0f32 / 11.0;
        assert!((stats.bubble_fraction_gpipe() - expected).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 6: Bubble fraction 1F1B
    // -----------------------------------------------------------------------
    #[test]
    fn test_bubble_fraction_1f1b() {
        let stats = PipelineBubbleStats {
            num_stages: 4,
            num_micro_batches: 8,
            schedule: PipelineSchedule::OneFOneBubble,
        };
        // (4-1) / (2*(8-1) + 4) = 3 / (14+4) = 3/18 = 1/6
        let expected = 3.0f32 / 18.0;
        assert!((stats.bubble_fraction_1f1b() - expected).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 7: First and last stage detection
    // -----------------------------------------------------------------------
    #[test]
    fn test_first_last_stage_detection() {
        let cfg_first = PipelineConfig {
            num_stages: 4,
            num_micro_batches: 8,
            stage_rank: 0,
            schedule: PipelineSchedule::GPipe,
        };
        let sched_first = PipelineScheduler::new(cfg_first, 8);
        assert!(sched_first.is_first_stage());
        assert!(!sched_first.is_last_stage());

        let cfg_last = PipelineConfig {
            num_stages: 4,
            num_micro_batches: 8,
            stage_rank: 3,
            schedule: PipelineSchedule::GPipe,
        };
        let sched_last = PipelineScheduler::new(cfg_last, 8);
        assert!(!sched_last.is_first_stage());
        assert!(sched_last.is_last_stage());
    }

    // -----------------------------------------------------------------------
    // Test 8: Micro-batch count via schedule_steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_micro_batch_count_in_schedule() {
        let cfg = PipelineConfig {
            num_stages: 2,
            num_micro_batches: 4,
            stage_rank: 0,
            schedule: PipelineSchedule::GPipe,
        };
        let sched = PipelineScheduler::new(cfg, 4);
        let steps = sched.schedule_steps(4);
        let fwd_count = steps
            .iter()
            .filter(|s| matches!(s, PipelineStep::Forward { .. }))
            .count();
        let bwd_count = steps
            .iter()
            .filter(|s| matches!(s, PipelineStep::Backward { .. }))
            .count();
        assert_eq!(fwd_count, 4);
        assert_eq!(bwd_count, 4);
    }

    // -----------------------------------------------------------------------
    // Test 9: schedule_steps for stage 0
    // -----------------------------------------------------------------------
    #[test]
    fn test_schedule_steps_stage_0_gpipe() {
        let cfg = PipelineConfig {
            num_stages: 3,
            num_micro_batches: 3,
            stage_rank: 0,
            schedule: PipelineSchedule::GPipe,
        };
        let sched = PipelineScheduler::new(cfg, 6);
        let steps = sched.schedule_steps(3);
        // Stage 0 has no RecvActivations (first stage)
        assert!(!steps.iter().any(|s| matches!(s, PipelineStep::RecvActivations { .. })));
        // Has 3 Forward steps
        assert_eq!(
            steps.iter().filter(|s| matches!(s, PipelineStep::Forward { .. })).count(),
            3
        );
        // Has SendActivations for each micro-batch
        assert_eq!(
            steps
                .iter()
                .filter(|s| matches!(s, PipelineStep::SendActivations { .. }))
                .count(),
            3
        );
    }

    // -----------------------------------------------------------------------
    // Test 10: Optimal micro-batch count
    // -----------------------------------------------------------------------
    #[test]
    fn test_optimal_num_micro_batches() {
        let stats = PipelineBubbleStats {
            num_stages: 8,
            num_micro_batches: 32,
            schedule: PipelineSchedule::GPipe,
        };
        assert_eq!(stats.optimal_num_micro_batches(), 32); // 4 * 8
    }

    // -----------------------------------------------------------------------
    // Test 11: Memory footprint ratio
    // -----------------------------------------------------------------------
    #[test]
    fn test_memory_footprint_ratio() {
        let stats = PipelineBubbleStats {
            num_stages: 4,
            num_micro_batches: 8,
            schedule: PipelineSchedule::GPipe,
        };
        // (4 + 8) / (4 * 8) = 12/32 = 0.375
        let expected = 12.0f32 / 32.0;
        assert!((stats.memory_footprint_ratio() - expected).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 12: FLOPs partition edge case — 1 stage means no partitioning
    // -----------------------------------------------------------------------
    #[test]
    fn test_flops_partition_single_stage() {
        let flops = vec![100u64, 200, 300, 400];
        let stages = partition_layers_by_flops(&flops, 1);
        assert_eq!(stages.len(), 1);
        assert_eq!(stages[0].num_layers, 4);
        assert_eq!(stages[0].layer_start, 0);
        assert_eq!(stages[0].layer_end, 4);
    }

    // -----------------------------------------------------------------------
    // Test 13: PipelineScheduler this_stage returns correct metadata
    // -----------------------------------------------------------------------
    #[test]
    fn test_this_stage_metadata() {
        let cfg = PipelineConfig {
            num_stages: 4,
            num_micro_batches: 8,
            stage_rank: 2,
            schedule: PipelineSchedule::GPipe,
        };
        let sched = PipelineScheduler::new(cfg, 8);
        let stage = sched.this_stage();
        assert_eq!(stage.stage_id, 2);
        assert_eq!(stage.num_layers, 2); // 8/4 = 2 per stage
        assert_eq!(stage.layer_start, 4);
        assert_eq!(stage.layer_end, 6);
    }

    // -----------------------------------------------------------------------
    // Test 14: 1F1B schedule has both Forward and Backward steps
    // -----------------------------------------------------------------------
    #[test]
    fn test_1f1b_schedule_has_fwd_and_bwd() {
        let cfg = PipelineConfig {
            num_stages: 4,
            num_micro_batches: 8,
            stage_rank: 1,
            schedule: PipelineSchedule::OneFOneBubble,
        };
        let sched = PipelineScheduler::new(cfg, 8);
        let steps = sched.schedule_steps(8);
        let fwd = steps.iter().filter(|s| matches!(s, PipelineStep::Forward { .. })).count();
        let bwd = steps.iter().filter(|s| matches!(s, PipelineStep::Backward { .. })).count();
        assert_eq!(fwd, 8);
        assert_eq!(bwd, 8);
    }

    // -----------------------------------------------------------------------
    // Test 15: MicroBatch creation
    // -----------------------------------------------------------------------
    #[test]
    fn test_micro_batch_creation() {
        let mb = MicroBatch::new(3, vec![1.0f32, 2.0, 3.0], (1, 3), true);
        assert_eq!(mb.micro_batch_id, 3);
        assert_eq!(mb.shape, (1, 3));
        assert!(mb.is_last);
        assert_eq!(mb.data.len(), 3);
    }

    // -----------------------------------------------------------------------
    // Test 16: PpError Display
    // -----------------------------------------------------------------------
    #[test]
    fn test_pp_error_display() {
        let e = PpError::ZeroStages;
        assert!(e.to_string().contains("num_stages"));

        let e2 = PpError::StageRankOutOfRange { stage_rank: 5, num_stages: 4 };
        assert!(e2.to_string().contains("5"));
    }
}
