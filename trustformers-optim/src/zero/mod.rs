//! ZeRO (Zero Redundancy Optimizer) Implementation for TrustformeRS
//!
//! ZeRO is a memory-efficient training technique that partitions optimizer states,
//! gradients, and parameters across devices to reduce memory usage while maintaining
//! training efficiency.
//!
//! Implements three stages:
//! - Stage 1: Partition optimizer states
//! - Stage 2: Partition optimizer states + gradients
//! - Stage 3: Partition optimizer states + gradients + parameters

pub mod zero_optimizer;
pub mod zero_stage1;
pub mod zero_stage2;
pub mod zero_stage3;
pub mod zero_stage3_overlap;
pub mod zero_utils;

pub use zero_optimizer::{ZeROConfig, ZeROOptimizer, ZeROStage};
pub use zero_stage1::ZeROStage1;
pub use zero_stage2::ZeROStage2;
pub use zero_stage3::ZeROStage3;
pub use zero_utils::{
    all_gather_gradients, gather_parameters, partition_gradients, partition_parameters,
    reduce_scatter_gradients, GradientBuffer, ParameterGroup, ParameterPartition, ZeROState,
};

/// ZeRO optimization stages
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZeROImplementationStage {
    /// Stage 1: Partition optimizer states only
    Stage1,
    /// Stage 2: Partition optimizer states + gradients
    Stage2,
    /// Stage 3: Partition optimizer states + gradients + parameters
    Stage3,
}

/// Memory statistics for ZeRO optimization
#[derive(Debug, Clone)]
pub struct ZeROMemoryStats {
    /// Memory saved by partitioning optimizer states
    pub optimizer_memory_saved: usize,
    /// Memory saved by partitioning gradients
    pub gradient_memory_saved: usize,
    /// Memory saved by partitioning parameters
    pub parameter_memory_saved: usize,
    /// Total memory saved
    pub total_memory_saved: usize,
    /// Memory overhead from communication buffers
    pub communication_overhead: usize,
}

impl Default for ZeROMemoryStats {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeROMemoryStats {
    pub fn new() -> Self {
        Self {
            optimizer_memory_saved: 0,
            gradient_memory_saved: 0,
            parameter_memory_saved: 0,
            total_memory_saved: 0,
            communication_overhead: 0,
        }
    }

    pub fn update_totals(&mut self) {
        self.total_memory_saved =
            self.optimizer_memory_saved + self.gradient_memory_saved + self.parameter_memory_saved;
    }
}

// ─── Flat partition helpers (no Tensor dependency) ───────────────────────────

/// Partition optimizer states across `world_size` ranks.
///
/// `state` is a slice of optimizer state vectors (e.g., Adam m, v per parameter).
/// Returns `Vec<Vec<Vec<f32>>>` where `result[rank][param_idx]` is that rank's slice
/// of the optimizer state for parameter `param_idx`.
pub fn partition_optimizer_state(state: &[Vec<f32>], world_size: usize) -> Vec<Vec<Vec<f32>>> {
    assert!(world_size > 0, "world_size must be > 0");
    let mut result: Vec<Vec<Vec<f32>>> = vec![Vec::new(); world_size];
    for param_state in state {
        let total = param_state.len();
        let chunk_size = total.div_ceil(world_size);
        for rank in 0..world_size {
            let start = rank * chunk_size;
            let end = (start + chunk_size).min(total);
            let shard = if start < total { param_state[start..end].to_vec() } else { Vec::new() };
            result[rank].push(shard);
        }
    }
    result
}

/// Partition gradients across `world_size` ranks.
///
/// Returns `result[rank][param_idx]` = that rank's slice of grad for param `param_idx`.
pub fn partition_gradients_flat(grads: &[Vec<f32>], world_size: usize) -> Vec<Vec<Vec<f32>>> {
    assert!(world_size > 0, "world_size must be > 0");
    let mut result: Vec<Vec<Vec<f32>>> = vec![Vec::new(); world_size];
    for grad in grads {
        let total = grad.len();
        let chunk_size = total.div_ceil(world_size);
        for rank in 0..world_size {
            let start = rank * chunk_size;
            let end = (start + chunk_size).min(total);
            let shard = if start < total { grad[start..end].to_vec() } else { Vec::new() };
            result[rank].push(shard);
        }
    }
    result
}

/// Partition parameters across `world_size` ranks.
///
/// Returns `result[rank][param_idx]` = that rank's slice of the parameter.
pub fn partition_parameters_flat(params: &[Vec<f32>], world_size: usize) -> Vec<Vec<Vec<f32>>> {
    assert!(world_size > 0, "world_size must be > 0");
    let mut result: Vec<Vec<Vec<f32>>> = vec![Vec::new(); world_size];
    for param in params {
        let total = param.len();
        let chunk_size = total.div_ceil(world_size);
        for rank in 0..world_size {
            let start = rank * chunk_size;
            let end = (start + chunk_size).min(total);
            let shard = if start < total { param[start..end].to_vec() } else { Vec::new() };
            result[rank].push(shard);
        }
    }
    result
}

/// Gather (reconstruct) parameters from their partitioned shards.
///
/// `partitioned[rank][param_idx]` = rank's shard for that param.
/// Returns `Vec<Vec<f32>>` indexed by param_idx with full concatenated values.
pub fn gather_parameters_flat(partitioned: &[Vec<Vec<f32>>]) -> Vec<Vec<f32>> {
    if partitioned.is_empty() {
        return Vec::new();
    }
    let num_params = partitioned[0].len();
    let mut result: Vec<Vec<f32>> = vec![Vec::new(); num_params];
    for rank_data in partitioned {
        for (param_idx, shard) in rank_data.iter().enumerate() {
            if param_idx < result.len() {
                result[param_idx].extend_from_slice(shard);
            }
        }
    }
    result
}

/// Calculate memory reduction ratio for a given ZeRO stage.
///
/// Returns the fraction of total baseline memory that is saved (0.0 to 1.0).
/// - Stage 1: saves optimizer_bytes * (world_size - 1) / world_size
/// - Stage 2: saves (optimizer_bytes + grad_bytes) * (world_size - 1) / world_size
/// - Stage 3: saves all bytes * (world_size - 1) / world_size
pub fn zero_stage_memory_reduction(
    stage: u8,
    world_size: usize,
    param_bytes: usize,
    grad_bytes: usize,
    opt_bytes: usize,
) -> f32 {
    if world_size <= 1 {
        return 0.0;
    }
    let total_bytes = (param_bytes + grad_bytes + opt_bytes) as f32;
    if total_bytes == 0.0 {
        return 0.0;
    }
    let ws = world_size as f32;
    let save_fraction = (ws - 1.0) / ws;
    let saved_bytes = match stage {
        1 => opt_bytes as f32 * save_fraction,
        2 => (opt_bytes + grad_bytes) as f32 * save_fraction,
        3 => (param_bytes + grad_bytes + opt_bytes) as f32 * save_fraction,
        _ => 0.0,
    };
    saved_bytes / total_bytes
}

// ─── ZeroConfig ─────────────────────────────────────────────────────────────

/// Simple configuration struct for ZeRO stage selection and validation.
#[derive(Debug, Clone)]
pub struct ZeroConfig {
    /// ZeRO stage: 1, 2, or 3
    pub stage: u8,
    /// Number of distributed ranks
    pub world_size: usize,
    /// Overlap communication with computation
    pub overlap_comm: bool,
    /// Number of gradient elements per reduce bucket
    pub reduce_bucket_size: usize,
}

impl Default for ZeroConfig {
    fn default() -> Self {
        Self {
            stage: 1,
            world_size: 1,
            overlap_comm: true,
            reduce_bucket_size: 500_000_000,
        }
    }
}

impl ZeroConfig {
    /// Validate the configuration.
    ///
    /// Returns `Err` with a descriptive message if:
    /// - `stage` is not in 1..=3
    /// - `world_size` is 0
    pub fn validate(&self) -> Result<(), String> {
        if self.stage == 0 || self.stage > 3 {
            return Err(format!("ZeRO stage must be 1, 2, or 3; got {}", self.stage));
        }
        if self.world_size == 0 {
            return Err("world_size must be >= 1".to_string());
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper ───────────────────────────────────────────────────────────────

    fn make_params(n: usize) -> Vec<f32> {
        (0..n).map(|i| i as f32).collect()
    }

    // ── partition_optimizer_state ─────────────────────────────────────────────

    #[test]
    fn test_partition_optimizer_state_basic() {
        // 1 state vec of 8 floats, world_size=4 → each rank gets 2
        let state = vec![make_params(8)];
        let partitioned = partition_optimizer_state(&state, 4);
        assert_eq!(partitioned.len(), 4);
        for rank in 0..4 {
            assert_eq!(
                partitioned[rank][0].len(),
                2,
                "rank {rank} should have 2 elements"
            );
        }
        // verify values: rank 0 = [0,1], rank 1 = [2,3], ...
        assert_eq!(partitioned[0][0], vec![0.0, 1.0]);
        assert_eq!(partitioned[1][0], vec![2.0, 3.0]);
        assert_eq!(partitioned[2][0], vec![4.0, 5.0]);
        assert_eq!(partitioned[3][0], vec![6.0, 7.0]);
    }

    #[test]
    fn test_partition_optimizer_state_uneven() {
        // 7 elements, world_size=3 → chunks of ceil(7/3)=3, ranks get [3, 3, 1]
        let state = vec![make_params(7)];
        let partitioned = partition_optimizer_state(&state, 3);
        assert_eq!(partitioned.len(), 3);
        assert_eq!(partitioned[0][0].len(), 3);
        assert_eq!(partitioned[1][0].len(), 3);
        assert_eq!(partitioned[2][0].len(), 1);
        // total elements = 7
        let total: usize = partitioned.iter().map(|r| r[0].len()).sum();
        assert_eq!(total, 7);
    }

    #[test]
    fn test_partition_optimizer_state_multiple_states() {
        // world_size=2, 3 state vecs of different lengths
        let state = vec![make_params(4), make_params(6), make_params(2)];
        let partitioned = partition_optimizer_state(&state, 2);
        assert_eq!(partitioned.len(), 2);
        for rank_data in &partitioned {
            assert_eq!(rank_data.len(), 3, "each rank should have 3 param states");
        }
    }

    #[test]
    fn test_partition_optimizer_state_rank_sizes_sum_to_original() {
        let state = vec![make_params(10), make_params(7)];
        let partitioned = partition_optimizer_state(&state, 4);
        for param_idx in 0..2 {
            let total: usize = partitioned.iter().map(|r| r[param_idx].len()).sum();
            assert_eq!(total, state[param_idx].len());
        }
    }

    // ── partition_gradients_flat ──────────────────────────────────────────────

    #[test]
    fn test_partition_gradients_basic() {
        let grads = vec![make_params(16)];
        let partitioned = partition_gradients_flat(&grads, 4);
        assert_eq!(partitioned.len(), 4);
        for rank in 0..4 {
            assert_eq!(partitioned[rank][0].len(), 4);
        }
    }

    #[test]
    fn test_partition_gradients_multi() {
        let grads = vec![make_params(8), make_params(4)];
        let partitioned = partition_gradients_flat(&grads, 2);
        // rank 0: first 4 of param0, first 2 of param1
        assert_eq!(partitioned[0][0].len(), 4);
        assert_eq!(partitioned[0][1].len(), 2);
    }

    #[test]
    fn test_partition_gradients_size_check() {
        let grads = vec![make_params(9), make_params(5)];
        let partitioned = partition_gradients_flat(&grads, 3);
        for (param_idx, original) in grads.iter().enumerate() {
            let total: usize = partitioned.iter().map(|r| r[param_idx].len()).sum();
            assert_eq!(total, original.len());
        }
    }

    // ── partition_parameters_flat ─────────────────────────────────────────────

    #[test]
    fn test_partition_parameters_basic() {
        let params = vec![make_params(12)];
        let partitioned = partition_parameters_flat(&params, 4);
        assert_eq!(partitioned.len(), 4);
        for rank in 0..4 {
            assert_eq!(partitioned[rank][0].len(), 3);
        }
    }

    #[test]
    fn test_partition_parameters_no_duplicate() {
        // Total elements across all ranks must equal original count
        let params = vec![make_params(20)];
        let partitioned = partition_parameters_flat(&params, 4);
        let total: usize = partitioned.iter().map(|r| r[0].len()).sum();
        assert_eq!(total, 20);
    }

    #[test]
    fn test_partition_parameters_world_size_1() {
        let params = vec![make_params(10)];
        let partitioned = partition_parameters_flat(&params, 1);
        assert_eq!(partitioned.len(), 1);
        assert_eq!(partitioned[0][0], make_params(10));
    }

    // ── gather_parameters_flat ────────────────────────────────────────────────

    #[test]
    fn test_gather_is_inverse_of_partition() {
        let original = vec![make_params(12), make_params(8)];
        let partitioned = partition_parameters_flat(&original, 4);
        let gathered = gather_parameters_flat(&partitioned);
        assert_eq!(gathered.len(), original.len());
        for (idx, orig) in original.iter().enumerate() {
            assert_eq!(&gathered[idx], orig, "param {idx} mismatch after gather");
        }
    }

    #[test]
    fn test_gather_inverse_uneven() {
        let original = vec![make_params(7), make_params(11)];
        let partitioned = partition_parameters_flat(&original, 3);
        let gathered = gather_parameters_flat(&partitioned);
        for (idx, orig) in original.iter().enumerate() {
            assert_eq!(&gathered[idx], orig);
        }
    }

    #[test]
    fn test_gather_empty() {
        let gathered = gather_parameters_flat(&[]);
        assert!(gathered.is_empty());
    }

    // ── zero_stage_memory_reduction ───────────────────────────────────────────

    #[test]
    fn test_stage1_memory_reduction() {
        // Stage 1 saves opt_bytes * (ws-1)/ws
        // world_size=4: saves 3/4 of opt_bytes
        let ratio = zero_stage_memory_reduction(1, 4, 1000, 1000, 1000);
        // saved = 1000 * 0.75 = 750, total = 3000, ratio = 750/3000 = 0.25
        let expected = (1000.0f32 * 0.75) / 3000.0;
        assert!(
            (ratio - expected).abs() < 1e-5,
            "got {ratio}, expected {expected}"
        );
    }

    #[test]
    fn test_stage2_memory_reduction() {
        let ratio = zero_stage_memory_reduction(2, 4, 1000, 1000, 1000);
        // saved = (opt+grad) * 0.75 = 2000 * 0.75 = 1500, total=3000, ratio=0.5
        let expected = (2000.0f32 * 0.75) / 3000.0;
        assert!(
            (ratio - expected).abs() < 1e-5,
            "got {ratio}, expected {expected}"
        );
    }

    #[test]
    fn test_stage3_memory_reduction() {
        let ratio = zero_stage_memory_reduction(3, 4, 1000, 1000, 1000);
        // saved = 3000 * 0.75 = 2250, total=3000, ratio=0.75
        let expected = 3000.0f32 * 0.75 / 3000.0;
        assert!(
            (ratio - expected).abs() < 1e-5,
            "got {ratio}, expected {expected}"
        );
    }

    #[test]
    fn test_memory_reduction_world_size_1() {
        let ratio = zero_stage_memory_reduction(3, 1, 1000, 1000, 1000);
        assert_eq!(ratio, 0.0);
    }

    #[test]
    fn test_memory_reduction_stage3_is_greater_than_stage1() {
        let r1 = zero_stage_memory_reduction(1, 4, 1000, 1000, 1000);
        let r3 = zero_stage_memory_reduction(3, 4, 1000, 1000, 1000);
        assert!(r3 > r1, "stage3 should save more than stage1");
    }

    // ── ZeroConfig validation ─────────────────────────────────────────────────

    #[test]
    fn test_zero_config_valid() {
        let cfg = ZeroConfig {
            stage: 2,
            world_size: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_zero_config_invalid_stage_zero() {
        let cfg = ZeroConfig {
            stage: 0,
            world_size: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_config_invalid_stage_four() {
        let cfg = ZeroConfig {
            stage: 4,
            world_size: 4,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_config_invalid_world_size() {
        let cfg = ZeroConfig {
            stage: 1,
            world_size: 0,
            ..Default::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_config_all_stages_valid() {
        for stage in 1u8..=3 {
            let cfg = ZeroConfig {
                stage,
                world_size: 8,
                ..Default::default()
            };
            assert!(cfg.validate().is_ok(), "stage {stage} should be valid");
        }
    }

    // ── ZeROMemoryStats ───────────────────────────────────────────────────────

    #[test]
    fn test_zero_memory_stats_new() {
        let stats = ZeROMemoryStats::new();
        assert_eq!(stats.optimizer_memory_saved, 0);
        assert_eq!(stats.gradient_memory_saved, 0);
        assert_eq!(stats.parameter_memory_saved, 0);
        assert_eq!(stats.total_memory_saved, 0);
        assert_eq!(stats.communication_overhead, 0);
    }

    #[test]
    fn test_zero_memory_stats_update_totals() {
        let mut stats = ZeROMemoryStats::new();
        stats.optimizer_memory_saved = 100;
        stats.gradient_memory_saved = 200;
        stats.parameter_memory_saved = 300;
        stats.update_totals();
        assert_eq!(stats.total_memory_saved, 600);
    }

    #[test]
    fn test_partition_large_vectors() {
        let params: Vec<Vec<f32>> =
            (0..5).map(|p| (0..1000).map(|i| (p * 1000 + i) as f32).collect()).collect();
        let partitioned = partition_parameters_flat(&params, 8);
        assert_eq!(partitioned.len(), 8);
        // Each rank should hold 1000/8 = ceil(1000/8) = 125 elements per param
        assert_eq!(partitioned[0][0].len(), 125);
        // Gather should recover original
        let gathered = gather_parameters_flat(&partitioned);
        for (idx, orig) in params.iter().enumerate() {
            assert_eq!(&gathered[idx], orig, "param {idx} mismatch");
        }
    }
}
