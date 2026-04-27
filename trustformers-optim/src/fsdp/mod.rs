//! Fully Sharded Data Parallel (FSDP) Simulation
//!
//! FSDP shards parameters, gradients, and optimizer state across all processes,
//! providing maximum memory efficiency for large model training.
//!
//! This module simulates FSDP behavior for testing and algorithm development
//! without requiring actual distributed infrastructure.

use std::collections::HashMap;

/// FSDP wrapping strategy
#[derive(Debug, Clone, PartialEq)]
pub enum WrappingPolicy {
    /// Wrap entire model as one FSDP unit
    WholeProgramWrap,
    /// Wrap each transformer layer separately (recommended)
    TransformerLayerWrap {
        /// Minimum number of parameters to wrap a layer
        min_params: usize,
    },
    /// Wrap by module size threshold
    SizeBasedWrap {
        /// Minimum number of parameters to trigger wrapping
        min_num_params: usize,
    },
}

/// Configuration for FSDP training
#[derive(Debug, Clone)]
pub struct FsdpConfig {
    /// Total number of processes
    pub world_size: usize,
    /// This process's rank
    pub local_rank: usize,
    /// Strategy for deciding which modules to wrap
    pub wrapping_policy: WrappingPolicy,
    /// Offload parameter shards to CPU between forward and backward passes
    pub cpu_offload: bool,
    /// Keep full-precision master weights, use fp16 for forward pass
    pub mixed_precision: bool,
    /// Prefetch the next layer's params during the backward pass
    pub backward_prefetch: bool,
    /// Prefetch params ahead of forward pass
    pub forward_prefetch: bool,
}

impl Default for FsdpConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            local_rank: 0,
            wrapping_policy: WrappingPolicy::WholeProgramWrap,
            cpu_offload: false,
            mixed_precision: false,
            backward_prefetch: true,
            forward_prefetch: false,
        }
    }
}

/// Sharding strategy for FSDP — analogous to ZeRO stages.
#[derive(Debug, Clone, PartialEq)]
pub enum ShardingStrategy {
    /// ZeRO-3: shard params + grads + opt state (maximum memory savings).
    FullShard,
    /// ZeRO-2: shard grads + opt state only; parameters are replicated.
    ShardGradOp,
    /// DDP: replicate everything — no sharding.
    NoShard,
    /// ZeRO-3 within a node, replicate model across nodes.
    HybridShard { num_model_replicas: usize },
}

/// Memory analyzer for FSDP configurations.
///
/// Provides estimates of peak per-rank memory and ratio vs DDP baseline.
pub struct FsdpMemoryAnalyzer;

impl FsdpMemoryAnalyzer {
    /// Estimate peak memory per rank in bytes.
    ///
    /// Assumes f32 (4 bytes/param) and Adam optimizer (2× param bytes for m and v).
    /// Uses FullShard accounting by default:
    ///   - param_bytes / world_size + grad_bytes / world_size + opt_bytes / world_size
    pub fn peak_memory(config: &FsdpConfig, total_params: usize) -> usize {
        let param_bytes = total_params * 4; // f32
        let grad_bytes = total_params * 4;
        let opt_bytes = total_params * 8; // Adam: m + v = 2× params
        let ws = config.world_size.max(1);
        if ws == 1 {
            param_bytes + grad_bytes + opt_bytes
        } else {
            // FullShard: all three components are divided by world_size
            param_bytes / ws + grad_bytes / ws + opt_bytes / ws
        }
    }

    /// Ratio of per-rank memory vs DDP (no sharding) baseline.
    ///
    /// Returns a value in `(0.0, 1.0]` where `1.0` means same as DDP and
    /// `1/world_size` means maximum reduction from FullShard.
    pub fn memory_vs_ddp_ratio(world_size: usize) -> f32 {
        if world_size <= 1 {
            1.0
        } else {
            1.0 / world_size as f32
        }
    }
}

/// One FSDP unit — wraps a group of parameters that are collectively sharded.
#[derive(Debug, Clone)]
pub struct FsdpUnit {
    /// Unique identifier for this unit
    pub unit_id: usize,
    /// Names of parameters in this unit
    pub param_names: Vec<String>,
    /// Total number of parameters across all ranks
    pub total_params: usize,
    /// Number of parameters this rank holds (shard_size = ceil(total/world_size))
    pub shard_size: usize,
    /// Whether the shard is currently offloaded to CPU
    pub is_offloaded: bool,
}

impl FsdpUnit {
    /// Create a new FSDP unit.
    pub fn new(
        unit_id: usize,
        param_names: Vec<String>,
        total_params: usize,
        world_size: usize,
    ) -> Self {
        let shard_size =
            if world_size == 0 { total_params } else { total_params.div_ceil(world_size) };

        Self {
            unit_id,
            param_names,
            total_params,
            shard_size,
            is_offloaded: false,
        }
    }

    /// Memory in bytes used by this rank's shard.
    ///
    /// Uses 8 bytes per parameter (f64).
    pub fn memory_bytes_per_rank(&self) -> usize {
        self.shard_size * std::mem::size_of::<f64>()
    }

    /// Number of parameters held locally by this rank (= `shard_size`).
    pub fn local_params(&self) -> usize {
        self.shard_size
    }

    /// Size of the AllGather output buffer (= `total_params` — full reconstruction needed for forward pass).
    pub fn all_gather_buffer_size(&self) -> usize {
        self.total_params
    }

    /// Size of the ReduceScatter input/output buffer (= `total_params`).
    pub fn reduce_scatter_buffer_size(&self) -> usize {
        self.total_params
    }
}

/// FSDP state tracking for one training step.
pub struct FsdpState {
    config: FsdpConfig,
    units: Vec<FsdpUnit>,
    /// param_name -> unit_id
    param_registry: HashMap<String, usize>,
    /// unit_id -> gathered params (during forward pass)
    gather_cache: HashMap<usize, Vec<f64>>,
    /// unit_id -> shard values
    shard_storage: HashMap<usize, Vec<f64>>,
}

impl FsdpState {
    /// Create a new FSDP state with the given configuration.
    pub fn new(config: FsdpConfig) -> Self {
        Self {
            config,
            units: Vec::new(),
            param_registry: HashMap::new(),
            gather_cache: HashMap::new(),
            shard_storage: HashMap::new(),
        }
    }

    /// Register a group of parameters as one FSDP unit.
    ///
    /// Returns the `unit_id` of the newly created unit.
    pub fn wrap_unit(
        &mut self,
        param_names: Vec<String>,
        param_values: HashMap<String, Vec<f64>>,
    ) -> Result<usize, FsdpError> {
        if param_names.is_empty() {
            return Err(FsdpError::EmptyUnit);
        }

        // Check for duplicate registrations
        for name in &param_names {
            if self.param_registry.contains_key(name) {
                return Err(FsdpError::AlreadyRegistered(name.clone()));
            }
        }

        let unit_id = self.units.len();
        let total_params: usize =
            param_names.iter().filter_map(|n| param_values.get(n)).map(|v| v.len()).sum();

        let unit = FsdpUnit::new(
            unit_id,
            param_names.clone(),
            total_params,
            self.config.world_size,
        );

        // Build the flat parameter vector (ordered by param_names)
        let mut flat_params: Vec<f64> = Vec::with_capacity(total_params);
        for name in &param_names {
            if let Some(vals) = param_values.get(name) {
                flat_params.extend_from_slice(vals);
            }
        }

        // Store local shard
        let shard_start = self.config.local_rank * unit.shard_size;
        let shard_end = (shard_start + unit.shard_size).min(flat_params.len());
        let shard: Vec<f64> = if shard_start < flat_params.len() {
            flat_params[shard_start..shard_end].to_vec()
        } else {
            Vec::new()
        };

        self.shard_storage.insert(unit_id, shard);

        // Register param names
        for name in &param_names {
            self.param_registry.insert(name.clone(), unit_id);
        }

        self.units.push(unit);
        Ok(unit_id)
    }

    /// AllGather parameters for a unit (before forward pass).
    ///
    /// Simulates reconstruction by tiling the shard `world_size` times,
    /// then truncating to total_params length.
    pub fn allgather_unit(&mut self, unit_id: usize) -> Result<Vec<f64>, FsdpError> {
        let unit = self.units.get(unit_id).ok_or(FsdpError::UnitNotFound(unit_id))?;

        let total_params = unit.total_params;
        let world_size = self.config.world_size;

        let shard = self.shard_storage.get(&unit_id).cloned().unwrap_or_default();

        // Simulate AllGather: repeat shard world_size times and truncate
        let mut gathered: Vec<f64> = Vec::with_capacity(total_params);
        if world_size == 0 || shard.is_empty() {
            // fallback: empty
        } else {
            for _ in 0..world_size {
                gathered.extend_from_slice(&shard);
                if gathered.len() >= total_params {
                    break;
                }
            }
            gathered.truncate(total_params);
        }

        self.gather_cache.insert(unit_id, gathered.clone());
        Ok(gathered)
    }

    /// Discard gathered parameters for a unit (after forward pass, to save memory).
    pub fn discard_unit_params(&mut self, unit_id: usize) -> Result<(), FsdpError> {
        if unit_id >= self.units.len() {
            return Err(FsdpError::UnitNotFound(unit_id));
        }
        self.gather_cache.remove(&unit_id);
        Ok(())
    }

    /// ReduceScatter gradients for a unit (after backward pass).
    ///
    /// Simulates by averaging `grads / world_size` and returning the local shard.
    pub fn reduce_scatter_grads(
        &mut self,
        unit_id: usize,
        grads: Vec<f64>,
    ) -> Result<Vec<f64>, FsdpError> {
        let unit = self.units.get(unit_id).ok_or(FsdpError::UnitNotFound(unit_id))?;

        let shard_size = unit.shard_size;
        let world_size = self.config.world_size.max(1) as f64;

        // Average gradient over all ranks (simulate reduce)
        let averaged: Vec<f64> = grads.iter().map(|&g| g / world_size).collect();

        // Return the local shard portion
        let rank = self.config.local_rank;
        let start = rank * shard_size;
        let end = (start + shard_size).min(averaged.len());

        let local_grads: Vec<f64> =
            if start < averaged.len() { averaged[start..end].to_vec() } else { Vec::new() };

        Ok(local_grads)
    }

    /// Total memory saved vs DDP (replicated params) as a ratio.
    ///
    /// Returns `world_size` since each rank holds `1/world_size` of total params.
    pub fn memory_saving_ratio(&self) -> f64 {
        self.config.world_size as f64
    }

    /// Per-rank memory footprint in bytes across all units.
    pub fn per_rank_memory_bytes(&self) -> usize {
        self.units.iter().map(|u| u.memory_bytes_per_rank()).sum()
    }

    /// Number of FSDP units registered.
    pub fn unit_count(&self) -> usize {
        self.units.len()
    }

    /// Total parameter count across all units.
    pub fn total_params(&self) -> usize {
        self.units.iter().map(|u| u.total_params).sum()
    }
}

/// Errors that can occur in FSDP operations.
#[derive(Debug, thiserror::Error)]
pub enum FsdpError {
    #[error("Unit not found: {0}")]
    UnitNotFound(usize),
    #[error("Param already registered: {0}")]
    AlreadyRegistered(String),
    #[error("Empty unit")]
    EmptyUnit,
    #[error("Not gathered: unit {0}")]
    NotGathered(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_param_values(names: &[&str], size: usize) -> HashMap<String, Vec<f64>> {
        names
            .iter()
            .enumerate()
            .map(|(i, &name)| {
                let vals: Vec<f64> = (0..size).map(|j| (i * size + j) as f64).collect();
                (name.to_string(), vals)
            })
            .collect()
    }

    // Test 1: wrap_unit basic functionality
    #[test]
    fn test_wrap_unit_basic() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["w1".to_string(), "b1".to_string()];
        let values = make_param_values(&["w1", "b1"], 16);
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        assert_eq!(unit_id, 0);
        assert_eq!(state.unit_count(), 1);
    }

    // Test 2: unit shard_size = ceil(total/world_size)
    #[test]
    fn test_unit_shard_size() {
        let world_size = 4;
        let config = FsdpConfig {
            world_size,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["weight".to_string()];
        let total = 100usize;
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("weight".to_string(), vec![1.0f64; total]);
            m
        };
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        let unit = &state.units[unit_id];
        let expected_shard = total.div_ceil(world_size); // ceil(100/4) = 25
        assert_eq!(unit.shard_size, expected_shard);
        assert_eq!(unit.total_params, total);
    }

    // Test 3: allgather_unit reconstruction (returns non-empty vec)
    #[test]
    fn test_allgather_unit_reconstruction() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["p".to_string()];
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("p".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
            m
        };
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        let gathered = state.allgather_unit(unit_id).expect("gather failed");
        // Should have total_params = 4 elements
        assert_eq!(gathered.len(), 4);
    }

    // Test 4: discard clears gather_cache
    #[test]
    fn test_discard_clears_cache() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["q".to_string()];
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("q".to_string(), vec![1.0, 2.0, 3.0, 4.0]);
            m
        };
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        state.allgather_unit(unit_id).expect("gather failed");
        assert!(state.gather_cache.contains_key(&unit_id));
        state.discard_unit_params(unit_id).expect("discard failed");
        assert!(!state.gather_cache.contains_key(&unit_id));
    }

    // Test 5: reduce_scatter averages grads / world_size
    #[test]
    fn test_reduce_scatter_averages() {
        let world_size = 4;
        let config = FsdpConfig {
            world_size,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["r".to_string()];
        let total = 8usize;
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("r".to_string(), vec![1.0f64; total]);
            m
        };
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        // All grads = 4.0 → after avg = 1.0
        let grads = vec![4.0f64; total];
        let local_grads = state.reduce_scatter_grads(unit_id, grads).expect("scatter failed");
        // Each element should be 4.0 / 4.0 = 1.0
        for &g in &local_grads {
            assert!((g - 1.0).abs() < 1e-9, "Expected 1.0, got {}", g);
        }
    }

    // Test 6: memory_saving_ratio ≈ world_size
    #[test]
    fn test_memory_saving_ratio() {
        let world_size = 8;
        let config = FsdpConfig {
            world_size,
            local_rank: 2,
            ..Default::default()
        };
        let state = FsdpState::new(config);
        let ratio = state.memory_saving_ratio();
        assert!((ratio - world_size as f64).abs() < 1e-9);
    }

    // Test 7: per_rank_memory_bytes calculation
    #[test]
    fn test_per_rank_memory_bytes() {
        let world_size = 4;
        let config = FsdpConfig {
            world_size,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["w".to_string()];
        let total = 100usize;
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("w".to_string(), vec![1.0f64; total]);
            m
        };
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");
        let unit = &state.units[unit_id];
        let expected_bytes = unit.shard_size * 8; // f64 = 8 bytes
        assert_eq!(state.per_rank_memory_bytes(), expected_bytes);
    }

    // Test 8: unit_count and total_params accessors
    #[test]
    fn test_unit_count_and_total_params() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        assert_eq!(state.unit_count(), 0);
        assert_eq!(state.total_params(), 0);

        let names = vec!["a".to_string()];
        let values: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("a".to_string(), vec![0.0f64; 50]);
            m
        };
        state.wrap_unit(names, values).expect("wrap failed");
        assert_eq!(state.unit_count(), 1);
        assert_eq!(state.total_params(), 50);
    }

    // Test 9: TransformerLayerWrap policy
    #[test]
    fn test_transformer_layer_wrap_policy() {
        let policy = WrappingPolicy::TransformerLayerWrap { min_params: 1024 };
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 1,
            wrapping_policy: policy.clone(),
            ..Default::default()
        };
        let state = FsdpState::new(config);
        assert_eq!(
            state.config.wrapping_policy,
            WrappingPolicy::TransformerLayerWrap { min_params: 1024 }
        );
    }

    // Test 10: FsdpUnit memory_bytes_per_rank
    #[test]
    fn test_fsdp_unit_memory_bytes_per_rank() {
        let unit = FsdpUnit::new(0, vec!["w".to_string()], 1000, 4);
        // shard_size = ceil(1000/4) = 250
        assert_eq!(unit.shard_size, 250);
        // memory = 250 * 8 = 2000 bytes
        assert_eq!(unit.memory_bytes_per_rank(), 250 * 8);
    }

    // Test 11: double-register error (AlreadyRegistered)
    #[test]
    fn test_double_register_error() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);

        let names1 = vec!["shared_param".to_string()];
        let values1: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("shared_param".to_string(), vec![1.0f64; 10]);
            m
        };
        state.wrap_unit(names1, values1).expect("first wrap should succeed");

        let names2 = vec!["shared_param".to_string()];
        let values2: HashMap<String, Vec<f64>> = {
            let mut m = HashMap::new();
            m.insert("shared_param".to_string(), vec![2.0f64; 10]);
            m
        };
        let result = state.wrap_unit(names2, values2);
        assert!(matches!(result, Err(FsdpError::AlreadyRegistered(_))));
    }

    // Test 12: unit_not_found error
    #[test]
    fn test_unit_not_found_error() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let result = state.allgather_unit(99);
        assert!(matches!(result, Err(FsdpError::UnitNotFound(99))));
    }

    // Test 13: multi-unit model (multiple wrap_unit calls)
    #[test]
    fn test_multi_unit_model() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);

        let layers = [
            (vec!["l0.w".to_string(), "l0.b".to_string()], 128, 4),
            (vec!["l1.w".to_string(), "l1.b".to_string()], 256, 8),
            (vec!["l2.w".to_string(), "l2.b".to_string()], 512, 16),
        ];

        let mut total_expected = 0usize;
        for (names, w_size, b_size) in &layers {
            let mut values: HashMap<String, Vec<f64>> = HashMap::new();
            values.insert(names[0].clone(), vec![1.0f64; *w_size]);
            values.insert(names[1].clone(), vec![0.0f64; *b_size]);
            total_expected += w_size + b_size;
            state.wrap_unit(names.clone(), values).expect("wrap failed");
        }

        assert_eq!(state.unit_count(), 3);
        assert_eq!(state.total_params(), total_expected);
    }

    // Test 14: cpu_offload flag in config
    #[test]
    fn test_cpu_offload_flag() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 1,
            cpu_offload: true,
            mixed_precision: true,
            ..Default::default()
        };
        assert!(config.cpu_offload);
        assert!(config.mixed_precision);

        let state = FsdpState::new(config);
        assert!(state.config.cpu_offload);
        assert!(state.config.mixed_precision);

        // Check default has cpu_offload false
        let default_config = FsdpConfig::default();
        assert!(!default_config.cpu_offload);
    }

    // ── ShardingStrategy tests ────────────────────────────────────────────────

    // Test 15: FullShard variant equality
    #[test]
    fn test_sharding_strategy_full_shard() {
        let s = ShardingStrategy::FullShard;
        assert_eq!(s, ShardingStrategy::FullShard);
    }

    // Test 16: NoShard != FullShard
    #[test]
    fn test_sharding_strategy_no_shard() {
        assert_ne!(ShardingStrategy::NoShard, ShardingStrategy::FullShard);
    }

    // Test 17: ShardGradOp variant
    #[test]
    fn test_sharding_strategy_shard_grad_op() {
        let s = ShardingStrategy::ShardGradOp;
        assert_eq!(s, ShardingStrategy::ShardGradOp);
        assert_ne!(s, ShardingStrategy::FullShard);
    }

    // Test 18: HybridShard with num_model_replicas
    #[test]
    fn test_hybrid_shard_replicas() {
        let s = ShardingStrategy::HybridShard {
            num_model_replicas: 4,
        };
        assert_eq!(
            s,
            ShardingStrategy::HybridShard {
                num_model_replicas: 4
            }
        );
        assert_ne!(
            s,
            ShardingStrategy::HybridShard {
                num_model_replicas: 2
            }
        );
    }

    // ── FsdpUnit extra methods ────────────────────────────────────────────────

    // Test 19: local_params == shard_size
    #[test]
    fn test_fsdp_unit_local_params() {
        let unit = FsdpUnit::new(0, vec!["w".to_string()], 100, 4);
        assert_eq!(unit.local_params(), unit.shard_size);
    }

    // Test 20: all_gather_buffer_size == total_params
    #[test]
    fn test_fsdp_unit_all_gather_buffer() {
        let unit = FsdpUnit::new(0, vec!["w".to_string()], 100, 4);
        assert_eq!(unit.all_gather_buffer_size(), 100);
    }

    // Test 21: reduce_scatter_buffer_size == total_params
    #[test]
    fn test_fsdp_unit_reduce_scatter_buffer() {
        let unit = FsdpUnit::new(0, vec!["w".to_string()], 64, 8);
        assert_eq!(unit.reduce_scatter_buffer_size(), 64);
    }

    // ── FsdpMemoryAnalyzer tests ──────────────────────────────────────────────

    // Test 22: peak_memory world_size=1 = 16 bytes/param (4+4+8)
    #[test]
    fn test_memory_analyzer_peak_memory_world_size_1() {
        let config = FsdpConfig {
            world_size: 1,
            local_rank: 0,
            ..Default::default()
        };
        let peak = FsdpMemoryAnalyzer::peak_memory(&config, 1000);
        // 4+4+8 = 16 bytes per param * 1000 = 16000
        assert_eq!(peak, 16_000);
    }

    // Test 23: peak_memory world_size=4 = 16/4 = 4 bytes/param (integer division)
    #[test]
    fn test_memory_analyzer_peak_memory_world_size_4() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 0,
            ..Default::default()
        };
        let peak = FsdpMemoryAnalyzer::peak_memory(&config, 1000);
        // 1000*4/4 + 1000*4/4 + 1000*8/4 = 250 + 250 + 500 = 1000 ... wait:
        // (4+4+8)*1000 / 4 = 16000/4 = 4000
        // but integer: 4000/4 + 4000/4 + 8000/4 = 1000+1000+2000 = 4000
        assert_eq!(peak, 4_000);
    }

    // Test 24: memory_vs_ddp_ratio world_size=1 => 1.0
    #[test]
    fn test_memory_vs_ddp_ratio_world_size_1() {
        let ratio = FsdpMemoryAnalyzer::memory_vs_ddp_ratio(1);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    // Test 25: memory_vs_ddp_ratio world_size=8 => 0.125
    #[test]
    fn test_memory_vs_ddp_ratio_world_size_8() {
        let ratio = FsdpMemoryAnalyzer::memory_vs_ddp_ratio(8);
        assert!((ratio - 0.125).abs() < 1e-6);
    }

    // Test 26: memory_vs_ddp_ratio world_size=0 treated as 1 => 1.0
    #[test]
    fn test_memory_vs_ddp_ratio_world_size_0() {
        let ratio = FsdpMemoryAnalyzer::memory_vs_ddp_ratio(0);
        assert!((ratio - 1.0).abs() < 1e-6);
    }

    // ── Shard storage tests ───────────────────────────────────────────────────

    // Test 27: rank 2 of 4 wrapping 100 params — shard_size=25, start=50
    #[test]
    fn test_fsdp_state_rank_gets_correct_shard() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 2,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["params".to_string()];
        let vals: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let mut values = HashMap::new();
        values.insert("params".to_string(), vals.clone());
        let unit_id = state.wrap_unit(names, values).expect("wrap failed");

        let unit = &state.units[unit_id];
        assert_eq!(unit.shard_size, 25);

        // rank 2: should hold elements [50..75]
        let shard = state.shard_storage.get(&unit_id).expect("shard missing");
        assert_eq!(shard.len(), 25);
        assert!(
            (shard[0] - 50.0).abs() < 1e-9,
            "first element should be 50.0, got {}",
            shard[0]
        );
    }

    // Test 28: forward_prefetch flag in config
    #[test]
    fn test_forward_prefetch_flag() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 0,
            forward_prefetch: true,
            ..Default::default()
        };
        assert!(config.forward_prefetch);
        let state = FsdpState::new(config);
        assert!(state.config.forward_prefetch);
        assert!(!FsdpConfig::default().forward_prefetch);
    }

    // Test 29: reduce_scatter rank=1 of world_size=4, 8 params, grads=[4.0;8], averaged=1.0
    #[test]
    fn test_reduce_scatter_rank1_of_4() {
        let config = FsdpConfig {
            world_size: 4,
            local_rank: 1,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let names = vec!["r".to_string()];
        let mut values = HashMap::new();
        values.insert("r".to_string(), vec![1.0f64; 8]);
        let unit_id = state.wrap_unit(names, values).expect("wrap");
        // shard_size = ceil(8/4) = 2
        let grads = vec![4.0f64; 8]; // all 4.0, averaged → 1.0
        let local = state.reduce_scatter_grads(unit_id, grads).expect("scatter");
        assert_eq!(local.len(), 2, "rank 1 should hold 2 elements");
        for &g in &local {
            assert!((g - 1.0).abs() < 1e-9, "Expected 1.0, got {g}");
        }
    }

    // Test 30: wrapping empty param_names returns EmptyUnit error
    #[test]
    fn test_wrap_empty_unit_error() {
        let config = FsdpConfig {
            world_size: 2,
            local_rank: 0,
            ..Default::default()
        };
        let mut state = FsdpState::new(config);
        let result = state.wrap_unit(vec![], HashMap::new());
        assert!(matches!(result, Err(FsdpError::EmptyUnit)));
    }
}
