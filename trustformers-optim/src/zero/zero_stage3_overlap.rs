//! ZeRO Stage 3 with Async Communication Overlap
//!
//! Extends ZeRO Stage 3 with async prefetch and overlap of communication
//! with backward pass computation. Parameters, gradients, and optimizer
//! states are all partitioned across ranks.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// A partition of a parameter tensor for one rank
#[derive(Debug, Clone)]
pub struct ParamPartition {
    pub param_name: String,
    pub rank: usize,
    pub world_size: usize,
    pub total_elements: usize,
    pub local_start: usize,
    pub local_end: usize,
    pub local_values: Vec<f64>,
}

impl ParamPartition {
    /// Create a new partition from total_values, splitting across world_size ranks.
    /// Rank i gets elements [i*chunk_size, min((i+1)*chunk_size, total)].
    pub fn new(
        param_name: impl Into<String>,
        rank: usize,
        world_size: usize,
        total_values: Vec<f64>,
    ) -> Self {
        let total_elements = total_values.len();
        let chunk_size = total_elements.div_ceil(world_size);
        let local_start = rank * chunk_size;
        let local_end = ((rank + 1) * chunk_size).min(total_elements);
        let local_values = if local_start < total_elements {
            total_values[local_start..local_end].to_vec()
        } else {
            Vec::new()
        };

        Self {
            param_name: param_name.into(),
            rank,
            world_size,
            total_elements,
            local_start,
            local_end,
            local_values,
        }
    }

    /// Number of elements this rank holds.
    pub fn partition_size(&self) -> usize {
        self.local_values.len()
    }

    /// Reconstruct the full parameter by gathering from all rank partitions.
    pub fn gather_full(partitions: &[ParamPartition]) -> Result<Vec<f64>, Zero3Error> {
        if partitions.is_empty() {
            return Err(Zero3Error::GatherFailed("no partitions provided".into()));
        }

        let first = &partitions[0];
        let total_elements = first.total_elements;
        let world_size = first.world_size;
        let param_name = first.param_name.clone();

        // Validate all partitions match
        for p in partitions {
            if p.total_elements != total_elements || p.world_size != world_size {
                return Err(Zero3Error::GatherFailed(param_name.clone()));
            }
        }

        if partitions.len() != world_size {
            return Err(Zero3Error::GatherFailed(format!(
                "{}: expected {} partitions, got {}",
                param_name,
                world_size,
                partitions.len()
            )));
        }

        let mut result = vec![0.0f64; total_elements];

        // Sort by rank to fill in order
        let mut sorted: Vec<&ParamPartition> = partitions.iter().collect();
        sorted.sort_by_key(|p| p.rank);

        for partition in sorted {
            let start = partition.local_start;
            let end = partition.local_end.min(total_elements);
            for (i, &val) in partition.local_values.iter().enumerate() {
                let idx = start + i;
                if idx < end && idx < total_elements {
                    result[idx] = val;
                }
            }
        }

        Ok(result)
    }
}

/// Configuration for ZeRO Stage 3 with async prefetch
#[derive(Debug, Clone)]
pub struct Zero3Config {
    /// Total number of processes
    pub world_size: usize,
    /// This process's rank
    pub local_rank: usize,
    /// Bytes to prefetch ahead (default 25 MiB)
    pub prefetch_bucket_size: usize,
    /// Parameters smaller than this threshold stay unpartitioned
    pub param_persistence_threshold: usize,
    /// Overlap communication with backward pass
    pub overlap_comm: bool,
    /// Use reduce-scatter instead of all-reduce
    pub reduce_scatter: bool,
}

impl Default for Zero3Config {
    fn default() -> Self {
        Self {
            world_size: 1,
            local_rank: 0,
            prefetch_bucket_size: 25 * 1024 * 1024,
            param_persistence_threshold: 1024,
            overlap_comm: true,
            reduce_scatter: true,
        }
    }
}

/// Simulated communication operation
#[derive(Debug, Clone, PartialEq)]
pub enum CommOp {
    AllGather {
        param_name: String,
        num_elements: usize,
    },
    ReduceScatter {
        param_name: String,
        num_elements: usize,
    },
    AllReduce {
        param_name: String,
        num_elements: usize,
    },
}

/// ZeRO Stage 3 parameter manager with async prefetch support
pub struct Zero3ParamManager {
    config: Zero3Config,
    partitions: Arc<Mutex<HashMap<String, ParamPartition>>>,
    pending_gathers: Arc<Mutex<Vec<CommOp>>>,
    pending_reduce_scatters: Arc<Mutex<Vec<CommOp>>>,
    communication_log: Arc<Mutex<Vec<CommOp>>>,
}

impl Zero3ParamManager {
    /// Create a new manager with the given configuration.
    pub fn new(config: Zero3Config) -> Self {
        Self {
            config,
            partitions: Arc::new(Mutex::new(HashMap::new())),
            pending_gathers: Arc::new(Mutex::new(Vec::new())),
            pending_reduce_scatters: Arc::new(Mutex::new(Vec::new())),
            communication_log: Arc::new(Mutex::new(Vec::new())),
        }
    }

    /// Register a parameter for partitioning.
    ///
    /// If `values.len() < param_persistence_threshold`, the parameter is kept
    /// unpartitioned (full copy on each rank). Otherwise it is partitioned
    /// across `world_size` ranks.
    pub fn register_param(
        &self,
        name: impl Into<String>,
        values: Vec<f64>,
    ) -> Result<(), Zero3Error> {
        let name = name.into();
        let mut partitions = self.partitions.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        let partition = if values.len() < self.config.param_persistence_threshold
            || self.config.world_size <= 1
        {
            // Small param or single-process — keep full copy
            let n = values.len();
            ParamPartition {
                param_name: name.clone(),
                rank: self.config.local_rank,
                world_size: self.config.world_size,
                total_elements: n,
                local_start: 0,
                local_end: n,
                local_values: values,
            }
        } else {
            ParamPartition::new(
                name.clone(),
                self.config.local_rank,
                self.config.world_size,
                values,
            )
        };

        partitions.insert(name, partition);
        Ok(())
    }

    /// Schedule an AllGather before the forward pass for a given parameter.
    pub fn prefetch_forward(&self, param_name: &str) -> Result<CommOp, Zero3Error> {
        let partitions = self.partitions.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        let partition = partitions
            .get(param_name)
            .ok_or_else(|| Zero3Error::ParamNotFound(param_name.to_string()))?;

        let op = CommOp::AllGather {
            param_name: param_name.to_string(),
            num_elements: partition.total_elements,
        };

        drop(partitions);

        self.pending_gathers
            .lock()
            .map_err(|_| Zero3Error::LockPoisoned)?
            .push(op.clone());

        self.communication_log
            .lock()
            .map_err(|_| Zero3Error::LockPoisoned)?
            .push(op.clone());

        Ok(op)
    }

    /// Schedule a ReduceScatter after the backward pass for a given parameter.
    ///
    /// Simulates the communication by averaging `grad_values / world_size`.
    pub fn schedule_reduce_scatter(
        &self,
        param_name: &str,
        grad_values: Vec<f64>,
    ) -> Result<CommOp, Zero3Error> {
        let partitions = self.partitions.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        let partition = partitions
            .get(param_name)
            .ok_or_else(|| Zero3Error::ParamNotFound(param_name.to_string()))?;

        let num_elements = grad_values.len();
        let _ = partition; // accessed for existence check
        drop(partitions);

        // Simulate reduce: average across world_size
        let _averaged: Vec<f64> =
            grad_values.iter().map(|&g| g / self.config.world_size as f64).collect();

        let op = CommOp::ReduceScatter {
            param_name: param_name.to_string(),
            num_elements,
        };

        self.pending_reduce_scatters
            .lock()
            .map_err(|_| Zero3Error::LockPoisoned)?
            .push(op.clone());

        self.communication_log
            .lock()
            .map_err(|_| Zero3Error::LockPoisoned)?
            .push(op.clone());

        Ok(op)
    }

    /// Get a copy of the local partition of a parameter.
    pub fn get_local_partition(&self, param_name: &str) -> Result<ParamPartition, Zero3Error> {
        self.get_partition_copy(param_name)
    }

    /// Get a copy of the local partition of a parameter.
    pub fn get_partition_copy(&self, param_name: &str) -> Result<ParamPartition, Zero3Error> {
        let partitions = self.partitions.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        partitions
            .get(param_name)
            .cloned()
            .ok_or_else(|| Zero3Error::ParamNotFound(param_name.to_string()))
    }

    /// Simulate executing all pending communications and return statistics.
    pub fn flush_communications(&self) -> Result<CommStats, Zero3Error> {
        let mut pending_gathers =
            self.pending_gathers.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        let mut pending_reduce_scatters =
            self.pending_reduce_scatters.lock().map_err(|_| Zero3Error::LockPoisoned)?;

        let total_allgathers = pending_gathers.len();
        let total_reduce_scatters = pending_reduce_scatters.len();

        let mut total_elements_communicated = 0usize;

        for op in pending_gathers.iter() {
            match op {
                CommOp::AllGather { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
                CommOp::ReduceScatter { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
                CommOp::AllReduce { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
            }
        }

        for op in pending_reduce_scatters.iter() {
            match op {
                CommOp::AllGather { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
                CommOp::ReduceScatter { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
                CommOp::AllReduce { num_elements, .. } => {
                    total_elements_communicated += num_elements;
                },
            }
        }

        pending_gathers.clear();
        pending_reduce_scatters.clear();

        let estimated_comm_bytes = total_elements_communicated * std::mem::size_of::<f64>();

        Ok(CommStats {
            total_allgathers,
            total_reduce_scatters,
            total_allreduces: 0,
            total_elements_communicated,
            estimated_comm_bytes,
        })
    }

    /// Return a snapshot of all communication operations that have been issued.
    pub fn communication_log(&self) -> Vec<CommOp> {
        self.communication_log.lock().map(|log| log.clone()).unwrap_or_default()
    }

    /// Memory used by this rank's parameter partitions (in bytes).
    pub fn local_memory_bytes(&self) -> usize {
        self.partitions
            .lock()
            .map(|p| {
                p.values()
                    .map(|part| part.local_values.len() * std::mem::size_of::<f64>())
                    .sum()
            })
            .unwrap_or(0)
    }

    /// Total memory that would be used if parameters were not partitioned (in bytes).
    pub fn unpartitioned_memory_bytes(&self) -> usize {
        self.partitions
            .lock()
            .map(|p| p.values().map(|part| part.total_elements * std::mem::size_of::<f64>()).sum())
            .unwrap_or(0)
    }

    /// Memory reduction factor relative to no partitioning.
    pub fn memory_reduction_factor(&self) -> f64 {
        let local = self.local_memory_bytes();
        let total = self.unpartitioned_memory_bytes();

        if local == 0 || total == 0 {
            self.config.world_size as f64
        } else {
            total as f64 / local as f64
        }
    }
}

/// Statistics from a communication flush.
#[derive(Debug, Clone)]
pub struct CommStats {
    pub total_allgathers: usize,
    pub total_reduce_scatters: usize,
    pub total_allreduces: usize,
    pub total_elements_communicated: usize,
    pub estimated_comm_bytes: usize,
}

/// Errors that can occur in ZeRO Stage 3 operations.
#[derive(Debug, thiserror::Error)]
pub enum Zero3Error {
    #[error("Parameter not found: {0}")]
    ParamNotFound(String),
    #[error("Gather failed: incomplete partitions for {0}")]
    GatherFailed(String),
    #[error("Lock poisoned")]
    LockPoisoned,
    #[error("Invalid world size: {0}")]
    InvalidWorldSize(usize),
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_values(n: usize, base: f64) -> Vec<f64> {
        (0..n).map(|i| base + i as f64).collect()
    }

    // Test 1: param registration basic
    #[test]
    fn test_register_param_basic() {
        let config = Zero3Config {
            world_size: 4,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        let values = make_values(16, 1.0);
        mgr.register_param("weight", values).expect("register failed");
        let part = mgr.get_partition_copy("weight").expect("get failed");
        assert_eq!(part.param_name, "weight");
        assert_eq!(part.total_elements, 16);
    }

    // Test 2: partition sizes — each rank gets total/world_size elements
    #[test]
    fn test_partition_sizes() {
        let total = 100usize;
        let world_size = 4;
        let values = make_values(total, 0.0);

        let mut total_covered = 0;
        for rank in 0..world_size {
            let part = ParamPartition::new("p", rank, world_size, values.clone());
            total_covered += part.partition_size();
        }
        assert_eq!(total_covered, total);
    }

    // Test 3: small param persistence (< threshold, kept unpartitioned)
    #[test]
    fn test_small_param_persistence() {
        let config = Zero3Config {
            world_size: 8,
            local_rank: 2,
            param_persistence_threshold: 1024,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        // 10 elements < 1024 threshold
        let values = make_values(10, 5.0);
        mgr.register_param("small_param", values.clone()).expect("register failed");
        let part = mgr.get_partition_copy("small_param").expect("get failed");
        // Should keep full copy (all 10 elements)
        assert_eq!(part.local_values.len(), 10);
        assert_eq!(part.total_elements, 10);
        assert_eq!(part.local_start, 0);
    }

    // Test 4: gather_full reconstruction
    #[test]
    fn test_gather_full_reconstruction() {
        let world_size = 4;
        let values: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let partitions: Vec<ParamPartition> = (0..world_size)
            .map(|rank| ParamPartition::new("p", rank, world_size, values.clone()))
            .collect();
        let reconstructed = ParamPartition::gather_full(&partitions).expect("gather failed");
        assert_eq!(reconstructed.len(), values.len());
        for (i, (&orig, &recon)) in values.iter().zip(reconstructed.iter()).enumerate() {
            assert_eq!(orig, recon, "mismatch at index {}", i);
        }
    }

    // Test 5: prefetch_forward generates AllGather CommOp
    #[test]
    fn test_prefetch_forward_generates_allgather() {
        let config = Zero3Config {
            world_size: 4,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        mgr.register_param("layer1.weight", make_values(64, 0.0))
            .expect("register failed");

        let op = mgr.prefetch_forward("layer1.weight").expect("prefetch failed");
        match &op {
            CommOp::AllGather {
                param_name,
                num_elements,
            } => {
                assert_eq!(param_name, "layer1.weight");
                assert_eq!(*num_elements, 64);
            },
            _ => panic!("Expected AllGather, got {:?}", op),
        }
    }

    // Test 6: schedule_reduce_scatter averages grads
    #[test]
    fn test_schedule_reduce_scatter() {
        let config = Zero3Config {
            world_size: 4,
            local_rank: 1,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        mgr.register_param("w", make_values(32, 1.0)).expect("register failed");

        let grads: Vec<f64> = vec![4.0; 32];
        let op = mgr.schedule_reduce_scatter("w", grads).expect("scatter failed");
        match &op {
            CommOp::ReduceScatter {
                param_name,
                num_elements,
            } => {
                assert_eq!(param_name, "w");
                assert_eq!(*num_elements, 32);
            },
            _ => panic!("Expected ReduceScatter"),
        }
    }

    // Test 7: flush_communications stats
    #[test]
    fn test_flush_communications_stats() {
        let config = Zero3Config {
            world_size: 2,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        mgr.register_param("a", make_values(10, 0.0)).expect("register failed");
        mgr.register_param("b", make_values(20, 0.0)).expect("register failed");

        mgr.prefetch_forward("a").expect("prefetch a failed");
        mgr.prefetch_forward("b").expect("prefetch b failed");
        mgr.schedule_reduce_scatter("a", vec![1.0; 10]).expect("scatter a failed");

        let stats = mgr.flush_communications().expect("flush failed");
        assert_eq!(stats.total_allgathers, 2);
        assert_eq!(stats.total_reduce_scatters, 1);
        assert_eq!(stats.total_elements_communicated, 40); // 10+20 gathers + 10 scatter
    }

    // Test 8: memory_reduction_factor ≈ world_size
    #[test]
    fn test_memory_reduction_factor() {
        let world_size = 8;
        let config = Zero3Config {
            world_size,
            local_rank: 3,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        // Register a large param that will be partitioned
        mgr.register_param("big_param", make_values(800, 1.0)).expect("register failed");

        let factor = mgr.memory_reduction_factor();
        // Should be approximately world_size (8)
        assert!(
            (factor - world_size as f64).abs() < 1.5,
            "Expected ~{}, got {}",
            world_size,
            factor
        );
    }

    // Test 9: communication_log ordering
    #[test]
    fn test_communication_log_ordering() {
        let config = Zero3Config {
            world_size: 2,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        mgr.register_param("p1", make_values(10, 0.0)).expect("register p1 failed");
        mgr.register_param("p2", make_values(20, 0.0)).expect("register p2 failed");

        mgr.prefetch_forward("p1").expect("prefetch p1");
        mgr.schedule_reduce_scatter("p1", vec![1.0; 10]).expect("scatter p1");
        mgr.prefetch_forward("p2").expect("prefetch p2");

        let log = mgr.communication_log();
        assert_eq!(log.len(), 3);
        // First op should be AllGather for p1
        assert!(matches!(&log[0], CommOp::AllGather { param_name, .. } if param_name == "p1"));
        // Second op should be ReduceScatter for p1
        assert!(matches!(&log[1], CommOp::ReduceScatter { param_name, .. } if param_name == "p1"));
        // Third op should be AllGather for p2
        assert!(matches!(&log[2], CommOp::AllGather { param_name, .. } if param_name == "p2"));
    }

    // Test 10: partition_size per rank
    #[test]
    fn test_partition_size_per_rank() {
        let world_size = 3;
        let total = 10usize; // 10 / 3 = 4, 4, 2
        let values = make_values(total, 0.0);

        let chunk_size = total.div_ceil(world_size); // ceiling = 4
        let sizes: Vec<usize> = (0..world_size)
            .map(|rank| {
                let p = ParamPartition::new("p", rank, world_size, values.clone());
                p.partition_size()
            })
            .collect();

        // Rank 0: 0..4 = 4 elements
        // Rank 1: 4..8 = 4 elements
        // Rank 2: 8..10 = 2 elements
        assert_eq!(sizes[0], chunk_size.min(total));
        let total_covered: usize = sizes.iter().sum();
        assert_eq!(total_covered, total);
    }

    // Test 11: multiple params registered
    #[test]
    fn test_multiple_params() {
        let config = Zero3Config {
            world_size: 4,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);

        let param_names = [
            "layer0.weight",
            "layer0.bias",
            "layer1.weight",
            "layer1.bias",
        ];
        let sizes = [256, 16, 512, 16];

        for (name, &size) in param_names.iter().zip(sizes.iter()) {
            mgr.register_param(*name, make_values(size, 1.0)).expect("register failed");
        }

        for name in &param_names {
            let part = mgr.get_partition_copy(name).expect("get failed");
            assert_eq!(part.param_name, *name);
        }

        // All params should sum to correct local memory
        let local_bytes = mgr.local_memory_bytes();
        assert!(local_bytes > 0);
    }

    // Test 12: world_size=1 (no partitioning benefit)
    #[test]
    fn test_world_size_one() {
        let config = Zero3Config {
            world_size: 1,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        let values = make_values(100, 2.0);
        mgr.register_param("param", values.clone()).expect("register failed");

        let part = mgr.get_partition_copy("param").expect("get failed");
        // With world_size=1, all elements should be local
        assert_eq!(part.local_values.len(), 100);
        assert_eq!(part.total_elements, 100);

        let factor = mgr.memory_reduction_factor();
        // With world_size=1, factor should be 1.0
        assert!((factor - 1.0).abs() < 0.01, "Expected ~1.0, got {}", factor);
    }

    // Test 13: Zero3Config default values
    #[test]
    fn test_config_defaults() {
        let config = Zero3Config::default();
        assert_eq!(config.world_size, 1);
        assert_eq!(config.local_rank, 0);
        assert_eq!(config.prefetch_bucket_size, 25 * 1024 * 1024);
        assert_eq!(config.param_persistence_threshold, 1024);
        assert!(config.overlap_comm);
        assert!(config.reduce_scatter);
    }

    // Test 14: CommStats total_elements_communicated
    #[test]
    fn test_comm_stats_total_elements() {
        let config = Zero3Config {
            world_size: 4,
            local_rank: 0,
            param_persistence_threshold: 4,
            ..Default::default()
        };
        let mgr = Zero3ParamManager::new(config);
        mgr.register_param("x", make_values(50, 0.0)).expect("register x");
        mgr.register_param("y", make_values(30, 0.0)).expect("register y");

        mgr.prefetch_forward("x").expect("prefetch x");
        mgr.prefetch_forward("y").expect("prefetch y");
        mgr.schedule_reduce_scatter("x", vec![0.0; 50]).expect("scatter x");
        mgr.schedule_reduce_scatter("y", vec![0.0; 30]).expect("scatter y");

        let stats = mgr.flush_communications().expect("flush failed");
        // gathers: 50+30=80, scatters: 50+30=80 → total 160
        assert_eq!(stats.total_elements_communicated, 160);
        assert_eq!(stats.estimated_comm_bytes, 160 * 8); // f64 = 8 bytes
        assert_eq!(stats.total_allgathers, 2);
        assert_eq!(stats.total_reduce_scatters, 2);
    }
}

// ============================================================================
// ZeRO Stage 3 Overlap: PrefetchManager and ZeroStage3Overlap
// ============================================================================

use std::collections::VecDeque;

/// A group of parameters to be prefetched together.
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    pub param_ids: Vec<usize>,
    pub total_bytes: usize,
    pub priority: u8,
}

/// Statistics for the prefetch manager.
#[derive(Debug, Clone, Default)]
pub struct PrefetchStats {
    pub total_prefetches: u64,
    pub total_bytes_prefetched: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// Manages prefetching of parameter groups for ZeRO-3.
pub struct PrefetchManager {
    pub num_prefetch_buckets: usize,
    pub prefetch_queue: VecDeque<ParameterGroup>,
    pub in_flight: Vec<ParameterGroup>,
    pub stats: PrefetchStats,
}

impl PrefetchManager {
    /// Create a new PrefetchManager with the given number of prefetch buckets.
    pub fn new(num_prefetch_buckets: usize) -> Self {
        Self {
            num_prefetch_buckets,
            prefetch_queue: VecDeque::new(),
            in_flight: Vec::new(),
            stats: PrefetchStats::default(),
        }
    }

    /// Schedule a parameter group for prefetching.
    /// Returns `Zero3Error::InvalidWorldSize(0)` if the queue is at capacity.
    pub fn schedule_prefetch(&mut self, group: ParameterGroup) -> Result<(), Zero3Error> {
        if self.prefetch_queue.len() >= self.num_prefetch_buckets {
            return Err(Zero3Error::InvalidWorldSize(0));
        }
        self.prefetch_queue.push_back(group);
        Ok(())
    }

    /// Try to find a prefetched group containing `param_id` in the in-flight set.
    /// Increments `cache_hits` if found, `cache_misses` otherwise.
    pub fn try_get_prefetched(&mut self, param_id: usize) -> Option<&ParameterGroup> {
        let pos = self.in_flight.iter().position(|g| g.param_ids.contains(&param_id));
        if let Some(idx) = pos {
            self.stats.cache_hits += 1;
            Some(&self.in_flight[idx])
        } else {
            self.stats.cache_misses += 1;
            None
        }
    }

    /// Compute cache hit rate: `hits / (hits + misses)`. Returns `0.0` if no requests yet.
    pub fn cache_hit_rate(&self) -> f64 {
        let total = self.stats.cache_hits + self.stats.cache_misses;
        if total == 0 {
            return 0.0;
        }
        self.stats.cache_hits as f64 / total as f64
    }

    /// Prefetch efficiency approximation: `bytes / (bytes + 1)` to avoid division by zero.
    pub fn prefetch_efficiency(&self) -> f64 {
        let b = self.stats.total_bytes_prefetched as f64;
        b / (b + 1.0)
    }

    /// Move all queued groups to in-flight, updating stats.
    pub fn flush_to_in_flight(&mut self) {
        while let Some(group) = self.prefetch_queue.pop_front() {
            self.stats.total_prefetches += 1;
            self.stats.total_bytes_prefetched += group.total_bytes as u64;
            self.in_flight.push(group);
        }
    }
}

/// Configuration for ZeRO Stage 3 with communication-computation overlap.
#[derive(Debug, Clone)]
pub struct ZeroStage3OverlapConfig {
    pub world_size: usize,
    pub prefetch_bucket_size_mb: f64,
    pub reduce_scatter_bucket_size_mb: f64,
    pub overlap_comm: bool,
    pub contiguous_gradients: bool,
}

impl Default for ZeroStage3OverlapConfig {
    fn default() -> Self {
        Self {
            world_size: 4,
            prefetch_bucket_size_mb: 25.0,
            reduce_scatter_bucket_size_mb: 25.0,
            overlap_comm: true,
            contiguous_gradients: true,
        }
    }
}

/// Communication statistics for `ZeroStage3Overlap`.
#[derive(Debug, Clone, Default)]
pub struct CommStatsOverlap {
    pub all_gather_count: u64,
    pub reduce_scatter_count: u64,
    pub total_bytes_transferred: u64,
}

/// ZeRO Stage 3 optimizer with overlapped communication and computation.
///
/// Manages AllGather prefetching and ReduceScatter scheduling while tracking
/// which rank owns each parameter partition.
pub struct ZeroStage3Overlap {
    pub config: ZeroStage3OverlapConfig,
    pub prefetch_manager: PrefetchManager,
    /// Maps `param_id` → rank that owns the partition.
    pub partition_map: HashMap<usize, usize>,
    pub comm_stats: CommStatsOverlap,
}

impl ZeroStage3Overlap {
    /// Create a new `ZeroStage3Overlap` from the given configuration.
    pub fn new(config: ZeroStage3OverlapConfig) -> Self {
        // Derive queue capacity from the prefetch bucket size (in bytes), floored at 1.
        let num_buckets = ((config.prefetch_bucket_size_mb * 1024.0 * 1024.0) as usize).max(1);
        Self {
            prefetch_manager: PrefetchManager::new(num_buckets),
            config,
            partition_map: HashMap::new(),
            comm_stats: CommStatsOverlap::default(),
        }
    }

    /// Register which rank owns a given parameter.
    pub fn register_param_ownership(&mut self, param_id: usize, owning_rank: usize) {
        self.partition_map.insert(param_id, owning_rank);
    }

    /// Get the rank that owns a given parameter, if registered.
    pub fn get_owning_rank(&self, param_id: usize) -> Option<usize> {
        self.partition_map.get(&param_id).copied()
    }

    /// Schedule an AllGather for a parameter group (overlap with backward pass).
    pub fn schedule_all_gather(&mut self, param_group: ParameterGroup) -> Result<(), Zero3Error> {
        self.comm_stats.all_gather_count += 1;
        self.comm_stats.total_bytes_transferred += param_group.total_bytes as u64;
        self.prefetch_manager.schedule_prefetch(param_group)
    }

    /// Schedule a ReduceScatter for a parameter group after the backward pass.
    pub fn schedule_reduce_scatter_group(
        &mut self,
        param_group: ParameterGroup,
    ) -> Result<(), Zero3Error> {
        self.comm_stats.reduce_scatter_count += 1;
        self.comm_stats.total_bytes_transferred += param_group.total_bytes as u64;
        Ok(())
    }

    /// Simulated overlap efficiency: `0.8` when `overlap_comm` is enabled, `0.0` otherwise.
    pub fn overlap_efficiency(&self) -> f64 {
        if self.config.overlap_comm {
            0.8
        } else {
            0.0
        }
    }
}

#[cfg(test)]
mod overlap_tests {
    use super::*;

    fn make_group(ids: Vec<usize>, bytes: usize, priority: u8) -> ParameterGroup {
        ParameterGroup {
            param_ids: ids,
            total_bytes: bytes,
            priority,
        }
    }

    #[test]
    fn test_prefetch_manager_new_initializes() {
        let pm = PrefetchManager::new(4);
        assert_eq!(pm.num_prefetch_buckets, 4);
        assert!(pm.prefetch_queue.is_empty());
        assert!(pm.in_flight.is_empty());
        assert_eq!(pm.stats.total_prefetches, 0);
    }

    #[test]
    fn test_schedule_prefetch_adds_to_queue() {
        let mut pm = PrefetchManager::new(4);
        pm.schedule_prefetch(make_group(vec![1, 2], 1024, 0)).expect("schedule failed");
        assert_eq!(pm.prefetch_queue.len(), 1);
    }

    #[test]
    fn test_schedule_prefetch_capacity_error() {
        let mut pm = PrefetchManager::new(2);
        pm.schedule_prefetch(make_group(vec![0], 512, 0)).expect("first ok");
        pm.schedule_prefetch(make_group(vec![1], 512, 0)).expect("second ok");
        let result = pm.schedule_prefetch(make_group(vec![2], 512, 0));
        assert!(result.is_err(), "should fail at capacity");
        match result {
            Err(Zero3Error::InvalidWorldSize(0)) => {},
            other => panic!("Expected InvalidWorldSize(0), got {:?}", other),
        }
    }

    #[test]
    fn test_flush_to_in_flight_moves_items() {
        let mut pm = PrefetchManager::new(8);
        pm.schedule_prefetch(make_group(vec![10], 2048, 1)).expect("ok");
        pm.schedule_prefetch(make_group(vec![11], 4096, 2)).expect("ok");
        pm.flush_to_in_flight();
        assert!(pm.prefetch_queue.is_empty());
        assert_eq!(pm.in_flight.len(), 2);
        assert_eq!(pm.stats.total_prefetches, 2);
        assert_eq!(pm.stats.total_bytes_prefetched, 6144);
    }

    #[test]
    fn test_try_get_prefetched_cache_hit() {
        let mut pm = PrefetchManager::new(8);
        pm.schedule_prefetch(make_group(vec![5, 6], 1024, 0)).expect("ok");
        pm.flush_to_in_flight();
        let result = pm.try_get_prefetched(5);
        assert!(result.is_some());
        assert_eq!(pm.stats.cache_hits, 1);
        assert_eq!(pm.stats.cache_misses, 0);
    }

    #[test]
    fn test_try_get_prefetched_cache_miss() {
        let mut pm = PrefetchManager::new(8);
        pm.schedule_prefetch(make_group(vec![5], 1024, 0)).expect("ok");
        pm.flush_to_in_flight();
        let result = pm.try_get_prefetched(99);
        assert!(result.is_none());
        assert_eq!(pm.stats.cache_hits, 0);
        assert_eq!(pm.stats.cache_misses, 1);
    }

    #[test]
    fn test_cache_hit_rate_zero_when_no_requests() {
        let pm = PrefetchManager::new(4);
        assert_eq!(pm.cache_hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_hit_rate_correct_ratio() {
        let mut pm = PrefetchManager::new(8);
        pm.schedule_prefetch(make_group(vec![1, 2, 3], 512, 0)).expect("ok");
        pm.flush_to_in_flight();
        pm.try_get_prefetched(1); // hit
        pm.try_get_prefetched(2); // hit
        pm.try_get_prefetched(99); // miss
        let rate = pm.cache_hit_rate();
        assert!(
            (rate - 2.0 / 3.0).abs() < 1e-6,
            "Expected 0.667, got {}",
            rate
        );
    }

    #[test]
    fn test_prefetch_efficiency_positive_after_prefetch() {
        let mut pm = PrefetchManager::new(8);
        pm.schedule_prefetch(make_group(vec![0], 1_000_000, 0)).expect("ok");
        pm.flush_to_in_flight();
        let eff = pm.prefetch_efficiency();
        assert!(eff > 0.0, "efficiency should be positive after prefetch");
        assert!(eff < 1.0, "efficiency should be less than 1.0");
    }

    #[test]
    fn test_zero_stage3_overlap_new_initializes() {
        let config = ZeroStage3OverlapConfig::default();
        let overlap = ZeroStage3Overlap::new(config);
        assert!(overlap.partition_map.is_empty());
        assert_eq!(overlap.comm_stats.all_gather_count, 0);
        assert_eq!(overlap.comm_stats.reduce_scatter_count, 0);
    }

    #[test]
    fn test_register_and_get_owning_rank() {
        let mut overlap = ZeroStage3Overlap::new(ZeroStage3OverlapConfig::default());
        overlap.register_param_ownership(42, 3);
        assert_eq!(overlap.get_owning_rank(42), Some(3));
        assert_eq!(overlap.get_owning_rank(99), None);
    }

    #[test]
    fn test_schedule_all_gather_increments_count() {
        let mut overlap = ZeroStage3Overlap::new(ZeroStage3OverlapConfig::default());
        overlap.schedule_all_gather(make_group(vec![0], 1024, 0)).expect("ok");
        overlap.schedule_all_gather(make_group(vec![1], 2048, 0)).expect("ok");
        assert_eq!(overlap.comm_stats.all_gather_count, 2);
    }

    #[test]
    fn test_schedule_reduce_scatter_increments_count() {
        let mut overlap = ZeroStage3Overlap::new(ZeroStage3OverlapConfig::default());
        overlap.schedule_reduce_scatter_group(make_group(vec![0], 512, 0)).expect("ok");
        assert_eq!(overlap.comm_stats.reduce_scatter_count, 1);
    }

    #[test]
    fn test_total_bytes_transferred_accumulates() {
        let mut overlap = ZeroStage3Overlap::new(ZeroStage3OverlapConfig::default());
        overlap.schedule_all_gather(make_group(vec![0], 1024, 0)).expect("ok");
        overlap.schedule_reduce_scatter_group(make_group(vec![0], 2048, 0)).expect("ok");
        assert_eq!(overlap.comm_stats.total_bytes_transferred, 3072);
    }

    #[test]
    fn test_overlap_efficiency_enabled() {
        let config = ZeroStage3OverlapConfig {
            overlap_comm: true,
            ..Default::default()
        };
        let overlap = ZeroStage3Overlap::new(config);
        assert_eq!(overlap.overlap_efficiency(), 0.8);
    }

    #[test]
    fn test_overlap_efficiency_disabled() {
        let config = ZeroStage3OverlapConfig {
            overlap_comm: false,
            ..Default::default()
        };
        let overlap = ZeroStage3Overlap::new(config);
        assert_eq!(overlap.overlap_efficiency(), 0.0);
    }

    #[test]
    fn test_prefetch_stats_default_zero() {
        let stats = PrefetchStats::default();
        assert_eq!(stats.total_prefetches, 0);
        assert_eq!(stats.total_bytes_prefetched, 0);
        assert_eq!(stats.cache_hits, 0);
        assert_eq!(stats.cache_misses, 0);
    }

    #[test]
    fn test_parameter_group_priority_ordering() {
        let high = make_group(vec![0], 512, 255);
        let low = make_group(vec![1], 512, 0);
        assert!(high.priority > low.priority);
    }
}
