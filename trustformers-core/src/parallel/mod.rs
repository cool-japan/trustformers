//! Parallel execution support for TrustformeRS
//!
//! This module provides infrastructure for various parallelism strategies including:
//! - Data parallelism
//! - Model parallelism (tensor and pipeline)
//! - Hybrid parallelism
//! - NUMA-aware optimization

pub mod model_parallel;
pub mod parallel_layers;
pub mod pipeline_parallel;
pub mod tensor_parallel;

pub mod mpi_communicator;

#[cfg(feature = "nccl")]
pub mod nccl_communicator;

pub use model_parallel::{
    CommunicationBackend, Communicator, DeviceMesh, DistributedTensor, ModelParallelConfig,
    ModelParallelContext, ModelParallelStrategy, PipelineOp, PipelineSchedule,
    PipelineScheduleType, TensorPartition,
};

pub use parallel_layers::{
    ActivationType, ColumnParallelLinear, ParallelMLP, ParallelMultiHeadAttention,
    RowParallelLinear,
};

pub use tensor_parallel::{
    AsyncTensorParallel, InitMethod, TensorParallelInit, TensorParallelOps, TensorParallelShapes,
};

pub use pipeline_parallel::{
    MicrobatchManager, PipelineExecutor, PipelineLayer, PipelineModel, PipelineOptimizer,
    PipelineStage,
};

pub use mpi_communicator::{mpi_utils, MpiCommunicatorImpl};

#[cfg(feature = "nccl")]
pub use nccl_communicator::{create_nccl_communicator, NcclCommunicator};

use crate::errors::{runtime_error, Result};
use parking_lot::RwLock;
use std::sync::Arc;

/// Core parallelism strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ParallelismStrategy {
    /// Data parallelism only
    Data,
    /// Model parallelism (tensor or pipeline)
    Model,
    /// Hybrid (data + model)
    Hybrid,
    /// No parallelism (single device)
    None,
}

/// Parallel execution context
#[derive(Clone)]
pub struct ParallelContext {
    strategy: ParallelismStrategy,
    num_devices: usize,
    device_id: usize,
    numa_config: Option<NumaConfig>,
}

/// NUMA configuration for CPU optimization
#[derive(Debug, Clone)]
pub struct NumaConfig {
    pub node_id: usize,
    pub cpu_affinity: Vec<usize>,
    pub memory_policy: MemoryPolicy,
}

#[derive(Debug, Clone, Copy)]
pub enum MemoryPolicy {
    /// Bind memory to local NUMA node
    BindLocal,
    /// Interleave memory across nodes
    Interleave,
    /// Prefer local but allow remote
    PreferLocal,
}

impl ParallelContext {
    pub fn new(strategy: ParallelismStrategy, num_devices: usize) -> Self {
        Self {
            strategy,
            num_devices,
            device_id: 0,
            numa_config: None,
        }
    }

    pub fn with_device_id(mut self, device_id: usize) -> Self {
        self.device_id = device_id;
        self
    }

    pub fn with_numa_config(mut self, numa_config: NumaConfig) -> Self {
        self.numa_config = Some(numa_config);
        self
    }

    pub fn strategy(&self) -> ParallelismStrategy {
        self.strategy
    }

    pub fn num_devices(&self) -> usize {
        self.num_devices
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
}

/// Parallel operations trait
pub trait ParallelOps {
    /// Execute operation in parallel context
    fn parallel_execute<F, T>(&self, f: F) -> Result<T>
    where
        F: FnOnce(&ParallelContext) -> Result<T>;

    /// Map operation across parallel devices
    fn parallel_map<F, T>(&self, items: Vec<T>, f: F) -> Result<Vec<T>>
    where
        F: Fn(T, &ParallelContext) -> Result<T> + Send + Sync,
        T: Send;
}

/// Global parallel context
static PARALLEL_CONTEXT: RwLock<Option<Arc<ParallelContext>>> = RwLock::new(None);

/// Initialize global parallel context
pub fn init_parallelism(context: ParallelContext) {
    *PARALLEL_CONTEXT.write() = Some(Arc::new(context));
}

/// Get global parallel context
pub fn parallel_context() -> Option<Arc<ParallelContext>> {
    PARALLEL_CONTEXT.read().clone()
}

/// Execute function in parallel context
pub fn parallel_execute<F, T>(f: F) -> Result<T>
where
    F: FnOnce(&ParallelContext) -> Result<T>,
{
    let context =
        parallel_context().ok_or_else(|| runtime_error("Parallel context not initialized"))?;
    f(&context)
}

/// Map function across items in parallel
pub fn parallel_map<F, T>(items: Vec<T>, f: F) -> Result<Vec<T>>
where
    F: Fn(T, &ParallelContext) -> Result<T> + Send + Sync,
    T: Send,
{
    let context =
        parallel_context().ok_or_else(|| runtime_error("Parallel context not initialized"))?;

    // Simple implementation - in practice would use thread pool
    items.into_iter().map(|item| f(item, &context)).collect()
}

/// Parallel chunk mapping for large datasets
pub fn parallel_chunk_map<F, T>(items: Vec<T>, chunk_size: usize, f: F) -> Result<Vec<T>>
where
    F: Fn(Vec<T>, &ParallelContext) -> Result<Vec<T>> + Send + Sync,
    T: Send + Clone,
{
    let context =
        parallel_context().ok_or_else(|| runtime_error("Parallel context not initialized"))?;

    let mut chunks = Vec::new();
    let mut i = 0;
    while i < items.len() {
        let end = (i + chunk_size).min(items.len());
        chunks.push(items[i..end].to_vec());
        i = end;
    }

    let results: Result<Vec<Vec<T>>> = chunks.into_iter().map(|chunk| f(chunk, &context)).collect();

    results.map(|vecs| vecs.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── 1. ParallelismStrategy variants are distinct ──────────────────────────

    #[test]
    fn test_parallelism_strategy_variants_distinct() {
        assert_ne!(ParallelismStrategy::Data, ParallelismStrategy::Model);
        assert_ne!(ParallelismStrategy::Hybrid, ParallelismStrategy::None);
        assert_eq!(ParallelismStrategy::Data, ParallelismStrategy::Data);
    }

    // ── 2. ParallelContext constructs with correct strategy ───────────────────

    #[test]
    fn test_parallel_context_strategy() {
        let ctx = ParallelContext::new(ParallelismStrategy::Data, 4);
        assert_eq!(ctx.strategy(), ParallelismStrategy::Data);
    }

    // ── 3. ParallelContext num_devices ────────────────────────────────────────

    #[test]
    fn test_parallel_context_num_devices() {
        let ctx = ParallelContext::new(ParallelismStrategy::Model, 8);
        assert_eq!(ctx.num_devices(), 8);
    }

    // ── 4. ParallelContext default device_id is 0 ────────────────────────────

    #[test]
    fn test_parallel_context_default_device_id() {
        let ctx = ParallelContext::new(ParallelismStrategy::None, 1);
        assert_eq!(ctx.device_id(), 0);
    }

    // ── 5. with_device_id builder ─────────────────────────────────────────────

    #[test]
    fn test_parallel_context_with_device_id() {
        let ctx = ParallelContext::new(ParallelismStrategy::Hybrid, 4).with_device_id(3);
        assert_eq!(ctx.device_id(), 3);
    }

    // ── 6. with_numa_config sets numa config ─────────────────────────────────

    #[test]
    fn test_parallel_context_with_numa_config() {
        let numa = NumaConfig {
            node_id: 1,
            cpu_affinity: vec![0, 1, 2, 3],
            memory_policy: MemoryPolicy::BindLocal,
        };
        let ctx = ParallelContext::new(ParallelismStrategy::None, 1).with_numa_config(numa);
        assert!(ctx.numa_config.is_some(), "numa_config must be set");
    }

    // ── 7. MemoryPolicy variants can be cloned ────────────────────────────────

    #[test]
    fn test_memory_policy_clone() {
        let p = MemoryPolicy::Interleave;
        let q = p;
        let _ = q;
    }

    // ── 8. NumaConfig node_id stored correctly ────────────────────────────────

    #[test]
    fn test_numa_config_node_id() {
        let numa = NumaConfig {
            node_id: 2,
            cpu_affinity: vec![4, 5],
            memory_policy: MemoryPolicy::PreferLocal,
        };
        assert_eq!(numa.node_id, 2);
    }

    // ── 9. init_parallelism + parallel_context round-trip ────────────────────

    #[test]
    fn test_init_and_get_parallel_context() {
        let ctx = ParallelContext::new(ParallelismStrategy::Data, 2);
        init_parallelism(ctx);
        let retrieved = parallel_context();
        assert!(
            retrieved.is_some(),
            "parallel_context must return Some after init"
        );
        let c = retrieved.unwrap_or_else(|| panic!("context is None"));
        assert_eq!(c.strategy(), ParallelismStrategy::Data);
    }

    // ── 10. parallel_execute returns error when not initialized ───────────────
    // NOTE: Since global state may be set from test 9, this tests the happy path.

    #[test]
    fn test_parallel_execute_runs_closure() {
        init_parallelism(ParallelContext::new(ParallelismStrategy::Data, 1));
        let result = parallel_execute(|ctx| {
            assert_eq!(ctx.num_devices(), 1);
            Ok(42u32)
        });
        assert_eq!(result.unwrap_or(0), 42, "parallel_execute must run closure");
    }

    // ── 11. ParallelismStrategy is Copy ───────────────────────────────────────

    #[test]
    fn test_parallelism_strategy_is_copy() {
        let s = ParallelismStrategy::Hybrid;
        let t = s; // copy
        assert_eq!(s, t);
    }

    // ── 12. parallel_map with initialized context ─────────────────────────────

    #[test]
    fn test_parallel_map_doubles_values() {
        init_parallelism(ParallelContext::new(ParallelismStrategy::None, 1));
        let items = vec![1u32, 2, 3, 4];
        let result = parallel_map(items, |item, _ctx| Ok(item * 2));
        let values = result.unwrap_or_default();
        assert_eq!(
            values,
            vec![2u32, 4, 6, 8],
            "parallel_map must double values"
        );
    }

    // ── 13. NumaConfig cpu_affinity stored correctly ──────────────────────────

    #[test]
    fn test_numa_config_cpu_affinity() {
        let affinity = vec![0usize, 2, 4, 6];
        let numa = NumaConfig {
            node_id: 0,
            cpu_affinity: affinity.clone(),
            memory_policy: MemoryPolicy::Interleave,
        };
        assert_eq!(numa.cpu_affinity, affinity);
    }

    // ── 14. ParallelContext clone works ───────────────────────────────────────

    #[test]
    fn test_parallel_context_clone() {
        let ctx = ParallelContext::new(ParallelismStrategy::Hybrid, 3);
        let _cloned = ctx.clone();
    }
}
