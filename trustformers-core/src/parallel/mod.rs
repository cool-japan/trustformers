//! Parallel execution support for TrustformeRS
//!
//! This module provides infrastructure for various parallelism strategies including:
//! - Data parallelism
//! - Model parallelism (tensor and pipeline)
//! - Hybrid parallelism
//! - NUMA-aware optimization

pub mod local_communicator;
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

pub use local_communicator::LocalCommunicator;
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

/// Map function across items in parallel.
///
/// Genuinely parallel: dispatches through `scirs2_core::parallel_ops`
/// (rayon), which preserves input order in its output (`Vec`'s
/// `into_par_iter` is an `IndexedParallelIterator`). Falls back to the
/// crate's sequential rayon shim automatically when the `parallel` feature
/// is off, so this compiles and behaves correctly either way.
pub fn parallel_map<F, T>(items: Vec<T>, f: F) -> Result<Vec<T>>
where
    F: Fn(T, &ParallelContext) -> Result<T> + Send + Sync,
    T: Send,
{
    let context =
        parallel_context().ok_or_else(|| runtime_error("Parallel context not initialized"))?;

    use scirs2_core::parallel_ops::*;
    items.into_par_iter().map(|item| f(item, &context)).collect()
}

/// Parallel chunk mapping for large datasets.
///
/// Each chunk is processed by a separate task via
/// `scirs2_core::parallel_ops` (rayon); chunk order is preserved before
/// flattening.
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

    use scirs2_core::parallel_ops::*;
    let results: Result<Vec<Vec<T>>> =
        chunks.into_par_iter().map(|chunk| f(chunk, &context)).collect();

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

    // ── 15. parallel_map actually uses more than one thread ───────────────────
    //
    // Regression test: before this fix, `parallel_map` was
    // `items.into_iter().map(...)` - fully sequential despite its name and
    // `Send + Sync` bounds. Every closure invocation would have run on the
    // calling thread, so this would see exactly one distinct `ThreadId`
    // (`seen.len() == 1`) against the old implementation. With a real rayon
    // dispatch, closures that overlap in time (via `sleep`) get distributed
    // across the global thread pool's worker threads.

    #[test]
    fn test_parallel_map_uses_multiple_threads() {
        use std::collections::HashSet;
        use std::sync::Mutex as StdMutex;
        use std::thread;

        if thread::available_parallelism().map(|n| n.get()).unwrap_or(1) < 2 {
            // Cannot observe >1 thread in use on a single-hardware-thread
            // machine; skip rather than produce a flaky failure.
            return;
        }

        init_parallelism(ParallelContext::new(ParallelismStrategy::Data, 4));

        let thread_ids: Arc<StdMutex<HashSet<thread::ThreadId>>> =
            Arc::new(StdMutex::new(HashSet::new()));
        let items: Vec<u32> = (0..64).collect();

        let ids_for_closure = Arc::clone(&thread_ids);
        let result = parallel_map(items, move |item, _ctx| {
            ids_for_closure
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .insert(thread::current().id());
            // Encourage the scheduler to actually overlap work across
            // threads instead of draining the queue on whichever thread
            // grabs it first.
            thread::sleep(std::time::Duration::from_millis(1));
            Ok(item)
        });

        assert!(result.is_ok());
        let seen = thread_ids.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        assert!(
            seen.len() > 1,
            "parallel_map should run its closure from more than one thread when \
             available_parallelism() > 1; saw {} distinct thread(s) - looks sequential",
            seen.len()
        );
    }

    // ── 16. parallel_chunk_map preserves order and chunk boundaries ───────────

    #[test]
    fn test_parallel_chunk_map_preserves_order() {
        init_parallelism(ParallelContext::new(ParallelismStrategy::Data, 2));
        let items: Vec<u32> = (0..10).collect();
        let result = parallel_chunk_map(items, 3, |chunk, _ctx| {
            Ok(chunk.into_iter().map(|v| v * 10).collect())
        });
        assert_eq!(
            result.unwrap_or_default(),
            vec![0u32, 10, 20, 30, 40, 50, 60, 70, 80, 90],
            "parallel_chunk_map must preserve item order across chunk boundaries"
        );
    }
}
