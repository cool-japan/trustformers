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
