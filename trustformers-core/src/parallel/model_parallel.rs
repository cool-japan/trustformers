//! Model Parallel Support for Large Models
//!
//! This module provides infrastructure for distributing model layers and tensors
//! across multiple devices (GPUs) to enable training and inference of models
//! that are too large to fit on a single device.

#![allow(unused_variables)] // Model parallelism implementation

#[allow(unused_imports)] // Used conditionally based on feature gates
use crate::errors::{runtime_error, tensor_op_error, Result};
use crate::Tensor;
use std::sync::Arc;

/// Model parallelism strategy
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelParallelStrategy {
    /// Pipeline parallelism - split model by layers
    Pipeline,
    /// Tensor parallelism - split individual layers
    Tensor,
    /// Hybrid approach combining both
    Hybrid,
}

/// Configuration for model parallel execution
#[derive(Debug, Clone)]
pub struct ModelParallelConfig {
    /// Number of devices to use
    pub num_devices: usize,
    /// Parallelism strategy
    pub strategy: ModelParallelStrategy,
    /// Device IDs to use (e.g., [0, 1, 2, 3] for 4 GPUs)
    pub device_ids: Vec<usize>,
    /// Pipeline depth for pipeline parallelism
    pub pipeline_depth: Option<usize>,
    /// Tensor split dimension for tensor parallelism
    pub tensor_split_dim: Option<usize>,
    /// Enable gradient checkpointing to save memory
    pub gradient_checkpointing: bool,
    /// Communication backend
    pub comm_backend: CommunicationBackend,
}

impl Default for ModelParallelConfig {
    fn default() -> Self {
        Self {
            num_devices: 1,
            strategy: ModelParallelStrategy::Pipeline,
            device_ids: vec![0],
            pipeline_depth: None,
            tensor_split_dim: None,
            gradient_checkpointing: false,
            comm_backend: CommunicationBackend::Nccl,
        }
    }
}

/// Communication backend for model parallel
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum CommunicationBackend {
    /// NVIDIA Collective Communication Library
    Nccl,
    /// Message Passing Interface
    Mpi,
    /// Gloo (CPU communication)
    Gloo,
    /// Custom implementation
    Custom,
}

/// Distributed tensor that can be split across devices
#[derive(Debug, Clone)]
pub struct DistributedTensor {
    /// Local shard of the tensor
    pub local_shard: Tensor,
    /// Global shape of the full tensor
    pub global_shape: Vec<usize>,
    /// Partition info
    pub partition: TensorPartition,
    /// Device ID where this shard resides
    pub device_id: usize,
}

/// Information about how a tensor is partitioned
#[derive(Debug, Clone)]
pub struct TensorPartition {
    /// Dimension along which tensor is split
    pub split_dim: usize,
    /// Start index in the global tensor
    pub start_idx: usize,
    /// End index in the global tensor
    pub end_idx: usize,
    /// Total number of partitions
    pub num_partitions: usize,
    /// This partition's rank
    pub partition_rank: usize,
}

impl DistributedTensor {
    /// Create a new distributed tensor from a local shard
    pub fn new(
        local_shard: Tensor,
        global_shape: Vec<usize>,
        partition: TensorPartition,
        device_id: usize,
    ) -> Self {
        Self {
            local_shard,
            global_shape,
            partition,
            device_id,
        }
    }

    /// Get the local shape of this shard
    pub fn local_shape(&self) -> Vec<usize> {
        self.local_shard.shape()
    }

    /// Check if this tensor needs communication for operations
    pub fn requires_communication(&self) -> bool {
        self.partition.num_partitions > 1
    }
}

/// Model parallel context managing distributed execution
pub struct ModelParallelContext {
    config: ModelParallelConfig,
    rank: usize,
    world_size: usize,
    pub(crate) communicator: Arc<dyn Communicator>,
    #[allow(dead_code)]
    device_mesh: DeviceMesh,
}

impl ModelParallelContext {
    /// Initialize model parallel context
    pub fn new(config: ModelParallelConfig) -> Result<Self> {
        let world_size = config.num_devices;
        let rank = 0; // Will be set by init process

        let communicator = create_communicator(&config.comm_backend)?;
        let device_mesh = DeviceMesh::new(&config.device_ids, config.strategy)?;

        Ok(Self {
            config,
            rank,
            world_size,
            communicator,
            device_mesh,
        })
    }

    /// Get current process rank
    pub fn rank(&self) -> usize {
        self.rank
    }

    /// Get total world size
    pub fn world_size(&self) -> usize {
        self.world_size
    }

    /// Partition a tensor across devices
    pub fn partition_tensor(&self, tensor: &Tensor, split_dim: usize) -> Result<DistributedTensor> {
        let shape = tensor.shape();
        if split_dim >= shape.len() {
            return Err(tensor_op_error(
                "split_tensor",
                format!(
                    "Split dimension {} out of bounds for tensor with {} dimensions",
                    split_dim,
                    shape.len()
                ),
            ));
        }

        let dim_size = shape[split_dim];
        let chunk_size = (dim_size + self.world_size - 1) / self.world_size;
        let start_idx = self.rank * chunk_size;
        let end_idx = ((self.rank + 1) * chunk_size).min(dim_size);

        // Extract local shard by slicing the tensor along the split dimension
        let local_shard = tensor.slice(split_dim, start_idx, end_idx)?;

        let partition = TensorPartition {
            split_dim,
            start_idx,
            end_idx,
            num_partitions: self.world_size,
            partition_rank: self.rank,
        };

        Ok(DistributedTensor::new(
            local_shard,
            shape.to_vec(),
            partition,
            self.config.device_ids[self.rank],
        ))
    }

    /// Gather distributed tensor to full tensor
    pub fn all_gather(&self, distributed: &DistributedTensor) -> Result<Tensor> {
        if !distributed.requires_communication() {
            return Ok(distributed.local_shard.clone());
        }

        self.communicator
            .all_gather(&distributed.local_shard, distributed.partition.split_dim)
    }

    /// Reduce scattered tensor across devices
    pub fn reduce_scatter(&self, tensor: &Tensor, split_dim: usize) -> Result<Tensor> {
        self.communicator.reduce_scatter(tensor, split_dim)
    }

    /// All-reduce operation for gradient synchronization
    pub fn all_reduce(&self, tensor: &mut Tensor) -> Result<()> {
        self.communicator.all_reduce(tensor)
    }

    /// Broadcast tensor from root rank to all other ranks
    pub fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()> {
        self.communicator.broadcast(tensor, root)
    }
}

/// Device mesh for organizing devices in model parallel
#[derive(Debug, Clone)]
pub struct DeviceMesh {
    /// Device IDs in the mesh
    device_ids: Vec<usize>,
    /// Topology of the mesh
    topology: MeshTopology,
}

#[derive(Debug, Clone)]
enum MeshTopology {
    /// Linear arrangement (for pipeline parallel)
    Linear,
    /// 2D mesh (for tensor parallel)
    Grid2D { rows: usize, cols: usize },
    /// 3D mesh (for hybrid parallel)
    #[allow(dead_code)]
    Grid3D { x: usize, y: usize, z: usize },
}

impl DeviceMesh {
    fn new(device_ids: &[usize], strategy: ModelParallelStrategy) -> Result<Self> {
        let topology = match strategy {
            ModelParallelStrategy::Pipeline => MeshTopology::Linear,
            ModelParallelStrategy::Tensor => {
                // For tensor parallel, try to create a balanced 2D grid
                let n = device_ids.len();
                let rows = (n as f64).sqrt().ceil() as usize;
                let cols = (n + rows - 1) / rows;
                MeshTopology::Grid2D { rows, cols }
            },
            ModelParallelStrategy::Hybrid => {
                // For hybrid, create a 3D mesh if possible
                // This is a simplified version
                MeshTopology::Linear
            },
        };

        Ok(Self {
            device_ids: device_ids.to_vec(),
            topology,
        })
    }

    /// Get device ID at a given coordinate
    pub fn device_at(&self, coord: &[usize]) -> Option<usize> {
        match &self.topology {
            MeshTopology::Linear => {
                coord.first().and_then(|&idx| self.device_ids.get(idx).copied())
            },
            MeshTopology::Grid2D { rows, cols } => {
                if coord.len() >= 2 {
                    let idx = coord[0] * cols + coord[1];
                    self.device_ids.get(idx).copied()
                } else {
                    None
                }
            },
            MeshTopology::Grid3D { x, y, z } => {
                if coord.len() >= 3 {
                    let idx = coord[0] * y * z + coord[1] * z + coord[2];
                    self.device_ids.get(idx).copied()
                } else {
                    None
                }
            },
        }
    }
}

/// Communication interface for model parallel operations
pub trait Communicator: Send + Sync {
    /// All-gather operation
    fn all_gather(&self, tensor: &Tensor, split_dim: usize) -> Result<Tensor>;

    /// Reduce-scatter operation
    fn reduce_scatter(&self, tensor: &Tensor, split_dim: usize) -> Result<Tensor>;

    /// All-reduce operation
    fn all_reduce(&self, tensor: &mut Tensor) -> Result<()>;

    /// Point-to-point send
    fn send(&self, tensor: &Tensor, dest: usize) -> Result<()>;

    /// Point-to-point receive
    fn recv(&self, shape: &[usize], src: usize) -> Result<Tensor>;

    /// Broadcast from root
    fn broadcast(&self, tensor: &mut Tensor, root: usize) -> Result<()>;
}

/// Create appropriate communicator based on backend
fn create_communicator(backend: &CommunicationBackend) -> Result<Arc<dyn Communicator>> {
    match backend {
        CommunicationBackend::Nccl => {
            #[cfg(feature = "nccl")]
            {
                use super::nccl_communicator::create_nccl_communicator;
                // Default to rank 0, world_size 1, device 0 for single-process case
                // In a real distributed setup, these would come from environment or config
                let rank =
                    std::env::var("RANK").unwrap_or_else(|_| "0".to_string()).parse().unwrap_or(0);
                let world_size = std::env::var("WORLD_SIZE")
                    .unwrap_or_else(|_| "1".to_string())
                    .parse()
                    .unwrap_or(1);
                let device_id = std::env::var("LOCAL_RANK")
                    .unwrap_or_else(|_| "0".to_string())
                    .parse()
                    .unwrap_or(0);

                create_nccl_communicator(rank, world_size, device_id)
            }

            #[cfg(not(feature = "nccl"))]
            return Err(runtime_error(
                "NCCL backend requested but not compiled with nccl feature",
            ));
        },
        CommunicationBackend::Mpi => {
            use super::mpi_communicator::MpiCommunicatorImpl;
            Ok(Arc::new(MpiCommunicatorImpl::new()?))
        },
        CommunicationBackend::Gloo => {
            // Fallback to mock for now
            Ok(Arc::new(MockCommunicator::new()))
        },
        CommunicationBackend::Custom => Ok(Arc::new(MockCommunicator::new())),
    }
}

/// Mock communicator for testing
struct MockCommunicator;

impl MockCommunicator {
    fn new() -> Self {
        Self
    }
}

impl Communicator for MockCommunicator {
    fn all_gather(&self, tensor: &Tensor, _split_dim: usize) -> Result<Tensor> {
        // In mock mode, just return the tensor as-is
        Ok(tensor.clone())
    }

    fn reduce_scatter(&self, tensor: &Tensor, _split_dim: usize) -> Result<Tensor> {
        Ok(tensor.clone())
    }

    fn all_reduce(&self, _tensor: &mut Tensor) -> Result<()> {
        Ok(())
    }

    fn send(&self, _tensor: &Tensor, _dest: usize) -> Result<()> {
        Ok(())
    }

    fn recv(&self, shape: &[usize], _src: usize) -> Result<Tensor> {
        Tensor::zeros(shape)
    }

    fn broadcast(&self, _tensor: &mut Tensor, _root: usize) -> Result<()> {
        Ok(())
    }
}

/// Pipeline parallel schedule for forward/backward passes
#[derive(Debug, Clone)]
pub struct PipelineSchedule {
    /// Number of pipeline stages
    pub num_stages: usize,
    /// Number of microbatches
    pub num_microbatches: usize,
    /// Schedule type
    pub schedule_type: PipelineScheduleType,
}

#[derive(Debug, Clone, Copy)]
pub enum PipelineScheduleType {
    /// Forward then backward (simple but inefficient)
    Sequential,
    /// 1F1B schedule (one forward, one backward)
    OneForwardOneBackward,
    /// Interleaved 1F1B for better efficiency
    InterleavedOneF1B,
}

impl PipelineSchedule {
    /// Create a new pipeline schedule
    pub fn new(
        num_stages: usize,
        num_microbatches: usize,
        schedule_type: PipelineScheduleType,
    ) -> Self {
        Self {
            num_stages,
            num_microbatches,
            schedule_type,
        }
    }

    /// Get the schedule for a specific stage
    pub fn get_stage_schedule(&self, stage_id: usize) -> Vec<PipelineOp> {
        match self.schedule_type {
            PipelineScheduleType::Sequential => self.sequential_schedule(stage_id),
            PipelineScheduleType::OneForwardOneBackward => self.one_f1b_schedule(stage_id),
            PipelineScheduleType::InterleavedOneF1B => self.interleaved_1f1b_schedule(stage_id),
        }
    }

    fn sequential_schedule(&self, stage_id: usize) -> Vec<PipelineOp> {
        let mut ops = Vec::new();

        // All forwards first
        for mb in 0..self.num_microbatches {
            ops.push(PipelineOp::Forward { microbatch_id: mb });
        }

        // Then all backwards
        for mb in (0..self.num_microbatches).rev() {
            ops.push(PipelineOp::Backward { microbatch_id: mb });
        }

        ops
    }

    fn one_f1b_schedule(&self, stage_id: usize) -> Vec<PipelineOp> {
        let mut ops = Vec::new();
        let num_warmup = self.num_stages - stage_id - 1;

        // Warmup phase - only forward
        for mb in 0..num_warmup.min(self.num_microbatches) {
            ops.push(PipelineOp::Forward { microbatch_id: mb });
        }

        // Steady state - 1F1B
        let steady_state_mbs = self.num_microbatches.saturating_sub(num_warmup);
        for i in 0..steady_state_mbs {
            let forward_mb = num_warmup + i;
            let backward_mb = i;

            if forward_mb < self.num_microbatches {
                ops.push(PipelineOp::Forward {
                    microbatch_id: forward_mb,
                });
            }
            ops.push(PipelineOp::Backward {
                microbatch_id: backward_mb,
            });
        }

        // Cooldown phase - only backward
        for mb in steady_state_mbs..self.num_microbatches {
            ops.push(PipelineOp::Backward { microbatch_id: mb });
        }

        ops
    }

    fn interleaved_1f1b_schedule(&self, _stage_id: usize) -> Vec<PipelineOp> {
        // Simplified version - can be optimized further
        self.one_f1b_schedule(_stage_id)
    }
}

#[derive(Debug, Clone)]
pub enum PipelineOp {
    Forward { microbatch_id: usize },
    Backward { microbatch_id: usize },
    SendActivation { to_stage: usize },
    RecvActivation { from_stage: usize },
    SendGradient { to_stage: usize },
    RecvGradient { from_stage: usize },
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_partition() {
        let ctx = ModelParallelContext::new(ModelParallelConfig {
            num_devices: 4,
            device_ids: vec![0, 1, 2, 3],
            comm_backend: CommunicationBackend::Custom, // Use mock backend for tests
            ..Default::default()
        })
        .unwrap();

        let tensor = Tensor::zeros(&[128, 512]).unwrap();
        let distributed = ctx.partition_tensor(&tensor, 0).unwrap();

        // Verify partition metadata is correct
        assert_eq!(distributed.global_shape, vec![128, 512]);
        assert_eq!(distributed.partition.split_dim, 0);
        assert_eq!(distributed.partition.start_idx, 0);
        assert_eq!(distributed.partition.end_idx, 32);
        assert_eq!(distributed.partition.num_partitions, 4);

        // Check local tensor shape after slicing
        let local_shape = distributed.local_shard.shape();
        assert_eq!(local_shape, vec![32, 512]); // First dimension should be sliced to 32
    }

    #[test]
    fn test_device_mesh() {
        let mesh = DeviceMesh::new(&[0, 1, 2, 3], ModelParallelStrategy::Tensor).unwrap();

        assert_eq!(mesh.device_at(&[0, 0]), Some(0));
        assert_eq!(mesh.device_at(&[0, 1]), Some(1));
        assert_eq!(mesh.device_at(&[1, 0]), Some(2));
        assert_eq!(mesh.device_at(&[1, 1]), Some(3));
    }

    #[test]
    fn test_pipeline_schedule() {
        let schedule = PipelineSchedule::new(4, 8, PipelineScheduleType::OneForwardOneBackward);
        let stage0_ops = schedule.get_stage_schedule(0);

        // Stage 0 should have 3 warmup forwards
        let forward_ops: Vec<_> = stage0_ops
            .iter()
            .filter(|op| matches!(op, PipelineOp::Forward { .. }))
            .collect();
        assert!(!forward_ops.is_empty());
    }
}
