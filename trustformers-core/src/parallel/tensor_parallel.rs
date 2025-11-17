//! Tensor Parallelism Operations and Utilities
//!
//! This module provides low-level tensor parallel operations for distributing
//! computations across multiple devices within a single layer.

#![allow(dead_code)] // Distributed parallelism features under development
#![allow(deprecated)] // Using rand legacy API, will migrate to scirs2_core

use super::model_parallel::{DistributedTensor, ModelParallelContext, TensorPartition};
use crate::errors::{invalid_input, Result};
use crate::Tensor;
use std::sync::Arc;

/// Tensor parallel operations
pub struct TensorParallelOps {
    mp_context: Arc<ModelParallelContext>,
}

impl TensorParallelOps {
    pub fn new(mp_context: Arc<ModelParallelContext>) -> Self {
        Self { mp_context }
    }

    /// Copy tensor to all devices (broadcast)
    pub fn broadcast_tensor(&self, tensor: &Tensor, root_rank: usize) -> Result<Tensor> {
        let mut tensor_copy = tensor.clone();
        self.mp_context.communicator.broadcast(&mut tensor_copy, root_rank)?;
        Ok(tensor_copy)
    }

    /// Scatter tensor along a dimension to all devices
    pub fn scatter_tensor(&self, tensor: &Tensor, scatter_dim: usize) -> Result<DistributedTensor> {
        self.mp_context.partition_tensor(tensor, scatter_dim)
    }

    /// Gather distributed tensor from all devices
    pub fn gather_tensor(&self, distributed: &DistributedTensor) -> Result<Tensor> {
        self.mp_context.all_gather(distributed)
    }

    /// All-to-all communication for tensor redistribution
    ///
    /// This implements a sophisticated tensor redistribution pattern where each process
    /// sends different data to every other process and receives different data from
    /// every other process. This is crucial for efficient tensor resharding in
    /// distributed transformers, particularly for attention mechanism parallelism.
    ///
    /// # Algorithm Overview
    /// 1. Split input tensor along split_dim into world_size chunks
    /// 2. Each process i sends chunk j to process j
    /// 3. Each process receives chunks from all other processes
    /// 4. Concatenate received chunks along concat_dim
    ///
    /// # Arguments
    /// * `tensor` - Input tensor to redistribute
    /// * `split_dim` - Dimension to split tensor for sending
    /// * `concat_dim` - Dimension to concatenate received chunks
    ///
    /// # Returns
    /// Redistributed tensor with new sharding pattern
    pub fn all_to_all(
        &self,
        tensor: &Tensor,
        split_dim: usize,
        concat_dim: usize,
    ) -> Result<Tensor> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();
        let shape = tensor.shape();

        // Validate dimensions
        if split_dim >= shape.len() || concat_dim >= shape.len() {
            return Err(invalid_input(format!(
                "Invalid dimensions: split_dim={}, concat_dim={}, tensor_dims={}",
                split_dim,
                concat_dim,
                shape.len()
            )));
        }

        // Early return for single process
        if world_size == 1 {
            return Ok(tensor.clone());
        }

        // Calculate split sizes along split_dim
        let split_size = shape[split_dim];
        if split_size < world_size {
            return Err(invalid_input(format!(
                "Split dimension size ({}) must be at least world_size ({})",
                split_size, world_size
            )));
        }

        let chunk_size = split_size / world_size;
        let remainder = split_size % world_size;

        // Create send buffers by splitting tensor along split_dim
        let mut send_chunks = Vec::with_capacity(world_size);
        let mut current_offset = 0;

        for i in 0..world_size {
            // Handle uneven splits - first 'remainder' processes get chunk_size + 1
            let current_chunk_size = if i < remainder { chunk_size + 1 } else { chunk_size };

            if current_chunk_size > 0 {
                let chunk = tensor.slice(
                    split_dim,
                    current_offset,
                    current_offset + current_chunk_size,
                )?;
                send_chunks.push(chunk);
            } else {
                // Create empty tensor for processes that don't receive data
                let mut empty_shape = shape.to_vec();
                empty_shape[split_dim] = 0;
                send_chunks.push(Tensor::zeros(&empty_shape)?);
            }

            current_offset += current_chunk_size;
        }

        // Perform all-to-all exchange using point-to-point communications
        // This implements a decentralized algorithm to avoid bottlenecks
        let mut receive_chunks = Vec::with_capacity(world_size);

        // Initialize receive chunks with proper sizing
        for i in 0..world_size {
            let sender_chunk_size = if i < remainder { chunk_size + 1 } else { chunk_size };

            if sender_chunk_size > 0 {
                let mut recv_shape = shape.to_vec();
                recv_shape[split_dim] = sender_chunk_size;
                receive_chunks.push(Tensor::zeros(&recv_shape)?);
            } else {
                let mut empty_shape = shape.to_vec();
                empty_shape[split_dim] = 0;
                receive_chunks.push(Tensor::zeros(&empty_shape)?);
            }
        }

        // Execute the all-to-all communication pattern
        // Use a ring-based algorithm for better scalability
        for phase in 0..world_size {
            let send_to = (rank + phase) % world_size;
            let recv_from = (rank + world_size - phase) % world_size;

            // In practice, this would use MPI_Isend/MPI_Irecv or NCCL send/recv
            // For simulation, we perform the data exchange conceptually
            if phase == 0 {
                // Self-exchange (no communication needed)
                receive_chunks[rank] = send_chunks[rank].clone();
            } else {
                // Simulate non-blocking send/recv
                // In real implementation:
                // - MPI_Isend(send_chunks[send_to], send_to, tag, comm, &send_request)
                // - MPI_Irecv(receive_chunks[recv_from], recv_from, tag, comm, &recv_request)
                // - MPI_Wait(&recv_request, MPI_STATUS_IGNORE)

                // For now, simulate the exchange using communicator
                let send_data = send_chunks[send_to].clone();
                receive_chunks[recv_from] =
                    self.simulate_point_to_point_exchange(&send_data, recv_from)?;
            }
        }

        // Concatenate received chunks along concat_dim
        // Handle the case where some chunks might be empty
        let non_empty_chunks: Vec<_> = receive_chunks
            .into_iter()
            .filter(|chunk| {
                let chunk_shape = chunk.shape();
                chunk_shape.iter().all(|&dim| dim > 0)
            })
            .collect();

        if non_empty_chunks.is_empty() {
            // Return zero tensor with appropriate shape
            let mut result_shape = shape.to_vec();
            result_shape[concat_dim] = 0;
            return Tensor::zeros(&result_shape);
        }

        // Concatenate along concat_dim
        let result = if non_empty_chunks.len() == 1 {
            non_empty_chunks.into_iter().next().unwrap()
        } else {
            self.concatenate_tensors(&non_empty_chunks, concat_dim)?
        };

        Ok(result)
    }

    /// Simulate point-to-point data exchange (placeholder for actual MPI/NCCL implementation)
    fn simulate_point_to_point_exchange(
        &self,
        data: &Tensor,
        _target_rank: usize,
    ) -> Result<Tensor> {
        // In a real implementation, this would perform actual network communication
        // For now, return the data as-is to simulate successful exchange
        Ok(data.clone())
    }

    /// Concatenate tensors along specified dimension with proper error handling
    fn concatenate_tensors(&self, tensors: &[Tensor], concat_dim: usize) -> Result<Tensor> {
        if tensors.is_empty() {
            return Err(invalid_input(
                "Cannot concatenate empty tensor list".to_string(),
            ));
        }

        if tensors.len() == 1 {
            return Ok(tensors[0].clone());
        }

        // Validate that all tensors have compatible shapes
        let first_shape = tensors[0].shape();
        for (i, tensor) in tensors.iter().enumerate().skip(1) {
            let shape = tensor.shape();
            if shape.len() != first_shape.len() {
                return Err(invalid_input(format!(
                    "Tensor {} has {} dimensions, expected {}",
                    i,
                    shape.len(),
                    first_shape.len()
                )));
            }
            for (dim_idx, (&dim_size, &expected_size)) in
                shape.iter().zip(first_shape.iter()).enumerate()
            {
                if dim_idx != concat_dim && dim_size != expected_size {
                    return Err(invalid_input(format!(
                        "Tensor {} has size {} in dimension {}, expected {}",
                        i, dim_size, dim_idx, expected_size
                    )));
                }
            }
        }

        // Perform concatenation using tensor operations
        let result = crate::tensor::Tensor::concat(tensors, concat_dim)?;

        Ok(result)
    }

    /// Split tensor for column-wise parallelism
    pub fn split_column_wise(&self, tensor: &Tensor) -> Result<Tensor> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        let shape = tensor.shape();
        let last_dim = shape.len() - 1;
        let columns = shape[last_dim];

        let columns_per_rank = (columns + world_size - 1) / world_size;
        let start_col = rank * columns_per_rank;
        let end_col = ((rank + 1) * columns_per_rank).min(columns);

        // Slice the tensor along the last dimension (columns)
        tensor.slice(last_dim, start_col, end_col)
    }

    /// Split tensor for row-wise parallelism
    pub fn split_row_wise(&self, tensor: &Tensor) -> Result<Tensor> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        let shape = tensor.shape();
        let second_last_dim = shape.len() - 2;
        let rows = shape[second_last_dim];

        let rows_per_rank = (rows + world_size - 1) / world_size;
        let start_row = rank * rows_per_rank;
        let end_row = ((rank + 1) * rows_per_rank).min(rows);

        // Slice the tensor along the second-to-last dimension (rows)
        tensor.slice(second_last_dim, start_row, end_row)
    }

    /// Reduce-scatter operation for summing and scattering
    pub fn reduce_scatter_sum(&self, tensor: &Tensor, scatter_dim: usize) -> Result<Tensor> {
        self.mp_context.reduce_scatter(tensor, scatter_dim)
    }

    /// All-reduce sum across all devices
    pub fn all_reduce_sum(&self, tensor: &mut Tensor) -> Result<()> {
        self.mp_context.all_reduce(tensor)
    }

    /// Create a distributed tensor with advanced sharding strategies
    pub fn create_distributed_tensor(
        &self,
        global_shape: &[usize],
        sharding_strategy: ShardingStrategy,
    ) -> Result<DistributedTensor> {
        let world_size = self.mp_context.world_size();
        let rank = self.mp_context.rank();

        match sharding_strategy {
            ShardingStrategy::RowWise(dim) => {
                if dim >= global_shape.len() {
                    return Err(invalid_input(format!(
                        "Sharding dimension {} out of bounds for shape {:?}",
                        dim, global_shape
                    )));
                }

                let dim_size = global_shape[dim];
                let chunk_size = (dim_size + world_size - 1) / world_size;
                let start_idx = rank * chunk_size;
                let end_idx = (start_idx + chunk_size).min(dim_size);

                let mut local_shape = global_shape.to_vec();
                local_shape[dim] = end_idx - start_idx;

                let local_shard = Tensor::zeros(&local_shape)?;

                Ok(DistributedTensor {
                    local_shard,
                    global_shape: global_shape.to_vec(),
                    partition: TensorPartition {
                        split_dim: dim,
                        start_idx,
                        end_idx,
                        num_partitions: world_size,
                        partition_rank: rank,
                    },
                    device_id: rank,
                })
            },
            ShardingStrategy::ColumnWise(dim) => {
                // Similar to row-wise but for column dimension
                if dim >= global_shape.len() {
                    return Err(invalid_input(format!(
                        "Sharding dimension {} out of bounds for shape {:?}",
                        dim, global_shape
                    )));
                }

                let dim_size = global_shape[dim];
                let chunk_size = (dim_size + world_size - 1) / world_size;
                let start_idx = rank * chunk_size;
                let end_idx = (start_idx + chunk_size).min(dim_size);

                let mut local_shape = global_shape.to_vec();
                local_shape[dim] = end_idx - start_idx;

                let local_shard = Tensor::zeros(&local_shape)?;

                Ok(DistributedTensor {
                    local_shard,
                    global_shape: global_shape.to_vec(),
                    partition: TensorPartition {
                        split_dim: dim,
                        start_idx,
                        end_idx,
                        num_partitions: world_size,
                        partition_rank: rank,
                    },
                    device_id: rank,
                })
            },
            ShardingStrategy::Replicated => {
                // Full replication across all devices
                let local_shard = Tensor::zeros(global_shape)?;

                Ok(DistributedTensor {
                    local_shard,
                    global_shape: global_shape.to_vec(),
                    partition: TensorPartition {
                        split_dim: 0,
                        start_idx: 0,
                        end_idx: global_shape[0],
                        num_partitions: 1, // No actual partitioning
                        partition_rank: rank,
                    },
                    device_id: rank,
                })
            },
            ShardingStrategy::Block2D { row_dim, col_dim } => {
                // 2D block sharding for matrices
                if row_dim >= global_shape.len() || col_dim >= global_shape.len() {
                    return Err(invalid_input(format!(
                        "Sharding dimensions ({}, {}) out of bounds for shape {:?}",
                        row_dim, col_dim, global_shape
                    )));
                }

                // Simple 2D grid layout
                let grid_size = (world_size as f32).sqrt() as usize;
                let row_rank = rank / grid_size;
                let col_rank = rank % grid_size;

                let row_size = global_shape[row_dim];
                let col_size = global_shape[col_dim];
                let row_chunk = (row_size + grid_size - 1) / grid_size;
                let col_chunk = (col_size + grid_size - 1) / grid_size;

                let row_start = row_rank * row_chunk;
                let row_end = (row_start + row_chunk).min(row_size);
                let col_start = col_rank * col_chunk;
                let col_end = (col_start + col_chunk).min(col_size);

                let mut local_shape = global_shape.to_vec();
                local_shape[row_dim] = row_end - row_start;
                local_shape[col_dim] = col_end - col_start;

                let local_shard = Tensor::zeros(&local_shape)?;

                Ok(DistributedTensor {
                    local_shard,
                    global_shape: global_shape.to_vec(),
                    partition: TensorPartition {
                        split_dim: row_dim,
                        start_idx: row_start,
                        end_idx: row_end,
                        num_partitions: world_size,
                        partition_rank: rank,
                    },
                    device_id: rank,
                })
            },
        }
    }

    /// Synchronize gradients across all devices with optional gradient clipping
    pub fn sync_gradients(&self, tensors: &mut [Tensor], clip_norm: Option<f32>) -> Result<()> {
        // Apply gradient clipping if specified
        if let Some(max_norm) = clip_norm {
            let total_norm = Self::compute_gradient_norm(tensors)?;
            if total_norm > max_norm {
                let clip_coef = max_norm / total_norm;
                for tensor in tensors.iter_mut() {
                    let data = tensor.data_mut()?;
                    for value in data.iter_mut() {
                        *value *= clip_coef;
                    }
                }
            }
        }

        // All-reduce each tensor
        for tensor in tensors.iter_mut() {
            self.all_reduce_sum(tensor)?;

            // Average gradients
            let world_size = self.mp_context.world_size() as f32;
            let data = tensor.data_mut()?;
            for value in data.iter_mut() {
                *value /= world_size;
            }
        }

        Ok(())
    }

    /// Compute gradient norm across multiple tensors
    fn compute_gradient_norm(tensors: &[Tensor]) -> Result<f32> {
        let mut total_norm_sq = 0.0f32;

        for tensor in tensors {
            let data = tensor.data()?;
            for value in data.iter() {
                total_norm_sq += value * value;
            }
        }

        Ok(total_norm_sq.sqrt())
    }
}

/// Utilities for initializing weights in tensor parallel settings
pub struct TensorParallelInit;

impl TensorParallelInit {
    /// Initialize column-parallel weight matrix
    pub fn column_parallel_weight(
        in_features: usize,
        out_features: usize,
        world_size: usize,
        rank: usize,
        init_method: InitMethod,
    ) -> Result<Tensor> {
        let out_features_per_rank = (out_features + world_size - 1) / world_size;
        let local_out_features = if rank == world_size - 1 {
            out_features - rank * out_features_per_rank
        } else {
            out_features_per_rank
        };

        Self::init_weight(&[in_features, local_out_features], init_method)
    }

    /// Initialize row-parallel weight matrix
    pub fn row_parallel_weight(
        in_features: usize,
        out_features: usize,
        world_size: usize,
        rank: usize,
        init_method: InitMethod,
    ) -> Result<Tensor> {
        let in_features_per_rank = (in_features + world_size - 1) / world_size;
        let local_in_features = if rank == world_size - 1 {
            in_features - rank * in_features_per_rank
        } else {
            in_features_per_rank
        };

        Self::init_weight(&[local_in_features, out_features], init_method)
    }

    fn init_weight(shape: &[usize], method: InitMethod) -> Result<Tensor> {
        use scirs2_core::random::*;

        let mut rng = thread_rng();
        let size = shape.iter().product();

        match method {
            InitMethod::Normal { mean, std } => {
                let normal = Normal::new(mean as f64, std as f64)
                    .map_err(|e| invalid_input(format!("Normal distribution parameters: {}", e)))?;
                let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
                Ok(Tensor::from_data(data, shape)?)
            },
            InitMethod::Uniform { low, high } => {
                if low >= high {
                    return Err(invalid_input(format!(
                        "Uniform distribution bounds: low ({}) must be less than high ({})",
                        low, high
                    )));
                }
                let uniform = UniformDist::new(low as f64, high as f64).map_err(|e| {
                    invalid_input(format!("Uniform distribution parameters: {}", e))
                })?;
                let data: Vec<f32> = (0..size).map(|_| uniform.sample(&mut rng) as f32).collect();
                Ok(Tensor::from_data(data, shape)?)
            },
            InitMethod::Xavier => {
                let fan_in = shape[0] as f32;
                let fan_out = shape[1] as f32;
                let std = (2.0 / (fan_in + fan_out)).sqrt();
                let normal = Normal::new(0.0, std as f64).map_err(|e| {
                    invalid_input(format!("Xavier initialization parameters: {}", e))
                })?;
                let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
                Ok(Tensor::from_data(data, shape)?)
            },
            InitMethod::Kaiming => {
                let fan_in = shape[0] as f32;
                let std = (2.0 / fan_in).sqrt();
                let normal = Normal::new(0.0, std as f64).map_err(|e| {
                    invalid_input(format!("Kaiming initialization parameters: {}", e))
                })?;
                let data: Vec<f32> = (0..size).map(|_| normal.sample(&mut rng) as f32).collect();
                Ok(Tensor::from_data(data, shape)?)
            },
        }
    }
}

/// Different strategies for sharding tensors across devices
#[derive(Debug, Clone, Copy)]
pub enum ShardingStrategy {
    /// Shard along rows (specified dimension)
    RowWise(usize),
    /// Shard along columns (specified dimension)
    ColumnWise(usize),
    /// Replicate tensor on all devices
    Replicated,
    /// 2D block sharding for matrices
    Block2D { row_dim: usize, col_dim: usize },
}

#[derive(Debug, Clone, Copy)]
pub enum InitMethod {
    Normal { mean: f32, std: f32 },
    Uniform { low: f32, high: f32 },
    Xavier,
    Kaiming,
}

/// Helper for computing tensor parallel shapes
pub struct TensorParallelShapes;

impl TensorParallelShapes {
    /// Get local tensor shape for column-parallel distribution
    pub fn column_parallel_shape(
        global_shape: &[usize],
        world_size: usize,
        rank: usize,
    ) -> Vec<usize> {
        let mut local_shape = global_shape.to_vec();
        let last_dim = local_shape.len() - 1;
        let global_columns = global_shape[last_dim];

        let columns_per_rank = (global_columns + world_size - 1) / world_size;
        local_shape[last_dim] = if rank == world_size - 1 {
            global_columns - rank * columns_per_rank
        } else {
            columns_per_rank
        };

        local_shape
    }

    /// Get local tensor shape for row-parallel distribution
    pub fn row_parallel_shape(
        global_shape: &[usize],
        world_size: usize,
        rank: usize,
    ) -> Vec<usize> {
        let mut local_shape = global_shape.to_vec();
        let second_last_dim = local_shape.len() - 2;
        let global_rows = global_shape[second_last_dim];

        let rows_per_rank = (global_rows + world_size - 1) / world_size;
        local_shape[second_last_dim] = if rank == world_size - 1 {
            global_rows - rank * rows_per_rank
        } else {
            rows_per_rank
        };

        local_shape
    }

    /// Calculate the split sizes for each rank
    pub fn split_sizes(total_size: usize, world_size: usize) -> Vec<usize> {
        let base_size = total_size / world_size;
        let remainder = total_size % world_size;

        (0..world_size).map(|i| base_size + if i < remainder { 1 } else { 0 }).collect()
    }
}

/// Async tensor parallel operations for overlapping computation and communication
pub struct AsyncTensorParallel {
    mp_context: Arc<ModelParallelContext>,
}

impl AsyncTensorParallel {
    pub fn new(mp_context: Arc<ModelParallelContext>) -> Self {
        Self { mp_context }
    }

    /// Start async all-reduce (returns handle)
    pub async fn all_reduce_async(&self, tensor: Tensor) -> Result<AllReduceHandle> {
        // In practice, this would start NCCL async operation
        Ok(AllReduceHandle {
            tensor,
            completed: false,
        })
    }

    /// Start async all-gather
    pub async fn all_gather_async(
        &self,
        distributed: DistributedTensor,
    ) -> Result<AllGatherHandle> {
        Ok(AllGatherHandle {
            distributed,
            completed: false,
        })
    }
}

/// Handle for async all-reduce operation
pub struct AllReduceHandle {
    tensor: Tensor,
    completed: bool,
}

impl AllReduceHandle {
    /// Wait for completion and get result
    pub async fn wait(mut self) -> Result<Tensor> {
        // In practice, wait for NCCL operation
        self.completed = true;
        Ok(self.tensor)
    }

    /// Check if operation is complete (non-blocking)
    pub fn is_complete(&self) -> bool {
        self.completed
    }
}

/// Handle for async all-gather operation
pub struct AllGatherHandle {
    distributed: DistributedTensor,
    completed: bool,
}

impl AllGatherHandle {
    /// Wait for completion and get result
    pub async fn wait(mut self) -> Result<Tensor> {
        self.completed = true;
        // In practice, would gather from all ranks
        Ok(self.distributed.local_shard.clone())
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn test_tensor_parallel_shapes() {
        let global_shape = vec![32, 64, 768];
        let world_size = 4;

        // Test column parallel
        let col_shape = TensorParallelShapes::column_parallel_shape(&global_shape, world_size, 0);
        assert_eq!(col_shape, vec![32, 64, 192]); // 768 / 4 = 192

        // Test row parallel
        let row_shape = TensorParallelShapes::row_parallel_shape(&global_shape, world_size, 0);
        assert_eq!(row_shape, vec![32, 16, 768]); // 64 / 4 = 16
    }

    #[test]
    fn test_split_sizes() {
        assert_eq!(
            TensorParallelShapes::split_sizes(100, 4),
            vec![25, 25, 25, 25]
        );

        assert_eq!(
            TensorParallelShapes::split_sizes(101, 4),
            vec![26, 25, 25, 25]
        );
    }

    #[test]
    fn test_weight_initialization() {
        let weight =
            TensorParallelInit::column_parallel_weight(512, 2048, 4, 0, InitMethod::Xavier)
                .unwrap();

        assert_eq!(weight.shape(), &[512, 512]); // 2048 / 4 = 512
    }
}
