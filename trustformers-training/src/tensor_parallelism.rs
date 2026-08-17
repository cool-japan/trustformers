use crate::distributed::ProcessGroup;
use anyhow::{anyhow, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};
use trustformers_core::tensor::Tensor;

/// Tensor Parallelism Configuration
///
/// Tensor parallelism distributes individual tensors (weights, activations) across multiple devices,
/// enabling the training of models where individual layers are too large to fit on a single device.
/// This is particularly effective for large linear layers and attention mechanisms.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorParallelismConfig {
    /// Number of devices for tensor parallelism
    pub tensor_parallel_size: usize,
    /// Tensor partitioning strategy
    pub partitioning_strategy: TensorPartitioningStrategy,
    /// Whether to use column parallelism for linear layers
    pub column_parallel: bool,
    /// Whether to use row parallelism for linear layers
    pub row_parallel: bool,
    /// Communication pattern for tensor operations
    pub communication_pattern: TensorCommunicationPattern,
    /// Whether to use asynchronous communication
    pub async_communication: bool,
    /// Communication fusion threshold (operations below this size are fused)
    pub fusion_threshold_bytes: usize,
    /// Whether to use gradient accumulation across tensor chunks
    pub gradient_accumulation: bool,
    /// Memory optimization level for tensor parallelism
    pub memory_optimization: TensorMemoryOptimization,
    /// Whether to use mixed precision for tensor operations
    pub mixed_precision: bool,
}

impl Default for TensorParallelismConfig {
    fn default() -> Self {
        Self {
            tensor_parallel_size: 1,
            partitioning_strategy: TensorPartitioningStrategy::ColumnWise,
            column_parallel: true,
            row_parallel: true,
            communication_pattern: TensorCommunicationPattern::AllReduce,
            async_communication: true,
            fusion_threshold_bytes: 1024 * 1024, // 1MB
            gradient_accumulation: true,
            memory_optimization: TensorMemoryOptimization::Medium,
            mixed_precision: false,
        }
    }
}

/// Tensor partitioning strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorPartitioningStrategy {
    /// Split tensors column-wise
    ColumnWise,
    /// Split tensors row-wise
    RowWise,
    /// Split tensors along batch dimension
    BatchWise,
    /// Split tensors along sequence dimension
    SequenceWise,
    /// Dynamic partitioning based on tensor shape
    Dynamic,
    /// Block-wise partitioning for 2D tensors
    BlockWise,
    /// Custom partitioning strategy
    Custom,
}

/// Communication patterns for tensor parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorCommunicationPattern {
    /// All-reduce for gradient synchronization
    AllReduce,
    /// All-gather for activation collection
    AllGather,
    /// Reduce-scatter for distributed computation
    ReduceScatter,
    /// Point-to-point for custom patterns
    PointToPoint,
    /// Hierarchical communication
    Hierarchical,
}

/// Memory optimization strategies for tensor parallelism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TensorMemoryOptimization {
    None,
    Low,
    Medium,
    High,
    Extreme,
}

/// Tensor partition information
#[derive(Debug, Clone)]
pub struct TensorPartition {
    /// Partition ID
    pub partition_id: usize,
    /// Device rank where this partition is stored
    pub device_rank: usize,
    /// Tensor name/identifier
    pub tensor_name: String,
    /// Partition shape
    pub shape: Vec<usize>,
    /// Offset in the original tensor
    pub offset: Vec<usize>,
    /// Whether this partition needs communication
    pub needs_communication: bool,
    /// Communication dependencies (other partitions needed for computation)
    pub dependencies: Vec<usize>,
}

/// Tensor operation for distributed computation
#[derive(Debug, Clone)]
pub struct TensorOperation {
    /// Operation ID
    pub operation_id: usize,
    /// Operation type
    pub operation_type: TensorOperationType,
    /// Input tensor partitions
    pub input_partitions: Vec<usize>,
    /// Output tensor partitions
    pub output_partitions: Vec<usize>,
    /// Communication requirements
    pub communication_requirements: Vec<CommunicationRequirement>,
    /// Memory requirements in bytes
    pub memory_requirements: usize,
}

/// Types of tensor operations
#[derive(Debug, Clone, Hash, Eq, PartialEq)]
pub enum TensorOperationType {
    MatMul,
    Add,
    Attention,
    Linear,
    Embedding,
    LayerNorm,
    Activation,
    Custom(String),
}

/// Communication requirement for tensor operations
#[derive(Debug, Clone)]
pub struct CommunicationRequirement {
    /// Source partition ID
    pub source_partition: usize,
    /// Target partition ID
    pub target_partition: usize,
    /// Communication type
    pub communication_type: TensorCommunicationPattern,
    /// Data size in bytes
    pub data_size: usize,
}

/// Tensor parallelism coordinator
pub struct TensorParallelism {
    config: TensorParallelismConfig,
    global_rank: usize,
    world_size: usize,

    // Tensor partition management
    tensor_partitions: HashMap<String, Vec<TensorPartition>>,
    local_partitions: HashMap<String, Vec<usize>>, // tensor_name -> local partition IDs

    // Process groups for tensor parallelism
    tensor_group: Arc<dyn ProcessGroup>,
    column_group: Option<Arc<dyn ProcessGroup>>,
    row_group: Option<Arc<dyn ProcessGroup>>,

    // Operation scheduling
    operation_scheduler: Arc<RwLock<OperationScheduler>>,

    // Communication optimization
    communication_optimizer: Arc<Mutex<CommunicationOptimizer>>,

    // Statistics tracking
    statistics: Arc<Mutex<TensorParallelismStats>>,

    // Tensor data held by this rank, keyed by partition id. The partition table
    // above records shapes and ownership; the actual payloads live here.
    partition_data: Arc<RwLock<HashMap<usize, Tensor>>>,
}

/// Deterministic message tag for a point-to-point partition transfer.
///
/// Both endpoints compute the same value from the partition pair, so the
/// transfer does not depend on either side's position in a call sequence.
fn point_to_point_tag(source_partition: usize, target_partition: usize) -> u64 {
    (source_partition as u64) << 20 | (target_partition as u64 & 0xF_FFFF)
}

/// Deterministic message tag for one phase of a hierarchical reduction.
fn hierarchical_tag(partition: usize, phase: u64, member: usize) -> u64 {
    1 << 40 | (partition as u64) << 20 | phase << 18 | (member as u64 & 0x3_FFFF)
}

/// Operation scheduler for tensor operations
#[derive(Debug, Default)]
struct OperationScheduler {
    pending_operations: Vec<TensorOperation>,
    running_operations: Vec<TensorOperation>,
    completed_operations: Vec<TensorOperation>,
    operation_graph: HashMap<usize, Vec<usize>>, // operation_id -> dependencies
}

/// Communication optimizer for reducing communication overhead
#[derive(Debug, Default)]
struct CommunicationOptimizer {
    fusion_buffer: Vec<CommunicationRequirement>,
    communication_schedule: Vec<Vec<CommunicationRequirement>>, // Batched communications
    async_handles: Vec<AsyncCommHandle>,
    bandwidth_usage: f32,
    latency_estimates: HashMap<TensorCommunicationPattern, Duration>,
}

/// Async communication handle (placeholder)
#[derive(Debug)]
struct AsyncCommHandle {
    id: usize,
    completion_time: Instant,
}

/// Tensor parallelism statistics
#[derive(Debug, Default)]
struct TensorParallelismStats {
    total_communication_time: Duration,
    computation_time: Duration,
    memory_usage_per_device: HashMap<usize, u64>,
    communication_volume: u64,
    operation_count: HashMap<TensorOperationType, usize>,
    efficiency_score: f32,
}

impl TensorParallelism {
    /// Create a new tensor parallelism coordinator
    pub fn new(
        config: TensorParallelismConfig,
        global_rank: usize,
        world_size: usize,
        tensor_group: Arc<dyn ProcessGroup>,
    ) -> Result<Self> {
        // Validate configuration
        if config.tensor_parallel_size > world_size {
            return Err(anyhow!(
                "Tensor parallel size ({}) cannot exceed world size ({})",
                config.tensor_parallel_size,
                world_size
            ));
        }

        if !world_size.is_multiple_of(config.tensor_parallel_size) {
            return Err(anyhow!(
                "World size ({}) must be divisible by tensor parallel size ({})",
                world_size,
                config.tensor_parallel_size
            ));
        }

        // Initialize column and row process groups for different parallelism types
        let column_group = if config.column_parallel {
            // In practice, would create specific process groups for column parallelism
            Some(tensor_group.clone())
        } else {
            None
        };

        let row_group = if config.row_parallel {
            // In practice, would create specific process groups for row parallelism
            Some(tensor_group.clone())
        } else {
            None
        };

        Ok(Self {
            config,
            global_rank,
            world_size,
            tensor_partitions: HashMap::new(),
            local_partitions: HashMap::new(),
            tensor_group,
            column_group,
            row_group,
            operation_scheduler: Arc::new(RwLock::new(OperationScheduler::default())),
            communication_optimizer: Arc::new(Mutex::new(CommunicationOptimizer::default())),
            statistics: Arc::new(Mutex::new(TensorParallelismStats::default())),
            partition_data: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Partition a tensor across devices
    pub fn partition_tensor(
        &mut self,
        tensor_name: &str,
        tensor_shape: &[usize],
        strategy: Option<TensorPartitioningStrategy>,
    ) -> Result<Vec<TensorPartition>> {
        let partitioning_strategy = strategy.unwrap_or(self.config.partitioning_strategy.clone());

        let partitions = match partitioning_strategy {
            TensorPartitioningStrategy::ColumnWise => {
                self.partition_column_wise(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::RowWise => {
                self.partition_row_wise(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::BatchWise => {
                self.partition_batch_wise(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::SequenceWise => {
                self.partition_sequence_wise(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::Dynamic => {
                self.partition_dynamic(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::BlockWise => {
                self.partition_block_wise(tensor_name, tensor_shape)?
            },
            TensorPartitioningStrategy::Custom => {
                self.partition_custom(tensor_name, tensor_shape)?
            },
        };

        // Update local partition tracking
        let local_partition_ids: Vec<usize> = partitions
            .iter()
            .enumerate()
            .filter(|(_, partition)| partition.device_rank == self.global_rank)
            .map(|(i, _)| i)
            .collect();

        self.tensor_partitions.insert(tensor_name.to_string(), partitions.clone());
        self.local_partitions.insert(tensor_name.to_string(), local_partition_ids);

        Ok(partitions)
    }

    /// Column-wise tensor partitioning
    fn partition_column_wise(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        if tensor_shape.len() < 2 {
            return Err(anyhow!(
                "Column-wise partitioning requires at least 2D tensor"
            ));
        }

        let num_partitions = self.config.tensor_parallel_size;
        let columns = tensor_shape[tensor_shape.len() - 1];
        let columns_per_partition = columns.div_ceil(num_partitions);

        let mut partitions = Vec::new();

        for partition_id in 0..num_partitions {
            let start_col = partition_id * columns_per_partition;
            let end_col = std::cmp::min(start_col + columns_per_partition, columns);

            if start_col < columns {
                let mut partition_shape = tensor_shape.to_vec();
                partition_shape[tensor_shape.len() - 1] = end_col - start_col;

                let mut offset = vec![0; tensor_shape.len()];
                offset[tensor_shape.len() - 1] = start_col;

                let partition = TensorPartition {
                    partition_id,
                    device_rank: partition_id % self.world_size,
                    tensor_name: tensor_name.to_string(),
                    shape: partition_shape,
                    offset,
                    needs_communication: true,
                    dependencies: Vec::new(),
                };

                partitions.push(partition);
            }
        }

        Ok(partitions)
    }

    /// Row-wise tensor partitioning
    fn partition_row_wise(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        if tensor_shape.len() < 2 {
            return Err(anyhow!("Row-wise partitioning requires at least 2D tensor"));
        }

        let num_partitions = self.config.tensor_parallel_size;
        let rows = tensor_shape[tensor_shape.len() - 2];
        let rows_per_partition = rows.div_ceil(num_partitions);

        let mut partitions = Vec::new();

        for partition_id in 0..num_partitions {
            let start_row = partition_id * rows_per_partition;
            let end_row = std::cmp::min(start_row + rows_per_partition, rows);

            if start_row < rows {
                let mut partition_shape = tensor_shape.to_vec();
                partition_shape[tensor_shape.len() - 2] = end_row - start_row;

                let mut offset = vec![0; tensor_shape.len()];
                offset[tensor_shape.len() - 2] = start_row;

                let partition = TensorPartition {
                    partition_id,
                    device_rank: partition_id % self.world_size,
                    tensor_name: tensor_name.to_string(),
                    shape: partition_shape,
                    offset,
                    needs_communication: true,
                    dependencies: Vec::new(),
                };

                partitions.push(partition);
            }
        }

        Ok(partitions)
    }

    /// Batch-wise tensor partitioning
    fn partition_batch_wise(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        if tensor_shape.is_empty() {
            return Err(anyhow!(
                "Batch-wise partitioning requires at least 1D tensor"
            ));
        }

        let num_partitions = self.config.tensor_parallel_size;
        let batch_size = tensor_shape[0];
        let batch_per_partition = batch_size.div_ceil(num_partitions);

        let mut partitions = Vec::new();

        for partition_id in 0..num_partitions {
            let start_batch = partition_id * batch_per_partition;
            let end_batch = std::cmp::min(start_batch + batch_per_partition, batch_size);

            if start_batch < batch_size {
                let mut partition_shape = tensor_shape.to_vec();
                partition_shape[0] = end_batch - start_batch;

                let mut offset = vec![0; tensor_shape.len()];
                offset[0] = start_batch;

                let partition = TensorPartition {
                    partition_id,
                    device_rank: partition_id % self.world_size,
                    tensor_name: tensor_name.to_string(),
                    shape: partition_shape,
                    offset,
                    needs_communication: false, // Batch parallelism doesn't need communication for most ops
                    dependencies: Vec::new(),
                };

                partitions.push(partition);
            }
        }

        Ok(partitions)
    }

    /// Sequence-wise tensor partitioning
    fn partition_sequence_wise(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        if tensor_shape.len() < 2 {
            return Err(anyhow!(
                "Sequence-wise partitioning requires at least 2D tensor"
            ));
        }

        // Assume sequence dimension is the second dimension
        let num_partitions = self.config.tensor_parallel_size;
        let sequence_length = tensor_shape[1];
        let seq_per_partition = sequence_length.div_ceil(num_partitions);

        let mut partitions = Vec::new();

        for partition_id in 0..num_partitions {
            let start_seq = partition_id * seq_per_partition;
            let end_seq = std::cmp::min(start_seq + seq_per_partition, sequence_length);

            if start_seq < sequence_length {
                let mut partition_shape = tensor_shape.to_vec();
                partition_shape[1] = end_seq - start_seq;

                let mut offset = vec![0; tensor_shape.len()];
                offset[1] = start_seq;

                let partition = TensorPartition {
                    partition_id,
                    device_rank: partition_id % self.world_size,
                    tensor_name: tensor_name.to_string(),
                    shape: partition_shape,
                    offset,
                    needs_communication: true,
                    dependencies: Vec::new(),
                };

                partitions.push(partition);
            }
        }

        Ok(partitions)
    }

    /// Dynamic tensor partitioning based on tensor properties
    fn partition_dynamic(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        // Choose partitioning strategy based on tensor shape
        if tensor_shape.len() >= 2 {
            let last_dim = tensor_shape[tensor_shape.len() - 1];
            let second_last_dim = tensor_shape[tensor_shape.len() - 2];

            if last_dim > second_last_dim {
                // More columns than rows, use column-wise
                self.partition_column_wise(tensor_name, tensor_shape)
            } else {
                // More rows than columns, use row-wise
                self.partition_row_wise(tensor_name, tensor_shape)
            }
        } else {
            // 1D tensor, use batch-wise
            self.partition_batch_wise(tensor_name, tensor_shape)
        }
    }

    /// Block-wise tensor partitioning for 2D tensors
    fn partition_block_wise(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        if tensor_shape.len() != 2 {
            return Err(anyhow!("Block-wise partitioning only supports 2D tensors"));
        }

        let num_partitions = self.config.tensor_parallel_size;
        let grid_size = (num_partitions as f64).sqrt().ceil() as usize;

        if grid_size * grid_size != num_partitions {
            // Fallback to column-wise if not a perfect square
            return self.partition_column_wise(tensor_name, tensor_shape);
        }

        let rows = tensor_shape[0];
        let cols = tensor_shape[1];
        let rows_per_block = rows.div_ceil(grid_size);
        let cols_per_block = cols.div_ceil(grid_size);

        let mut partitions = Vec::new();
        let mut partition_id = 0;

        for row_block in 0..grid_size {
            for col_block in 0..grid_size {
                let start_row = row_block * rows_per_block;
                let end_row = std::cmp::min(start_row + rows_per_block, rows);
                let start_col = col_block * cols_per_block;
                let end_col = std::cmp::min(start_col + cols_per_block, cols);

                if start_row < rows && start_col < cols {
                    let partition_shape = vec![end_row - start_row, end_col - start_col];
                    let offset = vec![start_row, start_col];

                    let partition = TensorPartition {
                        partition_id,
                        device_rank: partition_id % self.world_size,
                        tensor_name: tensor_name.to_string(),
                        shape: partition_shape,
                        offset,
                        needs_communication: true,
                        dependencies: Vec::new(),
                    };

                    partitions.push(partition);
                    partition_id += 1;
                }
            }
        }

        Ok(partitions)
    }

    /// Custom tensor partitioning (placeholder)
    fn partition_custom(
        &self,
        tensor_name: &str,
        tensor_shape: &[usize],
    ) -> Result<Vec<TensorPartition>> {
        // For now, fallback to column-wise
        self.partition_column_wise(tensor_name, tensor_shape)
    }

    /// Execute a distributed tensor operation
    pub fn execute_operation(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let start_time = Instant::now();

        // Execute the operation based on its type
        let outputs = match &operation.operation_type {
            TensorOperationType::MatMul => self.execute_matmul(operation, inputs)?,
            TensorOperationType::Add => self.execute_add(operation, inputs)?,
            TensorOperationType::Attention => self.execute_attention(operation, inputs)?,
            TensorOperationType::Linear => self.execute_linear(operation, inputs)?,
            TensorOperationType::Embedding => self.execute_embedding(operation, inputs)?,
            TensorOperationType::LayerNorm => self.execute_layernorm(operation, inputs)?,
            TensorOperationType::Activation => self.execute_activation(operation, inputs)?,
            TensorOperationType::Custom(name) => self.execute_custom(name, operation, inputs)?,
        };

        // Handle communication requirements
        self.handle_communication_requirements(&operation.communication_requirements)?;

        // Update statistics
        {
            let mut stats = self.statistics.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.computation_time += start_time.elapsed();
            *stats.operation_count.entry(operation.operation_type.clone()).or_insert(0) += 1;
        }

        Ok(outputs)
    }

    /// Fetch a required input tensor, or fail with a message naming the key.
    fn require_input<'a>(
        inputs: &'a HashMap<String, Tensor>,
        key: &str,
        operation: &TensorOperationType,
    ) -> Result<&'a Tensor> {
        inputs.get(key).ok_or_else(|| {
            anyhow!(
                "operation {:?} requires input `{}`; provided keys: {:?}",
                operation,
                key,
                {
                    let mut keys: Vec<&str> = inputs.keys().map(String::as_str).collect();
                    keys.sort_unstable();
                    keys
                }
            )
        })
    }

    /// Execute matrix multiplication with tensor parallelism
    fn execute_matmul(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let a = Self::require_input(inputs, "A", &operation.operation_type)?;
        let b = Self::require_input(inputs, "B", &operation.operation_type)?;

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), a.matmul(b)?);
        Ok(outputs)
    }

    /// Execute tensor addition
    fn execute_add(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let a = Self::require_input(inputs, "A", &operation.operation_type)?;
        let b = Self::require_input(inputs, "B", &operation.operation_type)?;

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), a.add(b)?);
        Ok(outputs)
    }

    /// Execute scaled dot-product attention over this rank's shard of the
    /// attention heads.
    ///
    /// Inputs are the already-sharded projections `query`, `key` and `value`,
    /// each shaped `[sequence, local_heads * head_dim]`, plus the optional row-
    /// parallel output projection `weight` shaped
    /// `[local_heads * head_dim, model_dim]`.
    ///
    /// Per head `h`:
    ///
    /// ```text
    /// scores_h = Q_h · K_hᵀ / sqrt(head_dim)
    /// A_h      = softmax(scores_h)          (row-wise, numerically stabilised)
    /// out_h    = A_h · V_h
    /// ```
    ///
    /// The per-head outputs are concatenated along the feature axis. When
    /// `weight` is supplied the concatenated result is projected and the
    /// partial products are summed across the tensor-parallel group with an
    /// all-reduce — the standard Megatron row-parallel attention output.
    ///
    /// `num_heads` may be supplied as a single-element tensor under the key
    /// `num_heads`; it defaults to one head spanning the whole feature axis.
    fn execute_attention(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let query = Self::require_input(inputs, "query", &operation.operation_type)?;
        let key = Self::require_input(inputs, "key", &operation.operation_type)?;
        let value = Self::require_input(inputs, "value", &operation.operation_type)?;

        let query_shape = query.shape();
        if query_shape.len() != 2 {
            return Err(anyhow!(
                "attention expects 2-D [sequence, local_heads * head_dim] tensors, got {:?}",
                query_shape
            ));
        }
        if key.shape() != query_shape || value.shape() != query_shape {
            return Err(anyhow!(
                "attention expects matching query/key/value shapes, got {:?}, {:?}, {:?}",
                query_shape,
                key.shape(),
                value.shape()
            ));
        }

        let sequence = query_shape[0];
        let features = query_shape[1];

        let num_heads = match inputs.get("num_heads") {
            Some(tensor) => {
                let heads = tensor.to_vec_f32()?;
                let heads = *heads
                    .first()
                    .ok_or_else(|| anyhow!("`num_heads` must contain at least one value"))?
                    as usize;
                if heads == 0 {
                    return Err(anyhow!("`num_heads` must be >= 1"));
                }
                heads
            },
            None => 1,
        };
        if !features.is_multiple_of(num_heads) {
            return Err(anyhow!(
                "feature dimension {} is not divisible by {} local heads",
                features,
                num_heads
            ));
        }
        let head_dim = features / num_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let query_values = query.to_vec_f32()?;
        let key_values = key.to_vec_f32()?;
        let value_values = value.to_vec_f32()?;
        let mut context = vec![0.0f32; sequence * features];
        let mut scores = vec![0.0f32; sequence];

        for head in 0..num_heads {
            let head_offset = head * head_dim;
            for row in 0..sequence {
                // scores[col] = <Q[row, head], K[col, head]> * scale
                let mut max_score = f32::NEG_INFINITY;
                for (col, score) in scores.iter_mut().enumerate() {
                    let mut dot = 0.0f32;
                    for feature in 0..head_dim {
                        dot += query_values[row * features + head_offset + feature]
                            * key_values[col * features + head_offset + feature];
                    }
                    *score = dot * scale;
                    max_score = max_score.max(*score);
                }

                // Numerically stable softmax over the key axis.
                let mut denominator = 0.0f32;
                for score in scores.iter_mut() {
                    *score = (*score - max_score).exp();
                    denominator += *score;
                }
                let inverse = if denominator > 0.0 { 1.0 / denominator } else { 0.0 };

                for (col, score) in scores.iter().enumerate() {
                    let weight = score * inverse;
                    if weight == 0.0 {
                        continue;
                    }
                    for feature in 0..head_dim {
                        context[row * features + head_offset + feature] +=
                            weight * value_values[col * features + head_offset + feature];
                    }
                }
            }
        }

        let context = Tensor::from_slice(&context, &[sequence, features])?;
        let mut outputs = HashMap::new();

        match inputs.get("weight") {
            Some(weight) => {
                // Row-parallel output projection: each rank holds a slice of the
                // contraction dimension, so the partial products must be summed
                // across the tensor-parallel group.
                let mut projected = vec![context.matmul(weight)?];
                if self.tensor_group.world_size() > 1 {
                    self.tensor_group.all_reduce(&mut projected)?;
                }
                outputs.insert("output".to_string(), projected.remove(0));
            },
            None => {
                outputs.insert("output".to_string(), context.clone());
            },
        }

        outputs.insert("context".to_string(), context);
        Ok(outputs)
    }

    /// Execute linear layer
    fn execute_linear(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let input = Self::require_input(inputs, "input", &operation.operation_type)?;
        let weight = Self::require_input(inputs, "weight", &operation.operation_type)?;

        let mut result = input.matmul(weight)?;
        if let Some(bias) = inputs.get("bias") {
            result = result.add(bias)?;
        }

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), result);
        Ok(outputs)
    }

    /// Execute an embedding lookup over this rank's vocabulary shard.
    ///
    /// `input` holds flat token ids; `weight` is the `[local_vocab, embed_dim]`
    /// shard starting at `vocab_offset` (default `0`). Ids outside this shard
    /// contribute zeros, so an all-reduce across the tensor-parallel group
    /// reconstructs the full embedding — the standard vocabulary-parallel
    /// embedding.
    fn execute_embedding(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let input = Self::require_input(inputs, "input", &operation.operation_type)?;
        let weight = Self::require_input(inputs, "weight", &operation.operation_type)?;

        let weight_shape = weight.shape();
        if weight_shape.len() != 2 {
            return Err(anyhow!(
                "embedding weight must be 2-D [local_vocab, embed_dim], got {:?}",
                weight_shape
            ));
        }
        let (local_vocab, embed_dim) = (weight_shape[0], weight_shape[1]);

        let vocab_offset = match inputs.get("vocab_offset") {
            Some(tensor) => *tensor
                .to_vec_f32()?
                .first()
                .ok_or_else(|| anyhow!("`vocab_offset` must contain at least one value"))?
                as usize,
            None => 0,
        };

        let ids = input.to_vec_f32()?;
        let weight_values = weight.to_vec_f32()?;
        let mut gathered = vec![0.0f32; ids.len() * embed_dim];

        for (position, id) in ids.iter().enumerate() {
            if *id < 0.0 {
                return Err(anyhow!("embedding ids must be non-negative, got {id}"));
            }
            let global_id = *id as usize;
            if global_id < vocab_offset {
                continue;
            }
            let local_id = global_id - vocab_offset;
            if local_id >= local_vocab {
                continue; // owned by another rank
            }
            gathered[position * embed_dim..(position + 1) * embed_dim]
                .copy_from_slice(&weight_values[local_id * embed_dim..(local_id + 1) * embed_dim]);
        }

        let mut shape = input.shape();
        shape.push(embed_dim);
        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), Tensor::from_slice(&gathered, &shape)?);
        Ok(outputs)
    }

    /// Execute layer normalization over the last dimension.
    ///
    /// LayerNorm is *not* sharded in tensor parallelism (every rank keeps a full
    /// copy), so this is a local computation: `(x - mean) / sqrt(var + eps)`
    /// followed by the optional affine `weight`/`bias`.
    fn execute_layernorm(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let input = Self::require_input(inputs, "input", &operation.operation_type)?;

        let shape = input.shape();
        let last_dim = *shape
            .last()
            .ok_or_else(|| anyhow!("layernorm input must have at least one dimension"))?;
        if last_dim == 0 {
            return Err(anyhow!("layernorm last dimension must be non-empty"));
        }

        let epsilon = match inputs.get("epsilon") {
            Some(tensor) => *tensor
                .to_vec_f32()?
                .first()
                .ok_or_else(|| anyhow!("`epsilon` must contain at least one value"))?,
            None => 1e-5,
        };
        let gain = match inputs.get("weight") {
            Some(tensor) => Some(tensor.to_vec_f32()?),
            None => None,
        };
        let bias = match inputs.get("bias") {
            Some(tensor) => Some(tensor.to_vec_f32()?),
            None => None,
        };

        let mut values = input.to_vec_f32()?;
        for row in values.chunks_mut(last_dim) {
            let mean = row.iter().sum::<f32>() / last_dim as f32;
            let variance =
                row.iter().map(|value| (value - mean).powi(2)).sum::<f32>() / last_dim as f32;
            let inverse_std = 1.0 / (variance + epsilon).sqrt();
            for (index, value) in row.iter_mut().enumerate() {
                let mut normalized = (*value - mean) * inverse_std;
                if let Some(gain) = &gain {
                    normalized *= gain.get(index).copied().unwrap_or(1.0);
                }
                if let Some(bias) = &bias {
                    normalized += bias.get(index).copied().unwrap_or(0.0);
                }
                *value = normalized;
            }
        }

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), Tensor::from_slice(&values, &shape)?);
        Ok(outputs)
    }

    /// Execute an element-wise activation.
    ///
    /// The activation is selected by the operation's `Custom` name when present,
    /// otherwise ReLU. Supported: `relu`, `gelu`, `silu`/`swish`, `tanh`,
    /// `sigmoid`.
    fn execute_activation(
        &self,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let input = Self::require_input(inputs, "input", &operation.operation_type)?;
        let kind = match &operation.operation_type {
            TensorOperationType::Custom(name) => name.to_ascii_lowercase(),
            _ => "relu".to_string(),
        };

        let shape = input.shape();
        let mut values = input.to_vec_f32()?;
        for value in values.iter_mut() {
            *value = match kind.as_str() {
                "relu" => value.max(0.0),
                "gelu" => {
                    // Tanh approximation of the Gaussian error linear unit.
                    let x = *value;
                    let inner = (2.0f32 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x * x * x);
                    0.5 * x * (1.0 + inner.tanh())
                },
                "silu" | "swish" => *value / (1.0 + (-*value).exp()),
                "tanh" => value.tanh(),
                "sigmoid" => 1.0 / (1.0 + (-*value).exp()),
                other => {
                    return Err(anyhow!(
                        "unsupported activation `{other}`; expected one of relu, gelu, silu, \
                         swish, tanh, sigmoid"
                    ))
                },
            };
        }

        let mut outputs = HashMap::new();
        outputs.insert("output".to_string(), Tensor::from_slice(&values, &shape)?);
        Ok(outputs)
    }

    /// Execute a custom operation.
    ///
    /// Activation names are dispatched to [`Self::execute_activation`]; anything
    /// else has no definition here and returns an error rather than silently
    /// echoing its input.
    fn execute_custom(
        &self,
        operation_name: &str,
        operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        const ACTIVATIONS: [&str; 6] = ["relu", "gelu", "silu", "swish", "tanh", "sigmoid"];
        if ACTIVATIONS.contains(&operation_name.to_ascii_lowercase().as_str()) {
            return self.execute_activation(operation, inputs);
        }

        Err(anyhow!(
            "custom tensor operation `{operation_name}` has no implementation; register it or \
             use one of the built-in operation types"
        ))
    }

    /// Store this rank's data for `partition_id`.
    ///
    /// The partition table records shapes and ownership; the tensors themselves
    /// live here. Collectives operate on this registry, so data must be stored
    /// before a communication requirement referencing the partition is
    /// executed.
    pub fn store_partition_data(&self, partition_id: usize, tensor: Tensor) -> Result<()> {
        let mut data = self.partition_data.write().unwrap_or_else(|poisoned| poisoned.into_inner());
        data.insert(partition_id, tensor);
        Ok(())
    }

    /// This rank's data for `partition_id`, if any.
    pub fn partition_data(&self, partition_id: usize) -> Option<Tensor> {
        self.partition_data
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .get(&partition_id)
            .cloned()
    }

    fn require_partition_data(&self, partition_id: usize, operation: &str) -> Result<Tensor> {
        self.partition_data(partition_id).ok_or_else(|| {
            anyhow!(
                "no tensor data registered for partition {partition_id}; call \
                 TensorParallelism::store_partition_data before running `{operation}`"
            )
        })
    }

    fn find_partition(&self, partition_id: usize, operation: &str) -> Result<&TensorPartition> {
        self.tensor_partitions
            .values()
            .flatten()
            .find(|partition| partition.partition_id == partition_id)
            .ok_or_else(|| anyhow!("partition {partition_id} not found for `{operation}`"))
    }

    /// Handle communication requirements for tensor operations
    fn handle_communication_requirements(
        &self,
        requirements: &[CommunicationRequirement],
    ) -> Result<()> {
        let start_time = Instant::now();

        for requirement in requirements {
            match requirement.communication_type {
                TensorCommunicationPattern::AllReduce => {
                    self.handle_all_reduce(requirement)?;
                },
                TensorCommunicationPattern::AllGather => {
                    self.handle_all_gather(requirement)?;
                },
                TensorCommunicationPattern::ReduceScatter => {
                    self.handle_reduce_scatter(requirement)?;
                },
                TensorCommunicationPattern::PointToPoint => {
                    self.handle_point_to_point(requirement)?;
                },
                TensorCommunicationPattern::Hierarchical => {
                    self.handle_hierarchical(requirement)?;
                },
            }
        }

        // Update communication statistics
        {
            let mut stats = self.statistics.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
            stats.total_communication_time += start_time.elapsed();
            stats.communication_volume +=
                requirements.iter().map(|r| r.data_size as u64).sum::<u64>();
        }

        Ok(())
    }

    /// Sum the source partition's tensor across the communication group and
    /// write the result back into the partition registry.
    fn handle_all_reduce(&self, requirement: &CommunicationRequirement) -> Result<()> {
        let partition_id = requirement.source_partition;
        let partition = self.find_partition(partition_id, "all-reduce")?;

        let group: &Arc<dyn ProcessGroup> =
            if self.config.column_parallel && partition.needs_communication {
                self.column_group.as_ref().unwrap_or(&self.tensor_group)
            } else {
                &self.tensor_group
            };

        let mut tensors = vec![self.require_partition_data(partition_id, "all-reduce")?];
        group.all_reduce(&mut tensors)?;
        self.store_partition_data(partition_id, tensors.remove(0))?;

        Ok(())
    }

    /// Gather every rank's slice of the source partition and store the
    /// concatenation into the target partition.
    fn handle_all_gather(&self, requirement: &CommunicationRequirement) -> Result<()> {
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;
        self.find_partition(source_partition, "all-gather")?;

        let group: &Arc<dyn ProcessGroup> = if self.config.row_parallel {
            self.row_group.as_ref().unwrap_or(&self.tensor_group)
        } else {
            &self.tensor_group
        };

        let local = self.require_partition_data(source_partition, "all-gather")?;
        let gathered = group.all_gather(&local)?;

        // Concatenate along the leading axis, which is how row-parallel shards
        // reassemble into the full tensor.
        let mut values = Vec::new();
        let mut rows = 0usize;
        let mut trailing = local.shape();
        for shard in &gathered {
            let shard_shape = shard.shape();
            rows += shard_shape.first().copied().unwrap_or(0);
            values.extend(shard.to_vec_f32()?);
        }
        if trailing.is_empty() {
            trailing = vec![values.len()];
        } else {
            trailing[0] = rows;
        }

        self.store_partition_data(target_partition, Tensor::from_slice(&values, &trailing)?)?;
        Ok(())
    }

    /// Reduce the source partition across the group and keep only this rank's
    /// chunk, storing it in the target partition.
    fn handle_reduce_scatter(&self, requirement: &CommunicationRequirement) -> Result<()> {
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;
        self.find_partition(source_partition, "reduce-scatter")?;

        let local = self.require_partition_data(source_partition, "reduce-scatter")?;
        let chunk = self.tensor_group.reduce_scatter(&local)?;
        self.store_partition_data(target_partition, chunk)?;

        Ok(())
    }

    /// Move the source partition's tensor directly from its owner to the target
    /// partition's owner.
    fn handle_point_to_point(&self, requirement: &CommunicationRequirement) -> Result<()> {
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;

        let source = self.find_partition(source_partition, "point-to-point")?;
        let target = self.find_partition(target_partition, "point-to-point")?;
        let source_rank = source.device_rank;
        let target_rank = target.device_rank;
        let shape = source.shape.clone();

        if source_rank == target_rank {
            // Purely local move; no message needed.
            if self.global_rank == source_rank {
                let tensor = self.require_partition_data(source_partition, "point-to-point")?;
                self.store_partition_data(target_partition, tensor)?;
            }
            return Ok(());
        }

        if !self.tensor_group.supports_point_to_point() {
            return Err(anyhow!(
                "point-to-point transfer from partition {source_partition} to \
                 {target_partition} requires a process group with a real transport"
            ));
        }

        // Both endpoints derive the same tag from the partition pair, so no
        // call-ordering convention is required.
        let tag = point_to_point_tag(source_partition, target_partition);

        if self.global_rank == source_rank {
            let tensor = self.require_partition_data(source_partition, "point-to-point")?;
            self.tensor_group.send(target_rank, tag, &tensor)?;
        } else if self.global_rank == target_rank {
            let tensor = self.tensor_group.recv(source_rank, tag, &shape)?;
            self.store_partition_data(target_partition, tensor)?;
        }

        Ok(())
    }

    /// Two-level all-reduce: reduce within each node to its leader, all-reduce
    /// among the leaders, then broadcast the result back inside each node.
    ///
    /// Nodes are inferred from the rank layout (`sqrt(world_size)` ranks per
    /// node), matching how the topology is modelled elsewhere in this module.
    /// With a single node this degenerates to a plain group all-reduce.
    fn handle_hierarchical(&self, requirement: &CommunicationRequirement) -> Result<()> {
        let source_partition = requirement.source_partition;
        self.find_partition(source_partition, "hierarchical")?;

        let ranks_per_node = ((self.world_size as f64).sqrt().ceil() as usize).max(1);
        if ranks_per_node >= self.world_size || self.world_size <= 1 {
            // One node: the hierarchy collapses to a flat all-reduce.
            let mut tensors = vec![self.require_partition_data(source_partition, "hierarchical")?];
            self.tensor_group.all_reduce(&mut tensors)?;
            self.store_partition_data(source_partition, tensors.remove(0))?;
            return Ok(());
        }

        if !self.tensor_group.supports_point_to_point() {
            return Err(anyhow!(
                "hierarchical communication requires a process group with a real transport"
            ));
        }

        let node_id = self.global_rank / ranks_per_node;
        let local_rank = self.global_rank % ranks_per_node;
        let leader = node_id * ranks_per_node;

        let mut tensor = self.require_partition_data(source_partition, "hierarchical")?;
        let shape = tensor.shape();
        let node_members: Vec<usize> = (leader..self.world_size)
            .take(ranks_per_node)
            .filter(|rank| *rank != leader)
            .collect();

        if local_rank == 0 {
            // Phase 1: intra-node reduce into the leader.
            for member in &node_members {
                let contribution = self.tensor_group.recv(
                    *member,
                    hierarchical_tag(source_partition, 0, *member),
                    &shape,
                )?;
                tensor = tensor.add(&contribution)?;
            }

            // Phase 2: all-reduce among the leaders. Every rank participates in
            // the group collective, so non-leaders contribute a zero tensor and
            // the sum is exactly the leaders' sum.
            let mut leaders = vec![tensor.clone()];
            self.tensor_group.all_reduce(&mut leaders)?;
            tensor = leaders.remove(0);

            // Phase 3: broadcast the result back inside the node.
            for member in &node_members {
                self.tensor_group.send(
                    *member,
                    hierarchical_tag(source_partition, 1, *member),
                    &tensor,
                )?;
            }
        } else {
            // Phase 1: contribute to the node leader.
            self.tensor_group.send(
                leader,
                hierarchical_tag(source_partition, 0, self.global_rank),
                &tensor,
            )?;

            // Phase 2: participate with zeros so the group collective stays in
            // lock-step without double-counting this rank's data.
            let mut zeros = vec![Tensor::zeros(&shape)?];
            self.tensor_group.all_reduce(&mut zeros)?;

            // Phase 3: receive the node result.
            tensor = self.tensor_group.recv(
                leader,
                hierarchical_tag(source_partition, 1, self.global_rank),
                &shape,
            )?;
        }

        self.store_partition_data(source_partition, tensor)?;
        Ok(())
    }

    /// Get tensor parallelism statistics
    pub fn get_statistics(&self) -> TensorParallelismStatistics {
        let stats = self.statistics.lock().unwrap_or_else(|poisoned| poisoned.into_inner());

        TensorParallelismStatistics {
            total_partitions: self.tensor_partitions.values().map(|v| v.len()).sum(),
            local_partitions: self.local_partitions.values().map(|v| v.len()).sum(),
            communication_time: stats.total_communication_time,
            computation_time: stats.computation_time,
            communication_volume: stats.communication_volume,
            efficiency_score: stats.efficiency_score,
            memory_usage_per_device: stats.memory_usage_per_device.clone(),
        }
    }

    /// Get configuration
    pub fn config(&self) -> &TensorParallelismConfig {
        &self.config
    }

    /// Get local partitions for a tensor
    pub fn get_local_partitions(&self, tensor_name: &str) -> Option<&Vec<usize>> {
        self.local_partitions.get(tensor_name)
    }

    /// Get tensor partitions
    pub fn get_tensor_partitions(&self, tensor_name: &str) -> Option<&Vec<TensorPartition>> {
        self.tensor_partitions.get(tensor_name)
    }
}

/// Tensor parallelism statistics
#[derive(Debug, Clone)]
pub struct TensorParallelismStatistics {
    pub total_partitions: usize,
    pub local_partitions: usize,
    pub communication_time: Duration,
    pub computation_time: Duration,
    pub communication_volume: u64,
    pub efficiency_score: f32,
    pub memory_usage_per_device: HashMap<usize, u64>,
}

/// Tensor parallelism utilities
pub mod utils {
    use super::*;

    /// Calculate optimal tensor parallelism configuration
    pub fn calculate_optimal_tensor_config(
        model_size_params: u64,
        memory_per_device: u64,
        world_size: usize,
    ) -> Result<TensorParallelismConfig> {
        let memory_per_param = 4; // 4 bytes per float32 parameter
        let model_memory_size = model_size_params * memory_per_param;

        let required_devices = model_memory_size.div_ceil(memory_per_device);
        let tensor_parallel_size = std::cmp::min(required_devices as usize, world_size);

        Ok(TensorParallelismConfig {
            tensor_parallel_size,
            ..Default::default()
        })
    }

    /// Estimate communication overhead for tensor parallelism
    pub fn estimate_communication_overhead(
        config: &TensorParallelismConfig,
        tensor_size_bytes: usize,
        operations_per_step: usize,
    ) -> f32 {
        let communication_per_operation = match config.communication_pattern {
            TensorCommunicationPattern::AllReduce => tensor_size_bytes * 2, // Send + receive
            TensorCommunicationPattern::AllGather => {
                tensor_size_bytes * config.tensor_parallel_size
            },
            TensorCommunicationPattern::ReduceScatter => tensor_size_bytes,
            _ => tensor_size_bytes,
        };

        (communication_per_operation * operations_per_step) as f32 / (1024.0 * 1024.0)
        // Convert to MB
    }

    /// Calculate memory savings from tensor parallelism
    pub fn calculate_memory_savings(model_params: u64, tensor_parallel_size: usize) -> f32 {
        if tensor_parallel_size <= 1 {
            return 0.0;
        }

        let memory_per_device = model_params / tensor_parallel_size as u64;
        let total_memory_without_tp = model_params;

        1.0 - (memory_per_device as f32 / total_memory_without_tp as f32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributed::SimulatedProcessGroup;
    use std::sync::Arc;

    #[test]
    fn test_tensor_parallelism_config() {
        let config = TensorParallelismConfig::default();
        assert_eq!(config.tensor_parallel_size, 1);
        assert!(config.column_parallel);
        assert!(config.row_parallel);
    }

    #[test]
    fn test_tensor_parallelism_creation() {
        let config = TensorParallelismConfig {
            tensor_parallel_size: 4,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 4));
        let tensor_parallelism = TensorParallelism::new(config, 0, 4, process_group);

        assert!(tensor_parallelism.is_ok());
    }

    #[test]
    fn test_column_wise_partitioning() {
        let config = TensorParallelismConfig {
            tensor_parallel_size: 2,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
        let mut tensor_parallelism =
            TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

        let partitions = tensor_parallelism
            .partition_tensor("test", &[100, 200], None)
            .expect("tensor operation failed");
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].shape, vec![100, 100]);
        assert_eq!(partitions[1].shape, vec![100, 100]);
    }

    #[test]
    fn test_row_wise_partitioning() {
        let config = TensorParallelismConfig {
            tensor_parallel_size: 2,
            partitioning_strategy: TensorPartitioningStrategy::RowWise,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
        let mut tensor_parallelism =
            TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

        let partitions = tensor_parallelism
            .partition_tensor("test", &[100, 200], None)
            .expect("tensor operation failed");
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].shape, vec![50, 200]);
        assert_eq!(partitions[1].shape, vec![50, 200]);
    }

    #[test]
    fn test_batch_wise_partitioning() {
        let config = TensorParallelismConfig {
            tensor_parallel_size: 2,
            partitioning_strategy: TensorPartitioningStrategy::BatchWise,
            ..Default::default()
        };

        let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
        let mut tensor_parallelism =
            TensorParallelism::new(config, 0, 2, process_group).expect("tensor operation failed");

        let partitions = tensor_parallelism
            .partition_tensor("test", &[64, 100, 200], None)
            .expect("tensor operation failed");
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].shape, vec![32, 100, 200]);
        assert_eq!(partitions[1].shape, vec![32, 100, 200]);
    }

    #[test]
    fn test_tensor_operation_execution() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism =
            TensorParallelism::new(config, 0, 1, process_group).expect("tensor operation failed");

        let operation = TensorOperation {
            operation_id: 0,
            operation_type: TensorOperationType::Add,
            input_partitions: vec![0, 1],
            output_partitions: vec![0],
            communication_requirements: vec![],
            memory_requirements: 1024,
        };

        let mut inputs = HashMap::new();
        inputs.insert(
            "A".to_string(),
            Tensor::ones(&[10, 10]).expect("tensor operation failed"),
        );
        inputs.insert(
            "B".to_string(),
            Tensor::ones(&[10, 10]).expect("tensor operation failed"),
        );

        let result = tensor_parallelism.execute_operation(&operation, &inputs);
        assert!(result.is_ok());
    }

    /// Naive single-head scaled dot-product attention reference.
    fn reference_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, dim: usize) -> Vec<f32> {
        let scale = 1.0 / (dim as f32).sqrt();
        let mut out = vec![0.0f32; seq * dim];
        for row in 0..seq {
            let mut scores = vec![0.0f32; seq];
            for (col, score) in scores.iter_mut().enumerate() {
                let mut dot = 0.0f32;
                for f in 0..dim {
                    dot += q[row * dim + f] * k[col * dim + f];
                }
                *score = dot * scale;
            }
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f32 = exp.iter().sum();
            for (col, weight) in exp.iter().enumerate() {
                let weight = weight / sum;
                for f in 0..dim {
                    out[row * dim + f] += weight * v[col * dim + f];
                }
            }
        }
        out
    }

    #[test]
    fn attention_is_not_the_identity_and_matches_the_reference() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism =
            TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

        let seq = 3;
        let dim = 2;
        let q = vec![1.0, 0.0, 0.0, 1.0, 0.5, 0.5];
        let k = vec![1.0, 0.0, 0.0, 1.0, -1.0, 0.0];
        let v = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];

        let mut inputs = HashMap::new();
        inputs.insert(
            "query".to_string(),
            Tensor::from_slice(&q, &[seq, dim]).expect("tensor must build in test"),
        );
        inputs.insert(
            "key".to_string(),
            Tensor::from_slice(&k, &[seq, dim]).expect("tensor must build in test"),
        );
        inputs.insert(
            "value".to_string(),
            Tensor::from_slice(&v, &[seq, dim]).expect("tensor must build in test"),
        );

        let operation = TensorOperation {
            operation_id: 1,
            operation_type: TensorOperationType::Attention,
            input_partitions: vec![],
            output_partitions: vec![],
            communication_requirements: vec![],
            memory_requirements: 0,
        };

        let outputs = tensor_parallelism
            .execute_operation(&operation, &inputs)
            .expect("attention must execute in test");
        let produced = outputs
            .get("output")
            .expect("attention must produce an output")
            .to_vec_f32()
            .expect("tensor read must succeed in test");

        let expected = reference_attention(&q, &k, &v, seq, dim);
        for (got, want) in produced.iter().zip(&expected) {
            approx::assert_relative_eq!(got, want, epsilon = 1e-5);
        }

        // Regression: the previous implementation returned `inputs["input"]`
        // unchanged, so assert the output genuinely differs from the query.
        assert_ne!(produced, q, "attention must not be the identity");
    }

    #[test]
    fn attention_errors_on_missing_inputs() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism =
            TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

        let operation = TensorOperation {
            operation_id: 1,
            operation_type: TensorOperationType::Attention,
            input_partitions: vec![],
            output_partitions: vec![],
            communication_requirements: vec![],
            memory_requirements: 0,
        };

        // Empty input map used to yield an empty output map instead of an error.
        assert!(tensor_parallelism.execute_operation(&operation, &HashMap::new()).is_err());
    }

    #[test]
    fn matmul_and_linear_error_on_missing_inputs() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism =
            TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

        for operation_type in [
            TensorOperationType::MatMul,
            TensorOperationType::Linear,
            TensorOperationType::Add,
        ] {
            let operation = TensorOperation {
                operation_id: 0,
                operation_type,
                input_partitions: vec![],
                output_partitions: vec![],
                communication_requirements: vec![],
                memory_requirements: 0,
            };
            assert!(tensor_parallelism.execute_operation(&operation, &HashMap::new()).is_err());
        }
    }

    #[test]
    fn layernorm_normalises_and_activation_is_real() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism =
            TensorParallelism::new(config, 0, 1, process_group).expect("must build in test");

        let mut inputs = HashMap::new();
        inputs.insert(
            "input".to_string(),
            Tensor::from_slice(&[1.0, 2.0, 3.0, 4.0], &[2, 2]).expect("tensor must build in test"),
        );

        let layernorm = TensorOperation {
            operation_id: 0,
            operation_type: TensorOperationType::LayerNorm,
            input_partitions: vec![],
            output_partitions: vec![],
            communication_requirements: vec![],
            memory_requirements: 0,
        };
        let normalized = tensor_parallelism
            .execute_operation(&layernorm, &inputs)
            .expect("layernorm must execute in test")
            .get("output")
            .expect("layernorm must produce an output")
            .to_vec_f32()
            .expect("tensor read must succeed in test");

        // Each row has zero mean and unit variance (up to epsilon).
        for row in normalized.chunks(2) {
            approx::assert_relative_eq!(row[0] + row[1], 0.0f32, epsilon = 1e-4);
            approx::assert_relative_eq!(row[0], -1.0f32, epsilon = 1e-3);
        }

        inputs.insert(
            "input".to_string(),
            Tensor::from_slice(&[-2.0, -0.5, 0.5, 2.0], &[4]).expect("tensor must build in test"),
        );
        let relu = TensorOperation {
            operation_id: 1,
            operation_type: TensorOperationType::Activation,
            input_partitions: vec![],
            output_partitions: vec![],
            communication_requirements: vec![],
            memory_requirements: 0,
        };
        let activated = tensor_parallelism
            .execute_operation(&relu, &inputs)
            .expect("activation must execute in test")
            .get("output")
            .expect("activation must produce an output")
            .to_vec_f32()
            .expect("tensor read must succeed in test");
        assert_eq!(activated, vec![0.0, 0.0, 0.5, 2.0]);
    }

    #[test]
    fn all_reduce_requirement_moves_real_partition_data() {
        use crate::distributed_collective::InProcessProcessGroup;

        let world_size = 2;
        let groups = InProcessProcessGroup::group(world_size).expect("groups must build in test");

        let handles: Vec<_> = groups
            .into_iter()
            .enumerate()
            .map(|(rank, group)| {
                std::thread::spawn(move || {
                    let config = TensorParallelismConfig {
                        tensor_parallel_size: world_size,
                        column_parallel: false,
                        ..Default::default()
                    };
                    let group: Arc<dyn ProcessGroup> = Arc::new(group);
                    let mut tp = TensorParallelism::new(config, rank, world_size, group)
                        .expect("must build in test");
                    let partitions = tp
                        .partition_tensor("w", &[4, 4], None)
                        .expect("partitioning must succeed in test");
                    let partition_id = partitions[0].partition_id;

                    tp.store_partition_data(
                        partition_id,
                        Tensor::from_slice(&[rank as f32 + 1.0; 4], &[4])
                            .expect("tensor must build in test"),
                    )
                    .expect("store must succeed in test");

                    let operation = TensorOperation {
                        operation_id: 0,
                        operation_type: TensorOperationType::Add,
                        input_partitions: vec![],
                        output_partitions: vec![],
                        communication_requirements: vec![CommunicationRequirement {
                            source_partition: partition_id,
                            target_partition: partition_id,
                            communication_type: TensorCommunicationPattern::AllReduce,
                            data_size: 16,
                        }],
                        memory_requirements: 0,
                    };

                    let mut inputs = HashMap::new();
                    inputs.insert(
                        "A".to_string(),
                        Tensor::ones(&[2, 2]).expect("tensor must build in test"),
                    );
                    inputs.insert(
                        "B".to_string(),
                        Tensor::ones(&[2, 2]).expect("tensor must build in test"),
                    );
                    tp.execute_operation(&operation, &inputs)
                        .expect("operation must execute in test");

                    tp.partition_data(partition_id)
                        .expect("partition data must survive")
                        .to_vec_f32()
                        .expect("tensor read must succeed in test")
                })
            })
            .collect();

        for handle in handles {
            let values = handle.join().expect("rank thread must not panic in test");
            // 1 + 2 summed across the two ranks.
            assert_eq!(values, vec![3.0f32; 4]);
        }
    }

    #[test]
    fn all_reduce_requirement_errors_without_registered_data() {
        let config = TensorParallelismConfig {
            tensor_parallel_size: 2,
            ..Default::default()
        };
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 2));
        let mut tp =
            TensorParallelism::new(config, 0, 2, process_group).expect("must build in test");
        let partitions = tp
            .partition_tensor("w", &[4, 4], None)
            .expect("partitioning must succeed in test");

        let requirement = CommunicationRequirement {
            source_partition: partitions[0].partition_id,
            target_partition: partitions[0].partition_id,
            communication_type: TensorCommunicationPattern::AllReduce,
            data_size: 16,
        };
        assert!(
            tp.handle_communication_requirements(&[requirement]).is_err(),
            "an all-reduce with no registered tensor must fail loudly"
        );
    }

    #[test]
    fn test_optimal_tensor_config_calculation() {
        // Use 10B parameters (40GB memory) with 8GB per device
        // This requires at least 5 devices, so tensor_parallel_size > 1
        let config = utils::calculate_optimal_tensor_config(
            10_000_000_000,         // 10B parameters (40GB memory)
            8 * 1024 * 1024 * 1024, // 8GB memory per device
            8,                      // world size
        )
        .expect("operation failed in test");

        assert!(
            config.tensor_parallel_size > 1,
            "Expected tensor_parallel_size > 1, got {}",
            config.tensor_parallel_size
        );
    }

    #[test]
    fn test_communication_overhead_estimation() {
        let config = TensorParallelismConfig::default();
        let overhead = utils::estimate_communication_overhead(&config, 1024 * 1024, 100);
        assert!(overhead > 0.0);
    }

    #[test]
    fn test_memory_savings_calculation() {
        let savings = utils::calculate_memory_savings(1_000_000_000, 4);
        assert!(savings > 0.0 && savings < 1.0);
    }
}
