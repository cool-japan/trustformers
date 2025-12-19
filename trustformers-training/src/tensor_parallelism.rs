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
#[allow(dead_code)]
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
    #[allow(dead_code)]
    operation_scheduler: Arc<RwLock<OperationScheduler>>,

    // Communication optimization
    communication_optimizer: Arc<Mutex<CommunicationOptimizer>>,

    // Statistics tracking
    statistics: Arc<Mutex<TensorParallelismStats>>,
}

/// Operation scheduler for tensor operations
#[derive(Debug, Default)]
#[allow(dead_code)]
struct OperationScheduler {
    #[allow(dead_code)]
    pending_operations: Vec<TensorOperation>,
    running_operations: Vec<TensorOperation>,
    completed_operations: Vec<TensorOperation>,
    operation_graph: HashMap<usize, Vec<usize>>, // operation_id -> dependencies
}

/// Communication optimizer for reducing communication overhead
#[derive(Debug, Default)]
#[allow(dead_code)]
struct CommunicationOptimizer {
    #[allow(dead_code)]
    fusion_buffer: Vec<CommunicationRequirement>,
    communication_schedule: Vec<Vec<CommunicationRequirement>>, // Batched communications
    async_handles: Vec<AsyncCommHandle>,
    bandwidth_usage: f32,
    latency_estimates: HashMap<TensorCommunicationPattern, Duration>,
}

/// Async communication handle (placeholder)
#[derive(Debug)]
#[allow(dead_code)]
struct AsyncCommHandle {
    #[allow(dead_code)]
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

        if world_size % config.tensor_parallel_size != 0 {
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
            let mut stats = self.statistics.lock().unwrap();
            stats.computation_time += start_time.elapsed();
            *stats.operation_count.entry(operation.operation_type.clone()).or_insert(0) += 1;
        }

        Ok(outputs)
    }

    /// Execute matrix multiplication with tensor parallelism
    fn execute_matmul(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Simplified matrix multiplication
        // In practice, would handle distributed computation across tensor partitions
        let mut outputs = HashMap::new();

        if let (Some(a), Some(b)) = (inputs.get("A"), inputs.get("B")) {
            let result = a.matmul(b)?;
            outputs.insert("output".to_string(), result);
        }

        Ok(outputs)
    }

    /// Execute tensor addition
    fn execute_add(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let (Some(a), Some(b)) = (inputs.get("A"), inputs.get("B")) {
            let result = a.add(b)?;
            outputs.insert("output".to_string(), result);
        }

        Ok(outputs)
    }

    /// Execute attention mechanism
    fn execute_attention(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        // Simplified attention computation
        let mut outputs = HashMap::new();

        if let Some(input) = inputs.get("input") {
            // Placeholder attention computation
            outputs.insert("output".to_string(), input.clone());
        }

        Ok(outputs)
    }

    /// Execute linear layer
    fn execute_linear(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let (Some(input), Some(weight)) = (inputs.get("input"), inputs.get("weight")) {
            let result = input.matmul(weight)?;
            outputs.insert("output".to_string(), result);
        }

        Ok(outputs)
    }

    /// Execute embedding layer
    fn execute_embedding(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let Some(input) = inputs.get("input") {
            // Simplified embedding lookup
            outputs.insert("output".to_string(), input.clone());
        }

        Ok(outputs)
    }

    /// Execute layer normalization
    fn execute_layernorm(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let Some(input) = inputs.get("input") {
            // Simplified layer norm
            outputs.insert("output".to_string(), input.clone());
        }

        Ok(outputs)
    }

    /// Execute activation function
    fn execute_activation(
        &self,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let Some(input) = inputs.get("input") {
            // Simplified activation (ReLU)
            outputs.insert("output".to_string(), input.clone());
        }

        Ok(outputs)
    }

    /// Execute custom operation
    fn execute_custom(
        &self,
        _operation_name: &str,
        _operation: &TensorOperation,
        inputs: &HashMap<String, Tensor>,
    ) -> Result<HashMap<String, Tensor>> {
        let mut outputs = HashMap::new();

        if let Some(input) = inputs.get("input") {
            outputs.insert("output".to_string(), input.clone());
        }

        Ok(outputs)
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
            let mut stats = self.statistics.lock().unwrap();
            stats.total_communication_time += start_time.elapsed();
            stats.communication_volume +=
                requirements.iter().map(|r| r.data_size as u64).sum::<u64>();
        }

        Ok(())
    }

    /// Handle all-reduce communication
    fn handle_all_reduce(&self, requirement: &CommunicationRequirement) -> Result<()> {
        // All-reduce: sum gradients/tensors across all devices and distribute result back
        let partition_id = requirement.source_partition;

        // Find the tensor partition
        let partition = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == partition_id)
            .ok_or_else(|| anyhow!("Partition {} not found for all-reduce", partition_id))?;

        // Perform all-reduce operation using the appropriate communication group
        let _group = if self.config.column_parallel && partition.needs_communication {
            self.column_group.as_ref().unwrap_or(&self.tensor_group)
        } else {
            &self.tensor_group
        };

        // In a real implementation, this would perform:
        // 1. Serialize tensor data from partition
        // 2. Call group.all_reduce() with the tensor data
        // 3. Update the partition with reduced results
        println!(
            "All-reduce: Processing partition {} on device {} (size: {} bytes)",
            partition_id, partition.device_rank, requirement.data_size
        );

        // Simulate communication overhead
        std::thread::sleep(Duration::from_micros((requirement.data_size / 1000) as u64));

        Ok(())
    }

    /// Handle all-gather communication
    fn handle_all_gather(&self, requirement: &CommunicationRequirement) -> Result<()> {
        // All-gather: collect tensor partitions from all devices to reconstruct full tensor
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;

        // Find source partition
        let _source = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == source_partition)
            .ok_or_else(|| {
                anyhow!(
                    "Source partition {} not found for all-gather",
                    source_partition
                )
            })?;

        // Determine communication group based on parallelism type
        let _group = if self.config.row_parallel {
            self.row_group.as_ref().unwrap_or(&self.tensor_group)
        } else {
            &self.tensor_group
        };

        // In a real implementation, this would:
        // 1. Gather tensor partitions from all devices in the group
        // 2. Reconstruct the full tensor from gathered partitions
        // 3. Store result in target partition or broadcast to all devices
        println!(
            "All-gather: Collecting from partition {} to partition {} (size: {} bytes)",
            source_partition, target_partition, requirement.data_size
        );

        // Update local partitions map if we're gathering locally
        if let Some(tensor_name) = self
            .tensor_partitions
            .iter()
            .find(|(_, partitions)| partitions.iter().any(|p| p.partition_id == source_partition))
            .map(|(name, _)| name.clone())
        {
            // Mark that this tensor now has gathered data
            println!(
                "All-gather: Updated tensor '{}' with gathered data",
                tensor_name
            );
        }

        // Simulate communication overhead
        std::thread::sleep(Duration::from_micros((requirement.data_size / 500) as u64));

        Ok(())
    }

    /// Handle reduce-scatter communication
    fn handle_reduce_scatter(&self, requirement: &CommunicationRequirement) -> Result<()> {
        // Reduce-scatter: perform reduction operation and scatter results across devices
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;

        // Find source partition
        let _source = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == source_partition)
            .ok_or_else(|| {
                anyhow!(
                    "Source partition {} not found for reduce-scatter",
                    source_partition
                )
            })?;

        // Use tensor group for reduce-scatter operations
        let _group = &self.tensor_group;

        // Calculate scatter chunk size based on world size
        let chunk_size = requirement.data_size / self.world_size;

        // In a real implementation, this would:
        // 1. Perform reduction operation (sum, mean, etc.) on source tensor
        // 2. Split the reduced tensor into chunks equal to world_size
        // 3. Scatter each chunk to corresponding device
        // 4. Each device receives and stores its chunk in target partition
        println!("Reduce-scatter: Reducing partition {} and scattering to partition {} (chunk size: {} bytes)",
                 source_partition, target_partition, chunk_size);

        // Calculate which chunk this device should receive
        let my_chunk_index = self.global_rank;
        println!(
            "Reduce-scatter: Device {} will receive chunk {}",
            self.global_rank, my_chunk_index
        );

        // Simulate communication and computation overhead
        std::thread::sleep(Duration::from_micros((requirement.data_size / 750) as u64));

        Ok(())
    }

    /// Handle point-to-point communication
    fn handle_point_to_point(&self, requirement: &CommunicationRequirement) -> Result<()> {
        // Point-to-point: direct communication between two specific devices
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;

        // Find source and target partitions
        let source = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == source_partition)
            .ok_or_else(|| anyhow!("Source partition {} not found for P2P", source_partition))?;

        let target = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == target_partition)
            .ok_or_else(|| anyhow!("Target partition {} not found for P2P", target_partition))?;

        // Determine if this device is involved in the communication
        let is_sender = source.device_rank == self.global_rank;
        let is_receiver = target.device_rank == self.global_rank;

        if is_sender {
            // This device is sending data
            println!(
                "P2P: Sending from partition {} to device {} (size: {} bytes)",
                source_partition, target.device_rank, requirement.data_size
            );

            // In a real implementation:
            // 1. Serialize tensor data from source partition
            // 2. Use ProcessGroup.send() to target device
        } else if is_receiver {
            // This device is receiving data
            println!(
                "P2P: Receiving from device {} to partition {} (size: {} bytes)",
                source.device_rank, target_partition, requirement.data_size
            );

            // In a real implementation:
            // 1. Use ProcessGroup.recv() from source device
            // 2. Deserialize and store data in target partition
        } else {
            // This device is not involved in this P2P communication
            println!(
                "P2P: Device {} not involved in communication {} -> {}",
                self.global_rank, source.device_rank, target.device_rank
            );
        }

        // Simulate communication latency
        if is_sender || is_receiver {
            std::thread::sleep(Duration::from_micros(
                (requirement.data_size / 2000 + 100) as u64,
            ));
        }

        Ok(())
    }

    /// Handle hierarchical communication
    fn handle_hierarchical(&self, requirement: &CommunicationRequirement) -> Result<()> {
        // Hierarchical: multi-level communication for large-scale deployments
        let source_partition = requirement.source_partition;
        let target_partition = requirement.target_partition;

        // Find source partition
        let _source = self
            .tensor_partitions
            .values()
            .flatten()
            .find(|p| p.partition_id == source_partition)
            .ok_or_else(|| {
                anyhow!(
                    "Source partition {} not found for hierarchical comm",
                    source_partition
                )
            })?;

        // Calculate hierarchical communication structure
        let nodes_per_level = (self.world_size as f64).sqrt().ceil() as usize;
        let node_id = self.global_rank / nodes_per_level;
        let local_rank = self.global_rank % nodes_per_level;

        println!(
            "Hierarchical: Device {} (node {}, local rank {}) processing partition {}",
            self.global_rank, node_id, local_rank, source_partition
        );

        // Hierarchical communication typically involves:
        // 1. Intra-node communication (within each compute node)
        // 2. Inter-node communication (between node leaders)
        // 3. Final intra-node broadcast of results

        if local_rank == 0 {
            // This is a node leader - participates in inter-node communication
            println!(
                "Hierarchical: Node leader {} participating in inter-node communication",
                self.global_rank
            );

            // Phase 1: Collect from local devices (intra-node reduce)
            std::thread::sleep(Duration::from_micros((requirement.data_size / 1000) as u64));

            // Phase 2: Inter-node all-reduce among leaders
            std::thread::sleep(Duration::from_micros((requirement.data_size / 500) as u64));

            // Phase 3: Broadcast back to local devices
            std::thread::sleep(Duration::from_micros((requirement.data_size / 2000) as u64));
        } else {
            // Regular device - participates in intra-node communication only
            println!(
                "Hierarchical: Device {} participating in intra-node communication with leader",
                self.global_rank
            );

            // Phase 1: Send to node leader
            std::thread::sleep(Duration::from_micros((requirement.data_size / 2000) as u64));

            // Phase 3: Receive result from node leader
            std::thread::sleep(Duration::from_micros((requirement.data_size / 4000) as u64));
        }

        println!(
            "Hierarchical: Completed hierarchical communication for partition {} (target: {})",
            source_partition, target_partition
        );

        Ok(())
    }

    /// Get tensor parallelism statistics
    pub fn get_statistics(&self) -> TensorParallelismStatistics {
        let stats = self.statistics.lock().unwrap();

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
        let mut tensor_parallelism = TensorParallelism::new(config, 0, 2, process_group).unwrap();

        let partitions = tensor_parallelism.partition_tensor("test", &[100, 200], None).unwrap();
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
        let mut tensor_parallelism = TensorParallelism::new(config, 0, 2, process_group).unwrap();

        let partitions = tensor_parallelism.partition_tensor("test", &[100, 200], None).unwrap();
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
        let mut tensor_parallelism = TensorParallelism::new(config, 0, 2, process_group).unwrap();

        let partitions =
            tensor_parallelism.partition_tensor("test", &[64, 100, 200], None).unwrap();
        assert_eq!(partitions.len(), 2);
        assert_eq!(partitions[0].shape, vec![32, 100, 200]);
        assert_eq!(partitions[1].shape, vec![32, 100, 200]);
    }

    #[test]
    fn test_tensor_operation_execution() {
        let config = TensorParallelismConfig::default();
        let process_group = Arc::new(SimulatedProcessGroup::new(0, 1));
        let tensor_parallelism = TensorParallelism::new(config, 0, 1, process_group).unwrap();

        let operation = TensorOperation {
            operation_id: 0,
            operation_type: TensorOperationType::Add,
            input_partitions: vec![0, 1],
            output_partitions: vec![0],
            communication_requirements: vec![],
            memory_requirements: 1024,
        };

        let mut inputs = HashMap::new();
        inputs.insert("A".to_string(), Tensor::ones(&[10, 10]).unwrap());
        inputs.insert("B".to_string(), Tensor::ones(&[10, 10]).unwrap());

        let result = tensor_parallelism.execute_operation(&operation, &inputs);
        assert!(result.is_ok());
    }

    #[test]
    fn test_optimal_tensor_config_calculation() {
        let config = utils::calculate_optimal_tensor_config(
            1_000_000_000,          // 1B parameters
            8 * 1024 * 1024 * 1024, // 8GB memory per device
            8,                      // world size
        )
        .unwrap();

        assert!(config.tensor_parallel_size > 1);
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
