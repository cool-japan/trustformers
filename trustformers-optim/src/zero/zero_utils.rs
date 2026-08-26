//! Utility functions and data structures for ZeRO optimization

use std::collections::HashMap;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::parallel::ModelParallelContext;
use trustformers_core::tensor::Tensor;

/// ZeRO optimizer state management
#[derive(Debug, Clone)]
pub struct ZeROState {
    /// Current step number
    pub step: usize,
    /// Partitioned optimizer states per parameter group
    pub optimizer_states: HashMap<String, HashMap<String, Tensor>>,
    /// Partitioned gradients (for Stage 2+)
    pub gradient_partitions: HashMap<String, GradientBuffer>,
    /// Partitioned parameters (for Stage 3)
    pub parameter_partitions: HashMap<String, ParameterPartition>,
    /// Communication buffers for all-gather operations
    pub communication_buffers: HashMap<String, Tensor>,
}

impl Default for ZeROState {
    fn default() -> Self {
        Self::new()
    }
}

impl ZeROState {
    pub fn new() -> Self {
        Self {
            step: 0,
            optimizer_states: HashMap::new(),
            gradient_partitions: HashMap::new(),
            parameter_partitions: HashMap::new(),
            communication_buffers: HashMap::new(),
        }
    }

    /// Reset gradients for next iteration
    pub fn zero_grad(&mut self) {
        for buffer in self.gradient_partitions.values_mut() {
            buffer.zero();
        }
    }

    /// Increment step counter
    pub fn step(&mut self) {
        self.step += 1;
    }

    /// Get memory usage statistics
    pub fn memory_usage(&self) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        // Calculate optimizer state memory
        let mut optimizer_memory = 0;
        for states in self.optimizer_states.values() {
            for tensor in states.values() {
                optimizer_memory += tensor.memory_usage();
            }
        }
        stats.insert("optimizer_states".to_string(), optimizer_memory);

        // Calculate gradient memory
        let mut gradient_memory = 0;
        for buffer in self.gradient_partitions.values() {
            gradient_memory += buffer.memory_usage();
        }
        stats.insert("gradient_partitions".to_string(), gradient_memory);

        // Calculate parameter memory
        let mut parameter_memory = 0;
        for partition in self.parameter_partitions.values() {
            parameter_memory += partition.memory_usage();
        }
        stats.insert("parameter_partitions".to_string(), parameter_memory);

        // Calculate communication buffer memory
        let mut comm_memory = 0;
        for tensor in self.communication_buffers.values() {
            comm_memory += tensor.memory_usage();
        }
        stats.insert("communication_buffers".to_string(), comm_memory);

        stats
    }
}

/// Parameter group for ZeRO optimization
#[derive(Debug, Clone)]
pub struct ParameterGroup {
    /// Group name/identifier
    pub name: String,
    /// Parameter names in this group
    pub parameter_names: Vec<String>,
    /// Local partition of parameters
    pub local_parameters: HashMap<String, Tensor>,
    /// Metadata for parameter partitioning
    pub partition_info: PartitionInfo,
}

impl ParameterGroup {
    pub fn new(name: String, parameter_names: Vec<String>) -> Self {
        Self {
            name,
            parameter_names,
            local_parameters: HashMap::new(),
            partition_info: PartitionInfo::default(),
        }
    }

    /// Add a parameter to this group
    pub fn add_parameter(&mut self, name: String, tensor: Tensor) {
        self.local_parameters.insert(name.clone(), tensor);
        if !self.parameter_names.contains(&name) {
            self.parameter_names.push(name);
        }
    }

    /// Get total memory usage of this group
    pub fn memory_usage(&self) -> usize {
        self.local_parameters.values().map(|t| t.memory_usage()).sum()
    }
}

/// Partition information for distributed parameters
#[derive(Debug, Clone)]
pub struct PartitionInfo {
    /// Rank of this partition
    pub rank: usize,
    /// Total number of partitions
    pub world_size: usize,
    /// Start index in global parameter
    pub start_idx: usize,
    /// End index in global parameter
    pub end_idx: usize,
    /// Global shape of full parameter
    pub global_shape: Vec<usize>,
    /// Local shape of this partition
    pub local_shape: Vec<usize>,
}

impl Default for PartitionInfo {
    fn default() -> Self {
        Self {
            rank: 0,
            world_size: 1,
            start_idx: 0,
            end_idx: 0,
            global_shape: vec![],
            local_shape: vec![],
        }
    }
}

/// Parameter partition for ZeRO Stage 3
#[derive(Debug, Clone)]
pub struct ParameterPartition {
    /// Parameter name
    pub name: String,
    /// Local shard of the parameter
    pub local_shard: Tensor,
    /// Partition metadata
    pub partition_info: PartitionInfo,
    /// Whether this parameter is currently gathered
    pub is_gathered: bool,
    /// Full parameter (only valid when is_gathered = true)
    pub full_parameter: Option<Tensor>,
}

impl ParameterPartition {
    pub fn new(name: String, local_shard: Tensor, partition_info: PartitionInfo) -> Self {
        Self {
            name,
            local_shard,
            partition_info,
            is_gathered: false,
            full_parameter: None,
        }
    }

    /// Get memory usage of this partition
    pub fn memory_usage(&self) -> usize {
        let mut usage = self.local_shard.memory_usage();
        if let Some(full_param) = &self.full_parameter {
            usage += full_param.memory_usage();
        }
        usage
    }

    /// Gather full parameter from all partitions
    pub fn gather(&mut self, mp_context: &ModelParallelContext) -> Result<()> {
        if self.is_gathered {
            return Ok(());
        }

        // Use model parallel context to gather the parameter
        let full_param =
            mp_context.all_gather(&trustformers_core::parallel::DistributedTensor::new(
                self.local_shard.clone(),
                self.partition_info.global_shape.clone(),
                trustformers_core::parallel::TensorPartition {
                    split_dim: 0, // Assume partitioning along first dimension
                    start_idx: self.partition_info.start_idx,
                    end_idx: self.partition_info.end_idx,
                    num_partitions: self.partition_info.world_size,
                    partition_rank: self.partition_info.rank,
                },
                self.partition_info.rank,
            ))?;

        self.full_parameter = Some(full_param);
        self.is_gathered = true;
        Ok(())
    }

    /// Release gathered parameter to save memory
    pub fn release(&mut self) {
        self.full_parameter = None;
        self.is_gathered = false;
    }
}

/// Gradient buffer for ZeRO Stage 2+
#[derive(Debug, Clone)]
pub struct GradientBuffer {
    /// Buffer name
    pub name: String,
    /// Local gradient shard
    pub local_gradient: Tensor,
    /// Accumulated gradients
    pub accumulated_gradient: Option<Tensor>,
    /// Number of accumulated steps
    pub accumulation_steps: usize,
    /// Partition metadata
    pub partition_info: PartitionInfo,
}

impl GradientBuffer {
    pub fn new(name: String, local_gradient: Tensor, partition_info: PartitionInfo) -> Self {
        Self {
            name,
            local_gradient,
            accumulated_gradient: None,
            accumulation_steps: 0,
            partition_info,
        }
    }

    /// Zero the gradient buffer
    pub fn zero(&mut self) {
        // The shape is taken from an existing valid tensor, so `zeros` cannot
        // fail in practice; on the theoretical error path we leave the buffer
        // unchanged rather than panicking.
        if let Ok(zeros) = Tensor::zeros(&self.local_gradient.shape()) {
            self.local_gradient = zeros;
        }
        self.accumulated_gradient = None;
        self.accumulation_steps = 0;
    }

    /// Accumulate gradient
    pub fn accumulate(&mut self, gradient: &Tensor) -> Result<()> {
        if let Some(acc_grad) = &mut self.accumulated_gradient {
            *acc_grad = acc_grad.add(gradient)?;
        } else {
            self.accumulated_gradient = Some(gradient.clone());
        }
        self.accumulation_steps += 1;
        Ok(())
    }

    /// Get the accumulated gradient (averaged if needed)
    pub fn get_accumulated(&self) -> Option<Tensor> {
        if let Some(acc_grad) = &self.accumulated_gradient {
            if self.accumulation_steps > 1 {
                acc_grad.scalar_div(self.accumulation_steps as f32).ok()
            } else {
                Some(acc_grad.clone())
            }
        } else {
            None
        }
    }

    /// Get memory usage of this buffer
    pub fn memory_usage(&self) -> usize {
        let mut usage = self.local_gradient.memory_usage();
        if let Some(acc_grad) = &self.accumulated_gradient {
            usage += acc_grad.memory_usage();
        }
        usage
    }
}

/// Partition parameters across devices for ZeRO Stage 3
pub fn partition_parameters(
    parameters: &HashMap<String, Tensor>,
    world_size: usize,
    rank: usize,
) -> Result<HashMap<String, ParameterPartition>> {
    let mut partitions = HashMap::new();

    for (name, param) in parameters {
        let shape = param.shape();
        let (start_idx, end_idx) = shard_range(shape.iter().product::<usize>(), world_size, rank)?;
        let local_shard = slice_flat(param, start_idx, end_idx)?;

        let partition_info = PartitionInfo {
            rank,
            world_size,
            start_idx,
            end_idx,
            global_shape: shape.to_vec(),
            local_shape: local_shard.shape().to_vec(),
        };

        let partition = ParameterPartition::new(name.clone(), local_shard, partition_info);
        partitions.insert(name.clone(), partition);
    }

    Ok(partitions)
}

/// Computes the half-open element range `[start, end)` owned by `rank`.
///
/// Ranks are assigned contiguous, balanced slices of the *flattened* tensor:
/// rank `r` owns `[r·⌈n/W⌉, min((r+1)·⌈n/W⌉, n))`. Trailing ranks may own an empty
/// range when `n` is not divisible by `W`; the ranges are disjoint and their lengths
/// sum to exactly `n`, which is what makes ZeRO's memory saving real.
///
/// # Errors
///
/// Returns an error for `world_size == 0` or `rank >= world_size`.
pub fn shard_range(
    total_elements: usize,
    world_size: usize,
    rank: usize,
) -> Result<(usize, usize)> {
    if world_size == 0 {
        return Err(TrustformersError::invalid_input(
            "ZeRO partitioning requires world_size > 0".to_string(),
        ));
    }
    if rank >= world_size {
        return Err(TrustformersError::invalid_input(format!(
            "rank {rank} is out of range for world_size {world_size}"
        )));
    }

    let elements_per_rank = total_elements.div_ceil(world_size);
    let start_idx = (rank * elements_per_rank).min(total_elements);
    let end_idx = ((rank + 1) * elements_per_rank).min(total_elements);
    Ok((start_idx, end_idx))
}

/// Extracts `[start, end)` of a tensor's flattened data as a genuinely smaller 1-D
/// tensor.
///
/// The whole point of ZeRO is that a rank only *retains* its own slice, so this really
/// allocates `end - start` elements rather than rescaling the original. (The source
/// tensor is read in full here because partitioning happens once, at setup, from a
/// materialised parameter map.)
///
/// # Errors
///
/// Returns an error when the range is out of bounds or the tensor is not
/// `f32`-readable.
pub fn slice_flat(tensor: &Tensor, start: usize, end: usize) -> Result<Tensor> {
    let data = tensor.data_f32()?;
    if end > data.len() || start > end {
        return Err(TrustformersError::invalid_input(format!(
            "shard range {start}..{end} is out of bounds for a tensor of {} elements",
            data.len()
        )));
    }
    let shard = data[start..end].to_vec();
    let len = shard.len();
    Tensor::from_vec(shard, &[len])
}

/// Reassembles a full tensor from shards presented in rank order.
///
/// This is the inverse of [`partition_parameters`]/[`partition_gradients`]: the shards
/// are concatenated and reshaped back to `global_shape`.
///
/// # Errors
///
/// Returns an error when the concatenated length does not match `global_shape`.
pub fn gather_shards(shards: &[Tensor], global_shape: &[usize]) -> Result<Tensor> {
    let mut values = Vec::new();
    for shard in shards {
        values.extend_from_slice(&shard.data_f32()?);
    }

    let expected: usize = global_shape.iter().product();
    if values.len() != expected {
        return Err(TrustformersError::invalid_input(format!(
            "gathered {} elements but the global shape {global_shape:?} needs {expected}",
            values.len()
        )));
    }

    Tensor::from_vec(values, global_shape)
}

/// Gather parameters from all devices
pub fn gather_parameters(
    partitions: &mut HashMap<String, ParameterPartition>,
    mp_context: &ModelParallelContext,
) -> Result<HashMap<String, Tensor>> {
    let mut gathered = HashMap::new();

    for (name, partition) in partitions.iter_mut() {
        partition.gather(mp_context)?;
        if let Some(full_param) = &partition.full_parameter {
            gathered.insert(name.clone(), full_param.clone());
        }
    }

    Ok(gathered)
}

/// Partition gradients across devices for ZeRO Stage 2+
pub fn partition_gradients(
    gradients: &HashMap<String, Tensor>,
    world_size: usize,
    rank: usize,
) -> Result<HashMap<String, GradientBuffer>> {
    let mut buffers = HashMap::new();

    for (name, grad) in gradients {
        let shape = grad.shape();
        let (start_idx, end_idx) = shard_range(shape.iter().product::<usize>(), world_size, rank)?;
        let local_gradient = slice_flat(grad, start_idx, end_idx)?;

        let partition_info = PartitionInfo {
            rank,
            world_size,
            start_idx,
            end_idx,
            global_shape: shape.to_vec(),
            local_shape: local_gradient.shape().to_vec(),
        };

        let buffer = GradientBuffer::new(name.clone(), local_gradient, partition_info);
        buffers.insert(name.clone(), buffer);
    }

    Ok(buffers)
}

/// All-gather gradients from all devices
pub fn all_gather_gradients(
    buffers: &HashMap<String, GradientBuffer>,
    mp_context: &ModelParallelContext,
) -> Result<HashMap<String, Tensor>> {
    let mut gathered = HashMap::new();

    for (name, buffer) in buffers {
        let distributed_tensor = trustformers_core::parallel::DistributedTensor::new(
            buffer.local_gradient.clone(),
            buffer.partition_info.global_shape.clone(),
            trustformers_core::parallel::TensorPartition {
                split_dim: 0,
                start_idx: buffer.partition_info.start_idx,
                end_idx: buffer.partition_info.end_idx,
                num_partitions: buffer.partition_info.world_size,
                partition_rank: buffer.partition_info.rank,
            },
            buffer.partition_info.rank,
        );

        let full_gradient = mp_context.all_gather(&distributed_tensor)?;
        gathered.insert(name.clone(), full_gradient);
    }

    Ok(gathered)
}

/// Reduce-scatter gradients across devices
pub fn reduce_scatter_gradients(
    gradients: &HashMap<String, Tensor>,
    mp_context: &ModelParallelContext,
) -> Result<HashMap<String, Tensor>> {
    let mut scattered = HashMap::new();

    for (name, grad) in gradients {
        let scattered_grad = mp_context.reduce_scatter(grad, 0)?;
        scattered.insert(name.clone(), scattered_grad);
    }

    Ok(scattered)
}

/// Calculate optimal bucket size for gradient communication
pub fn calculate_bucket_size(
    parameter_sizes: &[usize],
    target_bucket_size: usize,
) -> Vec<Vec<usize>> {
    let mut buckets = Vec::new();
    let mut current_bucket = Vec::new();
    let mut current_size = 0;

    for (i, &size) in parameter_sizes.iter().enumerate() {
        if current_size + size > target_bucket_size && !current_bucket.is_empty() {
            buckets.push(current_bucket);
            current_bucket = Vec::new();
            current_size = 0;
        }

        current_bucket.push(i);
        current_size += size;
    }

    if !current_bucket.is_empty() {
        buckets.push(current_bucket);
    }

    buckets
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_zero_state_creation() {
        let state = ZeROState::new();
        assert_eq!(state.step, 0);
        assert!(state.optimizer_states.is_empty());
        assert!(state.gradient_partitions.is_empty());
        assert!(state.parameter_partitions.is_empty());
    }

    #[test]
    fn test_parameter_group() {
        let mut group = ParameterGroup::new("test_group".to_string(), vec!["param1".to_string()]);
        let tensor = Tensor::ones(&[2, 2]).expect("Failed to create tensor");
        group.add_parameter("param1".to_string(), tensor);

        assert_eq!(group.parameter_names.len(), 1);
        assert_eq!(group.local_parameters.len(), 1);
        assert!(group.memory_usage() > 0);
    }

    #[test]
    fn test_gradient_buffer() {
        let tensor = Tensor::ones(&[2, 2]).expect("Failed to create tensor");
        let partition_info = PartitionInfo::default();
        let mut buffer = GradientBuffer::new("test_grad".to_string(), tensor, partition_info);

        let grad = Tensor::ones(&[2, 2]).expect("Failed to create tensor");
        buffer.accumulate(&grad).expect("Operation failed in test");

        assert_eq!(buffer.accumulation_steps, 1);
        assert!(buffer.get_accumulated().is_some());
    }

    #[test]
    fn test_partition_parameters() {
        let mut params = HashMap::new();
        params.insert(
            "param1".to_string(),
            Tensor::ones(&[4, 4]).expect("Failed to create tensor"),
        );
        params.insert(
            "param2".to_string(),
            Tensor::ones(&[2, 2]).expect("Failed to create tensor"),
        );

        let partitions = partition_parameters(&params, 2, 0).expect("Operation failed in test");
        assert_eq!(partitions.len(), 2);

        for partition in partitions.values() {
            assert_eq!(partition.partition_info.world_size, 2);
            assert_eq!(partition.partition_info.rank, 0);
        }
    }

    #[test]
    fn test_calculate_bucket_size() {
        let sizes = vec![100, 200, 150, 300, 50];
        let buckets = calculate_bucket_size(&sizes, 400);

        assert!(!buckets.is_empty());

        // Check that no bucket exceeds the target size
        for bucket in &buckets {
            let bucket_size: usize = bucket.iter().map(|&i| sizes[i]).sum();
            assert!(bucket_size <= 400 || bucket.len() == 1); // Single large item allowed
        }
    }

    /// Regression: partitioning used to scale the whole tensor by `1/world_size` on
    /// every rank, so no rank ever held less than the full parameter.
    #[test]
    fn shards_are_disjoint_and_cover_every_element() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "w".to_string(),
            Tensor::from_vec((0..10).map(|i| i as f32).collect(), &[10]).expect("tensor"),
        );

        let world_size = 4;
        let mut total = 0_usize;
        let mut seen = Vec::new();
        for rank in 0..world_size {
            let partitions =
                partition_parameters(&parameters, world_size, rank).expect("partition");
            let shard = &partitions.get("w").expect("shard").local_shard;
            let values = shard.data_f32().expect("values");
            total += values.len();
            seen.extend(values);
        }

        assert_eq!(total, 10, "shard lengths must sum to the element count");
        let expected: Vec<f32> = (0..10).map(|i| i as f32).collect();
        assert_eq!(seen, expected, "rank r must own slice r, in order");
    }

    /// Every rank's shard must be strictly smaller than the global tensor — that is
    /// the entire memory saving ZeRO promises.
    #[test]
    fn each_shard_is_smaller_than_the_global_tensor() {
        let mut parameters = HashMap::new();
        parameters.insert(
            "w".to_string(),
            Tensor::from_vec(vec![1.0_f32; 16], &[4, 4]).expect("tensor"),
        );

        let partitions = partition_parameters(&parameters, 4, 1).expect("partition");
        let partition = partitions.get("w").expect("partition");
        assert_eq!(partition.local_shard.len(), 4);
        assert_eq!(partition.partition_info.start_idx, 4);
        assert_eq!(partition.partition_info.end_idx, 8);
        assert_eq!(partition.partition_info.global_shape, vec![4, 4]);
        assert_eq!(partition.partition_info.local_shape, vec![4]);
    }

    /// Gathering the shards in rank order must reproduce the original tensor exactly.
    #[test]
    fn gather_round_trips_the_original_tensor() {
        let original =
            Tensor::from_vec((0..12).map(|i| i as f32 * 0.5).collect(), &[3, 4]).expect("tensor");
        let mut parameters = HashMap::new();
        parameters.insert("w".to_string(), original.clone());

        let world_size = 5;
        let shards: Vec<Tensor> = (0..world_size)
            .map(|rank| {
                partition_parameters(&parameters, world_size, rank)
                    .expect("partition")
                    .get("w")
                    .expect("shard")
                    .local_shard
                    .clone()
            })
            .collect();

        let gathered = gather_shards(&shards, &[3, 4]).expect("gather");
        assert_eq!(gathered.shape(), original.shape());
        assert_eq!(
            gathered.data_f32().expect("values"),
            original.data_f32().expect("values")
        );
    }

    /// Gradients shard exactly like parameters.
    #[test]
    fn gradients_are_sharded_not_rescaled() {
        let mut gradients = HashMap::new();
        gradients.insert(
            "g".to_string(),
            Tensor::from_vec(vec![2.0_f32; 8], &[8]).expect("tensor"),
        );

        let buffers = partition_gradients(&gradients, 2, 0).expect("partition");
        let local = &buffers.get("g").expect("buffer").local_gradient;
        assert_eq!(local.len(), 4, "the shard must be half the gradient");
        assert!(
            local.data_f32().expect("values").iter().all(|v| (*v - 2.0).abs() < 1e-6),
            "values must be copied verbatim, not scaled by 1/world_size"
        );
    }

    /// A trailing rank with nothing left to own gets an empty shard, not a copy.
    #[test]
    fn trailing_ranks_may_own_nothing() {
        assert_eq!(shard_range(3, 4, 3).expect("range"), (3, 3));
        assert_eq!(shard_range(3, 4, 0).expect("range"), (0, 1));
        assert!(shard_range(3, 0, 0).is_err());
        assert!(shard_range(3, 2, 2).is_err());
    }
}
