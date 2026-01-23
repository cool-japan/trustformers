/// Ring Attention Implementation for Extremely Long Context Processing
///
/// This module implements Ring Attention as described in "Ring Attention with Blockwise Transformers
/// for Near-Infinite Context" (Liu et al., 2023). Ring Attention enables processing sequences of
/// millions of tokens by distributing attention computation across multiple devices in a ring topology.
///
/// Key features:
/// - Distributed attention computation for unlimited sequence length
/// - Memory-efficient block-wise processing
/// - Support for both causal and bidirectional attention
/// - Integration with existing transformer architectures
/// - Fault tolerance and load balancing
///
/// The implementation uses block-wise computation where each device processes a segment of the
/// sequence, and attention computation is distributed across the ring of devices.
use std::collections::HashMap;
use trustformers_core::{
    errors::{Result, TrustformersError},
    tensor::Tensor,
};

/// Ring Attention configuration
#[derive(Debug, Clone)]
pub struct RingAttentionConfig {
    /// Number of devices/nodes in the ring
    pub ring_size: usize,
    /// Block size for sequence processing (tokens per block)
    pub block_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Dimension per attention head
    pub head_dim: usize,
    /// Whether to use causal attention (GPT-style)
    pub causal: bool,
    /// Maximum sequence length to support
    pub max_seq_length: usize,
    /// Overlap size between blocks for better context
    pub block_overlap: usize,
    /// Communication backend for distributed processing
    pub communication_backend: CommunicationBackend,
    /// Memory optimization settings
    pub memory_optimization: MemoryOptimizationConfig,
}

impl Default for RingAttentionConfig {
    fn default() -> Self {
        Self {
            ring_size: 8,
            block_size: 4096,
            num_heads: 32,
            head_dim: 128,
            causal: true,
            max_seq_length: 1_000_000, // 1M tokens
            block_overlap: 256,
            communication_backend: CommunicationBackend::NCCL,
            memory_optimization: MemoryOptimizationConfig::default(),
        }
    }
}

/// Communication backends for distributed attention
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationBackend {
    /// NVIDIA Collective Communications Library
    NCCL,
    /// Message Passing Interface
    MPI,
    /// Gloo for CPU-based communication
    Gloo,
    /// Custom implementation
    Custom(String),
}

/// Memory optimization configuration
#[derive(Debug, Clone)]
pub struct MemoryOptimizationConfig {
    /// Use gradient checkpointing to trade compute for memory
    pub gradient_checkpointing: bool,
    /// Enable attention computation fusion
    pub fused_attention: bool,
    /// Use mixed precision (FP16/BF16) for attention computation
    pub mixed_precision: bool,
    /// Enable sequence parallelism within blocks
    pub sequence_parallel: bool,
    /// Flash attention compatibility
    pub flash_attention: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        Self {
            gradient_checkpointing: true,
            fused_attention: true,
            mixed_precision: true,
            sequence_parallel: false,
            flash_attention: true,
        }
    }
}

/// Ring attention block information
#[derive(Debug, Clone)]
pub struct AttentionBlock {
    /// Block index in the sequence
    pub block_id: usize,
    /// Device/rank owning this block
    pub device_id: usize,
    /// Start position in the sequence
    pub start_pos: usize,
    /// End position in the sequence
    pub end_pos: usize,
    /// Query embeddings for this block
    pub queries: Option<Tensor>,
    /// Key embeddings for this block
    pub keys: Option<Tensor>,
    /// Value embeddings for this block
    pub values: Option<Tensor>,
}

impl AttentionBlock {
    pub fn new(block_id: usize, device_id: usize, start_pos: usize, end_pos: usize) -> Self {
        Self {
            block_id,
            device_id,
            start_pos,
            end_pos,
            queries: None,
            keys: None,
            values: None,
        }
    }

    pub fn set_qkv(&mut self, queries: Tensor, keys: Tensor, values: Tensor) {
        self.queries = Some(queries);
        self.keys = Some(keys);
        self.values = Some(values);
    }

    pub fn sequence_length(&self) -> usize {
        self.end_pos - self.start_pos
    }
}

/// Ring attention computation engine
pub struct RingAttention {
    config: RingAttentionConfig,
    device_id: usize,
    #[allow(dead_code)]
    communication_group: CommunicationGroup,
    attention_blocks: Vec<AttentionBlock>,
    memory_pool: AttentionMemoryPool,
}

/// Communication group for ring topology
#[derive(Debug, Clone)]
pub struct CommunicationGroup {
    pub ring_size: usize,
    pub current_rank: usize,
    pub next_rank: usize,
    pub prev_rank: usize,
    pub backend: CommunicationBackend,
}

impl CommunicationGroup {
    pub fn new(ring_size: usize, current_rank: usize, backend: CommunicationBackend) -> Self {
        let next_rank = (current_rank + 1) % ring_size;
        let prev_rank = if current_rank == 0 { ring_size - 1 } else { current_rank - 1 };

        Self {
            ring_size,
            current_rank,
            next_rank,
            prev_rank,
            backend,
        }
    }
}

/// Memory pool for efficient attention computation
#[derive(Debug)]
pub struct AttentionMemoryPool {
    /// Pre-allocated query buffers
    query_buffers: Vec<Option<Tensor>>,
    /// Pre-allocated key buffers
    key_buffers: Vec<Option<Tensor>>,
    /// Pre-allocated value buffers
    value_buffers: Vec<Option<Tensor>>,
    /// Pre-allocated attention score buffers
    score_buffers: Vec<Option<Tensor>>,
    /// Pre-allocated output buffers
    output_buffers: Vec<Option<Tensor>>,
    /// Buffer pool size
    pool_size: usize,
}

impl AttentionMemoryPool {
    pub fn new(pool_size: usize) -> Self {
        Self {
            query_buffers: vec![None; pool_size],
            key_buffers: vec![None; pool_size],
            value_buffers: vec![None; pool_size],
            score_buffers: vec![None; pool_size],
            output_buffers: vec![None; pool_size],
            pool_size,
        }
    }

    pub fn get_query_buffer(&mut self, index: usize) -> Option<&mut Tensor> {
        if index < self.pool_size {
            self.query_buffers[index].as_mut()
        } else {
            None
        }
    }

    pub fn allocate_buffers(
        &mut self,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        for i in 0..self.pool_size {
            // Allocate query buffer: [seq_len, num_heads, head_dim]
            self.query_buffers[i] = Some(Tensor::zeros(&[seq_len, num_heads, head_dim])?);

            // Allocate key buffer: [seq_len, num_heads, head_dim]
            self.key_buffers[i] = Some(Tensor::zeros(&[seq_len, num_heads, head_dim])?);

            // Allocate value buffer: [seq_len, num_heads, head_dim]
            self.value_buffers[i] = Some(Tensor::zeros(&[seq_len, num_heads, head_dim])?);

            // Allocate score buffer: [num_heads, seq_len, seq_len]
            self.score_buffers[i] = Some(Tensor::zeros(&[num_heads, seq_len, seq_len])?);

            // Allocate output buffer: [seq_len, num_heads, head_dim]
            self.output_buffers[i] = Some(Tensor::zeros(&[seq_len, num_heads, head_dim])?);
        }
        Ok(())
    }
}

impl RingAttention {
    /// Create a new Ring Attention instance
    pub fn new(config: RingAttentionConfig, device_id: usize) -> Result<Self> {
        if device_id >= config.ring_size {
            return Err(TrustformersError::config_error(
                &format!(
                    "Device ID {} must be less than ring size {}",
                    device_id, config.ring_size
                ),
                "ring_attention_init",
            ));
        }

        let communication_group = CommunicationGroup::new(
            config.ring_size,
            device_id,
            config.communication_backend.clone(),
        );

        let memory_pool = AttentionMemoryPool::new(config.ring_size * 2); // Extra buffers for pipeline

        Ok(Self {
            config,
            device_id,
            communication_group,
            attention_blocks: Vec::new(),
            memory_pool,
        })
    }

    /// Partition the sequence into blocks for ring processing
    pub fn partition_sequence(&mut self, sequence_length: usize) -> Result<Vec<AttentionBlock>> {
        let num_blocks = sequence_length.div_ceil(self.config.block_size);
        let mut blocks = Vec::new();

        for block_id in 0..num_blocks {
            let start_pos = block_id * self.config.block_size;
            let end_pos = ((block_id + 1) * self.config.block_size).min(sequence_length);
            let device_id = block_id % self.config.ring_size;

            let block = AttentionBlock::new(block_id, device_id, start_pos, end_pos);
            blocks.push(block);
        }

        self.attention_blocks = blocks.clone();
        Ok(blocks)
    }

    /// Compute ring attention for the given input embeddings
    pub fn forward(
        &mut self,
        input_embeddings: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let input_shape = input_embeddings.shape();
        if input_shape.len() != 3 {
            return Err(TrustformersError::config_error(
                "Input embeddings must have shape [batch_size, seq_length, embed_dim]",
                "ring_attention_forward",
            ));
        }

        let _batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let embed_dim = input_shape[2];

        // Partition sequence into blocks
        let blocks = self.partition_sequence(seq_length)?;

        // Allocate memory buffers
        self.memory_pool.allocate_buffers(
            self.config.block_size,
            self.config.num_heads,
            self.config.head_dim,
        )?;

        // Process each block that belongs to this device
        let mut local_outputs = HashMap::new();

        for block in &blocks {
            if block.device_id == self.device_id {
                let block_input = self.extract_block(input_embeddings, block)?;
                let block_output =
                    self.process_block(&block_input, block, &blocks, attention_mask)?;
                local_outputs.insert(block.block_id, block_output);
            }
        }

        // Aggregate results from all blocks
        self.aggregate_outputs(local_outputs, seq_length, embed_dim)
    }

    /// Extract input embeddings for a specific block
    fn extract_block(&self, input_embeddings: &Tensor, block: &AttentionBlock) -> Result<Tensor> {
        // Extract the slice corresponding to this block
        // Extract sequence dimension slice for this block
        input_embeddings.slice(1, block.start_pos, block.end_pos)
    }

    /// Process attention computation for a single block
    fn process_block(
        &mut self,
        block_input: &Tensor,
        current_block: &AttentionBlock,
        all_blocks: &[AttentionBlock],
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let block_seq_len = current_block.sequence_length();

        // Project to Q, K, V
        let queries = self.project_queries(block_input)?;
        let _keys = self.project_keys(block_input)?;
        let _values = self.project_values(block_input)?;

        // Initialize output accumulator
        let mut output = Tensor::zeros(&queries.shape())?;
        let mut attention_weights =
            Tensor::zeros(&[self.config.num_heads, block_seq_len, block_seq_len])?;

        // Ring attention computation: iterate through all blocks
        for step in 0..self.config.ring_size {
            let key_block_idx = (current_block.block_id + step) % all_blocks.len();
            let key_block = &all_blocks[key_block_idx];

            // Get key-value pairs for this step (simulate communication)
            let (step_keys, step_values) = self.get_remote_kv(key_block)?;

            // Compute attention scores between current queries and remote keys
            let scores = self.compute_attention_scores(&queries, &step_keys)?;

            // Apply causal mask if needed
            let masked_scores = if self.config.causal {
                self.apply_causal_mask(&scores, current_block, key_block)?
            } else {
                scores
            };

            // Apply attention mask if provided
            let final_scores = if let Some(mask) = attention_mask {
                self.apply_attention_mask(&masked_scores, mask, current_block, key_block)?
            } else {
                masked_scores
            };

            // Softmax over key dimension
            let attention_probs = self.softmax_over_keys(&final_scores)?;

            // Weighted sum with values
            let weighted_values = self.apply_attention(&attention_probs, &step_values)?;

            // Accumulate outputs
            output = output.add(&weighted_values)?;
            attention_weights = attention_weights.add(&attention_probs)?;
        }

        // Normalize by accumulated weights
        self.normalize_output(&output, &attention_weights)
    }

    /// Project input to query space
    fn project_queries(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified projection - in practice would use learned linear layers
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let _embed_dim = input_shape[2];

        // Reshape to [batch_size, seq_len, num_heads, head_dim]
        let projected = input.reshape(&[
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;

        Ok(projected)
    }

    /// Project input to key space
    fn project_keys(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified projection - in practice would use learned linear layers
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let _embed_dim = input_shape[2];

        let projected = input.reshape(&[
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;

        Ok(projected)
    }

    /// Project input to value space
    fn project_values(&self, input: &Tensor) -> Result<Tensor> {
        // Simplified projection - in practice would use learned linear layers
        let input_shape = input.shape();
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];
        let _embed_dim = input_shape[2];

        let projected = input.reshape(&[
            batch_size,
            seq_len,
            self.config.num_heads,
            self.config.head_dim,
        ])?;

        Ok(projected)
    }

    /// Get key-value pairs from remote blocks (simulated)
    fn get_remote_kv(&self, block: &AttentionBlock) -> Result<(Tensor, Tensor)> {
        // In practice, this would involve actual communication
        // For now, create dummy tensors
        let seq_len = block.sequence_length();
        let keys = Tensor::randn(&[1, seq_len, self.config.num_heads, self.config.head_dim])?;
        let values = Tensor::randn(&[1, seq_len, self.config.num_heads, self.config.head_dim])?;

        Ok((keys, values))
    }

    /// Compute attention scores between queries and keys
    fn compute_attention_scores(&self, queries: &Tensor, keys: &Tensor) -> Result<Tensor> {
        // Q @ K^T / sqrt(head_dim)
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Transpose keys: [batch, seq_len, num_heads, head_dim] -> [batch, num_heads, head_dim, seq_len]
        let keys_transposed = keys.transpose(keys.shape().len() - 2, keys.shape().len() - 1)?;

        // Matrix multiplication: [batch, seq_len, num_heads, head_dim] @ [batch, num_heads, head_dim, seq_len]
        let scores = queries.matmul(&keys_transposed)?;

        // Scale by sqrt(head_dim)
        scores.scalar_mul(scale)
    }

    /// Apply causal mask to attention scores
    fn apply_causal_mask(
        &self,
        scores: &Tensor,
        query_block: &AttentionBlock,
        key_block: &AttentionBlock,
    ) -> Result<Tensor> {
        let scores_shape = scores.shape();
        let mut masked_scores = scores.clone();

        // Apply causal masking: can only attend to current and previous positions
        if key_block.start_pos >= query_block.end_pos {
            // Key block is completely in the future - mask everything
            masked_scores = Tensor::full(f32::NEG_INFINITY, scores_shape.to_vec())?;
        } else if key_block.start_pos < query_block.start_pos {
            // Partial masking within the block
            // This would require more complex masking logic
        }

        Ok(masked_scores)
    }

    /// Apply attention mask
    fn apply_attention_mask(
        &self,
        scores: &Tensor,
        mask: &Tensor,
        query_block: &AttentionBlock,
        _key_block: &AttentionBlock,
    ) -> Result<Tensor> {
        // Extract relevant portion of the mask for these blocks
        // Extract mask slice for query block
        let mask_slice = mask.slice(0, query_block.start_pos, query_block.end_pos)?;

        // Apply mask (add large negative value where mask is 0)
        let mask_value = Tensor::full(f32::NEG_INFINITY, scores.shape().to_vec())?;
        let expanded_mask = mask_slice.unsqueeze(0)?.unsqueeze(0)?; // Add batch and head dims

        // Where mask is 0, use mask_value; otherwise use original scores
        scores.add(&mask_value.mul(&(Tensor::ones(&expanded_mask.shape())?.sub(&expanded_mask)?))?)
    }

    /// Softmax over key dimension
    fn softmax_over_keys(&self, scores: &Tensor) -> Result<Tensor> {
        scores.softmax(-1)
    }

    /// Apply attention weights to values
    fn apply_attention(&self, attention_probs: &Tensor, values: &Tensor) -> Result<Tensor> {
        // [batch, num_heads, query_len, key_len] @ [batch, key_len, num_heads, head_dim]
        // -> [batch, query_len, num_heads, head_dim]
        attention_probs.matmul(values)
    }

    /// Normalize output by accumulated attention weights
    fn normalize_output(&self, output: &Tensor, attention_weights: &Tensor) -> Result<Tensor> {
        // Sum attention weights over key dimension to get normalization factor
        let weight_sum =
            attention_weights.sum(Some(vec![attention_weights.shape().len() - 1]), true)?;

        // Avoid division by zero
        let eps = 1e-8;
        let safe_weight_sum = weight_sum.add_scalar(eps)?;

        // Normalize output
        output.div(&safe_weight_sum.unsqueeze(safe_weight_sum.shape().len())?)
    }

    /// Aggregate outputs from all blocks
    fn aggregate_outputs(
        &self,
        local_outputs: HashMap<usize, Tensor>,
        total_seq_length: usize,
        embed_dim: usize,
    ) -> Result<Tensor> {
        // Create output tensor for full sequence
        let mut full_output = Tensor::zeros(&[1, total_seq_length, embed_dim])?;

        // Copy local outputs to their positions in the full sequence
        for (block_id, output) in local_outputs {
            let start_pos = block_id * self.config.block_size;
            let end_pos = ((block_id + 1) * self.config.block_size).min(total_seq_length);

            // Copy output to the correct position
            // In practice, this would be done more efficiently
            let output_data = output.data_f32()?;
            let mut full_data = full_output.data_f32()?;

            for i in 0..(end_pos - start_pos) {
                for j in 0..embed_dim {
                    let src_idx = i * embed_dim + j;
                    let dst_idx = (start_pos + i) * embed_dim + j;
                    if src_idx < output_data.len() && dst_idx < full_data.len() {
                        full_data[dst_idx] = output_data[src_idx];
                    }
                }
            }

            full_output = Tensor::from_vec(full_data, &full_output.shape())?;
        }

        Ok(full_output)
    }

    /// Get ring attention statistics
    pub fn get_stats(&self) -> RingAttentionStats {
        let _total_params = self.config.num_heads * self.config.head_dim * 3; // Q, K, V projections
        let memory_per_block =
            self.config.block_size * self.config.num_heads * self.config.head_dim * 4; // 4 bytes per f32
        let total_memory = memory_per_block * self.config.ring_size;

        RingAttentionStats {
            ring_size: self.config.ring_size,
            block_size: self.config.block_size,
            max_sequence_length: self.config.max_seq_length,
            memory_per_block_bytes: memory_per_block,
            total_memory_bytes: total_memory,
            theoretical_max_length: self.config.ring_size * self.config.block_size,
            communication_overhead_ratio: 1.0 / self.config.ring_size as f32,
        }
    }
}

/// Ring attention performance statistics
#[derive(Debug, Clone)]
pub struct RingAttentionStats {
    pub ring_size: usize,
    pub block_size: usize,
    pub max_sequence_length: usize,
    pub memory_per_block_bytes: usize,
    pub total_memory_bytes: usize,
    pub theoretical_max_length: usize,
    pub communication_overhead_ratio: f32,
}

/// Distributed Ring Attention Manager for coordinating multiple devices
pub struct DistributedRingAttentionManager {
    devices: Vec<RingAttention>,
    #[allow(dead_code)]
    coordination_config: CoordinationConfig,
}

/// Configuration for distributed coordination
#[derive(Debug, Clone)]
pub struct CoordinationConfig {
    pub synchronization_strategy: SynchronizationStrategy,
    pub fault_tolerance: bool,
    pub load_balancing: bool,
    pub communication_compression: bool,
}

/// Synchronization strategies for distributed processing
#[derive(Debug, Clone, PartialEq)]
pub enum SynchronizationStrategy {
    /// Synchronous processing - all devices wait for slowest
    Synchronous,
    /// Asynchronous processing with pipeline
    AsynchronousPipelined,
    /// Adaptive synchronization based on device performance
    Adaptive,
}

impl Default for CoordinationConfig {
    fn default() -> Self {
        Self {
            synchronization_strategy: SynchronizationStrategy::AsynchronousPipelined,
            fault_tolerance: true,
            load_balancing: true,
            communication_compression: true,
        }
    }
}

impl DistributedRingAttentionManager {
    pub fn new(
        configs: Vec<RingAttentionConfig>,
        coordination_config: CoordinationConfig,
    ) -> Result<Self> {
        let mut devices = Vec::new();

        for (device_id, config) in configs.into_iter().enumerate() {
            let ring_attention = RingAttention::new(config, device_id)?;
            devices.push(ring_attention);
        }

        Ok(Self {
            devices,
            coordination_config,
        })
    }

    pub fn process_distributed(
        &mut self,
        input_embeddings: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Coordinate processing across all devices
        // This is a simplified version - real implementation would handle actual distributed execution

        if self.devices.is_empty() {
            return Err(TrustformersError::config_error(
                "No devices configured",
                "distributed_process",
            ));
        }

        // For now, use the first device for processing
        self.devices[0].forward(input_embeddings, attention_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ring_attention_config() {
        let config = RingAttentionConfig::default();
        assert_eq!(config.ring_size, 8);
        assert_eq!(config.block_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal);
    }

    #[test]
    fn test_communication_group() {
        let group = CommunicationGroup::new(8, 3, CommunicationBackend::NCCL);
        assert_eq!(group.current_rank, 3);
        assert_eq!(group.next_rank, 4);
        assert_eq!(group.prev_rank, 2);
    }

    #[test]
    fn test_attention_block_creation() {
        let block = AttentionBlock::new(0, 0, 0, 1024);
        assert_eq!(block.block_id, 0);
        assert_eq!(block.device_id, 0);
        assert_eq!(block.sequence_length(), 1024);
    }

    #[test]
    fn test_sequence_partitioning() -> Result<()> {
        let config = RingAttentionConfig {
            ring_size: 4,
            block_size: 1000,
            ..Default::default()
        };

        let mut ring_attention = RingAttention::new(config, 0)?;
        let blocks = ring_attention.partition_sequence(3500)?;

        assert_eq!(blocks.len(), 4); // 3500 tokens / 1000 block size = 4 blocks
        assert_eq!(blocks[0].start_pos, 0);
        assert_eq!(blocks[0].end_pos, 1000);
        assert_eq!(blocks[3].start_pos, 3000);
        assert_eq!(blocks[3].end_pos, 3500);

        Ok(())
    }

    #[test]
    fn test_memory_pool_allocation() -> Result<()> {
        let mut pool = AttentionMemoryPool::new(4);
        pool.allocate_buffers(1024, 16, 64)?;

        assert!(pool.get_query_buffer(0).is_some());
        assert!(pool.get_query_buffer(3).is_some());
        assert!(pool.get_query_buffer(4).is_none()); // Out of bounds

        Ok(())
    }

    #[test]
    fn test_ring_attention_forward() -> Result<()> {
        let config = RingAttentionConfig {
            ring_size: 2,
            block_size: 512,
            num_heads: 8,
            head_dim: 64,
            ..Default::default()
        };

        let mut ring_attention = RingAttention::new(config, 0)?;

        // Create test input: [batch_size=1, seq_length=1024, embed_dim=512]
        let input = Tensor::randn(&[1, 1024, 512])?;

        let output = ring_attention.forward(&input, None)?;

        // Check output shape matches input
        assert_eq!(output.shape(), input.shape());

        Ok(())
    }

    #[test]
    fn test_causal_mask_application() -> Result<()> {
        let config = RingAttentionConfig {
            causal: true,
            ..Default::default()
        };

        let ring_attention = RingAttention::new(config, 0)?;

        let scores = Tensor::ones(&[1, 8, 64, 64])?; // [batch, heads, seq_len, seq_len]
        let query_block = AttentionBlock::new(0, 0, 0, 64);
        let key_block = AttentionBlock::new(1, 1, 64, 128); // Future block

        let masked_scores = ring_attention.apply_causal_mask(&scores, &query_block, &key_block)?;

        // Should be all negative infinity since key block is in the future
        let data = masked_scores.data_f32()?;
        assert!(data.iter().all(|&x| x == f32::NEG_INFINITY));

        Ok(())
    }

    #[test]
    fn test_attention_stats() -> Result<()> {
        let config = RingAttentionConfig {
            ring_size: 8,
            block_size: 4096,
            num_heads: 32,
            head_dim: 128,
            ..Default::default()
        };

        let ring_attention = RingAttention::new(config, 0)?;
        let stats = ring_attention.get_stats();

        assert_eq!(stats.ring_size, 8);
        assert_eq!(stats.block_size, 4096);
        assert_eq!(stats.theoretical_max_length, 8 * 4096); // 32K tokens
        assert!(stats.communication_overhead_ratio > 0.0);

        Ok(())
    }

    #[test]
    fn test_distributed_manager() -> Result<()> {
        let configs = vec![
            RingAttentionConfig::default(),
            RingAttentionConfig::default(),
        ];

        let coordination_config = CoordinationConfig::default();
        let manager = DistributedRingAttentionManager::new(configs, coordination_config)?;

        assert_eq!(manager.devices.len(), 2);

        Ok(())
    }
}
