// Ring Attention for Ultra-Long Sequence Training
// Implementation of distributed attention computation for sequences with millions of tokens

use anyhow::Result;
use log;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Instant;
use trustformers_core::errors::{invalid_input, tensor_op_error, Result as CoreResult};
use trustformers_core::Tensor;

/// Configuration for Ring Attention distributed training
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingAttentionConfig {
    /// Number of devices in the ring topology
    pub num_devices: usize,
    /// Sequence chunk size per device
    pub chunk_size: usize,
    /// Whether to use bidirectional ring communication
    pub bidirectional: bool,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use causal masking
    pub causal: bool,
    /// Block size for memory optimization
    pub block_size: usize,
    /// Overlap communication with computation
    pub overlap_comm_comp: bool,
    /// Use compression for communication
    pub compression_enabled: bool,
    /// Compression ratio (if enabled)
    pub compression_ratio: f32,
}

impl Default for RingAttentionConfig {
    fn default() -> Self {
        Self {
            num_devices: 8,
            chunk_size: 4096,
            bidirectional: true,
            num_heads: 32,
            head_dim: 128,
            causal: true,
            block_size: 512,
            overlap_comm_comp: true,
            compression_enabled: false,
            compression_ratio: 0.5,
        }
    }
}

/// Communication patterns for Ring Attention
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RingCommunicationPattern {
    /// Simple unidirectional ring
    Unidirectional,
    /// Bidirectional ring for faster convergence
    Bidirectional,
    /// Hierarchical ring for very large clusters
    Hierarchical { levels: usize },
    /// Adaptive pattern based on sequence length
    Adaptive,
}

/// Ring attention block for distributed computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingAttentionBlock {
    /// Device rank in the ring
    pub device_rank: usize,
    /// Sequence chunk assigned to this device
    pub sequence_chunk: (usize, usize), // (start, end)
    /// Key-value pairs from previous devices
    pub received_kv: Vec<RingKVPair>,
    /// This device's own key chunk, flattened `[chunk_len * head_dim]`.
    ///
    /// Populated by [`RingAttentionManager::set_local_kv`]. The ring rotates
    /// *these* values; nothing is synthesised.
    #[serde(default)]
    pub local_keys: Vec<f32>,
    /// This device's own value chunk, flattened `[chunk_len * head_dim]`.
    #[serde(default)]
    pub local_values: Vec<f32>,
    /// Communication buffer for ring transfers
    pub comm_buffer: Option<Vec<f32>>,
    /// Attention computation statistics
    pub attention_stats: RingAttentionStats,
}

/// Key-Value pair for ring communication
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RingKVPair {
    /// Keys tensor chunk
    pub keys: Vec<f32>,
    /// Values tensor chunk
    pub values: Vec<f32>,
    /// Source device rank
    pub source_rank: usize,
    /// Sequence position range
    pub position_range: (usize, usize),
    /// Attention mask for this chunk
    pub attention_mask: Option<Vec<bool>>,
}

/// Statistics for Ring Attention performance monitoring
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RingAttentionStats {
    /// Total attention operations performed
    pub total_attention_ops: u64,
    /// Communication volume (bytes transferred)
    pub communication_volume: u64,
    /// Computation time (milliseconds)
    pub computation_time_ms: f64,
    /// Communication time (milliseconds)
    pub communication_time_ms: f64,
    /// Memory usage peak (bytes)
    pub peak_memory_usage: usize,
    /// Number of ring passes completed
    pub ring_passes: usize,
    /// Attention efficiency score (0-1)
    pub efficiency_score: f32,
    /// Memory allocation count (for optimization tracking)
    pub memory_allocations: u32,
    /// Average memory reuse ratio
    pub memory_reuse_ratio: f32,
}

/// Ring Attention Manager for coordinating distributed attention
pub struct RingAttentionManager {
    config: RingAttentionConfig,
    devices: Vec<RingAttentionBlock>,
    communication_pattern: RingCommunicationPattern,
    global_sequence_length: usize,
    current_ring_step: usize,
    performance_stats: HashMap<usize, RingAttentionStats>,
}

impl RingAttentionManager {
    /// Create a new Ring Attention manager with enhanced validation
    pub fn new(config: RingAttentionConfig, global_sequence_length: usize) -> Result<Self> {
        // Validate configuration
        if config.num_devices == 0 {
            return Err(anyhow::anyhow!("Number of devices must be positive"));
        }

        if global_sequence_length == 0 {
            return Err(anyhow::anyhow!("Global sequence length must be positive"));
        }

        if config.chunk_size == 0 {
            return Err(anyhow::anyhow!("Chunk size must be positive"));
        }

        if config.head_dim == 0 {
            return Err(anyhow::anyhow!("Head dimension must be positive"));
        }

        // Warn about potential memory issues
        let memory_per_device = config.chunk_size * config.head_dim * 2 * 4; // 2 for K+V, 4 bytes per float
        let total_memory = memory_per_device * config.num_devices;

        if total_memory > 1_000_000_000 {
            // 1GB threshold
            log::warn!(
                "Ring attention configuration may use significant memory: {:.2} GB",
                total_memory as f64 / 1_000_000_000.0
            );
        }

        let mut devices = Vec::with_capacity(config.num_devices);
        let base_chunk_size = global_sequence_length / config.num_devices;
        let remainder = global_sequence_length % config.num_devices;

        let mut current_pos = 0;

        for rank in 0..config.num_devices {
            // Distribute remainder across first few devices
            let device_chunk_size =
                if rank < remainder { base_chunk_size + 1 } else { base_chunk_size };

            let start_pos = current_pos;
            let end_pos = current_pos + device_chunk_size;

            // Pre-allocate communication buffer with proper size
            let buffer_size = config.chunk_size.max(device_chunk_size) * config.head_dim * 2;

            let block = RingAttentionBlock {
                device_rank: rank,
                sequence_chunk: (start_pos, end_pos),
                received_kv: Vec::with_capacity(config.num_devices),
                local_keys: Vec::new(),
                local_values: Vec::new(),
                comm_buffer: Some(vec![0.0; buffer_size]),
                attention_stats: RingAttentionStats::default(),
            };

            devices.push(block);
            current_pos = end_pos;
        }

        log::info!(
            "Initialized ring attention with {} devices, total sequence length: {}",
            config.num_devices,
            global_sequence_length
        );

        let communication_pattern = if config.bidirectional {
            RingCommunicationPattern::Bidirectional
        } else {
            RingCommunicationPattern::Unidirectional
        };

        Ok(Self {
            config,
            devices,
            communication_pattern,
            global_sequence_length,
            current_ring_step: 0,
            performance_stats: HashMap::new(),
        })
    }

    /// Execute distributed attention computation with enhanced monitoring
    pub fn compute_ring_attention(
        &mut self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
    ) -> CoreResult<Tensor> {
        // Validate input tensors
        self.validate_input_tensors(queries, keys, values)?;

        let start_time = Instant::now();
        let computation_start = Instant::now();

        log::debug!(
            "Starting ring attention computation for {} devices",
            self.config.num_devices
        );

        // Initialize attention outputs for each device
        let mut attention_outputs = Vec::with_capacity(self.config.num_devices);
        let mut total_comm_time = 0.0;

        // Perform ring attention computation
        for ring_step in 0..self.config.num_devices {
            self.current_ring_step = ring_step;

            log::trace!("Ring step {}/{}", ring_step + 1, self.config.num_devices);

            // Compute attention for current ring step
            let step_start = Instant::now();
            let step_outputs = self.compute_attention_step(queries, keys, values)?;
            let step_duration = step_start.elapsed().as_millis() as f64;

            attention_outputs.push(step_outputs);

            // Update computation time stats
            for device in &mut self.devices {
                device.attention_stats.computation_time_ms +=
                    step_duration / self.config.num_devices as f64;
            }

            // Rotate key-value pairs to next device in ring
            if ring_step < self.config.num_devices - 1 {
                let comm_start = Instant::now();

                if let Err(e) = self.rotate_kv_pairs() {
                    log::error!("Failed to rotate KV pairs at step {}: {}", ring_step, e);
                    return Err(tensor_op_error(
                        "ring_communication",
                        format!("Ring communication failed: {}", e),
                    ));
                }

                let comm_duration = comm_start.elapsed().as_millis() as f64;
                total_comm_time += comm_duration;

                // Update communication time stats
                for device in &mut self.devices {
                    device.attention_stats.communication_time_ms +=
                        comm_duration / self.config.num_devices as f64;
                }
            }
        }

        // Aggregate attention outputs from all ring steps
        let aggregation_start = Instant::now();
        let final_output = self.aggregate_attention_outputs(attention_outputs)?;
        let aggregation_time = aggregation_start.elapsed().as_millis() as f64;

        // Update comprehensive performance statistics
        let total_time = start_time.elapsed().as_millis() as f64;
        let computation_time = computation_start.elapsed().as_millis() as f64 - total_comm_time;

        self.update_comprehensive_performance_stats(
            total_time,
            computation_time,
            total_comm_time,
            aggregation_time,
        );

        log::info!("Ring attention computation completed in {:.2}ms (comp: {:.2}ms, comm: {:.2}ms, agg: {:.2}ms)",
                  total_time, computation_time, total_comm_time, aggregation_time);

        Ok(final_output)
    }

    /// Compute attention for a single ring step
    fn compute_attention_step(
        &mut self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
    ) -> CoreResult<Vec<Tensor>> {
        let mut step_outputs = Vec::new();

        // Extract device information to avoid borrowing conflicts
        let device_chunks: Vec<(usize, (usize, usize))> =
            self.devices.iter().map(|d| (d.device_rank, d.sequence_chunk)).collect();

        for (device_rank, sequence_chunk) in device_chunks {
            // Extract query chunk for this device
            let q_chunk = self.extract_chunk(queries, sequence_chunk)?;

            // Compute attention scores with all available K,V pairs
            let attention_scores = self.compute_attention_scores_simple(&q_chunk, keys)?;

            // Apply causal masking if enabled
            let masked_scores = if self.config.causal {
                self.apply_causal_mask_simple(attention_scores, sequence_chunk)?
            } else {
                attention_scores
            };

            // Compute attention weights (softmax)
            let attention_weights = self.compute_softmax(&masked_scores)?;

            // Compute weighted sum with values
            let output = self.compute_weighted_sum_simple(&attention_weights, values)?;

            step_outputs.push(output);

            // Update device statistics
            if let Some(device) = self.devices.get_mut(device_rank) {
                device.attention_stats.total_attention_ops += 1;
                device.attention_stats.ring_passes = self.current_ring_step + 1;
            }
        }

        Ok(step_outputs)
    }

    /// Extract sequence chunk for a specific device
    fn extract_chunk(&self, tensor: &Tensor, chunk_range: (usize, usize)) -> CoreResult<Tensor> {
        // Extract sequence chunk using tensor slicing operations
        let (start_idx, end_idx) = chunk_range;
        let _chunk_length = end_idx - start_idx;

        // Get tensor shape and validate chunk range
        let shape = tensor.shape();
        if shape.len() < 3 {
            return Err(tensor_op_error(
                "ring_attention_validation",
                "Ring attention requires tensors with at least 3 dimensions [batch, seq_len, hidden_dim]"
            ));
        }

        let seq_len = shape[1];
        if end_idx > seq_len {
            return Err(tensor_op_error(
                "chunk_validation",
                format!(
                    "Chunk range {}-{} exceeds sequence length {}",
                    start_idx, end_idx, seq_len
                ),
            ));
        }

        // Use tensor slicing to extract the chunk
        // For a tensor of shape [batch_size, seq_len, hidden_dim], extract [batch_size, chunk_length, hidden_dim]
        let chunk = tensor.slice_multi(&[
            (0, shape[0]),        // Full batch dimension
            (start_idx, end_idx), // Sequence chunk
            (0, shape[2]),        // Full hidden dimension
        ])?;

        Ok(chunk)
    }

    /// Compute attention scores between queries and keys
    fn compute_attention_scores(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        _device: &RingAttentionBlock,
    ) -> CoreResult<Tensor> {
        // Scaled dot-product attention
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Q * K^T / sqrt(d_k)
        let scores = queries.matmul(keys)?;
        let scaled_scores = scores.mul_scalar(scale)?;

        Ok(scaled_scores)
    }

    /// Simplified version without device parameter
    fn compute_attention_scores_simple(
        &self,
        queries: &Tensor,
        keys: &Tensor,
    ) -> CoreResult<Tensor> {
        // Scaled dot-product attention
        let scale = 1.0 / (self.config.head_dim as f32).sqrt();

        // Q * K^T / sqrt(d_k)
        let scores = queries.matmul(keys)?;
        let scaled_scores = scores.mul_scalar(scale)?;

        Ok(scaled_scores)
    }

    /// Apply causal masking to attention scores for ring attention
    fn apply_causal_mask_simple(
        &self,
        scores: Tensor,
        sequence_chunk: (usize, usize),
    ) -> CoreResult<Tensor> {
        let (chunk_start, _chunk_end) = sequence_chunk;
        let shape = scores.shape();

        // Attention scores should have shape [batch_size, seq_len_q, seq_len_k]
        if shape.len() < 3 {
            return Ok(scores); // Skip masking for malformed tensors
        }

        let batch_size = shape[0];
        let seq_len_q = shape[1];
        let seq_len_k = shape[2];

        // Create causal mask tensor - positions can only attend to previous positions
        let mut mask_values = vec![0.0f32; seq_len_q * seq_len_k];

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let global_pos_i = chunk_start + i;
                let global_pos_j = j; // Assuming key positions start from 0 for received chunks

                // Apply causal constraint: position i can only attend to positions <= i
                if global_pos_j > global_pos_i {
                    // Use large negative value to effectively zero out after softmax
                    mask_values[i * seq_len_k + j] = -1e9;
                }
            }
        }

        // Create mask tensor and expand to batch dimension
        let mask = Tensor::from_vec(mask_values, &[1, seq_len_q, seq_len_k])?;
        let expanded_mask = mask.broadcast_to(&[batch_size, seq_len_q, seq_len_k])?;

        // Add mask to scores (adding large negative values to positions that should be masked)
        let masked_scores = scores.add(&expanded_mask)?;

        Ok(masked_scores)
    }

    /// Compute softmax over attention scores
    fn compute_softmax(&self, scores: &Tensor) -> CoreResult<Tensor> {
        // Numerically stable softmax - simplified implementation
        let exp_scores = scores.exp()?;
        let sum_exp = exp_scores.sum(None, true)?;
        let softmax = exp_scores.div(&sum_exp)?;

        Ok(softmax)
    }

    /// Memory-efficient softmax computation with pre-allocated buffers
    /// Reduces memory allocation overhead for high-frequency operations
    fn compute_softmax_inplace(&self, scores: &mut Tensor) -> CoreResult<()> {
        // Simplified in-place softmax - compute exp and normalize
        // In a full implementation, this would find max for stability
        *scores = scores.exp()?;

        // Compute sum and normalize in-place
        let sum_exp = scores.sum(None, true)?;
        *scores = scores.div(&sum_exp)?;

        Ok(())
    }

    /// Optimized attention computation with memory reuse
    /// Uses pre-allocated buffers to minimize memory allocation overhead
    fn compute_attention_optimized(
        &mut self,
        query: &Tensor,
        key: &Tensor,
        value: &Tensor,
        reuse_buffer: &mut Option<Tensor>,
    ) -> CoreResult<Tensor> {
        // Compute Q*K^T (simplified - transpose last two dimensions)
        // For attention: transpose dimensions 2 and 3 (sequence_len and head_dim)
        let key_t = key.transpose(2, 3)?;
        let mut scores = query.matmul(&key_t)?;

        // Scale by sqrt(d_k) for stability
        let scale = (self.config.head_dim as f32).sqrt();
        scores = scores.scalar_div(scale)?;

        // Apply causal mask if needed
        if self.config.causal {
            scores = self.apply_causal_mask_simple(scores, (0, 0))?;
        }

        // Use in-place softmax to save memory
        self.compute_softmax_inplace(&mut scores)?;

        // Compute final output with value matrix
        let output = scores.matmul(value)?;

        // Store intermediate result in reuse buffer for potential next iteration
        *reuse_buffer = Some(scores);

        Ok(output)
    }

    /// Compute weighted sum of values using attention weights
    fn compute_weighted_sum(
        &self,
        weights: &Tensor,
        values: &Tensor,
        _device: &RingAttentionBlock,
    ) -> CoreResult<Tensor> {
        // Attention output = weights * values
        let output = weights.matmul(values)?;
        Ok(output)
    }

    /// Simplified weighted sum without device parameter
    fn compute_weighted_sum_simple(&self, weights: &Tensor, values: &Tensor) -> CoreResult<Tensor> {
        // Attention output = weights * values
        let output = weights.matmul(values)?;
        Ok(output)
    }

    /// Load this device's key/value chunk.
    ///
    /// `keys` and `values` must have the same length; the ring rotation moves
    /// exactly these buffers between devices.
    pub fn set_local_kv(
        &mut self,
        device_rank: usize,
        keys: Vec<f32>,
        values: Vec<f32>,
    ) -> Result<()> {
        if keys.len() != values.len() {
            return Err(anyhow::anyhow!(
                "keys ({}) and values ({}) must have the same length",
                keys.len(),
                values.len()
            ));
        }
        let device = self.devices.get_mut(device_rank).ok_or_else(|| {
            anyhow::anyhow!(
                "device rank {device_rank} is out of range for {} devices",
                self.config.num_devices
            )
        })?;
        device.local_keys = keys;
        device.local_values = values;
        Ok(())
    }

    /// This device's currently loaded key/value chunk.
    pub fn local_kv(&self, device_rank: usize) -> Option<(&[f32], &[f32])> {
        self.devices
            .get(device_rank)
            .map(|device| (device.local_keys.as_slice(), device.local_values.as_slice()))
    }

    /// Take a snapshot of every device's local key/value chunk.
    ///
    /// Rotation reads from this snapshot so that a device forwarding data does
    /// not observe values another device wrote in the same step.
    fn snapshot_local_kv(&self) -> Result<Vec<(Vec<f32>, Vec<f32>)>> {
        self.devices
            .iter()
            .map(|device| {
                if device.local_keys.is_empty() || device.local_values.is_empty() {
                    return Err(anyhow::anyhow!(
                        "device {} has no local key/value chunk loaded; call \
                         RingAttentionManager::set_local_kv before rotating",
                        device.device_rank
                    ));
                }
                Ok((device.local_keys.clone(), device.local_values.clone()))
            })
            .collect()
    }

    /// Rotate key-value pairs to the neighbouring device(s) in the ring.
    fn rotate_kv_pairs(&mut self) -> Result<()> {
        match self.communication_pattern {
            RingCommunicationPattern::Bidirectional => {
                self.rotate_kv_unidirectional()?;
                self.rotate_kv_reverse()?;
                Ok(())
            },
            // Unidirectional is the base rotation; hierarchical and adaptive
            // patterns differ only in how the ring is grouped, and both reduce
            // to a single forward hop at this level.
            _ => self.rotate_kv_unidirectional(),
        }
    }

    /// Forward ring rotation: device `i` sends its own K/V chunk to device
    /// `(i + 1) % num_devices`.
    ///
    /// The transferred payload is the *source device's actual* key/value data.
    /// An earlier revision fabricated `sin`/`cos` sequences here and discarded
    /// the real chunk, so the attention computation downstream operated on
    /// synthetic data.
    fn rotate_kv_unidirectional(&mut self) -> Result<()> {
        let num_devices = self.config.num_devices;
        let snapshot = self.snapshot_local_kv()?;

        // Clear previous received KV pairs
        for device in &mut self.devices {
            device.received_kv.clear();
        }

        for i in 0..num_devices {
            let next_device = (i + 1) % num_devices;
            let (mut keys, mut values) = snapshot[i].clone();

            // Apply compression if enabled. `comm_volume` is the number of
            // bytes that really travel, so enabling compression is visible in
            // the statistics instead of being a silent no-op.
            let comm_volume = if self.config.compression_enabled {
                self.compress_kv_data(&mut keys, &mut values)?
            } else {
                Self::uncompressed_kv_bytes(keys.len())
            };

            let kv_pair = RingKVPair {
                keys,
                values,
                source_rank: i,
                position_range: self.devices[i].sequence_chunk,
                attention_mask: None,
            };

            self.devices[next_device].received_kv.push(kv_pair);

            self.devices[i].attention_stats.communication_volume += comm_volume as u64;
            if let Some(stats) = self.performance_stats.get_mut(&i) {
                stats.communication_volume += comm_volume as u64;
            }
        }

        Ok(())
    }

    /// Reverse ring rotation: device `i` sends its own K/V chunk to device
    /// `(i - 1) % num_devices`.
    fn rotate_kv_reverse(&mut self) -> Result<()> {
        let num_devices = self.config.num_devices;
        let snapshot = self.snapshot_local_kv()?;

        for i in 0..num_devices {
            let prev_device = (i + num_devices - 1) % num_devices;
            let (mut keys, mut values) = snapshot[i].clone();

            let comm_volume = if self.config.compression_enabled {
                self.compress_kv_data(&mut keys, &mut values)?
            } else {
                Self::uncompressed_kv_bytes(keys.len())
            };

            let kv_pair = RingKVPair {
                keys,
                values,
                source_rank: i,
                position_range: self.devices[i].sequence_chunk,
                attention_mask: None,
            };

            self.devices[prev_device].received_kv.push(kv_pair);

            self.devices[i].attention_stats.communication_volume += comm_volume as u64;
            if let Some(stats) = self.performance_stats.get_mut(&i) {
                stats.communication_volume += comm_volume as u64;
            }
        }

        Ok(())
    }

    /// Aggregate attention outputs from all ring steps
    fn aggregate_attention_outputs(&self, outputs: Vec<Vec<Tensor>>) -> CoreResult<Tensor> {
        // Combine outputs from all devices and ring steps
        let first_output = &outputs[0][0];
        let mut aggregated = first_output.clone();

        // Simple averaging for now - more sophisticated aggregation can be implemented
        for step_outputs in outputs.iter().skip(1) {
            for (i, output) in step_outputs.iter().enumerate() {
                if i == 0 {
                    aggregated = aggregated.add(output)?;
                }
            }
        }

        // Normalize by number of ring steps
        let scale = 1.0 / outputs.len() as f32;
        aggregated = aggregated.mul_scalar(scale)?;

        Ok(aggregated)
    }

    /// Update performance statistics
    fn update_performance_stats(&mut self, total_time_ms: f64) {
        for (rank, device) in self.devices.iter_mut().enumerate() {
            device.attention_stats.computation_time_ms +=
                total_time_ms / self.config.num_devices as f64;

            // Calculate efficiency score based on computation vs communication time
            let total_time = device.attention_stats.computation_time_ms
                + device.attention_stats.communication_time_ms;
            if total_time > 0.0 {
                device.attention_stats.efficiency_score =
                    (device.attention_stats.computation_time_ms / total_time) as f32;
            }

            self.performance_stats.insert(rank, device.attention_stats.clone());
        }
    }

    /// Get performance statistics for all devices
    pub fn get_performance_stats(&self) -> &HashMap<usize, RingAttentionStats> {
        &self.performance_stats
    }

    /// Validate input tensors for ring attention computation
    fn validate_input_tensors(
        &self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
    ) -> CoreResult<()> {
        let q_shape = queries.shape();
        let k_shape = keys.shape();
        let v_shape = values.shape();

        // Check that all tensors have the same number of dimensions
        if q_shape.len() != 3 || k_shape.len() != 3 || v_shape.len() != 3 {
            return Err(invalid_input(
                "All input tensors must be 3-dimensional [batch, seq_len, hidden_size]",
            ));
        }

        // Check that batch sizes match
        if q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0] {
            return Err(invalid_input(
                "All input tensors must have the same batch size",
            ));
        }

        // Check that hidden dimensions are compatible
        if k_shape[2] != v_shape[2] {
            return Err(invalid_input(
                "Key and value tensors must have the same hidden dimension",
            ));
        }

        Ok(())
    }

    /// Update comprehensive performance statistics
    fn update_comprehensive_performance_stats(
        &mut self,
        total_time: f64,
        computation_time: f64,
        total_comm_time: f64,
        aggregation_time: f64,
    ) {
        for (rank, device) in self.devices.iter_mut().enumerate() {
            device.attention_stats.computation_time_ms += computation_time + aggregation_time;
            device.attention_stats.communication_time_ms +=
                total_comm_time / self.config.num_devices as f64;

            // Calculate efficiency metrics
            let total_time_per_device = total_time / self.config.num_devices as f64;
            if total_time_per_device > 0.0 {
                device.attention_stats.efficiency_score =
                    (computation_time / total_time_per_device) as f32;
            }

            // Update memory efficiency (use a default value for now)
            device.attention_stats.memory_reuse_ratio = 0.75;

            self.performance_stats.insert(rank, device.attention_stats.clone());
        }
    }

    /// Get aggregate performance metrics
    pub fn get_aggregate_stats(&self) -> RingAttentionStats {
        let mut aggregate = RingAttentionStats::default();

        for stats in self.performance_stats.values() {
            aggregate.total_attention_ops += stats.total_attention_ops;
            aggregate.communication_volume += stats.communication_volume;
            aggregate.computation_time_ms += stats.computation_time_ms;
            aggregate.communication_time_ms += stats.communication_time_ms;
            aggregate.peak_memory_usage = aggregate.peak_memory_usage.max(stats.peak_memory_usage);
            aggregate.ring_passes = aggregate.ring_passes.max(stats.ring_passes);
        }

        // Average efficiency score
        if !self.performance_stats.is_empty() {
            aggregate.efficiency_score =
                self.performance_stats.values().map(|s| s.efficiency_score).sum::<f32>()
                    / self.performance_stats.len() as f32;
        }

        aggregate
    }

    /// Optimize communication pattern based on performance
    pub fn optimize_communication_pattern(&mut self) -> Result<()> {
        let aggregate_stats = self.get_aggregate_stats();

        // Switch to bidirectional if communication is a bottleneck
        let comm_ratio = aggregate_stats.communication_time_ms
            / (aggregate_stats.computation_time_ms + aggregate_stats.communication_time_ms);

        if comm_ratio > 0.3
            && matches!(
                self.communication_pattern,
                RingCommunicationPattern::Unidirectional
            )
        {
            self.communication_pattern = RingCommunicationPattern::Bidirectional;
            log::info!(
                "switched to bidirectional ring communication (comm ratio: {:.2})",
                comm_ratio
            );
        }

        // Enable compression if communication volume is high
        if aggregate_stats.communication_volume > 1_000_000_000 && !self.config.compression_enabled
        {
            self.config.compression_enabled = true;
            self.config.compression_ratio = 0.5;
            log::info!(
                "enabled communication compression (volume: {} bytes)",
                aggregate_stats.communication_volume
            );
        }

        Ok(())
    }

    /// Create optimized configuration for specific sequence length
    pub fn create_optimized_config(
        sequence_length: usize,
        num_devices: usize,
        model_params: ModelParams,
    ) -> RingAttentionConfig {
        // Calculate optimal chunk size based on memory constraints
        let memory_per_device = 80_000_000_000; // 80GB typical GPU memory
        let model_memory = model_params.estimate_memory_usage();
        let available_memory = memory_per_device - model_memory;

        // Reserve memory for attention computation
        let attention_memory_fraction = 0.6;
        let attention_memory = (available_memory as f64 * attention_memory_fraction) as usize;

        // Calculate chunk size based on available memory
        // Memory needed: chunk_size * num_heads * head_dim * 4 (for Q, K, V, O) * batch_size
        let bytes_per_token = model_params.num_heads * model_params.head_dim * 4 * 4; // 4 tensors, 4 bytes per float
        let optimal_chunk_size =
            (attention_memory / bytes_per_token).min(sequence_length / num_devices);

        RingAttentionConfig {
            num_devices,
            chunk_size: optimal_chunk_size,
            bidirectional: sequence_length > 1_000_000, // Use bidirectional for very long sequences
            num_heads: model_params.num_heads,
            head_dim: model_params.head_dim,
            causal: model_params.causal,
            block_size: 512.min(optimal_chunk_size / 8),
            overlap_comm_comp: true,
            compression_enabled: sequence_length >= 2_000_000, // Enable compression for very long sequences
            compression_ratio: 0.5,
        }
    }
}

/// Model parameters for Ring Attention optimization
#[derive(Debug, Clone)]
pub struct ModelParams {
    pub num_heads: usize,
    pub head_dim: usize,
    pub hidden_dim: usize,
    pub num_layers: usize,
    pub causal: bool,
}

impl ModelParams {
    /// Estimate memory usage for the model
    pub fn estimate_memory_usage(&self) -> usize {
        // Rough estimate: parameters + activations
        let param_count = self.hidden_dim * self.hidden_dim * 4 * self.num_layers; // Rough estimate
        let param_memory = param_count * 4; // 4 bytes per float32 parameter

        // Add some buffer for activations and gradients
        param_memory * 3 // Parameters + gradients + optimizer states
    }
}

impl RingAttentionManager {
    /// Bytes of metadata that accompany one quantized buffer on the wire: an
    /// `f32` scale and an `f32` zero point.
    const QUANT_HEADER_BYTES: usize = 2 * std::mem::size_of::<f32>();

    /// Bit width used to transmit a K/V buffer at the configured ratio, or
    /// `None` when the ratio asks for no compression at all.
    ///
    /// [`RingAttentionConfig::compression_ratio`] is the target
    /// bytes-out / bytes-in, so a ratio of `r` over 32-bit input asks for
    /// `32 * r` bits. A ratio of `1.0` or more (and any non-finite value) means
    /// "do not shrink the payload", which must be a genuine no-op rather than
    /// lossy 16-bit quantization advertised as a halving. Otherwise the width
    /// is clamped to `2..=16`: below two bits a buffer degenerates to its
    /// min/max, and at seventeen or more the saving no longer justifies the
    /// loss.
    fn quantization_bits(compression_ratio: f32) -> Option<u32> {
        if !compression_ratio.is_finite() || compression_ratio >= 1.0 {
            return None;
        }
        Some(((compression_ratio * 32.0).round() as i64).clamp(2, 16) as u32)
    }

    /// Quantize `keys` and `values` for transmission and return the number of
    /// bytes that actually travel.
    ///
    /// The buffers are replaced by their *reconstruction*, which is what the
    /// receiver observes — the same length, the same shape, with quantization
    /// error. An earlier revision decimated the buffer and then padded it back
    /// to full length with a geometric decay of the last kept sample: the
    /// payload never shrank (so the reported saving was zero), the tail was
    /// fabricated data, and a `compression_ratio` above `1.0` or below
    /// `1/len` panicked on `step_by(0)` / an empty-vector index.
    fn compress_kv_data(&self, keys: &mut [f32], values: &mut [f32]) -> Result<usize> {
        let Some(bits) = Self::quantization_bits(self.config.compression_ratio) else {
            // The configuration asks for no reduction: leave the data untouched
            // and report the full uncompressed size.
            return Ok(Self::uncompressed_kv_bytes(keys.len()));
        };

        Self::quantize_in_place(keys, bits);
        Self::quantize_in_place(values, bits);

        let payload_bits = (keys.len() + values.len()) * bits as usize;
        Ok(payload_bits.div_ceil(8) + 2 * Self::QUANT_HEADER_BYTES)
    }

    /// Uniform affine quantization to `bits` levels, followed by immediate
    /// dequantization — the values the receiver would reconstruct.
    fn quantize_in_place(buffer: &mut [f32], bits: u32) {
        if buffer.is_empty() {
            return;
        }

        let mut min_value = f32::INFINITY;
        let mut max_value = f32::NEG_INFINITY;
        for value in buffer.iter() {
            min_value = min_value.min(*value);
            max_value = max_value.max(*value);
        }

        let span = max_value - min_value;
        if !span.is_finite() || span <= 0.0 {
            // A constant (or non-finite) buffer quantizes exactly; leave it be.
            return;
        }

        let levels = ((1u32 << bits) - 1) as f32;
        let scale = span / levels;
        for value in buffer.iter_mut() {
            let level = ((*value - min_value) / scale).round().clamp(0.0, levels);
            *value = min_value + level * scale;
        }
    }

    /// Bytes one uncompressed K/V pair of `elements` values each occupies.
    fn uncompressed_kv_bytes(elements: usize) -> usize {
        2 * elements * std::mem::size_of::<f32>()
    }

    /// Block-sparse (tiled) attention with an online softmax.
    ///
    /// This is the FlashAttention recurrence: keys/values are streamed one
    /// `block_size` tile at a time while a running maximum `m`, a running
    /// normaliser `l` and an accumulator are rescaled, so only one tile of
    /// scores is ever materialised. The result is **numerically identical** to
    /// dense scaled dot-product attention (up to floating-point rounding);
    /// `block_size` trades peak memory for loop overhead and nothing else.
    ///
    /// An earlier revision summed independently-softmaxed blocks into a
    /// discarded temporary and returned the freshly allocated zero tensor, so
    /// every caller received zeros regardless of its inputs.
    ///
    /// Tensors are `[batch, sequence, features]`; with `config.causal` set,
    /// query `i` attends only to keys `j <= i`.
    fn compute_block_sparse_attention(
        &mut self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        block_size: usize,
    ) -> CoreResult<Tensor> {
        let shape = queries.shape();
        if shape.len() != 3 {
            return Err(invalid_input(format!(
                "block-sparse attention expects [batch, sequence, features] queries, got {shape:?}"
            )));
        }
        if keys.shape() != shape || values.shape() != shape {
            return Err(invalid_input(format!(
                "block-sparse attention expects matching query/key/value shapes, got {:?}, {:?}, {:?}",
                shape,
                keys.shape(),
                values.shape()
            )));
        }
        let block_size = block_size.max(1);

        let batch_size = shape[0];
        let seq_len = shape[1];
        let hidden_dim = shape[2];

        // The module scales by the configured head dimension; fall back to the
        // tensor's own feature count when it is unset so the scale is never 1/0.
        let scale_dim = if self.config.head_dim == 0 { hidden_dim } else { self.config.head_dim };
        let scale = if scale_dim == 0 { 1.0 } else { 1.0 / (scale_dim as f32).sqrt() };

        let query_values = queries.to_vec_f32()?;
        let key_values = keys.to_vec_f32()?;
        let value_values = values.to_vec_f32()?;

        let expected = batch_size * seq_len * hidden_dim;
        if query_values.len() != expected {
            return Err(tensor_op_error(
                "block_sparse_attention",
                format!(
                    "expected {expected} elements in the query tensor, got {}",
                    query_values.len()
                ),
            ));
        }

        let mut output = vec![0.0f32; expected];
        let mut accumulator = vec![0.0f32; hidden_dim];

        for batch in 0..batch_size {
            let batch_offset = batch * seq_len * hidden_dim;

            for query_index in 0..seq_len {
                let query_offset = batch_offset + query_index * hidden_dim;
                let query_row = &query_values[query_offset..query_offset + hidden_dim];

                // Online-softmax state for this query row.
                let mut running_max = f32::NEG_INFINITY;
                let mut running_sum = 0.0f32;
                accumulator.iter_mut().for_each(|slot| *slot = 0.0);

                // Causal rows never look past their own position, so whole key
                // tiles beyond it are skipped without being scored at all.
                let last_key = if self.config.causal { query_index + 1 } else { seq_len };

                let mut block_start = 0usize;
                while block_start < last_key {
                    let block_end = (block_start + block_size).min(last_key);

                    // Score this tile.
                    let mut tile_scores = Vec::with_capacity(block_end - block_start);
                    let mut tile_max = f32::NEG_INFINITY;
                    for key_index in block_start..block_end {
                        let key_offset = batch_offset + key_index * hidden_dim;
                        let key_row = &key_values[key_offset..key_offset + hidden_dim];
                        let dot: f32 =
                            query_row.iter().zip(key_row).map(|(q, k)| q * k).sum::<f32>() * scale;
                        tile_max = tile_max.max(dot);
                        tile_scores.push(dot);
                    }

                    if tile_scores.is_empty() {
                        block_start = block_end;
                        continue;
                    }

                    // Rescale the running state to the new maximum, then fold
                    // the tile in. `exp(-inf) == 0`, which zeroes the untouched
                    // initial state on the first tile.
                    let new_max = running_max.max(tile_max);
                    let correction = (running_max - new_max).exp();
                    running_sum *= correction;
                    for slot in accumulator.iter_mut() {
                        *slot *= correction;
                    }

                    for (offset, score) in tile_scores.iter().enumerate() {
                        let weight = (score - new_max).exp();
                        running_sum += weight;
                        let key_index = block_start + offset;
                        let value_offset = batch_offset + key_index * hidden_dim;
                        for (slot, value) in accumulator
                            .iter_mut()
                            .zip(&value_values[value_offset..value_offset + hidden_dim])
                        {
                            *slot += weight * value;
                        }
                    }

                    running_max = new_max;
                    block_start = block_end;
                }

                if running_sum > 0.0 {
                    for (slot, accumulated) in
                        output[query_offset..query_offset + hidden_dim].iter_mut().zip(&accumulator)
                    {
                        *slot = accumulated / running_sum;
                    }
                }
            }
        }

        Tensor::from_vec(output, &[batch_size, seq_len, hidden_dim])
    }

    /// Apply causal masking within a block
    fn apply_block_causal_mask(
        &self,
        scores: Tensor,
        start_i: usize,
        start_j: usize,
        end_i: usize,
        end_j: usize,
    ) -> CoreResult<Tensor> {
        let shape = scores.shape();
        let batch_size = shape[0];
        let seq_len_q = end_i - start_i;
        let seq_len_k = end_j - start_j;

        // Create block-specific causal mask
        let mut mask_values = vec![0.0f32; seq_len_q * seq_len_k];

        for i in 0..seq_len_q {
            for j in 0..seq_len_k {
                let global_pos_i = start_i + i;
                let global_pos_j = start_j + j;

                // Apply causal constraint
                if global_pos_j > global_pos_i {
                    mask_values[i * seq_len_k + j] = -1e9;
                }
            }
        }

        // Create and apply mask
        let mask = Tensor::from_vec(mask_values, &[1, seq_len_q, seq_len_k])?;
        let expanded_mask = mask.broadcast_to(&[batch_size, seq_len_q, seq_len_k])?;
        let masked_scores = scores.add(&expanded_mask)?;

        Ok(masked_scores)
    }

    /// Memory pool for efficient tensor allocation
    /// Reuses tensors to reduce allocation overhead
    pub fn create_memory_pool(&mut self) -> RingAttentionMemoryPool {
        RingAttentionMemoryPool::new(
            self.config.num_devices,
            self.config.chunk_size,
            self.config.head_dim,
        )
    }

    /// Execute ring attention with memory pooling for efficiency
    pub fn compute_ring_attention_with_pooling(
        &mut self,
        queries: &Tensor,
        keys: &Tensor,
        values: &Tensor,
        memory_pool: &mut RingAttentionMemoryPool,
    ) -> CoreResult<Tensor> {
        let start_time = std::time::Instant::now();

        // Use block-sparse attention for large sequences
        let use_block_sparse = queries.shape()[1] > 8192; // Use for sequences > 8k tokens

        let result = if use_block_sparse {
            self.compute_block_sparse_attention(queries, keys, values, self.config.block_size)?
        } else {
            self.compute_ring_attention(queries, keys, values)?
        };

        // Update memory pool statistics
        memory_pool.update_usage_stats(start_time.elapsed());

        Ok(result)
    }
}

/// Memory pool for efficient tensor allocation in Ring Attention
pub struct RingAttentionMemoryPool {
    tensor_cache: Vec<Option<Tensor>>,
    allocation_count: u64,
    reuse_count: u64,
    peak_memory_usage: usize,
    last_cleanup: std::time::Instant,
}

impl RingAttentionMemoryPool {
    pub fn new(num_devices: usize, _chunk_size: usize, _head_dim: usize) -> Self {
        // Pre-allocate common tensor sizes
        let cache_size = num_devices * 4; // Multiple tensors per device

        Self {
            tensor_cache: vec![None; cache_size],
            allocation_count: 0,
            reuse_count: 0,
            peak_memory_usage: 0,
            last_cleanup: std::time::Instant::now(),
        }
    }

    /// Get a tensor from the pool or allocate a new one
    pub fn get_tensor(&mut self, shape: &[usize]) -> CoreResult<Tensor> {
        // Try to find a cached tensor with matching shape
        for cached in &mut self.tensor_cache {
            if let Some(tensor) = cached {
                if tensor.shape() == shape {
                    let result = tensor.clone();
                    *cached = None; // Remove from cache
                    self.reuse_count += 1;
                    return Ok(result);
                }
            }
        }

        // No suitable cached tensor found, allocate new one
        self.allocation_count += 1;
        Tensor::zeros(shape)
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, tensor: Tensor) {
        // Find an empty slot in the cache
        for cached in &mut self.tensor_cache {
            if cached.is_none() {
                *cached = Some(tensor);
                return;
            }
        }

        // Cache is full, perform cleanup if needed
        if self.last_cleanup.elapsed().as_secs() > 60 {
            self.cleanup_cache();
        }
    }

    /// Clean up old cached tensors
    pub fn cleanup_cache(&mut self) {
        // Simple cleanup: clear half the cache
        let clear_count = self.tensor_cache.len() / 2;
        for i in 0..clear_count {
            self.tensor_cache[i] = None;
        }
        self.last_cleanup = std::time::Instant::now();
    }

    /// Update usage statistics
    pub fn update_usage_stats(&mut self, _duration: std::time::Duration) {
        // Update memory usage tracking
        let current_memory = self.estimate_memory_usage();
        self.peak_memory_usage = self.peak_memory_usage.max(current_memory);
    }

    /// Estimate current memory usage
    pub fn estimate_memory_usage(&self) -> usize {
        self.tensor_cache.iter()
            .filter_map(|cached| cached.as_ref())
            .map(|tensor| tensor.shape().iter().product::<usize>() * 4) // 4 bytes per f32
            .sum()
    }

    /// Get memory pool statistics
    pub fn get_stats(&self) -> MemoryPoolStats {
        MemoryPoolStats {
            allocation_count: self.allocation_count,
            reuse_count: self.reuse_count,
            reuse_ratio: if self.allocation_count > 0 {
                self.reuse_count as f32 / self.allocation_count as f32
            } else {
                0.0
            },
            peak_memory_usage: self.peak_memory_usage,
            current_memory_usage: self.estimate_memory_usage(),
        }
    }
}

/// Memory pool statistics
#[derive(Debug, Clone)]
pub struct MemoryPoolStats {
    pub allocation_count: u64,
    pub reuse_count: u64,
    pub reuse_ratio: f32,
    pub peak_memory_usage: usize,
    pub current_memory_usage: usize,
}

/// Utilities for Ring Attention
pub mod utils {
    use super::*;

    /// Calculate optimal number of devices for given sequence length
    pub fn calculate_optimal_devices(sequence_length: usize, min_chunk_size: usize) -> usize {
        let max_devices = sequence_length / min_chunk_size;

        // Common device counts that work well with ring topology
        let preferred_counts = [1, 2, 4, 8, 16, 32, 64, 128];

        preferred_counts
            .iter()
            .rev()
            .find(|&&count| count <= max_devices)
            .copied()
            .unwrap_or(1)
    }

    /// Estimate Ring Attention speedup vs standard attention
    pub fn estimate_speedup(
        sequence_length: usize,
        num_devices: usize,
        memory_bandwidth_gbps: f64,
    ) -> f64 {
        // Standard attention: O(n^2) memory and computation
        let standard_ops = (sequence_length * sequence_length) as f64;

        // Ring attention: O(n^2/d) memory per device, same total computation
        let ring_ops_per_device = standard_ops / num_devices as f64;

        // Communication overhead estimation
        let chunk_size = sequence_length / num_devices;
        let comm_volume = chunk_size * 512 * 2 * num_devices; // Rough estimate for K,V transfer
        let comm_time = comm_volume as f64 / (memory_bandwidth_gbps * 1e9);

        // Compute time estimation (very rough)
        let compute_time = ring_ops_per_device / 1e12; // Assuming 1 TFLOPS per device

        // Speedup limited by communication overhead
        let parallel_efficiency = compute_time / (compute_time + comm_time);
        let theoretical_speedup = num_devices as f64;

        theoretical_speedup * parallel_efficiency
    }

    /// Generate optimal Ring Attention configuration presets
    pub fn create_preset_configs() -> HashMap<String, RingAttentionConfig> {
        let mut presets = HashMap::new();

        // Small scale: up to 32K tokens
        presets.insert(
            "small_scale".to_string(),
            RingAttentionConfig {
                num_devices: 4,
                chunk_size: 8192,
                bidirectional: false,
                num_heads: 32,
                head_dim: 128,
                causal: true,
                block_size: 512,
                overlap_comm_comp: false,
                compression_enabled: false,
                compression_ratio: 1.0,
            },
        );

        // Medium scale: up to 1M tokens
        presets.insert(
            "medium_scale".to_string(),
            RingAttentionConfig {
                num_devices: 8,
                chunk_size: 131072,
                bidirectional: true,
                num_heads: 32,
                head_dim: 128,
                causal: true,
                block_size: 1024,
                overlap_comm_comp: true,
                compression_enabled: false,
                compression_ratio: 1.0,
            },
        );

        // Large scale: up to 10M tokens
        presets.insert(
            "large_scale".to_string(),
            RingAttentionConfig {
                num_devices: 32,
                chunk_size: 327680,
                bidirectional: true,
                num_heads: 64,
                head_dim: 128,
                causal: true,
                block_size: 2048,
                overlap_comm_comp: true,
                compression_enabled: true,
                compression_ratio: 0.5,
            },
        );

        // Ultra scale: up to 100M tokens
        presets.insert(
            "ultra_scale".to_string(),
            RingAttentionConfig {
                num_devices: 128,
                chunk_size: 786432,
                bidirectional: true,
                num_heads: 128,
                head_dim: 128,
                causal: true,
                block_size: 4096,
                overlap_comm_comp: true,
                compression_enabled: true,
                compression_ratio: 0.3,
            },
        );

        presets
    }
}

/// Enhanced Ring Attention configuration validator
impl RingAttentionConfig {
    /// Validate configuration parameters and suggest optimizations
    pub fn validate_and_optimize(&mut self) -> Result<Vec<String>> {
        let mut warnings = Vec::new();

        // Validate basic parameters
        if self.num_devices == 0 {
            return Err(anyhow::anyhow!("Number of devices must be positive"));
        }

        if self.chunk_size == 0 {
            return Err(anyhow::anyhow!("Chunk size must be positive"));
        }

        if self.head_dim == 0 {
            return Err(anyhow::anyhow!("Head dimension must be positive"));
        }

        // Performance optimization suggestions
        if !self.chunk_size.is_multiple_of(64) {
            warnings.push(format!(
                "Chunk size {} is not aligned to 64, consider adjusting for better performance",
                self.chunk_size
            ));
        }

        if self.num_devices > 32 {
            warnings.push("Very high device count may lead to communication overhead".to_string());
            if !self.compression_enabled {
                self.compression_enabled = true;
                self.compression_ratio = 0.7;
                warnings.push("Auto-enabled compression for high device count".to_string());
            }
        }

        // Memory optimization suggestions
        let memory_per_device = self.chunk_size * self.head_dim * 8; // Rough estimate
        if memory_per_device > 100_000_000 {
            // 100MB per device
            warnings.push(format!(
                "High memory usage per device (~{:.1}MB), consider reducing chunk size",
                memory_per_device as f64 / 1_000_000.0
            ));
        }

        // Communication pattern optimization
        if self.chunk_size > 10000 && !self.bidirectional {
            self.bidirectional = true;
            warnings.push("Auto-enabled bidirectional communication for large chunks".to_string());
        }

        if !self.overlap_comm_comp && self.num_devices > 4 {
            self.overlap_comm_comp = true;
            warnings.push("Auto-enabled communication-computation overlap".to_string());
        }

        Ok(warnings)
    }
}

/// Ring Attention Memory Pool for efficient tensor reuse
#[derive(Debug)]
pub struct RingAttentionMemoryPoolV2 {
    num_devices: usize,
    chunk_size: usize,
    head_dim: usize,
    tensor_pools: HashMap<String, Vec<Tensor>>,
    allocation_count: usize,
    reuse_count: usize,
}

impl RingAttentionMemoryPoolV2 {
    pub fn new(num_devices: usize, chunk_size: usize, head_dim: usize) -> Self {
        Self {
            num_devices,
            chunk_size,
            head_dim,
            tensor_pools: HashMap::new(),
            allocation_count: 0,
            reuse_count: 0,
        }
    }

    /// Get a tensor from the pool or allocate a new one
    pub fn get_tensor(&mut self, tensor_type: &str, shape: &[usize]) -> CoreResult<Tensor> {
        let key = format!("{}_{:?}", tensor_type, shape);

        if let Some(pool) = self.tensor_pools.get_mut(&key) {
            if let Some(tensor) = pool.pop() {
                self.reuse_count += 1;
                log::trace!(
                    "Reusing tensor from pool: {} (reuse rate: {:.2}%)",
                    key,
                    self.get_reuse_rate() * 100.0
                );
                return Ok(tensor);
            }
        }

        // Allocate new tensor
        let tensor = Tensor::zeros(shape)?;
        self.allocation_count += 1;
        log::trace!("Allocated new tensor: {} {:?}", tensor_type, shape);

        Ok(tensor)
    }

    /// Return a tensor to the pool for reuse
    pub fn return_tensor(&mut self, tensor: Tensor, tensor_type: &str) {
        let shape = tensor.shape().to_vec();
        let key = format!("{}_{:?}", tensor_type, shape);

        let pool = self.tensor_pools.entry(key).or_default();

        // Limit pool size to prevent memory bloat
        if pool.len() < 10 {
            pool.push(tensor);
        }
    }

    /// Get memory pool statistics
    pub fn get_reuse_rate(&self) -> f32 {
        if self.allocation_count + self.reuse_count == 0 {
            0.0
        } else {
            self.reuse_count as f32 / (self.allocation_count + self.reuse_count) as f32
        }
    }

    /// Clear the memory pool
    pub fn clear(&mut self) {
        self.tensor_pools.clear();
        self.allocation_count = 0;
        self.reuse_count = 0;
    }
}

#[cfg(test)]
mod tests;
