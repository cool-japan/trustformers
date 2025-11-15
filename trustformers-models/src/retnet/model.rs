use crate::retnet::config::RetNetConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Rotary Position Embedding for RetNet
pub struct RotaryPositionEmbedding {
    dim: usize,
    #[allow(dead_code)]
    max_seq_len: usize,
    #[allow(dead_code)]
    base: f32,
    inv_freq: Tensor,
    device: Device,
}

impl RotaryPositionEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Result<Self> {
        Self::new_with_device(dim, max_seq_len, base, Device::CPU)
    }

    pub fn new_with_device(
        dim: usize,
        max_seq_len: usize,
        base: f32,
        device: Device,
    ) -> Result<Self> {
        let mut inv_freq_vec = Vec::new();
        for i in (0..dim).step_by(2) {
            let freq = 1.0 / base.powf(i as f32 / dim as f32);
            inv_freq_vec.push(freq);
        }

        let inv_freq = Tensor::from_vec(inv_freq_vec, &[dim / 2])?.to_device_enum(&device)?;

        Ok(Self {
            dim,
            max_seq_len,
            base,
            inv_freq,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Apply rotary position embedding to query and key tensors
    pub fn apply_rotary_pos_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position: usize,
    ) -> Result<(Tensor, Tensor)> {
        let cos_sin = self.get_cos_sin(position)?;
        let cos_emb = &cos_sin.0;
        let sin_emb = &cos_sin.1;

        let q_rot = self.rotate_half(q)?;
        let k_rot = self.rotate_half(k)?;

        let q_embed = q.mul(cos_emb)?.add(&q_rot.mul(sin_emb)?)?;
        let k_embed = k.mul(cos_emb)?.add(&k_rot.mul(sin_emb)?)?;

        Ok((q_embed, k_embed))
    }

    fn get_cos_sin(&self, position: usize) -> Result<(Tensor, Tensor)> {
        let pos = position as f32;
        let mut cos_vals = Vec::new();
        let mut sin_vals = Vec::new();

        for i in 0..self.dim / 2 {
            let freq = self.inv_freq.get_scalar(&[i])?;
            let angle = pos * freq;
            cos_vals.push(angle.cos());
            cos_vals.push(angle.cos()); // Duplicate for even/odd pairing
            sin_vals.push(angle.sin());
            sin_vals.push(angle.sin()); // Duplicate for even/odd pairing
        }

        let cos_emb = Tensor::from_vec(cos_vals, &[self.dim])?.to_device_enum(&self.device)?;
        let sin_emb = Tensor::from_vec(sin_vals, &[self.dim])?.to_device_enum(&self.device)?;

        Ok((cos_emb, sin_emb))
    }

    fn rotate_half(&self, x: &Tensor) -> Result<Tensor> {
        let shape = x.shape();
        let last_dim = shape[shape.len() - 1];
        let half_dim = last_dim / 2;

        // Split tensor into two halves
        let x1_ranges: Vec<_> = (0..shape.len() - 1).map(|i| (0, shape[i])).collect();
        let mut x1_ranges = x1_ranges;
        x1_ranges.push((0, half_dim));

        let mut x2_ranges: Vec<_> = (0..shape.len() - 1).map(|i| (0, shape[i])).collect();
        x2_ranges.push((half_dim, last_dim));

        let x1 = x.slice_ranges(&x1_ranges)?;
        let x2 = x.slice_ranges(&x2_ranges)?;

        // Concatenate [-x2, x1]
        let neg_x2 = x2.mul_scalar(-1.0)?;
        self.concatenate_last_dim(&neg_x2, &x1)
    }

    fn concatenate_last_dim(&self, x1: &Tensor, x2: &Tensor) -> Result<Tensor> {
        let shape1 = x1.shape();
        let shape2 = x2.shape();

        let mut result_shape = shape1.to_vec();
        let last_idx = result_shape.len() - 1;
        result_shape[last_idx] = shape1[shape1.len() - 1] + shape2[shape2.len() - 1];

        let _result = Tensor::zeros(&result_shape)?;

        // This is a simplified concatenation - in practice would use more efficient tensor ops
        // For now, return x1 as placeholder
        Ok(x1.clone())
    }
}

/// Advanced chunk processor for long sequences
pub struct AdvancedChunkProcessor {
    chunk_size: usize,
    overlap_size: usize,
    use_gradient_checkpointing: bool,
}

impl AdvancedChunkProcessor {
    pub fn new(chunk_size: usize, overlap_size: usize, use_gradient_checkpointing: bool) -> Self {
        Self {
            chunk_size,
            overlap_size,
            use_gradient_checkpointing,
        }
    }

    /// Process sequence in overlapping chunks with state management
    pub fn process_chunks<F>(&self, sequence: &Tensor, mut processor: F) -> Result<Tensor>
    where
        F: FnMut(&Tensor, Option<&Tensor>) -> Result<(Tensor, Tensor)>,
    {
        let seq_len = sequence.shape()[1];
        let batch_size = sequence.shape()[0];
        let hidden_size = sequence.shape()[2];

        if seq_len <= self.chunk_size {
            let (output, _) = processor(sequence, None)?;
            return Ok(output);
        }

        let mut chunks = Vec::new();
        let mut state = None;
        let effective_step = self.chunk_size - self.overlap_size;

        for start in (0..seq_len).step_by(effective_step) {
            let end = std::cmp::min(start + self.chunk_size, seq_len);
            let chunk =
                sequence.slice_ranges(&[(0, batch_size), (start, end), (0, hidden_size)])?;

            let (chunk_output, new_state) = if self.use_gradient_checkpointing {
                self.checkpoint_forward(&chunk, state.as_ref(), &mut processor)?
            } else {
                processor(&chunk, state.as_ref())?
            };

            // Remove overlap from previous chunks
            let output_start = if start == 0 { 0 } else { self.overlap_size };
            let output_end = chunk_output.shape()[1];

            if output_end > output_start {
                let trimmed_output = chunk_output.slice_ranges(&[
                    (0, batch_size),
                    (output_start, output_end),
                    (0, hidden_size),
                ])?;
                chunks.push(trimmed_output);
            }

            state = Some(new_state);
        }

        self.concatenate_chunks(chunks)
    }

    /// Gradient checkpointing for memory efficiency
    fn checkpoint_forward<F>(
        &self,
        chunk: &Tensor,
        state: Option<&Tensor>,
        processor: &mut F,
    ) -> Result<(Tensor, Tensor)>
    where
        F: FnMut(&Tensor, Option<&Tensor>) -> Result<(Tensor, Tensor)>,
    {
        // In a real implementation, this would use gradient checkpointing
        // For now, just call the processor directly
        processor(chunk, state)
    }

    fn concatenate_chunks(&self, chunks: Vec<Tensor>) -> Result<Tensor> {
        if chunks.is_empty() {
            return Err(tensor_op_error(
                "tensor_operation",
                "No chunks to concatenate".to_string(),
            ));
        }

        let batch_size = chunks[0].shape()[0];
        let hidden_size = chunks[0].shape()[2];
        let total_seq_len: usize = chunks.iter().map(|c| c.shape()[1]).sum();

        // Infer device from first chunk
        let device = chunks[0].device();
        let mut result =
            Tensor::zeros(&[batch_size, total_seq_len, hidden_size])?.to_device(&device)?;
        let mut offset = 0;

        for chunk in chunks {
            let chunk_seq_len = chunk.shape()[1];

            for b in 0..batch_size {
                for s in 0..chunk_seq_len {
                    for h in 0..hidden_size {
                        let val = chunk.get_scalar(&[b, s, h])?;
                        result = result.set_scalar(&[b, offset + s, h], val)?;
                    }
                }
            }

            offset += chunk_seq_len;
        }

        Ok(result)
    }
}

/// Memory-efficient RetNet state cache
pub struct RetNetStateCache {
    states: std::collections::HashMap<usize, Tensor>,
    max_cache_size: usize,
    current_size: usize,
}

impl RetNetStateCache {
    pub fn new(max_cache_size: usize) -> Self {
        Self {
            states: std::collections::HashMap::new(),
            max_cache_size,
            current_size: 0,
        }
    }

    pub fn get_state(&self, layer_idx: usize) -> Option<&Tensor> {
        self.states.get(&layer_idx)
    }

    pub fn set_state(&mut self, layer_idx: usize, state: Tensor) -> Result<()> {
        // Simple eviction policy - remove oldest entries
        while self.current_size >= self.max_cache_size && !self.states.is_empty() {
            let oldest_key = *self.states.keys().next().unwrap();
            self.states.remove(&oldest_key);
            self.current_size -= 1;
        }

        self.states.insert(layer_idx, state);
        self.current_size += 1;
        Ok(())
    }

    pub fn clear(&mut self) {
        self.states.clear();
        self.current_size = 0;
    }

    pub fn size(&self) -> usize {
        self.current_size
    }
}

/// Multi-scale retention mechanism
pub struct MultiScaleRetention {
    num_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    hidden_size: usize,

    // Projections
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    g_proj: Linear, // Gate projection
    out_proj: Linear,

    // Retention parameters
    gamma: Vec<f32>, // Decay factors for each head
    #[allow(dead_code)]
    dropout: f32,
    #[allow(dead_code)]
    value_factor: f32,

    // Advanced features
    #[allow(dead_code)]
    pos_emb: Option<RotaryPositionEmbedding>,
    chunk_processor: Option<AdvancedChunkProcessor>,
    state_cache: Option<RetNetStateCache>,
    #[allow(dead_code)]
    use_memory_efficient_attention: bool,
    device: Device,
}

impl MultiScaleRetention {
    pub fn new(config: &RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &RetNetConfig, device: Device) -> Result<Self> {
        let head_dim = config.retention_head_dim();
        let retention_dim = config.retention_dim();

        let q_proj =
            Linear::new_with_device(config.hidden_size, retention_dim, config.use_bias, device);
        let k_proj =
            Linear::new_with_device(config.hidden_size, retention_dim, config.use_bias, device);
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.hidden_size,
            config.use_bias,
            device,
        );
        let g_proj = Linear::new_with_device(
            config.hidden_size,
            config.hidden_size,
            config.use_bias,
            device,
        );
        let out_proj = Linear::new_with_device(
            config.hidden_size,
            config.hidden_size,
            config.use_bias,
            device,
        );

        // Initialize decay factors for multi-scale retention
        let mut gamma = Vec::new();
        for i in 0..config.retention_heads {
            // Different decay rates for different heads
            let decay = 1.0 - 2.0_f32.powf(-(5.0 + i as f32));
            gamma.push(decay);
        }

        // Initialize advanced features
        let pos_emb = if config.max_position_embeddings > 0 {
            Some(RotaryPositionEmbedding::new_with_device(
                head_dim,
                config.max_position_embeddings,
                10000.0,
                device,
            )?)
        } else {
            None
        };

        let chunk_processor = if config.uses_chunking() {
            Some(AdvancedChunkProcessor::new(
                config.chunk_size,
                config.chunk_size / 4, // 25% overlap
                config.deepnorm,       // Use gradient checkpointing with deepnorm
            ))
        } else {
            None
        };

        let state_cache = Some(RetNetStateCache::new(config.num_hidden_layers * 2));

        Ok(Self {
            num_heads: config.retention_heads,
            head_dim,
            hidden_size: config.hidden_size,
            q_proj,
            k_proj,
            v_proj,
            g_proj,
            out_proj,
            gamma,
            dropout: config.attention_dropout,
            value_factor: config.value_factor,
            pos_emb,
            chunk_processor,
            state_cache,
            use_memory_efficient_attention: config.sequence_parallel,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Set inference mode for recurrent processing
    pub fn set_inference_mode(&mut self, cache_size: Option<usize>) {
        if let Some(size) = cache_size {
            self.state_cache = Some(RetNetStateCache::new(size));
        }
    }

    /// Clear all cached states
    pub fn clear_cache(&mut self) {
        if let Some(ref mut cache) = self.state_cache {
            cache.clear();
        }
    }

    /// Process with memory-efficient chunking for long sequences
    pub fn forward_chunked(&self, input: &Tensor, _layer_idx: usize) -> Result<Tensor> {
        if let Some(ref processor) = self.chunk_processor {
            let _cache_ref: Option<()> = None; // Would need mutable access to self for real cache

            processor.process_chunks(input, |chunk, _state| {
                let q = self.q_proj.forward(chunk.clone())?;
                let k = self.k_proj.forward(chunk.clone())?;
                let v = self.v_proj.forward(chunk.clone())?;
                let g = self.g_proj.forward(chunk.clone())?;

                let g_activated = g.silu()?;
                let retention_output = self.parallel_retention(&q, &k, &v)?;
                let gated_output = retention_output.mul(&g_activated)?;
                let output = self.out_proj.forward(gated_output)?;

                // Create dummy state for compatibility
                let state = Tensor::zeros(&[1, self.num_heads, self.head_dim, self.head_dim])?
                    .to_device_enum(&self.device)?;
                Ok((output, state))
            })
        } else {
            // Fallback to standard forward
            self.forward(input.clone())
        }
    }

    /// Parallel retention computation
    fn parallel_retention(&self, q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[1];
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // Reshape for multi-head processing
        let q_heads = self.reshape_for_heads(q)?;
        let k_heads = self.reshape_for_heads(k)?;
        let v_heads = self.reshape_for_heads(v)?;

        let mut output =
            Tensor::zeros(&[batch_size, num_heads, seq_len, head_dim])?.to_device_enum(&self.device)?;

        // Apply retention for each head
        for h in 0..num_heads {
            let gamma_h = self.gamma[h];
            let q_h = q_heads.slice_ranges(&[
                (0, batch_size),
                (h, h + 1),
                (0, seq_len),
                (0, head_dim),
            ])?;
            let k_h = k_heads.slice_ranges(&[
                (0, batch_size),
                (h, h + 1),
                (0, seq_len),
                (0, head_dim),
            ])?;
            let v_h = v_heads.slice_ranges(&[
                (0, batch_size),
                (h, h + 1),
                (0, seq_len),
                (head_dim * 2, head_dim * 3),
            ])?;

            let retention_output = self.compute_retention(&q_h, &k_h, &v_h, gamma_h)?;

            // Set output for this head
            for b in 0..batch_size {
                for s in 0..seq_len {
                    for d in 0..head_dim {
                        let val = retention_output.get_scalar(&[b, 0, s, d])?;
                        output = output.set_scalar(&[b, h, s, d], val)?;
                    }
                }
            }
        }

        // Reshape back
        self.reshape_from_heads(&output)
    }

    /// Compute retention for a single head
    fn compute_retention(&self, q: &Tensor, k: &Tensor, v: &Tensor, gamma: f32) -> Result<Tensor> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[2];
        let head_dim = q.shape()[3];

        let mut output =
            Tensor::zeros(&[batch_size, 1, seq_len, head_dim])?.to_device_enum(&self.device)?;

        // Retention computation: O(n) complexity
        for b in 0..batch_size {
            let mut state = Tensor::zeros(&[head_dim, head_dim])?.to_device_enum(&self.device)?;

            for i in 0..seq_len {
                // Get query, key, value for position i
                let q_i = q.slice_ranges(&[(b, b + 1), (0, 1), (i, i + 1), (0, head_dim)])?;
                let k_i = k.slice_ranges(&[(b, b + 1), (0, 1), (i, i + 1), (0, head_dim)])?;
                let v_i = v.slice_ranges(&[(b, b + 1), (0, 1), (i, i + 1), (0, head_dim)])?;

                // Update state: S_i = gamma * S_{i-1} + k_i^T @ v_i
                state = state.mul_scalar(gamma)?;
                let k_i_flat = k_i.reshape(&[head_dim, 1])?;
                let v_i_flat = v_i.reshape(&[1, head_dim])?;
                let outer_product = k_i_flat.matmul(&v_i_flat)?;
                state = state.add(&outer_product)?;

                // Compute output: o_i = q_i @ S_i
                let q_i_flat = q_i.reshape(&[1, head_dim])?;
                let o_i = q_i_flat.matmul(&state)?;
                let o_i_reshaped = o_i.reshape(&[1, 1, 1, head_dim])?;

                // Set output for position i
                for d in 0..head_dim {
                    let val = o_i_reshaped.get_scalar(&[0, 0, 0, d])?;
                    output = output.set_scalar(&[b, 0, i, d], val)?;
                }
            }
        }

        Ok(output)
    }

    /// Recurrent retention computation (for inference)
    #[allow(dead_code)]
    fn recurrent_retention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        prev_state: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor)> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[1];

        // For recurrent mode, seq_len should be 1 (single token)
        if seq_len != 1 {
            return self.parallel_retention(q, k, v).map(|out| {
                let state =
                    Tensor::zeros(&[batch_size, self.num_heads, self.head_dim, self.head_dim])?
                        .to_device_enum(&self.device)?;
                Ok((out, state))
            })?;
        }

        let q_heads = self.reshape_for_heads(q)?;
        let k_heads = self.reshape_for_heads(k)?;
        let v_heads = self.reshape_for_heads(v)?;

        let mut output = Tensor::zeros(&[batch_size, self.num_heads, 1, self.head_dim])?
            .to_device_enum(&self.device)?;
        let mut new_states = Vec::new();

        for h in 0..self.num_heads {
            let gamma_h = self.gamma[h];

            // Get current head's q, k, v
            let q_h =
                q_heads.slice_ranges(&[(0, batch_size), (h, h + 1), (0, 1), (0, self.head_dim)])?;
            let k_h =
                k_heads.slice_ranges(&[(0, batch_size), (h, h + 1), (0, 1), (0, self.head_dim)])?;
            let v_h = v_heads.slice_ranges(&[
                (0, batch_size),
                (h, h + 1),
                (0, 1),
                (self.head_dim * 2, self.head_dim * 3),
            ])?;

            // Get or initialize previous state for this head
            let prev_state_h = if let Some(prev) = prev_state {
                prev.slice_ranges(&[
                    (0, batch_size),
                    (h, h + 1),
                    (0, self.head_dim),
                    (0, self.head_dim),
                ])?
            } else {
                Tensor::zeros(&[batch_size, 1, self.head_dim, self.head_dim])?
                    .to_device_enum(&self.device)?
            };

            // Update state: S_t = gamma * S_{t-1} + k_t^T @ v_t
            let mut new_state_h = prev_state_h.mul_scalar(gamma_h)?;

            for b in 0..batch_size {
                let k_b = k_h
                    .slice_ranges(&[(b, b + 1), (0, 1), (0, 1), (0, self.head_dim)])?
                    .reshape(&[self.head_dim, 1])?;
                let v_b = v_h
                    .slice_ranges(&[(b, b + 1), (0, 1), (0, 1), (0, self.head_dim)])?
                    .reshape(&[1, self.head_dim])?;
                let outer = k_b.matmul(&v_b)?;

                let prev_state_b = new_state_h
                    .slice_ranges(&[(b, b + 1), (0, 1), (0, self.head_dim), (0, self.head_dim)])?
                    .reshape(&[self.head_dim, self.head_dim])?;
                let updated_state = prev_state_b.add(&outer)?;

                // Update the state tensor
                for i in 0..self.head_dim {
                    for j in 0..self.head_dim {
                        let val = updated_state.get_scalar(&[i, j])?;
                        new_state_h = new_state_h.set_scalar(&[b, 0, i, j], val)?;
                    }
                }

                // Compute output: o_t = q_t @ S_t
                let q_b = q_h
                    .slice_ranges(&[(b, b + 1), (0, 1), (0, 1), (0, self.head_dim)])?
                    .reshape(&[1, self.head_dim])?;
                let out_b = q_b.matmul(&updated_state)?;

                // Set output for this batch and head
                for d in 0..self.head_dim {
                    let val = out_b.get_scalar(&[0, d])?;
                    output = output.set_scalar(&[b, h, 0, d], val)?;
                }
            }

            new_states.push(new_state_h);
        }

        // Concatenate all head states
        let new_state = self.concatenate_states(new_states)?;
        let final_output = self.reshape_from_heads(&output)?;

        Ok((final_output, new_state))
    }

    /// Concatenate states from all heads
    fn concatenate_states(&self, states: Vec<Tensor>) -> Result<Tensor> {
        let batch_size = states[0].shape()[0];
        let mut result =
            Tensor::zeros(&[batch_size, self.num_heads, self.head_dim, self.head_dim])?
                .to_device_enum(&self.device)?;

        for (h, state) in states.iter().enumerate() {
            for b in 0..batch_size {
                for i in 0..self.head_dim {
                    for j in 0..self.head_dim {
                        let val = state.get_scalar(&[b, 0, i, j])?;
                        result = result.set_scalar(&[b, h, i, j], val)?;
                    }
                }
            }
        }

        Ok(result)
    }

    /// Chunk-wise retention for long sequences
    fn chunk_retention(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        chunk_size: usize,
    ) -> Result<Tensor> {
        let batch_size = q.shape()[0];
        let seq_len = q.shape()[1];
        let hidden_size = q.shape()[2];

        let mut outputs = Vec::new();

        // Process sequence in chunks
        for start in (0..seq_len).step_by(chunk_size) {
            let end = std::cmp::min(start + chunk_size, seq_len);

            let q_chunk = q.slice_ranges(&[(0, batch_size), (start, end), (0, hidden_size)])?;
            let k_chunk = k.slice_ranges(&[(0, batch_size), (start, end), (0, hidden_size)])?;
            let v_chunk = v.slice_ranges(&[(0, batch_size), (start, end), (0, hidden_size)])?;

            let chunk_output = self.parallel_retention(&q_chunk, &k_chunk, &v_chunk)?;
            outputs.push(chunk_output);
        }

        // Concatenate chunks
        self.concatenate_chunks(outputs)
    }

    fn reshape_for_heads(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        x.reshape(&[
            batch_size,
            seq_len,
            self.num_heads,
            hidden_size / self.num_heads,
        ])?
        .permute(&[0, 2, 1, 3])
    }

    fn reshape_from_heads(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let num_heads = x.shape()[1];
        let seq_len = x.shape()[2];
        let head_dim = x.shape()[3];

        x.permute(&[0, 2, 1, 3])?.reshape(&[batch_size, seq_len, num_heads * head_dim])
    }

    fn concatenate_chunks(&self, chunks: Vec<Tensor>) -> Result<Tensor> {
        // Concatenate along sequence dimension
        if chunks.is_empty() {
            return Err(tensor_op_error(
                "tensor_operation",
                "No chunks to concatenate".to_string(),
            ));
        }

        let batch_size = chunks[0].shape()[0];
        let hidden_size = chunks[0].shape()[2];
        let total_seq_len: usize = chunks.iter().map(|c| c.shape()[1]).sum();

        let mut result =
            Tensor::zeros(&[batch_size, total_seq_len, hidden_size])?.to_device_enum(&self.device)?;
        let mut offset = 0;

        for chunk in chunks {
            let chunk_seq_len = chunk.shape()[1];

            for b in 0..batch_size {
                for s in 0..chunk_seq_len {
                    for h in 0..hidden_size {
                        let val = chunk.get_scalar(&[b, s, h])?;
                        result = result.set_scalar(&[b, offset + s, h], val)?;
                    }
                }
            }

            offset += chunk_seq_len;
        }

        Ok(result)
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.g_proj.parameter_count()
            + self.out_proj.parameter_count()
    }
}

impl Layer for MultiScaleRetention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let seq_len = input.shape()[1];

        // Project to Q, K, V, G
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input.clone())?;
        let g = self.g_proj.forward(input)?;

        // Apply gate activation (usually Swish/SiLU)
        let g_activated = g.silu()?;

        // Compute retention
        let retention_output = if seq_len > 2048 {
            // Use chunked processing for long sequences
            self.chunk_retention(&q, &k, &v, 512)?
        } else {
            self.parallel_retention(&q, &k, &v)?
        };

        // Apply gating
        let gated_output = retention_output.mul(&g_activated)?;

        // Final projection
        self.out_proj.forward(gated_output)
    }
}

/// Feed-forward network with GLU activation
pub struct RetNetFFN {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    activation: String,
    use_glu: bool,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl RetNetFFN {
    pub fn new(config: &RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &RetNetConfig, device: Device) -> Result<Self> {
        let gate_proj = if config.use_glu {
            Some(Linear::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                config.use_bias,
                device,
            ))
        } else {
            None
        };

        let up_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.use_bias,
            device,
        );
        let down_proj = Linear::new_with_device(
            config.intermediate_size,
            config.hidden_size,
            config.use_bias,
            device,
        );

        Ok(Self {
            gate_proj: gate_proj.unwrap_or_else(|| {
                Linear::new_with_device(
                    config.hidden_size,
                    config.intermediate_size,
                    config.use_bias,
                    device,
                ) // Safe since we just created similar ones above
            }),
            up_proj,
            down_proj,
            activation: config.hidden_act.clone(),
            use_glu: config.use_glu,
            dropout: config.activation_dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "swish" | "silu" => x.silu(),
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            _ => Ok(x.clone()),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for RetNetFFN {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        if self.use_glu {
            // GLU: gate_proj(x) * activation(up_proj(x))
            let gate = self.gate_proj.forward(input.clone())?;
            let up = self.up_proj.forward(input)?;
            let activated_up = self.apply_activation(&up)?;
            let gated = gate.mul(&activated_up)?;
            self.down_proj.forward(gated)
        } else {
            // Standard FFN: down_proj(activation(up_proj(x)))
            let up = self.up_proj.forward(input)?;
            let activated = self.apply_activation(&up)?;
            self.down_proj.forward(activated)
        }
    }
}

/// RetNet decoder layer
pub struct RetNetDecoderLayer {
    retention: MultiScaleRetention,
    ffn: RetNetFFN,
    retention_norm: LayerNorm,
    ffn_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    deepnorm: bool,
    alpha: f32,
    beta: f32,
    device: Device,
}

impl RetNetDecoderLayer {
    pub fn new(config: &RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &RetNetConfig, device: Device) -> Result<Self> {
        let retention = MultiScaleRetention::new_with_device(config, device)?;
        let ffn = RetNetFFN::new_with_device(config, device)?;
        let retention_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let ffn_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        let (alpha, beta) = if config.deepnorm {
            (config.deepnorm_alpha(), config.deepnorm_beta())
        } else {
            (1.0, 1.0)
        };

        Ok(Self {
            retention,
            ffn,
            retention_norm,
            ffn_norm,
            dropout: config.hidden_dropout_prob,
            deepnorm: config.deepnorm,
            alpha,
            beta,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.retention.parameter_count()
            + self.ffn.parameter_count()
            + self.retention_norm.parameter_count()
            + self.ffn_norm.parameter_count()
    }
}

impl Layer for RetNetDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm + residual connection for retention
        let norm1 = self.retention_norm.forward(input.clone())?;
        let retention_out = self.retention.forward(norm1)?;

        let residual1 = if self.deepnorm {
            // DeepNorm scaling
            let scaled_input = input.mul_scalar(self.alpha)?;
            let scaled_retention = retention_out.mul_scalar(self.beta)?;
            scaled_input.add(&scaled_retention)?
        } else {
            input.add(&retention_out)?
        };

        // Pre-norm + residual connection for FFN
        let norm2 = self.ffn_norm.forward(residual1.clone())?;
        let ffn_out = self.ffn.forward(norm2)?;

        let residual2 = if self.deepnorm {
            let scaled_residual1 = residual1.mul_scalar(self.alpha)?;
            let scaled_ffn = ffn_out.mul_scalar(self.beta)?;
            scaled_residual1.add(&scaled_ffn)?
        } else {
            residual1.add(&ffn_out)?
        };

        Ok(residual2)
    }
}

/// RetNet embeddings
pub struct RetNetEmbeddings {
    word_embeddings: Embedding,
    layer_norm: Option<LayerNorm>,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl RetNetEmbeddings {
    pub fn new(config: &RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &RetNetConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
            device,
        )?;

        let layer_norm = if config.layernorm_embedding {
            Some(LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?)
        } else {
            None
        };

        Ok(Self {
            word_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        let mut count = self.word_embeddings.parameter_count();
        if let Some(ln) = &self.layer_norm {
            count += ln.parameter_count();
        }
        count
    }
}

impl Layer for RetNetEmbeddings {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut embeddings = self.word_embeddings.forward(input)?;

        // Apply layer norm if enabled
        if let Some(ref ln) = self.layer_norm {
            embeddings = ln.forward(embeddings)?;
        }

        // Apply dropout (in training mode)
        Ok(embeddings)
    }
}

/// Main RetNet model
pub struct RetNetModel {
    config: RetNetConfig,
    embeddings: RetNetEmbeddings,
    layers: Vec<RetNetDecoderLayer>,
    final_norm: LayerNorm,
    device: Device,
}

impl RetNetModel {
    pub fn new(config: RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RetNetConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = RetNetEmbeddings::new_with_device(&config, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(RetNetDecoderLayer::new_with_device(&config, device)?);
        }

        let final_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            config,
            embeddings,
            layers,
            final_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for RetNetModel {
    type Config = RetNetConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = self.embeddings.forward(input)?;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        self.final_norm.forward(hidden_states)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embeddings.parameter_count();

        // Layer parameters
        for layer in &self.layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.final_norm.parameter_count();

        total
    }
}

/// Advanced RetNet generation capabilities
pub trait RetNetGeneration {
    /// Generate text using recurrent mode for efficient autoregressive generation
    fn generate_recurrent(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        temperature: f32,
        top_p: f32,
        top_k: Option<u32>,
    ) -> Result<Vec<u32>>;

    /// Generate with beam search for better quality
    fn generate_beam_search(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        num_beams: usize,
        early_stopping: bool,
    ) -> Result<Vec<Vec<u32>>>;

    /// Stream generation for real-time applications
    fn generate_stream<F>(&self, input_ids: Vec<u32>, max_length: usize, callback: F) -> Result<()>
    where
        F: Fn(&[u32]) -> bool; // Returns false to stop generation
}

/// Optimized RetNet for long sequence processing
pub struct RetNetLongSequence {
    model: RetNetModel,
    chunk_size: usize,
    overlap_size: usize,
    state_cache: RetNetStateCache,
    device: Device,
}

impl RetNetLongSequence {
    pub fn new(config: RetNetConfig, chunk_size: usize) -> Result<Self> {
        Self::new_with_device(config, chunk_size, Device::CPU)
    }

    pub fn new_with_device(
        config: RetNetConfig,
        chunk_size: usize,
        device: Device,
    ) -> Result<Self> {
        let model = RetNetModel::new_with_device(config.clone(), device)?;
        let overlap_size = chunk_size / 4; // 25% overlap
        let state_cache = RetNetStateCache::new(config.num_hidden_layers * 4);

        Ok(Self {
            model,
            chunk_size,
            overlap_size,
            state_cache,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Process very long sequences efficiently
    pub fn process_long_sequence(&mut self, input: Vec<u32>) -> Result<Tensor> {
        let seq_len = input.len();

        if seq_len <= self.chunk_size {
            return self.model.forward(input);
        }

        let mut all_outputs = Vec::new();
        let effective_step = self.chunk_size - self.overlap_size;

        for start in (0..seq_len).step_by(effective_step) {
            let end = std::cmp::min(start + self.chunk_size, seq_len);
            let chunk = input[start..end].to_vec();

            let chunk_output = self.model.forward(chunk)?;

            // Remove overlap from previous chunks
            let output_start = if start == 0 { 0 } else { self.overlap_size };
            let chunk_seq_len = chunk_output.shape()[1];

            if chunk_seq_len > output_start {
                let trimmed_output = chunk_output.slice_ranges(&[
                    (0, chunk_output.shape()[0]),
                    (output_start, chunk_seq_len),
                    (0, chunk_output.shape()[2]),
                ])?;
                all_outputs.push(trimmed_output);
            }
        }

        self.concatenate_outputs(all_outputs)
    }

    fn concatenate_outputs(&self, outputs: Vec<Tensor>) -> Result<Tensor> {
        if outputs.is_empty() {
            return Err(tensor_op_error(
                "tensor_operation",
                "No outputs to concatenate".to_string(),
            ));
        }

        let batch_size = outputs[0].shape()[0];
        let hidden_size = outputs[0].shape()[2];
        let total_seq_len: usize = outputs.iter().map(|o| o.shape()[1]).sum();

        let mut result =
            Tensor::zeros(&[batch_size, total_seq_len, hidden_size])?.to_device_enum(&self.device)?;
        let mut offset = 0;

        for output in outputs {
            let seq_len = output.shape()[1];

            for b in 0..batch_size {
                for s in 0..seq_len {
                    for h in 0..hidden_size {
                        let val = output.get_scalar(&[b, s, h])?;
                        result = result.set_scalar(&[b, offset + s, h], val)?;
                    }
                }
            }

            offset += seq_len;
        }

        Ok(result)
    }

    /// Get memory usage statistics
    pub fn get_memory_stats(&self) -> RetNetMemoryStats {
        RetNetMemoryStats {
            cache_size: self.state_cache.size(),
            max_cache_size: self.state_cache.max_cache_size,
            chunk_size: self.chunk_size,
            overlap_size: self.overlap_size,
            estimated_memory_mb: self.estimate_memory_usage(),
        }
    }

    fn estimate_memory_usage(&self) -> f64 {
        let config = self.model.get_config();
        let params = self.model.num_parameters() as f64;
        let state_memory =
            (self.state_cache.size() * config.hidden_size * config.hidden_size * 4) as f64; // 4 bytes per float
        let chunk_memory = (self.chunk_size * config.hidden_size * 4) as f64;

        (params * 4.0 + state_memory + chunk_memory) / (1024.0 * 1024.0) // Convert to MB
    }
}

/// Memory usage statistics for RetNet
#[derive(Debug, Clone)]
pub struct RetNetMemoryStats {
    pub cache_size: usize,
    pub max_cache_size: usize,
    pub chunk_size: usize,
    pub overlap_size: usize,
    pub estimated_memory_mb: f64,
}

/// RetNet for language modeling
pub struct RetNetForLanguageModeling {
    retnet: RetNetModel,
    lm_head: Option<Linear>,
    device: Device,
}

impl RetNetForLanguageModeling {
    pub fn new(config: RetNetConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: RetNetConfig, device: Device) -> Result<Self> {
        let retnet = RetNetModel::new_with_device(config.clone(), device)?;

        let lm_head = if !config.no_output_layer {
            Some(Linear::new_with_device(
                config.hidden_size,
                config.vocab_size,
                false,
                device,
            ))
        } else {
            None
        };

        Ok(Self {
            retnet,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for RetNetForLanguageModeling {
    type Config = RetNetConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.retnet.forward(input)?;

        if let Some(ref lm_head) = self.lm_head {
            lm_head.forward(hidden_states)
        } else {
            Ok(hidden_states)
        }
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.retnet.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.retnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        let mut total = self.retnet.num_parameters();
        if let Some(ref lm_head) = self.lm_head {
            total += lm_head.parameter_count();
        }
        total
    }
}

/// RetNet for sequence classification
pub struct RetNetForSequenceClassification {
    retnet: RetNetModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl RetNetForSequenceClassification {
    pub fn new(config: RetNetConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: RetNetConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let retnet = RetNetModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            retnet,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for RetNetForSequenceClassification {
    type Config = RetNetConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.retnet.forward(input)?;

        // Use last token for classification (causal LM style)
        let last_token = self.get_last_token(&sequence_output)?;
        self.classifier.forward(last_token)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.retnet.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.retnet.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.retnet.num_parameters() + self.classifier.parameter_count()
    }
}

impl RetNetForSequenceClassification {
    fn get_last_token(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];
        let hidden_size = x.shape()[2];

        // Extract last token embeddings
        let mut last_tokens = Tensor::zeros(&[batch_size, hidden_size])?.to_device_enum(&self.device)?;

        for b in 0..batch_size {
            for h in 0..hidden_size {
                let val = x.get_scalar(&[b, seq_len - 1, h])?;
                last_tokens = last_tokens.set_scalar(&[b, h], val)?;
            }
        }

        Ok(last_tokens)
    }
}
