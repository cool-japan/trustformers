//! Ring Attention for extremely long context processing.
//!
//! This module implements Ring Attention as described in "Ring Attention with
//! Blockwise Transformers for Near-Infinite Context" (Liu et al., 2023): the
//! sequence is partitioned into blocks, every block owns its key/value shard, and
//! the shards are rotated around a ring so that each query block eventually sees
//! every key block exactly once. Partial results are combined with an online
//! (flash-attention style) softmax, so the ring never materialises the full
//! `seq_len × seq_len` score matrix and the result is numerically identical to
//! dense attention.
//!
//! ## What this implementation actually does
//!
//! * **Real projections.** Learned `Linear` layers produce Q/K/V and the output
//!   projection. They are created on the first `forward` call, when the input
//!   embedding width becomes known.
//! * **Real ring rotation.** Every ring position starts holding its own K/V shard;
//!   after each step the shards are rotated by one position, so a shard visits
//!   every query block over the `num_blocks` steps. No shard is ever fabricated.
//! * **Real attention math.** `Q·Kᵀ / sqrt(head_dim)` → causal / padding masking →
//!   online softmax accumulation with running max and running sum → `P·V`.
//! * **In-process execution.** The ring is executed sequentially inside one
//!   process: this engine owns every shard and rotates them itself. No collective
//!   communication backend is wired up, and requesting one is an error rather than
//!   a silent no-op.

use trustformers_core::{
    errors::{not_implemented, Result, TrustformersError},
    layers::Linear,
    tensor::Tensor,
    traits::Layer,
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
    /// Overlap size between blocks.
    ///
    /// The online-softmax accumulation is exact, so overlapping blocks would
    /// double-count keys rather than add context. Only `0` is supported.
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
            block_overlap: 0,
            communication_backend: CommunicationBackend::InProcess,
            memory_optimization: MemoryOptimizationConfig::default(),
        }
    }
}

/// Communication backends for distributed attention
#[derive(Debug, Clone, PartialEq)]
pub enum CommunicationBackend {
    /// Sequential in-process ring: this engine owns and rotates every shard.
    InProcess,
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
    /// Use gradient checkpointing to trade compute for memory.
    ///
    /// Not supported by this inference-only engine.
    pub gradient_checkpointing: bool,
    /// Fuse the score/softmax/output steps into a single pass over each block.
    pub fused_attention: bool,
    /// Use mixed precision (FP16/BF16) for attention computation.
    ///
    /// Not supported: the CPU path accumulates in f32 throughout.
    pub mixed_precision: bool,
    /// Enable sequence parallelism within blocks.
    ///
    /// Not supported by the sequential in-process ring.
    pub sequence_parallel: bool,
    /// Flash-attention style online softmax (running max / running sum).
    pub flash_attention: bool,
}

impl Default for MemoryOptimizationConfig {
    fn default() -> Self {
        // Only the options this engine genuinely implements are enabled.
        Self {
            gradient_checkpointing: false,
            fused_attention: true,
            mixed_precision: false,
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
        let next_rank = (current_rank + 1) % ring_size.max(1);
        let prev_rank =
            if current_rank == 0 { ring_size.saturating_sub(1) } else { current_rank - 1 };

        Self {
            ring_size,
            current_rank,
            next_rank,
            prev_rank,
            backend,
        }
    }
}

/// Memory pool for pre-allocating attention working buffers.
///
/// Standalone helper for callers that want to reuse buffers across forward passes;
/// the ring engine itself accumulates in place and does not require it.
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

    pub fn get_key_buffer(&mut self, index: usize) -> Option<&mut Tensor> {
        if index < self.pool_size {
            self.key_buffers[index].as_mut()
        } else {
            None
        }
    }

    pub fn get_value_buffer(&mut self, index: usize) -> Option<&mut Tensor> {
        if index < self.pool_size {
            self.value_buffers[index].as_mut()
        } else {
            None
        }
    }

    pub fn get_score_buffer(&mut self, index: usize) -> Option<&mut Tensor> {
        if index < self.pool_size {
            self.score_buffers[index].as_mut()
        } else {
            None
        }
    }

    pub fn get_output_buffer(&mut self, index: usize) -> Option<&mut Tensor> {
        if index < self.pool_size {
            self.output_buffers[index].as_mut()
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

/// Learned Q/K/V/output projections.
///
/// Created lazily on the first `forward`, because the input embedding width is
/// only known then; the inner attention width is always `num_heads · head_dim`.
#[derive(Debug)]
struct RingProjections {
    embed_dim: usize,
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
}

impl RingProjections {
    fn new(embed_dim: usize, inner_dim: usize) -> Self {
        Self {
            embed_dim,
            query: Linear::new(embed_dim, inner_dim, false),
            key: Linear::new(embed_dim, inner_dim, false),
            value: Linear::new(embed_dim, inner_dim, false),
            output: Linear::new(inner_dim, embed_dim, false),
        }
    }

    fn parameter_count(&self) -> usize {
        self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output.parameter_count()
    }
}

/// A key/value shard: the real K/V rows of one sequence block.
///
/// Shards are the objects that travel around the ring; each carries the absolute
/// sequence offset of the block it came from, which is what causal masking and the
/// block-level skip compare against.
#[derive(Debug, Clone)]
struct KvShard {
    start_pos: usize,
    seq_len: usize,
    /// `[seq_len, num_heads · head_dim]`, row-major.
    keys: Vec<f32>,
    /// `[seq_len, num_heads · head_dim]`, row-major.
    values: Vec<f32>,
}

/// Online (flash-attention style) softmax accumulator for one query block.
///
/// Holds, per `(query row, head)`, the running maximum score `m`, the running
/// denominator `l`, and the unnormalised output `o`. Absorbing a new key block
/// rescales the existing state by `exp(m_old - m_new)` before adding the block's
/// contribution, which makes the blockwise result exactly equal to a single dense
/// softmax over all keys.
#[derive(Debug)]
struct OnlineSoftmax {
    rows: usize,
    heads: usize,
    head_dim: usize,
    running_max: Vec<f32>,
    running_sum: Vec<f32>,
    output: Vec<f32>,
    scratch: Vec<f32>,
}

impl OnlineSoftmax {
    fn new(rows: usize, heads: usize, head_dim: usize) -> Self {
        Self {
            rows,
            heads,
            head_dim,
            running_max: vec![f32::NEG_INFINITY; rows * heads],
            running_sum: vec![0.0; rows * heads],
            output: vec![0.0; rows * heads * head_dim],
            scratch: Vec::new(),
        }
    }

    /// Fold one K/V shard into the accumulator.
    ///
    /// `queries` is the whole `[seq_len, heads · head_dim]` query buffer for this
    /// batch element; `query_block` selects the rows this accumulator owns.
    #[allow(clippy::too_many_arguments)]
    fn absorb(
        &mut self,
        queries: &[f32],
        query_block: &AttentionBlock,
        shard: &KvShard,
        scale: f32,
        causal: bool,
        key_mask: Option<&[bool]>,
    ) {
        if shard.seq_len == 0 || self.rows == 0 {
            return;
        }

        // Block-level causal skip: the entire shard lies strictly after every
        // query in this block. Skipping *before* touching the running state is
        // what keeps `exp(-inf - -inf)` out of the accumulator.
        if causal && shard.start_pos >= query_block.end_pos {
            return;
        }

        let inner = self.heads * self.head_dim;
        if self.scratch.len() < shard.seq_len {
            self.scratch.resize(shard.seq_len, 0.0);
        }

        for (row, query_pos) in (query_block.start_pos..query_block.end_pos).enumerate() {
            for head in 0..self.heads {
                let query_base = query_pos * inner + head * self.head_dim;

                // Scores for this (row, head) against the whole shard.
                let mut block_max = f32::NEG_INFINITY;
                for j in 0..shard.seq_len {
                    let key_pos = shard.start_pos + j;
                    if causal && key_pos > query_pos {
                        self.scratch[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    if let Some(mask) = key_mask {
                        if !mask[key_pos] {
                            self.scratch[j] = f32::NEG_INFINITY;
                            continue;
                        }
                    }

                    let key_base = j * inner + head * self.head_dim;
                    let mut dot = 0.0f32;
                    for d in 0..self.head_dim {
                        dot += queries[query_base + d] * shard.keys[key_base + d];
                    }
                    let score = dot * scale;
                    self.scratch[j] = score;
                    if score > block_max {
                        block_max = score;
                    }
                }

                // Every key in this shard was masked out for this row.
                if !block_max.is_finite() {
                    continue;
                }

                let state_idx = row * self.heads + head;
                let previous_max = self.running_max[state_idx];
                let new_max =
                    if previous_max.is_finite() { previous_max.max(block_max) } else { block_max };
                let correction =
                    if previous_max.is_finite() { (previous_max - new_max).exp() } else { 0.0 };

                let out_base = state_idx * self.head_dim;
                if correction != 1.0 {
                    for d in 0..self.head_dim {
                        self.output[out_base + d] *= correction;
                    }
                    self.running_sum[state_idx] *= correction;
                }

                let mut block_sum = 0.0f32;
                for j in 0..shard.seq_len {
                    let score = self.scratch[j];
                    if !score.is_finite() {
                        continue;
                    }
                    let weight = (score - new_max).exp();
                    block_sum += weight;
                    let value_base = j * inner + head * self.head_dim;
                    for d in 0..self.head_dim {
                        self.output[out_base + d] += weight * shard.values[value_base + d];
                    }
                }

                self.running_sum[state_idx] += block_sum;
                self.running_max[state_idx] = new_max;
            }
        }
    }

    /// Normalise by the running denominator and write into `destination`.
    ///
    /// `destination` is the `[rows, heads · head_dim]` slice of the context buffer
    /// belonging to this query block.
    fn finish_into(&self, destination: &mut [f32]) {
        let inner = self.heads * self.head_dim;
        for row in 0..self.rows {
            for head in 0..self.heads {
                let state_idx = row * self.heads + head;
                let denominator = self.running_sum[state_idx];
                let src = state_idx * self.head_dim;
                let dst = row * inner + head * self.head_dim;
                if denominator > 0.0 {
                    for d in 0..self.head_dim {
                        destination[dst + d] = self.output[src + d] / denominator;
                    }
                } else {
                    // Fully masked row (e.g. an all-padding key set): no evidence
                    // to attend to, so the context stays zero rather than NaN.
                    for d in 0..self.head_dim {
                        destination[dst + d] = 0.0;
                    }
                }
            }
        }
    }
}

/// Ring attention computation engine.
///
/// One instance owns the whole sequence and executes the ring sequentially: the
/// K/V shards are rotated between ring positions until every query block has seen
/// every key block. `device_id` names which ring position is considered local for
/// topology bookkeeping and statistics; it does not change the numerical result,
/// which is always the full attention output for the whole input.
pub struct RingAttention {
    config: RingAttentionConfig,
    device_id: usize,
    communication_group: CommunicationGroup,
    attention_blocks: Vec<AttentionBlock>,
    projections: Option<RingProjections>,
}

impl RingAttention {
    /// Create a new Ring Attention instance
    pub fn new(config: RingAttentionConfig, device_id: usize) -> Result<Self> {
        if config.ring_size == 0 {
            return Err(TrustformersError::config_error(
                "Ring size must be greater than zero",
                "ring_attention_init",
            ));
        }
        if device_id >= config.ring_size {
            return Err(TrustformersError::config_error(
                &format!(
                    "Device ID {} must be less than ring size {}",
                    device_id, config.ring_size
                ),
                "ring_attention_init",
            ));
        }
        if config.block_size == 0 {
            return Err(TrustformersError::config_error(
                "Block size must be greater than zero",
                "ring_attention_init",
            ));
        }
        if config.num_heads == 0 || config.head_dim == 0 {
            return Err(TrustformersError::config_error(
                "num_heads and head_dim must be greater than zero",
                "ring_attention_init",
            ));
        }
        if config.block_overlap != 0 {
            return Err(not_implemented(
                "overlapping ring blocks: the online-softmax accumulation is exact, so \
                 overlapping blocks would double-count keys; set block_overlap = 0",
            ));
        }
        if config.communication_backend != CommunicationBackend::InProcess {
            return Err(not_implemented(format!(
                "communication backend {:?}: no collective backend is wired up, this engine \
                 executes the ring in-process; use CommunicationBackend::InProcess",
                config.communication_backend
            )));
        }
        if config.memory_optimization.mixed_precision {
            return Err(not_implemented(
                "mixed-precision ring attention: the CPU path accumulates in f32 throughout",
            ));
        }
        if config.memory_optimization.gradient_checkpointing {
            return Err(not_implemented(
                "gradient checkpointing: this ring-attention engine is inference-only",
            ));
        }
        if config.memory_optimization.sequence_parallel {
            return Err(not_implemented(
                "sequence parallelism inside a block: the in-process ring is sequential",
            ));
        }

        let communication_group = CommunicationGroup::new(
            config.ring_size,
            device_id,
            config.communication_backend.clone(),
        );

        Ok(Self {
            config,
            device_id,
            communication_group,
            attention_blocks: Vec::new(),
            projections: None,
        })
    }

    /// Ring position considered local to this engine.
    pub fn device_id(&self) -> usize {
        self.device_id
    }

    /// Ring topology (rank, next rank, previous rank) of this engine.
    pub fn communication_group(&self) -> &CommunicationGroup {
        &self.communication_group
    }

    /// Blocks produced by the most recent `partition_sequence` / `forward`.
    pub fn attention_blocks(&self) -> &[AttentionBlock] {
        &self.attention_blocks
    }

    /// Number of learned parameters; `0` until the first `forward` sizes them.
    pub fn parameter_count(&self) -> usize {
        self.projections.as_ref().map_or(0, RingProjections::parameter_count)
    }

    /// Partition the sequence into blocks for ring processing
    pub fn partition_sequence(&mut self, sequence_length: usize) -> Result<Vec<AttentionBlock>> {
        let num_blocks = sequence_length.div_ceil(self.config.block_size);
        let mut blocks = Vec::with_capacity(num_blocks);

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

    /// Create the Q/K/V/output projections for an embedding width, once.
    fn ensure_projections(&mut self, embed_dim: usize) -> Result<()> {
        let inner_dim = self.config.num_heads * self.config.head_dim;

        if let Some(projections) = &self.projections {
            if projections.embed_dim != embed_dim {
                return Err(TrustformersError::shape_error(format!(
                    "ring attention was initialised for embedding width {} but received {}",
                    projections.embed_dim, embed_dim
                )));
            }
            return Ok(());
        }

        self.projections = Some(RingProjections::new(embed_dim, inner_dim));
        Ok(())
    }

    /// Compute ring attention for the given input embeddings.
    ///
    /// `input_embeddings` is `[batch, seq_len, embed_dim]`; the output has the same
    /// shape. `attention_mask` is an optional key-padding mask of shape `[seq_len]`
    /// or `[batch, seq_len]`, where a zero marks a position that must not be
    /// attended to.
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

        let batch_size = input_shape[0];
        let seq_length = input_shape[1];
        let embed_dim = input_shape[2];

        if seq_length == 0 || batch_size == 0 {
            return Err(TrustformersError::config_error(
                "Ring attention requires a non-empty batch and sequence",
                "ring_attention_forward",
            ));
        }
        if seq_length > self.config.max_seq_length {
            return Err(TrustformersError::config_error(
                &format!(
                    "Sequence length {} exceeds configured max_seq_length {}",
                    seq_length, self.config.max_seq_length
                ),
                "ring_attention_forward",
            ));
        }

        self.ensure_projections(embed_dim)?;
        let blocks = self.partition_sequence(seq_length)?;
        let key_mask = decode_key_padding_mask(attention_mask, batch_size, seq_length)?;

        let projections = self.projections.as_ref().ok_or_else(|| {
            TrustformersError::tensor_op_error(
                "ring attention projections were not initialised",
                "ring_attention_forward",
            )
        })?;

        // Real learned projections — the ring operates on these, never on noise.
        let queries = projections.query.forward(input_embeddings.clone())?.data()?;
        let keys = projections.key.forward(input_embeddings.clone())?.data()?;
        let values = projections.value.forward(input_embeddings.clone())?.data()?;

        let inner_dim = self.config.num_heads * self.config.head_dim;
        let stride = seq_length * inner_dim;
        let mut context = vec![0.0f32; batch_size * stride];

        for batch in 0..batch_size {
            let range = batch * stride..(batch + 1) * stride;
            let batch_mask = key_mask
                .as_ref()
                .map(|mask| &mask[batch * seq_length..(batch + 1) * seq_length]);
            self.run_ring(
                &queries[range.clone()],
                &keys[range.clone()],
                &values[range.clone()],
                batch_mask,
                &blocks,
                &mut context[range],
            );
        }

        let context_tensor = Tensor::from_vec(context, &[batch_size, seq_length, inner_dim])?;
        projections.output.forward(context_tensor)
    }

    /// Execute the ring for one batch element.
    ///
    /// Every ring position starts holding its own K/V shard. On each of the
    /// `num_blocks` steps each position folds the shard it currently holds into its
    /// query block's online-softmax accumulator, then the shards rotate by one
    /// position — the in-process equivalent of the ring send/recv. After
    /// `num_blocks` steps every query block has seen every key block exactly once.
    fn run_ring(
        &self,
        queries: &[f32],
        keys: &[f32],
        values: &[f32],
        key_mask: Option<&[bool]>,
        blocks: &[AttentionBlock],
        context: &mut [f32],
    ) {
        let heads = self.config.num_heads;
        let head_dim = self.config.head_dim;
        let inner_dim = heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        // The real K/V rows of each block: these buffers are what travels.
        let mut resident: Vec<KvShard> =
            blocks.iter().map(|block| self.shard_for_block(keys, values, block)).collect();

        let mut accumulators: Vec<OnlineSoftmax> = blocks
            .iter()
            .map(|block| OnlineSoftmax::new(block.sequence_length(), heads, head_dim))
            .collect();

        for _step in 0..blocks.len() {
            for (position, shard) in resident.iter().enumerate() {
                accumulators[position].absorb(
                    queries,
                    &blocks[position],
                    shard,
                    scale,
                    self.config.causal,
                    key_mask,
                );
            }
            // Ring exchange: each shard moves on to the next ring position.
            resident.rotate_right(1);
        }

        for (position, accumulator) in accumulators.iter().enumerate() {
            let block = &blocks[position];
            accumulator
                .finish_into(&mut context[block.start_pos * inner_dim..block.end_pos * inner_dim]);
        }
    }

    /// Fetch the K/V shard currently assigned to a block.
    ///
    /// The shard is materialised from the block's own rows of the projected K/V
    /// buffers — the real data for that slice of the sequence.
    fn shard_for_block(&self, keys: &[f32], values: &[f32], block: &AttentionBlock) -> KvShard {
        let inner_dim = self.config.num_heads * self.config.head_dim;
        KvShard {
            start_pos: block.start_pos,
            seq_len: block.sequence_length(),
            keys: keys[block.start_pos * inner_dim..block.end_pos * inner_dim].to_vec(),
            values: values[block.start_pos * inner_dim..block.end_pos * inner_dim].to_vec(),
        }
    }

    /// Get ring attention statistics
    pub fn get_stats(&self) -> RingAttentionStats {
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

/// Decode an optional key-padding mask into a per-position boolean keep flag.
///
/// Accepts `[seq_len]` (shared across the batch) or `[batch, seq_len]`. Anything
/// else — in particular a full `[batch, heads, q, k]` additive mask — is reported
/// as unimplemented rather than silently ignored.
fn decode_key_padding_mask(
    attention_mask: Option<&Tensor>,
    batch_size: usize,
    seq_length: usize,
) -> Result<Option<Vec<bool>>> {
    let Some(mask) = attention_mask else {
        return Ok(None);
    };

    let shape = mask.shape();
    let values = mask.data()?;

    let flags = match shape.as_slice() {
        [len] if *len == seq_length => {
            let row: Vec<bool> = values.iter().map(|v| *v != 0.0).collect();
            row.repeat(batch_size)
        },
        [batch, len] if *batch == batch_size && *len == seq_length => {
            values.iter().map(|v| *v != 0.0).collect()
        },
        _ => {
            return Err(not_implemented(format!(
                "attention mask of shape {shape:?}: ring attention supports a key-padding mask \
                 of shape [seq_len] or [batch, seq_len]"
            )))
        },
    };

    Ok(Some(flags))
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

/// Coordinates several in-process ring-attention engines.
///
/// Each engine still executes its whole ring locally; this manager owns the set of
/// engines and dispatches work to them. It performs no cross-process communication.
pub struct DistributedRingAttentionManager {
    devices: Vec<RingAttention>,
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
        // The in-process manager runs its engines one after another; only the
        // synchronous strategy describes what actually happens.
        Self {
            synchronization_strategy: SynchronizationStrategy::Synchronous,
            fault_tolerance: false,
            load_balancing: false,
            communication_compression: false,
        }
    }
}

impl DistributedRingAttentionManager {
    pub fn new(
        configs: Vec<RingAttentionConfig>,
        coordination_config: CoordinationConfig,
    ) -> Result<Self> {
        if configs.is_empty() {
            return Err(TrustformersError::config_error(
                "At least one ring attention configuration is required",
                "distributed_ring_init",
            ));
        }
        if coordination_config.synchronization_strategy != SynchronizationStrategy::Synchronous {
            return Err(not_implemented(format!(
                "{:?} scheduling: the in-process manager executes its engines synchronously",
                coordination_config.synchronization_strategy
            )));
        }
        if coordination_config.communication_compression {
            return Err(not_implemented(
                "communication compression: the in-process ring exchanges no wire messages",
            ));
        }
        if coordination_config.fault_tolerance {
            return Err(not_implemented(
                "fault tolerance: there is no peer to fail over to in the in-process ring",
            ));
        }
        if coordination_config.load_balancing {
            return Err(not_implemented(
                "dynamic load balancing: blocks are assigned round-robin at partition time",
            ));
        }

        let mut devices = Vec::with_capacity(configs.len());
        for (device_id, config) in configs.into_iter().enumerate() {
            let device_id = device_id % config.ring_size.max(1);
            let ring_attention = RingAttention::new(config, device_id)?;
            devices.push(ring_attention);
        }

        Ok(Self {
            devices,
            coordination_config,
        })
    }

    /// Number of engines under this manager.
    pub fn device_count(&self) -> usize {
        self.devices.len()
    }

    /// The coordination policy this manager was built with.
    pub fn coordination_config(&self) -> &CoordinationConfig {
        &self.coordination_config
    }

    /// Run the ring on the engine at `device_index`.
    pub fn process_on_device(
        &mut self,
        device_index: usize,
        input_embeddings: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let device = self.devices.get_mut(device_index).ok_or_else(|| {
            TrustformersError::config_error(
                &format!("Device index {device_index} is out of range"),
                "distributed_process",
            )
        })?;
        device.forward(input_embeddings, attention_mask)
    }

    /// Run the ring on the first configured engine.
    ///
    /// Every engine owns the whole sequence, so this returns the full attention
    /// output; the remaining engines exist for topology bookkeeping.
    pub fn process_distributed(
        &mut self,
        input_embeddings: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        self.process_on_device(0, input_embeddings, attention_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config(seq_blocks: usize) -> RingAttentionConfig {
        RingAttentionConfig {
            ring_size: seq_blocks,
            block_size: 8,
            num_heads: 2,
            head_dim: 4,
            causal: true,
            max_seq_length: 4096,
            block_overlap: 0,
            communication_backend: CommunicationBackend::InProcess,
            memory_optimization: MemoryOptimizationConfig::default(),
        }
    }

    /// Deterministic pseudo-random input so tests never depend on global RNG state.
    fn deterministic_input(batch: usize, seq: usize, embed: usize) -> Tensor {
        let mut state: u64 = 0x2545_F491_4F6C_DD1D;
        let data: Vec<f32> = (0..batch * seq * embed)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (u32::MAX as f32 / 2.0)) - 1.0
            })
            .collect();
        Tensor::from_vec(data, &[batch, seq, embed]).expect("input tensor")
    }

    /// Dense reference attention using the engine's own learned projections.
    ///
    /// `q·kᵀ/sqrt(d)` → mask → softmax over all keys → `p·v` → output projection.
    fn dense_reference(
        engine: &RingAttention,
        input: &Tensor,
        key_mask: Option<&[bool]>,
    ) -> Vec<f32> {
        let projections = engine.projections.as_ref().expect("projections initialised");
        let shape = input.shape();
        let (batch, seq) = (shape[0], shape[1]);
        let heads = engine.config.num_heads;
        let head_dim = engine.config.head_dim;
        let inner = heads * head_dim;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q = projections.query.forward(input.clone()).expect("q").data().expect("q data");
        let k = projections.key.forward(input.clone()).expect("k").data().expect("k data");
        let v = projections.value.forward(input.clone()).expect("v").data().expect("v data");

        let mut context = vec![0.0f32; batch * seq * inner];
        for b in 0..batch {
            let base = b * seq * inner;
            for i in 0..seq {
                for h in 0..heads {
                    let q_base = base + i * inner + h * head_dim;
                    let mut scores = vec![f32::NEG_INFINITY; seq];
                    for j in 0..seq {
                        if engine.config.causal && j > i {
                            continue;
                        }
                        if let Some(mask) = key_mask {
                            if !mask[b * seq + j] {
                                continue;
                            }
                        }
                        let k_base = base + j * inner + h * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += q[q_base + d] * k[k_base + d];
                        }
                        scores[j] = dot * scale;
                    }
                    let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    if !max.is_finite() {
                        continue;
                    }
                    let mut denominator = 0.0f32;
                    let out_base = base + i * inner + h * head_dim;
                    for j in 0..seq {
                        if !scores[j].is_finite() {
                            continue;
                        }
                        let weight = (scores[j] - max).exp();
                        denominator += weight;
                        let v_base = base + j * inner + h * head_dim;
                        for d in 0..head_dim {
                            context[out_base + d] += weight * v[v_base + d];
                        }
                    }
                    for d in 0..head_dim {
                        context[out_base + d] /= denominator;
                    }
                }
            }
        }

        let context_tensor =
            Tensor::from_vec(context, &[batch, seq, inner]).expect("context tensor");
        projections
            .output
            .forward(context_tensor)
            .expect("output projection")
            .data()
            .expect("output data")
    }

    #[test]
    fn test_ring_attention_config() {
        let config = RingAttentionConfig::default();
        assert_eq!(config.ring_size, 8);
        assert_eq!(config.block_size, 4096);
        assert_eq!(config.num_heads, 32);
        assert_eq!(config.head_dim, 128);
        assert!(config.causal);
        assert_eq!(config.block_overlap, 0);
        assert_eq!(
            config.communication_backend,
            CommunicationBackend::InProcess
        );
    }

    #[test]
    fn test_communication_group() {
        let group = CommunicationGroup::new(8, 3, CommunicationBackend::InProcess);
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
            num_heads: 2,
            head_dim: 4,
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
        pool.allocate_buffers(16, 2, 4)?;

        assert!(pool.get_query_buffer(0).is_some());
        assert!(pool.get_query_buffer(3).is_some());
        assert!(pool.get_query_buffer(4).is_none()); // Out of bounds
        assert!(pool.get_score_buffer(0).is_some());

        Ok(())
    }

    /// The ring output must equal dense causal attention within f32 tolerance.
    ///
    /// This is the regression test for the previous implementation, which averaged
    /// freshly generated `Tensor::randn` "key/value" tensors and never used the
    /// queries at all.
    #[test]
    fn test_ring_attention_matches_dense_causal_attention() -> Result<()> {
        let config = tiny_config(4);
        let mut ring = RingAttention::new(config, 0)?;

        // 32 tokens over blocks of 8 -> 4 ring positions, so shards really rotate.
        let input = deterministic_input(1, 32, 8);
        let ring_output = ring.forward(&input, None)?;
        assert_eq!(ring_output.shape(), input.shape());
        assert_eq!(ring.attention_blocks().len(), 4);

        let reference = dense_reference(&ring, &input, None);
        let actual = ring_output.data()?;
        assert_eq!(actual.len(), reference.len());
        for (i, (got, want)) in actual.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: ring {got} vs dense attention {want}"
            );
        }

        Ok(())
    }

    /// Non-causal (bidirectional) ring attention must also match dense attention.
    #[test]
    fn test_ring_attention_matches_dense_bidirectional_attention() -> Result<()> {
        let config = RingAttentionConfig {
            causal: false,
            ..tiny_config(3)
        };
        let mut ring = RingAttention::new(config, 0)?;

        let input = deterministic_input(1, 24, 8);
        let ring_output = ring.forward(&input, None)?;
        let reference = dense_reference(&ring, &input, None);
        let actual = ring_output.data()?;

        for (i, (got, want)) in actual.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: ring {got} vs dense attention {want}"
            );
        }

        Ok(())
    }

    /// A multi-element batch must match dense attention for every element.
    #[test]
    fn test_ring_attention_matches_dense_attention_batched() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;

        let input = deterministic_input(3, 16, 8);
        let ring_output = ring.forward(&input, None)?;
        assert_eq!(ring_output.shape(), vec![3, 16, 8]);

        let reference = dense_reference(&ring, &input, None);
        let actual = ring_output.data()?;
        for (i, (got, want)) in actual.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: ring {got} vs dense attention {want}"
            );
        }

        Ok(())
    }

    /// A key-padding mask must be honoured, and match dense attention under the
    /// same mask.
    #[test]
    fn test_ring_attention_honours_key_padding_mask() -> Result<()> {
        let config = RingAttentionConfig {
            causal: false,
            ..tiny_config(2)
        };
        let mut ring = RingAttention::new(config, 0)?;

        let seq = 16usize;
        let input = deterministic_input(1, seq, 8);
        // Mask out the second half of the sequence.
        let mask_values: Vec<f32> = (0..seq).map(|i| if i < seq / 2 { 1.0 } else { 0.0 }).collect();
        let mask = Tensor::from_vec(mask_values.clone(), &[seq])?;

        let masked = ring.forward(&input, Some(&mask))?.data()?;
        let flags: Vec<bool> = mask_values.iter().map(|v| *v != 0.0).collect();
        let reference = dense_reference(&ring, &input, Some(&flags));
        for (i, (got, want)) in masked.iter().zip(reference.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-3,
                "element {i}: masked ring {got} vs masked dense {want}"
            );
        }

        // The mask must actually change the result.
        let unmasked = ring.forward(&input, None)?.data()?;
        assert!(
            masked.iter().zip(unmasked.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "the key-padding mask had no effect"
        );

        Ok(())
    }

    /// Unsupported mask layouts are reported instead of ignored.
    #[test]
    fn test_ring_attention_rejects_unsupported_mask_shape() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;
        let input = deterministic_input(1, 16, 8);
        let bad_mask = Tensor::zeros(&[1, 2, 16, 16])?;
        assert!(ring.forward(&input, Some(&bad_mask)).is_err());
        Ok(())
    }

    /// The forward pass must be deterministic.
    ///
    /// The previous implementation drew fresh `Tensor::randn` keys and values on
    /// every ring step, so two identical calls disagreed.
    #[test]
    fn test_ring_attention_forward_is_deterministic() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;
        let input = deterministic_input(1, 16, 8);

        let first = ring.forward(&input, None)?.data()?;
        let second = ring.forward(&input, None)?.data()?;

        assert_eq!(first.len(), second.len());
        for (a, b) in first.iter().zip(second.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "forward is not deterministic: {a} vs {b}"
            );
        }

        Ok(())
    }

    /// The output must depend on the input (queries and keys are really used).
    #[test]
    fn test_ring_attention_output_depends_on_input() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;

        let input_a = deterministic_input(1, 16, 8);
        let mut data_b = input_a.data()?;
        for value in data_b.iter_mut() {
            *value *= -1.0;
        }
        let input_b = Tensor::from_vec(data_b, &[1, 16, 8])?;

        let out_a = ring.forward(&input_a, None)?.data()?;
        let out_b = ring.forward(&input_b, None)?.data()?;

        assert!(
            out_a.iter().zip(out_b.iter()).any(|(a, b)| (a - b).abs() > 1e-4),
            "ring attention ignored its input"
        );

        Ok(())
    }

    /// Causal ring attention must not let a later token influence an earlier one.
    #[test]
    fn test_ring_attention_is_causal() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;

        let seq = 16usize;
        let embed = 8usize;
        let input_a = deterministic_input(1, seq, embed);
        let mut data_b = input_a.data()?;
        // Perturb only the final token.
        for value in data_b.iter_mut().skip((seq - 1) * embed) {
            *value += 5.0;
        }
        let input_b = Tensor::from_vec(data_b, &[1, seq, embed])?;

        let out_a = ring.forward(&input_a, None)?.data()?;
        let out_b = ring.forward(&input_b, None)?.data()?;

        for i in 0..(seq - 1) * embed {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-4,
                "position {} changed when only the last token was perturbed",
                i / embed
            );
        }
        assert!(
            out_a[(seq - 1) * embed..]
                .iter()
                .zip(out_b[(seq - 1) * embed..].iter())
                .any(|(a, b)| (a - b).abs() > 1e-4),
            "the perturbed final token did not change its own output"
        );

        Ok(())
    }

    /// The block partition must not change the result: the same weights split
    /// into three blocks and into one block must agree.
    #[test]
    fn test_ring_result_is_independent_of_block_size() -> Result<()> {
        let seq = 24usize;
        let input = deterministic_input(1, seq, 8);

        let mut ring = RingAttention::new(tiny_config(3), 0)?;
        let blockwise = ring.forward(&input, None)?.data()?;
        assert_eq!(ring.attention_blocks().len(), 3);

        // Same instance, same weights, one block covering the whole sequence.
        ring.config.block_size = seq;
        let single_block = ring.forward(&input, None)?.data()?;
        assert_eq!(ring.attention_blocks().len(), 1);

        for (i, (blocked, single)) in blockwise.iter().zip(single_block.iter()).enumerate() {
            assert!(
                (blocked - single).abs() < 1e-3,
                "element {i}: 3 blocks {blocked} vs 1 block {single}"
            );
        }

        Ok(())
    }

    /// Every shard visits every ring position exactly once per forward pass.
    #[test]
    fn test_ring_rotation_visits_every_block_once() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(4), 0)?;
        let blocks = ring.partition_sequence(32)?;
        assert_eq!(blocks.len(), 4);

        let inner = ring.config.num_heads * ring.config.head_dim;
        let keys: Vec<f32> = (0..32 * inner).map(|i| i as f32).collect();
        let values = keys.clone();

        let mut resident: Vec<KvShard> =
            blocks.iter().map(|block| ring.shard_for_block(&keys, &values, block)).collect();

        let block_size = ring.config.block_size;
        let mut seen = vec![vec![false; blocks.len()]; blocks.len()];
        for _ in 0..blocks.len() {
            for (position, shard) in resident.iter().enumerate() {
                let block_id = shard.start_pos / block_size;
                assert!(
                    !seen[position][block_id],
                    "block {block_id} reached position {position} twice"
                );
                seen[position][block_id] = true;
                // The shard still carries its block's real key rows.
                assert_eq!(shard.keys[0], (shard.start_pos * inner) as f32);
            }
            resident.rotate_right(1);
        }

        for row in &seen {
            assert!(row.iter().all(|visited| *visited), "a shard never arrived");
        }

        // The rotation returns the ring to its starting arrangement.
        for (position, shard) in resident.iter().enumerate() {
            assert_eq!(shard.start_pos / block_size, position);
        }

        Ok(())
    }

    /// Shards carry the real projected K/V rows of their block.
    #[test]
    fn test_shard_carries_real_block_data() -> Result<()> {
        let ring = RingAttention::new(tiny_config(2), 0)?;
        let inner = ring.config.num_heads * ring.config.head_dim;
        let keys: Vec<f32> = (0..16 * inner).map(|i| i as f32).collect();
        let values: Vec<f32> = keys.iter().map(|v| -v).collect();

        let block = AttentionBlock::new(1, 1, 8, 16);
        let shard = ring.shard_for_block(&keys, &values, &block);

        assert_eq!(shard.start_pos, 8);
        assert_eq!(shard.seq_len, 8);
        assert_eq!(shard.keys.len(), 8 * inner);
        assert_eq!(shard.keys[0], (8 * inner) as f32);
        assert_eq!(shard.values[0], -((8 * inner) as f32));

        Ok(())
    }

    /// Configurations describing capabilities this engine does not have are
    /// rejected instead of being silently ignored.
    #[test]
    fn test_unsupported_configurations_are_rejected() {
        let overlap = RingAttentionConfig {
            block_overlap: 128,
            ..tiny_config(2)
        };
        assert!(RingAttention::new(overlap, 0).is_err());

        let nccl = RingAttentionConfig {
            communication_backend: CommunicationBackend::NCCL,
            ..tiny_config(2)
        };
        assert!(RingAttention::new(nccl, 0).is_err());

        let mixed = RingAttentionConfig {
            memory_optimization: MemoryOptimizationConfig {
                mixed_precision: true,
                ..MemoryOptimizationConfig::default()
            },
            ..tiny_config(2)
        };
        assert!(RingAttention::new(mixed, 0).is_err());
    }

    /// Sequences longer than the configured maximum are rejected.
    #[test]
    fn test_forward_rejects_oversized_sequence() -> Result<()> {
        let config = RingAttentionConfig {
            max_seq_length: 8,
            ..tiny_config(2)
        };
        let mut ring = RingAttention::new(config, 0)?;
        let input = deterministic_input(1, 16, 8);
        assert!(ring.forward(&input, None).is_err());
        Ok(())
    }

    /// Projections are sized once and a different embedding width is an error.
    #[test]
    fn test_projection_width_is_pinned_after_first_forward() -> Result<()> {
        let mut ring = RingAttention::new(tiny_config(2), 0)?;
        assert_eq!(ring.parameter_count(), 0);

        let input = deterministic_input(1, 16, 8);
        ring.forward(&input, None)?;
        let inner = ring.config.num_heads * ring.config.head_dim;
        assert_eq!(ring.parameter_count(), 4 * 8 * inner);

        let wider = deterministic_input(1, 16, 12);
        assert!(ring.forward(&wider, None).is_err());

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
        let configs = vec![tiny_config(2), tiny_config(2)];

        let coordination_config = CoordinationConfig::default();
        let mut manager = DistributedRingAttentionManager::new(configs, coordination_config)?;

        assert_eq!(manager.device_count(), 2);

        let input = deterministic_input(1, 16, 8);
        let output = manager.process_distributed(&input, None)?;
        assert_eq!(output.shape(), input.shape());
        assert!(output.data()?.iter().all(|v| v.is_finite()));

        Ok(())
    }

    /// Coordination features that are not implemented must be rejected.
    #[test]
    fn test_distributed_manager_rejects_unimplemented_coordination() {
        let configs = vec![tiny_config(2)];
        let coordination = CoordinationConfig {
            fault_tolerance: true,
            ..CoordinationConfig::default()
        };
        assert!(DistributedRingAttentionManager::new(configs, coordination).is_err());

        let configs = vec![tiny_config(2)];
        let coordination = CoordinationConfig {
            synchronization_strategy: SynchronizationStrategy::AsynchronousPipelined,
            ..CoordinationConfig::default()
        };
        assert!(DistributedRingAttentionManager::new(configs, coordination).is_err());
    }
}
