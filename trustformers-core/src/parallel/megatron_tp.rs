//! Tensor parallelism (Megatron-LM style) for distributed model execution.
//!
//! Columns split along output dim (AllReduce after), rows split along input dim
//! (ReduceScatter after). Designed for simulation of multi-rank execution in a
//! single process (testing / benchmarking) as well as real distributed settings.

use std::fmt;

// ---------------------------------------------------------------------------
// Error types
// ---------------------------------------------------------------------------

/// Errors that can arise when constructing or using tensor-parallel layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TpError {
    /// The given dimension is not evenly divisible by `world_size`.
    WorldSizeNotDivisible { dim: usize, world_size: usize },
    /// `rank` is outside [0, world_size).
    RankOutOfRange { rank: usize, world_size: usize },
    /// A dimension mismatch was detected during a forward pass.
    DimensionMismatch,
}

impl fmt::Display for TpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::WorldSizeNotDivisible { dim, world_size } => write!(
                f,
                "dimension {dim} is not divisible by world_size {world_size}"
            ),
            Self::RankOutOfRange { rank, world_size } => write!(
                f,
                "rank {rank} is out of range for world_size {world_size}"
            ),
            Self::DimensionMismatch => write!(f, "tensor dimension mismatch"),
        }
    }
}

impl std::error::Error for TpError {}

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Configuration shared by all tensor-parallel layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorParallelConfig {
    /// Total number of parallel ranks.  Must be ≥ 1.
    pub world_size: usize,
    /// This rank's index, in `0..world_size`.
    pub rank: usize,
    /// When `true`, use scatter-gather instead of all-reduce after column-parallel
    /// layers (activates sequence parallelism path).
    pub scatter_gather: bool,
    /// Sequence parallelism: distribute the sequence dimension across ranks for
    /// long-context workloads.
    pub sequence_parallel: bool,
}

impl Default for TensorParallelConfig {
    fn default() -> Self {
        Self {
            world_size: 1,
            rank: 0,
            scatter_gather: false,
            sequence_parallel: false,
        }
    }
}

impl TensorParallelConfig {
    /// Convenience constructor.
    pub fn new(world_size: usize, rank: usize) -> Result<Self, TpError> {
        if rank >= world_size {
            return Err(TpError::RankOutOfRange { rank, world_size });
        }
        Ok(Self { world_size, rank, scatter_gather: false, sequence_parallel: false })
    }

    /// Validate that `dim` is evenly divisible by `world_size`.
    fn check_divisible(&self, dim: usize) -> Result<(), TpError> {
        if dim % self.world_size != 0 {
            Err(TpError::WorldSizeNotDivisible { dim, world_size: self.world_size })
        } else {
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// ColumnParallelLinear
// ---------------------------------------------------------------------------

/// Column-parallel linear layer (Megatron-LM style).
///
/// Splits the weight matrix along the **output** dimension.  Each rank holds
/// a shard of size `[in_features, local_out]` where `local_out = out_features /
/// world_size`.  After the forward pass, ranks call `all_gather_output` to
/// reconstruct the full `[batch, out_features]` tensor.
pub struct ColumnParallelLinear {
    /// Weight shard: flattened row-major `[in_features, local_out]`.
    pub weight: Vec<f32>,
    /// Full input dimension.
    pub in_features: usize,
    /// Full output dimension (across all ranks).
    pub out_features: usize,
    /// Local (per-rank) output dimension = `out_features / world_size`.
    pub local_out: usize,
    /// Parallelism configuration.
    pub config: TensorParallelConfig,
}

impl ColumnParallelLinear {
    /// Create a new `ColumnParallelLinear`.  Weights are zero-initialised;
    /// call sites should overwrite `weight` with actual values.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: TensorParallelConfig,
    ) -> Result<Self, TpError> {
        if config.rank >= config.world_size {
            return Err(TpError::RankOutOfRange {
                rank: config.rank,
                world_size: config.world_size,
            });
        }
        config.check_divisible(out_features)?;
        let local_out = out_features / config.world_size;
        let weight = vec![0.0f32; in_features * local_out];
        Ok(Self { weight, in_features, out_features, local_out, config })
    }

    /// Forward pass: compute `x @ W_shard` for this rank.
    ///
    /// `x` is a flat row-major tensor of shape `[batch_size, in_features]`.
    /// Returns a flat row-major tensor of shape `[batch_size, local_out]`.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; batch_size * self.local_out];
        // Matrix multiply: out[b, j] = sum_k x[b, k] * weight[k, j]
        for b in 0..batch_size {
            for j in 0..self.local_out {
                let mut acc = 0.0f32;
                for k in 0..self.in_features {
                    acc += x[b * self.in_features + k] * self.weight[k * self.local_out + j];
                }
                out[b * self.local_out + j] = acc;
            }
        }
        out
    }

    /// Simulate an All-Gather across `world_size` ranks by concatenating the
    /// per-rank output shards along the column dimension.
    ///
    /// `local_outputs[r]` must have length `batch_size * local_out` with equal
    /// length across all ranks (equal partitioning assumed).  When `batch_size ==
    /// 1` this is simply a flat concatenation; for `batch_size > 1` use the
    /// `all_gather_output_batched` variant which additionally takes `batch_size`
    /// and `local_out` to correctly interleave columns.
    ///
    /// The `world_size` parameter is accepted for API symmetry but unused when
    /// the number of shards already equals the world size.
    pub fn all_gather_output(local_outputs: &[Vec<f32>], _world_size: usize) -> Vec<f32> {
        // Flat concatenation of equal-length shards.
        // Correct for batch_size=1 (each shard IS the full column block).
        let total = local_outputs.iter().map(|v| v.len()).sum();
        let mut result = Vec::with_capacity(total);
        for shard in local_outputs {
            result.extend_from_slice(shard);
        }
        result
    }

    /// Interleave-gather: correctly reassemble `[batch_size, out_features]` from
    /// per-rank `[batch_size, local_out]` shards when `batch_size > 1`.
    pub fn all_gather_output_batched(
        local_outputs: &[Vec<f32>],
        batch_size: usize,
        local_out: usize,
    ) -> Vec<f32> {
        let world_size = local_outputs.len();
        let out_features = world_size * local_out;
        let mut result = vec![0.0f32; batch_size * out_features];
        for (r, shard) in local_outputs.iter().enumerate() {
            for b in 0..batch_size {
                for j in 0..local_out {
                    result[b * out_features + r * local_out + j] =
                        shard[b * local_out + j];
                }
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// RowParallelLinear
// ---------------------------------------------------------------------------

/// Row-parallel linear layer (Megatron-LM style).
///
/// Splits the weight matrix along the **input** dimension.  Each rank holds a
/// weight shard `[local_in, out_features]`.  Each rank also receives the
/// corresponding input shard `x[:, rank*local_in:(rank+1)*local_in]`.  After the
/// forward pass, ranks call `all_reduce_output` to sum the partial results and
/// obtain the full `[batch, out_features]` output.
pub struct RowParallelLinear {
    /// Weight shard: flattened row-major `[local_in, out_features]`.
    pub weight: Vec<f32>,
    /// Full input dimension.
    pub in_features: usize,
    /// Local (per-rank) input dimension = `in_features / world_size`.
    pub local_in: usize,
    /// Full output dimension.
    pub out_features: usize,
    /// Parallelism configuration.
    pub config: TensorParallelConfig,
}

impl RowParallelLinear {
    /// Create a new `RowParallelLinear`.  Weights are zero-initialised.
    pub fn new(
        in_features: usize,
        out_features: usize,
        config: TensorParallelConfig,
    ) -> Result<Self, TpError> {
        if config.rank >= config.world_size {
            return Err(TpError::RankOutOfRange {
                rank: config.rank,
                world_size: config.world_size,
            });
        }
        config.check_divisible(in_features)?;
        let local_in = in_features / config.world_size;
        let weight = vec![0.0f32; local_in * out_features];
        Ok(Self { weight, in_features, local_in, out_features, config })
    }

    /// Forward pass: extract this rank's input shard from `x` and compute the
    /// partial matrix product.
    ///
    /// `x` is a flat row-major tensor of shape `[batch_size, in_features]`.
    /// Returns a partial sum of shape `[batch_size, out_features]`.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        let rank = self.config.rank;
        let start = rank * self.local_in;
        let mut out = vec![0.0f32; batch_size * self.out_features];
        for b in 0..batch_size {
            for j in 0..self.out_features {
                let mut acc = 0.0f32;
                for k in 0..self.local_in {
                    acc += x[b * self.in_features + start + k]
                        * self.weight[k * self.out_features + j];
                }
                out[b * self.out_features + j] = acc;
            }
        }
        out
    }

    /// Simulate an All-Reduce (sum) across ranks.
    ///
    /// `partial_outputs[r]` is the partial result from rank `r`, of shape
    /// `[batch_size, out_features]` (flat).  Returns their element-wise sum.
    pub fn all_reduce_output(partial_outputs: &[Vec<f32>]) -> Vec<f32> {
        if partial_outputs.is_empty() {
            return Vec::new();
        }
        let len = partial_outputs[0].len();
        let mut result = vec![0.0f32; len];
        for partial in partial_outputs {
            for (r, p) in result.iter_mut().zip(partial.iter()) {
                *r += p;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// VocabParallelEmbedding
// ---------------------------------------------------------------------------

/// Vocabulary-parallel embedding table.
///
/// The embedding table of shape `[vocab_size, hidden_size]` is partitioned across
/// ranks: rank `r` holds rows `[r*local_vocab_size, (r+1)*local_vocab_size)`.
/// For each token ID, exactly one rank has a non-zero embedding; an all-reduce
/// (element-wise sum) across ranks recovers the full embedding for every token.
pub struct VocabParallelEmbedding {
    /// Embedding shard: flattened row-major `[local_vocab_size, hidden_size]`.
    pub embedding: Vec<f32>,
    /// Number of vocab rows owned by this rank.
    pub local_vocab_size: usize,
    /// Embedding vector width.
    pub hidden_size: usize,
    /// First token index owned by this rank.
    pub vocab_start_idx: usize,
    /// One-past-last token index owned by this rank.
    pub vocab_end_idx: usize,
    /// Parallelism configuration.
    pub config: TensorParallelConfig,
}

impl VocabParallelEmbedding {
    /// Create a new `VocabParallelEmbedding`.  Embeddings are zero-initialised.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        config: TensorParallelConfig,
    ) -> Result<Self, TpError> {
        if config.rank >= config.world_size {
            return Err(TpError::RankOutOfRange {
                rank: config.rank,
                world_size: config.world_size,
            });
        }
        config.check_divisible(vocab_size)?;
        let local_vocab_size = vocab_size / config.world_size;
        let vocab_start_idx = config.rank * local_vocab_size;
        let vocab_end_idx = vocab_start_idx + local_vocab_size;
        let embedding = vec![0.0f32; local_vocab_size * hidden_size];
        Ok(Self {
            embedding,
            local_vocab_size,
            hidden_size,
            vocab_start_idx,
            vocab_end_idx,
            config,
        })
    }

    /// Forward pass: look up each token.  Tokens outside this rank's range produce
    /// a zero vector; the caller must call `all_reduce_embeddings` over all ranks.
    ///
    /// Returns a flat row-major tensor of shape `[num_tokens, hidden_size]`.
    pub fn forward(&self, token_ids: &[u32]) -> Vec<f32> {
        let n = token_ids.len();
        let mut out = vec![0.0f32; n * self.hidden_size];
        for (i, &tok) in token_ids.iter().enumerate() {
            let tok_usize = tok as usize;
            if tok_usize >= self.vocab_start_idx && tok_usize < self.vocab_end_idx {
                let local_idx = tok_usize - self.vocab_start_idx;
                let src = local_idx * self.hidden_size;
                let dst = i * self.hidden_size;
                out[dst..dst + self.hidden_size]
                    .copy_from_slice(&self.embedding[src..src + self.hidden_size]);
            }
            // else: zero (already initialised)
        }
        out
    }

    /// Simulate an All-Reduce (element-wise sum) over per-rank partial embeddings.
    ///
    /// Since exactly one rank has a non-zero embedding per token, the sum equals the
    /// correct embedding for every token.
    pub fn all_reduce_embeddings(partial: &[Vec<f32>]) -> Vec<f32> {
        if partial.is_empty() {
            return Vec::new();
        }
        let len = partial[0].len();
        let mut result = vec![0.0f32; len];
        for p in partial {
            for (r, v) in result.iter_mut().zip(p.iter()) {
                *r += v;
            }
        }
        result
    }
}

// ---------------------------------------------------------------------------
// TensorParallelLinear dispatch enum
// ---------------------------------------------------------------------------

/// Unified enum that dispatches to either `ColumnParallelLinear` or
/// `RowParallelLinear`.
pub enum TensorParallelLinear {
    Column(ColumnParallelLinear),
    Row(RowParallelLinear),
}

impl TensorParallelLinear {
    /// Forward pass — delegates to the inner variant.
    pub fn forward(&self, x: &[f32], batch_size: usize) -> Vec<f32> {
        match self {
            Self::Column(layer) => layer.forward(x, batch_size),
            Self::Row(layer) => layer.forward(x, batch_size),
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a weight matrix (identity-ish) for testing
    fn identity_weight(rows: usize, cols: usize) -> Vec<f32> {
        let mut w = vec![0.0f32; rows * cols];
        for i in 0..rows.min(cols) {
            w[i * cols + i] = 1.0;
        }
        w
    }

    // -----------------------------------------------------------------------
    // Test 1: ColumnParallelLinear output shape
    // -----------------------------------------------------------------------
    #[test]
    fn test_column_parallel_output_shape() {
        let cfg = TensorParallelConfig::new(4, 0).expect("valid config");
        let layer = ColumnParallelLinear::new(8, 16, cfg).expect("valid layer");
        let x = vec![1.0f32; 2 * 8]; // batch=2, in=8
        let out = layer.forward(&x, 2);
        assert_eq!(out.len(), 2 * 4); // batch=2, local_out=4
    }

    // -----------------------------------------------------------------------
    // Test 2: ColumnParallelLinear correct numerical output
    // -----------------------------------------------------------------------
    #[test]
    fn test_column_parallel_correct_output() {
        let cfg = TensorParallelConfig::new(2, 0).expect("valid config");
        let mut layer = ColumnParallelLinear::new(4, 4, cfg).expect("valid layer");
        // weight[k, j] = 1 if k == j else 0  (identity for first 2 cols)
        layer.weight = identity_weight(4, 2);
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, in=4
        let out = layer.forward(&x, 1);
        // out[0] = x dot col0 = 1.0, out[1] = x dot col1 = 2.0
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 3: RowParallelLinear correct output
    // -----------------------------------------------------------------------
    #[test]
    fn test_row_parallel_correct_output() {
        // world_size=2, rank=0: owns input cols 0..2
        let cfg = TensorParallelConfig::new(2, 0).expect("valid config");
        let mut layer = RowParallelLinear::new(4, 4, cfg).expect("valid layer");
        // weight[local_k, j] = 1 if local_k == j else 0
        layer.weight = identity_weight(2, 4);
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // batch=1, in=4
        let partial = layer.forward(&x, 1);
        // rank 0 input shard = [1.0, 2.0]; partial[0]=1.0, partial[1]=2.0, rest 0
        assert!((partial[0] - 1.0).abs() < 1e-6);
        assert!((partial[1] - 2.0).abs() < 1e-6);
        assert!((partial[2]).abs() < 1e-6);
        assert!((partial[3]).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 4: VocabParallel in-range lookup
    // -----------------------------------------------------------------------
    #[test]
    fn test_vocab_parallel_in_range() {
        // world_size=4, rank=1: owns tokens 2..4 (vocab=8, hidden=3)
        let cfg = TensorParallelConfig::new(4, 1).expect("valid config");
        let mut emb = VocabParallelEmbedding::new(8, 3, cfg).expect("valid embedding");
        // Set embedding for local idx 0 (global token 2)
        emb.embedding[0] = 1.0;
        emb.embedding[1] = 2.0;
        emb.embedding[2] = 3.0;
        let out = emb.forward(&[2u32]);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 5: VocabParallel out-of-range returns zeros
    // -----------------------------------------------------------------------
    #[test]
    fn test_vocab_parallel_out_of_range() {
        let cfg = TensorParallelConfig::new(4, 0).expect("valid config");
        let emb = VocabParallelEmbedding::new(8, 3, cfg).expect("valid embedding");
        // Token 5 is owned by rank 2 (tokens 4..6); rank 0 should return zeros
        let out = emb.forward(&[5u32]);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    // -----------------------------------------------------------------------
    // Test 6: all_gather concatenation
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_gather_output_concatenation() {
        let shard0 = vec![1.0f32, 2.0, 3.0]; // batch=1, local_out=3
        let shard1 = vec![4.0f32, 5.0, 6.0];
        let gathered =
            ColumnParallelLinear::all_gather_output(&[shard0, shard1], 2);
        assert_eq!(gathered, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    }

    // -----------------------------------------------------------------------
    // Test 6b: all_gather_output_batched row-interleave
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_gather_output_batched() {
        // batch=2, local_out=2, world_size=2
        // rank0 output: rows [a0,a1] [a2,a3]
        // rank1 output: rows [b0,b1] [b2,b3]
        // expected: [[a0,a1,b0,b1],[a2,a3,b2,b3]]
        let shard0 = vec![1.0f32, 2.0, 3.0, 4.0]; // batch=2, local_out=2
        let shard1 = vec![5.0f32, 6.0, 7.0, 8.0];
        let gathered = ColumnParallelLinear::all_gather_output_batched(
            &[shard0, shard1],
            2,
            2,
        );
        // row 0: [1,2, 5,6], row 1: [3,4, 7,8]
        assert_eq!(gathered, vec![1.0, 2.0, 5.0, 6.0, 3.0, 4.0, 7.0, 8.0]);
    }

    // -----------------------------------------------------------------------
    // Test 7: all_reduce sum
    // -----------------------------------------------------------------------
    #[test]
    fn test_all_reduce_output_sum() {
        let p0 = vec![1.0f32, 0.0, 0.0, 0.0];
        let p1 = vec![0.0f32, 2.0, 0.0, 0.0];
        let p2 = vec![0.0f32, 0.0, 3.0, 0.0];
        let reduced = RowParallelLinear::all_reduce_output(&[p0, p1, p2]);
        assert!((reduced[0] - 1.0).abs() < 1e-6);
        assert!((reduced[1] - 2.0).abs() < 1e-6);
        assert!((reduced[2] - 3.0).abs() < 1e-6);
        assert!((reduced[3]).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 8: Single rank (world_size=1) behaves like regular linear
    // -----------------------------------------------------------------------
    #[test]
    fn test_single_rank_column_linear() {
        let cfg = TensorParallelConfig::default(); // world_size=1, rank=0
        let mut layer = ColumnParallelLinear::new(3, 3, cfg).expect("valid layer");
        layer.weight = identity_weight(3, 3);
        let x = vec![1.0f32, 2.0, 3.0];
        let out = layer.forward(&x, 1);
        // Identity: out == x
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 3.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 9: Divisibility error
    // -----------------------------------------------------------------------
    #[test]
    fn test_world_size_not_divisible_error() {
        let cfg = TensorParallelConfig::new(4, 0).expect("valid config");
        let result = ColumnParallelLinear::new(8, 7, cfg); // 7 not divisible by 4
        assert!(matches!(
            result,
            Err(TpError::WorldSizeNotDivisible { dim: 7, world_size: 4 })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 10: Rank out of range error
    // -----------------------------------------------------------------------
    #[test]
    fn test_rank_out_of_range_error() {
        let result = TensorParallelConfig::new(4, 4); // rank == world_size
        assert!(matches!(
            result,
            Err(TpError::RankOutOfRange { rank: 4, world_size: 4 })
        ));
    }

    // -----------------------------------------------------------------------
    // Test 11: VocabParallel each rank handles its range
    // -----------------------------------------------------------------------
    #[test]
    fn test_vocab_parallel_each_rank() {
        // vocab=8, hidden=2, world_size=4
        // rank r handles tokens [2r, 2r+2)
        let hidden = 2usize;
        let mut all_results: Vec<Vec<f32>> = Vec::new();
        for r in 0..4usize {
            let cfg = TensorParallelConfig::new(4, r).expect("valid config");
            let mut emb = VocabParallelEmbedding::new(8, hidden, cfg)
                .expect("valid embedding");
            // Give each row a distinctive value
            for local_i in 0..2usize {
                let global_tok = r * 2 + local_i;
                emb.embedding[local_i * hidden] = global_tok as f32 * 10.0;
                emb.embedding[local_i * hidden + 1] = global_tok as f32 * 10.0 + 1.0;
            }
            // Look up all 8 tokens
            let out = emb.forward(&[0u32, 1, 2, 3, 4, 5, 6, 7]);
            all_results.push(out);
        }
        // all-reduce: sum
        let final_emb = VocabParallelEmbedding::all_reduce_embeddings(&all_results);
        // Token 0 embedding: [0.0, 1.0]
        assert!((final_emb[0] - 0.0).abs() < 1e-6);
        assert!((final_emb[1] - 1.0).abs() < 1e-6);
        // Token 3 embedding: [30.0, 31.0]
        assert!((final_emb[3 * hidden] - 30.0).abs() < 1e-6);
        assert!((final_emb[3 * hidden + 1] - 31.0).abs() < 1e-6);
        // Token 7 embedding: [70.0, 71.0]
        assert!((final_emb[7 * hidden] - 70.0).abs() < 1e-6);
        assert!((final_emb[7 * hidden + 1] - 71.0).abs() < 1e-6);
    }

    // -----------------------------------------------------------------------
    // Test 12: TensorParallelLinear enum dispatch
    // -----------------------------------------------------------------------
    #[test]
    fn test_tensor_parallel_linear_dispatch() {
        let cfg = TensorParallelConfig::new(2, 0).expect("valid config");
        let layer = TensorParallelLinear::Column(
            ColumnParallelLinear::new(4, 4, cfg).expect("valid layer"),
        );
        let x = vec![0.0f32; 1 * 4];
        let out = layer.forward(&x, 1);
        assert_eq!(out.len(), 2); // local_out = 4/2 = 2
    }

    // -----------------------------------------------------------------------
    // Test 13: Tensor parallel attention head split simulation
    // -----------------------------------------------------------------------
    #[test]
    fn test_attention_head_split() {
        // Simulate splitting attention heads across 4 ranks.
        // num_heads=8, head_dim=16 => out_features = 8*16 = 128
        // Each rank handles 2 heads => local_out = 32
        let world_size = 4usize;
        let in_features = 64usize;
        let out_features = 128usize; // num_heads * head_dim
        let mut shards: Vec<Vec<f32>> = Vec::new();
        let mut layers: Vec<ColumnParallelLinear> = Vec::new();

        for r in 0..world_size {
            let cfg = TensorParallelConfig::new(world_size, r).expect("valid config");
            let layer = ColumnParallelLinear::new(in_features, out_features, cfg)
                .expect("valid layer");
            layers.push(layer);
        }

        let x = vec![1.0f32; in_features]; // batch=1
        for layer in &layers {
            shards.push(layer.forward(&x, 1));
        }

        // Each shard should have 32 elements (128 / 4)
        assert!(shards.iter().all(|s| s.len() == 32));

        // After gather we should get 128 elements total
        let gathered = ColumnParallelLinear::all_gather_output_batched(&shards, 1, 32);
        assert_eq!(gathered.len(), out_features);
    }

    // -----------------------------------------------------------------------
    // Test 14: Row parallel all-reduce multi-rank simulation
    // -----------------------------------------------------------------------
    #[test]
    fn test_row_parallel_multi_rank_simulation() {
        let world_size = 2usize;
        let in_features = 4usize;
        let out_features = 4usize;

        // Build layers for each rank
        let mut partials: Vec<Vec<f32>> = Vec::new();
        let x: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0]; // batch=1

        for r in 0..world_size {
            let cfg = TensorParallelConfig::new(world_size, r).expect("valid config");
            let mut layer =
                RowParallelLinear::new(in_features, out_features, cfg).expect("valid layer");
            // rank 0: weight = identity for first 2 inputs
            // rank 1: weight = identity for last 2 inputs
            if r == 0 {
                layer.weight = identity_weight(2, 4);
            } else {
                // local_in=2, out=4; identity for local rows 0,1 -> global cols 2,3
                let mut w = vec![0.0f32; 2 * 4];
                w[0 * 4 + 2] = 1.0; // local input 0 -> output col 2
                w[1 * 4 + 3] = 1.0; // local input 1 -> output col 3
                layer.weight = w;
            }
            partials.push(layer.forward(&x, 1));
        }

        let result = RowParallelLinear::all_reduce_output(&partials);
        // rank0 shard=[1,2] -> output=[1,2,0,0]
        // rank1 shard=[3,4] -> output=[0,0,3,4]
        // reduced: [1,2,3,4]
        assert!((result[0] - 1.0).abs() < 1e-6);
        assert!((result[1] - 2.0).abs() < 1e-6);
        assert!((result[2] - 3.0).abs() < 1e-6);
        assert!((result[3] - 4.0).abs() < 1e-6);
    }
}
