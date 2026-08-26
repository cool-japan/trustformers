use trustformers_core::{
    errors::{invalid_input, unsupported_operation, Result},
    tensor::Tensor,
};

/// Output structure for cross-attention operations
#[derive(Debug, Clone)]
pub struct CrossAttentionOutput {
    /// Attention output tensor
    pub output: Tensor,
    /// Attention weights (optional)
    pub attention_weights: Option<Tensor>,
    /// Attention statistics (optional)
    pub attention_stats: Option<AttentionStats>,
}

/// Statistics about attention patterns
#[derive(Debug, Clone)]
pub struct AttentionStats {
    /// Average attention entropy
    pub entropy: f32,
    /// Maximum attention weight
    pub max_weight: f32,
    /// Minimum attention weight
    pub min_weight: f32,
    /// Attention sparsity ratio
    pub sparsity: f32,
    /// Head-wise statistics
    pub head_stats: Vec<HeadStats>,
}

/// Per-head attention statistics
#[derive(Debug, Clone)]
pub struct HeadStats {
    /// Head index
    pub head_idx: usize,
    /// Head entropy
    pub entropy: f32,
    /// Head sparsity
    pub sparsity: f32,
    /// Most attended positions
    pub top_positions: Vec<usize>,
}

/// Create attention mask for cross-attention
pub fn create_attention_mask(
    query_len: usize,
    key_len: usize,
    mask_type: MaskType,
) -> Result<Tensor> {
    match mask_type {
        MaskType::None => {
            // No masking - all positions visible
            Ok(Tensor::zeros(&[query_len, key_len])?)
        },
        MaskType::Causal => {
            // Causal mask - can only attend to previous positions
            let mut mask = vec![vec![0.0f32; key_len]; query_len];
            for (i, row) in mask.iter_mut().enumerate() {
                for j in (i + 1)..key_len.min(query_len) {
                    row[j] = f32::NEG_INFINITY;
                }
            }
            let flattened: Vec<f32> = mask.into_iter().flatten().collect();
            Ok(Tensor::from_vec(flattened, &[query_len, key_len])?)
        },
        MaskType::Local(window_size) => {
            // Local attention mask - only attend to nearby positions
            let mut mask = vec![vec![f32::NEG_INFINITY; key_len]; query_len];
            for (i, row) in mask.iter_mut().enumerate() {
                let start = i.saturating_sub(window_size / 2);
                let end = (i + window_size / 2 + 1).min(key_len);
                for j in start..end {
                    row[j] = 0.0;
                }
            }
            let flattened: Vec<f32> = mask.into_iter().flatten().collect();
            Ok(Tensor::from_vec(flattened, &[query_len, key_len])?)
        },
        MaskType::Custom(custom_mask) => Ok(custom_mask),
    }
}

/// Types of attention masks
#[derive(Debug, Clone)]
pub enum MaskType {
    /// No masking
    None,
    /// Causal masking for autoregressive models
    Causal,
    /// Local window masking
    Local(usize),
    /// Custom mask tensor
    Custom(Tensor),
}

/// Create sparse attention mask
pub fn create_sparse_mask(
    query_len: usize,
    key_len: usize,
    sparsity_ratio: f32,
    pattern: SparsePattern,
) -> Result<Tensor> {
    match pattern {
        SparsePattern::Random { connections, seed } => {
            create_random_sparse_mask(query_len, key_len, sparsity_ratio, connections, seed)
        },
        SparsePattern::Block(block_size) => {
            create_block_sparse_mask(query_len, key_len, block_size)
        },
        SparsePattern::Strided(stride) => create_strided_sparse_mask(query_len, key_len, stride),
        SparsePattern::TopK(k) => create_topk_sparse_mask(query_len, key_len, k),
    }
}

/// Sparse attention patterns
#[derive(Debug, Clone)]
pub enum SparsePattern {
    /// Random sparse connections, deterministic given `seed`.
    ///
    /// When `connections` is `Some(k)`, every query row keeps exactly `k`
    /// randomly chosen keys (bounded random attention, e.g. BigBird-style).
    /// When `None`, every cell is kept independently with probability
    /// `1 - sparsity_ratio` (the ratio [`create_sparse_mask`] was called
    /// with).
    Random {
        connections: Option<usize>,
        seed: u64,
    },
    /// Block-based sparse connections
    Block(usize),
    /// Strided sparse connections
    Strided(usize),
    /// Top-k sparse connections
    TopK(usize),
}

/// Builds a genuinely random (seeded, reproducible) sparse attention mask.
///
/// `connections = Some(k)`: each query row keeps exactly `min(k, key_len)`
/// keys, chosen without replacement via a partial Fisher-Yates shuffle --
/// the standard bounded-random-attention pattern (BigBird's "random"
/// component). `connections = None`: each cell is kept independently with
/// probability `1 - sparsity_ratio` (a Bernoulli mask). Either way, the same
/// `seed` (with the same `query_len`/`key_len`/`sparsity_ratio`/
/// `connections`) always reproduces the same mask -- this replaced a fixed
/// `(i + j) % 10` stripe pattern that was identical on every call and never
/// read `connections` at all.
fn create_random_sparse_mask(
    query_len: usize,
    key_len: usize,
    sparsity_ratio: f32,
    connections: Option<usize>,
    seed: u64,
) -> Result<Tensor> {
    let mut rng = fastrand::Rng::with_seed(seed);
    let mut mask = vec![f32::NEG_INFINITY; query_len * key_len];

    match connections {
        Some(k) => {
            let k = k.min(key_len);
            let mut candidates: Vec<usize> = (0..key_len).collect();
            for i in 0..query_len {
                // Partial Fisher-Yates: after this loop, candidates[..k] is
                // a uniformly random k-subset of 0..key_len, in random
                // order. Re-shuffled per row so each query gets its own
                // random subset rather than every row sharing one.
                for a in 0..k {
                    let b = a + rng.usize(0..(key_len - a));
                    candidates.swap(a, b);
                }
                let row_start = i * key_len;
                for &j in &candidates[..k] {
                    mask[row_start + j] = 0.0;
                }
            }
        },
        None => {
            let keep_prob = (1.0 - sparsity_ratio).clamp(0.0, 1.0);
            for cell in &mut mask {
                if rng.f32() < keep_prob {
                    *cell = 0.0;
                }
            }
        },
    }

    Tensor::from_vec(mask, &[query_len, key_len])
}

fn create_block_sparse_mask(query_len: usize, key_len: usize, block_size: usize) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; key_len]; query_len];

    for (i, row) in mask.iter_mut().enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            let qi = i / block_size;
            let kj = j / block_size;

            // Allow attention within blocks and to adjacent blocks
            if qi == kj || qi.abs_diff(kj) <= 1 {
                *val = 0.0;
            }
        }
    }

    let flattened: Vec<f32> = mask.into_iter().flatten().collect();
    Tensor::from_vec(flattened, &[query_len, key_len])
}

fn create_strided_sparse_mask(query_len: usize, key_len: usize, stride: usize) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; key_len]; query_len];

    for (i, row) in mask.iter_mut().enumerate() {
        for (j, val) in row.iter_mut().enumerate() {
            // Allow attention to positions at regular intervals
            if j % stride == i % stride {
                *val = 0.0;
            }
        }
    }

    let flattened: Vec<f32> = mask.into_iter().flatten().collect();
    Tensor::from_vec(flattened, &[query_len, key_len])
}

fn create_topk_sparse_mask(query_len: usize, key_len: usize, k: usize) -> Result<Tensor> {
    let mut mask = vec![vec![f32::NEG_INFINITY; key_len]; query_len];

    for (i, row) in mask.iter_mut().enumerate() {
        // Allow attention to k nearest positions
        let start = i.saturating_sub(k / 2);
        let end = (i + k / 2 + 1).min(key_len);

        for j in start..end {
            row[j] = 0.0;
        }
    }

    let flattened: Vec<f32> = mask.into_iter().flatten().collect();
    Tensor::from_vec(flattened, &[query_len, key_len])
}

/// Create hierarchical attention mask
pub fn create_hierarchical_mask(
    query_len: usize,
    key_len: usize,
    num_levels: usize,
    pooling_factor: usize,
) -> Result<Vec<Tensor>> {
    let mut masks = Vec::new();

    for level in 0..num_levels {
        let level_pooling = pooling_factor.pow(level as u32);
        let level_query_len = query_len.div_ceil(level_pooling);
        let level_key_len = key_len.div_ceil(level_pooling);

        let mask = create_attention_mask(level_query_len, level_key_len, MaskType::None)?;
        masks.push(mask);
    }

    Ok(masks)
}

/// Compute attention statistics.
///
/// `attention_weights` is expected to be `[batch_size, num_heads, seq_len, seq_len]` when
/// `num_heads > 0` (per-head stats select along dimension 1); the overall statistics only
/// require at least one dimension. Returns a structured error, rather than panicking, when
/// the tensor's rank doesn't support the requested `num_heads`.
pub fn compute_attention_stats(
    attention_weights: &Tensor,
    num_heads: usize,
) -> Result<AttentionStats> {
    let shape = attention_weights.shape();
    if shape.is_empty() {
        return Err(invalid_input(
            "compute_attention_stats: attention_weights tensor has no dimensions",
        ));
    }
    if num_heads > 0 && shape.len() < 2 {
        return Err(invalid_input(format!(
            "compute_attention_stats: num_heads={num_heads} > 0 requires attention_weights to \
             have at least 2 dimensions (expected [batch_size, num_heads, seq_len, seq_len]), \
             got shape {shape:?}"
        )));
    }

    // Compute overall statistics
    let entropy = compute_entropy(attention_weights)?;
    let (min_weight, max_weight) = compute_min_max(attention_weights)?;
    let sparsity = compute_sparsity(attention_weights, 1e-6)?;

    // Compute per-head statistics
    let mut head_stats = Vec::new();
    for head in 0..num_heads {
        // Select the specific head weights from the attention tensor
        // Shape: [batch_size, num_heads, seq_len, seq_len] -> [batch_size, seq_len, seq_len]
        let head_weights = attention_weights.select(1, head as i64)?;
        let head_entropy = compute_entropy(&head_weights)?;
        let head_sparsity = compute_sparsity(&head_weights, 1e-6)?;
        let top_positions = compute_top_positions(&head_weights, 5)?;

        head_stats.push(HeadStats {
            head_idx: head,
            entropy: head_entropy,
            sparsity: head_sparsity,
            top_positions,
        });
    }

    Ok(AttentionStats {
        entropy,
        max_weight,
        min_weight,
        sparsity,
        head_stats,
    })
}

/// Shannon entropy of the attention distribution, in nats, averaged across every attended-from
/// position in `tensor`.
///
/// `tensor` holds one or more attention-weight rows along its last dimension (the softmax
/// axis: for each query position, the weights over key positions sum to ~1). Entropy is
/// computed per row as `-sum(p * ln(p))` over that last dimension (zero-weight entries
/// contribute 0, matching the standard `0 * ln(0) = 0` convention), then averaged over all
/// rows so the result is a single scalar summarizing the whole tensor. A uniform distribution
/// over `n` positions has entropy `ln(n)` (maximum); a one-hot distribution has entropy `0`
/// (minimum).
fn compute_entropy(tensor: &Tensor) -> Result<f32> {
    let shape = tensor.shape();
    let last_dim = *shape
        .last()
        .ok_or_else(|| invalid_input("attention tensor has no dimensions"))?;
    if last_dim == 0 {
        return Err(invalid_input("attention tensor's last dimension is empty"));
    }

    let data = tensor.data()?;
    if data.is_empty() {
        return Err(invalid_input("attention tensor has no elements"));
    }

    let mut total_entropy = 0.0f64;
    let mut num_rows = 0usize;
    for row in data.chunks(last_dim) {
        let mut row_entropy = 0.0f64;
        for &p in row {
            let p = p as f64;
            if p > 0.0 {
                row_entropy -= p * p.ln();
            }
        }
        total_entropy += row_entropy;
        num_rows += 1;
    }

    Ok((total_entropy / num_rows as f64) as f32)
}

/// Minimum and maximum attention weight anywhere in `tensor`.
fn compute_min_max(tensor: &Tensor) -> Result<(f32, f32)> {
    let data = tensor.data()?;
    let mut iter = data.iter().copied();
    let first = iter.next().ok_or_else(|| invalid_input("attention tensor has no elements"))?;

    let (min, max) = iter.fold((first, first), |(min, max), v| (min.min(v), max.max(v)));
    Ok((min, max))
}

/// Fraction of attention weights at or below `threshold`, i.e. the fraction of (query, key)
/// pairs the model effectively ignores.
fn compute_sparsity(tensor: &Tensor, threshold: f32) -> Result<f32> {
    let data = tensor.data()?;
    if data.is_empty() {
        return Err(invalid_input("attention tensor has no elements"));
    }

    let below = data.iter().filter(|&&v| v <= threshold).count();
    Ok(below as f32 / data.len() as f32)
}

/// Indices of the `k` most-attended positions along `tensor`'s last dimension, averaged over
/// every attended-from row and ranked by that average weight (descending; ties broken by
/// ascending index). When multiple query positions are present, this reports which key
/// positions receive the most attention on average across them, not per-query top-k.
fn compute_top_positions(tensor: &Tensor, k: usize) -> Result<Vec<usize>> {
    let shape = tensor.shape();
    let last_dim = *shape
        .last()
        .ok_or_else(|| invalid_input("attention tensor has no dimensions"))?;
    if last_dim == 0 {
        return Err(invalid_input("attention tensor's last dimension is empty"));
    }

    let data = tensor.data()?;
    if data.is_empty() {
        return Err(invalid_input("attention tensor has no elements"));
    }

    let mut sums = vec![0.0f64; last_dim];
    let mut num_rows = 0usize;
    for row in data.chunks(last_dim) {
        for (pos, &v) in row.iter().enumerate() {
            sums[pos] += v as f64;
        }
        num_rows += 1;
    }

    let mut averaged: Vec<(usize, f64)> =
        sums.into_iter().map(|s| s / num_rows as f64).enumerate().collect();
    // Descending by average weight; ascending index breaks ties so the result is deterministic.
    averaged.sort_by(|(idx_a, val_a), (idx_b, val_b)| {
        val_b
            .partial_cmp(val_a)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(idx_a.cmp(idx_b))
    });

    Ok(averaged.into_iter().take(k.min(last_dim)).map(|(idx, _)| idx).collect())
}

/// Apply attention dropout
pub fn apply_attention_dropout(
    attention_weights: Tensor,
    dropout_rate: f32,
    training: bool,
) -> Result<Tensor> {
    if training && dropout_rate > 0.0 {
        attention_weights.dropout(dropout_rate)
    } else {
        Ok(attention_weights)
    }
}

/// Compute scaled dot-product attention
pub fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    mask: Option<&Tensor>,
    scale: f32,
    dropout_rate: f32,
    training: bool,
) -> Result<CrossAttentionOutput> {
    // Compute attention scores
    let key_shape = key.shape();
    let dim0 = key_shape.len().saturating_sub(2);
    let dim1 = key_shape.len().saturating_sub(1);
    let scores = query.matmul(&key.transpose(dim0, dim1)?)?;
    let scores = scores.mul_scalar(scale)?;

    // Apply mask if provided
    let scores = if let Some(mask) = mask { scores.add(mask)? } else { scores };

    // Apply softmax
    let attention_weights = scores.softmax(-1)?;

    // Apply dropout
    let attention_weights = apply_attention_dropout(attention_weights, dropout_rate, training)?;

    // Apply attention to values
    let output = attention_weights.matmul(value)?;

    Ok(CrossAttentionOutput {
        output,
        attention_weights: Some(attention_weights),
        attention_stats: None,
    })
}

/// Reshape tensor for multi-head attention
pub fn reshape_for_multihead(tensor: Tensor, num_heads: usize, head_dim: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    let batch_size = shape[0];
    let seq_len = shape[1];

    tensor.reshape(&[batch_size, seq_len, num_heads, head_dim])?.transpose(1, 2)
}

/// Reshape tensor back from multi-head attention
pub fn reshape_from_multihead(tensor: Tensor, hidden_size: usize) -> Result<Tensor> {
    let shape = tensor.shape();
    let batch_size = shape[0];
    let seq_len = shape[2];

    tensor.transpose(1, 2)?.reshape(&[batch_size, seq_len, hidden_size])
}

/// Pool tensor for hierarchical attention.
pub fn pool_tensor(tensor: Tensor, pooling_factor: usize, method: PoolingMethod) -> Result<Tensor> {
    match method {
        PoolingMethod::Average => average_pool_1d(tensor, pooling_factor),
        PoolingMethod::Max => max_pool_1d(tensor, pooling_factor),
        PoolingMethod::Learnable => Err(unsupported_operation(
            "pool_tensor(PoolingMethod::Learnable)",
            "this free function carries no learned-weight parameter to pool with -- real \
             learnable pooling in this crate is HierarchicalCrossAttention's per-level \
             `pooling_layers` (a Linear applied after PoolingMethod::Average pooling; see \
             HierarchicalAttentionConfig::learnable_pooling). Use PoolingMethod::Average or \
             PoolingMethod::Max with this function instead of silently substituting one of them.",
        )),
    }
}

/// Downsamples an additive attention mask's key axis to track pooled K/V.
///
/// `HierarchicalCrossAttention` shrinks `current_key`/`current_value`'s
/// sequence axis by `pooling_factor` between levels (via [`pool_tensor`]
/// with [`PoolingMethod::Average`]); a caller-supplied mask's key axis does
/// not shrink on its own, so without this, the next level's
/// `MultiHeadCrossAttention::forward` tries to broadcast a stale,
/// larger-than-`key_len` mask against the now-smaller key/value tensors.
///
/// Reuses [`pool_tensor`] with [`PoolingMethod::Average`] itself (the same
/// pooling the K/V tensors go through, so the two end up with identical
/// pooled lengths, both `seq_len.div_ceil(pooling_factor)`), by reshaping
/// the mask so its key axis plays the role of `pool_tensor`'s `seq_len`
/// axis. This is also the conservative combination for a `0.0`/`-inf`
/// additive mask: a window pools to `f32::NEG_INFINITY` (masked) as soon as
/// ANY position inside it was masked, and only stays `0.0` (visible) when
/// every position in the window was -- `NEG_INFINITY` is absorbing under
/// averaging with any finite number of finite addends (`0.0` is the only
/// other value ever present here, so there is no `inf - inf = NaN` risk).
///
/// Accepts a 2-D `[query_len, key_len]` or 3-D `[batch, query_len, key_len]`
/// mask -- the two shapes [`MultiHeadCrossAttention`](super::layers::MultiHeadCrossAttention)'s
/// `forward` itself accepts; other ranks are refused.
pub fn pool_mask_key_axis(mask: Tensor, pooling_factor: usize) -> Result<Tensor> {
    let shape = mask.shape();
    match shape.len() {
        2 => {
            let (query_len, key_len) = (shape[0], shape[1]);
            // Each query row becomes an independent "batch" element; the
            // key axis becomes the "seq_len" being pooled; a dummy
            // "hidden"=1 axis satisfies `pool_tensor`'s
            // [batch, seq_len, hidden] contract unchanged.
            let reshaped = mask.reshape(&[query_len, key_len, 1])?;
            let pooled = pool_tensor(reshaped, pooling_factor, PoolingMethod::Average)?;
            let pooled_key_len = pooled.shape()[1];
            pooled.reshape(&[query_len, pooled_key_len])
        },
        3 => {
            let (batch, query_len, key_len) = (shape[0], shape[1], shape[2]);
            let reshaped = mask.reshape(&[batch * query_len, key_len, 1])?;
            let pooled = pool_tensor(reshaped, pooling_factor, PoolingMethod::Average)?;
            let pooled_key_len = pooled.shape()[1];
            pooled.reshape(&[batch, query_len, pooled_key_len])
        },
        other => Err(invalid_input(format!(
            "pool_mask_key_axis expects a 2-D [query_len, key_len] or 3-D \
             [batch, query_len, key_len] mask, got a {other}-D shape {shape:?}"
        ))),
    }
}

/// Pooling methods for hierarchical attention.
#[derive(Debug, Clone)]
pub enum PoolingMethod {
    /// Average pooling (implemented; see `average_pool_1d`).
    Average,
    /// Max pooling (implemented; see `max_pool_1d`).
    Max,
    /// Learned pooling weights. **Not implemented by [`pool_tensor`]**: this free function has
    /// no parameter through which to supply learned weights, so selecting this variant returns
    /// a structured error rather than silently behaving like [`PoolingMethod::Average`]. Real
    /// learnable pooling in this crate is `HierarchicalCrossAttention`'s `pooling_layers` (see
    /// `HierarchicalAttentionConfig::learnable_pooling`).
    Learnable,
}

/// Validates that `tensor` is rank-3 `[batch, seq_len, hidden]` -- the shape convention every
/// pooling/interpolation helper below assumes, matching [`reshape_for_multihead`]'s convention
/// for the same tensors -- and returns its three dimensions.
fn expect_batch_seq_hidden(tensor: &Tensor, op: &str) -> Result<(usize, usize, usize)> {
    let shape = tensor.shape();
    if shape.len() != 3 {
        return Err(invalid_input(format!(
            "{op} expects a [batch, seq_len, hidden] (rank-3) tensor, got shape {shape:?}"
        )));
    }
    let (batch, seq_len, hidden) = (shape[0], shape[1], shape[2]);
    if seq_len == 0 {
        return Err(invalid_input(format!(
            "{op}: seq_len must be greater than 0, got shape {shape:?}"
        )));
    }
    Ok((batch, seq_len, hidden))
}

/// Real 1-D average pooling over the sequence axis (dim 1) of a `[batch, seq_len, hidden]`
/// tensor.
///
/// Uses **fixed, non-overlapping windows** of `pooling_factor` sequence positions (stride equal
/// to the window size) rather than resampling to a fixed target length -- [`interpolate_tensor`]
/// is the fixed-target-length operation. The output sequence length is
/// `seq_len.div_ceil(pooling_factor)`, matching [`create_hierarchical_mask`]'s per-level length
/// formula (`pooling_factor.pow(level)` combined with `div_ceil`), so a mask built for hierarchy
/// level `L+1` lines up with a tensor pooled one step from level `L` -- this is the contract
/// `HierarchicalCrossAttention::forward`'s per-level pooling actually needs. A trailing partial
/// window (when `seq_len` doesn't evenly divide by `pooling_factor`) is averaged over only its
/// actual members; it is never zero-padded.
fn average_pool_1d(tensor: Tensor, pooling_factor: usize) -> Result<Tensor> {
    pool_1d_windows(tensor, pooling_factor, "average_pool_1d", |window| {
        window.iter().sum::<f32>() / window.len() as f32
    })
}

/// Real 1-D max pooling over the sequence axis. See [`average_pool_1d`] for the window-size /
/// output-length contract (identical here; each window is reduced with `max` instead of
/// `average`).
fn max_pool_1d(tensor: Tensor, pooling_factor: usize) -> Result<Tensor> {
    pool_1d_windows(tensor, pooling_factor, "max_pool_1d", |window| {
        window.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    })
}

/// Shared windowed-pooling implementation for [`average_pool_1d`] and [`max_pool_1d`]: splits
/// the sequence axis into fixed, non-overlapping windows of `pooling_factor` positions each
/// (the last window truncated, not padded, when `seq_len` doesn't divide evenly) and reduces
/// every window -- independently per batch element and per hidden channel -- with
/// `reduce_window`.
fn pool_1d_windows(
    tensor: Tensor,
    pooling_factor: usize,
    op: &str,
    reduce_window: impl Fn(&[f32]) -> f32,
) -> Result<Tensor> {
    if pooling_factor == 0 {
        return Err(invalid_input(format!(
            "{op}: pooling_factor must be greater than 0"
        )));
    }
    let (batch, seq_len, hidden) = expect_batch_seq_hidden(&tensor, op)?;
    let data = tensor.data()?;

    let out_seq_len = seq_len.div_ceil(pooling_factor);
    let mut out = vec![0.0f32; batch * out_seq_len * hidden];
    let mut window = Vec::with_capacity(pooling_factor);

    for b in 0..batch {
        for os in 0..out_seq_len {
            let start = os * pooling_factor;
            let end = (start + pooling_factor).min(seq_len);
            for h in 0..hidden {
                window.clear();
                window.extend((start..end).map(|s| data[(b * seq_len + s) * hidden + h]));
                out[(b * out_seq_len + os) * hidden + h] = reduce_window(&window);
            }
        }
    }

    Tensor::from_vec(out, &[batch, out_seq_len, hidden])
}

/// Interpolate tensor for hierarchical attention.
pub fn interpolate_tensor(
    tensor: Tensor,
    target_length: usize,
    method: InterpolationMethod,
) -> Result<Tensor> {
    match method {
        InterpolationMethod::Linear => linear_interpolate(tensor, target_length),
        InterpolationMethod::Nearest => nearest_interpolate(tensor, target_length),
    }
}

/// Interpolation methods.
#[derive(Debug, Clone)]
pub enum InterpolationMethod {
    /// Linear interpolation (implemented; see `linear_interpolate`).
    Linear,
    /// Nearest neighbor interpolation (implemented; see `nearest_interpolate`).
    Nearest,
}

/// Continuous source-axis positions for resampling a length-`src_len` sequence to `target_len`
/// positions, "align corners" style: when `target_len > 1` and `src_len > 1`, the first and last
/// output positions map exactly to the first and last source positions
/// (`src_pos(i) = i * (src_len - 1) / (target_len - 1)`); a single output position, or any
/// resampling of a single-element source, maps to the source's midpoint. Shared by
/// [`linear_interpolate`] and [`nearest_interpolate`] so both agree on "where in the source
/// sequence output position `i` comes from" -- they differ only in how they turn that continuous
/// position into a value.
fn resample_positions_align_corners(src_len: usize, target_len: usize) -> Vec<f32> {
    if target_len == 0 {
        return Vec::new();
    }
    if target_len == 1 || src_len <= 1 {
        let mid = src_len.saturating_sub(1) as f32 / 2.0;
        return vec![mid; target_len];
    }
    let scale = (src_len - 1) as f32 / (target_len - 1) as f32;
    (0..target_len).map(|i| i as f32 * scale).collect()
}

/// Real linear interpolation resampling the sequence axis (dim 1) of a `[batch, seq_len,
/// hidden]` tensor to exactly `target_length` positions -- a **fixed target length**, unlike
/// [`average_pool_1d`]/[`max_pool_1d`]'s fixed-window contract (their `pooling_factor` sets a
/// window size, not a target length). Each output position linearly blends its two nearest
/// source positions (see [`resample_positions_align_corners`]); output positions that land
/// exactly on a source position -- including both sequence endpoints when `target_length > 1` --
/// reproduce that source value exactly.
fn linear_interpolate(tensor: Tensor, target_length: usize) -> Result<Tensor> {
    if target_length == 0 {
        return Err(invalid_input(
            "linear_interpolate: target_length must be greater than 0",
        ));
    }
    let (batch, seq_len, hidden) = expect_batch_seq_hidden(&tensor, "linear_interpolate")?;
    let data = tensor.data()?;
    let positions = resample_positions_align_corners(seq_len, target_length);

    let mut out = vec![0.0f32; batch * target_length * hidden];
    for b in 0..batch {
        for (ti, &pos) in positions.iter().enumerate() {
            let pos = pos.max(0.0);
            let s0 = (pos.floor() as usize).min(seq_len - 1);
            let s1 = (s0 + 1).min(seq_len - 1);
            let frac = (pos - s0 as f32).clamp(0.0, 1.0);
            for h in 0..hidden {
                let v0 = data[(b * seq_len + s0) * hidden + h];
                let v1 = data[(b * seq_len + s1) * hidden + h];
                out[(b * target_length + ti) * hidden + h] = v0 * (1.0 - frac) + v1 * frac;
            }
        }
    }

    Tensor::from_vec(out, &[batch, target_length, hidden])
}

/// Real nearest-neighbor interpolation resampling the sequence axis to exactly `target_length`
/// positions (fixed target length; see [`linear_interpolate`]'s doc for the pooling-vs-
/// interpolation contract distinction). Each output position takes the value of the closest
/// source position from [`resample_positions_align_corners`], rounding **half away from zero**
/// ([`f32::round`]'s convention) when a position lands exactly between two source indices.
fn nearest_interpolate(tensor: Tensor, target_length: usize) -> Result<Tensor> {
    if target_length == 0 {
        return Err(invalid_input(
            "nearest_interpolate: target_length must be greater than 0",
        ));
    }
    let (batch, seq_len, hidden) = expect_batch_seq_hidden(&tensor, "nearest_interpolate")?;
    let data = tensor.data()?;
    let positions = resample_positions_align_corners(seq_len, target_length);

    let mut out = vec![0.0f32; batch * target_length * hidden];
    for b in 0..batch {
        for (ti, &pos) in positions.iter().enumerate() {
            let s = (pos.round().max(0.0) as usize).min(seq_len - 1);
            for h in 0..hidden {
                out[(b * target_length + ti) * hidden + h] = data[(b * seq_len + s) * hidden + h];
            }
        }
    }

    Tensor::from_vec(out, &[batch, target_length, hidden])
}

#[cfg(test)]
mod attention_stats_tests {
    use super::*;

    /// A uniform distribution over `n` positions has closed-form entropy `ln(n)`.
    #[test]
    fn test_compute_entropy_uniform_distribution_equals_ln_n() {
        let n = 4;
        let uniform = vec![1.0f32 / n as f32; n];
        let tensor = Tensor::from_vec(uniform, &[1, n]).expect("tensor construction");

        let entropy = compute_entropy(&tensor).expect("compute_entropy");
        assert!(
            (entropy - (n as f32).ln()).abs() < 1e-5,
            "uniform-{n} entropy should be ln({n}) = {}, got {entropy}",
            (n as f32).ln()
        );
    }

    /// A one-hot distribution (all mass on a single position) has entropy exactly 0.
    #[test]
    fn test_compute_entropy_one_hot_is_zero() {
        let one_hot = vec![0.0f32, 0.0, 1.0, 0.0, 0.0];
        let tensor = Tensor::from_vec(one_hot, &[1, 5]).expect("tensor construction");

        let entropy = compute_entropy(&tensor).expect("compute_entropy");
        assert!(
            entropy.abs() < 1e-6,
            "one-hot entropy should be 0, got {entropy}"
        );
    }

    /// Entropy is averaged across every row when more than one attention distribution is
    /// present: one uniform-2 row (entropy ln(2)) and one one-hot row (entropy 0) average to
    /// ln(2) / 2.
    #[test]
    fn test_compute_entropy_averages_across_rows() {
        let data = vec![0.5f32, 0.5, 1.0, 0.0];
        let tensor = Tensor::from_vec(data, &[2, 2]).expect("tensor construction");

        let entropy = compute_entropy(&tensor).expect("compute_entropy");
        let expected = (2.0f32).ln() / 2.0;
        assert!(
            (entropy - expected).abs() < 1e-5,
            "expected average entropy {expected}, got {entropy}"
        );
    }

    #[test]
    fn test_compute_min_max_known_values() {
        let data = vec![0.3f32, 0.1, 0.9, 0.4, 0.05, 0.6];
        let tensor = Tensor::from_vec(data, &[2, 3]).expect("tensor construction");

        let (min, max) = compute_min_max(&tensor).expect("compute_min_max");
        assert!((min - 0.05).abs() < 1e-6, "expected min 0.05, got {min}");
        assert!((max - 0.9).abs() < 1e-6, "expected max 0.9, got {max}");
    }

    #[test]
    fn test_compute_min_max_constant_tensor() {
        let data = vec![0.25f32; 6];
        let tensor = Tensor::from_vec(data, &[2, 3]).expect("tensor construction");

        let (min, max) = compute_min_max(&tensor).expect("compute_min_max");
        assert!((min - 0.25).abs() < 1e-6);
        assert!((max - 0.25).abs() < 1e-6);
    }

    /// Fraction of weights at/below the threshold: 3 of 6 values here are <= 0.1.
    #[test]
    fn test_compute_sparsity_known_fraction() {
        let data = vec![0.05f32, 0.5, 0.1, 0.6, 0.02, 0.9];
        let tensor = Tensor::from_vec(data, &[2, 3]).expect("tensor construction");

        let sparsity = compute_sparsity(&tensor, 0.1).expect("compute_sparsity");
        assert!(
            (sparsity - 0.5).abs() < 1e-6,
            "expected sparsity 3/6 = 0.5, got {sparsity}"
        );
    }

    #[test]
    fn test_compute_sparsity_all_above_threshold_is_zero() {
        let data = vec![0.5f32, 0.6, 0.7, 0.8];
        let tensor = Tensor::from_vec(data, &[1, 4]).expect("tensor construction");

        let sparsity = compute_sparsity(&tensor, 0.1).expect("compute_sparsity");
        assert!(sparsity.abs() < 1e-6);
    }

    #[test]
    fn test_compute_sparsity_all_below_threshold_is_one() {
        let data = vec![0.0f32, 0.01, 0.02, 0.03];
        let tensor = Tensor::from_vec(data, &[1, 4]).expect("tensor construction");

        let sparsity = compute_sparsity(&tensor, 0.1).expect("compute_sparsity");
        assert!((sparsity - 1.0).abs() < 1e-6);
    }

    /// Single-row top-k: the 3 largest weights are at positions 4 (0.9), 1 (0.7), 3 (0.5),
    /// strictly in that descending order.
    #[test]
    fn test_compute_top_positions_known_top_k() {
        let data = vec![0.1f32, 0.7, 0.05, 0.5, 0.9, 0.2];
        let tensor = Tensor::from_vec(data, &[1, 6]).expect("tensor construction");

        let top = compute_top_positions(&tensor, 3).expect("compute_top_positions");
        assert_eq!(top, vec![4, 1, 3]);
    }

    /// k larger than the available positions is clamped to the dimension size rather than
    /// erroring or padding.
    #[test]
    fn test_compute_top_positions_k_larger_than_dimension_is_clamped() {
        let data = vec![0.2f32, 0.5, 0.3];
        let tensor = Tensor::from_vec(data, &[1, 3]).expect("tensor construction");

        let top = compute_top_positions(&tensor, 10).expect("compute_top_positions");
        assert_eq!(top, vec![1, 2, 0]);
    }

    /// With two rows, top positions are ranked by the average weight across rows: position 0
    /// averages (0.9+0.1)/2=0.5, position 1 averages (0.1+0.9)/2=0.5 (tie -> lower index
    /// first), position 2 averages (0.0+0.0)/2=0.0.
    #[test]
    fn test_compute_top_positions_averages_across_rows() {
        let data = vec![0.9f32, 0.1, 0.0, 0.1, 0.9, 0.0];
        let tensor = Tensor::from_vec(data, &[2, 3]).expect("tensor construction");

        let top = compute_top_positions(&tensor, 2).expect("compute_top_positions");
        assert_eq!(top, vec![0, 1]);
    }

    /// End-to-end: compute_attention_stats on a shape [batch=1, heads=2, seq_q=1, seq_k=4]
    /// tensor where head 0 is uniform (entropy ln(4)) and head 1 is one-hot (entropy 0)
    /// produces per-head stats matching the closed-form values, and overall stats computed
    /// over the whole flattened tensor.
    #[test]
    fn test_compute_attention_stats_end_to_end_per_head() {
        // [batch=1, heads=2, seq_q=1, seq_k=4]
        let data = vec![
            0.25f32, 0.25, 0.25, 0.25, // head 0: uniform
            0.0, 0.0, 1.0, 0.0, // head 1: one-hot at position 2
        ];
        let tensor = Tensor::from_vec(data, &[1, 2, 1, 4]).expect("tensor construction");

        let stats = compute_attention_stats(&tensor, 2).expect("compute_attention_stats");
        assert_eq!(stats.head_stats.len(), 2);

        let head0 = &stats.head_stats[0];
        assert_eq!(head0.head_idx, 0);
        assert!(
            (head0.entropy - (4.0f32).ln()).abs() < 1e-5,
            "head 0 (uniform-4) entropy should be ln(4), got {}",
            head0.entropy
        );

        let head1 = &stats.head_stats[1];
        assert_eq!(head1.head_idx, 1);
        assert!(
            head1.entropy.abs() < 1e-6,
            "head 1 (one-hot) entropy should be 0, got {}",
            head1.entropy
        );
        assert_eq!(
            head1.top_positions[0], 2,
            "head 1's top position should be index 2"
        );

        // Overall min/max is taken over the whole flattened tensor: min 0.0, max 1.0.
        assert!((stats.min_weight - 0.0).abs() < 1e-6);
        assert!((stats.max_weight - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_compute_entropy_rejects_empty_tensor() {
        let tensor = Tensor::from_vec(Vec::new(), &[0]).expect("tensor construction");
        assert!(compute_entropy(&tensor).is_err());
    }

    /// Regression test: `compute_attention_stats` used to index `shape[0]`/`shape[2]`
    /// unconditionally and would panic on any tensor of rank < 3 (e.g. a caller passing a
    /// single already-selected 2-D [seq_q, seq_k] slice with num_heads > 0). It must now
    /// return a structured error instead of panicking.
    #[test]
    fn test_compute_attention_stats_rejects_low_rank_tensor_instead_of_panicking() {
        let data = vec![0.5f32, 0.5];
        let tensor = Tensor::from_vec(data, &[2]).expect("tensor construction");

        let result = compute_attention_stats(&tensor, 2);
        assert!(
            result.is_err(),
            "rank-1 tensor with num_heads=2 must error, not panic"
        );
    }

    /// num_heads=0 does not need to select along a head dimension, so a 1-D tensor is valid
    /// input for the overall-only statistics path.
    #[test]
    fn test_compute_attention_stats_zero_heads_accepts_1d_tensor() {
        let data = vec![0.25f32, 0.25, 0.25, 0.25];
        let tensor = Tensor::from_vec(data, &[4]).expect("tensor construction");

        let stats = compute_attention_stats(&tensor, 0).expect("compute_attention_stats");
        assert!(stats.head_stats.is_empty());
        assert!((stats.entropy - (4.0f32).ln()).abs() < 1e-5);
    }
}

#[cfg(test)]
mod pooling_and_interpolation_tests {
    use super::*;

    /// Builds a `[batch=1, seq_len=values.len(), hidden=1]` tensor so `seq_data` below can read
    /// the pooled/interpolated sequence straight back out in order.
    fn seq_tensor(values: &[f32]) -> Tensor {
        Tensor::from_vec(values.to_vec(), &[1, values.len(), 1]).expect("tensor construction")
    }

    fn seq_data(tensor: &Tensor) -> Vec<f32> {
        tensor.data().expect("tensor data")
    }

    fn assert_close(actual: &[f32], expected: &[f32], tol: f32) {
        assert_eq!(
            actual.len(),
            expected.len(),
            "length mismatch: actual {actual:?}, expected {expected:?}"
        );
        for (i, (&a, &e)) in actual.iter().zip(expected).enumerate() {
            assert!(
                (a - e).abs() < tol,
                "index {i}: expected {e}, got {a} (full actual={actual:?}, expected={expected:?})"
            );
        }
    }

    // -- average_pool_1d (via pool_tensor / PoolingMethod::Average) --

    /// Even ratio: seq_len=4, pooling_factor=2 -> two full, non-overlapping windows.
    #[test]
    fn test_average_pool_even_ratio() {
        let input = seq_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let out = pool_tensor(input, 2, PoolingMethod::Average).expect("pooling must succeed");
        assert_eq!(out.shape(), vec![1, 2, 1]);
        assert_close(&seq_data(&out), &[1.5, 3.5], 1e-6);
    }

    /// Odd ratio: seq_len=5, pooling_factor=2 -> output length ceil(5/2)=3, and the last window
    /// has only one member (averaged over just that member, not zero-padded).
    #[test]
    fn test_average_pool_odd_ratio_has_partial_last_window() {
        let input = seq_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = pool_tensor(input, 2, PoolingMethod::Average).expect("pooling must succeed");
        assert_eq!(out.shape(), vec![1, 3, 1]);
        assert_close(&seq_data(&out), &[1.5, 3.5, 5.0], 1e-6);
    }

    /// Length-1 edge case: a single sequence position pools to itself, unchanged.
    #[test]
    fn test_average_pool_length_one_is_unchanged() {
        let input = seq_tensor(&[7.0]);
        let out = pool_tensor(input, 3, PoolingMethod::Average).expect("pooling must succeed");
        assert_eq!(out.shape(), vec![1, 1, 1]);
        assert_close(&seq_data(&out), &[7.0], 1e-6);
    }

    /// Pooling must reduce only the sequence axis: each (batch, hidden-channel) pair is pooled
    /// independently, so batch 1's values must not leak into batch 0's windows or vice versa,
    /// and hidden channel 0/1 must not mix.
    #[test]
    fn test_average_pool_independent_per_batch_and_channel() {
        // shape [batch=2, seq=4, hidden=2]
        let data = vec![
            1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0, // batch 0
            100.0, 1.0, 200.0, 2.0, 300.0, 3.0, 400.0, 4.0, // batch 1
        ];
        let input = Tensor::from_vec(data, &[2, 4, 2]).expect("tensor construction");
        let out = pool_tensor(input, 2, PoolingMethod::Average).expect("pooling must succeed");
        assert_eq!(out.shape(), vec![2, 2, 2]);
        assert_close(
            &out.data().expect("data"),
            &[1.5, 15.0, 3.5, 35.0, 150.0, 1.5, 350.0, 3.5],
            1e-6,
        );
    }

    // -- max_pool_1d --

    #[test]
    fn test_max_pool_even_ratio() {
        let input = seq_tensor(&[1.0, 5.0, 3.0, 7.0]);
        let out = pool_tensor(input, 2, PoolingMethod::Max).expect("pooling must succeed");
        assert_close(&seq_data(&out), &[5.0, 7.0], 1e-6);
    }

    #[test]
    fn test_max_pool_odd_ratio_has_partial_last_window() {
        let input = seq_tensor(&[1.0, 5.0, 3.0, 7.0, 2.0]);
        let out = pool_tensor(input, 2, PoolingMethod::Max).expect("pooling must succeed");
        assert_close(&seq_data(&out), &[5.0, 7.0, 2.0], 1e-6);
    }

    #[test]
    fn test_max_pool_length_one_is_unchanged() {
        let input = seq_tensor(&[9.0]);
        let out = pool_tensor(input, 5, PoolingMethod::Max).expect("pooling must succeed");
        assert_close(&seq_data(&out), &[9.0], 1e-6);
    }

    // -- PoolingMethod::Learnable: structured refusal, never a silent Average alias --

    #[test]
    fn test_pool_learnable_returns_structured_error_not_average_alias() {
        let input = seq_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let result = pool_tensor(input, 2, PoolingMethod::Learnable);
        assert!(
            result.is_err(),
            "Learnable must not silently succeed by aliasing Average"
        );
        let message = result.expect_err("checked above").to_string().to_lowercase();
        assert!(
            message.contains("unsupported"),
            "expected an unsupported-operation refusal, got: {message}"
        );
    }

    #[test]
    fn test_pooling_factor_zero_is_rejected() {
        let input = seq_tensor(&[1.0, 2.0, 3.0]);
        assert!(pool_tensor(input, 0, PoolingMethod::Average).is_err());
    }

    // -- linear_interpolate --

    /// Upsampling an already-linear sequence: every output position (on- or off-grid) matches
    /// the linear ramp exactly.
    #[test]
    fn test_linear_interpolate_upsample_matches_hand_computed_values() {
        let input = seq_tensor(&[0.0, 1.0, 2.0]);
        let out = interpolate_tensor(input, 5, InterpolationMethod::Linear)
            .expect("interpolation must succeed");
        assert_eq!(out.shape(), vec![1, 5, 1]);
        // positions = [0, 0.5, 1, 1.5, 2] -> values = [0, 0.5, 1, 1.5, 2]
        assert_close(&seq_data(&out), &[0.0, 0.5, 1.0, 1.5, 2.0], 1e-5);
    }

    /// Downsampling with output positions that land strictly between source grid points.
    #[test]
    fn test_linear_interpolate_downsample_off_grid_matches_hand_computed_values() {
        let input = seq_tensor(&[0.0, 10.0, 20.0, 30.0, 40.0]);
        let out = interpolate_tensor(input, 4, InterpolationMethod::Linear)
            .expect("interpolation must succeed");
        // positions = [0, 4/3, 8/3, 4] -> values = [0, 40/3, 80/3, 40]
        assert_close(&seq_data(&out), &[0.0, 40.0 / 3.0, 80.0 / 3.0, 40.0], 1e-3);
    }

    /// A single-element source has nothing to interpolate between: every output position must
    /// broadcast that one value.
    #[test]
    fn test_linear_interpolate_single_element_source_broadcasts() {
        let input = seq_tensor(&[42.0]);
        let out = interpolate_tensor(input, 4, InterpolationMethod::Linear)
            .expect("interpolation must succeed");
        assert_close(&seq_data(&out), &[42.0, 42.0, 42.0, 42.0], 1e-6);
    }

    /// target_length == 1 resamples to the source's midpoint.
    #[test]
    fn test_linear_interpolate_target_length_one_is_the_midpoint() {
        let input = seq_tensor(&[1.0, 2.0, 3.0, 4.0]);
        let out = interpolate_tensor(input, 1, InterpolationMethod::Linear)
            .expect("interpolation must succeed");
        // midpoint position = (4-1)/2 = 1.5 -> blend of data[1]=2 and data[2]=3 -> 2.5
        assert_close(&seq_data(&out), &[2.5], 1e-6);
    }

    // -- nearest_interpolate --

    /// Upsampling; also locks in the round-half-away-from-zero tie-break at the two positions
    /// that land exactly halfway between source indices.
    #[test]
    fn test_nearest_interpolate_upsample_matches_hand_computed_values() {
        let input = seq_tensor(&[10.0, 20.0, 30.0]);
        let out = interpolate_tensor(input, 5, InterpolationMethod::Nearest)
            .expect("interpolation must succeed");
        // positions [0, 0.5, 1, 1.5, 2] round (half-away-from-zero) to indices [0,1,1,2,2].
        assert_close(&seq_data(&out), &[10.0, 20.0, 20.0, 30.0, 30.0], 1e-6);
    }

    #[test]
    fn test_nearest_interpolate_downsample_matches_hand_computed_values() {
        let input = seq_tensor(&[1.0, 2.0, 3.0, 4.0, 5.0]);
        let out = interpolate_tensor(input, 4, InterpolationMethod::Nearest)
            .expect("interpolation must succeed");
        // positions [0, 4/3, 8/3, 4] round to indices [0, 1, 3, 4].
        assert_close(&seq_data(&out), &[1.0, 2.0, 4.0, 5.0], 1e-6);
    }

    #[test]
    fn test_nearest_interpolate_length_one_source_broadcasts() {
        let input = seq_tensor(&[7.0]);
        let out = interpolate_tensor(input, 3, InterpolationMethod::Nearest)
            .expect("interpolation must succeed");
        assert_close(&seq_data(&out), &[7.0, 7.0, 7.0], 1e-6);
    }

    // -- shape validation --

    #[test]
    fn test_pool_tensor_rejects_non_rank_3_input() {
        let input = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]).expect("tensor construction");
        assert!(pool_tensor(input, 2, PoolingMethod::Average).is_err());
    }

    #[test]
    fn test_interpolate_tensor_rejects_zero_target_length() {
        let input = seq_tensor(&[1.0, 2.0, 3.0]);
        assert!(interpolate_tensor(input, 0, InterpolationMethod::Linear).is_err());
    }
}

#[cfg(test)]
mod sparse_mask_and_mask_pooling_tests {
    use super::*;

    fn count_unmasked(data: &[f32]) -> usize {
        data.iter().filter(|&&v| v == 0.0).count()
    }

    /// The default `SparsePattern::Random` config (`connections: None`) keeps
    /// each cell independently with probability `1 - sparsity_ratio`. Over a
    /// large grid the empirical keep fraction must land near that
    /// probability -- a loose bound (this is one seeded, deterministic draw,
    /// not a statistical claim about the RNG), but nowhere near what a
    /// broken implementation (e.g. all-kept, all-masked, or a fixed stripe
    /// with a very different ratio) could produce.
    #[test]
    fn random_mask_without_connections_keeps_close_to_the_target_ratio() {
        let (query_len, key_len) = (100, 100);
        let sparsity_ratio = 0.3;
        let mask = create_sparse_mask(
            query_len,
            key_len,
            sparsity_ratio,
            SparsePattern::Random {
                connections: None,
                seed: 7,
            },
        )
        .expect("mask construction");
        let data = mask.data().expect("tensor data");
        let kept = count_unmasked(&data);
        let expected = (query_len * key_len) as f32 * (1.0 - sparsity_ratio);
        assert!(
            (kept as f32 - expected).abs() < expected * 0.15,
            "kept {kept} of {} cells, expected close to {expected}",
            query_len * key_len
        );
    }

    /// Regression: the mask used to be `(i + j) % 10 < (keep_ratio * 10.0)
    /// as usize` -- a fixed diagonal stripe, identical on every call
    /// regardless of seed. This compares directly against that exact old
    /// formula (same `keep_ratio`, so density alone can't distinguish them)
    /// and requires the new, real implementation to disagree with it on at
    /// least one cell.
    #[test]
    fn random_mask_is_not_the_old_fixed_diagonal_stripe() {
        let (query_len, key_len) = (20, 20);
        let sparsity_ratio = 0.3;
        let mask = create_sparse_mask(
            query_len,
            key_len,
            sparsity_ratio,
            SparsePattern::Random {
                connections: None,
                seed: 42,
            },
        )
        .expect("mask construction");
        let data = mask.data().expect("tensor data");

        let keep_ratio = 1.0 - sparsity_ratio;
        let mut differs = false;
        for i in 0..query_len {
            for j in 0..key_len {
                let old_stripe_kept = (i + j) % 10 < (keep_ratio * 10.0) as usize;
                let new_kept = data[i * key_len + j] == 0.0;
                if old_stripe_kept != new_kept {
                    differs = true;
                }
            }
        }
        assert!(
            differs,
            "the new mask must not reproduce the old fixed stripe pattern"
        );
    }

    /// The same seed (with the same shape/ratio/connections) must always
    /// reproduce the exact same mask.
    #[test]
    fn random_mask_is_deterministic_for_a_fixed_seed() {
        let pattern = || SparsePattern::Random {
            connections: None,
            seed: 123,
        };
        let first = create_sparse_mask(16, 16, 0.4, pattern()).expect("first mask");
        let second = create_sparse_mask(16, 16, 0.4, pattern()).expect("second mask");
        assert_eq!(
            first.data().expect("data"),
            second.data().expect("data"),
            "the same seed must reproduce the exact same mask"
        );
    }

    /// A different seed must (verified empirically for these two seeds)
    /// produce a different mask -- otherwise `seed` would be decorative.
    #[test]
    fn random_mask_differs_for_a_different_seed() {
        let first = create_sparse_mask(
            16,
            16,
            0.4,
            SparsePattern::Random {
                connections: None,
                seed: 1,
            },
        )
        .expect("first mask");
        let second = create_sparse_mask(
            16,
            16,
            0.4,
            SparsePattern::Random {
                connections: None,
                seed: 2,
            },
        )
        .expect("second mask");
        assert_ne!(
            first.data().expect("data"),
            second.data().expect("data"),
            "different seeds must (for these two, verified) produce different masks"
        );
    }

    /// `connections = Some(k)` must keep EXACTLY `k` keys per query row, not
    /// an approximate fraction -- and must actually read `random_connections`
    /// at all, which the old `(i + j) % 10` formula never did.
    #[test]
    fn random_mask_with_connections_keeps_exactly_k_per_row() {
        let (query_len, key_len, k) = (10, 50, 7);
        let mask = create_sparse_mask(
            query_len,
            key_len,
            0.9, // sparsity_ratio is ignored when `connections` is Some
            SparsePattern::Random {
                connections: Some(k),
                seed: 99,
            },
        )
        .expect("mask construction");
        let data = mask.data().expect("tensor data");

        for i in 0..query_len {
            let row = &data[i * key_len..(i + 1) * key_len];
            let kept = count_unmasked(row);
            assert_eq!(kept, k, "row {i} must keep exactly {k} keys, got {kept}");
        }
    }

    /// `connections` larger than `key_len` must clamp to `key_len` (keep
    /// everything) rather than panicking or under/over-shooting.
    #[test]
    fn random_mask_with_connections_larger_than_key_len_keeps_everything() {
        let (query_len, key_len) = (4, 5);
        let mask = create_sparse_mask(
            query_len,
            key_len,
            0.9,
            SparsePattern::Random {
                connections: Some(100),
                seed: 5,
            },
        )
        .expect("mask construction");
        let data = mask.data().expect("tensor data");
        assert_eq!(count_unmasked(&data), query_len * key_len);
    }

    // -- pool_mask_key_axis --

    /// Hand-computed: seq_len=5, pooling_factor=2 -> windows [0,1] [2,3] [4].
    /// `NEG_INFINITY` is absorbing under averaging with any finite addends
    /// (only `0.0` and `NEG_INFINITY` ever appear in an additive mask), so a
    /// window pools to masked iff ANY position in it was masked.
    #[test]
    fn pool_mask_key_axis_2d_matches_hand_computed_values() {
        let neg_inf = f32::NEG_INFINITY;
        #[rustfmt::skip]
        let mask = Tensor::from_vec(
            vec![
                0.0,     0.0,     neg_inf, 0.0,     neg_inf, // row 0
                neg_inf, neg_inf, neg_inf, 0.0,     0.0,     // row 1
            ],
            &[2, 5],
        )
        .expect("tensor construction");

        let pooled = pool_mask_key_axis(mask, 2).expect("pooling must succeed");
        assert_eq!(pooled.shape(), vec![2, 3]);
        let data = pooled.data().expect("tensor data");
        // row 0: [0,0]->0.0, [-inf,0]->-inf, [-inf]->-inf
        assert_eq!(data[0], 0.0);
        assert!(data[1].is_infinite() && data[1] < 0.0);
        assert!(data[2].is_infinite() && data[2] < 0.0);
        // row 1: [-inf,-inf]->-inf, [-inf,0]->-inf, [0]->0.0
        assert!(data[3].is_infinite() && data[3] < 0.0);
        assert!(data[4].is_infinite() && data[4] < 0.0);
        assert_eq!(data[5], 0.0);
    }

    #[test]
    fn pool_mask_key_axis_3d_pools_each_batch_and_row_independently() {
        let neg_inf = f32::NEG_INFINITY;
        // shape [batch=2, query_len=1, key_len=4]; batch 0 all-visible,
        // batch 1 all-masked -- pooling must not mix them.
        let mask = Tensor::from_vec(
            vec![0.0, 0.0, 0.0, 0.0, neg_inf, neg_inf, neg_inf, neg_inf],
            &[2, 1, 4],
        )
        .expect("tensor construction");

        let pooled = pool_mask_key_axis(mask, 2).expect("pooling must succeed");
        assert_eq!(pooled.shape(), vec![2, 1, 2]);
        let data = pooled.data().expect("tensor data");
        assert_eq!(&data[0..2], &[0.0, 0.0]);
        assert!(data[2].is_infinite() && data[2] < 0.0);
        assert!(data[3].is_infinite() && data[3] < 0.0);
    }

    #[test]
    fn pool_mask_key_axis_rejects_other_ranks() {
        let mask =
            Tensor::from_vec(vec![0.0, 0.0, 0.0, 0.0], &[2, 2, 1, 1]).expect("tensor construction");
        assert!(pool_mask_key_axis(mask, 2).is_err());
    }
}
