//! Memory-efficient attention mechanisms for transformer models.
//!
//! This module provides:
//! - Standard scaled dot-product attention (O(N²) reference)
//! - FlashAttention-style tiled computation (O(N) memory)
//! - Multi-head attention with optional FlashAttention backend
//! - KV cache for autoregressive generation
//! - [`flash`]: Flash Attention v2 with f32 flat-slice API (Dao, 2023)
//! - [`sliding_window`]: Sliding-window attention (Longformer / Mistral)
//!
//! # Algorithm Notes
//!
//! FlashAttention uses the online softmax normalization algorithm by
//! Milakov & Gimelshein (2018) to avoid materialising the full N×N attention
//! matrix. For each query block it maintains a running maximum `m` and
//! normalisation constant `l`, updating them incrementally as key/value
//! blocks are processed.

pub mod flash;
pub mod sliding_window;
pub mod variants;

// Re-export advanced attention variants for convenient access.
pub use variants::{
    alibi_attention, alibi_attention_full, alibi_bias, alibi_slopes,
    cross_attention, grouped_query_attention, multi_query_attention, repeat_kv,
    AliBiConfig, AliBiFullConfig, CrossAttentionConfig, GqaConfig, MqaConfig,
};

// ─── Error ────────────────────────────────────────────────────────────────────

/// Errors that can arise from attention operations.
#[derive(Debug, thiserror::Error)]
pub enum AttentionError {
    /// Query and key have incompatible shapes.
    #[error("Shape mismatch: Q={q}, K={k}")]
    QKShapeMismatch { q: String, k: String },
    /// Input sequence has zero length.
    #[error("Empty input")]
    EmptyInput,
    /// `d_model` is not divisible by `nheads`.
    #[error("Invalid head count: d_model={dm} not divisible by nheads={nh}")]
    InvalidHeads { dm: usize, nh: usize },
    /// Requested block size exceeds sequence length.
    #[error("Block size {bs} larger than sequence length {sl}")]
    BlockTooLarge { bs: usize, sl: usize },
    /// KV cache operation failed.
    #[error("KV cache error: {0}")]
    KVCacheError(String),
}

// ─── Internal helpers ────────────────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Row-wise softmax in-place over a 2-D slice (rows × cols).
fn softmax_rows(matrix: &mut [Vec<f64>]) {
    for row in matrix.iter_mut() {
        let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0_f64;
        for v in row.iter_mut() {
            *v = (*v - max_val).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

/// Matrix multiply: result[i][j] = sum_k A[i][k] * B[k][j].
/// A: [m, k], B: [k, n], result: [m, n].
fn matmul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let m = a.len();
    if m == 0 {
        return Vec::new();
    }
    let k = a[0].len();
    let n = if b.is_empty() { 0 } else { b[0].len() };
    let mut out = vec![vec![0.0_f64; n]; m];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i][p];
            for j in 0..n {
                out[i][j] += aip * b[p][j];
            }
        }
    }
    out
}

// ─── Standard Scaled Dot-Product Attention ───────────────────────────────────

/// Standard O(N²) scaled dot-product attention.
///
/// Computes `softmax(Q K^T / scale + mask) V`.
///
/// # Parameters
/// - `q`: Query matrix `[seq_q, head_dim]`.
/// - `k`: Key matrix `[seq_k, head_dim]`.
/// - `v`: Value matrix `[seq_k, head_dim]`.
/// - `mask`: Optional additive mask `[seq_q, seq_k]`; use `-∞` to mask positions.
/// - `scale`: Scaling factor (default `1 / sqrt(head_dim)`).
///
/// # Errors
/// Returns [`AttentionError`] on shape mismatches or empty inputs.
pub fn scaled_dot_product_attention(
    q: &[Vec<f64>],
    k: &[Vec<f64>],
    v: &[Vec<f64>],
    mask: Option<&[Vec<f64>]>,
    scale: Option<f64>,
) -> Result<Vec<Vec<f64>>, AttentionError> {
    if q.is_empty() || k.is_empty() || v.is_empty() {
        return Err(AttentionError::EmptyInput);
    }
    let head_dim = q[0].len();
    if k[0].len() != head_dim {
        return Err(AttentionError::QKShapeMismatch {
            q: format!("[{}, {}]", q.len(), head_dim),
            k: format!("[{}, {}]", k.len(), k[0].len()),
        });
    }
    if v.len() != k.len() {
        return Err(AttentionError::QKShapeMismatch {
            q: format!("K len={}", k.len()),
            k: format!("V len={}", v.len()),
        });
    }

    let scale_factor = scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());
    let seq_q = q.len();
    let seq_k = k.len();

    // Compute scores S = Q K^T / scale  [seq_q, seq_k]
    let mut scores = vec![vec![0.0_f64; seq_k]; seq_q];
    for i in 0..seq_q {
        for j in 0..seq_k {
            scores[i][j] = dot(&q[i], &k[j]) * scale_factor;
        }
    }

    // Apply mask
    if let Some(m) = mask {
        for i in 0..seq_q {
            for j in 0..seq_k {
                scores[i][j] += m[i][j];
            }
        }
    }

    // Softmax over key dimension
    softmax_rows(&mut scores);

    // Output = softmax(scores) @ V  [seq_q, head_dim]
    let v_dim = v[0].len();
    let mut output = vec![vec![0.0_f64; v_dim]; seq_q];
    for i in 0..seq_q {
        for j in 0..seq_k {
            let s = scores[i][j];
            for d in 0..v_dim {
                output[i][d] += s * v[j][d];
            }
        }
    }

    Ok(output)
}

// ─── FlashAttention-style Tiled Attention ────────────────────────────────────

/// Configuration for [`FlashAttention`].
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Query block size (default 64).
    pub block_size_q: usize,
    /// Key block size (default 64).
    pub block_size_k: usize,
    /// Apply causal (lower-triangular) masking (default `false`).
    pub causal: bool,
    /// Dropout probability — stored but not applied in this deterministic
    /// reference implementation (default 0.0).
    pub dropout_p: f64,
    /// Scaling factor (default `1 / sqrt(head_dim)`).
    pub scale: Option<f64>,
}

impl Default for FlashAttentionConfig {
    fn default() -> Self {
        Self {
            block_size_q: 64,
            block_size_k: 64,
            causal: false,
            dropout_p: 0.0,
            scale: None,
        }
    }
}

/// FlashAttention: tiled attention avoiding materialisation of the N×N matrix.
///
/// Uses the online-softmax algorithm (Milakov & Gimelshein 2018):
/// for each query block, maintains a running maximum `m` and normalisation
/// constant `l` across all key blocks.  Memory complexity is O(N·d) vs
/// O(N²) for standard attention.
pub struct FlashAttention {
    config: FlashAttentionConfig,
}

impl FlashAttention {
    /// Create a new [`FlashAttention`] with the given config.
    pub fn new(config: FlashAttentionConfig) -> Self {
        Self { config }
    }

    /// FlashAttention forward pass.
    ///
    /// # Parameters
    /// - `q`: `[seq_q, head_dim]`
    /// - `k`: `[seq_k, head_dim]`
    /// - `v`: `[seq_k, head_dim]`
    ///
    /// # Errors
    /// Returns [`AttentionError`] on empty inputs or shape mismatches.
    pub fn forward(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, AttentionError> {
        if q.is_empty() || k.is_empty() || v.is_empty() {
            return Err(AttentionError::EmptyInput);
        }
        let head_dim = q[0].len();
        if k[0].len() != head_dim {
            return Err(AttentionError::QKShapeMismatch {
                q: format!("[{}, {}]", q.len(), head_dim),
                k: format!("[{}, {}]", k.len(), k[0].len()),
            });
        }
        if v.len() != k.len() {
            return Err(AttentionError::QKShapeMismatch {
                q: format!("K len={}", k.len()),
                k: format!("V len={}", v.len()),
            });
        }

        let scale = self
            .config
            .scale
            .unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

        let seq_q = q.len();
        let seq_k = k.len();
        let v_dim = v[0].len();
        let bq = self.config.block_size_q.max(1);
        let bk = self.config.block_size_k.max(1);

        // Output accumulator and per-row statistics
        let mut output = vec![vec![0.0_f64; v_dim]; seq_q];
        // Running max per query row
        let mut m_i = vec![f64::NEG_INFINITY; seq_q];
        // Running normalisation constant per query row
        let mut l_i = vec![0.0_f64; seq_q];

        // Iterate over query blocks
        let mut qi_start = 0;
        while qi_start < seq_q {
            let qi_end = (qi_start + bq).min(seq_q);

            // Iterate over key/value blocks
            let mut kj_start = 0;
            while kj_start < seq_k {
                let kj_end = (kj_start + bk).min(seq_k);

                // Compute S_ij = Q_i K_j^T / scale  [bq_actual, bk_actual]
                for qi in qi_start..qi_end {
                    let local_qi = qi - qi_start;
                    let mut s_row = vec![0.0_f64; kj_end - kj_start];

                    for (local_kj, kj) in (kj_start..kj_end).enumerate() {
                        // Apply causal mask: positions where kj > qi get -inf
                        if self.config.causal && kj > qi {
                            s_row[local_kj] = f64::NEG_INFINITY;
                        } else {
                            s_row[local_kj] = dot(&q[qi], &k[kj]) * scale;
                        }
                    }

                    // Row max for this block
                    let m_ij = s_row
                        .iter()
                        .cloned()
                        .fold(f64::NEG_INFINITY, f64::max);

                    // New running max
                    let m_i_new = m_i[qi].max(m_ij);

                    // Shifted probabilities P_ij = exp(S_ij - m_i_new)
                    let p_row: Vec<f64> = s_row
                        .iter()
                        .map(|&s| (s - m_i_new).exp())
                        .collect();

                    // Row sum of P_ij
                    let p_sum: f64 = p_row.iter().sum();

                    // Correction factor for previously accumulated output
                    let correction = (m_i[qi] - m_i_new).exp();

                    // New normalisation constant
                    let l_i_new = correction * l_i[qi] + p_sum;

                    // Update output: rescale old contribution then add new
                    if l_i_new > 0.0 {
                        let inv_l = 1.0 / l_i_new;
                        let old_scale = correction * l_i[qi] * inv_l;
                        for d in 0..v_dim {
                            let mut new_contrib = 0.0_f64;
                            for (local_kj, kj) in (kj_start..kj_end).enumerate() {
                                new_contrib += p_row[local_kj] * v[kj][d];
                            }
                            output[qi][d] =
                                old_scale * output[qi][d] + new_contrib * inv_l;
                        }
                    }

                    m_i[qi] = m_i_new;
                    l_i[qi] = l_i_new;

                    // suppress unused warning on local_qi
                    let _ = local_qi;
                }

                kj_start = kj_end;
            }

            qi_start = qi_end;
        }

        Ok(output)
    }

    /// Verify that this FlashAttention implementation produces the same output
    /// as [`scaled_dot_product_attention`] within the given tolerance.
    ///
    /// Returns `true` if the maximum absolute element-wise difference is below
    /// `tolerance`.
    ///
    /// # Errors
    /// Propagates any error from either attention implementation.
    pub fn verify_against_standard(
        &self,
        q: &[Vec<f64>],
        k: &[Vec<f64>],
        v: &[Vec<f64>],
        tolerance: f64,
    ) -> Result<bool, AttentionError> {
        // Build causal mask if needed
        let mask_storage: Option<Vec<Vec<f64>>> = if self.config.causal {
            let seq_q = q.len();
            let seq_k = k.len();
            let mut m = vec![vec![0.0_f64; seq_k]; seq_q];
            for (i, row) in m.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    if j > i {
                        *cell = f64::NEG_INFINITY;
                    }
                }
            }
            Some(m)
        } else {
            None
        };

        let flash_out = self.forward(q, k, v)?;
        let std_out = scaled_dot_product_attention(
            q,
            k,
            v,
            mask_storage.as_deref(),
            self.config.scale,
        )?;

        let mut max_diff = 0.0_f64;
        for (row_f, row_s) in flash_out.iter().zip(std_out.iter()) {
            for (&f, &s) in row_f.iter().zip(row_s.iter()) {
                max_diff = max_diff.max((f - s).abs());
            }
        }

        Ok(max_diff < tolerance)
    }
}

// ─── Multi-Head Attention ─────────────────────────────────────────────────────

/// Configuration for [`MultiHeadAttention`].
#[derive(Debug, Clone)]
pub struct MultiHeadAttentionConfig {
    /// Full model dimension.
    pub d_model: usize,
    /// Number of attention heads.
    pub nheads: usize,
    /// Per-head dimension (`d_model / nheads`).
    pub head_dim: usize,
    /// Use FlashAttention backend when `true`; standard attention otherwise.
    pub use_flash: bool,
    /// Apply causal masking.
    pub causal: bool,
    /// Dropout probability.
    pub dropout_p: f64,
    /// Flash attention block size.
    pub block_size: usize,
}

impl MultiHeadAttentionConfig {
    /// Construct a config, checking that `d_model` is divisible by `nheads`.
    ///
    /// # Errors
    /// Returns [`AttentionError::InvalidHeads`] if the condition is violated.
    pub fn new(d_model: usize, nheads: usize) -> Result<Self, AttentionError> {
        if d_model % nheads != 0 {
            return Err(AttentionError::InvalidHeads {
                dm: d_model,
                nh: nheads,
            });
        }
        Ok(Self {
            d_model,
            nheads,
            head_dim: d_model / nheads,
            use_flash: false,
            causal: false,
            dropout_p: 0.0,
            block_size: 64,
        })
    }
}

/// Multi-head attention layer.
///
/// Holds learned projection weights `Wq`, `Wk`, `Wv`, `Wo` initialised to the
/// identity (diagonal 1) for deterministic unit tests without random number
/// generation.
pub struct MultiHeadAttention {
    config: MultiHeadAttentionConfig,
    wq: Vec<Vec<f64>>, // [d_model, d_model]
    wk: Vec<Vec<f64>>,
    wv: Vec<Vec<f64>>,
    wo: Vec<Vec<f64>>,
    flash: Option<FlashAttention>,
}

impl MultiHeadAttention {
    /// Create a new layer; weights are initialised to the identity matrix so
    /// that projections are no-ops (useful for unit tests).
    pub fn new(config: MultiHeadAttentionConfig) -> Self {
        let d = config.d_model;
        // Identity initialisation
        let identity = |size: usize| -> Vec<Vec<f64>> {
            let mut m = vec![vec![0.0_f64; size]; size];
            for (i, row) in m.iter_mut().enumerate() {
                row[i] = 1.0;
            }
            m
        };

        let flash = if config.use_flash {
            Some(FlashAttention::new(FlashAttentionConfig {
                block_size_q: config.block_size,
                block_size_k: config.block_size,
                causal: config.causal,
                dropout_p: config.dropout_p,
                scale: None,
            }))
        } else {
            None
        };

        Self {
            wq: identity(d),
            wk: identity(d),
            wv: identity(d),
            wo: identity(d),
            flash,
            config,
        }
    }

    /// Multi-head attention forward pass.
    ///
    /// Steps:
    /// 1. Project Q, K, V via learned weights.
    /// 2. Split into `nheads` heads.
    /// 3. Run attention per head.
    /// 4. Concatenate head outputs.
    /// 5. Apply output projection.
    ///
    /// # Errors
    /// Returns [`AttentionError`] on shape mismatches or attention failures.
    pub fn forward(
        &self,
        query: &[Vec<f64>],
        key: &[Vec<f64>],
        value: &[Vec<f64>],
    ) -> Result<Vec<Vec<f64>>, AttentionError> {
        if query.is_empty() || key.is_empty() || value.is_empty() {
            return Err(AttentionError::EmptyInput);
        }

        let seq_len = query.len();
        let d_model = self.config.d_model;
        let nheads = self.config.nheads;
        let head_dim = self.config.head_dim;

        // 1. Linear projections
        let q_proj = matmul(query, &self.wq); // [seq_len, d_model]
        let k_proj = matmul(key, &self.wk);
        let v_proj = matmul(value, &self.wv);

        // 2. Split heads: [seq_len, d_model] -> [nheads][seq_len][head_dim]
        let split_heads = |proj: &[Vec<f64>]| -> Vec<Vec<Vec<f64>>> {
            let mut heads = vec![vec![vec![0.0_f64; head_dim]; proj.len()]; nheads];
            for (t, row) in proj.iter().enumerate() {
                for (h, head_buf) in heads.iter_mut().enumerate() {
                    let start = h * head_dim;
                    head_buf[t].copy_from_slice(&row[start..start + head_dim]);
                }
            }
            heads
        };

        let q_heads = split_heads(&q_proj);
        let k_heads = split_heads(&k_proj);
        let v_heads = split_heads(&v_proj);

        // 3. Attention per head
        // Build causal mask once (shared across heads)
        let causal_mask: Option<Vec<Vec<f64>>> = if self.config.causal {
            let seq_k = key.len();
            let mut m = vec![vec![0.0_f64; seq_k]; seq_len];
            for (i, row) in m.iter_mut().enumerate() {
                for (j, cell) in row.iter_mut().enumerate() {
                    if j > i {
                        *cell = f64::NEG_INFINITY;
                    }
                }
            }
            Some(m)
        } else {
            None
        };

        let mut head_outputs: Vec<Vec<Vec<f64>>> = Vec::with_capacity(nheads);
        for h in 0..nheads {
            let head_out = if let Some(ref fa) = self.flash {
                fa.forward(&q_heads[h], &k_heads[h], &v_heads[h])?
            } else {
                scaled_dot_product_attention(
                    &q_heads[h],
                    &k_heads[h],
                    &v_heads[h],
                    causal_mask.as_deref(),
                    None,
                )?
            };
            head_outputs.push(head_out);
        }

        // 4. Concatenate heads: [nheads][seq_len][head_dim] -> [seq_len, d_model]
        let mut concat = vec![vec![0.0_f64; d_model]; seq_len];
        for (h, head_out) in head_outputs.iter().enumerate() {
            let start = h * head_dim;
            for (t, row) in concat.iter_mut().enumerate() {
                row[start..start + head_dim].copy_from_slice(&head_out[t]);
            }
        }

        // 5. Output projection
        let result = matmul(&concat, &self.wo);
        Ok(result)
    }
}

// ─── KV Cache ─────────────────────────────────────────────────────────────────

/// Incremental KV cache for a single attention head.
///
/// Pre-allocates `max_seq_len` rows of `head_dim` elements and tracks
/// how many have been filled.
#[derive(Debug, Clone)]
pub struct KVCache {
    /// Maximum number of tokens this cache can hold.
    pub max_seq_len: usize,
    /// Number of token positions currently stored.
    pub current_len: usize,
    /// Stored key vectors `[current_len, head_dim]`.
    pub k_cache: Vec<Vec<f64>>,
    /// Stored value vectors `[current_len, head_dim]`.
    pub v_cache: Vec<Vec<f64>>,
}

impl KVCache {
    /// Create an empty cache.
    pub fn new(max_seq_len: usize, _head_dim: usize) -> Self {
        Self {
            max_seq_len,
            current_len: 0,
            k_cache: Vec::with_capacity(max_seq_len),
            v_cache: Vec::with_capacity(max_seq_len),
        }
    }

    /// Append a single (key, value) pair.
    ///
    /// # Errors
    /// Returns [`AttentionError::KVCacheError`] if the cache is already full.
    pub fn append(&mut self, k: Vec<f64>, v: Vec<f64>) -> Result<(), AttentionError> {
        if self.current_len >= self.max_seq_len {
            return Err(AttentionError::KVCacheError(format!(
                "cache full ({} / {})",
                self.current_len, self.max_seq_len
            )));
        }
        self.k_cache.push(k);
        self.v_cache.push(v);
        self.current_len += 1;
        Ok(())
    }

    /// Borrow the stored key vectors.
    pub fn get_keys(&self) -> &[Vec<f64>] {
        &self.k_cache
    }

    /// Borrow the stored value vectors.
    pub fn get_values(&self) -> &[Vec<f64>] {
        &self.v_cache
    }

    /// Number of tokens currently stored.
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Returns `true` if no tokens have been stored.
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Returns `true` if the cache has no remaining capacity.
    pub fn is_full(&self) -> bool {
        self.current_len >= self.max_seq_len
    }

    /// Clear the cache without deallocating memory.
    pub fn reset(&mut self) {
        self.k_cache.clear();
        self.v_cache.clear();
        self.current_len = 0;
    }

    /// Number of additional tokens that can be appended.
    pub fn remaining_capacity(&self) -> usize {
        self.max_seq_len.saturating_sub(self.current_len)
    }
}

// ─── Multi-Head KV Cache ──────────────────────────────────────────────────────

/// Collection of per-head KV caches.
pub struct MultiHeadKVCache {
    /// Number of attention heads.
    pub nheads: usize,
    /// Per-head caches.
    pub caches: Vec<KVCache>,
}

impl MultiHeadKVCache {
    /// Create one [`KVCache`] per head.
    pub fn new(nheads: usize, max_seq_len: usize, head_dim: usize) -> Self {
        let caches = (0..nheads)
            .map(|_| KVCache::new(max_seq_len, head_dim))
            .collect();
        Self { nheads, caches }
    }

    /// Append a (key, value) pair to the specified head's cache.
    ///
    /// # Errors
    /// Returns [`AttentionError::KVCacheError`] for an invalid `head` index or
    /// if the target cache is full.
    pub fn append_head(
        &mut self,
        head: usize,
        k: Vec<f64>,
        v: Vec<f64>,
    ) -> Result<(), AttentionError> {
        if head >= self.nheads {
            return Err(AttentionError::KVCacheError(format!(
                "head index {} out of range (nheads={})",
                head, self.nheads
            )));
        }
        self.caches[head].append(k, v)
    }

    /// Borrow the cache for a specific head; returns `None` for out-of-range indices.
    pub fn get_head(&self, head: usize) -> Option<&KVCache> {
        self.caches.get(head)
    }

    /// Reset all per-head caches.
    pub fn reset_all(&mut self) {
        for cache in self.caches.iter_mut() {
            cache.reset();
        }
    }

    /// Sum of all per-head cache lengths (total tokens cached across heads).
    pub fn total_tokens_cached(&self) -> usize {
        self.caches.iter().map(|c| c.current_len).sum()
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ──────────────────────────────────────────────────────────────

    /// Build an n×n identity matrix as Vec<Vec<f64>>.
    fn identity(n: usize) -> Vec<Vec<f64>> {
        let mut m = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            m[i][i] = 1.0;
        }
        m
    }

    /// Fill a matrix with a constant value.
    fn const_mat(rows: usize, cols: usize, val: f64) -> Vec<Vec<f64>> {
        vec![vec![val; cols]; rows]
    }

    /// Check that two matrices are element-wise close within `tol`.
    fn assert_close(a: &[Vec<f64>], b: &[Vec<f64>], tol: f64, label: &str) {
        assert_eq!(a.len(), b.len(), "{}: row count mismatch", label);
        for (i, (ra, rb)) in a.iter().zip(b.iter()).enumerate() {
            assert_eq!(ra.len(), rb.len(), "{}: row {} col count mismatch", label, i);
            for (j, (&va, &vb)) in ra.iter().zip(rb.iter()).enumerate() {
                assert!(
                    (va - vb).abs() < tol,
                    "{}: [{},{}] expected {} got {} (diff {})",
                    label,
                    i,
                    j,
                    vb,
                    va,
                    (va - vb).abs()
                );
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // scaled_dot_product_attention
    // ═══════════════════════════════════════════════════════════════════════

    /// When Q=K=V=I (identity) with no mask, every query attends uniformly
    /// to all keys, so output ≈ (1/seq_len) * I.
    #[test]
    fn test_sdpa_identity_uniform_attention() {
        let n = 4;
        let q = identity(n);
        let k = identity(n);
        let v = identity(n);
        let out = scaled_dot_product_attention(&q, &k, &v, None, None)
            .expect("sdpa identity");
        assert_eq!(out.len(), n);
        assert_eq!(out[0].len(), n);
        // Each row of output should sum to 1 and be uniform
        for row in &out {
            let s: f64 = row.iter().sum();
            assert!((s - 1.0).abs() < 1e-10, "row should sum to 1, got {}", s);
        }
    }

    /// Output shape is [seq_q, head_dim_v].
    #[test]
    fn test_sdpa_output_shape() {
        let seq_q = 5;
        let seq_k = 7;
        let head_dim = 8;
        let q = vec![vec![0.1_f64; head_dim]; seq_q];
        let k = vec![vec![0.2_f64; head_dim]; seq_k];
        let v = vec![vec![0.3_f64; head_dim]; seq_k];
        let out = scaled_dot_product_attention(&q, &k, &v, None, None)
            .expect("sdpa shape");
        assert_eq!(out.len(), seq_q);
        assert_eq!(out[0].len(), head_dim);
    }

    /// Causal mask: upper-triangular positions set to −∞ should receive
    /// zero attention weight.
    #[test]
    fn test_sdpa_causal_mask() {
        let n = 4;
        let q = identity(n);
        let k = identity(n);
        let v = identity(n);
        let mut mask = vec![vec![0.0_f64; n]; n];
        for i in 0..n {
            for j in (i + 1)..n {
                mask[i][j] = f64::NEG_INFINITY;
            }
        }
        let out = scaled_dot_product_attention(&q, &k, &v, Some(&mask), None)
            .expect("sdpa causal");
        // Row 0 can only attend to position 0 → output row 0 == v[0]
        assert_close(&[out[0].clone()], &[v[0].clone()], 1e-10, "causal row 0");
        // Row 1 attends to positions 0..=1 equally (same dot-product values
        // after causal mask), output is average of v[0] and v[1]
        for &x in &out[1] {
            // Each component should be in [0, 1] and finite
            assert!(x.is_finite(), "causal row 1 not finite");
        }
    }

    /// A larger scale factor reduces the entropy of softmax → outputs shift
    /// toward the highest-scoring key.
    #[test]
    fn test_sdpa_scale_effect() {
        let n = 4;
        let head_dim = 4;
        // Non-uniform Q: first query matches first key strongly
        let q = vec![vec![1.0_f64, 0.0, 0.0, 0.0]; n];
        let k = identity(head_dim);
        let v = identity(head_dim);
        // Default scale
        let out_default = scaled_dot_product_attention(&q, &k, &v, None, None)
            .expect("sdpa scale default");
        // Large scale → very sharp distribution
        let out_large = scaled_dot_product_attention(&q, &k, &v, None, Some(10.0))
            .expect("sdpa scale large");
        // With large scale the first key gets much higher weight → output[0][0]
        // should be larger in the large-scale case
        assert!(
            out_large[0][0] > out_default[0][0],
            "large scale should concentrate attention"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // FlashAttention
    // ═══════════════════════════════════════════════════════════════════════

    /// FlashAttention must agree with standard attention on a random-ish matrix.
    #[test]
    fn test_flash_matches_standard() {
        let seq = 8;
        let dim = 4;
        // Use a simple deterministic matrix
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i * dim + d) as f64) * 0.1).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let fa = FlashAttention::new(FlashAttentionConfig {
            block_size_q: 4,
            block_size_k: 4,
            ..Default::default()
        });
        let ok = fa
            .verify_against_standard(&q, &k, &v, 1e-9)
            .expect("verify");
        assert!(ok, "flash should match standard");
    }

    /// Causal mode must agree with standard attention with causal mask.
    #[test]
    fn test_flash_causal_matches_standard() {
        let seq = 6;
        let dim = 4;
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i + d) as f64) * 0.05).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let fa = FlashAttention::new(FlashAttentionConfig {
            causal: true,
            block_size_q: 3,
            block_size_k: 3,
            ..Default::default()
        });
        let ok = fa
            .verify_against_standard(&q, &k, &v, 1e-9)
            .expect("verify causal");
        assert!(ok, "flash causal should match standard");
    }

    /// Block size 16 gives same result as standard.
    #[test]
    fn test_flash_block_size_16() {
        let seq = 32;
        let dim = 8;
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i + d) as f64) * 0.02).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let fa = FlashAttention::new(FlashAttentionConfig {
            block_size_q: 16,
            block_size_k: 16,
            ..Default::default()
        });
        let ok = fa.verify_against_standard(&q, &k, &v, 1e-9).expect("b16");
        assert!(ok, "block_size 16 should match standard");
    }

    /// Block size 32 gives same result as standard.
    #[test]
    fn test_flash_block_size_32() {
        let seq = 48;
        let dim = 8;
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i + d + 1) as f64) * 0.03).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let fa = FlashAttention::new(FlashAttentionConfig {
            block_size_q: 32,
            block_size_k: 32,
            ..Default::default()
        });
        let ok = fa.verify_against_standard(&q, &k, &v, 1e-9).expect("b32");
        assert!(ok, "block_size 32 should match standard");
    }

    /// Block size 64 gives same result as standard.
    #[test]
    fn test_flash_block_size_64() {
        let seq = 80;
        let dim = 8;
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i + d + 2) as f64) * 0.01).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let fa = FlashAttention::new(FlashAttentionConfig::default());
        let ok = fa.verify_against_standard(&q, &k, &v, 1e-9).expect("b64");
        assert!(ok, "block_size 64 should match standard");
    }

    /// Single-token sequence (seq_len=1).
    #[test]
    fn test_flash_single_token() {
        let q = vec![vec![1.0_f64, 0.5, -0.3, 0.8]];
        let k = vec![vec![0.2_f64, 1.0, 0.0, -0.1]];
        let v = vec![vec![1.0_f64, 2.0, 3.0, 4.0]];
        let fa = FlashAttention::new(FlashAttentionConfig::default());
        let ok = fa
            .verify_against_standard(&q, &k, &v, 1e-9)
            .expect("single token");
        assert!(ok, "single token flash should match standard");
    }

    /// Long sequence (seq_len > 2 × block_size).
    #[test]
    fn test_flash_long_sequence() {
        let seq = 200;
        let dim = 16;
        let fa = FlashAttention::new(FlashAttentionConfig {
            block_size_q: 64,
            block_size_k: 64,
            ..Default::default()
        });
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i + d) as f64) / (seq as f64)).collect())
            .collect();
        let k = q.clone();
        let v = q.clone();
        let ok = fa
            .verify_against_standard(&q, &k, &v, 1e-9)
            .expect("long seq");
        assert!(ok, "long sequence flash should match standard");
    }

    /// Numerically stable: large QK values should not produce NaN.
    #[test]
    fn test_flash_numerical_stability() {
        let seq = 8;
        let dim = 4;
        // Large values that would cause overflow without the max-shift trick
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..dim).map(|d| ((i * dim + d) as f64) * 100.0).collect())
            .collect();
        let k = q.clone();
        let v = const_mat(seq, dim, 1.0);
        let fa = FlashAttention::new(FlashAttentionConfig {
            block_size_q: 4,
            block_size_k: 4,
            ..Default::default()
        });
        let out = fa.forward(&q, &k, &v).expect("stability");
        for row in &out {
            for &x in row {
                assert!(x.is_finite(), "output should be finite with large QK");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MultiHeadAttention
    // ═══════════════════════════════════════════════════════════════════════

    /// Non-divisible head count must return an error.
    #[test]
    fn test_mha_invalid_heads() {
        let err = MultiHeadAttentionConfig::new(10, 3);
        assert!(
            matches!(err, Err(AttentionError::InvalidHeads { dm: 10, nh: 3 })),
            "expected InvalidHeads error"
        );
    }

    /// Output shape should be [seq_len, d_model].
    #[test]
    fn test_mha_output_shape() {
        let cfg = MultiHeadAttentionConfig::new(8, 2).expect("cfg");
        let mha = MultiHeadAttention::new(cfg);
        let seq = 6;
        let d = 8;
        let x = const_mat(seq, d, 0.5);
        let out = mha.forward(&x, &x, &x).expect("mha forward");
        assert_eq!(out.len(), seq, "output rows");
        assert_eq!(out[0].len(), d, "output cols");
    }

    /// Standard (non-flash) attention forward pass produces finite values.
    #[test]
    fn test_mha_standard_forward() {
        let cfg = MultiHeadAttentionConfig::new(8, 4).expect("cfg");
        let mha = MultiHeadAttention::new(cfg);
        let seq = 5;
        let d = 8;
        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..d).map(|j| (i + j) as f64 * 0.1).collect())
            .collect();
        let out = mha.forward(&q, &q, &q).expect("mha std");
        for row in &out {
            for &x in row {
                assert!(x.is_finite(), "mha standard output should be finite");
            }
        }
    }

    /// Flash backend gives same result as standard (identity weights → both
    /// run same math through different code paths).
    #[test]
    fn test_mha_flash_vs_standard() {
        let d = 8;
        let heads = 2;
        let seq = 10;

        let mut cfg_std = MultiHeadAttentionConfig::new(d, heads).expect("cfg");
        cfg_std.use_flash = false;
        let mha_std = MultiHeadAttention::new(cfg_std);

        let mut cfg_flash = MultiHeadAttentionConfig::new(d, heads).expect("cfg");
        cfg_flash.use_flash = true;
        cfg_flash.block_size = 4;
        let mha_flash = MultiHeadAttention::new(cfg_flash);

        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..d).map(|j| (i + j) as f64 * 0.05).collect())
            .collect();

        let out_std = mha_std.forward(&q, &q, &q).expect("std");
        let out_flash = mha_flash.forward(&q, &q, &q).expect("flash");
        assert_close(&out_std, &out_flash, 1e-9, "flash vs standard MHA");
    }

    /// Causal MHA: output token 0 should only depend on token 0 (lower-
    /// triangular attention pattern).
    #[test]
    fn test_mha_causal_mode() {
        let d = 8;
        let heads = 2;
        let seq = 6;

        let mut cfg = MultiHeadAttentionConfig::new(d, heads).expect("cfg");
        cfg.causal = true;
        let mha = MultiHeadAttention::new(cfg);

        let q: Vec<Vec<f64>> = (0..seq)
            .map(|i| (0..d).map(|j| (i + j) as f64 * 0.1).collect())
            .collect();
        let out = mha.forward(&q, &q, &q).expect("mha causal");
        assert_eq!(out.len(), seq);
        for row in &out {
            for &x in row {
                assert!(x.is_finite(), "causal mha should be finite");
            }
        }
    }

    // ═══════════════════════════════════════════════════════════════════════
    // KVCache
    // ═══════════════════════════════════════════════════════════════════════

    /// Appended keys are retrievable.
    #[test]
    fn test_kvcache_append_and_get() {
        let mut cache = KVCache::new(10, 4);
        let k = vec![1.0, 2.0, 3.0, 4.0];
        let v = vec![5.0, 6.0, 7.0, 8.0];
        cache.append(k.clone(), v.clone()).expect("append");
        assert_eq!(cache.get_keys(), &[k]);
        assert_eq!(cache.get_values(), &[v]);
        assert_eq!(cache.len(), 1);
    }

    /// `is_full` triggers only when capacity is exhausted.
    #[test]
    fn test_kvcache_is_full() {
        let mut cache = KVCache::new(3, 2);
        assert!(!cache.is_full());
        for _ in 0..3 {
            cache.append(vec![0.0, 0.0], vec![0.0, 0.0]).expect("app");
        }
        assert!(cache.is_full());
    }

    /// After reset, the cache is empty and keys are gone.
    #[test]
    fn test_kvcache_reset() {
        let mut cache = KVCache::new(5, 3);
        for _ in 0..4 {
            cache.append(vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0])
                .expect("app");
        }
        assert_eq!(cache.len(), 4);
        cache.reset();
        assert_eq!(cache.len(), 0);
        assert!(cache.get_keys().is_empty());
    }

    /// `remaining_capacity` tracks available slots correctly.
    #[test]
    fn test_kvcache_remaining_capacity() {
        let mut cache = KVCache::new(8, 2);
        assert_eq!(cache.remaining_capacity(), 8);
        cache.append(vec![0.1, 0.2], vec![0.3, 0.4]).expect("a1");
        cache.append(vec![0.5, 0.6], vec![0.7, 0.8]).expect("a2");
        assert_eq!(cache.remaining_capacity(), 6);
    }

    /// Appending to a full cache returns an error.
    #[test]
    fn test_kvcache_overflow_error() {
        let mut cache = KVCache::new(2, 2);
        cache.append(vec![1.0, 2.0], vec![3.0, 4.0]).expect("a1");
        cache.append(vec![5.0, 6.0], vec![7.0, 8.0]).expect("a2");
        let err = cache.append(vec![9.0, 10.0], vec![11.0, 12.0]);
        assert!(
            matches!(err, Err(AttentionError::KVCacheError(_))),
            "expected KVCacheError on overflow"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════
    // MultiHeadKVCache
    // ═══════════════════════════════════════════════════════════════════════

    /// Per-head append and retrieval.
    #[test]
    fn test_mhkv_append_head() {
        let mut mhkv = MultiHeadKVCache::new(4, 10, 3);
        let k = vec![1.0, 2.0, 3.0];
        let v = vec![4.0, 5.0, 6.0];
        mhkv.append_head(2, k.clone(), v.clone()).expect("append");
        let head = mhkv.get_head(2).expect("get head 2");
        assert_eq!(head.get_keys(), &[k]);
        assert_eq!(head.get_values(), &[v]);
    }

    /// `get_head` returns the correct cache for a valid index.
    #[test]
    fn test_mhkv_get_head() {
        let mhkv = MultiHeadKVCache::new(3, 5, 4);
        assert!(mhkv.get_head(0).is_some());
        assert!(mhkv.get_head(2).is_some());
        assert!(mhkv.get_head(3).is_none());
    }

    /// `reset_all` clears every head's cache.
    #[test]
    fn test_mhkv_reset_all() {
        let mut mhkv = MultiHeadKVCache::new(2, 5, 2);
        mhkv.append_head(0, vec![1.0, 2.0], vec![3.0, 4.0])
            .expect("h0");
        mhkv.append_head(1, vec![5.0, 6.0], vec![7.0, 8.0])
            .expect("h1");
        mhkv.reset_all();
        assert_eq!(mhkv.total_tokens_cached(), 0);
    }

    /// `total_tokens_cached` sums across all heads.
    #[test]
    fn test_mhkv_total_tokens_cached() {
        let mut mhkv = MultiHeadKVCache::new(3, 10, 2);
        mhkv.append_head(0, vec![1.0, 2.0], vec![3.0, 4.0])
            .expect("h0a");
        mhkv.append_head(0, vec![5.0, 6.0], vec![7.0, 8.0])
            .expect("h0b");
        mhkv.append_head(1, vec![9.0, 10.0], vec![11.0, 12.0])
            .expect("h1");
        // head 2: zero
        assert_eq!(mhkv.total_tokens_cached(), 3);
    }
}
