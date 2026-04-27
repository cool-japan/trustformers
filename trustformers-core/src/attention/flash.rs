//! Pure-Rust Flash Attention v2 algorithm.
//!
//! Computes softmax(QK^T/√d)V in tiles without materializing the full attention
//! matrix.  Reference: "FlashAttention-2: Faster Attention with Better
//! Parallelism and Work Partitioning" (Dao, 2023).
//!
//! Memory complexity: O(seq_len * head_dim) instead of O(seq_len²).

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors returned by flash-attention operations.
#[derive(Debug, Clone, PartialEq)]
pub enum FlashAttnError {
    /// An input slice had the wrong number of elements.
    InvalidDimensions {
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        got: usize,
    },
    /// Block size is not usable (e.g. zero).
    InvalidBlockSize(String),
    /// Query and key/value sequence lengths are inconsistent with the config.
    SeqLenMismatch,
}

impl fmt::Display for FlashAttnError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidDimensions { expected, got } => {
                write!(f, "invalid dimensions: expected {expected}, got {got}")
            }
            Self::InvalidBlockSize(msg) => write!(f, "invalid block size: {msg}"),
            Self::SeqLenMismatch => write!(f, "sequence length mismatch between Q and K/V"),
        }
    }
}

impl std::error::Error for FlashAttnError {}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for the Flash Attention kernel.
#[derive(Debug, Clone)]
pub struct FlashAttentionConfig {
    /// Query block size (tile rows).  Default: 64.
    pub block_size_q: usize,
    /// Key/value block size (tile columns).  Default: 64.
    pub block_size_kv: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Apply causal (lower-triangular) masking.  Default: `true`.
    pub causal: bool,
    /// Scale factor applied to QK^T.  Default: `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// Attention dropout probability (stored; not applied in this reference
    /// implementation).  Default: 0.0.
    pub dropout_prob: f32,
}

impl FlashAttentionConfig {
    /// Build a config with all fields required.
    pub fn new(head_dim: usize, num_heads: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            block_size_q: 64,
            block_size_kv: 64,
            head_dim,
            num_heads,
            causal: true,
            scale,
            dropout_prob: 0.0,
        }
    }
}

// ── Internal helpers ───────────────────────────────────────────────────────

/// Row-wise maximum of a flat row-major matrix `[rows, cols]`.
pub fn row_max(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![f32::NEG_INFINITY; rows];
    for r in 0..rows {
        for c in 0..cols {
            let v = matrix[r * cols + c];
            if v > out[r] {
                out[r] = v;
            }
        }
    }
    out
}

/// Row-wise sum of a flat row-major matrix `[rows, cols]`.
pub fn row_sum(matrix: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows];
    for r in 0..rows {
        for c in 0..cols {
            out[r] += matrix[r * cols + c];
        }
    }
    out
}

/// Matrix multiply: C = A * B,  A: [m, k],  B: [k, n],  C: [m, n].
///
/// All matrices are flat row-major slices.
pub fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut c = vec![0.0_f32; m * n];
    for i in 0..m {
        for p in 0..k {
            let aip = a[i * k + p];
            for j in 0..n {
                c[i * n + j] += aip * b[p * n + j];
            }
        }
    }
    c
}

/// In-place per-row softmax of a flat row-major matrix `[rows, cols]`.
pub fn softmax_rows(matrix: &mut [f32], rows: usize, cols: usize) {
    for r in 0..rows {
        let start = r * cols;
        let end = start + cols;
        let row = &mut matrix[start..end];

        let max_v = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0.0_f32;
        for v in row.iter_mut() {
            *v = (*v - max_v).exp();
            sum += *v;
        }
        if sum > 0.0 {
            for v in row.iter_mut() {
                *v /= sum;
            }
        }
    }
}

// ── Naive reference implementation (used in tests) ─────────────────────────

/// Naive O(N²) scaled-dot-product attention for a **single head**.
///
/// - `q`: `[seq_len_q, head_dim]`
/// - `k`: `[seq_len_kv, head_dim]`
/// - `v`: `[seq_len_kv, head_dim]`
/// - Returns: `[seq_len_q, head_dim]`
pub fn naive_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    scale: f32,
    seq_len_q: usize,
    seq_len_kv: usize,
    head_dim: usize,
    causal: bool,
) -> Vec<f32> {
    // Compute S = Q K^T * scale  [seq_len_q, seq_len_kv]
    let mut s = matmul(q, &transpose(k, seq_len_kv, head_dim), seq_len_q, head_dim, seq_len_kv);

    // Apply scale
    for v_s in s.iter_mut() {
        *v_s *= scale;
    }

    // Apply causal mask
    if causal {
        for i in 0..seq_len_q {
            for j in 0..seq_len_kv {
                if j > i {
                    s[i * seq_len_kv + j] = f32::NEG_INFINITY;
                }
            }
        }
    }

    // Softmax
    softmax_rows(&mut s, seq_len_q, seq_len_kv);

    // Output = S @ V  [seq_len_q, head_dim]
    matmul(&s, v, seq_len_q, seq_len_kv, head_dim)
}

/// Transpose a flat row-major matrix `[rows, cols]` → `[cols, rows]`.
fn transpose(a: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = a[r * cols + c];
        }
    }
    out
}

// ── Flash Attention Forward ────────────────────────────────────────────────

/// Flash Attention forward pass.
///
/// Computes multi-head attention in tiles, never materialising the full
/// `[seq_len_q, seq_len_kv]` attention matrix.
///
/// ## Layout
/// - `q`:  `[seq_len_q,  num_heads, head_dim]`  (row-major)
/// - `k`:  `[seq_len_kv, num_heads, head_dim]`
/// - `v`:  `[seq_len_kv, num_heads, head_dim]`
/// - output: `[seq_len_q,  num_heads, head_dim]`
///
/// # Errors
/// Returns [`FlashAttnError`] on shape or configuration violations.
pub fn flash_attention_forward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &FlashAttentionConfig,
    seq_len_q: usize,
    seq_len_kv: usize,
) -> Result<Vec<f32>, FlashAttnError> {
    let h = config.num_heads;
    let d = config.head_dim;

    // Validate block sizes
    if config.block_size_q == 0 {
        return Err(FlashAttnError::InvalidBlockSize(
            "block_size_q must be > 0".to_string(),
        ));
    }
    if config.block_size_kv == 0 {
        return Err(FlashAttnError::InvalidBlockSize(
            "block_size_kv must be > 0".to_string(),
        ));
    }

    let expected_q = seq_len_q * h * d;
    if q.len() != expected_q {
        return Err(FlashAttnError::InvalidDimensions {
            expected: expected_q,
            got: q.len(),
        });
    }
    let expected_kv = seq_len_kv * h * d;
    if k.len() != expected_kv {
        return Err(FlashAttnError::InvalidDimensions {
            expected: expected_kv,
            got: k.len(),
        });
    }
    if v.len() != expected_kv {
        return Err(FlashAttnError::InvalidDimensions {
            expected: expected_kv,
            got: v.len(),
        });
    }

    let bq = config.block_size_q;
    let bkv = config.block_size_kv;
    let scale = config.scale;

    let mut output = vec![0.0_f32; seq_len_q * h * d];

    // Process each head independently
    for head in 0..h {
        // Slice out this head's data as flat [seq_len, head_dim] matrices
        // Q_h[i, d] = q[i * h * d + head * d + d_idx]
        let q_h = extract_head(q, seq_len_q, h, d, head);
        let k_h = extract_head(k, seq_len_kv, h, d, head);
        let v_h = extract_head(v, seq_len_kv, h, d, head);

        // Running statistics for online softmax
        let mut m = vec![f32::NEG_INFINITY; seq_len_q]; // running max
        let mut l = vec![0.0_f32; seq_len_q]; // running normalizer
        let mut o_h = vec![0.0_f32; seq_len_q * d]; // output accumulator

        // Outer loop: query blocks
        let mut qi_start = 0;
        while qi_start < seq_len_q {
            let qi_end = (qi_start + bq).min(seq_len_q);
            let br = qi_end - qi_start; // actual rows in this Q block

            // Inner loop: key/value blocks
            let mut kj_start = 0;
            while kj_start < seq_len_kv {
                let kj_end = (kj_start + bkv).min(seq_len_kv);
                let bc = kj_end - kj_start; // actual cols in this KV block

                // S_block = Q_block @ K_block^T * scale  [br, bc]
                let q_block = &q_h[qi_start * d..qi_end * d];
                let k_block = &k_h[kj_start * d..kj_end * d];
                let k_block_t = transpose(k_block, bc, d);

                let mut s_block = matmul(q_block, &k_block_t, br, d, bc);

                // Scale
                for val in s_block.iter_mut() {
                    *val *= scale;
                }

                // Causal mask: position j+kj_start > i+qi_start → -inf
                if config.causal {
                    for r in 0..br {
                        let abs_row = qi_start + r;
                        for c in 0..bc {
                            let abs_col = kj_start + c;
                            if abs_col > abs_row {
                                s_block[r * bc + c] = f32::NEG_INFINITY;
                            }
                        }
                    }
                }

                // Per-row max of S_block [br]
                let m_block = row_max(&s_block, br, bc);

                // Update running max and compute correction factors
                // m_new[r] = max(m[qi_start+r], m_block[r])
                let mut m_new = vec![0.0_f32; br];
                for r in 0..br {
                    m_new[r] = m[qi_start + r].max(m_block[r]);
                }

                // P_block[r,c] = exp(S_block[r,c] - m_new[r])  [br, bc]
                let mut p_block = vec![0.0_f32; br * bc];
                for r in 0..br {
                    for c in 0..bc {
                        p_block[r * bc + c] = (s_block[r * bc + c] - m_new[r]).exp();
                    }
                }

                // Row sum of P_block
                let p_sum = row_sum(&p_block, br, bc);

                // l_new[r] = exp(m[r] - m_new[r]) * l[r] + p_sum[r]
                // O[r] = (exp(m[r] - m_new[r]) * l[r] * O[r] + P_block @ V_block) / l_new
                let v_block = &v_h[kj_start * d..kj_end * d];
                let pv = matmul(&p_block, v_block, br, bc, d); // [br, d]

                for r in 0..br {
                    let abs_r = qi_start + r;
                    let correction = (m[abs_r] - m_new[r]).exp();
                    let l_new = correction * l[abs_r] + p_sum[r];

                    if l_new > 0.0 {
                        let inv_l = 1.0 / l_new;
                        let old_weight = correction * l[abs_r] * inv_l;
                        for dd in 0..d {
                            o_h[abs_r * d + dd] = old_weight * o_h[abs_r * d + dd]
                                + pv[r * d + dd] * inv_l;
                        }
                    }

                    m[abs_r] = m_new[r];
                    l[abs_r] = l_new;
                }

                kj_start = kj_end;
            }

            qi_start = qi_end;
        }

        // Write head output back into interleaved layout
        insert_head(&mut output, &o_h, seq_len_q, h, d, head);
    }

    Ok(output)
}

// ── Flash Attention Backward ───────────────────────────────────────────────

/// Simplified Flash Attention backward pass.
///
/// Recomputes the attention weights from Q, K, V and the saved output O, then
/// computes gradients dQ, dK, dV.
///
/// Returns `(dQ, dK, dV)` with the same layouts as Q, K, V respectively.
///
/// # Errors
/// Returns [`FlashAttnError`] on shape violations.
pub fn flash_attention_backward(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    o: &[f32],
    do_: &[f32],
    config: &FlashAttentionConfig,
    seq_len_q: usize,
    seq_len_kv: usize,
) -> Result<(Vec<f32>, Vec<f32>, Vec<f32>), FlashAttnError> {
    let h = config.num_heads;
    let d = config.head_dim;

    // Dimension validation
    let expected_q = seq_len_q * h * d;
    let expected_kv = seq_len_kv * h * d;

    if q.len() != expected_q {
        return Err(FlashAttnError::InvalidDimensions { expected: expected_q, got: q.len() });
    }
    if k.len() != expected_kv {
        return Err(FlashAttnError::InvalidDimensions { expected: expected_kv, got: k.len() });
    }
    if v.len() != expected_kv {
        return Err(FlashAttnError::InvalidDimensions { expected: expected_kv, got: v.len() });
    }
    if o.len() != expected_q {
        return Err(FlashAttnError::InvalidDimensions { expected: expected_q, got: o.len() });
    }
    if do_.len() != expected_q {
        return Err(FlashAttnError::InvalidDimensions { expected: expected_q, got: do_.len() });
    }

    let scale = config.scale;

    let mut dq = vec![0.0_f32; expected_q];
    let mut dk = vec![0.0_f32; expected_kv];
    let mut dv = vec![0.0_f32; expected_kv];

    for head in 0..h {
        let q_h = extract_head(q, seq_len_q, h, d, head);
        let k_h = extract_head(k, seq_len_kv, h, d, head);
        let v_h = extract_head(v, seq_len_kv, h, d, head);
        let _o_h = extract_head(o, seq_len_q, h, d, head);
        let do_h = extract_head(do_, seq_len_q, h, d, head);

        // Recompute attention weights P [seq_len_q, seq_len_kv]
        let k_h_t = transpose(&k_h, seq_len_kv, d);
        let mut s = matmul(&q_h, &k_h_t, seq_len_q, d, seq_len_kv);

        for val in s.iter_mut() {
            *val *= scale;
        }

        if config.causal {
            for i in 0..seq_len_q {
                for j in 0..seq_len_kv {
                    if j > i {
                        s[i * seq_len_kv + j] = f32::NEG_INFINITY;
                    }
                }
            }
        }

        softmax_rows(&mut s, seq_len_q, seq_len_kv);
        // P = s  [seq_len_q, seq_len_kv]

        // dV = P^T @ dO  [seq_len_kv, d]
        let p_t = transpose(&s, seq_len_q, seq_len_kv);
        let dv_h = matmul(&p_t, &do_h, seq_len_kv, seq_len_q, d);

        // dP = dO @ V^T  [seq_len_q, seq_len_kv]
        let v_h_t = transpose(&v_h, seq_len_kv, d);
        let dp = matmul(&do_h, &v_h_t, seq_len_q, d, seq_len_kv);

        // dS = P * (dP - sum_j(P * dP))  (softmax backward)
        // sum per row of (P * dP)
        let mut ds = vec![0.0_f32; seq_len_q * seq_len_kv];
        for i in 0..seq_len_q {
            let mut row_dot = 0.0_f32;
            for j in 0..seq_len_kv {
                row_dot += s[i * seq_len_kv + j] * dp[i * seq_len_kv + j];
            }
            for j in 0..seq_len_kv {
                ds[i * seq_len_kv + j] =
                    s[i * seq_len_kv + j] * (dp[i * seq_len_kv + j] - row_dot) * scale;
            }
        }

        // dQ = dS @ K  [seq_len_q, d]
        let dq_h = matmul(&ds, &k_h, seq_len_q, seq_len_kv, d);

        // dK = dS^T @ Q  [seq_len_kv, d]
        let ds_t = transpose(&ds, seq_len_q, seq_len_kv);
        let dk_h = matmul(&ds_t, &q_h, seq_len_kv, seq_len_q, d);

        insert_head(&mut dq, &dq_h, seq_len_q, h, d, head);
        insert_head(&mut dk, &dk_h, seq_len_kv, h, d, head);
        insert_head(&mut dv, &dv_h, seq_len_kv, h, d, head);
    }

    Ok((dq, dk, dv))
}

// ── Layout helpers ─────────────────────────────────────────────────────────

/// Extract a single head's data from interleaved layout.
///
/// Input layout: `[seq_len, num_heads, head_dim]`
/// Output: `[seq_len, head_dim]`  (contiguous, row-major)
fn extract_head(data: &[f32], seq_len: usize, num_heads: usize, head_dim: usize, head: usize) -> Vec<f32> {
    let mut out = vec![0.0_f32; seq_len * head_dim];
    for i in 0..seq_len {
        for d in 0..head_dim {
            out[i * head_dim + d] = data[i * num_heads * head_dim + head * head_dim + d];
        }
    }
    out
}

/// Write a single head's data back into interleaved layout.
fn insert_head(data: &mut [f32], head_data: &[f32], seq_len: usize, num_heads: usize, head_dim: usize, head: usize) {
    for i in 0..seq_len {
        for d in 0..head_dim {
            data[i * num_heads * head_dim + head * head_dim + d] = head_data[i * head_dim + d];
        }
    }
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // Helper: build a small random-ish deterministic tensor
    fn make_tensor(seq: usize, heads: usize, dim: usize, seed: f32) -> Vec<f32> {
        let n = seq * heads * dim;
        (0..n)
            .map(|i| ((i as f32 * seed * 0.1).sin() * 0.5 + 0.5) * 0.2)
            .collect()
    }

    // Helper: compare two slices within tolerance
    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{label}: index {i}: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    // ── Test 1: Basic forward pass output shape ─────────────────────────

    #[test]
    fn test_basic_output_shape() {
        let seq_q = 4;
        let seq_kv = 4;
        let heads = 2;
        let dim = 8;

        let q = make_tensor(seq_q, heads, dim, 1.0);
        let k = make_tensor(seq_kv, heads, dim, 2.0);
        let v = make_tensor(seq_kv, heads, dim, 3.0);

        let config = FlashAttentionConfig::new(dim, heads);
        let out = flash_attention_forward(&q, &k, &v, &config, seq_q, seq_kv)
            .expect("forward should succeed");

        assert_eq!(out.len(), seq_q * heads * dim);
    }

    // ── Test 2: Flash attention matches naive (no causal) ───────────────

    #[test]
    fn test_matches_naive_no_causal() {
        let seq = 6;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.1);
        let k = make_tensor(seq, heads, dim, 2.2);
        let v = make_tensor(seq, heads, dim, 3.3);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = false;
        config.block_size_q = 3;
        config.block_size_kv = 3;

        let flash_out = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("forward");

        // Naive (single head, no interleaving needed for heads=1)
        let naive_out = naive_attention(&q, &k, &v, config.scale, seq, seq, dim, false);

        assert_close(&flash_out, &naive_out, 1e-4, "flash vs naive (no causal)");
    }

    // ── Test 3: Flash attention matches naive (causal) ──────────────────

    #[test]
    fn test_matches_naive_causal() {
        let seq = 8;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.3);
        let k = make_tensor(seq, heads, dim, 2.4);
        let v = make_tensor(seq, heads, dim, 3.5);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = true;
        config.block_size_q = 4;
        config.block_size_kv = 4;

        let flash_out = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("forward");

        let naive_out = naive_attention(&q, &k, &v, config.scale, seq, seq, dim, true);

        assert_close(&flash_out, &naive_out, 1e-4, "flash vs naive (causal)");
    }

    // ── Test 4: Single query token ──────────────────────────────────────

    #[test]
    fn test_single_query_token() {
        let seq_q = 1;
        let seq_kv = 8;
        let heads = 2;
        let dim = 8;

        let q = make_tensor(seq_q, heads, dim, 0.7);
        let k = make_tensor(seq_kv, heads, dim, 0.8);
        let v = make_tensor(seq_kv, heads, dim, 0.9);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = false;

        let out = flash_attention_forward(&q, &k, &v, &config, seq_q, seq_kv)
            .expect("forward with single query");

        assert_eq!(out.len(), seq_q * heads * dim);
        // Output should be finite
        for &val in &out {
            assert!(val.is_finite(), "output should be finite");
        }
    }

    // ── Test 5: Block size = 1 (degenerate) ────────────────────────────

    #[test]
    fn test_block_size_one() {
        let seq = 5;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = true;
        config.block_size_q = 1;
        config.block_size_kv = 1;

        let flash_out = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("block_size=1 forward");

        let naive_out = naive_attention(&q, &k, &v, config.scale, seq, seq, dim, true);

        assert_close(&flash_out, &naive_out, 1e-4, "block=1 vs naive");
    }

    // ── Test 6: Different head dims ─────────────────────────────────────

    #[test]
    fn test_different_head_dims() {
        for &dim in &[1usize, 4, 16, 32] {
            let seq = 4;
            let heads = 1;

            let q = make_tensor(seq, heads, dim, 1.5);
            let k = make_tensor(seq, heads, dim, 2.5);
            let v = make_tensor(seq, heads, dim, 3.5);

            let mut config = FlashAttentionConfig::new(dim, heads);
            config.causal = false;

            let flash_out = flash_attention_forward(&q, &k, &v, &config, seq, seq)
                .unwrap_or_else(|e| panic!("dim={dim} failed: {e}"));

            let naive_out = naive_attention(&q, &k, &v, config.scale, seq, seq, dim, false);

            assert_close(
                &flash_out,
                &naive_out,
                1e-4,
                &format!("head_dim={dim}"),
            );
        }
    }

    // ── Test 7: Multiple heads match naive per-head ─────────────────────

    #[test]
    fn test_multi_head_matches_naive() {
        let seq = 6;
        let heads = 4;
        let dim = 8;

        let q = make_tensor(seq, heads, dim, 1.1);
        let k = make_tensor(seq, heads, dim, 2.2);
        let v = make_tensor(seq, heads, dim, 3.3);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = false;
        config.block_size_q = 3;
        config.block_size_kv = 3;

        let flash_out = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("multi-head forward");

        // Verify each head independently
        for h in 0..heads {
            let q_h = extract_head(&q, seq, heads, dim, h);
            let k_h = extract_head(&k, seq, heads, dim, h);
            let v_h = extract_head(&v, seq, heads, dim, h);
            let naive_h = naive_attention(&q_h, &k_h, &v_h, config.scale, seq, seq, dim, false);

            let flash_h = extract_head(&flash_out, seq, heads, dim, h);
            assert_close(&flash_h, &naive_h, 1e-4, &format!("head {h}"));
        }
    }

    // ── Test 8: Config defaults ─────────────────────────────────────────

    #[test]
    fn test_config_defaults() {
        let config = FlashAttentionConfig::new(64, 8);
        assert_eq!(config.block_size_q, 64);
        assert_eq!(config.block_size_kv, 64);
        assert!(config.causal);
        assert_eq!(config.dropout_prob, 0.0);
        let expected_scale = 1.0 / (64.0_f32).sqrt();
        assert!((config.scale - expected_scale).abs() < 1e-6);
    }

    // ── Test 9: Error cases ─────────────────────────────────────────────

    #[test]
    fn test_error_wrong_q_size() {
        let config = FlashAttentionConfig::new(4, 2);
        let bad_q = vec![0.0_f32; 5]; // wrong size
        let k = make_tensor(4, 2, 4, 1.0);
        let v = make_tensor(4, 2, 4, 1.0);
        let err = flash_attention_forward(&bad_q, &k, &v, &config, 4, 4);
        assert!(matches!(err, Err(FlashAttnError::InvalidDimensions { .. })));
    }

    #[test]
    fn test_error_zero_block_size() {
        let mut config = FlashAttentionConfig::new(4, 1);
        config.block_size_q = 0;
        let q = make_tensor(4, 1, 4, 1.0);
        let k = make_tensor(4, 1, 4, 1.0);
        let v = make_tensor(4, 1, 4, 1.0);
        let err = flash_attention_forward(&q, &k, &v, &config, 4, 4);
        assert!(matches!(err, Err(FlashAttnError::InvalidBlockSize(_))));
    }

    #[test]
    fn test_error_zero_block_size_kv() {
        let mut config = FlashAttentionConfig::new(4, 1);
        config.block_size_kv = 0;
        let q = make_tensor(4, 1, 4, 1.0);
        let k = make_tensor(4, 1, 4, 1.0);
        let v = make_tensor(4, 1, 4, 1.0);
        let err = flash_attention_forward(&q, &k, &v, &config, 4, 4);
        assert!(matches!(err, Err(FlashAttnError::InvalidBlockSize(_))));
    }

    // ── Test 10: Backward dV correctness via finite differences ─────────

    #[test]
    fn test_backward_dv_finite_diff() {
        let seq = 4;
        let heads = 1;
        let dim = 4;
        let eps = 1e-3_f32;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = false;

        // Forward pass
        let o = flash_attention_forward(&q, &k, &v, &config, seq, seq).expect("forward");

        // Use all-ones dO (loss = sum of outputs)
        let do_ = vec![1.0_f32; o.len()];

        let (_, _, dv) = flash_attention_backward(
            &q, &k, &v, &o, &do_, &config, seq, seq,
        )
        .expect("backward");

        // Finite-difference check for dV: d(sum(O))/dV[i]
        // We check just the first few elements
        for idx in 0..(seq * heads * dim).min(8) {
            let mut v_plus = v.clone();
            v_plus[idx] += eps;
            let o_plus = flash_attention_forward(&q, &k, &v_plus, &config, seq, seq)
                .expect("forward+");

            let loss_plus: f32 = o_plus.iter().sum();
            let loss_base: f32 = o.iter().sum();
            let fd_grad = (loss_plus - loss_base) / eps;

            assert!(
                (dv[idx] - fd_grad).abs() < 1e-2,
                "dV[{idx}] mismatch: analytic={} fd={} diff={}",
                dv[idx],
                fd_grad,
                (dv[idx] - fd_grad).abs()
            );
        }
    }

    // ── Test 11: Causal mask blocks future tokens ────────────────────────

    #[test]
    fn test_causal_mask_blocks_future() {
        // With causal=true, the first query should only see the first key
        let seq = 4;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        // Make k[1..] very large so they dominate if not masked
        let mut k = vec![0.0_f32; seq * heads * dim];
        // k[0] = 1.0, k[1..] = 100.0
        for i in 0..dim {
            k[i] = 1.0;
            for t in 1..seq {
                k[t * dim + i] = 100.0;
            }
        }
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = true;

        let out_causal = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("causal forward");

        config.causal = false;
        let out_no_causal = flash_attention_forward(&q, &k, &v, &config, seq, seq)
            .expect("non-causal forward");

        // First token output should differ significantly between causal/non-causal
        let first_token_causal: f32 = out_causal[..dim].iter().sum();
        let first_token_no_causal: f32 = out_no_causal[..dim].iter().sum();
        assert!(
            (first_token_causal - first_token_no_causal).abs() > 1e-3,
            "causal mask should affect first token output"
        );
    }

    // ── Test 12: Non-square Q/KV sequence lengths ────────────────────────

    #[test]
    fn test_non_square_seq_lens() {
        let seq_q = 3;
        let seq_kv = 7;
        let heads = 2;
        let dim = 8;

        let q = make_tensor(seq_q, heads, dim, 1.0);
        let k = make_tensor(seq_kv, heads, dim, 2.0);
        let v = make_tensor(seq_kv, heads, dim, 3.0);

        let mut config = FlashAttentionConfig::new(dim, heads);
        config.causal = false;
        config.block_size_q = 2;
        config.block_size_kv = 3;

        let out = flash_attention_forward(&q, &k, &v, &config, seq_q, seq_kv)
            .expect("non-square forward");

        assert_eq!(out.len(), seq_q * heads * dim);

        // Also verify against naive per head
        for h in 0..heads {
            let q_h = extract_head(&q, seq_q, heads, dim, h);
            let k_h = extract_head(&k, seq_kv, heads, dim, h);
            let v_h = extract_head(&v, seq_kv, heads, dim, h);
            let naive = naive_attention(&q_h, &k_h, &v_h, config.scale, seq_q, seq_kv, dim, false);
            let flash_h = extract_head(&out, seq_q, heads, dim, h);
            assert_close(&flash_h, &naive, 1e-4, &format!("non-square head {h}"));
        }
    }
}
