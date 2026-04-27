//! Sliding-window attention (Beltagy et al., 2020 — Longformer / Mistral).
//!
//! Each token attends only to a local window of neighbouring tokens, reducing
//! the quadratic O(n²) attention cost to O(n·w) where w is the window size.
//! A Longformer-style extension allows the first N tokens to attend globally.

use std::fmt;

// ── Error type ─────────────────────────────────────────────────────────────

/// Errors returned by sliding-window attention operations.
#[derive(Debug, Clone, PartialEq)]
pub enum SwaError {
    /// Window size is invalid (e.g. zero).
    InvalidWindowSize(String),
    /// Input tensor dimensions are inconsistent.
    DimensionMismatch,
    /// Sequence is shorter than required for the operation.
    SeqLenTooShort {
        /// Minimum acceptable sequence length.
        min: usize,
        /// Actual sequence length.
        got: usize,
    },
}

impl fmt::Display for SwaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindowSize(msg) => write!(f, "invalid window size: {msg}"),
            Self::DimensionMismatch => write!(f, "dimension mismatch in sliding-window attention"),
            Self::SeqLenTooShort { min, got } => {
                write!(f, "sequence too short: min={min}, got={got}")
            }
        }
    }
}

impl std::error::Error for SwaError {}

// ── Configuration ──────────────────────────────────────────────────────────

/// Configuration for sliding-window attention.
#[derive(Debug, Clone)]
pub struct SlidingWindowConfig {
    /// Number of tokens on each side of the current position to attend to.
    /// Default: 512.
    pub window_size: usize,
    /// Attention head dimension.
    pub head_dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Causal variant: each token attends to `[pos-window, pos]`.  Default: `true`.
    pub causal: bool,
    /// Attention scale.  Default: `1 / sqrt(head_dim)`.
    pub scale: f32,
    /// First N tokens use global attention (Longformer-style).  Default: 0.
    pub include_global_tokens: usize,
}

impl SlidingWindowConfig {
    /// Construct a config with sensible defaults.
    pub fn new(head_dim: usize, num_heads: usize) -> Self {
        let scale = 1.0 / (head_dim as f32).sqrt();
        Self {
            window_size: 512,
            head_dim,
            num_heads,
            causal: true,
            scale,
            include_global_tokens: 0,
        }
    }
}

// ── Attention span helper ──────────────────────────────────────────────────

/// Compute the `[start, end)` range of positions attended by `pos`.
///
/// - Non-causal: `[pos - window, pos + window + 1)` clamped to `[0, seq_len)`.
/// - Causal:     `[pos - window, pos + 1)` clamped to `[0, seq_len)`.
///
/// Global tokens (index < `include_global_tokens`) always attend the full
/// sequence (end = seq_len).
pub fn compute_attention_span(
    pos: usize,
    seq_len: usize,
    window: usize,
    causal: bool,
) -> (usize, usize) {
    let start = pos.saturating_sub(window);
    let end = if causal {
        (pos + 1).min(seq_len)
    } else {
        (pos + window + 1).min(seq_len)
    };
    (start, end)
}

// ── Statistics ─────────────────────────────────────────────────────────────

/// Statistics describing the attention pattern for a given configuration.
#[derive(Debug, Clone)]
pub struct AttentionPatternStats {
    /// Average number of tokens each query attends to.
    pub mean_attention_span: f32,
    /// Maximum number of tokens any query attends to.
    pub max_attention_span: usize,
    /// Number of global tokens (from config).
    pub num_global_tokens: usize,
    /// Approximate fraction of the full attention matrix that is computed.
    /// Computed as `effective_window / seq_len`.
    pub effective_memory_reduction: f32,
}

/// Compute attention pattern statistics for the given config and sequence length.
pub fn compute_pattern_stats(config: &SlidingWindowConfig, seq_len: usize) -> AttentionPatternStats {
    let mut total_span: usize = 0;
    let mut max_span: usize = 0;

    for pos in 0..seq_len {
        let span = if pos < config.include_global_tokens {
            // Global token: attends to everything
            seq_len
        } else {
            let (start, end) = compute_attention_span(pos, seq_len, config.window_size, config.causal);
            // Also count global tokens that are always visible
            let global_extra = config.include_global_tokens.min(start); // global tokens before window
            (end - start) + global_extra
        };

        total_span += span;
        if span > max_span {
            max_span = span;
        }
    }

    let mean_span = if seq_len > 0 {
        total_span as f32 / seq_len as f32
    } else {
        0.0
    };

    let effective_window = if config.causal {
        config.window_size + 1
    } else {
        2 * config.window_size + 1
    };

    let memory_reduction = if seq_len > 0 {
        (effective_window.min(seq_len) as f32) / (seq_len as f32)
    } else {
        1.0
    };

    AttentionPatternStats {
        mean_attention_span: mean_span,
        max_attention_span: max_span,
        num_global_tokens: config.include_global_tokens,
        effective_memory_reduction: memory_reduction,
    }
}

// ── Core forward pass ──────────────────────────────────────────────────────

/// Sliding-window attention forward pass.
///
/// ## Layout
/// - `q`, `k`, `v`: `[seq_len, num_heads, head_dim]`  (row-major)
/// - Output:         `[seq_len, num_heads, head_dim]`
///
/// For each query position `i`:
/// - If `i < include_global_tokens`: attends to **all** positions.
/// - Otherwise: attends to `[max(0, i-window), min(seq_len, i+window+1))` (non-causal)
///   or `[max(0, i-window), i+1)` (causal), plus any global tokens that fall
///   outside that window.
///
/// # Errors
/// Returns [`SwaError`] on invalid configuration or dimension mismatches.
pub fn sliding_window_attention(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    config: &SlidingWindowConfig,
    seq_len: usize,
) -> Result<Vec<f32>, SwaError> {
    if config.window_size == 0 {
        return Err(SwaError::InvalidWindowSize(
            "window_size must be > 0".to_string(),
        ));
    }
    if seq_len == 0 {
        return Err(SwaError::SeqLenTooShort { min: 1, got: 0 });
    }

    let h = config.num_heads;
    let d = config.head_dim;
    let expected = seq_len * h * d;

    if q.len() != expected || k.len() != expected || v.len() != expected {
        return Err(SwaError::DimensionMismatch);
    }

    let scale = config.scale;
    let global_n = config.include_global_tokens.min(seq_len);

    let mut output = vec![0.0_f32; expected];

    for head in 0..h {
        for pos in 0..seq_len {
            // Determine the set of attended positions
            let (win_start, win_end) = if pos < global_n {
                // Global token: full attention
                (0, seq_len)
            } else {
                compute_attention_span(pos, seq_len, config.window_size, config.causal)
            };

            // Collect all attended key indices: window range ∪ global tokens
            // Global tokens outside the window are included separately
            let mut attended: Vec<usize> = (win_start..win_end).collect();

            if pos >= global_n && global_n > 0 {
                // Add global tokens that are before win_start
                for g in 0..global_n {
                    if g < win_start {
                        attended.push(g);
                    }
                }
                // Sort to keep order (for determinism)
                attended.sort_unstable();
                attended.dedup();
            }

            let n_attended = attended.len();
            if n_attended == 0 {
                continue;
            }

            // Compute Q[pos] dot K[j] for each j in attended → scores
            let q_offset = pos * h * d + head * d;
            let q_vec = &q[q_offset..q_offset + d];

            let mut scores: Vec<f32> = attended
                .iter()
                .map(|&j| {
                    let k_offset = j * h * d + head * d;
                    let k_vec = &k[k_offset..k_offset + d];
                    q_vec.iter().zip(k_vec.iter()).map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
                })
                .collect();

            // Softmax over scores
            let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let mut sum_exp = 0.0_f32;
            for s in scores.iter_mut() {
                *s = (*s - max_s).exp();
                sum_exp += *s;
            }
            if sum_exp > 0.0 {
                for s in scores.iter_mut() {
                    *s /= sum_exp;
                }
            }

            // Output = sum_j scores[j] * V[j]
            let out_offset = pos * h * d + head * d;
            for (idx, &j) in attended.iter().enumerate() {
                let v_offset = j * h * d + head * d;
                let weight = scores[idx];
                for dd in 0..d {
                    output[out_offset + dd] += weight * v[v_offset + dd];
                }
            }
        }
    }

    Ok(output)
}

// ── Tests ──────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(seq: usize, heads: usize, dim: usize, seed: f32) -> Vec<f32> {
        let n = seq * heads * dim;
        (0..n)
            .map(|i| ((i as f32 * seed * 0.1).sin() * 0.5 + 0.5) * 0.2)
            .collect()
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, label: &str) {
        assert_eq!(a.len(), b.len(), "{label}: length mismatch");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!(
                (x - y).abs() < tol,
                "{label}[{i}]: {x} vs {y} (diff={})",
                (x - y).abs()
            );
        }
    }

    // Reference full-attention for comparison (single-head or multi-head)
    fn full_attention(q: &[f32], k: &[f32], v: &[f32], seq: usize, h: usize, d: usize, scale: f32) -> Vec<f32> {
        let mut out = vec![0.0_f32; seq * h * d];
        for head in 0..h {
            for i in 0..seq {
                let q_off = i * h * d + head * d;
                let q_vec = &q[q_off..q_off + d];

                let mut scores: Vec<f32> = (0..seq)
                    .map(|j| {
                        let k_off = j * h * d + head * d;
                        q_vec.iter().zip(k[k_off..k_off + d].iter())
                            .map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
                    })
                    .collect();

                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum_exp = 0.0_f32;
                for s in scores.iter_mut() {
                    *s = (*s - max_s).exp();
                    sum_exp += *s;
                }
                if sum_exp > 0.0 {
                    for s in scores.iter_mut() {
                        *s /= sum_exp;
                    }
                }

                let out_off = i * h * d + head * d;
                for j in 0..seq {
                    let v_off = j * h * d + head * d;
                    for dd in 0..d {
                        out[out_off + dd] += scores[j] * v[v_off + dd];
                    }
                }
            }
        }
        out
    }

    // ── Test 1: compute_attention_span causal ──────────────────────────

    #[test]
    fn test_attention_span_causal() {
        // window=3, seq=10, causal
        assert_eq!(compute_attention_span(0, 10, 3, true), (0, 1));
        assert_eq!(compute_attention_span(2, 10, 3, true), (0, 3));
        assert_eq!(compute_attention_span(5, 10, 3, true), (2, 6));
        assert_eq!(compute_attention_span(9, 10, 3, true), (6, 10));
    }

    // ── Test 2: compute_attention_span non-causal ──────────────────────

    #[test]
    fn test_attention_span_non_causal() {
        // window=2, seq=8, non-causal
        assert_eq!(compute_attention_span(0, 8, 2, false), (0, 3));
        assert_eq!(compute_attention_span(3, 8, 2, false), (1, 6));
        assert_eq!(compute_attention_span(7, 8, 2, false), (5, 8));
    }

    // ── Test 3: Output shape ──────────────────────────────────────────

    #[test]
    fn test_output_shape() {
        let seq = 10;
        let heads = 2;
        let dim = 8;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = 3;

        let out = sliding_window_attention(&q, &k, &v, &config, seq)
            .expect("output shape test");

        assert_eq!(out.len(), seq * heads * dim);
    }

    // ── Test 4: Large window == full attention (non-causal) ───────────

    #[test]
    fn test_large_window_equals_full_attention_non_causal() {
        let seq = 6;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.1);
        let k = make_tensor(seq, heads, dim, 2.2);
        let v = make_tensor(seq, heads, dim, 3.3);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = seq; // window >= seq_len → full attention
        config.causal = false;

        let swa_out = sliding_window_attention(&q, &k, &v, &config, seq)
            .expect("large window non-causal");
        let full_out = full_attention(&q, &k, &v, seq, heads, dim, config.scale);

        assert_close(&swa_out, &full_out, 1e-5, "large window == full (non-causal)");
    }

    // ── Test 5: Large window == full attention (causal) ───────────────

    #[test]
    fn test_large_window_equals_full_attention_causal() {
        let seq = 6;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.5);
        let k = make_tensor(seq, heads, dim, 2.5);
        let v = make_tensor(seq, heads, dim, 3.5);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = seq; // window >= seq_len
        config.causal = true;

        let swa_out = sliding_window_attention(&q, &k, &v, &config, seq)
            .expect("large window causal");

        // Causal full attention
        let scale = config.scale;
        let mut full_out = vec![0.0_f32; seq * heads * dim];
        for head in 0..heads {
            for i in 0..seq {
                let q_off = i * heads * dim + head * dim;
                let q_vec = &q[q_off..q_off + dim];
                let mut scores: Vec<f32> = (0..=i)
                    .map(|j| {
                        let k_off = j * heads * dim + head * dim;
                        q_vec.iter().zip(k[k_off..k_off + dim].iter())
                            .map(|(&qi, &ki)| qi * ki).sum::<f32>() * scale
                    })
                    .collect();
                let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut s_exp = 0.0_f32;
                for s in scores.iter_mut() { *s = (*s - max_s).exp(); s_exp += *s; }
                if s_exp > 0.0 { for s in scores.iter_mut() { *s /= s_exp; } }
                let out_off = i * heads * dim + head * dim;
                for (j, &w) in scores.iter().enumerate() {
                    let v_off = j * heads * dim + head * dim;
                    for dd in 0..dim { full_out[out_off + dd] += w * v[v_off + dd]; }
                }
            }
        }

        assert_close(&swa_out, &full_out, 1e-5, "large window == full (causal)");
    }

    // ── Test 6: Small window limits attention (causal) ────────────────

    #[test]
    fn test_small_window_causal() {
        let seq = 10;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = 1;
        config.causal = true;

        let out = sliding_window_attention(&q, &k, &v, &config, seq)
            .expect("small window causal");
        assert_eq!(out.len(), seq * heads * dim);

        // All outputs must be finite
        for &val in &out {
            assert!(val.is_finite(), "output must be finite");
        }
    }

    // ── Test 7: Global tokens (Longformer-style) ─────────────────────

    #[test]
    fn test_global_tokens() {
        let seq = 8;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = 1; // small window
        config.causal = false;
        config.include_global_tokens = 2; // first 2 tokens are global

        let out = sliding_window_attention(&q, &k, &v, &config, seq)
            .expect("global tokens");
        assert_eq!(out.len(), seq * heads * dim);

        // Global tokens (pos 0,1) should attend the full sequence:
        // their output should match full non-causal attention for those positions
        let full_out = full_attention(&q, &k, &v, seq, heads, dim, config.scale);

        for pos in 0..2 {
            let off = pos * heads * dim;
            assert_close(
                &out[off..off + dim],
                &full_out[off..off + dim],
                1e-5,
                &format!("global token {pos}"),
            );
        }
    }

    // ── Test 8: Memory reduction formula ─────────────────────────────

    #[test]
    fn test_memory_reduction_formula() {
        let seq = 100;
        let mut config = SlidingWindowConfig::new(64, 8);
        config.window_size = 10;
        config.causal = true; // effective window = 11

        let stats = compute_pattern_stats(&config, seq);

        let expected = 11.0_f32 / 100.0_f32;
        assert!(
            (stats.effective_memory_reduction - expected).abs() < 1e-5,
            "memory reduction: {} vs {}",
            stats.effective_memory_reduction,
            expected
        );
    }

    // ── Test 9: Pattern stats struct ─────────────────────────────────

    #[test]
    fn test_pattern_stats() {
        let seq = 20;
        let mut config = SlidingWindowConfig::new(64, 4);
        config.window_size = 3;
        config.causal = true;
        config.include_global_tokens = 1;

        let stats = compute_pattern_stats(&config, seq);

        assert_eq!(stats.num_global_tokens, 1);
        assert!(stats.mean_attention_span > 0.0);
        assert!(stats.max_attention_span >= 4); // window=3 causal → up to 4 tokens
        assert!(stats.effective_memory_reduction > 0.0);
        assert!(stats.effective_memory_reduction <= 1.0);
    }

    // ── Test 10: Error — zero window size ─────────────────────────────

    #[test]
    fn test_error_zero_window() {
        let seq = 8;
        let heads = 1;
        let dim = 4;
        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config = SlidingWindowConfig::new(dim, heads);
        config.window_size = 0;

        let err = sliding_window_attention(&q, &k, &v, &config, seq);
        assert!(matches!(err, Err(SwaError::InvalidWindowSize(_))));
    }

    // ── Test 11: Error — dimension mismatch ──────────────────────────

    #[test]
    fn test_error_dimension_mismatch() {
        let seq = 8;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = vec![0.0_f32; 5]; // wrong size
        let v = make_tensor(seq, heads, dim, 3.0);

        let config = SlidingWindowConfig::new(dim, heads);

        let err = sliding_window_attention(&q, &k, &v, &config, seq);
        assert!(matches!(err, Err(SwaError::DimensionMismatch)));
    }

    // ── Test 12: Error — zero seq len ─────────────────────────────────

    #[test]
    fn test_error_zero_seq_len() {
        let config = SlidingWindowConfig::new(4, 1);
        let err = sliding_window_attention(&[], &[], &[], &config, 0);
        assert!(matches!(err, Err(SwaError::SeqLenTooShort { .. })));
    }

    // ── Test 13: Different window sizes produce different outputs ─────

    #[test]
    fn test_different_window_sizes() {
        let seq = 12;
        let heads = 1;
        let dim = 4;

        let q = make_tensor(seq, heads, dim, 1.0);
        let k = make_tensor(seq, heads, dim, 2.0);
        let v = make_tensor(seq, heads, dim, 3.0);

        let mut config1 = SlidingWindowConfig::new(dim, heads);
        config1.window_size = 1;
        config1.causal = false;

        let mut config2 = SlidingWindowConfig::new(dim, heads);
        config2.window_size = 4;
        config2.causal = false;

        let out1 = sliding_window_attention(&q, &k, &v, &config1, seq).expect("window=1");
        let out2 = sliding_window_attention(&q, &k, &v, &config2, seq).expect("window=4");

        // Middle tokens should differ
        let mid = (seq / 2) * heads * dim;
        let diff: f32 = out1[mid..mid + dim]
            .iter()
            .zip(out2[mid..mid + dim].iter())
            .map(|(&a, &b)| (a - b).abs())
            .sum();
        assert!(diff > 1e-5, "different windows should produce different outputs");
    }
}
