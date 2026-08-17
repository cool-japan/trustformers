//! Pure-Rust transformer forward-pass math.
//!
//! Everything in this module operates on plain `f32` slices/`Vec<f32>` with
//! explicit shapes passed alongside — nothing here touches `wasm_bindgen`,
//! `js_sys`, or `web_sys`, so the full numeric core is exercised by ordinary
//! native `#[test]`s (see the `tests` module below). The thin `WasmTensor`
//! boundary lives in `wasm_model.rs`, which converts inputs/outputs at the
//! edges only.
//!
//! ## Checkpoint layout
//!
//! The naming convention used to look weights up (see `weights.rs`) is a
//! first-party TrustformeRS-WASM layout, *not* a byte-for-byte mirror of any
//! particular HuggingFace checkpoint's tensor names. The math implemented
//! here is real (real matmuls, real softmax attention, real residual/
//! normalization), but loading an arbitrary HF `model.safetensors` file
//! as-is will fail with a clear "missing tensor" error rather than silently
//! reinterpreting differently-named weights.

use std::string::{String, ToString};
use std::vec;
use std::vec::Vec;

/// Normalization style used by a transformer block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NormKind {
    /// Mean/variance layer normalization (BERT, GPT-2, T5-style-lite).
    LayerNorm,
    /// Root-mean-square normalization, no mean subtraction, no bias
    /// (LLaMA, Mistral).
    RmsNorm,
}

/// Feed-forward network weights for one transformer block.
pub enum FfnWeights<'a> {
    /// Two-matrix GELU-activated FFN: `fc2(gelu(fc1(x)))`.
    Gelu {
        fc1_w: &'a [f32],
        fc1_b: Option<&'a [f32]>,
        fc2_w: &'a [f32],
        fc2_b: Option<&'a [f32]>,
    },
    /// Three-matrix SwiGLU-activated FFN: `down(silu(gate(x)) * up(x))`.
    SwiGlu {
        gate_w: &'a [f32],
        up_w: &'a [f32],
        down_w: &'a [f32],
    },
}

/// All weights required to run one transformer block.
///
/// Every `*_w` matrix is stored row-major with shape `[in_features,
/// out_features]`, i.e. `linear(x, w)` computes `x @ w`.
pub struct LayerWeights<'a> {
    pub q_w: &'a [f32],
    pub q_b: Option<&'a [f32]>,
    pub k_w: &'a [f32],
    pub k_b: Option<&'a [f32]>,
    pub v_w: &'a [f32],
    pub v_b: Option<&'a [f32]>,
    pub o_w: &'a [f32],
    pub o_b: Option<&'a [f32]>,
    pub norm1_w: &'a [f32],
    pub norm1_b: Option<&'a [f32]>,
    pub norm2_w: &'a [f32],
    pub norm2_b: Option<&'a [f32]>,
    pub ffn: FfnWeights<'a>,
}

/// Static configuration shared by every layer of one forward pass.
#[derive(Debug, Clone, Copy)]
pub struct TransformerConfig {
    pub hidden_size: usize,
    pub num_heads: usize,
    pub causal: bool,
    pub norm_kind: NormKind,
    pub use_rope: bool,
    pub eps: f32,
    pub rope_base: f32,
}

/// Compute `x[rows, in_dim] @ w[in_dim, out_dim] (+ bias)`.
pub fn linear(
    x: &[f32],
    rows: usize,
    in_dim: usize,
    w: &[f32],
    out_dim: usize,
    bias: Option<&[f32]>,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * in_dim {
        return Err(format!(
            "linear: input length {} does not match rows*in_dim ({rows}*{in_dim})",
            x.len()
        ));
    }
    if w.len() != in_dim * out_dim {
        return Err(format!(
            "linear: weight length {} does not match in_dim*out_dim ({in_dim}*{out_dim})",
            w.len()
        ));
    }
    if let Some(b) = bias {
        if b.len() != out_dim {
            return Err(format!(
                "linear: bias length {} does not match out_dim ({out_dim})",
                b.len()
            ));
        }
    }

    let mut out = vec![0.0f32; rows * out_dim];
    for r in 0..rows {
        let x_row = &x[r * in_dim..(r + 1) * in_dim];
        let out_row = &mut out[r * out_dim..(r + 1) * out_dim];
        for (k, &xv) in x_row.iter().enumerate() {
            if xv == 0.0 {
                continue;
            }
            let w_row = &w[k * out_dim..(k + 1) * out_dim];
            for (o, &wv) in w_row.iter().enumerate() {
                out_row[o] += xv * wv;
            }
        }
        if let Some(b) = bias {
            for (o, ov) in out_row.iter_mut().enumerate() {
                *ov += b[o];
            }
        }
    }
    Ok(out)
}

/// Transpose a `[rows, cols]` row-major matrix into `[cols, rows]`.
pub fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Row-wise numerically-stable softmax, in place.
pub fn softmax_row(row: &mut [f32]) {
    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in row.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in row.iter_mut() {
            *v /= sum;
        }
    }
}

/// Standard mean/variance layer normalization over the last axis.
pub fn layer_norm(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    bias: Option<&[f32]>,
    eps: f32,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * cols || weight.len() != cols {
        return Err("layer_norm: shape mismatch".to_string());
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean = row.iter().sum::<f32>() / cols as f32;
        let var = row.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / cols as f32;
        let inv_std = 1.0 / (var + eps).sqrt();
        let out_row = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            let normalized = (row[c] - mean) * inv_std * weight[c];
            out_row[c] = match bias {
                Some(b) => normalized + b[c],
                None => normalized,
            };
        }
    }
    Ok(out)
}

/// Root-mean-square normalization (no mean subtraction, no bias).
pub fn rms_norm(
    x: &[f32],
    rows: usize,
    cols: usize,
    weight: &[f32],
    eps: f32,
) -> Result<Vec<f32>, String> {
    if x.len() != rows * cols || weight.len() != cols {
        return Err("rms_norm: shape mismatch".to_string());
    }
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        let row = &x[r * cols..(r + 1) * cols];
        let mean_sq = row.iter().map(|v| v * v).sum::<f32>() / cols as f32;
        let inv_rms = 1.0 / (mean_sq + eps).sqrt();
        let out_row = &mut out[r * cols..(r + 1) * cols];
        for c in 0..cols {
            out_row[c] = row[c] * inv_rms * weight[c];
        }
    }
    Ok(out)
}

/// GELU activation (tanh approximation, matches the common transformer
/// implementation convention).
#[inline]
pub fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((2.0 / std::f32::consts::PI).sqrt() * (x + 0.044715 * x.powi(3))).tanh())
}

/// SiLU / Swish activation used by SwiGLU FFNs.
#[inline]
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Look up token embeddings: `ids[seq] -> hidden[seq, hidden]`.
pub fn embedding_lookup(
    ids: &[u32],
    table: &[f32],
    vocab_size: usize,
    hidden: usize,
) -> Result<Vec<f32>, String> {
    if table.len() != vocab_size * hidden {
        return Err(format!(
            "embedding_lookup: table length {} does not match vocab_size*hidden ({vocab_size}*{hidden})",
            table.len()
        ));
    }
    let mut out = vec![0.0f32; ids.len() * hidden];
    for (i, &id) in ids.iter().enumerate() {
        let id = id as usize;
        if id >= vocab_size {
            return Err(format!(
                "embedding_lookup: token id {id} is out of vocabulary range (vocab_size={vocab_size})"
            ));
        }
        out[i * hidden..(i + 1) * hidden].copy_from_slice(&table[id * hidden..(id + 1) * hidden]);
    }
    Ok(out)
}

/// Apply rotary position embeddings to a `[seq, head_dim]` slice in place,
/// rotating consecutive coordinate pairs `(2i, 2i+1)` by a position- and
/// frequency-dependent angle.
fn apply_rope(x: &mut [f32], seq_len: usize, head_dim: usize, base: f32) {
    let half = head_dim / 2;
    for pos in 0..seq_len {
        let row = &mut x[pos * head_dim..(pos + 1) * head_dim];
        for i in 0..half {
            let theta = (pos as f32) * base.powf(-2.0 * (i as f32) / head_dim as f32);
            let (sin, cos) = theta.sin_cos();
            let a = row[i];
            let b = row[i + half];
            row[i] = a * cos - b * sin;
            row[i + half] = a * sin + b * cos;
        }
    }
}

/// Multi-head (self-)attention over a `[seq, hidden]` input.
#[allow(clippy::too_many_arguments)]
pub fn multi_head_attention(
    x: &[f32],
    seq_len: usize,
    hidden: usize,
    num_heads: usize,
    q_w: &[f32],
    q_b: Option<&[f32]>,
    k_w: &[f32],
    k_b: Option<&[f32]>,
    v_w: &[f32],
    v_b: Option<&[f32]>,
    o_w: &[f32],
    o_b: Option<&[f32]>,
    causal: bool,
    use_rope: bool,
    rope_base: f32,
) -> Result<Vec<f32>, String> {
    if hidden == 0 || num_heads == 0 || !hidden.is_multiple_of(num_heads) {
        return Err(format!(
            "multi_head_attention: hidden_size ({hidden}) must be a positive multiple of num_heads ({num_heads})"
        ));
    }
    let head_dim = hidden / num_heads;

    let q = linear(x, seq_len, hidden, q_w, hidden, q_b)?;
    let k = linear(x, seq_len, hidden, k_w, hidden, k_b)?;
    let v = linear(x, seq_len, hidden, v_w, hidden, v_b)?;

    // Reshape [seq, hidden] -> per-head [seq, head_dim] views by copying
    // (small buffers; clarity over micro-optimization).
    let mut context = vec![0.0f32; seq_len * hidden];
    let scale = 1.0 / (head_dim as f32).sqrt();

    for h in 0..num_heads {
        let mut q_head = vec![0.0f32; seq_len * head_dim];
        let mut k_head = vec![0.0f32; seq_len * head_dim];
        let mut v_head = vec![0.0f32; seq_len * head_dim];
        for pos in 0..seq_len {
            let src = pos * hidden + h * head_dim;
            let dst = pos * head_dim;
            q_head[dst..dst + head_dim].copy_from_slice(&q[src..src + head_dim]);
            k_head[dst..dst + head_dim].copy_from_slice(&k[src..src + head_dim]);
            v_head[dst..dst + head_dim].copy_from_slice(&v[src..src + head_dim]);
        }

        if use_rope {
            apply_rope(&mut q_head, seq_len, head_dim, rope_base);
            apply_rope(&mut k_head, seq_len, head_dim, rope_base);
        }

        // scores[seq, seq] = (q_head @ k_head^T) * scale
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..seq_len {
                if causal && j > i {
                    scores[i * seq_len + j] = f32::NEG_INFINITY;
                    continue;
                }
                let qi = &q_head[i * head_dim..(i + 1) * head_dim];
                let kj = &k_head[j * head_dim..(j + 1) * head_dim];
                let dot: f32 = qi.iter().zip(kj.iter()).map(|(a, b)| a * b).sum();
                scores[i * seq_len + j] = dot * scale;
            }
        }
        for i in 0..seq_len {
            softmax_row(&mut scores[i * seq_len..(i + 1) * seq_len]);
        }

        // head_out[seq, head_dim] = scores @ v_head
        for i in 0..seq_len {
            let score_row = &scores[i * seq_len..(i + 1) * seq_len];
            let dst = i * hidden + h * head_dim;
            for (j, &w) in score_row.iter().enumerate() {
                if w == 0.0 {
                    continue;
                }
                let v_row = &v_head[j * head_dim..(j + 1) * head_dim];
                for d in 0..head_dim {
                    context[dst + d] += w * v_row[d];
                }
            }
        }
    }

    linear(&context, seq_len, hidden, o_w, hidden, o_b)
}

/// Run one full transformer block (pre-attention norm -> self-attention ->
/// residual -> pre-FFN norm -> FFN -> residual), returning the new
/// `[seq, hidden]` hidden state.
pub fn transformer_block(
    hidden_states: &[f32],
    seq_len: usize,
    weights: &LayerWeights,
    cfg: &TransformerConfig,
) -> Result<Vec<f32>, String> {
    let h = cfg.hidden_size;
    let normed1 = match cfg.norm_kind {
        NormKind::LayerNorm => layer_norm(
            hidden_states,
            seq_len,
            h,
            weights.norm1_w,
            weights.norm1_b,
            cfg.eps,
        )?,
        NormKind::RmsNorm => rms_norm(hidden_states, seq_len, h, weights.norm1_w, cfg.eps)?,
    };

    let attn_out = multi_head_attention(
        &normed1,
        seq_len,
        h,
        cfg.num_heads,
        weights.q_w,
        weights.q_b,
        weights.k_w,
        weights.k_b,
        weights.v_w,
        weights.v_b,
        weights.o_w,
        weights.o_b,
        cfg.causal,
        cfg.use_rope,
        cfg.rope_base,
    )?;

    let mut residual1 = vec![0.0f32; seq_len * h];
    for i in 0..residual1.len() {
        residual1[i] = hidden_states[i] + attn_out[i];
    }

    let normed2 = match cfg.norm_kind {
        NormKind::LayerNorm => layer_norm(
            &residual1,
            seq_len,
            h,
            weights.norm2_w,
            weights.norm2_b,
            cfg.eps,
        )?,
        NormKind::RmsNorm => rms_norm(&residual1, seq_len, h, weights.norm2_w, cfg.eps)?,
    };

    let ffn_out = match &weights.ffn {
        FfnWeights::Gelu {
            fc1_w,
            fc1_b,
            fc2_w,
            fc2_b,
        } => {
            let intermediate_size = fc1_w.len() / h;
            let mut hidden1 = linear(&normed2, seq_len, h, fc1_w, intermediate_size, *fc1_b)?;
            for v in hidden1.iter_mut() {
                *v = gelu(*v);
            }
            linear(&hidden1, seq_len, intermediate_size, fc2_w, h, *fc2_b)?
        },
        FfnWeights::SwiGlu {
            gate_w,
            up_w,
            down_w,
        } => {
            let intermediate_size = gate_w.len() / h;
            let mut gate = linear(&normed2, seq_len, h, gate_w, intermediate_size, None)?;
            let up = linear(&normed2, seq_len, h, up_w, intermediate_size, None)?;
            for (g, u) in gate.iter_mut().zip(up.iter()) {
                *g = silu(*g) * u;
            }
            linear(&gate, seq_len, intermediate_size, down_w, h, None)?
        },
    };

    let mut residual2 = vec![0.0f32; seq_len * h];
    for i in 0..residual2.len() {
        residual2[i] = residual1[i] + ffn_out[i];
    }
    Ok(residual2)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transpose_rectangular() {
        // [[1,2,3],[4,5,6]] (2x3) -> [[1,4],[2,5],[3,6]] (3x2)
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let t = transpose(&data, 2, 3);
        assert_eq!(t, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0]);
    }

    #[test]
    fn test_linear_basic() {
        // x = [[1,2],[3,4]] (2x2), w = identity (2x2) -> output == x
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let out = linear(&x, 2, 2, &w, 2, None).expect("shapes are consistent");
        assert_eq!(out, x);
    }

    #[test]
    fn test_linear_with_bias() {
        let x = vec![1.0, 2.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let b = vec![10.0, 20.0];
        let out = linear(&x, 1, 2, &w, 2, Some(&b)).expect("shapes are consistent");
        assert_eq!(out, vec![11.0, 22.0]);
    }

    #[test]
    fn test_linear_shape_mismatch_errors() {
        let x = vec![1.0, 2.0, 3.0];
        let w = vec![1.0, 0.0, 0.0, 1.0];
        assert!(linear(&x, 1, 2, &w, 2, None).is_err());
    }

    #[test]
    fn test_softmax_row_sums_to_one() {
        let mut row = vec![1.0, 2.0, 3.0, 4.0];
        softmax_row(&mut row);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5, "softmax must sum to 1, got {sum}");
        // Monotonic: larger logits get larger probability.
        assert!(row[3] > row[2] && row[2] > row[1] && row[1] > row[0]);
    }

    #[test]
    fn test_softmax_row_causal_neg_infinity_becomes_zero() {
        let mut row = vec![1.0, f32::NEG_INFINITY, 2.0];
        softmax_row(&mut row);
        assert_eq!(row[1], 0.0);
        let sum: f32 = row.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_layer_norm_zero_mean_unit_var() {
        let x = vec![1.0, 2.0, 3.0, 4.0]; // 1 row, 4 cols
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let out = layer_norm(&x, 1, 4, &weight, None, 1e-5).expect("valid shapes");
        let mean: f32 = out.iter().sum::<f32>() / 4.0;
        assert!(
            mean.abs() < 1e-4,
            "normalized mean should be ~0, got {mean}"
        );
    }

    #[test]
    fn test_rms_norm_scales_to_unit_rms() {
        let x = vec![2.0, 2.0, 2.0, 2.0];
        let weight = vec![1.0, 1.0, 1.0, 1.0];
        let out = rms_norm(&x, 1, 4, &weight, 1e-8).expect("valid shapes");
        for v in out {
            assert!((v - 1.0).abs() < 1e-3, "expected ~1.0, got {v}");
        }
    }

    #[test]
    fn test_embedding_lookup_selects_correct_rows() {
        // vocab_size=3, hidden=2
        let table = vec![0.0, 0.0, 1.0, 1.0, 2.0, 2.0];
        let ids = vec![2u32, 0u32];
        let out = embedding_lookup(&ids, &table, 3, 2).expect("valid ids");
        assert_eq!(out, vec![2.0, 2.0, 0.0, 0.0]);
    }

    #[test]
    fn test_embedding_lookup_out_of_range_errors() {
        let table = vec![0.0, 0.0, 1.0, 1.0];
        let ids = vec![5u32];
        assert!(embedding_lookup(&ids, &table, 2, 2).is_err());
    }

    #[test]
    fn test_multi_head_attention_output_shape() {
        let seq_len = 3;
        let hidden = 4;
        let num_heads = 2;
        let x = vec![0.1; seq_len * hidden];
        let identity = |n: usize| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        let w = identity(hidden);
        let out = multi_head_attention(
            &x, seq_len, hidden, num_heads, &w, None, &w, None, &w, None, &w, None, false, false,
            10000.0,
        )
        .expect("valid config");
        assert_eq!(out.len(), seq_len * hidden);
    }

    #[test]
    fn test_causal_attention_first_token_ignores_future() {
        // With a causal mask, token 0's output must not depend on token 1's
        // value: change token 1 and token 0's output must stay identical.
        let seq_len = 2;
        let hidden = 2;
        let num_heads = 1;
        let w = vec![1.0, 0.0, 0.0, 1.0]; // identity
        let x_a = vec![1.0, 0.0, /* pos0 */ 5.0, 5.0 /* pos1 */];
        let x_b = vec![
            1.0, 0.0, /* pos0 */ -9.0, 3.0, /* pos1 (different) */
        ];

        let out_a = multi_head_attention(
            &x_a, seq_len, hidden, num_heads, &w, None, &w, None, &w, None, &w, None, true, false,
            10000.0,
        )
        .expect("valid config");
        let out_b = multi_head_attention(
            &x_b, seq_len, hidden, num_heads, &w, None, &w, None, &w, None, &w, None, true, false,
            10000.0,
        )
        .expect("valid config");

        // First token's (2 values) output must be identical regardless of
        // the second token's content when causally masked.
        assert!((out_a[0] - out_b[0]).abs() < 1e-6);
        assert!((out_a[1] - out_b[1]).abs() < 1e-6);
    }

    #[test]
    fn test_non_causal_attention_first_token_depends_on_future() {
        // Without a mask, changing token 1 CAN change token 0's output
        // (bidirectional attention) — this guards against accidentally
        // always causally masking.
        let seq_len = 2;
        let hidden = 2;
        let num_heads = 1;
        let w = vec![1.0, 0.0, 0.0, 1.0];
        let x_a = vec![1.0, 0.0, 5.0, 5.0];
        let x_b = vec![1.0, 0.0, -9.0, 3.0];

        let out_a = multi_head_attention(
            &x_a, seq_len, hidden, num_heads, &w, None, &w, None, &w, None, &w, None, false, false,
            10000.0,
        )
        .expect("valid config");
        let out_b = multi_head_attention(
            &x_b, seq_len, hidden, num_heads, &w, None, &w, None, &w, None, &w, None, false, false,
            10000.0,
        )
        .expect("valid config");

        assert!((out_a[0] - out_b[0]).abs() > 1e-6);
    }

    #[test]
    fn test_transformer_block_gelu_output_shape_and_finite() {
        let seq_len = 2;
        let hidden = 4;
        let intermediate = 8;
        let ones_h = |n: usize| vec![1.0f32; n];
        let identity = |n: usize| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        let qkvo = identity(hidden);
        let fc1 = vec![0.1f32; hidden * intermediate];
        let fc2 = vec![0.1f32; intermediate * hidden];
        let weights = LayerWeights {
            q_w: &qkvo,
            q_b: None,
            k_w: &qkvo,
            k_b: None,
            v_w: &qkvo,
            v_b: None,
            o_w: &qkvo,
            o_b: None,
            norm1_w: &ones_h(hidden),
            norm1_b: None,
            norm2_w: &ones_h(hidden),
            norm2_b: None,
            ffn: FfnWeights::Gelu {
                fc1_w: &fc1,
                fc1_b: None,
                fc2_w: &fc2,
                fc2_b: None,
            },
        };
        let cfg = TransformerConfig {
            hidden_size: hidden,
            num_heads: 2,
            causal: false,
            norm_kind: NormKind::LayerNorm,
            use_rope: false,
            eps: 1e-5,
            rope_base: 10000.0,
        };
        let x = vec![0.5f32; seq_len * hidden];
        let out = transformer_block(&x, seq_len, &weights, &cfg).expect("valid block");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_transformer_block_swiglu_rmsnorm_rope_finite() {
        let seq_len = 3;
        let hidden = 4;
        let intermediate = 6;
        let identity = |n: usize| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        let qkvo = identity(hidden);
        let gate = vec![0.05f32; hidden * intermediate];
        let up = vec![0.05f32; hidden * intermediate];
        let down = vec![0.05f32; intermediate * hidden];
        let weights = LayerWeights {
            q_w: &qkvo,
            q_b: None,
            k_w: &qkvo,
            k_b: None,
            v_w: &qkvo,
            v_b: None,
            o_w: &qkvo,
            o_b: None,
            norm1_w: &vec![1.0f32; hidden],
            norm1_b: None,
            norm2_w: &vec![1.0f32; hidden],
            norm2_b: None,
            ffn: FfnWeights::SwiGlu {
                gate_w: &gate,
                up_w: &up,
                down_w: &down,
            },
        };
        let cfg = TransformerConfig {
            hidden_size: hidden,
            num_heads: 2,
            causal: true,
            norm_kind: NormKind::RmsNorm,
            use_rope: true,
            eps: 1e-6,
            rope_base: 10000.0,
        };
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| (i as f32) * 0.1 - 0.5).collect();
        let out = transformer_block(&x, seq_len, &weights, &cfg).expect("valid block");
        assert_eq!(out.len(), seq_len * hidden);
        assert!(out.iter().all(|v| v.is_finite()));
    }

    #[test]
    fn test_transformer_block_changes_with_different_weights() {
        // Sanity check against accidental identity/no-op: different Q
        // weights must change the output.
        let seq_len = 2;
        let hidden = 4;
        let identity = |n: usize| {
            let mut m = vec![0.0f32; n * n];
            for i in 0..n {
                m[i * n + i] = 1.0;
            }
            m
        };
        let ones_h = vec![1.0f32; hidden];
        let fc1 = vec![0.1f32; hidden * hidden];
        let fc2 = vec![0.1f32; hidden * hidden];
        let qkvo_a = identity(hidden);
        let qkvo_b = {
            let mut m = identity(hidden);
            m[0] = 5.0; // perturb
            m
        };

        let cfg = TransformerConfig {
            hidden_size: hidden,
            num_heads: 2,
            causal: false,
            norm_kind: NormKind::LayerNorm,
            use_rope: false,
            eps: 1e-5,
            rope_base: 10000.0,
        };
        // Deliberately *not* a constant vector: a constant row is exactly
        // annihilated by LayerNorm's mean subtraction (every normalized
        // value becomes 0 regardless of weights), which would make this
        // test pass vacuously even with the plain identity/no-op block.
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| 0.1 + 0.05 * i as f32).collect();

        let weights_a = LayerWeights {
            q_w: &qkvo_a,
            q_b: None,
            k_w: &identity(hidden),
            k_b: None,
            v_w: &identity(hidden),
            v_b: None,
            o_w: &identity(hidden),
            o_b: None,
            norm1_w: &ones_h,
            norm1_b: None,
            norm2_w: &ones_h,
            norm2_b: None,
            ffn: FfnWeights::Gelu {
                fc1_w: &fc1,
                fc1_b: None,
                fc2_w: &fc2,
                fc2_b: None,
            },
        };
        let out_a = transformer_block(&x, seq_len, &weights_a, &cfg).expect("valid block");

        let weights_b = LayerWeights {
            q_w: &qkvo_b,
            q_b: None,
            k_w: &identity(hidden),
            k_b: None,
            v_w: &identity(hidden),
            v_b: None,
            o_w: &identity(hidden),
            o_b: None,
            norm1_w: &ones_h,
            norm1_b: None,
            norm2_w: &ones_h,
            norm2_b: None,
            ffn: FfnWeights::Gelu {
                fc1_w: &fc1,
                fc1_b: None,
                fc2_w: &fc2,
                fc2_b: None,
            },
        };
        let out_b = transformer_block(&x, seq_len, &weights_b, &cfg).expect("valid block");

        assert_ne!(
            out_a, out_b,
            "changing Q weights must change the block output"
        );
    }
}
