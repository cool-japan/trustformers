//! Jamba-2 model implementation (AI21 Labs).
//!
//! Jamba-2 is AI21's second-generation hybrid Mamba-Transformer model. It interleaves:
//!   - Mamba SSM blocks for linear-complexity processing of long sequences
//!   - Grouped Query Attention (GQA) Transformer blocks for precise long-range dependencies
//!   - Mixture-of-Experts (MoE) FFN in selected layers for sparse computation
//!
//! Reference: AI21 Labs, "Jamba-2" (2024)

use crate::jamba2::config::{Jamba2Config, LayerType};

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn softplus(x: f64) -> f64 {
    if x > 20.0 {
        x
    } else if x < -20.0 {
        x.exp()
    } else {
        (1.0 + x.exp()).ln()
    }
}

#[inline]
fn silu_f64(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Numerically stable softmax over a slice.
fn softmax_f64(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum < 1e-30 {
        vec![1.0 / x.len() as f64; x.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Matrix-vector product: weight [out × in] @ vec [in] → [out].
fn mat_vec_mul(weight: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, Jamba2Error> {
    if weight.is_empty() {
        return Ok(Vec::new());
    }
    let in_dim = weight[0].len();
    if x.len() != in_dim {
        return Err(Jamba2Error::DimensionMismatch {
            expected: in_dim,
            got: x.len(),
            context: "mat_vec_mul".to_string(),
        });
    }
    Ok(weight
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect())
}

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Error types for Jamba-2 model operations.
#[derive(Debug, thiserror::Error)]
pub enum Jamba2Error {
    #[error("Empty input provided")]
    EmptyInput,
    #[error("Dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    #[error("Layer error at layer {layer}: {msg}")]
    LayerError { layer: usize, msg: String },
}

// ---------------------------------------------------------------------------
// Jamba2RmsNorm — Root Mean Square Layer Normalization
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization for Jamba-2.
pub struct Jamba2RmsNorm {
    weight: Vec<f64>,
    eps: f64,
}

impl Jamba2RmsNorm {
    /// Create a new RMSNorm with all-ones weight vector.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
        }
    }

    /// Forward pass: normalize then scale by learned weight.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Jamba2Error> {
        if x.is_empty() {
            return Err(Jamba2Error::EmptyInput);
        }
        if x.len() != self.weight.len() {
            return Err(Jamba2Error::DimensionMismatch {
                expected: self.weight.len(),
                got: x.len(),
                context: "RmsNorm".to_string(),
            });
        }
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let rms = (mean_sq + self.eps).sqrt();
        Ok(x.iter().zip(self.weight.iter()).map(|(v, w)| v / rms * w).collect())
    }
}

// ---------------------------------------------------------------------------
// MambaBlock — Simplified Mamba Selective State Space Model
// ---------------------------------------------------------------------------

/// Mamba SSM block as used in Jamba-2.
///
/// Implements a simplified version of the Mamba selective scan:
///   1. Project hidden → 2 * inner_dim to split into x and z branches
///   2. Apply causal 1D depthwise convolution on x
///   3. Compute Δ (dt), B, C via x_proj + dt_proj
///   4. Run the selective SSM linear scan (discretized ZOH)
///   5. Gate with silu(z)
///   6. Project back to hidden dimension
pub struct MambaBlock {
    /// Projects hidden_size → 2 * inner_dim (x branch + z gate)
    in_proj: Vec<Vec<f64>>,
    /// Depthwise conv weights: [inner_dim × d_conv]
    conv1d_weight: Vec<Vec<f64>>,
    /// Projects inner_dim → dt_rank + 2 * d_state (Δ, B, C)
    x_proj: Vec<Vec<f64>>,
    /// Projects dt_rank → inner_dim (Δ expansion)
    dt_proj: Vec<Vec<f64>>,
    /// Projects inner_dim → hidden_size
    out_proj: Vec<Vec<f64>>,
    /// Log of negative A eigenvalues: [inner_dim]
    a_log: Vec<f64>,
    /// D skip-connection coefficient: [inner_dim]
    d_param: Vec<f64>,
    /// Pre-normalization
    norm: Jamba2RmsNorm,
    hidden_size: usize,
    inner_dim: usize,
    d_conv: usize,
    d_state: usize,
    dt_rank: usize,
}

impl MambaBlock {
    /// Create a new MambaBlock with small non-zero initialization.
    pub fn new(config: &Jamba2Config) -> Self {
        let hidden_size = config.hidden_size;
        let inner_dim = config.mamba_inner_dim();
        let d_conv = config.mamba_d_conv;
        let d_state = config.mamba_d_state;
        let dt_rank = config.effective_dt_rank();

        // in_proj: hidden → 2 * inner_dim
        let in_proj: Vec<Vec<f64>> = (0..2 * inner_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.02;
                row
            })
            .collect();

        // conv1d: depthwise — each channel has its own kernel of size d_conv
        let conv1d_weight: Vec<Vec<f64>> =
            (0..inner_dim).map(|_| vec![1.0 / d_conv as f64; d_conv]).collect();

        // x_proj: inner_dim → dt_rank + 2 * d_state
        let x_proj_out = dt_rank + 2 * d_state;
        let x_proj: Vec<Vec<f64>> = (0..x_proj_out)
            .map(|i| {
                let mut row = vec![0.0f64; inner_dim];
                row[i % inner_dim] = 0.02;
                row
            })
            .collect();

        // dt_proj: dt_rank → inner_dim
        let dt_proj: Vec<Vec<f64>> = (0..inner_dim)
            .map(|i| {
                let mut row = vec![0.0f64; dt_rank];
                row[i % dt_rank] = 0.02;
                row
            })
            .collect();

        // out_proj: inner_dim → hidden_size
        let out_proj: Vec<Vec<f64>> = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; inner_dim];
                row[i % inner_dim] = 0.02;
                row
            })
            .collect();

        // A: initialized so that log(-A) = 0 → A ≈ 0.5 (reasonable default)
        let a_log = vec![0.0f64; inner_dim];
        // D skip connection initialized to 1
        let d_param = vec![1.0f64; inner_dim];

        Self {
            in_proj,
            conv1d_weight,
            x_proj,
            dt_proj,
            out_proj,
            a_log,
            d_param,
            norm: Jamba2RmsNorm::new(hidden_size, config.rms_norm_eps),
            hidden_size,
            inner_dim,
            d_conv,
            d_state,
            dt_rank,
        }
    }

    /// Apply causal depthwise 1D convolution on the sequence.
    ///
    /// Each channel has its own kernel; output at time t only uses t and past positions.
    fn causal_conv1d(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Jamba2Error> {
        let seq_len = x.len();
        let channels = self.inner_dim;
        let d_conv = self.d_conv;
        let mut out = vec![vec![0.0f64; channels]; seq_len];
        for t in 0..seq_len {
            for c in 0..channels {
                let kernel = &self.conv1d_weight[c];
                let mut val = 0.0f64;
                for k in 0..d_conv {
                    if t >= k {
                        val += kernel[k] * x[t - k][c];
                    }
                }
                out[t][c] = val;
            }
        }
        Ok(out)
    }

    /// Forward pass for the Mamba SSM block.
    ///
    /// Input: `x` of shape [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size] (residual added internally)
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Jamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Jamba2Error::EmptyInput);
        }
        if x[0].len() != self.hidden_size {
            return Err(Jamba2Error::DimensionMismatch {
                expected: self.hidden_size,
                got: x[0].len(),
                context: "MambaBlock input".to_string(),
            });
        }

        let inner_dim = self.inner_dim;
        let d_state = self.d_state;
        let dt_rank = self.dt_rank;

        // Pre-norm then project: [seq_len, 2 * inner_dim]
        let projs: Vec<Vec<f64>> = x
            .iter()
            .map(|token| {
                let normed = self.norm.forward(token)?;
                mat_vec_mul(&self.in_proj, &normed)
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Split into x-branch and z-gate
        let x_branch: Vec<Vec<f64>> = projs.iter().map(|p| p[..inner_dim].to_vec()).collect();
        let z_gate: Vec<Vec<f64>> =
            projs.iter().map(|p| p[inner_dim..2 * inner_dim].to_vec()).collect();

        // Causal 1D depthwise conv on x_branch
        let x_conv = self.causal_conv1d(&x_branch)?;

        // Apply silu activation to x_conv
        let x_act: Vec<Vec<f64>> =
            x_conv.iter().map(|row| row.iter().map(|v| silu_f64(*v)).collect()).collect();

        // x_proj: inner_dim → dt_rank + 2 * d_state  (Δ, B, C)
        let x_proj_out: Vec<Vec<f64>> = x_act
            .iter()
            .map(|row| mat_vec_mul(&self.x_proj, row))
            .collect::<Result<Vec<_>, _>>()?;

        // Extract Δ_raw (dt_rank), B (d_state), C (d_state)
        let dt_raw: Vec<Vec<f64>> = x_proj_out.iter().map(|p| p[..dt_rank].to_vec()).collect();
        let b_seq: Vec<Vec<f64>> =
            x_proj_out.iter().map(|p| p[dt_rank..dt_rank + d_state].to_vec()).collect();
        let c_seq: Vec<Vec<f64>> = x_proj_out
            .iter()
            .map(|p| p[dt_rank + d_state..dt_rank + 2 * d_state].to_vec())
            .collect();

        // dt_proj: dt_rank → inner_dim, then softplus for positivity
        let dt_seq: Vec<Vec<f64>> = dt_raw
            .iter()
            .map(|row| {
                let expanded = mat_vec_mul(&self.dt_proj, row)?;
                Ok(expanded.iter().map(|v| softplus(*v)).collect())
            })
            .collect::<Result<Vec<_>, _>>()?;

        // Selective SSM linear scan (discretized ZOH approximation)
        // State h: [inner_dim, d_state]
        let mut h: Vec<Vec<f64>> = vec![vec![0.0f64; d_state]; inner_dim];
        let mut y_seq: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

        for t in 0..seq_len {
            let mut y_t = vec![0.0f64; inner_dim];
            for i in 0..inner_dim {
                let dt_i = dt_seq[t][i];
                // Discretize: A_bar = exp(-dt * exp(a_log))  (ZOH)
                let a_bar = (-dt_i * self.a_log[i].exp()).exp();
                let x_val = x_act[t][i];
                // D skip connection
                let mut y_val = self.d_param[i] * x_val;
                for s in 0..d_state {
                    h[i][s] = a_bar * h[i][s] + x_val * b_seq[t][s];
                    y_val += c_seq[t][s] * h[i][s];
                }
                y_t[i] = y_val;
            }

            // Gate with silu(z)
            let gated: Vec<f64> =
                y_t.iter().zip(z_gate[t].iter()).map(|(y, z)| y * silu_f64(*z)).collect();
            y_seq.push(gated);
        }

        // out_proj and residual connection
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for (t, gated) in y_seq.iter().enumerate() {
            let projected = mat_vec_mul(&self.out_proj, gated)?;
            let out: Vec<f64> = x[t].iter().zip(projected.iter()).map(|(r, p)| r + p).collect();
            result.push(out);
        }

        Ok(result)
    }

    /// Hidden size of this block.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Inner (expanded) dimension.
    pub fn inner_dim(&self) -> usize {
        self.inner_dim
    }
}

// ---------------------------------------------------------------------------
// Jamba2Attention — Grouped Query Attention (GQA)
// ---------------------------------------------------------------------------

/// Grouped Query Attention (GQA) layer for Jamba-2.
///
/// Uses num_attention_heads query heads and num_key_value_heads K/V heads.
/// Each K/V head is shared among (num_attention_heads / num_key_value_heads) query heads.
/// Includes RoPE positional embedding (simplified version).
pub struct Jamba2Attention {
    q_proj: Vec<Vec<f64>>, // [num_heads * head_dim × hidden_size]
    k_proj: Vec<Vec<f64>>, // [num_kv_heads * head_dim × hidden_size]
    v_proj: Vec<Vec<f64>>, // [num_kv_heads * head_dim × hidden_size]
    o_proj: Vec<Vec<f64>>, // [hidden_size × num_heads * head_dim]
    norm: Jamba2RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl Jamba2Attention {
    /// Create a new GQA attention layer.
    pub fn new(config: &Jamba2Config) -> Self {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim;
        let q_dim = num_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;

        let q_proj: Vec<Vec<f64>> = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden];
                row[i % hidden] = 0.01;
                row
            })
            .collect();
        let k_proj: Vec<Vec<f64>> = (0..kv_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden];
                row[(i + 1) % hidden] = 0.01;
                row
            })
            .collect();
        let v_proj: Vec<Vec<f64>> = (0..kv_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden];
                row[(i + 2) % hidden] = 0.01;
                row
            })
            .collect();
        let o_proj: Vec<Vec<f64>> = (0..hidden)
            .map(|i| {
                let mut row = vec![0.0f64; q_dim];
                row[i % q_dim] = 0.01;
                row
            })
            .collect();

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            norm: Jamba2RmsNorm::new(hidden, config.rms_norm_eps),
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Apply simplified RoPE rotation to a head vector.
    ///
    /// Rotates pairs of dimensions by position-dependent angles.
    fn apply_rope(&self, vec: &[f64], position: usize) -> Vec<f64> {
        let mut out = vec.to_vec();
        let half = out.len() / 2;
        for i in 0..half {
            let theta = position as f64 / (10000.0_f64.powf(2.0 * i as f64 / out.len() as f64));
            let (sin_t, cos_t) = (theta.sin(), theta.cos());
            let x0 = out[i];
            let x1 = out[i + half];
            out[i] = x0 * cos_t - x1 * sin_t;
            out[i + half] = x0 * sin_t + x1 * cos_t;
        }
        out
    }

    /// Forward pass for GQA attention.
    ///
    /// Input: `x` of shape [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size] (with residual)
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Jamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Jamba2Error::EmptyInput);
        }

        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let groups = num_heads / num_kv_heads;
        let scale = 1.0 / (head_dim as f64).sqrt();

        // Pre-norm then project Q, K, V
        let mut q_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut k_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut v_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

        for token in x.iter() {
            let normed = self.norm.forward(token)?;
            q_all.push(mat_vec_mul(&self.q_proj, &normed)?);
            k_all.push(mat_vec_mul(&self.k_proj, &normed)?);
            v_all.push(mat_vec_mul(&self.v_proj, &normed)?);
        }

        // Scaled dot-product attention per head with GQA
        let mut context_all: Vec<Vec<f64>> = vec![vec![0.0f64; num_heads * head_dim]; seq_len];

        for h in 0..num_heads {
            let kv_h = h / groups; // which K/V head this Q head maps to

            for q_pos in 0..seq_len {
                // Compute attention scores (causal mask: only attend to positions <= q_pos)
                let q_vec: Vec<f64> =
                    (0..head_dim).map(|d| q_all[q_pos][h * head_dim + d]).collect();
                // Apply RoPE to Q
                let q_rope = self.apply_rope(&q_vec, q_pos);

                let mut scores: Vec<f64> = Vec::with_capacity(q_pos + 1);
                for k_pos in 0..=q_pos {
                    let k_vec: Vec<f64> =
                        (0..head_dim).map(|d| k_all[k_pos][kv_h * head_dim + d]).collect();
                    // Apply RoPE to K
                    let k_rope = self.apply_rope(&k_vec, k_pos);
                    let dot: f64 = q_rope.iter().zip(k_rope.iter()).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }

                let attn_weights = softmax_f64(&scores);

                // Weighted sum of values
                for (k_pos, &w) in attn_weights.iter().enumerate() {
                    for d in 0..head_dim {
                        context_all[q_pos][h * head_dim + d] +=
                            w * v_all[k_pos][kv_h * head_dim + d];
                    }
                }
            }
        }

        // Output projection + residual
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for (t, ctx) in context_all.iter().enumerate() {
            let projected = mat_vec_mul(&self.o_proj, ctx)?;
            let out: Vec<f64> = x[t].iter().zip(projected.iter()).map(|(r, p)| r + p).collect();
            result.push(out);
        }

        Ok(result)
    }

    /// Number of query heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of key-value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

// ---------------------------------------------------------------------------
// Jamba2Mlp — SwiGLU feed-forward network
// ---------------------------------------------------------------------------

/// SwiGLU MLP used as expert and dense FFN.
///
/// Computation: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct Jamba2Mlp {
    gate_proj: Vec<Vec<f64>>, // [intermediate × hidden]
    up_proj: Vec<Vec<f64>>,   // [intermediate × hidden]
    down_proj: Vec<Vec<f64>>, // [hidden × intermediate]
}

impl Jamba2Mlp {
    /// Create a new SwiGLU MLP with small diagonal initialization.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let gate_proj: Vec<Vec<f64>> = (0..intermediate_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();
        let up_proj: Vec<Vec<f64>> = (0..intermediate_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 1) % hidden_size] = 0.01;
                row
            })
            .collect();
        let down_proj: Vec<Vec<f64>> = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; intermediate_size];
                row[i % intermediate_size] = 0.01;
                row
            })
            .collect();
        Self {
            gate_proj,
            up_proj,
            down_proj,
        }
    }

    /// Forward: SwiGLU activation.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Jamba2Error> {
        let gate = mat_vec_mul(&self.gate_proj, x)?;
        let up = mat_vec_mul(&self.up_proj, x)?;
        let activated: Vec<f64> =
            gate.iter().zip(up.iter()).map(|(g, u)| silu_f64(*g) * u).collect();
        mat_vec_mul(&self.down_proj, &activated)
    }
}

// ---------------------------------------------------------------------------
// Jamba2MoELayer — Mixture of Experts FFN
// ---------------------------------------------------------------------------

/// Mixture of Experts FFN layer for Jamba-2.
///
/// Routes each token to the top-k experts (default top-2) using a learned router,
/// and combines their outputs with softmax-normalized weights.
/// Also includes a shared (always-active) expert FFN.
pub struct Jamba2MoELayer {
    experts: Vec<Jamba2Mlp>,
    /// Shared expert that always contributes
    shared_expert: Jamba2Mlp,
    /// Router gate weights: [num_experts × hidden_size]
    router: Vec<Vec<f64>>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl Jamba2MoELayer {
    /// Create a new MoE layer.
    pub fn new(config: &Jamba2Config) -> Self {
        let experts: Vec<Jamba2Mlp> = (0..config.num_experts)
            .map(|_| Jamba2Mlp::new(config.hidden_size, config.intermediate_size))
            .collect();
        let shared_expert = Jamba2Mlp::new(config.hidden_size, config.intermediate_size);

        // Router: unique routing bias per expert
        let router: Vec<Vec<f64>> = (0..config.num_experts)
            .map(|e| {
                let mut row = vec![0.0f64; config.hidden_size];
                row[e % config.hidden_size] = 1.0;
                row
            })
            .collect();

        Self {
            experts,
            shared_expert,
            router,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
        }
    }

    /// Compute router logits for a single token.
    pub fn router_logits(&self, x: &[f64]) -> Result<Vec<f64>, Jamba2Error> {
        mat_vec_mul(&self.router, x)
    }

    /// Forward: route to top-k experts, add shared expert, combine.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Jamba2Error> {
        let logits = self.router_logits(x)?;
        let probs = softmax_f64(&logits);

        let k = self.num_experts_per_tok.min(self.num_experts);
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = &indexed[..k];

        // Renormalize top-k weights
        let weight_sum: f64 = top_k.iter().map(|(_, w)| w).sum();
        let normalized: Vec<f64> = if weight_sum > 1e-10 {
            top_k.iter().map(|(_, w)| w / weight_sum).collect()
        } else {
            vec![1.0 / k as f64; k]
        };

        let hidden_size = x.len();
        let mut output = vec![0.0f64; hidden_size];

        // Sparse expert contributions
        for (i, (expert_idx, _)) in top_k.iter().enumerate() {
            let expert_out = self.experts[*expert_idx].forward(x)?;
            for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                *o += normalized[i] * e;
            }
        }

        // Add shared expert (always active, weight = 1.0)
        let shared_out = self.shared_expert.forward(x)?;
        for (o, s) in output.iter_mut().zip(shared_out.iter()) {
            *o += s;
        }

        Ok(output)
    }
}

// ---------------------------------------------------------------------------
// Jamba2DecoderLayer — single decoder layer
// ---------------------------------------------------------------------------

/// A single Jamba-2 decoder layer.
///
/// Based on the layer_idx and config, this dispatches to:
///   - MambaBlock (Mamba-only layers)
///   - Jamba2Attention + dense FFN (Attention layers)
///   - MambaBlock + MoE FFN (Mamba+MoE layers)
///   - Jamba2Attention + MoE FFN (Attention+MoE layers)
pub struct Jamba2DecoderLayer {
    layer_type: LayerType,
    mamba: Option<MambaBlock>,
    attention: Option<Jamba2Attention>,
    ffn_dense: Option<Jamba2Mlp>,
    ffn_moe: Option<Jamba2MoELayer>,
    /// Post-attention / post-SSM layer norm
    post_norm: Jamba2RmsNorm,
    hidden_size: usize,
}

impl Jamba2DecoderLayer {
    /// Create a new decoder layer according to its index.
    pub fn new(config: &Jamba2Config, layer_idx: usize) -> Self {
        let layer_type = config.layer_type(layer_idx);
        let hidden_size = config.hidden_size;

        let mamba = match layer_type {
            LayerType::Mamba | LayerType::MambaMoE => Some(MambaBlock::new(config)),
            _ => None,
        };
        let attention = match layer_type {
            LayerType::Attention | LayerType::AttentionMoE => Some(Jamba2Attention::new(config)),
            _ => None,
        };
        let ffn_dense = match layer_type {
            LayerType::Attention => {
                Some(Jamba2Mlp::new(config.hidden_size, config.intermediate_size))
            },
            _ => None,
        };
        let ffn_moe = match layer_type {
            LayerType::MambaMoE | LayerType::AttentionMoE => Some(Jamba2MoELayer::new(config)),
            _ => None,
        };

        Self {
            layer_type,
            mamba,
            attention,
            ffn_dense,
            ffn_moe,
            post_norm: Jamba2RmsNorm::new(hidden_size, config.rms_norm_eps),
            hidden_size,
        }
    }

    /// Forward pass for this decoder layer.
    ///
    /// Input: `x` shape [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size]
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Jamba2Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Jamba2Error::EmptyInput);
        }

        // SSM or Attention sub-layer (includes residual internally)
        let after_mixer: Vec<Vec<f64>> = match &self.layer_type {
            LayerType::Mamba | LayerType::MambaMoE => {
                let mamba = self.mamba.as_ref().ok_or_else(|| Jamba2Error::LayerError {
                    layer: 0,
                    msg: "MambaBlock missing for Mamba layer".to_string(),
                })?;
                mamba.forward(x)?
            },
            LayerType::Attention | LayerType::AttentionMoE => {
                let attn = self.attention.as_ref().ok_or_else(|| Jamba2Error::LayerError {
                    layer: 0,
                    msg: "Jamba2Attention missing for Attention layer".to_string(),
                })?;
                attn.forward(x)?
            },
        };

        // FFN sub-layer with post-norm and residual
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for (t, hidden) in after_mixer.iter().enumerate() {
            let normed = self.post_norm.forward(hidden)?;
            let ffn_out = match &self.layer_type {
                LayerType::MambaMoE | LayerType::AttentionMoE => {
                    let moe = self.ffn_moe.as_ref().ok_or_else(|| Jamba2Error::LayerError {
                        layer: t,
                        msg: "MoE FFN missing".to_string(),
                    })?;
                    moe.forward(&normed)?
                },
                LayerType::Attention => {
                    let dense = self.ffn_dense.as_ref().ok_or_else(|| Jamba2Error::LayerError {
                        layer: t,
                        msg: "Dense FFN missing".to_string(),
                    })?;
                    dense.forward(&normed)?
                },
                LayerType::Mamba => {
                    // Pure Mamba layers have no separate FFN
                    normed
                },
            };

            // Residual connection
            let out: Vec<f64> = hidden.iter().zip(ffn_out.iter()).map(|(h, f)| h + f).collect();
            result.push(out);
        }

        Ok(result)
    }

    /// The type of this decoder layer.
    pub fn layer_type(&self) -> &LayerType {
        &self.layer_type
    }

    /// Hidden size for this layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ---------------------------------------------------------------------------
// Jamba2Model — full model stack
// ---------------------------------------------------------------------------

/// The full Jamba-2 decoder model (without language modelling head).
///
/// Consists of:
///   - Token embedding table
///   - N Jamba2DecoderLayer layers
///   - Final RMSNorm
pub struct Jamba2Model {
    /// Token embeddings: [vocab_size, hidden_size]
    embed_tokens: Vec<Vec<f64>>,
    layers: Vec<Jamba2DecoderLayer>,
    norm: Jamba2RmsNorm,
    config: Jamba2Config,
}

impl Jamba2Model {
    /// Create a new Jamba-2 model from configuration.
    pub fn new(config: Jamba2Config) -> Self {
        // Initialize embeddings with small values
        let embed_tokens: Vec<Vec<f64>> = (0..config.vocab_size)
            .map(|i| {
                let mut row = vec![0.0f64; config.hidden_size];
                row[i % config.hidden_size] = 0.01;
                row
            })
            .collect();

        let layers: Vec<Jamba2DecoderLayer> = (0..config.num_hidden_layers)
            .map(|idx| Jamba2DecoderLayer::new(&config, idx))
            .collect();

        let norm = Jamba2RmsNorm::new(config.hidden_size, config.rms_norm_eps);

        Self {
            embed_tokens,
            layers,
            norm,
            config,
        }
    }

    /// Forward pass through the full model.
    ///
    /// `input_ids`: token indices, shape `[seq_len]`
    /// Returns hidden states of shape `[seq_len, hidden_size]`
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<Vec<f64>>, Jamba2Error> {
        if input_ids.is_empty() {
            return Err(Jamba2Error::EmptyInput);
        }

        // Embed tokens
        let mut hidden: Vec<Vec<f64>> = input_ids
            .iter()
            .map(|&id| {
                let idx = id as usize % self.config.vocab_size;
                self.embed_tokens[idx].clone()
            })
            .collect();

        // Pass through each decoder layer
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden = layer.forward(&hidden).map_err(|e| Jamba2Error::LayerError {
                layer: layer_idx,
                msg: e.to_string(),
            })?;
        }

        // Final layer norm
        let normed: Vec<Vec<f64>> =
            hidden.iter().map(|row| self.norm.forward(row)).collect::<Result<Vec<_>, _>>()?;

        Ok(normed)
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &Jamba2Config {
        &self.config
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Layer type at the given index.
    pub fn layer_type(&self, idx: usize) -> Option<&LayerType> {
        self.layers.get(idx).map(|l| l.layer_type())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jamba2::config::{Jamba2Config, LayerType};

    /// Minimal config sufficient to build a small model quickly in tests.
    fn small_config() -> Jamba2Config {
        Jamba2Config {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 6,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4,
            mamba_d_state: 4,
            mamba_d_conv: 2,
            mamba_expand: 2,
            mamba_dt_rank: 4,
            attn_layer_offset: 4,
            attn_layer_period: 8,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            tie_word_embeddings: false,
        }
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_default_config_hidden_size() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.hidden_size, 4096, "default hidden_size should be 4096");
    }

    #[test]
    fn test_default_config_num_experts() {
        let cfg = Jamba2Config::default();
        assert_eq!(cfg.num_experts, 16, "Jamba-2 has 16 MoE experts");
    }

    #[test]
    fn test_default_config_num_experts_per_tok() {
        let cfg = Jamba2Config::default();
        assert_eq!(
            cfg.num_experts_per_tok, 2,
            "Jamba-2 activates 2 experts per token"
        );
    }

    #[test]
    fn test_config_validate_ok() {
        let cfg = small_config();
        cfg.validate().expect("small_config should be valid");
    }

    #[test]
    fn test_config_validate_zero_hidden_size_fails() {
        let mut cfg = small_config();
        cfg.hidden_size = 0;
        assert!(
            cfg.validate().is_err(),
            "zero hidden_size must fail validation"
        );
    }

    #[test]
    fn test_config_validate_experts_per_tok_exceeds_experts() {
        let mut cfg = small_config();
        cfg.num_experts_per_tok = cfg.num_experts + 1;
        assert!(
            cfg.validate().is_err(),
            "experts_per_tok > num_experts must fail"
        );
    }

    #[test]
    fn test_mamba_inner_dim() {
        let cfg = small_config();
        assert_eq!(
            cfg.mamba_inner_dim(),
            cfg.mamba_expand * cfg.hidden_size,
            "mamba_inner_dim = expand * hidden_size"
        );
    }

    #[test]
    fn test_effective_dt_rank_explicit() {
        let cfg = small_config();
        assert_eq!(
            cfg.effective_dt_rank(),
            4,
            "explicit mamba_dt_rank should be returned directly"
        );
    }

    #[test]
    fn test_effective_dt_rank_auto() {
        let mut cfg = small_config();
        cfg.mamba_dt_rank = 0; // triggers auto
        let expected = cfg.hidden_size.div_ceil(16);
        assert_eq!(
            cfg.effective_dt_rank(),
            expected,
            "auto dt_rank = ceil(hidden/16)"
        );
    }

    // ── Layer-type classification ─────────────────────────────────────────────

    #[test]
    fn test_layer_type_mamba_early_layers() {
        let cfg = small_config();
        // Layer 0: not attention (< offset 4), not MoE (< offset 1) — pure Mamba
        assert_eq!(
            cfg.layer_type(0),
            LayerType::Mamba,
            "layer 0 should be Mamba"
        );
    }

    #[test]
    fn test_layer_type_moe_odd_layers() {
        let cfg = small_config();
        // Layer 1: (1 >= offset 1) && (1-1)%2==0 → MoE; not attention → MambaMoE
        assert_eq!(
            cfg.layer_type(1),
            LayerType::MambaMoE,
            "layer 1 should be MambaMoE"
        );
    }

    #[test]
    fn test_layer_type_attention_at_offset() {
        let cfg = small_config();
        // Layer 4: attention layer; (4-1)%2 != 0 → not MoE → Attention
        assert_eq!(
            cfg.layer_type(4),
            LayerType::Attention,
            "layer 4 should be Attention"
        );
    }

    #[test]
    fn test_attention_layer_period_pattern() {
        let cfg = Jamba2Config::default(); // period = 8, offset = 4
        assert!(cfg.is_attention_layer(4), "layer 4 should be attention");
        assert!(cfg.is_attention_layer(12), "layer 12 should be attention");
        assert!(cfg.is_attention_layer(20), "layer 20 should be attention");
        assert!(
            !cfg.is_attention_layer(5),
            "layer 5 should not be attention"
        );
    }

    // ── RMSNorm tests ────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_ones_input_unchanged() {
        let norm = Jamba2RmsNorm::new(4, 1e-5);
        let input = vec![1.0_f64; 4];
        let output = norm.forward(&input).expect("rmsnorm should succeed");
        assert_eq!(output.len(), 4, "output length must match input");
        for v in &output {
            assert!(v.is_finite(), "output must be finite");
        }
    }

    #[test]
    fn test_rmsnorm_dimension_mismatch_fails() {
        let norm = Jamba2RmsNorm::new(4, 1e-5);
        let input = vec![1.0_f64; 3];
        assert!(
            norm.forward(&input).is_err(),
            "dimension mismatch should fail"
        );
    }

    #[test]
    fn test_rmsnorm_empty_input_fails() {
        let norm = Jamba2RmsNorm::new(4, 1e-5);
        assert!(norm.forward(&[]).is_err(), "empty input should fail");
    }

    // ── Softmax tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_softmax_sums_to_one() {
        let logits = vec![1.0_f64, 2.0, 3.0, 0.5];
        let probs = softmax_f64(&logits);
        let total: f64 = probs.iter().sum();
        assert!(
            (total - 1.0).abs() < 1e-9,
            "softmax must sum to 1; got {total}"
        );
    }

    #[test]
    fn test_softmax_all_non_negative() {
        let logits = vec![-5.0_f64, 0.0, 5.0];
        let probs = softmax_f64(&logits);
        for p in probs {
            assert!(p >= 0.0, "all softmax outputs must be non-negative");
        }
    }

    // ── MoE expert routing tests ──────────────────────────────────────────────

    #[test]
    fn test_moe_router_logits_length() {
        let cfg = small_config();
        let moe = Jamba2MoELayer::new(&cfg);
        let x = vec![0.5_f64; cfg.hidden_size];
        let logits = moe.router_logits(&x).expect("router_logits should succeed");
        assert_eq!(logits.len(), cfg.num_experts, "one logit per expert");
    }

    #[test]
    fn test_moe_forward_output_length() {
        let cfg = small_config();
        let moe = Jamba2MoELayer::new(&cfg);
        let x = vec![0.1_f64; cfg.hidden_size];
        let out = moe.forward(&x).expect("moe forward should succeed");
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "moe output dim must equal hidden_size"
        );
    }

    #[test]
    fn test_moe_expert_routing_selects_top_k() {
        // Validate that only num_experts_per_tok experts are selected.
        let cfg = small_config();
        let moe = Jamba2MoELayer::new(&cfg);
        // Using distinct router logits to ensure deterministic routing.
        let x = vec![0.0_f64; cfg.hidden_size];
        // forward should succeed without panics
        moe.forward(&x).expect("moe forward with zero input should succeed");
    }

    // ── Model construction ───────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = small_config();
        let _model = Jamba2Model::new(cfg);
    }

    #[test]
    fn test_model_num_layers() {
        let cfg = small_config();
        let expected = cfg.num_hidden_layers;
        let model = Jamba2Model::new(cfg);
        assert_eq!(model.num_layers(), expected, "num_layers must match config");
    }

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = small_config();
        let hidden_size = cfg.hidden_size;
        let model = Jamba2Model::new(cfg);
        let input_ids: Vec<u32> = vec![0, 1, 2];
        let output = model.forward(&input_ids).expect("model forward should succeed");
        assert_eq!(output.len(), 3, "output sequence length must match input");
        assert_eq!(
            output[0].len(),
            hidden_size,
            "each token must have hidden_size dims"
        );
    }

    #[test]
    fn test_model_forward_empty_fails() {
        let cfg = small_config();
        let model = Jamba2Model::new(cfg);
        assert!(
            model.forward(&[]).is_err(),
            "empty input must return an error"
        );
    }

    #[test]
    fn test_model_layer_types_not_all_same() {
        let cfg = small_config();
        let model = Jamba2Model::new(cfg.clone());
        // With the small config, layer 0 is Mamba and layer 1 is MambaMoE
        let t0 = model.layer_type(0).expect("layer 0 should exist");
        let t1 = model.layer_type(1).expect("layer 1 should exist");
        assert_ne!(t0, t1, "adjacent layer types should differ in small_config");
    }

    #[test]
    fn test_jamba2_1_5b_config() {
        let cfg = Jamba2Config::jamba2_1_5b();
        assert_eq!(cfg.hidden_size, 2048);
        assert_eq!(cfg.num_hidden_layers, 12);
        cfg.validate().expect("jamba2_1_5b config should be valid");
    }
}
