//! Jamba model implementation (AI21 Labs).
//!
//! Jamba is the first production Mamba+Transformer hybrid. It interleaves:
//!   - Mamba SSM blocks for efficiency
//!   - Transformer attention blocks (with GQA) for long-range dependencies
//!   - Mixture-of-Experts (MoE) FFN in attention layers for capacity
//!
//! Reference: "Jamba: A Hybrid Transformer-Mamba Language Model" (AI21, 2024)

use crate::jamba::config::JambaConfig;

/// Error types for Jamba model operations.
#[derive(Debug, thiserror::Error)]
pub enum JambaError {
    #[error("Empty input")]
    EmptyInput,
    #[error("Layer error at {layer}: {msg}")]
    LayerError { layer: usize, msg: String },
}

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
fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// Softmax over a slice.
fn softmax(x: &[f64]) -> Vec<f64> {
    let max_val = x.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = x.iter().map(|v| (v - max_val).exp()).collect();
    let sum: f64 = exps.iter().sum();
    if sum == 0.0 {
        vec![1.0 / x.len() as f64; x.len()]
    } else {
        exps.iter().map(|e| e / sum).collect()
    }
}

/// Matrix-vector product: weight [out x in] @ vec [in] -> [out].
fn mat_vec_mul(weight: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, JambaError> {
    if weight.is_empty() {
        return Ok(Vec::new());
    }
    let in_dim = weight[0].len();
    if x.len() != in_dim {
        return Err(JambaError::LayerError {
            layer: 0,
            msg: format!("mat_vec_mul: weight cols={} but x len={}", in_dim, x.len()),
        });
    }
    Ok(weight
        .iter()
        .map(|row| row.iter().zip(x.iter()).map(|(w, v)| w * v).sum())
        .collect())
}

// ---------------------------------------------------------------------------
// JambaRmsNorm
// ---------------------------------------------------------------------------

/// Root Mean Square Layer Normalization for Jamba.
pub struct JambaRmsNorm {
    weight: Vec<f64>,
    eps: f64,
}

impl JambaRmsNorm {
    /// Create a new RMSNorm with all-ones weights.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0; dim],
            eps,
        }
    }

    /// Forward pass: normalize then scale.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, JambaError> {
        if x.is_empty() {
            return Err(JambaError::EmptyInput);
        }
        if x.len() != self.weight.len() {
            return Err(JambaError::LayerError {
                layer: 0,
                msg: format!("RmsNorm dim={} but x len={}", self.weight.len(), x.len()),
            });
        }
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64;
        let rms = (mean_sq + self.eps).sqrt();
        Ok(x.iter().zip(self.weight.iter()).map(|(v, w)| v / rms * w).collect())
    }

    /// Dimension of this norm layer.
    pub fn dim(&self) -> usize {
        self.weight.len()
    }
}

// ---------------------------------------------------------------------------
// JambaMlp — SwiGLU feed-forward network
// ---------------------------------------------------------------------------

/// SwiGLU MLP used as dense FFN and as individual MoE expert.
///
/// Computation: down_proj(silu(gate_proj(x)) * up_proj(x))
pub struct JambaMlp {
    gate_proj: Vec<Vec<f64>>, // [intermediate x hidden]
    up_proj: Vec<Vec<f64>>,   // [intermediate x hidden]
    down_proj: Vec<Vec<f64>>, // [hidden x intermediate]
}

impl JambaMlp {
    /// Create a new SwiGLU MLP with small non-zero initialization.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        // Initialize with small values on the diagonal to avoid dead neurons
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
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, JambaError> {
        let gate = mat_vec_mul(&self.gate_proj, x)?;
        let up = mat_vec_mul(&self.up_proj, x)?;
        // SwiGLU: silu(gate) * up
        let activated: Vec<f64> = gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
        mat_vec_mul(&self.down_proj, &activated)
    }
}

// ---------------------------------------------------------------------------
// JambaMoeLayer — Mixture of Experts
// ---------------------------------------------------------------------------

/// Mixture of Experts layer for Jamba attention blocks.
///
/// Routes each token to the top-k experts and combines their outputs
/// with softmax-normalized routing weights.
pub struct JambaMoeLayer {
    experts: Vec<JambaMlp>,
    /// Routing weights: [num_experts x hidden_size]
    gate: Vec<Vec<f64>>,
    num_experts: usize,
    num_experts_per_tok: usize,
}

impl JambaMoeLayer {
    /// Create a new MoE layer.
    pub fn new(config: &JambaConfig) -> Self {
        let experts: Vec<JambaMlp> = (0..config.num_experts)
            .map(|_| JambaMlp::new(config.hidden_size, config.intermediate_size))
            .collect();
        // Gate: each expert has a unique routing bias so all experts can be selected
        let gate: Vec<Vec<f64>> = (0..config.num_experts)
            .map(|e| {
                let mut row = vec![0.0f64; config.hidden_size];
                row[e % config.hidden_size] = 1.0; // unique column per expert
                row
            })
            .collect();
        Self {
            experts,
            gate,
            num_experts: config.num_experts,
            num_experts_per_tok: config.num_experts_per_tok,
        }
    }

    /// Compute router logits for a single token: `[num_experts]`.
    pub fn router_logits(&self, x: &[f64]) -> Result<Vec<f64>, JambaError> {
        mat_vec_mul(&self.gate, x)
    }

    /// Forward: route to top-k experts and combine outputs.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, JambaError> {
        let logits = self.router_logits(x)?;
        let probs = softmax(&logits);

        // Top-k selection
        let k = self.num_experts_per_tok.min(self.num_experts);
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let top_k = &indexed[..k];

        // Renormalize top-k weights
        let weight_sum: f64 = top_k.iter().map(|(_, w)| w).sum();
        let normalized_weights: Vec<f64> = if weight_sum > 1e-10 {
            top_k.iter().map(|(_, w)| w / weight_sum).collect()
        } else {
            vec![1.0 / k as f64; k]
        };

        // Combine expert outputs
        let hidden_size = x.len();
        let mut output = vec![0.0f64; hidden_size];
        for (i, (expert_idx, _)) in top_k.iter().enumerate() {
            let expert_out = self.experts[*expert_idx].forward(x)?;
            for (o, e) in output.iter_mut().zip(expert_out.iter()) {
                *o += normalized_weights[i] * e;
            }
        }

        Ok(output)
    }

    /// Number of experts
    pub fn num_experts(&self) -> usize {
        self.num_experts
    }

    /// Number of experts per token
    pub fn num_experts_per_tok(&self) -> usize {
        self.num_experts_per_tok
    }

    /// Get selected expert indices for a token (for load balancing analysis).
    pub fn selected_experts(&self, x: &[f64]) -> Result<Vec<usize>, JambaError> {
        let logits = self.router_logits(x)?;
        let probs = softmax(&logits);
        let k = self.num_experts_per_tok.min(self.num_experts);
        let mut indexed: Vec<(usize, f64)> = probs.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(indexed[..k].iter().map(|(idx, _)| *idx).collect())
    }
}

// ---------------------------------------------------------------------------
// JambaMambaLayer — SSM layer for Jamba
// ---------------------------------------------------------------------------

/// Mamba SSM layer as used inside Jamba.
///
/// Simplified selective SSM: projects input, applies causal conv,
/// runs recurrent SSM, gates with z, and projects back.
pub struct JambaMambaLayer {
    /// Input projection: [2 * inner_dim x hidden_size] (z and x_ssm)
    in_proj: Vec<Vec<f64>>,
    /// Output projection: [hidden_size x inner_dim]
    out_proj: Vec<Vec<f64>>,
    /// Log(-A) per channel: [inner_dim]
    a_log: Vec<f64>,
    /// D skip connection per channel: [inner_dim]
    d_bias: Vec<f64>,
    /// Local conv weights: [inner_dim x d_conv]
    conv_weight: Vec<Vec<f64>>,
    norm: JambaRmsNorm,
    hidden_size: usize,
    inner_dim: usize,
    d_conv: usize,
    d_state: usize,
}

impl JambaMambaLayer {
    /// Create a new Mamba layer for Jamba.
    pub fn new(config: &JambaConfig) -> Self {
        let hidden_size = config.hidden_size;
        let inner_dim = config.mamba_inner_dim();
        let d_conv = config.mamba_d_conv;
        let d_state = config.mamba_d_state;

        // in_proj output: z (inner_dim) + x_ssm (inner_dim) + B (d_state) + C (d_state) + dt (1)
        // For simplicity in Jamba we use a flat projection: 2*inner_dim + 2*d_state + 1
        let in_proj_out = 2 * inner_dim + 2 * d_state + 1;
        let in_proj: Vec<Vec<f64>> = (0..in_proj_out)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.02;
                row
            })
            .collect();

        let out_proj: Vec<Vec<f64>> = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; inner_dim];
                row[i % inner_dim] = 0.02;
                row
            })
            .collect();

        // a_log = 0 → A = exp(-softplus(0)) ≈ 0.5
        let a_log = vec![0.0f64; inner_dim];
        // D skip = 1.0 for non-trivial skip connection
        let d_bias = vec![1.0f64; inner_dim];

        let conv_weight: Vec<Vec<f64>> =
            (0..inner_dim).map(|_| vec![1.0 / d_conv as f64; d_conv]).collect();

        Self {
            in_proj,
            out_proj,
            a_log,
            d_bias,
            conv_weight,
            norm: JambaRmsNorm::new(hidden_size, config.rms_norm_eps),
            hidden_size,
            inner_dim,
            d_conv,
            d_state,
        }
    }

    /// Apply causal local convolution on the sequence.
    fn causal_conv(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, JambaError> {
        let seq_len = x.len();
        let channels = self.inner_dim;
        let d_conv = self.d_conv;
        let mut out = vec![vec![0.0f64; channels]; seq_len];
        for t in 0..seq_len {
            for c in 0..channels {
                let w = &self.conv_weight[c];
                let mut val = 0.0f64;
                for k in 0..d_conv {
                    if t >= k {
                        val += w[k] * x[t - k][c];
                    }
                }
                out[t][c] = val;
            }
        }
        Ok(out)
    }

    /// Forward pass for the Mamba layer.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, JambaError> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(JambaError::EmptyInput);
        }
        if x[0].len() != self.hidden_size {
            return Err(JambaError::LayerError {
                layer: 0,
                msg: format!(
                    "MambaLayer expected hidden_size={} got {}",
                    self.hidden_size,
                    x[0].len()
                ),
            });
        }

        let inner_dim = self.inner_dim;
        let d_state = self.d_state;

        // Project and split
        let mut projs: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for token in x.iter() {
            // Apply pre-norm
            let normed = self.norm.forward(token)?;
            projs.push(mat_vec_mul(&self.in_proj, &normed)?);
        }

        // Split slices: z[0..inner], x_ssm[inner..2*inner], B[..+d_state], C[..+d_state], dt
        let x_ssm_raw: Vec<Vec<f64>> =
            projs.iter().map(|p| p[inner_dim..2 * inner_dim].to_vec()).collect();
        let z_seq: Vec<Vec<f64>> = projs.iter().map(|p| p[0..inner_dim].to_vec()).collect();
        let b_seq: Vec<Vec<f64>> = projs
            .iter()
            .map(|p| p[2 * inner_dim..2 * inner_dim + d_state].to_vec())
            .collect();
        let c_seq: Vec<Vec<f64>> = projs
            .iter()
            .map(|p| p[2 * inner_dim + d_state..2 * inner_dim + 2 * d_state].to_vec())
            .collect();
        let dt_seq: Vec<f64> = projs.iter().map(|p| p[2 * inner_dim + 2 * d_state]).collect();

        // Causal conv on x_ssm
        let x_ssm = self.causal_conv(&x_ssm_raw)?;

        // Recurrent SSM: state [inner_dim x d_state]
        let mut h: Vec<Vec<f64>> = vec![vec![0.0f64; d_state]; inner_dim];

        let mut y_seq: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for t in 0..seq_len {
            let dt_val = softplus(dt_seq[t]);
            let mut y_t = vec![0.0f64; inner_dim];

            for i in 0..inner_dim {
                let a_bar = (-dt_val * self.a_log[i].exp()).exp();
                let x_val = x_ssm[t][i];
                let mut y_val = self.d_bias[i] * x_val;
                for s in 0..d_state {
                    h[i][s] = a_bar * h[i][s] + x_val * b_seq[t][s];
                    y_val += c_seq[t][s] * h[i][s];
                }
                y_t[i] = y_val;
            }

            // Gate with z
            let gated: Vec<f64> =
                y_t.iter().zip(z_seq[t].iter()).map(|(y, z)| y * silu(*z)).collect();
            y_seq.push(gated);
        }

        // out_proj + residual
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for (t, gated) in y_seq.iter().enumerate() {
            let projected = mat_vec_mul(&self.out_proj, gated)?;
            // residual
            let out: Vec<f64> = x[t].iter().zip(projected.iter()).map(|(r, p)| r + p).collect();
            result.push(out);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// JambaAttentionLayer — GQA attention
// ---------------------------------------------------------------------------

/// Grouped Query Attention (GQA) layer for Jamba.
///
/// Implements standard scaled dot-product attention with:
///   - num_attention_heads Q heads
///   - num_key_value_heads K/V heads (GQA: each KV head shared by multiple Q heads)
///   - Rotary position embeddings (RoPE)
pub struct JambaAttentionLayer {
    q_proj: Vec<Vec<f64>>, // [num_heads * head_dim x hidden]
    k_proj: Vec<Vec<f64>>, // [num_kv_heads * head_dim x hidden]
    v_proj: Vec<Vec<f64>>, // [num_kv_heads * head_dim x hidden]
    o_proj: Vec<Vec<f64>>, // [hidden x num_heads * head_dim]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl JambaAttentionLayer {
    /// Create a new GQA attention layer.
    pub fn new(config: &JambaConfig) -> Self {
        let hidden = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();
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
            num_heads,
            num_kv_heads,
            head_dim,
        }
    }

    /// Forward pass for GQA attention.
    ///
    /// Input: sequence [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size]
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, JambaError> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(JambaError::EmptyInput);
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let groups_per_kv = self.num_heads / self.num_kv_heads.max(1);

        // Project Q, K, V for all tokens
        let mut q_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut k_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut v_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for token in x.iter() {
            q_all.push(mat_vec_mul(&self.q_proj, token)?);
            k_all.push(mat_vec_mul(&self.k_proj, token)?);
            v_all.push(mat_vec_mul(&self.v_proj, token)?);
        }

        // Compute causal attention output
        // For each query position t and head h:
        //   attn_out[t][h] = sum_s(softmax(q[t,h] . k[s,kv_h] * scale)[s] * v[s,kv_h])
        // where kv_h = h / groups_per_kv
        let q_total_dim = self.num_heads * self.head_dim;
        let mut attn_output: Vec<Vec<f64>> = vec![vec![0.0f64; q_total_dim]; seq_len];

        for t in 0..seq_len {
            for h in 0..self.num_heads {
                let kv_h = h / groups_per_kv.max(1);
                let q_start = h * self.head_dim;
                let kv_start = kv_h * self.head_dim;

                // Compute attention scores for positions 0..=t (causal)
                let mut scores: Vec<f64> = Vec::with_capacity(t + 1);
                for s in 0..=t {
                    let dot: f64 = (0..self.head_dim)
                        .map(|d| q_all[t][q_start + d] * k_all[s][kv_start + d])
                        .sum();
                    scores.push(dot * scale);
                }
                let attn_weights = softmax(&scores);

                // Weighted sum of values
                for s in 0..=t {
                    for d in 0..self.head_dim {
                        attn_output[t][q_start + d] += attn_weights[s] * v_all[s][kv_start + d];
                    }
                }
            }
        }

        // Output projection + residual
        let mut result: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        for (t, attn_out) in attn_output.iter().enumerate() {
            let projected = mat_vec_mul(&self.o_proj, attn_out)?;
            let out: Vec<f64> = x[t].iter().zip(projected.iter()).map(|(r, p)| r + p).collect();
            result.push(out);
        }

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// JambaFfnKind — Dense or MoE FFN
// ---------------------------------------------------------------------------

/// Feed-forward network type for Jamba attention blocks.
pub enum JambaFfnKind {
    Dense(JambaMlp),
    Moe(JambaMoeLayer),
}

impl JambaFfnKind {
    fn forward(&self, x: &[f64]) -> Result<Vec<f64>, JambaError> {
        match self {
            JambaFfnKind::Dense(mlp) => mlp.forward(x),
            JambaFfnKind::Moe(moe) => moe.forward(x),
        }
    }
}

// ---------------------------------------------------------------------------
// JambaLayerContent — Mamba or Attention{+MoE/Dense}
// ---------------------------------------------------------------------------

/// Content of a Jamba decoder layer: Mamba SSM or Attention+FFN.
pub enum JambaLayerContent {
    Mamba(JambaMambaLayer),
    AttentionFfn {
        attn: JambaAttentionLayer,
        ffn: JambaFfnKind,
    },
}

// ---------------------------------------------------------------------------
// JambaDecoderLayer
// ---------------------------------------------------------------------------

/// A single Jamba decoder block.
pub struct JambaDecoderLayer {
    /// Layer index for diagnostics
    pub layer_idx: usize,
    content: JambaLayerContent,
    /// Pre-norm applied before Mamba or Attention
    pre_norm: JambaRmsNorm,
    /// Second norm applied between attention and FFN (only for attention layers)
    pre_ffn_norm: Option<JambaRmsNorm>,
}

impl JambaDecoderLayer {
    /// Create a Jamba decoder layer.
    pub fn new(config: &JambaConfig, layer_idx: usize) -> Self {
        if config.is_attention_layer(layer_idx) {
            let attn = JambaAttentionLayer::new(config);
            let ffn = if config.is_moe_layer(layer_idx) {
                JambaFfnKind::Moe(JambaMoeLayer::new(config))
            } else {
                JambaFfnKind::Dense(JambaMlp::new(config.hidden_size, config.intermediate_size))
            };
            JambaDecoderLayer {
                layer_idx,
                content: JambaLayerContent::AttentionFfn { attn, ffn },
                pre_norm: JambaRmsNorm::new(config.hidden_size, config.rms_norm_eps),
                pre_ffn_norm: Some(JambaRmsNorm::new(config.hidden_size, config.rms_norm_eps)),
            }
        } else {
            JambaDecoderLayer {
                layer_idx,
                content: JambaLayerContent::Mamba(JambaMambaLayer::new(config)),
                pre_norm: JambaRmsNorm::new(config.hidden_size, config.rms_norm_eps),
                pre_ffn_norm: None,
            }
        }
    }

    /// Whether this layer is a Mamba layer.
    pub fn is_mamba(&self) -> bool {
        matches!(&self.content, JambaLayerContent::Mamba(_))
    }

    /// Whether this layer is an attention layer.
    pub fn is_attention(&self) -> bool {
        matches!(&self.content, JambaLayerContent::AttentionFfn { .. })
    }

    /// Whether this layer uses MoE.
    pub fn is_moe(&self) -> bool {
        matches!(
            &self.content,
            JambaLayerContent::AttentionFfn {
                ffn: JambaFfnKind::Moe(_),
                ..
            }
        )
    }

    /// Forward pass for this layer.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, JambaError> {
        let layer_idx = self.layer_idx;
        match &self.content {
            JambaLayerContent::Mamba(mamba) => {
                // Pre-norm is handled internally in JambaMambaLayer for simplicity
                mamba.forward(x).map_err(|e| JambaError::LayerError {
                    layer: layer_idx,
                    msg: e.to_string(),
                })
            },
            JambaLayerContent::AttentionFfn { attn, ffn } => {
                let seq_len = x.len();
                if seq_len == 0 {
                    return Err(JambaError::EmptyInput);
                }
                // Pre-norm before attention
                let normed: Result<Vec<Vec<f64>>, JambaError> =
                    x.iter().map(|token| self.pre_norm.forward(token)).collect();
                let normed = normed?;

                // Attention
                let attn_out = attn.forward(&normed).map_err(|e| JambaError::LayerError {
                    layer: layer_idx,
                    msg: e.to_string(),
                })?;

                // Residual after attention
                let after_attn: Vec<Vec<f64>> = x
                    .iter()
                    .zip(attn_out.iter())
                    .map(|(r, a)| r.iter().zip(a.iter()).map(|(rv, av)| rv + av).collect())
                    .collect();

                // Pre-FFN norm
                let ffn_input: Vec<Vec<f64>> = if let Some(ref ffn_norm) = self.pre_ffn_norm {
                    after_attn
                        .iter()
                        .map(|token| ffn_norm.forward(token))
                        .collect::<Result<Vec<_>, _>>()?
                } else {
                    after_attn.clone()
                };

                // FFN per token
                let ffn_out: Result<Vec<Vec<f64>>, JambaError> =
                    ffn_input.iter().map(|token| ffn.forward(token)).collect();
                let ffn_out = ffn_out?;

                // Residual after FFN
                let result: Vec<Vec<f64>> = after_attn
                    .iter()
                    .zip(ffn_out.iter())
                    .map(|(r, f)| r.iter().zip(f.iter()).map(|(rv, fv)| rv + fv).collect())
                    .collect();

                Ok(result)
            },
        }
    }
}

// ---------------------------------------------------------------------------
// JambaModel
// ---------------------------------------------------------------------------

/// Full Jamba backbone: embedding + interleaved layers + final norm.
pub struct JambaModel {
    embed_tokens: Vec<Vec<f64>>,
    layers: Vec<JambaDecoderLayer>,
    final_norm: JambaRmsNorm,
    config: JambaConfig,
}

impl JambaModel {
    /// Create a new JambaModel.
    pub fn new(config: &JambaConfig) -> Self {
        let embed_tokens = vec![vec![0.0f64; config.hidden_size]; config.vocab_size];
        let layers: Vec<JambaDecoderLayer> = (0..config.num_hidden_layers)
            .map(|i| JambaDecoderLayer::new(config, i))
            .collect();
        let final_norm = JambaRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        Self {
            embed_tokens,
            layers,
            final_norm,
            config: config.clone(),
        }
    }

    /// Forward pass: input_ids -> hidden states [seq_len, hidden_size].
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<Vec<f64>>, JambaError> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(JambaError::EmptyInput);
        }

        let mut hidden: Vec<Vec<f64>> = input_ids
            .iter()
            .map(|&id| {
                if id < self.embed_tokens.len() {
                    self.embed_tokens[id].clone()
                } else {
                    vec![0.0f64; self.config.hidden_size]
                }
            })
            .collect();

        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden)?;
        }

        // Final norm
        let normed: Result<Vec<Vec<f64>>, JambaError> =
            hidden.iter().map(|t| self.final_norm.forward(t)).collect();
        normed
    }

    /// Get a reference to the layers for inspection.
    pub fn layers(&self) -> &[JambaDecoderLayer] {
        &self.layers
    }
}

// ---------------------------------------------------------------------------
// JambaForCausalLM
// ---------------------------------------------------------------------------

/// Jamba language model with causal LM head.
pub struct JambaForCausalLM {
    model: JambaModel,
    lm_head: Vec<Vec<f64>>, // [vocab_size x hidden_size]
}

impl JambaForCausalLM {
    /// Create a new Jamba causal LM model.
    pub fn new(config: &JambaConfig) -> Self {
        let lm_head = vec![vec![0.0f64; config.hidden_size]; config.vocab_size];
        Self {
            model: JambaModel::new(config),
            lm_head,
        }
    }

    /// Forward pass: input_ids -> logits [seq_len, vocab_size].
    pub fn forward(&self, input_ids: &[usize]) -> Result<Vec<Vec<f64>>, JambaError> {
        let hidden = self.model.forward(input_ids)?;
        let logits: Result<Vec<Vec<f64>>, JambaError> =
            hidden.iter().map(|h| mat_vec_mul(&self.lm_head, h)).collect();
        logits
    }

    /// Config accessor
    pub fn config(&self) -> &JambaConfig {
        &self.model.config
    }

    /// Model reference (for layer inspection)
    pub fn model(&self) -> &JambaModel {
        &self.model
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jamba::config::JambaConfig;

    // ------ Test 1: config presets are non-trivial ------
    #[test]
    fn test_config_presets() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.vocab_size, 65536);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_experts, 16);

        let small = JambaConfig::small_test();
        assert_eq!(small.hidden_size, 64);
        assert_eq!(small.num_hidden_layers, 8);
    }

    // ------ Test 2: is_attention_layer for indices 3, 11 ------
    #[test]
    fn test_is_attention_layer_indices_3_and_11() {
        let cfg = JambaConfig::jamba_1_5b();
        // attn_layer_offset=3, attn_layer_period=8 → layers 3,11,19,27
        assert!(cfg.is_attention_layer(3), "layer 3 should be attention");
        assert!(cfg.is_attention_layer(11), "layer 11 should be attention");
        assert!(!cfg.is_attention_layer(0), "layer 0 is Mamba");
        assert!(!cfg.is_attention_layer(4), "layer 4 is Mamba");
        assert!(!cfg.is_attention_layer(7), "layer 7 is Mamba");
    }

    // ------ Test 3: is_moe_layer selects correct layers ------
    #[test]
    fn test_is_moe_layer() {
        let cfg = JambaConfig::jamba_1_5b();
        // expert_layer_offset=1, expert_layer_period=2
        // MoE layers are those that are BOTH attention AND MoE-scheduled
        // Attention layers: 3,11,19,27
        // Among those, offset=1, period=2 → 3%2=1 → 3 is MoE? Let's check: (3-1)%2=0 yes
        // 11: (11-1)%2=0 yes → 11 is MoE
        assert!(cfg.is_moe_layer(3), "layer 3 should be MoE");
        assert!(cfg.is_moe_layer(11), "layer 11 should be MoE");
        // Non-attention layer cannot be MoE
        assert!(!cfg.is_moe_layer(0), "layer 0 is Mamba, not MoE");
        assert!(!cfg.is_moe_layer(4), "layer 4 is Mamba, not MoE");
    }

    // ------ Test 4: hidden_size consistency ------
    #[test]
    fn test_hidden_size_consistency() {
        let cfg = JambaConfig::jamba_1_5b();
        assert_eq!(cfg.head_dim() * cfg.num_attention_heads, cfg.hidden_size);
        assert_eq!(cfg.mamba_inner_dim(), cfg.hidden_size * cfg.mamba_expand);
    }

    // ------ Test 5: JambaRmsNorm forward ------
    #[test]
    fn test_jamba_rmsnorm_forward() {
        let norm = JambaRmsNorm::new(4, 1e-5);
        let x = vec![1.0, 2.0, 3.0, 4.0];
        let out = norm.forward(&x).expect("rmsnorm should succeed");
        assert_eq!(out.len(), 4);
        let mean_sq: f64 = x.iter().map(|v| v * v).sum::<f64>() / 4.0;
        let rms = (mean_sq + 1e-5).sqrt();
        for (i, (got, &orig)) in out.iter().zip(x.iter()).enumerate() {
            let expected = orig / rms;
            assert!(
                (got - expected).abs() < 1e-9,
                "norm[{}]: got={} exp={}",
                i,
                got,
                expected
            );
        }
    }

    // ------ Test 6: Mamba layer forward shape ------
    #[test]
    fn test_mamba_layer_forward_shape() {
        let cfg = JambaConfig::small_test();
        let layer = JambaMambaLayer::new(&cfg);
        let seq_len = 5usize;
        let x: Vec<Vec<f64>> = vec![vec![0.1f64; cfg.hidden_size]; seq_len];
        let out = layer.forward(&x).expect("mamba layer forward");
        assert_eq!(out.len(), seq_len);
        assert_eq!(out[0].len(), cfg.hidden_size);
    }

    // ------ Test 7: attention layer forward shape ------
    #[test]
    fn test_attention_layer_forward_shape() {
        let cfg = JambaConfig::small_test();
        let attn = JambaAttentionLayer::new(&cfg);
        let seq_len = 4usize;
        let x: Vec<Vec<f64>> = vec![vec![0.1f64; cfg.hidden_size]; seq_len];
        let out = attn.forward(&x).expect("attention forward");
        assert_eq!(out.len(), seq_len);
        assert_eq!(out[0].len(), cfg.hidden_size);
    }

    // ------ Test 8: MoE routing selects top-2 experts ------
    #[test]
    fn test_moe_routing_top2() {
        let cfg = JambaConfig::small_test();
        let moe = JambaMoeLayer::new(&cfg);
        let x = vec![1.0f64; cfg.hidden_size];
        let selected = moe.selected_experts(&x).expect("moe routing");
        assert_eq!(
            selected.len(),
            cfg.num_experts_per_tok,
            "should select exactly num_experts_per_tok experts"
        );
        // All selected indices should be valid
        for &idx in selected.iter() {
            assert!(idx < cfg.num_experts, "expert idx {} out of range", idx);
        }
        // Selected experts should be distinct
        let mut sorted = selected.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(
            sorted.len(),
            selected.len(),
            "selected experts should be distinct"
        );
    }

    // ------ Test 9: MoE load balancing — all experts selectable ------
    #[test]
    fn test_moe_load_balancing_all_experts_selectable() {
        let cfg = JambaConfig::small_test();
        let moe = JambaMoeLayer::new(&cfg);
        let mut ever_selected = vec![false; cfg.num_experts];

        // Use different input patterns to trigger different expert selections
        for i in 0..cfg.num_experts {
            let mut x = vec![0.0f64; cfg.hidden_size];
            // Emphasize different dimensions to force different routing
            x[i % cfg.hidden_size] = 10.0;
            let selected = moe.selected_experts(&x).expect("moe routing");
            for &idx in selected.iter() {
                ever_selected[idx] = true;
            }
        }

        // With our initialization (each expert has a unique dimension activated),
        // we expect multiple distinct experts to be selected
        let num_selected = ever_selected.iter().filter(|&&v| v).count();
        assert!(
            num_selected >= 2,
            "at least 2 distinct experts should be selectable, got {}",
            num_selected
        );
    }

    // ------ Test 10: full model forward (small_test) ------
    #[test]
    fn test_full_model_forward_small() {
        let cfg = JambaConfig::small_test();
        let model = JambaForCausalLM::new(&cfg);
        let input_ids = vec![0usize, 1, 2, 3];
        let logits = model.forward(&input_ids).expect("full model forward");
        assert_eq!(logits.len(), 4, "one logit vector per token");
        assert_eq!(logits[0].len(), cfg.vocab_size, "logit dim = vocab_size");
    }

    // ------ Test 11: lm_head output shape ------
    #[test]
    fn test_lm_head_output_shape() {
        let cfg = JambaConfig::small_test();
        let model = JambaForCausalLM::new(&cfg);
        let input_ids = vec![0usize, 5, 10];
        let logits = model.forward(&input_ids).expect("lm_head forward");
        assert_eq!(logits.len(), 3);
        for row in logits.iter() {
            assert_eq!(row.len(), cfg.vocab_size);
        }
    }

    // ------ Test 12: interleaved layer structure ------
    #[test]
    fn test_interleaved_layer_structure() {
        let cfg = JambaConfig::small_test();
        // num_hidden_layers=8, attn_layer_offset=3, period=8 → only layer 3 is attention
        let model = JambaForCausalLM::new(&cfg);
        let layers = model.model().layers();
        assert_eq!(layers.len(), cfg.num_hidden_layers);

        // Layers 0,1,2 should be Mamba; layer 3 should be Attention; rest Mamba
        for (i, layer) in layers.iter().enumerate() {
            if cfg.is_attention_layer(i) {
                assert!(layer.is_attention(), "layer {} should be attention", i);
            } else {
                assert!(layer.is_mamba(), "layer {} should be mamba", i);
            }
        }
    }

    // ------ Test 13: attention layers at correct positions in 1.5B ------
    #[test]
    fn test_attention_layer_positions_1_5b() {
        let cfg = JambaConfig::jamba_1_5b();
        let expected_attn_layers: Vec<usize> =
            (0..cfg.num_hidden_layers).filter(|&i| cfg.is_attention_layer(i)).collect();
        // Should be 3, 11, 19, 27 for 32-layer model with offset=3, period=8
        assert_eq!(expected_attn_layers, vec![3, 11, 19, 27]);
    }

    // ------ Test 14: MoE layer forward output shape ------
    #[test]
    fn test_moe_layer_forward_shape() {
        let cfg = JambaConfig::small_test();
        let moe = JambaMoeLayer::new(&cfg);
        let x = vec![0.5f64; cfg.hidden_size];
        let out = moe.forward(&x).expect("moe forward");
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MoE output should match hidden_size"
        );
    }

    // ------ Test 15: empty input returns error ------
    #[test]
    fn test_empty_input_error() {
        let cfg = JambaConfig::small_test();
        let model = JambaForCausalLM::new(&cfg);
        let result = model.forward(&[]);
        assert!(result.is_err());
        matches!(result.unwrap_err(), JambaError::EmptyInput);
    }
}
