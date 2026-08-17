//! InternLM-2 model core components.
//!
//! Implements the transformer backbone with:
//! - RoPE with optional NTK dynamic scaling
//! - RMSNorm
//! - Grouped Query Attention (GQA)
//! - SwiGLU MLP

use crate::internlm2::config::InternLm2Config;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by InternLM-2 operations.
#[derive(Debug)]
pub enum InternLm2Error {
    /// Invalid input (e.g., empty token list, mismatched dimensions)
    InvalidInput(String),
    /// Error during a forward pass computation
    ForwardError(String),
}

impl fmt::Display for InternLm2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            InternLm2Error::InvalidInput(msg) => write!(f, "InternLM-2 invalid input: {msg}"),
            InternLm2Error::ForwardError(msg) => write!(f, "InternLM-2 forward error: {msg}"),
        }
    }
}

impl std::error::Error for InternLm2Error {}

// ─────────────────────────────────────────────────────────────────────────────
// RoPE with optional NTK dynamic scaling
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding (RoPE) for InternLM-2.
///
/// Optionally applies NTK-aware dynamic scaling by multiplying each frequency
/// theta_i by `scale^(2i/d)` when `rope_scaling` is `Some(scale)`.
pub struct InternLm2RotaryEmbedding {
    theta: f64,
    scaling: Option<f64>,
}

impl InternLm2RotaryEmbedding {
    /// Create a new RoPE embedding.
    pub fn new(theta: f64, scaling: Option<f64>) -> Self {
        Self { theta, scaling }
    }

    /// Compute base frequencies for each head-dim pair.
    ///
    /// `i` runs over half the head dimension; returns `theta_i = theta^(-2i/d)`.
    fn compute_freqs(&self, head_dim: usize) -> Vec<f64> {
        let half = head_dim / 2;
        (0..half)
            .map(|i| {
                let base_freq = 1.0 / self.theta.powf(2.0 * i as f64 / head_dim as f64);
                match self.scaling {
                    Some(scale) => base_freq * scale.powf(2.0 * i as f64 / head_dim as f64),
                    None => base_freq,
                }
            })
            .collect()
    }

    /// Apply RoPE to query and key tensors.
    ///
    /// `q` and `k` are flat arrays with shape `[seq_len, num_heads, head_dim]` encoded
    /// as row-major. Returns `(rotated_q, rotated_k)` with the same shape.
    ///
    /// If `q` or `k` are empty the function returns empty vectors without error
    /// (callers can pass placeholder tensors from mock models).
    pub fn apply(
        &self,
        q: &[f32],
        k: &[f32],
        seq_len: usize,
        head_dim: usize,
    ) -> (Vec<f32>, Vec<f32>) {
        if q.is_empty() || k.is_empty() || head_dim < 2 || seq_len == 0 {
            return (q.to_vec(), k.to_vec());
        }

        let mut q_out = q.to_vec();
        let mut k_out = k.to_vec();
        self.rotate_in_place(&mut q_out, seq_len, head_dim);
        self.rotate_in_place(&mut k_out, seq_len, head_dim);
        (q_out, k_out)
    }

    /// Rotate a flat `[seq_len, num_heads, head_dim]` buffer in place.
    ///
    /// `num_heads` is inferred from the buffer length, so every head of a given
    /// token is rotated with that token's position (query and key tensors may
    /// carry different head counts under GQA).
    pub fn rotate_in_place(&self, data: &mut [f32], seq_len: usize, head_dim: usize) {
        if data.is_empty() || head_dim < 2 || seq_len == 0 {
            return;
        }
        let num_vectors = data.len() / head_dim;
        if num_vectors == 0 || !num_vectors.is_multiple_of(seq_len) {
            return;
        }
        let num_heads = num_vectors / seq_len;
        let freqs = self.compute_freqs(head_dim);
        let half = head_dim / 2;

        for pos in 0..seq_len {
            for head in 0..num_heads {
                let base = (pos * num_heads + head) * head_dim;
                for i in 0..half {
                    let angle = pos as f32 * freqs[i] as f32;
                    let (sin_a, cos_a) = angle.sin_cos();
                    let x0 = data[base + i];
                    let x1 = data[base + i + half];
                    data[base + i] = x0 * cos_a - x1 * sin_a;
                    data[base + i + half] = x0 * sin_a + x1 * cos_a;
                }
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Dense weight matrices
// ─────────────────────────────────────────────────────────────────────────────

/// A row-major `[out_dim, in_dim]` weight matrix with a real matrix-vector product.
#[derive(Clone, Debug)]
pub struct InternLm2Linear {
    weight: Vec<f32>,
    out_dim: usize,
    in_dim: usize,
}

impl InternLm2Linear {
    /// Create a matrix with deterministic pseudo-random weights.
    ///
    /// Zero-initialised projections make every output identical regardless of
    /// the input, which is indistinguishable from a broken layer; a reproducible
    /// `1/sqrt(fan_in)`-scaled draw keeps the module exercisable before real
    /// checkpoint weights are loaded through [`set_weight`](Self::set_weight).
    pub fn new(out_dim: usize, in_dim: usize, seed: u64) -> Self {
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(0x1234_5678);
        let scale = 1.0 / (in_dim.max(1) as f32).sqrt();
        let weight = (0..out_dim * in_dim)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let unit = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
                unit * scale
            })
            .collect();
        Self {
            weight,
            out_dim,
            in_dim,
        }
    }

    /// Output dimension (rows).
    pub fn out_dim(&self) -> usize {
        self.out_dim
    }

    /// Input dimension (columns).
    pub fn in_dim(&self) -> usize {
        self.in_dim
    }

    /// Raw row-major weights.
    pub fn weight(&self) -> &[f32] {
        &self.weight
    }

    /// Replace the weights; the buffer must be `out_dim * in_dim` long.
    pub fn set_weight(&mut self, weight: Vec<f32>) -> Result<(), InternLm2Error> {
        if weight.len() != self.out_dim * self.in_dim {
            return Err(InternLm2Error::InvalidInput(format!(
                "expected {} weights ({}x{}), got {}",
                self.out_dim * self.in_dim,
                self.out_dim,
                self.in_dim,
                weight.len()
            )));
        }
        self.weight = weight;
        Ok(())
    }

    /// `out = W x` for a single vector of length `in_dim`.
    pub fn forward_vec(&self, input: &[f32]) -> Vec<f32> {
        let mut out = vec![0.0f32; self.out_dim];
        for (row, slot) in out.iter_mut().enumerate() {
            let base = row * self.in_dim;
            let mut acc = 0.0f32;
            for (i, &x) in input.iter().take(self.in_dim).enumerate() {
                acc += self.weight[base + i] * x;
            }
            *slot = acc;
        }
        out
    }

    /// Apply the matrix to every `in_dim`-sized row of a flat buffer.
    pub fn forward_rows(&self, input: &[f32]) -> Vec<f32> {
        let mut out = Vec::with_capacity(input.len() / self.in_dim.max(1) * self.out_dim);
        for row in input.chunks(self.in_dim) {
            out.extend_from_slice(&self.forward_vec(row));
        }
        out
    }

    /// Number of stored parameters.
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// RMS layer normalization used in InternLM-2.
pub struct InternLm2RmsNorm;

impl InternLm2RmsNorm {
    /// Normalise `x` using its RMS then scale by `weight`.
    ///
    /// `x` and `weight` must have the same length.
    pub fn forward(x: &[f32], weight: &[f32], eps: f64) -> Vec<f32> {
        let len = x.len();
        if len == 0 {
            return Vec::new();
        }
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / len as f32;
        let rms = (mean_sq + eps as f32).sqrt();
        x.iter().zip(weight.iter()).map(|(xi, wi)| xi / rms * wi).collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Attention (GQA)
// ─────────────────────────────────────────────────────────────────────────────

/// InternLM-2 attention layer with Grouped Query Attention (GQA).
///
/// `num_kv_heads < num_attention_heads` — each KV head is shared by
/// `gqa_ratio = num_attention_heads / num_kv_heads` query heads.
pub struct InternLm2Attention {
    /// Cloned model config for dimension information.
    pub config: InternLm2Config,
    /// Index of this layer (0-based).
    pub layer_idx: usize,
    /// Query projection `[num_attention_heads * head_dim, hidden_size]`.
    q_proj: InternLm2Linear,
    /// Key projection `[num_key_value_heads * head_dim, hidden_size]`.
    k_proj: InternLm2Linear,
    /// Value projection `[num_key_value_heads * head_dim, hidden_size]`.
    v_proj: InternLm2Linear,
    /// Output projection `[hidden_size, num_attention_heads * head_dim]`.
    o_proj: InternLm2Linear,
    /// RoPE module.
    rope: InternLm2RotaryEmbedding,
    /// Layer-norm weight for pre-attention norm.
    norm_weight: Vec<f32>,
}

impl InternLm2Attention {
    /// Create a new attention layer.
    ///
    /// Projections carry deterministic pseudo-random weights (see
    /// [`InternLm2Linear::new`]) and the pre-attention norm starts at ones.
    pub fn new(config: InternLm2Config, layer_idx: usize) -> Self {
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        let q_dim = config.num_attention_heads * head_dim;
        let kv_dim = config.num_key_value_heads * head_dim;
        let norm_weight = vec![1.0_f32; h];
        let rope = InternLm2RotaryEmbedding::new(config.rope_theta, config.rope_scaling);
        let seed = 0x51E1_0000 ^ (layer_idx as u64 + 1);
        Self {
            q_proj: InternLm2Linear::new(q_dim, h, seed),
            k_proj: InternLm2Linear::new(kv_dim, h, seed ^ 0x11),
            v_proj: InternLm2Linear::new(kv_dim, h, seed ^ 0x22),
            o_proj: InternLm2Linear::new(h, q_dim, seed ^ 0x33),
            rope,
            norm_weight,
            config,
            layer_idx,
        }
    }

    /// Map query head index to its corresponding KV head index.
    pub fn kv_head_for_q(&self, q_head: usize) -> usize {
        let ratio = self.config.gqa_ratio();
        q_head / ratio
    }

    /// Mutable access to the query projection (for weight loading).
    pub fn q_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.q_proj
    }

    /// Mutable access to the key projection (for weight loading).
    pub fn k_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.k_proj
    }

    /// Mutable access to the value projection (for weight loading).
    pub fn v_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.v_proj
    }

    /// Mutable access to the output projection (for weight loading).
    pub fn o_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.o_proj
    }

    /// Replace the pre-attention RMSNorm weight (`hidden_size` values).
    pub fn set_norm_weight(&mut self, weight: Vec<f32>) -> Result<(), InternLm2Error> {
        if weight.len() != self.config.hidden_size {
            return Err(InternLm2Error::InvalidInput(format!(
                "attention norm weight must have {} values, got {}",
                self.config.hidden_size,
                weight.len()
            )));
        }
        self.norm_weight = weight;
        Ok(())
    }

    /// Number of parameters in this attention block.
    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
            + self.norm_weight.len()
    }

    /// Causal grouped-query self-attention.
    ///
    /// `hidden_states` is a flat `[seq_len * hidden_size]` buffer. The block
    /// performs a pre-attention RMSNorm, projects Q/K/V with real weights, rotates
    /// Q and K with RoPE, and computes
    /// `softmax(mask(Q Kᵀ) / sqrt(head_dim)) V` where query head `q` reads KV head
    /// `q / gqa_ratio`. The result is projected back with `o_proj`.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        match self.try_forward(hidden_states, seq_len) {
            Ok(output) => output,
            // The public signature is infallible; a malformed input yields an
            // all-zero block rather than a panic. `try_forward` reports the
            // reason for callers that can handle it.
            Err(_) => vec![0.0_f32; seq_len * self.config.hidden_size],
        }
    }

    /// Fallible variant of [`forward`](Self::forward).
    pub fn try_forward(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, InternLm2Error> {
        let h = self.config.hidden_size;
        if seq_len == 0 {
            return Ok(Vec::new());
        }
        if hidden_states.len() != seq_len * h {
            return Err(InternLm2Error::InvalidInput(format!(
                "expected {} hidden values, got {}",
                seq_len * h,
                hidden_states.len()
            )));
        }

        // Pre-attention RMS norm applied per token.
        let normed: Vec<f32> = hidden_states
            .chunks(h)
            .flat_map(|chunk| {
                InternLm2RmsNorm::forward(chunk, &self.norm_weight, self.config.rms_norm_eps)
            })
            .collect();

        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;
        let q_width = num_q_heads * head_dim;
        let kv_width = num_kv_heads * head_dim;

        // Real projections: [seq_len, heads * head_dim]
        let mut queries = self.q_proj.forward_rows(&normed);
        let mut keys = self.k_proj.forward_rows(&normed);
        let values = self.v_proj.forward_rows(&normed);

        // RoPE on queries and keys (values are never rotated).
        self.rope.rotate_in_place(&mut queries, seq_len, head_dim);
        self.rope.rotate_in_place(&mut keys, seq_len, head_dim);

        // Causal scaled dot-product attention with GQA head sharing.
        let scale = (head_dim as f32).sqrt().recip();
        let mut context = vec![0.0_f32; seq_len * q_width];
        let mut scores = vec![0.0_f32; seq_len];

        for q_head in 0..num_q_heads {
            let kv_head = self.kv_head_for_q(q_head).min(num_kv_heads.saturating_sub(1));
            for pos in 0..seq_len {
                let q_base = pos * q_width + q_head * head_dim;
                let mut max_score = f32::NEG_INFINITY;
                for (key_pos, score) in scores.iter_mut().take(pos + 1).enumerate() {
                    let k_base = key_pos * kv_width + kv_head * head_dim;
                    let dot: f32 =
                        (0..head_dim).map(|i| queries[q_base + i] * keys[k_base + i]).sum::<f32>()
                            * scale;
                    *score = dot;
                    if dot > max_score {
                        max_score = dot;
                    }
                }

                let mut sum = 0.0_f32;
                for score in scores.iter_mut().take(pos + 1) {
                    *score = (*score - max_score).exp();
                    sum += *score;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                for (key_pos, score) in scores.iter().take(pos + 1).enumerate() {
                    let weight = score * inv_sum;
                    let v_base = key_pos * kv_width + kv_head * head_dim;
                    for i in 0..head_dim {
                        context[q_base + i] += weight * values[v_base + i];
                    }
                }
            }
        }

        Ok(self.o_proj.forward_rows(&context))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────

/// InternLM-2 feed-forward network using SwiGLU activation.
///
/// `out = down_proj(gate_proj(x) * silu(up_proj(x)))`
pub struct InternLm2MLP {
    hidden_size: usize,
    intermediate_size: usize,
    /// Gate projection `[intermediate_size, hidden_size]`.
    gate_proj: InternLm2Linear,
    /// Up projection `[intermediate_size, hidden_size]`.
    up_proj: InternLm2Linear,
    /// Down projection `[hidden_size, intermediate_size]`.
    down_proj: InternLm2Linear,
    /// Layer-norm weight for pre-MLP norm.
    norm_weight: Vec<f32>,
    rms_norm_eps: f64,
}

impl InternLm2MLP {
    /// Create a new MLP with deterministic projections and ones norm weights.
    pub fn new(config: &InternLm2Config) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Self {
            hidden_size: h,
            intermediate_size: i,
            gate_proj: InternLm2Linear::new(i, h, 0x3F1E_0001),
            up_proj: InternLm2Linear::new(i, h, 0x3F1E_0002),
            down_proj: InternLm2Linear::new(h, i, 0x3F1E_0003),
            norm_weight: vec![1.0_f32; h],
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    /// SiLU activation: `x * sigmoid(x)`.
    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Hidden size this MLP maps from and back to.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Width of the SwiGLU intermediate projection.
    pub fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }

    /// Mutable access to the gate projection (for weight loading).
    pub fn gate_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.gate_proj
    }

    /// Mutable access to the up projection (for weight loading).
    pub fn up_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.up_proj
    }

    /// Mutable access to the down projection (for weight loading).
    pub fn down_proj_mut(&mut self) -> &mut InternLm2Linear {
        &mut self.down_proj
    }

    /// Replace the pre-MLP RMSNorm weight (`hidden_size` values).
    pub fn set_norm_weight(&mut self, weight: Vec<f32>) -> Result<(), InternLm2Error> {
        if weight.len() != self.hidden_size {
            return Err(InternLm2Error::InvalidInput(format!(
                "MLP norm weight must have {} values, got {}",
                self.hidden_size,
                weight.len()
            )));
        }
        self.norm_weight = weight;
        Ok(())
    }

    /// Number of parameters in this MLP.
    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
            + self.norm_weight.len()
    }

    /// Forward pass: pre-norm → gate/up projection → SwiGLU → down projection.
    ///
    /// Accepts a flat `[seq_len * hidden_size]` input and returns the same shape.
    /// The SwiGLU is `down(silu(gate(x)) * up(x))`, matching the reference
    /// implementation.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let total = x.len();
        if total == 0 || !total.is_multiple_of(self.hidden_size) {
            return Vec::new();
        }
        let h = self.hidden_size;
        let mut out = Vec::with_capacity(total);

        for x_tok in x.chunks(h) {
            let normed = InternLm2RmsNorm::forward(x_tok, &self.norm_weight, self.rms_norm_eps);
            let gate = self.gate_proj.forward_vec(&normed);
            let up = self.up_proj.forward_vec(&normed);
            let activated: Vec<f32> =
                gate.iter().zip(up.iter()).map(|(g, u)| Self::silu(*g) * u).collect();
            out.extend_from_slice(&self.down_proj.forward_vec(&activated));
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder Layer
// ─────────────────────────────────────────────────────────────────────────────

/// One InternLM-2 transformer decoder block.
pub struct InternLm2DecoderLayer {
    attention: InternLm2Attention,
    mlp: InternLm2MLP,
}

impl InternLm2DecoderLayer {
    /// Create a new decoder layer with given layer index.
    pub fn new(config: InternLm2Config, layer_idx: usize) -> Self {
        let mlp = InternLm2MLP::new(&config);
        let attention = InternLm2Attention::new(config, layer_idx);
        Self { attention, mlp }
    }

    /// Forward pass with residual connections.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        // Self-attention with residual
        let attn_out = self.attention.forward(hidden_states, seq_len);
        let after_attn: Vec<f32> =
            hidden_states.iter().zip(attn_out.iter()).map(|(h, a)| h + a).collect();

        // MLP with residual
        let mlp_out = self.mlp.forward(&after_attn);
        after_attn.iter().zip(mlp_out.iter()).map(|(h, m)| h + m).collect()
    }

    /// Mutable access to the attention block (for weight loading).
    pub fn attention_mut(&mut self) -> &mut InternLm2Attention {
        &mut self.attention
    }

    /// Mutable access to the MLP block (for weight loading).
    pub fn mlp_mut(&mut self) -> &mut InternLm2MLP {
        &mut self.mlp
    }

    /// Number of parameters in this decoder layer.
    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count() + self.mlp.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// InternLM-2 base language model (without LM head).
pub struct InternLm2Model {
    /// Model configuration.
    pub config: InternLm2Config,
    /// Decoder layers.
    pub layers: Vec<InternLm2DecoderLayer>,
    /// Final RMS norm weight.
    final_norm_weight: Vec<f32>,
    /// Token embedding table, row-major `[vocab_size, hidden_size]`.
    embed_weight: Vec<f32>,
}

impl InternLm2Model {
    /// Create a new model with the given configuration.
    ///
    /// The embedding table and every projection start from a deterministic
    /// pseudo-random draw so the forward pass is exercisable before checkpoint
    /// weights are loaded; a zero table would make every token embed identically.
    pub fn new(config: InternLm2Config) -> Self {
        let num_layers = config.num_hidden_layers;
        let h = config.hidden_size;
        let v = config.vocab_size;

        let layers = (0..num_layers)
            .map(|idx| InternLm2DecoderLayer::new(config.clone(), idx))
            .collect();

        // Reuse the matrix initialiser: an embedding table is a [vocab, hidden] matrix.
        let embed_weight = InternLm2Linear::new(v, h, 0x0EBE_D000).weight().to_vec();

        Self {
            final_norm_weight: vec![1.0_f32; h],
            embed_weight,
            layers,
            config,
        }
    }

    /// Replace the token embedding table (`vocab_size * hidden_size` values).
    pub fn set_embed_weight(&mut self, weight: Vec<f32>) -> Result<(), InternLm2Error> {
        let expected = self.config.vocab_size * self.config.hidden_size;
        if weight.len() != expected {
            return Err(InternLm2Error::InvalidInput(format!(
                "embedding table must have {expected} values, got {}",
                weight.len()
            )));
        }
        self.embed_weight = weight;
        Ok(())
    }

    /// Replace the final RMSNorm weight (`hidden_size` values).
    pub fn set_final_norm_weight(&mut self, weight: Vec<f32>) -> Result<(), InternLm2Error> {
        if weight.len() != self.config.hidden_size {
            return Err(InternLm2Error::InvalidInput(format!(
                "final norm weight must have {} values, got {}",
                self.config.hidden_size,
                weight.len()
            )));
        }
        self.final_norm_weight = weight;
        Ok(())
    }

    /// Look up the embedding row of `token_id`.
    pub fn embed_token(&self, token_id: usize) -> Result<&[f32], InternLm2Error> {
        let h = self.config.hidden_size;
        let start = token_id * h;
        self.embed_weight.get(start..start + h).ok_or_else(|| {
            InternLm2Error::InvalidInput(format!(
                "token id {token_id} is out of vocabulary range {}",
                self.config.vocab_size
            ))
        })
    }

    /// Total number of parameters held by the base model.
    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_weight.len() + layer_params + self.final_norm_weight.len()
    }

    /// Run the model on a sequence of token IDs.
    ///
    /// Returns the final hidden states `[seq_len * hidden_size]`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, InternLm2Error> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(InternLm2Error::InvalidInput(
                "input_ids must not be empty".to_string(),
            ));
        }

        let h = self.config.hidden_size;
        let v = self.config.vocab_size;

        // Token embedding lookup from the real embedding table.
        let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * h);
        for &tok in input_ids {
            let tok_id = tok as usize;
            if tok_id >= v {
                return Err(InternLm2Error::InvalidInput(format!(
                    "token id {tok_id} is out of vocabulary range {v}"
                )));
            }
            hidden.extend_from_slice(self.embed_token(tok_id)?);
        }

        // Pass through decoder layers.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final RMS norm.
        hidden = hidden
            .chunks(h)
            .flat_map(|chunk| {
                InternLm2RmsNorm::forward(chunk, &self.final_norm_weight, self.config.rms_norm_eps)
            })
            .collect();

        Ok(hidden)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::internlm2::config::InternLm2Config;

    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|_| lcg_next(&mut state)).collect()
    }

    fn tiny_internlm2_config() -> InternLm2Config {
        InternLm2Config {
            vocab_size: 64,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            intermediate_size: 16,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rope_scaling: None,
            hidden_act: "silu".to_string(),
            rms_norm_eps: 1e-5,
            tie_word_embeddings: false,
            use_cache: true,
        }
    }

    // -- InternLm2RmsNorm --

    #[test]
    fn test_internlm2_rmsnorm_unit_rms() {
        let weight = vec![1.0_f32; 4];
        let x = vec![3.0_f32, 4.0, 0.0, 0.0];
        let output = InternLm2RmsNorm::forward(&x, &weight, 1e-5);
        let rms = (output.iter().map(|v| v * v).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-4,
            "RMSNorm output rms must be ~1.0, got {rms}"
        );
    }

    #[test]
    fn test_internlm2_rmsnorm_empty_input_returns_empty() {
        let output = InternLm2RmsNorm::forward(&[], &[], 1e-5);
        assert!(output.is_empty(), "empty input must return empty output");
    }

    #[test]
    fn test_internlm2_rmsnorm_preserves_length() {
        let x = lcg_vec(8, 70);
        let w = vec![1.0_f32; 8];
        let output = InternLm2RmsNorm::forward(&x, &w, 1e-5);
        assert_eq!(output.len(), 8, "RMSNorm must preserve input length");
    }

    // -- InternLm2RotaryEmbedding --

    #[test]
    fn test_internlm2_rope_output_length_matches_input() {
        let rope = InternLm2RotaryEmbedding::new(1_000_000.0, None);
        let head_dim = 8;
        let seq_len = 4;
        let q = lcg_vec(seq_len * head_dim, 71);
        let k = lcg_vec(seq_len * head_dim, 72);
        let (q_out, k_out) = rope.apply(&q, &k, seq_len, head_dim);
        assert_eq!(q_out.len(), q.len(), "Q output length must match input");
        assert_eq!(k_out.len(), k.len(), "K output length must match input");
    }

    #[test]
    fn test_internlm2_rope_empty_input_passthrough() {
        let rope = InternLm2RotaryEmbedding::new(10000.0, None);
        let (q_out, k_out) = rope.apply(&[], &[], 0, 8);
        assert!(q_out.is_empty(), "empty Q must pass through");
        assert!(k_out.is_empty(), "empty K must pass through");
    }

    #[test]
    fn test_internlm2_rope_norm_preserving() {
        let rope = InternLm2RotaryEmbedding::new(10000.0, None);
        let head_dim = 8;
        let q = lcg_vec(head_dim, 73);
        let k = lcg_vec(head_dim, 74);
        let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let (q_out, _) = rope.apply(&q, &k, 1, head_dim);
        let q_norm_after: f32 = q_out.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "RoPE must preserve norm, before={q_norm_before} after={q_norm_after}",
        );
    }

    #[test]
    fn test_internlm2_rope_with_ntk_scaling() {
        let rope = InternLm2RotaryEmbedding::new(1_000_000.0, Some(2.0));
        let head_dim = 8;
        let q = lcg_vec(head_dim, 75);
        let k = lcg_vec(head_dim, 76);
        let (q_out, k_out) = rope.apply(&q, &k, 1, head_dim);
        assert_eq!(q_out.len(), head_dim, "NTK-scaled RoPE Q length must match");
        assert_eq!(k_out.len(), head_dim, "NTK-scaled RoPE K length must match");
    }

    // -- InternLm2Config --

    #[test]
    fn test_internlm2_config_gqa_ratio() {
        let cfg = tiny_internlm2_config();
        let ratio = cfg.gqa_ratio();
        assert_eq!(
            ratio,
            cfg.num_attention_heads / cfg.num_key_value_heads,
            "gqa_ratio must equal nh/nkv",
        );
    }

    #[test]
    fn test_internlm2_config_head_dim() {
        let cfg = tiny_internlm2_config();
        let hd = cfg.head_dim();
        assert_eq!(
            hd,
            cfg.hidden_size / cfg.num_attention_heads,
            "head_dim = hidden_size/num_heads"
        );
    }

    // -- InternLm2Attention --

    #[test]
    fn test_internlm2_attention_kv_head_mapping() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        // For 4 Q heads and 2 KV heads, ratio=2
        // Q head 0,1 → KV head 0; Q head 2,3 → KV head 1
        assert_eq!(attn.kv_head_for_q(0), 0, "Q head 0 must map to KV head 0");
        assert_eq!(attn.kv_head_for_q(1), 0, "Q head 1 must map to KV head 0");
        assert_eq!(attn.kv_head_for_q(2), 1, "Q head 2 must map to KV head 1");
        assert_eq!(attn.kv_head_for_q(3), 1, "Q head 3 must map to KV head 1");
    }

    #[test]
    fn test_internlm2_attention_forward_shape() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        let hidden = lcg_vec(cfg.hidden_size, 80);
        let out = attn.forward(&hidden, 1);
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "attention output must have hidden_size elements"
        );
    }

    // -- Real attention: weights, cross-token reads, causality, reference math --

    /// Zero-initialised projections would make the output identically zero.
    #[test]
    fn test_internlm2_attention_output_is_not_zero() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        let hidden = lcg_vec(2 * cfg.hidden_size, 81);
        let out = attn.forward(&hidden, 2);
        let magnitude = out.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(
            magnitude > 1e-6,
            "attention must use real projection weights (max |out| = {magnitude})"
        );
    }

    /// Each token must attend over the whole causal prefix. The old code computed
    /// a single self-score and discarded it, so earlier tokens had no influence.
    #[test]
    fn test_internlm2_attention_reads_previous_tokens() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        let seq_len = 3;
        let h = cfg.hidden_size;
        let base = lcg_vec(seq_len * h, 82);

        let out_a = attn.forward(&base, seq_len);
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().take(h) {
            *value += 1.0;
        }
        let out_b = attn.forward(&perturbed, seq_len);

        let last = (seq_len - 1) * h;
        let diff = out_a[last..]
            .iter()
            .zip(out_b[last..].iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            diff > 1e-6,
            "the last token must attend to earlier tokens (diff {diff})"
        );
    }

    #[test]
    fn test_internlm2_attention_is_causal() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        let seq_len = 3;
        let h = cfg.hidden_size;
        let base = lcg_vec(seq_len * h, 83);

        let out_a = attn.forward(&base, seq_len);
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().skip((seq_len - 1) * h) {
            *value += 1.5;
        }
        let out_b = attn.forward(&perturbed, seq_len);

        for i in 0..(seq_len - 1) * h {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-5,
                "position {} must not see the future",
                i / h
            );
        }
    }

    /// Recompute the block by hand from its own weights and compare.
    #[test]
    fn test_internlm2_attention_matches_naive_reference() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg.clone(), 0);
        let seq_len = 3;
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim();
        let num_q = cfg.num_attention_heads;
        let num_kv = cfg.num_key_value_heads;
        let ratio = cfg.gqa_ratio();
        let x = lcg_vec(seq_len * h, 84);

        let got = attn.try_forward(&x, seq_len).expect("forward");

        // Reference implementation.
        let normed: Vec<f32> = x
            .chunks(h)
            .flat_map(|chunk| InternLm2RmsNorm::forward(chunk, &attn.norm_weight, cfg.rms_norm_eps))
            .collect();
        let mut q = attn.q_proj.forward_rows(&normed);
        let mut k = attn.k_proj.forward_rows(&normed);
        let v = attn.v_proj.forward_rows(&normed);
        attn.rope.rotate_in_place(&mut q, seq_len, head_dim);
        attn.rope.rotate_in_place(&mut k, seq_len, head_dim);

        let q_width = num_q * head_dim;
        let kv_width = num_kv * head_dim;
        let scale = (head_dim as f32).sqrt().recip();
        let mut context = vec![0.0f32; seq_len * q_width];
        for q_head in 0..num_q {
            let kv_head = q_head / ratio;
            for pos in 0..seq_len {
                let q_base = pos * q_width + q_head * head_dim;
                let mut weights = Vec::with_capacity(pos + 1);
                for key_pos in 0..=pos {
                    let k_base = key_pos * kv_width + kv_head * head_dim;
                    weights.push(
                        (0..head_dim).map(|i| q[q_base + i] * k[k_base + i]).sum::<f32>() * scale,
                    );
                }
                let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = weights.iter().map(|w| (w - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (key_pos, e) in exps.iter().enumerate() {
                    let weight = e / sum;
                    let v_base = key_pos * kv_width + kv_head * head_dim;
                    for i in 0..head_dim {
                        context[q_base + i] += weight * v[v_base + i];
                    }
                }
            }
        }
        let expected = attn.o_proj.forward_rows(&context);

        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: got {a}, expected {b}");
        }
    }

    #[test]
    fn test_internlm2_attention_rejects_bad_input_length() {
        let cfg = tiny_internlm2_config();
        let attn = InternLm2Attention::new(cfg, 0);
        assert!(attn.try_forward(&[0.0, 1.0, 2.0], 2).is_err());
    }

    #[test]
    fn test_internlm2_linear_matvec_reference() {
        let mut linear = InternLm2Linear::new(2, 3, 5);
        linear.set_weight(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).expect("weights");
        let out = linear.forward_vec(&[1.0, 0.5, -1.0]);
        // Row 0: 1*1 + 2*0.5 + 3*(-1) = -1 ; Row 1: 4*1 + 5*0.5 + 6*(-1) = 0.5
        assert!((out[0] + 1.0).abs() < 1e-6, "got {}", out[0]);
        assert!((out[1] - 0.5).abs() < 1e-6, "got {}", out[1]);
        assert!(linear.set_weight(vec![1.0]).is_err());
    }

    /// RoPE must derive the angle from the **token** position, not from the flat
    /// vector index: every head of a token shares one position.
    #[test]
    fn test_internlm2_rope_position_is_per_token_not_per_head() {
        let rope = InternLm2RotaryEmbedding::new(10000.0, None);
        let head_dim = 2;
        let seq_len = 2;
        let num_heads = 2;
        let original: Vec<f32> = vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
        let mut data = original.clone();
        rope.rotate_in_place(&mut data, seq_len, head_dim);

        // Token 0 (both heads): angle 0 → unchanged.
        for i in 0..num_heads * head_dim {
            assert!(
                (data[i] - original[i]).abs() < 1e-6,
                "head {} of token 0 must not be rotated",
                i / head_dim
            );
        }
        // Token 1 (both heads): rotated by the same non-zero angle.
        let head0 = &data[num_heads * head_dim..num_heads * head_dim + head_dim];
        let head1 = &data[num_heads * head_dim + head_dim..];
        assert!((head0[0] - head1[0]).abs() < 1e-6);
        assert!((head0[1] - head1[1]).abs() < 1e-6);
        assert!(
            (head0[0] - 1.0).abs() > 1e-6 || head0[1].abs() > 1e-6,
            "token 1 must actually be rotated"
        );
    }

    // -- MLP with real weights --

    #[test]
    fn test_internlm2_mlp_output_is_input_dependent() {
        let cfg = tiny_internlm2_config();
        let mlp = InternLm2MLP::new(&cfg);
        let a = mlp.forward(&lcg_vec(cfg.hidden_size, 85));
        let b = mlp.forward(&lcg_vec(cfg.hidden_size, 86));
        let magnitude = a.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(magnitude > 1e-6, "the MLP must not return all zeros");
        let diff = a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(diff > 1e-6, "the MLP output must depend on its input");
    }

    #[test]
    fn test_internlm2_mlp_matches_naive_reference() {
        let cfg = tiny_internlm2_config();
        let mlp = InternLm2MLP::new(&cfg);
        let x = lcg_vec(cfg.hidden_size, 87);
        let got = mlp.forward(&x);

        let normed = InternLm2RmsNorm::forward(&x, &mlp.norm_weight, cfg.rms_norm_eps);
        let gate = mlp.gate_proj.forward_vec(&normed);
        let up = mlp.up_proj.forward_vec(&normed);
        let activated: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(g, u)| (g / (1.0 + (-g).exp())) * u).collect();
        let expected = mlp.down_proj.forward_vec(&activated);

        for (i, (a, b)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "element {i}: {a} vs {b}");
        }
    }

    // -- Embeddings --

    #[test]
    fn test_internlm2_embeddings_differ_per_token() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg.clone());
        let first = model.embed_token(0).expect("token 0").to_vec();
        let second = model.embed_token(1).expect("token 1").to_vec();
        let magnitude = first.iter().fold(0.0f32, |acc, v| acc.max(v.abs()));
        assert!(magnitude > 1e-6, "embeddings must not be all zero");
        let diff = first
            .iter()
            .zip(second.iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(diff > 1e-6, "different tokens need different embeddings");
    }

    #[test]
    fn test_internlm2_set_embed_weight_validates_length() {
        let cfg = tiny_internlm2_config();
        let mut model = InternLm2Model::new(cfg.clone());
        let good = vec![0.25f32; cfg.vocab_size * cfg.hidden_size];
        assert!(model.set_embed_weight(good).is_ok());
        assert!(model.set_embed_weight(vec![0.0; 3]).is_err());
    }

    #[test]
    fn test_internlm2_model_output_depends_on_input_ids() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg.clone());
        let a = model.forward(&[1u32, 2, 3]).expect("forward a");
        let b = model.forward(&[3u32, 2, 1]).expect("forward b");
        let diff = a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(diff > 1e-6, "hidden states must depend on the input tokens");
        for value in a {
            assert!(value.is_finite(), "hidden states must stay finite");
        }
    }

    // -- InternLm2MLP --

    #[test]
    fn test_internlm2_mlp_output_length() {
        let cfg = tiny_internlm2_config();
        let mlp = InternLm2MLP::new(&cfg);
        let x = lcg_vec(cfg.hidden_size, 81);
        let out = mlp.forward(&x);
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MLP output must have hidden_size elements"
        );
    }

    #[test]
    fn test_internlm2_mlp_empty_input_returns_empty() {
        let cfg = tiny_internlm2_config();
        let mlp = InternLm2MLP::new(&cfg);
        let out = mlp.forward(&[]);
        assert!(
            out.is_empty(),
            "MLP with empty input must return empty output"
        );
    }

    // -- InternLm2Model --

    #[test]
    fn test_internlm2_model_construction() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg);
        assert_eq!(model.layers.len(), 2, "model must have 2 layers");
    }

    #[test]
    fn test_internlm2_model_forward_single_token() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg.clone());
        let output = model.forward(&[0u32]).expect("forward must succeed");
        assert_eq!(
            output.len(),
            cfg.hidden_size,
            "output length must equal hidden_size"
        );
    }

    #[test]
    fn test_internlm2_model_forward_multi_token() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg.clone());
        let output = model.forward(&[0u32, 1, 2]).expect("multi-token forward must succeed");
        assert_eq!(
            output.len(),
            3 * cfg.hidden_size,
            "output length must be seq_len * hidden_size"
        );
    }

    #[test]
    fn test_internlm2_model_empty_input_fails() {
        let cfg = tiny_internlm2_config();
        let model = InternLm2Model::new(cfg);
        let result = model.forward(&[]);
        assert!(result.is_err(), "empty input must return an error");
    }

    #[test]
    fn test_internlm2_model_out_of_vocab_fails() {
        let cfg = tiny_internlm2_config(); // vocab_size=64
        let model = InternLm2Model::new(cfg);
        let result = model.forward(&[100u32]); // 100 >= 64
        assert!(result.is_err(), "out-of-vocab token must return an error");
    }
}
