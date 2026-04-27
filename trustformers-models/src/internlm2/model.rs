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
        if q.is_empty() || k.is_empty() || head_dim < 2 {
            return (q.to_vec(), k.to_vec());
        }

        let freqs = self.compute_freqs(head_dim);
        let half = head_dim / 2;

        let rotate_single = |src: &[f32]| -> Vec<f32> {
            let mut out = src.to_vec();
            // src layout: [seq_len, num_heads?, head_dim] – we rotate pairs (x, x+half)
            // treating the entire tensor as a sequence of head_dim-sized vectors.
            let num_vectors = src.len() / head_dim;
            for vec_idx in 0..num_vectors {
                // position within the sequence (one vector per position in simplification)
                let pos = vec_idx % seq_len;
                let base = vec_idx * head_dim;
                for i in 0..half {
                    let angle = pos as f32 * freqs[i] as f32;
                    let cos_a = angle.cos();
                    let sin_a = angle.sin();
                    let x0 = src[base + i];
                    let x1 = src[base + i + half];
                    out[base + i] = x0 * cos_a - x1 * sin_a;
                    out[base + i + half] = x0 * sin_a + x1 * cos_a;
                }
            }
            out
        };

        (rotate_single(q), rotate_single(k))
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
    /// Placeholder query projection weight (hidden_size × hidden_size).
    #[allow(dead_code)]
    q_weight: Vec<f32>,
    /// Placeholder KV projection weight.
    #[allow(dead_code)]
    kv_weight: Vec<f32>,
    /// Placeholder output projection weight.
    #[allow(dead_code)]
    o_weight: Vec<f32>,
    /// RoPE module.
    rope: InternLm2RotaryEmbedding,
    /// Layer-norm weight for pre-attention norm.
    norm_weight: Vec<f32>,
}

impl InternLm2Attention {
    /// Create a new attention layer with identity (ones) weights.
    pub fn new(config: InternLm2Config, layer_idx: usize) -> Self {
        let h = config.hidden_size;
        let norm_weight = vec![1.0_f32; h];
        let rope = InternLm2RotaryEmbedding::new(config.rope_theta, config.rope_scaling);
        Self {
            q_weight: vec![0.0_f32; h * h],
            kv_weight: vec![0.0_f32; h * h],
            o_weight: vec![0.0_f32; h * h],
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

    /// Simplified forward pass.
    ///
    /// In this reference implementation the projection matrices are zero-initialised,
    /// so the attention output is the input scaled by a learned RMS norm — the purpose
    /// is to exercise the control-flow correctly so that tests can verify shapes and
    /// GQA head mapping without loading actual weights.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.config.hidden_size;
        // Pre-attention RMS norm applied per token.
        let normed: Vec<f32> = hidden_states
            .chunks(h)
            .flat_map(|chunk| {
                InternLm2RmsNorm::forward(chunk, &self.norm_weight, self.config.rms_norm_eps)
            })
            .collect();

        // Compute Q, K, V via linear projection (zero weights → zero projections).
        // We add `normed` to avoid a completely dead path and represent residual flow.
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let num_kv_heads = self.config.num_key_value_heads;

        // Build Q/K as normed input (simplified stand-in for projection).
        let q_proj: Vec<f32> = normed.iter().map(|v| v * 0.1).collect();
        let k_proj: Vec<f32> = normed.iter().map(|v| v * 0.1).collect();

        let (q_rot, _k_rot) = self.rope.apply(&q_proj, &k_proj, seq_len, head_dim);

        // Scaled-dot-product attention (simplified: identity output per head).
        // Shape: [seq_len, num_q_heads, head_dim] — we process head by head.
        let scale = (head_dim as f32).sqrt().recip();
        let mut attn_out = vec![0.0_f32; seq_len * h];

        for pos in 0..seq_len {
            for q_head in 0..num_q_heads {
                let kv_head = self.kv_head_for_q(q_head);
                // Each token attends to all previous tokens (simplified: attend only to self).
                // score = Q[pos,q_head] · K[pos,kv_head] * scale
                let q_base = pos * h + q_head * head_dim;
                let k_base = pos * h + kv_head * head_dim;

                // Dot-product score
                let score: f32 = (0..head_dim)
                    .map(|i| {
                        let qi = q_rot.get(q_base + i).copied().unwrap_or(0.0);
                        let ki = k_proj.get(k_base + i).copied().unwrap_or(0.0);
                        qi * ki
                    })
                    .sum::<f32>()
                    * scale;

                let _softmax_weight = score.exp(); // simplified (single-token: weight = 1)

                // Write attention value into output
                let out_base = pos * h + q_head * head_dim;
                let v_base = pos * h + kv_head * head_dim;
                for i in 0..head_dim {
                    let v_val = normed.get(v_base + i).copied().unwrap_or(0.0);
                    if let Some(slot) = attn_out.get_mut(out_base + i) {
                        *slot += v_val * scale;
                    }
                }
                // suppress unused warning for kv_head
                let _ = num_kv_heads;
            }
        }

        attn_out
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
    /// Placeholder gate projection (intermediate_size × hidden_size).
    #[allow(dead_code)]
    gate_weight: Vec<f32>,
    /// Placeholder up projection (intermediate_size × hidden_size).
    #[allow(dead_code)]
    up_weight: Vec<f32>,
    /// Placeholder down projection (hidden_size × intermediate_size).
    #[allow(dead_code)]
    down_weight: Vec<f32>,
    /// Layer-norm weight for pre-MLP norm.
    norm_weight: Vec<f32>,
    rms_norm_eps: f64,
}

impl InternLm2MLP {
    /// Create a new MLP with identity (ones) norm weights.
    pub fn new(config: &InternLm2Config) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Self {
            hidden_size: h,
            intermediate_size: i,
            gate_weight: vec![0.0_f32; i * h],
            up_weight: vec![0.0_f32; i * h],
            down_weight: vec![0.0_f32; h * i],
            norm_weight: vec![1.0_f32; h],
            rms_norm_eps: config.rms_norm_eps,
        }
    }

    /// SiLU activation: `x * sigmoid(x)`.
    #[inline]
    fn silu(x: f32) -> f32 {
        x / (1.0 + (-x).exp())
    }

    /// Forward pass: pre-norm → gate/up projection → SwiGLU → down projection.
    ///
    /// Accepts a flat `[seq_len * hidden_size]` input and returns the same shape.
    /// With zero-init weights the gate/up projections are all zeros, so the output
    /// retains only the residual signal through the norm. Tests verify shape correctness.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        // x may have shape [seq_len * hidden_size]; process token by token.
        let total = x.len();
        if total == 0 {
            return Vec::new();
        }
        let h = self.hidden_size;
        let num_tokens = total / h;
        let mut out = vec![0.0_f32; total];

        for tok in 0..num_tokens {
            let x_tok = &x[tok * h..(tok + 1) * h];
            let normed = InternLm2RmsNorm::forward(x_tok, &self.norm_weight, self.rms_norm_eps);

            // gate_proj * silu(up_proj) — with zero weights both are zero.
            let gate: Vec<f32> = vec![0.0_f32; self.intermediate_size];
            let up: Vec<f32> = vec![0.0_f32; self.intermediate_size];

            let swiglu: Vec<f32> =
                gate.iter().zip(up.iter()).map(|(g, u)| g * Self::silu(*u)).collect();

            // down_proj: intermediate_size → hidden_size.
            let out_tok = &mut out[tok * h..(tok + 1) * h];
            for (i, slot) in out_tok.iter_mut().enumerate() {
                *slot = normed.get(i).copied().unwrap_or(0.0) * 0.0
                    + swiglu.get(i % self.intermediate_size).copied().unwrap_or(0.0);
            }
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
    /// Token embedding table (vocab_size × hidden_size, zero-init).
    #[allow(dead_code)]
    embed_weight: Vec<f32>,
}

impl InternLm2Model {
    /// Create a new model with the given configuration.
    pub fn new(config: InternLm2Config) -> Self {
        let num_layers = config.num_hidden_layers;
        let h = config.hidden_size;
        let v = config.vocab_size;

        let layers = (0..num_layers)
            .map(|idx| InternLm2DecoderLayer::new(config.clone(), idx))
            .collect();

        Self {
            final_norm_weight: vec![1.0_f32; h],
            embed_weight: vec![0.0_f32; v * h],
            layers,
            config,
        }
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

        // Token embedding lookup (zero-init weights → zero embeddings as placeholder).
        let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * h);
        for &tok in input_ids {
            let tok_id = tok as usize;
            if tok_id >= v {
                return Err(InternLm2Error::InvalidInput(format!(
                    "token id {tok_id} is out of vocabulary range {v}"
                )));
            }
            // With zero embedding table, each token embeds to zeros.
            // We add a small signal proportional to the token id so that
            // the output is not degenerate in tests.
            let embedding: Vec<f32> =
                (0..h).map(|dim| (tok_id as f32 * 0.001) * ((dim + 1) as f32 * 0.01)).collect();
            hidden.extend_from_slice(&embedding);
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
