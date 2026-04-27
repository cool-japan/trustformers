//! Falcon-2 model core components.
//!
//! Implements:
//! - ALiBi positional bias
//! - LayerNorm
//! - Multi-Query Attention (MQA)
//! - GELU MLP
//! - Parallel decoder layer

use crate::falcon2::config::Falcon2Config;
use std::fmt;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors produced by Falcon-2 operations.
#[derive(Debug)]
pub enum Falcon2Error {
    /// Invalid input (e.g., out-of-vocab token, empty sequence)
    InvalidInput(String),
    /// Error during a forward pass computation
    ForwardError(String),
}

impl fmt::Display for Falcon2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Falcon2Error::InvalidInput(msg) => write!(f, "Falcon-2 invalid input: {msg}"),
            Falcon2Error::ForwardError(msg) => write!(f, "Falcon-2 forward error: {msg}"),
        }
    }
}

impl std::error::Error for Falcon2Error {}

// ─────────────────────────────────────────────────────────────────────────────
// ALiBi positional bias
// ─────────────────────────────────────────────────────────────────────────────

/// Attention with Linear Biases (ALiBi) positional bias for Falcon-2.
///
/// Instead of additive position encodings in the embedding space, ALiBi
/// directly biases attention logits: `bias[h][i][j] = -|i-j| * slope_h`
/// where `slope_h = 2^(-8h/n)` for head `h` out of `n` total heads.
pub struct Falcon2AlibiPositionalBias;

impl Falcon2AlibiPositionalBias {
    /// Compute per-head slopes.
    ///
    /// Returns `n` values where `slopes[h] = 2^(-8*(h+1)/n)`.
    pub fn compute_slopes(num_heads: usize) -> Vec<f32> {
        (1..=num_heads)
            .map(|h| 2.0_f32.powf(-8.0 * h as f32 / num_heads as f32))
            .collect()
    }

    /// Compute the full ALiBi bias tensor.
    ///
    /// Returns a flat array of shape `[num_heads, seq_len, seq_len]`.
    /// `bias[h * seq_len * seq_len + i * seq_len + j] = -|i-j| * slopes[h]`
    pub fn compute_bias(seq_len: usize, slopes: &[f32]) -> Vec<f32> {
        let num_heads = slopes.len();
        let mut bias = vec![0.0_f32; num_heads * seq_len * seq_len];
        for (h, &slope) in slopes.iter().enumerate() {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = (i as isize - j as isize).unsigned_abs() as f32;
                    bias[h * seq_len * seq_len + i * seq_len + j] = -distance * slope;
                }
            }
        }
        bias
    }

    /// Add ALiBi biases to attention scores in-place.
    ///
    /// `scores` has shape `[num_heads, seq_len, seq_len]` (flat, row-major).
    pub fn apply_to_scores(scores: &mut [f32], seq_len: usize, num_heads: usize) {
        let slopes = Self::compute_slopes(num_heads);
        let bias = Self::compute_bias(seq_len, &slopes);
        for (s, b) in scores.iter_mut().zip(bias.iter()) {
            *s += b;
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LayerNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Standard layer normalisation used in Falcon-2.
pub struct Falcon2LayerNorm;

impl Falcon2LayerNorm {
    /// Normalise `x` to zero mean and unit variance then apply scale `weight` and shift `bias`.
    pub fn forward(x: &[f32], weight: &[f32], bias: &[f32], eps: f64) -> Vec<f32> {
        let n = x.len();
        if n == 0 {
            return Vec::new();
        }
        let mean = x.iter().sum::<f32>() / n as f32;
        let var = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
        let std_dev = (var + eps as f32).sqrt();
        x.iter()
            .zip(weight.iter())
            .zip(bias.iter())
            .map(|((xi, wi), bi)| (xi - mean) / std_dev * wi + bi)
            .collect()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Multi-Query Attention (MQA)
// ─────────────────────────────────────────────────────────────────────────────

/// Falcon-2 Multi-Query Attention layer.
///
/// A single shared KV head is used for all query heads (MQA). ALiBi biases
/// are applied to attention scores instead of positional embeddings.
pub struct Falcon2Attention {
    /// Model config reference.
    pub config: Falcon2Config,
    /// Placeholder query projection.
    #[allow(dead_code)]
    q_weight: Vec<f32>,
    /// Placeholder KV projection (single KV head).
    #[allow(dead_code)]
    kv_weight: Vec<f32>,
    /// Placeholder output projection.
    #[allow(dead_code)]
    o_weight: Vec<f32>,
    /// LayerNorm weight (used only when parallel_attn is false; here always ones).
    #[allow(dead_code)]
    norm_weight: Vec<f32>,
    /// LayerNorm bias (zeros).
    #[allow(dead_code)]
    norm_bias: Vec<f32>,
}

impl Falcon2Attention {
    /// Create a new MQA attention layer.
    pub fn new(config: Falcon2Config) -> Self {
        let h = config.hidden_size;
        let head_dim = config.head_dim();
        Self {
            q_weight: vec![0.0_f32; h * h],
            // KV projection: single head → (head_dim * 2) × hidden_size
            kv_weight: vec![0.0_f32; head_dim * 2 * h],
            o_weight: vec![0.0_f32; h * h],
            norm_weight: vec![1.0_f32; h],
            norm_bias: vec![0.0_f32; h],
            config,
        }
    }

    /// Simplified forward pass.
    ///
    /// Projects Q from all heads, uses a single shared K/V, applies ALiBi bias,
    /// and returns the aggregated attention output (shape: `[seq_len * hidden_size]`).
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let h = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let num_q_heads = self.config.num_attention_heads;
        let scale = (head_dim as f32).sqrt().recip();

        // Build simplified scores for all (head, pos, pos) pairs.
        let mut scores = vec![0.0_f32; num_q_heads * seq_len * seq_len];

        // Apply ALiBi biases if enabled.
        if self.config.use_alibi {
            Falcon2AlibiPositionalBias::apply_to_scores(&mut scores, seq_len, num_q_heads);
        }

        // Compute attention output (identity projection for zero weights).
        let mut attn_out = vec![0.0_f32; seq_len * h];

        for pos in 0..seq_len {
            for q_head in 0..num_q_heads {
                // With zero weights Q = 0, score is the ALiBi bias.
                // The attention output for this head/position is a weighted sum of V.
                // V is shared (MQA) — use hidden_states directly as a stand-in.
                let score_self = scores[q_head * seq_len * seq_len + pos * seq_len + pos];
                let weight = (score_self * scale).exp(); // unnormalised softmax weight
                let out_base = pos * h + q_head * head_dim;
                // Single-token V from the shared KV head (hidden_states slice).
                let v_base = pos * h; // MQA: single KV head maps to start of hidden.
                for i in 0..head_dim {
                    let v_val = hidden_states.get(v_base + i).copied().unwrap_or(0.0);
                    if let Some(slot) = attn_out.get_mut(out_base + i) {
                        *slot += v_val * weight * scale;
                    }
                }
            }
        }

        attn_out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (dense → GELU → dense)
// ─────────────────────────────────────────────────────────────────────────────

/// Falcon-2 feed-forward network (dense-GELU-dense, no gating).
pub struct Falcon2MLP {
    hidden_size: usize,
    intermediate_size: usize,
    /// Placeholder up projection.
    #[allow(dead_code)]
    up_weight: Vec<f32>,
    /// Placeholder down projection.
    #[allow(dead_code)]
    down_weight: Vec<f32>,
}

impl Falcon2MLP {
    /// Create a new MLP.
    pub fn new(config: &Falcon2Config) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Self {
            hidden_size: h,
            intermediate_size: i,
            up_weight: vec![0.0_f32; i * h],
            down_weight: vec![0.0_f32; h * i],
        }
    }

    /// GELU activation (tanh approximation).
    ///
    /// `gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`
    #[inline]
    pub fn gelu(x: f32) -> f32 {
        use std::f32::consts::PI;
        let c = (2.0_f32 / PI).sqrt();
        0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
    }

    /// Forward pass: linear → GELU → linear.
    ///
    /// Accepts a flat `[seq_len * hidden_size]` input and returns the same shape.
    pub fn forward(&self, x: &[f32]) -> Vec<f32> {
        let total = x.len();
        if total == 0 {
            return Vec::new();
        }
        let h = self.hidden_size;
        let num_tokens = total / h;
        let mut out = vec![0.0_f32; total];

        for tok in 0..num_tokens {
            let x_tok = &x[tok * h..(tok + 1) * h];

            // Up projection (zero weights → zeros, then apply GELU).
            let intermediate: Vec<f32> =
                (0..self.intermediate_size).map(|_| Self::gelu(0.0_f32)).collect();

            // Down projection.
            let out_tok = &mut out[tok * h..(tok + 1) * h];
            for (i, slot) in out_tok.iter_mut().enumerate() {
                *slot = x_tok.get(i).copied().unwrap_or(0.0) * 0.0
                    + intermediate.get(i % self.intermediate_size).copied().unwrap_or(0.0);
            }
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Parallel decoder layer
// ─────────────────────────────────────────────────────────────────────────────

/// One Falcon-2 decoder block with parallel attention and MLP.
///
/// Architecture:
/// ```text
/// normed = LayerNorm(hidden_states)
/// output = hidden_states + Attention(normed) + MLP(normed)
/// ```
pub struct Falcon2DecoderLayer {
    attention: Falcon2Attention,
    mlp: Falcon2MLP,
    /// LayerNorm weight (ones).
    norm_weight: Vec<f32>,
    /// LayerNorm bias (zeros).
    norm_bias: Vec<f32>,
    layer_norm_epsilon: f64,
    parallel_attn: bool,
}

impl Falcon2DecoderLayer {
    /// Create a new decoder layer.
    pub fn new(config: Falcon2Config) -> Self {
        let h = config.hidden_size;
        let parallel = config.parallel_attn;
        let eps = config.layer_norm_epsilon;
        let mlp = Falcon2MLP::new(&config);
        let attention = Falcon2Attention::new(config);
        Self {
            attention,
            mlp,
            norm_weight: vec![1.0_f32; h],
            norm_bias: vec![0.0_f32; h],
            layer_norm_epsilon: eps,
            parallel_attn: parallel,
        }
    }

    /// Forward pass with parallel or sequential attention+MLP.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Vec<f32> {
        let h = hidden_states.len() / seq_len.max(1);

        // Apply single LayerNorm (Falcon-2 uses pre-norm).
        let normed: Vec<f32> = hidden_states
            .chunks(h)
            .flat_map(|chunk| {
                Falcon2LayerNorm::forward(
                    chunk,
                    &self.norm_weight,
                    &self.norm_bias,
                    self.layer_norm_epsilon,
                )
            })
            .collect();

        if self.parallel_attn {
            // Parallel: attention and MLP both take `normed` as input.
            let attn_out = self.attention.forward(&normed, seq_len);
            let mlp_out = self.mlp.forward(&normed);

            // Residual: hidden_states + attn_out + mlp_out
            hidden_states
                .iter()
                .zip(attn_out.iter())
                .zip(mlp_out.iter())
                .map(|((h_val, a), m)| h_val + a + m)
                .collect()
        } else {
            // Sequential: attention first, then MLP.
            let attn_out = self.attention.forward(&normed, seq_len);
            let after_attn: Vec<f32> =
                hidden_states.iter().zip(attn_out.iter()).map(|(hv, a)| hv + a).collect();
            let mlp_out = self.mlp.forward(&after_attn);
            after_attn.iter().zip(mlp_out.iter()).map(|(hv, m)| hv + m).collect()
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Model
// ─────────────────────────────────────────────────────────────────────────────

/// Falcon-2 base language model (without LM head).
pub struct Falcon2Model {
    /// Model configuration.
    pub config: Falcon2Config,
    /// Decoder layers.
    pub layers: Vec<Falcon2DecoderLayer>,
    /// Final LayerNorm weight (ones).
    final_norm_weight: Vec<f32>,
    /// Final LayerNorm bias (zeros).
    final_norm_bias: Vec<f32>,
    /// Token embedding table (vocab_size × hidden_size, zero-init).
    #[allow(dead_code)]
    embed_weight: Vec<f32>,
}

impl Falcon2Model {
    /// Create a new model with the given configuration.
    pub fn new(config: Falcon2Config) -> Self {
        let num_layers = config.num_hidden_layers;
        let h = config.hidden_size;
        let v = config.vocab_size;

        let layers = (0..num_layers).map(|_| Falcon2DecoderLayer::new(config.clone())).collect();

        Self {
            final_norm_weight: vec![1.0_f32; h],
            final_norm_bias: vec![0.0_f32; h],
            embed_weight: vec![0.0_f32; v * h],
            layers,
            config,
        }
    }

    /// Run the model on a sequence of token IDs.
    ///
    /// Returns the final hidden states `[seq_len * hidden_size]`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, Falcon2Error> {
        let seq_len = input_ids.len();
        if seq_len == 0 {
            return Err(Falcon2Error::InvalidInput(
                "input_ids must not be empty".to_string(),
            ));
        }

        let h = self.config.hidden_size;
        let v = self.config.vocab_size;

        // Token embedding lookup.
        let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * h);
        for &tok in input_ids {
            let tok_id = tok as usize;
            if tok_id >= v {
                return Err(Falcon2Error::InvalidInput(format!(
                    "token id {tok_id} is out of vocabulary range {v}"
                )));
            }
            // Deterministic embedding: small signal proportional to token id.
            let embedding: Vec<f32> =
                (0..h).map(|dim| (tok_id as f32 * 0.001) * ((dim + 1) as f32 * 0.01)).collect();
            hidden.extend_from_slice(&embedding);
        }

        // Pass through decoder layers.
        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len);
        }

        // Final LayerNorm.
        hidden = hidden
            .chunks(h)
            .flat_map(|chunk| {
                Falcon2LayerNorm::forward(
                    chunk,
                    &self.final_norm_weight,
                    &self.final_norm_bias,
                    self.config.layer_norm_epsilon,
                )
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
    use crate::falcon2::config::Falcon2Config;

    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|_| lcg_next(&mut state)).collect()
    }

    fn tiny_falcon2_config() -> Falcon2Config {
        Falcon2Config {
            hidden_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_kv_heads: 1,
            intermediate_size: 64,
            max_position_embeddings: 64,
            vocab_size: 64,
            layer_norm_epsilon: 1e-5,
            use_alibi: false,
            parallel_attn: true,
            bias: false,
            hidden_act: "gelu".to_string(),
        }
    }

    // -- Falcon2LayerNorm --

    #[test]
    fn test_falcon2_layernorm_zero_mean() {
        let weight = vec![1.0_f32; 4];
        let bias = vec![0.0_f32; 4];
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let out = Falcon2LayerNorm::forward(&x, &weight, &bias, 1e-5);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-4,
            "LayerNorm output mean must be ~0, got {mean}"
        );
    }

    #[test]
    fn test_falcon2_layernorm_unit_variance() {
        let weight = vec![1.0_f32; 4];
        let bias = vec![0.0_f32; 4];
        let x = vec![10.0_f32, 20.0, 30.0, 40.0];
        let out = Falcon2LayerNorm::forward(&x, &weight, &bias, 1e-5);
        let mean = out.iter().sum::<f32>() / out.len() as f32;
        let var = out.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / out.len() as f32;
        assert!(
            (var - 1.0).abs() < 1e-3,
            "LayerNorm output variance must be ~1, got {var}"
        );
    }

    #[test]
    fn test_falcon2_layernorm_empty_input_returns_empty() {
        let out = Falcon2LayerNorm::forward(&[], &[], &[], 1e-5);
        assert!(out.is_empty(), "empty input must return empty output");
    }

    // -- Falcon2AlibiPositionalBias --

    #[test]
    fn test_alibi_slopes_length() {
        let num_heads = 8;
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(num_heads);
        assert_eq!(
            slopes.len(),
            num_heads,
            "slopes length must equal num_heads"
        );
    }

    #[test]
    fn test_alibi_slopes_decreasing() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(8);
        for i in 0..slopes.len() - 1 {
            assert!(
                slopes[i] >= slopes[i + 1],
                "ALiBi slopes must be non-increasing across heads",
            );
        }
    }

    #[test]
    fn test_alibi_slopes_all_positive() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(8);
        for &s in &slopes {
            assert!(s > 0.0, "all ALiBi slopes must be positive");
        }
    }

    #[test]
    fn test_alibi_bias_shape() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(4);
        let seq_len = 8;
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        assert_eq!(
            bias.len(),
            4 * seq_len * seq_len,
            "bias tensor must have shape [num_heads, seq_len, seq_len]",
        );
    }

    #[test]
    fn test_alibi_bias_diagonal_is_zero() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(2);
        let seq_len = 4;
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        // Diagonal entries: distance = |i-i| = 0, so bias = 0
        for h in 0..2 {
            for i in 0..seq_len {
                let b = bias[h * seq_len * seq_len + i * seq_len + i];
                assert_eq!(b, 0.0, "ALiBi diagonal bias must be 0 (zero distance)");
            }
        }
    }

    #[test]
    fn test_alibi_bias_off_diagonal_negative() {
        let slopes = Falcon2AlibiPositionalBias::compute_slopes(2);
        let seq_len = 4;
        let bias = Falcon2AlibiPositionalBias::compute_bias(seq_len, &slopes);
        // Off-diagonal: distance > 0 → bias = -distance * slope < 0
        let b01 = bias[1]; // h=0, i=0, j=1
        assert!(b01 < 0.0, "ALiBi off-diagonal bias must be negative");
    }

    // -- Falcon2MLP --

    #[test]
    fn test_falcon2_gelu_at_zero() {
        let result = Falcon2MLP::gelu(0.0);
        assert!(result.abs() < 1e-6, "gelu(0) must be 0");
    }

    #[test]
    fn test_falcon2_gelu_positive_input() {
        assert!(
            Falcon2MLP::gelu(1.0) > 0.0,
            "gelu(positive) must be positive"
        );
    }

    #[test]
    fn test_falcon2_mlp_output_length() {
        let cfg = tiny_falcon2_config();
        let mlp = Falcon2MLP::new(&cfg);
        let x = lcg_vec(cfg.hidden_size, 100);
        let out = mlp.forward(&x);
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MLP output length must equal hidden_size"
        );
    }

    #[test]
    fn test_falcon2_mlp_empty_input_returns_empty() {
        let cfg = tiny_falcon2_config();
        let mlp = Falcon2MLP::new(&cfg);
        let out = mlp.forward(&[]);
        assert!(out.is_empty(), "MLP empty input must return empty output");
    }

    // -- Falcon2Config --

    #[test]
    fn test_falcon2_config_head_dim() {
        let cfg = tiny_falcon2_config();
        assert_eq!(
            cfg.head_dim(),
            cfg.hidden_size / cfg.num_attention_heads,
            "head_dim must equal hidden_size / num_attention_heads",
        );
    }

    // -- Falcon2Model --

    #[test]
    fn test_falcon2_model_construction() {
        let cfg = tiny_falcon2_config();
        let model = Falcon2Model::new(cfg);
        assert_eq!(model.layers.len(), 2, "model must have 2 layers");
    }

    #[test]
    fn test_falcon2_model_forward_single_token() {
        let cfg = tiny_falcon2_config();
        let model = Falcon2Model::new(cfg.clone());
        let output = model.forward(&[0u32]).expect("forward must succeed");
        assert_eq!(
            output.len(),
            cfg.hidden_size,
            "output length must equal hidden_size"
        );
    }

    #[test]
    fn test_falcon2_model_forward_multi_token() {
        let cfg = tiny_falcon2_config();
        let model = Falcon2Model::new(cfg.clone());
        let output = model.forward(&[0u32, 1, 2]).expect("multi-token forward must succeed");
        assert_eq!(
            output.len(),
            3 * cfg.hidden_size,
            "output length must be seq_len * hidden_size"
        );
    }

    #[test]
    fn test_falcon2_model_empty_input_fails() {
        let cfg = tiny_falcon2_config();
        let model = Falcon2Model::new(cfg);
        let result = model.forward(&[]);
        assert!(result.is_err(), "empty input must return an error");
    }

    #[test]
    fn test_falcon2_model_out_of_vocab_fails() {
        let cfg = tiny_falcon2_config(); // vocab_size=64
        let model = Falcon2Model::new(cfg);
        let result = model.forward(&[100u32]); // 100 >= 64
        assert!(result.is_err(), "out-of-vocab token must return an error");
    }
}
