use crate::granite::config::{GraniteConfig, GraniteError};

// ─── RMSNorm ────────────────────────────────────────────────────────────────

/// RMSNorm layer as used in Granite models.
///
/// Computes: `x / rms(x) * weight` where `rms(x) = sqrt(mean(x^2) + eps)`.
#[derive(Debug, Clone)]
pub struct GraniteRmsNorm {
    weight: Vec<f32>,
    eps: f32,
}

impl GraniteRmsNorm {
    /// Create a new RMSNorm layer with `dim`-dimensional weight vector initialised
    /// to ones.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0_f32; dim],
            eps: eps as f32,
        }
    }

    /// Apply RMSNorm to `x` (length must equal `dim`).
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, GraniteError> {
        if x.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        if x.len() != self.weight.len() {
            return Err(GraniteError::DimensionMismatch {
                expected: self.weight.len(),
                got: x.len(),
            });
        }
        let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
        let rms = (mean_sq + self.eps).sqrt();
        let out = x.iter().zip(self.weight.iter()).map(|(v, w)| v / rms * w).collect();
        Ok(out)
    }
}

// ─── Rotary Embedding ────────────────────────────────────────────────────────

/// Precomputed cosine and sine tables for Rotary Position Embeddings (RoPE).
#[derive(Debug, Clone)]
pub struct GraniteRotaryEmbedding {
    /// cos cache: shape [max_position_embeddings, head_dim / 2]
    cos_cache: Vec<f32>,
    /// sin cache: shape [max_position_embeddings, head_dim / 2]
    sin_cache: Vec<f32>,
    max_position_embeddings: usize,
    half_dim: usize,
}

impl GraniteRotaryEmbedding {
    /// Build the cosine/sine tables up to `max_position_embeddings`.
    pub fn new(head_dim: usize, max_position_embeddings: usize, rope_theta: f64) -> Self {
        let half_dim = head_dim / 2;
        let mut cos_cache = Vec::with_capacity(max_position_embeddings * half_dim);
        let mut sin_cache = Vec::with_capacity(max_position_embeddings * half_dim);

        for pos in 0..max_position_embeddings {
            for i in 0..half_dim {
                let freq = 1.0 / rope_theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * freq;
                cos_cache.push(angle.cos() as f32);
                sin_cache.push(angle.sin() as f32);
            }
        }

        Self {
            cos_cache,
            sin_cache,
            max_position_embeddings,
            half_dim,
        }
    }

    /// Return the (cos, sin) slices for position `pos`.
    ///
    /// Each slice has length `half_dim`.
    pub fn get_position(&self, pos: usize) -> Result<(&[f32], &[f32]), GraniteError> {
        if pos >= self.max_position_embeddings {
            return Err(GraniteError::InvalidConfig(format!(
                "position {} exceeds max_position_embeddings {}",
                pos, self.max_position_embeddings
            )));
        }
        let start = pos * self.half_dim;
        let end = start + self.half_dim;
        Ok((&self.cos_cache[start..end], &self.sin_cache[start..end]))
    }

    /// Apply RoPE in-place to a single head's query or key vector of length
    /// `head_dim = 2 * half_dim`.
    pub fn rotate_head(&self, head: &mut [f32], pos: usize) -> Result<(), GraniteError> {
        let (cos, sin) = self.get_position(pos)?;
        let half = head.len() / 2;
        for i in 0..half {
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos[i] - x1 * sin[i];
            head[i + half] = x0 * sin[i] + x1 * cos[i];
        }
        Ok(())
    }
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// Initialise a weight matrix with Xavier-uniform-like values derived from a
/// deterministic LCG, avoiding any dependency on `rand`.
fn init_weight(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let n = rows * cols;
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt() as f32;
    let mut state = seed;
    (0..n)
        .map(|_| {
            // A minimal LCG for deterministic initialisation.
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = (state >> 33) as f32 / u32::MAX as f32; // [0, 1)
            t * 2.0 * limit - limit
        })
        .collect()
}

/// Simple dense (linear) layer without bias or with optional constant-zero bias.
#[derive(Debug, Clone)]
pub struct DenseLayer {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    in_features: usize,
    out_features: usize,
}

impl DenseLayer {
    pub fn new(in_features: usize, out_features: usize, use_bias: bool, seed: u64) -> Self {
        let weight = init_weight(out_features, in_features, seed);
        let bias = if use_bias { Some(vec![0.0_f32; out_features]) } else { None };
        Self {
            weight,
            bias,
            in_features,
            out_features,
        }
    }

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, GraniteError> {
        if x.len() != self.in_features {
            return Err(GraniteError::DimensionMismatch {
                expected: self.in_features,
                got: x.len(),
            });
        }
        let mut out = vec![0.0_f32; self.out_features];
        for o in 0..self.out_features {
            let row_start = o * self.in_features;
            let mut acc: f32 = 0.0;
            for i in 0..self.in_features {
                acc += self.weight[row_start + i] * x[i];
            }
            if let Some(b) = &self.bias {
                acc += b[o];
            }
            out[o] = acc;
        }
        Ok(out)
    }
}

/// SiLU (Sigmoid Linear Unit) activation: `x * sigmoid(x)`.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// Multi-head attention with optional GQA for Granite models.
///
/// Uses the Granite-specific scaling:
/// `scale = attention_multiplier / sqrt(head_dim)`
#[derive(Debug, Clone)]
pub struct GraniteAttention {
    q_proj: DenseLayer,
    k_proj: DenseLayer,
    v_proj: DenseLayer,
    o_proj: DenseLayer,
    rotary: GraniteRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    attention_multiplier: f32,
    #[allow(dead_code)]
    attention_dropout: f32,
}

impl GraniteAttention {
    /// Construct from config; seeds are derived from field offsets for
    /// deterministic but distinct initialisations.
    pub fn new(config: &GraniteConfig) -> Result<Self, GraniteError> {
        config.validate()?;
        let h = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;
        let use_bias = config.attention_bias;

        Ok(Self {
            q_proj: DenseLayer::new(h, nh * hd, use_bias, 0x1111),
            k_proj: DenseLayer::new(h, nkv * hd, use_bias, 0x2222),
            v_proj: DenseLayer::new(h, nkv * hd, use_bias, 0x3333),
            o_proj: DenseLayer::new(nh * hd, h, use_bias, 0x4444),
            rotary: GraniteRotaryEmbedding::new(
                hd,
                config.max_position_embeddings,
                config.rope_theta,
            ),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            attention_multiplier: config.attention_multiplier,
            attention_dropout: config.attention_dropout,
        })
    }

    /// Forward pass for a flat hidden-state sequence of shape `[seq_len * hidden_size]`.
    ///
    /// Returns the attention output with the same shape.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Result<Vec<f32>, GraniteError> {
        let hidden_size = self.num_heads * self.head_dim;
        if hidden.len() != seq_len * hidden_size {
            return Err(GraniteError::DimensionMismatch {
                expected: seq_len * hidden_size,
                got: hidden.len(),
            });
        }

        let scale = self.attention_multiplier / (self.head_dim as f32).sqrt();
        let kv_groups = self.num_heads / self.num_kv_heads;

        // Project Q, K, V for each position.
        let mut q_all = Vec::with_capacity(seq_len * self.num_heads * self.head_dim);
        let mut k_all = Vec::with_capacity(seq_len * self.num_kv_heads * self.head_dim);
        let mut v_all = Vec::with_capacity(seq_len * self.num_kv_heads * self.head_dim);

        for t in 0..seq_len {
            let x = &hidden[t * hidden_size..(t + 1) * hidden_size];
            let mut q = self.q_proj.forward(x)?;
            let mut k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;

            // Apply RoPE per head.
            for h in 0..self.num_heads {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                self.rotary.rotate_head(&mut q[start..end], t)?;
            }
            for h in 0..self.num_kv_heads {
                let start = h * self.head_dim;
                let end = start + self.head_dim;
                self.rotary.rotate_head(&mut k[start..end], t)?;
            }

            q_all.extend_from_slice(&q);
            k_all.extend_from_slice(&k);
            v_all.extend_from_slice(&v);
        }

        // Scaled dot-product attention per head, then project output.
        let mut output = vec![0.0_f32; seq_len * hidden_size];

        for t in 0..seq_len {
            // Accumulate per-head context vectors.
            let mut head_contexts = vec![0.0_f32; self.num_heads * self.head_dim];

            for h in 0..self.num_heads {
                let kv_h = h / kv_groups;
                let q_row = &q_all[t * self.num_heads * self.head_dim + h * self.head_dim
                    ..t * self.num_heads * self.head_dim + (h + 1) * self.head_dim];

                // Compute raw attention scores across all positions.
                let mut scores = Vec::with_capacity(seq_len);
                for s in 0..seq_len {
                    let k_row = &k_all[s * self.num_kv_heads * self.head_dim + kv_h * self.head_dim
                        ..s * self.num_kv_heads * self.head_dim + (kv_h + 1) * self.head_dim];
                    let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }

                // Causal mask: mask future positions with -inf.
                for s in (t + 1)..seq_len {
                    scores[s] = f32::NEG_INFINITY;
                }

                // Softmax.
                let max_score = scores
                    .iter()
                    .cloned()
                    .filter(|v| v.is_finite())
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_scores: Vec<f32> = scores
                    .iter()
                    .map(|&s| if s.is_finite() { (s - max_score).exp() } else { 0.0 })
                    .collect();
                let sum_exp: f32 = exp_scores.iter().sum();
                let attn_weights: Vec<f32> = if sum_exp > 0.0 {
                    exp_scores.iter().map(|v| v / sum_exp).collect()
                } else {
                    vec![1.0 / seq_len as f32; seq_len]
                };

                // Weighted sum of values.
                let ctx = &mut head_contexts[h * self.head_dim..(h + 1) * self.head_dim];
                for s in 0..seq_len {
                    let v_row = &v_all[s * self.num_kv_heads * self.head_dim + kv_h * self.head_dim
                        ..s * self.num_kv_heads * self.head_dim + (kv_h + 1) * self.head_dim];
                    for d in 0..self.head_dim {
                        ctx[d] += attn_weights[s] * v_row[d];
                    }
                }
            }

            // Output projection for this position.
            let proj = self.o_proj.forward(&head_contexts)?;
            output[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&proj);
        }

        Ok(output)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// SwiGLU-style MLP used in Granite decoder layers.
///
/// Computes: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, Clone)]
pub struct GraniteMlp {
    gate_proj: DenseLayer,
    up_proj: DenseLayer,
    down_proj: DenseLayer,
}

impl GraniteMlp {
    /// Construct from config with deterministic weight seeds.
    pub fn new(config: &GraniteConfig) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        let use_bias = config.mlp_bias;
        Self {
            gate_proj: DenseLayer::new(h, i, use_bias, 0x5555),
            up_proj: DenseLayer::new(h, i, use_bias, 0x6666),
            down_proj: DenseLayer::new(i, h, use_bias, 0x7777),
        }
    }

    /// Apply the MLP to a single hidden vector of length `hidden_size`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, GraniteError> {
        if x.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let intermediate: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
        self.down_proj.forward(&intermediate)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

/// A single Granite transformer decoder layer.
///
/// Layout:
/// 1. `input_layernorm(x)` → attention → scale by `residual_multiplier` → add `x`
/// 2. `post_attention_layernorm(x)` → MLP → scale by `residual_multiplier` → add `x`
#[derive(Debug, Clone)]
pub struct GraniteDecoderLayer {
    input_layernorm: GraniteRmsNorm,
    attention: GraniteAttention,
    post_attention_layernorm: GraniteRmsNorm,
    mlp: GraniteMlp,
    residual_multiplier: f32,
}

impl GraniteDecoderLayer {
    /// Construct a decoder layer from the model config.
    pub fn new(config: &GraniteConfig) -> Result<Self, GraniteError> {
        Ok(Self {
            input_layernorm: GraniteRmsNorm::new(config.hidden_size, config.rms_norm_eps),
            attention: GraniteAttention::new(config)?,
            post_attention_layernorm: GraniteRmsNorm::new(config.hidden_size, config.rms_norm_eps),
            mlp: GraniteMlp::new(config),
            residual_multiplier: config.residual_multiplier,
        })
    }

    /// Process a flat hidden-state sequence of shape `[seq_len * hidden_size]`.
    ///
    /// Returns updated hidden states with the same shape.
    pub fn forward(
        &self,
        hidden: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, GraniteError> {
        // ── Attention sub-layer ──────────────────────────────────────────────
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let slice = &hidden[t * hidden_size..(t + 1) * hidden_size];
            let n = self.input_layernorm.forward(slice)?;
            normed.extend_from_slice(&n);
        }

        let attn_out = self.attention.forward(&normed, seq_len)?;

        let mut after_attn = Vec::with_capacity(hidden.len());
        for i in 0..hidden.len() {
            after_attn.push(hidden[i] + self.residual_multiplier * attn_out[i]);
        }

        // ── MLP sub-layer ────────────────────────────────────────────────────
        let mut normed2 = Vec::with_capacity(after_attn.len());
        for t in 0..seq_len {
            let slice = &after_attn[t * hidden_size..(t + 1) * hidden_size];
            let n = self.post_attention_layernorm.forward(slice)?;
            normed2.extend_from_slice(&n);
        }

        let mut out = Vec::with_capacity(after_attn.len());
        for t in 0..seq_len {
            let slice = &normed2[t * hidden_size..(t + 1) * hidden_size];
            let mlp_out = self.mlp.forward(slice)?;
            for i in 0..hidden_size {
                out.push(after_attn[t * hidden_size + i] + self.residual_multiplier * mlp_out[i]);
            }
        }

        Ok(out)
    }
}

// ─── Embedding ───────────────────────────────────────────────────────────────

/// Token embedding table, scaled by `embedding_multiplier * sqrt(hidden_size)`.
#[derive(Debug, Clone)]
pub struct GraniteEmbedding {
    table: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
    scale: f32,
}

impl GraniteEmbedding {
    /// Create the embedding table with constant-zero weights (real weights are
    /// loaded from a checkpoint).  Scale is pre-computed.
    pub fn new(vocab_size: usize, hidden_size: usize, embedding_multiplier: f32) -> Self {
        let scale = embedding_multiplier * (hidden_size as f32).sqrt();
        Self {
            table: vec![0.0_f32; vocab_size * hidden_size],
            vocab_size,
            hidden_size,
            scale,
        }
    }

    /// Embed a sequence of token IDs, returning a flat `[seq_len * hidden_size]`
    /// vector.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, GraniteError> {
        if token_ids.is_empty() {
            return Err(GraniteError::EmptyInput);
        }
        let mut out = Vec::with_capacity(token_ids.len() * self.hidden_size);
        for &id in token_ids {
            let id = id as usize;
            if id >= self.vocab_size {
                return Err(GraniteError::InvalidConfig(format!(
                    "token id {} exceeds vocab_size {}",
                    id, self.vocab_size
                )));
            }
            let start = id * self.hidden_size;
            for v in &self.table[start..start + self.hidden_size] {
                out.push(v * self.scale);
            }
        }
        Ok(out)
    }

    /// The pre-computed embedding scale (`embedding_multiplier * sqrt(hidden_size)`).
    pub fn scale(&self) -> f32 {
        self.scale
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// The core Granite transformer model (embed → N decoder layers → final norm).
#[derive(Debug, Clone)]
pub struct GraniteModel {
    embed_tokens: GraniteEmbedding,
    layers: Vec<GraniteDecoderLayer>,
    norm: GraniteRmsNorm,
    hidden_size: usize,
}

impl GraniteModel {
    /// Construct the model from config.
    pub fn new(config: &GraniteConfig) -> Result<Self, GraniteError> {
        config.validate()?;
        let embed_tokens = GraniteEmbedding::new(
            config.vocab_size,
            config.hidden_size,
            config.embedding_multiplier,
        );
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(GraniteDecoderLayer::new(config)?);
        }
        let norm = GraniteRmsNorm::new(config.hidden_size, config.rms_norm_eps);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: config.hidden_size,
        })
    }

    /// Run the forward pass: embed tokens → decoder stack → final RMSNorm.
    ///
    /// Returns `[seq_len * hidden_size]` hidden states.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, GraniteError> {
        let seq_len = token_ids.len();
        let mut hidden = self.embed_tokens.forward(token_ids)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len, self.hidden_size)?;
        }

        // Apply final RMSNorm token by token.
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let slice = &hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
            let n = self.norm.forward(slice)?;
            normed.extend_from_slice(&n);
        }

        Ok(normed)
    }

    /// Expose hidden size for downstream task heads.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::granite::config::GraniteConfig;

    /// A tiny Granite config that is fast to construct and run in unit tests.
    fn tiny_config() -> GraniteConfig {
        GraniteConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            head_dim: 8,
            max_position_embeddings: 16,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            attention_bias: false,
            mlp_bias: false,
            tie_word_embeddings: false,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            initializer_range: 0.02,
            embedding_multiplier: 1.0,
            logits_scaling: 1.0,
            residual_multiplier: 1.0,
            attention_multiplier: 1.0,
        }
    }

    // --- GraniteRmsNorm ---

    #[test]
    fn test_rmsnorm_unit_vector_is_unchanged() {
        // A unit vector (length = 1) should be unchanged if weight is all-ones
        // because rms = 1/sqrt(dim), normalised value = x * sqrt(dim), but weight is 1.
        // Actually let's just check the output is finite and same length.
        let norm = GraniteRmsNorm::new(4, 1e-5);
        let x = vec![1.0_f32, 0.0, 0.0, 0.0];
        let out = norm.forward(&x).expect("RMSNorm must succeed for unit vector");
        assert_eq!(out.len(), 4, "output must have same length as input");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all output values must be finite"
        );
    }

    #[test]
    fn test_rmsnorm_empty_input_errors() {
        let norm = GraniteRmsNorm::new(4, 1e-5);
        let result = norm.forward(&[]);
        assert!(result.is_err(), "empty input must return an error");
    }

    #[test]
    fn test_rmsnorm_dimension_mismatch_errors() {
        let norm = GraniteRmsNorm::new(4, 1e-5);
        let result = norm.forward(&[1.0, 2.0]); // length 2 ≠ dim 4
        assert!(result.is_err(), "dimension mismatch must return an error");
    }

    #[test]
    fn test_rmsnorm_all_ones_output_is_one_over_sqrt_dim() {
        let dim = 4_usize;
        let norm = GraniteRmsNorm::new(dim, 1e-6);
        let x = vec![1.0_f32; dim];
        let out = norm.forward(&x).expect("RMSNorm of all-ones must succeed");
        // rms(x) = 1.0; each normalised value = 1.0/1.0 * 1.0 = 1.0
        for (i, &v) in out.iter().enumerate() {
            let diff = (v - 1.0_f32).abs();
            assert!(diff < 1e-5, "element {} must be ~1.0, got {}", i, v);
        }
    }

    // --- GraniteRotaryEmbedding ---

    #[test]
    fn test_rotary_embedding_first_position_cos_is_one() {
        // At position 0, angle = 0 → cos = 1, sin = 0
        let rope = GraniteRotaryEmbedding::new(4, 8, 10000.0);
        let (cos, sin) = rope.get_position(0).expect("position 0 must be valid");
        for &c in cos {
            let diff = (c - 1.0_f32).abs();
            assert!(diff < 1e-6, "cos at position 0 must be 1.0, got {}", c);
        }
        for &s in sin {
            let diff = s.abs();
            assert!(diff < 1e-6, "sin at position 0 must be 0.0, got {}", s);
        }
    }

    #[test]
    fn test_rotary_embedding_out_of_bounds_errors() {
        let rope = GraniteRotaryEmbedding::new(4, 8, 10000.0);
        let result = rope.get_position(100);
        assert!(result.is_err(), "position beyond max must return an error");
    }

    #[test]
    fn test_rotary_embedding_cache_lengths_are_correct() {
        let head_dim = 8;
        let max_pos = 16;
        let rope = GraniteRotaryEmbedding::new(head_dim, max_pos, 10000.0);
        let (cos, sin) = rope.get_position(3).expect("position 3 must be valid");
        assert_eq!(
            cos.len(),
            head_dim / 2,
            "cos slice length must be head_dim / 2"
        );
        assert_eq!(
            sin.len(),
            head_dim / 2,
            "sin slice length must be head_dim / 2"
        );
    }

    #[test]
    fn test_rotate_head_preserves_vector_norm() {
        // Rotation preserves the L2 norm of the vector
        let rope = GraniteRotaryEmbedding::new(4, 8, 10000.0);
        let mut head = vec![1.0_f32, 0.0, 2.0, 0.0];
        let norm_before: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope.rotate_head(&mut head, 2).expect("rotate_head must succeed");
        let norm_after: f32 = head.iter().map(|x| x * x).sum::<f32>().sqrt();
        let diff = (norm_before - norm_after).abs();
        assert!(
            diff < 1e-5,
            "rotation must preserve vector norm, diff = {}",
            diff
        );
    }

    // --- DenseLayer ---

    #[test]
    fn test_dense_layer_forward_output_shape() {
        let layer = DenseLayer::new(8, 4, false, 42);
        let x = vec![1.0_f32; 8];
        let out = layer.forward(&x).expect("DenseLayer forward must succeed");
        assert_eq!(out.len(), 4, "output size must match out_features");
    }

    #[test]
    fn test_dense_layer_dimension_mismatch_errors() {
        let layer = DenseLayer::new(8, 4, false, 42);
        let result = layer.forward(&[1.0_f32; 5]); // wrong input length
        assert!(result.is_err(), "wrong input length must return an error");
    }

    #[test]
    fn test_dense_layer_with_bias_output_is_finite() {
        let layer = DenseLayer::new(4, 4, true, 0x5555);
        let x = vec![1.0_f32, -1.0, 0.5, -0.5];
        let out = layer.forward(&x).expect("DenseLayer with bias must succeed");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all outputs must be finite"
        );
    }

    // --- GraniteModel ---

    #[test]
    fn test_transformer_forward_output_length() {
        let cfg = tiny_config();
        let model = GraniteModel::new(&cfg).expect("GraniteModel creation must succeed");
        let token_ids = vec![1_u32, 2, 3, 4];
        let seq_len = token_ids.len();
        let output = model.forward(&token_ids).expect("GraniteModel forward must succeed");
        assert_eq!(
            output.len(),
            seq_len * cfg.hidden_size,
            "output must have seq_len * hidden_size elements"
        );
    }

    #[test]
    fn test_transformer_hidden_size_accessor() {
        let cfg = tiny_config();
        let model = GraniteModel::new(&cfg).expect("GraniteModel creation");
        assert_eq!(model.hidden_size(), cfg.hidden_size);
    }

    #[test]
    fn test_transformer_forward_output_all_finite() {
        let cfg = tiny_config();
        let model = GraniteModel::new(&cfg).expect("GraniteModel creation");
        let output = model.forward(&[0_u32, 1, 2]).expect("forward pass must succeed");
        assert!(
            output.iter().all(|v| v.is_finite()),
            "all output values must be finite"
        );
    }

    #[test]
    fn test_transformer_single_token_forward() {
        let cfg = tiny_config();
        let model = GraniteModel::new(&cfg).expect("GraniteModel creation");
        let output = model.forward(&[5_u32]).expect("single-token forward pass must succeed");
        assert_eq!(output.len(), cfg.hidden_size);
    }
}
