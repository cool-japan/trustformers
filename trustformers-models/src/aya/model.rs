use crate::aya::config::{AyaConfig, AyaError};

// ─── LayerNorm ───────────────────────────────────────────────────────────────

/// Standard LayerNorm (not RMSNorm) as used in Aya-23.
///
/// Computes: `(x - mean) / sqrt(var + eps) * weight + bias`.
#[derive(Debug, Clone)]
pub struct AyaLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl AyaLayerNorm {
    /// Create a new LayerNorm with `dim`-dimensional weight=1 and bias=0.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0_f32; dim],
            bias: vec![0.0_f32; dim],
            eps: eps as f32,
        }
    }

    /// Apply LayerNorm to `x` using the provided `weight` and `bias` slices.
    ///
    /// All three slices must have identical lengths.
    pub fn forward(
        x: &[f32],
        weight: &[f32],
        bias: &[f32],
        eps: f32,
    ) -> Result<Vec<f32>, AyaError> {
        if x.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        if x.len() != weight.len() || x.len() != bias.len() {
            return Err(AyaError::DimensionMismatch {
                expected: x.len(),
                got: if weight.len() != x.len() { weight.len() } else { bias.len() },
            });
        }
        let n = x.len() as f32;
        let mean: f32 = x.iter().sum::<f32>() / n;
        let var: f32 = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
        let std = (var + eps).sqrt();
        let out = x
            .iter()
            .zip(weight.iter())
            .zip(bias.iter())
            .map(|((v, w), b)| (v - mean) / std * w + b)
            .collect();
        Ok(out)
    }

    /// Apply this layer's stored weight and bias.
    pub fn apply(&self, x: &[f32]) -> Result<Vec<f32>, AyaError> {
        Self::forward(x, &self.weight, &self.bias, self.eps)
    }
}

// ─── Rotary Embedding ────────────────────────────────────────────────────────

/// Precomputed cosine and sine tables for RoPE.
#[derive(Debug, Clone)]
pub struct AyaRotaryEmbedding {
    cos_cache: Vec<f32>,
    sin_cache: Vec<f32>,
    max_position_embeddings: usize,
    half_dim: usize,
}

impl AyaRotaryEmbedding {
    /// Build cosine/sine tables up to `max_position_embeddings`.
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

    /// Apply RoPE in-place to a head vector of length `head_dim = 2 * half_dim`.
    pub fn rotate_head(&self, head: &mut [f32], pos: usize) -> Result<(), AyaError> {
        if pos >= self.max_position_embeddings {
            return Err(AyaError::InvalidConfig(format!(
                "position {} exceeds max_position_embeddings {}",
                pos, self.max_position_embeddings
            )));
        }
        let half = head.len() / 2;
        let start = pos * self.half_dim;
        for i in 0..half {
            let cos = self.cos_cache[start + i];
            let sin = self.sin_cache[start + i];
            let x0 = head[i];
            let x1 = head[i + half];
            head[i] = x0 * cos - x1 * sin;
            head[i + half] = x0 * sin + x1 * cos;
        }
        Ok(())
    }
}

// ─── Dense Layer ─────────────────────────────────────────────────────────────

/// Deterministic LCG weight initialisation.
fn init_weight(rows: usize, cols: usize, seed: u64) -> Vec<f32> {
    let n = rows * cols;
    let limit = (6.0_f64 / (rows + cols) as f64).sqrt() as f32;
    let mut state = seed;
    (0..n)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let t = (state >> 33) as f32 / u32::MAX as f32;
            t * 2.0 * limit - limit
        })
        .collect()
}

/// Simple dense (linear) layer.
#[derive(Debug, Clone)]
pub struct AyaDenseLayer {
    weight: Vec<f32>,
    bias: Option<Vec<f32>>,
    pub in_features: usize,
    pub out_features: usize,
}

impl AyaDenseLayer {
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

    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, AyaError> {
        if x.len() != self.in_features {
            return Err(AyaError::DimensionMismatch {
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

/// SiLU activation.
#[inline]
fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

// ─── Attention ───────────────────────────────────────────────────────────────

/// Multi-head attention with GQA and optional QK-normalization for Aya-23.
///
/// Applies `logit_scale` to the attention output.
#[derive(Debug, Clone)]
pub struct AyaAttention {
    q_proj: AyaDenseLayer,
    k_proj: AyaDenseLayer,
    v_proj: AyaDenseLayer,
    o_proj: AyaDenseLayer,
    /// Optional QK-norm applied to query/key heads.
    q_norm: Option<AyaLayerNorm>,
    k_norm: Option<AyaLayerNorm>,
    rotary: AyaRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    logit_scale: f32,
    #[allow(dead_code)]
    attention_dropout: f32,
}

impl AyaAttention {
    /// Construct from config.
    pub fn new(config: &AyaConfig) -> Result<Self, AyaError> {
        config.validate()?;
        let h = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;

        let (q_norm, k_norm) = if config.use_qk_norm {
            (
                Some(AyaLayerNorm::new(hd, config.layer_norm_eps)),
                Some(AyaLayerNorm::new(hd, config.layer_norm_eps)),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj: AyaDenseLayer::new(h, nh * hd, false, 0x0001_111A),
            k_proj: AyaDenseLayer::new(h, nkv * hd, false, 0x0002_222A),
            v_proj: AyaDenseLayer::new(h, nkv * hd, false, 0x0003_333A),
            o_proj: AyaDenseLayer::new(nh * hd, h, false, 0x0004_444A),
            q_norm,
            k_norm,
            rotary: AyaRotaryEmbedding::new(hd, config.max_position_embeddings, config.rope_theta),
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            logit_scale: config.logit_scale,
            attention_dropout: config.attention_dropout,
        })
    }

    /// Forward pass for flat `[seq_len * hidden_size]` input.
    pub fn forward(&self, hidden: &[f32], seq_len: usize) -> Result<Vec<f32>, AyaError> {
        let hidden_size = self.num_heads * self.head_dim;
        if hidden.len() != seq_len * hidden_size {
            return Err(AyaError::DimensionMismatch {
                expected: seq_len * hidden_size,
                got: hidden.len(),
            });
        }

        let scale = self.logit_scale / (self.head_dim as f32).sqrt();
        let kv_groups = self.num_heads / self.num_kv_heads;

        let mut q_all = Vec::with_capacity(seq_len * self.num_heads * self.head_dim);
        let mut k_all = Vec::with_capacity(seq_len * self.num_kv_heads * self.head_dim);
        let mut v_all = Vec::with_capacity(seq_len * self.num_kv_heads * self.head_dim);

        for t in 0..seq_len {
            let x = &hidden[t * hidden_size..(t + 1) * hidden_size];
            let mut q = self.q_proj.forward(x)?;
            let mut k = self.k_proj.forward(x)?;
            let v = self.v_proj.forward(x)?;

            // Apply optional QK-norm per head.
            if let (Some(qn), Some(kn)) = (&self.q_norm, &self.k_norm) {
                for h in 0..self.num_heads {
                    let s = h * self.head_dim;
                    let e = s + self.head_dim;
                    let normed = qn.apply(&q[s..e])?;
                    q[s..e].copy_from_slice(&normed);
                }
                for h in 0..self.num_kv_heads {
                    let s = h * self.head_dim;
                    let e = s + self.head_dim;
                    let normed = kn.apply(&k[s..e])?;
                    k[s..e].copy_from_slice(&normed);
                }
            }

            // Apply RoPE.
            for h in 0..self.num_heads {
                let s = h * self.head_dim;
                let e = s + self.head_dim;
                self.rotary.rotate_head(&mut q[s..e], t)?;
            }
            for h in 0..self.num_kv_heads {
                let s = h * self.head_dim;
                let e = s + self.head_dim;
                self.rotary.rotate_head(&mut k[s..e], t)?;
            }

            q_all.extend_from_slice(&q);
            k_all.extend_from_slice(&k);
            v_all.extend_from_slice(&v);
        }

        let mut output = vec![0.0_f32; seq_len * hidden_size];

        for t in 0..seq_len {
            let mut head_contexts = vec![0.0_f32; self.num_heads * self.head_dim];

            for h in 0..self.num_heads {
                let kv_h = h / kv_groups;
                let q_row = &q_all[t * self.num_heads * self.head_dim + h * self.head_dim
                    ..t * self.num_heads * self.head_dim + (h + 1) * self.head_dim];

                let mut scores = Vec::with_capacity(seq_len);
                for s in 0..seq_len {
                    let k_row = &k_all[s * self.num_kv_heads * self.head_dim + kv_h * self.head_dim
                        ..s * self.num_kv_heads * self.head_dim + (kv_h + 1) * self.head_dim];
                    let dot: f32 = q_row.iter().zip(k_row.iter()).map(|(a, b)| a * b).sum();
                    scores.push(dot * scale);
                }

                // Causal mask.
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

                let ctx = &mut head_contexts[h * self.head_dim..(h + 1) * self.head_dim];
                for s in 0..seq_len {
                    let v_row = &v_all[s * self.num_kv_heads * self.head_dim + kv_h * self.head_dim
                        ..s * self.num_kv_heads * self.head_dim + (kv_h + 1) * self.head_dim];
                    for d in 0..self.head_dim {
                        ctx[d] += attn_weights[s] * v_row[d];
                    }
                }
            }

            let proj = self.o_proj.forward(&head_contexts)?;
            output[t * hidden_size..(t + 1) * hidden_size].copy_from_slice(&proj);
        }

        Ok(output)
    }
}

// ─── MLP ─────────────────────────────────────────────────────────────────────

/// SwiGLU MLP: `down_proj(silu(gate_proj(x)) * up_proj(x))`.
#[derive(Debug, Clone)]
pub struct AyaMlp {
    gate_proj: AyaDenseLayer,
    up_proj: AyaDenseLayer,
    down_proj: AyaDenseLayer,
}

impl AyaMlp {
    /// Construct from config.
    pub fn new(config: &AyaConfig) -> Self {
        let h = config.hidden_size;
        let i = config.intermediate_size;
        Self {
            gate_proj: AyaDenseLayer::new(h, i, false, 0x0005_555A),
            up_proj: AyaDenseLayer::new(h, i, false, 0x0006_666A),
            down_proj: AyaDenseLayer::new(i, h, false, 0x0007_777A),
        }
    }

    /// Apply MLP to a single hidden vector.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, AyaError> {
        if x.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        let gate = self.gate_proj.forward(x)?;
        let up = self.up_proj.forward(x)?;
        let intermediate: Vec<f32> =
            gate.iter().zip(up.iter()).map(|(g, u)| silu(*g) * u).collect();
        self.down_proj.forward(&intermediate)
    }
}

// ─── Decoder Layer ───────────────────────────────────────────────────────────

/// A single Aya-23 transformer decoder layer.
///
/// Pre-LayerNorm layout:
/// 1. `input_layernorm(x)` → attention → add `x`
/// 2. `post_attention_layernorm(x)` → MLP → add `x`
#[derive(Debug, Clone)]
pub struct AyaDecoderLayer {
    input_layernorm: AyaLayerNorm,
    attention: AyaAttention,
    post_attention_layernorm: AyaLayerNorm,
    mlp: AyaMlp,
}

impl AyaDecoderLayer {
    /// Construct a decoder layer from config.
    pub fn new(config: &AyaConfig) -> Result<Self, AyaError> {
        Ok(Self {
            input_layernorm: AyaLayerNorm::new(config.hidden_size, config.layer_norm_eps),
            attention: AyaAttention::new(config)?,
            post_attention_layernorm: AyaLayerNorm::new(config.hidden_size, config.layer_norm_eps),
            mlp: AyaMlp::new(config),
        })
    }

    /// Process flat `[seq_len * hidden_size]` hidden states.
    pub fn forward(
        &self,
        hidden: &[f32],
        seq_len: usize,
        hidden_size: usize,
    ) -> Result<Vec<f32>, AyaError> {
        // ── Attention sub-layer ──────────────────────────────────────────────
        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let slice = &hidden[t * hidden_size..(t + 1) * hidden_size];
            let n = self.input_layernorm.apply(slice)?;
            normed.extend_from_slice(&n);
        }

        let attn_out = self.attention.forward(&normed, seq_len)?;

        let mut after_attn = Vec::with_capacity(hidden.len());
        for i in 0..hidden.len() {
            after_attn.push(hidden[i] + attn_out[i]);
        }

        // ── MLP sub-layer ────────────────────────────────────────────────────
        let mut normed2 = Vec::with_capacity(after_attn.len());
        for t in 0..seq_len {
            let slice = &after_attn[t * hidden_size..(t + 1) * hidden_size];
            let n = self.post_attention_layernorm.apply(slice)?;
            normed2.extend_from_slice(&n);
        }

        let mut out = Vec::with_capacity(after_attn.len());
        for t in 0..seq_len {
            let slice = &normed2[t * hidden_size..(t + 1) * hidden_size];
            let mlp_out = self.mlp.forward(slice)?;
            for i in 0..hidden_size {
                out.push(after_attn[t * hidden_size + i] + mlp_out[i]);
            }
        }

        Ok(out)
    }
}

// ─── Embedding ───────────────────────────────────────────────────────────────

/// Token embedding table.
#[derive(Debug, Clone)]
pub struct AyaEmbedding {
    table: Vec<f32>,
    vocab_size: usize,
    hidden_size: usize,
}

impl AyaEmbedding {
    pub fn new(vocab_size: usize, hidden_size: usize) -> Self {
        Self {
            table: vec![0.0_f32; vocab_size * hidden_size],
            vocab_size,
            hidden_size,
        }
    }

    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, AyaError> {
        if token_ids.is_empty() {
            return Err(AyaError::EmptyInput);
        }
        let mut out = Vec::with_capacity(token_ids.len() * self.hidden_size);
        for &id in token_ids {
            let id = id as usize;
            if id >= self.vocab_size {
                return Err(AyaError::InvalidConfig(format!(
                    "token id {} exceeds vocab_size {}",
                    id, self.vocab_size
                )));
            }
            let start = id * self.hidden_size;
            out.extend_from_slice(&self.table[start..start + self.hidden_size]);
        }
        Ok(out)
    }
}

// ─── Model ───────────────────────────────────────────────────────────────────

/// The core Aya-23 transformer model.
#[derive(Debug, Clone)]
pub struct AyaModel {
    embed_tokens: AyaEmbedding,
    layers: Vec<AyaDecoderLayer>,
    norm: AyaLayerNorm,
    hidden_size: usize,
}

impl AyaModel {
    /// Construct the model from config.
    pub fn new(config: &AyaConfig) -> Result<Self, AyaError> {
        config.validate()?;
        let embed_tokens = AyaEmbedding::new(config.vocab_size, config.hidden_size);
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(AyaDecoderLayer::new(config)?);
        }
        let norm = AyaLayerNorm::new(config.hidden_size, config.layer_norm_eps);
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass: embed → decoder stack → final LayerNorm.
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>, AyaError> {
        let seq_len = token_ids.len();
        let mut hidden = self.embed_tokens.forward(token_ids)?;

        for layer in &self.layers {
            hidden = layer.forward(&hidden, seq_len, self.hidden_size)?;
        }

        let mut normed = Vec::with_capacity(hidden.len());
        for t in 0..seq_len {
            let slice = &hidden[t * self.hidden_size..(t + 1) * self.hidden_size];
            let n = self.norm.apply(slice)?;
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
    use crate::aya::config::AyaConfig;

    // --- LCG ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_f32(state: &mut u64) -> f32 {
        let v = lcg_next(state);
        ((v >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    }

    /// Tiny config to keep tests fast (single layer, small dims).
    fn tiny_config() -> AyaConfig {
        AyaConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 4, // 16 / 4 = 4
            max_position_embeddings: 32,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            logit_scale: 0.0625,
            use_qk_norm: false,
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            supported_languages: vec!["en".to_string(), "fr".to_string()],
            tokenizer_class: "PreTrainedTokenizer".to_string(),
        }
    }

    // --- AyaLayerNorm ---

    #[test]
    fn test_aya_layer_norm_output_length() {
        let ln = AyaLayerNorm::new(8, 1e-5);
        let input: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let out = ln.apply(&input).expect("AyaLayerNorm must succeed");
        assert_eq!(out.len(), 8, "LayerNorm output length must match input");
    }

    #[test]
    fn test_aya_layer_norm_mean_near_zero() {
        let ln = AyaLayerNorm::new(16, 1e-5);
        let mut state: u64 = 42;
        let input: Vec<f32> = (0..16).map(|_| lcg_f32(&mut state)).collect();
        let out = ln.apply(&input).expect("AyaLayerNorm must succeed");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        assert!(
            mean.abs() < 1e-5,
            "LayerNorm output mean must be near 0, got {}",
            mean
        );
    }

    #[test]
    fn test_aya_layer_norm_std_near_one() {
        let ln = AyaLayerNorm::new(32, 1e-5);
        let mut state: u64 = 123;
        let input: Vec<f32> = (0..32).map(|_| lcg_f32(&mut state)).collect();
        let out = ln.apply(&input).expect("AyaLayerNorm must succeed");
        let mean: f32 = out.iter().sum::<f32>() / out.len() as f32;
        let var: f32 = out.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / out.len() as f32;
        let std = var.sqrt();
        assert!(
            (std - 1.0).abs() < 0.01,
            "LayerNorm output std must be near 1, got {}",
            std
        );
    }

    #[test]
    fn test_aya_layer_norm_empty_input_errors() {
        let result = AyaLayerNorm::forward(&[], &[1.0], &[0.0], 1e-5);
        assert!(result.is_err(), "LayerNorm must reject empty input");
    }

    #[test]
    fn test_aya_layer_norm_dimension_mismatch_errors() {
        let x = vec![1.0_f32; 4];
        let w = vec![1.0_f32; 3];
        let b = vec![0.0_f32; 4];
        let result = AyaLayerNorm::forward(&x, &w, &b, 1e-5);
        assert!(
            result.is_err(),
            "LayerNorm must reject mismatched dimensions"
        );
    }

    // --- AyaRotaryEmbedding ---

    #[test]
    fn test_aya_rotary_cache_size() {
        let rope = AyaRotaryEmbedding::new(8, 16, 10000.0);
        // cos_cache: max_position_embeddings * half_dim = 16 * 4 = 64
        assert_eq!(rope.cos_cache.len(), 64);
        assert_eq!(rope.sin_cache.len(), 64);
    }

    #[test]
    fn test_aya_rotary_rotate_head_no_error() {
        let rope = AyaRotaryEmbedding::new(8, 16, 10000.0);
        let mut head: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let result = rope.rotate_head(&mut head, 0);
        assert!(result.is_ok(), "rotate_head at pos 0 must succeed");
    }

    #[test]
    fn test_aya_rotary_rotate_head_out_of_range_errors() {
        let rope = AyaRotaryEmbedding::new(8, 4, 10000.0);
        let mut head: Vec<f32> = vec![0.0_f32; 8];
        let result = rope.rotate_head(&mut head, 100);
        assert!(
            result.is_err(),
            "rotate_head must error when pos exceeds max"
        );
    }

    // --- AyaDenseLayer ---

    #[test]
    fn test_aya_dense_layer_output_size() {
        let layer = AyaDenseLayer::new(8, 4, true, 0xABCD);
        let input: Vec<f32> = vec![1.0_f32; 8];
        let out = layer.forward(&input).expect("AyaDenseLayer forward must succeed");
        assert_eq!(
            out.len(),
            4,
            "Dense layer output size must equal out_features"
        );
    }

    #[test]
    fn test_aya_dense_layer_dimension_mismatch_errors() {
        let layer = AyaDenseLayer::new(8, 4, true, 0x1);
        let bad_input: Vec<f32> = vec![1.0_f32; 5];
        let result = layer.forward(&bad_input);
        assert!(result.is_err(), "Dense layer must reject wrong input size");
    }

    // --- AyaMlp (SwiGLU) ---

    #[test]
    fn test_aya_mlp_output_size() {
        let cfg = tiny_config();
        let mlp = AyaMlp::new(&cfg);
        let input: Vec<f32> = vec![0.5_f32; cfg.hidden_size];
        let out = mlp.forward(&input).expect("AyaMlp forward must succeed");
        assert_eq!(
            out.len(),
            cfg.hidden_size,
            "MLP output must preserve hidden_size"
        );
    }

    #[test]
    fn test_aya_mlp_empty_input_errors() {
        let cfg = tiny_config();
        let mlp = AyaMlp::new(&cfg);
        let result = mlp.forward(&[]);
        assert!(result.is_err(), "MLP must reject empty input");
    }

    // --- AyaEmbedding ---

    #[test]
    fn test_aya_embedding_output_shape() {
        let cfg = tiny_config();
        let emb = AyaEmbedding::new(cfg.vocab_size, cfg.hidden_size);
        let token_ids: Vec<u32> = vec![0, 1, 2, 3];
        let out = emb.forward(&token_ids).expect("AyaEmbedding forward must succeed");
        assert_eq!(
            out.len(),
            4 * cfg.hidden_size,
            "Embedding output must be seq_len * hidden_size"
        );
    }

    #[test]
    fn test_aya_embedding_empty_input_errors() {
        let emb = AyaEmbedding::new(64, 16);
        let result = emb.forward(&[]);
        assert!(result.is_err(), "Embedding must reject empty input");
    }

    #[test]
    fn test_aya_embedding_out_of_range_token_errors() {
        let emb = AyaEmbedding::new(64, 16);
        let result = emb.forward(&[100]);
        assert!(
            result.is_err(),
            "Embedding must reject token_id >= vocab_size"
        );
    }

    // --- AyaAttention (GQA) ---

    #[test]
    fn test_aya_attention_gqa_kv_groups() {
        let cfg = tiny_config();
        // num_attention_heads=4, num_key_value_heads=2 -> kv_groups=2
        let kv_groups = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(kv_groups, 2, "GQA: num_heads / num_kv_heads must equal 2");
    }

    #[test]
    fn test_aya_attention_output_size() {
        let cfg = tiny_config();
        let attn = AyaAttention::new(&cfg).expect("AyaAttention::new must succeed");
        let hidden_size = cfg.hidden_size;
        let seq_len = 3usize;
        let input: Vec<f32> = vec![0.1_f32; seq_len * hidden_size];
        let out = attn.forward(&input, seq_len).expect("AyaAttention forward must succeed");
        assert_eq!(
            out.len(),
            seq_len * hidden_size,
            "Attention output must be seq_len * hidden_size"
        );
    }

    // --- AyaModel ---

    #[test]
    fn test_aya_model_new() {
        let cfg = tiny_config();
        let model = AyaModel::new(&cfg).expect("AyaModel::new must succeed");
        assert_eq!(model.hidden_size(), cfg.hidden_size);
    }

    #[test]
    fn test_aya_model_forward_output_shape() {
        let cfg = tiny_config();
        let model = AyaModel::new(&cfg).expect("AyaModel::new must succeed");
        let token_ids: Vec<u32> = vec![1, 2, 3];
        let out = model.forward(&token_ids).expect("AyaModel forward must succeed");
        assert_eq!(
            out.len(),
            token_ids.len() * cfg.hidden_size,
            "AyaModel output must be seq_len * hidden_size"
        );
    }

    #[test]
    fn test_aya_model_single_token_forward() {
        let cfg = tiny_config();
        let model = AyaModel::new(&cfg).expect("AyaModel::new must succeed");
        let out = model.forward(&[5]).expect("single-token forward must succeed");
        assert_eq!(out.len(), cfg.hidden_size);
    }
}
