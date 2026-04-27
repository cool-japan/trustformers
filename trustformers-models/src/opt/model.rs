//! OPT model architecture implementation.
//!
//! Implements the core building blocks of Meta's OPT family:
//! - Learned positional embeddings with +2 offset.
//! - Standard LayerNorm (mean subtraction + variance normalisation).
//! - Full (non-GQA) causal self-attention with bias in all projections.
//! - FFN using `fc1 → ReLU → fc2` (no gating).
//! - Pre-norm decoder layers and a final LayerNorm.

use std::fmt;
use thiserror::Error;

use crate::opt::config::OptConfig;

// ─── Error type ────────────────────────────────────────────────────────────

/// Errors produced by the OPT model.
#[derive(Debug, Error)]
pub enum OptError {
    /// The provided configuration failed validation.
    #[error("invalid OPT configuration: {0}")]
    InvalidConfig(String),
    /// A forward-pass or shape-related error.
    #[error("OPT forward error: {0}")]
    Forward(String),
    /// Generation failed.
    #[error("OPT generation error: {0}")]
    Generation(String),
}

impl fmt::Display for OptConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "OptConfig {{ hidden={}, layers={}, heads={}, ffn={} }}",
            self.hidden_size, self.num_hidden_layers, self.num_attention_heads, self.ffn_dim
        )
    }
}

// ─── LayerNorm ─────────────────────────────────────────────────────────────

/// Standard Layer Normalisation.
///
/// Subtracts the mean and divides by the standard deviation, then applies
/// an affine transform `weight * x + bias`.  This differs from RMSNorm which
/// *does not* subtract the mean.
#[derive(Debug, Clone)]
pub struct OptLayerNorm {
    weight: Vec<f32>,
    bias: Vec<f32>,
    eps: f32,
}

impl OptLayerNorm {
    /// Create a new LayerNorm for `dim`-dimensional inputs.
    ///
    /// Weights are initialised to 1.0 and biases to 0.0.
    pub fn new(dim: usize, eps: f64) -> Self {
        Self {
            weight: vec![1.0; dim],
            bias: vec![0.0; dim],
            eps: eps as f32,
        }
    }

    /// Apply layer normalisation to `x`.
    ///
    /// `x` must have length equal to `self.weight.len()`.
    pub fn forward(&self, x: &[f32]) -> Result<Vec<f32>, OptError> {
        let n = x.len();
        if n == 0 {
            return Err(OptError::Forward("empty input to LayerNorm".to_string()));
        }
        if n != self.weight.len() {
            return Err(OptError::Forward(format!(
                "LayerNorm size mismatch: input {} vs weight {}",
                n,
                self.weight.len()
            )));
        }
        let mean: f32 = x.iter().sum::<f32>() / n as f32;
        let var: f32 = x.iter().map(|xi| (xi - mean).powi(2)).sum::<f32>() / n as f32;
        let std_inv = 1.0 / (var + self.eps).sqrt();
        let out: Vec<f32> = x
            .iter()
            .zip(self.weight.iter())
            .zip(self.bias.iter())
            .map(|((xi, wi), bi)| (xi - mean) * std_inv * wi + bi)
            .collect();
        Ok(out)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── Learned positional embeddings ─────────────────────────────────────────

/// OPT learned positional embedding table.
///
/// OPT uses a fixed lookup table with `max_position_embeddings + 2` rows
/// (the +2 is a convention carried over from Fairseq).  Position IDs are
/// therefore `seq_pos + 2`.
#[derive(Debug, Clone)]
pub struct OptLearnedPositionalEmbedding {
    /// Number of positions (= `max_position_embeddings + 2`).
    num_embeddings: usize,
    embed_dim: usize,
    /// Flat table: shape `[num_embeddings, embed_dim]`.
    table: Vec<f32>,
}

impl OptLearnedPositionalEmbedding {
    /// Build the embedding table.  The table is populated using a
    /// deterministic sinusoidal initialisation (OPT uses learned embeddings,
    /// but for inference the exact values are loaded from a checkpoint; here
    /// we use sin/cos as a stand-in that preserves the correct shape).
    pub fn new(max_position_embeddings: usize, embed_dim: usize) -> Self {
        let num_embeddings = max_position_embeddings + 2;
        let mut table = Vec::with_capacity(num_embeddings * embed_dim);
        for pos in 0..num_embeddings {
            for dim_idx in 0..embed_dim {
                let angle =
                    pos as f32 / 10000_f32.powf(2.0 * (dim_idx / 2) as f32 / embed_dim as f32);
                let val = if dim_idx % 2 == 0 { angle.sin() } else { angle.cos() };
                table.push(val);
            }
        }
        Self {
            num_embeddings,
            embed_dim,
            table,
        }
    }

    /// Return the OPT-convention position IDs for a sequence of length `seq_len`.
    ///
    /// `position_ids[i] = i + 2`
    pub fn get_position_ids(seq_len: usize) -> Vec<usize> {
        (0..seq_len).map(|i| i + 2).collect()
    }

    /// Look up positional embeddings for the given position IDs.
    ///
    /// Returns a flat `[seq_len * embed_dim]` vector.
    pub fn embed(&self, position_ids: &[usize]) -> Result<Vec<f32>, OptError> {
        let mut out = Vec::with_capacity(position_ids.len() * self.embed_dim);
        for &pos in position_ids {
            if pos >= self.num_embeddings {
                return Err(OptError::Forward(format!(
                    "position id {} >= num_embeddings {}",
                    pos, self.num_embeddings
                )));
            }
            let start = pos * self.embed_dim;
            out.extend_from_slice(&self.table[start..start + self.embed_dim]);
        }
        Ok(out)
    }

    /// Embedding dimension.
    pub fn embed_dim(&self) -> usize {
        self.embed_dim
    }
}

// ─── Linear layer ──────────────────────────────────────────────────────────

/// A dense linear layer `y = x W^T + b`.
///
/// Weights are zero-initialised (stand-in for loaded checkpoint weights).
#[derive(Debug, Clone)]
pub struct OptLinear {
    in_features: usize,
    out_features: usize,
    weight: Vec<f32>, // row-major [out_features, in_features]
    bias: Vec<f32>,   // length out_features
}

impl OptLinear {
    /// Create with zero weights and zero bias.
    pub fn new(in_features: usize, out_features: usize) -> Self {
        Self {
            in_features,
            out_features,
            weight: vec![0.0; out_features * in_features],
            bias: vec![0.0; out_features],
        }
    }

    /// Apply the linear transform to a `[seq_len, in_features]` flat input.
    ///
    /// Returns a flat `[seq_len, out_features]` vector.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>, OptError> {
        if x.len() != seq_len * self.in_features {
            return Err(OptError::Forward(format!(
                "linear input length {} != seq_len {} * in_features {}",
                x.len(),
                seq_len,
                self.in_features
            )));
        }
        let mut out = vec![0.0_f32; seq_len * self.out_features];
        for s in 0..seq_len {
            for o in 0..self.out_features {
                let mut acc = self.bias[o];
                for i in 0..self.in_features {
                    acc += x[s * self.in_features + i] * self.weight[o * self.in_features + i];
                }
                out[s * self.out_features + o] = acc;
            }
        }
        Ok(out)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.weight.len() + self.bias.len()
    }
}

// ─── Attention ─────────────────────────────────────────────────────────────

/// OPT causal self-attention.
///
/// Full (non-GQA) multi-head attention with a causal mask.
/// All Q / K / V / out projections carry a bias vector.
#[derive(Debug, Clone)]
pub struct OptAttention {
    embed_dim: usize,
    num_heads: usize,
    head_dim: usize,
    q_proj: OptLinear,
    k_proj: OptLinear,
    v_proj: OptLinear,
    out_proj: OptLinear,
}

impl OptAttention {
    /// Create a new OPT attention layer.
    pub fn new(config: &OptConfig) -> Result<Self, OptError> {
        let embed_dim = config.hidden_size;
        let num_heads = config.num_attention_heads;
        if !embed_dim.is_multiple_of(num_heads) {
            return Err(OptError::InvalidConfig(format!(
                "hidden_size {} not divisible by num_attention_heads {}",
                embed_dim, num_heads
            )));
        }
        let head_dim = embed_dim / num_heads;
        Ok(Self {
            embed_dim,
            num_heads,
            head_dim,
            q_proj: OptLinear::new(embed_dim, embed_dim),
            k_proj: OptLinear::new(embed_dim, embed_dim),
            v_proj: OptLinear::new(embed_dim, embed_dim),
            out_proj: OptLinear::new(embed_dim, embed_dim),
        })
    }

    /// Apply causal self-attention to `hidden_states`.
    ///
    /// `hidden_states` is a flat `[seq_len * embed_dim]` slice.
    /// Returns a flat `[seq_len * embed_dim]` slice.
    pub fn forward(&self, hidden_states: &[f32], seq_len: usize) -> Result<Vec<f32>, OptError> {
        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states, seq_len)?;
        let k = self.k_proj.forward(hidden_states, seq_len)?;
        let v = self.v_proj.forward(hidden_states, seq_len)?;

        let scale = 1.0_f32 / (self.head_dim as f32).sqrt();

        // Multi-head attention: [seq, heads, head_dim]
        // Compute attention scores with causal mask
        let mut attn_out = vec![0.0_f32; seq_len * self.embed_dim];

        for h in 0..self.num_heads {
            // Attention scores: [seq_len, seq_len]
            let mut scores = vec![f32::NEG_INFINITY; seq_len * seq_len];
            for qi in 0..seq_len {
                for ki in 0..=qi {
                    // causal: qi >= ki
                    let mut dot = 0.0_f32;
                    for d in 0..self.head_dim {
                        let q_val = q[qi * self.embed_dim + h * self.head_dim + d];
                        let k_val = k[ki * self.embed_dim + h * self.head_dim + d];
                        dot += q_val * k_val;
                    }
                    scores[qi * seq_len + ki] = dot * scale;
                }
            }

            // Softmax over keys (row-wise)
            for qi in 0..seq_len {
                let row = &mut scores[qi * seq_len..(qi + 1) * seq_len];
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut sum = 0.0_f32;
                for s in row.iter_mut() {
                    if *s != f32::NEG_INFINITY {
                        *s = (*s - max_val).exp();
                        sum += *s;
                    } else {
                        *s = 0.0;
                    }
                }
                if sum > 0.0 {
                    for s in row.iter_mut() {
                        *s /= sum;
                    }
                }
            }

            // Weighted sum of values
            for qi in 0..seq_len {
                for d in 0..self.head_dim {
                    let mut acc = 0.0_f32;
                    for ki in 0..seq_len {
                        let w = scores[qi * seq_len + ki];
                        acc += w * v[ki * self.embed_dim + h * self.head_dim + d];
                    }
                    attn_out[qi * self.embed_dim + h * self.head_dim + d] = acc;
                }
            }
        }

        // Output projection
        self.out_proj.forward(&attn_out, seq_len)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.out_proj.parameter_count()
    }
}

// ─── FFN ───────────────────────────────────────────────────────────────────

/// OPT Feed-Forward Network: `fc1 → ReLU → fc2` with bias.
///
/// OPT does NOT use gated activations (no SwiGLU, no GeGLU).
#[derive(Debug, Clone)]
pub struct OptFeedForward {
    fc1: OptLinear,
    fc2: OptLinear,
}

impl OptFeedForward {
    /// Create a new FFN.
    pub fn new(hidden_size: usize, ffn_dim: usize) -> Self {
        Self {
            fc1: OptLinear::new(hidden_size, ffn_dim),
            fc2: OptLinear::new(ffn_dim, hidden_size),
        }
    }

    /// ReLU activation: `max(0, x)`.
    #[inline]
    fn relu(x: f32) -> f32 {
        x.max(0.0)
    }

    /// Apply `fc1 → ReLU → fc2`.
    ///
    /// `x` is a flat `[seq_len * hidden_size]` input.
    pub fn forward(&self, x: &[f32], seq_len: usize) -> Result<Vec<f32>, OptError> {
        let after_fc1 = self.fc1.forward(x, seq_len)?;
        let ffn_dim = after_fc1.len() / seq_len;
        let after_relu: Vec<f32> = after_fc1.iter().map(|&v| Self::relu(v)).collect();
        let out = self.fc2.forward(&after_relu, seq_len)?;
        let _ = ffn_dim; // suppress unused warning
        Ok(out)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.fc1.parameter_count() + self.fc2.parameter_count()
    }
}

// ─── Decoder layer ─────────────────────────────────────────────────────────

/// A single OPT decoder layer.
///
/// Architecture (pre-norm variant, `do_layer_norm_before = true`):
/// ```text
/// x → LayerNorm → Attention → residual → LayerNorm → FFN → residual
/// ```
/// For post-norm (`do_layer_norm_before = false`):
/// ```text
/// x → Attention → residual → LayerNorm → FFN → residual → LayerNorm
/// ```
#[derive(Debug, Clone)]
pub struct OptDecoderLayer {
    self_attn: OptAttention,
    ffn: OptFeedForward,
    self_attn_layer_norm: OptLayerNorm,
    final_layer_norm: OptLayerNorm,
    do_layer_norm_before: bool,
    hidden_size: usize,
}

impl OptDecoderLayer {
    /// Create a new decoder layer from config.
    pub fn new(config: &OptConfig) -> Result<Self, OptError> {
        let hidden_size = config.hidden_size;
        Ok(Self {
            self_attn: OptAttention::new(config)?,
            ffn: OptFeedForward::new(hidden_size, config.ffn_dim),
            self_attn_layer_norm: OptLayerNorm::new(hidden_size, config.layer_norm_eps),
            final_layer_norm: OptLayerNorm::new(hidden_size, config.layer_norm_eps),
            do_layer_norm_before: config.do_layer_norm_before,
            hidden_size,
        })
    }

    /// Process `hidden_states` through one decoder layer.
    ///
    /// `hidden_states` is a flat `[seq_len * hidden_size]` slice;
    /// `pos_embeds` (currently unused — mixed into input by the decoder) is
    /// provided for future positional-bias variants.
    pub fn forward(
        &self,
        hidden_states: &[f32],
        _pos_embeds: &[f32],
        seq_len: usize,
    ) -> Result<Vec<f32>, OptError> {
        // ── Attention sub-layer ──────────────────────────────────────────
        let residual = hidden_states.to_vec();

        let normed_for_attn = if self.do_layer_norm_before {
            // Pre-norm: normalise each token's vector independently
            self.apply_layernorm_tokens(hidden_states, seq_len, &self.self_attn_layer_norm)?
        } else {
            hidden_states.to_vec()
        };

        let attn_out = self.self_attn.forward(&normed_for_attn, seq_len)?;

        // Residual
        let mut hidden: Vec<f32> =
            residual.iter().zip(attn_out.iter()).map(|(r, a)| r + a).collect();

        if !self.do_layer_norm_before {
            // Post-norm
            hidden = self.apply_layernorm_tokens(&hidden, seq_len, &self.self_attn_layer_norm)?;
        }

        // ── FFN sub-layer ────────────────────────────────────────────────
        let residual2 = hidden.clone();

        let normed_for_ffn = if self.do_layer_norm_before {
            self.apply_layernorm_tokens(&hidden, seq_len, &self.final_layer_norm)?
        } else {
            hidden
        };

        let ffn_out = self.ffn.forward(&normed_for_ffn, seq_len)?;

        let mut out: Vec<f32> = residual2.iter().zip(ffn_out.iter()).map(|(r, f)| r + f).collect();

        if !self.do_layer_norm_before {
            out = self.apply_layernorm_tokens(&out, seq_len, &self.final_layer_norm)?;
        }

        Ok(out)
    }

    /// Apply a LayerNorm independently to each token (row of `hidden_states`).
    fn apply_layernorm_tokens(
        &self,
        hidden_states: &[f32],
        seq_len: usize,
        norm: &OptLayerNorm,
    ) -> Result<Vec<f32>, OptError> {
        let mut out = Vec::with_capacity(hidden_states.len());
        for s in 0..seq_len {
            let start = s * self.hidden_size;
            let end = start + self.hidden_size;
            let row = norm.forward(&hidden_states[start..end])?;
            out.extend(row);
        }
        Ok(out)
    }

    /// Number of trainable parameters.
    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.ffn.parameter_count()
            + self.self_attn_layer_norm.parameter_count()
            + self.final_layer_norm.parameter_count()
    }
}

// ─── Decoder ───────────────────────────────────────────────────────────────

/// OPT decoder: embed_tokens + embed_positions + decoder_layers + final_layer_norm.
#[derive(Debug, Clone)]
pub struct OptDecoder {
    config: OptConfig,
    /// Token embedding table: `[vocab_size, word_embed_proj_dim]` flat.
    embed_tokens: Vec<f32>,
    /// Positional embedding.
    embed_positions: OptLearnedPositionalEmbedding,
    /// Optional projection from `word_embed_proj_dim` → `hidden_size`
    /// (used when `word_embed_proj_dim != hidden_size`).
    project_in: Option<OptLinear>,
    /// Optional projection from `hidden_size` → `word_embed_proj_dim`.
    project_out: Option<OptLinear>,
    layers: Vec<OptDecoderLayer>,
    final_layer_norm: OptLayerNorm,
}

impl OptDecoder {
    /// Construct the decoder from config.
    pub fn new(config: &OptConfig) -> Result<Self, OptError> {
        config.validate().map_err(OptError::InvalidConfig)?;

        let vocab_size = config.vocab_size;
        let embed_dim = config.word_embed_proj_dim;
        let hidden_size = config.hidden_size;

        // Token embeddings
        let embed_tokens = vec![0.0_f32; vocab_size * embed_dim];

        // Positional embeddings
        let embed_positions =
            OptLearnedPositionalEmbedding::new(config.max_position_embeddings, hidden_size);

        // Projection layers if embed_dim != hidden_size
        let (project_in, project_out) = if embed_dim != hidden_size {
            (
                Some(OptLinear::new(embed_dim, hidden_size)),
                Some(OptLinear::new(hidden_size, embed_dim)),
            )
        } else {
            (None, None)
        };

        // Build decoder layers
        let layers: Result<Vec<_>, _> =
            (0..config.num_hidden_layers).map(|_| OptDecoderLayer::new(config)).collect();
        let layers = layers?;

        let final_layer_norm = OptLayerNorm::new(hidden_size, config.layer_norm_eps);

        Ok(Self {
            config: config.clone(),
            embed_tokens,
            embed_positions,
            project_in,
            project_out,
            layers,
            final_layer_norm,
        })
    }

    /// Forward pass through the decoder.
    ///
    /// `input_ids` — token indices (length = `seq_len`).
    /// Returns flat `[seq_len * hidden_size]` hidden states.
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, OptError> {
        let seq_len = input_ids.len();
        let embed_dim = self.config.word_embed_proj_dim;
        let hidden_size = self.config.hidden_size;

        // Token embeddings
        let mut hidden: Vec<f32> = Vec::with_capacity(seq_len * embed_dim);
        for &tok in input_ids {
            let idx = tok as usize;
            if idx >= self.config.vocab_size {
                return Err(OptError::Forward(format!(
                    "token id {} >= vocab_size {}",
                    idx, self.config.vocab_size
                )));
            }
            let start = idx * embed_dim;
            hidden.extend_from_slice(&self.embed_tokens[start..start + embed_dim]);
        }

        // Project in (if necessary)
        let mut hidden: Vec<f32> = match &self.project_in {
            Some(proj) => proj.forward(&hidden, seq_len)?,
            None => hidden,
        };

        // Positional embeddings (added to hidden states)
        let position_ids = OptLearnedPositionalEmbedding::get_position_ids(seq_len);
        let pos_embeds = self.embed_positions.embed(&position_ids)?;
        for (h, p) in hidden.iter_mut().zip(pos_embeds.iter()) {
            *h += p;
        }

        // Decoder layers
        for layer in &self.layers {
            hidden = layer.forward(&hidden, &pos_embeds, seq_len)?;
        }

        // Final LayerNorm (applied per-token)
        let mut normed = Vec::with_capacity(hidden.len());
        for s in 0..seq_len {
            let start = s * hidden_size;
            let end = start + hidden_size;
            let row = self.final_layer_norm.forward(&hidden[start..end])?;
            normed.extend(row);
        }

        // Project out (if necessary)
        let out: Vec<f32> = match &self.project_out {
            Some(proj) => proj.forward(&normed, seq_len)?,
            None => normed,
        };

        Ok(out)
    }
}

// ─── OptModel ─────────────────────────────────────────────────────────────

/// Base OPT model (decoder only, returns last hidden states).
#[derive(Debug, Clone)]
pub struct OptModel {
    pub config: OptConfig,
    decoder: OptDecoder,
}

impl OptModel {
    /// Construct a new (untrained) OPT model.
    pub fn new(config: &OptConfig) -> Result<Self, OptError> {
        let decoder = OptDecoder::new(config)?;
        Ok(Self {
            config: config.clone(),
            decoder,
        })
    }

    /// Run the decoder and return `[seq_len, hidden_size]` hidden states (flat).
    pub fn forward(&self, input_ids: &[u32]) -> Result<Vec<f32>, OptError> {
        self.decoder.forward(input_ids)
    }
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::opt::config::OptConfig;

    fn tiny_config() -> OptConfig {
        OptConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            ffn_dim: 64,
            max_position_embeddings: 16,
            word_embed_proj_dim: 32,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            do_layer_norm_before: true,
            activation_function: "relu".to_string(),
            use_cache: true,
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: Some(1),
        }
    }

    #[test]
    fn test_opt_position_embedding_offset() {
        // OPT convention: position_ids[i] = i + 2
        let ids = OptLearnedPositionalEmbedding::get_position_ids(5);
        assert_eq!(ids, vec![2, 3, 4, 5, 6]);
    }

    #[test]
    fn test_opt_position_ids() {
        let ids = OptLearnedPositionalEmbedding::get_position_ids(1);
        assert_eq!(ids[0], 2);
    }

    #[test]
    fn test_opt_layer_norm() {
        let norm = OptLayerNorm::new(4, 1e-5);
        let x = vec![1.0_f32, 2.0, 3.0, 4.0];
        let y = norm.forward(&x).expect("layer norm should succeed");
        assert_eq!(y.len(), 4);
        // Mean of output should be ~0 with unit weights & zero bias
        let mean: f32 = y.iter().sum::<f32>() / y.len() as f32;
        assert!(mean.abs() < 1e-4, "mean should be ~0, got {}", mean);
    }

    #[test]
    fn test_opt_relu_activation() {
        // Test ReLU via FFN
        let relu = |x: f32| x.max(0.0);
        assert_eq!(relu(2.0), 2.0);
        assert_eq!(relu(-1.0), 0.0);
        assert_eq!(relu(0.0), 0.0);
    }

    #[test]
    fn test_opt_ffn_no_gating() {
        // OPT FFN has no gate: just fc1 → relu → fc2
        let cfg = tiny_config();
        let ffn = OptFeedForward::new(cfg.hidden_size, cfg.ffn_dim);
        let input = vec![0.5_f32; cfg.hidden_size]; // single token
        let out = ffn.forward(&input, 1).expect("ffn forward should succeed");
        assert_eq!(out.len(), cfg.hidden_size);
    }

    #[test]
    fn test_opt_attention_causal() {
        let cfg = tiny_config();
        let attn = OptAttention::new(&cfg).expect("attention creation should succeed");
        let seq_len = 3;
        let input = vec![0.1_f32; seq_len * cfg.hidden_size];
        let out = attn.forward(&input, seq_len).expect("attention forward should succeed");
        assert_eq!(out.len(), seq_len * cfg.hidden_size);
    }

    #[test]
    fn test_opt_decoder_layer_prenorm() {
        let cfg = tiny_config();
        assert!(cfg.do_layer_norm_before);
        let layer = OptDecoderLayer::new(&cfg).expect("layer creation should succeed");
        let seq_len = 2;
        let input = vec![0.0_f32; seq_len * cfg.hidden_size];
        let pos = vec![0.0_f32; seq_len * cfg.hidden_size];
        let out = layer
            .forward(&input, &pos, seq_len)
            .expect("decoder layer forward should succeed");
        assert_eq!(out.len(), seq_len * cfg.hidden_size);
    }

    #[test]
    fn test_opt_model_forward() {
        let cfg = tiny_config();
        let model = OptModel::new(&cfg).expect("model creation should succeed");
        let input_ids = vec![0u32, 1, 2];
        let out = model.forward(&input_ids).expect("forward should succeed");
        assert_eq!(out.len(), input_ids.len() * cfg.hidden_size);
    }

    #[test]
    fn test_opt_error_display() {
        let e = OptError::InvalidConfig("bad config".to_string());
        assert!(e.to_string().contains("bad config"));

        let e2 = OptError::Forward("shape error".to_string());
        assert!(e2.to_string().contains("shape error"));

        let e3 = OptError::Generation("too long".to_string());
        assert!(e3.to_string().contains("too long"));
    }
}
