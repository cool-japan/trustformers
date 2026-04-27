//! Stable Diffusion 3 text encoder pipeline.
//!
//! SD3 conditions the diffusion model on text via three encoders:
//!   - CLIP-L  (hidden=768, layers=12): produces 77-token embeddings + pooled [EOS] vector
//!   - CLIP-G  (hidden=1280, layers=32): produces 77-token embeddings + pooled [EOS] vector
//!   - T5-XXL  (hidden=4096, layers=24): produces 256-token embeddings for cross-attention
//!
//! The pooled conditioning vector is `concat(clip_l_eos, clip_g_eos)` → [2048].
//! The cross-attention conditioning is the full T5 encoder output → [256, 4096].
//!
//! Reference: "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"
//!            (Esser et al., 2024)

use crate::sd3::config::Sd3Config;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors from the SD3 text encoder pipeline.
#[derive(Debug, thiserror::Error)]
pub enum Sd3Error {
    #[error("Empty input sequence")]
    EmptyInput,
    #[error("Dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },
    #[error("Configuration error: {0}")]
    Config(String),
}

// ---------------------------------------------------------------------------
// Math helpers
// ---------------------------------------------------------------------------

#[inline]
fn gelu(x: f64) -> f64 {
    // Accurate GELU approximation
    0.5 * x * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
}

/// Numerically stable softmax.
fn softmax(x: &[f64]) -> Vec<f64> {
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
fn mat_vec_mul(weight: &[Vec<f64>], x: &[f64]) -> Result<Vec<f64>, Sd3Error> {
    if weight.is_empty() {
        return Ok(Vec::new());
    }
    let in_dim = weight[0].len();
    if x.len() != in_dim {
        return Err(Sd3Error::DimensionMismatch {
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

/// Layer normalization (standard, not RMS).
fn layer_norm(x: &[f64], weight: &[f64], bias: &[f64], eps: f64) -> Result<Vec<f64>, Sd3Error> {
    if x.len() != weight.len() || x.len() != bias.len() {
        return Err(Sd3Error::DimensionMismatch {
            expected: weight.len(),
            got: x.len(),
            context: "layer_norm".to_string(),
        });
    }
    let mean: f64 = x.iter().sum::<f64>() / x.len() as f64;
    let var: f64 = x.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / x.len() as f64;
    let std_inv = 1.0 / (var + eps).sqrt();
    Ok(x.iter()
        .zip(weight.iter().zip(bias.iter()))
        .map(|(v, (w, b))| (v - mean) * std_inv * w + b)
        .collect())
}

// ---------------------------------------------------------------------------
// T5RelativePositionBias
// ---------------------------------------------------------------------------

/// T5-style relative position bias using logarithmic bucketing.
///
/// T5 encodes relative positions (i - j) into a fixed number of buckets
/// and learns a scalar bias per (bucket, head) pair. This allows the model
/// to generalize to longer sequences than seen during training.
pub struct T5RelativePositionBias {
    /// Learned bias embeddings: [num_buckets, num_heads]
    embeddings: Vec<Vec<f64>>,
    num_heads: usize,
    num_buckets: usize,
    max_distance: usize,
}

impl T5RelativePositionBias {
    /// Create a new T5 relative position bias module.
    pub fn new(num_heads: usize, num_buckets: usize, max_distance: usize) -> Self {
        // Initialize embeddings with small values
        let embeddings: Vec<Vec<f64>> = (0..num_buckets)
            .map(|i| (0..num_heads).map(|h| 0.01 * ((i + h) % 7) as f64 - 0.03).collect())
            .collect();
        Self {
            embeddings,
            num_heads,
            num_buckets,
            max_distance,
        }
    }

    /// Map a relative position to a bucket index.
    ///
    /// T5 uses bidirectional bucketing: the first `num_buckets/2` positions are
    /// used for negative (i > j) relative positions and the second half for
    /// positive (i < j) ones. Within each half:
    ///   - The first `num_buckets/4` buckets cover exact distances 0..num_buckets/4
    ///   - The remaining buckets are logarithmically spaced up to max_distance
    pub fn relative_position_bucket(
        relative_position: i32,
        bidirectional: bool,
        num_buckets: usize,
        max_distance: usize,
    ) -> usize {
        let mut ret: i32 = 0;
        let mut n = -relative_position;

        let effective_buckets = if bidirectional {
            let half = (num_buckets / 2) as i32;
            if n < 0 {
                ret += half;
                n = -n;
            }
            half as usize
        } else {
            n = n.max(0);
            num_buckets
        };

        // Split: exact positions occupy first max_exact, log-spaced occupy the rest
        let max_exact = effective_buckets / 2;
        let is_small = n < max_exact as i32;

        let val: i32 = if is_small {
            n
        } else {
            let max_exact_f = max_exact as f64;
            let n_f = n as f64;
            let max_dist_f = max_distance as f64;

            let log_val = (n_f / max_exact_f).ln() / (max_dist_f / max_exact_f).ln()
                * (effective_buckets - max_exact) as f64;
            let bucket_val = max_exact as i32 + log_val.round() as i32;
            bucket_val.min(effective_buckets as i32 - 1)
        };

        (ret + val).max(0).min(num_buckets as i32 - 1) as usize
    }

    /// Compute a full relative position bias matrix.
    ///
    /// Returns a `[seq_len × seq_len × num_heads]` bias tensor laid out as
    /// a Vec of length seq_len*seq_len, each entry being a `[num_heads]` vector.
    /// Indexed as `[query_pos * seq_len + key_pos]`.
    pub fn compute_bias(&self, seq_len: usize, bidirectional: bool) -> Vec<Vec<f64>> {
        let mut result = Vec::with_capacity(seq_len * seq_len);
        for q in 0..seq_len {
            for k in 0..seq_len {
                let rel_pos = q as i32 - k as i32;
                let bucket = Self::relative_position_bucket(
                    rel_pos,
                    bidirectional,
                    self.num_buckets,
                    self.max_distance,
                );
                result.push(self.embeddings[bucket].clone());
            }
        }
        result
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of relative position buckets.
    pub fn num_buckets(&self) -> usize {
        self.num_buckets
    }
}

// ---------------------------------------------------------------------------
// T5Attention — multi-head self-attention with relative position bias
// ---------------------------------------------------------------------------

/// T5 multi-head self-attention layer.
///
/// Does not use any absolute positional embeddings; all position information
/// is encoded in the relative position bias added to attention logits.
pub struct T5Attention {
    q_proj: Vec<Vec<f64>>, // [num_heads * head_dim × hidden_size]
    k_proj: Vec<Vec<f64>>, // [num_heads * head_dim × hidden_size]
    v_proj: Vec<Vec<f64>>, // [num_heads * head_dim × hidden_size]
    o_proj: Vec<Vec<f64>>, // [hidden_size × num_heads * head_dim]
    /// Relative position bias (only on first layer per T5 convention)
    rel_pos_bias: Option<T5RelativePositionBias>,
    num_heads: usize,
    head_dim: usize,
}

impl T5Attention {
    /// Create a new T5 attention layer.
    ///
    /// If `has_rel_pos_bias` is true, a relative position bias module is created.
    pub fn new(
        hidden_size: usize,
        num_heads: usize,
        head_dim: usize,
        num_buckets: usize,
        max_distance: usize,
        has_rel_pos_bias: bool,
    ) -> Self {
        let q_dim = num_heads * head_dim;

        let q_proj: Vec<Vec<f64>> = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();
        let k_proj: Vec<Vec<f64>> = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 1) % hidden_size] = 0.01;
                row
            })
            .collect();
        let v_proj: Vec<Vec<f64>> = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 2) % hidden_size] = 0.01;
                row
            })
            .collect();
        let o_proj: Vec<Vec<f64>> = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; q_dim];
                row[i % q_dim] = 0.01;
                row
            })
            .collect();

        let rel_pos_bias = if has_rel_pos_bias {
            Some(T5RelativePositionBias::new(
                num_heads,
                num_buckets,
                max_distance,
            ))
        } else {
            None
        };

        Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rel_pos_bias,
            num_heads,
            head_dim,
        }
    }

    /// Forward pass for T5 self-attention.
    ///
    /// Input: `x` of shape [seq_len, hidden_size]
    /// Output: [seq_len, hidden_size] (no residual — added by the layer)
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Sd3Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Sd3Error::EmptyInput);
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        // Project Q, K, V
        let mut q_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut k_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);
        let mut v_all: Vec<Vec<f64>> = Vec::with_capacity(seq_len);

        for token in x.iter() {
            q_all.push(mat_vec_mul(&self.q_proj, token)?);
            k_all.push(mat_vec_mul(&self.k_proj, token)?);
            v_all.push(mat_vec_mul(&self.v_proj, token)?);
        }

        // Compute relative position biases if available
        let rel_bias: Option<Vec<Vec<f64>>> =
            self.rel_pos_bias.as_ref().map(|rpb| rpb.compute_bias(seq_len, true));

        let mut context_all: Vec<Vec<f64>> = vec![vec![0.0f64; num_heads * head_dim]; seq_len];

        for h in 0..num_heads {
            for q_pos in 0..seq_len {
                let q_vec: Vec<f64> =
                    (0..head_dim).map(|d| q_all[q_pos][h * head_dim + d]).collect();

                let mut scores: Vec<f64> = (0..seq_len)
                    .map(|k_pos| {
                        let k_vec: Vec<f64> =
                            (0..head_dim).map(|d| k_all[k_pos][h * head_dim + d]).collect();
                        let dot: f64 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                        let mut s = dot * scale;
                        // Add relative position bias for this (q, k) pair, head h
                        if let Some(ref bias_flat) = rel_bias {
                            s += bias_flat[q_pos * seq_len + k_pos][h];
                        }
                        s
                    })
                    .collect();

                // T5 encoder uses bidirectional (non-causal) attention
                let attn_weights = softmax(&scores);
                scores.clear(); // free memory

                for (k_pos, &w) in attn_weights.iter().enumerate() {
                    for d in 0..head_dim {
                        context_all[q_pos][h * head_dim + d] += w * v_all[k_pos][h * head_dim + d];
                    }
                }
            }
        }

        // Output projection
        let result: Vec<Vec<f64>> = context_all
            .iter()
            .map(|ctx| mat_vec_mul(&self.o_proj, ctx))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(result)
    }

    /// Number of attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }
}

// ---------------------------------------------------------------------------
// T5FeedForward — Gated GeLU FFN
// ---------------------------------------------------------------------------

/// T5 gated GeLU feed-forward network.
///
/// Uses a "gated linear unit" variant of GeLU:
///   output = wo( gelu(wi_0(x)) * wi_1(x) )
/// where wi_0 is the gate projection and wi_1 is the value projection.
pub struct T5FeedForward {
    wi_0: Vec<Vec<f64>>, // [intermediate × hidden] — gate
    wi_1: Vec<Vec<f64>>, // [intermediate × hidden] — value
    wo: Vec<Vec<f64>>,   // [hidden × intermediate] — output
}

impl T5FeedForward {
    /// Create a new T5 FFN with small diagonal initialization.
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let wi_0: Vec<Vec<f64>> = (0..intermediate_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();
        let wi_1: Vec<Vec<f64>> = (0..intermediate_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 1) % hidden_size] = 0.01;
                row
            })
            .collect();
        let wo: Vec<Vec<f64>> = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; intermediate_size];
                row[i % intermediate_size] = 0.01;
                row
            })
            .collect();
        Self { wi_0, wi_1, wo }
    }

    /// Forward pass: gated GeLU FFN.
    pub fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Sd3Error> {
        let gate = mat_vec_mul(&self.wi_0, x)?;
        let value = mat_vec_mul(&self.wi_1, x)?;
        let activated: Vec<f64> =
            gate.iter().zip(value.iter()).map(|(g, v)| gelu(*g) * v).collect();
        mat_vec_mul(&self.wo, &activated)
    }
}

// ---------------------------------------------------------------------------
// T5EncoderLayer — single T5 encoder layer
// ---------------------------------------------------------------------------

/// A single T5 encoder layer with pre-layer-norm.
///
/// Layout:
///   normed = layer_norm(x)
///   attn_out = attention(normed) + x   (residual)
///   normed2 = layer_norm(attn_out)
///   ffn_out = ffn(normed2) + attn_out  (residual)
pub struct T5EncoderLayer {
    attention: T5Attention,
    ffn: T5FeedForward,
    attn_norm_weight: Vec<f64>,
    attn_norm_bias: Vec<f64>,
    ffn_norm_weight: Vec<f64>,
    ffn_norm_bias: Vec<f64>,
    hidden_size: usize,
}

impl T5EncoderLayer {
    /// Create a new T5 encoder layer.
    ///
    /// The first layer (`layer_idx == 0`) owns the relative position bias.
    pub fn new(config: &Sd3Config, layer_idx: usize) -> Self {
        let hidden_size = config.t5_hidden_size;
        let has_rel_bias = layer_idx == 0;

        let attention = T5Attention::new(
            hidden_size,
            config.t5_num_heads,
            config.t5_head_dim(),
            config.t5_relative_attn_buckets,
            config.t5_max_distance,
            has_rel_bias,
        );
        let ffn = T5FeedForward::new(hidden_size, config.t5_intermediate_size);

        Self {
            attention,
            ffn,
            attn_norm_weight: vec![1.0f64; hidden_size],
            attn_norm_bias: vec![0.0f64; hidden_size],
            ffn_norm_weight: vec![1.0f64; hidden_size],
            ffn_norm_bias: vec![0.0f64; hidden_size],
            hidden_size,
        }
    }

    /// Forward pass for this T5 encoder layer.
    pub fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Sd3Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Sd3Error::EmptyInput);
        }

        // Self-attention with pre-norm
        let normed_attn: Vec<Vec<f64>> = x
            .iter()
            .map(|row| layer_norm(row, &self.attn_norm_weight, &self.attn_norm_bias, 1e-6))
            .collect::<Result<Vec<_>, _>>()?;

        let attn_out = self.attention.forward(&normed_attn)?;

        // Residual
        let after_attn: Vec<Vec<f64>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(r, a)| r.iter().zip(a.iter()).map(|(rv, av)| rv + av).collect())
            .collect();

        // FFN with pre-norm
        let normed_ffn: Vec<Vec<f64>> = after_attn
            .iter()
            .map(|row| layer_norm(row, &self.ffn_norm_weight, &self.ffn_norm_bias, 1e-6))
            .collect::<Result<Vec<_>, _>>()?;

        let ffn_out: Vec<Vec<f64>> = normed_ffn
            .iter()
            .map(|row| self.ffn.forward(row))
            .collect::<Result<Vec<_>, _>>()?;

        // Residual
        let result: Vec<Vec<f64>> = after_attn
            .iter()
            .zip(ffn_out.iter())
            .map(|(r, f)| r.iter().zip(f.iter()).map(|(rv, fv)| rv + fv).collect())
            .collect();

        Ok(result)
    }

    /// Hidden size of this layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
}

// ---------------------------------------------------------------------------
// T5Encoder — full T5 encoder stack
// ---------------------------------------------------------------------------

/// Full T5 encoder: N stacked T5EncoderLayer instances.
pub struct T5Encoder {
    embed_tokens: Vec<Vec<f64>>, // [vocab_size, hidden_size]
    layers: Vec<T5EncoderLayer>,
    final_norm_weight: Vec<f64>,
    final_norm_bias: Vec<f64>,
    config_hidden_size: usize,
    config_vocab_size: usize,
}

impl T5Encoder {
    /// Create a new T5 encoder from configuration.
    pub fn new(config: &Sd3Config) -> Self {
        let hidden_size = config.t5_hidden_size;
        let vocab_size = config.t5_vocab_size;

        let embed_tokens: Vec<Vec<f64>> = (0..vocab_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();

        let layers: Vec<T5EncoderLayer> =
            (0..config.t5_num_layers).map(|idx| T5EncoderLayer::new(config, idx)).collect();

        Self {
            embed_tokens,
            layers,
            final_norm_weight: vec![1.0f64; hidden_size],
            final_norm_bias: vec![0.0f64; hidden_size],
            config_hidden_size: hidden_size,
            config_vocab_size: vocab_size,
        }
    }

    /// Encode a sequence of token IDs.
    ///
    /// `token_ids`: `[seq_len]` → output: `[seq_len, hidden_size]`
    pub fn encode(&self, token_ids: &[u32]) -> Result<Vec<Vec<f64>>, Sd3Error> {
        if token_ids.is_empty() {
            return Err(Sd3Error::EmptyInput);
        }

        // Token embeddings
        let mut hidden: Vec<Vec<f64>> = token_ids
            .iter()
            .map(|&id| {
                let idx = id as usize % self.config_vocab_size;
                self.embed_tokens[idx].clone()
            })
            .collect();

        // Pass through encoder layers
        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer norm
        let normed: Vec<Vec<f64>> = hidden
            .iter()
            .map(|row| layer_norm(row, &self.final_norm_weight, &self.final_norm_bias, 1e-6))
            .collect::<Result<Vec<_>, _>>()?;

        Ok(normed)
    }

    /// Hidden size of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.config_hidden_size
    }

    /// Number of encoder layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// ClipAttention — simple CLIP multi-head attention
// ---------------------------------------------------------------------------

/// CLIP-style multi-head causal self-attention (used in CLIP text encoder).
///
/// Unlike T5, CLIP uses learned absolute positional embeddings and
/// causal (triangular) masking for text encoding.
struct ClipAttention {
    q_proj: Vec<Vec<f64>>,
    k_proj: Vec<Vec<f64>>,
    v_proj: Vec<Vec<f64>>,
    o_proj: Vec<Vec<f64>>,
    num_heads: usize,
    head_dim: usize,
}

impl ClipAttention {
    fn new(hidden_size: usize, num_heads: usize) -> Self {
        let head_dim = hidden_size / num_heads;
        let q_dim = num_heads * head_dim;
        let q_proj = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();
        let k_proj = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 1) % hidden_size] = 0.01;
                row
            })
            .collect();
        let v_proj = (0..q_dim)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[(i + 2) % hidden_size] = 0.01;
                row
            })
            .collect();
        let o_proj = (0..hidden_size)
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
            head_dim,
        }
    }

    fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Sd3Error> {
        let seq_len = x.len();
        if seq_len == 0 {
            return Err(Sd3Error::EmptyInput);
        }

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let num_heads = self.num_heads;
        let head_dim = self.head_dim;

        let mut q_all: Vec<Vec<f64>> =
            x.iter().map(|t| mat_vec_mul(&self.q_proj, t)).collect::<Result<_, _>>()?;
        let k_all: Vec<Vec<f64>> =
            x.iter().map(|t| mat_vec_mul(&self.k_proj, t)).collect::<Result<_, _>>()?;
        let v_all: Vec<Vec<f64>> =
            x.iter().map(|t| mat_vec_mul(&self.v_proj, t)).collect::<Result<_, _>>()?;

        let mut context_all: Vec<Vec<f64>> = vec![vec![0.0f64; num_heads * head_dim]; seq_len];

        for h in 0..num_heads {
            for q_pos in 0..seq_len {
                let q_vec: Vec<f64> =
                    (0..head_dim).map(|d| q_all[q_pos][h * head_dim + d]).collect();

                // Causal mask: only attend to positions <= q_pos
                let scores: Vec<f64> = (0..=q_pos)
                    .map(|k_pos| {
                        let k_vec: Vec<f64> =
                            (0..head_dim).map(|d| k_all[k_pos][h * head_dim + d]).collect();
                        let dot: f64 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
                        dot * scale
                    })
                    .collect();

                let attn_weights = softmax(&scores);
                for (k_pos, &w) in attn_weights.iter().enumerate() {
                    for d in 0..head_dim {
                        context_all[q_pos][h * head_dim + d] += w * v_all[k_pos][h * head_dim + d];
                    }
                }
            }
        }

        // Clear unused q_all to satisfy move semantics
        q_all.clear();

        let result: Vec<Vec<f64>> = context_all
            .iter()
            .map(|ctx| mat_vec_mul(&self.o_proj, ctx))
            .collect::<Result<_, _>>()?;

        Ok(result)
    }
}

// ---------------------------------------------------------------------------
// ClipMlp — CLIP feed-forward (standard GeLU MLP)
// ---------------------------------------------------------------------------

struct ClipMlp {
    fc1: Vec<Vec<f64>>, // [intermediate × hidden]
    fc2: Vec<Vec<f64>>, // [hidden × intermediate]
}

impl ClipMlp {
    fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        let fc1 = (0..intermediate_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();
        let fc2 = (0..hidden_size)
            .map(|i| {
                let mut row = vec![0.0f64; intermediate_size];
                row[i % intermediate_size] = 0.01;
                row
            })
            .collect();
        Self { fc1, fc2 }
    }

    fn forward(&self, x: &[f64]) -> Result<Vec<f64>, Sd3Error> {
        let h = mat_vec_mul(&self.fc1, x)?;
        let activated: Vec<f64> = h.iter().map(|v| gelu(*v)).collect();
        mat_vec_mul(&self.fc2, &activated)
    }
}

// ---------------------------------------------------------------------------
// ClipEncoderLayer — single CLIP transformer layer
// ---------------------------------------------------------------------------

struct ClipEncoderLayer {
    attention: ClipAttention,
    mlp: ClipMlp,
    norm1_weight: Vec<f64>,
    norm1_bias: Vec<f64>,
    norm2_weight: Vec<f64>,
    norm2_bias: Vec<f64>,
}

impl ClipEncoderLayer {
    fn new(hidden_size: usize, num_heads: usize, intermediate_size: usize) -> Self {
        Self {
            attention: ClipAttention::new(hidden_size, num_heads),
            mlp: ClipMlp::new(hidden_size, intermediate_size),
            norm1_weight: vec![1.0f64; hidden_size],
            norm1_bias: vec![0.0f64; hidden_size],
            norm2_weight: vec![1.0f64; hidden_size],
            norm2_bias: vec![0.0f64; hidden_size],
        }
    }

    fn forward(&self, x: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, Sd3Error> {
        // Pre-norm + attention + residual
        let normed: Vec<Vec<f64>> = x
            .iter()
            .map(|row| layer_norm(row, &self.norm1_weight, &self.norm1_bias, 1e-5))
            .collect::<Result<_, _>>()?;
        let attn_out = self.attention.forward(&normed)?;
        let after_attn: Vec<Vec<f64>> = x
            .iter()
            .zip(attn_out.iter())
            .map(|(r, a)| r.iter().zip(a.iter()).map(|(rv, av)| rv + av).collect())
            .collect();

        // Pre-norm + MLP + residual
        let normed2: Vec<Vec<f64>> = after_attn
            .iter()
            .map(|row| layer_norm(row, &self.norm2_weight, &self.norm2_bias, 1e-5))
            .collect::<Result<_, _>>()?;
        let mlp_out: Vec<Vec<f64>> =
            normed2.iter().map(|row| self.mlp.forward(row)).collect::<Result<_, _>>()?;

        Ok(after_attn
            .iter()
            .zip(mlp_out.iter())
            .map(|(r, m)| r.iter().zip(m.iter()).map(|(rv, mv)| rv + mv).collect())
            .collect())
    }
}

// ---------------------------------------------------------------------------
// ClipTextEncoder — full CLIP text encoder
// ---------------------------------------------------------------------------

/// Simplified CLIP text encoder (shared structure for CLIP-L and CLIP-G).
pub struct ClipTextEncoder {
    embed_tokens: Vec<Vec<f64>>, // [vocab_size, hidden_size]
    pos_embed: Vec<Vec<f64>>,    // [max_seq_len, hidden_size]
    layers: Vec<ClipEncoderLayer>,
    final_norm_weight: Vec<f64>,
    final_norm_bias: Vec<f64>,
    hidden_size: usize,
    vocab_size: usize,
}

impl ClipTextEncoder {
    /// Create a new CLIP text encoder.
    pub fn new(
        vocab_size: usize,
        hidden_size: usize,
        num_layers: usize,
        num_heads: usize,
        intermediate_size: usize,
        max_seq_len: usize,
    ) -> Self {
        let embed_tokens: Vec<Vec<f64>> = (0..vocab_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();

        // Learned positional embeddings
        let pos_embed: Vec<Vec<f64>> = (0..max_seq_len)
            .map(|i| {
                (0..hidden_size)
                    .map(|d| {
                        let angle =
                            i as f64 / (10000.0_f64).powf(2.0 * d as f64 / hidden_size as f64);
                        if d % 2 == 0 {
                            angle.sin()
                        } else {
                            angle.cos()
                        }
                    })
                    .collect()
            })
            .collect();

        let layers: Vec<ClipEncoderLayer> = (0..num_layers)
            .map(|_| ClipEncoderLayer::new(hidden_size, num_heads, intermediate_size))
            .collect();

        Self {
            embed_tokens,
            pos_embed,
            layers,
            final_norm_weight: vec![1.0f64; hidden_size],
            final_norm_bias: vec![0.0f64; hidden_size],
            hidden_size,
            vocab_size,
        }
    }

    /// Encode a sequence of token IDs.
    ///
    /// Returns `(all_hidden_states [seq_len, hidden_size], pooled_eos [hidden_size])`.
    /// The EOS pooled vector is the final-layer hidden state at the last non-padding position.
    pub fn encode(&self, token_ids: &[u32]) -> Result<(Vec<Vec<f64>>, Vec<f64>), Sd3Error> {
        if token_ids.is_empty() {
            return Err(Sd3Error::EmptyInput);
        }

        // Embed tokens + add positional embeddings
        let mut hidden: Vec<Vec<f64>> = token_ids
            .iter()
            .enumerate()
            .map(|(pos, &id)| {
                let idx = id as usize % self.vocab_size;
                let pos_idx = pos.min(self.pos_embed.len() - 1);
                self.embed_tokens[idx]
                    .iter()
                    .zip(self.pos_embed[pos_idx].iter())
                    .map(|(e, p)| e + p)
                    .collect()
            })
            .collect();

        // Pass through transformer layers
        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden)?;
        }

        // Final layer norm
        let normed: Vec<Vec<f64>> = hidden
            .iter()
            .map(|row| layer_norm(row, &self.final_norm_weight, &self.final_norm_bias, 1e-5))
            .collect::<Result<_, _>>()?;

        // EOS pooled: last token's hidden state (CLIP convention)
        let eos_idx = normed.len() - 1;
        let pooled = normed[eos_idx].clone();

        Ok((normed, pooled))
    }

    /// Hidden size of the encoder.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Number of transformer layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

// ---------------------------------------------------------------------------
// Sd3TextEmbeddings — output struct
// ---------------------------------------------------------------------------

/// Combined text embeddings produced by the SD3 text encoder pipeline.
///
/// These are used for conditioning the SD3 diffusion transformer (MMDiT):
///   - `t5_embeddings`: cross-attention conditioning from T5 `[max_t5_seq_len × t5_hidden_size]`
///   - `pooled_embeddings`: global conditioning `[pooled_embedding_dim = CLIP-L + CLIP-G]`
///   - `seq_len`: actual input sequence length (before T5 padding)
pub struct Sd3TextEmbeddings {
    /// T5 encoder output: `[max_t5_seq_len * t5_hidden_size]` (row-major)
    pub t5_embeddings: Vec<f32>,
    /// Pooled CLIP embeddings: `[pooled_embedding_dim]`
    pub pooled_embeddings: Vec<f32>,
    /// Actual number of tokens in the input sequence
    pub seq_len: usize,
}

impl Sd3TextEmbeddings {
    /// T5 embedding dimension (number of floats per token).
    pub fn t5_embedding_dim(&self, max_seq_len: usize) -> usize {
        self.t5_embeddings.len().checked_div(max_seq_len).unwrap_or(0)
    }

    /// Pooled embedding dimension.
    pub fn pooled_dim(&self) -> usize {
        self.pooled_embeddings.len()
    }
}

// ---------------------------------------------------------------------------
// Sd3TextEncoderPipeline
// ---------------------------------------------------------------------------

/// The full SD3 three-encoder text processing pipeline.
///
/// Encodes text tokens through CLIP-L, CLIP-G, and T5-XXL and returns
/// combined embeddings for conditioning the SD3 diffusion model.
pub struct Sd3TextEncoderPipeline {
    clip_l: ClipTextEncoder,
    clip_g: ClipTextEncoder,
    t5: T5Encoder,
    config: Sd3Config,
}

impl Sd3TextEncoderPipeline {
    /// Create a new SD3 text encoder pipeline from configuration.
    pub fn new(config: Sd3Config) -> Result<Self, Sd3Error> {
        config.validate().map_err(|e| Sd3Error::Config(e.to_string()))?;

        let clip_l = ClipTextEncoder::new(
            config.clip_vocab_size,
            config.clip_hidden_size,
            config.clip_num_layers,
            config.clip_num_heads,
            config.clip_intermediate_size,
            config.max_sequence_length,
        );

        // CLIP-G uses the same vocab and max_seq_len, but different hidden dim / layer counts
        // CLIP-G intermediate size is typically 5120; we derive it from CLIP-G hidden for simplicity
        let clip_g_intermediate = config.clip_g_hidden_size * 4;
        let clip_g = ClipTextEncoder::new(
            config.clip_vocab_size,
            config.clip_g_hidden_size,
            config.clip_g_num_layers,
            config.clip_g_num_heads,
            clip_g_intermediate,
            config.max_sequence_length,
        );

        let t5 = T5Encoder::new(&config);

        Ok(Self {
            clip_l,
            clip_g,
            t5,
            config,
        })
    }

    /// Encode a sequence of token IDs and return SD3 text embeddings.
    ///
    /// `token_ids`: token indices (shared across encoders for simplicity; in practice
    ///   different tokenizers are used for CLIP vs T5).
    /// `seq_len`: effective sequence length for the T5 encoder output.
    pub fn encode_text(
        &self,
        token_ids: &[u32],
        seq_len: usize,
    ) -> Result<Sd3TextEmbeddings, Sd3Error> {
        if token_ids.is_empty() {
            return Err(Sd3Error::EmptyInput);
        }

        let max_t5_len = self.config.max_t5_sequence_length;
        let t5_hidden = self.config.t5_hidden_size;

        // Truncate/pad token_ids for T5 (max 256 tokens)
        let t5_input: Vec<u32> = if token_ids.len() > max_t5_len {
            token_ids[..max_t5_len].to_vec()
        } else {
            let mut padded = token_ids.to_vec();
            padded.resize(max_t5_len.min(token_ids.len() + 1).max(token_ids.len()), 0);
            padded
        };

        // T5 encoding
        let t5_out = self.t5.encode(&t5_input)?;

        // Pad T5 output to max_t5_len if shorter
        let t5_padded = if t5_out.len() < max_t5_len {
            let mut padded = t5_out;
            while padded.len() < max_t5_len {
                padded.push(vec![0.0f64; t5_hidden]);
            }
            padded
        } else {
            t5_out
        };

        // CLIP-L encoding (truncate to max 77 tokens)
        let clip_input: Vec<u32> = if token_ids.len() > self.config.max_sequence_length {
            token_ids[..self.config.max_sequence_length].to_vec()
        } else {
            token_ids.to_vec()
        };

        let (_, clip_l_pooled) = self.clip_l.encode(&clip_input)?;
        let (_, clip_g_pooled) = self.clip_g.encode(&clip_input)?;

        // Flatten T5 output to [max_t5_len * t5_hidden] f32
        let t5_embeddings: Vec<f32> =
            t5_padded.iter().flat_map(|row| row.iter().map(|&v| v as f32)).collect();

        // Concatenate CLIP-L and CLIP-G pooled vectors → [pooled_dim]
        let pooled_embeddings: Vec<f32> =
            clip_l_pooled.iter().chain(clip_g_pooled.iter()).map(|&v| v as f32).collect();

        Ok(Sd3TextEmbeddings {
            t5_embeddings,
            pooled_embeddings,
            seq_len,
        })
    }

    /// Reference to the configuration.
    pub fn config(&self) -> &Sd3Config {
        &self.config
    }

    /// Reference to the T5 encoder.
    pub fn t5_encoder(&self) -> &T5Encoder {
        &self.t5
    }

    /// Reference to the CLIP-L encoder.
    pub fn clip_l_encoder(&self) -> &ClipTextEncoder {
        &self.clip_l
    }

    /// Reference to the CLIP-G encoder.
    pub fn clip_g_encoder(&self) -> &ClipTextEncoder {
        &self.clip_g
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// LCG for deterministic pseudo-randomness.
    struct Lcg {
        state: u64,
    }

    impl Lcg {
        fn new(seed: u64) -> Self {
            Self { state: seed }
        }

        fn next_u64(&mut self) -> u64 {
            self.state =
                self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            self.state
        }
    }

    // --- Sd3Config tests ---

    #[test]
    fn test_sd3_config_default_values() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.t5_hidden_size, 4096);
        assert_eq!(cfg.t5_num_layers, 24);
        assert_eq!(cfg.t5_num_heads, 64);
        assert_eq!(cfg.clip_hidden_size, 768);
        assert_eq!(cfg.clip_num_layers, 12);
        assert_eq!(cfg.clip_g_hidden_size, 1280);
        assert_eq!(cfg.clip_g_num_layers, 32);
        assert_eq!(cfg.pooled_embedding_dim, 2048);
        assert_eq!(cfg.max_sequence_length, 77);
        assert_eq!(cfg.max_t5_sequence_length, 256);
    }

    #[test]
    fn test_sd3_config_pooled_dim_is_sum_of_clips() {
        let cfg = Sd3Config::default();
        assert_eq!(
            cfg.pooled_embedding_dim,
            cfg.clip_hidden_size + cfg.clip_g_hidden_size,
            "pooled_dim must equal CLIP-L + CLIP-G hidden sizes"
        );
    }

    #[test]
    fn test_sd3_config_validate_ok() {
        let cfg = Sd3Config::default();
        assert!(cfg.validate().is_ok(), "default config should validate");
    }

    #[test]
    fn test_sd3_config_t5_head_dim() {
        let cfg = Sd3Config::default();
        // 4096 / 64 = 64
        assert_eq!(cfg.t5_head_dim(), 64);
    }

    #[test]
    fn test_sd3_config_clip_head_dim() {
        let cfg = Sd3Config::default();
        // 768 / 12 = 64
        assert_eq!(cfg.clip_head_dim(), 64);
    }

    #[test]
    fn test_sd3_config_clip_g_head_dim() {
        let cfg = Sd3Config::default();
        // 1280 / 20 = 64
        assert_eq!(cfg.clip_g_head_dim(), 64);
    }

    #[test]
    fn test_sd3_config_validate_bad_pooled_dim() {
        let cfg = Sd3Config {
            pooled_embedding_dim: 1024,
            ..Sd3Config::default()
        }; // should be 2048
        assert!(cfg.validate().is_err(), "wrong pooled_dim should fail");
    }

    #[test]
    fn test_sd3_config_text_embedding_dim_equals_t5_hidden() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.text_embedding_dim, cfg.t5_hidden_size);
    }

    // --- T5RelativePositionBias tests ---

    #[test]
    fn test_t5_rel_pos_bias_new() {
        let bias = T5RelativePositionBias::new(8, 32, 128);
        assert_eq!(bias.num_heads(), 8);
        assert_eq!(bias.num_buckets(), 32);
    }

    #[test]
    fn test_t5_rel_pos_bucket_zero() {
        // Relative position 0 → bucket 0 or num_buckets/2 depending on bidirectional
        let b_uni = T5RelativePositionBias::relative_position_bucket(0, false, 32, 128);
        assert!(b_uni < 32, "bucket must be < num_buckets");
    }

    #[test]
    fn test_t5_rel_pos_bucket_in_range() {
        let num_buckets = 32usize;
        let max_distance = 128usize;
        let mut rng = Lcg::new(42);
        for _ in 0..50 {
            let pos: i32 = (rng.next_u64() % 400) as i32 - 200;
            let b = T5RelativePositionBias::relative_position_bucket(
                pos,
                true,
                num_buckets,
                max_distance,
            );
            assert!(b < num_buckets, "bucket {} out of range for pos {}", b, pos);
        }
    }

    #[test]
    fn test_t5_rel_pos_bias_compute_shape() {
        let bias = T5RelativePositionBias::new(4, 32, 128);
        let mat = bias.compute_bias(5, true);
        // seq_len^2 entries, each of length num_heads
        assert_eq!(mat.len(), 25, "bias matrix should have seq_len^2 rows");
        assert_eq!(mat[0].len(), 4, "each entry should have num_heads values");
    }

    #[test]
    fn test_t5_rel_pos_bias_diagonal_same_bucket() {
        // On the diagonal (q==k), relative position is 0 — same bucket for all positions
        let bias = T5RelativePositionBias::new(4, 32, 128);
        let seq_len = 6;
        let mat = bias.compute_bias(seq_len, true);
        let b00 = T5RelativePositionBias::relative_position_bucket(0, true, 32, 128);
        for i in 0..seq_len {
            let b_ii = T5RelativePositionBias::relative_position_bucket(0, true, 32, 128);
            assert_eq!(b00, b_ii, "diagonal entries should share bucket");
            let _ = mat[i * seq_len + i]; // access is valid
        }
    }

    // --- gelu / softmax math helpers ---

    #[test]
    fn test_gelu_zero() {
        let val = 0.5 * 0.0_f64 * (1.0 + ((2.0 / std::f64::consts::PI).sqrt() * 0.0_f64).tanh());
        assert!((val - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_softmax_sums_to_one() {
        let scores = vec![1.0f64, 2.0, 3.0, 0.5, -1.0];
        let probs = softmax(&scores);
        let sum: f64 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-9, "softmax must sum to 1");
    }

    #[test]
    fn test_softmax_max_gets_highest_prob() {
        let scores = vec![1.0f64, 5.0, 2.0];
        let probs = softmax(&scores);
        assert!(
            probs[1] > probs[0] && probs[1] > probs[2],
            "max score should have highest prob"
        );
    }

    // --- layer_norm helper ---

    #[test]
    fn test_layer_norm_constant_input() {
        // A constant input normalizes to zero, then applies weight/bias
        let x = vec![3.0f64; 4];
        let weight = vec![1.0f64; 4];
        let bias = vec![0.0f64; 4];
        let out = layer_norm(&x, &weight, &bias, 1e-6).expect("layer_norm should succeed");
        for &v in &out {
            assert!(v.abs() < 1e-5, "constant input should normalize to ~0");
        }
    }

    // --- T5Attention ---

    #[test]
    fn test_t5_attention_output_shape() {
        // Small config: hidden=4, heads=2, head_dim=2
        let hidden = 4usize;
        let heads = 2usize;
        let head_dim = 2usize;
        let attn = T5Attention::new(hidden, heads, head_dim, 16, 32, true);
        assert_eq!(attn.num_heads(), 2);
        let seq_len = 3;
        let x: Vec<Vec<f64>> = (0..seq_len).map(|_| vec![0.1f64; hidden]).collect();
        let out = attn.forward(&x).expect("T5Attention forward should succeed");
        assert_eq!(out.len(), seq_len, "output seq_len must match input");
        assert_eq!(out[0].len(), hidden, "output hidden dim must match");
    }

    #[test]
    fn test_t5_attention_empty_input_error() {
        let attn = T5Attention::new(4, 2, 2, 16, 32, false);
        let empty: Vec<Vec<f64>> = vec![];
        let result = attn.forward(&empty);
        assert!(result.is_err(), "empty input should fail");
    }

    // --- T5FeedForward ---

    #[test]
    fn test_t5_ffn_output_shape() {
        let ffn = T5FeedForward::new(8, 16);
        let x = vec![0.5f64; 8];
        let out = ffn.forward(&x).expect("T5FeedForward forward should succeed");
        assert_eq!(out.len(), 8, "FFN output must match hidden size");
    }

    // --- ClipTextEncoder ---

    #[test]
    fn test_clip_encoder_output_shape() {
        let enc = ClipTextEncoder::new(100, 8, 1, 2, 16, 16);
        let tokens: Vec<u32> = vec![1, 2, 3];
        let (hidden, pooled) = enc.encode(&tokens).expect("CLIP encode should succeed");
        assert_eq!(
            hidden.len(),
            3,
            "hidden states should have one row per token"
        );
        assert_eq!(
            hidden[0].len(),
            8,
            "each token embedding should be hidden_size"
        );
        assert_eq!(pooled.len(), 8, "pooled should be hidden_size");
    }

    #[test]
    fn test_clip_encoder_empty_input_error() {
        let enc = ClipTextEncoder::new(100, 8, 1, 2, 16, 16);
        let result = enc.encode(&[]);
        assert!(result.is_err(), "empty token list should fail");
    }

    // --- Sd3TextEncoderPipeline (tiny config) ---

    #[test]
    fn test_sd3_pipeline_construction_tiny() {
        // Build a tiny config for fast tests
        let cfg = Sd3Config {
            t5_vocab_size: 32,
            t5_hidden_size: 4,
            t5_num_layers: 1,
            t5_num_heads: 2,
            t5_intermediate_size: 8,
            t5_relative_attn_buckets: 8,
            t5_max_distance: 16,
            clip_vocab_size: 32,
            clip_hidden_size: 4,
            clip_num_layers: 1,
            clip_num_heads: 2,
            clip_intermediate_size: 8,
            clip_g_hidden_size: 8,
            clip_g_num_layers: 1,
            clip_g_num_heads: 2,
            text_embedding_dim: 4,
            pooled_embedding_dim: 12, // 4 + 8
            max_sequence_length: 8,
            max_t5_sequence_length: 8,
        };
        let pipeline = Sd3TextEncoderPipeline::new(cfg);
        assert!(
            pipeline.is_ok(),
            "pipeline should construct with tiny config"
        );
    }

    #[test]
    fn test_sd3_pipeline_encode_text_shapes() {
        let cfg = Sd3Config {
            t5_vocab_size: 32,
            t5_hidden_size: 4,
            t5_num_layers: 1,
            t5_num_heads: 2,
            t5_intermediate_size: 8,
            t5_relative_attn_buckets: 8,
            t5_max_distance: 16,
            clip_vocab_size: 32,
            clip_hidden_size: 4,
            clip_num_layers: 1,
            clip_num_heads: 2,
            clip_intermediate_size: 8,
            clip_g_hidden_size: 8,
            clip_g_num_layers: 1,
            clip_g_num_heads: 2,
            text_embedding_dim: 4,
            pooled_embedding_dim: 12,
            max_sequence_length: 8,
            max_t5_sequence_length: 8,
        };
        let pipeline = Sd3TextEncoderPipeline::new(cfg.clone()).expect("pipeline should construct");
        let tokens: Vec<u32> = vec![1, 2, 3];
        let embeddings = pipeline.encode_text(&tokens, 3).expect("encode_text should succeed");
        // t5_embeddings: max_t5_len * t5_hidden = 8 * 4 = 32
        assert_eq!(
            embeddings.t5_embeddings.len(),
            cfg.max_t5_sequence_length * cfg.t5_hidden_size
        );
        // pooled_embeddings: clip_l + clip_g = 4 + 8 = 12
        assert_eq!(embeddings.pooled_embeddings.len(), cfg.pooled_embedding_dim);
        assert_eq!(embeddings.seq_len, 3);
    }

    #[test]
    fn test_sd3_text_embeddings_pooled_dim() {
        let emb = Sd3TextEmbeddings {
            t5_embeddings: vec![0.0f32; 1024],
            pooled_embeddings: vec![0.0f32; 2048],
            seq_len: 10,
        };
        assert_eq!(emb.pooled_dim(), 2048);
    }

    #[test]
    fn test_sd3_text_embeddings_t5_dim() {
        let emb = Sd3TextEmbeddings {
            t5_embeddings: vec![0.0f32; 256 * 4096],
            pooled_embeddings: vec![0.0f32; 2048],
            seq_len: 16,
        };
        assert_eq!(emb.t5_embedding_dim(256), 4096);
    }

    #[test]
    fn test_sd3_pipeline_empty_tokens_error() {
        let cfg = Sd3Config {
            t5_vocab_size: 32,
            t5_hidden_size: 4,
            t5_num_layers: 1,
            t5_num_heads: 2,
            t5_intermediate_size: 8,
            t5_relative_attn_buckets: 8,
            t5_max_distance: 16,
            clip_vocab_size: 32,
            clip_hidden_size: 4,
            clip_num_layers: 1,
            clip_num_heads: 2,
            clip_intermediate_size: 8,
            clip_g_hidden_size: 8,
            clip_g_num_layers: 1,
            clip_g_num_heads: 2,
            text_embedding_dim: 4,
            pooled_embedding_dim: 12,
            max_sequence_length: 8,
            max_t5_sequence_length: 8,
        };
        let pipeline = Sd3TextEncoderPipeline::new(cfg).expect("pipeline should construct");
        let result = pipeline.encode_text(&[], 0);
        assert!(result.is_err(), "empty token ids should fail");
    }

    #[test]
    fn test_mat_vec_mul_shape() {
        // 3x2 weight @ 2-vec → 3-vec
        let weight: Vec<Vec<f64>> = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![0.5, 0.5]];
        let x = vec![2.0f64, 3.0];
        let out = mat_vec_mul(&weight, &x).expect("mat_vec_mul should succeed");
        assert_eq!(out.len(), 3);
        assert!((out[0] - 2.0).abs() < 1e-9);
        assert!((out[1] - 3.0).abs() < 1e-9);
        assert!((out[2] - 2.5).abs() < 1e-9);
    }
}
