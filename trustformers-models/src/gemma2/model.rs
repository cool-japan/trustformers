//! # Gemma-2 Model Implementation
//!
//! Core architecture components for Gemma-2:
//! - Soft-capping on attention and final logits
//! - Alternating local / global attention (even = local, odd = global)
//! - Grouped Query Attention (GQA)
//! - GEGLU activation (GELU gate × linear up)
//! - Pre- and post-normalization with RMSNorm

use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

use super::config::Gemma2Config;

// ---------------------------------------------------------------------------
// Soft-capping
// ---------------------------------------------------------------------------

/// Apply logit soft-capping: `tanh(x / cap) * cap`.
///
/// This bounds activations smoothly, preventing extreme values from
/// destabilising training or inference while preserving gradient flow.
pub fn soft_cap(x: f32, cap: f64) -> f32 {
    let c = cap as f32;
    (x / c).tanh() * c
}

/// Apply soft-capping element-wise to a mutable slice.
pub fn apply_soft_cap_inplace(data: &mut [f32], cap: f64) {
    for v in data.iter_mut() {
        *v = soft_cap(*v, cap);
    }
}

// ---------------------------------------------------------------------------
// Activations
// ---------------------------------------------------------------------------

/// Standard GELU: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³)))`.
pub fn gelu(x: f32) -> f32 {
    use std::f32::consts::PI;
    let c = (2.0f32 / PI).sqrt();
    0.5 * x * (1.0 + (c * (x + 0.044715 * x * x * x)).tanh())
}

/// GEGLU: element-wise `gelu(gate) * up`.
pub fn geglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| gelu(g) * u).collect()
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Gemma-2 RMSNorm layer.
///
/// Formula: `output = weight * (input / sqrt(mean(input²) + eps))`
/// Weight is initialised to all-ones.
pub struct Gemma2RmsNorm {
    weight: Tensor,
    eps: f32,
    device: Device,
}

impl Gemma2RmsNorm {
    /// Create a new `Gemma2RmsNorm` for vectors of length `size`.
    pub fn new(size: usize, eps: f64, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[size])?;
        Ok(Self {
            weight,
            eps: eps as f32,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Gemma2RmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let n = arr.len() as f32;
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / n;
                let rms = (mean_sq + self.eps).sqrt();
                let normed = arr.mapv(|x| x / rms);
                match &self.weight {
                    Tensor::F32(w) => Ok(Tensor::F32(&normed * w)),
                    _ => Err(tensor_op_error(
                        "gemma2_rmsnorm",
                        "weight tensor must be F32",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "gemma2_rmsnorm",
                "input tensor must be F32",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding
// ---------------------------------------------------------------------------

/// Standard RoPE for Gemma-2 (no scaling).
pub struct Gemma2RotaryEmbedding {
    head_dim: usize,
    rope_theta: f64,
    #[allow(dead_code)]
    device: Device,
}

impl Gemma2RotaryEmbedding {
    pub fn new(config: &Gemma2Config, device: Device) -> Self {
        Self {
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            device,
        }
    }

    /// Apply RoPE to flat query and key slices (in-place variant).
    ///
    /// Both `q` and `k` are expected to have shape
    /// `[seq_len * head_dim]` per head.
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], seq_len: usize) {
        let half = self.head_dim / 2;
        for pos in 0..seq_len {
            for i in 0..half {
                let freq = 1.0 / self.rope_theta.powf(2.0 * i as f64 / self.head_dim as f64);
                let angle = (pos as f64 * freq) as f32;
                let cos_v = angle.cos();
                let sin_v = angle.sin();

                let base = pos * self.head_dim;
                let q0 = q[base + i];
                let q1 = q[base + i + half];
                q[base + i] = q0 * cos_v - q1 * sin_v;
                q[base + i + half] = q0 * sin_v + q1 * cos_v;

                let k0 = k[base + i];
                let k1 = k[base + i + half];
                k[base + i] = k0 * cos_v - k1 * sin_v;
                k[base + i + half] = k0 * sin_v + k1 * cos_v;
            }
        }
    }
}

/// Apply RoPE to `data` (row-major, shape `[seq_len, num_heads * head_dim]`)
/// in place, rotating every one of the `num_heads` blocks in each row
/// independently so multi-head (and GQA, where `q` and `k` have a different
/// head count) tensors are fully rotated.
///
/// `position_ids[row]` gives the absolute position of `row`; a future
/// cache-aware caller can continue a sequence by passing positions offset by
/// the cache length instead of always starting at 0.
fn rotate_heads_rope(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    theta: f64,
    position_ids: &[usize],
) {
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    let row_width = num_heads * head_dim;
    for (row, &pos) in position_ids.iter().enumerate() {
        let row_off = row * row_width;
        for h in 0..num_heads {
            let head_off = row_off + h * head_dim;
            for i in 0..half {
                let freq = 1.0 / theta.powf(2.0 * i as f64 / head_dim as f64);
                let angle = (pos as f64 * freq) as f32;
                let cos_v = angle.cos();
                let sin_v = angle.sin();
                let x1 = data[head_off + i];
                let x2 = data[head_off + i + half];
                data[head_off + i] = x1 * cos_v - x2 * sin_v;
                data[head_off + i + half] = x1 * sin_v + x2 * cos_v;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// GEGLU MLP
// ---------------------------------------------------------------------------

/// Gemma-2 MLP block using GEGLU activation.
///
/// Architecture: gate_proj + up_proj (combined) → GEGLU → down_proj
pub struct Gemma2GegluMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    device: Device,
}

impl Gemma2GegluMlp {
    pub fn new(config: &Gemma2Config, device: Device) -> Result<Self> {
        let gate_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let up_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let down_proj =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, false, device);
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Gemma2GegluMlp {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;

        let activated = match (&gate_out, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => {
                let g_slice = g
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("gemma2_geglu", "gate tensor not contiguous"))?;
                let u_slice = u
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("gemma2_geglu", "up tensor not contiguous"))?;
                let result = geglu(g_slice, u_slice);
                let shape = g.shape().to_vec();
                Tensor::from_vec(result, &shape)?
            },
            _ => {
                return Err(tensor_op_error(
                    "gemma2_geglu",
                    "gate and up tensors must be F32",
                ))
            },
        };

        self.down_proj.forward(activated)
    }
}

// ---------------------------------------------------------------------------
// Attention scaling
// ---------------------------------------------------------------------------

/// Derive the attention scaling factor from `query_pre_attn_scalar`, per
/// Gemma-2's published formula: `scale = query_pre_attn_scalar ^ -0.5`.
///
/// `query_pre_attn_scalar` is the raw HuggingFace `config.json` value (see
/// [`Gemma2Config::query_pre_attn_scalar`](super::config::Gemma2Config)'s doc
/// comment); it must **not** be used directly as the scale, and it is not
/// always `head_dim` -- Gemma-2-27B deliberately sets it to `144` despite
/// `head_dim=128`. Callers must ensure `query_pre_attn_scalar > 0.0` (raising
/// zero or a negative number to the `-0.5` power yields infinity or NaN);
/// [`Gemma2Attention::new`] enforces this before calling.
pub fn attention_scale_from_query_pre_attn_scalar(query_pre_attn_scalar: f64) -> f32 {
    (query_pre_attn_scalar as f32).powf(-0.5)
}

// ---------------------------------------------------------------------------
// Attention
// ---------------------------------------------------------------------------

/// Gemma-2 attention layer.
///
/// Supports:
/// - Grouped Query Attention (GQA)
/// - Soft-capping of attention scores (`tanh(score / 50.0) * 50.0`)
/// - Alternating local (sliding window) and global attention
/// - Causal masking
pub struct Gemma2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Gemma2RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// `true` when this layer uses local (sliding window) attention
    is_local: bool,
    sliding_window: usize,
    attention_logit_softcapping: f64,
    /// Multiplier applied to raw `Q @ K^T` scores before soft-capping:
    /// `attention_scale_from_query_pre_attn_scalar(config.query_pre_attn_scalar)`,
    /// i.e. `query_pre_attn_scalar ^ -0.5`. This is *not* simply
    /// `config.query_pre_attn_scalar` used directly (that field holds the
    /// raw HF `config.json` value, not the scale -- see its doc comment).
    scale: f32,
    device: Device,
}

impl Gemma2Attention {
    pub fn new(config: &Gemma2Config, layer_idx: usize, device: Device) -> Result<Self> {
        if config.query_pre_attn_scalar.is_nan() || config.query_pre_attn_scalar <= 0.0 {
            // Raised to the -0.5 power below; zero, negative, or NaN would
            // silently produce an infinite or NaN attention scale instead of
            // a clear construction-time error.
            return Err(tensor_op_error(
                "gemma2_attn",
                "query_pre_attn_scalar must be > 0",
            ));
        }
        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            false,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            false,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            false,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            false,
            device,
        );
        let rotary_emb = Gemma2RotaryEmbedding::new(config, device);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            is_local: Gemma2Config::is_local_layer(layer_idx),
            sliding_window: config.sliding_window,
            attention_logit_softcapping: config.attention_logit_softcapping,
            scale: attention_scale_from_query_pre_attn_scalar(config.query_pre_attn_scalar),
            device,
        })
    }

    /// Returns `true` when this layer uses local (sliding window) attention.
    pub fn is_local(&self) -> bool {
        self.is_local
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Gemma2Attention {
    type Input = Tensor;
    type Output = Tensor;

    /// Real scaled dot-product attention: RoPE, `Q @ K^T`, attention-logit
    /// soft-capping (`tanh(s/cap)*cap`), causal masking (plus a sliding
    /// window on local layers), softmax, and `@ V`.
    ///
    /// Order matches the published Gemma-2 architecture: scale → soft-cap →
    /// mask → softmax.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let (mut q_data, mut k_data, v_data) = match (&q, &k, &v) {
            (Tensor::F32(qd), Tensor::F32(kd), Tensor::F32(vd)) => (
                qd.as_slice()
                    .ok_or_else(|| tensor_op_error("gemma2_attn", "q not contiguous"))?
                    .to_vec(),
                kd.as_slice()
                    .ok_or_else(|| tensor_op_error("gemma2_attn", "k not contiguous"))?
                    .to_vec(),
                vd.as_slice()
                    .ok_or_else(|| tensor_op_error("gemma2_attn", "v not contiguous"))?
                    .to_vec(),
            ),
            _ => return Err(tensor_op_error("gemma2_attn", "q, k, v must be F32")),
        };

        if self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            return Err(tensor_op_error(
                "gemma2_attn",
                "num_heads must be a positive multiple of num_kv_heads",
            ));
        }
        let q_width = self.num_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        if q_width == 0 || !q_data.len().is_multiple_of(q_width) {
            return Err(tensor_op_error(
                "gemma2_attn",
                "q size inconsistent with num_heads * head_dim",
            ));
        }
        let seq_len = q_data.len() / q_width;
        if k_data.len() != seq_len * kv_width || v_data.len() != seq_len * kv_width {
            return Err(tensor_op_error(
                "gemma2_attn",
                "k/v size inconsistent with num_kv_heads * head_dim",
            ));
        }

        // Sliding window only applies on local layers; a misconfigured
        // window of 0 would mask every key (including self), producing an
        // empty softmax, so reject it explicitly instead of emitting NaN.
        let window = if self.is_local {
            if self.sliding_window == 0 {
                return Err(tensor_op_error(
                    "gemma2_attn",
                    "sliding_window must be > 0 on a local attention layer",
                ));
            }
            Some(self.sliding_window)
        } else {
            None
        };

        let position_ids: Vec<usize> = (0..seq_len).collect();
        rotate_heads_rope(
            &mut q_data,
            self.num_heads,
            self.head_dim,
            self.rotary_emb.rope_theta,
            &position_ids,
        );
        rotate_heads_rope(
            &mut k_data,
            self.num_kv_heads,
            self.head_dim,
            self.rotary_emb.rope_theta,
            &position_ids,
        );

        let group = self.num_heads / self.num_kv_heads;
        let scale = self.scale;
        let mut out = vec![0f32; seq_len * q_width];

        for h in 0..self.num_heads {
            let kv_h = h / group;
            for i in 0..seq_len {
                let q_off = i * q_width + h * self.head_dim;
                // Causal (+ sliding window) key positions for query i.
                let mut scores = Vec::with_capacity(i + 1);
                let mut key_positions = Vec::with_capacity(i + 1);
                for j in 0..=i {
                    if let Some(w) = window {
                        if i - j >= w {
                            continue;
                        }
                    }
                    let k_off = j * kv_width + kv_h * self.head_dim;
                    let dot: f32 =
                        (0..self.head_dim).map(|d| q_data[q_off + d] * k_data[k_off + d]).sum();
                    let scaled = dot * scale;
                    scores.push(soft_cap(scaled, self.attention_logit_softcapping));
                    key_positions.push(j);
                }
                let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut weights = vec![0f32; scores.len()];
                let mut sum = 0f32;
                for (idx, &s) in scores.iter().enumerate() {
                    let e = (s - max_val).exp();
                    weights[idx] = e;
                    sum += e;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let out_off = i * q_width + h * self.head_dim;
                for (idx, &j) in key_positions.iter().enumerate() {
                    let wn = weights[idx] * inv_sum;
                    let v_off = j * kv_width + kv_h * self.head_dim;
                    for d in 0..self.head_dim {
                        out[out_off + d] += wn * v_data[v_off + d];
                    }
                }
            }
        }

        let attended = Tensor::from_vec(out, &[seq_len, q_width])?;
        self.o_proj.forward(attended)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

/// Gemma-2 decoder layer with pre- and post-normalization.
///
/// Architecture:
/// ```text
/// x → pre_norm → attention → post_norm → + x → pre_ff_norm → mlp → post_ff_norm → + x
/// ```
pub struct Gemma2DecoderLayer {
    self_attn: Gemma2Attention,
    mlp: Gemma2GegluMlp,
    input_layernorm: Gemma2RmsNorm,
    post_attention_layernorm: Gemma2RmsNorm,
    pre_feedforward_layernorm: Gemma2RmsNorm,
    post_feedforward_layernorm: Gemma2RmsNorm,
    device: Device,
}

impl Gemma2DecoderLayer {
    pub fn new(config: &Gemma2Config, layer_idx: usize, device: Device) -> Result<Self> {
        let self_attn = Gemma2Attention::new(config, layer_idx, device)?;
        let mlp = Gemma2GegluMlp::new(config, device)?;
        let input_layernorm = Gemma2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let post_attention_layernorm =
            Gemma2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let pre_feedforward_layernorm =
            Gemma2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let post_feedforward_layernorm =
            Gemma2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            pre_feedforward_layernorm,
            post_feedforward_layernorm,
            device,
        })
    }

    pub fn is_local(&self) -> bool {
        self.self_attn.is_local()
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Gemma2DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // --- Attention sublayer with pre- and post-norm ---
        let pre_normed = self.input_layernorm.forward(input.clone())?;
        let attn_out = self.self_attn.forward(pre_normed)?;
        let post_attn = self.post_attention_layernorm.forward(attn_out)?;
        // Residual add
        let hidden = input.add(&post_attn)?;

        // --- MLP sublayer with pre- and post-norm ---
        let pre_ff_normed = self.pre_feedforward_layernorm.forward(hidden.clone())?;
        let mlp_out = self.mlp.forward(pre_ff_normed)?;
        let post_ff = self.post_feedforward_layernorm.forward(mlp_out)?;
        // Residual add
        hidden.add(&post_ff)
    }
}

// ---------------------------------------------------------------------------
// Gemma2Model
// ---------------------------------------------------------------------------

/// Gemma-2 base model (embedding + transformer layers + final norm).
pub struct Gemma2Model {
    config: Gemma2Config,
    embed_tokens: Embedding,
    layers: Vec<Gemma2DecoderLayer>,
    norm: Gemma2RmsNorm,
    device: Device,
}

impl Gemma2Model {
    pub fn new(config: Gemma2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Gemma2Config, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(Gemma2DecoderLayer::new(&config, layer_idx, device)?);
        }

        let norm = Gemma2RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            device,
        })
    }

    pub fn config(&self) -> &Gemma2Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for Gemma2Model {
    type Config = Gemma2Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let token_ids: Vec<u32> = match &input_ids {
            Tensor::I64(arr) => arr.as_slice().unwrap_or(&[]).iter().map(|&x| x as u32).collect(),
            Tensor::F32(arr) => {
                arr.as_slice().unwrap_or(&[]).iter().map(|&x| x.round() as u32).collect()
            },
            _ => {
                return Err(tensor_op_error(
                    "gemma2_forward",
                    "input_ids must be I64 or F32",
                ))
            },
        };

        let mut hidden_states = self.embed_tokens.forward(token_ids)?;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        self.norm.forward(hidden_states)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // `Model::load_pretrained(&mut dyn Read)` is a legacy interface with
        // no defined weight format (see the note on
        // `WeightLoader::load_weights_into_model` in
        // `trustformers_core::utils::weight_loading`: an earlier revision of
        // this trait method serialised tensors into an invented envelope
        // that no model could parse, so the weights never actually reached
        // the model). `Gemma2Model` has no weight-loading path implemented
        // at all (unlike e.g. `GemmaModel`/`MistralModel`/`QwenModel`, which
        // provide a real `load_from_path`/`load_from_huggingface` on their
        // `*ForCausalLM` wrapper). Silently returning `Ok(())` here would
        // leave the model's freshly-initialised (effectively random)
        // weights in place while claiming the load succeeded, so report
        // this honestly as unsupported instead.
        Err(TrustformersError::not_implemented(
            "Gemma2Model::load_pretrained: no weight-loading implementation exists for Gemma2 \
             yet; there is no `load_from_path`/`load_from_huggingface` to delegate to"
                .to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let hs = self.config.hidden_size;
        let is = self.config.intermediate_size;
        let nl = self.config.num_hidden_layers;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let vs = self.config.vocab_size;

        let embed = vs * hs;
        let attn = hs * nh * hd + hs * nkv * hd + hs * nkv * hd + nh * hd * hs;
        let mlp = hs * is + hs * is + is * hs;
        let norms = 4 * hs;
        let final_norm = hs;

        embed + nl * (attn + mlp + norms) + final_norm
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::device::Device;

    /// LCG-based pseudo-random number generator (no rand/ndarray).
    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|_| lcg_next(&mut state)).collect()
    }

    fn tiny_config() -> Gemma2Config {
        Gemma2Config {
            vocab_size: 64,
            hidden_size: 8,
            num_hidden_layers: 2,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            intermediate_size: 16,
            head_dim: 4,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: 16,
            attention_logit_softcapping: 50.0,
            final_logit_softcapping: 30.0,
            // Raw HF-style value (== head_dim, the typical convention), not
            // a precomputed scale -- see `Gemma2Config::query_pre_attn_scalar`.
            query_pre_attn_scalar: 4.0,
            model_type: "gemma2-test".to_string(),
        }
    }

    // -- soft_cap --

    #[test]
    fn test_soft_cap_at_zero() {
        assert_eq!(soft_cap(0.0, 50.0), 0.0, "soft_cap(0) must be 0");
    }

    #[test]
    fn test_soft_cap_positive_saturates_below_cap() {
        let cap = 50.0_f64;
        let large = 10_000.0_f32;
        let result = soft_cap(large, cap);
        // f32 tanh saturates to exactly 1.0 for large inputs, so result may equal cap
        assert!(
            result <= cap as f32,
            "soft_cap must be at most cap for large input"
        );
        assert!(result > 0.0, "soft_cap of positive must be positive");
    }

    #[test]
    fn test_soft_cap_negative_saturates_above_neg_cap() {
        let cap = 50.0_f64;
        let result = soft_cap(-10_000.0, cap);
        // f32 tanh saturates to exactly -1.0 for large negatives, so result may equal -cap
        assert!(
            result >= -(cap as f32),
            "soft_cap must be at least -cap for large negative input"
        );
        assert!(result < 0.0, "soft_cap of negative must be negative");
    }

    #[test]
    fn test_soft_cap_identity_region() {
        // For very small x/cap, tanh(x/c)*c ≈ x
        let x = 0.01_f32;
        let cap = 50.0_f64;
        let result = soft_cap(x, cap);
        assert!(
            (result - x).abs() < 1e-4,
            "soft_cap should be near identity for small inputs"
        );
    }

    #[test]
    fn test_apply_soft_cap_inplace_length_preserved() {
        let mut data = lcg_vec(16, 42);
        let original_len = data.len();
        apply_soft_cap_inplace(&mut data, 30.0);
        assert_eq!(
            data.len(),
            original_len,
            "inplace soft-cap must not change length"
        );
    }

    #[test]
    fn test_apply_soft_cap_inplace_bounds() {
        let mut data: Vec<f32> = vec![100.0, -100.0, 0.0, 50.0, -50.0];
        apply_soft_cap_inplace(&mut data, 30.0);
        for &v in &data {
            assert!(
                v < 30.0 && v > -30.0,
                "all values must be within cap bounds after inplace"
            );
        }
    }

    // -- gelu --

    #[test]
    fn test_gelu_at_zero() {
        assert!((gelu(0.0)).abs() < 1e-6, "gelu(0) must be ~0");
    }

    #[test]
    fn test_gelu_positive_for_positive_input() {
        assert!(gelu(1.0) > 0.0, "gelu(1) must be positive");
    }

    #[test]
    fn test_gelu_approaches_identity_for_large_input() {
        // For large x, gelu(x) ≈ x
        let x = 10.0_f32;
        let result = gelu(x);
        assert!(
            (result - x).abs() < 0.01,
            "gelu approaches identity for large positive"
        );
    }

    // -- geglu --

    #[test]
    fn test_geglu_zero_gate_gives_zero() {
        let gate = vec![0.0_f32; 8];
        let up = lcg_vec(8, 1);
        let result = geglu(&gate, &up);
        for &v in &result {
            assert!(v.abs() < 1e-6, "geglu with zero gate must output zeros");
        }
    }

    #[test]
    fn test_geglu_length_matches_input() {
        let gate = lcg_vec(8, 2);
        let up = lcg_vec(8, 3);
        let result = geglu(&gate, &up);
        assert_eq!(
            result.len(),
            8,
            "geglu output length must match input length"
        );
    }

    // -- Gemma2RmsNorm --

    #[test]
    fn test_rmsnorm_ones_weight_normalises_to_unit_rms() {
        let norm =
            Gemma2RmsNorm::new(4, 1e-6, Device::CPU).expect("RmsNorm construction must succeed");
        let input = Tensor::from_vec(vec![3.0_f32, 4.0, 0.0, 0.0], &[4])
            .expect("tensor creation must succeed");
        let output = norm.forward(input).expect("rmsnorm forward must succeed");
        let vals: Vec<f32> = match &output {
            Tensor::F32(arr) => arr.as_slice().expect("output must be contiguous").to_vec(),
            _ => panic!("expected F32 output"),
        };
        assert_eq!(vals.len(), 4, "output length must match input");
        // When weight=ones, output = input/rms; check rms ~ 1
        let rms = (vals.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-4,
            "RMSNorm output rms must be ~1.0, got {rms}"
        );
    }

    #[test]
    fn test_rmsnorm_device_accessor() {
        let norm = Gemma2RmsNorm::new(8, 1e-6, Device::CPU).expect("RmsNorm must build");
        assert_eq!(
            norm.device(),
            Device::CPU,
            "device accessor must return CPU"
        );
    }

    // -- Gemma2Config helpers --

    #[test]
    fn test_is_local_layer_even_layers() {
        assert!(
            Gemma2Config::is_local_layer(0),
            "layer 0 (even) must be local"
        );
        assert!(
            Gemma2Config::is_local_layer(2),
            "layer 2 (even) must be local"
        );
        assert!(
            Gemma2Config::is_local_layer(4),
            "layer 4 (even) must be local"
        );
    }

    #[test]
    fn test_is_local_layer_odd_layers() {
        assert!(
            !Gemma2Config::is_local_layer(1),
            "layer 1 (odd) must be global"
        );
        assert!(
            !Gemma2Config::is_local_layer(3),
            "layer 3 (odd) must be global"
        );
        assert!(
            !Gemma2Config::is_local_layer(5),
            "layer 5 (odd) must be global"
        );
    }

    #[test]
    fn test_kv_group_size_gemma2_2b() {
        let cfg = Gemma2Config::gemma2_2b();
        assert_eq!(
            cfg.kv_group_size(),
            cfg.num_attention_heads / cfg.num_key_value_heads,
            "kv_group_size must equal num_attention_heads / num_key_value_heads",
        );
    }

    #[test]
    fn test_gemma2_2b_config_values() {
        let cfg = Gemma2Config::gemma2_2b();
        assert_eq!(cfg.hidden_size, 2304, "2B hidden_size must be 2304");
        assert_eq!(cfg.num_hidden_layers, 26, "2B must have 26 layers");
        assert_eq!(cfg.num_attention_heads, 8, "2B must have 8 heads");
        assert_eq!(
            cfg.head_dim, 256,
            "head_dim must be 256 for all Gemma-2 variants"
        );
    }

    // -- Gemma2RotaryEmbedding --

    #[test]
    fn test_rope_apply_preserves_norm() {
        let cfg = tiny_config();
        let rope = Gemma2RotaryEmbedding::new(&cfg, Device::CPU);
        let n = cfg.head_dim;
        let mut q = lcg_vec(n, 7);
        let mut k = lcg_vec(n, 8);
        let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_before: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope.apply(&mut q, &mut k, 1);
        let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        let k_norm_after: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "RoPE must be norm-preserving for Q, before={q_norm_before} after={q_norm_after}",
        );
        assert!(
            (k_norm_before - k_norm_after).abs() < 1e-4,
            "RoPE must be norm-preserving for K",
        );
    }

    #[test]
    fn test_rope_zero_position_is_identity() {
        let cfg = tiny_config();
        let rope = Gemma2RotaryEmbedding::new(&cfg, Device::CPU);
        let n = cfg.head_dim;
        let mut q = lcg_vec(n, 9);
        let mut k = lcg_vec(n, 10);
        let q_orig = q.clone();
        let _k_orig = k.clone();
        rope.apply(&mut q, &mut k, 1); // seq_len=1 → only position 0
                                       // Position 0 → angle=0 → cos=1, sin=0 → identity rotation
        for i in 0..n / 2 {
            assert!(
                (q[i] - q_orig[i]).abs() < 1e-5,
                "RoPE at pos=0 must be identity for q[{i}]",
            );
        }
    }

    // -- Gemma2Model --

    #[test]
    fn test_gemma2_model_construction() {
        let cfg = tiny_config();
        let model = Gemma2Model::new(cfg).expect("Gemma2Model construction must succeed");
        assert_eq!(model.layers.len(), 2, "model must have 2 decoder layers");
    }

    #[test]
    fn test_gemma2_model_forward_shape() {
        let cfg = tiny_config();
        let model = Gemma2Model::new(cfg.clone()).expect("model must build");
        // Single token forward: I64 input
        let input = Tensor::from_vec(vec![0_f32], &[1]).expect("token tensor must build");
        let output = model.forward(input).expect("forward must succeed");
        let out_len = output.shape().iter().product::<usize>();
        assert!(out_len > 0, "output must be non-empty");
        // hidden_size tokens = 8 elements
        assert_eq!(
            out_len, cfg.hidden_size,
            "output length must equal hidden_size for single token"
        );
    }

    #[test]
    fn test_gemma2_model_num_parameters_positive() {
        let cfg = tiny_config();
        let model = Gemma2Model::new(cfg).expect("model must build");
        let params = model.num_parameters();
        assert!(params > 0, "model must have a positive parameter count");
    }

    /// Regression: `load_pretrained` must NOT silently report success while
    /// leaving the model's random initial weights untouched. It previously
    /// read the buffer, checked it was non-empty, and returned `Ok(())`
    /// without parsing anything (a fabricated success). It must now report
    /// a structured "not implemented" error instead.
    #[test]
    fn test_gemma2_model_load_pretrained_reports_not_implemented_instead_of_fake_success() {
        use trustformers_core::traits::Model;
        let cfg = tiny_config();
        let mut model = Gemma2Model::new(cfg).expect("model must build");
        let mut data: &[u8] = b"not a real checkpoint, but not empty either";
        let result = model.load_pretrained(&mut data);
        assert!(
            result.is_err(),
            "load_pretrained must fail rather than silently succeed with no weights loaded"
        );
    }

    #[test]
    fn test_gemma2_decoder_layer_is_local_reflects_layer_idx() {
        let cfg = tiny_config();
        let layer0 = Gemma2DecoderLayer::new(&cfg, 0, Device::CPU)
            .expect("layer 0 construction must succeed");
        let layer1 = Gemma2DecoderLayer::new(&cfg, 1, Device::CPU)
            .expect("layer 1 construction must succeed");
        assert!(layer0.is_local(), "layer 0 must be local");
        assert!(!layer1.is_local(), "layer 1 must be global");
    }

    #[test]
    fn test_gemma2_attention_layer_0_is_local() {
        let cfg = tiny_config();
        let attn = Gemma2Attention::new(&cfg, 0, Device::CPU)
            .expect("attention construction must succeed");
        assert!(attn.is_local(), "attention at layer 0 must be local");
    }

    #[test]
    fn test_gemma2_attention_layer_1_is_global() {
        let cfg = tiny_config();
        let attn = Gemma2Attention::new(&cfg, 1, Device::CPU)
            .expect("attention construction must succeed");
        assert!(!attn.is_local(), "attention at layer 1 must be global");
    }

    #[test]
    fn test_gemma2_config_validate_ok() {
        let cfg = tiny_config();
        assert!(cfg.validate().is_ok(), "valid config must pass validation");
    }

    #[test]
    fn test_gemma2_config_validate_zero_heads_fails() {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = 0;
        assert!(
            cfg.validate().is_err(),
            "zero attention heads must fail validation"
        );
    }

    // -- rotate_heads_rope (multi-head RoPE primitive) --
    //
    // Regression coverage for the bug class where RoPE only rotated the
    // first `head_dim`-wide block of a multi-head row and left every other
    // head positionally blind.

    #[test]
    fn test_rotate_heads_rope_rotates_every_head() {
        let head_dim = 4;
        let num_heads = 2;
        let mut data = vec![1.0f32; num_heads * head_dim]; // seq_len=1
        let original = data.clone();
        rotate_heads_rope(&mut data, num_heads, head_dim, 10000.0, &[7]);
        let head0_changed = data[0..head_dim]
            .iter()
            .zip(&original[0..head_dim])
            .any(|(a, b)| (a - b).abs() > 1e-4);
        let head1_changed = data[head_dim..2 * head_dim]
            .iter()
            .zip(&original[head_dim..2 * head_dim])
            .any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(head0_changed, "head 0 must rotate at a non-zero position");
        assert!(
            head1_changed,
            "head 1 must ALSO rotate at a non-zero position, not just head 0"
        );
    }

    #[test]
    fn test_rotate_heads_rope_position_zero_is_identity() {
        let head_dim = 4;
        let mut data = vec![1.0f32, 2.0, 3.0, 4.0];
        let original = data.clone();
        rotate_heads_rope(&mut data, 1, head_dim, 10000.0, &[0]);
        for (a, b) in data.iter().zip(original.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "position 0 must be an identity rotation"
            );
        }
    }

    #[test]
    fn test_rotate_heads_rope_differs_by_position() {
        let head_dim = 4;
        let mut at_pos0 = vec![1.0f32; head_dim];
        let mut at_pos5 = vec![1.0f32; head_dim];
        rotate_heads_rope(&mut at_pos0, 1, head_dim, 10000.0, &[0]);
        rotate_heads_rope(&mut at_pos5, 1, head_dim, 10000.0, &[5]);
        let differs = at_pos0.iter().zip(at_pos5.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "RoPE must rotate differently at different positions"
        );
    }

    #[test]
    fn test_gemma2_model_forward_f32_input() {
        let cfg = tiny_config();
        let model = Gemma2Model::new(cfg.clone()).expect("model must build");
        let input = Tensor::from_vec(vec![1.0_f32], &[1]).expect("f32 token tensor must build");
        let output = model.forward(input).expect("forward with f32 input must succeed");
        let out_len = output.shape().iter().product::<usize>();
        assert_eq!(
            out_len, cfg.hidden_size,
            "f32 input must produce correct output shape"
        );
    }

    // -- attention_scale_from_query_pre_attn_scalar / query_pre_attn_scalar --
    //
    // Regression coverage for `Gemma2Config::query_pre_attn_scalar` holding
    // the raw HuggingFace `config.json` value (the denominator the scale is
    // derived from via `^-0.5`) rather than a precomputed scale. Before this
    // fix, `Gemma2Attention` used `config.query_pre_attn_scalar` directly as
    // the multiplicative scale; deserializing a real Gemma-2 `config.json`
    // (which carries `"query_pre_attn_scalar": 256` for 2B/9B) into that
    // field would then have scaled every attention score by 256 instead of
    // by `256^-0.5 == 1/16` -- a 4096x error with no error returned.

    #[test]
    fn test_attention_scale_matches_hf_formula_for_2b_9b() {
        // Published HF config value for Gemma-2 2B/9B: query_pre_attn_scalar
        // == head_dim == 256, so scale == 256^-0.5 == 1/16.
        let scale = attention_scale_from_query_pre_attn_scalar(256.0);
        assert!(
            (scale - 0.0625).abs() < 1e-6,
            "expected 0.0625, got {scale}"
        );
    }

    #[test]
    fn test_attention_scale_27b_diverges_from_naive_head_dim_formula() {
        // Gemma-2-27B deliberately sets query_pre_attn_scalar=144 despite
        // head_dim=128, so the correct scale must differ from the naive (and
        // for 27B, wrong) `head_dim^-0.5` formula.
        let correct = attention_scale_from_query_pre_attn_scalar(144.0);
        let naive_head_dim_scale = 1.0_f32 / 128.0_f32.sqrt();
        assert!(
            (correct - naive_head_dim_scale).abs() > 1e-3,
            "27B's scale must differ from head_dim^-0.5: correct={correct}, naive={naive_head_dim_scale}"
        );
        assert!(
            (correct - 1.0 / 12.0_f32).abs() < 1e-4,
            "144^-0.5 must equal 1/12, got {correct}"
        );
    }

    #[test]
    fn test_gemma2_attention_rejects_non_positive_query_pre_attn_scalar() {
        let mut cfg = tiny_config();
        cfg.query_pre_attn_scalar = 0.0;
        let result = Gemma2Attention::new(&cfg, 0, Device::CPU);
        assert!(
            result.is_err(),
            "query_pre_attn_scalar=0 must be rejected at construction rather than silently \
             producing an infinite/NaN scale"
        );
    }

    #[test]
    fn test_gemma2_attention_scale_field_reflects_raw_hf_config_value() {
        // Constructing with the real HF raw value (256.0, as it would
        // appear if a real Gemma-2 `config.json` were deserialized) must
        // derive the correct scale (1/16), not use 256.0 as the scale
        // itself.
        let mut cfg = tiny_config();
        cfg.query_pre_attn_scalar = 256.0;
        let attn = Gemma2Attention::new(&cfg, 0, Device::CPU).expect("attention must build");
        assert!(
            (attn.scale - 0.0625).abs() < 1e-6,
            "derived scale must be 256^-0.5 = 0.0625, got {}",
            attn.scale
        );
    }

    fn scale_regression_config() -> Gemma2Config {
        Gemma2Config {
            vocab_size: 8,
            hidden_size: 2,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: 4,
            head_dim: 2,
            max_position_embeddings: 8,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-6,
            sliding_window: 16,
            attention_logit_softcapping: 1.0e6, // near-identity soft-cap for this test's small scores
            final_logit_softcapping: 30.0,
            query_pre_attn_scalar: 256.0, // the real HF raw value for 2B/9B
            model_type: "gemma2-scale-test".to_string(),
        }
    }

    /// End-to-end regression test through the public `Layer::forward` API:
    /// with `query_pre_attn_scalar` set to the real HF raw value (256.0),
    /// the attention output must match a hand-derived reference computed
    /// with the CORRECT scale (`256^-0.5 == 1/16`), not the value the old
    /// (buggy) code would have produced by using 256.0 directly as the
    /// scale (which saturates softmax toward a one-hot distribution).
    #[test]
    fn test_gemma2_attention_forward_uses_derived_scale_not_raw_query_pre_attn_scalar() {
        let cfg = scale_regression_config();
        let mut attn = Gemma2Attention::new(&cfg, 1, Device::CPU).expect("global attention layer");
        assert!(
            !attn.is_local(),
            "layer 1 must be global so a sliding window can't affect this test"
        );

        // Identity Q/K/V/O projections make Q == K == V == the raw input,
        // so the whole computation is hand-traceable.
        let identity =
            Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], &[2, 2]).expect("identity tensor");
        attn.q_proj.set_weight(identity.clone()).expect("set q_proj weight");
        attn.k_proj.set_weight(identity.clone()).expect("set k_proj weight");
        attn.v_proj.set_weight(identity.clone()).expect("set v_proj weight");
        attn.o_proj.set_weight(identity).expect("set o_proj weight");

        // token0 = [1, 0] at position 0, token1 = [0, 1] at position 1.
        let input = Tensor::from_vec(vec![1.0_f32, 0.0, 0.0, 1.0], &[2, 2]).expect("input tensor");
        let output = attn.forward(input).expect("attention forward");
        let out_data = match output {
            Tensor::F32(arr) => arr.as_slice().expect("contiguous").to_vec(),
            _ => panic!("expected F32 output"),
        };

        // --- Independent reference, computed directly from the published
        // equations (not by calling this module's internal helpers) ---
        //
        // RoPE with head_dim=2, theta=10000 has a single frequency of 1.0.
        // Position 0 is an identity rotation; position 1 rotates [0,1] by
        // angle=1.0 radian via (x1,x2) -> (x1*cos - x2*sin, x1*sin + x2*cos).
        let angle = 1.0_f32;
        let (sin_a, cos_a) = (angle.sin(), angle.cos());
        let q1_rot = [0.0 * cos_a - 1.0 * sin_a, 0.0 * sin_a + 1.0 * cos_a];
        let k0_rot = [1.0_f32, 0.0_f32];
        let k1_rot = q1_rot; // identical input and rotation to q1

        let correct_scale = attention_scale_from_query_pre_attn_scalar(cfg.query_pre_attn_scalar);
        assert!(
            (correct_scale - 0.0625).abs() < 1e-6,
            "sanity: 256^-0.5 must be 1/16, got {correct_scale}"
        );

        let dot = |a: [f32; 2], b: [f32; 2]| a[0] * b[0] + a[1] * b[1];
        let s0 = dot(q1_rot, k0_rot) * correct_scale;
        let s1 = dot(q1_rot, k1_rot) * correct_scale;
        let m = s0.max(s1);
        let (e0, e1) = ((s0 - m).exp(), (s1 - m).exp());
        let (w0, w1) = (e0 / (e0 + e1), e1 / (e0 + e1));
        // V is never rotated: V0=[1,0], V1=[0,1] (the raw input rows).
        let expected_row1 = [w0, w1];

        // Sanity: the reference itself must not already be a saturated
        // ~[0, 1] distribution, or this test couldn't tell the fix apart
        // from the bug it targets.
        assert!(
            expected_row1[1] < 0.9,
            "sanity: reference must not be saturated, got {expected_row1:?}"
        );

        // Row 0 attends only to itself (causal, single-key softmax): its
        // output is exactly V[0] regardless of scale, so it only checks
        // basic wiring, not the scale itself.
        assert!(
            (out_data[0] - 1.0).abs() < 1e-4,
            "row0[0]: got {}",
            out_data[0]
        );
        assert!(
            (out_data[1] - 0.0).abs() < 1e-4,
            "row0[1]: got {}",
            out_data[1]
        );

        // Row 1 is the discriminating check. If `forward` used the raw
        // `query_pre_attn_scalar` (256.0) directly as the scale instead of
        // `256^-0.5`, the pre-softmax scores would be ~4096x larger,
        // saturating softmax toward `[~0, ~1]` instead of matching
        // `expected_row1`.
        assert!(
            (out_data[2] - expected_row1[0]).abs() < 1e-3,
            "row1[0]: expected {}, got {}",
            expected_row1[0],
            out_data[2]
        );
        assert!(
            (out_data[3] - expected_row1[1]).abs() < 1e-3,
            "row1[1]: expected {}, got {} (a value near 1.0 indicates the raw \
             query_pre_attn_scalar was used directly as the scale)",
            expected_row1[1],
            out_data[3]
        );
    }
}
