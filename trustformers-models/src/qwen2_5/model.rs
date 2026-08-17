//! # Qwen2.5 Model Implementation
//!
//! Core architecture components:
//! - `Qwen25RmsNorm` — standard RMS normalisation
//! - `Qwen25RotaryEmbedding` — RoPE with configurable theta
//! - `Qwen25Attention` — Grouped Query Attention (GQA) with optional sliding window
//! - `Qwen25MLP` — SwiGLU FFN (gate_proj × silu, up_proj, down_proj)
//! - `Qwen25DecoderLayer` — single transformer layer
//! - `Qwen25Model` — full stack of decoder layers

use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

use super::config::Qwen25Config;

// ---------------------------------------------------------------------------
// Activation helpers
// ---------------------------------------------------------------------------

/// SiLU (Swish): `x * sigmoid(x)`.
pub fn silu(x: f32) -> f32 {
    x / (1.0 + (-x).exp())
}

/// SwiGLU gate: `silu(gate) * up`.
pub fn swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter().zip(up.iter()).map(|(&g, &u)| silu(g) * u).collect()
}

// ---------------------------------------------------------------------------
// RMSNorm
// ---------------------------------------------------------------------------

/// Qwen2.5 RMSNorm layer.
///
/// `output = weight * (input / sqrt(mean(input²) + eps))`
pub struct Qwen25RmsNorm {
    weight: Tensor,
    eps: f32,
    size: usize,
    device: Device,
}

impl Qwen25RmsNorm {
    pub fn new(size: usize, eps: f64, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[size])?;
        Ok(Self {
            weight,
            eps: eps as f32,
            size,
            device,
        })
    }

    /// Dimension this norm operates on.
    pub fn size(&self) -> usize {
        self.size
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Qwen25RmsNorm {
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
                        "qwen25_rmsnorm",
                        "weight tensor must be F32",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "qwen25_rmsnorm",
                "input tensor must be F32",
            )),
        }
    }
}

// ---------------------------------------------------------------------------
// Rotary Position Embedding
// ---------------------------------------------------------------------------

/// RoPE for Qwen2.5 with configurable base frequency.
///
/// Qwen2.5 uses `rope_theta = 1_000_000` by default, enabling long-context stability.
pub struct Qwen25RotaryEmbedding {
    head_dim: usize,
    rope_theta: f64,
    #[allow(dead_code)]
    device: Device,
}

impl Qwen25RotaryEmbedding {
    pub fn new(config: &Qwen25Config, device: Device) -> Self {
        Self {
            head_dim: config.head_dim,
            rope_theta: config.rope_theta,
            device,
        }
    }

    /// Apply RoPE in-place to flat Q and K slices of size `seq_len * head_dim` each.
    pub fn apply(&self, q: &mut [f32], k: &mut [f32], seq_len: usize) {
        let half = self.head_dim / 2;
        if half == 0 {
            return;
        }
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
/// head count) tensors are fully rotated, not just the first head.
///
/// `pub(crate)` (rather than private) solely so the regression test in
/// `qwen2_5::tests` can exercise it directly.
pub(crate) fn rotate_heads_rope(
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
// Grouped Query Attention
// ---------------------------------------------------------------------------

/// Qwen2.5 Grouped Query Attention (GQA) with optional sliding window.
///
/// Projection dimensions:
/// - `q_proj`: `hidden_size → num_attention_heads * head_dim`
/// - `k_proj`: `hidden_size → num_key_value_heads * head_dim`
/// - `v_proj`: `hidden_size → num_key_value_heads * head_dim`
/// - `o_proj`: `num_attention_heads * head_dim → hidden_size`
pub struct Qwen25Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: Qwen25RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// `Some(window)` when this layer uses sliding window attention.
    sliding_window: Option<usize>,
    device: Device,
}

impl Qwen25Attention {
    pub fn new(config: &Qwen25Config, layer_idx: usize, device: Device) -> Result<Self> {
        let hs = config.hidden_size;
        let nh = config.num_attention_heads;
        let nkv = config.num_key_value_heads;
        let hd = config.head_dim;

        let q_proj = Linear::new_with_device(hs, nh * hd, false, device);
        let k_proj = Linear::new_with_device(hs, nkv * hd, false, device);
        let v_proj = Linear::new_with_device(hs, nkv * hd, false, device);
        let o_proj = Linear::new_with_device(nh * hd, hs, false, device);
        let rotary_emb = Qwen25RotaryEmbedding::new(config, device);

        let sliding_window = if config.layer_uses_sliding_window(layer_idx) {
            config.sliding_window
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: nh,
            num_kv_heads: nkv,
            head_dim: hd,
            sliding_window,
            device,
        })
    }

    /// Returns `true` when this layer uses sliding window attention.
    pub fn uses_sliding_window(&self) -> bool {
        self.sliding_window.is_some()
    }

    /// Returns the effective sliding window size, if configured.
    pub fn sliding_window(&self) -> Option<usize> {
        self.sliding_window
    }

    /// Number of query attention heads.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of key/value heads.
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Qwen25Attention {
    type Input = Tensor;
    type Output = Tensor;

    /// Real grouped-query scaled dot-product attention: RoPE, `Q @ K^T`
    /// (repeating each KV head across its `num_heads / num_kv_heads` query
    /// heads), causal masking combined with a sliding window when this
    /// layer is configured for one, softmax, and `@ V`.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let (mut q_data, mut k_data, v_data) = match (&q, &k, &v) {
            (Tensor::F32(qd), Tensor::F32(kd), Tensor::F32(vd)) => (
                qd.as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_attn", "q not contiguous"))?
                    .to_vec(),
                kd.as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_attn", "k not contiguous"))?
                    .to_vec(),
                vd.as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_attn", "v not contiguous"))?
                    .to_vec(),
            ),
            _ => return Err(tensor_op_error("qwen25_attn", "q, k, v must be F32")),
        };

        if self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            return Err(tensor_op_error(
                "qwen25_attn",
                "num_heads must be a positive multiple of num_kv_heads",
            ));
        }
        let q_width = self.num_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        if q_width == 0 || !q_data.len().is_multiple_of(q_width) {
            return Err(tensor_op_error(
                "qwen25_attn",
                "q size inconsistent with num_heads * head_dim",
            ));
        }
        let seq_len = q_data.len() / q_width;
        if k_data.len() != seq_len * kv_width || v_data.len() != seq_len * kv_width {
            return Err(tensor_op_error(
                "qwen25_attn",
                "k/v size inconsistent with num_kv_heads * head_dim",
            ));
        }
        if let Some(w) = self.sliding_window {
            if w == 0 {
                return Err(tensor_op_error("qwen25_attn", "sliding_window must be > 0"));
            }
        }

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
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        let mut out = vec![0f32; seq_len * q_width];

        for h in 0..self.num_heads {
            let kv_h = h / group;
            for i in 0..seq_len {
                let q_off = i * q_width + h * self.head_dim;
                let mut scores = Vec::with_capacity(i + 1);
                let mut key_positions = Vec::with_capacity(i + 1);
                for j in 0..=i {
                    if let Some(w) = self.sliding_window {
                        if i - j >= w {
                            continue;
                        }
                    }
                    let k_off = j * kv_width + kv_h * self.head_dim;
                    let dot: f32 =
                        (0..self.head_dim).map(|d| q_data[q_off + d] * k_data[k_off + d]).sum();
                    scores.push(dot * scale);
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
// SwiGLU MLP
// ---------------------------------------------------------------------------

/// Qwen2.5 MLP with SwiGLU activation.
///
/// Architecture: `down_proj(silu(gate_proj(x)) * up_proj(x))`
pub struct Qwen25MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    device: Device,
}

impl Qwen25MLP {
    pub fn new(config: &Qwen25Config, device: Device) -> Self {
        let hs = config.hidden_size;
        let is = config.intermediate_size;
        let gate_proj = Linear::new_with_device(hs, is, false, device);
        let up_proj = Linear::new_with_device(hs, is, false, device);
        let down_proj = Linear::new_with_device(is, hs, false, device);
        Self {
            gate_proj,
            up_proj,
            down_proj,
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Qwen25MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;

        let activated = match (&gate_out, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => {
                let g_slice = g
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_mlp", "gate tensor not contiguous"))?;
                let u_slice = u
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_mlp", "up tensor not contiguous"))?;
                let result = swiglu(g_slice, u_slice);
                let shape = g.shape().to_vec();
                Tensor::from_vec(result, &shape)?
            },
            _ => {
                return Err(tensor_op_error(
                    "qwen25_mlp",
                    "gate and up tensors must be F32",
                ))
            },
        };
        self.down_proj.forward(activated)
    }
}

// ---------------------------------------------------------------------------
// Decoder Layer
// ---------------------------------------------------------------------------

/// Qwen2.5 transformer decoder layer.
///
/// Pre-RMSNorm architecture:
/// ```text
/// x → input_layernorm → attention → + x → post_attention_layernorm → mlp → + x
/// ```
pub struct Qwen25DecoderLayer {
    self_attn: Qwen25Attention,
    mlp: Qwen25MLP,
    input_layernorm: Qwen25RmsNorm,
    post_attention_layernorm: Qwen25RmsNorm,
    device: Device,
}

impl Qwen25DecoderLayer {
    pub fn new(config: &Qwen25Config, layer_idx: usize, device: Device) -> Result<Self> {
        let self_attn = Qwen25Attention::new(config, layer_idx, device)?;
        let mlp = Qwen25MLP::new(config, device);
        let input_layernorm = Qwen25RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        let post_attention_layernorm =
            Qwen25RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            device,
        })
    }

    /// Returns `true` when this layer uses sliding window attention.
    pub fn uses_sliding_window(&self) -> bool {
        self.self_attn.uses_sliding_window()
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for Qwen25DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Attention sublayer
        let normed = self.input_layernorm.forward(input.clone())?;
        let attn_out = self.self_attn.forward(normed)?;
        let hidden = input.add(&attn_out)?;

        // MLP sublayer
        let normed_ff = self.post_attention_layernorm.forward(hidden.clone())?;
        let mlp_out = self.mlp.forward(normed_ff)?;
        hidden.add(&mlp_out)
    }
}

// ---------------------------------------------------------------------------
// Qwen25Model
// ---------------------------------------------------------------------------

/// Qwen2.5 base model: token embedding + decoder layers + final RMSNorm.
pub struct Qwen25Model {
    config: Qwen25Config,
    embed_tokens: Embedding,
    layers: Vec<Qwen25DecoderLayer>,
    norm: Qwen25RmsNorm,
    device: Device,
}

impl Qwen25Model {
    pub fn new(config: Qwen25Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Qwen25Config, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for layer_idx in 0..config.num_hidden_layers {
            layers.push(Qwen25DecoderLayer::new(&config, layer_idx, device)?);
        }

        let norm = Qwen25RmsNorm::new(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            device,
        })
    }

    pub fn config(&self) -> &Qwen25Config {
        &self.config
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for Qwen25Model {
    type Config = Qwen25Config;
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
                    "qwen25_forward",
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
        // the model). `Qwen25Model` has no weight-loading path implemented
        // at all (unlike e.g. `GemmaModel`/`MistralModel`/`QwenModel`, which
        // provide a real `load_from_path`/`load_from_huggingface` on their
        // `*ForCausalLM` wrapper). Silently returning `Ok(())` here would
        // leave the model's freshly-initialised (effectively random)
        // weights in place while claiming the load succeeded, so report
        // this honestly as unsupported instead.
        Err(TrustformersError::not_implemented(
            "Qwen25Model::load_pretrained: no weight-loading implementation exists for Qwen2.5 \
             yet; there is no `load_from_path`/`load_from_huggingface` to delegate to"
                .to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let hs = self.config.hidden_size;
        let vs = self.config.vocab_size;
        let nl = self.config.num_hidden_layers;
        let nh = self.config.num_attention_heads;
        let nkv = self.config.num_key_value_heads;
        let hd = self.config.head_dim;
        let is = self.config.intermediate_size;

        let embed = vs * hs;
        let attn = hs * nh * hd + hs * nkv * hd + hs * nkv * hd + nh * hd * hs;
        let mlp = 3 * hs * is;
        let norms = 2 * hs;
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

    fn lcg_next(state: &mut u64) -> f32 {
        *state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((*state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }

    fn lcg_vec(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n).map(|_| lcg_next(&mut state)).collect()
    }

    fn tiny_qwen25_config() -> Qwen25Config {
        Qwen25Config {
            vocab_size: 64,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 2,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            sliding_window: None,
            max_window_layers: 2,
            use_sliding_window: false,
            rms_norm_eps: 1e-6,
            hidden_act: "silu".to_string(),
            initializer_range: 0.02,
            tie_word_embeddings: false,
            use_mrope: false,
        }
    }

    // -- silu --

    #[test]
    fn test_silu_at_zero_is_zero() {
        assert!((silu(0.0)).abs() < 1e-6, "silu(0) must be 0");
    }

    #[test]
    fn test_silu_positive_input_positive_output() {
        assert!(silu(1.0) > 0.0, "silu(positive) must be positive");
    }

    #[test]
    fn test_silu_large_approches_identity() {
        // For large positive x, sigmoid(x) -> 1 so silu(x) -> x
        let x = 20.0_f32;
        assert!((silu(x) - x).abs() < 0.01, "silu(20) must be close to 20");
    }

    // -- swiglu --

    #[test]
    fn test_swiglu_length_matches_input() {
        let gate = lcg_vec(8, 30);
        let up = lcg_vec(8, 31);
        let result = swiglu(&gate, &up);
        assert_eq!(result.len(), 8, "swiglu output length must match input");
    }

    #[test]
    fn test_swiglu_zero_gate_gives_zero() {
        let gate = vec![0.0_f32; 8];
        let up = lcg_vec(8, 32);
        let result = swiglu(&gate, &up);
        for &v in &result {
            assert!(
                v.abs() < 1e-6,
                "swiglu with zero gate must give zero output"
            );
        }
    }

    // -- Qwen25RmsNorm --

    #[test]
    fn test_qwen25_rmsnorm_unit_rms() {
        let norm = Qwen25RmsNorm::new(4, 1e-6, Device::CPU).expect("Qwen25RmsNorm must build");
        let input =
            Tensor::from_vec(vec![3.0_f32, 4.0, 0.0, 0.0], &[4]).expect("tensor must build");
        let output = norm.forward(input).expect("forward must succeed");
        let vals: Vec<f32> = match &output {
            Tensor::F32(arr) => arr.as_slice().expect("must be contiguous").to_vec(),
            _ => panic!("expected F32"),
        };
        let rms = (vals.iter().map(|x| x * x).sum::<f32>() / 4.0).sqrt();
        assert!(
            (rms - 1.0).abs() < 1e-4,
            "RMSNorm output rms must be ~1.0, got {rms}"
        );
    }

    #[test]
    fn test_qwen25_rmsnorm_size_accessor() {
        let norm = Qwen25RmsNorm::new(16, 1e-6, Device::CPU).expect("must build");
        assert_eq!(norm.size(), 16, "size() must return the norm dimension");
    }

    #[test]
    fn test_qwen25_rmsnorm_device_accessor() {
        let norm = Qwen25RmsNorm::new(8, 1e-6, Device::CPU).expect("must build");
        assert_eq!(norm.device(), Device::CPU, "device() must return CPU");
    }

    // -- Qwen25Config --

    #[test]
    fn test_qwen25_config_validate_ok() {
        let cfg = tiny_qwen25_config();
        assert!(cfg.validate().is_ok(), "valid config must pass validation");
    }

    #[test]
    fn test_qwen25_config_validate_sliding_window_missing_fails() {
        let mut cfg = tiny_qwen25_config();
        cfg.use_sliding_window = true;
        cfg.sliding_window = None;
        assert!(
            cfg.validate().is_err(),
            "use_sliding_window=true with no window must fail"
        );
    }

    #[test]
    fn test_qwen25_kv_group_size() {
        let cfg = tiny_qwen25_config();
        let expected = cfg.num_attention_heads / cfg.num_key_value_heads;
        assert_eq!(
            cfg.kv_group_size(),
            expected,
            "kv_group_size must be nh/nkv"
        );
    }

    #[test]
    fn test_qwen25_0_5b_config_tied_embeddings() {
        let cfg = Qwen25Config::qwen25_0_5b();
        assert!(
            cfg.tie_word_embeddings,
            "0.5B model must have tied word embeddings"
        );
    }

    #[test]
    fn test_qwen25_7b_config_untied_embeddings() {
        let cfg = Qwen25Config::qwen25_7b();
        assert!(
            !cfg.tie_word_embeddings,
            "7B model must NOT have tied word embeddings"
        );
    }

    #[test]
    fn test_qwen25_rope_theta_large() {
        let cfg = Qwen25Config::qwen25_7b();
        // Qwen2.5 uses rope_theta=1_000_000 for extended context
        assert_eq!(
            cfg.rope_theta, 1_000_000.0,
            "Qwen2.5 must use rope_theta=1_000_000"
        );
    }

    #[test]
    fn test_qwen25_layer_uses_sliding_window_false_by_default() {
        let cfg = tiny_qwen25_config();
        assert!(
            !cfg.layer_uses_sliding_window(0),
            "no layer should use sliding window when use_sliding_window=false",
        );
    }

    #[test]
    fn test_qwen25_layer_uses_sliding_window_activates_beyond_max_window_layers() {
        let mut cfg = tiny_qwen25_config();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(32);
        cfg.max_window_layers = 1; // layers >= 1 get sliding window
        assert!(
            cfg.layer_uses_sliding_window(1),
            "layer 1 >= max_window_layers=1 must use sliding window",
        );
        assert!(
            !cfg.layer_uses_sliding_window(0),
            "layer 0 < max_window_layers=1 must not use sliding window",
        );
    }

    // -- Qwen25RotaryEmbedding --

    #[test]
    fn test_qwen25_rope_preserves_norm() {
        let cfg = tiny_qwen25_config();
        let rope = Qwen25RotaryEmbedding::new(&cfg, Device::CPU);
        let n = cfg.head_dim;
        let mut q = lcg_vec(n, 40);
        let mut k = lcg_vec(n, 41);
        let q_norm_before: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        rope.apply(&mut q, &mut k, 1);
        let q_norm_after: f32 = q.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (q_norm_before - q_norm_after).abs() < 1e-4,
            "RoPE must preserve Q norm, before={q_norm_before} after={q_norm_after}",
        );
    }

    // `rotate_heads_rope` is the function `Qwen25Attention::forward` actually
    // calls for both Q and K (see `forward` above); `Qwen25RotaryEmbedding::apply`
    // exercised above is not on that path. `crate::qwen2_5::tests` (in
    // `qwen2_5/mod.rs`) already covers "every head rotates"
    // (`test_qwen25_rotate_heads_rope_rotates_every_head`); the two tests
    // below add the position-dependence coverage that suite does not have —
    // identity at position 0, and two non-zero positions actually differing
    // — using a non-degenerate `theta`/`head_dim` (the attention-level
    // config's `head_dim=2` collapses to a single frequency band of exactly
    // 1.0 regardless of `theta`, which would mask a broken frequency
    // computation). Both would have FAILED against a no-op RoPE that left
    // `data` unchanged, matching the fake implementation the original audit
    // found.

    /// RoPE at position 0 must be the identity rotation (angle = 0).
    #[test]
    fn test_rotate_heads_rope_position_zero_is_identity() {
        let original = vec![1.0f32, 2.0, 3.0, 4.0];
        let mut q = original.clone();
        rotate_heads_rope(&mut q, 1, 4, 10000.0, &[0]);
        for (a, b) in original.iter().zip(q.iter()) {
            assert!(
                (a - b).abs() < 1e-5,
                "position 0 must be identity: {a} vs {b}"
            );
        }
    }

    /// Two different non-zero positions on the same input vector must rotate
    /// to different outputs. This fails against a no-op RoPE that just
    /// leaves `data` unchanged regardless of `position_ids`.
    #[test]
    fn test_rotate_heads_rope_differs_by_position() {
        let base = vec![1.0f32; 4];
        let mut at_pos0 = base.clone();
        let mut at_pos5 = base.clone();
        rotate_heads_rope(&mut at_pos0, 1, 4, 10000.0, &[0]);
        rotate_heads_rope(&mut at_pos5, 1, 4, 10000.0, &[5]);
        let differs = at_pos0.iter().zip(at_pos5.iter()).any(|(a, b)| (a - b).abs() > 1e-4);
        assert!(
            differs,
            "RoPE must rotate differently at different positions"
        );
    }

    // -- Qwen25Attention --

    #[test]
    fn test_qwen25_attention_no_sliding_window_when_full_attn() {
        let cfg = tiny_qwen25_config();
        let attn = Qwen25Attention::new(&cfg, 0, Device::CPU).expect("attention must build");
        assert!(
            !attn.uses_sliding_window(),
            "full-attn layer must not use sliding window"
        );
        assert_eq!(
            attn.sliding_window(),
            None,
            "sliding_window must be None for full-attn layer"
        );
    }

    #[test]
    fn test_qwen25_attention_gqa_heads() {
        let cfg = tiny_qwen25_config();
        let attn = Qwen25Attention::new(&cfg, 0, Device::CPU).expect("attention must build");
        assert_eq!(
            attn.num_heads(),
            cfg.num_attention_heads,
            "Q head count must match config"
        );
        assert_eq!(
            attn.num_kv_heads(),
            cfg.num_key_value_heads,
            "KV head count must match config"
        );
    }

    // Real grouped-query scaled dot-product attention regression tests.
    // `crate::qwen2_5::tests` (in `qwen2_5/mod.rs`) already covers output
    // shape, early-token-change propagation, causal masking, prefix
    // extension, and a `window=1` sliding-window exclusion test, all against
    // `Qwen25Attention::forward` end to end (RoPE, repeated-KV-head Q@K^T,
    // causal/sliding-window masking, softmax, @V) — every one of those would
    // have FAILED against the old fake path, which discarded V and fed a
    // zero-padded, resized RoPE'd query straight into `o_proj`. The test
    // below adds a `window=2` variant that is not implied by that suite:

    /// Sliding-window attention must actually restrict the attention span,
    /// not just be wired through config accessors (`uses_sliding_window`/
    /// `sliding_window`, already covered above). With `window=2` and
    /// `seq_len=4`: token 1 (sees positions `{0,1}`, since `1-0=1 < 2`) must
    /// still be affected by a change to token 0, but token 3 (sees positions
    /// `{2,3}` only, since `3-0=3 >= 2` excludes position 0) must NOT be —
    /// the discriminating half that a window which merely gates a flag
    /// (without actually excluding out-of-window keys) would fail.
    #[test]
    fn test_qwen25_attention_sliding_window_restricts_span() {
        let mut cfg = tiny_qwen25_config();
        cfg.use_sliding_window = true;
        cfg.sliding_window = Some(2);
        cfg.max_window_layers = 0; // layer_idx(0) >= max_window_layers(0) => uses window
        let attn = Qwen25Attention::new(&cfg, 0, Device::CPU).expect("attention must build");
        assert!(
            attn.uses_sliding_window(),
            "layer 0 must use the sliding window with this config"
        );
        assert_eq!(attn.sliding_window(), Some(2));

        let seq_len = 4;
        let hidden = cfg.hidden_size;
        let base = lcg_vec(seq_len * hidden, 55);
        let mut modified = base.clone();
        for x in modified[0..hidden].iter_mut() {
            *x += 5.0; // perturb only token 0
        }

        let out_base = attn
            .forward(Tensor::from_vec(base, &[seq_len, hidden]).expect("tensor"))
            .expect("forward base");
        let out_mod = attn
            .forward(Tensor::from_vec(modified, &[seq_len, hidden]).expect("tensor"))
            .expect("forward modified");

        let (a, b) = match (&out_base, &out_mod) {
            (Tensor::F32(x), Tensor::F32(y)) => (
                x.as_slice().expect("contiguous").to_vec(),
                y.as_slice().expect("contiguous").to_vec(),
            ),
            _ => panic!("expected F32 outputs"),
        };

        let row1_differs = a[hidden..2 * hidden]
            .iter()
            .zip(&b[hidden..2 * hidden])
            .any(|(x, y)| (x - y).abs() > 1e-5);
        assert!(
            row1_differs,
            "token 1 is within the sliding window of token 0 and must be affected by its change"
        );

        let row3_a = &a[3 * hidden..4 * hidden];
        let row3_b = &b[3 * hidden..4 * hidden];
        for (x, y) in row3_a.iter().zip(row3_b.iter()) {
            assert!(
                (x - y).abs() < 1e-6,
                "token 3 is outside the sliding window of token 0 and must be unaffected"
            );
        }
    }

    // -- Qwen25Model --

    #[test]
    fn test_qwen25_model_construction() {
        let cfg = tiny_qwen25_config();
        let model = Qwen25Model::new(cfg).expect("Qwen25Model must build");
        assert_eq!(
            model.config().num_hidden_layers,
            2,
            "model must have 2 layers"
        );
    }

    #[test]
    fn test_qwen25_model_forward_single_token() {
        let cfg = tiny_qwen25_config();
        let model = Qwen25Model::new(cfg.clone()).expect("model must build");
        let input = Tensor::from_vec(vec![0_f32], &[1]).expect("i64 token must build");
        let output = model.forward(input).expect("forward must succeed");
        let out_len = output.shape().iter().product::<usize>();
        assert!(out_len > 0, "output must be non-empty");
    }

    #[test]
    fn test_qwen25_model_num_parameters_positive() {
        let cfg = tiny_qwen25_config();
        let model = Qwen25Model::new(cfg).expect("model must build");
        assert!(
            model.num_parameters() > 0,
            "num_parameters must be positive"
        );
    }

    /// Regression: `load_pretrained` must NOT silently report success while
    /// leaving the model's random initial weights untouched. It previously
    /// read the buffer, checked it was non-empty, and returned `Ok(())`
    /// without parsing anything (a fabricated success). It must now report
    /// a structured "not implemented" error instead.
    #[test]
    fn test_qwen25_model_load_pretrained_reports_not_implemented_instead_of_fake_success() {
        let cfg = tiny_qwen25_config();
        let mut model = Qwen25Model::new(cfg).expect("model must build");
        let mut data: &[u8] = b"not a real checkpoint, but not empty either";
        let result = model.load_pretrained(&mut data);
        assert!(
            result.is_err(),
            "load_pretrained must fail rather than silently succeed with no weights loaded"
        );
    }

    #[test]
    fn test_qwen25_decoder_layer_no_sliding_window() {
        let cfg = tiny_qwen25_config();
        let layer =
            Qwen25DecoderLayer::new(&cfg, 0, Device::CPU).expect("decoder layer must build");
        assert!(
            !layer.uses_sliding_window(),
            "layer must not use sliding window"
        );
    }
}
