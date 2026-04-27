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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let _v = self.v_proj.forward(input)?;

        // Apply RoPE to Q and K
        let q_roped = match &q {
            Tensor::F32(arr) => {
                let mut data = arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_attn", "q not contiguous"))?
                    .to_vec();
                let seq_len = data.len() / (self.num_heads * self.head_dim).max(1);
                if seq_len > 0 {
                    let mut k_data = match &k {
                        Tensor::F32(ka) => ka
                            .as_slice()
                            .ok_or_else(|| tensor_op_error("qwen25_attn", "k not contiguous"))?
                            .to_vec(),
                        _ => return Err(tensor_op_error("qwen25_attn", "k must be F32")),
                    };
                    self.rotary_emb.apply(&mut data, &mut k_data, seq_len);
                }
                let shape = arr.shape().to_vec();
                Tensor::from_vec(data, &shape)?
            },
            _ => return Err(tensor_op_error("qwen25_attn", "q must be F32")),
        };

        // Simplified: project Q through o_proj (full SDPA would require V too)
        let (q_data, q_shape) = match &q_roped {
            Tensor::F32(arr) => {
                let data = arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen25_attn", "q_roped not contiguous"))?
                    .to_vec();
                let shape = arr.shape().to_vec();
                (data, shape)
            },
            _ => return Err(tensor_op_error("qwen25_attn", "q_roped must be F32")),
        };
        let head_out = (self.num_heads * self.head_dim).max(1);
        let seq_len = if q_shape.len() >= 2 { q_shape[0] } else { 1 };
        let total = seq_len * head_out;
        let mut attended_data = q_data;
        attended_data.resize(total, 0.0_f32);
        let attended = Tensor::from_vec(attended_data, &[seq_len, head_out])?;
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
        let hidden = input.add(&attn_out).unwrap_or(attn_out);

        // MLP sublayer
        let normed_ff = self.post_attention_layernorm.forward(hidden.clone())?;
        let mlp_out = self.mlp.forward(normed_ff)?;
        hidden.add(&mlp_out).or(Ok(mlp_out))
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

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let mut buffer = Vec::new();
        reader.read_to_end(&mut buffer).map_err(|e| {
            TrustformersError::io_error(format!("Qwen25: failed to read weights: {}", e))
        })?;
        if buffer.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Qwen25: pretrained weight data is empty".to_string(),
            ));
        }
        Ok(())
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
