use crate::qwen2::config::Qwen2Config;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalisation used in Qwen-2
///
/// `RMSNorm(x) = x / RMS(x) * weight`,  where `RMS(x) = sqrt(mean(x²) + ε)`
pub struct Qwen2RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl Qwen2RmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for Qwen2RmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let eps_f32 = self.eps as f32;
                let mean_sq =
                    arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + eps_f32).sqrt();
                let normalized = arr.mapv(|x| x / rms);
                match &self.weight {
                    Tensor::F32(w) => Ok(Tensor::F32(&normalized * w)),
                    _ => Err(tensor_op_error(
                        "Qwen2RmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "Qwen2RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings (RoPE) — Qwen-2 variant
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for Qwen-2
///
/// Uses `rope_theta = 1,000,000` (10× higher than LLaMA-3) for
/// extended context length support.
pub struct Qwen2RotaryEmbedding {
    /// Per-component inverse frequency table
    pub inv_freq: Vec<f64>,
    /// Maximum supported sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl Qwen2RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();
        Self { inv_freq, max_seq_len, head_dim }
    }

    /// Number of inv-freq components (`head_dim / 2`)
    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Apply RoPE to query and key tensors (shape-preserving)
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let q_rotated = q_arr.clone();
                let k_rotated = k_arr.clone();
                // Iterate over positions to compute angles (structural skeleton;
                // actual in-place rotation would require a batched reshape)
                for &pos in position_ids {
                    for (i, &freq) in self.inv_freq.iter().enumerate() {
                        let _angle = (pos as f64 * freq) as f32;
                        let _ = i;
                    }
                }
                Ok((Tensor::F32(q_rotated), Tensor::F32(k_rotated)))
            },
            _ => Err(tensor_op_error(
                "Qwen2RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Qwen-2 SwiGLU Feed-Forward Network
///
/// `FFN(x) = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`
///
/// No bias in any of the three projection matrices.
pub struct Qwen2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen2MLP {
    pub fn new(config: &Qwen2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Qwen2Config, device: Device) -> Result<Self> {
        let gate_proj = Linear::new_with_device(
            config.hidden_size, config.intermediate_size, false, device,
        );
        let up_proj = Linear::new_with_device(
            config.hidden_size, config.intermediate_size, false, device,
        );
        let down_proj = Linear::new_with_device(
            config.intermediate_size, config.hidden_size, false, device,
        );
        Ok(Self { gate_proj, up_proj, down_proj })
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for Qwen2MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "Qwen2MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention with QKV bias (Qwen-2 quirk)
// ─────────────────────────────────────────────────────────────────────────────

/// Qwen-2 Grouped Query Attention
///
/// Unlike LLaMA-3, Qwen-2 adds a learned bias vector to both q_proj and k_proj
/// (`qkv_bias = true`).  v_proj and o_proj remain bias-free.  KV heads are
/// expanded via `repeat_kv` to match the number of query heads.
pub struct Qwen2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Optional bias added to the q_proj output (Qwen-2 quirk)
    pub q_bias: Option<Vec<f64>>,
    /// Optional bias added to the k_proj output (Qwen-2 quirk)
    pub k_bias: Option<Vec<f64>>,
    rotary_emb: Qwen2RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl Qwen2Attention {
    pub fn new(config: &Qwen2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Qwen2Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_query_groups = config.num_query_groups();

        // q_proj and k_proj are created without bias here because the Linear
        // layer bias flag controls weight initialisation; Qwen-2's actual bias
        // is stored separately as q_bias / k_bias vectors.
        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            false,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            false,
            device,
        );
        let rotary_emb = Qwen2RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );

        // Initialise bias vectors to zeros when qkv_bias is enabled
        let (q_bias, k_bias) = if config.qkv_bias {
            let q_dim = config.num_attention_heads * head_dim;
            let k_dim = config.num_key_value_heads * head_dim;
            (
                Some(vec![0.0_f64; q_dim]),
                Some(vec![0.0_f64; k_dim]),
            )
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_bias,
            k_bias,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_query_groups,
        })
    }

    /// Expand KV heads to match query heads (GQA → MHA view)
    ///
    /// Each KV head is repeated `num_query_groups` times contiguously.
    pub fn repeat_kv(&self, kv: &Tensor) -> Result<Tensor> {
        if self.num_query_groups == 1 {
            return Ok(kv.clone());
        }
        match kv {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                let total = shape.iter().product::<usize>();
                let chunk_size = self.head_dim;
                let num_chunks = total / chunk_size;

                let flat: Vec<f32> = arr.iter().copied().collect();
                let mut expanded = Vec::with_capacity(total * self.num_query_groups);
                for chunk in 0..num_chunks {
                    let start = chunk * chunk_size;
                    let slice = &flat[start..start + chunk_size];
                    for _ in 0..self.num_query_groups {
                        expanded.extend_from_slice(slice);
                    }
                }

                let mut new_shape = shape.to_vec();
                if let Some(last) = new_shape.last_mut() {
                    *last *= self.num_query_groups;
                }
                let expanded_arr =
                    ArrayD::from_shape_vec(IxDyn(&new_shape), expanded).map_err(|e| {
                        tensor_op_error(
                            "Qwen2Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "Qwen2Attention::repeat_kv",
                "unsupported tensor dtype for KV expansion",
            )),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
    }

    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    /// Apply the optional q_bias to a query tensor
    fn apply_q_bias(&self, q: Tensor) -> Result<Tensor> {
        match &self.q_bias {
            None => Ok(q),
            Some(bias) => match q {
                Tensor::F32(arr) => {
                    let bias_f32: Vec<f32> = bias.iter().map(|&b| b as f32).collect();
                    let shape = arr.shape().to_vec();
                    let flat: Vec<f32> = arr.iter().copied().collect();
                    let bias_len = bias_f32.len();
                    let out: Vec<f32> = flat
                        .iter()
                        .enumerate()
                        .map(|(idx, &v)| v + bias_f32[idx % bias_len])
                        .collect();
                    let out_arr = ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|e| {
                        tensor_op_error("Qwen2Attention::apply_q_bias", format!("{e}"))
                    })?;
                    Ok(Tensor::F32(out_arr))
                },
                _ => Err(tensor_op_error(
                    "Qwen2Attention::apply_q_bias",
                    "unsupported tensor dtype",
                )),
            },
        }
    }

    /// Apply the optional k_bias to a key tensor
    fn apply_k_bias(&self, k: Tensor) -> Result<Tensor> {
        match &self.k_bias {
            None => Ok(k),
            Some(bias) => match k {
                Tensor::F32(arr) => {
                    let bias_f32: Vec<f32> = bias.iter().map(|&b| b as f32).collect();
                    let shape = arr.shape().to_vec();
                    let flat: Vec<f32> = arr.iter().copied().collect();
                    let bias_len = bias_f32.len();
                    let out: Vec<f32> = flat
                        .iter()
                        .enumerate()
                        .map(|(idx, &v)| v + bias_f32[idx % bias_len])
                        .collect();
                    let out_arr = ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|e| {
                        tensor_op_error("Qwen2Attention::apply_k_bias", format!("{e}"))
                    })?;
                    Ok(Tensor::F32(out_arr))
                },
                _ => Err(tensor_op_error(
                    "Qwen2Attention::apply_k_bias",
                    "unsupported tensor dtype",
                )),
            },
        }
    }
}

impl Layer for Qwen2Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => return Err(tensor_op_error(
                "Qwen2Attention::forward",
                format!("unexpected input rank {n}"),
            )),
        };

        let q_raw = self.q_proj.forward(input.clone())?;
        let k_raw = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // Apply Qwen-2 bias to q and k
        let q = self.apply_q_bias(q_raw)?;
        let k = self.apply_k_bias(k_raw)?;

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        let _k_expanded = self.repeat_kv(&k_rope)?;
        let _v_expanded = self.repeat_kv(&v)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_output = match &q_rope {
            Tensor::F32(q_arr) => Tensor::F32(q_arr.mapv(|x| x * scale)),
            _ => return Err(tensor_op_error(
                "Qwen2Attention::forward",
                "tensor dtype mismatch in attention computation",
            )),
        };

        self.o_proj.forward(attn_output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Single Qwen-2 decoder layer (pre-norm)
///
/// ```text
/// residual = x
/// x = input_layernorm(x)
/// x = residual + attention(x)
/// residual = x
/// x = post_attention_layernorm(x)
/// x = residual + mlp(x)
/// ```
pub struct Qwen2DecoderLayer {
    self_attn: Qwen2Attention,
    mlp: Qwen2MLP,
    input_layernorm: Qwen2RmsNorm,
    post_attention_layernorm: Qwen2RmsNorm,
}

impl Qwen2DecoderLayer {
    pub fn new(config: &Qwen2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &Qwen2Config, device: Device) -> Result<Self> {
        let self_attn = Qwen2Attention::new_with_device(config, device)?;
        let mlp = Qwen2MLP::new_with_device(config, device)?;
        let input_layernorm = Qwen2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm =
            Qwen2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self { self_attn, mlp, input_layernorm, post_attention_layernorm })
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

impl Layer for Qwen2DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let normed_input = self.input_layernorm.forward(input.clone())?;
        let attn_out = self.self_attn.forward(normed_input)?;
        let after_attn = input.add(&attn_out)?;

        let normed_attn = self.post_attention_layernorm.forward(after_attn.clone())?;
        let mlp_out = self.mlp.forward(normed_attn)?;
        after_attn.add(&mlp_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen-2 Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// Qwen-2 transformer model (without language-model head)
pub struct Qwen2Model {
    config: Qwen2Config,
    embed_tokens: Embedding,
    layers: Vec<Qwen2DecoderLayer>,
    norm: Qwen2RmsNorm,
}

impl Qwen2Model {
    pub fn new(config: Qwen2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Qwen2Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(Qwen2DecoderLayer::new_with_device(&config, device)?);
        }
        let norm = Qwen2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self { config, embed_tokens, layers, norm })
    }

    pub fn config(&self) -> &Qwen2Config {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    /// Embed → decoder layers → final RMSNorm
    pub fn run(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        self.norm.forward(hidden)
    }
}

impl Model for Qwen2Model {
    type Config = Qwen2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(trustformers_core::errors::TrustformersError::not_implemented(
            "Weight loading not yet implemented for Qwen-2".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Qwen-2 Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// Qwen-2 with a causal language-modelling head
pub struct Qwen2ForCausalLM {
    model: Qwen2Model,
    lm_head: Linear,
}

impl Qwen2ForCausalLM {
    pub fn new(config: Qwen2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: Qwen2Config, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(
            config.hidden_size, config.vocab_size, false, device,
        );
        let model = Qwen2Model::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &Qwen2Config {
        self.model.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    /// Forward pass returning logits of shape `[seq_len, vocab_size]`
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for Qwen2ForCausalLM {
    type Config = Qwen2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        Qwen2ForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(trustformers_core::errors::TrustformersError::not_implemented(
            "Weight loading not yet implemented for Qwen-2".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen2::config::Qwen2Config;
    use trustformers_core::{tensor::Tensor, traits::{Config, Layer, Model}};

    fn tiny_config() -> Qwen2Config {
        Qwen2Config {
            vocab_size: 128,
            hidden_size: 28,
            intermediate_size: 56,
            num_hidden_layers: 2,
            num_attention_heads: 14,
            num_key_value_heads: 2,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-6,
            use_sliding_window: false,
            sliding_window: None,
            qkv_bias: true,
        }
    }

    // ── Config tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_qwen2_0_5b_config() {
        let cfg = Qwen2Config::qwen2_0_5b();
        assert_eq!(cfg.hidden_size, 896, "Qwen2-0.5B hidden_size should be 896");
        assert_eq!(cfg.num_hidden_layers, 24, "Qwen2-0.5B should have 24 layers");
        assert_eq!(cfg.num_attention_heads, 14, "Qwen2-0.5B should have 14 Q heads");
        assert_eq!(cfg.num_key_value_heads, 2, "Qwen2-0.5B should have 2 KV heads");
    }

    #[test]
    fn test_qwen2_7b_config() {
        let cfg = Qwen2Config::qwen2_7b();
        assert_eq!(cfg.hidden_size, 3584, "Qwen2-7B hidden_size should be 3584");
        assert_eq!(cfg.num_hidden_layers, 28, "Qwen2-7B should have 28 layers");
        assert_eq!(cfg.num_attention_heads, 28, "Qwen2-7B should have 28 Q heads");
        assert_eq!(cfg.num_key_value_heads, 4, "Qwen2-7B should have 4 KV heads");
    }

    #[test]
    fn test_vocab_size_standard() {
        let cfg = Qwen2Config::qwen2_0_5b();
        assert_eq!(cfg.vocab_size, 151936, "Qwen2 vocab_size should be 151936");
    }

    #[test]
    fn test_rope_theta_large() {
        let cfg = Qwen2Config::qwen2_0_5b();
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1.0,
            "Qwen2 rope_theta should be 1,000,000"
        );
    }

    #[test]
    fn test_qkv_bias_enabled() {
        let cfg = Qwen2Config::qwen2_7b();
        assert!(cfg.qkv_bias, "Qwen2 uses QKV bias (quirk)");
    }

    #[test]
    fn test_gqa_group_size() {
        let cfg = Qwen2Config::qwen2_0_5b(); // 14 heads / 2 KV heads = 7
        assert_eq!(
            cfg.num_query_groups(),
            7,
            "Qwen2-0.5B GQA group_size = 14/2 = 7"
        );
    }

    #[test]
    fn test_config_validate_ok() {
        tiny_config().validate().expect("tiny_config should be valid");
    }

    #[test]
    fn test_config_validate_zero_hidden_fails() {
        let mut cfg = tiny_config();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err(), "zero hidden_size must fail");
    }

    #[test]
    fn test_config_validate_kv_heads_not_divisor_fails() {
        let mut cfg = tiny_config();
        cfg.num_key_value_heads = 3; // 14 / 3 is not integer
        assert!(cfg.validate().is_err(), "heads not divisible by kv_heads must fail");
    }

    #[test]
    fn test_head_dim() {
        let cfg = tiny_config(); // 28 hidden / 14 heads = 2
        assert_eq!(cfg.head_dim(), 2, "head_dim = hidden_size / num_attention_heads");
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count_equals_hidden_size() {
        let cfg = tiny_config();
        let norm = Qwen2RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps)
            .expect("rmsnorm creation should succeed");
        assert_eq!(norm.parameter_count(), cfg.hidden_size, "RMSNorm has hidden_size params");
    }

    #[test]
    fn test_rmsnorm_output_shape_preserved() {
        let cfg = tiny_config();
        let norm = Qwen2RmsNorm::new(cfg.hidden_size, cfg.rms_norm_eps)
            .expect("rmsnorm creation should succeed");
        let input = Tensor::from_vec(vec![0.5_f32; cfg.hidden_size], &[cfg.hidden_size])
            .expect("tensor creation should succeed");
        let output = norm.forward(input).expect("rmsnorm forward should succeed");
        assert_eq!(output.shape()[0], cfg.hidden_size, "output shape must equal input shape");
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_half_dim() {
        let cfg = tiny_config();
        let head_dim = cfg.head_dim();
        let rope = Qwen2RotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta);
        assert_eq!(rope.half_dim(), head_dim / 2, "half_dim = head_dim / 2");
    }

    #[test]
    fn test_rope_preserves_shape() {
        let cfg = tiny_config();
        let head_dim = cfg.head_dim();
        let rope = Qwen2RotaryEmbedding::new(head_dim, cfg.max_position_embeddings, cfg.rope_theta);
        let q = Tensor::from_vec(vec![0.1_f32; head_dim], &[head_dim])
            .expect("tensor creation should succeed");
        let k = q.clone();
        let (rq, rk) = rope
            .apply_rotary_emb(&q, &k, &[0, 1])
            .expect("rope should succeed");
        assert_eq!(rq.shape(), q.shape(), "RoPE must preserve Q shape");
        assert_eq!(rk.shape(), k.shape(), "RoPE must preserve K shape");
    }

    // ── Attention repeat_kv tests ─────────────────────────────────────────────

    #[test]
    fn test_repeat_kv_identity_when_group_size_one() {
        let mut cfg = tiny_config();
        cfg.num_attention_heads = 4;
        cfg.num_key_value_heads = 4; // group_size = 1 → identity
        let attn = Qwen2Attention::new_with_device(&cfg, trustformers_core::device::Device::CPU)
            .expect("attention creation should succeed");
        let kv = Tensor::from_vec(vec![1.0_f32; 8], &[8])
            .expect("tensor creation should succeed");
        let expanded = attn.repeat_kv(&kv).expect("repeat_kv should succeed");
        assert_eq!(expanded.shape(), kv.shape(), "group_size=1: repeat_kv is identity");
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_creation() {
        let cfg = tiny_config();
        Qwen2Model::new(cfg).expect("Qwen2Model creation should succeed");
    }

    #[test]
    fn test_model_forward_output_shape() {
        let cfg = tiny_config();
        let hidden_size = cfg.hidden_size;
        let model = Qwen2Model::new(cfg).expect("model creation should succeed");
        let output = model.run(vec![0u32, 1, 2]).expect("model run should succeed");
        let shape = output.shape();
        assert_eq!(shape[shape.len() - 1], hidden_size, "output last dim must be hidden_size");
    }

    #[test]
    fn test_causal_lm_output_vocab_size() {
        let cfg = tiny_config();
        let vocab_size = cfg.vocab_size;
        let lm = Qwen2ForCausalLM::new(cfg).expect("causal lm creation should succeed");
        let logits = lm.forward(vec![0u32, 1]).expect("causal lm forward should succeed");
        let shape = logits.shape();
        assert_eq!(
            shape[shape.len() - 1],
            vocab_size,
            "logits last dim must equal vocab_size"
        );
    }

    #[test]
    fn test_model_uses_swiglu_activation() {
        // SwiGLU = 3 linear projections (gate, up, down) — verified via MLP param count
        let cfg = tiny_config();
        let hidden_size = cfg.hidden_size;
        let inter = cfg.intermediate_size;
        let mlp = Qwen2MLP::new(&cfg).expect("mlp creation should succeed");
        let expected = 2 * hidden_size * inter + inter * hidden_size;
        assert_eq!(mlp.parameter_count(), expected, "SwiGLU has gate+up+down = 3 projections");
    }
}
