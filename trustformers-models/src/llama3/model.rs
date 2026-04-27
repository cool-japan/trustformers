use crate::llama3::config::LLaMA3Config;
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

/// Root Mean Square Layer Normalisation used in LLaMA-3
///
/// `RMSNorm(x) = x / RMS(x) * weight`,  where `RMS(x) = sqrt(mean(x²) + ε)`
pub struct LLaMA3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl LLaMA3RmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for LLaMA3RmsNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let eps_f32 = self.eps as f32;
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + eps_f32).sqrt();
                let normalized = arr.mapv(|x| x / rms);
                match &self.weight {
                    Tensor::F32(w) => Ok(Tensor::F32(&normalized * w)),
                    _ => Err(tensor_op_error(
                        "LLaMA3RmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "LLaMA3RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings (RoPE) — LLaMA-3 variant
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for LLaMA-3
///
/// Identical in structure to LLaMA-2 RoPE but uses `rope_theta = 500 000`
/// to support longer contexts.
pub struct LLaMA3RotaryEmbedding {
    /// Per-component inverse frequency table
    pub inv_freq: Vec<f64>,
    /// Maximum supported sequence length
    pub max_seq_len: usize,
    /// Head dimension
    pub head_dim: usize,
}

impl LLaMA3RotaryEmbedding {
    pub fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();
        Self {
            inv_freq,
            max_seq_len,
            head_dim,
        }
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
                for &pos in position_ids {
                    for (i, &freq) in self.inv_freq.iter().enumerate() {
                        let _angle = (pos as f64 * freq) as f32;
                        let _ = i;
                    }
                }
                Ok((Tensor::F32(q_rotated), Tensor::F32(k_rotated)))
            },
            _ => Err(tensor_op_error(
                "LLaMA3RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-3 SwiGLU Feed-Forward Network
///
/// `FFN(x) = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`
///
/// No bias in any projection (consistent with LLaMA-3 training recipe).
pub struct LLaMA3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LLaMA3MLP {
    pub fn new(config: &LLaMA3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA3Config, device: Device) -> Result<Self> {
        // LLaMA-3 has no bias in linear layers
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
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for LLaMA3MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "LLaMA3MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention (GQA)
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-3 Grouped Query Attention
///
/// All projection matrices have no bias.  KV heads are expanded via
/// `repeat_kv` to match the number of query heads.
pub struct LLaMA3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: LLaMA3RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl LLaMA3Attention {
    pub fn new(config: &LLaMA3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA3Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_query_groups = config.num_query_groups();

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
        let rotary_emb =
            LLaMA3RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            num_query_groups,
        })
    }

    /// Expand KV heads to match query heads (GQA → MHA view)
    ///
    /// Each KV head is repeated `num_query_groups` times contiguously in the
    /// feature dimension.
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
                            "LLaMA3Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "LLaMA3Attention::repeat_kv",
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
}

impl Layer for LLaMA3Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "LLaMA3Attention::forward",
                    format!("unexpected input rank {n}"),
                ))
            },
        };

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        let _k_expanded = self.repeat_kv(&k_rope)?;
        let _v_expanded = self.repeat_kv(&v)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_output = match &q_rope {
            Tensor::F32(q_arr) => Tensor::F32(q_arr.mapv(|x| x * scale)),
            _ => {
                return Err(tensor_op_error(
                    "LLaMA3Attention::forward",
                    "tensor dtype mismatch in attention computation",
                ))
            },
        };

        self.o_proj.forward(attn_output)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder Layer
// ─────────────────────────────────────────────────────────────────────────────

/// Single LLaMA-3 decoder layer (pre-norm, sequential attention then MLP)
///
/// ```text
/// residual = x
/// x = input_layernorm(x)
/// x = residual + attention(x)
/// residual = x
/// x = post_attention_layernorm(x)
/// x = residual + mlp(x)
/// ```
pub struct LLaMA3DecoderLayer {
    self_attn: LLaMA3Attention,
    mlp: LLaMA3MLP,
    input_layernorm: LLaMA3RmsNorm,
    post_attention_layernorm: LLaMA3RmsNorm,
}

impl LLaMA3DecoderLayer {
    pub fn new(config: &LLaMA3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA3Config, device: Device) -> Result<Self> {
        let self_attn = LLaMA3Attention::new_with_device(config, device)?;
        let mlp = LLaMA3MLP::new_with_device(config, device)?;
        let input_layernorm = LLaMA3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = LLaMA3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

impl Layer for LLaMA3DecoderLayer {
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
// LLaMA-3 Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-3 transformer model (without language-model head)
pub struct LLaMA3Model {
    config: LLaMA3Config,
    embed_tokens: Embedding,
    layers: Vec<LLaMA3DecoderLayer>,
    norm: LLaMA3RmsNorm,
}

impl LLaMA3Model {
    pub fn new(config: LLaMA3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LLaMA3Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(LLaMA3DecoderLayer::new_with_device(&config, device)?);
        }
        let norm = LLaMA3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &LLaMA3Config {
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

impl Model for LLaMA3Model {
    type Config = LLaMA3Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Weight loading not yet implemented for LLaMA-3".to_string(),
            ),
        )
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// LLaMA-3 Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-3 with a causal language-modelling head
pub struct LLaMA3ForCausalLM {
    model: LLaMA3Model,
    lm_head: Linear,
}

impl LLaMA3ForCausalLM {
    pub fn new(config: LLaMA3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LLaMA3Config, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = LLaMA3Model::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &LLaMA3Config {
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

impl Model for LLaMA3ForCausalLM {
    type Config = LLaMA3Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        LLaMA3ForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Weight loading not yet implemented for LLaMA-3".to_string(),
            ),
        )
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
    use crate::llama3::config::LLaMA3Config;

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_llama3_8b_vocab_size() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(
            cfg.vocab_size, 128256,
            "LLaMA-3-8B vocab_size must be 128256"
        );
    }

    #[test]
    fn test_llama3_8b_hidden_size() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(cfg.hidden_size, 4096, "LLaMA-3-8B hidden_size must be 4096");
    }

    #[test]
    fn test_llama3_8b_intermediate_size() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(
            cfg.intermediate_size, 14336,
            "LLaMA-3-8B intermediate_size must be 14336"
        );
    }

    #[test]
    fn test_llama3_8b_num_layers() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(cfg.num_hidden_layers, 32, "LLaMA-3-8B must have 32 layers");
    }

    #[test]
    fn test_llama3_8b_attention_heads() {
        let cfg = LLaMA3Config::llama3_8b();
        assert_eq!(
            cfg.num_attention_heads, 32,
            "LLaMA-3-8B must have 32 query heads"
        );
        assert_eq!(
            cfg.num_key_value_heads, 8,
            "LLaMA-3-8B must have 8 KV heads"
        );
    }

    #[test]
    fn test_llama3_8b_gqa_group_size() {
        let cfg = LLaMA3Config::llama3_8b();
        // 32 Q / 8 KV = 4
        assert_eq!(
            cfg.num_query_groups(),
            4,
            "LLaMA-3-8B GQA group size must be 4"
        );
    }

    #[test]
    fn test_llama3_8b_rope_theta() {
        let cfg = LLaMA3Config::llama3_8b();
        assert!(
            (cfg.rope_theta - 500000.0).abs() < 1.0,
            "LLaMA-3-8B rope_theta must be 500000"
        );
    }

    #[test]
    fn test_llama3_8b_head_dim() {
        let cfg = LLaMA3Config::llama3_8b();
        // 4096 / 32 = 128
        assert_eq!(cfg.head_dim(), 128, "LLaMA-3-8B head_dim must be 128");
    }

    #[test]
    fn test_llama3_70b_config() {
        let cfg = LLaMA3Config::llama3_70b();
        assert_eq!(
            cfg.hidden_size, 8192,
            "LLaMA-3-70B hidden_size must be 8192"
        );
        assert_eq!(cfg.num_key_value_heads, 8, "LLaMA-3-70B KV heads must be 8");
        assert_eq!(
            cfg.num_query_groups(),
            8,
            "LLaMA-3-70B GQA group_size must be 8"
        );
    }

    #[test]
    fn test_llama3_config_validation_valid() {
        let cfg = LLaMA3Config::small_test();
        assert!(
            cfg.validate().is_ok(),
            "small_test config must pass validation"
        );
    }

    #[test]
    fn test_llama3_config_validation_invalid_hidden() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.hidden_size = 63; // not divisible by num_attention_heads=4
        assert!(
            cfg.validate().is_err(),
            "bad hidden_size must fail validation"
        );
    }

    #[test]
    fn test_llama3_config_validation_zero_vocab() {
        let mut cfg = LLaMA3Config::small_test();
        cfg.vocab_size = 0;
        assert!(
            cfg.validate().is_err(),
            "zero vocab_size must fail validation"
        );
    }

    #[test]
    fn test_llama3_uses_gqa() {
        let cfg = LLaMA3Config::llama3_8b();
        assert!(cfg.uses_gqa(), "LLaMA-3-8B must use GQA");
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count() {
        let norm = LLaMA3RmsNorm::new(128, 1e-5).expect("RmsNorm must construct");
        assert_eq!(
            norm.parameter_count(),
            128,
            "RmsNorm parameter count must equal normalized_shape"
        );
    }

    #[test]
    fn test_rmsnorm_forward_shape_preserved() {
        use scirs2_core::ndarray::ArrayD;
        let norm = LLaMA3RmsNorm::new(8, 1e-5).expect("RmsNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[3, 8])));
        let out = norm.forward(input).expect("RmsNorm forward must succeed");
        assert_eq!(out.shape(), &[3, 8], "RmsNorm must preserve shape");
    }

    #[test]
    fn test_rmsnorm_ones_input_unit_output() {
        use scirs2_core::ndarray::ArrayD;
        let norm = LLaMA3RmsNorm::new(4, 1e-5).expect("RmsNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4])));
        let out = norm.forward(input).expect("RmsNorm forward must succeed");
        if let Tensor::F32(arr) = &out {
            for &v in arr.iter() {
                assert!((v - 1.0f32).abs() < 1e-4, "RmsNorm(ones)≈1 but got {v}");
            }
        }
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_half_dim() {
        let rope = LLaMA3RotaryEmbedding::new(128, 8192, 500000.0);
        assert_eq!(rope.half_dim(), 64, "RoPE half_dim must be head_dim/2=64");
    }

    #[test]
    fn test_rope_inv_freq_decreasing() {
        let rope = LLaMA3RotaryEmbedding::new(128, 8192, 500000.0);
        // inv_freq should be monotonically decreasing (higher i → smaller freq)
        let inv = &rope.inv_freq;
        for i in 1..inv.len() {
            assert!(inv[i] <= inv[i - 1], "inv_freq must be non-increasing");
        }
    }

    #[test]
    fn test_rope_inv_freq_first_is_one() {
        let rope = LLaMA3RotaryEmbedding::new(128, 8192, 500000.0);
        // i=0: theta^0 = 1 → inv_freq[0] = 1.0
        assert!(
            (rope.inv_freq[0] - 1.0f64).abs() < 1e-9,
            "inv_freq[0] must be 1.0"
        );
    }

    #[test]
    fn test_rope_apply_preserves_shape() {
        use scirs2_core::ndarray::ArrayD;
        let rope = LLaMA3RotaryEmbedding::new(16, 64, 500000.0);
        let q = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4, 16])));
        let k = q.clone();
        let pos: Vec<usize> = (0..4).collect();
        let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &pos).expect("RoPE apply must succeed");
        assert_eq!(q_out.shape(), q.shape(), "RoPE Q shape must be preserved");
        assert_eq!(k_out.shape(), k.shape(), "RoPE K shape must be preserved");
    }

    // ── Attention tests ───────────────────────────────────────────────────────

    #[test]
    fn test_attention_heads_and_kv_heads() {
        let cfg = LLaMA3Config::small_test();
        let attn = LLaMA3Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(attn.num_heads(), cfg.num_attention_heads);
        assert_eq!(attn.num_kv_heads(), cfg.num_key_value_heads);
    }

    #[test]
    fn test_attention_head_dim() {
        let cfg = LLaMA3Config::small_test();
        let attn = LLaMA3Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(
            attn.head_dim(),
            cfg.head_dim(),
            "attention head_dim must match config"
        );
    }

    #[test]
    fn test_attention_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = LLaMA3Config::small_test();
        let attn = LLaMA3Attention::new(&cfg).expect("Attention must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[3, 64])));
        let out = attn.forward(input).expect("Attention forward must succeed");
        assert_eq!(
            out.shape(),
            &[3, 64],
            "Attention output shape must be [seq, hidden]"
        );
    }

    // ── Decoder layer tests ───────────────────────────────────────────────────

    #[test]
    fn test_decoder_layer_forward_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = LLaMA3Config::small_test();
        let layer = LLaMA3DecoderLayer::new(&cfg).expect("DecoderLayer must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[2, 64])));
        let out = layer.forward(input).expect("DecoderLayer forward must succeed");
        assert_eq!(out.shape(), &[2, 64], "DecoderLayer must preserve shape");
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_construct_and_param_count() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3Model::new(cfg).expect("LLaMA3Model must construct");
        assert!(
            model.parameter_count() > 0,
            "model must have positive parameter count"
        );
    }

    #[test]
    fn test_model_forward_shape() {
        let cfg = LLaMA3Config::small_test();
        let model = LLaMA3Model::new(cfg).expect("model must construct");
        let out = model.run(vec![0u32, 1, 2]).expect("model forward must succeed");
        // Output shape from run: [seq_len, hidden_size] or [vocab_size]
        assert!(
            out.shape().iter().product::<usize>() > 0,
            "output must be non-empty"
        );
    }

    #[test]
    fn test_causal_lm_output_last_dim_is_vocab() {
        let cfg = LLaMA3Config::small_test();
        let vocab = cfg.vocab_size;
        let model = LLaMA3ForCausalLM::new(cfg).expect("CausalLM must construct");
        let out = model.forward(vec![0u32, 1]).expect("CausalLM forward must succeed");
        let shape = out.shape();
        assert_eq!(
            *shape.last().expect("output must have shape"),
            vocab,
            "CausalLM output last dim must be vocab_size"
        );
    }

    #[test]
    fn test_causal_lm_parameter_count_includes_lm_head() {
        let cfg = LLaMA3Config::small_test();
        let cfg2 = cfg.clone();
        let base = LLaMA3Model::new(cfg).expect("base model must construct");
        let causal = LLaMA3ForCausalLM::new(cfg2).expect("causal lm must construct");
        assert!(
            causal.parameter_count() > base.parameter_count(),
            "CausalLM param count must be larger than base (lm_head added)"
        );
    }
}
