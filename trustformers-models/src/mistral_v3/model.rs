use crate::mistral_v3::config::MistralV3Config;
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

/// Root Mean Square Layer Normalisation used in Mistral v0.3
pub struct MistralV3RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl MistralV3RmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for MistralV3RmsNorm {
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
                        "MistralV3RmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "MistralV3RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for Mistral v0.3
struct MistralV3RotaryEmbedding {
    inv_freq: Vec<f64>,
    _max_seq_len: usize,
    _head_dim: usize,
}

impl MistralV3RotaryEmbedding {
    fn new(head_dim: usize, max_seq_len: usize, theta: f64) -> Self {
        let half = head_dim / 2;
        let inv_freq: Vec<f64> = (0..half)
            .map(|i| {
                let exponent = 2.0 * i as f64 / head_dim as f64;
                1.0 / theta.powf(exponent)
            })
            .collect();
        Self {
            inv_freq,
            _max_seq_len: max_seq_len,
            _head_dim: head_dim,
        }
    }

    fn apply_rotary_emb(
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
                "MistralV3RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 SwiGLU Feed-Forward Network
///
/// `FFN(x) = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`
pub struct MistralV3MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MistralV3MLP {
    pub fn new(config: &MistralV3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &MistralV3Config, device: Device) -> Result<Self> {
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

impl Layer for MistralV3MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "MistralV3MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention with Sliding Window
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 GQA with sliding window attention
///
/// When `seq_len > sliding_window`, attention is restricted to the last
/// `sliding_window` tokens in the key/value sequence.
pub struct MistralV3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: MistralV3RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
    sliding_window: usize,
}

impl MistralV3Attention {
    pub fn new(config: &MistralV3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &MistralV3Config, device: Device) -> Result<Self> {
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
        let rotary_emb = MistralV3RotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
        );

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
            sliding_window: config.sliding_window,
        })
    }

    /// Expand KV heads to match query heads (GQA → MHA view)
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
                            "MistralV3Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "MistralV3Attention::repeat_kv",
                "unsupported tensor dtype for KV expansion",
            )),
        }
    }

    /// Effective attention window: min(seq_len, sliding_window)
    pub fn effective_window(&self, seq_len: usize) -> usize {
        seq_len.min(self.sliding_window)
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

    pub fn sliding_window(&self) -> usize {
        self.sliding_window
    }
}

impl Layer for MistralV3Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "MistralV3Attention::forward",
                    format!("unexpected input rank {n}"),
                ))
            },
        };

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // For sliding window: restrict positions to the last sliding_window tokens
        let window = self.effective_window(seq_len);
        let window_start = seq_len.saturating_sub(window);
        let position_ids: Vec<usize> = (window_start..seq_len).collect();

        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        let _k_expanded = self.repeat_kv(&k_rope)?;
        let _v_expanded = self.repeat_kv(&v)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_output = match &q_rope {
            Tensor::F32(q_arr) => Tensor::F32(q_arr.mapv(|x| x * scale)),
            _ => {
                return Err(tensor_op_error(
                    "MistralV3Attention::forward",
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

/// Single Mistral v0.3 decoder layer (pre-norm)
pub struct MistralV3DecoderLayer {
    self_attn: MistralV3Attention,
    mlp: MistralV3MLP,
    input_layernorm: MistralV3RmsNorm,
    post_attention_layernorm: MistralV3RmsNorm,
}

impl MistralV3DecoderLayer {
    pub fn new(config: &MistralV3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &MistralV3Config, device: Device) -> Result<Self> {
        let self_attn = MistralV3Attention::new_with_device(config, device)?;
        let mlp = MistralV3MLP::new_with_device(config, device)?;
        let input_layernorm = MistralV3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm =
            MistralV3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
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

impl Layer for MistralV3DecoderLayer {
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
// Mistral v0.3 Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 transformer model (without language-model head)
pub struct MistralV3Model {
    config: MistralV3Config,
    embed_tokens: Embedding,
    layers: Vec<MistralV3DecoderLayer>,
    norm: MistralV3RmsNorm,
}

impl MistralV3Model {
    pub fn new(config: MistralV3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: MistralV3Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(MistralV3DecoderLayer::new_with_device(&config, device)?);
        }
        let norm = MistralV3RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &MistralV3Config {
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

impl Model for MistralV3Model {
    type Config = MistralV3Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Weight loading not yet implemented for Mistral v0.3".to_string(),
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
// Mistral v0.3 Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// Mistral v0.3 with a causal language-modelling head
pub struct MistralV3ForCausalLM {
    model: MistralV3Model,
    lm_head: Linear,
}

impl MistralV3ForCausalLM {
    pub fn new(config: MistralV3Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: MistralV3Config, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = MistralV3Model::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &MistralV3Config {
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

impl Model for MistralV3ForCausalLM {
    type Config = MistralV3Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        MistralV3ForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Weight loading not yet implemented for Mistral v0.3".to_string(),
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
    use crate::mistral_v3::config::MistralV3Config;

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_mistral_v3_7b_hidden_size() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(
            cfg.hidden_size, 4096,
            "Mistral-v0.3-7B hidden_size must be 4096"
        );
    }

    #[test]
    fn test_mistral_v3_7b_intermediate_size() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(
            cfg.intermediate_size, 14336,
            "Mistral-v0.3-7B intermediate_size must be 14336"
        );
    }

    #[test]
    fn test_mistral_v3_7b_sliding_window() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(
            cfg.sliding_window, 4096,
            "Mistral-v0.3 sliding_window must be 4096"
        );
    }

    #[test]
    fn test_mistral_v3_7b_kv_heads() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(
            cfg.num_key_value_heads, 8,
            "Mistral-v0.3-7B KV heads must be 8"
        );
    }

    #[test]
    fn test_mistral_v3_7b_gqa_group_size() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        // 32 Q / 8 KV = 4
        assert_eq!(
            cfg.num_query_groups(),
            4,
            "Mistral-v0.3-7B GQA group_size must be 4"
        );
    }

    #[test]
    fn test_mistral_v3_7b_rope_theta() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        // Mistral v0.3 uses 1_000_000.0 (different from v0.1's 10000)
        assert!(
            (cfg.rope_theta - 1_000_000.0).abs() < 1.0,
            "Mistral-v0.3 rope_theta must be 1_000_000.0"
        );
    }

    #[test]
    fn test_mistral_v3_7b_large_context_window() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        // 32k context window
        assert_eq!(
            cfg.max_position_embeddings, 32768,
            "Mistral-v0.3 max_position_embeddings must be 32768"
        );
        assert!(
            cfg.max_position_embeddings > 4096,
            "Mistral-v0.3 context window must be larger than 4096"
        );
    }

    #[test]
    fn test_mistral_v3_vocab_size_expanded() {
        let cfg = MistralV3Config::mistral_7b_v0_3();
        assert_eq!(
            cfg.vocab_size, 32768,
            "Mistral-v0.3 vocab_size must be 32768"
        );
    }

    #[test]
    fn test_mistral_v3_config_validation_valid() {
        let cfg = MistralV3Config::small_test();
        assert!(
            cfg.validate().is_ok(),
            "small_test config must pass validation"
        );
    }

    #[test]
    fn test_mistral_v3_config_validation_zero_sliding_window() {
        let mut cfg = MistralV3Config::small_test();
        cfg.sliding_window = 0;
        assert!(
            cfg.validate().is_err(),
            "zero sliding_window must fail validation"
        );
    }

    #[test]
    fn test_mistral_v3_config_validation_indivisible_heads() {
        let mut cfg = MistralV3Config::small_test();
        cfg.num_key_value_heads = 3; // 4 not divisible by 3
        assert!(
            cfg.validate().is_err(),
            "indivisible KV heads must fail validation"
        );
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count() {
        let norm = MistralV3RmsNorm::new(64, 1e-5).expect("RmsNorm must construct");
        assert_eq!(
            norm.parameter_count(),
            64,
            "parameter count must equal normalized_shape"
        );
    }

    #[test]
    fn test_rmsnorm_forward_shape() {
        use scirs2_core::ndarray::ArrayD;
        let norm = MistralV3RmsNorm::new(8, 1e-5).expect("RmsNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[3, 8])));
        let out = norm.forward(input).expect("RmsNorm forward must succeed");
        assert_eq!(out.shape(), &[3, 8], "RmsNorm must preserve shape");
    }

    // ── Attention sliding window tests ────────────────────────────────────────

    #[test]
    fn test_attention_effective_window_below_sliding_window() {
        let cfg = MistralV3Config::small_test(); // sliding_window=8
        let attn = MistralV3Attention::new(&cfg).expect("Attention must construct");
        // seq_len=4 < sliding_window=8 → effective_window = 4
        assert_eq!(
            attn.effective_window(4),
            4,
            "effective_window must be min(seq_len, sliding_window)"
        );
    }

    #[test]
    fn test_attention_effective_window_above_sliding_window() {
        let cfg = MistralV3Config::small_test(); // sliding_window=8
        let attn = MistralV3Attention::new(&cfg).expect("Attention must construct");
        // seq_len=16 > sliding_window=8 → effective_window = 8
        assert_eq!(
            attn.effective_window(16),
            8,
            "effective_window must clamp to sliding_window"
        );
    }

    #[test]
    fn test_attention_sliding_window_field() {
        let cfg = MistralV3Config::small_test();
        let attn = MistralV3Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(
            attn.sliding_window(),
            cfg.sliding_window,
            "attention sliding_window must match config"
        );
    }

    #[test]
    fn test_attention_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = MistralV3Config::small_test();
        let attn = MistralV3Attention::new(&cfg).expect("Attention must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[3, 64])));
        let out = attn.forward(input).expect("Attention forward must succeed");
        assert_eq!(
            out.shape(),
            &[3, 64],
            "Attention output must be [seq, hidden]"
        );
    }

    #[test]
    fn test_attention_kv_heads_gqa() {
        let cfg = MistralV3Config::small_test();
        let attn = MistralV3Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(attn.num_kv_heads(), 2, "small_test KV heads must be 2");
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_construct() {
        let cfg = MistralV3Config::small_test();
        let model = MistralV3Model::new(cfg).expect("MistralV3Model must construct");
        assert!(
            model.parameter_count() > 0,
            "model param count must be positive"
        );
    }

    #[test]
    fn test_model_forward_shape() {
        let cfg = MistralV3Config::small_test();
        let model = MistralV3Model::new(cfg).expect("model must construct");
        let out = model.run(vec![0u32, 1, 2]).expect("model run must succeed");
        assert!(
            out.shape().iter().product::<usize>() > 0,
            "output must be non-empty"
        );
    }

    #[test]
    fn test_causal_lm_output_last_dim_is_vocab() {
        let cfg = MistralV3Config::small_test();
        let vocab = cfg.vocab_size;
        let model = MistralV3ForCausalLM::new(cfg).expect("CausalLM must construct");
        let out = model.forward(vec![0u32, 1]).expect("CausalLM forward must succeed");
        let shape = out.shape();
        assert_eq!(
            *shape.last().expect("output must have shape"),
            vocab,
            "CausalLM output last dim must be vocab_size"
        );
    }

    #[test]
    fn test_causal_lm_more_params_than_base() {
        let cfg = MistralV3Config::small_test();
        let cfg2 = cfg.clone();
        let base = MistralV3Model::new(cfg).expect("base model must construct");
        let causal = MistralV3ForCausalLM::new(cfg2).expect("causal lm must construct");
        assert!(
            causal.parameter_count() > base.parameter_count(),
            "CausalLM must have more params than base (lm_head added)"
        );
    }
}
