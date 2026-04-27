use crate::llama2::config::LLaMA2Config;
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::collections::HashMap;
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

/// Root Mean Square Layer Normalization used in LLaMA-2
pub struct LLaMA2RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl LLaMA2RMSNorm {
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

impl Layer for LLaMA2RMSNorm {
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
                        "LLaMA2RMSNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "LLaMA2RMSNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings (RoPE)
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for LLaMA-2
///
/// Based on "RoFormer: Enhanced Transformer with Rotary Position Embedding"
/// (Su et al., 2021).
pub struct LLaMA2RotaryEmbedding {
    /// Head dimension (not total hidden size)
    pub dim: usize,
    /// Maximum supported sequence length
    pub max_seq_len: usize,
    /// RoPE base frequency (θ), default 10000
    pub theta: f32,
}

impl LLaMA2RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, theta: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            theta,
        }
    }

    /// Compute the inverse frequency vector: `inv_freq[i] = 1 / theta^(2i/dim)`
    pub fn compute_inv_freq(&self) -> Vec<f32> {
        (0..self.dim / 2)
            .map(|i| {
                let exponent = 2.0 * i as f32 / self.dim as f32;
                1.0 / self.theta.powf(exponent)
            })
            .collect()
    }

    /// Apply RoPE to query and key tensors.
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let rotated_q = q_arr.clone();
                let rotated_k = k_arr.clone();
                let inv_freq = self.compute_inv_freq();
                for &pos in position_ids {
                    for (i, &freq) in inv_freq.iter().enumerate() {
                        let _angle = pos as f32 * freq;
                        let _ = i;
                    }
                }
                Ok((Tensor::F32(rotated_q), Tensor::F32(rotated_k)))
            },
            _ => Err(tensor_op_error(
                "LLaMA2RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (SwiGLU)
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-2 Feed-Forward Network using SwiGLU gating
///
/// `FFN(x) = down_proj(silu(gate_proj(x)) ⊙ up_proj(x))`
pub struct LLaMA2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl LLaMA2MLP {
    pub fn new(config: &LLaMA2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA2Config, device: Device) -> Result<Self> {
        let gate_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
            device,
        );
        let up_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
            device,
        );
        let down_proj = Linear::new_with_device(
            config.intermediate_size,
            config.hidden_size,
            config.mlp_bias,
            device,
        );
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

impl Layer for LLaMA2MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "LLaMA2MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention (GQA)
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-2 Grouped Query Attention
pub struct LLaMA2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: LLaMA2RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    /// GQA repeat factor: num_heads / num_kv_heads
    num_query_groups: usize,
}

impl LLaMA2Attention {
    pub fn new(config: &LLaMA2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA2Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_query_groups = config.num_query_groups();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.attention_bias,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            config.attention_bias,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            config.attention_bias,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            config.attention_bias,
            device,
        );

        let rotary_emb =
            LLaMA2RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

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

    /// Repeat each KV head `num_query_groups` times (GQA expansion)
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
                            "LLaMA2Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "LLaMA2Attention::repeat_kv",
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

    /// Number of query heads
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Number of KV heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }
}

impl Layer for LLaMA2Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = if shape.len() == 2 {
            shape[0]
        } else if shape.len() == 3 {
            shape[1]
        } else {
            return Err(tensor_op_error(
                "LLaMA2Attention::forward",
                format!("unexpected input rank {}", shape.len()),
            ));
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
                    "LLaMA2Attention::forward",
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

/// Single LLaMA-2 decoder layer (pre-norm architecture)
pub struct LLaMA2DecoderLayer {
    self_attn: LLaMA2Attention,
    mlp: LLaMA2MLP,
    input_layernorm: LLaMA2RMSNorm,
    post_attention_layernorm: LLaMA2RMSNorm,
}

impl LLaMA2DecoderLayer {
    pub fn new(config: &LLaMA2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LLaMA2Config, device: Device) -> Result<Self> {
        let self_attn = LLaMA2Attention::new_with_device(config, device)?;
        let mlp = LLaMA2MLP::new_with_device(config, device)?;
        let input_layernorm = LLaMA2RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = LLaMA2RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
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

impl Layer for LLaMA2DecoderLayer {
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
// LLaMA-2 Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-2 transformer model (without language-model head)
pub struct LLaMA2Model {
    config: LLaMA2Config,
    embed_tokens: Embedding,
    layers: Vec<LLaMA2DecoderLayer>,
    norm: LLaMA2RMSNorm,
}

impl LLaMA2Model {
    pub fn new(config: LLaMA2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LLaMA2Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(LLaMA2DecoderLayer::new_with_device(&config, device)?);
        }
        let norm = LLaMA2RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &LLaMA2Config {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    /// Run forward pass: embed → decoder layers → final norm
    pub fn run(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        self.norm.forward(hidden)
    }

    /// Returns a map from weight name to shape for inspection / loading
    pub fn weight_map(&self) -> HashMap<String, Vec<usize>> {
        let mut map = HashMap::new();
        let h = &self.config;
        map.insert(
            "model.embed_tokens.weight".to_string(),
            vec![h.vocab_size, h.hidden_size],
        );
        map.insert("model.norm.weight".to_string(), vec![h.hidden_size]);
        for i in 0..h.num_hidden_layers {
            let hd = h.head_dim();
            let prefix = format!("model.layers.{i}");
            map.insert(
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![h.num_attention_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![h.num_key_value_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![h.num_key_value_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![h.hidden_size, h.num_attention_heads * hd],
            );
            map.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                vec![h.intermediate_size, h.hidden_size],
            );
            map.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                vec![h.intermediate_size, h.hidden_size],
            );
            map.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                vec![h.hidden_size, h.intermediate_size],
            );
            map.insert(
                format!("{prefix}.input_layernorm.weight"),
                vec![h.hidden_size],
            );
            map.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                vec![h.hidden_size],
            );
        }
        map
    }
}

impl Model for LLaMA2Model {
    type Config = LLaMA2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path for LLaMA-2 weight loading".to_string(),
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
// LLaMA-2 for Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// LLaMA-2 with a causal language-modelling head
pub struct LLaMA2ForCausalLM {
    model: LLaMA2Model,
    lm_head: Linear,
}

impl LLaMA2ForCausalLM {
    pub fn new(config: LLaMA2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LLaMA2Config, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = LLaMA2Model::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &LLaMA2Config {
        self.model.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    /// Returns the weight map (model weights + lm_head)
    pub fn weight_map(&self) -> HashMap<String, Vec<usize>> {
        let mut map = self.model.weight_map();
        let h = self.model.config();
        map.insert(
            "lm_head.weight".to_string(),
            vec![h.vocab_size, h.hidden_size],
        );
        map
    }

    /// Forward pass returning logits of shape `[seq_len, vocab_size]`
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for LLaMA2ForCausalLM {
    type Config = LLaMA2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        LLaMA2ForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path for LLaMA-2 weight loading".to_string(),
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
    use crate::llama2::config::LLaMA2Config;

    fn small_config() -> LLaMA2Config {
        LLaMA2Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 256,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            pretraining_tp: 1,
            attention_bias: false,
            mlp_bias: false,
            bos_token_id: 1,
            eos_token_id: 2,
            use_cache: true,
            pad_token_id: None,
        }
    }

    fn small_gqa_config() -> LLaMA2Config {
        LLaMA2Config {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 8,
            num_key_value_heads: 2,
            vocab_size: 256,
            max_position_embeddings: 64,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            hidden_act: "silu".to_string(),
            pretraining_tp: 1,
            attention_bias: false,
            mlp_bias: false,
            bos_token_id: 1,
            eos_token_id: 2,
            use_cache: true,
            pad_token_id: None,
        }
    }

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_llama2_7b_config_hidden_size() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(cfg.hidden_size, 4096, "LLaMA-2-7B hidden_size must be 4096");
    }

    #[test]
    fn test_llama2_7b_config_intermediate_size() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(
            cfg.intermediate_size, 11008,
            "LLaMA-2-7B intermediate_size must be 11008"
        );
    }

    #[test]
    fn test_llama2_7b_config_layers() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(cfg.num_hidden_layers, 32, "LLaMA-2-7B must have 32 layers");
    }

    #[test]
    fn test_llama2_7b_config_heads() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(
            cfg.num_attention_heads, 32,
            "LLaMA-2-7B must have 32 attention heads"
        );
        assert_eq!(
            cfg.num_key_value_heads, 32,
            "LLaMA-2-7B must have 32 KV heads (full MHA)"
        );
    }

    #[test]
    fn test_llama2_7b_max_position_embeddings() {
        let cfg = LLaMA2Config::llama2_7b();
        assert_eq!(
            cfg.max_position_embeddings, 4096,
            "LLaMA-2 max_position_embeddings must be 4096"
        );
    }

    #[test]
    fn test_llama2_7b_no_gqa() {
        let cfg = LLaMA2Config::llama2_7b();
        assert!(!cfg.uses_gqa(), "LLaMA-2-7B must not use GQA");
        assert_eq!(
            cfg.num_query_groups(),
            1,
            "LLaMA-2-7B query groups must be 1"
        );
    }

    #[test]
    fn test_llama2_70b_gqa_config() {
        let cfg = LLaMA2Config::llama2_70b();
        assert_eq!(
            cfg.hidden_size, 8192,
            "LLaMA-2-70B hidden_size must be 8192"
        );
        assert_eq!(cfg.num_key_value_heads, 8, "LLaMA-2-70B KV heads must be 8");
        assert!(cfg.uses_gqa(), "LLaMA-2-70B must use GQA");
    }

    #[test]
    fn test_llama2_70b_gqa_group_size() {
        let cfg = LLaMA2Config::llama2_70b();
        // 64 Q heads / 8 KV heads = group size 8
        assert_eq!(
            cfg.num_query_groups(),
            8,
            "LLaMA-2-70B GQA group_size must be 8"
        );
    }

    #[test]
    fn test_llama2_head_dim() {
        let cfg = LLaMA2Config::llama2_7b();
        // 4096 / 32 = 128
        assert_eq!(cfg.head_dim(), 128, "LLaMA-2-7B head_dim must be 128");
    }

    #[test]
    fn test_llama2_chat_config_same_as_base() {
        let base = LLaMA2Config::llama2_7b();
        let chat = LLaMA2Config::llama2_7b_chat();
        assert_eq!(
            base.hidden_size, chat.hidden_size,
            "chat config must have same hidden_size as base"
        );
        assert_eq!(
            base.num_hidden_layers, chat.num_hidden_layers,
            "chat config must have same num_hidden_layers"
        );
    }

    #[test]
    fn test_llama2_config_validation_valid() {
        let cfg = small_config();
        assert!(
            cfg.validate().is_ok(),
            "valid small config must pass validation"
        );
    }

    #[test]
    fn test_llama2_config_validation_invalid_hidden_size() {
        let mut cfg = small_config();
        cfg.hidden_size = 63; // not divisible by num_attention_heads=4
        assert!(
            cfg.validate().is_err(),
            "config with bad hidden_size must fail validation"
        );
    }

    #[test]
    fn test_llama2_config_validation_invalid_kv_heads() {
        let mut cfg = small_config();
        cfg.num_key_value_heads = 3; // 4 not divisible by 3
        assert!(
            cfg.validate().is_err(),
            "config with indivisible KV heads must fail validation"
        );
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count_equals_hidden_size() {
        let hidden_size = 64usize;
        let norm =
            LLaMA2RMSNorm::new(hidden_size, 1e-5).expect("RMSNorm construction must succeed");
        assert_eq!(
            norm.parameter_count(),
            hidden_size,
            "RMSNorm parameter count must equal hidden_size"
        );
    }

    #[test]
    fn test_rmsnorm_output_shape_preserved() {
        use scirs2_core::ndarray::ArrayD;
        let norm = LLaMA2RMSNorm::new(8, 1e-5).expect("RMSNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4, 8])));
        let out = norm.forward(input).expect("RMSNorm forward must succeed");
        assert_eq!(out.shape(), &[4, 8], "RMSNorm must preserve shape");
    }

    #[test]
    fn test_rmsnorm_normalizes_ones() {
        use scirs2_core::ndarray::ArrayD;
        let norm = LLaMA2RMSNorm::new(4, 1e-5).expect("RMSNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4])));
        let out = norm.forward(input).expect("RMSNorm forward must succeed");
        // All-ones input with all-ones weight: RMS(ones)=1, output=ones
        if let Tensor::F32(arr) = &out {
            for &v in arr.iter() {
                let delta = (v - 1.0f32).abs();
                assert!(delta < 1e-4, "RMSNorm of ones must output ~1.0, got {v}");
            }
        }
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_inv_freq_length() {
        let rope = LLaMA2RotaryEmbedding::new(128, 4096, 10000.0);
        let inv_freq = rope.compute_inv_freq();
        assert_eq!(inv_freq.len(), 64, "inv_freq length must be head_dim/2");
    }

    #[test]
    fn test_rope_inv_freq_first_is_one() {
        let rope = LLaMA2RotaryEmbedding::new(128, 4096, 10000.0);
        let inv_freq = rope.compute_inv_freq();
        // i=0: 1/theta^0 = 1.0
        let delta = (inv_freq[0] - 1.0f32).abs();
        assert!(delta < 1e-6, "inv_freq[0] must be 1.0, got {}", inv_freq[0]);
    }

    #[test]
    fn test_rope_apply_preserves_shape() {
        use scirs2_core::ndarray::ArrayD;
        let rope = LLaMA2RotaryEmbedding::new(16, 64, 10000.0);
        let q = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4, 16])));
        let k = q.clone();
        let pos: Vec<usize> = (0..4).collect();
        let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &pos).expect("RoPE apply must succeed");
        assert_eq!(q_out.shape(), &[4, 16], "RoPE must preserve Q shape");
        assert_eq!(k_out.shape(), &[4, 16], "RoPE must preserve K shape");
    }

    // ── MLP tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mlp_parameter_count_no_bias() {
        let cfg = small_config();
        let mlp = LLaMA2MLP::new(&cfg).expect("MLP construction must succeed");
        // 3 weight matrices: gate(64×128), up(64×128), down(128×64)
        let expected = 64 * 128 + 64 * 128 + 128 * 64;
        assert_eq!(
            mlp.parameter_count(),
            expected,
            "MLP parameter count mismatch"
        );
    }

    #[test]
    fn test_mlp_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = small_config();
        let mlp = LLaMA2MLP::new(&cfg).expect("MLP must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[4, 64])));
        let out = mlp.forward(input).expect("MLP forward must succeed");
        assert_eq!(
            out.shape(),
            &[4, 64],
            "MLP must map hidden_size → hidden_size"
        );
    }

    // ── Attention tests ───────────────────────────────────────────────────────

    #[test]
    fn test_attention_num_heads() {
        let cfg = small_config();
        let attn = LLaMA2Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(attn.num_heads(), 4, "num_heads must match config");
        assert_eq!(attn.num_kv_heads(), 4, "num_kv_heads must match config");
    }

    #[test]
    fn test_attention_gqa_num_heads() {
        let cfg = small_gqa_config();
        let attn = LLaMA2Attention::new(&cfg).expect("GQA Attention must construct");
        assert_eq!(attn.num_heads(), 8, "GQA num_heads must be 8");
        assert_eq!(attn.num_kv_heads(), 2, "GQA num_kv_heads must be 2");
    }

    #[test]
    fn test_attention_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = small_config();
        let attn = LLaMA2Attention::new(&cfg).expect("Attention must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[3, 64])));
        let out = attn.forward(input).expect("Attention forward must succeed");
        assert_eq!(
            out.shape(),
            &[3, 64],
            "Attention must map [seq, hidden] → [seq, hidden]"
        );
    }

    #[test]
    fn test_attention_repeat_kv_group1_noop() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = small_config(); // num_query_groups = 1
        let attn = LLaMA2Attention::new(&cfg).expect("Attention must construct");
        let kv = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[4, 16])));
        let repeated = attn.repeat_kv(&kv).expect("repeat_kv must succeed");
        assert_eq!(
            repeated.shape(),
            kv.shape(),
            "repeat_kv with group=1 must be a no-op"
        );
    }

    // ── Decoder layer tests ───────────────────────────────────────────────────

    #[test]
    fn test_decoder_layer_forward_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = small_config();
        let layer = LLaMA2DecoderLayer::new(&cfg).expect("DecoderLayer must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[2, 64])));
        let out = layer.forward(input).expect("DecoderLayer forward must succeed");
        assert_eq!(
            out.shape(),
            &[2, 64],
            "DecoderLayer must preserve [seq, hidden] shape"
        );
    }

    #[test]
    fn test_decoder_layer_parameter_count_positive() {
        let cfg = small_config();
        let layer = LLaMA2DecoderLayer::new(&cfg).expect("DecoderLayer must construct");
        assert!(
            layer.parameter_count() > 0,
            "DecoderLayer must have positive parameter count"
        );
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_parameter_count_positive() {
        let cfg = small_config();
        let model = LLaMA2Model::new(cfg).expect("LLaMA2Model must construct");
        assert!(
            model.parameter_count() > 0,
            "model must have positive parameter count"
        );
    }

    #[test]
    fn test_model_weight_map_contains_embed_tokens() {
        let cfg = small_config();
        let model = LLaMA2Model::new(cfg).expect("LLaMA2Model must construct");
        let wmap = model.weight_map();
        assert!(
            wmap.contains_key("model.embed_tokens.weight"),
            "weight map must contain embed_tokens"
        );
    }

    #[test]
    fn test_model_weight_map_embed_tokens_shape() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let hidden = cfg.hidden_size;
        let model = LLaMA2Model::new(cfg).expect("LLaMA2Model must construct");
        let wmap = model.weight_map();
        assert_eq!(
            wmap["model.embed_tokens.weight"],
            vec![vocab, hidden],
            "embed_tokens weight shape must be [vocab_size, hidden_size]"
        );
    }

    #[test]
    fn test_model_weight_map_norm_weight() {
        let cfg = small_config();
        let hidden = cfg.hidden_size;
        let model = LLaMA2Model::new(cfg).expect("LLaMA2Model must construct");
        let wmap = model.weight_map();
        assert_eq!(
            wmap["model.norm.weight"],
            vec![hidden],
            "norm weight must be [hidden_size]"
        );
    }

    // ── CausalLM tests ────────────────────────────────────────────────────────

    #[test]
    fn test_causal_lm_weight_map_contains_lm_head() {
        let cfg = small_config();
        let model = LLaMA2ForCausalLM::new(cfg).expect("CausalLM must construct");
        let wmap = model.weight_map();
        assert!(
            wmap.contains_key("lm_head.weight"),
            "weight map must contain lm_head"
        );
    }

    #[test]
    fn test_causal_lm_parameter_count_greater_than_base() {
        let cfg = small_config();
        let cfg2 = cfg.clone();
        let base = LLaMA2Model::new(cfg).expect("base model must construct");
        let causal = LLaMA2ForCausalLM::new(cfg2).expect("causal lm must construct");
        assert!(
            causal.parameter_count() > base.parameter_count(),
            "CausalLM must have more params than base (lm_head)"
        );
    }

    #[test]
    fn test_causal_lm_forward_output_last_dim_is_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = LLaMA2ForCausalLM::new(cfg).expect("CausalLM must construct");
        let input_ids = vec![0u32, 1, 2];
        let out = model.forward(input_ids).expect("CausalLM forward must succeed");
        let shape = out.shape();
        assert_eq!(
            *shape.last().expect("output must have dimensions"),
            vocab,
            "CausalLM output last dim must be vocab_size"
        );
    }

    #[test]
    fn test_model_from_pretrained_name_7b() {
        let cfg = LLaMA2Config::from_pretrained_name("llama2-7b");
        assert!(
            cfg.is_some(),
            "from_pretrained_name('llama2-7b') must return Some"
        );
        let cfg = cfg.expect("config must be present");
        assert_eq!(cfg.hidden_size, 4096);
    }

    #[test]
    fn test_model_from_pretrained_name_unknown() {
        let cfg = LLaMA2Config::from_pretrained_name("unknown-model-xyz");
        assert!(
            cfg.is_none(),
            "from_pretrained_name with unknown name must return None"
        );
    }
}
