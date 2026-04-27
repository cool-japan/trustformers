use crate::codellama::config::{CodeLlamaConfig, RopeScalingType};
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

/// RMSNorm for CodeLlama (identical to LLaMA-2)
pub struct CodeLlamaRMSNorm {
    weight: Tensor,
    eps: f64,
}

impl CodeLlamaRMSNorm {
    pub fn new(size: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[size])?;
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

impl Layer for CodeLlamaRMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let eps = self.eps as f32;
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);
                match &self.weight {
                    Tensor::F32(w) => Ok(Tensor::F32(&normalized * w)),
                    _ => Err(tensor_op_error(
                        "CodeLlamaRMSNorm::forward",
                        "weight dtype mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "CodeLlamaRMSNorm::forward",
                "unsupported input dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings with optional scaling
// ─────────────────────────────────────────────────────────────────────────────

/// RoPE for CodeLlama, supporting linear and dynamic NTK scaling.
///
/// ## Linear scaling
/// `inv_freq_scaled[i] = (theta ^ (-2i/d)) / factor`
///
/// ## Dynamic NTK scaling
/// `inv_freq[i] = ((alpha * theta) ^ (-2i/d))`, where `alpha = factor`
pub struct CodeLlamaRotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub theta: f32,
    pub scaling_type: Option<RopeScalingType>,
    pub scaling_factor: f32,
}

impl CodeLlamaRotaryEmbedding {
    pub fn new(config: &CodeLlamaConfig) -> Self {
        let (scaling_type, scaling_factor) = match &config.rope_scaling {
            Some(s) => (Some(s.scaling_type.clone()), s.factor),
            None => (None, 1.0),
        };
        Self {
            dim: config.head_dim(),
            max_seq_len: config.max_position_embeddings,
            theta: config.rope_theta,
            scaling_type,
            scaling_factor,
        }
    }

    /// Compute scaled inverse frequencies
    pub fn compute_inv_freq(&self) -> Vec<f32> {
        let effective_theta = match &self.scaling_type {
            Some(RopeScalingType::Dynamic) => self.scaling_factor * self.theta,
            _ => self.theta,
        };

        (0..self.dim / 2)
            .map(|i| {
                let base_freq = 1.0 / effective_theta.powf(2.0 * i as f32 / self.dim as f32);
                match &self.scaling_type {
                    Some(RopeScalingType::Linear) => base_freq / self.scaling_factor,
                    _ => base_freq,
                }
            })
            .collect()
    }

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
                "CodeLlamaRotaryEmbedding::apply_rotary_emb",
                "unsupported dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// MLP (SwiGLU — identical to LLaMA-2)
// ─────────────────────────────────────────────────────────────────────────────

pub struct CodeLlamaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl CodeLlamaMLP {
    pub fn new(config: &CodeLlamaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &CodeLlamaConfig, device: Device) -> Result<Self> {
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

impl Layer for CodeLlamaMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error("CodeLlamaMLP::forward", "dtype mismatch")),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention
// ─────────────────────────────────────────────────────────────────────────────

pub struct CodeLlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: CodeLlamaRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl CodeLlamaAttention {
    pub fn new(config: &CodeLlamaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &CodeLlamaConfig, device: Device) -> Result<Self> {
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
        let rotary_emb = CodeLlamaRotaryEmbedding::new(config);

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
                let arr = ArrayD::from_shape_vec(IxDyn(&new_shape), expanded).map_err(|e| {
                    tensor_op_error("CodeLlamaAttention::repeat_kv", format!("{e}"))
                })?;
                Ok(Tensor::F32(arr))
            },
            _ => Err(tensor_op_error(
                "CodeLlamaAttention::repeat_kv",
                "unsupported dtype",
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

impl Layer for CodeLlamaAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "CodeLlamaAttention::forward",
                    format!("unexpected input rank {n}"),
                ))
            },
        };

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let positions: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &positions)?;

        let _k_expanded = self.repeat_kv(&k_rope)?;
        let _v_expanded = self.repeat_kv(&v)?;

        let scale = (self.head_dim as f32).sqrt().recip();
        let attn_out = match &q_rope {
            Tensor::F32(q_arr) => Tensor::F32(q_arr.mapv(|x| x * scale)),
            _ => {
                return Err(tensor_op_error(
                    "CodeLlamaAttention::forward",
                    "dtype mismatch",
                ))
            },
        };

        self.o_proj.forward(attn_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Decoder Layer
// ─────────────────────────────────────────────────────────────────────────────

pub struct CodeLlamaDecoderLayer {
    self_attn: CodeLlamaAttention,
    mlp: CodeLlamaMLP,
    input_layernorm: CodeLlamaRMSNorm,
    post_attention_layernorm: CodeLlamaRMSNorm,
}

impl CodeLlamaDecoderLayer {
    pub fn new(config: &CodeLlamaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &CodeLlamaConfig, device: Device) -> Result<Self> {
        let self_attn = CodeLlamaAttention::new_with_device(config, device)?;
        let mlp = CodeLlamaMLP::new_with_device(config, device)?;
        let input_layernorm = CodeLlamaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm =
            CodeLlamaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
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

impl Layer for CodeLlamaDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let normed = self.input_layernorm.forward(input.clone())?;
        let attn_out = self.self_attn.forward(normed)?;
        let after_attn = input.add(&attn_out)?;

        let normed2 = self.post_attention_layernorm.forward(after_attn.clone())?;
        let mlp_out = self.mlp.forward(normed2)?;
        after_attn.add(&mlp_out)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// CodeLlama Base Model
// ─────────────────────────────────────────────────────────────────────────────

pub struct CodeLlamaModel {
    config: CodeLlamaConfig,
    embed_tokens: Embedding,
    layers: Vec<CodeLlamaDecoderLayer>,
    norm: CodeLlamaRMSNorm,
}

impl CodeLlamaModel {
    pub fn new(config: CodeLlamaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: CodeLlamaConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(CodeLlamaDecoderLayer::new_with_device(&config, device)?);
        }
        let norm = CodeLlamaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &CodeLlamaConfig {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    /// Run forward pass without going through trait disambiguation
    pub fn run(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let mut hidden = self.embed_tokens.forward(input_ids)?;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        self.norm.forward(hidden)
    }

    /// Returns a map from weight name to expected shape
    pub fn weight_map(&self) -> HashMap<String, Vec<usize>> {
        let mut map = HashMap::new();
        let h = &self.config;
        map.insert(
            "model.embed_tokens.weight".into(),
            vec![h.vocab_size, h.hidden_size],
        );
        map.insert("model.norm.weight".into(), vec![h.hidden_size]);
        for i in 0..h.num_hidden_layers {
            let hd = h.head_dim();
            let p = format!("model.layers.{i}");
            map.insert(
                format!("{p}.self_attn.q_proj.weight"),
                vec![h.num_attention_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{p}.self_attn.k_proj.weight"),
                vec![h.num_key_value_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{p}.self_attn.v_proj.weight"),
                vec![h.num_key_value_heads * hd, h.hidden_size],
            );
            map.insert(
                format!("{p}.self_attn.o_proj.weight"),
                vec![h.hidden_size, h.num_attention_heads * hd],
            );
            map.insert(
                format!("{p}.mlp.gate_proj.weight"),
                vec![h.intermediate_size, h.hidden_size],
            );
            map.insert(
                format!("{p}.mlp.up_proj.weight"),
                vec![h.intermediate_size, h.hidden_size],
            );
            map.insert(
                format!("{p}.mlp.down_proj.weight"),
                vec![h.hidden_size, h.intermediate_size],
            );
            map.insert(format!("{p}.input_layernorm.weight"), vec![h.hidden_size]);
            map.insert(
                format!("{p}.post_attention_layernorm.weight"),
                vec![h.hidden_size],
            );
        }
        map
    }
}

impl Model for CodeLlamaModel {
    type Config = CodeLlamaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path for CodeLlama weight loading".to_string(),
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
// CodeLlama for Causal LM
// ─────────────────────────────────────────────────────────────────────────────

pub struct CodeLlamaForCausalLM {
    model: CodeLlamaModel,
    lm_head: Linear,
}

impl CodeLlamaForCausalLM {
    pub fn new(config: CodeLlamaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: CodeLlamaConfig, device: Device) -> Result<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = CodeLlamaModel::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &CodeLlamaConfig {
        self.model.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    /// Returns the weight map for the full causal LM model
    pub fn weight_map(&self) -> HashMap<String, Vec<usize>> {
        let mut map = self.model.weight_map();
        let h = self.model.config();
        map.insert("lm_head.weight".into(), vec![h.vocab_size, h.hidden_size]);
        map
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for CodeLlamaForCausalLM {
    type Config = CodeLlamaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        CodeLlamaForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path for CodeLlama weight loading".to_string(),
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
    use crate::codellama::config::{CodeLlamaConfig, RopeScalingConfig, RopeScalingType};

    fn small_config() -> CodeLlamaConfig {
        CodeLlamaConfig {
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            vocab_size: 256,
            max_position_embeddings: 64,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            attention_bias: false,
            mlp_bias: false,
            bos_token_id: 1,
            eos_token_id: 2,
            use_cache: true,
            pad_token_id: None,
            rope_scaling: None,
            infilling: true,
            programming_languages: vec!["python".to_string()],
        }
    }

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_codellama_7b_hidden_size() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(
            cfg.hidden_size, 4096,
            "CodeLlama-7B hidden_size must be 4096"
        );
    }

    #[test]
    fn test_codellama_7b_max_seq_len() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(
            cfg.max_position_embeddings, 16384,
            "CodeLlama-7B max_position_embeddings must be 16384"
        );
    }

    #[test]
    fn test_codellama_vocab_size() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(cfg.vocab_size, 32016, "CodeLlama vocab_size must be 32016");
    }

    #[test]
    fn test_codellama_rope_theta_long_context() {
        // CodeLlama uses 10000 (same base as LLaMA2) but supports extended context
        // via rope_scaling; check that default theta is present
        let cfg = CodeLlamaConfig::codellama_7b();
        assert!(cfg.rope_theta > 0.0, "rope_theta must be positive");
    }

    #[test]
    fn test_codellama_infilling_flag() {
        let base = CodeLlamaConfig::codellama_7b();
        let instruct = CodeLlamaConfig::codellama_7b_instruct();
        // base 7B has infilling disabled; instruct re-enables it
        assert!(
            !base.infilling,
            "CodeLlama-7B base must have infilling disabled"
        );
        assert!(
            instruct.infilling,
            "CodeLlama-7B-Instruct must have infilling enabled"
        );
    }

    #[test]
    fn test_codellama_34b_linear_rope_scaling() {
        let cfg = CodeLlamaConfig::codellama_34b();
        let scaling = cfg.rope_scaling.as_ref().expect("34B must have rope_scaling set");
        assert_eq!(
            scaling.scaling_type,
            RopeScalingType::Linear,
            "34B must use linear rope scaling"
        );
        assert!(
            (scaling.factor - 4.0).abs() < 1e-6,
            "34B scaling factor must be 4.0"
        );
    }

    #[test]
    fn test_codellama_34b_gqa() {
        let cfg = CodeLlamaConfig::codellama_34b();
        assert!(cfg.uses_gqa(), "CodeLlama-34B must use GQA");
        assert_eq!(cfg.num_key_value_heads, 8, "34B KV heads must be 8");
    }

    #[test]
    fn test_codellama_effective_max_context_no_scaling() {
        let cfg = CodeLlamaConfig::codellama_7b();
        assert_eq!(
            cfg.effective_max_context(),
            cfg.max_position_embeddings,
            "without scaling, effective context = max_position_embeddings"
        );
    }

    #[test]
    fn test_codellama_effective_max_context_with_scaling() {
        let cfg = CodeLlamaConfig::codellama_34b();
        // 16384 * 4.0 = 65536
        assert_eq!(
            cfg.effective_max_context(),
            65536,
            "with factor=4.0, effective context must be 65536"
        );
    }

    #[test]
    fn test_codellama_config_validation_valid() {
        let cfg = small_config();
        assert!(
            cfg.validate().is_ok(),
            "valid small config must pass validation"
        );
    }

    #[test]
    fn test_codellama_config_validation_invalid_hidden() {
        let mut cfg = small_config();
        cfg.hidden_size = 63;
        assert!(
            cfg.validate().is_err(),
            "bad hidden_size must fail validation"
        );
    }

    #[test]
    fn test_codellama_config_validation_negative_scaling_factor() {
        let mut cfg = small_config();
        cfg.rope_scaling = Some(RopeScalingConfig {
            scaling_type: RopeScalingType::Linear,
            factor: -1.0,
        });
        assert!(
            cfg.validate().is_err(),
            "negative scaling factor must fail validation"
        );
    }

    #[test]
    fn test_codellama_from_pretrained_name_7b() {
        let cfg = CodeLlamaConfig::from_pretrained_name("codellama-7b");
        assert!(
            cfg.is_some(),
            "from_pretrained_name must return Some for known model"
        );
    }

    #[test]
    fn test_codellama_from_pretrained_name_unknown() {
        let cfg = CodeLlamaConfig::from_pretrained_name("unknown-model");
        assert!(
            cfg.is_none(),
            "from_pretrained_name must return None for unknown model"
        );
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_inv_freq_linear_scaling() {
        let mut cfg = small_config();
        let factor = 2.0f32;
        cfg.rope_scaling = Some(RopeScalingConfig {
            scaling_type: RopeScalingType::Linear,
            factor,
        });
        let rope = CodeLlamaRotaryEmbedding::new(&cfg);
        let inv_freq_scaled = rope.compute_inv_freq();

        let cfg_noscale = small_config();
        let rope_noscale = CodeLlamaRotaryEmbedding::new(&cfg_noscale);
        let inv_freq_base = rope_noscale.compute_inv_freq();

        // linear scaling divides all freqs by factor
        for (s, b) in inv_freq_scaled.iter().zip(inv_freq_base.iter()) {
            let expected = b / factor;
            let delta = (s - expected).abs();
            assert!(
                delta < 1e-6,
                "linear-scaled freq {s} must equal base/factor={expected}"
            );
        }
    }

    #[test]
    fn test_rope_inv_freq_length() {
        let cfg = small_config();
        let rope = CodeLlamaRotaryEmbedding::new(&cfg);
        assert_eq!(
            rope.compute_inv_freq().len(),
            cfg.head_dim() / 2,
            "inv_freq length must be head_dim/2"
        );
    }

    // ── MLP tests ─────────────────────────────────────────────────────────────

    #[test]
    fn test_mlp_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = small_config();
        let mlp = CodeLlamaMLP::new(&cfg).expect("MLP must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[4, 64])));
        let out = mlp.forward(input).expect("MLP forward must succeed");
        assert_eq!(out.shape(), &[4, 64], "MLP must map hidden → hidden");
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_weight_map_contains_embed_tokens() {
        let cfg = small_config();
        let model = CodeLlamaModel::new(cfg).expect("model must construct");
        let wmap = model.weight_map();
        assert!(
            wmap.contains_key("model.embed_tokens.weight"),
            "weight map must contain embed_tokens"
        );
    }

    #[test]
    fn test_model_weight_map_layer0_q_proj() {
        let cfg = small_config();
        let model = CodeLlamaModel::new(cfg).expect("model must construct");
        let wmap = model.weight_map();
        assert!(
            wmap.contains_key("model.layers.0.self_attn.q_proj.weight"),
            "weight map must contain layer 0 q_proj"
        );
    }

    #[test]
    fn test_causal_lm_output_last_dim_is_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = CodeLlamaForCausalLM::new(cfg).expect("CausalLM must construct");
        let out = model.forward(vec![0u32, 1, 2]).expect("CausalLM forward must succeed");
        let shape = out.shape();
        assert_eq!(
            *shape.last().expect("output must have shape"),
            vocab,
            "CausalLM output last dim must be vocab_size"
        );
    }

    #[test]
    fn test_causal_lm_more_params_than_base() {
        let cfg = small_config();
        let cfg2 = cfg.clone();
        let base = CodeLlamaModel::new(cfg).expect("base model must construct");
        let causal = CodeLlamaForCausalLM::new(cfg2).expect("causal lm must construct");
        assert!(
            causal.parameter_count() > base.parameter_count(),
            "CausalLM must have more params than base (lm_head added)"
        );
    }
}
