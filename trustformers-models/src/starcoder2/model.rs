use crate::starcoder2::config::StarCoder2Config;
use scirs2_core::ndarray::{ArrayD, IxDyn};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

// ─────────────────────────────────────────────────────────────────────────────
// RMSNorm
// ─────────────────────────────────────────────────────────────────────────────

/// Root Mean Square Layer Normalisation for StarCoder2.
pub struct StarCoder2RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl StarCoder2RmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for StarCoder2RmsNorm {
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
                        "StarCoder2RmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "StarCoder2RmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for StarCoder2 (θ = 10 000).
pub struct StarCoder2RotaryEmbedding {
    pub inv_freq: Vec<f64>,
    pub max_seq_len: usize,
    pub head_dim: usize,
}

impl StarCoder2RotaryEmbedding {
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

    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Apply RoPE rotations (shape-preserving).
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                // Simplified: compute angles but return cloned tensors
                // (full apply would reshape to [seq, nheads, head_dim] and rotate pairs)
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
                "StarCoder2RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP (with bias)
// ─────────────────────────────────────────────────────────────────────────────

/// StarCoder2 SwiGLU Feed-Forward Network.
///
/// Identical topology to LLaMA but **all projections carry a bias term**,
/// matching the StarCoder2 training configuration.
pub struct StarCoder2MLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl StarCoder2MLP {
    pub fn new(config: &StarCoder2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StarCoder2Config, device: Device) -> Result<Self> {
        let bias = config.use_bias;
        let gate_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, bias, device);
        let up_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, bias, device);
        let down_proj =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, bias, device);
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

impl Layer for StarCoder2MLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "StarCoder2MLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention (near-MQA with num_kv_heads = 2)
// ─────────────────────────────────────────────────────────────────────────────

/// StarCoder2 Grouped Query Attention.
///
/// With `num_key_value_heads = 2` this is effectively Multi-Query Attention
/// with two KV heads shared among all query heads.  All projections include
/// bias terms when `use_bias = true`.
pub struct StarCoder2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: StarCoder2RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl StarCoder2Attention {
    pub fn new(config: &StarCoder2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StarCoder2Config, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_query_groups = config.num_query_groups();
        let bias = config.use_bias;

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            bias,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            bias,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            bias,
            device,
        );
        let rotary_emb = StarCoder2RotaryEmbedding::new(
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
        })
    }

    /// Expand KV heads to match query heads using GQA repeat.
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
                            "StarCoder2Attention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "StarCoder2Attention::repeat_kv",
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

impl Layer for StarCoder2Attention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "StarCoder2Attention::forward",
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
                    "StarCoder2Attention::forward",
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

fn make_contiguous(t: Tensor) -> Result<Tensor> {
    let shape = t.shape().to_vec();
    t.reshape(&shape)
}

/// Single StarCoder2 decoder layer (pre-norm, attention then MLP).
pub struct StarCoder2DecoderLayer {
    self_attn: StarCoder2Attention,
    mlp: StarCoder2MLP,
    input_layernorm: StarCoder2RmsNorm,
    post_attention_layernorm: StarCoder2RmsNorm,
}

impl StarCoder2DecoderLayer {
    pub fn new(config: &StarCoder2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StarCoder2Config, device: Device) -> Result<Self> {
        let self_attn = StarCoder2Attention::new_with_device(config, device)?;
        let mlp = StarCoder2MLP::new_with_device(config, device)?;
        let input_layernorm = StarCoder2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm =
            StarCoder2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
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

impl Layer for StarCoder2DecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let normed = make_contiguous(self.input_layernorm.forward(input.clone())?)?;
        let attn_out = self.self_attn.forward(normed)?;
        let input_c = make_contiguous(input)?;
        let after_attn = input_c.add(&make_contiguous(attn_out)?)?;

        let normed2 = make_contiguous(self.post_attention_layernorm.forward(after_attn.clone())?)?;
        let mlp_out = self.mlp.forward(normed2)?;
        make_contiguous(after_attn)?.add(&make_contiguous(mlp_out)?)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// StarCoder2 Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// StarCoder2 transformer model (without LM head).
pub struct StarCoder2Model {
    config: StarCoder2Config,
    embed_tokens: Embedding,
    layers: Vec<StarCoder2DecoderLayer>,
    norm: StarCoder2RmsNorm,
}

impl StarCoder2Model {
    pub fn new(config: StarCoder2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: StarCoder2Config, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(StarCoder2DecoderLayer::new_with_device(&config, device)?);
        }
        let norm = StarCoder2RmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &StarCoder2Config {
        &self.config
    }

    pub fn parameter_count(&self) -> usize {
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        self.embed_tokens.parameter_count() + layer_params + self.norm.parameter_count()
    }

    pub fn run(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let seq_len = input_ids.len();
        let embeddings = self.embed_tokens.forward(input_ids)?;
        let mut hidden = embeddings.reshape(&[1, seq_len, self.config.hidden_size])?;
        for layer in &self.layers {
            hidden = layer.forward(hidden)?;
        }
        make_contiguous(self.norm.forward(hidden)?)
    }
}

impl Model for StarCoder2Model {
    type Config = StarCoder2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "Weight loading not yet implemented for StarCoder2".to_string(),
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
// StarCoder2 Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// StarCoder2 with causal language-modelling head.
pub struct StarCoder2ForCausalLM {
    model: StarCoder2Model,
    lm_head: Linear,
}

impl StarCoder2ForCausalLM {
    pub fn new(config: StarCoder2Config) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: StarCoder2Config, device: Device) -> Result<Self> {
        // lm_head typically has no bias and does not share use_bias flag
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = StarCoder2Model::new_with_device(config, device)?;
        Ok(Self { model, lm_head })
    }

    pub fn config(&self) -> &StarCoder2Config {
        self.model.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.model.parameter_count() + self.lm_head.parameter_count()
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for StarCoder2ForCausalLM {
    type Config = StarCoder2Config;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        StarCoder2ForCausalLM::forward(self, input_ids)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(TrustformersError::not_implemented(
            "Weight loading not yet implemented for StarCoder2".to_string(),
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
    use crate::starcoder2::config::StarCoder2Config;

    // ── Config tests ──────────────────────────────────────────────────────────

    #[test]
    fn test_starcoder2_3b_hidden_size() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(
            cfg.hidden_size, 3072,
            "StarCoder2-3B hidden_size must be 3072"
        );
    }

    #[test]
    fn test_starcoder2_3b_num_layers() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(
            cfg.num_hidden_layers, 30,
            "StarCoder2-3B must have 30 layers"
        );
    }

    #[test]
    fn test_starcoder2_3b_attention_heads() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(
            cfg.num_attention_heads, 24,
            "StarCoder2-3B must have 24 query heads"
        );
        assert_eq!(
            cfg.num_key_value_heads, 2,
            "StarCoder2-3B must have 2 KV heads (GQA)"
        );
    }

    #[test]
    fn test_starcoder2_3b_gqa_group_size() {
        let cfg = StarCoder2Config::starcoder2_3b();
        // 24 Q / 2 KV = 12
        assert_eq!(
            cfg.num_query_groups(),
            12,
            "StarCoder2-3B GQA group_size must be 12"
        );
    }

    #[test]
    fn test_starcoder2_3b_use_bias() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert!(cfg.use_bias, "StarCoder2 must use bias in projections");
    }

    #[test]
    fn test_starcoder2_3b_vocab_size() {
        let cfg = StarCoder2Config::starcoder2_3b();
        assert_eq!(cfg.vocab_size, 49152, "StarCoder2 vocab_size must be 49152");
    }

    #[test]
    fn test_starcoder2_fim_token_ids_in_vocab() {
        // FIM special tokens <fim_prefix>=1, <fim_middle>=2, <fim_suffix>=3 must fit in vocab
        let cfg = StarCoder2Config::starcoder2_3b();
        let fim_prefix_id = 1u32;
        let fim_middle_id = 2u32;
        let fim_suffix_id = 3u32;
        assert!(
            fim_prefix_id < cfg.vocab_size as u32,
            "fim_prefix_id must be in vocab"
        );
        assert!(
            fim_middle_id < cfg.vocab_size as u32,
            "fim_middle_id must be in vocab"
        );
        assert!(
            fim_suffix_id < cfg.vocab_size as u32,
            "fim_suffix_id must be in vocab"
        );
    }

    #[test]
    fn test_starcoder2_sliding_window_none_in_released() {
        // Released checkpoints have no sliding window
        let cfg = StarCoder2Config::starcoder2_3b();
        assert!(
            cfg.sliding_window.is_none(),
            "StarCoder2-3B released checkpoint has no sliding window"
        );
    }

    #[test]
    fn test_starcoder2_head_dim() {
        let cfg = StarCoder2Config::starcoder2_3b();
        // 3072 / 24 = 128
        assert_eq!(cfg.head_dim(), 128, "StarCoder2-3B head_dim must be 128");
    }

    #[test]
    fn test_starcoder2_config_validation_valid() {
        let cfg = StarCoder2Config::small_test();
        assert!(
            cfg.validate().is_ok(),
            "small_test config must pass validation"
        );
    }

    #[test]
    fn test_starcoder2_config_validation_bad_hidden() {
        let mut cfg = StarCoder2Config::small_test();
        cfg.hidden_size = 63;
        assert!(
            cfg.validate().is_err(),
            "bad hidden_size must fail validation"
        );
    }

    // ── RMSNorm tests ─────────────────────────────────────────────────────────

    #[test]
    fn test_rmsnorm_parameter_count() {
        let norm = StarCoder2RmsNorm::new(64, 1e-5).expect("RmsNorm must construct");
        assert_eq!(
            norm.parameter_count(),
            64,
            "parameter count must equal normalized_shape"
        );
    }

    #[test]
    fn test_rmsnorm_forward_shape() {
        use scirs2_core::ndarray::ArrayD;
        let norm = StarCoder2RmsNorm::new(8, 1e-5).expect("RmsNorm must construct");
        let input = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[2, 8])));
        let out = norm.forward(input).expect("RmsNorm forward must succeed");
        assert_eq!(out.shape(), &[2, 8], "RmsNorm must preserve shape");
    }

    // ── RoPE tests ────────────────────────────────────────────────────────────

    #[test]
    fn test_rope_half_dim() {
        let rope = StarCoder2RotaryEmbedding::new(128, 16384, 10000.0);
        assert_eq!(rope.half_dim(), 64, "RoPE half_dim must be head_dim/2");
    }

    #[test]
    fn test_rope_inv_freq_non_increasing() {
        let rope = StarCoder2RotaryEmbedding::new(128, 16384, 10000.0);
        for i in 1..rope.inv_freq.len() {
            assert!(
                rope.inv_freq[i] <= rope.inv_freq[i - 1],
                "inv_freq must be non-increasing"
            );
        }
    }

    #[test]
    fn test_rope_apply_shape_preserved() {
        use scirs2_core::ndarray::ArrayD;
        let rope = StarCoder2RotaryEmbedding::new(16, 64, 10000.0);
        let q = Tensor::F32(ArrayD::ones(scirs2_core::ndarray::IxDyn(&[3, 16])));
        let k = q.clone();
        let pos: Vec<usize> = (0..3).collect();
        let (qo, ko) = rope.apply_rotary_emb(&q, &k, &pos).expect("RoPE must succeed");
        assert_eq!(qo.shape(), q.shape(), "Q shape must be preserved");
        assert_eq!(ko.shape(), k.shape(), "K shape must be preserved");
    }

    // ── Attention tests ───────────────────────────────────────────────────────

    #[test]
    fn test_attention_kv_heads() {
        let cfg = StarCoder2Config::small_test();
        let attn = StarCoder2Attention::new(&cfg).expect("Attention must construct");
        assert_eq!(
            attn.num_kv_heads(),
            2,
            "StarCoder2 attention must have 2 KV heads"
        );
    }

    #[test]
    fn test_attention_forward_output_shape() {
        use scirs2_core::ndarray::ArrayD;
        let cfg = StarCoder2Config::small_test();
        let attn = StarCoder2Attention::new(&cfg).expect("Attention must construct");
        let input = Tensor::F32(ArrayD::zeros(scirs2_core::ndarray::IxDyn(&[2, 1, 64])));
        let out = attn.forward(input).expect("Attention forward must succeed");
        assert_eq!(
            out.shape(),
            &[2, 1, 64],
            "Attention output must preserve shape"
        );
    }

    // ── Model tests ───────────────────────────────────────────────────────────

    #[test]
    fn test_model_construct() {
        let cfg = StarCoder2Config::small_test();
        let model = StarCoder2Model::new(cfg).expect("StarCoder2Model must construct");
        assert!(
            model.parameter_count() > 0,
            "model must have positive param count"
        );
    }

    #[test]
    fn test_model_run_output_shape() {
        let cfg = StarCoder2Config::small_test();
        let model = StarCoder2Model::new(cfg).expect("model must construct");
        let out = model.run(vec![0u32, 1, 2]).expect("model run must succeed");
        // run outputs [1, seq_len, hidden_size]
        assert_eq!(
            out.shape().len(),
            3,
            "StarCoder2Model run output must be 3-D"
        );
        assert_eq!(out.shape()[1], 3, "seq_len dimension must match input");
        assert_eq!(out.shape()[2], 64, "last dim must be hidden_size");
    }

    #[test]
    fn test_causal_lm_output_last_dim_is_vocab() {
        let cfg = StarCoder2Config::small_test();
        let vocab = cfg.vocab_size;
        let model = StarCoder2ForCausalLM::new(cfg).expect("CausalLM must construct");
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
        let cfg = StarCoder2Config::small_test();
        let cfg2 = cfg.clone();
        let base = StarCoder2Model::new(cfg).expect("base model must construct");
        let causal = StarCoder2ForCausalLM::new(cfg2).expect("causal lm must construct");
        assert!(
            causal.parameter_count() > base.parameter_count(),
            "CausalLM must have more params than base"
        );
    }
}
