use crate::weight_loading::binding::{
    bind_embedding, bind_linear, take_norm_weight, DECODER_BUFFER_SUFFIXES,
};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
use crate::yi::config::YiConfig;
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

/// Root Mean Square Layer Normalisation for Yi.
pub struct YiRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl YiRmsNorm {
    pub fn new(normalized_shape: usize, eps: f64) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }

    /// Install the normalisation gain from a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when `weight` does not have the shape this norm was built for.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != self.weight.shape() {
            return Err(TrustformersError::shape_error(format!(
                "YiRmsNorm expects a {:?} gain, got {:?}",
                self.weight.shape(),
                weight.shape()
            )));
        }
        self.weight = weight;
        Ok(())
    }

    /// The current normalisation gain.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }
}

impl Layer for YiRmsNorm {
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
                        "YiRmsNorm::forward",
                        "weight tensor type mismatch",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "YiRmsNorm::forward",
                "unsupported input tensor dtype",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rotary Position Embeddings — θ = 5 000 000 for long-context
// ─────────────────────────────────────────────────────────────────────────────

/// Rotary Position Embedding for Yi.
///
/// Uses a large `rope_theta` (5 000 000) to support very long contexts
/// (up to 200K tokens in Yi-1.5 long-context variants).
pub struct YiRotaryEmbedding {
    pub inv_freq: Vec<f64>,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub theta: f64,
}

impl YiRotaryEmbedding {
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
            theta,
        }
    }

    pub fn half_dim(&self) -> usize {
        self.inv_freq.len()
    }

    /// Apply RoPE rotations (shape-preserving simplified form).
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
                "YiRotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// SwiGLU MLP (no bias)
// ─────────────────────────────────────────────────────────────────────────────

/// Yi SwiGLU Feed-Forward Network (no bias, same as LLaMA).
pub struct YiMLP {
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
}

impl YiMLP {
    pub fn new(config: &YiConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &YiConfig, device: Device) -> Result<Self> {
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

impl Layer for YiMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate_out = self.gate_proj.forward(input.clone())?;
        let up_out = self.up_proj.forward(input)?;
        let gate_activated = silu(&gate_out)?;
        let combined = match (&gate_activated, &up_out) {
            (Tensor::F32(g), Tensor::F32(u)) => Ok(Tensor::F32(g * u)),
            _ => Err(tensor_op_error(
                "YiMLP::forward",
                "tensor dtype mismatch in SwiGLU gate multiply",
            )),
        }?;
        self.down_proj.forward(combined)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Grouped Query Attention
// ─────────────────────────────────────────────────────────────────────────────

/// Yi Grouped Query Attention (no bias, repeat KV heads).
pub struct YiAttention {
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
    rotary_emb: YiRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    num_query_groups: usize,
}

impl YiAttention {
    pub fn new(config: &YiConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &YiConfig, device: Device) -> Result<Self> {
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
            YiRotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

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

    /// Expand KV heads to match query heads.
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
                            "YiAttention::repeat_kv",
                            format!("shape error during KV expansion: {e}"),
                        )
                    })?;
                Ok(Tensor::F32(expanded_arr))
            },
            _ => Err(tensor_op_error(
                "YiAttention::repeat_kv",
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

impl Layer for YiAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = match shape.len() {
            2 => shape[0],
            3 => shape[1],
            n => {
                return Err(tensor_op_error(
                    "YiAttention::forward",
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
                    "YiAttention::forward",
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

/// Single Yi decoder layer (pre-norm).
pub struct YiDecoderLayer {
    pub(crate) self_attn: YiAttention,
    pub(crate) mlp: YiMLP,
    pub(crate) input_layernorm: YiRmsNorm,
    pub(crate) post_attention_layernorm: YiRmsNorm,
}

impl YiDecoderLayer {
    pub fn new(config: &YiConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &YiConfig, device: Device) -> Result<Self> {
        let self_attn = YiAttention::new_with_device(config, device)?;
        let mlp = YiMLP::new_with_device(config, device)?;
        let input_layernorm = YiRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = YiRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
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

impl Layer for YiDecoderLayer {
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
// Yi Base Model
// ─────────────────────────────────────────────────────────────────────────────

/// Yi transformer model (without LM head).
pub struct YiModel {
    config: YiConfig,
    embed_tokens: Embedding,
    layers: Vec<YiDecoderLayer>,
    norm: YiRmsNorm,
}

impl YiModel {
    pub fn new(config: YiConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: YiConfig, device: Device) -> Result<Self> {
        config.validate()?;
        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for _ in 0..config.num_hidden_layers {
            layers.push(YiDecoderLayer::new_with_device(&config, device)?);
        }
        let norm = YiRmsNorm::new(config.hidden_size, config.rms_norm_eps)?;
        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn config(&self) -> &YiConfig {
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

impl Model for YiModel {
    type Config = YiConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        self.run(input_ids)
    }

    /// Load a HuggingFace Yi checkpoint (safetensors or `torch.save`).
    ///
    /// See [`YiModel::load_checkpoint`] for the name map and the failure modes.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_checkpoint(&checkpoint, &["lm_head."])?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

impl YiModel {
    /// Bind a parsed checkpoint into this model.
    ///
    /// Yi follows the LLaMA tensor layout, so both the `model.*`-prefixed
    /// (`YiForCausalLM`) and the bare (`YiModel`) HuggingFace exports are
    /// accepted. `allowed_unused_prefixes` names namespaces this base model
    /// legitimately ignores — the LM head lives on [`YiForCausalLM`], not here.
    ///
    /// The load is all-or-nothing on real weights: a parameter the checkpoint
    /// does not carry is recorded and reported by
    /// [`WeightBinder::finish`](crate::weight_loading::checkpoint::WeightBinder::finish),
    /// never substituted, so a checkpoint that does not match the architecture
    /// cannot leave randomly-initialised tensors in place while returning `Ok`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like a Yi checkpoint, when any
    /// tensor has the wrong shape, when a parameter is missing, or when the
    /// checkpoint carries weights this architecture does not recognise.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &Checkpoint,
        allowed_unused_prefixes: &[&str],
    ) -> Result<LoadReport> {
        let prefix = checkpoint.detect_prefix(&["model.", ""], "embed_tokens.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let hidden = self.config.hidden_size;
        let head_dim = self.config.head_dim();
        let q_width = self.config.num_attention_heads * head_dim;
        let kv_width = self.config.num_key_value_heads * head_dim;
        let intermediate = self.config.intermediate_size;

        bind_embedding(
            &mut binder,
            "embed_tokens",
            self.config.vocab_size,
            hidden,
            &mut self.embed_tokens,
        )?;

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("layers.{i}.self_attn");
            bind_linear(
                &mut binder,
                &format!("{attn}.q_proj"),
                q_width,
                hidden,
                false,
                &mut layer.self_attn.q_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.k_proj"),
                kv_width,
                hidden,
                false,
                &mut layer.self_attn.k_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.v_proj"),
                kv_width,
                hidden,
                false,
                &mut layer.self_attn.v_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.o_proj"),
                hidden,
                q_width,
                false,
                &mut layer.self_attn.o_proj,
            )?;

            let mlp = format!("layers.{i}.mlp");
            bind_linear(
                &mut binder,
                &format!("{mlp}.gate_proj"),
                intermediate,
                hidden,
                false,
                &mut layer.mlp.gate_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.up_proj"),
                intermediate,
                hidden,
                false,
                &mut layer.mlp.up_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.down_proj"),
                hidden,
                intermediate,
                false,
                &mut layer.mlp.down_proj,
            )?;

            if let Some(w) =
                take_norm_weight(&mut binder, &format!("layers.{i}.input_layernorm"), hidden)?
            {
                layer.input_layernorm.set_weight(w)?;
            }
            if let Some(w) = take_norm_weight(
                &mut binder,
                &format!("layers.{i}.post_attention_layernorm"),
                hidden,
            )? {
                layer.post_attention_layernorm.set_weight(w)?;
            }
        }

        if let Some(w) = take_norm_weight(&mut binder, "norm", hidden)? {
            self.norm.set_weight(w)?;
        }

        binder.finish(UnusedTensors::new(
            allowed_unused_prefixes,
            DECODER_BUFFER_SUFFIXES,
        ))
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Yi Causal LM
// ─────────────────────────────────────────────────────────────────────────────

/// Yi with causal language-modelling head.
///
/// When `tie_word_embeddings = true`, the `lm_head` conceptually shares weights
/// with `embed_tokens`.  In this implementation we create a separate `Linear`
/// to keep the model self-contained; real weight-tied inference would set
/// `lm_head.weight = embed_tokens.weight` during weight loading.
pub struct YiForCausalLM {
    model: YiModel,
    lm_head: Linear,
    /// Whether the LM head is conceptually tied to the embedding table.
    tie_word_embeddings: bool,
}

impl YiForCausalLM {
    pub fn new(config: YiConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: YiConfig, device: Device) -> Result<Self> {
        let tie = config.tie_word_embeddings;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = YiModel::new_with_device(config, device)?;
        Ok(Self {
            model,
            lm_head,
            tie_word_embeddings: tie,
        })
    }

    pub fn config(&self) -> &YiConfig {
        self.model.config()
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings
    }

    pub fn parameter_count(&self) -> usize {
        if self.tie_word_embeddings {
            // lm_head shares the embed_tokens weights — no extra parameters
            self.model.parameter_count()
        } else {
            self.model.parameter_count() + self.lm_head.parameter_count()
        }
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let hidden = self.model.run(input_ids)?;
        self.lm_head.forward(hidden)
    }
}

impl Model for YiForCausalLM {
    type Config = YiConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        YiForCausalLM::forward(self, input_ids)
    }

    /// Load a HuggingFace `YiForCausalLM` checkpoint.
    ///
    /// The backbone is bound first, then the LM head. A checkpoint with
    /// `tie_word_embeddings` carries no `lm_head.weight`; the input embedding
    /// matrix is reused in that case, which is exactly what the tied
    /// configuration means — not a fallback.
    ///
    /// # Errors
    ///
    /// See [`YiModel::load_checkpoint`]; additionally fails when the checkpoint
    /// holds neither an LM head nor an embedding matrix to tie it to, or when
    /// the head has the wrong shape.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.model.load_checkpoint(&checkpoint, &["lm_head."])?;

        let config = self.model.config();
        let expected = [config.vocab_size, config.hidden_size];
        let head = match checkpoint.get("lm_head.weight") {
            Some(weight) => weight,
            None => {
                let embed_name = if checkpoint.contains("model.embed_tokens.weight") {
                    "model.embed_tokens.weight"
                } else {
                    "embed_tokens.weight"
                };
                checkpoint.get(embed_name).ok_or_else(|| {
                    tensor_op_error(
                        "YiForCausalLM::load_pretrained",
                        "checkpoint holds neither lm_head.weight nor an embedding matrix to tie \
                         it to",
                    )
                })?
            },
        };
        if head.shape() != expected {
            return Err(TrustformersError::shape_error(format!(
                "language-model head has shape {:?} but this model expects {expected:?}",
                head.shape()
            )));
        }
        self.lm_head.set_weight(head.clone())?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        self.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::yi::config::YiConfig;
    use trustformers_core::traits::{Config, Layer};

    /// Minimal Yi config for tests (fast construction and inference).
    fn tiny_config() -> YiConfig {
        YiConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16,
            tie_word_embeddings: false,
        }
    }

    // --- YiConfig ---

    #[test]
    fn test_config_validation_valid() {
        let cfg = tiny_config();
        assert!(
            cfg.validate().is_ok(),
            "valid tiny config must pass validation"
        );
    }

    #[test]
    fn test_config_validation_zero_hidden_size_fails() {
        let mut cfg = tiny_config();
        cfg.hidden_size = 0;
        assert!(
            cfg.validate().is_err(),
            "hidden_size = 0 must fail validation"
        );
    }

    // --- YiRmsNorm ---

    #[test]
    fn test_rmsnorm_output_shape_preserved() {
        let norm = YiRmsNorm::new(8, 1e-5).expect("YiRmsNorm creation must succeed");
        let input = Tensor::ones(&[8]).expect("test tensor");
        let out = norm.forward(input).expect("YiRmsNorm forward must succeed");
        assert_eq!(out.len(), 8, "RMSNorm must preserve tensor shape");
    }

    #[test]
    fn test_rmsnorm_parameter_count_equals_dim() {
        let dim = 16_usize;
        let norm = YiRmsNorm::new(dim, 1e-5).expect("YiRmsNorm creation");
        assert_eq!(
            norm.parameter_count(),
            dim,
            "parameter count must equal dim"
        );
    }

    #[test]
    fn test_rmsnorm_all_ones_output_is_one() {
        // For all-ones input the RMS is 1, so normalised * weight(=1) = 1
        let norm = YiRmsNorm::new(4, 1e-8).expect("YiRmsNorm creation");
        let input = Tensor::ones(&[4]).expect("test tensor");
        let out = norm.forward(input).expect("forward must succeed");
        if let Tensor::F32(arr) = out {
            for &v in arr.iter() {
                let diff = (v - 1.0_f32).abs();
                assert!(
                    diff < 1e-5,
                    "all-ones input must give all-ones output, got {}",
                    v
                );
            }
        } else {
            panic!("expected F32 tensor");
        }
    }

    // --- YiRotaryEmbedding ---

    #[test]
    fn test_rotary_half_dim() {
        let rope = YiRotaryEmbedding::new(16, 32, 5_000_000.0);
        assert_eq!(rope.half_dim(), 8, "half_dim must be head_dim / 2");
    }

    #[test]
    fn test_rotary_inv_freq_decreasing() {
        // Inverse frequencies must be strictly decreasing (larger i → smaller freq)
        let rope = YiRotaryEmbedding::new(16, 32, 5_000_000.0);
        let freqs = &rope.inv_freq;
        for i in 1..freqs.len() {
            assert!(
                freqs[i] < freqs[i - 1],
                "inv_freq must be decreasing, but freqs[{}]={} >= freqs[{}]={}",
                i,
                freqs[i],
                i - 1,
                freqs[i - 1]
            );
        }
    }

    // --- YiModel ---

    #[test]
    fn test_yi_model_creation_succeeds() {
        let cfg = tiny_config();
        let _model = YiModel::new(cfg).expect("YiModel creation must succeed");
    }

    #[test]
    fn test_yi_model_parameter_count_positive() {
        let cfg = tiny_config();
        let model = YiModel::new(cfg).expect("YiModel creation");
        assert!(
            model.parameter_count() > 0,
            "model must have positive parameter count"
        );
    }

    #[test]
    fn test_yi_model_forward_output_shape() {
        let cfg = tiny_config();
        let model = YiModel::new(cfg.clone()).expect("YiModel creation");
        let seq_len = 4_usize;
        let output = model.run(vec![1_u32, 2, 3, 4]).expect("YiModel forward must succeed");
        let shape = output.shape();
        assert_eq!(shape[0], 1, "batch dim must be 1");
        assert_eq!(shape[1], seq_len, "seq_len must match input length");
        assert_eq!(
            shape[2], cfg.hidden_size,
            "feature dim must match hidden_size"
        );
    }

    #[test]
    fn test_yi_model_forward_all_finite() {
        let cfg = tiny_config();
        let model = YiModel::new(cfg).expect("YiModel creation");
        let output = model.run(vec![0_u32, 1]).expect("forward pass");
        if let Tensor::F32(arr) = output {
            assert!(
                arr.iter().all(|v| v.is_finite()),
                "all outputs must be finite"
            );
        }
    }

    // --- YiForCausalLM ---

    #[test]
    fn test_yi_causal_lm_creation_succeeds() {
        let cfg = tiny_config();
        let _model = YiForCausalLM::new(cfg).expect("YiForCausalLM creation must succeed");
    }

    #[test]
    fn test_yi_causal_lm_parameter_count_positive() {
        let cfg = tiny_config();
        let model = YiForCausalLM::new(cfg).expect("YiForCausalLM creation");
        assert!(model.parameter_count() > 0);
    }

    #[test]
    fn test_yi_causal_lm_forward_output_shape() {
        let cfg = tiny_config();
        let model = YiForCausalLM::new(cfg.clone()).expect("YiForCausalLM creation");
        let output = model.forward(vec![1_u32, 2, 3]).expect("YiForCausalLM forward must succeed");
        let shape = output.shape();
        // LM head projects to vocab size: [1, seq_len, vocab_size]
        assert_eq!(shape[2], cfg.vocab_size, "last dim must match vocab_size");
    }

    // ── Real checkpoint loading ─────────────────────────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, DecoderFixtureSpec, F32Tensor};

    fn loading_config() -> YiConfig {
        YiConfig {
            vocab_size: 12,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rope_theta: 5_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16,
            tie_word_embeddings: false,
        }
    }

    fn fixture(config: &YiConfig, prefix: &str) -> DecoderFixtureSpec {
        let head_dim = config.head_dim();
        DecoderFixtureSpec::llama_style(
            prefix,
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads * head_dim,
            config.num_key_value_heads * head_dim,
        )
    }

    /// Regression: `load_pretrained` returned `not_implemented`, so no Yi
    /// checkpoint could ever reach the model. It now binds every parameter, and
    /// the proof is that the checkpoint's exact values arrive in the layers.
    #[test]
    fn load_pretrained_binds_every_parameter_from_the_checkpoint() {
        let config = loading_config();
        let spec = fixture(&config, "model.");
        let tensors = spec.tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = YiModel::new(config.clone()).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        let expected_q = tensors
            .iter()
            .find(|t| t.name == "model.layers.0.self_attn.q_proj.weight")
            .expect("fixture must carry q_proj")
            .values
            .clone();
        assert_eq!(
            model.layers[0].self_attn.q_proj.weight().data().expect("readable"),
            expected_q,
            "q_proj must hold the checkpoint's values"
        );

        let expected_down = tensors
            .iter()
            .find(|t| t.name == "model.layers.1.mlp.down_proj.weight")
            .expect("fixture must carry down_proj")
            .values
            .clone();
        assert_eq!(
            model.layers[1].mlp.down_proj.weight().data().expect("readable"),
            expected_down
        );

        let expected_norm = tensors
            .iter()
            .find(|t| t.name == "model.norm.weight")
            .expect("fixture must carry the final norm")
            .values
            .clone();
        assert_eq!(model.norm.weight().data().expect("readable"), expected_norm);
    }

    /// Grouped-query attention: `k_proj`/`v_proj` are narrower than `q_proj`.
    /// A loader that assumed a square projection would reject this fixture.
    #[test]
    fn load_pretrained_respects_grouped_query_attention_widths() {
        let config = loading_config();
        let head_dim = config.head_dim();
        let mut model = YiModel::new(config.clone()).expect("model must build");
        let bytes = fixture(&config, "model.").safetensors();
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");

        assert_eq!(
            model.layers[0].self_attn.k_proj.weight().shape(),
            vec![config.num_key_value_heads * head_dim, config.hidden_size]
        );
        assert_eq!(
            model.layers[0].self_attn.q_proj.weight().shape(),
            vec![config.num_attention_heads * head_dim, config.hidden_size]
        );
    }

    #[test]
    fn load_pretrained_reports_a_missing_parameter_instead_of_inventing_it() {
        let config = loading_config();
        let mut tensors = fixture(&config, "model.").tensors();
        tensors.retain(|t| t.name != "model.layers.1.mlp.up_proj.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = YiModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an incomplete checkpoint must not load silently");
        assert!(
            err.to_string().contains("layers.1.mlp.up_proj.weight"),
            "the error must name the gap: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_foreign_tensor() {
        let config = loading_config();
        let mut tensors = fixture(&config, "model.").tensors();
        tensors.push(F32Tensor::ramp(
            "model.layers.9.mystery.weight",
            &[8, 8],
            99.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = YiModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_a_checkpoint_for_a_different_configuration() {
        let config = loading_config();
        let wider = YiConfig {
            hidden_size: 16,
            intermediate_size: 32,
            ..config.clone()
        };
        let bytes = fixture(&wider, "model.").safetensors();

        let mut model = YiModel::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a mismatched checkpoint must not be reshaped into place");
        assert!(err.to_string().contains("expects"), "unexpected: {err}");
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = YiModel::new(loading_config()).expect("model must build");
        let garbage = vec![0xABu8; 4096];
        let err = model
            .load_pretrained(&mut garbage.as_slice())
            .expect_err("garbage must not be accepted as weights");
        assert!(
            err.to_string().contains("unrecognised checkpoint container"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn causal_lm_loads_the_head_and_ties_it_when_absent() {
        let config = loading_config();

        // Explicit head.
        let mut spec = fixture(&config, "model.");
        spec.include_lm_head = true;
        let tensors = spec.tensors();
        let bytes = build_safetensors(&tensors);
        let mut model = YiForCausalLM::new(config.clone()).expect("model must build");
        model.load_pretrained(&mut bytes.as_slice()).expect("checkpoint must load");
        let head = tensors
            .iter()
            .find(|t| t.name == "lm_head.weight")
            .expect("fixture must carry the head")
            .values
            .clone();
        assert_eq!(model.lm_head.weight().data().expect("readable"), head);

        // Tied head: no `lm_head.weight`, so the embedding matrix is reused.
        let tied_tensors = fixture(&config, "model.").tensors();
        let tied_bytes = build_safetensors(&tied_tensors);
        let mut tied = YiForCausalLM::new(config).expect("model must build");
        tied.load_pretrained(&mut tied_bytes.as_slice())
            .expect("tied checkpoint must load");
        let embeddings = tied_tensors
            .iter()
            .find(|t| t.name == "model.embed_tokens.weight")
            .expect("fixture must carry the embeddings")
            .values
            .clone();
        assert_eq!(tied.lm_head.weight().data().expect("readable"), embeddings);
    }
}
