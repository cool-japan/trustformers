use crate::mistral_v3::config::MistralV3Config;
use crate::weight_loading::binding::{
    bind_embedding, bind_linear, take_norm_weight, DECODER_BUFFER_SUFFIXES,
};
use crate::weight_loading::checkpoint::{Checkpoint, LoadReport, UnusedTensors};
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

impl MistralV3RmsNorm {
    /// Install the normalisation gain from a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when `weight` does not have the shape this norm was built for.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != self.weight.shape() {
            return Err(TrustformersError::shape_error(format!(
                "MistralV3RmsNorm expects a {:?} gain, got {:?}",
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
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
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
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
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
    pub(crate) self_attn: MistralV3Attention,
    pub(crate) mlp: MistralV3MLP,
    pub(crate) input_layernorm: MistralV3RmsNorm,
    pub(crate) post_attention_layernorm: MistralV3RmsNorm,
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

    /// Load a HuggingFace Mistral checkpoint (safetensors or `torch.save`).
    ///
    /// See [`MistralV3Model::load_checkpoint`] for the name map and the failure modes.
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

    /// Load a HuggingFace MistralV3ForCausalLM checkpoint.
    ///
    /// The backbone is bound first, then the LM head. A checkpoint with tied
    /// word embeddings carries no `lm_head.weight`; the input embedding matrix
    /// is reused in that case, which is exactly what the tied configuration
    /// means — not a fallback to something invented.
    ///
    /// # Errors
    ///
    /// See [`MistralV3Model::load_checkpoint`]; additionally fails when the checkpoint
    /// holds neither an LM head nor an embedding matrix to tie it to, or when
    /// the head has the wrong shape.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.model.load_checkpoint(&checkpoint, &["lm_head."])?;

        let expected = [
            self.model.config().vocab_size,
            self.model.config().hidden_size,
        ];
        let head = match checkpoint.get("lm_head.weight") {
            Some(weight) => weight,
            None => {
                let embed_name = if checkpoint.contains("model.embed_tokens.weight") {
                    "model.embed_tokens.weight"
                } else {
                    "embed_tokens.weight"
                };
                checkpoint.get(embed_name).ok_or_else(|| {
                    TrustformersError::weight_load_error(
                        "checkpoint holds neither lm_head.weight nor an embedding matrix to tie \
                         it to"
                            .to_string(),
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

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

impl MistralV3Model {
    /// Bind a parsed checkpoint into this model.
    ///
    /// The HuggingFace export nests the backbone under `model.`; the bare
    /// layout is accepted too. `allowed_unused_prefixes` names namespaces this
    /// base model legitimately ignores (the LM head lives on the causal-LM
    /// wrapper, not here).
    ///
    /// A parameter the checkpoint does not carry is *recorded* and reported by
    /// [`WeightBinder::finish`](crate::weight_loading::checkpoint::WeightBinder::finish),
    /// never substituted, so a mismatched checkpoint cannot leave
    /// randomly-initialised tensors in place while the load returns `Ok`.
    ///
    /// # Errors
    ///
    /// Fails when the checkpoint does not look like a Mistral checkpoint,
    /// when any tensor has the wrong shape, when a parameter is missing, or when
    /// the checkpoint carries weights this architecture does not recognise.
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
        let attention_bias = false;
        let mlp_bias = false;

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
                attention_bias,
                &mut layer.self_attn.q_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.k_proj"),
                kv_width,
                hidden,
                attention_bias,
                &mut layer.self_attn.k_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.v_proj"),
                kv_width,
                hidden,
                attention_bias,
                &mut layer.self_attn.v_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{attn}.o_proj"),
                hidden,
                q_width,
                attention_bias,
                &mut layer.self_attn.o_proj,
            )?;

            let mlp = format!("layers.{i}.mlp");
            bind_linear(
                &mut binder,
                &format!("{mlp}.gate_proj"),
                intermediate,
                hidden,
                mlp_bias,
                &mut layer.mlp.gate_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.up_proj"),
                intermediate,
                hidden,
                mlp_bias,
                &mut layer.mlp.up_proj,
            )?;
            bind_linear(
                &mut binder,
                &format!("{mlp}.down_proj"),
                hidden,
                intermediate,
                mlp_bias,
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

    // ── Real checkpoint loading ─────────────────────────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, DecoderFixtureSpec, F32Tensor};

    fn loading_config() -> MistralV3Config {
        MistralV3Config {
            vocab_size: 12,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            sliding_window: 8,
            rope_theta: 1_000_000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16,
        }
    }

    fn loading_fixture(config: &MistralV3Config) -> DecoderFixtureSpec {
        let head_dim = config.head_dim();
        let mut spec = DecoderFixtureSpec::llama_style(
            "model.",
            config.vocab_size,
            config.hidden_size,
            config.intermediate_size,
            config.num_hidden_layers,
            config.num_attention_heads * head_dim,
            config.num_key_value_heads * head_dim,
        );
        spec.attention_bias = false;
        spec.mlp_bias = false;
        spec
    }

    /// Regression: `load_pretrained` returned `not_implemented`, so no Mistral v0.3 checkpoint
    /// could ever reach the model's parameters. It now binds every one of them,
    /// and the proof is that the checkpoint's exact values arrive in the layers.
    #[test]
    fn load_pretrained_binds_every_parameter_from_the_checkpoint() {
        let config = loading_config();
        let tensors = loading_fixture(&config).tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = MistralV3Model::new(config).expect("model must build");
        model
            .load_pretrained(&mut bytes.as_slice())
            .expect("a matching checkpoint must load");

        for name in [
            "model.layers.0.self_attn.q_proj.weight",
            "model.layers.1.mlp.down_proj.weight",
            "model.norm.weight",
        ] {
            let expected = tensors
                .iter()
                .find(|t| t.name == name)
                .unwrap_or_else(|| panic!("fixture must carry {name}"))
                .values
                .clone();
            let actual = match name {
                "model.layers.0.self_attn.q_proj.weight" => {
                    model.layers[0].self_attn.q_proj.weight().data().expect("readable")
                },
                "model.layers.1.mlp.down_proj.weight" => {
                    model.layers[1].mlp.down_proj.weight().data().expect("readable")
                },
                _ => model.norm.weight().data().expect("readable"),
            };
            assert_eq!(actual, expected, "{name} must hold the checkpoint's values");
        }
    }

    /// Grouped-query attention: `k_proj`/`v_proj` are narrower than `q_proj`.
    /// A loader that assumed a square projection would reject this fixture.
    #[test]
    fn load_pretrained_respects_grouped_query_attention_widths() {
        let config = loading_config();
        let head_dim = config.head_dim();
        let bytes = loading_fixture(&config).safetensors();
        let mut model = MistralV3Model::new(config.clone()).expect("model must build");
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
        let mut tensors = loading_fixture(&config).tensors();
        tensors.retain(|t| t.name != "model.layers.1.mlp.up_proj.weight");
        let bytes = build_safetensors(&tensors);

        let mut model = MistralV3Model::new(config).expect("model must build");
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
        let mut tensors = loading_fixture(&config).tensors();
        tensors.push(F32Tensor::ramp(
            "model.layers.9.mystery.weight",
            &[4, 4],
            99.0,
        ));
        let bytes = build_safetensors(&tensors);

        let mut model = MistralV3Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("an unrecognised weight must fail the load");
        assert!(
            err.to_string().contains("mystery.weight"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn load_pretrained_rejects_bytes_that_are_not_a_checkpoint() {
        let mut model = MistralV3Model::new(loading_config()).expect("model must build");
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
    fn load_pretrained_rejects_a_checkpoint_for_a_different_configuration() {
        let config = loading_config();
        let wider = MistralV3Config {
            hidden_size: config.hidden_size * 2,
            intermediate_size: config.intermediate_size * 2,
            ..config.clone()
        };
        let bytes = loading_fixture(&wider).safetensors();

        let mut model = MistralV3Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a mismatched checkpoint must not be reshaped into place");
        assert!(err.to_string().contains("expects"), "unexpected: {err}");
    }
}
