use crate::starcoder2::config::StarCoder2Config;
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

impl StarCoder2RmsNorm {
    /// Install the normalisation gain from a checkpoint.
    ///
    /// # Errors
    ///
    /// Fails when `weight` does not have the shape this norm was built for.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.shape() != self.weight.shape() {
            return Err(TrustformersError::shape_error(format!(
                "StarCoder2RmsNorm expects a {:?} gain, got {:?}",
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

    /// Apply rotary position embeddings to `q` and `k` (shape-preserving).
    ///
    /// Each input is `[seq, n_heads * head_dim]` (the number of heads is inferred
    /// from the width, so the same routine serves the query and the narrower
    /// key/value projections). Within every head the `rotate_half` convention is
    /// used: dimension `i` and `i + head_dim/2` form a rotation pair driven by the
    /// angle `position · inv_freq[i]`, matching the reference StarCoder2/LLaMA RoPE:
    ///
    /// ```text
    ///   out[i]          = x[i]·cos − x[i+half]·sin
    ///   out[i + half]   = x[i+half]·cos + x[i]·sin
    /// ```
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => Ok((
                Tensor::F32(self.rotate(q_arr, position_ids)?),
                Tensor::F32(self.rotate(k_arr, position_ids)?),
            )),
            _ => Err(tensor_op_error(
                "StarCoder2RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }

    /// Rotate a projection tensor along its head pairs (rank-agnostic).
    ///
    /// Accepts any layout whose last dimension is `n_heads * head_dim`: a bare
    /// `[width]` head vector (treated as a single position), `[seq, width]`, or
    /// `[batch, seq, width]`. Positions are taken from the second-to-last axis
    /// (or position 0 for a 1-D input); every leading axis is an independent
    /// batch. Iteration is in row-major logical order so it matches
    /// `from_shape_vec` regardless of the input's physical memory layout.
    fn rotate(&self, arr: &ArrayD<f32>, position_ids: &[usize]) -> Result<ArrayD<f32>> {
        let shape = arr.shape().to_vec();
        if shape.is_empty() {
            return Err(tensor_op_error(
                "StarCoder2RotaryEmbedding::rotate",
                "RoPE input must have at least one dimension",
            ));
        }
        let width = shape[shape.len() - 1];
        let head_dim = self.head_dim;
        let half = head_dim / 2;
        if head_dim == 0 || !width.is_multiple_of(head_dim) {
            return Err(tensor_op_error(
                "StarCoder2RotaryEmbedding::rotate",
                "projection width is not a multiple of head_dim",
            ));
        }
        let n_heads = width / head_dim;
        let (outer, seq) = if shape.len() == 1 {
            (1usize, 1usize)
        } else {
            let seq = shape[shape.len() - 2];
            let outer: usize = shape[..shape.len() - 2].iter().product();
            (outer, seq)
        };

        let flat: Vec<f32> = arr.iter().copied().collect();
        let mut out = flat.clone();
        for o in 0..outer {
            for t in 0..seq {
                let pos = position_ids.get(t).copied().unwrap_or(t);
                let row_base = (o * seq + t) * width;
                for h in 0..n_heads {
                    let base = row_base + h * head_dim;
                    for i in 0..half {
                        let angle = pos as f64 * self.inv_freq[i];
                        let cos = angle.cos() as f32;
                        let sin = angle.sin() as f32;
                        let x1 = flat[base + i];
                        let x2 = flat[base + i + half];
                        out[base + i] = x1 * cos - x2 * sin;
                        out[base + i + half] = x2 * cos + x1 * sin;
                    }
                }
            }
        }

        ArrayD::from_shape_vec(IxDyn(&shape), out).map_err(|e| {
            tensor_op_error(
                "StarCoder2RotaryEmbedding::rotate",
                format!("failed to rebuild rotated tensor: {e}"),
            )
        })
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
    pub(crate) gate_proj: Linear,
    pub(crate) up_proj: Linear,
    pub(crate) down_proj: Linear,
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
    pub(crate) q_proj: Linear,
    pub(crate) k_proj: Linear,
    pub(crate) v_proj: Linear,
    pub(crate) o_proj: Linear,
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
        let (batch_size, seq_len, is_3d) = match shape.len() {
            2 => (1usize, shape[0], false),
            3 => (shape[0], shape[1], true),
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

        // Expand the grouped key/value heads to match the query heads (GQA).
        let k_expanded = self.repeat_kv(&k_rope)?;
        let v_expanded = self.repeat_kv(&v)?;

        let head_dim = self.head_dim;
        let num_heads = self.num_heads;

        // [.., num_heads * head_dim] -> [batch, num_heads, seq, head_dim].
        let to_heads = |t: &Tensor| -> Result<Tensor> {
            t.reshape(&[batch_size, seq_len, num_heads, head_dim])?.transpose(1, 2)
        };
        let q_h = to_heads(&q_rope)?;
        let k_h = to_heads(&k_expanded)?;
        let v_h = to_heads(&v_expanded)?;

        // Scaled dot-product scores: [batch, num_heads, seq, seq].
        let scale = (head_dim as f32).sqrt().recip();
        let scores = q_h.matmul(&k_h.transpose(2, 3)?)?.mul_scalar(scale)?;

        // Additive causal mask ([1,1,seq,seq] broadcasts over batch & heads),
        // softmax over the key axis, then weight the values.
        let scores = scores.add(&causal_mask(seq_len)?)?;
        let weights = scores.softmax(-1)?;
        let context = weights.matmul(&v_h)?; // [batch, num_heads, seq, head_dim]

        // -> [batch, seq, num_heads * head_dim], collapsing the batch axis for 2-D inputs.
        let context = context.transpose(1, 2)?;
        let context = if is_3d {
            context.reshape(&[batch_size, seq_len, num_heads * head_dim])?
        } else {
            context.reshape(&[seq_len, num_heads * head_dim])?
        };

        self.o_proj.forward(context)
    }
}

/// Build an additive causal mask of shape `[1, 1, seq, seq]`: `0` on and below the
/// diagonal, a large negative value above it so masked positions vanish under softmax.
fn causal_mask(seq_len: usize) -> Result<Tensor> {
    let mut mask = vec![0.0f32; seq_len * seq_len];
    for i in 0..seq_len {
        for j in (i + 1)..seq_len {
            mask[i * seq_len + j] = -1.0e9;
        }
    }
    Tensor::from_vec(mask, &[seq_len, seq_len])?.reshape(&[1, 1, seq_len, seq_len])
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
    pub(crate) self_attn: StarCoder2Attention,
    pub(crate) mlp: StarCoder2MLP,
    pub(crate) input_layernorm: StarCoder2RmsNorm,
    pub(crate) post_attention_layernorm: StarCoder2RmsNorm,
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

    /// Load a HuggingFace StarCoder2 checkpoint (safetensors or `torch.save`).
    ///
    /// See [`StarCoder2Model::load_checkpoint`] for the name map and the failure modes.
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

    /// Load a HuggingFace StarCoder2ForCausalLM checkpoint.
    ///
    /// The backbone is bound first, then the LM head. A checkpoint with tied
    /// word embeddings carries no `lm_head.weight`; the input embedding matrix
    /// is reused in that case, which is exactly what the tied configuration
    /// means — not a fallback to something invented.
    ///
    /// # Errors
    ///
    /// See [`StarCoder2Model::load_checkpoint`]; additionally fails when the checkpoint
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

impl StarCoder2Model {
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
    /// Fails when the checkpoint does not look like a StarCoder2 checkpoint,
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
        let attention_bias = self.config.use_bias;
        let mlp_bias = self.config.use_bias;

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

    // ── Real checkpoint loading ─────────────────────────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, DecoderFixtureSpec, F32Tensor};

    fn loading_config() -> StarCoder2Config {
        StarCoder2Config {
            vocab_size: 12,
            hidden_size: 8,
            intermediate_size: 16,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            sliding_window: None,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 16,
            use_bias: true,
        }
    }

    fn loading_fixture(config: &StarCoder2Config) -> DecoderFixtureSpec {
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
        spec.attention_bias = config.use_bias;
        spec.mlp_bias = config.use_bias;
        spec
    }

    /// Regression: `load_pretrained` returned `not_implemented`, so no StarCoder2 checkpoint
    /// could ever reach the model's parameters. It now binds every one of them,
    /// and the proof is that the checkpoint's exact values arrive in the layers.
    #[test]
    fn load_pretrained_binds_every_parameter_from_the_checkpoint() {
        let config = loading_config();
        let tensors = loading_fixture(&config).tensors();
        let bytes = build_safetensors(&tensors);

        let mut model = StarCoder2Model::new(config).expect("model must build");
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
        let mut model = StarCoder2Model::new(config.clone()).expect("model must build");
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

        let mut model = StarCoder2Model::new(config).expect("model must build");
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

        let mut model = StarCoder2Model::new(config).expect("model must build");
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
        let mut model = StarCoder2Model::new(loading_config()).expect("model must build");
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
        let wider = StarCoder2Config {
            hidden_size: config.hidden_size * 2,
            intermediate_size: config.intermediate_size * 2,
            ..config.clone()
        };
        let bytes = loading_fixture(&wider).safetensors();

        let mut model = StarCoder2Model::new(config).expect("model must build");
        let err = model
            .load_pretrained(&mut bytes.as_slice())
            .expect_err("a mismatched checkpoint must not be reshaped into place");
        assert!(err.to_string().contains("expects"), "unexpected: {err}");
    }
}
