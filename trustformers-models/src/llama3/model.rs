use crate::llama3::config::LLaMA3Config;
use crate::weight_loading::{Checkpoint, LoadReport};
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

    /// Replace the scale vector; it must keep the normalised width.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        if weight.len() != self.weight.len() {
            return Err(tensor_op_error(
                "LLaMA3RmsNorm::set_weight",
                format!(
                    "expected {} weights, got {}",
                    self.weight.len(),
                    weight.len()
                ),
            ));
        }
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

    /// Normalise every trailing `normalized_shape`-sized vector independently.
    ///
    /// RMSNorm is defined per token; pooling the mean square across the whole
    /// tensor would make each token's scale depend on its neighbours.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match (&input, &self.weight) {
            (Tensor::F32(arr), Tensor::F32(w)) => {
                let eps_f32 = self.eps as f32;
                let size = w.len();
                if size == 0 || !arr.len().is_multiple_of(size) {
                    return Err(tensor_op_error(
                        "LLaMA3RmsNorm::forward",
                        format!(
                            "tensor of {} elements is not a multiple of the norm size {size}",
                            arr.len()
                        ),
                    ));
                }
                let weight: Vec<f32> = w.iter().copied().collect();
                let values: Vec<f32> = arr.iter().copied().collect();
                let mut data = Vec::with_capacity(values.len());
                for chunk in values.chunks(size) {
                    let mean_sq = chunk.iter().map(|x| x * x).sum::<f32>() / size as f32;
                    let inv_rms = 1.0 / (mean_sq + eps_f32).sqrt();
                    for (value, scale) in chunk.iter().zip(weight.iter()) {
                        data.push(value * inv_rms * scale);
                    }
                }
                let out = ArrayD::from_shape_vec(IxDyn(arr.shape()), data).map_err(|e| {
                    tensor_op_error("LLaMA3RmsNorm::forward", format!("shape error: {e}"))
                })?;
                Ok(Tensor::F32(out))
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

    /// Apply RoPE to query and key tensors (shape-preserving).
    ///
    /// `q` and `k` are `[seq_len, heads * head_dim]` or
    /// `[batch, seq_len, heads * head_dim]`; under GQA the two tensors carry
    /// different head counts, so each is rotated independently. Within every
    /// `head_dim`-wide head block component `i` pairs with component
    /// `i + head_dim/2` and the pair is rotated by `angle = pos * inv_freq[i]`
    /// — the LLaMA "rotate-half" convention.
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // One (sin, cos) table shared by both tensors and every head.
        let mut table = Vec::with_capacity(position_ids.len() * self.inv_freq.len());
        for &pos in position_ids {
            for &freq in &self.inv_freq {
                let angle = pos as f64 * freq;
                table.push((angle.sin() as f32, angle.cos() as f32));
            }
        }

        let q_rotated = self.rotate(q, position_ids.len(), &table, "query")?;
        let k_rotated = self.rotate(k, position_ids.len(), &table, "key")?;
        Ok((q_rotated, k_rotated))
    }

    /// Rotate every head block of one tensor with the precomputed table.
    fn rotate(
        &self,
        tensor: &Tensor,
        positions: usize,
        table: &[(f32, f32)],
        role: &str,
    ) -> Result<Tensor> {
        let half = self.inv_freq.len();
        match tensor {
            Tensor::F32(arr) => {
                let shape = arr.shape().to_vec();
                let (batch, seq_len, width) =
                    split_sequence_shape(&shape, "LLaMA3RotaryEmbedding::apply_rotary_emb")?;
                if self.head_dim == 0 || !width.is_multiple_of(self.head_dim) {
                    return Err(tensor_op_error(
                        "LLaMA3RotaryEmbedding::apply_rotary_emb",
                        format!(
                            "{role} width {width} is not a multiple of head_dim {}",
                            self.head_dim
                        ),
                    ));
                }
                if seq_len != positions {
                    return Err(tensor_op_error(
                        "LLaMA3RotaryEmbedding::apply_rotary_emb",
                        format!(
                            "{role} has {seq_len} positions but {positions} position ids were given"
                        ),
                    ));
                }

                let heads = width / self.head_dim;
                let mut data: Vec<f32> = arr.iter().copied().collect();
                for b in 0..batch {
                    for t in 0..seq_len {
                        let row = (b * seq_len + t) * width;
                        for head in 0..heads {
                            let base = row + head * self.head_dim;
                            for i in 0..half {
                                let (sin, cos) = table[t * half + i];
                                let x = data[base + i];
                                let y = data[base + i + half];
                                data[base + i] = x * cos - y * sin;
                                data[base + i + half] = x * sin + y * cos;
                            }
                        }
                    }
                }

                let rotated = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
                    tensor_op_error(
                        "LLaMA3RotaryEmbedding::apply_rotary_emb",
                        format!("shape error while rebuilding the {role} tensor: {e}"),
                    )
                })?;
                Ok(Tensor::F32(rotated))
            },
            _ => Err(tensor_op_error(
                "LLaMA3RotaryEmbedding::apply_rotary_emb",
                "unsupported tensor dtype for RoPE",
            )),
        }
    }
}

/// Split a 2-D `[seq, features]` or 3-D `[batch, seq, features]` shape.
fn split_sequence_shape(shape: &[usize], context: &str) -> Result<(usize, usize, usize)> {
    match shape.len() {
        2 => Ok((1, shape[0], shape[1])),
        3 => Ok((shape[0], shape[1], shape[2])),
        _ => Err(tensor_op_error(
            context,
            format!("expected [seq, features] or [batch, seq, features], got {shape:?}"),
        )),
    }
}

/// Causal multi-head scaled dot-product attention over flat `[seq, heads*dim]`
/// buffers.
///
/// `queries`, `keys` and `values` must already have the same head count — under
/// GQA the caller expands the KV heads first. Returns `softmax(mask(Q Kᵀ) /
/// sqrt(head_dim)) V` flattened the same way as the inputs.
fn causal_multi_head_sdpa(
    queries: &[f32],
    keys: &[f32],
    values: &[f32],
    seq_len: usize,
    num_heads: usize,
    head_dim: usize,
) -> Result<Vec<f32>> {
    let width = num_heads * head_dim;
    let expected = seq_len * width;
    if queries.len() != expected || keys.len() != expected || values.len() != expected {
        return Err(tensor_op_error(
            "causal_multi_head_sdpa",
            format!(
                "shape mismatch: expected {expected} values each, got q {} / k {} / v {}",
                queries.len(),
                keys.len(),
                values.len()
            ),
        ));
    }
    if seq_len == 0 || width == 0 {
        return Ok(Vec::new());
    }

    let scale = (head_dim as f32).sqrt().recip();
    let mut output = vec![0.0f32; expected];
    let mut scores = vec![0.0f32; seq_len];

    for head in 0..num_heads {
        for query_pos in 0..seq_len {
            let visible = query_pos + 1;
            let q_base = query_pos * width + head * head_dim;

            let mut max_score = f32::NEG_INFINITY;
            for (key_pos, score) in scores.iter_mut().take(visible).enumerate() {
                let k_base = key_pos * width + head * head_dim;
                let dot: f32 =
                    (0..head_dim).map(|d| queries[q_base + d] * keys[k_base + d]).sum::<f32>()
                        * scale;
                *score = dot;
                if dot > max_score {
                    max_score = dot;
                }
            }

            let mut sum = 0.0f32;
            for score in scores.iter_mut().take(visible) {
                *score = (*score - max_score).exp();
                sum += *score;
            }
            let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

            for (key_pos, score) in scores.iter().take(visible).enumerate() {
                let weight = score * inv_sum;
                let v_base = key_pos * width + head * head_dim;
                for d in 0..head_dim {
                    output[q_base + d] += weight * values[v_base + d];
                }
            }
        }
    }

    Ok(output)
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

    /// Causal grouped-query self-attention.
    ///
    /// Accepts `[seq_len, hidden]` or `[batch, seq_len, hidden]` and returns the
    /// same rank. The computation is the real thing —
    /// `softmax(mask(Q Kᵀ) / sqrt(head_dim)) V` with RoPE applied to Q and K and
    /// each KV head shared by `num_query_groups` query heads. The keys and values
    /// are consumed, not discarded.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape().to_vec();
        let (batch, seq_len, hidden) = split_sequence_shape(&shape, "LLaMA3Attention::forward")?;
        let width = self.num_heads * self.head_dim;
        if hidden != width {
            return Err(tensor_op_error(
                "LLaMA3Attention::forward",
                format!(
                    "input hidden size {hidden} != num_heads {} * head_dim {}",
                    self.num_heads, self.head_dim
                ),
            ));
        }

        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        // GQA: expand the KV heads so query head `h` reads KV head
        // `h / num_query_groups`.
        let queries = q_rope.data()?;
        let keys = self.repeat_kv(&k_rope)?.data()?;
        let values = self.repeat_kv(&v)?.data()?;

        let stride = seq_len * width;
        let mut context = Vec::with_capacity(batch * stride);
        for b in 0..batch {
            let range = b * stride..(b + 1) * stride;
            context.extend_from_slice(&causal_multi_head_sdpa(
                &queries[range.clone()],
                &keys[range.clone()],
                &values[range],
                seq_len,
                self.num_heads,
                self.head_dim,
            )?);
        }

        let attn_output = Tensor::from_vec(context, &shape)?;
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

impl LLaMA3Model {
    /// Bind a parsed checkpoint into this model.
    ///
    /// Accepts both `model.*`-prefixed (`LlamaForCausalLM`) and bare
    /// (`LlamaModel`) HuggingFace layouts. `allowed_unused_prefixes` names the
    /// checkpoint namespaces this base model legitimately ignores — the LM head
    /// lives on [`LLaMA3ForCausalLM`], not here.
    ///
    /// Missing parameters are reported rather than substituted, so a checkpoint
    /// that does not match the architecture cannot silently leave random weights
    /// in place.
    pub fn load_checkpoint(
        &mut self,
        checkpoint: &Checkpoint,
        allowed_unused_prefixes: &[&str],
    ) -> Result<LoadReport> {
        let prefix = checkpoint.detect_prefix(&["model.", ""], "embed_tokens.weight")?;
        let mut binder = checkpoint.binder(&prefix);

        let hidden = self.config.hidden_size;
        let head_dim = self.config.hidden_size / self.config.num_attention_heads;
        let kv_width = self.config.num_key_value_heads * head_dim;
        let intermediate = self.config.intermediate_size;

        if let Some(weight) =
            binder.take_shaped("embed_tokens.weight", &[self.config.vocab_size, hidden])?
        {
            self.embed_tokens.set_weight(weight)?;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("layers.{i}.self_attn");
            if let Some(w) =
                binder.take_shaped(&format!("{attn}.q_proj.weight"), &[hidden, hidden])?
            {
                layer.self_attn.q_proj.set_weight(w)?;
            }
            if let Some(w) =
                binder.take_shaped(&format!("{attn}.k_proj.weight"), &[kv_width, hidden])?
            {
                layer.self_attn.k_proj.set_weight(w)?;
            }
            if let Some(w) =
                binder.take_shaped(&format!("{attn}.v_proj.weight"), &[kv_width, hidden])?
            {
                layer.self_attn.v_proj.set_weight(w)?;
            }
            if let Some(w) =
                binder.take_shaped(&format!("{attn}.o_proj.weight"), &[hidden, hidden])?
            {
                layer.self_attn.o_proj.set_weight(w)?;
            }

            let mlp = format!("layers.{i}.mlp");
            if let Some(w) =
                binder.take_shaped(&format!("{mlp}.gate_proj.weight"), &[intermediate, hidden])?
            {
                layer.mlp.gate_proj.set_weight(w)?;
            }
            if let Some(w) =
                binder.take_shaped(&format!("{mlp}.up_proj.weight"), &[intermediate, hidden])?
            {
                layer.mlp.up_proj.set_weight(w)?;
            }
            if let Some(w) =
                binder.take_shaped(&format!("{mlp}.down_proj.weight"), &[hidden, intermediate])?
            {
                layer.mlp.down_proj.set_weight(w)?;
            }

            if let Some(w) =
                binder.take_shaped(&format!("layers.{i}.input_layernorm.weight"), &[hidden])?
            {
                layer.input_layernorm.set_weight(w)?;
            }
            if let Some(w) = binder.take_shaped(
                &format!("layers.{i}.post_attention_layernorm.weight"),
                &[hidden],
            )? {
                layer.post_attention_layernorm.set_weight(w)?;
            }
        }

        if let Some(w) = binder.take_shaped("norm.weight", &[hidden])? {
            self.norm.set_weight(w)?;
        }

        binder.finish(allowed_unused_prefixes)
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

    /// Load a HuggingFace LLaMA-3 checkpoint (safetensors or `torch.save`).
    ///
    /// The base model is bound first, then the LM head. Checkpoints that tie the
    /// head to the input embeddings (`tie_word_embeddings`) carry no `lm_head`
    /// tensor; the embedding matrix is reused in that case, which is exactly what
    /// the tied configuration means.
    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.model.load_checkpoint(&checkpoint, &["lm_head."])?;

        let config = self.model.config().clone();
        let expected = [config.vocab_size, config.hidden_size];
        if let Some(weight) = checkpoint.get("lm_head.weight") {
            if weight.shape() != expected {
                return Err(tensor_op_error(
                    "LLaMA3ForCausalLM::load_pretrained",
                    format!(
                        "lm_head.weight has shape {:?} but this model expects {expected:?}",
                        weight.shape()
                    ),
                ));
            }
            self.lm_head.set_weight(weight.clone())?;
        } else {
            // Tied embeddings: reuse the input embedding matrix as the head.
            let embed_name = if checkpoint.contains("model.embed_tokens.weight") {
                "model.embed_tokens.weight"
            } else {
                "embed_tokens.weight"
            };
            let embeddings = checkpoint.get(embed_name).ok_or_else(|| {
                tensor_op_error(
                    "LLaMA3ForCausalLM::load_pretrained",
                    "checkpoint holds neither lm_head.weight nor an embedding matrix to tie it to"
                        .to_string(),
                )
            })?;
            if embeddings.shape() != expected {
                return Err(tensor_op_error(
                    "LLaMA3ForCausalLM::load_pretrained",
                    format!(
                        "tied embedding matrix has shape {:?} but the head expects {expected:?}",
                        embeddings.shape()
                    ),
                ));
            }
            self.lm_head.set_weight(embeddings.clone())?;
        }
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

    // ── Weight loading ────────────────────────────────────────────────────────

    use crate::weight_loading::test_support::{build_safetensors, F32Tensor};
    use std::io::Cursor;

    /// Every tensor a `LlamaForCausalLM` checkpoint of `cfg` would carry.
    fn checkpoint_tensors(cfg: &LLaMA3Config, with_lm_head: bool) -> Vec<F32Tensor> {
        let hidden = cfg.hidden_size;
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let kv_width = cfg.num_key_value_heads * head_dim;
        let inter = cfg.intermediate_size;
        let mut tensors = vec![F32Tensor::ramp(
            "model.embed_tokens.weight",
            &[cfg.vocab_size, hidden],
            0.0,
        )];
        for i in 0..cfg.num_hidden_layers {
            let seed = (i as f32 + 1.0) * 100.0;
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.self_attn.q_proj.weight"),
                &[hidden, hidden],
                seed,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.self_attn.k_proj.weight"),
                &[kv_width, hidden],
                seed + 1.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.self_attn.v_proj.weight"),
                &[kv_width, hidden],
                seed + 2.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.self_attn.o_proj.weight"),
                &[hidden, hidden],
                seed + 3.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.mlp.gate_proj.weight"),
                &[inter, hidden],
                seed + 4.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.mlp.up_proj.weight"),
                &[inter, hidden],
                seed + 5.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.mlp.down_proj.weight"),
                &[hidden, inter],
                seed + 6.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.input_layernorm.weight"),
                &[hidden],
                seed + 7.0,
            ));
            tensors.push(F32Tensor::ramp(
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
                &[hidden],
                seed + 8.0,
            ));
        }
        tensors.push(F32Tensor::ramp("model.norm.weight", &[hidden], 999.0));
        if with_lm_head {
            tensors.push(F32Tensor::ramp(
                "lm_head.weight",
                &[cfg.vocab_size, hidden],
                7.0,
            ));
        }
        tensors
    }

    /// `load_pretrained` used to return `not_implemented`; it must now bind a real
    /// safetensors checkpoint into the right parameters.
    #[test]
    fn test_load_pretrained_binds_real_weights() {
        let cfg = LLaMA3Config::small_test();
        let bytes = build_safetensors(&checkpoint_tensors(&cfg, true));
        let mut model = LLaMA3ForCausalLM::new(cfg.clone()).expect("model");

        model
            .load_pretrained(&mut Cursor::new(bytes))
            .expect("loading a matching checkpoint must succeed");

        // Layer 0's query projection must hold the ramp emitted for that name.
        let q = model.model.layers[0].self_attn.q_proj.weight().data().expect("q weights");
        assert_eq!(q.len(), cfg.hidden_size * cfg.hidden_size);
        assert!((q[0] - 100.0).abs() < 1e-6, "q_proj[0] = {}", q[0]);
        assert!((q[1] - 100.5).abs() < 1e-6, "q_proj[1] = {}", q[1]);

        // Layer 1 must not receive layer 0's tensors.
        let q1 = model.model.layers[1].self_attn.q_proj.weight().data().expect("q1 weights");
        assert!(
            (q1[0] - 200.0).abs() < 1e-6,
            "layer 1 q_proj[0] = {}",
            q1[0]
        );

        // The LM head is bound from its own tensor.
        let head = model.lm_head.weight().data().expect("lm head");
        assert!((head[0] - 7.0).abs() < 1e-6, "lm_head[0] = {}", head[0]);

        // And the model actually runs with the loaded weights.
        let logits = model.forward(vec![1u32, 2, 3]).expect("forward");
        assert_eq!(logits.shape()[logits.shape().len() - 1], cfg.vocab_size);
    }

    #[test]
    fn test_load_pretrained_ties_lm_head_when_absent() {
        let cfg = LLaMA3Config::small_test();
        let bytes = build_safetensors(&checkpoint_tensors(&cfg, false));
        let mut model = LLaMA3ForCausalLM::new(cfg.clone()).expect("model");
        model.load_pretrained(&mut Cursor::new(bytes)).expect("tied load");

        let head = model.lm_head.weight().data().expect("lm head");
        // The fixture ramp for `model.embed_tokens.weight` starts at 0.0 with a
        // 0.5 step, so a tied head holds exactly those values.
        assert_eq!(head.len(), cfg.vocab_size * cfg.hidden_size);
        assert!((head[0]).abs() < 1e-6, "tied head[0] = {}", head[0]);
        assert!((head[5] - 2.5).abs() < 1e-6, "tied head[5] = {}", head[5]);
    }

    #[test]
    fn test_load_pretrained_rejects_wrong_shapes() {
        let cfg = LLaMA3Config::small_test();
        let mut tensors = checkpoint_tensors(&cfg, true);
        tensors[0] = F32Tensor::ramp("model.embed_tokens.weight", &[cfg.vocab_size, 8], 0.0);
        let bytes = build_safetensors(&tensors);
        let mut model = LLaMA3ForCausalLM::new(cfg).expect("model");
        assert!(
            model.load_pretrained(&mut Cursor::new(bytes)).is_err(),
            "a mis-shaped tensor must be rejected, not silently ignored"
        );
    }

    #[test]
    fn test_load_pretrained_rejects_foreign_checkpoint() {
        let cfg = LLaMA3Config::small_test();
        let bytes = build_safetensors(&[F32Tensor::ramp("some.other.model.weight", &[2, 2], 1.0)]);
        let mut model = LLaMA3ForCausalLM::new(cfg).expect("model");
        assert!(
            model.load_pretrained(&mut Cursor::new(bytes)).is_err(),
            "a checkpoint for another architecture must be rejected"
        );
    }

    #[test]
    fn test_rms_norm_normalises_each_row() {
        let norm = LLaMA3RmsNorm::new(4, 1e-6).expect("norm");
        let input = Tensor::from_vec(vec![1.0, 1.0, 1.0, 1.0, 30.0, 30.0, 30.0, 30.0], &[2, 4])
            .expect("input");
        let out = norm.forward(input).expect("forward").data().expect("data");
        for value in out {
            assert!(
                (value - 1.0).abs() < 1e-3,
                "each row must be normalised on its own, got {value}"
            );
        }
    }

    // ── Real RoPE and real scaled dot-product attention ───────────────────────

    /// Tiny config: hidden 4 = 2 query heads × head_dim 2, 2 KV heads (no GQA).
    fn tiny_attention_config() -> LLaMA3Config {
        LLaMA3Config {
            vocab_size: 16,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 2,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32,
        }
    }

    /// `[n, n]` identity matrix as a `Linear` weight (`[out_features, in]`).
    fn identity_weight(n: usize) -> Tensor {
        let mut data = vec![0.0f32; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        Tensor::from_vec(data, &[n, n]).expect("identity weight")
    }

    /// RoPE must rotate *every* head block. The old implementation cloned the
    /// input and threw the angle away, so nothing moved at all.
    #[test]
    fn test_rope_rotates_every_head_and_matches_hand_computation() {
        // head_dim = 2 → half = 1 → inv_freq = [1/theta^0] = [1.0].
        let rope = LLaMA3RotaryEmbedding::new(2, 32, 10000.0);
        assert_eq!(rope.half_dim(), 1);

        // Two positions, two heads of width 2 → last dim = 4.
        let values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0];
        let q = Tensor::from_vec(values.clone(), &[2, 4]).expect("q");
        let k = Tensor::from_vec(values, &[2, 4]).expect("k");
        let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &[0, 1]).expect("rope");
        let q_data = q_out.data().expect("q data");
        let k_data = k_out.data().expect("k data");
        assert_eq!(q_data, k_data, "both tensors use the same angle table");

        // Position 0: angle 0 → identity.
        for (i, expected) in [1.0f32, 0.0, 0.0, 1.0].iter().enumerate() {
            assert!(
                (q_data[i] - expected).abs() < 1e-6,
                "position 0 must be the identity, got {}",
                q_data[i]
            );
        }

        // Position 1: angle = 1 rad.
        let (sin, cos) = (1.0f32.sin(), 1.0f32.cos());
        // Head 0 holds (x, y) = (1, 0) → (cos, sin).
        assert!((q_data[4] - cos).abs() < 1e-6, "head 0 x: {}", q_data[4]);
        assert!((q_data[5] - sin).abs() < 1e-6, "head 0 y: {}", q_data[5]);
        // Head 1 holds (x, y) = (0, 1) → (-sin, cos). This is the block the old
        // code never touched.
        assert!((q_data[6] + sin).abs() < 1e-6, "head 1 x: {}", q_data[6]);
        assert!((q_data[7] - cos).abs() < 1e-6, "head 1 y: {}", q_data[7]);
    }

    /// The attention output must depend on V. With `v_proj = 0` a real SDPA
    /// returns zeros; the old code returned `o_proj(scale * Q)`, which is not.
    #[test]
    fn test_attention_reads_values() {
        let cfg = tiny_attention_config();
        let mut attn = LLaMA3Attention::new(&cfg).expect("attention");
        attn.q_proj.set_weight(identity_weight(4)).expect("q");
        attn.k_proj.set_weight(identity_weight(4)).expect("k");
        attn.o_proj.set_weight(identity_weight(4)).expect("o");
        attn.v_proj
            .set_weight(Tensor::from_vec(vec![0.0; 16], &[4, 4]).expect("zero"))
            .expect("v");

        let input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).expect("input");
        let out = attn.forward(input).expect("forward").data().expect("data");
        for value in out {
            assert!(
                value.abs() < 1e-6,
                "with zero values the context must be zero, got {value}"
            );
        }
    }

    /// Full naive reference: identity projections, RoPE recomputed from scratch
    /// in the test, then causal softmax attention per head.
    #[test]
    fn test_attention_matches_naive_reference() {
        let cfg = tiny_attention_config();
        let mut attn = LLaMA3Attention::new(&cfg).expect("attention");
        attn.q_proj.set_weight(identity_weight(4)).expect("q");
        attn.k_proj.set_weight(identity_weight(4)).expect("k");
        attn.v_proj.set_weight(identity_weight(4)).expect("v");
        attn.o_proj.set_weight(identity_weight(4)).expect("o");

        let rows = [[1.0f32, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let input = Tensor::from_vec(flat, &[2, 4]).expect("input");
        let got = attn.forward(input).expect("forward").data().expect("data");

        // ── Independent reference ────────────────────────────────────────────
        let head_dim = 2usize;
        let half = head_dim / 2;
        let heads = 2usize;
        let seq = 2usize;
        // inv_freq[i] = 1 / theta^(2i/head_dim); with half = 1 this is [1.0].
        let inv_freq: Vec<f32> = (0..half)
            .map(|i| 1.0 / (10000.0f32).powf(2.0 * i as f32 / head_dim as f32))
            .collect();

        let mut q_ref = rows.iter().map(|r| r.to_vec()).collect::<Vec<_>>();
        let mut k_ref = q_ref.clone();
        let v_ref = q_ref.clone(); // values are never rotated
        for (pos, (q_row, k_row)) in q_ref.iter_mut().zip(k_ref.iter_mut()).enumerate() {
            for head in 0..heads {
                let base = head * head_dim;
                for (i, &freq) in inv_freq.iter().enumerate() {
                    let angle = pos as f32 * freq;
                    let (sin, cos) = (angle.sin(), angle.cos());
                    for row in [&mut *q_row, &mut *k_row] {
                        let x = row[base + i];
                        let y = row[base + i + half];
                        row[base + i] = x * cos - y * sin;
                        row[base + i + half] = x * sin + y * cos;
                    }
                }
            }
        }

        let scale = 1.0f32 / (head_dim as f32).sqrt();
        let mut expected = vec![0.0f32; seq * heads * head_dim];
        for head in 0..heads {
            let base = head * head_dim;
            for query_pos in 0..seq {
                let scores: Vec<f32> = (0..=query_pos)
                    .map(|key_pos| {
                        (0..head_dim)
                            .map(|d| q_ref[query_pos][base + d] * k_ref[key_pos][base + d])
                            .sum::<f32>()
                            * scale
                    })
                    .collect();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (key_pos, weight) in exps.iter().enumerate() {
                    for d in 0..head_dim {
                        expected[query_pos * heads * head_dim + base + d] +=
                            weight / sum * v_ref[key_pos][base + d];
                    }
                }
            }
        }

        for (i, (actual, want)) in got.iter().zip(expected.iter()).enumerate() {
            assert!(
                (actual - want).abs() < 1e-5,
                "element {i}: got {actual}, reference {want}"
            );
        }
    }

    /// Causality: a later token must not influence an earlier output.
    #[test]
    fn test_attention_is_causal() {
        let cfg = tiny_attention_config();
        let attn = LLaMA3Attention::new(&cfg).expect("attention");

        let first = Tensor::from_vec(vec![0.4, -0.3, 0.7, 0.1, 0.2, 0.9, -0.5, 0.3], &[2, 4])
            .expect("first");
        let second = Tensor::from_vec(vec![0.4, -0.3, 0.7, 0.1, -9.0, 4.0, 6.0, -2.0], &[2, 4])
            .expect("second");
        let a = attn.forward(first).expect("forward a").data().expect("a");
        let b = attn.forward(second).expect("forward b").data().expect("b");

        for i in 0..4 {
            assert!(
                (a[i] - b[i]).abs() < 1e-6,
                "token 0 must not see token 1: {} vs {}",
                a[i],
                b[i]
            );
        }
        assert!(
            (4..8).any(|i| (a[i] - b[i]).abs() > 1e-4),
            "token 1 must react to its own change"
        );
    }

    /// GQA: with fewer KV heads than query heads the expanded keys/values still
    /// have to feed the attention. A zeroed KV head must silence exactly the
    /// query heads that share it.
    #[test]
    fn test_grouped_query_attention_shares_kv_heads() {
        let cfg = LLaMA3Config {
            num_key_value_heads: 1,
            ..tiny_attention_config()
        };
        let mut attn = LLaMA3Attention::new(&cfg).expect("attention");
        assert_eq!(attn.num_kv_heads(), 1);
        attn.q_proj.set_weight(identity_weight(4)).expect("q");
        attn.o_proj.set_weight(identity_weight(4)).expect("o");
        // k_proj / v_proj map hidden 4 → 1 KV head × head_dim 2.
        attn.k_proj
            .set_weight(
                Tensor::from_vec(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], &[2, 4])
                    .expect("k weight"),
            )
            .expect("k");
        attn.v_proj
            .set_weight(
                Tensor::from_vec(vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0], &[2, 4])
                    .expect("v weight"),
            )
            .expect("v");

        let input =
            Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], &[2, 4]).expect("input");
        let out = attn.forward(input).expect("forward").data().expect("data");

        // The single KV head is shared by both query heads, so both output head
        // blocks are convex combinations of v = [(3,4), (7,8)] and must lie
        // inside that range — impossible if V had been discarded.
        for query_pos in 0..2 {
            for head in 0..2 {
                let x = out[query_pos * 4 + head * 2];
                let y = out[query_pos * 4 + head * 2 + 1];
                assert!((3.0..=7.0).contains(&x), "x out of the V hull: {x}");
                assert!((4.0..=8.0).contains(&y), "y out of the V hull: {y}");
            }
        }
    }

    /// Batched input must produce the same rows as running each batch item alone.
    #[test]
    fn test_attention_batched_matches_single() {
        let cfg = tiny_attention_config();
        let attn = LLaMA3Attention::new(&cfg).expect("attention");

        let row_a = vec![0.1f32, -0.2, 0.3, 0.4, 0.5, 0.6, -0.7, 0.8];
        let row_b = vec![-0.9f32, 0.2, 0.1, 0.0, 0.4, -0.4, 0.6, 0.2];
        let mut batched = row_a.clone();
        batched.extend_from_slice(&row_b);

        let out_batched = attn
            .forward(Tensor::from_vec(batched, &[2, 2, 4]).expect("batched"))
            .expect("batched forward")
            .data()
            .expect("data");
        let out_a = attn
            .forward(Tensor::from_vec(row_a, &[2, 4]).expect("a"))
            .expect("a forward")
            .data()
            .expect("data");
        let out_b = attn
            .forward(Tensor::from_vec(row_b, &[2, 4]).expect("b"))
            .expect("b forward")
            .data()
            .expect("data");

        for i in 0..8 {
            assert!((out_batched[i] - out_a[i]).abs() < 1e-6, "batch 0 item {i}");
            assert!(
                (out_batched[8 + i] - out_b[i]).abs() < 1e-6,
                "batch 1 item {i}"
            );
        }
    }
}
