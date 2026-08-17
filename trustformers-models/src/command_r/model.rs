use crate::command_r::config::CommandRConfig;
use crate::common::ActivationType;
use crate::generation_utils::GenerationUtils;
use crate::weight_loading::{Checkpoint, LoadReport};
use scirs2_core::random::{thread_rng, Rng}; // SciRS2 Integration Policy
use trustformers_core::{
    errors::{invalid_config, tensor_op_error, Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Command R Rotary Position Embedding.
///
/// The rotation follows the HuggingFace `rotate_half` convention: for a head
/// vector `x` of size `dim`, channel `i` is paired with channel `i + dim/2` and
/// the pair is rotated by `position * inv_freq[i]`.
#[derive(Debug, Clone)]
pub struct CommandRRoPE {
    dim: usize,
    max_seq_len: usize,
    base: f32,
    /// `inv_freq[i] = base^(-2i/dim)` for `i in 0..dim/2`
    inv_freq: Vec<f32>,
}

impl CommandRRoPE {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Result<Self> {
        if dim < 2 || !dim.is_multiple_of(2) {
            return Err(invalid_config(
                "CommandRRoPE::new",
                format!("head dimension must be even and >= 2, got {dim}"),
            ));
        }
        let inv_freq = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / dim as f32))
            .collect::<Vec<f32>>();

        Ok(Self {
            dim,
            max_seq_len,
            base,
            inv_freq,
        })
    }

    /// Head dimension this RoPE was built for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Maximum position this RoPE accepts.
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// RoPE base frequency (`theta`).
    pub fn base(&self) -> f32 {
        self.base
    }

    /// Inverse frequencies, one per rotated channel pair.
    pub fn inv_freq(&self) -> &[f32] {
        &self.inv_freq
    }

    /// Build the `cos`/`sin` tables for the given absolute positions.
    ///
    /// Both tensors have shape `[positions.len(), dim / 2]`.
    pub fn cos_sin(&self, positions: &[usize]) -> Result<(Tensor, Tensor)> {
        let half = self.dim / 2;
        let mut cos_vals = Vec::with_capacity(positions.len() * half);
        let mut sin_vals = Vec::with_capacity(positions.len() * half);
        for &pos in positions {
            for freq in &self.inv_freq {
                let angle = pos as f32 * freq;
                cos_vals.push(angle.cos());
                sin_vals.push(angle.sin());
            }
        }
        Ok((
            Tensor::from_vec(cos_vals, &[positions.len(), half])?,
            Tensor::from_vec(sin_vals, &[positions.len(), half])?,
        ))
    }

    /// Rotate a flat `[batch, seq_len, num_heads, head_dim]` buffer in place.
    ///
    /// `positions[t]` is the **absolute** position of token `t`, so incremental
    /// decoding with a KV cache rotates the new token with its true position.
    pub fn rotate_in_place(
        &self,
        data: &mut [f32],
        batch: usize,
        num_heads: usize,
        head_dim: usize,
        positions: &[usize],
    ) -> Result<()> {
        if head_dim != self.dim {
            return Err(tensor_op_error(
                "CommandRRoPE::rotate_in_place",
                format!("head_dim {head_dim} does not match RoPE dim {}", self.dim),
            ));
        }
        let seq_len = positions.len();
        let expected = batch * seq_len * num_heads * head_dim;
        if data.len() != expected {
            return Err(tensor_op_error(
                "CommandRRoPE::rotate_in_place",
                format!("expected {expected} elements, got {}", data.len()),
            ));
        }
        if let Some(&max_pos) = positions.iter().max() {
            if max_pos >= self.max_seq_len {
                return Err(tensor_op_error(
                    "CommandRRoPE::rotate_in_place",
                    format!(
                        "position {max_pos} exceeds max_sequence_length {}",
                        self.max_seq_len
                    ),
                ));
            }
        }

        let half = head_dim / 2;
        for b in 0..batch {
            for (t, &pos) in positions.iter().enumerate() {
                for head in 0..num_heads {
                    let base = ((b * seq_len + t) * num_heads + head) * head_dim;
                    for i in 0..half {
                        let angle = pos as f32 * self.inv_freq[i];
                        let (sin_val, cos_val) = angle.sin_cos();
                        let x0 = data[base + i];
                        let x1 = data[base + i + half];
                        data[base + i] = x0 * cos_val - x1 * sin_val;
                        data[base + i + half] = x0 * sin_val + x1 * cos_val;
                    }
                }
            }
        }
        Ok(())
    }
}

/// Command R Attention layer
#[derive(Debug, Clone)]
pub struct CommandRAttention {
    #[allow(dead_code)]
    config: CommandRConfig,
    hidden_size: usize,
    num_heads: usize,
    num_key_value_heads: usize,
    head_dim: usize,

    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,

    rope: CommandRRoPE,
    attention_dropout: f32,
    #[allow(dead_code)]
    use_flash_attention: bool,
}

impl CommandRAttention {
    pub fn new(config: &CommandRConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_key_value_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        let q_proj = Linear::new(hidden_size, num_heads * head_dim, config.use_bias);
        let k_proj = Linear::new(hidden_size, num_key_value_heads * head_dim, config.use_bias);
        let v_proj = Linear::new(hidden_size, num_key_value_heads * head_dim, config.use_bias);
        let o_proj = Linear::new(num_heads * head_dim, hidden_size, config.use_bias);

        let rope = CommandRRoPE::new(head_dim, config.max_sequence_length, config.rope_theta)?;

        Ok(Self {
            config: config.clone(),
            hidden_size,
            num_heads,
            num_key_value_heads,
            head_dim,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rope,
            attention_dropout: config.attention_dropout,
            use_flash_attention: config.use_flash_attention,
        })
    }

    /// Causal grouped-query attention with a real KV cache.
    ///
    /// * `hidden_states` — `[batch, seq_len, hidden_size]`
    /// * `attention_mask` — optional **additive** mask broadcastable to
    ///   `[batch, num_heads, seq_len, total_len]`
    /// * `position_ids` — absolute positions of the tokens in `hidden_states`;
    ///   when it does not carry one entry per token the positions default to
    ///   `past_len .. past_len + seq_len`
    /// * `past_key_value` — cached keys/values shaped
    ///   `[batch, past_len, num_key_value_heads, head_dim]`
    ///
    /// The returned cache is the **concatenation** of the past and the freshly
    /// computed keys/values, so incremental decoding attends to every token seen
    /// so far.
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let shape = hidden_states.shape().to_vec();
        if shape.len() != 3 {
            return Err(tensor_op_error(
                "CommandRAttention::forward",
                format!("expected [batch, seq_len, hidden_size], got {shape:?}"),
            ));
        }
        let (batch_size, seq_len, hidden) = (shape[0], shape[1], shape[2]);
        if hidden != self.hidden_size {
            return Err(tensor_op_error(
                "CommandRAttention::forward",
                format!(
                    "input hidden size {hidden} does not match config hidden size {}",
                    self.hidden_size
                ),
            ));
        }

        // Length of the cached prefix (0 when decoding from scratch).
        let past_len = match past_key_value {
            Some((past_key, past_value)) => {
                let key_shape = past_key.shape().to_vec();
                let value_shape = past_value.shape().to_vec();
                let expected_trailing = [batch_size, self.num_key_value_heads, self.head_dim];
                let valid = |s: &[usize]| {
                    s.len() == 4
                        && s[0] == expected_trailing[0]
                        && s[2] == expected_trailing[1]
                        && s[3] == expected_trailing[2]
                };
                if !valid(&key_shape) || !valid(&value_shape) || key_shape[1] != value_shape[1] {
                    return Err(tensor_op_error(
                        "CommandRAttention::forward",
                        format!(
                            "past key/value must be [{}, past_len, {}, {}], got {key_shape:?} / {value_shape:?}",
                            batch_size, self.num_key_value_heads, self.head_dim
                        ),
                    ));
                }
                key_shape[1]
            },
            None => 0,
        };

        // Absolute positions for the incoming tokens.
        let positions = Self::resolve_positions(position_ids, seq_len, past_len);

        // Project to queries, keys, and values.
        let query_states = self.q_proj.forward(hidden_states.clone())?;
        let key_states = self.k_proj.forward(hidden_states.clone())?;
        let value_states = self.v_proj.forward(hidden_states.clone())?;

        // Apply RoPE to the queries and the *new* keys only (cached keys were
        // already rotated when they were produced).
        let mut query_data = query_states.data()?;
        let mut key_data = key_states.data()?;
        self.rope.rotate_in_place(
            &mut query_data,
            batch_size,
            self.num_heads,
            self.head_dim,
            &positions,
        )?;
        self.rope.rotate_in_place(
            &mut key_data,
            batch_size,
            self.num_key_value_heads,
            self.head_dim,
            &positions,
        )?;

        let new_key = Tensor::from_vec(
            key_data,
            &[batch_size, seq_len, self.num_key_value_heads, self.head_dim],
        )?;
        let new_value = value_states.reshape(&[
            batch_size,
            seq_len,
            self.num_key_value_heads,
            self.head_dim,
        ])?;

        // Append the new keys/values to the cache along the sequence axis.
        let (key_cache, value_cache) = match past_key_value {
            Some((past_key, past_value)) => (
                Tensor::concat(&[past_key.clone(), new_key], 1)?,
                Tensor::concat(&[past_value.clone(), new_value], 1)?,
            ),
            None => (new_key, new_value),
        };

        let total_len = past_len + seq_len;
        let attn_output = self.scaled_dot_product_attention(
            &query_data,
            &key_cache,
            &value_cache,
            batch_size,
            seq_len,
            total_len,
            past_len,
            attention_mask,
        )?;

        let attn_output = attn_output.reshape(&[batch_size, seq_len, self.hidden_size])?;
        let attn_output = self.o_proj.forward(attn_output)?;

        Ok((attn_output, Some((key_cache, value_cache))))
    }

    /// Absolute positions for the incoming tokens.
    ///
    /// `position_ids` wins when it supplies one entry per token; otherwise the
    /// positions continue the cached prefix.
    fn resolve_positions(position_ids: &Tensor, seq_len: usize, past_len: usize) -> Vec<usize> {
        if let Ok(values) = position_ids.data() {
            if values.len() == seq_len {
                return values.iter().map(|&p| p.max(0.0) as usize).collect();
            }
            if values.len() > seq_len && values.len() % seq_len == 0 {
                // [batch, seq_len]: every row carries the same positions.
                return values[..seq_len].iter().map(|&p| p.max(0.0) as usize).collect();
            }
        }
        (past_len..past_len + seq_len).collect()
    }

    /// `softmax(mask(Q Kᵀ) / sqrt(head_dim)) V` over the cached keys/values.
    ///
    /// Queries are the flat `[batch, seq_len, num_heads, head_dim]` buffer that
    /// has already been rotated; keys/values are the full cache
    /// `[batch, total_len, num_key_value_heads, head_dim]`. Query head `h` reads
    /// KV head `h / num_query_groups`, which is grouped-query attention.
    ///
    /// Attention dropout is deliberately not applied: this is an inference path
    /// and randomised logits would make generation non-reproducible.
    #[allow(clippy::too_many_arguments)]
    fn scaled_dot_product_attention(
        &self,
        query_data: &[f32],
        key_cache: &Tensor,
        value_cache: &Tensor,
        batch_size: usize,
        seq_len: usize,
        total_len: usize,
        past_len: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let head_dim = self.head_dim;
        let num_heads = self.num_heads;
        let num_kv_heads = self.num_key_value_heads;
        if num_kv_heads == 0 || !num_heads.is_multiple_of(num_kv_heads) {
            return Err(invalid_config(
                "CommandRAttention",
                format!(
                    "num_attention_heads {num_heads} must be a multiple of num_key_value_heads {num_kv_heads}"
                ),
            ));
        }
        let groups = num_heads / num_kv_heads;

        let key_data = key_cache.data()?;
        let value_data = value_cache.data()?;

        // Optional additive mask, indexed as [batch?, head?, query, key].
        let mask_data = match attention_mask {
            Some(mask) => Some(mask.data()?),
            None => None,
        };
        let mask_len = mask_data.as_ref().map(|m| m.len()).unwrap_or(0);

        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut output = vec![0.0f32; batch_size * seq_len * num_heads * head_dim];
        let mut scores = vec![0.0f32; total_len];

        for b in 0..batch_size {
            for head in 0..num_heads {
                let kv_head = head / groups;
                for q_idx in 0..seq_len {
                    let q_base = ((b * seq_len + q_idx) * num_heads + head) * head_dim;
                    // Causal limit: a query at cache slot past_len + q_idx may
                    // read every key up to and including its own slot.
                    let visible = past_len + q_idx + 1;
                    let mut max_score = f32::NEG_INFINITY;
                    for (key_idx, score) in scores.iter_mut().take(visible).enumerate() {
                        let k_base =
                            ((b * total_len + key_idx) * num_kv_heads + kv_head) * head_dim;
                        let mut dot = 0.0f32;
                        for d in 0..head_dim {
                            dot += query_data[q_base + d] * key_data[k_base + d];
                        }
                        dot *= scale;
                        if let Some(mask) = &mask_data {
                            dot += Self::mask_value(
                                mask, mask_len, b, head, q_idx, key_idx, batch_size, num_heads,
                                seq_len, total_len,
                            );
                        }
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

                    let out_base = ((b * seq_len + q_idx) * num_heads + head) * head_dim;
                    for (key_idx, score) in scores.iter().take(visible).enumerate() {
                        let weight = score * inv_sum;
                        if weight == 0.0 {
                            continue;
                        }
                        let v_base =
                            ((b * total_len + key_idx) * num_kv_heads + kv_head) * head_dim;
                        for d in 0..head_dim {
                            output[out_base + d] += weight * value_data[v_base + d];
                        }
                    }
                }
            }
        }

        Tensor::from_vec(output, &[batch_size, seq_len, num_heads, head_dim])
    }

    /// Read an additive mask entry, tolerating the common broadcast layouts
    /// (`[q, k]`, `[batch, q, k]`, `[batch, 1, q, k]`, `[batch, heads, q, k]`).
    #[allow(clippy::too_many_arguments)]
    fn mask_value(
        mask: &[f32],
        mask_len: usize,
        batch: usize,
        head: usize,
        q_idx: usize,
        key_idx: usize,
        batch_size: usize,
        num_heads: usize,
        seq_len: usize,
        total_len: usize,
    ) -> f32 {
        let plane = seq_len * total_len;
        if mask_len == plane {
            return mask[q_idx * total_len + key_idx];
        }
        if mask_len == batch_size * plane {
            return mask[batch * plane + q_idx * total_len + key_idx];
        }
        if mask_len == batch_size * num_heads * plane {
            return mask[((batch * num_heads + head) * seq_len + q_idx) * total_len + key_idx];
        }
        0.0
    }

    /// Configured attention-dropout probability.
    ///
    /// Dropout is a training-time regulariser; this inference path never applies
    /// it so that decoding stays reproducible.
    pub fn attention_dropout(&self) -> f32 {
        self.attention_dropout
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
    }
}

/// Command R MLP (Feed-Forward Network)
#[derive(Debug, Clone)]
pub struct CommandRMLP {
    #[allow(dead_code)]
    config: CommandRConfig,
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    intermediate_size: usize,

    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,

    activation: ActivationType,
}

impl CommandRMLP {
    pub fn new(config: &CommandRConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        let gate_proj = Linear::new(hidden_size, intermediate_size, config.use_bias);
        let up_proj = Linear::new(hidden_size, intermediate_size, config.use_bias);
        let down_proj = Linear::new(intermediate_size, hidden_size, config.use_bias);

        Ok(Self {
            config: config.clone(),
            hidden_size,
            intermediate_size,
            gate_proj,
            up_proj,
            down_proj,
            // Parse once at construction; unknown identifiers fall back to GELU,
            // matching the previous `_ => gate_output.gelu()` default arm.
            activation: ActivationType::from_config_str_or(
                &config.activation_function,
                ActivationType::Gelu,
            ),
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Gate projection with activation
        let gate_output = self.gate_proj.forward(x.clone())?;
        let gate_output = self.activation.apply(&gate_output)?;

        // Up projection
        let up_output = self.up_proj.forward(x.clone())?;

        // Element-wise multiplication
        let intermediate = gate_output.mul(&up_output)?;

        // Down projection
        let output = self.down_proj.forward(intermediate)?;

        Ok(output)
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

/// Command R Decoder Layer
#[derive(Debug, Clone)]
pub struct CommandRDecoderLayer {
    #[allow(dead_code)]
    config: CommandRConfig,
    #[allow(dead_code)]
    hidden_size: usize,

    self_attn: CommandRAttention,
    mlp: CommandRMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl CommandRDecoderLayer {
    pub fn new(config: &CommandRConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;

        let self_attn = CommandRAttention::new(config)?;
        let mlp = CommandRMLP::new(config)?;

        let input_layernorm = LayerNorm::new(vec![hidden_size], config.rms_norm_eps)?;
        let post_attention_layernorm = LayerNorm::new(vec![hidden_size], config.rms_norm_eps)?;

        Ok(Self {
            config: config.clone(),
            hidden_size,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: &Tensor,
        past_key_value: Option<(&Tensor, &Tensor)>,
    ) -> Result<(Tensor, Option<(Tensor, Tensor)>)> {
        let residual = hidden_states.clone();

        // Pre-attention layer norm
        let hidden_states = self.input_layernorm.forward(hidden_states.clone())?;

        // Self-attention
        let (attn_output, present_key_value) =
            self.self_attn
                .forward(&hidden_states, attention_mask, position_ids, past_key_value)?;

        // Add residual connection
        let hidden_states = residual.add(&attn_output)?;
        let residual = hidden_states.clone();

        // Post-attention layer norm
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;

        // MLP
        let mlp_output = self.mlp.forward(&hidden_states)?;

        // Add residual connection
        let hidden_states = residual.add(&mlp_output)?;

        Ok((hidden_states, present_key_value))
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

/// Command R Model
#[derive(Debug, Clone)]
pub struct CommandRModel {
    config: CommandRConfig,
    #[allow(dead_code)]
    vocab_size: usize,
    #[allow(dead_code)]
    hidden_size: usize,
    #[allow(dead_code)]
    num_hidden_layers: usize,

    embed_tokens: Embedding,
    layers: Vec<CommandRDecoderLayer>,
    norm: LayerNorm,

    #[allow(dead_code)]
    pad_token_id: Option<usize>,
    #[allow(dead_code)]
    bos_token_id: Option<usize>,
    #[allow(dead_code)]
    eos_token_id: Option<usize>,
}

impl CommandRModel {
    pub fn new(config: &CommandRConfig) -> Result<Self> {
        config.validate().map_err(|e| invalid_config("config_validation", &e))?;

        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;
        let num_hidden_layers = config.num_hidden_layers;

        let embed_tokens = Embedding::new(vocab_size, hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..num_hidden_layers {
            layers.push(CommandRDecoderLayer::new(config)?);
        }

        let norm = LayerNorm::new(vec![hidden_size], config.rms_norm_eps)?;

        Ok(Self {
            config: config.clone(),
            vocab_size,
            hidden_size,
            num_hidden_layers,
            embed_tokens,
            layers,
            norm,
            pad_token_id: config.pad_token_id,
            bos_token_id: config.bos_token_id,
            eos_token_id: config.eos_token_id,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_values: Option<&[(Tensor, Tensor)]>,
    ) -> Result<CommandRModelOutput> {
        let input_shape = input_ids.shape().to_vec();
        if input_shape.len() != 2 {
            return Err(tensor_op_error(
                "CommandRModel::forward",
                format!("expected input ids of shape [batch, seq_len], got {input_shape:?}"),
            ));
        }
        let batch_size = input_shape[0];
        let seq_len = input_shape[1];

        // Length of the cached prefix, so positions continue where it left off.
        let past_len = past_key_values
            .and_then(|pkv| pkv.first())
            .map(|(key, _)| {
                let shape = key.shape().to_vec();
                if shape.len() == 4 {
                    shape[1]
                } else {
                    0
                }
            })
            .unwrap_or(0);

        // Create position IDs if not provided
        let position_ids = if let Some(pos_ids) = position_ids {
            pos_ids.clone()
        } else {
            let pos_ids: Vec<f32> =
                (past_len..past_len + seq_len).map(|position| position as f32).collect();
            Tensor::from_vec(pos_ids, &[1, seq_len])?
        };

        // Token embeddings
        // Convert tensor to vector of token IDs
        let input_ids_vec = match input_ids {
            Tensor::I64(arr) => arr.iter().map(|&x| x as u32).collect::<Vec<u32>>(),
            _ => {
                return Err(tensor_op_error(
                    "CommandRModel::forward",
                    "Input IDs must be integer tensor",
                ))
            },
        };
        // Embedding lookup returns [batch * seq_len, hidden]; restore the batch axis.
        let embedded = self.embed_tokens.forward(input_ids_vec)?;
        let mut hidden_states =
            embedded.reshape(&[batch_size, seq_len, self.config.hidden_size])?;

        // Process through transformer layers
        let mut present_key_values = Vec::new();
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let past_key_value = past_key_values.map(|pkv| (&pkv[layer_idx].0, &pkv[layer_idx].1));

            let (layer_output, present_key_value) = layer.forward(
                &hidden_states,
                attention_mask,
                &position_ids,
                past_key_value,
            )?;

            hidden_states = layer_output;
            if let Some(pkv) = present_key_value {
                present_key_values.push(pkv);
            }
        }

        // Final layer norm
        let hidden_states = self.norm.forward(hidden_states)?;

        Ok(CommandRModelOutput {
            last_hidden_state: hidden_states,
            past_key_values: if present_key_values.is_empty() {
                None
            } else {
                Some(present_key_values)
            },
            hidden_states: None,
            attentions: None,
        })
    }
}

impl Model for CommandRModel {
    type Config = CommandRConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Process input through model layers
        let mut hidden_states = input;

        // Pass through all decoder layers - note: layer.forward returns (hidden_states, past_key_value)
        // For the Model trait implementation, we ignore past_key_values and use default params
        let seq_len = hidden_states.shape().get(1).copied().unwrap_or(0);
        let position_ids = Tensor::from_vec(
            (0..seq_len).map(|position| position as f32).collect(),
            &[1, seq_len],
        )?;
        for layer in &self.layers {
            let (new_hidden_states, _) = layer.forward(
                &hidden_states,
                None, // attention_mask
                &position_ids,
                None, // past_key_value
            )?;
            hidden_states = new_hidden_states;
        }

        // Apply final normalization
        hidden_states = self.norm.forward(hidden_states)?;

        Ok(hidden_states)
    }

    /// Load a HuggingFace Command-R checkpoint (safetensors or `torch.save`).
    ///
    /// Every tensor is bound to a named parameter; nothing is invented. Tensors
    /// the architecture does not know about make the load fail rather than being
    /// logged and ignored, and parameters the checkpoint does not carry are
    /// reported in the [`LoadReport`] instead of silently keeping their
    /// randomly-initialised values.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.load_checkpoint(&checkpoint, &["lm_head."])?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embed_params = self.embed_tokens.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.norm.parameter_count();

        embed_params + layers_params + norm_params
    }
}

impl CommandRModel {
    /// Bind a parsed checkpoint into this model.
    ///
    /// Accepts both `model.*`-prefixed (`CohereForCausalLM`) and bare
    /// (`CohereModel`) HuggingFace layouts. `allowed_unused_prefixes` names the
    /// checkpoint namespaces this base model legitimately ignores.
    ///
    /// Command-R shares one layer norm across the parallel attention/MLP block.
    /// This implementation runs attention and MLP sequentially with two norms, so
    /// a released Cohere checkpoint carries no `post_attention_layernorm`; that
    /// norm keeps its unit scale and the block therefore does **not** reproduce
    /// Cohere's parallel-residual arithmetic bit for bit. Every tensor the
    /// checkpoint does carry is bound exactly, and nothing is invented.
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

        if let Some(weight) =
            binder.take_shaped("embed_tokens.weight", &[self.config.vocab_size, hidden])?
        {
            self.embed_tokens.set_weight(weight)?;
        }

        for (i, layer) in self.layers.iter_mut().enumerate() {
            let attn = format!("layers.{i}.self_attn");
            if let Some(w) =
                binder.take_shaped(&format!("{attn}.q_proj.weight"), &[q_width, hidden])?
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
                binder.take_shaped(&format!("{attn}.o_proj.weight"), &[hidden, q_width])?
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
            if let Some(b) = binder.take_optional(&format!("layers.{i}.input_layernorm.bias")) {
                layer.input_layernorm.set_bias(b)?;
            }
            // Cohere's decoder shares a single norm across the parallel
            // attention/MLP block, so this parameter is genuinely absent from
            // released checkpoints: request it optionally instead of reporting a
            // spurious "missing" for every layer.
            if let Some(w) =
                binder.take_optional(&format!("layers.{i}.post_attention_layernorm.weight"))
            {
                layer.post_attention_layernorm.set_weight(w)?;
            }
            if let Some(b) =
                binder.take_optional(&format!("layers.{i}.post_attention_layernorm.bias"))
            {
                layer.post_attention_layernorm.set_bias(b)?;
            }
        }

        if let Some(w) = binder.take_shaped("norm.weight", &[hidden])? {
            self.norm.set_weight(w)?;
        }
        if let Some(b) = binder.take_optional("norm.bias") {
            self.norm.set_bias(b)?;
        }

        binder.finish(allowed_unused_prefixes)
    }
}

/// Command R Model Output
#[derive(Debug, Clone)]
pub struct CommandRModelOutput {
    pub last_hidden_state: Tensor,
    pub past_key_values: Option<Vec<(Tensor, Tensor)>>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

/// Command R for Causal Language Modeling
#[derive(Debug, Clone)]
pub struct CommandRForCausalLM {
    model: CommandRModel,
    lm_head: Linear,
    config: CommandRConfig,
}

impl CommandRForCausalLM {
    pub fn new(config: &CommandRConfig) -> Result<Self> {
        let model = CommandRModel::new(config)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, config.use_bias);

        Ok(Self {
            model,
            lm_head,
            config: config.clone(),
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        position_ids: Option<&Tensor>,
        past_key_values: Option<&[(Tensor, Tensor)]>,
        labels: Option<&Tensor>,
    ) -> Result<CommandRCausalLMOutput> {
        let outputs =
            self.model.forward(input_ids, attention_mask, position_ids, past_key_values)?;

        let logits = self.lm_head.forward(outputs.last_hidden_state)?;

        let loss = match labels {
            Some(labels) => Some(Self::causal_lm_loss(&logits, labels)?),
            None => None,
        };

        Ok(CommandRCausalLMOutput {
            loss,
            logits,
            past_key_values: outputs.past_key_values,
            hidden_states: outputs.hidden_states,
            attentions: outputs.attentions,
        })
    }

    /// Shifted causal-LM cross-entropy loss.
    ///
    /// `logits` is `[batch, seq_len, vocab]` and `labels` holds `batch * seq_len`
    /// token ids. Position `t` predicts `labels[t + 1]`, so the loss is
    /// `mean(-log softmax(logits[t])[labels[t+1]])` over the batch. Labels equal
    /// to `-100` are ignored, matching the HuggingFace convention.
    fn causal_lm_loss(logits: &Tensor, labels: &Tensor) -> Result<Tensor> {
        let shape = logits.shape().to_vec();
        if shape.len() != 3 {
            return Err(tensor_op_error(
                "CommandRForCausalLM::causal_lm_loss",
                format!("expected logits [batch, seq_len, vocab], got {shape:?}"),
            ));
        }
        let (batch_size, seq_len, vocab_size) = (shape[0], shape[1], shape[2]);

        let label_values = labels.data()?;
        if label_values.len() != batch_size * seq_len {
            return Err(tensor_op_error(
                "CommandRForCausalLM::causal_lm_loss",
                format!(
                    "expected {} labels, got {}",
                    batch_size * seq_len,
                    label_values.len()
                ),
            ));
        }
        if seq_len < 2 {
            return Err(tensor_op_error(
                "CommandRForCausalLM::causal_lm_loss",
                "causal-LM loss needs at least two positions to shift".to_string(),
            ));
        }

        let logit_values = logits.data()?;
        let mut total = 0.0f32;
        let mut counted = 0usize;
        for b in 0..batch_size {
            for t in 0..seq_len - 1 {
                let target = label_values[b * seq_len + t + 1];
                if target < 0.0 {
                    continue; // ignore_index
                }
                let target_idx = target as usize;
                if target_idx >= vocab_size {
                    return Err(tensor_op_error(
                        "CommandRForCausalLM::causal_lm_loss",
                        format!("label {target_idx} is outside the vocabulary ({vocab_size})"),
                    ));
                }
                let row = &logit_values[(b * seq_len + t) * vocab_size..][..vocab_size];
                let max_logit = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let sum_exp: f32 = row.iter().map(|&x| (x - max_logit).exp()).sum();
                let log_prob = row[target_idx] - max_logit - sum_exp.ln();
                total -= log_prob;
                counted += 1;
            }
        }

        if counted == 0 {
            return Err(tensor_op_error(
                "CommandRForCausalLM::causal_lm_loss",
                "every label was masked out; no loss could be computed".to_string(),
            ));
        }
        Tensor::scalar(total / counted as f32)
    }

    /// Autoregressive generation with a real KV cache and real sampling.
    ///
    /// `temperature <= 0` selects greedy decoding; otherwise the logits are
    /// temperature-scaled and passed to top-k / top-p / full multinomial
    /// sampling from [`GenerationUtils`].
    pub fn generate(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Tensor> {
        let mut rng = thread_rng();
        self.generate_with_rng(input_ids, max_length, temperature, top_k, top_p, &mut rng)
    }

    /// Same as [`generate`](Self::generate) but with a caller-supplied RNG so
    /// sampling can be made reproducible in tests.
    pub fn generate_with_rng(
        &mut self,
        input_ids: &Tensor,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut impl Rng,
    ) -> Result<Tensor> {
        let input_shape = input_ids.shape().to_vec();
        if input_shape.len() != 2 || input_shape[0] != 1 {
            return Err(tensor_op_error(
                "CommandRForCausalLM::generate",
                format!("generation expects input ids of shape [1, seq_len], got {input_shape:?}"),
            ));
        }

        let mut token_ids: Vec<i64> = match input_ids {
            Tensor::I64(arr) => arr.iter().copied().collect(),
            other => other.data()?.iter().map(|&value| value as i64).collect(),
        };
        if token_ids.is_empty() {
            return Err(tensor_op_error(
                "CommandRForCausalLM::generate",
                "input_ids must not be empty".to_string(),
            ));
        }

        let mut past_key_values: Option<Vec<(Tensor, Tensor)>> = None;
        let mut step_input = Tensor::from_vec_i64(token_ids.clone(), &[1, token_ids.len()])?;

        for _ in 0..max_length {
            let outputs =
                self.forward(&step_input, None, None, past_key_values.as_deref(), None)?;

            let logits_shape = outputs.logits.shape().to_vec();
            let vocab_size = logits_shape[logits_shape.len() - 1];
            let step_len = logits_shape[1];
            let all_logits = outputs.logits.data()?;
            let mut next_logits = all_logits[(step_len - 1) * vocab_size..][..vocab_size].to_vec();

            if temperature > 0.0 {
                GenerationUtils::apply_temperature(&mut next_logits, temperature);
            }
            let next_token = Self::sample_next_token(&next_logits, temperature, top_k, top_p, rng)?;

            token_ids.push(next_token as i64);
            past_key_values = outputs.past_key_values;

            if let Some(eos_id) = self.config.eos_token_id {
                if next_token as usize == eos_id {
                    break;
                }
            }

            // With a populated cache only the newly generated token is fed back.
            step_input = if past_key_values.is_some() {
                Tensor::from_vec_i64(vec![next_token as i64], &[1, 1])?
            } else {
                Tensor::from_vec_i64(token_ids.clone(), &[1, token_ids.len()])?
            };
        }

        let generated_len = token_ids.len();
        Tensor::from_vec_i64(token_ids, &[1, generated_len])
    }

    /// Pick the next token from (already temperature-scaled) logits.
    ///
    /// Delegates to the crate-wide [`GenerationUtils`] implementations instead of
    /// re-deriving them: `top_k` and `top_p` are honoured, and with neither set
    /// the token is drawn from the full softmax distribution. `temperature <= 0`
    /// means deterministic greedy decoding.
    fn sample_next_token(
        logits: &[f32],
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
        rng: &mut impl Rng,
    ) -> Result<u32> {
        if logits.is_empty() {
            return Err(tensor_op_error(
                "CommandRForCausalLM::sample_next_token",
                "logits must not be empty".to_string(),
            ));
        }
        if temperature <= 0.0 {
            return Ok(GenerationUtils::sample_greedy(logits));
        }
        if let Some(k) = top_k {
            let k = k.clamp(1, logits.len());
            return GenerationUtils::sample_top_k(logits, k, rng);
        }
        if let Some(p) = top_p {
            return GenerationUtils::sample_top_p(logits, p, rng);
        }
        let probs = GenerationUtils::softmax(logits);
        Ok(GenerationUtils::sample_from_probs(&probs, rng)? as u32)
    }
}

/// Command R Causal LM Output
#[derive(Debug, Clone)]
pub struct CommandRCausalLMOutput {
    pub loss: Option<Tensor>,
    pub logits: Tensor,
    pub past_key_values: Option<Vec<(Tensor, Tensor)>>,
    pub hidden_states: Option<Vec<Tensor>>,
    pub attentions: Option<Vec<Tensor>>,
}

impl Model for CommandRForCausalLM {
    type Config = CommandRConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Forward through the model to get hidden states. The trait method takes
        // hidden states (the inherent `CommandRModel::forward` takes token ids),
        // so disambiguate explicitly.
        let hidden_states = <CommandRModel as Model>::forward(&self.model, input)?;

        // Apply language modeling head to get logits
        let logits = self.lm_head.forward(hidden_states)?;

        Ok(logits)
    }

    /// Load a HuggingFace Command-R checkpoint into the base model and the head.
    ///
    /// Cohere checkpoints tie the LM head to the input embeddings, so a
    /// checkpoint without `lm_head.weight` reuses the embedding matrix — that is
    /// what the tied configuration means, not a fallback guess.
    fn load_pretrained(&mut self, reader: &mut dyn std::io::Read) -> Result<()> {
        let checkpoint = Checkpoint::from_reader(reader)?;
        self.model.load_checkpoint(&checkpoint, &["lm_head."])?;

        let expected = [self.config.vocab_size, self.config.hidden_size];
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
                        "CommandRForCausalLM::load_pretrained",
                        "checkpoint holds neither lm_head.weight nor an embedding matrix to tie it to"
                            .to_string(),
                    )
                })?
            },
        };
        if head.shape() != expected {
            return Err(tensor_op_error(
                "CommandRForCausalLM::load_pretrained",
                format!(
                    "language-model head has shape {:?} but this model expects {expected:?}",
                    head.shape()
                ),
            ));
        }
        self.lm_head.set_weight(head.clone())?;
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters() + self.lm_head.parameter_count()
    }
}

impl CommandRForCausalLM {
    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::WeightLoadingConfig;

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };
        self.load_from_path_with_config(model_path, config)
    }

    /// Load model weights with an explicit [`WeightLoadingConfig`].
    pub fn load_from_path_with_config(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
        config: crate::weight_loading::WeightLoadingConfig,
    ) -> Result<()> {
        use crate::weight_loading::auto_create_loader;

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embedding weights
        if let Ok(embed_weights) = loader.load_tensor("model.embed_tokens.weight") {
            self.model.embed_tokens.set_weight(embed_weights)?;
        }

        // Load layer weights
        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            // Load attention weights
            let attn_prefix = format!("model.layers.{}.self_attn", i);

            if let Ok(q_weight) = loader.load_tensor(&format!("{}.q_proj.weight", attn_prefix)) {
                layer.self_attn.q_proj.set_weight(q_weight)?;
            }
            if let Ok(k_weight) = loader.load_tensor(&format!("{}.k_proj.weight", attn_prefix)) {
                layer.self_attn.k_proj.set_weight(k_weight)?;
            }
            if let Ok(v_weight) = loader.load_tensor(&format!("{}.v_proj.weight", attn_prefix)) {
                layer.self_attn.v_proj.set_weight(v_weight)?;
            }
            if let Ok(o_weight) = loader.load_tensor(&format!("{}.o_proj.weight", attn_prefix)) {
                layer.self_attn.o_proj.set_weight(o_weight)?;
            }

            // Load MLP weights
            let mlp_prefix = format!("model.layers.{}.mlp", i);

            if let Ok(gate_weight) = loader.load_tensor(&format!("{}.gate_proj.weight", mlp_prefix))
            {
                layer.mlp.gate_proj.set_weight(gate_weight)?;
            }
            if let Ok(up_weight) = loader.load_tensor(&format!("{}.up_proj.weight", mlp_prefix)) {
                layer.mlp.up_proj.set_weight(up_weight)?;
            }
            if let Ok(down_weight) = loader.load_tensor(&format!("{}.down_proj.weight", mlp_prefix))
            {
                layer.mlp.down_proj.set_weight(down_weight)?;
            }

            // Load layer norm weights
            if let Ok(ln1_weight) =
                loader.load_tensor(&format!("model.layers.{}.input_layernorm.weight", i))
            {
                layer.input_layernorm.set_weight(ln1_weight)?;
            }
            if let Ok(ln2_weight) = loader.load_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                i
            )) {
                layer.post_attention_layernorm.set_weight(ln2_weight)?;
            }
        }

        // Load final layer norm
        if let Ok(norm_weight) = loader.load_tensor("model.norm.weight") {
            self.model.norm.set_weight(norm_weight)?;
        }

        // Load LM head weights
        if let Ok(lm_head_weight) = loader.load_tensor("lm_head.weight") {
            self.lm_head.set_weight(lm_head_weight)?;
        }

        Ok(())
    }

    /// Load from HuggingFace Hub model name
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        // Check if model is cached locally
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .unwrap_or_else(|_| {
                std::env::var("HOME").unwrap_or_else(|_| ".".to_string())
                    + "/.cache/huggingface/hub"
            });

        let model_path = std::path::Path::new(&cache_dir)
            .join(format!("models--{}", model_name.replace("/", "--")));

        if model_path.exists() {
            self.load_from_path(&model_path)
        } else {
            // Attempt to download the model from HuggingFace Hub
            self.download_from_huggingface_hub(model_name, &model_path)?;
            self.load_from_path(&model_path)
        }
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface_hub(
        &self,
        model_name: &str,
        model_path: &std::path::Path,
    ) -> Result<()> {
        use std::process::Command;

        tracing::info!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name,
            model_path
        );

        // Create the model directory
        std::fs::create_dir_all(model_path).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to create model directory: {}",
                e
            ))
        })?;

        // List of essential files for Command-R models
        let essential_files = vec![
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "pytorch_model.bin", // Try .bin first
            "model.safetensors", // Fall back to safetensors
        ];

        let base_url = format!("https://huggingface.co/{}/resolve/main", model_name);

        // Try to download each essential file
        for file_name in &essential_files {
            let file_url = format!("{}/{}", base_url, file_name);
            let file_path = model_path.join(file_name);

            tracing::info!("Attempting to download {}", file_url);

            // Convert path to string once for both commands
            let file_path_str = file_path.to_str().ok_or_else(|| {
                TrustformersError::invalid_config(format!("Invalid UTF-8 in path: {:?}", file_path))
            })?;

            // Try using curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail on HTTP errors
                    "-o",
                    file_path_str,
                    &file_url,
                ])
                .output();

            match curl_result {
                Ok(output) if output.status.success() => {
                    tracing::info!("Successfully downloaded {}", file_name);
                    continue;
                },
                Ok(output) => {
                    tracing::warn!(
                        "Failed to download {} with curl: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    tracing::info!("curl not available: {}", e);
                },
            }

            // Try using wget as fallback
            let wget_result = Command::new("wget").args(["-O", file_path_str, &file_url]).output();

            match wget_result {
                Ok(output) if output.status.success() => {
                    tracing::info!("Successfully downloaded {} with wget", file_name);
                    continue;
                },
                Ok(output) => {
                    tracing::warn!(
                        "Failed to download {} with wget: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    tracing::info!("wget not available: {}", e);
                },
            }

            // If essential files like config.json or pytorch_model.bin fail, return error
            if matches!(file_name, &"config.json" | &"pytorch_model.bin") {
                return Err(trustformers_core::errors::TrustformersError::io_error(format!(
                    "Failed to download essential file {} for model {}. Please ensure curl or wget is installed and you have internet access.",
                    file_name, model_name
                )));
            }
        }

        tracing::info!(
            "Successfully downloaded model {} to {:?}",
            model_name,
            model_path
        );
        Ok(())
    }

    /// Load weights through a **memory-mapped** loader.
    ///
    /// The loading config asks for a memory-mapped reader, so tensors are
    /// materialised one at a time out of the mapping instead of through an
    /// intermediate copy of the whole checkpoint. It is *not* deferred loading:
    /// when this call returns, every weight the model knows about is resident.
    /// Model parameters are owned `Tensor`s, so there is nothing left to resolve
    /// on first access.
    pub fn load_with_mmap(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::WeightLoadingConfig;

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: true,
            streaming: false,
            ..Default::default()
        };
        self.load_from_path_with_config(model_path, config)
    }

    /// Deprecated alias for [`load_with_mmap`](Self::load_with_mmap).
    ///
    /// The old name promised on-demand tensor resolution that this loader has
    /// never performed; it loads every tensor eagerly.
    #[deprecated(
        since = "0.2.1",
        note = "renamed to `load_with_mmap`: loading is memory-mapped, not deferred"
    )]
    pub fn load_with_lazy_loading(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        self.load_with_mmap(model_path)
    }
}

impl Config for CommandRConfig {
    fn validate(&self) -> Result<()> {
        self.validate().map_err(|e| invalid_config("config_validation", &e))
    }

    fn architecture(&self) -> &'static str {
        "command-r"
    }
}

#[cfg(test)]
#[path = "model_tests.rs"]
mod tests;
