use crate::llama::config::LlamaConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{invalid_config, tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// RMSNorm layer (Root Mean Square Layer Normalization)
/// Used in LLaMA instead of standard LayerNorm
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
}

impl RMSNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // RMSNorm: x * weight / sqrt(mean(x^2) + eps)
        match &input {
            Tensor::F32(arr) => {
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + self.eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);

                // Apply learnable weight
                match &self.weight {
                    Tensor::F32(weight_arr) => {
                        let result = &normalized * weight_arr;
                        Ok(Tensor::F32(result))
                    },
                    _ => Err(tensor_op_error(
                        "RMSNorm::forward",
                        "Unsupported weight tensor type for RMSNorm",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "RMSNorm::forward",
                "Unsupported input tensor type for RMSNorm",
            )),
        }
    }
}

impl RMSNorm {
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

/// Rotary Position Embedding (RoPE)
/// Reference: "RoFormer: Enhanced Transformer with Rotary Position Embedding" (Su et al., 2021)
pub struct RotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
}

impl RotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            base,
        }
    }

    /// Apply rotary embedding to query and key tensors.
    ///
    /// Implements RoPE (Su et al. 2021): rotates pairs of adjacent dimensions
    /// using position-dependent angles derived from the base frequency.
    /// Each pair `(x[i], x[i + half_dim])` is rotated by angle `pos / base^(2i/dim)`.
    ///
    /// `q` and `k` are expected to have shape `[seq_len, num_heads * head_dim]`
    /// or `[batch, seq_len, num_heads * head_dim]`.  The rotation is applied
    /// to the first `self.dim` values in each head.
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let mut rotated_q = q_arr.clone();
                let mut rotated_k = k_arr.clone();

                // q_arr / k_arr shape: [seq_len, total_dim]  (2-D, no batch)
                // or [batch, seq_len, total_dim] (3-D).
                // position_ids has length seq_len.
                let ndim = self.dim;
                let half = ndim / 2;

                // Validate that we can rotate: total_dim must be >= ndim
                let total_dim = {
                    let s = q_arr.shape();
                    s[s.len() - 1]
                };
                if total_dim < ndim {
                    return Err(tensor_op_error(
                        "RotaryEmbedding::apply_rotary_emb",
                        format!(
                            "tensor last dim {} is smaller than rope dim {}",
                            total_dim, ndim
                        ),
                    ));
                }

                // Pre-compute (cos, sin) for each position × each frequency pair
                // freqs[i] = 1 / base^(2i / ndim)  for i in 0..half
                let freqs: Vec<f32> = (0..half)
                    .map(|i| 1.0_f32 / self.base.powf(2.0 * i as f32 / ndim as f32))
                    .collect();

                // Rotate in-place.  We iterate over positions provided by
                // position_ids. For 2-D tensors the first axis is seq_len;
                // for 3-D tensors position_ids still indexes along seq_len.
                let shape = q_arr.shape().to_vec();
                let rank = shape.len();

                for (seq_idx, &pos) in position_ids.iter().enumerate() {
                    for i in 0..half {
                        let j = i + half; // companion dimension

                        let cos_val = (pos as f32 * freqs[i]).cos();
                        let sin_val = (pos as f32 * freqs[i]).sin();

                        if rank == 2 {
                            // shape: [seq_len, total_dim]
                            let qi = rotated_q[[seq_idx, i]];
                            let qj = rotated_q[[seq_idx, j]];
                            rotated_q[[seq_idx, i]] = qi * cos_val - qj * sin_val;
                            rotated_q[[seq_idx, j]] = qi * sin_val + qj * cos_val;

                            let ki = rotated_k[[seq_idx, i]];
                            let kj = rotated_k[[seq_idx, j]];
                            rotated_k[[seq_idx, i]] = ki * cos_val - kj * sin_val;
                            rotated_k[[seq_idx, j]] = ki * sin_val + kj * cos_val;
                        } else if rank == 3 {
                            // shape: [batch, seq_len, total_dim]
                            for b in 0..shape[0] {
                                let qi = rotated_q[[b, seq_idx, i]];
                                let qj = rotated_q[[b, seq_idx, j]];
                                rotated_q[[b, seq_idx, i]] = qi * cos_val - qj * sin_val;
                                rotated_q[[b, seq_idx, j]] = qi * sin_val + qj * cos_val;

                                let ki = rotated_k[[b, seq_idx, i]];
                                let kj = rotated_k[[b, seq_idx, j]];
                                rotated_k[[b, seq_idx, i]] = ki * cos_val - kj * sin_val;
                                rotated_k[[b, seq_idx, j]] = ki * sin_val + kj * cos_val;
                            }
                        }
                        // ranks other than 2/3 leave the values unchanged
                    }
                }

                Ok((Tensor::F32(rotated_q), Tensor::F32(rotated_k)))
            },
            _ => Err(tensor_op_error(
                "RotaryEmbedding::apply_rotary_emb",
                "Unsupported tensor types for RoPE",
            )),
        }
    }
}

/// LLaMA MLP layer with SiLU activation
pub struct LlamaMLP {
    pub gate_proj: Linear, // Linear layer for gating
    pub up_proj: Linear,   // Up projection
    pub down_proj: Linear, // Down projection
}

impl LlamaMLP {
    pub fn new(config: &LlamaConfig) -> Result<Self> {
        let gate_proj = Linear::new(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
        );
        let up_proj = Linear::new(
            config.hidden_size,
            config.intermediate_size,
            config.mlp_bias,
        );
        let down_proj = Linear::new(
            config.intermediate_size,
            config.hidden_size,
            config.mlp_bias,
        );

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    pub fn new_with_device(config: &LlamaConfig, device: Device) -> Result<Self> {
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
}

impl Layer for LlamaMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // LLaMA MLP: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SiLU to gate output
        let gate_activated = silu(&gate_output)?;

        // Element-wise multiply gate and up outputs
        let combined = match (&gate_activated, &up_output) {
            (Tensor::F32(gate_arr), Tensor::F32(up_arr)) => Ok(Tensor::F32(gate_arr * up_arr)),
            _ => Err(tensor_op_error(
                "LlamaMLP::forward",
                "Unsupported tensor types for LLaMA MLP",
            )),
        }?;

        // Apply down projection
        self.down_proj.forward(combined)
    }
}

impl LlamaMLP {
    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

/// LLaMA Attention layer with optional grouped-query attention
#[allow(dead_code)]
pub struct LlamaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    #[allow(dead_code)]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
}

impl LlamaAttention {
    pub fn new(config: &LlamaConfig) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();

        let q_proj = Linear::new(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.attention_bias,
        );
        let k_proj = Linear::new(
            config.hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
        );
        let v_proj = Linear::new(
            config.hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
        );
        let o_proj = Linear::new(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            config.attention_bias,
        );

        let rotary_emb =
            RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
        })
    }

    pub fn new_with_device(config: &LlamaConfig, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.attention_bias,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            num_kv_heads * head_dim,
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
            RotaryEmbedding::new(head_dim, config.max_position_embeddings, config.rope_theta);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
        })
    }
}

impl Layer for LlamaAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Full scaled dot-product attention with Grouped Query Attention (GQA).
    ///
    /// Shapes (2-D input, no explicit batch):
    ///   input    : [seq_len, hidden_size]
    ///   q        : [seq_len, num_heads * head_dim]
    ///   k, v     : [seq_len, num_kv_heads * head_dim]
    ///   output   : [seq_len, hidden_size]
    ///
    /// GQA: each KV head is shared by `num_heads / num_kv_heads` query heads.
    /// Causal mask: future positions are masked to −∞ before softmax.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        use scirs2_core::ndarray::{s, Array2};

        let shape = input.shape();
        let seq_len = if shape.len() == 2 {
            shape[0]
        } else if shape.len() == 3 {
            shape[1]
        } else {
            return Err(tensor_op_error(
                "LlamaAttention::forward",
                format!("Unexpected input shape: {:?}", shape),
            ));
        };

        let num_heads = self.num_heads;
        let num_kv_heads = self.num_kv_heads;
        let head_dim = self.head_dim;
        let queries_per_kv = num_heads / num_kv_heads; // GQA repeat factor

        // --- Projections ---
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // --- Apply RoPE ---
        let position_ids: Vec<usize> = (0..seq_len).collect();
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        // --- Extract F32 arrays ---
        let (q_arr, k_arr, v_arr) = match (&q_rope, &k_rope, &v) {
            (Tensor::F32(qa), Tensor::F32(ka), Tensor::F32(va)) => {
                (qa.clone(), ka.clone(), va.clone())
            },
            _ => {
                return Err(tensor_op_error(
                    "LlamaAttention::forward",
                    "Unsupported tensor types for LLaMA attention",
                ))
            },
        };

        // q_arr: [seq_len, num_heads * head_dim]
        // k_arr: [seq_len, num_kv_heads * head_dim]
        // v_arr: [seq_len, num_kv_heads * head_dim]

        let scale = (head_dim as f32).sqrt();
        let hidden_size = num_heads * head_dim;
        let mut attn_output = Array2::<f32>::zeros((seq_len, hidden_size));

        for h in 0..num_heads {
            // Which KV head this query head maps to (GQA)
            let kv_h = h / queries_per_kv;

            // Extract [seq_len, head_dim] slices
            let q_start = h * head_dim;
            let kv_start = kv_h * head_dim;

            let q_head = q_arr.slice(s![.., q_start..q_start + head_dim]).to_owned();
            let k_head = k_arr.slice(s![.., kv_start..kv_start + head_dim]).to_owned();
            let v_head = v_arr.slice(s![.., kv_start..kv_start + head_dim]).to_owned();

            // Scores = Q @ K^T / sqrt(head_dim)  shape: [seq_len, seq_len]
            let mut scores = q_head.dot(&k_head.t()) / scale;

            // Causal mask: mask future positions to -inf
            for i in 0..seq_len {
                for j in (i + 1)..seq_len {
                    scores[[i, j]] = f32::NEG_INFINITY;
                }
            }

            // Numerically-stable softmax row-wise
            let mut attn_weights = Array2::<f32>::zeros((seq_len, seq_len));
            for i in 0..seq_len {
                let row = scores.row(i);
                let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                let sum: f32 = exp_row.iter().sum();
                let safe_sum = if sum == 0.0 { 1.0 } else { sum };
                for (j, &val) in exp_row.iter().enumerate() {
                    attn_weights[[i, j]] = val / safe_sum;
                }
            }

            // Head output = attn_weights @ V  shape: [seq_len, head_dim]
            let head_out = attn_weights.dot(&v_head);

            let out_start = h * head_dim;
            attn_output
                .slice_mut(s![.., out_start..out_start + head_dim])
                .assign(&head_out);
        }

        // --- Output projection ---
        self.o_proj.forward(Tensor::F32(attn_output.into_dyn()))
    }
}

impl LlamaAttention {
    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
        // Note: RotaryEmbedding doesn't have learnable parameters
    }
}

/// LLaMA decoder layer
pub struct LlamaDecoderLayer {
    self_attn: LlamaAttention,
    mlp: LlamaMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl LlamaDecoderLayer {
    pub fn new(config: &LlamaConfig) -> Result<Self> {
        let self_attn = LlamaAttention::new(config)?;
        let mlp = LlamaMLP::new(config)?;
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn new_with_device(config: &LlamaConfig, device: Device) -> Result<Self> {
        let self_attn = LlamaAttention::new_with_device(config, device)?;
        let mlp = LlamaMLP::new_with_device(config, device)?;
        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }
}

impl Layer for LlamaDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm architecture: norm -> attention -> residual
        let normalized_input = self.input_layernorm.forward(input.clone())?;
        let attn_output = self.self_attn.forward(normalized_input)?;
        let residual1 = input.add(&attn_output)?;

        // Pre-norm architecture: norm -> mlp -> residual
        let normalized_residual = self.post_attention_layernorm.forward(residual1.clone())?;
        let mlp_output = self.mlp.forward(normalized_residual)?;
        let residual2 = residual1.add(&mlp_output)?;

        Ok(residual2)
    }
}

impl LlamaDecoderLayer {
    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

/// LLaMA model
pub struct LlamaModel {
    config: LlamaConfig,
    embed_tokens: Embedding,
    layers: Vec<LlamaDecoderLayer>,
    norm: RMSNorm,
}

impl LlamaModel {
    pub fn new(config: LlamaConfig) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new(&config)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn new_with_device(config: LlamaConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlamaDecoderLayer::new_with_device(&config, device)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    // LLaMA 1 model variants
    pub fn llama_7b() -> Result<Self> {
        Self::new(LlamaConfig::llama_7b())
    }

    pub fn llama_13b() -> Result<Self> {
        Self::new(LlamaConfig::llama_13b())
    }

    pub fn llama_30b() -> Result<Self> {
        Self::new(LlamaConfig::llama_30b())
    }

    pub fn llama_65b() -> Result<Self> {
        Self::new(LlamaConfig::llama_65b())
    }

    // LLaMA 2 model variants
    pub fn llama2_7b() -> Result<Self> {
        Self::new(LlamaConfig::llama2_7b())
    }

    pub fn llama2_13b() -> Result<Self> {
        Self::new(LlamaConfig::llama2_13b())
    }

    pub fn llama2_70b() -> Result<Self> {
        Self::new(LlamaConfig::llama2_70b())
    }

    // Code LLaMA variants
    pub fn code_llama_7b() -> Result<Self> {
        Self::new(LlamaConfig::code_llama_7b())
    }

    // LLaMA 3 model variants
    pub fn llama3_8b() -> Result<Self> {
        Self::new(LlamaConfig::llama3_8b())
    }

    pub fn llama3_70b() -> Result<Self> {
        Self::new(LlamaConfig::llama3_70b())
    }

    pub fn llama3_405b() -> Result<Self> {
        Self::new(LlamaConfig::llama3_405b())
    }

    // LLaMA 3 Instruct model variants
    pub fn llama3_8b_instruct() -> Result<Self> {
        Self::new(LlamaConfig::llama3_8b_instruct())
    }

    pub fn llama3_70b_instruct() -> Result<Self> {
        Self::new(LlamaConfig::llama3_70b_instruct())
    }

    pub fn llama3_405b_instruct() -> Result<Self> {
        Self::new(LlamaConfig::llama3_405b_instruct())
    }

    /// Create a LLaMA model from a pretrained model name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let config = LlamaConfig::from_pretrained_name(name).ok_or_else(|| {
            invalid_config(
                "pretrained_model",
                format!("Unknown pretrained model: {}", name),
            )
        })?;
        Self::new(config)
    }
}

impl Model for LlamaModel {
    type Config = LlamaConfig;
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Convert token IDs to embeddings
        let mut hidden_states = self.embed_tokens.forward(input)?;

        // Pass through all decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final layer norm
        let output = self.norm.forward(hidden_states)?;

        Ok(output)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // Legacy interface - use load_from_path instead for new weight loading
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path or load_from_huggingface for enhanced weight loading"
                    .to_string(),
            ),
        )
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let mut total = 0;

        // Embedding parameters
        total += self.embed_tokens.parameter_count();

        // Layer parameters
        for layer in &self.layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.norm.parameter_count();

        total
    }
}

impl LlamaModel {
    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embedding weights
        if let Ok(embed_weights) = loader.load_tensor("model.embed_tokens.weight") {
            self.embed_tokens.set_weight(embed_weights)?;
        }

        // Load layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // Load attention weights
            let attn_prefix = format!("model.layers.{}.self_attn", i);

            if let Ok(q_weights) = loader.load_tensor(&format!("{}.q_proj.weight", attn_prefix)) {
                layer.self_attn.q_proj.set_weight(q_weights)?;
            }
            if let Ok(k_weights) = loader.load_tensor(&format!("{}.k_proj.weight", attn_prefix)) {
                layer.self_attn.k_proj.set_weight(k_weights)?;
            }
            if let Ok(v_weights) = loader.load_tensor(&format!("{}.v_proj.weight", attn_prefix)) {
                layer.self_attn.v_proj.set_weight(v_weights)?;
            }
            if let Ok(o_weights) = loader.load_tensor(&format!("{}.o_proj.weight", attn_prefix)) {
                layer.self_attn.o_proj.set_weight(o_weights)?;
            }

            // Load MLP weights
            let mlp_prefix = format!("model.layers.{}.mlp", i);

            if let Ok(gate_weights) =
                loader.load_tensor(&format!("{}.gate_proj.weight", mlp_prefix))
            {
                layer.mlp.gate_proj.set_weight(gate_weights)?;
            }
            if let Ok(up_weights) = loader.load_tensor(&format!("{}.up_proj.weight", mlp_prefix)) {
                layer.mlp.up_proj.set_weight(up_weights)?;
            }
            if let Ok(down_weights) =
                loader.load_tensor(&format!("{}.down_proj.weight", mlp_prefix))
            {
                layer.mlp.down_proj.set_weight(down_weights)?;
            }

            // Load layer norm weights
            if let Ok(input_norm) =
                loader.load_tensor(&format!("model.layers.{}.input_layernorm.weight", i))
            {
                layer.input_layernorm.set_weight(input_norm)?;
            }
            if let Ok(post_norm) = loader.load_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                i
            )) {
                layer.post_attention_layernorm.set_weight(post_norm)?;
            }
        }

        // Load final norm
        if let Ok(norm_weights) = loader.load_tensor("model.norm.weight") {
            self.norm.set_weight(norm_weights)?;
        }

        loader.close()?;
        Ok(())
    }

    /// Load model weights from HuggingFace Hub
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        let cache_dir = std::env::temp_dir().join("huggingface_cache");
        let model_path = cache_dir.join(format!("models--{}", model_name.replace("/", "--")));

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

        println!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name, model_path
        );

        // Create the model directory
        std::fs::create_dir_all(model_path).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to create model directory: {}",
                e
            ))
        })?;

        // List of essential files for LLaMA models
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

            println!("Attempting to download {}", file_url);

            // Try using curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail on HTTP errors
                    "-o",
                    file_path.to_str().expect("operation failed"),
                    &file_url,
                ])
                .output();

            match curl_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {}", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with curl: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("curl not available: {}", e);
                },
            }

            // Try using wget as fallback
            let wget_result = Command::new("wget")
                .args([
                    "-O",
                    file_path.to_str().expect("operation failed"),
                    &file_url,
                ])
                .output();

            match wget_result {
                Ok(output) if output.status.success() => {
                    println!("Successfully downloaded {} with wget", file_name);
                    continue;
                },
                Ok(output) => {
                    eprintln!(
                        "Failed to download {} with wget: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    println!("wget not available: {}", e);
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

        println!(
            "Successfully downloaded model {} from HuggingFace Hub",
            model_name
        );
        Ok(())
    }

    /// Load weights with lazy loading for large models
    pub fn load_with_lazy_loading(
        &mut self,
        model_path: impl AsRef<std::path::Path>,
    ) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: true,
            streaming: false,
            ..Default::default()
        };

        let loader = auto_create_loader(&model_path, Some(config))?;

        // For lazy loading, we'd store references to the loader and load tensors on-demand
        // This is a simplified example - a full implementation would need more complex state management

        println!("Lazy loading enabled - tensors will be loaded on-demand");

        // List available tensors
        let tensor_names = loader.list_tensors()?;
        println!("Found {} tensors in model", tensor_names.len());

        // For now, still load everything (in a real implementation, this would be truly lazy)
        self.load_from_path(model_path)
    }
}

/// LLaMA for causal language modeling (with LM head)
pub struct LlamaForCausalLM {
    model: LlamaModel,
    lm_head: Linear,
}

impl LlamaForCausalLM {
    pub fn new(config: LlamaConfig) -> Result<Self> {
        let model = LlamaModel::new(config.clone())?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self { model, lm_head })
    }

    pub fn new_with_device(config: LlamaConfig, device: Device) -> Result<Self> {
        let model = LlamaModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self { model, lm_head })
    }

    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        // Load base model weights
        self.model.load_from_path(model_path.as_ref())?;

        // Load lm_head weights
        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        if let Ok(lm_head_weights) = loader.load_tensor("lm_head.weight") {
            self.lm_head.set_weight(lm_head_weights)?;
        }

        loader.close()?;
        Ok(())
    }
}

impl Model for LlamaForCausalLM {
    type Config = LlamaConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input)?;
        let logits = self.lm_head.forward(hidden_states)?;
        Ok(logits)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.model.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.model.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters() + self.lm_head.parameter_count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Layer;

    /// Small test config to keep runtime fast.
    fn small_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: None, // MHA (no GQA)
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            ..LlamaConfig::default()
        }
    }

    /// GQA config: 4 query heads, 2 KV heads (repeat factor = 2).
    fn gqa_config() -> LlamaConfig {
        LlamaConfig {
            vocab_size: 64,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            rms_norm_eps: 1e-5,
            max_position_embeddings: 32,
            rope_theta: 10000.0,
            ..LlamaConfig::default()
        }
    }

    // -----------------------------------------------------------------------
    // RotaryEmbedding tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_rope_output_shape_matches_input() {
        let rope = RotaryEmbedding::new(8, 32, 10000.0);
        let seq = 5usize;
        let dim = 16usize;

        // Build a simple [seq, dim] F32 tensor
        let data: Vec<f32> = (0..seq * dim).map(|i| i as f32 * 0.01).collect();
        let arr = scirs2_core::ndarray::Array2::from_shape_vec((seq, dim), data)
            .expect("shape vec");
        let q = Tensor::F32(arr.into_dyn());
        let k = q.clone();

        let positions: Vec<usize> = (0..seq).collect();
        let (q_out, k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("rope ok");
        assert_eq!(q_out.shape(), q.shape(), "Q shape must be preserved");
        assert_eq!(k_out.shape(), k.shape(), "K shape must be preserved");
    }

    #[test]
    fn test_rope_position_zero_leaves_values_unchanged() {
        // At position 0, angle = 0*freq = 0, cos=1, sin=0 → rotation is identity.
        let rope = RotaryEmbedding::new(4, 32, 10000.0);
        let seq = 1usize;
        let dim = 4usize;
        let data: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0];
        let arr = scirs2_core::ndarray::Array2::from_shape_vec((seq, dim), data)
            .expect("shape vec");
        let q = Tensor::F32(arr.into_dyn());
        let k = q.clone();

        let positions = vec![0usize];
        let (q_out, _k_out) = rope.apply_rotary_emb(&q, &k, &positions).expect("rope ok");

        if let Tensor::F32(out_arr) = q_out {
            for (orig, rotated) in [1.0f32, 2.0, 3.0, 4.0].iter().zip(out_arr.iter()) {
                assert!(
                    (orig - rotated).abs() < 1e-5,
                    "Position 0 should be identity: {} vs {}",
                    orig,
                    rotated
                );
            }
        } else {
            panic!("Expected F32 tensor");
        }
    }

    #[test]
    fn test_rope_non_zero_position_changes_values() {
        // At a non-zero position, values should actually change.
        let rope = RotaryEmbedding::new(4, 32, 10000.0);
        let seq = 2usize;
        let dim = 4usize;
        let data: Vec<f32> = vec![1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
        let arr = scirs2_core::ndarray::Array2::from_shape_vec((seq, dim), data)
            .expect("shape vec");
        let q = Tensor::F32(arr.into_dyn());
        let k = q.clone();
        let positions = vec![0usize, 5]; // position 5 → non-trivial rotation
        let (q_out, _) = rope.apply_rotary_emb(&q, &k, &positions).expect("rope ok");

        if let Tensor::F32(out_arr) = q_out {
            // Row 0 (pos=0) should be unchanged; row 1 (pos=5) should differ.
            // Use slice for dyn array since .row() requires fixed dims.
            let row1: Vec<f32> = out_arr
                .slice(scirs2_core::ndarray::s![1, ..])
                .iter()
                .copied()
                .collect();
            let changed = row1.iter().any(|&v| (v - 1.0).abs() > 1e-4);
            assert!(changed, "Non-zero position must change values");
        } else {
            panic!("Expected F32 tensor");
        }
    }

    // -----------------------------------------------------------------------
    // LlamaAttention tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_attention_forward_output_shape_mha() {
        let config = small_config();
        let attn = LlamaAttention::new(&config).expect("attn");
        let input = Tensor::zeros(&[6, 16]).expect("zeros");
        let output = attn.forward(input).expect("forward");
        assert_eq!(
            output.shape(),
            &[6, 16],
            "MHA attention output shape must match [seq, hidden]"
        );
    }

    #[test]
    fn test_attention_forward_output_shape_gqa() {
        let config = gqa_config();
        let attn = LlamaAttention::new(&config).expect("attn");
        let input = Tensor::zeros(&[4, 16]).expect("zeros");
        let output = attn.forward(input).expect("forward");
        assert_eq!(
            output.shape(),
            &[4, 16],
            "GQA attention output shape must match [seq, hidden]"
        );
    }

    #[test]
    fn test_attention_single_token_does_not_panic() {
        let config = small_config();
        let attn = LlamaAttention::new(&config).expect("attn");
        let input = Tensor::zeros(&[1, 16]).expect("zeros");
        // A single-token input has a trivial causal mask; should succeed.
        let _output = attn.forward(input).expect("single-token forward");
    }

    #[test]
    fn test_attention_forward_values_differ_from_zeros_input() {
        // With randomised weights (default init) and non-zero input, output should not be all-zero.
        let config = small_config();
        let attn = LlamaAttention::new(&config).expect("attn");
        // Use a non-trivial input (ones)
        let data: Vec<f32> = vec![1.0f32; 3 * 16];
        let arr = scirs2_core::ndarray::ArrayD::from_shape_vec(vec![3, 16], data).expect("arr");
        let input = Tensor::F32(arr);
        let output = attn.forward(input).expect("forward");
        if let Tensor::F32(out_arr) = output {
            let all_zero = out_arr.iter().all(|&v| v == 0.0);
            assert!(!all_zero, "Attention output should not be all-zero for non-zero input");
        }
    }

    // -----------------------------------------------------------------------
    // LlamaDecoderLayer smoke test
    // -----------------------------------------------------------------------

    #[test]
    fn test_decoder_layer_forward_shape() {
        let config = small_config();
        let layer = LlamaDecoderLayer::new(&config).expect("layer");
        let input = Tensor::zeros(&[5, 16]).expect("zeros");
        let output = layer.forward(input).expect("forward");
        assert_eq!(output.shape(), &[5, 16]);
    }
}
