use crate::qwen::config::{QwenConfig, RopeScaling};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::silu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Qwen RMSNorm implementation
pub struct QwenRMSNorm {
    weight: Tensor,
    eps: f32,
}

impl QwenRMSNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }
}

impl Layer for QwenRMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let mean_sq = arr.iter().map(|x| x * x).sum::<f32>() / arr.len() as f32;
                let rms = (mean_sq + self.eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);

                match &self.weight {
                    Tensor::F32(weight_arr) => {
                        let result = &normalized * weight_arr;
                        Ok(Tensor::F32(result))
                    },
                    _ => Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported weight tensor type for QwenRMSNorm",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for QwenRMSNorm",
            )),
        }
    }
}

impl QwenRMSNorm {
    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

/// Qwen Rotary Position Embedding with enhanced scaling
pub struct QwenRotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
    pub scaling: Option<RopeScaling>,
}

impl QwenRotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32, scaling: Option<RopeScaling>) -> Self {
        Self {
            dim,
            max_seq_len,
            base,
            scaling,
        }
    }

    /// Resolve `(base, freq_scale)` for the configured `rope_scaling`, given
    /// the largest position seen in this call.
    ///
    /// - No scaling: `(base, 1.0)`.
    /// - `"linear"` (position interpolation, Chen et al. 2023): frequencies
    ///   are divided by `scaling_factor`, equivalent to compressing
    ///   positions by the same factor.
    /// - `"dynamic"` (NTK-aware dynamic scaling, /u/bloc97 and /u/emozilla;
    ///   matches HF `transformers`' `_compute_dynamic_ntk_parameters`):
    ///   `base' = base * ((factor * seq_len / max_seq_len) - (factor - 1)) ^ (dim / (dim - 2))`,
    ///   where `seq_len = max(max_position_seen + 1, max_seq_len)` — a no-op
    ///   until the sequence actually exceeds `max_seq_len`.
    /// - Any other `scaling_type` is rejected with an explicit error rather
    ///   than silently applying an unverified formula.
    fn effective_base_and_scale(&self, max_position_seen: usize) -> Result<(f32, f32)> {
        let Some(scaling) = &self.scaling else {
            return Ok((self.base, 1.0));
        };
        match scaling.scaling_type.as_str() {
            "linear" => {
                if scaling.scaling_factor == 0.0 {
                    return Err(tensor_op_error(
                        "qwen_rope",
                        "rope_scaling.scaling_factor must be non-zero for \"linear\" scaling",
                    ));
                }
                Ok((self.base, 1.0 / scaling.scaling_factor))
            },
            "dynamic" => {
                if self.dim <= 2 {
                    return Err(tensor_op_error(
                        "qwen_rope",
                        "\"dynamic\" rope_scaling requires head_dim > 2",
                    ));
                }
                let seq_len = ((max_position_seen + 1).max(self.max_seq_len)) as f32;
                let factor = scaling.scaling_factor;
                let dim = self.dim as f32;
                let ratio = factor * seq_len / self.max_seq_len as f32 - (factor - 1.0);
                if ratio <= 0.0 {
                    return Err(tensor_op_error(
                        "qwen_rope",
                        "\"dynamic\" rope_scaling produced a non-positive base ratio",
                    ));
                }
                let adjusted_base = self.base * ratio.powf(dim / (dim - 2.0));
                Ok((adjusted_base, 1.0))
            },
            other => Err(tensor_op_error(
                "qwen_rope",
                format!(
                    "unsupported rope_scaling.scaling_type \"{other}\" (supported: \"linear\", \"dynamic\")"
                ),
            )),
        }
    }

    /// Apply rotary embedding to query and key tensors with Qwen's RoPE
    /// scaling. `q`/`k` have shape `[seq_len, heads * head_dim]`; the head
    /// count is inferred independently for each tensor (so `q` and `k` may
    /// have a different number of heads under grouped-query attention), and
    /// every head is rotated, not just the first.
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                if self.dim == 0 {
                    return Err(tensor_op_error("qwen_rope", "head_dim must be > 0"));
                }
                let q_shape = q_arr.shape().to_vec();
                let k_shape = k_arr.shape().to_vec();
                let q_last = *q_shape
                    .last()
                    .ok_or_else(|| tensor_op_error("qwen_rope", "q tensor has no dimensions"))?;
                let k_last = *k_shape
                    .last()
                    .ok_or_else(|| tensor_op_error("qwen_rope", "k tensor has no dimensions"))?;
                if !q_last.is_multiple_of(self.dim) || !k_last.is_multiple_of(self.dim) {
                    return Err(tensor_op_error(
                        "qwen_rope",
                        format!(
                            "last dim (q={q_last}, k={k_last}) must be a multiple of head_dim={}",
                            self.dim
                        ),
                    ));
                }
                let q_heads = q_last / self.dim;
                let k_heads = k_last / self.dim;

                let mut q_data = q_arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen_rope", "q tensor not contiguous"))?
                    .to_vec();
                let mut k_data = k_arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen_rope", "k tensor not contiguous"))?
                    .to_vec();

                let seq_len_q = q_data.len() / q_last.max(1);
                let seq_len_k = k_data.len() / k_last.max(1);
                if seq_len_q != position_ids.len() || seq_len_k != position_ids.len() {
                    return Err(tensor_op_error(
                        "qwen_rope",
                        "position_ids length must match the sequence length of q and k",
                    ));
                }

                let max_position_seen = position_ids.iter().copied().max().unwrap_or(0);
                let (base, freq_scale) = self.effective_base_and_scale(max_position_seen)?;

                apply_rope_rotate_half(
                    &mut q_data,
                    q_heads,
                    self.dim,
                    base,
                    freq_scale,
                    position_ids,
                );
                apply_rope_rotate_half(
                    &mut k_data,
                    k_heads,
                    self.dim,
                    base,
                    freq_scale,
                    position_ids,
                );

                Ok((
                    Tensor::from_vec(q_data, &q_shape)?,
                    Tensor::from_vec(k_data, &k_shape)?,
                ))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for RoPE",
            )),
        }
    }
}

/// Rotate `data` (row-major, shape `[seq_len, num_heads * head_dim]`) in
/// place using the "rotate-half" RoPE convention, scaled by `freq_scale`
/// (linear position-interpolation scaling multiplies frequencies by
/// `1/scaling_factor`; pass `1.0` for no scaling).
fn apply_rope_rotate_half(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    base: f32,
    freq_scale: f32,
    position_ids: &[usize],
) {
    let half = head_dim / 2;
    if half == 0 {
        return;
    }
    let row_width = num_heads * head_dim;
    for (row, &pos) in position_ids.iter().enumerate() {
        let row_off = row * row_width;
        for h in 0..num_heads {
            let head_off = row_off + h * head_dim;
            for i in 0..half {
                let freq = freq_scale / base.powf(2.0 * i as f32 / head_dim as f32);
                let angle = pos as f32 * freq;
                let cos_v = angle.cos();
                let sin_v = angle.sin();
                let x1 = data[head_off + i];
                let x2 = data[head_off + i + half];
                data[head_off + i] = x1 * cos_v - x2 * sin_v;
                data[head_off + i + half] = x1 * sin_v + x2 * cos_v;
            }
        }
    }
}

/// Qwen MLP layer with SwiGLU activation
pub struct QwenMLP {
    gate_proj: Linear, // Gating projection
    up_proj: Linear,   // Up projection
    down_proj: Linear, // Down projection
}

impl QwenMLP {
    pub fn new(config: &QwenConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &QwenConfig, device: Device) -> Result<Self> {
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

impl Layer for QwenMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply SiLU to gate output (SwiGLU activation)
        let gate_activated = silu(&gate_output)?;

        // Element-wise multiply gate and up outputs
        let combined = match (&gate_activated, &up_output) {
            (Tensor::F32(gate_arr), Tensor::F32(up_arr)) => Ok(Tensor::F32(gate_arr * up_arr)),
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for Qwen MLP",
            )),
        }?;

        // Apply down projection
        self.down_proj.forward(combined)
    }
}

/// Qwen Attention layer with grouped-query attention and sliding window support
pub struct QwenAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: QwenRotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f32,
    use_sliding_window: bool,
    sliding_window: Option<usize>,
}

impl QwenAttention {
    pub fn new(config: &QwenConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &QwenConfig, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();
        let scaling = 1.0 / (head_dim as f32).sqrt();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            false,
            device,
        );
        let k_proj =
            Linear::new_with_device(config.hidden_size, num_kv_heads * head_dim, false, device);
        let v_proj =
            Linear::new_with_device(config.hidden_size, num_kv_heads * head_dim, false, device);
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            false,
            device,
        );

        let rotary_emb = QwenRotaryEmbedding::new(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling.clone(),
        );

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
            scaling,
            use_sliding_window: config.use_sliding_window,
            sliding_window: config.sliding_window,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
        // Note: RotaryEmbedding doesn't have learnable parameters
    }
}

impl Layer for QwenAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Real scaled dot-product attention: RoPE (with optional NTK/linear
    /// scaling), grouped-query `Q @ K^T`, causal masking combined with a
    /// sliding window when `use_sliding_window` is set, softmax, and `@ V`.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = shape[shape.len() - 2];

        // Project to Q, K, V
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // Generate position IDs
        let position_ids: Vec<usize> = (0..seq_len).collect();

        // Apply rotary embedding
        let (q_rope, k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?;

        match (&q_rope, &k_rope, &v) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr), Tensor::F32(v_arr)) => {
                let q_data = q_arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen_attn", "q tensor not contiguous"))?;
                let k_data = k_arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen_attn", "k tensor not contiguous"))?;
                let v_data = v_arr
                    .as_slice()
                    .ok_or_else(|| tensor_op_error("qwen_attn", "v tensor not contiguous"))?;

                if self.num_heads == 0
                    || self.num_kv_heads == 0
                    || !self.num_heads.is_multiple_of(self.num_kv_heads)
                {
                    return Err(tensor_op_error(
                        "qwen_attn",
                        "num_heads must be a positive multiple of num_kv_heads",
                    ));
                }
                let q_width = self.num_heads * self.head_dim;
                let kv_width = self.num_kv_heads * self.head_dim;
                if q_data.len() != seq_len * q_width {
                    return Err(tensor_op_error("qwen_attn", "unexpected q tensor size"));
                }
                if k_data.len() != seq_len * kv_width || v_data.len() != seq_len * kv_width {
                    return Err(tensor_op_error(
                        "qwen_attn",
                        "unexpected k/v tensor size for the configured num_kv_heads",
                    ));
                }

                let window = if self.use_sliding_window {
                    match self.sliding_window {
                        Some(0) | None => {
                            return Err(tensor_op_error(
                                "qwen_attn",
                                "use_sliding_window is set but sliding_window is None or 0",
                            ))
                        },
                        Some(w) => Some(w),
                    }
                } else {
                    None
                };

                let group = self.num_heads / self.num_kv_heads;
                let mut out = vec![0f32; seq_len * q_width];
                for h in 0..self.num_heads {
                    let kv_h = h / group;
                    for i in 0..seq_len {
                        let q_off = i * q_width + h * self.head_dim;
                        let mut scores = Vec::with_capacity(i + 1);
                        let mut key_positions = Vec::with_capacity(i + 1);
                        for j in 0..=i {
                            if let Some(w) = window {
                                if i - j >= w {
                                    continue;
                                }
                            }
                            let k_off = j * kv_width + kv_h * self.head_dim;
                            let dot: f32 = (0..self.head_dim)
                                .map(|d| q_data[q_off + d] * k_data[k_off + d])
                                .sum();
                            scores.push(dot * self.scaling);
                            key_positions.push(j);
                        }
                        let max_val = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut weights = vec![0f32; scores.len()];
                        let mut sum = 0f32;
                        for (idx, &s) in scores.iter().enumerate() {
                            let e = (s - max_val).exp();
                            weights[idx] = e;
                            sum += e;
                        }
                        let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                        let out_off = i * q_width + h * self.head_dim;
                        for (idx, &j) in key_positions.iter().enumerate() {
                            let wn = weights[idx] * inv_sum;
                            let v_off = j * kv_width + kv_h * self.head_dim;
                            for d in 0..self.head_dim {
                                out[out_off + d] += wn * v_data[v_off + d];
                            }
                        }
                    }
                }

                let attended = Tensor::from_vec(out, &[seq_len, q_width])?;
                self.o_proj.forward(attended)
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for Qwen attention",
            )),
        }
    }
}

/// Qwen decoder layer
pub struct QwenDecoderLayer {
    self_attn: QwenAttention,
    mlp: QwenMLP,
    input_layernorm: QwenRMSNorm,
    post_attention_layernorm: QwenRMSNorm,
}

impl QwenDecoderLayer {
    pub fn new(config: &QwenConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &QwenConfig, device: Device) -> Result<Self> {
        let self_attn = QwenAttention::new_with_device(config, device)?;
        let mlp = QwenMLP::new_with_device(config, device)?;
        let input_layernorm = QwenRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = QwenRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

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

impl Layer for QwenDecoderLayer {
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

/// Qwen model
pub struct QwenModel {
    config: QwenConfig,
    embed_tokens: Embedding,
    layers: Vec<QwenDecoderLayer>,
    norm: QwenRMSNorm,
}

impl QwenModel {
    pub fn new(config: QwenConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: QwenConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(QwenDecoderLayer::new_with_device(&config, device)?);
        }

        let norm = QwenRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    /// Create a Qwen model from a pretrained model name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        use trustformers_core::errors::invalid_config;

        let config = match name {
            "qwen2-0.5b" => QwenConfig::qwen2_0_5b(),
            "qwen2-1.5b" => QwenConfig::qwen2_1_5b(),
            "qwen2-7b" => QwenConfig::qwen2_7b(),
            "qwen2-72b" => QwenConfig::qwen2_72b(),
            "qwen2.5-7b" => QwenConfig::qwen2_5_7b(),
            "qwen2.5-14b" => QwenConfig::qwen2_5_14b(),
            "qwen2.5-32b" => QwenConfig::qwen2_5_32b(),
            "qwen2.5-72b" => QwenConfig::qwen2_5_72b(),
            "qwen2.5-coder-7b" => QwenConfig::qwen2_5_coder_7b(),
            _ => {
                return Err(invalid_config(
                    "pretrained_model",
                    format!("Unknown pretrained model: {}", name),
                ))
            },
        };
        Self::new(config)
    }
}

impl Model for QwenModel {
    type Config = QwenConfig;
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

        // Count embedding parameters
        total += self.embed_tokens.parameter_count();

        // Count parameters in each decoder layer
        for layer in &self.layers {
            total += layer.parameter_count();
        }

        // Final norm parameters
        total += self.norm.parameter_count();

        total
    }
}

/// Qwen for causal language modeling
pub struct QwenForCausalLM {
    model: QwenModel,
    lm_head: Linear,
}

impl QwenForCausalLM {
    pub fn new(config: QwenConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: QwenConfig, device: Device) -> Result<Self> {
        let model = QwenModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self { model, lm_head })
    }

    /// Create a Qwen for causal LM model from a pretrained model name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let model = QwenModel::from_pretrained_name(name)?;
        let config = model.get_config().clone();
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self { model, lm_head })
    }

    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embeddings
        if let Ok(token_embeddings) = loader.load_tensor("model.embed_tokens.weight") {
            self.model.embed_tokens.set_weight(token_embeddings)?;
        }

        // Load transformer layers
        for layer_idx in 0..self.model.config.num_hidden_layers {
            let layer_prefix = format!("model.layers.{}", layer_idx);

            if let Some(layer) = self.model.layers.get_mut(layer_idx) {
                // Load attention weights
                if let Ok(q_weight) =
                    loader.load_tensor(&format!("{}.self_attn.q_proj.weight", layer_prefix))
                {
                    layer.self_attn.q_proj.set_weight(q_weight)?;
                }
                if let Ok(k_weight) =
                    loader.load_tensor(&format!("{}.self_attn.k_proj.weight", layer_prefix))
                {
                    layer.self_attn.k_proj.set_weight(k_weight)?;
                }
                if let Ok(v_weight) =
                    loader.load_tensor(&format!("{}.self_attn.v_proj.weight", layer_prefix))
                {
                    layer.self_attn.v_proj.set_weight(v_weight)?;
                }
                if let Ok(o_weight) =
                    loader.load_tensor(&format!("{}.self_attn.o_proj.weight", layer_prefix))
                {
                    layer.self_attn.o_proj.set_weight(o_weight)?;
                }

                // Load MLP weights
                if let Ok(gate_weight) =
                    loader.load_tensor(&format!("{}.mlp.gate_proj.weight", layer_prefix))
                {
                    layer.mlp.gate_proj.set_weight(gate_weight)?;
                }
                if let Ok(up_weight) =
                    loader.load_tensor(&format!("{}.mlp.up_proj.weight", layer_prefix))
                {
                    layer.mlp.up_proj.set_weight(up_weight)?;
                }
                if let Ok(down_weight) =
                    loader.load_tensor(&format!("{}.mlp.down_proj.weight", layer_prefix))
                {
                    layer.mlp.down_proj.set_weight(down_weight)?;
                }

                // Load normalization weights
                if let Ok(input_norm) =
                    loader.load_tensor(&format!("{}.input_layernorm.weight", layer_prefix))
                {
                    layer.input_layernorm.weight = input_norm;
                }
                if let Ok(post_norm) =
                    loader.load_tensor(&format!("{}.post_attention_layernorm.weight", layer_prefix))
                {
                    layer.post_attention_layernorm.weight = post_norm;
                }
            }
        }

        // Load final norm and LM head
        if let Ok(norm_weight) = loader.load_tensor("model.norm.weight") {
            self.model.norm.weight = norm_weight;
        }
        if let Ok(lm_head_weight) = loader.load_tensor("lm_head.weight") {
            self.lm_head.set_weight(lm_head_weight)?;
        }

        Ok(())
    }

    /// Load model weights from HuggingFace Hub or local cache
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        use std::path::PathBuf;

        // Try to find model in HuggingFace cache directory
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{}/.cache/huggingface/hub", home)
            });

        let model_base_path =
            PathBuf::from(cache_dir).join(format!("models--{}", model_name.replace("/", "--")));
        let model_path = model_base_path.join("snapshots");

        if model_path.exists() {
            self.load_from_path(&model_path)
        } else {
            // Attempt to download the model from HuggingFace Hub
            self.download_from_huggingface_hub(model_name, &model_base_path)?;
            self.load_from_path(&model_path)
        }
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface_hub(
        &self,
        model_name: &str,
        model_base_path: &std::path::Path,
    ) -> Result<()> {
        use std::process::Command;

        tracing::info!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name,
            model_base_path
        );

        // Create the model directory and snapshots subdirectory
        let snapshots_path = model_base_path.join("snapshots").join("main");
        std::fs::create_dir_all(&snapshots_path).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to create model directory: {}",
                e
            ))
        })?;

        // List of essential files for Qwen models
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
            let file_path = snapshots_path.join(file_name);

            tracing::info!("Attempting to download {}", file_url);

            // Convert path to string once for both commands
            let file_path_str = file_path.to_str().ok_or_else(|| {
                trustformers_core::errors::TrustformersError::invalid_config(format!(
                    "Invalid UTF-8 in path: {:?}",
                    file_path
                ))
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
            ..Default::default()
        };

        let _loader = auto_create_loader(model_path.as_ref(), Some(config))?;

        // Store the loader in the model for later use
        // For now, just perform regular loading
        self.load_from_path(model_path)
    }
}

impl Model for QwenForCausalLM {
    type Config = QwenConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input)?;
        let logits = self.lm_head.forward(hidden_states)?;
        Ok(logits)
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
        self.model.get_config()
    }

    fn num_parameters(&self) -> usize {
        // Count parameters in the base model
        let mut total = self.model.num_parameters();

        // Add language model head parameters
        total += self.lm_head.parameter_count();

        total
    }
}
