use crate::stablelm::config::StableLMConfig;
use scirs2_core::ndarray::{Array1, Array2, ArrayD, Axis, IxDyn}; // SciRS2 Integration Policy (Array2 in tests)
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, Linear},
    ops::activations::{silu, swiglu},
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Root Mean Square Layer Normalization
pub struct RMSNorm {
    weight: Tensor,
    eps: f32,
    device: Device,
}

impl RMSNorm {
    pub fn new(hidden_size: usize, eps: f32) -> Result<Self> {
        Self::new_with_device(hidden_size, eps, Device::CPU)
    }

    pub fn new_with_device(hidden_size: usize, eps: f32, device: Device) -> Result<Self> {
        let weight = Tensor::ones(&[hidden_size])?.to_device_enum(&device)?;
        Ok(Self {
            weight,
            eps,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.shape().iter().product()
    }

    /// The learned per-channel gain.
    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    /// Install a checkpoint gain vector.
    ///
    /// The shape is checked against the norm this layer was built for: silently
    /// accepting a mis-shaped vector would make `forward` fail far away from the
    /// call that actually got it wrong, and skipping the load entirely (as the
    /// loader used to do) leaves an all-ones gain masquerading as trained
    /// weights.
    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        let expected: usize = self.weight.shape().iter().product();
        let got: usize = weight.shape().iter().product();
        if got != expected {
            return Err(tensor_op_error(
                "RMSNorm::set_weight",
                format!(
                    "expected {expected} gain values, got {got} (shape {:?})",
                    weight.shape()
                ),
            ));
        }
        self.weight = weight.to_device_enum(&self.device)?;
        Ok(())
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    /// Normalise **each hidden vector independently**.
    ///
    /// RMSNorm is defined per token: `x / sqrt(mean(x_i^2) + eps) * weight`,
    /// where the mean runs over the last (hidden) axis only. Averaging over the
    /// whole tensor instead would leak information across tokens and batches.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let arr = match &input {
            Tensor::F32(arr) => arr,
            _ => {
                return Err(tensor_op_error(
                    "RMSNorm::forward",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };
        let weight_arr = match &self.weight {
            Tensor::F32(weight_arr) => weight_arr,
            _ => {
                return Err(tensor_op_error(
                    "RMSNorm::forward",
                    "Unsupported weight tensor type".to_string(),
                ))
            },
        };

        let hidden = *arr.shape().last().ok_or_else(|| {
            tensor_op_error(
                "RMSNorm::forward",
                "input must have at least one axis".to_string(),
            )
        })?;
        if weight_arr.len() != hidden {
            return Err(tensor_op_error(
                "RMSNorm::forward",
                format!(
                    "weight length {} does not match hidden size {hidden}",
                    weight_arr.len()
                ),
            ));
        }

        let weight: Vec<f32> = weight_arr.iter().copied().collect();
        let data: Vec<f32> = arr.iter().copied().collect();
        let mut normalized = Vec::with_capacity(data.len());
        for chunk in data.chunks(hidden) {
            let mean_sq = chunk.iter().map(|x| x * x).sum::<f32>() / hidden as f32;
            let inv_rms = 1.0 / (mean_sq + self.eps).sqrt();
            for (value, w) in chunk.iter().zip(weight.iter()) {
                normalized.push(value * inv_rms * w);
            }
        }

        let out = ArrayD::from_shape_vec(arr.raw_dim(), normalized).map_err(|e| {
            tensor_op_error("RMSNorm::forward", format!("failed to rebuild tensor: {e}"))
        })?;
        Ok(Tensor::F32(out))
    }
}

/// Rotary Position Embeddings with partial rotary factor
pub struct RotaryEmbedding {
    sin_cached: Tensor,
    cos_cached: Tensor,
    max_seq_len: usize,
    head_dim: usize,
    #[allow(dead_code)]
    base: f32,
    partial_rotary_factor: f32,
    device: Device,
}

impl RotaryEmbedding {
    pub fn new(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        partial_rotary_factor: f32,
    ) -> Result<Self> {
        Self::new_with_device(
            head_dim,
            max_seq_len,
            base,
            partial_rotary_factor,
            Device::CPU,
        )
    }

    pub fn new_with_device(
        head_dim: usize,
        max_seq_len: usize,
        base: f32,
        partial_rotary_factor: f32,
        device: Device,
    ) -> Result<Self> {
        let rotary_dim = ((head_dim as f32) * partial_rotary_factor) as usize;

        // Pre-compute sin and cos values
        let inv_freq = Array1::range(0.0, rotary_dim as f32, 2.0)
            .mapv(|i| 1.0 / base.powf(i / rotary_dim as f32));

        let t = Array1::range(0.0, max_seq_len as f32, 1.0);
        let freqs = t.view().insert_axis(Axis(1)).dot(&inv_freq.view().insert_axis(Axis(0)));

        let sin_arr =
            Array2::from_shape_fn((max_seq_len, rotary_dim / 2), |(i, j)| freqs[[i, j]].sin());
        let cos_arr =
            Array2::from_shape_fn((max_seq_len, rotary_dim / 2), |(i, j)| freqs[[i, j]].cos());

        let sin_cached = Tensor::F32(sin_arr.into_dyn()).to_device_enum(&device)?;
        let cos_cached = Tensor::F32(cos_arr.into_dyn()).to_device_enum(&device)?;

        Ok(Self {
            sin_cached,
            cos_cached,
            max_seq_len,
            head_dim,
            base,
            partial_rotary_factor,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Number of head dimensions that are actually rotated (`partial_rotary_factor`).
    pub fn rotary_dim(&self) -> usize {
        (((self.head_dim as f32) * self.partial_rotary_factor) as usize).min(self.head_dim)
    }

    /// Rotate a flat `[batch, seq_len, num_heads, head_dim]` buffer in place.
    ///
    /// The rotation follows the HuggingFace `rotate_half` convention used by
    /// StableLM: for the first `rotary_dim` channels the pair `(i, i + rotary_dim/2)`
    /// is rotated by the angle cached for the token's position; channels beyond
    /// `rotary_dim` are left untouched (partial rotary embeddings).
    ///
    /// `num_heads` is a parameter rather than an assumption, so query heads and
    /// (fewer) key/value heads can both be rotated correctly under GQA.
    pub fn rotate_in_place(
        &self,
        data: &mut [f32],
        batch: usize,
        seq_len: usize,
        num_heads: usize,
        head_dim: usize,
    ) -> Result<()> {
        if head_dim != self.head_dim {
            return Err(tensor_op_error(
                "RotaryEmbedding::rotate_in_place",
                format!(
                    "head_dim {head_dim} does not match the cached head_dim {}",
                    self.head_dim
                ),
            ));
        }
        if seq_len > self.max_seq_len {
            return Err(tensor_op_error(
                "RotaryEmbedding::rotate_in_place",
                format!(
                    "sequence length {seq_len} exceeds max_position_embeddings {}",
                    self.max_seq_len
                ),
            ));
        }
        let expected = batch * seq_len * num_heads * head_dim;
        if data.len() != expected {
            return Err(tensor_op_error(
                "RotaryEmbedding::rotate_in_place",
                format!("expected {expected} elements, got {}", data.len()),
            ));
        }

        let half = self.rotary_dim() / 2;
        if half == 0 {
            return Ok(());
        }

        let (cos_arr, sin_arr) = match (&self.cos_cached, &self.sin_cached) {
            (Tensor::F32(cos_arr), Tensor::F32(sin_arr)) => (cos_arr, sin_arr),
            _ => {
                return Err(tensor_op_error(
                    "RotaryEmbedding::rotate_in_place",
                    "rotary caches must be F32".to_string(),
                ))
            },
        };

        for b in 0..batch {
            for pos in 0..seq_len {
                for head in 0..num_heads {
                    let base = ((b * seq_len + pos) * num_heads + head) * head_dim;
                    for i in 0..half {
                        let cos_val = cos_arr[[pos, i]];
                        let sin_val = sin_arr[[pos, i]];
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

    /// Apply RoPE to query and key tensors shaped `[batch, seq_len, heads, head_dim]`.
    ///
    /// Query and key may carry a different number of heads (grouped-query
    /// attention); each tensor is rotated according to its own head count.
    pub fn forward(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let rotate = |tensor: &Tensor| -> Result<Tensor> {
            let arr = match tensor {
                Tensor::F32(arr) => arr,
                _ => {
                    return Err(tensor_op_error(
                        "RotaryEmbedding::forward",
                        "Unsupported tensor type".to_string(),
                    ))
                },
            };
            let shape = arr.shape().to_vec();
            if shape.len() != 4 {
                return Err(tensor_op_error(
                    "RotaryEmbedding::forward",
                    format!("expected [batch, seq_len, heads, head_dim], got {shape:?}"),
                ));
            }
            let mut data: Vec<f32> = arr.iter().copied().collect();
            self.rotate_in_place(&mut data, shape[0], shape[1], shape[2], shape[3])?;
            let rotated = ArrayD::from_shape_vec(IxDyn(&shape), data).map_err(|e| {
                tensor_op_error(
                    "RotaryEmbedding::forward",
                    format!("failed to rebuild tensor: {e}"),
                )
            })?;
            Ok(Tensor::F32(rotated))
        };

        let _ = seq_len; // the sequence length is taken from the tensor shape
        Ok((rotate(q)?, rotate(k)?))
    }
}

/// Multi-Head Attention with optional grouped-query attention
pub struct StableLMAttention {
    #[allow(dead_code)]
    config: StableLMConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: RotaryEmbedding,
    #[allow(dead_code)]
    head_dim: usize,
    num_heads: usize,
    num_kv_heads: usize,
    device: Device,
}

impl StableLMAttention {
    pub fn new(config: &StableLMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads.unwrap_or(num_heads);
        let head_dim = hidden_size / num_heads;

        let q_proj =
            Linear::new_with_device(hidden_size, hidden_size, config.attention_bias, device);
        let k_proj = Linear::new_with_device(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            device,
        );
        let v_proj = Linear::new_with_device(
            hidden_size,
            num_kv_heads * head_dim,
            config.attention_bias,
            device,
        );
        let o_proj =
            Linear::new_with_device(hidden_size, hidden_size, config.attention_bias, device);

        let rotary_emb = RotaryEmbedding::new_with_device(
            head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.partial_rotary_factor,
            device,
        )?;

        Ok(Self {
            config: config.clone(),
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            rotary_emb,
            head_dim,
            num_heads,
            num_kv_heads,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Repeat each key/value head `n_rep` times along the head axis.
    ///
    /// `hidden_states` must be 4-D `[batch, seq_len, num_kv_heads, head_dim]`.
    /// The result is `[batch, seq_len, num_kv_heads * n_rep, head_dim]` in which
    /// KV head `i` occupies output heads `i*n_rep .. (i+1)*n_rep` — the same
    /// grouping HuggingFace's `repeat_kv` produces, so query head `q` pairs with
    /// KV head `q / n_rep`.
    pub fn repeat_kv(&self, hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 0 {
            return Err(tensor_op_error(
                "StableLMAttention::repeat_kv",
                "n_rep must be >= 1".to_string(),
            ));
        }
        if n_rep == 1 {
            return Ok(hidden_states.clone());
        }

        match hidden_states {
            Tensor::F32(arr) => {
                let shape = arr.shape().to_vec();
                if shape.len() != 4 {
                    return Err(tensor_op_error(
                        "StableLMAttention::repeat_kv",
                        format!("expected [batch, seq_len, num_kv_heads, head_dim], got {shape:?}"),
                    ));
                }
                let (batch, seq_len, num_kv_heads, head_dim) =
                    (shape[0], shape[1], shape[2], shape[3]);

                let data: Vec<f32> = arr.iter().copied().collect();
                let mut repeated = Vec::with_capacity(data.len() * n_rep);
                for b in 0..batch {
                    for t in 0..seq_len {
                        for kv_head in 0..num_kv_heads {
                            let base = ((b * seq_len + t) * num_kv_heads + kv_head) * head_dim;
                            for _ in 0..n_rep {
                                repeated.extend_from_slice(&data[base..base + head_dim]);
                            }
                        }
                    }
                }

                let out = ArrayD::from_shape_vec(
                    IxDyn(&[batch, seq_len, num_kv_heads * n_rep, head_dim]),
                    repeated,
                )
                .map_err(|e| {
                    tensor_op_error(
                        "StableLMAttention::repeat_kv",
                        format!("failed to build repeated tensor: {e}"),
                    )
                })?;
                Ok(Tensor::F32(out))
            },
            _ => Err(tensor_op_error(
                "StableLMAttention::repeat_kv",
                "Unsupported tensor type".to_string(),
            )),
        }
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.o_proj.parameter_count()
        // Note: rotary_emb typically doesn't have trainable parameters
    }
}

impl Layer for StableLMAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Causal grouped-query self-attention.
    ///
    /// Accepts `[seq_len, hidden_size]` or `[batch, seq_len, hidden_size]` and
    /// returns a tensor of the same rank. The computation is the real thing:
    /// `softmax(mask(Q Kᵀ) / sqrt(head_dim)) V` with RoPE applied to Q and K and
    /// KV heads repeated to match the query heads.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let in_shape = input.shape().to_vec();
        let (batch, seq_len) = match in_shape.len() {
            2 => (1usize, in_shape[0]),
            3 => (in_shape[0], in_shape[1]),
            _ => {
                return Err(tensor_op_error(
                    "StableLMAttention::forward",
                    format!(
                        "expected [seq_len, hidden] or [batch, seq_len, hidden], got {in_shape:?}"
                    ),
                ))
            },
        };
        let hidden = in_shape[in_shape.len() - 1];
        let head_dim = self.head_dim;
        if hidden != self.num_heads * head_dim {
            return Err(tensor_op_error(
                "StableLMAttention::forward",
                format!(
                    "input hidden size {hidden} != num_heads {} * head_dim {head_dim}",
                    self.num_heads
                ),
            ));
        }

        // Query, Key, Value projections
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // Reshape to [batch, seq_len, heads, head_dim]
        let q = q.reshape(&[batch, seq_len, self.num_heads, head_dim])?;
        let k = k.reshape(&[batch, seq_len, self.num_kv_heads, head_dim])?;
        let v = v.reshape(&[batch, seq_len, self.num_kv_heads, head_dim])?;

        // Apply rotary embeddings to queries and keys (values are not rotated)
        let (q_rot, k_rot) = self.rotary_emb.forward(&q, &k, seq_len)?;

        // Repeat KV heads if using grouped-query attention
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_repeated = self.repeat_kv(&k_rot, n_rep)?;
        let v_repeated = self.repeat_kv(&v, n_rep)?;

        let q_data = q_rot.data()?;
        let k_data = k_repeated.data()?;
        let v_data = v_repeated.data()?;

        // Scaled dot-product attention with a causal mask.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let num_heads = self.num_heads;
        let mut context = vec![0.0f32; batch * seq_len * num_heads * head_dim];
        let mut scores = vec![0.0f32; seq_len];

        for b in 0..batch {
            for head in 0..num_heads {
                for query_pos in 0..seq_len {
                    let q_base = ((b * seq_len + query_pos) * num_heads + head) * head_dim;

                    // Scores over the causal prefix [0, query_pos].
                    let mut max_score = f32::NEG_INFINITY;
                    for key_pos in 0..=query_pos {
                        let k_base = ((b * seq_len + key_pos) * num_heads + head) * head_dim;
                        let dot: f32 = (0..head_dim)
                            .map(|d| q_data[q_base + d] * k_data[k_base + d])
                            .sum::<f32>()
                            * scale;
                        scores[key_pos] = dot;
                        if dot > max_score {
                            max_score = dot;
                        }
                    }

                    // Softmax over the prefix (numerically stabilised).
                    let mut sum = 0.0f32;
                    for score in scores.iter_mut().take(query_pos + 1) {
                        *score = (*score - max_score).exp();
                        sum += *score;
                    }
                    let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };

                    // Weighted sum of the value vectors.
                    let out_base = ((b * seq_len + query_pos) * num_heads + head) * head_dim;
                    for key_pos in 0..=query_pos {
                        let weight = scores[key_pos] * inv_sum;
                        let v_base = ((b * seq_len + key_pos) * num_heads + head) * head_dim;
                        for d in 0..head_dim {
                            context[out_base + d] += weight * v_data[v_base + d];
                        }
                    }
                }
            }
        }

        // Merge heads back into the hidden dimension and restore the input rank.
        let merged_shape: Vec<usize> = if in_shape.len() == 2 {
            vec![seq_len, hidden]
        } else {
            vec![batch, seq_len, hidden]
        };
        let attn_output = Tensor::from_vec(context, &merged_shape)?;

        // Output projection
        self.o_proj.forward(attn_output)
    }
}

/// MLP with SwiGLU activation
pub struct StableLMMLP {
    config: StableLMConfig,
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    device: Device,
}

impl StableLMMLP {
    pub fn new(config: &StableLMConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Self {
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;

        Self {
            config: config.clone(),
            gate_proj: Linear::new_with_device(
                hidden_size,
                intermediate_size,
                config.mlp_bias,
                device,
            ),
            up_proj: Linear::new_with_device(
                hidden_size,
                intermediate_size,
                config.mlp_bias,
                device,
            ),
            down_proj: Linear::new_with_device(
                intermediate_size,
                hidden_size,
                config.mlp_bias,
                device,
            ),
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.gate_proj.parameter_count()
            + self.up_proj.parameter_count()
            + self.down_proj.parameter_count()
    }
}

impl Layer for StableLMMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let gate = self.gate_proj.forward(input.clone())?;
        let up = self.up_proj.forward(input)?;

        // Apply activation based on config
        let activated = match self.config.hidden_act.as_str() {
            "silu" => {
                let gate_act = silu(&gate)?;
                match (&gate_act, &up) {
                    (Tensor::F32(g), Tensor::F32(u)) => Tensor::F32(g * u),
                    _ => {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Unsupported tensor type".to_string(),
                        ))
                    },
                }
            },
            "swiglu" => swiglu(&gate, &up)?,
            _ => silu(&gate)?, // Default to SiLU
        };

        self.down_proj.forward(activated)
    }
}

/// StableLM Decoder Layer
pub struct StableLMDecoderLayer {
    #[allow(dead_code)]
    config: StableLMConfig,
    self_attn: StableLMAttention,
    mlp: StableLMMLP,
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
    device: Device,
}

impl StableLMDecoderLayer {
    pub fn new(config: &StableLMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            self_attn: StableLMAttention::new_with_device(config, device)?,
            mlp: StableLMMLP::new_with_device(config, device),
            input_layernorm: RMSNorm::new_with_device(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            )?,
            post_attention_layernorm: RMSNorm::new_with_device(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.self_attn.parameter_count()
            + self.mlp.parameter_count()
            + self.input_layernorm.parameter_count()
            + self.post_attention_layernorm.parameter_count()
    }
}

impl Layer for StableLMDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm architecture
        let residual = input.clone();
        let hidden_states = self.input_layernorm.forward(input)?;
        let attn_output = self.self_attn.forward(hidden_states)?;

        // First residual connection
        let hidden_states = match (&residual, &attn_output) {
            (Tensor::F32(r), Tensor::F32(a)) => Tensor::F32(r + a),
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };

        // MLP block
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(hidden_states)?;

        // Second residual connection
        match (&residual, &mlp_output) {
            (Tensor::F32(r), Tensor::F32(m)) => Ok(Tensor::F32(r + m)),
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type".to_string(),
            )),
        }
    }
}

/// StableLM Embeddings
pub struct StableLMEmbeddings {
    word_embeddings: Embedding,
    device: Device,
}

impl StableLMEmbeddings {
    pub fn new(config: &StableLMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Result<Self> {
        Ok(Self {
            word_embeddings: Embedding::new_with_device(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id.map(|x| x as usize),
                device,
            )?,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
    }
}

impl Layer for StableLMEmbeddings {
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.word_embeddings.forward(input)
    }
}

/// StableLM Model Output
#[derive(Debug)]
pub struct StableLMOutputs {
    pub last_hidden_state: Tensor,
}

/// StableLM Base Model
pub struct StableLMModel {
    pub config: StableLMConfig,
    pub embeddings: StableLMEmbeddings,
    pub layers: Vec<StableLMDecoderLayer>,
    pub norm: RMSNorm,
    device: Device,
}

impl StableLMModel {
    pub fn new(config: StableLMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: StableLMConfig, device: Device) -> Result<Self> {
        let embeddings = StableLMEmbeddings::new_with_device(&config, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(StableLMDecoderLayer::new_with_device(&config, device)?);
        }

        let norm = RMSNorm::new_with_device(config.hidden_size, config.rms_norm_eps, device)?;

        Ok(Self {
            config,
            embeddings,
            layers,
            norm,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward_with_outputs(&self, input_ids: &Tensor) -> Result<StableLMOutputs> {
        // Convert tensor to token IDs
        let input_ids_vec = match input_ids {
            Tensor::I64(ref arr) => arr.mapv(|x| x as u32).into_raw_vec_and_offset().0,
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };
        let mut hidden_states = self.embeddings.forward(input_ids_vec)?;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        let last_hidden_state = self.norm.forward(hidden_states)?;

        Ok(StableLMOutputs { last_hidden_state })
    }
}

impl Model for StableLMModel {
    type Config = StableLMConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let outputs = self.forward_with_outputs(&input)?;
        Ok(outputs.last_hidden_state)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
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
        let embeddings_params = self.embeddings.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.norm.parameter_count();

        embeddings_params + layers_params + norm_params
    }
}

/// StableLM Causal LM Output
#[derive(Debug)]
pub struct StableLMCausalLMOutputs {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

/// StableLM for Causal Language Modeling
pub struct StableLMForCausalLM {
    pub model: StableLMModel,
    pub lm_head: Linear,
    device: Device,
}

impl StableLMForCausalLM {
    pub fn new(config: StableLMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: StableLMConfig, device: Device) -> Result<Self> {
        let model = StableLMModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            model,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward_with_outputs(&self, input_ids: &Tensor) -> Result<StableLMCausalLMOutputs> {
        let outputs = self.model.forward_with_outputs(input_ids)?;
        let logits = self.lm_head.forward(outputs.last_hidden_state.clone())?;

        Ok(StableLMCausalLMOutputs {
            logits,
            hidden_states: Some(outputs.last_hidden_state),
        })
    }
}

impl Model for StableLMForCausalLM {
    type Config = StableLMConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let outputs = self.forward_with_outputs(&input)?;
        Ok(outputs.logits)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
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
        self.model.num_parameters() + self.lm_head.parameter_count()
    }
}

impl StableLMForCausalLM {
    /// Load model weights from a directory containing HuggingFace format weights.
    ///
    /// Every parameter this architecture owns is bound, **including** the two
    /// RMSNorm gains per decoder layer and the final norm; an earlier revision
    /// skipped those and left them at their all-ones initialisation, which is
    /// indistinguishable from a trained gain at the type level but is not the
    /// checkpoint's model.
    ///
    /// A missing tensor is an error, not a silent skip: the previous
    /// `if let Ok(..)` chain returned `Ok(())` after binding nothing at all when
    /// the checkpoint used different names, so a caller could run inference on
    /// randomly initialised weights believing the model was loaded.
    ///
    /// `lm_head.weight` is the one genuinely optional entry — StableLM ties the
    /// head to the input embeddings when it is absent, which is what the tied
    /// configuration means rather than a fallback guess.
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoader, WeightLoadingConfig};

        /// Load a tensor that the architecture requires, naming it on failure.
        fn required(loader: &mut dyn WeightLoader, name: &str) -> Result<Tensor> {
            loader.load_tensor(name).map_err(|e| {
                tensor_op_error(
                    "StableLMForCausalLM::load_from_path",
                    format!("checkpoint is missing required tensor `{name}`: {e}"),
                )
            })
        }

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;
        let loader = loader.as_mut();

        let attention_bias = self.model.config.attention_bias;
        let mlp_bias = self.model.config.mlp_bias;

        // Token embeddings (kept for the tied-head case below).
        let embed_weights = required(loader, "model.embed_tokens.weight")?;
        self.model.embeddings.word_embeddings.set_weight(embed_weights.clone())?;

        for (i, layer) in self.model.layers.iter_mut().enumerate() {
            let attn_prefix = format!("model.layers.{i}.self_attn");
            let mlp_prefix = format!("model.layers.{i}.mlp");

            for (name, projection) in [
                ("q_proj", &mut layer.self_attn.q_proj),
                ("k_proj", &mut layer.self_attn.k_proj),
                ("v_proj", &mut layer.self_attn.v_proj),
                ("o_proj", &mut layer.self_attn.o_proj),
            ] {
                projection
                    .set_weight(required(loader, &format!("{attn_prefix}.{name}.weight"))?)?;
                if attention_bias {
                    projection
                        .set_bias(required(loader, &format!("{attn_prefix}.{name}.bias"))?)?;
                }
            }

            for (name, projection) in [
                ("gate_proj", &mut layer.mlp.gate_proj),
                ("up_proj", &mut layer.mlp.up_proj),
                ("down_proj", &mut layer.mlp.down_proj),
            ] {
                projection.set_weight(required(loader, &format!("{mlp_prefix}.{name}.weight"))?)?;
                if mlp_bias {
                    projection.set_bias(required(loader, &format!("{mlp_prefix}.{name}.bias"))?)?;
                }
            }

            layer.input_layernorm.set_weight(required(
                loader,
                &format!("model.layers.{i}.input_layernorm.weight"),
            )?)?;
            layer.post_attention_layernorm.set_weight(required(
                loader,
                &format!("model.layers.{i}.post_attention_layernorm.weight"),
            )?)?;
        }

        self.model.norm.set_weight(required(loader, "model.norm.weight")?)?;

        // Tied heads store no `lm_head.weight`; reuse the embedding matrix.
        match loader.load_tensor("lm_head.weight") {
            Ok(lm_head_weight) => self.lm_head.set_weight(lm_head_weight)?,
            Err(_) => self.lm_head.set_weight(embed_weights)?,
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

        // List of essential files for StableLM models
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
            let wget_result = Command::new("wget").args(["-O", file_path_str, &file_url]).output();

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
}

#[cfg(test)]
mod tests {
    use super::*;
    // Array2 already imported via scirs2_core at top

    #[test]
    fn test_rms_norm() -> Result<()> {
        let norm = RMSNorm::new(768, 1e-5)?;
        let input = Tensor::F32(Array2::ones((2, 768)).into_dyn());
        let output = norm.forward(input);
        assert!(output.is_ok());
        Ok(())
    }

    #[test]
    fn test_rotary_embedding() -> Result<()> {
        let rope = RotaryEmbedding::new(64, 512, 10000.0, 0.25)?;
        assert_eq!(rope.head_dim, 64);
        assert_eq!(rope.max_seq_len, 512);
        assert_eq!(rope.partial_rotary_factor, 0.25);
        Ok(())
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B model creation, run with --ignored
    fn test_stablelm_model_creation() -> Result<()> {
        let config = StableLMConfig::stablelm_3b();
        let model = StableLMModel::new(config.clone())?;

        assert_eq!(model.layers.len(), config.num_hidden_layers);
        assert_eq!(model.config.hidden_size, 2560);
        Ok(())
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B CausalLM, run with --ignored
    fn test_stablelm_causal_lm() -> Result<()> {
        let config = StableLMConfig::stablelm_3b();
        let _model = StableLMForCausalLM::new(config.clone())?;

        // StableLM for CausalLM created successfully - LM head dimensions are internal
        Ok(())
    }

    #[test]
    fn test_grouped_query_attention() -> Result<()> {
        let mut config = StableLMConfig::stablelm_2_1_6b();
        config.num_key_value_heads = Some(4);

        let attn = StableLMAttention::new(&config)?;
        assert_eq!(attn.num_heads, 32);
        assert_eq!(attn.num_kv_heads, 4);

        // Grouped query attention created successfully - projection dimensions are internal
        Ok(())
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B device support, run with --ignored
    fn test_device_support() -> Result<()> {
        let config = StableLMConfig::stablelm_3b();

        // Test CPU device (default)
        let model_cpu = StableLMModel::new(config.clone())?;
        assert_eq!(*model_cpu.device(), Device::CPU);

        // Test explicit CPU device
        let model_cpu_explicit = StableLMModel::new_with_device(config.clone(), Device::CPU)?;
        assert_eq!(*model_cpu_explicit.device(), Device::CPU);

        // Test that all components have the correct device
        assert_eq!(*model_cpu.embeddings.device(), Device::CPU);
        assert_eq!(*model_cpu.norm.device(), Device::CPU);
        for layer in &model_cpu.layers {
            assert_eq!(*layer.device(), Device::CPU);
            assert_eq!(*layer.self_attn.device(), Device::CPU);
            assert_eq!(*layer.mlp.device(), Device::CPU);
        }
        Ok(())
    }

    // ---- Grouped-query attention: repeat_kv must actually repeat ----

    /// Tiny GQA config: 4 query heads, 2 KV heads, head_dim 4.
    fn tiny_gqa_config() -> StableLMConfig {
        StableLMConfig {
            vocab_size: 32,
            hidden_size: 16,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_key_value_heads: Some(2),
            max_position_embeddings: 16,
            partial_rotary_factor: 0.5,
            ..StableLMConfig::default()
        }
    }

    #[test]
    fn test_repeat_kv_repeats_along_head_axis() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;

        // [batch=1, seq_len=2, num_kv_heads=2, head_dim=2]
        let data: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let kv = Tensor::from_vec(data, &[1, 2, 2, 2])?;

        let repeated = attn.repeat_kv(&kv, 2)?;
        assert_eq!(
            repeated.shape(),
            &[1, 2, 4, 2],
            "repeat_kv must grow the head axis by n_rep"
        );

        let out = repeated.data()?;
        // Token 0: kv head 0 = [0,1] -> output heads 0,1; kv head 1 = [2,3] -> heads 2,3.
        assert_eq!(&out[0..2], &[0.0, 1.0]);
        assert_eq!(&out[2..4], &[0.0, 1.0]);
        assert_eq!(&out[4..6], &[2.0, 3.0]);
        assert_eq!(&out[6..8], &[2.0, 3.0]);
        // Token 1: kv head 0 = [4,5]; kv head 1 = [6,7].
        assert_eq!(&out[8..10], &[4.0, 5.0]);
        assert_eq!(&out[10..12], &[4.0, 5.0]);
        assert_eq!(&out[12..14], &[6.0, 7.0]);
        assert_eq!(&out[14..16], &[6.0, 7.0]);
        Ok(())
    }

    #[test]
    fn test_repeat_kv_identity_for_single_repeat() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;
        let data: Vec<f32> = (0..8).map(|i| i as f32 * 0.5).collect();
        let kv = Tensor::from_vec(data.clone(), &[1, 2, 2, 2])?;
        let repeated = attn.repeat_kv(&kv, 1)?;
        assert_eq!(repeated.shape(), &[1, 2, 2, 2]);
        assert_eq!(repeated.data()?, data);
        Ok(())
    }

    #[test]
    fn test_repeat_kv_rejects_non_4d() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;
        let kv = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[2, 2])?;
        assert!(
            attn.repeat_kv(&kv, 2).is_err(),
            "repeat_kv must reject tensors that are not [b, s, kv_heads, head_dim]"
        );
        Ok(())
    }

    // ---- Real attention: shape, causality, cross-token dependence ----

    #[test]
    fn test_gqa_attention_forward_shape() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;
        let seq_len = 3;
        let data: Vec<f32> = (0..seq_len * config.hidden_size)
            .map(|i| ((i % 5) as f32 - 2.0) * 0.1)
            .collect();
        let input = Tensor::from_vec(data, &[seq_len, config.hidden_size])?;
        let output = attn.forward(input)?;
        assert_eq!(output.shape(), &[seq_len, config.hidden_size]);
        Ok(())
    }

    #[test]
    fn test_attention_reads_previous_tokens() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;
        let seq_len = 3;
        let hidden = config.hidden_size;
        let base: Vec<f32> = (0..seq_len * hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let out_a = attn.forward(Tensor::from_vec(base.clone(), &[seq_len, hidden])?)?.data()?;

        // Perturb token 0 only; the last token attends to it, so its output must move.
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().take(hidden) {
            *value += 0.5;
        }
        let out_b = attn.forward(Tensor::from_vec(perturbed, &[seq_len, hidden])?)?.data()?;

        let last = (seq_len - 1) * hidden;
        let diff = out_a[last..]
            .iter()
            .zip(out_b[last..].iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            diff > 1e-6,
            "the last token must attend to earlier tokens (diff {diff})"
        );
        Ok(())
    }

    #[test]
    fn test_attention_is_causal() -> Result<()> {
        let config = tiny_gqa_config();
        let attn = StableLMAttention::new(&config)?;
        let seq_len = 3;
        let hidden = config.hidden_size;
        let base: Vec<f32> = (0..seq_len * hidden).map(|i| ((i % 7) as f32 - 3.0) * 0.1).collect();

        let out_a = attn.forward(Tensor::from_vec(base.clone(), &[seq_len, hidden])?)?.data()?;

        // Perturb the LAST token; earlier positions must be unaffected.
        let mut perturbed = base.clone();
        for value in perturbed.iter_mut().skip((seq_len - 1) * hidden) {
            *value += 0.75;
        }
        let out_b = attn.forward(Tensor::from_vec(perturbed, &[seq_len, hidden])?)?.data()?;

        let prefix = (seq_len - 1) * hidden;
        for i in 0..prefix {
            assert!(
                (out_a[i] - out_b[i]).abs() < 1e-5,
                "causal attention must not let position {} see the future",
                i / hidden
            );
        }
        Ok(())
    }

    /// Reference-math check: with identity projections and RoPE disabled, the
    /// attention output must equal a naive `softmax(x·xᵀ/sqrt(d)) x` computed
    /// directly in the test.
    #[test]
    fn test_attention_matches_naive_reference() -> Result<()> {
        let mut config = tiny_gqa_config();
        config.num_key_value_heads = Some(config.num_attention_heads); // MHA
        config.partial_rotary_factor = 0.0; // disable rotation for an exact reference
        let mut attn = StableLMAttention::new(&config)?;

        let hidden = config.hidden_size;
        let identity: Vec<f32> = (0..hidden * hidden)
            .map(|i| if i / hidden == i % hidden { 1.0 } else { 0.0 })
            .collect();
        for proj in [
            &mut attn.q_proj,
            &mut attn.k_proj,
            &mut attn.v_proj,
            &mut attn.o_proj,
        ] {
            proj.set_weight(Tensor::from_vec(identity.clone(), &[hidden, hidden])?)?;
        }

        let seq_len = 3;
        let head_dim = hidden / config.num_attention_heads;
        let x: Vec<f32> = (0..seq_len * hidden).map(|i| ((i % 9) as f32 - 4.0) * 0.15).collect();
        let output = attn.forward(Tensor::from_vec(x.clone(), &[seq_len, hidden])?)?.data()?;

        // Naive reference implementation.
        let scale = 1.0 / (head_dim as f32).sqrt();
        let mut expected = vec![0.0f32; seq_len * hidden];
        for head in 0..config.num_attention_heads {
            for query_pos in 0..seq_len {
                let q_base = query_pos * hidden + head * head_dim;
                let mut weights = Vec::with_capacity(query_pos + 1);
                for key_pos in 0..=query_pos {
                    let k_base = key_pos * hidden + head * head_dim;
                    let dot: f32 =
                        (0..head_dim).map(|d| x[q_base + d] * x[k_base + d]).sum::<f32>() * scale;
                    weights.push(dot);
                }
                let max = weights.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let exps: Vec<f32> = weights.iter().map(|w| (w - max).exp()).collect();
                let sum: f32 = exps.iter().sum();
                for (key_pos, e) in exps.iter().enumerate() {
                    let weight = e / sum;
                    let v_base = key_pos * hidden + head * head_dim;
                    for d in 0..head_dim {
                        expected[q_base + d] += weight * x[v_base + d];
                    }
                }
            }
        }

        for (i, (got, want)) in output.iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - want).abs() < 1e-5,
                "element {i}: got {got}, expected {want}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rms_norm_normalises_each_token_independently() -> Result<()> {
        let norm = RMSNorm::new(4, 1e-6)?;
        // Two tokens with very different magnitudes.
        let input = Tensor::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 100.0, 100.0, 100.0, 100.0],
            &[2, 4],
        )?;
        let output = norm.forward(input)?.data()?;
        for value in &output {
            assert!(
                (value - 1.0).abs() < 1e-3,
                "each token must be normalised to unit RMS, got {value}"
            );
        }
        Ok(())
    }

    #[test]
    fn test_rotary_rotation_preserves_norm_and_changes_values() -> Result<()> {
        let head_dim = 8;
        let rope = RotaryEmbedding::new(head_dim, 16, 10000.0, 1.0)?;
        let mut data: Vec<f32> = (0..2 * head_dim).map(|i| (i as f32 + 1.0) * 0.1).collect();
        let original = data.clone();
        // [batch=1, seq_len=2, heads=1, head_dim]
        rope.rotate_in_place(&mut data, 1, 2, 1, head_dim)?;

        // Position 0 has angle 0 → identity.
        for i in 0..head_dim {
            assert!((data[i] - original[i]).abs() < 1e-6);
        }
        // Position 1 must actually rotate.
        let changed = (head_dim..2 * head_dim).any(|i| (data[i] - original[i]).abs() > 1e-6);
        assert!(changed, "RoPE must modify the second position");
        // Rotations preserve the vector norm.
        let norm_before: f32 = original[head_dim..].iter().map(|v| v * v).sum::<f32>().sqrt();
        let norm_after: f32 = data[head_dim..].iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm_before - norm_after).abs() < 1e-5);
        Ok(())
    }

    // ── Checkpoint loading ────────────────────────────────────────────────

    /// A temporary HuggingFace-style model directory, removed on drop.
    struct TempModelDir {
        path: std::path::PathBuf,
    }

    impl TempModelDir {
        fn new(label: &str) -> Self {
            use std::sync::atomic::{AtomicU64, Ordering};
            static COUNTER: AtomicU64 = AtomicU64::new(0);
            let unique = COUNTER.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "trustformers_stablelm_{label}_{}_{unique}",
                std::process::id()
            ));
            std::fs::create_dir_all(&path).expect("temp model dir must be creatable");
            Self { path }
        }

        fn write_safetensors(&self, tensors: &[crate::weight_loading::test_support::F32Tensor]) {
            std::fs::write(
                self.path.join("model.safetensors"),
                crate::weight_loading::test_support::build_safetensors(tensors),
            )
            .expect("fixture checkpoint must be writable");
        }
    }

    impl Drop for TempModelDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.path);
        }
    }

    /// Every tensor a StableLM checkpoint for `config` carries.
    fn checkpoint_tensors(
        config: &StableLMConfig,
        with_lm_head: bool,
    ) -> Vec<crate::weight_loading::test_support::F32Tensor> {
        use crate::weight_loading::test_support::F32Tensor;
        let hidden = config.hidden_size;
        let heads = config.num_attention_heads;
        let kv_heads = config.num_key_value_heads.unwrap_or(heads);
        let kv_width = kv_heads * (hidden / heads);
        let inter = config.intermediate_size;

        let mut tensors = vec![F32Tensor::ramp(
            "model.embed_tokens.weight",
            &[config.vocab_size, hidden],
            0.25,
        )];
        for i in 0..config.num_hidden_layers {
            tensors.extend([
                F32Tensor::ramp(
                    &format!("model.layers.{i}.self_attn.q_proj.weight"),
                    &[hidden, hidden],
                    1.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.self_attn.k_proj.weight"),
                    &[kv_width, hidden],
                    2.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.self_attn.v_proj.weight"),
                    &[kv_width, hidden],
                    3.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.self_attn.o_proj.weight"),
                    &[hidden, hidden],
                    4.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.mlp.gate_proj.weight"),
                    &[inter, hidden],
                    5.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.mlp.up_proj.weight"),
                    &[inter, hidden],
                    6.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.mlp.down_proj.weight"),
                    &[hidden, inter],
                    7.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.input_layernorm.weight"),
                    &[hidden],
                    8.0,
                ),
                F32Tensor::ramp(
                    &format!("model.layers.{i}.post_attention_layernorm.weight"),
                    &[hidden],
                    9.0,
                ),
            ]);
        }
        tensors.push(F32Tensor::ramp("model.norm.weight", &[hidden], 10.0));
        if with_lm_head {
            tensors.push(F32Tensor::ramp(
                "lm_head.weight",
                &[config.vocab_size, hidden],
                11.0,
            ));
        }
        tensors
    }

    /// The RMSNorm gains must come from the checkpoint. The previous loader
    /// skipped them ("would be loaded here if RMSNorm supported set_weight"),
    /// leaving the all-ones initialisation in place while reporting success.
    #[test]
    fn test_load_from_path_binds_layer_norm_gains() -> Result<()> {
        let config = tiny_gqa_config();
        let dir = TempModelDir::new("norms");
        dir.write_safetensors(&checkpoint_tensors(&config, true));

        let mut model = StableLMForCausalLM::new(config.clone())?;
        // Before loading, every gain is the all-ones initialisation.
        assert!(model.model.norm.weight().data()?.iter().all(|v| (v - 1.0).abs() < 1e-9));

        model.load_from_path(&dir.path)?;

        let expect_ramp = |tensor: &Tensor, seed: f32, label: &str| -> Result<()> {
            let values = tensor.data()?;
            for (i, value) in values.iter().enumerate() {
                let expected = seed + i as f32 * 0.5;
                assert!(
                    (value - expected).abs() < 1e-5,
                    "{label}[{i}] = {value}, expected {expected}"
                );
            }
            Ok(())
        };

        expect_ramp(model.model.norm.weight(), 10.0, "model.norm")?;
        expect_ramp(
            model.model.layers[0].input_layernorm.weight(),
            8.0,
            "input_layernorm",
        )?;
        expect_ramp(
            model.model.layers[0].post_attention_layernorm.weight(),
            9.0,
            "post_attention_layernorm",
        )?;
        expect_ramp(model.lm_head.weight(), 11.0, "lm_head")?;
        Ok(())
    }

    /// Without `lm_head.weight` the head is tied to the input embeddings.
    #[test]
    fn test_load_from_path_ties_lm_head_to_embeddings() -> Result<()> {
        let config = tiny_gqa_config();
        let dir = TempModelDir::new("tied");
        dir.write_safetensors(&checkpoint_tensors(&config, false));

        let mut model = StableLMForCausalLM::new(config.clone())?;
        model.load_from_path(&dir.path)?;

        let head = model.lm_head.weight().data()?;
        assert_eq!(head.len(), config.vocab_size * config.hidden_size);
        for (i, value) in head.iter().enumerate() {
            let expected = 0.25 + i as f32 * 0.5; // the embedding ramp
            assert!(
                (value - expected).abs() < 1e-5,
                "tied head[{i}] = {value}, expected {expected}"
            );
        }
        Ok(())
    }

    /// A checkpoint missing a required tensor must fail loudly. The previous
    /// loader wrapped every lookup in `if let Ok(..)` and returned `Ok(())`
    /// after binding nothing, so inference ran on random initialisation.
    #[test]
    fn test_load_from_path_reports_missing_tensors() -> Result<()> {
        let config = tiny_gqa_config();
        let dir = TempModelDir::new("missing");
        let tensors: Vec<_> = checkpoint_tensors(&config, true)
            .into_iter()
            .filter(|t| !t.name.ends_with("post_attention_layernorm.weight"))
            .collect();
        dir.write_safetensors(&tensors);

        let mut model = StableLMForCausalLM::new(config)?;
        let error = model.load_from_path(&dir.path).expect_err("missing tensor must be reported");
        let message = error.to_string();
        assert!(
            message.contains("post_attention_layernorm.weight"),
            "the error must name the missing tensor, got: {message}"
        );
        Ok(())
    }

    /// A checkpoint whose names do not match the architecture at all must be
    /// rejected instead of silently leaving a randomly initialised model.
    #[test]
    fn test_load_from_path_rejects_unrelated_checkpoint() -> Result<()> {
        use crate::weight_loading::test_support::F32Tensor;
        let config = tiny_gqa_config();
        let dir = TempModelDir::new("unrelated");
        dir.write_safetensors(&[F32Tensor::ramp("some.other.weight", &[2, 2], 0.0)]);

        let mut model = StableLMForCausalLM::new(config)?;
        assert!(
            model.load_from_path(&dir.path).is_err(),
            "binding nothing must not be reported as a successful load"
        );
        Ok(())
    }

    #[test]
    fn test_rms_norm_set_weight_rejects_wrong_size() -> Result<()> {
        let mut norm = RMSNorm::new(4, 1e-6)?;
        assert!(norm.set_weight(Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3])?).is_err());
        norm.set_weight(Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0], &[4])?)?;
        assert_eq!(norm.weight().data()?, vec![1.0, 2.0, 3.0, 4.0]);
        Ok(())
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B CausalLM device support (SIGKILL risk), run with --ignored
    fn test_causal_lm_device_support() -> Result<()> {
        let config = StableLMConfig::stablelm_3b();

        // Test CPU device
        let model = StableLMForCausalLM::new(config.clone())?;
        assert_eq!(*model.device(), Device::CPU);
        assert_eq!(*model.model.device(), Device::CPU);

        // Test explicit device
        let model_explicit = StableLMForCausalLM::new_with_device(config, Device::CPU)?;
        assert_eq!(*model_explicit.device(), Device::CPU);
        Ok(())
    }
}
