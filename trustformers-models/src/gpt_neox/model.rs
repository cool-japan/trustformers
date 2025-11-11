use crate::gpt_neox::config::GPTNeoXConfig;
use crate::llama::model::RotaryEmbedding; // Reuse from LLaMA
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::gelu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// GPT-NeoX MLP layer
pub struct GPTNeoXMLP {
    pub dense_h_to_4h: Linear,
    pub dense_4h_to_h: Linear,
}

impl GPTNeoXMLP {
    pub fn new(config: &GPTNeoXConfig) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: Linear::new(config.hidden_size, config.intermediate_size, true),
            dense_4h_to_h: Linear::new(config.intermediate_size, config.hidden_size, true),
        })
    }

    pub fn new_with_device(config: &GPTNeoXConfig, device: Device) -> Result<Self> {
        Ok(Self {
            dense_h_to_4h: Linear::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                true,
                device,
            ),
            dense_4h_to_h: Linear::new_with_device(
                config.intermediate_size,
                config.hidden_size,
                true,
                device,
            ),
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.dense_h_to_4h.parameter_count() + self.dense_4h_to_h.parameter_count()
    }

    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(
        &mut self,
        device: &trustformers_core::device::Device,
    ) -> trustformers_core::errors::Result<()> {
        self.dense_h_to_4h.weights_to_gpu(device)?;
        self.dense_4h_to_h.weights_to_gpu(device)?;
        Ok(())
    }
}

impl Layer for GPTNeoXMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.dense_h_to_4h.forward(input)?;
        let activated = gelu(&hidden)?;
        self.dense_4h_to_h.forward(activated)
    }
}

/// GPT-NeoX Attention layer with Rotary Position Embeddings
pub struct GPTNeoXAttention {
    pub query_key_value: Linear,      // Combined QKV projection
    pub dense: Linear,                // Output projection
    pub _rotary_emb: RotaryEmbedding, // TODO: Use in full attention implementation
    pub _num_heads: usize,
    pub _head_dim: usize,
    pub _rotary_ndims: usize,
}

impl GPTNeoXAttention {
    pub fn new(config: &GPTNeoXConfig) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_ndims = (head_dim as f32 * config.rotary_pct) as usize;

        Ok(Self {
            query_key_value: Linear::new(config.hidden_size, config.hidden_size * 3, true),
            dense: Linear::new(config.hidden_size, config.hidden_size, true),
            _rotary_emb: RotaryEmbedding::new(
                rotary_ndims,
                config.max_position_embeddings,
                config.rotary_emb_base,
            ),
            _num_heads: config.num_attention_heads,
            _head_dim: head_dim,
            _rotary_ndims: rotary_ndims,
        })
    }

    pub fn new_with_device(config: &GPTNeoXConfig, device: Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_ndims = (head_dim as f32 * config.rotary_pct) as usize;

        Ok(Self {
            query_key_value: Linear::new_with_device(
                config.hidden_size,
                config.hidden_size * 3,
                true,
                device,
            ),
            dense: Linear::new_with_device(config.hidden_size, config.hidden_size, true, device),
            _rotary_emb: RotaryEmbedding::new(
                rotary_ndims,
                config.max_position_embeddings,
                config.rotary_emb_base,
            ),
            _num_heads: config.num_attention_heads,
            _head_dim: head_dim,
            _rotary_ndims: rotary_ndims,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.query_key_value.parameter_count() + self.dense.parameter_count()
    }

    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(
        &mut self,
        device: &trustformers_core::device::Device,
    ) -> trustformers_core::errors::Result<()> {
        self.query_key_value.weights_to_gpu(device)?;
        self.dense.weights_to_gpu(device)?;
        Ok(())
    }
}

impl Layer for GPTNeoXAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        use scirs2_core::ndarray::{s, Array2};

        // Temporary fallback: Convert Metal tensors to F32
        // TODO: Implement full Tensor::Metal support in Attention
        #[cfg(feature = "metal")]
        let input = match &input {
            Tensor::Metal(_) => input.to_device_enum(&trustformers_core::device::Device::CPU)?,
            _ => input,
        };

        #[cfg(not(feature = "metal"))]
        let input = input;

        let shape = input.shape();
        let seq_len = if shape.len() == 2 { shape[0] } else { shape[1] };

        // Project to combined QKV
        let qkv = self.query_key_value.forward(input.clone())?;

        match qkv {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() != 2 {
                    return Err(tensor_op_error(
                        "GPTNeoXAttention::forward",
                        &format!("Expected 2D tensor, got shape: {:?}", shape),
                    ));
                }

                // Split QKV using HuggingFace's approach:
                // 1. Reshape [seq_len, 3*hidden_size] → [seq_len, num_heads, 3*head_dim]
                // 2. Transpose → [num_heads, seq_len, 3*head_dim]
                // 3. Chunk → Q, K, V each [num_heads, seq_len, head_dim]

                let num_heads = self._num_heads;
                let head_dim = self._head_dim;

                // Step 1: Reshape to [seq_len, num_heads, 3*head_dim]
                let qkv_reshaped = arr
                    .into_shape((seq_len, num_heads, 3 * head_dim))
                    .map_err(|_| tensor_op_error("GPTNeoXAttention", "QKV reshape failed"))?;

                // Step 2: Transpose to [num_heads, seq_len, 3*head_dim]
                let qkv_transposed = qkv_reshaped.permuted_axes([1, 0, 2]);

                // Step 3: Split into Q, K, V along last dimension
                // Each will be [num_heads, seq_len, head_dim]
                let q = qkv_transposed.slice(s![.., .., 0..head_dim]).to_owned();
                let k = qkv_transposed.slice(s![.., .., head_dim..2 * head_dim]).to_owned();
                let v = qkv_transposed.slice(s![.., .., 2 * head_dim..3 * head_dim]).to_owned();

                // Q, K, V are now [num_heads, seq_len, head_dim]
                // We need to transpose to [seq_len, num_heads, head_dim] for RoPE
                let rotary_ndims = self._rotary_ndims;

                // Transpose to [seq_len, num_heads, head_dim]
                let q_transposed = q.permuted_axes([1, 0, 2]);
                let k_transposed = k.permuted_axes([1, 0, 2]);

                // Apply Rotary Position Embeddings (RoPE) to Q and K
                let mut q_rope = q_transposed.to_owned();
                let mut k_rope = k_transposed.to_owned();

                // Apply RoPE to each head
                // RoPE rotates pairs (i, i + D/2) where D is rotary_ndims
                // q_rope and k_rope are [seq_len, num_heads, head_dim]
                let half_rotary_ndims = rotary_ndims / 2;
                for pos in 0..seq_len {
                    for h in 0..num_heads {
                        // Apply rotation to pairs (i, i + D/2) for i in 0..D/2
                        for i in 0..half_rotary_ndims {
                            let j = i + half_rotary_ndims;

                            // Calculate rotation angle for this pair
                            let freq = 1.0
                                / self._rotary_emb.base.powf(2.0 * i as f32 / rotary_ndims as f32);
                            let angle = pos as f32 * freq;
                            let cos_val = angle.cos();
                            let sin_val = angle.sin();

                            // Rotate Q: (qi, qj) → (qi*cos - qj*sin, qi*sin + qj*cos)
                            let q_i = q_rope[[pos, h, i]];
                            let q_j = q_rope[[pos, h, j]];
                            q_rope[[pos, h, i]] = q_i * cos_val - q_j * sin_val;
                            q_rope[[pos, h, j]] = q_i * sin_val + q_j * cos_val;

                            // Rotate K: (ki, kj) → (ki*cos - kj*sin, ki*sin + kj*cos)
                            let k_i = k_rope[[pos, h, i]];
                            let k_j = k_rope[[pos, h, j]];
                            k_rope[[pos, h, i]] = k_i * cos_val - k_j * sin_val;
                            k_rope[[pos, h, j]] = k_i * sin_val + k_j * cos_val;
                        }
                    }
                }

                // After RoPE, q_rope and k_rope are [seq_len, num_heads, head_dim]
                // V is still [num_heads, seq_len, head_dim], transpose it
                let v_reshaped = v.permuted_axes([1, 0, 2]);

                let q_reshaped = q_rope;
                let k_reshaped = k_rope;

                // Compute attention scores for each head
                // scores = Q @ K^T / sqrt(head_dim)
                let scale = (head_dim as f32).sqrt();
                let hidden_size = num_heads * head_dim;
                let mut attn_output = Array2::<f32>::zeros((seq_len, hidden_size));

                // Helper function: Try Metal attention (returns Some on success, None on failure)
                #[cfg(feature = "metal")]
                let try_metal_attention = |q_head: &Array2<f32>,
                                           k_head: &Array2<f32>,
                                           v_head: &Array2<f32>,
                                           _h: usize|
                 -> Option<Array2<f32>> {
                    use trustformers_core::gpu_ops::metal::get_metal_backend;

                    let backend = get_metal_backend().ok()?;

                    // Convert to vecs
                    let q_vec: Vec<f32> = q_head.iter().copied().collect();
                    let k_t = k_head.t();
                    let k_t_vec: Vec<f32> = k_t.iter().copied().collect();
                    let v_vec: Vec<f32> = v_head.iter().copied().collect();

                    // Q@K^T on Metal: [seq_len, head_dim] @ [head_dim, seq_len] = [seq_len, seq_len]
                    let scores_vec =
                        backend.matmul_f32(&q_vec, &k_t_vec, seq_len, head_dim, seq_len).ok()?;

                    // Scale scores
                    let scores_scaled: Vec<f32> = scores_vec.iter().map(|&x| x / scale).collect();

                    // Softmax with causal mask on Metal
                    let attn_weights_vec =
                        backend.softmax_causal_f32(&scores_scaled, seq_len).ok()?;

                    // Attn@V on Metal: [seq_len, seq_len] @ [seq_len, head_dim] = [seq_len, head_dim]
                    let output_vec = backend
                        .matmul_f32(&attn_weights_vec, &v_vec, seq_len, seq_len, head_dim)
                        .ok()?;

                    // Convert back to Array2
                    Array2::from_shape_vec((seq_len, head_dim), output_vec).ok()
                };

                // Helper function: CPU attention (always succeeds)
                let cpu_attention = |q_head: Array2<f32>,
                                     k_head: Array2<f32>,
                                     v_head: Array2<f32>|
                 -> Array2<f32> {
                    let k_t = k_head.t();
                    let mut scores = q_head.dot(&k_t) / scale;

                    // Apply causal mask
                    for i in 0..seq_len {
                        for j in (i + 1)..seq_len {
                            scores[[i, j]] = f32::NEG_INFINITY;
                        }
                    }

                    // Softmax
                    let mut attn_weights = Array2::<f32>::zeros(scores.dim());
                    for i in 0..seq_len {
                        let row = scores.row(i);
                        let max_val = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                        let exp_row: Vec<f32> = row.iter().map(|&x| (x - max_val).exp()).collect();
                        let sum: f32 = exp_row.iter().sum();
                        for (j, &val) in exp_row.iter().enumerate() {
                            attn_weights[[i, j]] = val / sum;
                        }
                    }

                    attn_weights.dot(&v_head)
                };

                // Process each attention head
                for h in 0..num_heads {
                    // Extract head: [seq_len, head_dim]
                    let q_head = q_reshaped.slice(s![.., h, ..]).to_owned();
                    let k_head = k_reshaped.slice(s![.., h, ..]).to_owned();
                    let v_head = v_reshaped.slice(s![.., h, ..]).to_owned();

                    // Compute head_output (try Metal, fallback to CPU)
                    let head_output: Array2<f32> = {
                        #[cfg(feature = "metal")]
                        {
                            try_metal_attention(&q_head, &k_head, &v_head, h).unwrap_or_else(|| {
                                cpu_attention(q_head.clone(), k_head.clone(), v_head.clone())
                            })
                        }

                        #[cfg(not(feature = "metal"))]
                        {
                            cpu_attention(q_head, k_head, v_head)
                        }
                    };

                    // Place head output in the correct position
                    let start_idx = h * head_dim;
                    let end_idx = start_idx + head_dim;
                    attn_output.slice_mut(s![.., start_idx..end_idx]).assign(&head_output);
                }

                // Project through output layer
                self.dense.forward(Tensor::F32(attn_output.into_dyn()))
            },
            _ => Err(tensor_op_error(
                "GPTNeoXAttention::forward",
                "Unsupported tensor type",
            )),
        }
    }
}

/// GPT-NeoX Layer
pub struct GPTNeoXLayer {
    pub input_layernorm: LayerNorm,
    pub post_attention_layernorm: LayerNorm,
    pub attention: GPTNeoXAttention,
    pub mlp: GPTNeoXMLP,
    pub use_parallel_residual: bool,
}

impl GPTNeoXLayer {
    pub fn new(config: &GPTNeoXConfig) -> Result<Self> {
        Ok(Self {
            input_layernorm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            post_attention_layernorm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?,
            attention: GPTNeoXAttention::new(config)?,
            mlp: GPTNeoXMLP::new(config)?,
            use_parallel_residual: config.use_parallel_residual,
        })
    }

    pub fn new_with_device(config: &GPTNeoXConfig, device: Device) -> Result<Self> {
        Ok(Self {
            input_layernorm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            post_attention_layernorm: LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps,
            )?,
            attention: GPTNeoXAttention::new_with_device(config, device)?,
            mlp: GPTNeoXMLP::new_with_device(config, device)?,
            use_parallel_residual: config.use_parallel_residual,
        })
    }

    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count() + self.mlp.parameter_count()
    }

    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(
        &mut self,
        device: &trustformers_core::device::Device,
    ) -> trustformers_core::errors::Result<()> {
        self.input_layernorm.weights_to_gpu(device)?;
        self.attention.weights_to_gpu(device)?;
        self.post_attention_layernorm.weights_to_gpu(device)?;
        self.mlp.weights_to_gpu(device)?;
        Ok(())
    }
}

impl Layer for GPTNeoXLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        if self.use_parallel_residual {
            // Parallel: attn and mlp computed in parallel with different layer norms
            // Formula: x = x + attn(ln1(x)) + mlp(ln2(x))
            let ln1_out = self.input_layernorm.forward(input.clone())?;
            let attn_out = self.attention.forward(ln1_out)?;

            let ln2_out = self.post_attention_layernorm.forward(input.clone())?;
            let mlp_out = self.mlp.forward(ln2_out)?;

            // Add both outputs to residual: x + attn + mlp
            // Use Tensor::add() to support mixed types (F32 + Metal)
            let temp = input.add(&attn_out)?;
            temp.add(&mlp_out)
        } else {
            // Sequential: attn first, then mlp
            let ln1_out = self.input_layernorm.forward(input.clone())?;
            let attn_out = self.attention.forward(ln1_out)?;

            // Use Tensor::add() to support mixed types
            let residual = input.add(&attn_out)?;

            let ln2_out = self.post_attention_layernorm.forward(residual.clone())?;
            let mlp_out = self.mlp.forward(ln2_out)?;

            // Use Tensor::add() to support mixed types
            residual.add(&mlp_out)
        }
    }
}

/// GPT-NeoX Model
pub struct GPTNeoXModel {
    pub embed_in: Embedding,
    pub layers: Vec<GPTNeoXLayer>,
    pub final_layer_norm: LayerNorm,
    config: GPTNeoXConfig,
}

impl GPTNeoXModel {
    pub fn new(config: GPTNeoXConfig) -> Result<Self> {
        config.validate()?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(GPTNeoXLayer::new(&config)?);
        }

        Ok(Self {
            embed_in: Embedding::new(config.vocab_size, config.hidden_size, None)?,
            layers,
            final_layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            config,
        })
    }

    pub fn new_with_device(config: GPTNeoXConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(GPTNeoXLayer::new_with_device(&config, device)?);
        }

        Ok(Self {
            embed_in: Embedding::new_with_device(
                config.vocab_size,
                config.hidden_size,
                None,
                device,
            )?,
            layers,
            final_layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            config,
        })
    }

    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(
        &mut self,
        device: &trustformers_core::device::Device,
    ) -> trustformers_core::errors::Result<()> {
        // Upload embedding weights to GPU
        self.embed_in.weights_to_gpu(device)?;

        // Convert all transformer layers
        for layer in &mut self.layers {
            layer.weights_to_gpu(device)?;
        }
        self.final_layer_norm.weights_to_gpu(device)?;
        Ok(())
    }

    /// Load model weights from HuggingFace format
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let model_path = model_path.as_ref(); // Convert to &Path so we can reuse it

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config.clone()))?;

        // Load embedding weights
        if let Ok(embed_weights) = loader.load_tensor("gpt_neox.embed_in.weight") {
            self.embed_in.set_weight(embed_weights)?;
        }

        // Load final layer norm (now that the loader bug is fixed)
        match loader.load_tensor("gpt_neox.final_layer_norm.weight") {
            Ok(final_ln_weight) => {
                // Debug: check the weight values
                if let Tensor::F32(ref arr) = final_ln_weight {
                    use scirs2_core::ndarray::s;
                    let first_10 = arr.slice(s![0..10]);
                    eprintln!(
                        "[DEBUG] final_layer_norm.weight first 10: {:?}",
                        first_10.iter().take(10).collect::<Vec<_>>()
                    );
                    let mean = arr.mean().unwrap_or(0.0);
                    eprintln!(
                        "[DEBUG] final_layer_norm.weight mean: {:.3} (expected: 6.688)",
                        mean
                    );
                }
                self.final_layer_norm.set_weight(final_ln_weight)?;
            },
            Err(e) => {
                eprintln!("[ERROR] Failed to load final_layer_norm.weight: {:?}", e);
            },
        }
        match loader.load_tensor("gpt_neox.final_layer_norm.bias") {
            Ok(final_ln_bias) => {
                self.final_layer_norm.set_bias(final_ln_bias)?;
            },
            Err(e) => {
                eprintln!("[ERROR] Failed to load final_layer_norm.bias: {:?}", e);
            },
        }

        // Load layer weights
        for (i, layer) in self.layers.iter_mut().enumerate() {
            let prefix = format!("gpt_neox.layers.{}", i);

            // Attention weights (combined QKV)
            if let Ok(qkv_weights) =
                loader.load_tensor(&format!("{}.attention.query_key_value.weight", prefix))
            {
                layer.attention.query_key_value.set_weight(qkv_weights)?;
            }
            if let Ok(qkv_bias) =
                loader.load_tensor(&format!("{}.attention.query_key_value.bias", prefix))
            {
                layer.attention.query_key_value.set_bias(qkv_bias)?;
            }

            if let Ok(dense_weights) =
                loader.load_tensor(&format!("{}.attention.dense.weight", prefix))
            {
                layer.attention.dense.set_weight(dense_weights)?;
            }
            if let Ok(dense_bias) = loader.load_tensor(&format!("{}.attention.dense.bias", prefix))
            {
                layer.attention.dense.set_bias(dense_bias)?;
            }

            // MLP weights
            if let Ok(mlp_up_weights) =
                loader.load_tensor(&format!("{}.mlp.dense_h_to_4h.weight", prefix))
            {
                layer.mlp.dense_h_to_4h.set_weight(mlp_up_weights)?;
            }
            if let Ok(mlp_up_bias) =
                loader.load_tensor(&format!("{}.mlp.dense_h_to_4h.bias", prefix))
            {
                layer.mlp.dense_h_to_4h.set_bias(mlp_up_bias)?;
            }

            if let Ok(mlp_down_weights) =
                loader.load_tensor(&format!("{}.mlp.dense_4h_to_h.weight", prefix))
            {
                layer.mlp.dense_4h_to_h.set_weight(mlp_down_weights)?;
            }
            if let Ok(mlp_down_bias) =
                loader.load_tensor(&format!("{}.mlp.dense_4h_to_h.bias", prefix))
            {
                layer.mlp.dense_4h_to_h.set_bias(mlp_down_bias)?;
            }

            // Layer norms
            if let Ok(ln1_weight) =
                loader.load_tensor(&format!("{}.input_layernorm.weight", prefix))
            {
                layer.input_layernorm.set_weight(ln1_weight)?;
            }
            if let Ok(ln1_bias) = loader.load_tensor(&format!("{}.input_layernorm.bias", prefix)) {
                layer.input_layernorm.set_bias(ln1_bias)?;
            }

            if let Ok(ln2_weight) =
                loader.load_tensor(&format!("{}.post_attention_layernorm.weight", prefix))
            {
                layer.post_attention_layernorm.set_weight(ln2_weight)?;
            }
            if let Ok(ln2_bias) =
                loader.load_tensor(&format!("{}.post_attention_layernorm.bias", prefix))
            {
                layer.post_attention_layernorm.set_bias(ln2_bias)?;
            }
        }

        loader.close()?;
        Ok(())
    }
}

impl Model for GPTNeoXModel {
    type Config = GPTNeoXConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Convert token IDs to embeddings
        let mut hidden_states = self.embed_in.forward(input)?;

        // Pass through all layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final layer norm
        self.final_layer_norm.forward(hidden_states)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        Err(
            trustformers_core::errors::TrustformersError::not_implemented(
                "Use load_from_path for GPT-NeoX weight loading".to_string(),
            ),
        )
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embed_params = self.embed_in.parameter_count();
        let layer_params: usize = self.layers.iter().map(|l| l.parameter_count()).sum();
        embed_params + layer_params
    }
}

/// GPT-NeoX for Causal Language Modeling
pub struct GPTNeoXForCausalLM {
    pub gpt_neox: GPTNeoXModel,
    pub embed_out: Linear,
}

impl GPTNeoXForCausalLM {
    pub fn new(config: GPTNeoXConfig) -> Result<Self> {
        let gpt_neox = GPTNeoXModel::new(config.clone())?;
        let embed_out = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            gpt_neox,
            embed_out,
        })
    }

    pub fn new_with_device(config: GPTNeoXConfig, device: Device) -> Result<Self> {
        let gpt_neox = GPTNeoXModel::new_with_device(config.clone(), device)?;
        let embed_out =
            Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            gpt_neox,
            embed_out,
        })
    }

    #[cfg(feature = "metal")]
    pub fn weights_to_gpu(
        &mut self,
        device: &trustformers_core::device::Device,
    ) -> trustformers_core::errors::Result<()> {
        self.gpt_neox.weights_to_gpu(device)?;
        self.embed_out.weights_to_gpu(device)?;
        println!("✓ All model weights uploaded to GPU");
        Ok(())
    }

    /// Load model weights from HuggingFace format
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        // Load base model
        self.gpt_neox.load_from_path(model_path.as_ref())?;

        // Load embed_out (lm_head)
        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        eprintln!("[DEBUG] Loading embed_out.weight...");
        match loader.load_tensor("embed_out.weight") {
            Ok(embed_out_weights) => {
                eprintln!("[DEBUG] ✓ embed_out.weight loaded successfully");
                if let Tensor::F32(ref arr) = embed_out_weights {
                    use scirs2_core::ndarray::s;
                    eprintln!("[DEBUG] embed_out.weight shape: {:?}", arr.shape());
                    let first_5 = arr.slice(s![0, 0..5]);
                    eprintln!(
                        "[DEBUG] embed_out.weight[0, 0..5]: {:?}",
                        first_5.iter().take(5).collect::<Vec<_>>()
                    );
                }
                self.embed_out.set_weight(embed_out_weights)?;
            },
            Err(e) => {
                eprintln!("[ERROR] Failed to load embed_out.weight: {:?}", e);
                eprintln!("[WARNING] LM head will use uninitialized/default weights!");
            },
        }

        loader.close()?;
        Ok(())
    }
}

impl Model for GPTNeoXForCausalLM {
    type Config = GPTNeoXConfig;
    type Input = Vec<u32>;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.gpt_neox.forward(input)?;
        let logits = self.embed_out.forward(hidden_states)?;
        Ok(logits)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.gpt_neox.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.gpt_neox.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.gpt_neox.num_parameters() + self.embed_out.parameter_count()
    }
}
