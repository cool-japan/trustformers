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
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
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
    query_key_value: Linear, // Combined QKV projection
    dense: Linear,           // Output projection
    _rotary_emb: RotaryEmbedding, // TODO: Use in full attention implementation
    _num_heads: usize,
    _head_dim: usize,
    _rotary_ndims: usize,
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
}

impl Layer for GPTNeoXAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let seq_len = if shape.len() == 2 { shape[0] } else { shape[1] };

        // Project to QKV
        let qkv = self.query_key_value.forward(input)?;

        // Generate position IDs
        let _position_ids: Vec<usize> = (0..seq_len).collect();

        // Split QKV (simplified - just use Q for now)
        // TODO: Implement proper split and attention computation using _position_ids and RoPE
        match qkv {
            Tensor::F32(_) => {
                // Simplified: skip RoPE and attention, just project through dense
                self.dense.forward(qkv)
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
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    attention: GPTNeoXAttention,
    mlp: GPTNeoXMLP,
    use_parallel_residual: bool,
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
}

impl Layer for GPTNeoXLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        if self.use_parallel_residual {
            // Parallel: attn and mlp computed in parallel, then added
            let ln_out = self.input_layernorm.forward(input.clone())?;
            let attn_out = self.attention.forward(ln_out.clone())?;
            let mlp_out = self.mlp.forward(ln_out)?;

            // Add both outputs to residual
            match (&input, &attn_out, &mlp_out) {
                (Tensor::F32(inp), Tensor::F32(attn), Tensor::F32(mlp)) => {
                    Ok(Tensor::F32(inp + attn + mlp))
                },
                _ => Err(tensor_op_error(
                    "GPTNeoXLayer::forward",
                    "Unsupported tensor types",
                )),
            }
        } else {
            // Sequential: attn first, then mlp
            let ln1_out = self.input_layernorm.forward(input.clone())?;
            let attn_out = self.attention.forward(ln1_out)?;

            let residual = match (&input, &attn_out) {
                (Tensor::F32(inp), Tensor::F32(attn)) => Tensor::F32(inp + attn),
                _ => {
                    return Err(tensor_op_error(
                        "GPTNeoXLayer::forward",
                        "Unsupported tensor types",
                    ))
                },
            };

            let ln2_out = self.post_attention_layernorm.forward(residual.clone())?;
            let mlp_out = self.mlp.forward(ln2_out)?;

            match (&residual, &mlp_out) {
                (Tensor::F32(res), Tensor::F32(mlp)) => Ok(Tensor::F32(res + mlp)),
                _ => Err(tensor_op_error(
                    "GPTNeoXLayer::forward",
                    "Unsupported tensor types",
                )),
            }
        }
    }
}

/// GPT-NeoX Model
pub struct GPTNeoXModel {
    embed_in: Embedding,
    layers: Vec<GPTNeoXLayer>,
    final_layer_norm: LayerNorm,
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
            embed_in: Embedding::new(config.vocab_size, config.hidden_size, None)?,
            layers,
            final_layer_norm: LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?,
            config,
        })
    }

    /// Load model weights from HuggingFace format
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embedding weights
        if let Ok(embed_weights) = loader.load_tensor("gpt_neox.embed_in.weight") {
            self.embed_in.set_weight(embed_weights)?;
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
            if let Ok(_qkv_bias) =
                loader.load_tensor(&format!("{}.attention.query_key_value.bias", prefix))
            {
                // TODO: Set bias when Linear layer supports bias setting
            }

            if let Ok(dense_weights) =
                loader.load_tensor(&format!("{}.attention.dense.weight", prefix))
            {
                layer.attention.dense.set_weight(dense_weights)?;
            }
            if let Ok(_dense_bias) = loader.load_tensor(&format!("{}.attention.dense.bias", prefix))
            {
                // TODO: Set bias when Linear layer supports bias setting
            }

            // MLP weights
            if let Ok(mlp_up_weights) =
                loader.load_tensor(&format!("{}.mlp.dense_h_to_4h.weight", prefix))
            {
                layer.mlp.dense_h_to_4h.set_weight(mlp_up_weights)?;
            }
            if let Ok(_mlp_up_bias) =
                loader.load_tensor(&format!("{}.mlp.dense_h_to_4h.bias", prefix))
            {
                // TODO: Set bias when Linear layer supports bias setting
            }

            if let Ok(mlp_down_weights) =
                loader.load_tensor(&format!("{}.mlp.dense_4h_to_h.weight", prefix))
            {
                layer.mlp.dense_4h_to_h.set_weight(mlp_down_weights)?;
            }
            if let Ok(_mlp_down_bias) =
                loader.load_tensor(&format!("{}.mlp.dense_4h_to_h.bias", prefix))
            {
                // TODO: Set bias when Linear layer supports bias setting
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

        // Load final layer norm
        if let Ok(final_ln_weight) = loader.load_tensor("gpt_neox.final_layer_norm.weight") {
            self.final_layer_norm.set_weight(final_ln_weight)?;
        }
        if let Ok(final_ln_bias) = loader.load_tensor("gpt_neox.final_layer_norm.bias") {
            self.final_layer_norm.set_bias(final_ln_bias)?;
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
    gpt_neox: GPTNeoXModel,
    embed_out: Linear,
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

        if let Ok(embed_out_weights) = loader.load_tensor("embed_out.weight") {
            self.embed_out.set_weight(embed_out_weights)?;
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
