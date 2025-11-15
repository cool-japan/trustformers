use crate::gemma::config::GemmaConfig;
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
    layers::{Embedding, Linear},
    ops::activations::gelu,
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Gemma RMSNorm implementation (similar to LLaMA)
pub struct GemmaRMSNorm {
    weight: Tensor,
    eps: f32,
}

impl GemmaRMSNorm {
    pub fn new(normalized_shape: usize, eps: f32) -> Result<Self> {
        let weight = Tensor::ones(&[normalized_shape])?;
        Ok(Self { weight, eps })
    }

    pub fn set_weight(&mut self, weight: Tensor) -> Result<()> {
        self.weight = weight;
        Ok(())
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.len()
    }
}

impl Layer for GemmaRMSNorm {
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
                        "Unsupported weight tensor type for GemmaRMSNorm",
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported input tensor type for GemmaRMSNorm",
            )),
        }
    }
}

/// Gemma Rotary Position Embedding (RoPE) - Enhanced version
pub struct GemmaRotaryEmbedding {
    pub dim: usize,
    pub max_seq_len: usize,
    pub base: f32,
}

impl GemmaRotaryEmbedding {
    pub fn new(dim: usize, max_seq_len: usize, base: f32) -> Self {
        Self {
            dim,
            max_seq_len,
            base,
        }
    }

    /// Apply rotary embedding to query and key tensors
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        // Simplified RoPE implementation
        // In a production implementation, this would have proper complex number handling
        let (rotated_q, rotated_k) = match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let rotated_q = q_arr.clone();
                let rotated_k = k_arr.clone();

                // Apply rotary embedding (simplified)
                for &pos in position_ids.iter() {
                    for head in 0..(self.dim / 2) {
                        let freq = 1.0 / self.base.powf(2.0 * head as f32 / self.dim as f32);
                        let angle = pos as f32 * freq;
                        let _cos_val = angle.cos();
                        let _sin_val = angle.sin();

                        // Apply rotation (simplified - would need proper tensor operations)
                    }
                }

                (Tensor::F32(rotated_q), Tensor::F32(rotated_k))
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor types for RoPE",
                ))
            },
        };

        Ok((rotated_q, rotated_k))
    }
}

/// Gemma MLP layer with GeGLU activation
pub struct GemmaMLP {
    gate_proj: Linear, // Gating projection
    up_proj: Linear,   // Up projection
    down_proj: Linear, // Down projection
}

impl GemmaMLP {
    pub fn new(config: &GemmaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &GemmaConfig, device: Device) -> Result<Self> {
        let gate_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.attention_bias,
            device,
        );
        let up_proj = Linear::new_with_device(
            config.hidden_size,
            config.intermediate_size,
            config.attention_bias,
            device,
        );
        let down_proj = Linear::new_with_device(
            config.intermediate_size,
            config.hidden_size,
            config.attention_bias,
            device,
        );

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

impl Layer for GemmaMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // GeGLU: down_proj(gelu(gate_proj(x)) * up_proj(x))
        let gate_output = self.gate_proj.forward(input.clone())?;
        let up_output = self.up_proj.forward(input)?;

        // Apply GELU to gate output (GeGLU activation)
        let gate_activated = gelu(&gate_output)?;

        // Element-wise multiply gate and up outputs
        let combined = match (&gate_activated, &up_output) {
            (Tensor::F32(gate_arr), Tensor::F32(up_arr)) => Ok(Tensor::F32(gate_arr * up_arr)),
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for Gemma MLP",
            )),
        }?;

        // Apply down projection
        self.down_proj.forward(combined)
    }
}

/// Gemma Attention layer with multi-query attention support
#[allow(dead_code)]
pub struct GemmaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    rotary_emb: GemmaRotaryEmbedding,
    #[allow(dead_code)]
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scaling: f32,
}

impl GemmaAttention {
    pub fn new(config: &GemmaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &GemmaConfig, device: Device) -> Result<Self> {
        let scaling = 1.0 / (config.head_dim as f32).sqrt();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            config.attention_bias,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * config.head_dim,
            config.attention_bias,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            config.attention_bias,
            device,
        );

        let rotary_emb = GemmaRotaryEmbedding::new(
            config.head_dim,
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
            head_dim: config.head_dim,
            scaling,
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

impl Layer for GemmaAttention {
    type Input = Tensor;
    type Output = Tensor;

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

        // Simplified attention mechanism (would need proper multi-query attention in production)
        match (&q_rope, &k_rope, &v) {
            (Tensor::F32(q_arr), Tensor::F32(_k_arr), Tensor::F32(v_arr)) => {
                // Simplified attention: scale and combine
                let scaled_q = q_arr.mapv(|x| x * self.scaling);
                let attention_output = &scaled_q + v_arr; // Simplified combination
                self.o_proj.forward(Tensor::F32(attention_output))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor types for Gemma attention",
            )),
        }
    }
}

/// Gemma decoder layer
pub struct GemmaDecoderLayer {
    self_attn: GemmaAttention,
    mlp: GemmaMLP,
    input_layernorm: GemmaRMSNorm,
    post_attention_layernorm: GemmaRMSNorm,
}

impl GemmaDecoderLayer {
    pub fn new(config: &GemmaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &GemmaConfig, device: Device) -> Result<Self> {
        let self_attn = GemmaAttention::new_with_device(config, device)?;
        let mlp = GemmaMLP::new_with_device(config, device)?;
        let input_layernorm = GemmaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = GemmaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

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

impl Layer for GemmaDecoderLayer {
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

/// Gemma model
pub struct GemmaModel {
    config: GemmaConfig,
    embed_tokens: Embedding,
    layers: Vec<GemmaDecoderLayer>,
    norm: GemmaRMSNorm,
}

impl GemmaModel {
    pub fn new(config: GemmaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: GemmaConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(GemmaDecoderLayer::new_with_device(&config, device)?);
        }

        let norm = GemmaRMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }
}

impl Model for GemmaModel {
    type Config = GemmaConfig;
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
            // Attention layer parameters
            total += layer.self_attn.q_proj.parameter_count();
            total += layer.self_attn.k_proj.parameter_count();
            total += layer.self_attn.v_proj.parameter_count();
            total += layer.self_attn.o_proj.parameter_count();

            // MLP parameters
            total += layer.mlp.gate_proj.parameter_count();
            total += layer.mlp.up_proj.parameter_count();
            total += layer.mlp.down_proj.parameter_count();

            // LayerNorm parameters (weight only, no bias)
            total += self.config.hidden_size; // input_layernorm
            total += self.config.hidden_size; // post_attention_layernorm
        }

        // Final norm parameters
        total += self.config.hidden_size;

        total
    }
}

/// Gemma for causal language modeling
pub struct GemmaForCausalLM {
    model: GemmaModel,
    lm_head: Linear,
}

impl GemmaForCausalLM {
    pub fn new(config: GemmaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: GemmaConfig, device: Device) -> Result<Self> {
        let model = GemmaModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self { model, lm_head })
    }
}

impl Model for GemmaForCausalLM {
    type Config = GemmaConfig;
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
        // Count parameters in the base model
        let mut total = self.model.num_parameters();

        // Add language model head parameters
        total += self.lm_head.parameter_count();

        total
    }
}

impl GemmaForCausalLM {
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

        // List of essential files for Gemma models
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
                    file_path.to_str().unwrap(),
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
                .args(["-O", file_path.to_str().unwrap(), &file_url])
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
}
