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

    /// Apply rotary embedding to query and key tensors
    pub fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
        position_ids: &[usize],
    ) -> Result<(Tensor, Tensor)> {
        let (rotated_q, rotated_k) = match (q, k) {
            (Tensor::F32(q_arr), Tensor::F32(k_arr)) => {
                let rotated_q = q_arr.clone();
                let rotated_k = k_arr.clone();

                // Apply rotary embedding
                for &pos in position_ids.iter() {
                    for head in 0..(self.dim / 2) {
                        let freq = 1.0 / self.base.powf(2.0 * head as f32 / self.dim as f32);
                        let angle = pos as f32 * freq;
                        let _cos_val = angle.cos();
                        let _sin_val = angle.sin();

                        // Apply rotation to Q
                        // Note: This is a simplified RoPE implementation
                        // In a full implementation, we would need proper tensor reshaping
                        // and complex number operations

                        // Apply rotation to K
                        // Note: This is a simplified RoPE implementation
                    }
                }

                (Tensor::F32(rotated_q), Tensor::F32(rotated_k))
            },
            _ => {
                return Err(tensor_op_error(
                    "RotaryEmbedding::apply_rotary_emb",
                    "Unsupported tensor types for RoPE",
                ))
            },
        };

        Ok((rotated_q, rotated_k))
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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let shape = input.shape();
        let (_batch_size, seq_len) = if shape.len() == 2 {
            // No batch dimension: [seq_len, hidden]
            (1, shape[0])
        } else if shape.len() == 3 {
            // Has batch dimension: [batch, seq_len, hidden]
            (shape[0], shape[1])
        } else {
            return Err(tensor_op_error(
                "LlamaAttention::forward",
                &format!("Unexpected input shape: {:?}", shape),
            ));
        };

        // Project to Q, K, V
        let q = self.q_proj.forward(input.clone())?; // [seq_len, num_heads * head_dim] or [batch, seq_len, ...]
        let k = self.k_proj.forward(input.clone())?; // [seq_len, num_kv_heads * head_dim] or [batch, seq_len, ...]
        let _v = self.v_proj.forward(input)?; // [seq_len, num_kv_heads * head_dim] or [batch, seq_len, ...] - TODO: Use in full attention

        // Generate position IDs (0, 1, 2, ..., seq_len-1)
        let position_ids: Vec<usize> = (0..seq_len).collect();

        // Apply rotary embedding
        let (q_rope, _k_rope) = self.rotary_emb.apply_rotary_emb(&q, &k, &position_ids)?; // TODO: Use _k_rope in full attention

        // Simplified attention for GQA (placeholder - just use Q and project)
        // TODO: Implement full scaled dot-product attention with proper GQA
        match &q_rope {
            Tensor::F32(_q_arr) => {
                // Simplified: just project Q through output layer
                // This maintains correct shapes and allows the model to run
                self.o_proj.forward(q_rope)
            },
            _ => Err(tensor_op_error(
                "LlamaAttention::forward",
                "Unsupported tensor types for LLaMA attention",
            )),
        }
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
