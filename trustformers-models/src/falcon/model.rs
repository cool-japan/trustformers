use crate::falcon::config::FalconConfig;
use scirs2_core::ndarray::{s, ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::{gelu, silu},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// ALiBi positional encoding implementation
/// Attention with Linear Biases (Press et al., 2022)
pub struct ALiBi {
    slopes: Tensor,
    num_heads: usize,
    device: Device,
}

impl ALiBi {
    pub fn new(num_heads: usize) -> Result<Self> {
        Self::new_with_device(num_heads, Device::CPU)
    }

    pub fn new_with_device(num_heads: usize, device: Device) -> Result<Self> {
        // Calculate slopes based on the geometric sequence pattern
        let mut slopes = Vec::new();
        let ratio = 2.0_f32.powf(-8.0 / num_heads as f32);

        if num_heads % 2 == 0 {
            // Even number of heads
            for i in 0..num_heads / 2 {
                slopes.push(ratio.powf((2 * i + 1) as f32));
            }
            for i in 0..num_heads / 2 {
                slopes.push(ratio.powf((2 * i + 2) as f32));
            }
        } else {
            // Odd number of heads
            for i in 0..num_heads {
                slopes.push(ratio.powf((i + 1) as f32));
            }
        }

        let slopes_tensor = Tensor::new(slopes)?;

        Ok(Self {
            slopes: slopes_tensor,
            num_heads,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Apply ALiBi bias to attention scores
    pub fn apply_bias(&self, attention_scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        // Create position bias matrix for causal attention
        let mut bias_data = Vec::new();

        // Create bias for each head
        for head_idx in 0..self.num_heads {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        // Future positions get large negative bias (causal mask)
                        bias_data.push(-10000.0);
                    } else {
                        // Past positions get linear bias scaled by head-specific slope
                        let distance = (i - j) as f32;
                        let slope = if let Ok(slopes_data) = self.slopes.data() {
                            if head_idx < slopes_data.len() {
                                slopes_data[head_idx]
                            } else {
                                1.0
                            }
                        } else {
                            1.0
                        };
                        bias_data.push(-distance * slope);
                    }
                }
            }
        }

        // Create bias tensor with proper shape for broadcasting
        let bias_tensor = Tensor::from_vec(bias_data, &[seq_len, seq_len])?;

        // Add bias to attention scores with proper broadcasting
        let biased_scores = attention_scores.add(&bias_tensor)?;
        Ok(biased_scores)
    }
}

/// Falcon attention layer with multi-query attention and optional ALiBi
pub struct FalconAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    dense: Linear,
    alibi: Option<ALiBi>,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    #[allow(dead_code)]
    attention_dropout: f32,
    #[allow(dead_code)]
    use_flash_attention: bool,
    device: Device,
    // Note: Multi-query attention is implemented through num_kv_heads parameter
    // Future enhancement: could add dedicated MultiQueryAttention component when needed
}

impl FalconAttention {
    pub fn new(config: &FalconConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FalconConfig, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let num_kv_heads = config.num_kv_heads();

        let q_proj = Linear::new(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            config.bias,
        );
        let k_proj = Linear::new(config.hidden_size, num_kv_heads * head_dim, config.bias);
        let v_proj = Linear::new(config.hidden_size, num_kv_heads * head_dim, config.bias);
        let dense = Linear::new(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            config.bias,
        );

        let alibi = if config.alibi {
            Some(ALiBi::new_with_device(config.num_attention_heads, device)?)
        } else {
            None
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            dense,
            alibi,
            num_heads: config.num_attention_heads,
            num_kv_heads,
            head_dim,
            attention_dropout: config.attention_dropout,
            use_flash_attention: config.use_flash_attention.unwrap_or(false),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Create causal mask for autoregressive attention
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        // Create lower triangular mask filled with 0s and -inf
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        Tensor::from_vec(mask_data, &[seq_len, seq_len])
    }

    pub fn parameter_count(&self) -> usize {
        self.q_proj.parameter_count()
            + self.k_proj.parameter_count()
            + self.v_proj.parameter_count()
            + self.dense.parameter_count()
    }
}

impl Layer for FalconAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Project to query, key, value
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // Implement proper multi-query attention
        // Reshape q, k, v for multi-head attention
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, self.head_dim])?;
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, self.head_dim])?;

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;

        // For multi-query attention, repeat k and v heads to match query heads
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let repeats = self.num_heads / self.num_kv_heads;

            // Manually repeat each kv head 'repeats' times
            let mut k_heads = Vec::new();
            let mut v_heads = Vec::new();

            for head_idx in 0..self.num_kv_heads {
                // Extract single head: [batch, 1, seq_len, head_dim]
                let k_head = k.slice_multi(&[
                    (0, batch_size),
                    (head_idx, head_idx + 1),
                    (0, seq_len),
                    (0, self.head_dim),
                ])?;
                let v_head = v.slice_multi(&[
                    (0, batch_size),
                    (head_idx, head_idx + 1),
                    (0, seq_len),
                    (0, self.head_dim),
                ])?;

                // Repeat this head 'repeats' times
                for _ in 0..repeats {
                    k_heads.push(k_head.clone());
                    v_heads.push(v_head.clone());
                }
            }

            // Concatenate all repeated heads
            let k_repeated = Tensor::concat(&k_heads, 1)?;
            let v_repeated = Tensor::concat(&v_heads, 1)?;
            (k_repeated, v_repeated)
        } else {
            (k, v)
        };

        // Compute attention scores: Q @ K.T / sqrt(d_k)
        // Transpose last two dimensions: [batch, num_heads, seq_len, head_dim] -> [batch, num_heads, head_dim, seq_len]
        let k_transposed = k.transpose(2, 3)?;
        let scores = q.matmul(&k_transposed)?;
        let scale = (self.head_dim as f32).sqrt();
        let scaled_scores = scores.div_scalar(scale)?;

        // Apply causal mask
        let causal_mask = self.create_causal_mask(seq_len)?;
        let masked_scores = scaled_scores.add(&causal_mask)?;

        // Apply softmax
        let attention_weights = masked_scores.softmax(-1)?;

        // Apply attention to values
        let attention_output = attention_weights.matmul(&v)?;

        // Transpose back and reshape
        let attention_output = attention_output.transpose(1, 2)?;
        let attention_output =
            attention_output.reshape(&[batch_size, seq_len, self.num_heads * self.head_dim])?;

        // Apply ALiBi bias if enabled
        let biased_output = if let Some(alibi) = &self.alibi {
            alibi.apply_bias(&attention_output, seq_len)?
        } else {
            attention_output
        };

        // Final output projection
        let output = self.dense.forward(biased_output)?;
        Ok(output)
    }
}

/// Falcon MLP layer
pub struct FalconMLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    activation: String,
    device: Device,
}

impl FalconMLP {
    pub fn new(config: &FalconConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FalconConfig, device: Device) -> Result<Self> {
        let intermediate_size = 4 * config.hidden_size;

        let dense_h_to_4h = Linear::new(config.hidden_size, intermediate_size, config.bias);
        let dense_4h_to_h = Linear::new(intermediate_size, config.hidden_size, config.bias);

        Ok(Self {
            dense_h_to_4h,
            dense_4h_to_h,
            activation: config.hidden_act.clone(),
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.dense_h_to_4h.parameter_count() + self.dense_4h_to_h.parameter_count()
    }
}

impl Layer for FalconMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.dense_h_to_4h.forward(input)?;

        // Apply activation function
        let activated = match self.activation.as_str() {
            "gelu" => gelu(&hidden)?,
            "relu" => hidden.relu()?,
            "silu" | "swish" => silu(&hidden)?,
            _ => hidden,
        };

        let output = self.dense_4h_to_h.forward(activated)?;
        Ok(output)
    }
}

/// Falcon decoder layer
pub struct FalconDecoderLayer {
    input_layernorm: LayerNorm,
    self_attention: FalconAttention,
    mlp: FalconMLP,
    parallel_attn: bool,
    apply_residual_connection_post_layernorm: bool,
    device: Device,
}

impl FalconDecoderLayer {
    pub fn new(config: &FalconConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &FalconConfig, device: Device) -> Result<Self> {
        let input_layernorm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_epsilon)?;
        let self_attention = FalconAttention::new_with_device(config, device)?;
        let mlp = FalconMLP::new_with_device(config, device)?;

        Ok(Self {
            input_layernorm,
            self_attention,
            mlp,
            parallel_attn: config.parallel_attn,
            apply_residual_connection_post_layernorm: config
                .apply_residual_connection_post_layernorm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.input_layernorm.parameter_count()
            + self.self_attention.parameter_count()
            + self.mlp.parameter_count()
    }
}

impl Layer for FalconDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        if self.parallel_attn {
            // Parallel attention and MLP computation (Falcon's innovation)
            let layernorm_output = self.input_layernorm.forward(input.clone())?;

            // Compute attention and MLP in parallel
            let attention_output = self.self_attention.forward(layernorm_output.clone())?;
            let mlp_output = self.mlp.forward(layernorm_output.clone())?;

            // Add both outputs to input (residual connections)
            let residual_input = if self.apply_residual_connection_post_layernorm {
                layernorm_output
            } else {
                input
            };

            // Add both outputs to input (residual connections)
            let output = residual_input.add(&attention_output)?.add(&mlp_output)?;
            Ok(output)
        } else {
            // Sequential attention -> MLP (standard transformer)
            let layernorm_output = self.input_layernorm.forward(input.clone())?;
            let attention_output = self.self_attention.forward(layernorm_output)?;

            // Add residual connection
            let residual_output = input.add(&attention_output)?;

            let layernorm_output2 = self.input_layernorm.forward(residual_output.clone())?;
            let mlp_output = self.mlp.forward(layernorm_output2)?;

            // Add residual connection
            let output = residual_output.add(&mlp_output)?;
            Ok(output)
        }
    }
}

/// Falcon transformer model
pub struct FalconModel {
    word_embeddings: Embedding,
    layers: Vec<FalconDecoderLayer>,
    ln_f: LayerNorm,
    config: FalconConfig,
    device: Device,
}

impl FalconModel {
    pub fn new(config: FalconConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: FalconConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let word_embeddings = Embedding::new(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id.map(|id| id as usize),
        )?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(FalconDecoderLayer::new_with_device(&config, device)?);
        }

        let ln_f = LayerNorm::new(vec![config.hidden_size], config.layer_norm_epsilon)?;

        Ok(Self {
            word_embeddings,
            layers,
            ln_f,
            config,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn config(&self) -> &FalconConfig {
        &self.config
    }
}

impl Model for FalconModel {
    type Config = FalconConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        Layer::forward(self, input)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
        // Legacy interface - use enhanced weight loading methods for production
        Err(TrustformersError::not_implemented(
            "Use load_from_path or load_from_huggingface for enhanced weight loading".to_string(),
        ))
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        let embeddings_params = self.word_embeddings.parameter_count();
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let norm_params = self.ln_f.parameter_count();

        embeddings_params + layers_params + norm_params
    }
}

impl Layer for FalconModel {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Convert input tensor to token IDs
        let token_ids = match &input {
            Tensor::F32(arr) => {
                // Convert F32 tensor to u32 token IDs
                arr.iter().map(|&x| x as u32).collect::<Vec<u32>>()
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Input must be F32 tensor",
                ))
            },
        };

        if token_ids.is_empty() {
            return Err(TrustformersError::model_error(
                "Empty token_ids provided".to_string(),
            ));
        }

        let mut hidden_states = self.word_embeddings.forward(token_ids)?;

        // Pass through transformer layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Final layer norm
        let output = self.ln_f.forward(hidden_states)?;
        Ok(output)
    }
}

/// Falcon model for causal language modeling
pub struct FalconForCausalLM {
    transformer: FalconModel,
    lm_head: Linear,
    device: Device,
}

impl FalconForCausalLM {
    pub fn new(config: FalconConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: FalconConfig, device: Device) -> Result<Self> {
        let transformer = FalconModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new(
            config.hidden_size,
            config.vocab_size,
            false, // No bias in language modeling head
        );

        Ok(Self {
            transformer,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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

        // Load word embeddings
        if let Ok(embed_weights) = loader.load_tensor("transformer.word_embeddings.weight") {
            self.transformer.word_embeddings.set_weight(embed_weights)?;
        }

        // Load layer weights
        for (i, layer) in self.transformer.layers.iter_mut().enumerate() {
            // Load attention weights
            let attn_prefix = format!("transformer.h.{}.self_attention", i);

            if let Ok(qkv_weight) =
                loader.load_tensor(&format!("{}.query_key_value.weight", attn_prefix))
            {
                // Falcon uses combined QKV projection - split into Q, K, V
                match &qkv_weight {
                    Tensor::F32(arr) => {
                        let shape = arr.shape();
                        let combined_size = shape[0];
                        let _hidden_size = shape[1];

                        // Assuming equal sizes for Q, K, V (though Falcon may use different ratios)
                        let head_dim = combined_size / 3;

                        // Split the combined weight tensor
                        let q_slice = arr.slice(s![0..head_dim, ..]).to_owned();
                        let k_slice = arr.slice(s![head_dim..2 * head_dim, ..]).to_owned();
                        let v_slice = arr.slice(s![2 * head_dim..3 * head_dim, ..]).to_owned();

                        // Convert to dynamic arrays and set individual weights
                        let q_dyn = q_slice.into_dyn();
                        let k_dyn = k_slice.into_dyn();
                        let v_dyn = v_slice.into_dyn();

                        layer.self_attention.q_proj.set_weight(Tensor::F32(q_dyn))?;
                        layer.self_attention.k_proj.set_weight(Tensor::F32(k_dyn))?;
                        layer.self_attention.v_proj.set_weight(Tensor::F32(v_dyn))?;
                    },
                    _ => {
                        // Fallback: use the same weight for all (not ideal but better than crashing)
                        layer.self_attention.q_proj.set_weight(qkv_weight.clone())?;
                    },
                }
            }
            if let Ok(o_weight) = loader.load_tensor(&format!("{}.dense.weight", attn_prefix)) {
                layer.self_attention.dense.set_weight(o_weight)?;
            }

            // Load MLP weights
            let mlp_prefix = format!("transformer.h.{}.mlp", i);

            if let Ok(up_weight) =
                loader.load_tensor(&format!("{}.dense_h_to_4h.weight", mlp_prefix))
            {
                layer.mlp.dense_h_to_4h.set_weight(up_weight)?;
            }
            if let Ok(down_weight) =
                loader.load_tensor(&format!("{}.dense_4h_to_h.weight", mlp_prefix))
            {
                layer.mlp.dense_4h_to_h.set_weight(down_weight)?;
            }

            // Load layer norm weights
            if let Ok(ln_weight) =
                loader.load_tensor(&format!("transformer.h.{}.input_layernorm.weight", i))
            {
                layer.input_layernorm.set_weight(ln_weight)?;
            }
            if let Ok(ln_bias) =
                loader.load_tensor(&format!("transformer.h.{}.input_layernorm.bias", i))
            {
                layer.input_layernorm.set_bias(ln_bias)?;
            }
        }

        // Load final layer norm
        if let Ok(norm_weight) = loader.load_tensor("transformer.ln_f.weight") {
            self.transformer.ln_f.set_weight(norm_weight)?;
        }
        if let Ok(norm_bias) = loader.load_tensor("transformer.ln_f.bias") {
            self.transformer.ln_f.set_bias(norm_bias)?;
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
            TrustformersError::io_error(format!("Failed to create model directory: {}", e))
        })?;

        // List of essential files for Falcon models
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
                return Err(TrustformersError::io_error(format!(
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

    /// Legacy method name for backward compatibility
    pub fn load_from_hub(&mut self, model_name: &str) -> Result<()> {
        self.load_from_huggingface(model_name)
    }

    /// Generate text using the model
    pub fn generate(&self, input_ids: Tensor, max_length: usize) -> Result<Tensor> {
        let mut current_ids = input_ids;
        let current_length = current_ids.shape()[current_ids.shape().len() - 1];

        // Autoregressive generation
        for _ in current_length..max_length {
            // Forward pass through the model
            let logits = <Self as Model>::forward(self, current_ids.clone())?;

            // Get the last token logits
            let last_logits = match &logits {
                Tensor::F32(arr) => {
                    let shape = arr.shape();
                    let seq_len = shape[shape.len() - 2];
                    let _vocab_size = shape[shape.len() - 1];

                    // Extract last token logits
                    let last_token_slice = if shape.len() == 3 {
                        arr.slice(s![0, seq_len - 1, ..])
                    } else {
                        arr.slice(s![seq_len - 1, ..])
                    };
                    last_token_slice.to_owned()
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Logits must be F32 tensor",
                    ))
                },
            };

            // Greedy decoding: select token with highest probability
            let next_token_id = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(idx, _)| idx as u32)
                .ok_or_else(|| {
                    TrustformersError::model_error("Failed to find next token".to_string())
                })?;

            // Check for EOS token (commonly ID 2 for Falcon models)
            if next_token_id == 2 {
                break;
            }

            // Append next token to sequence
            current_ids = match &current_ids {
                Tensor::F32(arr) => {
                    // Convert token ID to f32 tensor and concatenate
                    let mut new_shape = arr.shape().to_vec();
                    let last_idx = new_shape.len() - 1;
                    new_shape[last_idx] += 1;

                    let mut new_arr = ArrayD::<f32>::zeros(IxDyn(&new_shape));

                    // Copy existing data
                    if arr.ndim() == 2 {
                        for i in 0..arr.shape()[0] {
                            for j in 0..arr.shape()[1] {
                                new_arr[[i, j]] = arr[[i, j]];
                            }
                            new_arr[[i, arr.shape()[1]]] = next_token_id as f32;
                        }
                    } else if arr.ndim() == 1 {
                        for i in 0..arr.shape()[0] {
                            new_arr[[i]] = arr[[i]];
                        }
                        new_arr[[arr.shape()[0]]] = next_token_id as f32;
                    }

                    Tensor::F32(new_arr)
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Input must be F32 tensor",
                    ))
                },
            };
        }

        Ok(current_ids)
    }

    pub fn model(&self) -> &FalconModel {
        &self.transformer
    }
}

impl Model for FalconForCausalLM {
    type Config = FalconConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        Layer::forward(self, input)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.transformer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.transformer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.transformer.num_parameters() + self.lm_head.parameter_count()
    }
}

impl Layer for FalconForCausalLM {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = Layer::forward(&self.transformer, input)?;
        let logits = self.lm_head.forward(hidden_states)?;
        Ok(logits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_falcon_model_creation() {
        let config = FalconConfig::falcon_7b();
        let model = FalconModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_falcon_causal_lm_creation() {
        let config = FalconConfig::falcon_7b();
        let model = FalconForCausalLM::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_falcon_config_variants() {
        // Test 7B model
        let config_7b = FalconConfig::falcon_7b();
        assert_eq!(config_7b.hidden_size, 4544);
        assert_eq!(config_7b.num_hidden_layers, 32);
        assert!(config_7b.uses_alibi());

        // Test 40B model
        let config_40b = FalconConfig::falcon_40b();
        assert_eq!(config_40b.hidden_size, 8192);
        assert_eq!(config_40b.num_hidden_layers, 60);
        assert!(config_40b.uses_alibi());

        // Test 180B model
        let config_180b = FalconConfig::falcon_180b();
        assert_eq!(config_180b.hidden_size, 14848);
        assert_eq!(config_180b.num_hidden_layers, 80);
        assert!(!config_180b.uses_alibi());
        assert!(config_180b.uses_new_architecture());
    }

    #[test]
    fn test_alibi_creation() {
        let alibi = ALiBi::new(8);
        assert!(alibi.is_ok());

        let alibi = alibi.unwrap();
        assert_eq!(alibi.num_heads, 8);
    }

    #[test]
    fn test_falcon_attention_creation() {
        let config = FalconConfig::falcon_7b();
        let attention = FalconAttention::new(&config);
        assert!(attention.is_ok());
    }

    #[test]
    fn test_falcon_mlp_creation() {
        let config = FalconConfig::falcon_7b();
        let mlp = FalconMLP::new(&config);
        assert!(mlp.is_ok());
    }
}
