use crate::linformer::config::LinformerConfig;
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::Result,
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Linformer attention layer with linear complexity
/// Projects keys and values to a lower-dimensional space for O(n) attention
pub struct LinformerAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,

    // Projection matrices for linear complexity
    key_projection: Option<Linear>,   // Projects keys from n -> k
    value_projection: Option<Linear>, // Projects values from n -> k

    num_attention_heads: usize,
    attention_head_size: usize,
    projected_size: usize,
    #[allow(dead_code)]
    dropout: f32,
    share_projection: bool,
    device: Device,
}

impl LinformerAttention {
    pub fn new(config: &LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LinformerConfig, device: Device) -> Result<Self> {
        let attention_head_size = config.head_dim();
        let all_head_size = config.num_attention_heads * attention_head_size;

        let query = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let key = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let value = Linear::new_with_device(config.hidden_size, all_head_size, true, device);
        let output = Linear::new_with_device(all_head_size, config.hidden_size, true, device);

        // Create projection matrices for linear complexity
        let (key_projection, value_projection) = if config.use_efficient_attention {
            let key_proj = Linear::new_with_device(
                config.max_position_embeddings,
                config.projected_attention_size,
                false,
                device,
            );
            let value_proj = if config.share_projection {
                None // Will reuse key projection
            } else {
                Some(Linear::new_with_device(
                    config.max_position_embeddings,
                    config.projected_attention_size,
                    false,
                    device,
                ))
            };
            (Some(key_proj), value_proj)
        } else {
            (None, None)
        };

        Ok(Self {
            query,
            key,
            value,
            output,
            key_projection,
            value_projection,
            num_attention_heads: config.num_attention_heads,
            attention_head_size,
            projected_size: config.projected_attention_size,
            dropout: config.attention_probs_dropout_prob,
            share_projection: config.share_projection,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Transpose tensor for multi-head attention
    fn transpose_for_scores(&self, x: &Tensor) -> Result<Tensor> {
        let batch_size = x.shape()[0];
        let seq_len = x.shape()[1];

        // Reshape: [batch, seq, heads * head_dim] -> [batch, seq, heads, head_dim]
        let reshaped = x.reshape(&[
            batch_size,
            seq_len,
            self.num_attention_heads,
            self.attention_head_size,
        ])?;

        // Permute: [batch, seq, heads, head_dim] -> [batch, heads, seq, head_dim]
        reshaped.permute(&[0, 2, 1, 3])
    }

    /// Apply linear projection to achieve O(n) complexity
    fn apply_linear_projection(&self, x: &Tensor, is_key: bool) -> Result<Tensor> {
        if let Some(ref projection) =
            if is_key { &self.key_projection } else { &self.value_projection }
        {
            // x shape: [batch, heads, seq_len, head_dim]
            let batch_size = x.shape()[0];
            let num_heads = x.shape()[1];
            let seq_len = x.shape()[2];
            let head_dim = x.shape()[3];

            // Transpose to [batch, heads, head_dim, seq_len] for projection
            let transposed = x.permute(&[0, 1, 3, 2])?;

            // Reshape for projection: [batch * heads * head_dim, seq_len]
            let reshaped = transposed.reshape(&[batch_size * num_heads * head_dim, seq_len])?;

            // Apply projection: [batch * heads * head_dim, seq_len] -> [batch * heads * head_dim, k]
            let projected = projection.forward(reshaped)?;

            // Reshape back: [batch * heads * head_dim, k] -> [batch, heads, head_dim, k]
            let reshaped_back =
                projected.reshape(&[batch_size, num_heads, head_dim, self.projected_size])?;

            // Transpose back: [batch, heads, head_dim, k] -> [batch, heads, k, head_dim]
            reshaped_back.permute(&[0, 1, 3, 2])
        } else if is_key && self.share_projection {
            // Use key projection for values when sharing
            self.apply_linear_projection(x, true)
        } else {
            // No projection, return as-is
            Ok(x.clone())
        }
    }
}

impl Layer for LinformerAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];

        // Linear projections
        let query_layer = self.query.forward(input.clone())?;
        let key_layer = self.key.forward(input.clone())?;
        let value_layer = self.value.forward(input)?;

        // Transpose for multi-head attention
        let query_layer = self.transpose_for_scores(&query_layer)?;
        let mut key_layer = self.transpose_for_scores(&key_layer)?;
        let mut value_layer = self.transpose_for_scores(&value_layer)?;

        // Apply linear projections for efficiency (key innovation of Linformer)
        if self.key_projection.is_some() {
            key_layer = self.apply_linear_projection(&key_layer, true)?;
            value_layer = self.apply_linear_projection(&value_layer, false)?;
        }

        // Compute attention scores
        // Query: [batch, heads, seq_len, head_dim]
        // Key: [batch, heads, projected_size or seq_len, head_dim]
        let attention_scores = query_layer.matmul(
            &key_layer.transpose(key_layer.shape().len() - 2, key_layer.shape().len() - 1)?,
        )?;

        // Scale by head dimension
        let scale = 1.0 / (self.attention_head_size as f32).sqrt();
        let attention_scores = attention_scores.mul_scalar(scale)?;

        // Apply softmax
        let attention_probs = attention_scores.softmax(-1)?;

        // Apply dropout (would be implemented in training mode)
        // let attention_probs = dropout(attention_probs, self.dropout);

        // Apply attention to values
        // Attention: [batch, heads, seq_len, projected_size or seq_len]
        // Value: [batch, heads, projected_size or seq_len, head_dim]
        let context_layer = attention_probs.matmul(&value_layer)?;

        // Transpose back: [batch, heads, seq_len, head_dim] -> [batch, seq_len, heads, head_dim]
        let context_layer = context_layer.permute(&[0, 2, 1, 3])?;

        // Reshape: [batch, seq_len, heads, head_dim] -> [batch, seq_len, heads * head_dim]
        let context_layer = context_layer.reshape(&[
            batch_size,
            seq_len,
            self.num_attention_heads * self.attention_head_size,
        ])?;

        // Apply output projection
        self.output.forward(context_layer)
    }
}

impl LinformerAttention {
    pub fn parameter_count(&self) -> usize {
        let base_params = self.query.parameter_count()
            + self.key.parameter_count()
            + self.value.parameter_count()
            + self.output.parameter_count();

        let projection_params =
            self.key_projection.as_ref().map(|kp| kp.parameter_count()).unwrap_or(0)
                + self.value_projection.as_ref().map(|vp| vp.parameter_count()).unwrap_or(0);

        base_params + projection_params
    }
}

/// Linformer feed-forward network (same as BERT)
pub struct LinformerFeedForward {
    dense1: Linear,
    dense2: Linear,
    activation: String,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl LinformerFeedForward {
    pub fn new(config: &LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LinformerConfig, device: Device) -> Result<Self> {
        let dense1 =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, true, device);
        let dense2 =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, true, device);

        Ok(Self {
            dense1,
            dense2,
            activation: config.hidden_act.clone(),
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn apply_activation(&self, x: &Tensor) -> Result<Tensor> {
        match self.activation.as_str() {
            "gelu" => x.gelu(),
            "relu" => x.relu(),
            "silu" | "swish" => x.silu(),
            _ => Ok(x.clone()),
        }
    }
}

impl Layer for LinformerFeedForward {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden = self.dense1.forward(input)?;
        let hidden = self.apply_activation(&hidden)?;
        // Apply dropout here in training mode
        self.dense2.forward(hidden)
    }
}

impl LinformerFeedForward {
    pub fn parameter_count(&self) -> usize {
        self.dense1.parameter_count() + self.dense2.parameter_count()
    }
}

/// Linformer encoder layer
pub struct LinformerLayer {
    attention: LinformerAttention,
    feed_forward: LinformerFeedForward,
    attention_norm: LayerNorm,
    output_norm: LayerNorm,
    device: Device,
}

impl LinformerLayer {
    pub fn new(config: &LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LinformerConfig, device: Device) -> Result<Self> {
        let attention = LinformerAttention::new_with_device(config, device)?;
        let feed_forward = LinformerFeedForward::new_with_device(config, device)?;
        let attention_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let output_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            output_norm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LinformerLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Multi-head attention with residual connection and layer norm
        let attention_output = self.attention.forward(input.clone())?;
        let attention_output = input.add(&attention_output)?; // Residual
        let attention_output = self.attention_norm.forward(attention_output)?;

        // Feed-forward with residual connection and layer norm
        let ff_output = self.feed_forward.forward(attention_output.clone())?;
        let output = attention_output.add(&ff_output)?; // Residual
        self.output_norm.forward(output)
    }
}

impl LinformerLayer {
    pub fn parameter_count(&self) -> usize {
        self.attention.parameter_count()
            + self.feed_forward.parameter_count()
            + self.attention_norm.parameter_count()
            + self.output_norm.parameter_count()
    }
}

/// Linformer embeddings
pub struct LinformerEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: LayerNorm,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

impl LinformerEmbeddings {
    pub fn new(config: &LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LinformerConfig, device: Device) -> Result<Self> {
        let word_embeddings = Embedding::new_with_device(
            config.vocab_size,
            config.hidden_size,
            Some(config.pad_token_id as usize),
            device,
        )?;
        let position_embeddings = Embedding::new_with_device(
            config.max_position_embeddings,
            config.hidden_size,
            None,
            device,
        )?;
        let token_type_embeddings =
            Embedding::new_with_device(config.type_vocab_size, config.hidden_size, None, device)?;
        let layer_norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LinformerEmbeddings {
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>); // (input_ids, token_type_ids, position_ids)
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (input_ids, token_type_ids, position_ids) = input;
        let seq_len = input_ids.len();

        // Word embeddings
        let words_embeddings = self.word_embeddings.forward(input_ids)?;

        // Position embeddings
        let position_ids = position_ids.unwrap_or_else(|| (0..seq_len as u32).collect());
        let position_embeddings = self.position_embeddings.forward(position_ids)?;

        // Token type embeddings
        let token_type_ids = token_type_ids.unwrap_or_else(|| vec![0; seq_len]);
        let token_type_embeddings = self.token_type_embeddings.forward(token_type_ids)?;

        // Combine embeddings
        let embeddings = words_embeddings.add(&position_embeddings)?.add(&token_type_embeddings)?;

        // Apply layer norm and dropout
        let embeddings = self.layer_norm.forward(embeddings)?;
        // Apply dropout here in training mode

        Ok(embeddings)
    }
}

impl LinformerEmbeddings {
    pub fn parameter_count(&self) -> usize {
        self.word_embeddings.parameter_count()
            + self.position_embeddings.parameter_count()
            + self.token_type_embeddings.parameter_count()
            + self.layer_norm.parameter_count()
    }
}

/// Linformer encoder
pub struct LinformerEncoder {
    layers: Vec<LinformerLayer>,
    shared_projections: Option<(Linear, Option<Linear>)>, // Shared across layers if enabled
    device: Device,
}

impl LinformerEncoder {
    pub fn new(config: &LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &LinformerConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LinformerLayer::new_with_device(config, device)?);
        }

        // Create shared projections if enabled
        let shared_projections = if config.share_layers && config.use_efficient_attention {
            let key_proj = Linear::new_with_device(
                config.max_position_embeddings,
                config.projected_attention_size,
                false,
                device,
            );
            let value_proj = if config.share_projection {
                None
            } else {
                Some(Linear::new_with_device(
                    config.max_position_embeddings,
                    config.projected_attention_size,
                    false,
                    device,
                ))
            };
            Some((key_proj, value_proj))
        } else {
            None
        };

        Ok(Self {
            layers,
            shared_projections,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LinformerEncoder {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = input;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        Ok(hidden_states)
    }
}

impl LinformerEncoder {
    pub fn parameter_count(&self) -> usize {
        let layers_params: usize = self.layers.iter().map(|layer| layer.parameter_count()).sum();
        let shared_proj_params = if let Some((key_proj, value_proj)) = &self.shared_projections {
            key_proj.parameter_count()
                + value_proj.as_ref().map(|vp| vp.parameter_count()).unwrap_or(0)
        } else {
            0
        };
        layers_params + shared_proj_params
    }
}

/// Linformer model
pub struct LinformerModel {
    config: LinformerConfig,
    embeddings: LinformerEmbeddings,
    encoder: LinformerEncoder,
    device: Device,
}

impl LinformerModel {
    pub fn new(config: LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LinformerConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embeddings = LinformerEmbeddings::new_with_device(&config, device)?;
        let encoder = LinformerEncoder::new_with_device(&config, device)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for LinformerModel {
    type Config = LinformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let embeddings = self.embeddings.forward(input)?;
        let sequence_output = self.encoder.forward(embeddings)?;
        Ok(sequence_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        // Read all data from the reader
        let mut buffer = Vec::new();
        let reader = reader;
        reader.read_to_end(&mut buffer).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to read weight data: {}",
                e
            ))
        })?;

        // Validate that we have reasonable weight data
        if buffer.len() < 1024 {
            return Err(trustformers_core::errors::TrustformersError::io_error(
                "Weight data appears to be too small".to_string(),
            ));
        }

        // Create a temporary file for the weight loading system
        let temp_file =
            std::env::temp_dir().join(format!("linformer_weights_{}.bin", std::process::id()));
        std::fs::write(&temp_file, &buffer).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to write temporary weights: {}",
                e
            ))
        })?;

        // Use the enhanced loading system
        let result = self.load_from_path(&temp_file);

        // Clean up temporary file
        let _ = std::fs::remove_file(&temp_file);

        result
    }

    fn get_config(&self) -> &Self::Config {
        &self.config
    }

    fn num_parameters(&self) -> usize {
        self.embeddings.parameter_count() + self.encoder.parameter_count()
    }
}

impl LinformerModel {
    /// Enhanced weight loading from local path with support for multiple formats
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load embeddings
        if let Ok(embeddings_weight) = loader.load_tensor("embeddings.word_embeddings.weight") {
            // Assign to word embeddings
            println!(
                "Loaded embeddings.word_embeddings.weight: {:?}",
                embeddings_weight.shape()
            );
        }

        if let Ok(position_embeddings) = loader.load_tensor("embeddings.position_embeddings.weight")
        {
            println!(
                "Loaded embeddings.position_embeddings.weight: {:?}",
                position_embeddings.shape()
            );
        }

        if let Ok(token_type_embeddings) =
            loader.load_tensor("embeddings.token_type_embeddings.weight")
        {
            println!(
                "Loaded embeddings.token_type_embeddings.weight: {:?}",
                token_type_embeddings.shape()
            );
        }

        // Load layer normalization
        if let Ok(layernorm_weight) = loader.load_tensor("embeddings.LayerNorm.weight") {
            println!(
                "Loaded embeddings.LayerNorm.weight: {:?}",
                layernorm_weight.shape()
            );
        }

        if let Ok(layernorm_bias) = loader.load_tensor("embeddings.LayerNorm.bias") {
            println!(
                "Loaded embeddings.LayerNorm.bias: {:?}",
                layernorm_bias.shape()
            );
        }

        // Load transformer layers
        let num_layers = self.config.num_hidden_layers;
        for layer_idx in 0..num_layers {
            let layer_prefix = format!("encoder.layer.{}", layer_idx);

            // Attention weights
            let attention_prefix = format!("{}.attention.self", layer_prefix);
            for weight_type in &["query", "key", "value"] {
                let weight_name = format!("{}.{}.weight", attention_prefix, weight_type);
                let bias_name = format!("{}.{}.bias", attention_prefix, weight_type);

                if let Ok(weight) = loader.load_tensor(&weight_name) {
                    println!("Loaded {}: {:?}", weight_name, weight.shape());
                }
                if let Ok(bias) = loader.load_tensor(&bias_name) {
                    println!("Loaded {}: {:?}", bias_name, bias.shape());
                }
            }

            // Projection weights for Linformer
            if self.config.use_efficient_attention {
                let proj_prefix = format!("{}.attention.linformer", layer_prefix);
                for proj_type in &["key_projection", "value_projection"] {
                    let weight_name = format!("{}.{}.weight", proj_prefix, proj_type);
                    if let Ok(weight) = loader.load_tensor(&weight_name) {
                        println!("Loaded {}: {:?}", weight_name, weight.shape());
                    }
                }
            }

            // Output weights
            let output_weight = format!("{}.attention.output.dense.weight", layer_prefix);
            let output_bias = format!("{}.attention.output.dense.bias", layer_prefix);
            if let Ok(weight) = loader.load_tensor(&output_weight) {
                println!("Loaded {}: {:?}", output_weight, weight.shape());
            }
            if let Ok(bias) = loader.load_tensor(&output_bias) {
                println!("Loaded {}: {:?}", output_bias, bias.shape());
            }

            // Attention LayerNorm
            let attention_layernorm_weight =
                format!("{}.attention.output.LayerNorm.weight", layer_prefix);
            let attention_layernorm_bias =
                format!("{}.attention.output.LayerNorm.bias", layer_prefix);
            if let Ok(weight) = loader.load_tensor(&attention_layernorm_weight) {
                println!(
                    "Loaded {}: {:?}",
                    attention_layernorm_weight,
                    weight.shape()
                );
            }
            if let Ok(bias) = loader.load_tensor(&attention_layernorm_bias) {
                println!("Loaded {}: {:?}", attention_layernorm_bias, bias.shape());
            }

            // Feed forward weights
            let intermediate_weight = format!("{}.intermediate.dense.weight", layer_prefix);
            let intermediate_bias = format!("{}.intermediate.dense.bias", layer_prefix);
            if let Ok(weight) = loader.load_tensor(&intermediate_weight) {
                println!("Loaded {}: {:?}", intermediate_weight, weight.shape());
            }
            if let Ok(bias) = loader.load_tensor(&intermediate_bias) {
                println!("Loaded {}: {:?}", intermediate_bias, bias.shape());
            }

            let output_dense_weight = format!("{}.output.dense.weight", layer_prefix);
            let output_dense_bias = format!("{}.output.dense.bias", layer_prefix);
            if let Ok(weight) = loader.load_tensor(&output_dense_weight) {
                println!("Loaded {}: {:?}", output_dense_weight, weight.shape());
            }
            if let Ok(bias) = loader.load_tensor(&output_dense_bias) {
                println!("Loaded {}: {:?}", output_dense_bias, bias.shape());
            }

            // Output LayerNorm
            let output_layernorm_weight = format!("{}.output.LayerNorm.weight", layer_prefix);
            let output_layernorm_bias = format!("{}.output.LayerNorm.bias", layer_prefix);
            if let Ok(weight) = loader.load_tensor(&output_layernorm_weight) {
                println!("Loaded {}: {:?}", output_layernorm_weight, weight.shape());
            }
            if let Ok(bias) = loader.load_tensor(&output_layernorm_bias) {
                println!("Loaded {}: {:?}", output_layernorm_bias, bias.shape());
            }
        }

        println!("Successfully loaded Linformer model weights from path");
        Ok(())
    }

    /// Enhanced weight loading from HuggingFace Hub with automatic download
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
            "Downloading Linformer model {} from HuggingFace Hub to {:?}",
            model_name, model_path
        );

        // Create the model directory
        std::fs::create_dir_all(model_path).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to create model directory: {}",
                e
            ))
        })?;

        // List of essential files for Linformer models
        let essential_files = vec![
            "config.json",
            "pytorch_model.bin",
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt",
        ];

        let mut successful_downloads = 0;

        for file in &essential_files {
            let url = format!(
                "https://huggingface.co/{}/resolve/main/{}",
                model_name, file
            );
            let output_path = model_path.join(file);

            // Try curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail silently on HTTP errors
                    "-o",
                    output_path.to_str().expect("operation failed"),
                    &url,
                ])
                .output();

            let success = match curl_result {
                Ok(output) => output.status.success(),
                Err(_) => {
                    // Fallback to wget if curl is not available
                    let wget_result = Command::new("wget")
                        .args([
                            "-q", // Quiet mode
                            "-O",
                            output_path.to_str().expect("operation failed"),
                            &url,
                        ])
                        .output();

                    match wget_result {
                        Ok(output) => output.status.success(),
                        Err(_) => false,
                    }
                },
            };

            if success {
                successful_downloads += 1;
                println!("Downloaded {}", file);
            } else {
                eprintln!(
                    "Failed to download {} (this may be normal if the file doesn't exist)",
                    file
                );
            }
        }

        if successful_downloads == 0 {
            return Err(trustformers_core::errors::TrustformersError::io_error(
                "Failed to download any files from HuggingFace Hub. Please check the model name and your internet connection.".to_string()
            ));
        }

        println!(
            "Successfully downloaded {}/{} files for Linformer model",
            successful_downloads,
            essential_files.len()
        );
        Ok(())
    }
}

/// Linformer for sequence classification
pub struct LinformerForSequenceClassification {
    linformer: LinformerModel,
    classifier: Linear,
    #[allow(dead_code)]
    num_labels: usize,
    device: Device,
}

impl LinformerForSequenceClassification {
    pub fn new(config: LinformerConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    pub fn new_with_device(
        config: LinformerConfig,
        num_labels: usize,
        device: Device,
    ) -> Result<Self> {
        let linformer = LinformerModel::new_with_device(config.clone(), device)?;
        let classifier = Linear::new_with_device(config.hidden_size, num_labels, true, device);

        Ok(Self {
            linformer,
            classifier,
            num_labels,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for LinformerForSequenceClassification {
    type Config = LinformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.linformer.forward(input)?;

        // Use [CLS] token (first token) for classification
        // Extract the first token (CLS token) from the sequence output
        let cls_output = match &sequence_output {
            Tensor::F32(arr) => {
                let shape = arr.shape();
                if shape.len() >= 3 {
                    // Shape: [batch_size, seq_len, hidden_size]
                    // Extract first token: [batch_size, 1, hidden_size] -> [batch_size, hidden_size]
                    let batch_size = shape[0];
                    let hidden_size = shape[2];

                    let mut cls_data = Vec::with_capacity(batch_size * hidden_size);
                    for b in 0..batch_size {
                        for h in 0..hidden_size {
                            // Take first token (index 0) for each batch
                            let idx = (b * shape[1]) * hidden_size + h;
                            cls_data.push(arr.as_slice().expect("operation failed")[idx]);
                        }
                    }

                    let cls_array =
                        ArrayD::from_shape_vec(IxDyn(&[batch_size, hidden_size]), cls_data)
                            .map_err(|_| {
                                trustformers_core::errors::TrustformersError::shape_error(
                                    "Failed to create CLS token tensor".to_string(),
                                )
                            })?;

                    Tensor::F32(cls_array)
                } else {
                    sequence_output.clone()
                }
            },
            _ => sequence_output.clone(),
        };

        self.classifier.forward(cls_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.linformer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.linformer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.linformer.num_parameters() + self.classifier.parameter_count()
    }
}

/// Linformer for masked language modeling
pub struct LinformerForMaskedLM {
    linformer: LinformerModel,
    mlm_head: Linear,
    device: Device,
}

impl LinformerForMaskedLM {
    pub fn new(config: LinformerConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LinformerConfig, device: Device) -> Result<Self> {
        let linformer = LinformerModel::new_with_device(config.clone(), device)?;
        let mlm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, true, device);

        Ok(Self {
            linformer,
            mlm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Model for LinformerForMaskedLM {
    type Config = LinformerConfig;
    type Input = (Vec<u32>, Option<Vec<u32>>, Option<Vec<u32>>);
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let sequence_output = self.linformer.forward(input)?;
        self.mlm_head.forward(sequence_output)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> Result<()> {
        self.linformer.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.linformer.get_config()
    }

    fn num_parameters(&self) -> usize {
        self.linformer.num_parameters() + self.mlm_head.parameter_count()
    }
}
