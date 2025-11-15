use crate::gpt_neo::config::GptNeoConfig;
use scirs2_core::ndarray::{s, ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::device::Device;
use trustformers_core::errors::{tensor_op_error, Result, TrustformersError};
use trustformers_core::layers::{Embedding, LayerNorm, Linear};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer, Model, TokenizedInput};

#[derive(Debug, Clone)]
pub struct GptNeoModel {
    config: GptNeoConfig,
    wte: Embedding,
    wpe: Embedding,
    layers: Vec<GptNeoBlock>,
    ln_f: LayerNorm,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct GptNeoBlock {
    ln_1: LayerNorm,
    attn: GptNeoAttention,
    ln_2: LayerNorm,
    mlp: GptNeoMLP,
    #[allow(dead_code)]
    attention_type: String,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct GptNeoAttention {
    attention: MultiHeadAttention,
    #[allow(dead_code)]
    attention_type: String,
    window_size: Option<usize>,
    device: Device,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct MultiHeadAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    #[allow(dead_code)]
    num_heads: usize,
    head_dim: usize,
    dropout: f32,
    device: Device,
}

#[derive(Debug, Clone)]
pub struct GptNeoMLP {
    c_fc: Linear,
    c_proj: Linear,
    activation: String,
    #[allow(dead_code)]
    dropout: f32,
    device: Device,
}

#[derive(Debug)]
pub struct GptNeoModelOutput {
    pub last_hidden_state: Tensor,
    pub hidden_states: Option<Vec<Tensor>>,
}

impl GptNeoModel {
    pub fn new(config: GptNeoConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: GptNeoConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let wte = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let wpe = Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for i in 0..config.num_layers {
            let attention_type = config
                .attention_types
                .get(i % config.attention_types.len())
                .cloned()
                .unwrap_or_else(|| "global".to_string());

            layers.push(GptNeoBlock::new_with_device(
                &config,
                attention_type,
                device,
            )?);
        }

        let ln_f = LayerNorm::new(vec![config.hidden_size], config.layer_norm_epsilon)?;

        Ok(Self {
            config,
            wte,
            wpe,
            layers,
            ln_f,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl GptNeoBlock {
    fn new(config: &GptNeoConfig, attention_type: String) -> Result<Self> {
        Self::new_with_device(config, attention_type, Device::CPU)
    }

    fn new_with_device(
        config: &GptNeoConfig,
        attention_type: String,
        device: Device,
    ) -> Result<Self> {
        let ln_1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_epsilon)?;
        let ln_2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_epsilon)?;

        let attn = GptNeoAttention::new_with_device(config, &attention_type, device)?;
        let mlp = GptNeoMLP::new_with_device(config, device)?;

        Ok(Self {
            ln_1,
            attn,
            ln_2,
            mlp,
            attention_type,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm architecture
        let normed_hidden_states = self.ln_1.forward(hidden_states.clone())?;
        let attn_output = self.attn.forward(normed_hidden_states, attention_mask)?;
        let hidden_states = hidden_states.add(&attn_output)?;

        let normed_hidden_states = self.ln_2.forward(hidden_states.clone())?;
        let mlp_output = self.mlp.forward(normed_hidden_states)?;
        let hidden_states = hidden_states.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

impl GptNeoAttention {
    fn new(config: &GptNeoConfig, attention_type: &str) -> Result<Self> {
        Self::new_with_device(config, attention_type, Device::CPU)
    }

    fn new_with_device(
        config: &GptNeoConfig,
        attention_type: &str,
        device: Device,
    ) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        let attention = MultiHeadAttention::new_with_device(config, device)?;

        let window_size = if attention_type == "local" { Some(config.window_size) } else { None };

        Ok(Self {
            attention,
            attention_type: attention_type.to_string(),
            window_size,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // For now, implement as global attention regardless of type
        // In a full implementation, local attention would use sliding window
        self.attention.forward(hidden_states, attention_mask)
    }
}

impl MultiHeadAttention {
    fn new(config: &GptNeoConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &GptNeoConfig, device: Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_heads;
        Ok(Self {
            q_proj: Linear::new(config.hidden_size, config.hidden_size, false),
            k_proj: Linear::new(config.hidden_size, config.hidden_size, false),
            v_proj: Linear::new(config.hidden_size, config.hidden_size, false),
            out_proj: Linear::new(config.hidden_size, config.hidden_size, true),
            num_heads: config.num_heads,
            head_dim,
            dropout: config.attention_dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor, _attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let _batch_size = hidden_states.shape()[0];
        let _seq_len = hidden_states.shape()[1];

        // Compute Q, K, V
        let _q = self.q_proj.forward(hidden_states.clone())?;
        let _k = self.k_proj.forward(hidden_states.clone())?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape to [batch_size, num_heads, seq_len, head_dim]
        // For now, we'll use a simplified attention computation
        // In a full implementation, we'd properly reshape and compute attention scores

        // Simplified attention: just use V values for now
        let output = self.out_proj.forward(v)?;

        Ok(output)
    }
}

impl GptNeoMLP {
    fn new(config: &GptNeoConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    fn new_with_device(config: &GptNeoConfig, device: Device) -> Result<Self> {
        Ok(Self {
            c_fc: Linear::new(config.hidden_size, config.intermediate_size, true),
            c_proj: Linear::new(config.intermediate_size, config.hidden_size, true),
            activation: config.activation_function.clone(),
            dropout: config.resid_dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    fn forward(&self, hidden_states: Tensor) -> Result<Tensor> {
        let hidden_states = self.c_fc.forward(hidden_states)?;

        // Apply activation function
        let hidden_states = match self.activation.as_str() {
            "gelu" | "gelu_new" => trustformers_core::ops::activations::gelu(&hidden_states)?,
            "relu" => trustformers_core::ops::activations::relu(&hidden_states)?,
            _ => {
                return Err(trustformers_core::errors::TrustformersError::model_error(
                    format!("Unsupported activation function: {}", self.activation),
                ));
            },
        };

        self.c_proj.forward(hidden_states)
    }
}

impl Model for GptNeoModel {
    type Config = GptNeoConfig;
    type Input = TokenizedInput;
    type Output = GptNeoModelOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Create position IDs
        let seq_len = input.input_ids.len();
        let position_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();

        // Get embeddings
        let inputs_embeds = self.wte.forward(input.input_ids)?;
        let position_embeds = self.wpe.forward(position_ids)?;
        let mut hidden_states = inputs_embeds.add(&position_embeds)?;

        // Pass through layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states, None)?;
        }

        // Final layer norm
        let last_hidden_state = self.ln_f.forward(hidden_states)?;

        Ok(GptNeoModelOutput {
            last_hidden_state,
            hidden_states: None,
        })
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
        let mut total = 0;

        // Word token embeddings
        total += self.wte.parameter_count();

        // Word position embeddings
        total += self.wpe.parameter_count();

        // Transformer blocks
        for layer in &self.layers {
            total += layer.ln_1.parameter_count();
            total += layer.attn.attention.q_proj.parameter_count();
            total += layer.attn.attention.k_proj.parameter_count();
            total += layer.attn.attention.v_proj.parameter_count();
            total += layer.attn.attention.out_proj.parameter_count();
            total += layer.ln_2.parameter_count();
            total += layer.mlp.c_fc.parameter_count();
            total += layer.mlp.c_proj.parameter_count();
        }

        // Final layer norm
        total += self.ln_f.parameter_count();

        total
    }
}

#[derive(Debug, Clone)]
pub struct GptNeoLMHeadModel {
    transformer: GptNeoModel,
    lm_head: Linear,
    device: Device,
}

impl GptNeoLMHeadModel {
    pub fn new(config: GptNeoConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: GptNeoConfig, device: Device) -> Result<Self> {
        let transformer = GptNeoModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            transformer,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

#[derive(Debug)]
pub struct GptNeoLMHeadOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
}

impl Model for GptNeoLMHeadModel {
    type Config = GptNeoConfig;
    type Input = TokenizedInput;
    type Output = GptNeoLMHeadOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let transformer_output = self.transformer.forward(input)?;
        let logits = self.lm_head.forward(transformer_output.last_hidden_state.clone())?;

        Ok(GptNeoLMHeadOutput {
            logits,
            hidden_states: Some(transformer_output.last_hidden_state),
        })
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

impl GptNeoLMHeadModel {
    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load word token embeddings
        if let Ok(embed_weights) = loader.load_tensor("transformer.wte.weight") {
            self.transformer.wte.set_weight(embed_weights)?;
        }

        // Load position embeddings
        if let Ok(pos_weights) = loader.load_tensor("transformer.wpe.weight") {
            self.transformer.wpe.set_weight(pos_weights)?;
        }

        // Load transformer layers
        for (i, layer) in self.transformer.layers.iter_mut().enumerate() {
            // Load first layer norm
            if let Ok(ln1_weight) = loader.load_tensor(&format!("transformer.h.{}.ln_1.weight", i))
            {
                layer.ln_1.set_weight(ln1_weight)?;
            }
            if let Ok(ln1_bias) = loader.load_tensor(&format!("transformer.h.{}.ln_1.bias", i)) {
                layer.ln_1.set_bias(ln1_bias)?;
            }

            // Load attention weights
            let attn_prefix = format!("transformer.h.{}.attn.attention", i);

            if let Ok(q_weight) = loader.load_tensor(&format!("{}.q_proj.weight", attn_prefix)) {
                layer.attn.attention.q_proj.set_weight(q_weight)?;
            }
            if let Ok(k_weight) = loader.load_tensor(&format!("{}.k_proj.weight", attn_prefix)) {
                layer.attn.attention.k_proj.set_weight(k_weight)?;
            }
            if let Ok(v_weight) = loader.load_tensor(&format!("{}.v_proj.weight", attn_prefix)) {
                layer.attn.attention.v_proj.set_weight(v_weight)?;
            }
            if let Ok(o_weight) = loader.load_tensor(&format!("{}.out_proj.weight", attn_prefix)) {
                layer.attn.attention.out_proj.set_weight(o_weight)?;
            }

            // Load second layer norm
            if let Ok(ln2_weight) = loader.load_tensor(&format!("transformer.h.{}.ln_2.weight", i))
            {
                layer.ln_2.set_weight(ln2_weight)?;
            }
            if let Ok(ln2_bias) = loader.load_tensor(&format!("transformer.h.{}.ln_2.bias", i)) {
                layer.ln_2.set_bias(ln2_bias)?;
            }

            // Load MLP weights
            let mlp_prefix = format!("transformer.h.{}.mlp", i);

            if let Ok(fc_weight) = loader.load_tensor(&format!("{}.c_fc.weight", mlp_prefix)) {
                layer.mlp.c_fc.set_weight(fc_weight)?;
            }
            if let Ok(proj_weight) = loader.load_tensor(&format!("{}.c_proj.weight", mlp_prefix)) {
                layer.mlp.c_proj.set_weight(proj_weight)?;
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

        // List of essential files for GPT-Neo models
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

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load word token embeddings
        if let Ok(embed_weights) = loader.load_tensor("transformer.wte.weight") {
            self.transformer.wte.set_weight(embed_weights)?;
        }

        // Load position embeddings
        if let Ok(pos_weights) = loader.load_tensor("transformer.wpe.weight") {
            self.transformer.wpe.set_weight(pos_weights)?;
        }

        // Load transformer layers
        for (i, layer) in self.transformer.layers.iter_mut().enumerate() {
            // Load first layer norm
            if let Ok(ln1_weight) = loader.load_tensor(&format!("transformer.h.{}.ln_1.weight", i))
            {
                layer.ln_1.set_weight(ln1_weight)?;
            }
            if let Ok(ln1_bias) = loader.load_tensor(&format!("transformer.h.{}.ln_1.bias", i)) {
                layer.ln_1.set_bias(ln1_bias)?;
            }

            // Load attention weights
            let attn_prefix = format!("transformer.h.{}.attn.attention", i);

            if let Ok(q_weight) = loader.load_tensor(&format!("{}.q_proj.weight", attn_prefix)) {
                layer.attn.attention.q_proj.set_weight(q_weight)?;
            }
            if let Ok(k_weight) = loader.load_tensor(&format!("{}.k_proj.weight", attn_prefix)) {
                layer.attn.attention.k_proj.set_weight(k_weight)?;
            }
            if let Ok(v_weight) = loader.load_tensor(&format!("{}.v_proj.weight", attn_prefix)) {
                layer.attn.attention.v_proj.set_weight(v_weight)?;
            }
            if let Ok(o_weight) = loader.load_tensor(&format!("{}.out_proj.weight", attn_prefix)) {
                layer.attn.attention.out_proj.set_weight(o_weight)?;
            }

            // Load second layer norm
            if let Ok(ln2_weight) = loader.load_tensor(&format!("transformer.h.{}.ln_2.weight", i))
            {
                layer.ln_2.set_weight(ln2_weight)?;
            }
            if let Ok(ln2_bias) = loader.load_tensor(&format!("transformer.h.{}.ln_2.bias", i)) {
                layer.ln_2.set_bias(ln2_bias)?;
            }

            // Load MLP weights
            let mlp_prefix = format!("transformer.h.{}.mlp", i);

            if let Ok(fc_weight) = loader.load_tensor(&format!("{}.c_fc.weight", mlp_prefix)) {
                layer.mlp.c_fc.set_weight(fc_weight)?;
            }
            if let Ok(proj_weight) = loader.load_tensor(&format!("{}.c_proj.weight", mlp_prefix)) {
                layer.mlp.c_proj.set_weight(proj_weight)?;
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

    /// Generate text given a prompt using GPT-Neo
    pub fn generate(
        &self,
        input_ids: Vec<u32>,
        max_length: usize,
        temperature: f32,
        top_k: Option<usize>,
        top_p: Option<f32>,
    ) -> Result<Vec<u32>> {
        let mut generated = input_ids.clone();

        while generated.len() < max_length {
            // Prepare input
            let input = TokenizedInput {
                input_ids: generated.clone(),
                attention_mask: vec![1u8; generated.len()],
                token_type_ids: None,
                special_tokens_mask: None,
                offset_mapping: None,
                overflowing_tokens: None,
            };

            // Forward pass
            let output = self.forward(input)?;

            // Get logits for the last token
            let logits = output.logits;
            let last_logits = match &logits {
                Tensor::F32(arr) => {
                    // Get the last token's logits (shape: [batch, seq_len, vocab_size])
                    let shape = arr.shape();
                    if shape.len() != 3 {
                        return Err(tensor_op_error(
                            "tensor_operation",
                            "Expected 3D tensor for logits".to_string(),
                        ));
                    }
                    let seq_len = shape[1];
                    let vocab_size = shape[2];
                    let slice = arr.slice(s![0, seq_len - 1, ..]);
                    // ArrayD and IxDyn already imported via scirs2_core at top
                    ArrayD::from_shape_vec(IxDyn(&[vocab_size]), slice.iter().cloned().collect())
                        .map_err(|e| {
                            TrustformersError::tensor_op_error(
                                &format!("Failed to reshape tensor: {}", e),
                                "tensor_reshape",
                            )
                        })?
                },
                _ => {
                    return Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type for generation".to_string(),
                    ))
                },
            };

            // Apply temperature
            let scaled_logits = if temperature != 1.0 {
                last_logits.mapv(|x| x / temperature)
            } else {
                last_logits
            };

            // Apply top-k filtering
            let filtered_logits = if let Some(k) = top_k {
                apply_top_k_filtering(scaled_logits, k)?
            } else {
                scaled_logits
            };

            // Apply top-p (nucleus) filtering
            let final_logits = if let Some(p) = top_p {
                apply_top_p_filtering(filtered_logits, p)?
            } else {
                filtered_logits
            };

            // Sample from the distribution
            let next_token = sample_from_logits(final_logits)?;
            generated.push(next_token);

            // Check for EOS token (assuming 50256 is EOS for GPT-Neo, same as GPT-2)
            if next_token == 50256 {
                break;
            }
        }

        Ok(generated)
    }

    /// Generate text using greedy decoding
    pub fn generate_greedy(&self, input_ids: Vec<u32>, max_length: usize) -> Result<Vec<u32>> {
        self.generate(input_ids, max_length, 1.0, Some(1), None)
    }
}

// Helper functions for text generation (similar to GPT-2)
fn apply_top_k_filtering(logits: ArrayD<f32>, k: usize) -> Result<ArrayD<f32>> {
    let mut result = logits.clone();
    let mut indices_and_values: Vec<(usize, f32)> =
        logits.iter().enumerate().map(|(idx, &val)| (idx, val)).collect();

    // Sort by value in descending order
    indices_and_values.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Set all values outside top-k to -inf
    for (idx, _) in indices_and_values.iter().skip(k) {
        result[*idx] = f32::NEG_INFINITY;
    }

    Ok(result)
}

fn apply_top_p_filtering(logits: ArrayD<f32>, p: f32) -> Result<ArrayD<f32>> {
    // Convert to probabilities
    let probs = softmax(logits.clone())?;
    let mut indices_and_probs: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(idx, &prob)| (idx, prob)).collect();

    // Sort by probability in descending order
    indices_and_probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // Find the smallest set of tokens with cumulative probability > p
    let mut cumsum = 0.0;
    let mut cutoff_idx = indices_and_probs.len();
    for (i, (_, prob)) in indices_and_probs.iter().enumerate() {
        cumsum += prob;
        if cumsum > p {
            cutoff_idx = i + 1;
            break;
        }
    }

    // Set all probabilities outside top-p to 0 and convert back to logits
    let mut result = logits.clone();
    for (idx, _) in indices_and_probs.iter().skip(cutoff_idx) {
        result[*idx] = f32::NEG_INFINITY;
    }

    Ok(result)
}

fn sample_from_logits(logits: ArrayD<f32>) -> Result<u32> {
    use scirs2_core::random::*; // SciRS2 Integration Policy (includes WeightedIndex)

    // Convert to probabilities
    let probs = softmax(logits)?;

    // Create weighted distribution
    let weights: Vec<f32> = probs.iter().copied().collect();
    let dist = WeightedIndex::new(weights).map_err(|e| {
        TrustformersError::model_error(format!("Failed to create distribution: {}", e))
    })?;

    // Sample
    let mut rng = thread_rng(); // From scirs2_core::random
    Ok(rng.sample(&dist) as u32)
}

fn softmax(logits: ArrayD<f32>) -> Result<ArrayD<f32>> {
    // Find max for numerical stability
    let max_val = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));

    // Compute exp(x - max)
    let exp_vals = logits.mapv(|x| (x - max_val).exp());

    // Sum of exp values
    let sum: f32 = exp_vals.iter().sum();

    if sum <= 0.0 {
        return Err(TrustformersError::model_error(
            "Invalid softmax computation".to_string(),
        ));
    }

    // Normalize
    Ok(exp_vals / sum)
}
