use crate::llama::model::{LlamaMLP, RMSNorm, RotaryEmbedding}; // Reuse LLaMA components
use crate::mistral::config::MistralConfig;
use crate::moe::{Expert, MoEConfig, SparseMoE};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{Result, TrustformersError},
    layers::{Embedding, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Mistral attention layer with sliding window attention and grouped-query attention
pub struct MistralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    #[allow(dead_code)]
    rotary_emb: RotaryEmbedding,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    sliding_window: Option<usize>,
    #[allow(dead_code)]
    attention_dropout: f32,
}

impl MistralAttention {
    pub fn new(config: &MistralConfig) -> Result<Self> {
        let head_dim = config.head_dim();

        let q_proj = Linear::new(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            false,
        );
        let k_proj = Linear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
        );
        let v_proj = Linear::new(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
        );
        let o_proj = Linear::new(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            false,
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
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            sliding_window: config.sliding_window,
            attention_dropout: config.attention_dropout,
        })
    }

    pub fn new_with_device(config: &MistralConfig, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();

        let q_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_attention_heads * head_dim,
            false,
            device,
        );
        let k_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let v_proj = Linear::new_with_device(
            config.hidden_size,
            config.num_key_value_heads * head_dim,
            false,
            device,
        );
        let o_proj = Linear::new_with_device(
            config.num_attention_heads * head_dim,
            config.hidden_size,
            false,
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
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            sliding_window: config.sliding_window,
            attention_dropout: config.attention_dropout,
        })
    }

    /// Apply sliding window attention mask
    fn apply_sliding_window_mask(
        &self,
        attention_scores: &Tensor,
        seq_len: usize,
    ) -> Result<Tensor> {
        if let Some(window_size) = self.sliding_window {
            // Create sliding window mask
            let mut mask_data = vec![0.0f32; seq_len * seq_len];
            for i in 0..seq_len {
                for j in 0..seq_len {
                    let distance = j.saturating_sub(i);
                    if distance > window_size {
                        mask_data[i * seq_len + j] = f32::NEG_INFINITY;
                    }
                }
            }
            // Reshape to [1, 1, seq_len, seq_len] for broadcasting across batch and num_heads
            let mask = Tensor::from_vec(mask_data, &[seq_len, seq_len])?
                .reshape(&[1, 1, seq_len, seq_len])?;
            attention_scores.add(&mask)
        } else {
            Ok(attention_scores.clone())
        }
    }

    /// Apply rotary position embedding (simplified implementation)
    fn apply_rotary_embedding(&self, tensor: &Tensor, _seq_len: usize) -> Result<Tensor> {
        // For now, return the tensor unchanged
        // A full implementation would apply rotary embedding based on position
        Ok(tensor.clone())
    }

    /// Create causal mask for autoregressive attention
    fn create_causal_mask(&self, seq_len: usize) -> Result<Tensor> {
        let mut mask_data = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in (i + 1)..seq_len {
                mask_data[i * seq_len + j] = f32::NEG_INFINITY;
            }
        }
        // Reshape to [1, 1, seq_len, seq_len] for broadcasting across batch and num_heads
        Tensor::from_vec(mask_data, &[seq_len, seq_len])?.reshape(&[1, 1, seq_len, seq_len])
    }
}

impl Layer for MistralAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];
        let seq_len = input.shape()[1];
        eprintln!("[Mistral] Input shape: {:?}", input.shape());

        // Project to Q, K, V
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;
        eprintln!(
            "[Mistral] After projection - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Reshape for multi-head attention with grouped-query attention
        let head_dim = self.head_dim;
        let q = q.reshape(&[batch_size, seq_len, self.num_heads, head_dim])?;
        let k = k.reshape(&[batch_size, seq_len, self.num_kv_heads, head_dim])?;
        let v = v.reshape(&[batch_size, seq_len, self.num_kv_heads, head_dim])?;
        eprintln!(
            "[Mistral] After reshape - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Transpose to [batch, num_heads, seq_len, head_dim]
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        eprintln!(
            "[Mistral] After transpose - Q: {:?}, K: {:?}, V: {:?}",
            q.shape(),
            k.shape(),
            v.shape()
        );

        // Apply rotary embedding (simplified implementation)
        let q = self.apply_rotary_embedding(&q, seq_len)?;
        let k = self.apply_rotary_embedding(&k, seq_len)?;

        // Repeat k and v heads for grouped-query attention
        let (k, v) = if self.num_kv_heads < self.num_heads {
            eprintln!(
                "[Mistral GQA] Expanding {} KV heads to {} query heads (repeats={})",
                self.num_kv_heads,
                self.num_heads,
                self.num_heads / self.num_kv_heads
            );
            let repeats = self.num_heads / self.num_kv_heads;
            let mut k_heads = Vec::new();
            let mut v_heads = Vec::new();

            for head_idx in 0..self.num_kv_heads {
                let k_head = k.slice_multi(&[
                    (0, batch_size),
                    (head_idx, head_idx + 1),
                    (0, seq_len),
                    (0, head_dim),
                ])?;
                let v_head = v.slice_multi(&[
                    (0, batch_size),
                    (head_idx, head_idx + 1),
                    (0, seq_len),
                    (0, head_dim),
                ])?;
                eprintln!(
                    "[Mistral GQA] Head {} - k_head: {:?}, v_head: {:?}",
                    head_idx,
                    k_head.shape(),
                    v_head.shape()
                );

                for _ in 0..repeats {
                    k_heads.push(k_head.clone());
                    v_heads.push(v_head.clone());
                }
            }

            let k_repeated = Tensor::concat(&k_heads, 1)?;
            let v_repeated = Tensor::concat(&v_heads, 1)?;
            eprintln!(
                "[Mistral GQA] After concat - K: {:?}, V: {:?}",
                k_repeated.shape(),
                v_repeated.shape()
            );
            (k_repeated, v_repeated)
        } else {
            eprintln!("[Mistral] No GQA expansion needed (num_kv_heads == num_heads)");
            (k, v)
        };

        // Compute attention scores
        let k_transposed = k.transpose(2, 3)?;
        eprintln!("[Mistral] K transposed shape: {:?}", k_transposed.shape());
        let scores = q.matmul(&k_transposed)?;
        eprintln!("[Mistral] Attention scores shape: {:?}", scores.shape());
        let scale = (head_dim as f32).sqrt();
        let scaled_scores = scores.div_scalar(scale)?;

        // Apply sliding window attention mask
        let masked_scores = self.apply_sliding_window_mask(&scaled_scores, seq_len)?;
        eprintln!(
            "[Mistral] After sliding window mask: {:?}",
            masked_scores.shape()
        );

        // Apply causal mask
        let causal_mask = self.create_causal_mask(seq_len)?;
        eprintln!("[Mistral] Causal mask shape: {:?}", causal_mask.shape());
        let final_scores = masked_scores.add(&causal_mask)?;
        eprintln!("[Mistral] Final scores shape: {:?}", final_scores.shape());

        // Apply softmax
        let attention_weights = final_scores.softmax(-1)?;

        // Apply attention to values
        let attention_output = attention_weights.matmul(&v)?;

        // Transpose back and reshape
        let attention_output = attention_output.transpose(1, 2)?;
        let attention_output =
            attention_output.reshape(&[batch_size, seq_len, self.num_heads * head_dim])?;

        // Apply output projection
        self.o_proj.forward(attention_output)
    }
}

/// Mistral decoder layer
pub struct MistralDecoderLayer {
    self_attn: MistralAttention,
    mlp: LlamaMLP, // Reuse LLaMA MLP
    input_layernorm: RMSNorm,
    post_attention_layernorm: RMSNorm,
}

impl MistralDecoderLayer {
    pub fn new(config: &MistralConfig) -> Result<Self> {
        let self_attn = MistralAttention::new(config)?;

        // Convert MistralConfig to LlamaConfig-like structure for MLP
        let llama_config = crate::llama::config::LlamaConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            mlp_bias: false, // Mistral doesn't use bias in MLP
            ..Default::default()
        };
        let mlp = LlamaMLP::new(&llama_config)?;

        let input_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;
        let post_attention_layernorm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    pub fn new_with_device(config: &MistralConfig, device: Device) -> Result<Self> {
        let self_attn = MistralAttention::new_with_device(config, device)?;

        // Convert MistralConfig to LlamaConfig-like structure for MLP
        let llama_config = crate::llama::config::LlamaConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            mlp_bias: false, // Mistral doesn't use bias in MLP
            ..Default::default()
        };
        let mlp = LlamaMLP::new_with_device(&llama_config, device)?;

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

impl Layer for MistralDecoderLayer {
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

/// Mistral model
pub struct MistralModel {
    config: MistralConfig,
    embed_tokens: Embedding,
    layers: Vec<MistralDecoderLayer>,
    norm: RMSNorm,
}

impl MistralModel {
    pub fn new(config: MistralConfig) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(MistralDecoderLayer::new(&config)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }

    pub fn new_with_device(config: MistralConfig, device: Device) -> Result<Self> {
        config.validate()?;

        let embed_tokens = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(MistralDecoderLayer::new_with_device(&config, device)?);
        }

        let norm = RMSNorm::new(config.hidden_size, config.rms_norm_eps)?;

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
        })
    }
}

impl Model for MistralModel {
    type Config = MistralConfig;
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
        // Legacy interface - use enhanced weight loading methods for production
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
        let config = &self.config;
        let hidden_size = config.hidden_size;
        let intermediate_size = config.intermediate_size;
        let vocab_size = config.vocab_size;
        let num_layers = config.num_hidden_layers;
        let num_heads = config.num_attention_heads;
        let num_kv_heads = config.num_key_value_heads;
        let head_dim = config.head_dim();

        // Embedding: vocab_size * hidden_size
        let embedding_params = vocab_size * hidden_size;

        // Per layer parameters
        let per_layer_params = {
            // Attention: q_proj, k_proj, v_proj, o_proj
            let q_proj = hidden_size * (num_heads * head_dim);
            let k_proj = hidden_size * (num_kv_heads * head_dim);
            let v_proj = hidden_size * (num_kv_heads * head_dim);
            let o_proj = (num_heads * head_dim) * hidden_size;
            let attention_params = q_proj + k_proj + v_proj + o_proj;

            // MLP: gate_proj, up_proj, down_proj
            let gate_proj = hidden_size * intermediate_size;
            let up_proj = hidden_size * intermediate_size;
            let down_proj = intermediate_size * hidden_size;
            let mlp_params = gate_proj + up_proj + down_proj;

            // LayerNorms: input_layernorm, post_attention_layernorm (just hidden_size each)
            let layernorm_params = hidden_size * 2;

            attention_params + mlp_params + layernorm_params
        };

        // Final layer norm
        let final_norm_params = hidden_size;

        // Total
        embedding_params + (per_layer_params * num_layers) + final_norm_params
    }
}

/// Mistral for causal language modeling (with LM head)
pub struct MistralForCausalLM {
    model: MistralModel,
    lm_head: Linear,
}

impl MistralForCausalLM {
    pub fn new(config: MistralConfig) -> Result<Self> {
        let model = MistralModel::new(config.clone())?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self { model, lm_head })
    }

    pub fn new_with_device(config: MistralConfig, device: Device) -> Result<Self> {
        let model = MistralModel::new_with_device(config.clone(), device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self { model, lm_head })
    }
}

impl Model for MistralForCausalLM {
    type Config = MistralConfig;
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
        let model_params = self.model.num_parameters();

        // LM head: hidden_size * vocab_size (no bias)
        let config = self.model.get_config();
        let lm_head_params = config.hidden_size * config.vocab_size;

        model_params + lm_head_params
    }
}

impl MistralForCausalLM {
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

            // Load layer norm weights
            if let Ok(ln1_weight) =
                loader.load_tensor(&format!("model.layers.{}.input_layernorm.weight", i))
            {
                layer.input_layernorm.set_weight(ln1_weight)?;
            }
            if let Ok(ln2_weight) = loader.load_tensor(&format!(
                "model.layers.{}.post_attention_layernorm.weight",
                i
            )) {
                layer.post_attention_layernorm.set_weight(ln2_weight)?;
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
            TrustformersError::io_error(format!("Failed to create model directory: {}", e))
        })?;

        // List of essential files for Mistral models
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

        let _loader = auto_create_loader(&model_path, Some(config))?;

        // Use the same weight loading logic as load_from_path
        self.load_from_path(model_path)
    }
}

/// Mixtral Expert implementation using the shared MoE infrastructure
pub struct MixtralExpert {
    id: usize,
    mlp: LlamaMLP,
}

impl MixtralExpert {
    pub fn new(id: usize, config: &MistralConfig) -> Result<Self> {
        let llama_config = crate::llama::config::LlamaConfig {
            hidden_size: config.hidden_size,
            intermediate_size: config.intermediate_size,
            mlp_bias: false,
            ..Default::default()
        };
        let mlp = LlamaMLP::new(&llama_config)?;

        Ok(Self { id, mlp })
    }
}

impl Layer for MixtralExpert {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.mlp.forward(input)
    }
}

impl Expert for MixtralExpert {
    fn expert_id(&self) -> usize {
        self.id
    }
}

/// Mixtral Sparse Mixture of Experts layer using the shared infrastructure
pub type MixtralSparseMoE = SparseMoE<MixtralExpert>;

impl MixtralSparseMoE {
    /// Create a new Mixtral MoE layer for Mixtral 8x7B configuration
    pub fn new_mixtral_8x7b(config: &MistralConfig) -> Result<Self> {
        let num_experts = 8;
        let num_experts_per_token = 2;

        // Create experts
        let mut experts = Vec::new();
        for i in 0..num_experts {
            experts.push(MixtralExpert::new(i, config)?);
        }

        // Create MoE configuration
        let moe_config = MoEConfig {
            hidden_size: config.hidden_size,
            num_experts,
            num_experts_per_token,
            load_balancing_loss_coeff: 0.01,
            router_z_loss_coeff: 0.001,
            use_auxiliary_loss: true,
            jitter_noise: 1e-2,
            ..Default::default()
        };

        SparseMoE::new(experts, moe_config)
    }

    /// Create a new Mixtral MoE layer with custom configuration
    pub fn new_custom(
        config: &MistralConfig,
        num_experts: usize,
        num_experts_per_token: usize,
    ) -> Result<Self> {
        let mut experts = Vec::new();
        for i in 0..num_experts {
            experts.push(MixtralExpert::new(i, config)?);
        }

        let moe_config = MoEConfig {
            hidden_size: config.hidden_size,
            num_experts,
            num_experts_per_token,
            ..Default::default()
        };

        SparseMoE::new(experts, moe_config)
    }
}
