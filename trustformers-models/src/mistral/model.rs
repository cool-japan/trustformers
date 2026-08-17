use crate::llama::model::{LlamaMLP, RMSNorm}; // Reuse LLaMA components
use crate::mistral::config::MistralConfig;
use crate::moe::{Expert, MoEConfig, SparseMoE};
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
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
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope_theta: f32,
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_theta: config.rope_theta,
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

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: config.num_attention_heads,
            num_kv_heads: config.num_key_value_heads,
            head_dim,
            rope_theta: config.rope_theta,
            sliding_window: config.sliding_window,
            attention_dropout: config.attention_dropout,
        })
    }
}

/// Rotate `data` (row-major, shape `[seq_len, num_heads * head_dim]`) in
/// place using the standard "rotate-half" RoPE convention (Su et al. 2021):
/// each head's `head_dim`-wide channel vector is split into two halves
/// `(x1, x2)` and rotated as `(x1*cos - x2*sin, x1*sin + x2*cos)` using a
/// position-dependent angle `pos * theta^(-2i/head_dim)`. Every head in the
/// row is rotated independently, so `q` (with `num_heads`) and `k` (with
/// `num_kv_heads` under GQA) must be rotated with separate calls.
///
/// `pub(crate)` (rather than private) solely so the regression tests in
/// `mistral::tests` can exercise it directly.
pub(crate) fn apply_rope_rotate_half(
    data: &mut [f32],
    num_heads: usize,
    head_dim: usize,
    theta: f32,
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
                let freq = 1.0 / theta.powf(2.0 * i as f32 / head_dim as f32);
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

impl Layer for MistralAttention {
    type Input = Tensor;
    type Output = Tensor;

    /// Real scaled dot-product attention: RoPE, grouped-query `Q @ K^T`,
    /// causal masking combined with a sliding window (when configured), and
    /// softmax over `V`.
    ///
    /// Accepts `input` shaped `[seq_len, hidden_size]` (the shape produced
    /// by `Embedding::forward`/`MistralModel::forward`); `seq_len` is
    /// derived from the projected `Q` width rather than assumed from a fixed
    /// tensor rank, so this also works if a leading batch=1 axis is present.
    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        let (mut q_data, mut k_data, v_data) = match (&q, &k, &v) {
            (Tensor::F32(qd), Tensor::F32(kd), Tensor::F32(vd)) => (
                qd.as_slice()
                    .ok_or_else(|| tensor_op_error("mistral_attn", "q tensor not contiguous"))?
                    .to_vec(),
                kd.as_slice()
                    .ok_or_else(|| tensor_op_error("mistral_attn", "k tensor not contiguous"))?
                    .to_vec(),
                vd.as_slice()
                    .ok_or_else(|| tensor_op_error("mistral_attn", "v tensor not contiguous"))?
                    .to_vec(),
            ),
            _ => return Err(tensor_op_error("mistral_attn", "q, k, v must be F32")),
        };

        if self.num_heads == 0
            || self.num_kv_heads == 0
            || !self.num_heads.is_multiple_of(self.num_kv_heads)
        {
            return Err(tensor_op_error(
                "mistral_attn",
                "num_heads must be a positive multiple of num_kv_heads",
            ));
        }
        let q_width = self.num_heads * self.head_dim;
        let kv_width = self.num_kv_heads * self.head_dim;
        if q_width == 0 || !q_data.len().is_multiple_of(q_width) {
            return Err(tensor_op_error(
                "mistral_attn",
                "q size inconsistent with num_heads * head_dim",
            ));
        }
        let seq_len = q_data.len() / q_width;
        if k_data.len() != seq_len * kv_width || v_data.len() != seq_len * kv_width {
            return Err(tensor_op_error(
                "mistral_attn",
                "k/v size inconsistent with num_kv_heads * head_dim",
            ));
        }
        if let Some(w) = self.sliding_window {
            if w == 0 {
                return Err(tensor_op_error(
                    "mistral_attn",
                    "sliding_window must be > 0",
                ));
            }
        }

        let position_ids: Vec<usize> = (0..seq_len).collect();
        apply_rope_rotate_half(
            &mut q_data,
            self.num_heads,
            self.head_dim,
            self.rope_theta,
            &position_ids,
        );
        apply_rope_rotate_half(
            &mut k_data,
            self.num_kv_heads,
            self.head_dim,
            self.rope_theta,
            &position_ids,
        );

        let group = self.num_heads / self.num_kv_heads;
        let scale = 1.0 / (self.head_dim as f32).sqrt();
        // A window >= seq_len never excludes any causally-valid key, so this
        // single code path (shared with the tested `tasks::apply_sliding_window_mask`)
        // covers both plain causal attention and real sliding-window attention.
        let effective_window = self.sliding_window.unwrap_or(seq_len.max(1));

        let mut out = vec![0f32; seq_len * q_width];
        for h in 0..self.num_heads {
            let kv_h = h / group;
            let mut scores = vec![0f32; seq_len * seq_len];
            for i in 0..seq_len {
                let q_off = i * q_width + h * self.head_dim;
                for j in 0..seq_len {
                    let k_off = j * kv_width + kv_h * self.head_dim;
                    let dot: f32 =
                        (0..self.head_dim).map(|d| q_data[q_off + d] * k_data[k_off + d]).sum();
                    scores[i * seq_len + j] = dot * scale;
                }
            }
            crate::mistral::tasks::apply_sliding_window_mask(
                &mut scores,
                seq_len,
                effective_window,
            );

            for i in 0..seq_len {
                let row = &scores[i * seq_len..(i + 1) * seq_len];
                let max_val = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let mut weights = vec![0f32; seq_len];
                let mut sum = 0f32;
                for (j, &s) in row.iter().enumerate() {
                    let e = (s - max_val).exp();
                    weights[j] = e;
                    sum += e;
                }
                let inv_sum = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let out_off = i * q_width + h * self.head_dim;
                for (j, &w) in weights.iter().enumerate() {
                    let wn = w * inv_sum;
                    let v_off = j * kv_width + kv_h * self.head_dim;
                    for d in 0..self.head_dim {
                        out[out_off + d] += wn * v_data[v_off + d];
                    }
                }
            }
        }

        let attended = Tensor::from_vec(out, &[seq_len, q_width])?;
        self.o_proj.forward(attended)
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
