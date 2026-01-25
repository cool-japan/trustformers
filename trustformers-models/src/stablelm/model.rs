use crate::stablelm::config::StableLMConfig;
use scirs2_core::ndarray::{Array1, Array2, Axis}; // SciRS2 Integration Policy (Array2 in tests)
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result},
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
    pub fn new(hidden_size: usize, eps: f32) -> Self {
        Self::new_with_device(hidden_size, eps, Device::CPU)
    }

    pub fn new_with_device(hidden_size: usize, eps: f32, device: Device) -> Self {
        let weight = Tensor::ones(&[hidden_size])
            .expect("operation failed")
            .to_device_enum(&device)
            .expect("operation failed");
        Self {
            weight,
            eps,
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn parameter_count(&self) -> usize {
        self.weight.shape().iter().product()
    }
}

impl Layer for RMSNorm {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        match &input {
            Tensor::F32(arr) => {
                let mean_sq = arr.mapv(|x| x * x).mean().unwrap_or(0.0);
                let rms = (mean_sq + self.eps).sqrt();
                let normalized = arr.mapv(|x| x / rms);

                match &self.weight {
                    Tensor::F32(weight_arr) => {
                        let result = &normalized * weight_arr;
                        Ok(Tensor::F32(result))
                    },
                    _ => Err(tensor_op_error(
                        "tensor_operation",
                        "Unsupported tensor type".to_string(),
                    )),
                }
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type".to_string(),
            )),
        }
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
    pub fn new(head_dim: usize, max_seq_len: usize, base: f32, partial_rotary_factor: f32) -> Self {
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
    ) -> Self {
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

        let sin_cached = Tensor::F32(sin_arr.into_dyn())
            .to_device_enum(&device)
            .expect("operation failed");
        let cos_cached = Tensor::F32(cos_arr.into_dyn())
            .to_device_enum(&device)
            .expect("operation failed");

        Self {
            sin_cached,
            cos_cached,
            max_seq_len,
            head_dim,
            base,
            partial_rotary_factor,
            device,
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn forward(&self, q: &Tensor, k: &Tensor, seq_len: usize) -> Result<(Tensor, Tensor)> {
        let rotary_dim = ((self.head_dim as f32) * self.partial_rotary_factor) as usize;

        match (q, k, &self.sin_cached, &self.cos_cached) {
            (
                Tensor::F32(q_arr),
                Tensor::F32(k_arr),
                Tensor::F32(sin_arr),
                Tensor::F32(cos_arr),
            ) => {
                // Apply partial rotary embeddings
                let mut q_rot = q_arr.clone();
                let mut k_rot = k_arr.clone();

                // Only rotate the first rotary_dim dimensions
                if rotary_dim > 0 && seq_len <= self.max_seq_len {
                    // Get shapes before mutable operations to avoid borrow checker issues
                    let q_shape = q_rot.shape().to_vec();
                    let _k_shape = k_rot.shape().to_vec();

                    // Apply RoPE to query and key tensors
                    for seq_idx in 0..seq_len {
                        for dim_idx in 0..(rotary_dim / 2) {
                            let cos_val = cos_arr[[seq_idx, dim_idx]];
                            let sin_val = sin_arr[[seq_idx, dim_idx]];

                            // Apply rotation: [x1, x2] -> [x1*cos - x2*sin, x1*sin + x2*cos]
                            // This is a simplified implementation for the core rotation logic
                            for batch in 0..q_shape[0] {
                                for head in 0..q_shape[1] {
                                    if seq_idx < q_shape[2] && dim_idx < rotary_dim / 2 {
                                        let x1_idx = [batch, head, seq_idx, dim_idx * 2];
                                        let x2_idx = [batch, head, seq_idx, dim_idx * 2 + 1];

                                        if x2_idx[3] < q_shape[3] {
                                            let q_x1 = q_rot[x1_idx];
                                            let q_x2 = q_rot[x2_idx];
                                            let k_x1 = k_rot[x1_idx];
                                            let k_x2 = k_rot[x2_idx];

                                            // Query rotation
                                            q_rot[x1_idx] = q_x1 * cos_val - q_x2 * sin_val;
                                            q_rot[x2_idx] = q_x1 * sin_val + q_x2 * cos_val;

                                            // Key rotation
                                            k_rot[x1_idx] = k_x1 * cos_val - k_x2 * sin_val;
                                            k_rot[x2_idx] = k_x1 * sin_val + k_x2 * cos_val;
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Ok((Tensor::F32(q_rot), Tensor::F32(k_rot)))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Unsupported tensor type".to_string(),
            )),
        }
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
    pub fn new(config: &StableLMConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Self {
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
        );

        Self {
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
        }
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    fn repeat_kv(&self, hidden_states: &Tensor, n_rep: usize) -> Result<Tensor> {
        if n_rep == 1 {
            return Ok(hidden_states.clone());
        }

        match hidden_states {
            Tensor::F32(arr) => {
                // Repeat key/value heads for grouped-query attention
                let _shape = arr.shape();
                let mut repeated = arr.clone();

                // Simplified - actual implementation would properly repeat along head dimension
                for _ in 1..n_rep {
                    repeated = repeated.clone(); // Placeholder
                }

                Ok(Tensor::F32(repeated))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
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

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let _batch_size = 1; // Simplified
        let seq_len = 1; // Simplified

        // Query, Key, Value projections
        let q = self.q_proj.forward(input.clone())?;
        let k = self.k_proj.forward(input.clone())?;
        let v = self.v_proj.forward(input)?;

        // Apply rotary embeddings
        let (q_rot, k_rot) = self.rotary_emb.forward(&q, &k, seq_len)?;

        // Repeat KV heads if using grouped-query attention
        let n_rep = self.num_heads / self.num_kv_heads;
        let k_repeated = self.repeat_kv(&k_rot, n_rep)?;
        let v_repeated = self.repeat_kv(&v, n_rep)?;

        // Compute attention scores
        // Simplified - actual implementation would compute proper attention
        let attn_output = match (&q_rot, &k_repeated, &v_repeated) {
            (Tensor::F32(q_arr), Tensor::F32(_k_arr), Tensor::F32(_v_arr)) => {
                // Placeholder for attention computation
                Tensor::F32(q_arr.clone())
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor type".to_string(),
                ))
            },
        };

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
    pub fn new(config: &StableLMConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &StableLMConfig, device: Device) -> Self {
        Self {
            config: config.clone(),
            self_attn: StableLMAttention::new_with_device(config, device),
            mlp: StableLMMLP::new_with_device(config, device),
            input_layernorm: RMSNorm::new_with_device(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            ),
            post_attention_layernorm: RMSNorm::new_with_device(
                config.hidden_size,
                config.rms_norm_eps,
                device,
            ),
            device,
        }
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
            layers.push(StableLMDecoderLayer::new_with_device(&config, device));
        }

        let norm = RMSNorm::new_with_device(config.hidden_size, config.rms_norm_eps, device);

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
            self.model.embeddings.word_embeddings.set_weight(embed_weights)?;
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

            // Layer norm weights would be loaded here if RMSNorm supported set_weight
            // For now, skipping layer norm weight loading
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

            // Try using curl first
            let curl_result = Command::new("curl")
                .args([
                    "-L", // Follow redirects
                    "-f", // Fail on HTTP errors
                    "-o",
                    file_path.to_str().expect("operation failed"),
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
                .args([
                    "-O",
                    file_path.to_str().expect("operation failed"),
                    &file_url,
                ])
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

#[cfg(test)]
mod tests {
    use super::*;
    // Array2 already imported via scirs2_core at top

    #[test]
    fn test_rms_norm() {
        let norm = RMSNorm::new(768, 1e-5);
        let input = Tensor::F32(Array2::ones((2, 768)).into_dyn());
        let output = norm.forward(input);
        assert!(output.is_ok());
    }

    #[test]
    fn test_rotary_embedding() {
        let rope = RotaryEmbedding::new(64, 512, 10000.0, 0.25);
        assert_eq!(rope.head_dim, 64);
        assert_eq!(rope.max_seq_len, 512);
        assert_eq!(rope.partial_rotary_factor, 0.25);
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B model creation, run with --ignored
    fn test_stablelm_model_creation() {
        let config = StableLMConfig::stablelm_3b();
        let model = StableLMModel::new(config.clone()).expect("operation failed");

        assert_eq!(model.layers.len(), config.num_hidden_layers);
        assert_eq!(model.config.hidden_size, 2560);
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B CausalLM, run with --ignored
    fn test_stablelm_causal_lm() {
        let config = StableLMConfig::stablelm_3b();
        let _model = StableLMForCausalLM::new(config.clone()).expect("operation failed");

        // StableLM for CausalLM created successfully - LM head dimensions are internal
    }

    #[test]
    fn test_grouped_query_attention() {
        let mut config = StableLMConfig::stablelm_2_1_6b();
        config.num_key_value_heads = Some(4);

        let attn = StableLMAttention::new(&config);
        assert_eq!(attn.num_heads, 32);
        assert_eq!(attn.num_kv_heads, 4);

        // Grouped query attention created successfully - projection dimensions are internal
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B device support, run with --ignored
    fn test_device_support() {
        let config = StableLMConfig::stablelm_3b();

        // Test CPU device (default)
        let model_cpu = StableLMModel::new(config.clone()).expect("operation failed");
        assert_eq!(*model_cpu.device(), Device::CPU);

        // Test explicit CPU device
        let model_cpu_explicit =
            StableLMModel::new_with_device(config.clone(), Device::CPU).expect("operation failed");
        assert_eq!(*model_cpu_explicit.device(), Device::CPU);

        // Test that all components have the correct device
        assert_eq!(*model_cpu.embeddings.device(), Device::CPU);
        assert_eq!(*model_cpu.norm.device(), Device::CPU);
        for layer in &model_cpu.layers {
            assert_eq!(*layer.device(), Device::CPU);
            assert_eq!(*layer.self_attn.device(), Device::CPU);
            assert_eq!(*layer.mlp.device(), Device::CPU);
        }
    }

    #[test]
    #[ignore] // Heavy test - StableLM 3B CausalLM device support (SIGKILL risk), run with --ignored
    fn test_causal_lm_device_support() {
        let config = StableLMConfig::stablelm_3b();

        // Test CPU device
        let model = StableLMForCausalLM::new(config.clone()).expect("operation failed");
        assert_eq!(*model.device(), Device::CPU);
        assert_eq!(*model.model.device(), Device::CPU);

        // Test explicit device
        let model_explicit =
            StableLMForCausalLM::new_with_device(config, Device::CPU).expect("operation failed");
        assert_eq!(*model_explicit.device(), Device::CPU);
    }
}
