use crate::common::ActivationType;
use crate::falcon::config::FalconConfig;
use scirs2_core::ndarray::{s, ArrayD, IxDyn}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    device::Device,
    errors::{tensor_op_error, Result, TrustformersError},
    layers::{Embedding, LayerNorm, Linear},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// Geometric ALiBi slopes for a **power-of-two** head count.
///
/// `start = 2^(-8/n)` and `slopes[i] = start^(i + 1)`, i.e. the geometric
/// sequence from Press et al. (2022) in head order.
fn alibi_slopes_power_of_two(num_heads: usize) -> Vec<f64> {
    let start = 2.0_f64.powf(-8.0 / num_heads as f64);
    let mut slopes = Vec::with_capacity(num_heads);
    let mut value = start;
    for _ in 0..num_heads {
        slopes.push(value);
        value *= start;
    }
    slopes
}

/// Reference ALiBi slopes for `num_heads` heads.
///
/// This is the construction from the ALiBi reference implementation (Press
/// et al., 2022), which HuggingFace's Falcon and BLOOM both reproduce:
///
/// * a power-of-two head count uses the geometric sequence
///   `2^(-8/n), 2^(-16/n), …, 2^(-8)`, **in head order**;
/// * otherwise the slopes for the largest power of two below `num_heads` are
///   used first, then extended with every other slope of the next power of two
///   (`get_slopes(2 * closest)[0::2]`).
///
/// Head order matters: the slope assigned to head `h` has to be the same one the
/// pretrained checkpoint assumed for that head, so a permutation of the correct
/// set is still wrong.
pub fn alibi_slopes(num_heads: usize) -> Vec<f32> {
    fn slopes_f64(num_heads: usize) -> Vec<f64> {
        if num_heads == 0 {
            return Vec::new();
        }
        if num_heads.is_power_of_two() {
            return alibi_slopes_power_of_two(num_heads);
        }
        // 2^floor(log2(num_heads)); `num_heads >= 1` so `ilog2` is defined.
        let closest = 1usize << num_heads.ilog2();
        let mut slopes = alibi_slopes_power_of_two(closest);
        let extra = alibi_slopes_power_of_two(2 * closest);
        slopes.extend(extra.iter().step_by(2).take(num_heads - closest));
        slopes
    }

    slopes_f64(num_heads).into_iter().map(|value| value as f32).collect()
}

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
        if num_heads == 0 {
            return Err(tensor_op_error(
                "ALiBi::new_with_device",
                "num_heads must be at least 1".to_string(),
            ));
        }
        let slopes_tensor = Tensor::new(alibi_slopes(num_heads))?;

        Ok(Self {
            slopes: slopes_tensor,
            num_heads,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Per-head ALiBi slopes (one entry per attention head).
    pub fn slopes(&self) -> &Tensor {
        &self.slopes
    }

    /// Number of attention heads this bias was built for.
    pub fn num_heads(&self) -> usize {
        self.num_heads
    }

    /// Build the ALiBi bias tensor of shape `[1, num_heads, seq_len, seq_len]`.
    ///
    /// Following Press et al. (2022), the bias for head `h` at query position
    /// `i` and key position `j` is `-slope_h * (i - j)` — a linear penalty that
    /// grows with the query/key distance. Positions in the future (`j > i`) are
    /// left at `0.0`: causal masking is a separate, additive concern handled by
    /// `FalconAttention::create_causal_mask`, so this function contributes the
    /// positional bias and nothing else.
    ///
    /// The leading singleton axis lets the result broadcast over the batch when
    /// added to scores shaped `[batch, num_heads, seq_len, seq_len]`.
    pub fn build_bias(&self, seq_len: usize) -> Result<Tensor> {
        // Hoist the slope lookup out of the element loop: one read, not one per
        // element.
        let slopes = self.slopes.data()?;
        if slopes.len() != self.num_heads {
            return Err(tensor_op_error(
                "ALiBi::build_bias",
                format!(
                    "slope count {} does not match num_heads {}",
                    slopes.len(),
                    self.num_heads
                ),
            ));
        }

        let mut bias_data = Vec::with_capacity(self.num_heads * seq_len * seq_len);
        for &slope in slopes.iter() {
            for i in 0..seq_len {
                for j in 0..seq_len {
                    if j > i {
                        // Masked by the causal mask; contribute no positional bias.
                        bias_data.push(0.0);
                    } else {
                        let distance = (i - j) as f32;
                        bias_data.push(-distance * slope);
                    }
                }
            }
        }

        Tensor::from_vec(bias_data, &[1, self.num_heads, seq_len, seq_len])
    }

    /// Add the ALiBi bias to *pre-softmax* attention scores.
    ///
    /// `attention_scores` must have shape `[batch, num_heads, seq_len, seq_len]`
    /// — i.e. raw `Q Kᵀ / sqrt(d)` scores, before masking and before softmax.
    /// Applying ALiBi anywhere else (in particular to the post-softmax attention
    /// output) is not the ALiBi mechanism and is rejected here.
    pub fn apply_bias(&self, attention_scores: &Tensor, seq_len: usize) -> Result<Tensor> {
        let shape = attention_scores.shape();
        if shape.len() != 4
            || shape[1] != self.num_heads
            || shape[2] != seq_len
            || shape[3] != seq_len
        {
            return Err(tensor_op_error(
                "ALiBi::apply_bias",
                format!(
                    "expected pre-softmax scores of shape [batch, {}, {seq_len}, {seq_len}], got {:?}",
                    self.num_heads, shape
                ),
            ));
        }

        let bias_tensor = self.build_bias(seq_len)?;
        attention_scores.add(&bias_tensor)
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

        // Apply ALiBi positional bias to the PRE-softmax scores. ALiBi is a bias
        // on the attention logits (Press et al., 2022); adding it after softmax
        // would not be ALiBi at all.
        let scaled_scores = if let Some(alibi) = &self.alibi {
            alibi.apply_bias(&scaled_scores, seq_len)?
        } else {
            scaled_scores
        };

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

        // NOTE: ALiBi is *not* applied here. The bias belongs on the pre-softmax
        // logits (see above); adding it to the post-softmax context vectors would
        // corrupt the output instead of biasing the attention distribution.

        // Final output projection
        let output = self.dense.forward(attention_output)?;
        Ok(output)
    }
}

/// Falcon MLP layer
pub struct FalconMLP {
    dense_h_to_4h: Linear,
    dense_4h_to_h: Linear,
    activation: ActivationType,
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
            activation: ActivationType::from_config_str_or(
                &config.hidden_act,
                ActivationType::Identity,
            ),
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
        let activated = self.activation.apply(&hidden)?;

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

        tracing::info!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name,
            model_path
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
            let file_path_str = file_path.to_str().ok_or_else(|| {
                TrustformersError::io_error(format!("Non-UTF-8 file path: {}", file_path.display()))
            })?;

            tracing::info!("Attempting to download {}", file_url);

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
                    tracing::info!("Successfully downloaded {}", file_name);
                    continue;
                },
                Ok(output) => {
                    tracing::warn!(
                        "Failed to download {} with curl: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    tracing::info!("curl not available: {}", e);
                },
            }

            // Try using wget as fallback
            let wget_result = Command::new("wget").args(["-O", file_path_str, &file_url]).output();

            match wget_result {
                Ok(output) if output.status.success() => {
                    tracing::info!("Successfully downloaded {} with wget", file_name);
                    continue;
                },
                Ok(output) => {
                    tracing::warn!(
                        "Failed to download {} with wget: {}",
                        file_name,
                        String::from_utf8_lossy(&output.stderr)
                    );
                },
                Err(e) => {
                    tracing::info!("wget not available: {}", e);
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

        tracing::info!(
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

    // ---- Tiny config helper for cheap in-test model instantiation ----
    fn tiny_falcon_config() -> FalconConfig {
        FalconConfig {
            vocab_size: 64,
            hidden_size: 64,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            num_kv_heads: Some(1), // multi-query: 1 KV head
            max_position_embeddings: 32,
            alibi: false, // avoid extra bias allocation
            parallel_attn: true,
            ..FalconConfig::default()
        }
    }

    #[test]
    #[ignore] // Very heavy test - Falcon 7B model (SIGKILL risk), run with --ignored
    fn test_falcon_model_creation() {
        let config = FalconConfig::falcon_7b();
        let model = FalconModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    #[ignore] // Very heavy test - Falcon 7B CausalLM (SIGKILL risk), run with --ignored
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

        let alibi = alibi.expect("operation failed");
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

    // ---- ALiBi slope properties ----

    #[test]
    fn test_alibi_even_heads() {
        // Even number of heads must not panic
        let alibi = ALiBi::new(8).expect("ALiBi with 8 heads");
        assert_eq!(alibi.num_heads, 8);
    }

    #[test]
    fn test_alibi_odd_heads() {
        // Odd number of heads (e.g. 71 for Falcon-7B) must succeed
        let alibi = ALiBi::new(7).expect("ALiBi with 7 heads");
        assert_eq!(alibi.num_heads, 7);
    }

    #[test]
    fn test_alibi_device_cpu() {
        let alibi = ALiBi::new(4).expect("ALiBi with 4 heads");
        assert_eq!(alibi.device(), Device::CPU);
    }

    // ---- Multi-query attention (num_kv_heads = 1) ----

    #[test]
    fn test_falcon_7b_num_kv_heads_is_one() {
        let config = FalconConfig::falcon_7b();
        assert_eq!(
            config.num_kv_heads(),
            1,
            "Falcon-7B must use 1 KV head (multi-query)"
        );
    }

    #[test]
    fn test_falcon_attention_tiny_creation() {
        let config = tiny_falcon_config();
        let attn = FalconAttention::new(&config);
        assert!(
            attn.is_ok(),
            "FalconAttention construction with tiny config failed"
        );
    }

    #[test]
    fn test_falcon_attention_parameter_count_positive() {
        let config = tiny_falcon_config();
        let attn = FalconAttention::new(&config).expect("FalconAttention construction");
        assert!(attn.parameter_count() > 0);
    }

    // ---- Parallel attention flag ----

    #[test]
    fn test_falcon_decoder_layer_parallel_attn_flag() {
        let config = tiny_falcon_config();
        let layer = FalconDecoderLayer::new(&config).expect("FalconDecoderLayer construction");
        assert!(layer.parallel_attn, "Tiny config sets parallel_attn=true");
    }

    #[test]
    fn test_falcon_decoder_layer_sequential_attn() {
        let mut config = tiny_falcon_config();
        config.parallel_attn = false;
        let layer = FalconDecoderLayer::new(&config).expect("FalconDecoderLayer construction");
        assert!(!layer.parallel_attn);
    }

    // ---- new_decoder_architecture flag ----

    #[test]
    fn test_falcon_180b_new_decoder_architecture() {
        let config = FalconConfig::falcon_180b();
        assert!(config.new_decoder_architecture);
    }

    #[test]
    fn test_falcon_7b_old_decoder_architecture() {
        let config = FalconConfig::falcon_7b();
        assert!(!config.new_decoder_architecture);
    }

    // ---- MLP ----

    #[test]
    fn test_falcon_mlp_tiny_creation() {
        let config = tiny_falcon_config();
        let mlp = FalconMLP::new(&config).expect("FalconMLP tiny creation");
        assert!(mlp.parameter_count() > 0);
    }

    #[test]
    fn test_falcon_mlp_device_cpu() {
        let config = tiny_falcon_config();
        let mlp = FalconMLP::new(&config).expect("FalconMLP tiny creation");
        assert_eq!(mlp.device(), Device::CPU);
    }

    // ---- FalconModel creation and parameter count ----

    #[test]
    fn test_falcon_model_tiny_creation() {
        let config = tiny_falcon_config();
        let model = FalconModel::new(config);
        assert!(model.is_ok(), "FalconModel with tiny config must succeed");
    }

    #[test]
    fn test_falcon_model_num_parameters_positive() {
        let config = tiny_falcon_config();
        let model = FalconModel::new(config).expect("FalconModel tiny");
        assert!(model.num_parameters() > 0);
    }

    #[test]
    fn test_falcon_causal_lm_tiny_creation() {
        let config = tiny_falcon_config();
        let model = FalconForCausalLM::new(config);
        assert!(
            model.is_ok(),
            "FalconForCausalLM with tiny config must succeed"
        );
    }

    #[test]
    fn test_falcon_causal_lm_parameter_count_exceeds_base() {
        let config = tiny_falcon_config();
        let base = FalconModel::new(config.clone()).expect("FalconModel");
        let lm_head_model = FalconForCausalLM::new(config).expect("FalconForCausalLM");
        // CausalLM adds an lm_head on top → more parameters
        assert!(lm_head_model.num_parameters() > base.num_parameters());
    }

    // ---- ALiBi slopes are well-formed ----

    #[test]
    fn test_alibi_slopes_positive() {
        // Slopes must be positive (they decay attention over distance)
        let alibi = ALiBi::new(4).expect("ALiBi with 4 heads");
        let data = alibi.slopes.data().expect("slope data");
        for (i, &s) in data.iter().enumerate() {
            assert!(s > 0.0, "Slope[{}] = {} must be positive", i, s);
        }
    }

    // ---- ALiBi slopes: reference construction ----

    /// Power-of-two head counts must reproduce the geometric sequence *in head
    /// order*. The old code emitted the odd powers first and the even powers
    /// afterwards — the right set of slopes attached to the wrong heads.
    #[test]
    fn test_alibi_slopes_power_of_two_are_in_head_order() {
        let slopes = alibi_slopes(8);
        assert_eq!(slopes.len(), 8);
        let start = 2.0f32.powf(-1.0); // 2^(-8/8)
        for (i, &slope) in slopes.iter().enumerate() {
            let expected = start.powi(i as i32 + 1);
            assert!(
                (slope - expected).abs() < 1e-7,
                "slope[{i}] = {slope}, reference {expected}"
            );
        }
        // Strictly decreasing, which the interleaved order was not.
        for window in slopes.windows(2) {
            assert!(
                window[0] > window[1],
                "slopes must decrease with head index: {} !> {}",
                window[0],
                window[1]
            );
        }
    }

    /// Non-power-of-two head counts follow the reference recursion: the first
    /// `2^floor(log2 n)` slopes are exactly the power-of-two construction for
    /// that smaller count, then every other slope of the next power of two.
    #[test]
    fn test_alibi_slopes_non_power_of_two_extends_the_reference() {
        let slopes = alibi_slopes(12);
        assert_eq!(slopes.len(), 12);

        let base = alibi_slopes(8);
        for (i, &slope) in slopes.iter().take(8).enumerate() {
            assert!(
                (slope - base[i]).abs() < 1e-7,
                "head {i} must match the 8-head construction: {slope} vs {}",
                base[i]
            );
        }

        let next = alibi_slopes(16);
        for (offset, &slope) in slopes.iter().skip(8).enumerate() {
            let expected = next[offset * 2];
            assert!(
                (slope - expected).abs() < 1e-7,
                "head {} must be next[{}] = {expected}, got {slope}",
                8 + offset,
                offset * 2
            );
        }
    }

    /// Falcon-7B has 71 heads — the odd branch must not panic and must stay
    /// positive and finite for every head.
    #[test]
    fn test_alibi_slopes_seventy_one_heads() {
        let slopes = alibi_slopes(71);
        assert_eq!(slopes.len(), 71);
        assert!(slopes.iter().all(|s| *s > 0.0 && s.is_finite()));
        // The first 64 come from the 64-head construction.
        let base = alibi_slopes(64);
        for (i, &slope) in slopes.iter().take(64).enumerate() {
            assert!((slope - base[i]).abs() < 1e-9, "head {i}");
        }
    }

    #[test]
    fn test_alibi_rejects_zero_heads() {
        assert!(ALiBi::new(0).is_err(), "zero heads is not a valid ALiBi");
    }

    // ---- ALiBi bias: reference math and placement ----

    /// Hand-computed reference for `num_heads = 2`:
    /// `ratio = 2^(-8/2) = 0.0625`, so slopes are `[0.0625, 0.0625^2]`.
    #[test]
    fn test_alibi_slopes_two_heads_reference_values() {
        let alibi = ALiBi::new(2).expect("ALiBi with 2 heads");
        let slopes = alibi.slopes().data().expect("slope data");
        assert_eq!(slopes.len(), 2, "one slope per head");
        assert!(
            (slopes[0] - 0.0625).abs() < 1e-7,
            "slope[0] = {}",
            slopes[0]
        );
        assert!(
            (slopes[1] - 0.003_906_25).abs() < 1e-9,
            "slope[1] = {}",
            slopes[1]
        );
    }

    /// The bias must be `[1, num_heads, seq_len, seq_len]` — the old code produced
    /// `num_heads * seq_len * seq_len` values but declared a `[seq_len, seq_len]`
    /// shape, which is an element-count mismatch.
    #[test]
    fn test_alibi_build_bias_shape_and_reference_values() {
        let alibi = ALiBi::new(2).expect("ALiBi with 2 heads");
        let bias = alibi.build_bias(3).expect("bias");
        assert_eq!(
            bias.shape(),
            &[1, 2, 3, 3],
            "bias must be [1, num_heads, seq_len, seq_len]"
        );

        match &bias {
            Tensor::F32(arr) => {
                // Head 0, slope 0.0625: bias[i][j] = -(i - j) * slope for j <= i.
                assert_eq!(arr[[0, 0, 0, 0]], 0.0);
                assert!((arr[[0, 0, 1, 0]] + 0.0625).abs() < 1e-7);
                assert_eq!(arr[[0, 0, 1, 1]], 0.0);
                assert!((arr[[0, 0, 2, 0]] + 0.125).abs() < 1e-7);
                assert!((arr[[0, 0, 2, 1]] + 0.0625).abs() < 1e-7);
                assert_eq!(arr[[0, 0, 2, 2]], 0.0);
                // Future positions carry no positional bias (causal mask handles them).
                assert_eq!(arr[[0, 0, 0, 1]], 0.0);
                // Head 1 uses the steeper-decaying slope 0.0625^2.
                assert!((arr[[0, 1, 2, 0]] + 2.0 * 0.003_906_25).abs() < 1e-8);
            },
            _ => panic!("expected F32 bias"),
        }
    }

    /// ALiBi must land on the pre-softmax logits. Adding it to the post-softmax
    /// attention output (shape `[batch, seq_len, num_heads * head_dim]`) is the
    /// bug this rejects.
    #[test]
    fn test_alibi_apply_bias_rejects_post_softmax_output_shape() {
        let alibi = ALiBi::new(4).expect("ALiBi with 4 heads");
        let post_softmax_output = Tensor::zeros(&[1, 3, 4 * 8]).expect("output tensor");
        let result = alibi.apply_bias(&post_softmax_output, 3);
        assert!(
            result.is_err(),
            "ALiBi must refuse anything that is not [batch, num_heads, seq, seq]"
        );
    }

    /// Applying the bias to zero logits and taking a softmax must yield the
    /// hand-computed ALiBi distribution: closer keys get more probability mass.
    #[test]
    fn test_alibi_biased_softmax_matches_hand_computation() {
        let alibi = ALiBi::new(2).expect("ALiBi with 2 heads");
        let scores = Tensor::zeros(&[1, 2, 3, 3]).expect("zero scores");
        let biased = alibi.apply_bias(&scores, 3).expect("biased scores");
        // Mask the future so the softmax is over the causal prefix only.
        let mut mask_data = vec![0.0f32; 9];
        for i in 0..3 {
            for j in (i + 1)..3 {
                mask_data[i * 3 + j] = f32::NEG_INFINITY;
            }
        }
        let mask = Tensor::from_vec(mask_data, &[3, 3]).expect("mask");
        let weights = biased.add(&mask).expect("masked").softmax(-1).expect("softmax");

        // Head 0, query position 2: logits are [-0.125, -0.0625, 0.0].
        let slope = 0.0625f32;
        let raw = [(-2.0 * slope).exp(), (-slope).exp(), 1.0f32];
        let denom: f32 = raw.iter().sum();
        match &weights {
            Tensor::F32(arr) => {
                for (j, &r) in raw.iter().enumerate() {
                    let expected = r / denom;
                    let got = arr[[0, 0, 2, j]];
                    assert!(
                        (got - expected).abs() < 1e-6,
                        "weight[2][{j}] = {got}, expected {expected}"
                    );
                }
                // Monotone decay with distance is the defining ALiBi property.
                assert!(arr[[0, 0, 2, 2]] > arr[[0, 0, 2, 1]]);
                assert!(arr[[0, 0, 2, 1]] > arr[[0, 0, 2, 0]]);
            },
            _ => panic!("expected F32 weights"),
        }
    }

    /// End-to-end: an attention layer with ALiBi enabled must (a) run at all and
    /// (b) produce a different result from the identical layer without ALiBi.
    /// The pre-fix code failed (a) — the bias was applied to the post-softmax
    /// output, whose shape does not match.
    #[test]
    fn test_falcon_attention_alibi_changes_output() {
        let mut config_alibi = tiny_falcon_config();
        config_alibi.alibi = true;
        let config_plain = tiny_falcon_config();

        let with_alibi = FalconAttention::new(&config_alibi).expect("attention with ALiBi");
        let mut without_alibi =
            FalconAttention::new(&config_plain).expect("attention without ALiBi");

        // Share the projection weights so the only difference is the bias.
        for (dst, src) in [
            (&mut without_alibi.q_proj, &with_alibi.q_proj),
            (&mut without_alibi.k_proj, &with_alibi.k_proj),
            (&mut without_alibi.v_proj, &with_alibi.v_proj),
            (&mut without_alibi.dense, &with_alibi.dense),
        ] {
            dst.set_weight(src.weight().clone()).expect("copy weight");
        }
        // Re-borrow immutably after the weight copy.
        let with_alibi = &with_alibi;
        let without_alibi = &without_alibi;

        let seq_len = 4;
        let hidden = config_alibi.hidden_size;
        let input_data: Vec<f32> =
            (0..seq_len * hidden).map(|i| ((i % 13) as f32 - 6.0) * 0.05).collect();
        let input = Tensor::from_vec(input_data, &[1, seq_len, hidden]).expect("input");

        let biased = with_alibi.forward(input.clone()).expect("ALiBi forward must succeed");
        let plain = without_alibi.forward(input).expect("plain forward");

        let biased_data = biased.data().expect("biased data");
        let plain_data = plain.data().expect("plain data");
        assert_eq!(biased_data.len(), plain_data.len());
        let max_diff = biased_data
            .iter()
            .zip(plain_data.iter())
            .fold(0.0f32, |acc, (a, b)| acc.max((a - b).abs()));
        assert!(
            max_diff > 1e-6,
            "ALiBi must change the attention output (max diff {max_diff})"
        );
    }

    /// The bias is built as `[1, num_heads, seq, seq]` and must broadcast over a
    /// real batch dimension.
    #[test]
    fn test_falcon_attention_alibi_batched_forward() {
        let mut config = tiny_falcon_config();
        config.alibi = true;
        let attn = FalconAttention::new(&config).expect("attention");

        let batch = 2;
        let seq_len = 3;
        let hidden = config.hidden_size;
        let data: Vec<f32> =
            (0..batch * seq_len * hidden).map(|i| ((i % 11) as f32 - 5.0) * 0.05).collect();
        let input = Tensor::from_vec(data.clone(), &[batch, seq_len, hidden]).expect("input");
        let out = attn.forward(input).expect("batched ALiBi forward must succeed");
        assert_eq!(out.shape(), &[batch, seq_len, hidden]);

        // Each batch element is independent: running item 0 alone must match.
        let single = Tensor::from_vec(data[..seq_len * hidden].to_vec(), &[1, seq_len, hidden])
            .expect("single");
        let single_out = attn.forward(single).expect("single forward").data().expect("data");
        let batched = out.data().expect("batched data");
        for (i, expected) in single_out.iter().enumerate() {
            assert!(
                (batched[i] - expected).abs() < 1e-5,
                "batch element 0 diverged at {i}: {} vs {expected}",
                batched[i]
            );
        }
    }

    /// The causal mask contributes `-inf` to the masked positions and ALiBi adds
    /// a finite bias on top; the softmax must still yield finite probabilities
    /// (a fully-masked row, or `-inf + -inf` leaking into the sum, would produce
    /// NaN and quietly poison every downstream layer).
    #[test]
    fn test_falcon_attention_alibi_output_is_finite() {
        let mut config = tiny_falcon_config();
        config.alibi = true;
        let attn = FalconAttention::new(&config).expect("attention");

        let batch = 2;
        let seq_len = 5;
        let hidden = config.hidden_size;
        let data: Vec<f32> =
            (0..batch * seq_len * hidden).map(|i| ((i % 17) as f32 - 8.0) * 0.25).collect();
        let input = Tensor::from_vec(data, &[batch, seq_len, hidden]).expect("input");
        let out = attn.forward(input).expect("forward").data().expect("data");
        assert_eq!(out.len(), batch * seq_len * hidden);
        for (i, value) in out.iter().enumerate() {
            assert!(
                value.is_finite(),
                "output[{i}] = {value} is not finite: the masked softmax produced NaN/inf"
            );
        }
    }

    #[test]
    fn test_falcon_attention_alibi_output_depends_on_input() {
        let mut config = tiny_falcon_config();
        config.alibi = true;
        let attn = FalconAttention::new(&config).expect("attention");

        let seq_len = 3;
        let hidden = config.hidden_size;
        let make = |scale: f32| {
            let data: Vec<f32> =
                (0..seq_len * hidden).map(|i| ((i % 7) as f32 - 3.0) * scale).collect();
            Tensor::from_vec(data, &[1, seq_len, hidden]).expect("input")
        };

        let out_a = attn.forward(make(0.05)).expect("forward a");
        let out_b = attn.forward(make(0.5)).expect("forward b");
        let a = out_a.data().expect("data a");
        let b = out_b.data().expect("data b");
        let max_diff = a.iter().zip(b.iter()).fold(0.0f32, |acc, (x, y)| acc.max((x - y).abs()));
        assert!(max_diff > 1e-6, "output must depend on the input");
    }

    // ---- Causal mask ----

    #[test]
    fn test_causal_mask_upper_triangle_is_neg_inf() {
        let config = tiny_falcon_config();
        let attn = FalconAttention::new(&config).expect("FalconAttention");
        let mask = attn.create_causal_mask(4).expect("causal mask");
        match &mask {
            Tensor::F32(arr) => {
                // Upper triangle [0,1], [0,2], [0,3], [1,2], etc. must be -inf
                assert!(arr[[0, 1]].is_infinite() && arr[[0, 1]] < 0.0);
                assert!(arr[[0, 2]].is_infinite() && arr[[0, 2]] < 0.0);
                assert!(arr[[1, 2]].is_infinite() && arr[[1, 2]] < 0.0);
                // Diagonal and below must be 0
                assert_eq!(arr[[0, 0]], 0.0);
                assert_eq!(arr[[1, 1]], 0.0);
                assert_eq!(arr[[2, 1]], 0.0);
            },
            _ => panic!("Expected F32 mask"),
        }
    }
}
