use crate::llava::config::{LlavaConfig, LlavaVisionConfig};
use trustformers_core::{
    device::Device,
    errors::Result,
    layers::{Embedding, LayerNorm, Linear},
    ops::activations::{gelu, silu},
    tensor::{DType, Tensor},
    traits::Layer,
};

/// LLaVA Vision Transformer implementation
pub struct LlavaVisionTransformer {
    #[allow(dead_code)]
    config: LlavaVisionConfig,
    embeddings: LlavaVisionEmbeddings,
    encoder: LlavaVisionEncoder,
    post_layernorm: LayerNorm,
    device: Device,
}

impl LlavaVisionTransformer {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let embeddings = LlavaVisionEmbeddings::new_with_device(config.clone(), device)?;
        let encoder = LlavaVisionEncoder::new_with_device(config.clone(), device)?;
        let post_layernorm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            config,
            embeddings,
            encoder,
            post_layernorm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionTransformer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, pixel_values: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.embeddings.forward(pixel_values)?;
        let hidden_states = self.encoder.forward(hidden_states)?;
        let pooled_output = self.post_layernorm.forward(hidden_states)?;
        Ok(pooled_output)
    }
}

/// Vision embeddings with patch embedding and position encoding
pub struct LlavaVisionEmbeddings {
    config: LlavaVisionConfig,
    patch_embedding: Linear,
    position_embedding: Embedding,
    class_embedding: Tensor,
    device: Device,
}

impl LlavaVisionEmbeddings {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let patch_size = config.patch_size;
        let patch_embedding = Linear::new_with_device(
            config.num_channels * patch_size * patch_size,
            config.hidden_size,
            false,
            device,
        );

        let num_patches = (config.image_size / patch_size).pow(2);
        let num_positions = num_patches + 1; // +1 for class token
        let position_embedding =
            Embedding::new_with_device(num_positions, config.hidden_size, None, device)?;

        let class_embedding = Tensor::randn(&[config.hidden_size])?;

        Ok(Self {
            config,
            patch_embedding,
            position_embedding,
            class_embedding,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionEmbeddings {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, pixel_values: Self::Input) -> Result<Self::Output> {
        let batch_size = pixel_values.shape()[0];
        let patch_size = self.config.patch_size;
        let image_size = self.config.image_size;
        let _num_patches = (image_size / patch_size).pow(2);

        // Extract patches
        let patches = extract_patches(&pixel_values, patch_size)?;
        let patch_embeds = self.patch_embedding.forward(patches)?;

        // Add class token
        let class_embeds = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?.broadcast_to(&[
            batch_size,
            1,
            self.config.hidden_size,
        ])?;
        let embeddings = Tensor::concat(&[class_embeds, patch_embeds], 1)?;

        // Add position embeddings
        let seq_len = embeddings.shape()[1];
        let position_ids = Tensor::range(0, seq_len as i64, DType::I64)?;
        let position_ids_vec: Vec<u32> =
            position_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let position_embeds = self.position_embedding.forward(position_ids_vec)?;
        let embeddings = embeddings.add(&position_embeds.unsqueeze(0)?)?;

        Ok(embeddings)
    }
}

/// Vision transformer encoder
pub struct LlavaVisionEncoder {
    pub layers: Vec<LlavaVisionEncoderLayer>,
    device: Device,
}

impl LlavaVisionEncoder {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlavaVisionEncoderLayer::new_with_device(
                config.clone(),
                device,
            )?);
        }

        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionEncoder {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = hidden_states;

        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// Vision transformer encoder layer
pub struct LlavaVisionEncoderLayer {
    self_attn: LlavaVisionAttention,
    mlp: LlavaVisionMLP,
    layer_norm1: LayerNorm,
    layer_norm2: LayerNorm,
    device: Device,
}

impl LlavaVisionEncoderLayer {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let self_attn = LlavaVisionAttention::new_with_device(config.clone(), device)?;
        let mlp = LlavaVisionMLP::new_with_device(config.clone(), device)?;
        let layer_norm1 =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;
        let layer_norm2 =
            LayerNorm::new_with_device(vec![config.hidden_size], config.layer_norm_eps, device)?;

        Ok(Self {
            self_attn,
            mlp,
            layer_norm1,
            layer_norm2,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionEncoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let residual = hidden_states.clone();

        // Self-attention with pre-norm
        let hidden_states = self.layer_norm1.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(hidden_states)?;
        let hidden_states = residual.add(&attn_output)?;

        let residual = hidden_states.clone();

        // MLP with pre-norm
        let hidden_states = self.layer_norm2.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// Vision attention mechanism
pub struct LlavaVisionAttention {
    config: LlavaVisionConfig,
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    pub head_dim: usize,
    scale: f32,
    device: Device,
}

impl LlavaVisionAttention {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let head_dim = config.hidden_size / config.num_attention_heads;
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, true, device);
        let k_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, true, device);
        let v_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, true, device);
        let out_proj =
            Linear::new_with_device(config.hidden_size, config.hidden_size, true, device);

        Ok(Self {
            config,
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            head_dim,
            scale,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let batch_size = hidden_states.shape()[0];
        let seq_len = hidden_states.shape()[1];
        let num_heads = self.config.num_attention_heads;

        // Project to query, key, value
        let query = self.q_proj.forward(hidden_states.clone())?;
        let key = self.k_proj.forward(hidden_states.clone())?;
        let value = self.v_proj.forward(hidden_states)?;

        // Reshape for multi-head attention
        let query = query
            .reshape(&[batch_size, seq_len, num_heads, self.head_dim])?
            .transpose(1, 2)?;
        let key = key.reshape(&[batch_size, seq_len, num_heads, self.head_dim])?.transpose(1, 2)?;
        let value = value
            .reshape(&[batch_size, seq_len, num_heads, self.head_dim])?
            .transpose(1, 2)?;

        // Compute attention scores
        let attn_weights = query.matmul(&key.transpose_i64(-2, -1)?)?;
        let attn_weights = attn_weights.mul_scalar(self.scale)?;
        let attn_weights = attn_weights.softmax(-1)?;

        // Apply dropout
        let attn_weights = if self.config.attention_dropout > 0.0 {
            attn_weights.dropout(self.config.attention_dropout)?
        } else {
            attn_weights
        };

        // Apply attention to values
        let attn_output = attn_weights.matmul(&value)?;
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape(&[
            batch_size,
            seq_len,
            self.config.hidden_size,
        ])?;

        let output = self.out_proj.forward(attn_output)?;
        Ok(output)
    }
}

/// Vision MLP with GELU activation
pub struct LlavaVisionMLP {
    fc1: Linear,
    fc2: Linear,
    dropout: f32,
    device: Device,
}

impl LlavaVisionMLP {
    pub fn new(config: LlavaVisionConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaVisionConfig, device: Device) -> Result<Self> {
        let fc1 =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, true, device);
        let fc2 =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, true, device);

        Ok(Self {
            fc1,
            fc2,
            dropout: config.dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaVisionMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.fc1.forward(hidden_states)?;
        let hidden_states = gelu(&hidden_states)?;

        let hidden_states = if self.dropout > 0.0 {
            hidden_states.dropout(self.dropout)?
        } else {
            hidden_states
        };

        let hidden_states = self.fc2.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

/// Multimodal projector to connect vision and language models
pub struct LlavaMultiModalProjector {
    projector_type: String,
    layers: Vec<Linear>,
    device: Device,
}

impl LlavaMultiModalProjector {
    pub fn new(projector_type: String, input_dim: usize, output_dim: usize) -> Result<Self> {
        Self::new_with_device(projector_type, input_dim, output_dim, Device::CPU)
    }

    pub fn new_with_device(
        projector_type: String,
        input_dim: usize,
        output_dim: usize,
        device: Device,
    ) -> Result<Self> {
        let mut layers = Vec::new();

        match projector_type.as_str() {
            "linear" => {
                layers.push(Linear::new_with_device(input_dim, output_dim, true, device));
            },
            "mlp2x_gelu" => {
                let hidden_dim = output_dim;
                layers.push(Linear::new_with_device(input_dim, hidden_dim, true, device));
                layers.push(Linear::new_with_device(
                    hidden_dim, output_dim, true, device,
                ));
            },
            "mlp2x_relu" => {
                let hidden_dim = output_dim;
                layers.push(Linear::new_with_device(input_dim, hidden_dim, true, device));
                layers.push(Linear::new_with_device(
                    hidden_dim, output_dim, true, device,
                ));
            },
            _ => {
                return Err(trustformers_core::errors::invalid_config(
                    "projector_type",
                    format!("Unsupported projector type: {}", projector_type),
                ));
            },
        }

        Ok(Self {
            projector_type,
            layers,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaMultiModalProjector {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, image_features: Self::Input) -> Result<Self::Output> {
        let mut hidden_states = image_features;

        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(hidden_states)?;

            // Apply activation between layers
            if i < self.layers.len() - 1 {
                hidden_states = match self.projector_type.as_str() {
                    "mlp2x_gelu" => gelu(&hidden_states)?,
                    "mlp2x_relu" => hidden_states.relu()?,
                    _ => hidden_states,
                };
            }
        }

        Ok(hidden_states)
    }
}

/// Main LLaVA model combining vision and language
pub struct LlavaForConditionalGeneration {
    config: LlavaConfig,
    vision_tower: LlavaVisionTransformer,
    language_model: LlavaLanguageModel,
    mm_projector: LlavaMultiModalProjector,
    device: Device,
}

impl LlavaForConditionalGeneration {
    pub fn new(config: LlavaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaConfig, device: Device) -> Result<Self> {
        let vision_tower =
            LlavaVisionTransformer::new_with_device(config.vision_config.clone(), device)?;
        let language_model = LlavaLanguageModel::new_with_device(config.clone(), device)?;
        let mm_projector = LlavaMultiModalProjector::new_with_device(
            config.mm_projector_type.clone(),
            config.vision_config.hidden_size,
            config.mm_hidden_size,
            device,
        )?;

        Ok(Self {
            config,
            vision_tower,
            language_model,
            mm_projector,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Process images and text together
    pub fn forward_multimodal(
        &self,
        input_ids: Tensor,
        pixel_values: Option<Tensor>,
        attention_mask: Option<Tensor>,
    ) -> Result<LlavaOutput> {
        let mut inputs_embeds = self.language_model.get_input_embeddings(input_ids.clone())?;

        if let Some(pixel_values) = pixel_values {
            // Extract image features
            let image_features = self.vision_tower.forward(pixel_values)?;

            // Select features from specified layer
            let selected_features = if self.config.mm_vision_select_layer >= 0 {
                // Select from specific layer (not implemented in this simplified version)
                image_features
            } else {
                // Select from last N layers
                image_features
            };

            // Project image features to language model dimension
            let projected_features = self.mm_projector.forward(selected_features)?;

            // Merge image and text embeddings
            inputs_embeds =
                self.merge_multimodal_embeddings(inputs_embeds, projected_features, &input_ids)?;
        }

        // Forward through language model
        let outputs = self.language_model.forward_with_embeddings(inputs_embeds, attention_mask)?;

        Ok(LlavaOutput {
            logits: outputs.logits,
            hidden_states: outputs.hidden_states,
            attentions: outputs.attentions,
        })
    }

    fn merge_multimodal_embeddings(
        &self,
        text_embeds: Tensor,
        image_embeds: Tensor,
        _input_ids: &Tensor,
    ) -> Result<Tensor> {
        // This is a simplified implementation
        // In practice, this would involve more sophisticated merging
        // based on special image tokens in the input

        let _batch_size = text_embeds.shape()[0];
        let _text_seq_len = text_embeds.shape()[1];
        let _image_seq_len = image_embeds.shape()[1];
        let _hidden_size = text_embeds.shape()[2];

        // For now, concatenate image and text embeddings
        let merged = Tensor::concat(&[image_embeds, text_embeds], 1)?;

        Ok(merged)
    }
}

/// Simplified language model component
pub struct LlavaLanguageModel {
    #[allow(dead_code)]
    config: LlavaConfig,
    embed_tokens: Embedding,
    pub layers: Vec<LlavaDecoderLayer>,
    norm: LayerNorm,
    lm_head: Linear,
    device: Device,
}

impl LlavaLanguageModel {
    pub fn new(config: LlavaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaConfig, device: Device) -> Result<Self> {
        let embed_tokens =
            Embedding::new_with_device(config.vocab_size, config.hidden_size, None, device)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(LlavaDecoderLayer::new_with_device(config.clone(), device)?);
        }

        let norm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.rms_norm_eps, device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            config,
            embed_tokens,
            layers,
            norm,
            lm_head,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn get_input_embeddings(&self, input_ids: Tensor) -> Result<Tensor> {
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        self.embed_tokens.forward(input_ids_vec)
    }

    pub fn forward_with_embeddings(
        &self,
        inputs_embeds: Tensor,
        _attention_mask: Option<Tensor>,
    ) -> Result<LlavaLanguageOutput> {
        let mut hidden_states = inputs_embeds;

        // Pass through all decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(hidden_states)?;
        }

        // Apply final layer norm
        hidden_states = self.norm.forward(hidden_states)?;

        // Compute logits
        let logits = self.lm_head.forward(hidden_states.clone())?;

        Ok(LlavaLanguageOutput {
            logits,
            hidden_states: Some(hidden_states),
            attentions: None,
        })
    }
}

/// Simplified decoder layer (would use actual LLaMA/Vicuna layer in practice)
pub struct LlavaDecoderLayer {
    self_attn: LlavaAttention,
    mlp: LlavaMLP,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
    device: Device,
}

impl LlavaDecoderLayer {
    pub fn new(config: LlavaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaConfig, device: Device) -> Result<Self> {
        let self_attn = LlavaAttention::new_with_device(config.clone(), device)?;
        let mlp = LlavaMLP::new_with_device(config.clone(), device)?;
        let input_layernorm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.rms_norm_eps, device)?;
        let post_attention_layernorm =
            LayerNorm::new_with_device(vec![config.hidden_size], config.rms_norm_eps, device)?;

        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaDecoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let residual = hidden_states.clone();

        // Self-attention with pre-norm
        let hidden_states = self.input_layernorm.forward(hidden_states)?;
        let attn_output = self.self_attn.forward(hidden_states)?;
        let hidden_states = residual.add(&attn_output)?;

        let residual = hidden_states.clone();

        // MLP with pre-norm
        let hidden_states = self.post_attention_layernorm.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// Simplified attention (would use RoPE and other optimizations in practice)
pub struct LlavaAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    pub head_dim: usize,
    pub num_heads: usize,
    scale: f32,
    device: Device,
}

impl LlavaAttention {
    pub fn new(config: LlavaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaConfig, device: Device) -> Result<Self> {
        let head_dim = config.head_dim();
        let scale = 1.0 / (head_dim as f32).sqrt();

        let q_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, false, device);
        let k_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, false, device);
        let v_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, false, device);
        let o_proj = Linear::new_with_device(config.hidden_size, config.hidden_size, false, device);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            head_dim,
            num_heads: config.num_attention_heads,
            scale,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaAttention {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        // Simplified attention implementation
        let query = self.q_proj.forward(hidden_states.clone())?;
        let key = self.k_proj.forward(hidden_states.clone())?;
        let value = self.v_proj.forward(hidden_states)?;

        // Apply scaled dot-product attention (simplified)
        let attn_output = scaled_dot_product_attention(&query, &key, &value, self.scale)?;
        let output = self.o_proj.forward(attn_output)?;

        Ok(output)
    }
}

/// Simplified MLP
pub struct LlavaMLP {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    device: Device,
}

impl LlavaMLP {
    pub fn new(config: LlavaConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: LlavaConfig, device: Device) -> Result<Self> {
        let gate_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let up_proj =
            Linear::new_with_device(config.hidden_size, config.intermediate_size, false, device);
        let down_proj =
            Linear::new_with_device(config.intermediate_size, config.hidden_size, false, device);

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }
}

impl Layer for LlavaMLP {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, hidden_states: Self::Input) -> Result<Self::Output> {
        let gate_output = self.gate_proj.forward(hidden_states.clone())?;
        let up_output = self.up_proj.forward(hidden_states)?;

        let gate_output = silu(&gate_output)?;
        let intermediate = gate_output.mul(&up_output)?;
        let output = self.down_proj.forward(intermediate)?;

        Ok(output)
    }
}

/// Output structures
#[derive(Debug)]
pub struct LlavaOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Tensor>,
}

#[derive(Debug)]
pub struct LlavaLanguageOutput {
    pub logits: Tensor,
    pub hidden_states: Option<Tensor>,
    pub attentions: Option<Tensor>,
}

// Helper functions

fn extract_patches(pixel_values: &Tensor, _patch_size: usize) -> Result<Tensor> {
    // Simplified patch extraction
    // In practice, this would use proper convolution or unfold operations
    Ok(pixel_values.clone())
}

fn scaled_dot_product_attention(
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    scale: f32,
) -> Result<Tensor> {
    // Simplified attention computation
    let scores = query.matmul(&key.transpose_i64(-2, -1)?)?;
    let scores = scores.mul_scalar(scale)?;
    let attn_weights = scores.softmax(-1)?;
    let output = attn_weights.matmul(value)?;
    Ok(output)
}
