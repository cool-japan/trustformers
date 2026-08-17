use crate::llava::config::{LlavaConfig, LlavaVisionConfig};
use trustformers_core::{
    device::Device,
    errors::{Result, TrustformersError},
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

    /// Splice the projected image features into the text embedding sequence at
    /// the positions of the image placeholder token.
    ///
    /// LLaVA's prompt format inserts a single `<image>` token
    /// (`config.mm_patch_token`) where the picture belongs; the model replaces
    /// that one position with the whole run of projected patch embeddings, so
    /// the image lands *inside* the sentence — "USER: `<image>` what is in this
    /// picture?" must put the picture between "USER:" and "what". Everything
    /// before and after the placeholder keeps its order.
    ///
    /// # What this replaces
    ///
    /// The previous body took `_input_ids`, ignored it entirely, and returned
    /// `concat([image_embeds, text_embeds], dim=1)` — the image bolted onto the
    /// *front* of the prompt, regardless of where the user put the placeholder.
    /// The `<image>` token itself stayed in the text as an ordinary embedding,
    /// so the sequence contained both a stray placeholder and a misplaced image,
    /// and any prompt whose instruction preceded the picture was silently
    /// reordered.
    ///
    /// # Errors
    ///
    /// Fails when the tensors are not 3-D `F32`, when their batch or hidden
    /// dimensions disagree, or when the prompt contains no image placeholder
    /// while image features were supplied — that combination means the caller
    /// built the prompt wrongly, and appending the image somewhere arbitrary
    /// would hide it.
    fn merge_multimodal_embeddings(
        &self,
        text_embeds: Tensor,
        image_embeds: Tensor,
        input_ids: &Tensor,
    ) -> Result<Tensor> {
        let text_shape = text_embeds.shape();
        let image_shape = image_embeds.shape();
        let [batch_size, text_seq_len, hidden_size] = text_shape[..] else {
            return Err(TrustformersError::shape_error(format!(
                "text embeddings must be [batch, seq, hidden], got {text_shape:?}"
            )));
        };
        let [image_batch, image_seq_len, image_hidden] = image_shape[..] else {
            return Err(TrustformersError::shape_error(format!(
                "image embeddings must be [batch, patches, hidden], got {image_shape:?}"
            )));
        };
        if image_batch != batch_size || image_hidden != hidden_size {
            return Err(TrustformersError::shape_error(format!(
                "image embeddings {image_shape:?} do not match text embeddings {text_shape:?}"
            )));
        }

        let ids: Vec<u32> = input_ids
            .data()
            .map_err(|e| {
                TrustformersError::tensor_op_error(
                    "merge_multimodal_embeddings",
                    &format!("failed to read input_ids: {e}"),
                )
            })?
            .into_iter()
            .map(|v| v as u32)
            .collect();
        if ids.len() != batch_size * text_seq_len {
            return Err(TrustformersError::shape_error(format!(
                "input_ids hold {} id(s) but the text embeddings describe {} position(s)",
                ids.len(),
                batch_size * text_seq_len
            )));
        }

        let placeholder = self.config.mm_patch_token;
        let text_values = text_embeds.data().map_err(|e| {
            TrustformersError::tensor_op_error(
                "merge_multimodal_embeddings",
                &format!("failed to read text embeddings: {e}"),
            )
        })?;
        let image_values = image_embeds.data().map_err(|e| {
            TrustformersError::tensor_op_error(
                "merge_multimodal_embeddings",
                &format!("failed to read image embeddings: {e}"),
            )
        })?;

        // Every batch row must expand to the same length for the result to be a
        // dense tensor, so the placeholder count has to agree across the batch.
        let mut placeholder_counts = Vec::with_capacity(batch_size);
        for b in 0..batch_size {
            let row = &ids[b * text_seq_len..(b + 1) * text_seq_len];
            placeholder_counts.push(row.iter().filter(|&&id| id == placeholder).count());
        }
        let placeholders = placeholder_counts[0];
        if placeholders == 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "image features were supplied but the prompt contains no image placeholder \
                 token ({placeholder}); the image would have to be inserted at an arbitrary \
                 position"
            )));
        }
        if placeholder_counts.iter().any(|&count| count != placeholders) {
            return Err(TrustformersError::invalid_input_simple(format!(
                "every prompt in the batch must carry the same number of image placeholders, \
                 got {placeholder_counts:?}"
            )));
        }
        if !image_seq_len.is_multiple_of(placeholders) {
            return Err(TrustformersError::shape_error(format!(
                "{image_seq_len} image feature(s) cannot be split evenly across {placeholders} \
                 placeholder(s)"
            )));
        }
        let features_per_placeholder = image_seq_len / placeholders;

        // Each placeholder is replaced by `features_per_placeholder` positions.
        let merged_seq_len = text_seq_len - placeholders + image_seq_len;
        let mut merged = vec![0.0f32; batch_size * merged_seq_len * hidden_size];

        for b in 0..batch_size {
            let mut out_position = 0usize;
            let mut image_cursor = 0usize;
            for t in 0..text_seq_len {
                if ids[b * text_seq_len + t] == placeholder {
                    for _ in 0..features_per_placeholder {
                        let src = (b * image_seq_len + image_cursor) * hidden_size;
                        let dst = (b * merged_seq_len + out_position) * hidden_size;
                        merged[dst..dst + hidden_size]
                            .copy_from_slice(&image_values[src..src + hidden_size]);
                        image_cursor += 1;
                        out_position += 1;
                    }
                } else {
                    let src = (b * text_seq_len + t) * hidden_size;
                    let dst = (b * merged_seq_len + out_position) * hidden_size;
                    merged[dst..dst + hidden_size]
                        .copy_from_slice(&text_values[src..src + hidden_size]);
                    out_position += 1;
                }
            }
        }

        Tensor::from_vec(merged, &[batch_size, merged_seq_len, hidden_size])
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

/// Cut an image batch into a flat grid of non-overlapping square patches.
///
/// Input `[batch, channels, height, width]` becomes
/// `[batch, num_patches, channels * patch_size * patch_size]`, where
/// `num_patches = (height / patch_size) * (width / patch_size)` and the patches
/// are emitted in row-major grid order. Each patch's feature vector is laid out
/// channel-major (`c, py, px`), matching the flattened `Conv2d` kernel that a
/// HuggingFace ViT patch embedding uses — so the projection's weights line up
/// with the values it is multiplied by.
///
/// # What this replaces
///
/// The previous body was `Ok(pixel_values.clone())` under the comment
/// "Simplified patch extraction". The tensor handed to `patch_embedding` was
/// therefore the raw `[batch, channels, height, width]` image, while that
/// `Linear` expects `[.., channels * patch_size²]` — so for any real image the
/// projection either failed on shape or silently multiplied the wrong numbers,
/// and the "vision tower" never saw a patch at all.
///
/// # Errors
///
/// Fails when the input is not a 4-D `F32` tensor, when `patch_size` is 0, or
/// when the spatial dimensions are not divisible by `patch_size` — a partial
/// patch at the edge would have to be padded or dropped, and doing either
/// silently changes what the model sees.
fn extract_patches(pixel_values: &Tensor, patch_size: usize) -> Result<Tensor> {
    if patch_size == 0 {
        return Err(TrustformersError::invalid_input_simple(
            "patch_size must be greater than 0".to_string(),
        ));
    }
    let shape = pixel_values.shape();
    let [batch, channels, height, width] = shape[..] else {
        return Err(TrustformersError::shape_error(format!(
            "patch extraction expects [batch, channels, height, width], got {shape:?}"
        )));
    };
    if !height.is_multiple_of(patch_size) || !width.is_multiple_of(patch_size) {
        return Err(TrustformersError::shape_error(format!(
            "image {height}x{width} is not divisible into {patch_size}x{patch_size} patches"
        )));
    }

    let grid_h = height / patch_size;
    let grid_w = width / patch_size;
    let num_patches = grid_h * grid_w;
    let patch_dim = channels * patch_size * patch_size;

    let values = pixel_values.data().map_err(|e| {
        TrustformersError::tensor_op_error(
            "extract_patches",
            &format!("failed to read the pixel tensor: {e}"),
        )
    })?;

    let mut out = vec![0.0f32; batch * num_patches * patch_dim];
    for b in 0..batch {
        let image_offset = b * channels * height * width;
        for gy in 0..grid_h {
            for gx in 0..grid_w {
                let patch_index = gy * grid_w + gx;
                let out_offset = (b * num_patches + patch_index) * patch_dim;
                for c in 0..channels {
                    let channel_offset = image_offset + c * height * width;
                    for py in 0..patch_size {
                        let row = gy * patch_size + py;
                        let row_offset = channel_offset + row * width + gx * patch_size;
                        let dest = out_offset + (c * patch_size + py) * patch_size;
                        out[dest..dest + patch_size]
                            .copy_from_slice(&values[row_offset..row_offset + patch_size]);
                    }
                }
            }
        }
    }

    Tensor::from_vec(out, &[batch, num_patches, patch_dim])
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

#[cfg(test)]
mod tests {
    use super::*;

    // ── Patch extraction ────────────────────────────────────────────────────

    /// Regression: `extract_patches` was `Ok(pixel_values.clone())`, so the raw
    /// `[batch, channels, height, width]` image reached a projection expecting
    /// `[.., channels * patch_size²]`. This test pins both the output shape and
    /// the exact patch contents, and fails on the shape alone against the old
    /// no-op.
    #[test]
    fn extract_patches_cuts_a_real_grid() {
        // One 1-channel 4x4 image whose pixels are their own flat index.
        let pixels: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let image = Tensor::from_vec(pixels, &[1, 1, 4, 4]).expect("image must build");

        let patches = extract_patches(&image, 2).expect("patch extraction must succeed");
        // 2x2 grid of 2x2 patches, each flattened to 1*2*2 = 4 values.
        assert_eq!(patches.shape(), vec![1, 4, 4]);

        let values = patches.data().expect("readable");
        // Row-major grid order; each patch is its own 2x2 block of the image.
        assert_eq!(&values[0..4], &[0.0, 1.0, 4.0, 5.0], "top-left patch");
        assert_eq!(&values[4..8], &[2.0, 3.0, 6.0, 7.0], "top-right patch");
        assert_eq!(&values[8..12], &[8.0, 9.0, 12.0, 13.0], "bottom-left patch");
        assert_eq!(
            &values[12..16],
            &[10.0, 11.0, 14.0, 15.0],
            "bottom-right patch"
        );
    }

    /// Channels are interleaved per patch in `(c, py, px)` order, matching the
    /// flattened `Conv2d` kernel the projection's weights correspond to.
    #[test]
    fn extract_patches_lays_out_channels_first_within_a_patch() {
        // Two channels of a 2x2 image: channel 0 is 0..4, channel 1 is 10..14.
        let pixels: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0];
        let image = Tensor::from_vec(pixels, &[1, 2, 2, 2]).expect("image must build");

        let patches = extract_patches(&image, 2).expect("patch extraction must succeed");
        assert_eq!(patches.shape(), vec![1, 1, 8]);
        assert_eq!(
            patches.data().expect("readable"),
            vec![0.0, 1.0, 2.0, 3.0, 10.0, 11.0, 12.0, 13.0],
            "the single patch must hold channel 0 then channel 1"
        );
    }

    #[test]
    fn extract_patches_rejects_an_indivisible_image() {
        let image = Tensor::zeros(&[1, 3, 5, 5]).expect("image must build");
        let err = extract_patches(&image, 2)
            .expect_err("a 5x5 image cannot be cut into 2x2 patches without padding");
        assert!(
            err.to_string().contains("not divisible"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn extract_patches_rejects_a_non_image_tensor() {
        let flat = Tensor::zeros(&[16]).expect("tensor must build");
        assert!(extract_patches(&flat, 2).is_err());
        let image = Tensor::zeros(&[1, 1, 4, 4]).expect("image must build");
        assert!(extract_patches(&image, 0).is_err());
    }

    // ── Multimodal merge ────────────────────────────────────────────────────

    fn merge_config(placeholder: u32) -> LlavaConfig {
        LlavaConfig {
            vocab_size: 64,
            hidden_size: 4,
            intermediate_size: 8,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            mm_patch_token: placeholder,
            vision_config: LlavaVisionConfig {
                hidden_size: 4,
                intermediate_size: 8,
                num_hidden_layers: 1,
                num_attention_heads: 2,
                num_channels: 1,
                patch_size: 2,
                image_size: 4,
                ..LlavaVisionConfig::default()
            },
            ..LlavaConfig::default()
        }
    }

    /// Regression: `merge_multimodal_embeddings` ignored `input_ids` entirely
    /// and returned `concat([image, text])` — the image always at the front,
    /// wherever the prompt actually placed it, with the `<image>` placeholder
    /// left in the text as a stray embedding.
    #[test]
    fn image_features_land_at_the_placeholder_position() {
        let placeholder = 5u32;
        let model = LlavaForConditionalGeneration::new(merge_config(placeholder))
            .expect("model must build");

        // Prompt: [t0, <image>, t2] with distinctive per-position embeddings.
        let hidden = 4usize;
        let text: Vec<f32> = vec![
            1.0, 1.0, 1.0, 1.0, // position 0
            9.0, 9.0, 9.0, 9.0, // position 1: the placeholder
            3.0, 3.0, 3.0, 3.0, // position 2
        ];
        let text_embeds = Tensor::from_vec(text, &[1, 3, hidden]).expect("text must build");
        // Two image patches.
        let image = vec![7.0, 7.0, 7.0, 7.0, 8.0, 8.0, 8.0, 8.0];
        let image_embeds = Tensor::from_vec(image, &[1, 2, hidden]).expect("image must build");
        let input_ids =
            Tensor::from_vec(vec![0.0, placeholder as f32, 2.0], &[1, 3]).expect("ids must build");

        let merged = model
            .merge_multimodal_embeddings(text_embeds, image_embeds, &input_ids)
            .expect("merge must succeed");

        // 3 text positions - 1 placeholder + 2 image patches = 4 positions.
        assert_eq!(merged.shape(), vec![1, 4, hidden]);
        let values = merged.data().expect("readable");
        assert_eq!(
            &values[0..4],
            &[1.0, 1.0, 1.0, 1.0],
            "text before the image"
        );
        assert_eq!(&values[4..8], &[7.0, 7.0, 7.0, 7.0], "first image patch");
        assert_eq!(&values[8..12], &[8.0, 8.0, 8.0, 8.0], "second image patch");
        assert_eq!(
            &values[12..16],
            &[3.0, 3.0, 3.0, 3.0],
            "text after the image"
        );
        assert!(
            !values.iter().any(|v| (*v - 9.0).abs() < 1e-6),
            "the placeholder's own embedding must be replaced, not kept"
        );
    }

    /// A prompt whose instruction precedes the picture must keep that order.
    /// The old front-concatenation reversed it for every such prompt.
    #[test]
    fn text_before_the_image_is_not_reordered() {
        let placeholder = 5u32;
        let model = LlavaForConditionalGeneration::new(merge_config(placeholder))
            .expect("model must build");
        let hidden = 4usize;

        // Prompt: [t0, t1, <image>] — the picture comes last.
        let text_embeds = Tensor::from_vec(
            vec![1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 9.0, 9.0, 9.0, 9.0],
            &[1, 3, hidden],
        )
        .expect("text must build");
        let image_embeds =
            Tensor::from_vec(vec![7.0, 7.0, 7.0, 7.0], &[1, 1, hidden]).expect("image must build");
        let input_ids =
            Tensor::from_vec(vec![0.0, 1.0, placeholder as f32], &[1, 3]).expect("ids must build");

        let merged = model
            .merge_multimodal_embeddings(text_embeds, image_embeds, &input_ids)
            .expect("merge must succeed");
        let values = merged.data().expect("readable");
        assert_eq!(merged.shape(), vec![1, 3, hidden]);
        assert_eq!(&values[0..4], &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(&values[4..8], &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(
            &values[8..12],
            &[7.0, 7.0, 7.0, 7.0],
            "the image must stay at the end, where the prompt put it"
        );
    }

    #[test]
    fn a_prompt_without_a_placeholder_is_rejected() {
        let model = LlavaForConditionalGeneration::new(merge_config(5)).expect("model must build");
        let text_embeds = Tensor::zeros(&[1, 3, 4]).expect("text must build");
        let image_embeds = Tensor::zeros(&[1, 2, 4]).expect("image must build");
        let input_ids = Tensor::from_vec(vec![0.0, 1.0, 2.0], &[1, 3]).expect("ids must build");
        let err = model
            .merge_multimodal_embeddings(text_embeds, image_embeds, &input_ids)
            .expect_err("image features with nowhere to go must be an error");
        assert!(
            err.to_string().contains("no image placeholder"),
            "unexpected: {err}"
        );
    }

    #[test]
    fn mismatched_hidden_sizes_are_rejected() {
        let model = LlavaForConditionalGeneration::new(merge_config(5)).expect("model must build");
        let text_embeds = Tensor::zeros(&[1, 3, 4]).expect("text must build");
        let image_embeds = Tensor::zeros(&[1, 2, 8]).expect("image must build");
        let input_ids = Tensor::from_vec(vec![0.0, 5.0, 2.0], &[1, 3]).expect("ids must build");
        assert!(model
            .merge_multimodal_embeddings(text_embeds, image_embeds, &input_ids)
            .is_err());
    }
}
