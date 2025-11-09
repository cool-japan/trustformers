use crate::blip2::config::{Blip2Config, Blip2QFormerConfig, Blip2TextConfig, Blip2VisionConfig};
use trustformers_core::{
    kernels::fused_ops::ActivationType,
    layers::{
        attention::{AttentionConfig, MultiHeadAttention},
        embedding::Embedding,
        layernorm::LayerNorm,
        linear::Linear,
    },
    tensor::{DType, Tensor},
    traits::Layer,
};

/// BLIP-2 model for vision-language tasks
#[derive(Debug, Clone)]
pub struct Blip2Model {
    /// Configuration
    pub config: Blip2Config,
    /// Vision encoder
    pub vision_model: Blip2VisionModel,
    /// Q-Former model
    pub qformer_model: Blip2QFormerModel,
    /// Vision-language projector
    pub vision_projection: Linear,
    /// Text projector
    pub text_projection: Linear,
    /// Learned query embeddings
    pub query_tokens: Tensor,
    /// Layer norm for queries
    pub query_layer_norm: LayerNorm,
}

impl Blip2Model {
    /// Create a new BLIP-2 model
    pub fn new(config: Blip2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let vision_model = Blip2VisionModel::new(config.vision_config.clone())?;
        let qformer_model = Blip2QFormerModel::new(config.qformer_config.clone())?;

        // Project Q-Former output (768) to vision space (1408)
        let vision_projection = Linear::new(
            config.qformer_config.hidden_size,
            config.vision_config.hidden_size,
            false,
        );

        let text_projection = Linear::new(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            false,
        );

        // Initialize learnable query tokens
        let query_tokens =
            Tensor::randn(&[config.num_query_tokens, config.qformer_config.hidden_size])?;

        let query_layer_norm = LayerNorm::new(
            vec![config.qformer_config.hidden_size],
            config.qformer_config.layer_norm_eps as f32,
        )?;

        Ok(Self {
            config,
            vision_model,
            qformer_model,
            vision_projection,
            text_projection,
            query_tokens,
            query_layer_norm,
        })
    }

    /// Forward pass through the model
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Blip2Output, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];

        // Process images if provided
        let image_embeds = if let Some(pixel_values) = pixel_values {
            Some(self.vision_model.forward(pixel_values)?)
        } else {
            None
        };

        // Get query embeddings
        let query_embeds = self.get_query_embeddings(batch_size)?;

        // Process through Q-Former
        let qformer_outputs = self.qformer_model.forward(
            input_ids,
            image_embeds.as_ref(),
            Some(&query_embeds),
            attention_mask,
        )?;

        // Extract image and text features
        let image_features = if image_embeds.is_some() {
            Some(self.vision_projection.forward(qformer_outputs.pooler_output.clone())?)
        } else {
            None
        };

        let text_features = self.text_projection.forward(qformer_outputs.pooler_output.clone())?;

        Ok(Blip2Output {
            last_hidden_state: qformer_outputs.last_hidden_state,
            pooler_output: qformer_outputs.pooler_output,
            image_features,
            text_features,
            logits: qformer_outputs.logits,
        })
    }

    /// Get query embeddings expanded for batch
    fn get_query_embeddings(
        &self,
        batch_size: usize,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let query_embeds = self.query_layer_norm.forward(self.query_tokens.clone())?;

        // Expand for batch
        let expanded_shape = vec![
            batch_size,
            self.config.num_query_tokens,
            self.config.qformer_config.hidden_size,
        ];
        let expanded_embeds = query_embeds.unsqueeze(0)?.broadcast_to(&expanded_shape)?;

        Ok(expanded_embeds)
    }

    /// Get image features
    pub fn get_image_features(
        &self,
        pixel_values: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let image_embeds = self.vision_model.forward(pixel_values)?;
        let batch_size = pixel_values.shape()[0];

        let query_embeds = self.get_query_embeddings(batch_size)?;

        // Create dummy text input for Q-Former
        let input_ids = Tensor::zeros(&[batch_size, 1])?;

        let qformer_outputs = self.qformer_model.forward(
            &input_ids,
            Some(&image_embeds),
            Some(&query_embeds),
            None,
        )?;

        self.vision_projection
            .forward(qformer_outputs.pooler_output.clone())
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    /// Get text features
    pub fn get_text_features(
        &self,
        input_ids: &Tensor,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        let query_embeds = self.get_query_embeddings(batch_size)?;

        let qformer_outputs =
            self.qformer_model.forward(input_ids, None, Some(&query_embeds), None)?;

        self.text_projection
            .forward(qformer_outputs.pooler_output.clone())
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }
}

/// BLIP-2 for conditional generation
pub struct Blip2ForConditionalGeneration {
    /// BLIP-2 model
    pub blip2_model: Blip2Model,
    /// Language model head
    pub language_model: Box<dyn LanguageModel>,
    /// Language projection
    pub language_projection: Linear,
}

impl Blip2ForConditionalGeneration {
    /// Create a new BLIP-2 for conditional generation
    pub fn new(config: Blip2Config) -> Result<Self, Box<dyn std::error::Error>> {
        let blip2_model = Blip2Model::new(config.clone())?;

        let language_model: Box<dyn LanguageModel> = if config.use_decoder_only_language_model {
            Box::new(Blip2OptLanguageModel::new(config.text_config.clone())?)
        } else {
            Box::new(Blip2T5LanguageModel::new(config.text_config.clone())?)
        };

        let language_projection = Linear::new(
            config.qformer_config.hidden_size,
            config.text_config.hidden_size,
            false,
        );

        Ok(Self {
            blip2_model,
            language_model,
            language_projection,
        })
    }

    /// Forward pass for conditional generation
    pub fn forward(
        &self,
        input_ids: &Tensor,
        pixel_values: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<Blip2ConditionalGenerationOutput, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];

        // Get image features if provided
        let image_features = if let Some(pixel_values) = pixel_values {
            let image_embeds = self.blip2_model.vision_model.forward(pixel_values)?;

            let query_embeds = self.blip2_model.get_query_embeddings(batch_size)?;

            // Create dummy text input for Q-Former
            let dummy_input_ids = Tensor::zeros(&[batch_size, 1])?;

            let qformer_outputs = self.blip2_model.qformer_model.forward(
                &dummy_input_ids,
                Some(&image_embeds),
                Some(&query_embeds),
                None,
            )?;

            let projected =
                self.language_projection.forward(qformer_outputs.last_hidden_state.clone())?;
            Some(projected)
        } else {
            None
        };

        // Process through language model
        let language_outputs = self.language_model.forward(
            input_ids,
            image_features.as_ref(),
            attention_mask,
            labels,
        )?;

        Ok(Blip2ConditionalGenerationOutput {
            loss: language_outputs.loss,
            logits: language_outputs.logits,
            hidden_states: language_outputs.hidden_states,
            image_features,
        })
    }

    /// Generate text from image
    pub fn generate(
        &self,
        pixel_values: &Tensor,
        input_ids: Option<&Tensor>,
        max_length: usize,
        temperature: f32,
        top_p: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = pixel_values.shape()[0];

        // Get image features
        let image_embeds = self.blip2_model.vision_model.forward(pixel_values)?;
        let query_embeds = self.blip2_model.get_query_embeddings(batch_size)?;

        // Create dummy text input for Q-Former
        let dummy_input_ids = Tensor::zeros(&[batch_size, 1])?;

        let qformer_outputs = self.blip2_model.qformer_model.forward(
            &dummy_input_ids,
            Some(&image_embeds),
            Some(&query_embeds),
            None,
        )?;

        let image_features =
            self.language_projection.forward(qformer_outputs.last_hidden_state.clone())?;

        // Initialize with input_ids or BOS token
        let mut generated_ids = if let Some(input_ids) = input_ids {
            input_ids.clone()
        } else {
            Tensor::full(
                self.blip2_model.config.text_config.bos_token_id as f32,
                vec![batch_size, 1],
            )?
        };

        // Generate tokens
        for _ in 0..max_length {
            let outputs =
                self.language_model.forward(&generated_ids, Some(&image_features), None, None)?;

            // Apply temperature and top-p sampling
            let next_token_logits = outputs.logits.select(1, -1)?;
            let next_token = self.sample_token(&next_token_logits, temperature, top_p)?;

            // Append to generated sequence
            generated_ids = Tensor::concat(&[generated_ids, next_token.clone()], 1)?;

            // Check for EOS token
            if self.check_eos_token(&next_token)? {
                break;
            }
        }

        Ok(generated_ids)
    }

    /// Sample token with temperature and top-p
    fn sample_token(
        &self,
        logits: &Tensor,
        temperature: f32,
        _top_p: f32,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        let scaled_logits = logits.div_scalar(temperature)?;
        let probabilities = scaled_logits.softmax(-1)?;

        // Simple sampling (in practice, you'd implement proper nucleus sampling)
        let token_id = probabilities.argmax(-1)?;
        token_id
            .unsqueeze_i64(-1)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)
    }

    /// Check if EOS token is generated
    fn check_eos_token(&self, token: &Tensor) -> Result<bool, Box<dyn std::error::Error>> {
        let token_id = token.item::<i32>()?;
        Ok(token_id == self.blip2_model.config.text_config.eos_token_id)
    }
}

/// Vision model for BLIP-2
#[derive(Debug, Clone)]
pub struct Blip2VisionModel {
    /// Configuration
    pub config: Blip2VisionConfig,
    /// Patch embedding
    pub patch_embedding: Blip2PatchEmbedding,
    /// Class embedding
    pub class_embedding: Tensor,
    /// Position embedding
    pub position_embedding: Tensor,
    /// Transformer layers
    pub layers: Vec<Blip2VisionTransformerLayer>,
    /// Layer normalization
    pub layer_norm: LayerNorm,
    /// Pooler
    pub pooler: Option<Linear>,
}

impl Blip2VisionModel {
    /// Create a new vision model
    pub fn new(config: Blip2VisionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let patch_embedding = Blip2PatchEmbedding::new(&config)?;

        let class_embedding = Tensor::randn(&[config.hidden_size])?;
        let position_embedding = Tensor::randn(&[config.seq_len(), config.hidden_size])?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(Blip2VisionTransformerLayer::new(&config)?);
        }

        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        let pooler = Some(Linear::new(config.hidden_size, config.hidden_size, true));

        Ok(Self {
            config,
            patch_embedding,
            class_embedding,
            position_embedding,
            layers,
            layer_norm,
            pooler,
        })
    }

    /// Forward pass through vision model
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = pixel_values.shape()[0];

        // Patch embedding
        let patch_embeds = self.patch_embedding.forward(pixel_values)?;

        // Add class token
        let class_token = self.class_embedding.unsqueeze(0)?.unsqueeze(0)?;
        let class_token = class_token
            .broadcast_to(&[batch_size, 1, self.config.hidden_size])?
            .contiguous()?;

        let embeddings = Tensor::concat(&[class_token, patch_embeds], 1)?;

        // Add position embeddings
        let position_embeds = self.position_embedding.unsqueeze(0)?;
        let position_embeds = position_embeds
            .broadcast_to(&[batch_size, self.config.seq_len(), self.config.hidden_size])?
            .contiguous()?;
        let embeddings = embeddings.add(&position_embeds)?;

        // Apply transformer layers
        let mut hidden_states = embeddings;
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        // Layer normalization
        let hidden_states = self.layer_norm.forward(hidden_states)?;

        // Pooling
        if let Some(pooler) = &self.pooler {
            // select(1, 0) already reduces dimensions from [B, S, H] to [B, H]
            let cls_token = hidden_states.select(1, 0)?;
            let pooled = pooler.forward(cls_token)?;
            let pooled = pooled.tanh()?;
            let broadcasted = pooled.unsqueeze(1)?.broadcast_to(&[
                batch_size,
                hidden_states.shape()[1],
                self.config.hidden_size,
            ])?;
            // Ensure contiguous layout
            Ok(broadcasted.contiguous()?)
        } else {
            Ok(hidden_states)
        }
    }
}

/// Patch embedding for vision model
#[derive(Debug, Clone)]
pub struct Blip2PatchEmbedding {
    /// Projection layer
    pub projection: Linear,
    /// Patch size
    pub patch_size: usize,
    /// Hidden size
    pub hidden_size: usize,
}

impl Blip2PatchEmbedding {
    /// Create patch embedding
    pub fn new(config: &Blip2VisionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let projection = Linear::new(
            config.patch_size * config.patch_size * config.num_channels,
            config.hidden_size,
            true,
        );

        Ok(Self {
            projection,
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
        })
    }

    /// Forward pass
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let batch_size = pixel_values.shape()[0];
        let channels = pixel_values.shape()[1];
        let height = pixel_values.shape()[2];
        let width = pixel_values.shape()[3];

        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let num_patches = num_patches_h * num_patches_w;

        // Reshape to patches
        let patches = pixel_values.reshape(&[
            batch_size,
            channels,
            num_patches_h,
            self.patch_size,
            num_patches_w,
            self.patch_size,
        ])?;

        let patches = patches.permute(&[0, 2, 4, 1, 3, 5])?;

        // Convert to Vec and back to force contiguous standard layout
        let patches_vec = patches.to_vec_f32()?;
        let target_shape = vec![
            batch_size,
            num_patches,
            channels * self.patch_size * self.patch_size,
        ];
        let patches = Tensor::from_vec(patches_vec, &target_shape)?;

        // Project to hidden size
        let result = self
            .projection
            .forward(patches)
            .map_err(|e| Box::new(e) as Box<dyn std::error::Error>)?;
        Ok(result)
    }
}

/// Vision transformer layer
#[derive(Debug, Clone)]
pub struct Blip2VisionTransformerLayer {
    /// Self attention
    pub self_attention: MultiHeadAttention,
    /// Layer norm 1
    pub layer_norm1: LayerNorm,
    /// MLP
    pub mlp: Blip2MLP,
    /// Layer norm 2
    pub layer_norm2: LayerNorm,
}

impl Blip2VisionTransformerLayer {
    /// Create new layer
    pub fn new(config: &Blip2VisionConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let attention_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_dropout as f32,
            bias: true,
            max_seq_len: None,
        };

        let self_attention = MultiHeadAttention::new(
            attention_config.hidden_size,
            attention_config.num_heads,
            attention_config.dropout_prob,
            attention_config.bias,
        )?;
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let mlp = Blip2MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
        )?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            self_attention,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self attention with residual connection
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states.clone())?;
        let attention_output = self.self_attention.forward_attention(
            &hidden_states,
            &hidden_states,
            &hidden_states,
            None,
            false, // bidirectional attention
        )?;
        let hidden_states = residual.add(&attention_output)?;

        // MLP with residual connection
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// Q-Former model for BLIP-2
#[derive(Debug, Clone)]
pub struct Blip2QFormerModel {
    /// Configuration
    pub config: Blip2QFormerConfig,
    /// Embeddings
    pub embeddings: Blip2QFormerEmbeddings,
    /// Encoder layers
    pub encoder_layers: Vec<Blip2QFormerLayer>,
    /// Pooler
    pub pooler: Linear,
}

impl Blip2QFormerModel {
    /// Create new Q-Former model
    pub fn new(config: Blip2QFormerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let embeddings = Blip2QFormerEmbeddings::new(&config)?;

        let mut encoder_layers = Vec::new();
        for layer_idx in 0..config.num_hidden_layers {
            let has_cross_attention = layer_idx % config.cross_attention_frequency == 0;
            encoder_layers.push(Blip2QFormerLayer::new(&config, has_cross_attention)?);
        }

        let pooler = Linear::new(config.hidden_size, config.hidden_size, true);

        Ok(Self {
            config,
            embeddings,
            encoder_layers,
            pooler,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        query_embeds: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Blip2QFormerOutput, Box<dyn std::error::Error>> {
        // Get embeddings
        let mut hidden_states = self.embeddings.forward(input_ids)?;

        // Concatenate query embeddings if provided
        if let Some(query_embeds) = query_embeds {
            hidden_states = Tensor::concat(&[query_embeds.clone(), hidden_states], 1)?;
        }

        if let Some(_enc_hidden) = encoder_hidden_states {}

        // Apply encoder layers
        for layer in self.encoder_layers.iter() {
            hidden_states = layer.forward(&hidden_states, encoder_hidden_states, attention_mask)?;
        }

        // Pooling - select(1, 0) already reduces dimensions, no squeeze needed
        let pooler_output = self.pooler.forward(hidden_states.select(1, 0)?)?;
        let pooler_output = pooler_output.tanh()?;

        // Create logits (placeholder for language modeling head)
        let logits = Linear::new(self.config.hidden_size, self.config.vocab_size, false)
            .forward(hidden_states.clone())?;

        Ok(Blip2QFormerOutput {
            last_hidden_state: hidden_states,
            pooler_output,
            logits,
        })
    }
}

/// Q-Former embeddings
#[derive(Debug, Clone)]
pub struct Blip2QFormerEmbeddings {
    /// Word embeddings
    pub word_embeddings: Embedding,
    /// Position embeddings
    pub position_embeddings: Embedding,
    /// Token type embeddings
    pub token_type_embeddings: Embedding,
    /// Layer norm
    pub layer_norm: LayerNorm,
    /// Dropout
    pub dropout: f64,
}

impl Blip2QFormerEmbeddings {
    /// Create new embeddings
    pub fn new(config: &Blip2QFormerConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let word_embeddings = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let position_embeddings =
            Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;
        let token_type_embeddings =
            Embedding::new(config.type_vocab_size, config.hidden_size, None)?;
        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
            dropout: config.hidden_dropout_prob,
        })
    }

    /// Forward pass
    pub fn forward(&self, input_ids: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        let seq_length = input_ids.shape()[1];
        let batch_size = input_ids.shape()[0];

        // Word embeddings
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let word_embeds = self.word_embeddings.forward(input_ids_vec)?;
        let word_embeds = if word_embeds.shape().len() == 2
            && word_embeds.shape()[0] == batch_size * seq_length
        {
            // Reshape from [batch*seq, hidden] to [batch, seq, hidden]
            word_embeds.reshape(&[batch_size, seq_length, word_embeds.shape()[1]])?
        } else {
            word_embeds
        };

        // Position embeddings
        let position_ids = Tensor::range(0, seq_length as i64, DType::I64)?
            .unsqueeze(0)?
            .broadcast_to(&[batch_size, seq_length])?;
        let position_ids_vec: Vec<u32> =
            position_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let position_embeds = self.position_embeddings.forward(position_ids_vec)?;
        let position_embeds = if position_embeds.shape().len() == 2
            && position_embeds.shape()[0] == batch_size * seq_length
        {
            // Reshape from [batch*seq, hidden] to [batch, seq, hidden]
            position_embeds.reshape(&[batch_size, seq_length, position_embeds.shape()[1]])?
        } else {
            position_embeds
        };

        // Token type embeddings (default to 0)
        let token_type_ids = Tensor::zeros(&[batch_size, seq_length])?;
        let token_type_ids_vec: Vec<u32> =
            token_type_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let token_type_embeds = self.token_type_embeddings.forward(token_type_ids_vec)?;
        let token_type_embeds = if token_type_embeds.shape().len() == 2
            && token_type_embeds.shape()[0] == batch_size * seq_length
        {
            // Reshape from [batch*seq, hidden] to [batch, seq, hidden]
            token_type_embeds.reshape(&[batch_size, seq_length, token_type_embeds.shape()[1]])?
        } else {
            token_type_embeds
        };

        // Sum embeddings
        let embeddings = word_embeds.add(&position_embeds)?.add(&token_type_embeds)?;

        // Layer norm and dropout
        let embeddings = self.layer_norm.forward(embeddings)?;

        // Apply dropout (placeholder - would need proper dropout implementation)
        Ok(embeddings)
    }
}

/// Q-Former layer
#[derive(Debug, Clone)]
pub struct Blip2QFormerLayer {
    /// Self attention
    pub self_attention: MultiHeadAttention,
    /// Cross attention (optional)
    pub cross_attention: Option<MultiHeadAttention>,
    /// Projection for encoder hidden states (vision 1408 -> qformer 768)
    pub encoder_projection: Option<Linear>,
    /// Layer norm 1
    pub layer_norm1: LayerNorm,
    /// Layer norm 2
    pub layer_norm2: LayerNorm,
    /// Layer norm 3 (for cross attention)
    pub layer_norm3: Option<LayerNorm>,
    /// MLP
    pub mlp: Blip2MLP,
}

impl Blip2QFormerLayer {
    /// Create new layer
    pub fn new(
        config: &Blip2QFormerConfig,
        has_cross_attention: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let attention_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_probs_dropout_prob as f32,
            bias: true,
            max_seq_len: None,
        };

        let self_attention = MultiHeadAttention::new(
            attention_config.hidden_size,
            attention_config.num_heads,
            attention_config.dropout_prob,
            attention_config.bias,
        )?;
        let (cross_attention, encoder_projection) = if has_cross_attention {
            (
                Some(MultiHeadAttention::new(
                    attention_config.hidden_size,
                    attention_config.num_heads,
                    attention_config.dropout_prob,
                    attention_config.bias,
                )?),
                // Project vision encoder output (1408) to Q-Former hidden size (768)
                Some(Linear::new(1408, config.hidden_size, false)),
            )
        } else {
            (None, None)
        };

        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let layer_norm3 = if has_cross_attention {
            Some(LayerNorm::new(
                vec![config.hidden_size],
                config.layer_norm_eps as f32,
            )?)
        } else {
            None
        };

        let mlp = Blip2MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
        )?;

        Ok(Self {
            self_attention,
            cross_attention,
            encoder_projection,
            layer_norm1,
            layer_norm2,
            layer_norm3,
            mlp,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self attention
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states.clone())?;
        let attention_output = self.self_attention.forward_attention(
            &hidden_states,
            &hidden_states,
            &hidden_states,
            attention_mask,
            false, // bidirectional attention
        )?;
        let hidden_states = residual.add(&attention_output)?;

        // Cross attention
        let hidden_states = if let (
            Some(cross_attention),
            Some(encoder_hidden_states),
            Some(layer_norm3),
            Some(encoder_proj),
        ) = (
            &self.cross_attention,
            encoder_hidden_states,
            &self.layer_norm3,
            &self.encoder_projection,
        ) {
            let residual = hidden_states.clone();
            let hidden_states_norm = layer_norm3.forward(hidden_states.clone())?;

            // Project encoder hidden states to match Q-Former dimension
            let projected_encoder = encoder_proj.forward(encoder_hidden_states.clone())?;

            let cross_attention_output = cross_attention.forward_attention(
                &hidden_states_norm,
                &projected_encoder,
                &projected_encoder,
                None,
                false, // bidirectional cross-attention
            )?;
            residual.add(&cross_attention_output)?
        } else {
            hidden_states
        };

        // MLP
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// Multi-layer perceptron
#[derive(Debug, Clone)]
pub struct Blip2MLP {
    /// Linear 1
    pub linear1: Linear,
    /// Linear 2
    pub linear2: Linear,
    /// Activation
    pub activation: ActivationType,
}

impl Blip2MLP {
    /// Create new MLP
    pub fn new(
        hidden_size: usize,
        intermediate_size: usize,
        activation: &str,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let linear1 = Linear::new(hidden_size, intermediate_size, true);
        let linear2 = Linear::new(intermediate_size, hidden_size, true);
        let activation = match activation {
            "gelu" => ActivationType::GELU,
            "relu" => ActivationType::ReLU,
            "silu" | "swish" => ActivationType::SiLU,
            "tanh" => ActivationType::Tanh,
            "sigmoid" => ActivationType::Sigmoid,
            _ => ActivationType::GELU, // default
        };

        Ok(Self {
            linear1,
            linear2,
            activation,
        })
    }

    /// Forward pass
    pub fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Box<dyn std::error::Error>> {
        use trustformers_core::ops::activations::*;

        let hidden_states = self.linear1.forward(hidden_states.clone())?;
        let hidden_states = match self.activation {
            ActivationType::GELU => gelu(&hidden_states)?,
            ActivationType::ReLU => relu(&hidden_states)?,
            ActivationType::SiLU => silu(&hidden_states)?,
            ActivationType::Tanh => tanh(&hidden_states)?,
            ActivationType::Sigmoid => sigmoid(&hidden_states)?,
        };
        let hidden_states = self.linear2.forward(hidden_states)?;
        Ok(hidden_states)
    }
}

/// Language model trait
pub trait LanguageModel: Send + Sync {
    /// Forward pass
    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<LanguageModelOutput, Box<dyn std::error::Error>>;
}

/// OPT language model for BLIP-2
#[derive(Debug, Clone)]
pub struct Blip2OptLanguageModel {
    /// Configuration
    pub config: Blip2TextConfig,
    /// Embeddings
    pub embeddings: Embedding,
    /// Decoder layers
    pub layers: Vec<Blip2OptLayer>,
    /// Layer norm
    pub layer_norm: LayerNorm,
    /// LM head
    pub lm_head: Linear,
}

impl Blip2OptLanguageModel {
    /// Create new OPT language model
    pub fn new(config: Blip2TextConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut layers = Vec::new();
        for _ in 0..config.num_hidden_layers {
            layers.push(Blip2OptLayer::new(&config)?);
        }

        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            config,
            embeddings,
            layers,
            layer_norm,
            lm_head,
        })
    }
}

impl LanguageModel for Blip2OptLanguageModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<LanguageModelOutput, Box<dyn std::error::Error>> {
        let batch_size = input_ids.shape()[0];
        let seq_length = input_ids.shape()[1];

        // Get embeddings
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let hidden_states = self.embeddings.forward(input_ids_vec)?;

        // Reshape from [batch*seq, hidden] to [batch, seq, hidden]
        let mut hidden_states = if hidden_states.shape().len() == 2
            && hidden_states.shape()[0] == batch_size * seq_length
        {
            hidden_states.reshape(&[batch_size, seq_length, hidden_states.shape()[1]])?
        } else {
            hidden_states
        };

        // Prepend encoder hidden states if provided
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            hidden_states = Tensor::concat(&[encoder_hidden_states.clone(), hidden_states], 1)?;
        }

        // Apply decoder layers
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        // Layer norm
        let hidden_states = self.layer_norm.forward(hidden_states)?;

        // Language modeling head
        let logits = self.lm_head.forward(hidden_states.clone())?;

        // Calculate loss if labels provided
        let loss = if let Some(_labels) = labels {
            // Cross entropy loss (placeholder)
            Some(Tensor::scalar(1.0)?)
        } else {
            None
        };

        Ok(LanguageModelOutput {
            loss,
            logits,
            hidden_states,
        })
    }
}

/// OPT decoder layer
#[derive(Debug, Clone)]
pub struct Blip2OptLayer {
    /// Self attention
    pub self_attention: MultiHeadAttention,
    /// Layer norm 1
    pub layer_norm1: LayerNorm,
    /// MLP
    pub mlp: Blip2MLP,
    /// Layer norm 2
    pub layer_norm2: LayerNorm,
}

impl Blip2OptLayer {
    /// Create new layer
    pub fn new(config: &Blip2TextConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let attention_config = AttentionConfig {
            hidden_size: config.hidden_size,
            num_heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
            dropout_prob: config.attention_dropout as f32,
            bias: true,
            max_seq_len: None,
        };

        let self_attention = MultiHeadAttention::new(
            attention_config.hidden_size,
            attention_config.num_heads,
            attention_config.dropout_prob,
            attention_config.bias,
        )?;
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let mlp = Blip2MLP::new(
            config.hidden_size,
            config.intermediate_size,
            &config.hidden_act,
        )?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;

        Ok(Self {
            self_attention,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }

    /// Forward pass
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Box<dyn std::error::Error>> {
        // Self attention
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm1.forward(hidden_states.clone())?;
        let attention_output = self.self_attention.forward_attention(
            &hidden_states,
            &hidden_states,
            &hidden_states,
            attention_mask,
            false, // bidirectional attention
        )?;
        let hidden_states = residual.add(&attention_output)?;

        // MLP
        let residual = hidden_states.clone();
        let hidden_states = self.layer_norm2.forward(hidden_states)?;
        let mlp_output = self.mlp.forward(&hidden_states)?;
        let hidden_states = residual.add(&mlp_output)?;

        Ok(hidden_states)
    }
}

/// T5 language model for BLIP-2
#[derive(Debug, Clone)]
pub struct Blip2T5LanguageModel {
    /// Configuration
    pub config: Blip2TextConfig,
    /// Embeddings
    pub embeddings: Embedding,
    /// Encoder layers
    pub encoder_layers: Vec<Blip2OptLayer>,
    /// Decoder layers
    pub decoder_layers: Vec<Blip2OptLayer>,
    /// Layer norm
    pub layer_norm: LayerNorm,
    /// LM head
    pub lm_head: Linear,
}

impl Blip2T5LanguageModel {
    /// Create new T5 language model
    pub fn new(config: Blip2TextConfig) -> Result<Self, Box<dyn std::error::Error>> {
        let embeddings = Embedding::new(config.vocab_size, config.hidden_size, None)?;

        let mut encoder_layers = Vec::new();
        let mut decoder_layers = Vec::new();

        for _ in 0..config.num_hidden_layers {
            encoder_layers.push(Blip2OptLayer::new(&config)?);
            decoder_layers.push(Blip2OptLayer::new(&config)?);
        }

        let layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps as f32)?;
        let lm_head = Linear::new(config.hidden_size, config.vocab_size, false);

        Ok(Self {
            config,
            embeddings,
            encoder_layers,
            decoder_layers,
            layer_norm,
            lm_head,
        })
    }
}

impl LanguageModel for Blip2T5LanguageModel {
    fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        labels: Option<&Tensor>,
    ) -> Result<LanguageModelOutput, Box<dyn std::error::Error>> {
        // Get embeddings
        let input_ids_vec: Vec<u32> =
            input_ids.to_vec_f32()?.into_iter().map(|x| x as u32).collect();
        let mut hidden_states = self.embeddings.forward(input_ids_vec)?;

        // Encoder
        for layer in &self.encoder_layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        // Decoder with cross-attention to encoder states
        if let Some(encoder_hidden_states) = encoder_hidden_states {
            hidden_states = Tensor::concat(&[encoder_hidden_states.clone(), hidden_states], 1)?;
        }

        for layer in &self.decoder_layers {
            hidden_states = layer.forward(&hidden_states, attention_mask)?;
        }

        // Layer norm
        let hidden_states = self.layer_norm.forward(hidden_states)?;

        // Language modeling head
        let logits = self.lm_head.forward(hidden_states.clone())?;

        // Calculate loss if labels provided
        let loss = if let Some(_labels) = labels {
            // Cross entropy loss (placeholder)
            Some(Tensor::scalar(1.0)?)
        } else {
            None
        };

        Ok(LanguageModelOutput {
            loss,
            logits,
            hidden_states,
        })
    }
}

/// Output structures
#[derive(Debug, Clone)]
pub struct Blip2Output {
    /// Last hidden state
    pub last_hidden_state: Tensor,
    /// Pooler output
    pub pooler_output: Tensor,
    /// Image features
    pub image_features: Option<Tensor>,
    /// Text features
    pub text_features: Tensor,
    /// Logits
    pub logits: Tensor,
}

#[derive(Debug, Clone)]
pub struct Blip2ConditionalGenerationOutput {
    /// Loss
    pub loss: Option<Tensor>,
    /// Logits
    pub logits: Tensor,
    /// Hidden states
    pub hidden_states: Tensor,
    /// Image features
    pub image_features: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct Blip2QFormerOutput {
    /// Last hidden state
    pub last_hidden_state: Tensor,
    /// Pooler output
    pub pooler_output: Tensor,
    /// Logits
    pub logits: Tensor,
}

#[derive(Debug, Clone)]
pub struct LanguageModelOutput {
    /// Loss
    pub loss: Option<Tensor>,
    /// Logits
    pub logits: Tensor,
    /// Hidden states
    pub hidden_states: Tensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blip2_model_creation() {
        let config = Blip2Config::default();
        let model = Blip2Model::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_vision_model() {
        let config = Blip2VisionConfig::default();
        let model = Blip2VisionModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_qformer_model() {
        let config = Blip2QFormerConfig::default();
        let model = Blip2QFormerModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_patch_embedding() {
        let config = Blip2VisionConfig::default();
        let embedding = Blip2PatchEmbedding::new(&config);
        assert!(embedding.is_ok());
    }

    #[test]
    fn test_blip2_mlp() {
        let mlp = Blip2MLP::new(768, 3072, "gelu");
        assert!(mlp.is_ok());
    }

    #[test]
    fn test_blip2_opt_language_model() {
        let config = Blip2TextConfig::opt_2_7b();
        let model = Blip2OptLanguageModel::new(config);
        assert!(model.is_ok());
    }

    #[test]
    fn test_blip2_t5_language_model() {
        let config = Blip2TextConfig::flan_t5_xl();
        let model = Blip2T5LanguageModel::new(config);
        assert!(model.is_ok());
    }
}
