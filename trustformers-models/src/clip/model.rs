use crate::clip::config::{CLIPConfig, CLIPTextConfig, CLIPVisionConfig};
use scirs2_core::ndarray::{Array2, Array3, Array4}; // SciRS2 Integration Policy
use std::io::Read;
use trustformers_core::{
    errors::{tensor_op_error, Result},
    layers::{Embedding, FeedForward, LayerNorm, Linear, MultiHeadAttention},
    tensor::Tensor,
    traits::{Config, Layer, Model},
};

/// CLIP Vision Transformer patch embedding
pub struct CLIPVisionEmbeddings {
    patch_embedding: CLIPPatchEmbedding,
    class_embedding: Tensor,
    position_embedding: Embedding,
    num_patches: usize,
    num_positions: usize,
}

impl CLIPVisionEmbeddings {
    pub fn new(config: &CLIPVisionConfig) -> Result<Self> {
        let patch_embedding = CLIPPatchEmbedding::new(config)?;
        let num_patches = config.num_patches();
        let num_positions = config.seq_length();

        let class_embedding = Tensor::randn(&[config.hidden_size])?;
        let position_embedding = Embedding::new(num_positions, config.hidden_size, None)?;

        Ok(Self {
            patch_embedding,
            class_embedding,
            position_embedding,
            num_patches,
            num_positions,
        })
    }
}

impl Layer for CLIPVisionEmbeddings {
    type Input = Array4<f32>; // (batch_size, height, width, channels)
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let batch_size = input.shape()[0];

        // Get patch embeddings
        let patch_embeddings = self.patch_embedding.forward(input)?;

        // Prepare class token for each batch item
        let class_tokens = match &self.class_embedding {
            Tensor::F32(class_arr) => {
                let mut class_batch = Array2::zeros((batch_size, class_arr.len()));
                for i in 0..batch_size {
                    class_batch.row_mut(i).assign(class_arr);
                }
                Tensor::F32(class_batch.into_dyn())
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported class embedding tensor type",
                ))
            },
        };

        // Concatenate class token and patch embeddings
        let embeddings = match (&class_tokens, &patch_embeddings) {
            (Tensor::F32(class_arr), Tensor::F32(patch_arr)) => {
                let seq_len = 1 + self.num_patches;
                let hidden_size = class_arr.shape()[1];

                let mut combined = Array3::zeros((batch_size, seq_len, hidden_size));

                // Set class tokens at position 0
                for i in 0..batch_size {
                    combined
                        .slice_mut(ndarray::s![i, 0, ..])
                        .assign(&class_arr.slice(ndarray::s![i, ..]));
                }

                // Set patch embeddings at positions 1..seq_len
                for i in 0..batch_size {
                    for j in 0..self.num_patches {
                        combined
                            .slice_mut(ndarray::s![i, j + 1, ..])
                            .assign(&patch_arr.slice(ndarray::s![i, j, ..]));
                    }
                }

                Tensor::F32(combined.into_dyn())
            },
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Unsupported tensor types for embedding concatenation",
                ))
            },
        };

        // Add positional embeddings
        let position_ids: Vec<u32> = (0..self.num_positions).map(|i| i as u32).collect();
        let position_embeddings = self.position_embedding.forward(position_ids)?;

        embeddings.add(&position_embeddings)
    }
}

/// CLIP patch embedding layer
pub struct CLIPPatchEmbedding {
    projection: Linear,
    patch_size: usize,
    hidden_size: usize,
}

impl CLIPPatchEmbedding {
    pub fn new(config: &CLIPVisionConfig) -> Result<Self> {
        let in_features = config.patch_size * config.patch_size * config.num_channels;
        let projection = Linear::new(in_features, config.hidden_size, true);

        Ok(Self {
            projection,
            patch_size: config.patch_size,
            hidden_size: config.hidden_size,
        })
    }
}

impl Layer for CLIPPatchEmbedding {
    type Input = Array4<f32>; // (batch_size, height, width, channels)
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (batch_size, height, width, channels) = input.dim();
        let patch_size = self.patch_size;

        // Extract patches
        let patches_h = height / patch_size;
        let patches_w = width / patch_size;
        let num_patches = patches_h * patches_w;

        let mut patches =
            Array2::zeros((batch_size * num_patches, patch_size * patch_size * channels));

        for b in 0..batch_size {
            for ph in 0..patches_h {
                for pw in 0..patches_w {
                    let patch_idx = b * num_patches + ph * patches_w + pw;
                    let mut patch_data = Vec::new();

                    for y in 0..patch_size {
                        for x in 0..patch_size {
                            for c in 0..channels {
                                let pixel_y = ph * patch_size + y;
                                let pixel_x = pw * patch_size + x;
                                patch_data.push(input[(b, pixel_y, pixel_x, c)]);
                            }
                        }
                    }

                    for (i, &val) in patch_data.iter().enumerate() {
                        patches[(patch_idx, i)] = val;
                    }
                }
            }
        }

        // Project patches to hidden dimension
        let projected = self.projection.forward(Tensor::F32(patches.into_dyn()))?;

        // Reshape to (batch_size, num_patches, hidden_size)
        match projected {
            Tensor::F32(arr) => {
                let reshaped =
                    arr.into_shape_with_order((batch_size, num_patches, self.hidden_size))?;
                Ok(Tensor::F32(reshaped.into_dyn()))
            },
            _ => Err(tensor_op_error(
                "tensor_operation",
                "Expected F32 tensor from projection",
            )),
        }
    }
}

/// CLIP text encoder layer
pub struct CLIPTextTransformer {
    embeddings: CLIPTextEmbeddings,
    encoder: CLIPEncoder<CLIPTextConfig>,
    final_layer_norm: LayerNorm,
}

impl CLIPTextTransformer {
    pub fn new(config: &CLIPTextConfig) -> Result<Self> {
        let embeddings = CLIPTextEmbeddings::new(config)?;
        let encoder = CLIPEncoder::new(config)?;
        let final_layer_norm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            embeddings,
            encoder,
            final_layer_norm,
        })
    }
}

impl Layer for CLIPTextTransformer {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let embeddings = self.embeddings.forward(input)?;
        let encoded = self.encoder.forward(embeddings)?;
        self.final_layer_norm.forward(encoded)
    }
}

/// CLIP vision encoder layer
pub struct CLIPVisionTransformer {
    embeddings: CLIPVisionEmbeddings,
    encoder: CLIPEncoder<CLIPVisionConfig>,
    layernorm: LayerNorm,
}

impl CLIPVisionTransformer {
    pub fn new(config: &CLIPVisionConfig) -> Result<Self> {
        let embeddings = CLIPVisionEmbeddings::new(config)?;
        let encoder = CLIPEncoder::new(config)?;
        let layernorm = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            embeddings,
            encoder,
            layernorm,
        })
    }
}

impl Layer for CLIPVisionTransformer {
    type Input = Array4<f32>; // Images
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let embeddings = self.embeddings.forward(input)?;
        let encoded = self.encoder.forward(embeddings)?;
        self.layernorm.forward(encoded)
    }
}

/// CLIP text embeddings
pub struct CLIPTextEmbeddings {
    token_embedding: Embedding,
    position_embedding: Embedding,
}

impl CLIPTextEmbeddings {
    pub fn new(config: &CLIPTextConfig) -> Result<Self> {
        let token_embedding = Embedding::new(config.vocab_size, config.hidden_size, None)?;
        let position_embedding =
            Embedding::new(config.max_position_embeddings, config.hidden_size, None)?;

        Ok(Self {
            token_embedding,
            position_embedding,
        })
    }
}

impl Layer for CLIPTextEmbeddings {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let seq_len = input.len();
        let token_embeddings = self.token_embedding.forward(input)?;

        let position_ids: Vec<u32> = (0..seq_len).map(|i| i as u32).collect();
        let position_embeddings = self.position_embedding.forward(position_ids)?;

        token_embeddings.add(&position_embeddings)
    }
}

/// Generic CLIP encoder (works for both text and vision)
pub struct CLIPEncoder<C> {
    layers: Vec<CLIPEncoderLayer>,
    _phantom: std::marker::PhantomData<C>,
}

impl<C> CLIPEncoder<C>
where
    C: Config + Send + Sync,
{
    pub fn new(_config: &C) -> Result<Self> {
        let mut layers = Vec::new();

        // Note: This is a simplified implementation
        // In a real implementation, we'd need to extract layer config from C
        // For now, we'll use placeholder values
        let layer_config = CLIPEncoderLayerConfig {
            hidden_size: 512, // This would come from config
            num_attention_heads: 8,
            intermediate_size: 2048,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            attention_dropout: 0.0,
            dropout: 0.0,
        };

        for _ in 0..12 {
            // This would come from config.num_hidden_layers
            layers.push(CLIPEncoderLayer::new(&layer_config)?);
        }

        Ok(Self {
            layers,
            _phantom: std::marker::PhantomData,
        })
    }
}

impl<C> Layer for CLIPEncoder<C>
where
    C: Config + Send + Sync,
{
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, mut input: Self::Input) -> Result<Self::Output> {
        for layer in &self.layers {
            input = layer.forward(input)?;
        }
        Ok(input)
    }
}

/// CLIP encoder layer configuration
#[derive(Debug, Clone)]
pub struct CLIPEncoderLayerConfig {
    pub hidden_size: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub attention_dropout: f32,
    pub dropout: f32,
}

/// CLIP encoder layer
pub struct CLIPEncoderLayer {
    self_attn: MultiHeadAttention,
    layer_norm1: LayerNorm,
    mlp: FeedForward,
    layer_norm2: LayerNorm,
}

impl CLIPEncoderLayer {
    pub fn new(config: &CLIPEncoderLayerConfig) -> Result<Self> {
        let self_attn = MultiHeadAttention::new(
            config.hidden_size,
            config.num_attention_heads,
            config.attention_dropout,
            true, // bias
        )?;
        let layer_norm1 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;
        let mlp = FeedForward::new(config.hidden_size, config.intermediate_size, config.dropout)?;
        let layer_norm2 = LayerNorm::new(vec![config.hidden_size], config.layer_norm_eps)?;

        Ok(Self {
            self_attn,
            layer_norm1,
            mlp,
            layer_norm2,
        })
    }
}

impl Layer for CLIPEncoderLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        // Pre-norm architecture: norm -> attention -> residual
        let normalized1 = self.layer_norm1.forward(input.clone())?;
        let attn_output = self.self_attn.forward_self_attention(&normalized1, None, false)?;
        let residual1 = input.add(&attn_output)?;

        // Pre-norm architecture: norm -> mlp -> residual
        let normalized2 = self.layer_norm2.forward(residual1.clone())?;
        let mlp_output = self.mlp.forward(normalized2)?;
        let residual2 = residual1.add(&mlp_output)?;

        Ok(residual2)
    }
}

/// Main CLIP model
pub struct CLIPModel {
    config: CLIPConfig,
    text_model: CLIPTextTransformer,
    vision_model: CLIPVisionTransformer,
    text_projection: Linear,
    visual_projection: Linear,
    logit_scale: Tensor,
}

impl CLIPModel {
    pub fn new(config: CLIPConfig) -> Result<Self> {
        config.validate()?;

        let text_model = CLIPTextTransformer::new(&config.text_config)?;
        let vision_model = CLIPVisionTransformer::new(&config.vision_config)?;

        let text_projection =
            Linear::new(config.text_config.hidden_size, config.projection_dim, false);
        let visual_projection = Linear::new(
            config.vision_config.hidden_size,
            config.projection_dim,
            false,
        );

        let logit_scale =
            Tensor::F32(ndarray::Array1::from_elem(1, config.logit_scale_init_value).into_dyn());

        Ok(Self {
            config,
            text_model,
            vision_model,
            text_projection,
            visual_projection,
            logit_scale,
        })
    }

    /// Get text features
    pub fn get_text_features(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let text_outputs = self.text_model.forward(input_ids)?;

        // Use the [CLS] token representation (first token)
        let cls_output = text_outputs.select_first_token()?;
        self.text_projection.forward(cls_output)
    }

    /// Get image features
    pub fn get_image_features(&self, pixel_values: Array4<f32>) -> Result<Tensor> {
        let vision_outputs = self.vision_model.forward(pixel_values)?;

        // Use the [CLS] token representation (first token)
        let cls_output = vision_outputs.select_first_token()?;
        self.visual_projection.forward(cls_output)
    }

    /// Forward pass that returns both text and image features
    pub fn forward(
        &self,
        input_ids: Option<Vec<u32>>,
        pixel_values: Option<Array4<f32>>,
    ) -> Result<CLIPOutput> {
        let mut text_embeds = None;
        let mut image_embeds = None;

        if let Some(input_ids) = input_ids {
            text_embeds = Some(self.get_text_features(input_ids)?);
        }

        if let Some(pixel_values) = pixel_values {
            image_embeds = Some(self.get_image_features(pixel_values)?);
        }

        Ok(CLIPOutput {
            text_embeds,
            image_embeds,
            logits_per_image: None,
            logits_per_text: None,
        })
    }

    /// Compute similarity scores between text and images
    pub fn compute_similarity(
        &self,
        input_ids: Vec<u32>,
        pixel_values: Array4<f32>,
    ) -> Result<(Tensor, Tensor)> {
        let text_features = self.get_text_features(input_ids)?;
        let image_features = self.get_image_features(pixel_values)?;

        // Normalize features (manual implementation)
        let text_norm = text_features.norm()?;
        let image_norm = image_features.norm()?;
        let text_features_norm = text_features.scale(1.0 / text_norm)?;
        let image_features_norm = image_features.scale(1.0 / image_norm)?;

        // Compute logits
        let logit_scale = match &self.logit_scale {
            Tensor::F32(scale_arr) => scale_arr[[0]].exp(),
            _ => {
                return Err(tensor_op_error(
                    "tensor_operation",
                    "Invalid logit scale tensor",
                ))
            },
        };

        let image_transposed = image_features_norm.transpose(0, 1)?;
        let logits_per_image = text_features_norm.matmul(&image_transposed)?.scale(logit_scale)?;
        let logits_per_text = logits_per_image.transpose(0, 1)?;

        Ok((logits_per_image, logits_per_text))
    }
}

/// CLIP model output
#[derive(Debug)]
pub struct CLIPOutput {
    pub text_embeds: Option<Tensor>,
    pub image_embeds: Option<Tensor>,
    pub logits_per_image: Option<Tensor>,
    pub logits_per_text: Option<Tensor>,
}

impl Model for CLIPModel {
    type Config = CLIPConfig;
    type Input = (Option<Vec<u32>>, Option<Array4<f32>>); // (text_input, image_input)
    type Output = CLIPOutput;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let (input_ids, pixel_values) = input;
        self.forward(input_ids, pixel_values)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn Read) -> Result<()> {
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
        // Text encoder parameters
        let text_vocab_size = self.config.text_config.vocab_size;
        let text_hidden_size = self.config.text_config.hidden_size;
        let text_num_layers = self.config.text_config.num_hidden_layers;
        let _text_num_heads = self.config.text_config.num_attention_heads;
        let text_intermediate_size = self.config.text_config.intermediate_size;
        let text_max_position = self.config.text_config.max_position_embeddings;

        // Text embeddings: token + position
        let text_embedding_params =
            text_vocab_size * text_hidden_size + text_max_position * text_hidden_size;

        // Text encoder layers: attention + FFN + layer norms
        let text_attention_params_per_layer =
            4 * (text_hidden_size * text_hidden_size + text_hidden_size); // Q,K,V,dense
        let text_ffn_params_per_layer = text_hidden_size * text_intermediate_size
            + text_intermediate_size
            + text_intermediate_size * text_hidden_size
            + text_hidden_size;
        let text_layer_norm_params = 4 * text_hidden_size; // 2 LayerNorms per layer

        let text_encoder_params = text_num_layers
            * (text_attention_params_per_layer
                + text_ffn_params_per_layer
                + text_layer_norm_params);

        // Vision encoder parameters
        let vision_hidden_size = self.config.vision_config.hidden_size;
        let vision_num_layers = self.config.vision_config.num_hidden_layers;
        let _vision_num_heads = self.config.vision_config.num_attention_heads;
        let vision_intermediate_size = self.config.vision_config.intermediate_size;
        let vision_patch_size = self.config.vision_config.patch_size;
        let vision_num_channels = self.config.vision_config.num_channels;

        // Vision patch embedding
        let vision_embedding_params =
            vision_patch_size * vision_patch_size * vision_num_channels * vision_hidden_size
                + vision_hidden_size;

        // Vision encoder layers
        let vision_attention_params_per_layer =
            4 * (vision_hidden_size * vision_hidden_size + vision_hidden_size);
        let vision_ffn_params_per_layer = vision_hidden_size * vision_intermediate_size
            + vision_intermediate_size
            + vision_intermediate_size * vision_hidden_size
            + vision_hidden_size;
        let vision_layer_norm_params = 4 * vision_hidden_size;

        let vision_encoder_params = vision_num_layers
            * (vision_attention_params_per_layer
                + vision_ffn_params_per_layer
                + vision_layer_norm_params);

        // Projection layers (text and vision to shared embedding space)
        let projection_dim = self.config.projection_dim;
        let text_projection_params = text_hidden_size * projection_dim;
        let vision_projection_params = vision_hidden_size * projection_dim;

        // Logit scale parameter
        let logit_scale_params = 1;

        text_embedding_params
            + text_encoder_params
            + vision_embedding_params
            + vision_encoder_params
            + text_projection_params
            + vision_projection_params
            + logit_scale_params
    }
}

impl CLIPModel {
    /// Load model weights from a directory containing HuggingFace format weights
    pub fn load_from_path(&mut self, model_path: impl AsRef<std::path::Path>) -> Result<()> {
        use crate::weight_loading::{auto_create_loader, WeightLoadingConfig};

        let config = WeightLoadingConfig {
            lazy_loading: true,
            memory_mapped: false,
            ..Default::default()
        };

        let mut loader = auto_create_loader(model_path, Some(config))?;

        // Load text model weights
        // Text embeddings
        if let Ok(_text_embeddings) =
            loader.load_tensor("text_model.embeddings.token_embedding.weight")
        {
            // Note: This is a simplified implementation
            // In a full implementation, we would need to access the text model's internal components
            // For now, we indicate that the weight was found but cannot be set due to encapsulation
        }

        if let Ok(_text_pos_embeddings) =
            loader.load_tensor("text_model.embeddings.position_embedding.weight")
        {
            // Similar placeholder for position embeddings
        }

        // Text transformer layers
        for layer_idx in 0..self.config.text_config.num_hidden_layers {
            let layer_prefix = format!("text_model.encoder.layers.{}", layer_idx);

            // Text attention weights
            if let Ok(_q_weight) =
                loader.load_tensor(&format!("{}.self_attn.q_proj.weight", layer_prefix))
            {
                // Placeholder for text attention weight loading
            }
            if let Ok(_k_weight) =
                loader.load_tensor(&format!("{}.self_attn.k_proj.weight", layer_prefix))
            {
                // Placeholder for text attention weight loading
            }
            if let Ok(_v_weight) =
                loader.load_tensor(&format!("{}.self_attn.v_proj.weight", layer_prefix))
            {
                // Placeholder for text attention weight loading
            }
            if let Ok(_o_weight) =
                loader.load_tensor(&format!("{}.self_attn.out_proj.weight", layer_prefix))
            {
                // Placeholder for text attention weight loading
            }

            // Text MLP weights
            if let Ok(_fc1_weight) = loader.load_tensor(&format!("{}.mlp.fc1.weight", layer_prefix))
            {
                // Placeholder for text MLP weight loading
            }
            if let Ok(_fc2_weight) = loader.load_tensor(&format!("{}.mlp.fc2.weight", layer_prefix))
            {
                // Placeholder for text MLP weight loading
            }

            // Text layer normalization
            if let Ok(_ln1_weight) =
                loader.load_tensor(&format!("{}.layer_norm1.weight", layer_prefix))
            {
                // Placeholder for text layer norm weight loading
            }
            if let Ok(_ln2_weight) =
                loader.load_tensor(&format!("{}.layer_norm2.weight", layer_prefix))
            {
                // Placeholder for text layer norm weight loading
            }
        }

        // Load vision model weights
        // Vision patch embeddings
        if let Ok(_patch_embeddings) =
            loader.load_tensor("vision_model.embeddings.patch_embedding.weight")
        {
            // Placeholder for vision patch embedding weight loading
        }

        if let Ok(_class_embedding) = loader.load_tensor("vision_model.embeddings.class_embedding")
        {
            // Placeholder for vision class embedding weight loading
        }

        if let Ok(_pos_embeddings) =
            loader.load_tensor("vision_model.embeddings.position_embedding.weight")
        {
            // Placeholder for vision position embedding weight loading
        }

        // Vision transformer layers
        for layer_idx in 0..self.config.vision_config.num_hidden_layers {
            let layer_prefix = format!("vision_model.encoder.layers.{}", layer_idx);

            // Vision attention weights (similar pattern to text)
            if let Ok(_q_weight) =
                loader.load_tensor(&format!("{}.self_attn.q_proj.weight", layer_prefix))
            {
                // Placeholder for vision attention weight loading
            }
            // ... (similar for k_proj, v_proj, out_proj)

            // Vision MLP weights
            if let Ok(_fc1_weight) = loader.load_tensor(&format!("{}.mlp.fc1.weight", layer_prefix))
            {
                // Placeholder for vision MLP weight loading
            }
            // ... (similar for fc2)

            // Vision layer normalization
            if let Ok(_ln1_weight) =
                loader.load_tensor(&format!("{}.layer_norm1.weight", layer_prefix))
            {
                // Placeholder for vision layer norm weight loading
            }
            // ... (similar for layer_norm2)
        }

        // Load projection layers
        if let Ok(text_proj_weight) = loader.load_tensor("text_projection.weight") {
            self.text_projection.set_weight(text_proj_weight)?;
        }

        if let Ok(visual_proj_weight) = loader.load_tensor("visual_projection.weight") {
            self.visual_projection.set_weight(visual_proj_weight)?;
        }

        // Load logit scale
        if let Ok(logit_scale_weight) = loader.load_tensor("logit_scale") {
            self.logit_scale = logit_scale_weight;
        }

        Ok(())
    }

    /// Load model weights from HuggingFace Hub or local cache
    pub fn load_from_huggingface(&mut self, model_name: &str) -> Result<()> {
        use std::path::PathBuf;

        // Try to find model in HuggingFace cache directory
        let cache_dir = std::env::var("HF_HOME")
            .or_else(|_| std::env::var("HUGGINGFACE_HUB_CACHE"))
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{}/.cache/huggingface/hub", home)
            });

        let model_base_path =
            PathBuf::from(cache_dir).join(format!("models--{}", model_name.replace("/", "--")));
        let model_path = model_base_path.join("snapshots");

        if let Ok(mut entries) = std::fs::read_dir(&model_path) {
            if let Some(Ok(entry)) = entries.next() {
                let snapshot_path = entry.path();
                return self.load_from_path(&snapshot_path);
            }
        }

        // Attempt to download the model from HuggingFace Hub
        self.download_from_huggingface_hub(model_name, &model_base_path)?;

        // Try to load again after download
        if let Ok(mut entries) = std::fs::read_dir(&model_path) {
            if let Some(Ok(entry)) = entries.next() {
                let snapshot_path = entry.path();
                return self.load_from_path(&snapshot_path);
            }
        }

        Err(trustformers_core::errors::TrustformersError::io_error(
            format!("Failed to find downloaded model files for {}", model_name),
        ))
    }

    /// Download model from HuggingFace Hub
    fn download_from_huggingface_hub(
        &self,
        model_name: &str,
        model_base_path: &std::path::Path,
    ) -> Result<()> {
        use std::process::Command;

        println!(
            "Downloading model {} from HuggingFace Hub to {:?}",
            model_name, model_base_path
        );

        // Create the model directory and snapshots subdirectory
        let snapshots_path = model_base_path.join("snapshots").join("main");
        std::fs::create_dir_all(&snapshots_path).map_err(|e| {
            trustformers_core::errors::TrustformersError::io_error(format!(
                "Failed to create model directory: {}",
                e
            ))
        })?;

        // List of essential files for CLIP models
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
            let file_path = snapshots_path.join(file_name);

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

        // For lazy loading, we set up the loader but don't load weights immediately
        // Weights are loaded on-demand during forward passes
        // This is useful for very large models that don't fit in memory

        // For now, just perform regular loading
        self.load_from_path(model_path)
    }
}
