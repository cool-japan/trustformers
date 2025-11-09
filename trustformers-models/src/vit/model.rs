use crate::vit::config::ViTConfig;
use scirs2_core::ndarray::{s, Array1, Array2, Array3, Array4, Axis}; // SciRS2 Integration Policy
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::layers::{
    attention::MultiHeadAttention, embedding::Embedding, feedforward::FeedForward,
    layernorm::LayerNorm, linear::Linear,
};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::{Config, Layer};

/// Patch embedding layer for Vision Transformer
#[derive(Debug, Clone)]
pub struct PatchEmbedding {
    pub projection: Linear,
    pub patch_size: usize,
    pub num_channels: usize,
    pub hidden_size: usize,
}

impl PatchEmbedding {
    pub fn new(config: &ViTConfig) -> Self {
        let input_size = config.patch_size * config.patch_size * config.num_channels;

        Self {
            projection: Linear::new(input_size, config.hidden_size, config.use_patch_bias),
            patch_size: config.patch_size,
            num_channels: config.num_channels,
            hidden_size: config.hidden_size,
        }
    }

    /// Convert image to patches and embed them
    /// Input: (batch_size, height, width, channels)
    /// Output: (batch_size, num_patches, hidden_size)
    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        let (batch_size, height, width, channels) = images.dim();

        if height % self.patch_size != 0 || width % self.patch_size != 0 {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Image size {}x{} is not divisible by patch size {}",
                height, width, self.patch_size
            )));
        }

        if channels != self.num_channels {
            return Err(TrustformersError::invalid_input_simple(format!(
                "Expected {} channels, got {}",
                self.num_channels, channels
            )));
        }

        let num_patches_h = height / self.patch_size;
        let num_patches_w = width / self.patch_size;
        let num_patches = num_patches_h * num_patches_w;

        // Extract patches
        let mut patches = Array3::zeros((
            batch_size,
            num_patches,
            self.patch_size * self.patch_size * channels,
        ));

        for b in 0..batch_size {
            let mut patch_idx = 0;
            for i in 0..num_patches_h {
                for j in 0..num_patches_w {
                    let start_h = i * self.patch_size;
                    let start_w = j * self.patch_size;

                    // Extract patch and flatten
                    let patch = images.slice(s![
                        b,
                        start_h..start_h + self.patch_size,
                        start_w..start_w + self.patch_size,
                        ..
                    ]);

                    // Flatten patch (patch_size * patch_size * channels)
                    let flattened: Array1<f32> = patch.iter().cloned().collect();
                    patches.slice_mut(s![b, patch_idx, ..]).assign(&flattened);
                    patch_idx += 1;
                }
            }
        }

        // Project patches to hidden dimension
        let patches_tensor = Tensor::F32(patches.into_dyn());
        match self.projection.forward(patches_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor".to_string(),
            )),
        }
    }
}

/// Vision Transformer embeddings (patches + position + class token)
#[derive(Debug, Clone)]
pub struct ViTEmbeddings {
    pub patch_embeddings: PatchEmbedding,
    pub position_embeddings: Embedding,
    pub class_token: Option<Array1<f32>>,
    pub dropout: f32,
    pub config: ViTConfig,
}

impl ViTEmbeddings {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        let patch_embeddings = PatchEmbedding::new(config);
        let position_embeddings = Embedding::new(config.seq_length(), config.hidden_size, None)?;

        let class_token = if config.use_class_token {
            Some(Array1::zeros(config.hidden_size))
        } else {
            None
        };

        Ok(Self {
            patch_embeddings,
            position_embeddings,
            class_token,
            dropout: config.hidden_dropout_prob,
            config: config.clone(),
        })
    }

    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        let batch_size = images.dim().0;

        // Get patch embeddings
        let mut embeddings = self.patch_embeddings.forward(images)?;

        // Add class token if used
        if let Some(ref class_token) = self.class_token {
            let class_tokens =
                Array3::from_shape_fn((batch_size, 1, self.config.hidden_size), |(_, _, k)| {
                    class_token[k]
                });

            // Concatenate class token with patch embeddings
            embeddings = ndarray::concatenate![Axis(1), class_tokens, embeddings];
        }

        // Add position embeddings
        let seq_len = embeddings.dim().1;
        let pos_ids: Vec<u32> = (0..seq_len as u32).collect();
        let pos_embeddings = self.position_embeddings.forward(pos_ids)?;

        // Extract array from Tensor
        let pos_emb_array = match pos_embeddings {
            Tensor::F32(arr) => arr,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 tensor".to_string(),
                ))
            },
        };

        // Broadcast position embeddings to batch size
        for b in 0..batch_size {
            embeddings
                .slice_mut(s![b, .., ..])
                .zip_mut_with(&pos_emb_array, |a, &b| *a += b);
        }

        // Apply dropout
        if self.dropout > 0.0 {
            embeddings *= 1.0 - self.dropout;
        }

        Ok(embeddings)
    }
}

/// Vision Transformer attention layer
#[derive(Debug, Clone)]
pub struct ViTAttention {
    pub attention: MultiHeadAttention,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl ViTAttention {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
            )?,
            layer_norm: LayerNorm::new_simple(config.hidden_size, config.layer_norm_eps),
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // Self-attention
        let hidden_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let attention_output = self.attention.forward(hidden_tensor)?;

        // Extract array and apply dropout
        let attention_output = match attention_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 tensor".to_string(),
                ))
            },
        };

        let attention_output = if self.dropout > 0.0 {
            attention_output * (1.0 - self.dropout)
        } else {
            attention_output
        };

        // Residual connection + layer norm
        let output = hidden_states + &attention_output;
        let output_tensor = Tensor::F32(output.into_dyn());
        match self.layer_norm.forward(output_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor".to_string(),
            )),
        }
    }
}

/// Vision Transformer MLP (feed-forward) layer
#[derive(Debug, Clone)]
pub struct ViTMLP {
    pub feed_forward: FeedForward,
    pub layer_norm: LayerNorm,
    pub dropout: f32,
}

impl ViTMLP {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Ok(Self {
            feed_forward: FeedForward::new(
                config.hidden_size,
                config.intermediate_size,
                0.0, // dropout is handled separately
            )?,
            layer_norm: LayerNorm::new_simple(config.hidden_size, config.layer_norm_eps),
            dropout: config.hidden_dropout_prob,
        })
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // Feed-forward
        let hidden_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let ff_output = self.feed_forward.forward(hidden_tensor)?;

        // Extract array and apply dropout
        let ff_output = match ff_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 tensor".to_string(),
                ))
            },
        };

        let ff_output =
            if self.dropout > 0.0 { ff_output * (1.0 - self.dropout) } else { ff_output };

        // Residual connection + layer norm
        let output = hidden_states + &ff_output;
        let output_tensor = Tensor::F32(output.into_dyn());
        match self.layer_norm.forward(output_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor".to_string(),
            )),
        }
    }
}

/// Vision Transformer encoder layer
#[derive(Debug, Clone)]
pub struct ViTLayer {
    pub attention: ViTAttention,
    pub mlp: ViTMLP,
}

impl ViTLayer {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Ok(Self {
            attention: ViTAttention::new(config)?,
            mlp: ViTMLP::new(config)?,
        })
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // Attention sub-layer
        let attention_output = self.attention.forward(hidden_states)?;

        // MLP sub-layer
        let output = self.mlp.forward(&attention_output)?;

        Ok(output)
    }
}

/// Vision Transformer encoder
#[derive(Debug, Clone)]
pub struct ViTEncoder {
    pub layers: Vec<ViTLayer>,
}

impl ViTEncoder {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| ViTLayer::new(config))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers })
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        let mut hidden_states = hidden_states.clone();

        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states)?;
        }

        Ok(hidden_states)
    }
}

/// Vision Transformer model
#[derive(Debug, Clone)]
pub struct ViTModel {
    pub embeddings: ViTEmbeddings,
    pub encoder: ViTEncoder,
    pub layer_norm: LayerNorm,
    pub config: ViTConfig,
}

impl ViTModel {
    pub fn new(config: ViTConfig) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            embeddings: ViTEmbeddings::new(&config)?,
            encoder: ViTEncoder::new(&config)?,
            layer_norm: LayerNorm::new_simple(config.hidden_size, config.layer_norm_eps),
            config,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ViTConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(&self, images: &Array4<f32>) -> Result<Array3<f32>> {
        // Embeddings
        let embeddings = self.embeddings.forward(images)?;

        // Encoder
        let encoder_output = self.encoder.forward(&embeddings)?;

        // Final layer norm
        let output_tensor = Tensor::F32(encoder_output.into_dyn());
        let output = match self.layer_norm.forward(output_tensor)? {
            Tensor::F32(result) => result
                .into_dimensionality::<ndarray::Ix3>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
            _ => {
                return Err(TrustformersError::invalid_input_simple(
                    "Expected F32 tensor".to_string(),
                ))
            },
        };

        Ok(output)
    }

    /// Get the class token representation (for classification)
    pub fn get_class_token_output(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        let output = self.forward(images)?;

        if self.config.use_class_token {
            // Extract class token (first token)
            Ok(output.slice(s![.., 0, ..]).to_owned())
        } else {
            // Use mean of all patch tokens
            Ok(output.mean_axis(Axis(1)).unwrap())
        }
    }
}

/// Vision Transformer for image classification
#[derive(Debug, Clone)]
pub struct ViTForImageClassification {
    pub vit: ViTModel,
    pub classifier: Linear,
    pub dropout: f32,
}

impl ViTForImageClassification {
    pub fn new(config: ViTConfig) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(0.0);

        Ok(Self {
            vit: ViTModel::new(config.clone())?,
            classifier: Linear::new(config.hidden_size, config.num_labels, true),
            dropout,
        })
    }

    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ViTConfig::from_pretrained_name(model_name);
        Self::new(config)
    }

    pub fn forward(&self, images: &Array4<f32>) -> Result<Array2<f32>> {
        // Get class token representation
        let class_output = self.vit.get_class_token_output(images)?;

        // Apply dropout
        let class_output = if self.dropout > 0.0 {
            class_output * (1.0 - self.dropout)
        } else {
            class_output
        };

        // Classification head
        let class_tensor = Tensor::F32(class_output.into_dyn());
        match self.classifier.forward(class_tensor)? {
            Tensor::F32(result) => Ok(result
                .into_dimensionality::<ndarray::Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor".to_string(),
            )),
        }
    }
}
