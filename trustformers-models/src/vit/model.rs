use crate::vit::config::ViTConfig;
use scirs2_core::ndarray::{concatenate, s, Array1, Array2, Array3, Array4, Axis, Ix2, Ix3}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
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
    device: Device,
}

impl PatchEmbedding {
    pub fn new(config: &ViTConfig) -> Self {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Self {
        let input_size = config.patch_size * config.patch_size * config.num_channels;

        Self {
            projection: Linear::new_with_device(
                input_size,
                config.hidden_size,
                config.use_patch_bias,
                device,
            ),
            patch_size: config.patch_size,
            num_channels: config.num_channels,
            hidden_size: config.hidden_size,
            device,
        }
    }

    pub fn device(&self) -> Device {
        self.device
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
                .into_dimensionality::<Ix3>()
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
    device: Device,
}

impl ViTEmbeddings {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Result<Self> {
        let patch_embeddings = PatchEmbedding::new_with_device(config, device);
        let position_embeddings =
            Embedding::new_with_device(config.seq_length(), config.hidden_size, None, device)?;

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
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
            embeddings = concatenate![Axis(1), class_tokens, embeddings];
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
    device: Device,
}

impl ViTAttention {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: MultiHeadAttention::new_with_device(
                config.hidden_size,
                config.num_attention_heads,
                config.attention_probs_dropout_prob,
                true,
                device,
            )?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // Self-attention
        let hidden_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let attention_output = self.attention.forward(hidden_tensor)?;

        // Extract array and apply dropout
        let attention_output = match attention_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
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
                .into_dimensionality::<Ix3>()
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
    device: Device,
}

impl ViTMLP {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            feed_forward: FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                0.0, // dropout is handled separately
                device,
            ),
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            dropout: config.hidden_dropout_prob,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, hidden_states: &Array3<f32>) -> Result<Array3<f32>> {
        // Feed-forward
        let hidden_tensor = Tensor::F32(hidden_states.clone().into_dyn());
        let ff_output = self.feed_forward.forward(hidden_tensor)?;

        // Extract array and apply dropout
        let ff_output = match ff_output {
            Tensor::F32(arr) => arr
                .into_dimensionality::<Ix3>()
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
                .into_dimensionality::<Ix3>()
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
    device: Device,
}

impl ViTLayer {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Result<Self> {
        Ok(Self {
            attention: ViTAttention::new_with_device(config, device)?,
            mlp: ViTMLP::new_with_device(config, device)?,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl ViTEncoder {
    pub fn new(config: &ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: &ViTConfig, device: Device) -> Result<Self> {
        let layers = (0..config.num_hidden_layers)
            .map(|_| ViTLayer::new_with_device(config, device))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { layers, device })
    }

    pub fn device(&self) -> Device {
        self.device
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
    device: Device,
}

impl ViTModel {
    pub fn new(config: ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ViTConfig, device: Device) -> Result<Self> {
        config.validate()?;

        Ok(Self {
            embeddings: ViTEmbeddings::new_with_device(&config, device)?,
            encoder: ViTEncoder::new_with_device(&config, device)?,
            layer_norm: LayerNorm::new_with_device(
                vec![config.hidden_size],
                config.layer_norm_eps,
                device,
            )?,
            config,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
                .into_dimensionality::<Ix3>()
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
            Ok(output.mean_axis(Axis(1)).expect("operation failed"))
        }
    }
}

/// Vision Transformer for image classification
#[derive(Debug, Clone)]
pub struct ViTForImageClassification {
    pub vit: ViTModel,
    pub classifier: Linear,
    pub dropout: f32,
    device: Device,
}

impl ViTForImageClassification {
    pub fn new(config: ViTConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    pub fn new_with_device(config: ViTConfig, device: Device) -> Result<Self> {
        let dropout = config.classifier_dropout.unwrap_or(0.0);

        Ok(Self {
            vit: ViTModel::new_with_device(config.clone(), device)?,
            classifier: Linear::new_with_device(
                config.hidden_size,
                config.num_labels,
                true,
                device,
            ),
            dropout,
            device,
        })
    }

    pub fn device(&self) -> Device {
        self.device
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
                .into_dimensionality::<Ix2>()
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?),
            _ => Err(TrustformersError::invalid_input_simple(
                "Expected F32 tensor".to_string(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vit::config::ViTConfig;
    use scirs2_core::ndarray::Array4;
    use trustformers_core::traits::Config;

    /// A tiny ViT config fast enough for unit tests.
    /// Uses the same dimensions as the working tests in tests.rs.
    fn tiny_config() -> ViTConfig {
        ViTConfig {
            image_size: 32,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            intermediate_size: 256,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            encoder_stride: 16,
            num_labels: 10,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: false,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    fn make_images(batch: usize, cfg: &ViTConfig) -> Array4<f32> {
        Array4::zeros((batch, cfg.image_size, cfg.image_size, cfg.num_channels))
    }

    // --- ViTConfig ---

    #[test]
    fn test_config_validation_valid() {
        let cfg = tiny_config();
        assert!(
            cfg.validate().is_ok(),
            "valid tiny config must pass validation"
        );
    }

    #[test]
    fn test_config_validation_indivisible_image_patch_fails() {
        let mut cfg = tiny_config();
        cfg.patch_size = 7; // 16 % 7 != 0
        assert!(
            cfg.validate().is_err(),
            "image_size not divisible by patch_size must fail"
        );
    }

    #[test]
    fn test_config_num_patches() {
        let cfg = tiny_config();
        let expected = (cfg.image_size / cfg.patch_size).pow(2);
        assert_eq!(cfg.num_patches(), expected);
    }

    #[test]
    fn test_config_architecture_is_vit() {
        let cfg = tiny_config();
        assert_eq!(cfg.architecture(), "ViT");
    }

    // --- PatchEmbedding ---

    #[test]
    fn test_patch_embedding_output_shape() {
        let cfg = tiny_config();
        let pe = PatchEmbedding::new(&cfg);
        let images = make_images(2, &cfg);
        let out = pe.forward(&images).expect("PatchEmbedding forward must succeed");
        let expected_patches = cfg.num_patches();
        assert_eq!(out.shape()[0], 2, "batch dim must match");
        assert_eq!(out.shape()[1], expected_patches, "patch count must match");
        assert_eq!(
            out.shape()[2],
            cfg.hidden_size,
            "embedding dim must match hidden_size"
        );
    }

    #[test]
    fn test_patch_embedding_wrong_image_size_fails() {
        let cfg = tiny_config();
        let pe = PatchEmbedding::new(&cfg);
        // 15x15 is not divisible by patch_size 8
        let bad_images = Array4::zeros((1_usize, 15, 15, 3));
        let result = pe.forward(&bad_images);
        assert!(
            result.is_err(),
            "non-divisible image size must return an error"
        );
    }

    #[test]
    fn test_patch_embedding_wrong_channel_count_fails() {
        let cfg = tiny_config();
        let pe = PatchEmbedding::new(&cfg);
        let bad_images = Array4::zeros((1_usize, 16, 16, 1)); // 1 channel instead of 3
        let result = pe.forward(&bad_images);
        assert!(result.is_err(), "wrong channel count must return an error");
    }

    // --- ViTEmbeddings ---

    #[test]
    fn test_vit_embeddings_with_class_token_shape() {
        let cfg = tiny_config();
        assert!(cfg.use_class_token, "tiny config must use class token");
        let emb = ViTEmbeddings::new(&cfg).expect("ViTEmbeddings creation must succeed");
        let images = make_images(1, &cfg);
        let out = emb.forward(&images).expect("ViTEmbeddings forward must succeed");
        let expected_seq_len = cfg.num_patches() + 1;
        assert_eq!(
            out.shape()[1],
            expected_seq_len,
            "seq_len must be num_patches+1 with class token"
        );
    }

    #[test]
    fn test_vit_embeddings_without_class_token_shape() {
        let mut cfg = tiny_config();
        cfg.use_class_token = false;
        let emb = ViTEmbeddings::new(&cfg).expect("ViTEmbeddings creation must succeed");
        let images = make_images(1, &cfg);
        let out = emb.forward(&images).expect("ViTEmbeddings forward must succeed");
        assert_eq!(
            out.shape()[1],
            cfg.num_patches(),
            "seq_len must equal num_patches without class token"
        );
    }

    // --- ViTModel ---

    #[test]
    fn test_vit_model_forward_output_shape_with_class_token() {
        let cfg = tiny_config();
        let model = ViTModel::new(cfg.clone()).expect("ViTModel creation must succeed");
        // Use batch_size=1 to avoid incompatible memory layout on reshape
        let images = make_images(1, &cfg);
        let out = model.forward(&images).expect("ViTModel forward must succeed");
        assert_eq!(out.shape()[0], 1, "batch dimension must match");
        let expected_seq = cfg.num_patches() + 1;
        assert_eq!(
            out.shape()[1],
            expected_seq,
            "seq_len must match patches + class token"
        );
        assert_eq!(
            out.shape()[2],
            cfg.hidden_size,
            "feature dim must match hidden_size"
        );
    }

    #[test]
    fn test_vit_model_class_token_output_shape() {
        let cfg = tiny_config();
        let model = ViTModel::new(cfg.clone()).expect("ViTModel creation");
        // Use batch_size=1 to avoid incompatible memory layout on reshape
        let images = make_images(1, &cfg);
        let cls_out = model
            .get_class_token_output(&images)
            .expect("class token extraction must succeed");
        assert_eq!(cls_out.shape()[0], 1, "batch dim must match");
        assert_eq!(
            cls_out.shape()[1],
            cfg.hidden_size,
            "feature dim must match"
        );
    }

    // --- ViTForImageClassification ---

    #[test]
    fn test_vit_classification_output_shape() {
        let cfg = tiny_config();
        let model = ViTForImageClassification::new(cfg.clone())
            .expect("ViTForImageClassification creation must succeed");
        // Use batch_size=1 to avoid incompatible memory layout on reshape
        let images = make_images(1, &cfg);
        let out = model.forward(&images).expect("classification forward must succeed");
        assert_eq!(out.shape()[0], 1, "batch dim must match");
        assert_eq!(
            out.shape()[1],
            cfg.num_labels,
            "logit count must match num_labels"
        );
    }

    #[test]
    fn test_vit_classification_output_all_finite() {
        let cfg = tiny_config();
        let model = ViTForImageClassification::new(cfg.clone())
            .expect("ViTForImageClassification creation");
        let images = make_images(1, &cfg);
        let out = model.forward(&images).expect("classification forward");
        assert!(
            out.iter().all(|v| v.is_finite()),
            "all classification logits must be finite"
        );
    }
}
