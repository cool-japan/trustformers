use serde::{Deserialize, Serialize};

/// Configuration for the DALL-E model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DalleConfig {
    /// Text encoder configuration
    pub text_config: DalleTextConfig,
    /// Image encoder configuration
    pub image_config: DalleImageConfig,
    /// Vision transformer configuration for CLIP-style alignment
    pub vision_config: DalleVisionConfig,
    /// Diffusion configuration for image generation
    pub diffusion_config: DalleDiffusionConfig,
    /// Number of text tokens
    pub text_vocab_size: usize,
    /// Image resolution
    pub image_size: usize,
    /// Number of diffusion timesteps
    pub num_diffusion_steps: usize,
    /// Cross-attention between text and image features
    pub use_cross_attention: bool,
    /// Guidance scale for classifier-free guidance
    pub guidance_scale: f64,
    /// Whether to use CLIP pretraining
    pub use_clip_loss: bool,
}

impl Default for DalleConfig {
    fn default() -> Self {
        Self {
            text_config: DalleTextConfig::default(),
            image_config: DalleImageConfig::default(),
            vision_config: DalleVisionConfig::default(),
            diffusion_config: DalleDiffusionConfig::default(),
            text_vocab_size: 49408, // CLIP vocab size
            image_size: 512,
            num_diffusion_steps: 1000,
            use_cross_attention: true,
            guidance_scale: 7.5,
            use_clip_loss: true,
        }
    }
}

impl DalleConfig {
    /// DALL-E 2 configuration
    pub fn dalle_2() -> Self {
        Self {
            text_config: DalleTextConfig::clip_large(),
            image_config: DalleImageConfig::dalle_2(),
            vision_config: DalleVisionConfig::clip_vit_l(),
            diffusion_config: DalleDiffusionConfig::dalle_2(),
            text_vocab_size: 49408,
            image_size: 512,
            num_diffusion_steps: 1000,
            use_cross_attention: true,
            guidance_scale: 7.5,
            use_clip_loss: true,
        }
    }

    /// DALL-E 3 configuration
    pub fn dalle_3() -> Self {
        Self {
            text_config: DalleTextConfig::t5_xxl(),
            image_config: DalleImageConfig::dalle_3(),
            vision_config: DalleVisionConfig::clip_vit_g(),
            diffusion_config: DalleDiffusionConfig::dalle_3(),
            text_vocab_size: 32128, // T5 vocab size
            image_size: 1024,
            num_diffusion_steps: 1000,
            use_cross_attention: true,
            guidance_scale: 10.0,
            use_clip_loss: true,
        }
    }

    /// DALL-E Mini configuration (smaller variant)
    pub fn dalle_mini() -> Self {
        Self {
            text_config: DalleTextConfig::clip_base(),
            image_config: DalleImageConfig::dalle_mini(),
            vision_config: DalleVisionConfig::clip_vit_b(),
            diffusion_config: DalleDiffusionConfig::dalle_mini(),
            text_vocab_size: 49408,
            image_size: 256,
            num_diffusion_steps: 250,
            use_cross_attention: true,
            guidance_scale: 5.0,
            use_clip_loss: true,
        }
    }
}

/// Text encoder configuration for DALL-E
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DalleTextConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length (max sequence length)
    pub context_length: usize,
    /// Hidden size / embedding dimension
    pub hidden_size: usize,
    /// Number of transformer layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// MLP hidden size
    pub intermediate_size: usize,
    /// Activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}

impl Default for DalleTextConfig {
    fn default() -> Self {
        Self::clip_large()
    }
}

impl DalleTextConfig {
    /// CLIP Large text encoder
    pub fn clip_large() -> Self {
        Self {
            vocab_size: 49408,
            context_length: 77,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// CLIP Base text encoder
    pub fn clip_base() -> Self {
        Self {
            vocab_size: 49408,
            context_length: 77,
            hidden_size: 512,
            num_hidden_layers: 8,
            num_attention_heads: 8,
            intermediate_size: 2048,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// T5-XXL configuration for DALL-E 3
    pub fn t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            context_length: 512,
            hidden_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            intermediate_size: 10240,
            hidden_act: "relu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-6,
            initializer_range: 1.0,
        }
    }
}

/// Image encoder/decoder configuration for DALL-E
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DalleImageConfig {
    /// Image resolution
    pub image_size: usize,
    /// Number of channels (usually 3 for RGB)
    pub num_channels: usize,
    /// Patch size for vision transformer
    pub patch_size: usize,
    /// Hidden size for image processing
    pub hidden_size: usize,
    /// Number of layers in image encoder/decoder
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// MLP hidden size
    pub intermediate_size: usize,
    /// Activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Number of latent dimensions for diffusion
    pub latent_channels: usize,
    /// Downsampling factor from image to latent space
    pub downsampling_factor: usize,
}

impl Default for DalleImageConfig {
    fn default() -> Self {
        Self::dalle_2()
    }
}

impl DalleImageConfig {
    /// DALL-E 2 image configuration
    pub fn dalle_2() -> Self {
        Self {
            image_size: 512,
            num_channels: 3,
            patch_size: 16,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            latent_channels: 4,
            downsampling_factor: 8,
        }
    }

    /// DALL-E 3 image configuration
    pub fn dalle_3() -> Self {
        Self {
            image_size: 1024,
            num_channels: 3,
            patch_size: 16,
            hidden_size: 1536,
            num_hidden_layers: 32,
            num_attention_heads: 24,
            intermediate_size: 6144,
            hidden_act: "gelu".to_string(),
            dropout: 0.0,
            layer_norm_eps: 1e-5,
            latent_channels: 4,
            downsampling_factor: 8,
        }
    }

    /// DALL-E Mini image configuration
    pub fn dalle_mini() -> Self {
        Self {
            image_size: 256,
            num_channels: 3,
            patch_size: 16,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            dropout: 0.1,
            layer_norm_eps: 1e-5,
            latent_channels: 4,
            downsampling_factor: 8,
        }
    }

    /// Number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Latent image size
    pub fn latent_size(&self) -> usize {
        self.image_size / self.downsampling_factor
    }

    /// Number of latent patches
    pub fn num_latent_patches(&self) -> usize {
        self.latent_size().pow(2)
    }
}

/// Vision transformer configuration for CLIP alignment
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DalleVisionConfig {
    /// Image size
    pub image_size: usize,
    /// Patch size
    pub patch_size: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// MLP hidden size
    pub intermediate_size: usize,
    /// Activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}

impl Default for DalleVisionConfig {
    fn default() -> Self {
        Self::clip_vit_l()
    }
}

impl DalleVisionConfig {
    /// CLIP ViT-L/14 configuration
    pub fn clip_vit_l() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// CLIP ViT-B/16 configuration
    pub fn clip_vit_b() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// CLIP ViT-G for DALL-E 3
    pub fn clip_vit_g() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1664,
            num_hidden_layers: 48,
            num_attention_heads: 16,
            intermediate_size: 8192,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// Number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Sequence length (patches + CLS token)
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }
}

/// Diffusion model configuration for DALL-E
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DalleDiffusionConfig {
    /// Number of diffusion timesteps
    pub num_timesteps: usize,
    /// Beta schedule type
    pub beta_schedule: String,
    /// Linear beta start value
    pub beta_start: f64,
    /// Linear beta end value
    pub beta_end: f64,
    /// Whether to use learned variance
    pub learned_variance: bool,
    /// Loss type (l1, l2, huber)
    pub loss_type: String,
    /// Variance type (learned, fixed_small, fixed_large)
    pub variance_type: String,
    /// Whether to use v-parameterization
    pub v_parameterization: bool,
    /// Classifier-free guidance dropout probability
    pub guidance_dropout: f64,
    /// Number of inference steps
    pub num_inference_steps: usize,
    /// ETA parameter for DDIM
    pub eta: f64,
}

impl Default for DalleDiffusionConfig {
    fn default() -> Self {
        Self::dalle_2()
    }
}

impl DalleDiffusionConfig {
    /// DALL-E 2 diffusion configuration
    pub fn dalle_2() -> Self {
        Self {
            num_timesteps: 1000,
            beta_schedule: "scaled_linear".to_string(),
            beta_start: 0.00085,
            beta_end: 0.012,
            learned_variance: false,
            loss_type: "l2".to_string(),
            variance_type: "fixed_small".to_string(),
            v_parameterization: false,
            guidance_dropout: 0.1,
            num_inference_steps: 50,
            eta: 0.0,
        }
    }

    /// DALL-E 3 diffusion configuration
    pub fn dalle_3() -> Self {
        Self {
            num_timesteps: 1000,
            beta_schedule: "scaled_linear".to_string(),
            beta_start: 0.00085,
            beta_end: 0.012,
            learned_variance: true,
            loss_type: "l2".to_string(),
            variance_type: "learned".to_string(),
            v_parameterization: true,
            guidance_dropout: 0.1,
            num_inference_steps: 50,
            eta: 0.0,
        }
    }

    /// DALL-E Mini diffusion configuration
    pub fn dalle_mini() -> Self {
        Self {
            num_timesteps: 250,
            beta_schedule: "linear".to_string(),
            beta_start: 0.0001,
            beta_end: 0.02,
            learned_variance: false,
            loss_type: "l2".to_string(),
            variance_type: "fixed_small".to_string(),
            v_parameterization: false,
            guidance_dropout: 0.1,
            num_inference_steps: 25,
            eta: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dalle_config_default() {
        let config = DalleConfig::default();
        assert_eq!(config.text_vocab_size, 49408);
        assert_eq!(config.image_size, 512);
        assert_eq!(config.num_diffusion_steps, 1000);
        assert!(config.use_cross_attention);
        assert!(config.use_clip_loss);
    }

    #[test]
    fn test_dalle_2_config() {
        let config = DalleConfig::dalle_2();
        assert_eq!(config.text_config.vocab_size, 49408);
        assert_eq!(config.text_config.context_length, 77);
        assert_eq!(config.image_config.image_size, 512);
        assert_eq!(config.vision_config.hidden_size, 1024);
        assert_eq!(config.diffusion_config.num_timesteps, 1000);
    }

    #[test]
    fn test_dalle_3_config() {
        let config = DalleConfig::dalle_3();
        assert_eq!(config.text_config.vocab_size, 32128); // T5 vocab
        assert_eq!(config.image_config.image_size, 1024);
        assert_eq!(config.vision_config.hidden_size, 1664); // ViT-G
        assert!(config.diffusion_config.learned_variance);
        assert!(config.diffusion_config.v_parameterization);
    }

    #[test]
    fn test_dalle_mini_config() {
        let config = DalleConfig::dalle_mini();
        assert_eq!(config.image_size, 256);
        assert_eq!(config.diffusion_config.num_timesteps, 250);
        assert_eq!(config.guidance_scale, 5.0);
    }

    #[test]
    fn test_image_config_calculations() {
        let config = DalleImageConfig::dalle_2();
        assert_eq!(config.num_patches(), 1024); // (512/16)^2 = 32^2 = 1024
        assert_eq!(config.latent_size(), 64); // 512/8 = 64
        assert_eq!(config.num_latent_patches(), 4096); // 64^2 = 4096
    }

    #[test]
    fn test_vision_config_calculations() {
        let config = DalleVisionConfig::clip_vit_l();
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
        assert_eq!(config.seq_len(), 257); // 256 patches + 1 CLS token
    }

    #[test]
    fn test_text_config_variants() {
        let clip_large = DalleTextConfig::clip_large();
        let clip_base = DalleTextConfig::clip_base();
        let t5_xxl = DalleTextConfig::t5_xxl();

        assert_eq!(clip_large.hidden_size, 768);
        assert_eq!(clip_base.hidden_size, 512);
        assert_eq!(t5_xxl.hidden_size, 4096);

        assert_eq!(clip_large.vocab_size, 49408);
        assert_eq!(clip_base.vocab_size, 49408);
        assert_eq!(t5_xxl.vocab_size, 32128);
    }

    #[test]
    fn test_diffusion_config_variants() {
        let dalle_2 = DalleDiffusionConfig::dalle_2();
        let dalle_3 = DalleDiffusionConfig::dalle_3();
        let dalle_mini = DalleDiffusionConfig::dalle_mini();

        assert!(!dalle_2.learned_variance);
        assert!(dalle_3.learned_variance);
        assert!(!dalle_mini.learned_variance);

        assert!(!dalle_2.v_parameterization);
        assert!(dalle_3.v_parameterization);
        assert!(!dalle_mini.v_parameterization);

        assert_eq!(dalle_2.num_timesteps, 1000);
        assert_eq!(dalle_3.num_timesteps, 1000);
        assert_eq!(dalle_mini.num_timesteps, 250);
    }

    #[test]
    fn test_config_serialization() {
        let config = DalleConfig::dalle_2();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: DalleConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.text_vocab_size, deserialized.text_vocab_size);
        assert_eq!(config.image_size, deserialized.image_size);
        assert_eq!(config.guidance_scale, deserialized.guidance_scale);
        assert_eq!(
            config.text_config.hidden_size,
            deserialized.text_config.hidden_size
        );
        assert_eq!(
            config.image_config.latent_channels,
            deserialized.image_config.latent_channels
        );
        assert_eq!(
            config.diffusion_config.num_timesteps,
            deserialized.diffusion_config.num_timesteps
        );
    }
}
