use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// CLIP model configuration
/// Reference: "Learning Transferable Visual Representations with Contrastive Language-Image Pre-training" (Radford et al., 2021)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPConfig {
    // Text config
    pub text_config: CLIPTextConfig,
    // Vision config
    pub vision_config: CLIPVisionConfig,
    // Projection dimensions
    pub projection_dim: usize,
    pub logit_scale_init_value: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
}

/// CLIP text encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub max_position_embeddings: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub dropout: f32,
    pub attention_dropout: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
    pub pad_token_id: u32,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
}

/// CLIP vision encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CLIPVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f32,
    pub dropout: f32,
    pub attention_dropout: f32,
    pub initializer_range: f32,
    pub initializer_factor: f32,
}

impl Default for CLIPConfig {
    fn default() -> Self {
        Self {
            text_config: CLIPTextConfig::default(),
            vision_config: CLIPVisionConfig::default(),
            projection_dim: 512,
            logit_scale_init_value: 2.6592, // ln(1/0.07)
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}

impl Default for CLIPTextConfig {
    fn default() -> Self {
        Self {
            vocab_size: 49408,
            hidden_size: 512,
            intermediate_size: 2048,
            num_hidden_layers: 12,
            num_attention_heads: 8,
            max_position_embeddings: 77,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
            pad_token_id: 1,
            bos_token_id: 49406,
            eos_token_id: 49407,
        }
    }
}

impl Default for CLIPVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            intermediate_size: 3072,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            num_channels: 3,
            image_size: 224,
            patch_size: 32,
            hidden_act: "quick_gelu".to_string(),
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            attention_dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }
}

impl Config for CLIPConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        self.text_config.validate()?;
        self.vision_config.validate()?;

        if self.projection_dim == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "projection_dim",
                "projection_dim must be greater than 0",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CLIP"
    }
}

impl Config for CLIPTextConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.vocab_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "vocab_size",
                "vocab_size must be greater than 0",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CLIPText"
    }
}

impl Config for CLIPVisionConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.image_size % self.patch_size != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "image_size",
                "image_size must be divisible by patch_size",
            ));
        }

        if self.patch_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "patch_size",
                "patch_size must be greater than 0",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CLIPVision"
    }
}

impl CLIPConfig {
    /// CLIP ViT-B/32 configuration
    pub fn vit_b_32() -> Self {
        Self {
            text_config: CLIPTextConfig {
                hidden_size: 512,
                intermediate_size: 2048,
                num_hidden_layers: 12,
                num_attention_heads: 8,
                ..CLIPTextConfig::default()
            },
            vision_config: CLIPVisionConfig {
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                patch_size: 32,
                ..CLIPVisionConfig::default()
            },
            projection_dim: 512,
            ..Self::default()
        }
    }

    /// CLIP ViT-B/16 configuration
    pub fn vit_b_16() -> Self {
        Self {
            text_config: CLIPTextConfig {
                hidden_size: 512,
                intermediate_size: 2048,
                num_hidden_layers: 12,
                num_attention_heads: 8,
                ..CLIPTextConfig::default()
            },
            vision_config: CLIPVisionConfig {
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                patch_size: 16,
                ..CLIPVisionConfig::default()
            },
            projection_dim: 512,
            ..Self::default()
        }
    }

    /// CLIP ViT-L/14 configuration
    pub fn vit_l_14() -> Self {
        Self {
            text_config: CLIPTextConfig {
                hidden_size: 768,
                intermediate_size: 3072,
                num_hidden_layers: 12,
                num_attention_heads: 12,
                ..CLIPTextConfig::default()
            },
            vision_config: CLIPVisionConfig {
                hidden_size: 1024,
                intermediate_size: 4096,
                num_hidden_layers: 24,
                num_attention_heads: 16,
                patch_size: 14,
                ..CLIPVisionConfig::default()
            },
            projection_dim: 768,
            ..Self::default()
        }
    }
}

impl CLIPVisionConfig {
    /// Get the number of patches per dimension
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Get the total number of patches
    pub fn num_patches(&self) -> usize {
        let patches_per_side = self.num_patches_per_side();
        patches_per_side * patches_per_side
    }

    /// Get sequence length (patches + class token)
    pub fn seq_length(&self) -> usize {
        self.num_patches() + 1
    }
}

impl CLIPTextConfig {
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

impl CLIPVisionConfig {
    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}
