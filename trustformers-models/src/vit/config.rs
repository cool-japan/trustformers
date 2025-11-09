use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ViTConfig {
    pub image_size: usize,
    pub patch_size: usize,
    pub num_channels: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub encoder_stride: usize,
    pub num_labels: usize,
    pub classifier_dropout: Option<f32>,
    pub model_type: String,

    // ViT-specific parameters
    pub qkv_bias: bool,
    pub use_patch_bias: bool,
    pub use_class_token: bool,
    pub interpolate_pos_encoding: bool,
}

impl Default for ViTConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            encoder_stride: 16,
            num_labels: 1000, // ImageNet classes
            classifier_dropout: None,
            model_type: "vit".to_string(),

            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }
}

impl Config for ViTConfig {
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
        "ViT"
    }
}

impl ViTConfig {
    /// Number of patches per dimension
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of patches
    pub fn num_patches(&self) -> usize {
        let patches_per_side = self.num_patches_per_side();
        patches_per_side * patches_per_side
    }

    /// Sequence length (patches + class token if used)
    pub fn seq_length(&self) -> usize {
        let num_patches = self.num_patches();
        if self.use_class_token {
            num_patches + 1
        } else {
            num_patches
        }
    }

    /// ViT-Tiny configuration (5.7M parameters)
    pub fn tiny() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 192,
            num_hidden_layers: 12,
            num_attention_heads: 3,
            intermediate_size: 768,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            encoder_stride: 16,
            num_labels: 1000,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    /// ViT-Small configuration (22M parameters)
    pub fn small() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 1536,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            encoder_stride: 16,
            num_labels: 1000,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    /// ViT-Base configuration (86M parameters)
    pub fn base() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            encoder_stride: 16,
            num_labels: 1000,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    /// ViT-Large configuration (307M parameters)
    pub fn large() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            encoder_stride: 16,
            num_labels: 1000,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    /// ViT-Huge configuration (632M parameters)
    pub fn huge() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1280,
            num_hidden_layers: 32,
            num_attention_heads: 16,
            intermediate_size: 5120,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            encoder_stride: 14,
            num_labels: 1000,
            classifier_dropout: None,
            model_type: "vit".to_string(),
            qkv_bias: true,
            use_patch_bias: true,
            use_class_token: true,
            interpolate_pos_encoding: false,
        }
    }

    /// Create ViT configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("tiny") {
            Self::tiny()
        } else if name_lower.contains("small") {
            Self::small()
        } else if name_lower.contains("large") {
            Self::large()
        } else if name_lower.contains("huge") {
            Self::huge()
        } else if name_lower.contains("base") {
            Self::base()
        } else {
            // Default to base for unknown variants
            Self::base()
        }
    }

    /// Create configuration for different patch sizes
    pub fn with_patch_size(&self, patch_size: usize) -> Self {
        let mut config = self.clone();
        config.patch_size = patch_size;
        config.encoder_stride = patch_size;
        config
    }

    /// Create configuration for different image sizes
    pub fn with_image_size(&self, image_size: usize) -> Self {
        let mut config = self.clone();
        config.image_size = image_size;
        config
    }
}
