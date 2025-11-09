use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// LLaVA model configuration
/// Reference: "Visual Instruction Tuning" (Liu et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlavaConfig {
    // Language model configuration
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>,
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32,

    // Vision model configuration
    pub vision_config: LlavaVisionConfig,

    // Multimodal projector configuration
    pub mm_projector_type: String,
    pub mm_hidden_size: usize,
    pub mm_vision_select_layer: i32,
    pub mm_vision_select_feature: String,
    pub mm_patch_merge_type: String,

    // Training configuration
    pub image_aspect_ratio: String,
    pub image_grid_pinpoints: Option<Vec<(usize, usize)>>,
    pub mm_use_im_start_end: bool,
    pub mm_use_im_patch_token: bool,
    pub mm_patch_token: u32,
    pub mm_vision_tower: String,

    // Model type
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlavaVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub image_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub hidden_act: String,
    pub model_type: String,
    pub attention_dropout: f32,
    pub dropout: f32,
}

impl Default for LlavaConfig {
    fn default() -> Self {
        Self {
            // Language model defaults (based on Vicuna/LLaMA)
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            hidden_act: "silu".to_string(),
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,

            // Vision model defaults (based on CLIP ViT)
            vision_config: LlavaVisionConfig::default(),

            // Multimodal projector defaults
            mm_projector_type: "mlp2x_gelu".to_string(),
            mm_hidden_size: 4096,
            mm_vision_select_layer: -2,
            mm_vision_select_feature: "patch".to_string(),
            mm_patch_merge_type: "flat".to_string(),

            // Training defaults
            image_aspect_ratio: "square".to_string(),
            image_grid_pinpoints: None,
            mm_use_im_start_end: false,
            mm_use_im_patch_token: true,
            mm_patch_token: 32000,
            mm_vision_tower: "openai/clip-vit-large-patch14-336".to_string(),

            model_type: "llava".to_string(),
        }
    }
}

impl Default for LlavaVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1024,
            intermediate_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 14,
            image_size: 336,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            hidden_act: "gelu".to_string(),
            model_type: "clip_vision_model".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
        }
    }
}

impl Config for LlavaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        // Validate language model configuration
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(trustformers_core::errors::invalid_config(
                    "num_attention_heads",
                    "num_attention_heads must be divisible by num_key_value_heads",
                ));
            }
        }

        // Validate vision model configuration
        if self.vision_config.hidden_size % self.vision_config.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "vision_hidden_size",
                "vision hidden_size must be divisible by num_attention_heads",
            ));
        }

        // Validate multimodal projector
        if self.mm_vision_select_layer >= self.vision_config.num_hidden_layers as i32 {
            return Err(trustformers_core::errors::invalid_config(
                "mm_vision_select_layer",
                "mm_vision_select_layer must be less than vision num_hidden_layers",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "LLaVA"
    }
}

impl LlavaConfig {
    /// LLaVA-7B configuration (based on Vicuna-7B)
    pub fn llava_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            model_type: "llava-7b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA-13B configuration (based on Vicuna-13B)
    pub fn llava_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 2048,
            model_type: "llava-13b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA-v1.5-7B configuration with improved vision
    pub fn llava_v1_5_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096, // Extended context
            vision_config: LlavaVisionConfig {
                image_size: 336, // Higher resolution
                ..LlavaVisionConfig::default()
            },
            mm_projector_type: "mlp2x_gelu".to_string(),
            mm_vision_select_layer: -2,
            model_type: "llava-v1.5-7b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA-v1.5-13B configuration with improved vision
    pub fn llava_v1_5_13b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            max_position_embeddings: 4096,
            vision_config: LlavaVisionConfig {
                image_size: 336,
                ..LlavaVisionConfig::default()
            },
            mm_projector_type: "mlp2x_gelu".to_string(),
            mm_vision_select_layer: -2,
            model_type: "llava-v1.5-13b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA-v1.6-7B (LLaVA-NeXT) with enhanced capabilities
    pub fn llava_v1_6_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096,
            vision_config: LlavaVisionConfig {
                image_size: 336,
                ..LlavaVisionConfig::default()
            },
            mm_projector_type: "mlp2x_gelu".to_string(),
            mm_vision_select_layer: -2,
            image_aspect_ratio: "anyres".to_string(),
            image_grid_pinpoints: Some(vec![
                (336, 672),
                (672, 336),
                (672, 672),
                (1008, 336),
                (336, 1008),
            ]),
            model_type: "llava-v1.6-7b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA-v1.6-34B (largest LLaVA model)
    pub fn llava_v1_6_34b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 8192,
            intermediate_size: 22016,
            num_hidden_layers: 60,
            num_attention_heads: 64,
            max_position_embeddings: 4096,
            vision_config: LlavaVisionConfig {
                image_size: 336,
                ..LlavaVisionConfig::default()
            },
            mm_projector_type: "mlp2x_gelu".to_string(),
            mm_vision_select_layer: -2,
            image_aspect_ratio: "anyres".to_string(),
            image_grid_pinpoints: Some(vec![
                (336, 672),
                (672, 336),
                (672, 672),
                (1008, 336),
                (336, 1008),
            ]),
            model_type: "llava-v1.6-34b".to_string(),
            ..Self::default()
        }
    }

    /// LLaVA with Phi-3 backend
    pub fn llava_phi3_mini() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32),
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            vision_config: LlavaVisionConfig {
                image_size: 336,
                ..LlavaVisionConfig::default()
            },
            mm_projector_type: "mlp2x_gelu".to_string(),
            model_type: "llava-phi3-mini".to_string(),
            ..Self::default()
        }
    }

    /// Get the head dimension for language model
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the vision head dimension
    pub fn vision_head_dim(&self) -> usize {
        self.vision_config.hidden_size / self.vision_config.num_attention_heads
    }

    /// Get the number of patches for vision
    pub fn num_patches(&self) -> usize {
        (self.vision_config.image_size / self.vision_config.patch_size).pow(2)
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "liuhaotian/llava-v1-0-7b" | "llava-7b" => Some(Self::llava_7b()),
            "liuhaotian/llava-v1-0-13b" | "llava-13b" => Some(Self::llava_13b()),
            "liuhaotian/llava-v1.5-7b" | "llava-v1.5-7b" => Some(Self::llava_v1_5_7b()),
            "liuhaotian/llava-v1.5-13b" | "llava-v1.5-13b" => Some(Self::llava_v1_5_13b()),
            "liuhaotian/llava-v1.6-mistral-7b" | "llava-v1.6-7b" => Some(Self::llava_v1_6_7b()),
            "liuhaotian/llava-v1.6-yi-34b" | "llava-v1.6-34b" => Some(Self::llava_v1_6_34b()),
            "microsoft/llava-phi-3-mini" | "llava-phi3-mini" => Some(Self::llava_phi3_mini()),
            _ => None,
        }
    }

    /// Configure for high-resolution images
    pub fn with_high_resolution(&mut self, enabled: bool) -> &mut Self {
        if enabled {
            self.image_aspect_ratio = "anyres".to_string();
            self.image_grid_pinpoints = Some(vec![
                (336, 672),
                (672, 336),
                (672, 672),
                (1008, 336),
                (336, 1008),
                (1344, 336),
                (336, 1344),
            ]);
        } else {
            self.image_aspect_ratio = "square".to_string();
            self.image_grid_pinpoints = None;
        }
        self
    }

    /// Configure vision tower
    pub fn with_vision_tower(&mut self, tower: &str) -> &mut Self {
        self.mm_vision_tower = tower.to_string();
        self
    }
}
