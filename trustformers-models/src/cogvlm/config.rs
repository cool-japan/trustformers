use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// CogVLM model configuration
/// Reference: "CogVLM: Visual Expert for Pretrained Language Models" (Wang et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogVlmConfig {
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
    pub rope_scaling: Option<RopeScaling>,

    // Vision model configuration
    pub vision_config: CogVlmVisionConfig,

    // Cross-modal attention configuration
    pub cross_hidden_size: usize,
    pub cross_compute_hidden_size: usize,
    pub cogvlm_stage: i32, // 1 for image understanding, 2 for chat
    pub template_version: String,

    // Visual expert configuration
    pub num_multi_token: usize,
    pub multi_token_key: String,
    pub vision_token_num: usize,
    pub image_patch_token_id: u32,

    // Model type and version
    pub model_type: String,
    pub use_lora: bool,
    pub lora_rank: Option<usize>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogVlmVisionConfig {
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
    pub use_flash_attn: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub type_: String,
    pub factor: f32,
}

impl Default for CogVlmConfig {
    fn default() -> Self {
        Self {
            // Language model defaults (based on ChatGLM/Vicuna)
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
            pad_token_id: Some(0),
            bos_token_id: 1,
            eos_token_id: 2,
            rope_theta: 10000.0,
            rope_scaling: None,

            // Vision model defaults
            vision_config: CogVlmVisionConfig::default(),

            // Cross-modal attention defaults
            cross_hidden_size: 4096,
            cross_compute_hidden_size: 4096,
            cogvlm_stage: 2,
            template_version: "chat".to_string(),

            // Visual expert defaults
            num_multi_token: 5,
            multi_token_key: "multi_token".to_string(),
            vision_token_num: 256,
            image_patch_token_id: 32000,

            model_type: "cogvlm".to_string(),
            use_lora: false,
            lora_rank: None,
        }
    }
}

impl Default for CogVlmVisionConfig {
    fn default() -> Self {
        Self {
            hidden_size: 1792,
            intermediate_size: 15360,
            num_hidden_layers: 63,
            num_attention_heads: 16,
            num_channels: 3,
            patch_size: 14,
            image_size: 490,
            initializer_range: 0.02,
            layer_norm_eps: 1e-6,
            hidden_act: "gelu".to_string(),
            model_type: "eva_clip_g".to_string(),
            attention_dropout: 0.0,
            dropout: 0.0,
            use_flash_attn: true,
        }
    }
}

impl Config for CogVlmConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        // Validate language model configuration
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(
                    trustformers_core::errors::TrustformersError::invalid_config(
                        "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                    ),
                );
            }
        }

        // Validate vision model configuration
        if self.vision_config.hidden_size % self.vision_config.num_attention_heads != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "vision hidden_size must be divisible by num_attention_heads".to_string(),
                ),
            );
        }

        // Validate cross-modal configuration
        if self.cross_hidden_size != self.hidden_size {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "cross_hidden_size must equal hidden_size".to_string(),
                ),
            );
        }

        if self.cogvlm_stage < 1 || self.cogvlm_stage > 2 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "cogvlm_stage must be 1 or 2".to_string(),
                ),
            );
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CogVLM"
    }
}

impl CogVlmConfig {
    /// CogVLM-Chat-17B configuration
    pub fn cogvlm_chat_17b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            cogvlm_stage: 2,
            template_version: "chat".to_string(),
            vision_config: CogVlmVisionConfig {
                hidden_size: 1792,
                intermediate_size: 15360,
                num_hidden_layers: 63,
                image_size: 490,
                ..CogVlmVisionConfig::default()
            },
            model_type: "cogvlm-chat-17b".to_string(),
            ..Self::default()
        }
    }

    /// CogVLM-Base-17B configuration
    pub fn cogvlm_base_17b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            cogvlm_stage: 1,
            template_version: "base".to_string(),
            model_type: "cogvlm-base-17b".to_string(),
            ..Self::default()
        }
    }

    /// CogVLM-Grounding-17B for visual grounding tasks
    pub fn cogvlm_grounding_17b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 2048,
            cogvlm_stage: 2,
            template_version: "grounding".to_string(),
            model_type: "cogvlm-grounding-17b".to_string(),
            ..Self::default()
        }
    }

    /// CogVideo configuration for video understanding
    pub fn cogvideo() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096, // Longer sequences for video
            vision_config: CogVlmVisionConfig {
                hidden_size: 1792,
                intermediate_size: 15360,
                num_hidden_layers: 63,
                image_size: 224, // Smaller frames for video
                ..CogVlmVisionConfig::default()
            },
            vision_token_num: 1024, // More tokens for video
            cogvlm_stage: 2,
            template_version: "video".to_string(),
            model_type: "cogvideo".to_string(),
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
            "THUDM/cogvlm-chat-hf" | "cogvlm-chat-17b" => Some(Self::cogvlm_chat_17b()),
            "THUDM/cogvlm-base-hf" | "cogvlm-base-17b" => Some(Self::cogvlm_base_17b()),
            "THUDM/cogvlm-grounding-generalist-hf" | "cogvlm-grounding-17b" => {
                Some(Self::cogvlm_grounding_17b())
            },
            "THUDM/cogvideo-chat" | "cogvideo" => Some(Self::cogvideo()),
            _ => None,
        }
    }

    /// Configure for LoRA fine-tuning
    pub fn with_lora(&mut self, enabled: bool, rank: Option<usize>) -> &mut Self {
        self.use_lora = enabled;
        self.lora_rank = rank;
        self
    }

    /// Configure vision token number
    pub fn with_vision_tokens(&mut self, num_tokens: usize) -> &mut Self {
        self.vision_token_num = num_tokens;
        self
    }

    /// Configure for different stages
    pub fn with_stage(&mut self, stage: i32, template: &str) -> &mut Self {
        self.cogvlm_stage = stage;
        self.template_version = template.to_string();
        self
    }

    /// Small configuration for testing (fast model creation)
    pub fn small_test_config() -> Self {
        Self {
            vocab_size: 1000,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            max_position_embeddings: 512,
            cross_hidden_size: 64,
            vision_config: CogVlmVisionConfig {
                hidden_size: 64,
                intermediate_size: 128,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                ..Default::default()
            },
            model_type: "cogvlm-test".to_string(),
            ..Self::default()
        }
    }
}

/// CogVideo specific configuration for video understanding
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CogVideoConfig {
    pub base_config: CogVlmConfig,
    pub video_frames: usize,
    pub frame_stride: usize,
    pub temporal_patch_size: usize,
    pub temporal_num_layers: usize,
    pub temporal_hidden_size: usize,
    pub use_temporal_attention: bool,
    pub max_video_length: usize,
}

impl Default for CogVideoConfig {
    fn default() -> Self {
        Self {
            base_config: CogVlmConfig::cogvideo(),
            video_frames: 16,
            frame_stride: 2,
            temporal_patch_size: 2,
            temporal_num_layers: 4,
            temporal_hidden_size: 4096,
            use_temporal_attention: true,
            max_video_length: 32,
        }
    }
}

impl Config for CogVideoConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        self.base_config.validate()?;

        if self.video_frames == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "video_frames must be greater than 0".to_string(),
                ),
            );
        }

        if self.temporal_patch_size == 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "temporal_patch_size must be greater than 0".to_string(),
                ),
            );
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "CogVideo"
    }
}
