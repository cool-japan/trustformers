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
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
                return Err(trustformers_core::errors::invalid_config(
                    "num_attention_heads",
                    "num_attention_heads must be divisible by num_key_value_heads",
                ));
            }
        }

        // Validate vision model configuration
        if !self
            .vision_config
            .hidden_size
            .is_multiple_of(self.vision_config.num_attention_heads)
        {
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

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    struct Lcg {
        state: u64,
    }
    impl Lcg {
        fn new(seed: u64) -> Self {
            Lcg { state: seed }
        }
        fn next(&mut self) -> u64 {
            self.state = self
                .state
                .wrapping_mul(6364136223846793005u64)
                .wrapping_add(1442695040888963407u64);
            self.state
        }
        fn next_f32(&mut self) -> f32 {
            (self.next() >> 11) as f32 / (1u64 << 53) as f32
        }
    }

    #[test]
    fn test_default_config_fields() {
        let cfg = LlavaConfig::default();
        assert_eq!(cfg.vocab_size, 32000);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.model_type, "llava");
        assert!(cfg.use_cache);
        assert_eq!(cfg.mm_projector_type, "mlp2x_gelu");
    }

    #[test]
    fn test_default_validate_passes() {
        let cfg = LlavaConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_architecture_name() {
        let cfg = LlavaConfig::default();
        assert_eq!(cfg.architecture(), "LLaVA");
    }

    #[test]
    fn test_hidden_size_not_divisible_fails() {
        let cfg = LlavaConfig {
            hidden_size: 100,
            num_attention_heads: 32,
            ..LlavaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_kv_heads_not_divisible_fails() {
        let cfg = LlavaConfig {
            num_key_value_heads: Some(7),
            ..LlavaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_vision_hidden_not_divisible_fails() {
        let cfg = LlavaConfig {
            vision_config: LlavaVisionConfig {
                hidden_size: 100,
                num_attention_heads: 16,
                ..LlavaVisionConfig::default()
            },
            ..LlavaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_mm_vision_select_layer_out_of_bounds_fails() {
        let cfg = LlavaConfig {
            vision_config: LlavaVisionConfig {
                num_hidden_layers: 24,
                ..LlavaVisionConfig::default()
            },
            mm_vision_select_layer: 25, // >= 24
            ..LlavaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_head_dim_computation() {
        let cfg = LlavaConfig::default();
        assert_eq!(cfg.head_dim(), 4096 / 32);
    }

    #[test]
    fn test_num_kv_heads_default() {
        let cfg = LlavaConfig::default();
        assert_eq!(cfg.num_kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn test_vision_head_dim() {
        let cfg = LlavaConfig::default();
        let expected = cfg.vision_config.hidden_size / cfg.vision_config.num_attention_heads;
        assert_eq!(cfg.vision_head_dim(), expected);
    }

    #[test]
    fn test_num_patches() {
        let cfg = LlavaConfig::default();
        let expected = (cfg.vision_config.image_size / cfg.vision_config.patch_size).pow(2);
        assert_eq!(cfg.num_patches(), expected);
    }

    #[test]
    fn test_llava_7b_config() {
        let cfg = LlavaConfig::llava_7b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llava_13b_config() {
        let cfg = LlavaConfig::llava_13b();
        assert_eq!(cfg.hidden_size, 5120);
        assert_eq!(cfg.num_hidden_layers, 40);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llava_v1_5_7b_config() {
        let cfg = LlavaConfig::llava_v1_5_7b();
        assert_eq!(cfg.max_position_embeddings, 4096);
        assert_eq!(cfg.mm_vision_select_layer, -2);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llava_v1_6_7b_has_anyres() {
        let cfg = LlavaConfig::llava_v1_6_7b();
        assert_eq!(cfg.image_aspect_ratio, "anyres");
        assert!(cfg.image_grid_pinpoints.is_some());
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llava_v1_6_34b_config() {
        let cfg = LlavaConfig::llava_v1_6_34b();
        assert_eq!(cfg.hidden_size, 8192);
        assert_eq!(cfg.num_hidden_layers, 60);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_llava_phi3_mini_config() {
        let cfg = LlavaConfig::llava_phi3_mini();
        assert_eq!(cfg.vocab_size, 32064);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_key_value_heads, Some(32));
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_from_pretrained_name_7b() {
        let result = LlavaConfig::from_pretrained_name("llava-7b");
        assert!(result.is_some());
    }

    #[test]
    fn test_from_pretrained_name_unknown() {
        let result = LlavaConfig::from_pretrained_name("no-such-model");
        assert!(result.is_none());
    }

    #[test]
    fn test_high_resolution_mode() {
        let mut cfg = LlavaConfig::default();
        cfg.with_high_resolution(true);
        assert_eq!(cfg.image_aspect_ratio, "anyres");
        assert!(cfg.image_grid_pinpoints.is_some());
    }

    #[test]
    fn test_disable_high_resolution() {
        let mut cfg = LlavaConfig::llava_v1_6_7b();
        cfg.with_high_resolution(false);
        assert_eq!(cfg.image_aspect_ratio, "square");
        assert!(cfg.image_grid_pinpoints.is_none());
    }

    #[test]
    fn test_with_vision_tower() {
        let mut cfg = LlavaConfig::default();
        cfg.with_vision_tower("openai/clip-vit-base-patch32");
        assert_eq!(cfg.mm_vision_tower, "openai/clip-vit-base-patch32");
    }

    #[test]
    fn test_vision_default_config() {
        let vcfg = LlavaVisionConfig::default();
        assert_eq!(vcfg.hidden_size, 1024);
        assert_eq!(vcfg.patch_size, 14);
        assert_eq!(vcfg.image_size, 336);
        assert_eq!(vcfg.num_channels, 3);
    }

    #[test]
    fn test_lcg_produces_values_in_range() {
        let mut rng = Lcg::new(54321);
        for _ in 0..100 {
            let v = rng.next_f32();
            assert!((0.0..1.0).contains(&v));
        }
    }
}
