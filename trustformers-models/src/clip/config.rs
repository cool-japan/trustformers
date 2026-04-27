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
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
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
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if !self.image_size.is_multiple_of(self.patch_size) {
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

/// Trait for extracting encoder-layer-level configuration from both
/// [`CLIPTextConfig`] and [`CLIPVisionConfig`].
///
/// This allows `CLIPEncoder` to be generic over the config type while
/// still reading concrete field values instead of relying on hardcoded
/// placeholder constants.
pub trait CLIPEncoderConfig: Config {
    fn hidden_size(&self) -> usize;
    fn num_attention_heads(&self) -> usize;
    fn intermediate_size(&self) -> usize;
    fn num_hidden_layers(&self) -> usize;
    fn hidden_act(&self) -> &str;
    fn layer_norm_eps(&self) -> f32;
    fn attention_dropout(&self) -> f32;
    fn dropout(&self) -> f32;
}

impl CLIPEncoderConfig for CLIPTextConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn hidden_act(&self) -> &str {
        &self.hidden_act
    }
    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
    fn attention_dropout(&self) -> f32 {
        self.attention_dropout
    }
    fn dropout(&self) -> f32 {
        self.dropout
    }
}

impl CLIPEncoderConfig for CLIPVisionConfig {
    fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    fn num_attention_heads(&self) -> usize {
        self.num_attention_heads
    }
    fn intermediate_size(&self) -> usize {
        self.intermediate_size
    }
    fn num_hidden_layers(&self) -> usize {
        self.num_hidden_layers
    }
    fn hidden_act(&self) -> &str {
        &self.hidden_act
    }
    fn layer_norm_eps(&self) -> f32 {
        self.layer_norm_eps
    }
    fn attention_dropout(&self) -> f32 {
        self.attention_dropout
    }
    fn dropout(&self) -> f32 {
        self.dropout
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    // ──────────────────────────────────────────────────────────────────────────
    // CLIPTextConfig defaults
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_text_config_default_vocab_size() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.vocab_size, 49408);
    }

    #[test]
    fn test_text_config_default_hidden_size() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.hidden_size, 512);
    }

    #[test]
    fn test_text_config_default_num_attention_heads() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.num_attention_heads, 8);
    }

    #[test]
    fn test_text_config_default_max_position_embeddings() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.max_position_embeddings, 77);
    }

    #[test]
    fn test_text_config_default_hidden_act() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.hidden_act, "quick_gelu");
    }

    #[test]
    fn test_text_config_default_bos_eos_token_ids() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.bos_token_id, 49406);
        assert_eq!(tc.eos_token_id, 49407);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CLIPVisionConfig defaults
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_vision_config_default_hidden_size() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.hidden_size, 768);
    }

    #[test]
    fn test_vision_config_default_image_size() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.image_size, 224);
    }

    #[test]
    fn test_vision_config_default_patch_size() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.patch_size, 32);
    }

    #[test]
    fn test_vision_config_default_num_channels() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.num_channels, 3);
    }

    #[test]
    fn test_vision_config_default_num_attention_heads() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.num_attention_heads, 12);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CLIPConfig defaults
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_config_default_projection_dim() {
        let cfg = CLIPConfig::default();
        assert_eq!(cfg.projection_dim, 512);
    }

    #[test]
    fn test_clip_config_default_logit_scale() {
        let cfg = CLIPConfig::default();
        // ln(1/0.07) ≈ 2.6592
        assert!((cfg.logit_scale_init_value - 2.6592).abs() < 1e-3);
    }

    #[test]
    fn test_clip_config_default_initializer_factor() {
        let cfg = CLIPConfig::default();
        assert!((cfg.initializer_factor - 1.0).abs() < 1e-9);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Factory constructors
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_vit_b_32_patch_size() {
        let cfg = CLIPConfig::vit_b_32();
        assert_eq!(cfg.vision_config.patch_size, 32);
        assert_eq!(cfg.vision_config.hidden_size, 768);
        assert_eq!(cfg.projection_dim, 512);
    }

    #[test]
    fn test_vit_b_16_patch_size() {
        let cfg = CLIPConfig::vit_b_16();
        assert_eq!(cfg.vision_config.patch_size, 16);
        assert_eq!(cfg.vision_config.hidden_size, 768);
        assert_eq!(cfg.projection_dim, 512);
    }

    #[test]
    fn test_vit_l_14_patch_size_and_projection() {
        let cfg = CLIPConfig::vit_l_14();
        assert_eq!(cfg.vision_config.patch_size, 14);
        assert_eq!(cfg.vision_config.hidden_size, 1024);
        assert_eq!(cfg.vision_config.num_attention_heads, 16);
        assert_eq!(cfg.projection_dim, 768);
    }

    #[test]
    fn test_vit_l_14_text_config() {
        let cfg = CLIPConfig::vit_l_14();
        assert_eq!(cfg.text_config.hidden_size, 768);
        assert_eq!(cfg.text_config.num_attention_heads, 12);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helper methods — CLIPVisionConfig
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_num_patches_per_side_b32() {
        let cfg = CLIPConfig::vit_b_32();
        // 224 / 32 = 7
        assert_eq!(cfg.vision_config.num_patches_per_side(), 7);
    }

    #[test]
    fn test_num_patches_b32() {
        let cfg = CLIPConfig::vit_b_32();
        // 7 * 7 = 49
        assert_eq!(cfg.vision_config.num_patches(), 49);
    }

    #[test]
    fn test_seq_length_b32_includes_cls_token() {
        let cfg = CLIPConfig::vit_b_32();
        // 49 + 1 = 50
        assert_eq!(cfg.vision_config.seq_length(), 50);
    }

    #[test]
    fn test_num_patches_per_side_b16() {
        let cfg = CLIPConfig::vit_b_16();
        // 224 / 16 = 14
        assert_eq!(cfg.vision_config.num_patches_per_side(), 14);
    }

    #[test]
    fn test_num_patches_b16() {
        let cfg = CLIPConfig::vit_b_16();
        // 14 * 14 = 196
        assert_eq!(cfg.vision_config.num_patches(), 196);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Helper methods — head_dim
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_text_head_dim_default() {
        let tc = CLIPTextConfig::default();
        // 512 / 8 = 64
        assert_eq!(tc.head_dim(), 64);
    }

    #[test]
    fn test_vision_head_dim_default() {
        let vc = CLIPVisionConfig::default();
        // 768 / 12 = 64
        assert_eq!(vc.head_dim(), 64);
    }

    #[test]
    fn test_vision_head_dim_vit_l_14() {
        let cfg = CLIPConfig::vit_l_14();
        // 1024 / 16 = 64
        assert_eq!(cfg.vision_config.head_dim(), 64);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // CLIPEncoderConfig trait
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_encoder_config_text_fields() {
        let tc = CLIPTextConfig::default();
        assert_eq!(CLIPEncoderConfig::hidden_size(&tc), 512);
        assert_eq!(CLIPEncoderConfig::num_attention_heads(&tc), 8);
        assert_eq!(CLIPEncoderConfig::intermediate_size(&tc), 2048);
        assert_eq!(CLIPEncoderConfig::num_hidden_layers(&tc), 12);
        assert_eq!(CLIPEncoderConfig::hidden_act(&tc), "quick_gelu");
        assert!((CLIPEncoderConfig::layer_norm_eps(&tc) - 1e-5).abs() < 1e-9);
        assert!((CLIPEncoderConfig::attention_dropout(&tc) - 0.0).abs() < 1e-9);
        assert!((CLIPEncoderConfig::dropout(&tc) - 0.0).abs() < 1e-9);
    }

    #[test]
    fn test_encoder_config_vision_fields() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(CLIPEncoderConfig::hidden_size(&vc), 768);
        assert_eq!(CLIPEncoderConfig::num_attention_heads(&vc), 12);
        assert_eq!(CLIPEncoderConfig::intermediate_size(&vc), 3072);
        assert_eq!(CLIPEncoderConfig::num_hidden_layers(&vc), 12);
    }

    // ──────────────────────────────────────────────────────────────────────────
    // validate – success
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_default_clip_config_succeeds() {
        let cfg = CLIPConfig::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_validate_all_named_configs() {
        let configs = [
            CLIPConfig::vit_b_32(),
            CLIPConfig::vit_b_16(),
            CLIPConfig::vit_l_14(),
        ];
        for cfg in &configs {
            assert!(cfg.validate().is_ok());
        }
    }

    #[test]
    fn test_validate_text_config_succeeds() {
        let tc = CLIPTextConfig::default();
        assert!(tc.validate().is_ok());
    }

    #[test]
    fn test_validate_vision_config_succeeds() {
        let vc = CLIPVisionConfig::default();
        assert!(vc.validate().is_ok());
    }

    // ──────────────────────────────────────────────────────────────────────────
    // validate – failures
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_validate_text_hidden_not_divisible_fails() {
        let mut cfg = CLIPConfig::default();
        cfg.text_config.hidden_size = 100;
        cfg.text_config.num_attention_heads = 8;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_text_zero_vocab_fails() {
        let mut cfg = CLIPConfig::default();
        cfg.text_config.vocab_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_vision_hidden_not_divisible_fails() {
        let mut cfg = CLIPConfig::default();
        cfg.vision_config.hidden_size = 100;
        cfg.vision_config.num_attention_heads = 12;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_vision_image_not_divisible_by_patch_fails() {
        let mut cfg = CLIPConfig::default();
        cfg.vision_config.image_size = 225; // not divisible by 32
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_vision_odd_image_size_for_patch14_fails() {
        // Verify that a patch_size that doesn't divide image_size is rejected.
        // Using patch_size=14 and image_size=225 (not divisible by 14).
        let mut cfg = CLIPConfig::default();
        cfg.vision_config.patch_size = 14;
        cfg.vision_config.image_size = 225; // 225 % 14 = 1, not divisible
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_projection_dim_fails() {
        let cfg = CLIPConfig {
            projection_dim: 0,
            ..CLIPConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Config::architecture
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_architecture_clip() {
        let cfg = CLIPConfig::default();
        assert_eq!(cfg.architecture(), "CLIP");
    }

    #[test]
    fn test_architecture_clip_text() {
        let tc = CLIPTextConfig::default();
        assert_eq!(tc.architecture(), "CLIPText");
    }

    #[test]
    fn test_architecture_clip_vision() {
        let vc = CLIPVisionConfig::default();
        assert_eq!(vc.architecture(), "CLIPVision");
    }

    // ──────────────────────────────────────────────────────────────────────────
    // Clone
    // ──────────────────────────────────────────────────────────────────────────

    #[test]
    fn test_clip_config_clone_preserves_fields() {
        let cfg = CLIPConfig::vit_l_14();
        let cloned = cfg.clone();
        assert_eq!(cloned.projection_dim, cfg.projection_dim);
        assert_eq!(
            cloned.vision_config.hidden_size,
            cfg.vision_config.hidden_size
        );
        assert_eq!(cloned.text_config.hidden_size, cfg.text_config.hidden_size);
    }
}
