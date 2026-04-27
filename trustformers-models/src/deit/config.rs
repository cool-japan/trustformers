//! DeiT (Data-efficient Image Transformers) configuration.
//!
//! DeiT extends ViT with knowledge distillation, adding a learnable distillation token
//! alongside the [CLS] token. During inference, predictions from both tokens are averaged
//! for improved accuracy.
//!
//! # References
//!
//! - Touvron et al., "Training data-efficient image transformers & distillation through attention"
//!   (2021). <https://arxiv.org/abs/2012.12877>

use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Configuration for DeiT (Data-efficient Image Transformers).
///
/// DeiT adds a distillation token on top of ViT, enabling knowledge distillation
/// from a larger teacher model. At inference time, the CLS and distillation token
/// predictions are averaged to produce the final output.
///
/// # Example
///
/// ```rust
/// use trustformers_models::deit::DeiTConfig;
///
/// let config = DeiTConfig::deit_tiny_patch16_224();
/// assert_eq!(config.hidden_size, 192);
/// assert_eq!(config.num_attention_heads, 3);
/// assert!(config.use_distillation_token);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeiTConfig {
    /// Input image size (height = width). Default: 224.
    pub image_size: usize,
    /// Patch size for patch embedding. Default: 16.
    pub patch_size: usize,
    /// Number of input channels (3 for RGB). Default: 3.
    pub num_channels: usize,
    /// Hidden dimension size. Default: 768.
    pub hidden_size: usize,
    /// Number of transformer encoder blocks. Default: 12.
    pub num_hidden_layers: usize,
    /// Number of self-attention heads. Default: 12.
    pub num_attention_heads: usize,
    /// Feed-forward intermediate size (typically 4× hidden_size). Default: 3072.
    pub intermediate_size: usize,
    /// Dropout probability applied to hidden states. Default: 0.0.
    pub hidden_dropout_prob: f32,
    /// Dropout probability applied to attention weights. Default: 0.0.
    pub attention_probs_dropout_prob: f32,
    /// Number of output labels for classification. Default: 1000.
    pub num_labels: usize,
    /// Whether to add bias to QKV projections. Default: true.
    pub qkv_bias: bool,
    /// Layer normalization epsilon. Default: 1e-6.
    pub layer_norm_eps: f32,
    /// Whether to add a learnable distillation token. Default: true.
    pub use_distillation_token: bool,
    /// Model type identifier (used for HuggingFace compatibility). Default: "deit".
    pub model_type: String,
}

impl Default for DeiTConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 1000,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }
}

impl Config for DeiTConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.patch_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "patch_size",
                "patch_size must be greater than 0",
            ));
        }

        if self.hidden_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be greater than 0",
            ));
        }

        if self.num_attention_heads == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "num_attention_heads",
                "num_attention_heads must be greater than 0",
            ));
        }

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

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "DeiT"
    }
}

impl DeiTConfig {
    /// Number of patches along one side of the image.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Total number of image patches.
    pub fn num_patches(&self) -> usize {
        let n = self.num_patches_per_side();
        n * n
    }

    /// Full sequence length: patches + CLS token + optional distillation token.
    pub fn seq_length(&self) -> usize {
        let num_special = if self.use_distillation_token { 2 } else { 1 };
        self.num_patches() + num_special
    }

    /// DeiT-Tiny/16 — 5.7M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::deit::DeiTConfig;
    ///
    /// let cfg = DeiTConfig::deit_tiny_patch16_224();
    /// assert_eq!(cfg.hidden_size, 192);
    /// ```
    pub fn deit_tiny_patch16_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 192,
            num_hidden_layers: 12,
            num_attention_heads: 3,
            intermediate_size: 768,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 1000,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    /// DeiT-Small/16 — 22M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::deit::DeiTConfig;
    ///
    /// let cfg = DeiTConfig::deit_small_patch16_224();
    /// assert_eq!(cfg.hidden_size, 384);
    /// ```
    pub fn deit_small_patch16_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 384,
            num_hidden_layers: 12,
            num_attention_heads: 6,
            intermediate_size: 1536,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 1000,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    /// DeiT-Base/16 — 86M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::deit::DeiTConfig;
    ///
    /// let cfg = DeiTConfig::deit_base_patch16_224();
    /// assert_eq!(cfg.hidden_size, 768);
    /// ```
    pub fn deit_base_patch16_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 1000,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }

    /// DeiT-Base/16 at 384×384 — same depth as base but higher resolution.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::deit::DeiTConfig;
    ///
    /// let cfg = DeiTConfig::deit_base_patch16_384();
    /// assert_eq!(cfg.image_size, 384);
    /// ```
    pub fn deit_base_patch16_384() -> Self {
        Self {
            image_size: 384,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            num_labels: 1000,
            qkv_bias: true,
            layer_norm_eps: 1e-6,
            use_distillation_token: true,
            model_type: "deit".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_fields() {
        let cfg = DeiTConfig::default();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 16);
        assert_eq!(cfg.hidden_size, 768);
        assert_eq!(cfg.num_hidden_layers, 12);
        assert_eq!(cfg.num_attention_heads, 12);
        assert_eq!(cfg.intermediate_size, 3072);
        assert!(cfg.qkv_bias);
        assert!(cfg.use_distillation_token);
        assert_eq!(cfg.model_type, "deit");
    }

    #[test]
    fn test_tiny_preset() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        assert_eq!(cfg.hidden_size, 192);
        assert_eq!(cfg.num_attention_heads, 3);
        assert_eq!(cfg.intermediate_size, 768);
    }

    #[test]
    fn test_small_preset() {
        let cfg = DeiTConfig::deit_small_patch16_224();
        assert_eq!(cfg.hidden_size, 384);
        assert_eq!(cfg.num_attention_heads, 6);
    }

    #[test]
    fn test_base_384_preset() {
        let cfg = DeiTConfig::deit_base_patch16_384();
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.hidden_size, 768);
    }

    #[test]
    fn test_num_patches_224_patch16() {
        assert_eq!(DeiTConfig::deit_base_patch16_224().num_patches(), 196);
    }

    #[test]
    fn test_num_patches_384_patch16() {
        assert_eq!(DeiTConfig::deit_base_patch16_384().num_patches(), 576);
    }

    #[test]
    fn test_num_patches_per_side_tiny() {
        assert_eq!(DeiTConfig::deit_tiny_patch16_224().num_patches_per_side(), 14);
    }

    #[test]
    fn test_seq_length_with_distillation() {
        assert_eq!(DeiTConfig::deit_base_patch16_224().seq_length(), 198);
    }

    #[test]
    fn test_seq_length_without_distillation() {
        let mut cfg = DeiTConfig::deit_base_patch16_224();
        cfg.use_distillation_token = false;
        assert_eq!(cfg.seq_length(), 197);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(DeiTConfig::default().architecture(), "DeiT");
    }

    #[test]
    fn test_validate_tiny_ok() {
        assert!(DeiTConfig::deit_tiny_patch16_224().validate().is_ok());
    }

    #[test]
    fn test_validate_base_224_ok() {
        assert!(DeiTConfig::deit_base_patch16_224().validate().is_ok());
    }

    #[test]
    fn test_validate_base_384_ok() {
        assert!(DeiTConfig::deit_base_patch16_384().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_patch_size() {
        let mut cfg = DeiTConfig::default();
        cfg.patch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_hidden_size() {
        let mut cfg = DeiTConfig::default();
        cfg.hidden_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_attention_heads() {
        let mut cfg = DeiTConfig::default();
        cfg.num_attention_heads = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_hidden_not_divisible_by_heads() {
        let mut cfg = DeiTConfig::default();
        cfg.hidden_size = 769;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_image_not_divisible_by_patch() {
        let mut cfg = DeiTConfig::default();
        cfg.image_size = 225;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_dropout_zero_by_default() {
        let cfg = DeiTConfig::default();
        assert_eq!(cfg.hidden_dropout_prob, 0.0);
        assert_eq!(cfg.attention_probs_dropout_prob, 0.0);
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = DeiTConfig::deit_tiny_patch16_224();
        let cloned = cfg.clone();
        assert_eq!(cfg.hidden_size, cloned.hidden_size);
        assert_eq!(cfg.use_distillation_token, cloned.use_distillation_token);
        assert_eq!(cfg.model_type, cloned.model_type);
    }

    #[test]
    fn test_lcg_varied_image_sizes() {
        let mut s = 61u64;
        let patch = 16usize;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let mult = ((s % 16) + 1) as usize;
            let img_size = mult * patch;
            let heads = 8usize;
            let mut cfg = DeiTConfig::default();
            cfg.image_size = img_size;
            cfg.hidden_size = heads * 16;
            cfg.num_attention_heads = heads;
            assert!(cfg.validate().is_ok(), "img={img_size} failed");
        }
    }
}
