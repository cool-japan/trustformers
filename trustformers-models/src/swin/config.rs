//! Swin Transformer configuration.
//!
//! The Swin Transformer introduces a hierarchical feature pyramid built from
//! shifted-window self-attention (SW-MSA). Unlike ViT, feature maps are
//! downsampled between stages, enabling dense prediction tasks.
//!
//! # References
//!
//! - Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
//!   (ICCV 2021). <https://arxiv.org/abs/2103.14030>

use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Configuration for the Swin Transformer family.
///
/// # Example
///
/// ```rust
/// use trustformers_models::swin::SwinConfig;
///
/// let cfg = SwinConfig::swin_tiny_patch4_window7_224();
/// assert_eq!(cfg.embed_dim, 96);
/// assert_eq!(cfg.depths, vec![2, 2, 6, 2]);
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwinConfig {
    /// Input image size (height = width). Default: 224.
    pub image_size: usize,
    /// Patch partition size. Default: 4.
    pub patch_size: usize,
    /// Number of input channels. Default: 3.
    pub num_channels: usize,
    /// Base embedding dimension (C). Channel dimension doubles each stage.
    pub embed_dim: usize,
    /// Number of transformer blocks per stage.
    /// E.g. `[2, 2, 6, 2]` for Swin-Tiny.
    pub depths: Vec<usize>,
    /// Number of attention heads per stage.
    /// E.g. `[3, 6, 12, 24]` for Swin-Tiny.
    pub num_heads: Vec<usize>,
    /// Local attention window size (window_size × window_size). Default: 7.
    pub window_size: usize,
    /// MLP expansion ratio. Default: 4.0.
    pub mlp_ratio: f32,
    /// Add bias to QKV projections. Default: true.
    pub qkv_bias: bool,
    /// Dropout applied to hidden states. Default: 0.0.
    pub drop_rate: f32,
    /// Dropout applied to attention weights. Default: 0.0.
    pub attn_drop_rate: f32,
    /// Stochastic depth drop-path rate (max over all blocks). Default: 0.1.
    pub drop_path_rate: f32,
    /// Number of output classes. Default: 1000.
    pub num_labels: usize,
    /// Layer-norm epsilon. Default: 1e-5.
    pub layer_norm_eps: f32,
}

impl Default for SwinConfig {
    fn default() -> Self {
        Self {
            image_size: 224,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.1,
            num_labels: 1000,
            layer_norm_eps: 1e-5,
        }
    }
}

impl Config for SwinConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.patch_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "patch_size",
                "patch_size must be greater than 0",
            ));
        }

        if self.image_size % self.patch_size != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "image_size",
                "image_size must be divisible by patch_size",
            ));
        }

        if self.depths.len() != self.num_heads.len() {
            return Err(trustformers_core::errors::invalid_config(
                "depths / num_heads",
                "depths and num_heads must have the same length",
            ));
        }

        if self.depths.is_empty() {
            return Err(trustformers_core::errors::invalid_config(
                "depths",
                "depths must not be empty",
            ));
        }

        if self.window_size == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "window_size",
                "window_size must be greater than 0",
            ));
        }

        if self.embed_dim == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "embed_dim",
                "embed_dim must be greater than 0",
            ));
        }

        // Check head divisibility at each stage
        for (stage, (&heads, &_depth)) in
            self.num_heads.iter().zip(self.depths.iter()).enumerate()
        {
            let dim = self.stage_dim(stage);
            if dim % heads != 0 {
                return Err(trustformers_core::errors::invalid_config(
                    "num_heads",
                    format!(
                        "Stage {} channel dim {} is not divisible by num_heads {}",
                        stage, dim, heads
                    ),
                ));
            }
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Swin"
    }
}

impl SwinConfig {
    /// Number of transformer stages (equals `depths.len()`).
    pub fn num_stages(&self) -> usize {
        self.depths.len()
    }

    /// Channel dimension at a given stage index (0-based).
    ///
    /// Doubles from `embed_dim` each stage: C, 2C, 4C, 8C, …
    pub fn stage_dim(&self, stage: usize) -> usize {
        self.embed_dim * (1 << stage)
    }

    /// Final (deepest) channel dimension, equal to `embed_dim * 2^(num_stages-1)`.
    pub fn final_dim(&self) -> usize {
        self.stage_dim(self.num_stages().saturating_sub(1))
    }

    /// Initial spatial resolution after patch partition: `image_size / patch_size`.
    pub fn initial_resolution(&self) -> usize {
        self.image_size / self.patch_size
    }

    /// Swin-Tiny — 28M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::swin::SwinConfig;
    ///
    /// let cfg = SwinConfig::swin_tiny_patch4_window7_224();
    /// assert_eq!(cfg.embed_dim, 96);
    /// ```
    pub fn swin_tiny_patch4_window7_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 6, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.2,
            num_labels: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    /// Swin-Small — 50M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::swin::SwinConfig;
    ///
    /// let cfg = SwinConfig::swin_small_patch4_window7_224();
    /// assert_eq!(cfg.depths, vec![2, 2, 18, 2]);
    /// ```
    pub fn swin_small_patch4_window7_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 96,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![3, 6, 12, 24],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.3,
            num_labels: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    /// Swin-Base — 88M parameters, 224×224 input.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::swin::SwinConfig;
    ///
    /// let cfg = SwinConfig::swin_base_patch4_window7_224();
    /// assert_eq!(cfg.embed_dim, 128);
    /// ```
    pub fn swin_base_patch4_window7_224() -> Self {
        Self {
            image_size: 224,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 7,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.5,
            num_labels: 1000,
            layer_norm_eps: 1e-5,
        }
    }

    /// Swin-Base at 384×384 with 12×12 windows — 88M parameters.
    ///
    /// # Example
    ///
    /// ```rust
    /// use trustformers_models::swin::SwinConfig;
    ///
    /// let cfg = SwinConfig::swin_base_patch4_window12_384();
    /// assert_eq!(cfg.image_size, 384);
    /// assert_eq!(cfg.window_size, 12);
    /// ```
    pub fn swin_base_patch4_window12_384() -> Self {
        Self {
            image_size: 384,
            patch_size: 4,
            num_channels: 3,
            embed_dim: 128,
            depths: vec![2, 2, 18, 2],
            num_heads: vec![4, 8, 16, 32],
            window_size: 12,
            mlp_ratio: 4.0,
            qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.5,
            num_labels: 1000,
            layer_norm_eps: 1e-5,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trustformers_core::traits::Config;

    #[test]
    fn test_default_is_swin_tiny() {
        let cfg = SwinConfig::default();
        assert_eq!(cfg.embed_dim, 96);
        assert_eq!(cfg.depths, vec![2, 2, 6, 2]);
        assert_eq!(cfg.num_heads, vec![3, 6, 12, 24]);
    }

    #[test]
    fn test_tiny_preset() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        assert_eq!(cfg.image_size, 224);
        assert_eq!(cfg.patch_size, 4);
        assert_eq!(cfg.embed_dim, 96);
        assert_eq!(cfg.window_size, 7);
        assert_eq!(cfg.depths.len(), 4);
    }

    #[test]
    fn test_small_preset() {
        let cfg = SwinConfig::swin_small_patch4_window7_224();
        assert_eq!(cfg.depths, vec![2, 2, 18, 2]);
        assert_eq!(cfg.embed_dim, 96);
    }

    #[test]
    fn test_base_preset() {
        let cfg = SwinConfig::swin_base_patch4_window7_224();
        assert_eq!(cfg.embed_dim, 128);
        assert_eq!(cfg.depths, vec![2, 2, 18, 2]);
        assert_eq!(cfg.num_heads, vec![4, 8, 16, 32]);
    }

    #[test]
    fn test_base_384_preset() {
        let cfg = SwinConfig::swin_base_patch4_window12_384();
        assert_eq!(cfg.image_size, 384);
        assert_eq!(cfg.window_size, 12);
        assert_eq!(cfg.embed_dim, 128);
    }

    #[test]
    fn test_num_stages_four() {
        assert_eq!(SwinConfig::default().num_stages(), 4);
    }

    #[test]
    fn test_stage_dim_doubling_tiny() {
        let cfg = SwinConfig::swin_tiny_patch4_window7_224();
        assert_eq!(cfg.stage_dim(0), 96);
        assert_eq!(cfg.stage_dim(1), 192);
        assert_eq!(cfg.stage_dim(2), 384);
        assert_eq!(cfg.stage_dim(3), 768);
    }

    #[test]
    fn test_final_dim_tiny() {
        // 96 * 2^3 = 768
        assert_eq!(SwinConfig::swin_tiny_patch4_window7_224().final_dim(), 768);
    }

    #[test]
    fn test_final_dim_base() {
        // 128 * 2^3 = 1024
        assert_eq!(SwinConfig::swin_base_patch4_window7_224().final_dim(), 1024);
    }

    #[test]
    fn test_initial_resolution_224() {
        // 224 / 4 = 56
        assert_eq!(SwinConfig::swin_tiny_patch4_window7_224().initial_resolution(), 56);
    }

    #[test]
    fn test_initial_resolution_384() {
        // 384 / 4 = 96
        assert_eq!(SwinConfig::swin_base_patch4_window12_384().initial_resolution(), 96);
    }

    #[test]
    fn test_architecture_label() {
        assert_eq!(SwinConfig::default().architecture(), "Swin");
    }

    #[test]
    fn test_validate_tiny_ok() {
        assert!(SwinConfig::swin_tiny_patch4_window7_224().validate().is_ok());
    }

    #[test]
    fn test_validate_base_384_ok() {
        assert!(SwinConfig::swin_base_patch4_window12_384().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_patch_size() {
        let mut cfg = SwinConfig::default();
        cfg.patch_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_image_not_divisible_by_patch() {
        let mut cfg = SwinConfig::default();
        cfg.image_size = 225;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_depths_heads_length_mismatch() {
        let mut cfg = SwinConfig::default();
        cfg.depths = vec![2, 2, 6];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_empty_depths() {
        let mut cfg = SwinConfig::default();
        cfg.depths = vec![];
        cfg.num_heads = vec![];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_window_size() {
        let mut cfg = SwinConfig::default();
        cfg.window_size = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_embed_dim() {
        let mut cfg = SwinConfig::default();
        cfg.embed_dim = 0;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_depths_and_heads() {
        let cfg = SwinConfig::swin_small_patch4_window7_224();
        let cloned = cfg.clone();
        assert_eq!(cfg.depths, cloned.depths);
        assert_eq!(cfg.num_heads, cloned.num_heads);
    }

    #[test]
    fn test_lcg_varied_embed_dims() {
        let mut s = 79u64;
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let factor = ((s % 4) + 1) as usize;
            let embed_dim = 32 * factor;
            let mut cfg = SwinConfig::default();
            cfg.embed_dim = embed_dim;
            cfg.num_heads = vec![factor, factor * 2, factor * 4, factor * 8];
            assert!(cfg.validate().is_ok(), "embed={embed_dim} failed");
        }
    }
}
