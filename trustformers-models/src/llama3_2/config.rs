use serde::{Deserialize, Serialize};
use std::fmt;
use trustformers_core::traits::Config;

// ─────────────────────────────────────────────────────────────────────────────
// Error type
// ─────────────────────────────────────────────────────────────────────────────

/// Errors specific to Llama-3.2 configuration and inference
#[derive(Debug)]
pub enum Llama32Error {
    /// A configuration value is invalid
    InvalidConfig(String),
    /// Vision-encoder shape mismatch
    VisionShapeMismatch { expected: usize, got: usize },
    /// Pixel-value buffer has the wrong size
    PixelBufferSize { expected: usize, got: usize },
    /// Cross-attention layer index out of range
    CrossAttentionIndexOutOfRange { index: usize, num_layers: usize },
    /// General tensor operation error
    TensorOp(String),
    /// Not implemented
    NotImplemented(String),
}

impl fmt::Display for Llama32Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidConfig(msg) => write!(f, "Llama32 invalid config: {msg}"),
            Self::VisionShapeMismatch { expected, got } => {
                write!(f, "vision shape mismatch: expected {expected}, got {got}")
            },
            Self::PixelBufferSize { expected, got } => {
                write!(f, "pixel buffer size: expected {expected}, got {got}")
            },
            Self::CrossAttentionIndexOutOfRange { index, num_layers } => {
                write!(
                    f,
                    "cross-attention layer index {index} out of range for {num_layers} layers"
                )
            },
            Self::TensorOp(msg) => write!(f, "tensor op error: {msg}"),
            Self::NotImplemented(msg) => write!(f, "not implemented: {msg}"),
        }
    }
}

impl std::error::Error for Llama32Error {}

// ─────────────────────────────────────────────────────────────────────────────
// Llama-3.2 Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// Configuration for Llama-3.2 multimodal (vision) models.
///
/// Llama-3.2 introduced multimodal variants that combine a ViT-style vision
/// encoder with a Llama-3 text backbone.  Cross-attention layers (injected at
/// every 4th decoder layer by default) allow the text decoder to attend to
/// visual features.  The 3B variant uses LongRoPE scaling (`use_scaled_rope =
/// true`, `rope_scaling_factor = 32.0`) to extend the effective context to
/// 131 072 tokens.
///
/// # References
/// * "Llama 3.2: Revolutionizing Edge AI and Vision" (Meta AI blog, Sept 2024)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Llama32Config {
    // ── Text backbone (same as Llama-3) ──────────────────────────────────────
    /// Vocabulary size (128 256 shared Tiktoken vocabulary)
    pub vocab_size: usize,
    /// Hidden dimension of the text backbone
    pub hidden_size: usize,
    /// SwiGLU intermediate dimension
    pub intermediate_size: usize,
    /// Number of transformer decoder layers
    pub num_hidden_layers: usize,
    /// Number of query attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (GQA)
    pub num_key_value_heads: usize,
    /// Per-head dimension (`hidden_size / num_attention_heads`)
    pub head_dim: usize,
    /// Maximum position embeddings (131 072 with LongRoPE)
    pub max_position_embeddings: usize,
    /// RMSNorm epsilon
    pub rms_norm_eps: f64,
    /// RoPE base frequency
    pub rope_theta: f64,
    /// LongRoPE scaling factor (32.0 for 3.2 models)
    pub rope_scaling_factor: f32,
    /// Whether to use scaled (LongRoPE) rotary embeddings
    pub use_scaled_rope: bool,

    // ── Vision encoder ───────────────────────────────────────────────────────
    /// Vision encoder hidden dimension
    pub vision_hidden_size: usize,
    /// Vision encoder attention heads
    pub vision_num_attention_heads: usize,
    /// Vision encoder transformer layers
    pub vision_num_hidden_layers: usize,
    /// Vision encoder MLP intermediate dimension
    pub vision_intermediate_size: usize,
    /// Input image size in pixels (assumed square)
    pub image_size: usize,
    /// Vision patch size in pixels
    pub patch_size: usize,
    /// Number of image patches (`(image_size / patch_size)²`)
    pub num_patches: usize,
    /// Projected vision output dimension fed to cross-attention
    /// (`6 * hidden_size` for the 3B model)
    pub vision_output_dim: usize,

    // ── Cross-attention ───────────────────────────────────────────────────────
    /// Indices of text-decoder layers that have cross-attention (every 4th by default)
    pub cross_attention_layers: Vec<usize>,
    /// Number of global cross-attention layers (8 for 3.2-3B)
    pub num_global_layers: usize,
}

impl Default for Llama32Config {
    /// Returns the Llama-3.2-3B configuration
    fn default() -> Self {
        Self::llama32_3b()
    }
}

impl Config for Llama32Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        self.validate_internal().map_err(|e| {
            trustformers_core::errors::TrustformersError::invalid_config(e.to_string())
        })
    }

    fn architecture(&self) -> &'static str {
        "Llama-3.2"
    }
}

impl Llama32Config {
    /// Llama-3.2-3B default configuration
    pub fn llama32_3b() -> Self {
        let num_hidden_layers = 28;
        let image_size = 560;
        let patch_size = 14;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let hidden_size = 3072_usize;
        let cross_attention_layers = Self::default_cross_attention_layers(num_hidden_layers);
        Self {
            vocab_size: 128256,
            hidden_size,
            intermediate_size: 8192,
            num_hidden_layers,
            num_attention_heads: 24,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling_factor: 32.0,
            use_scaled_rope: true,
            vision_hidden_size: 1280,
            vision_num_attention_heads: 16,
            vision_num_hidden_layers: 32,
            vision_intermediate_size: 5120,
            image_size,
            patch_size,
            num_patches,
            vision_output_dim: 6 * hidden_size, // 18432
            cross_attention_layers,
            num_global_layers: 8,
        }
    }

    /// Llama-3.2-11B configuration
    ///
    /// 11B uses a larger text backbone (hidden=4096, 32 Q-heads, 8 KV-heads)
    /// while keeping the same ViT-H vision encoder.
    pub fn llama32_11b() -> Self {
        let num_hidden_layers = 32;
        let image_size = 560;
        let patch_size = 14;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let hidden_size = 4096_usize;
        let cross_attention_layers = Self::default_cross_attention_layers(num_hidden_layers);
        Self {
            vocab_size: 128256,
            hidden_size,
            intermediate_size: 14336,
            num_hidden_layers,
            num_attention_heads: 32,
            num_key_value_heads: 8,
            head_dim: 128,
            max_position_embeddings: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling_factor: 32.0,
            use_scaled_rope: true,
            vision_hidden_size: 1280,
            vision_num_attention_heads: 16,
            vision_num_hidden_layers: 32,
            vision_intermediate_size: 5120,
            image_size,
            patch_size,
            num_patches,
            vision_output_dim: 6 * hidden_size, // 24576
            cross_attention_layers,
            num_global_layers: 8,
        }
    }

    /// Small configuration for unit tests (fast, minimal memory)
    pub fn small_test() -> Self {
        let num_hidden_layers = 4;
        let image_size = 28;
        let patch_size = 14;
        let num_patches = (image_size / patch_size) * (image_size / patch_size);
        let hidden_size = 64_usize;
        let cross_attention_layers = Self::default_cross_attention_layers(num_hidden_layers);
        Self {
            vocab_size: 256,
            hidden_size,
            intermediate_size: 128,
            num_hidden_layers,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 16,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 500000.0,
            rope_scaling_factor: 32.0,
            use_scaled_rope: true,
            vision_hidden_size: 32,
            vision_num_attention_heads: 2,
            vision_num_hidden_layers: 2,
            vision_intermediate_size: 64,
            image_size,
            patch_size,
            num_patches,
            vision_output_dim: 6 * hidden_size, // 384
            cross_attention_layers,
            num_global_layers: 1,
        }
    }

    /// Compute the number of patches for a given image / patch size.
    ///
    /// `num_patches = (image_size / patch_size)²`
    pub fn num_patches(image_size: usize, patch_size: usize) -> usize {
        let side = image_size / patch_size;
        side * side
    }

    /// Generate the default cross-attention layer indices for `num_layers`
    /// decoder layers: every 4th layer starting from index 3.
    ///
    /// E.g. for 28 layers → [3, 7, 11, 15, 19, 23, 27]
    pub fn default_cross_attention_layers(num_layers: usize) -> Vec<usize> {
        (0..num_layers).filter(|&i| (i + 1) % 4 == 0).collect()
    }

    /// Validate this configuration, returning a typed `Llama32Error` on failure.
    pub fn validate_internal(&self) -> Result<(), Llama32Error> {
        if self.hidden_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "hidden_size must be > 0".to_string(),
            ));
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Llama32Error::InvalidConfig(format!(
                "hidden_size ({}) must be divisible by num_attention_heads ({})",
                self.hidden_size, self.num_attention_heads
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(Llama32Error::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads
            )));
        }
        if self.vocab_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "vocab_size must be > 0".to_string(),
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(Llama32Error::InvalidConfig(
                "num_hidden_layers must be > 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "intermediate_size must be > 0".to_string(),
            ));
        }
        if self.vision_hidden_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "vision_hidden_size must be > 0".to_string(),
            ));
        }
        if self.patch_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "patch_size must be > 0".to_string(),
            ));
        }
        if self.image_size == 0 {
            return Err(Llama32Error::InvalidConfig(
                "image_size must be > 0".to_string(),
            ));
        }
        if !self.image_size.is_multiple_of(self.patch_size) {
            return Err(Llama32Error::InvalidConfig(format!(
                "image_size ({}) must be divisible by patch_size ({})",
                self.image_size, self.patch_size
            )));
        }
        if self.vision_num_attention_heads == 0 {
            return Err(Llama32Error::InvalidConfig(
                "vision_num_attention_heads must be > 0".to_string(),
            ));
        }
        if !self.vision_hidden_size.is_multiple_of(self.vision_num_attention_heads) {
            return Err(Llama32Error::InvalidConfig(format!(
                "vision_hidden_size ({}) must be divisible by vision_num_attention_heads ({})",
                self.vision_hidden_size, self.vision_num_attention_heads
            )));
        }
        // Validate cross-attention layer indices are in range
        for &idx in &self.cross_attention_layers {
            if idx >= self.num_hidden_layers {
                return Err(Llama32Error::CrossAttentionIndexOutOfRange {
                    index: idx,
                    num_layers: self.num_hidden_layers,
                });
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_llama32_default_is_3b() {
        let cfg = Llama32Config::default();
        assert_eq!(cfg.vocab_size, 128256);
        assert_eq!(cfg.hidden_size, 3072);
    }

    #[test]
    fn test_llama32_3b_preset_fields() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.num_hidden_layers, 28);
        assert_eq!(cfg.num_attention_heads, 24);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_llama32_11b_preset_fields() {
        let cfg = Llama32Config::llama32_11b();
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
    }

    #[test]
    fn test_llama32_small_test_config() {
        let cfg = Llama32Config::small_test();
        assert_eq!(cfg.vocab_size, 256);
        assert_eq!(cfg.hidden_size, 64);
        assert_eq!(cfg.num_hidden_layers, 4);
    }

    #[test]
    fn test_llama32_validate_passes_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert!(cfg.validate_internal().is_ok());
    }

    #[test]
    fn test_llama32_validate_passes_11b() {
        let cfg = Llama32Config::llama32_11b();
        assert!(cfg.validate_internal().is_ok());
    }

    #[test]
    fn test_llama32_validate_fails_zero_hidden_size() {
        let cfg = Llama32Config {
            hidden_size: 0,
            ..Llama32Config::small_test()
        };
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_validate_fails_zero_vocab_size() {
        let cfg = Llama32Config {
            vocab_size: 0,
            ..Llama32Config::small_test()
        };
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_validate_fails_hidden_not_divisible_by_heads() {
        let cfg = Llama32Config {
            hidden_size: 63,
            num_attention_heads: 4,
            ..Llama32Config::small_test()
        };
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_validate_fails_heads_not_divisible_by_kv_heads() {
        let cfg = Llama32Config {
            num_attention_heads: 4,
            num_key_value_heads: 3,
            ..Llama32Config::small_test()
        };
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_validate_fails_image_not_divisible_by_patch() {
        let cfg = Llama32Config {
            image_size: 30,
            patch_size: 14,
            ..Llama32Config::small_test()
        };
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_patch_calculation() {
        let n = Llama32Config::num_patches(560, 14);
        assert_eq!(n, 40 * 40);
    }

    #[test]
    fn test_llama32_cross_attention_layers_28() {
        let layers = Llama32Config::default_cross_attention_layers(28);
        assert!(layers.contains(&3));
        assert!(layers.contains(&7));
        assert!(layers.contains(&27));
        assert!(!layers.contains(&0));
    }

    #[test]
    fn test_llama32_max_position_embeddings_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn test_llama32_use_scaled_rope_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert!(cfg.use_scaled_rope);
        assert!((cfg.rope_scaling_factor - 32.0).abs() < 1e-6);
    }

    #[test]
    fn test_llama32_vision_output_dim_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.vision_output_dim, 6 * cfg.hidden_size);
    }

    #[test]
    fn test_llama32_architecture_name() {
        let cfg = Llama32Config::default();
        assert_eq!(cfg.architecture(), "Llama-3.2");
    }

    #[test]
    fn test_llama32_num_global_layers_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.num_global_layers, 8);
    }

    #[test]
    fn test_llama32_vision_heads_3b() {
        let cfg = Llama32Config::llama32_3b();
        assert_eq!(cfg.vision_num_attention_heads, 16);
        assert_eq!(cfg.vision_num_hidden_layers, 32);
    }

    #[test]
    fn test_llama32_validate_fails_cross_attn_out_of_range() {
        let mut cfg = Llama32Config::small_test();
        cfg.cross_attention_layers = vec![100];
        assert!(cfg.validate_internal().is_err());
    }

    #[test]
    fn test_llama32_small_test_validate_passes() {
        let cfg = Llama32Config::small_test();
        assert!(cfg.validate_internal().is_ok());
    }

    #[test]
    fn test_llama32_lcg_values_in_range() {
        let mut s = 42u64;
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        let v = (s % 1000) as f32 / 1000.0;
        assert!((0.0..1.0).contains(&v));
    }

    #[test]
    fn test_llama32_11b_vision_hidden_size() {
        let cfg = Llama32Config::llama32_11b();
        assert_eq!(cfg.vision_hidden_size, 1280);
        assert_eq!(cfg.vision_intermediate_size, 5120);
    }
}
