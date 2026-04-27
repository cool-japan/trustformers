/// Configuration for the Stable Diffusion 3 text encoder pipeline.
///
/// SD3 uses three text encoders:
///   - CLIP-L (clip_*) — 77-token CLIP text encoder (hidden_size=768)
///   - CLIP-G (clip_g_*) — 77-token CLIP text encoder (hidden_size=1280)
///   - T5-XXL (t5_*) — 256-token T5 encoder (hidden_size=4096)
///
/// The pooled embeddings are the concatenation of CLIP-L and CLIP-G last-layer
/// `[EOS]` token outputs (768 + 1280 = 2048).
/// The cross-attention conditioning comes from the T5 encoder output.
#[derive(Debug, Clone)]
pub struct Sd3Config {
    // ---- T5-XXL text encoder ----
    /// T5 vocabulary size (default 32128)
    pub t5_vocab_size: usize,
    /// T5 hidden dimension (default 4096)
    pub t5_hidden_size: usize,
    /// T5 number of encoder layers (default 24)
    pub t5_num_layers: usize,
    /// T5 number of attention heads (default 64)
    pub t5_num_heads: usize,
    /// T5 FFN intermediate size (default 10240)
    pub t5_intermediate_size: usize,
    /// T5 relative attention bucketing count (default 32)
    pub t5_relative_attn_buckets: usize,
    /// T5 maximum distance for relative position bias (default 128)
    pub t5_max_distance: usize,

    // ---- CLIP-L encoder ----
    /// CLIP vocabulary size (default 49408, same for L and G)
    pub clip_vocab_size: usize,
    /// CLIP-L hidden dimension (default 768)
    pub clip_hidden_size: usize,
    /// CLIP-L number of transformer layers (default 12)
    pub clip_num_layers: usize,
    /// CLIP-L number of attention heads (default 12)
    pub clip_num_heads: usize,
    /// CLIP-L MLP intermediate size (default 3072)
    pub clip_intermediate_size: usize,

    // ---- CLIP-G encoder ----
    /// CLIP-G hidden dimension (default 1280)
    pub clip_g_hidden_size: usize,
    /// CLIP-G number of transformer layers (default 32)
    pub clip_g_num_layers: usize,
    /// CLIP-G number of attention heads (default 20)
    pub clip_g_num_heads: usize,

    // ---- SD3 combined ----
    /// T5 output embedding dimension used for cross-attention (default 4096)
    pub text_embedding_dim: usize,
    /// Pooled embedding dimension: CLIP-L hidden + CLIP-G hidden (default 2048)
    pub pooled_embedding_dim: usize,
    /// Maximum token sequence length for CLIP encoders (default 77)
    pub max_sequence_length: usize,
    /// Maximum token sequence length for T5 encoder (default 256)
    pub max_t5_sequence_length: usize,
}

impl Default for Sd3Config {
    fn default() -> Self {
        Self {
            t5_vocab_size: 32128,
            t5_hidden_size: 4096,
            t5_num_layers: 24,
            t5_num_heads: 64,
            t5_intermediate_size: 10240,
            t5_relative_attn_buckets: 32,
            t5_max_distance: 128,
            clip_vocab_size: 49408,
            clip_hidden_size: 768,
            clip_num_layers: 12,
            clip_num_heads: 12,
            clip_intermediate_size: 3072,
            clip_g_hidden_size: 1280,
            clip_g_num_layers: 32,
            clip_g_num_heads: 20,
            text_embedding_dim: 4096,
            pooled_embedding_dim: 2048, // 768 + 1280
            max_sequence_length: 77,
            max_t5_sequence_length: 256,
        }
    }
}

impl Sd3Config {
    /// Validate the configuration for consistency.
    pub fn validate(&self) -> Result<(), Sd3ConfigError> {
        if self.t5_hidden_size == 0 {
            return Err(Sd3ConfigError::InvalidField(
                "t5_hidden_size must be > 0".to_string(),
            ));
        }
        if self.t5_num_heads == 0 {
            return Err(Sd3ConfigError::InvalidField(
                "t5_num_heads must be > 0".to_string(),
            ));
        }
        if !self.t5_hidden_size.is_multiple_of(self.t5_num_heads) {
            return Err(Sd3ConfigError::InvalidField(format!(
                "t5_hidden_size ({}) must be divisible by t5_num_heads ({})",
                self.t5_hidden_size, self.t5_num_heads
            )));
        }
        if self.clip_hidden_size == 0 {
            return Err(Sd3ConfigError::InvalidField(
                "clip_hidden_size must be > 0".to_string(),
            ));
        }
        if self.clip_num_heads == 0 {
            return Err(Sd3ConfigError::InvalidField(
                "clip_num_heads must be > 0".to_string(),
            ));
        }
        if !self.clip_hidden_size.is_multiple_of(self.clip_num_heads) {
            return Err(Sd3ConfigError::InvalidField(format!(
                "clip_hidden_size ({}) must be divisible by clip_num_heads ({})",
                self.clip_hidden_size, self.clip_num_heads
            )));
        }
        if !self.clip_g_hidden_size.is_multiple_of(self.clip_g_num_heads) {
            return Err(Sd3ConfigError::InvalidField(format!(
                "clip_g_hidden_size ({}) must be divisible by clip_g_num_heads ({})",
                self.clip_g_hidden_size, self.clip_g_num_heads
            )));
        }
        let expected_pooled = self.clip_hidden_size + self.clip_g_hidden_size;
        if self.pooled_embedding_dim != expected_pooled {
            return Err(Sd3ConfigError::InvalidField(format!(
                "pooled_embedding_dim ({}) must equal clip_hidden_size ({}) + clip_g_hidden_size ({})",
                self.pooled_embedding_dim, self.clip_hidden_size, self.clip_g_hidden_size
            )));
        }
        if self.text_embedding_dim != self.t5_hidden_size {
            return Err(Sd3ConfigError::InvalidField(format!(
                "text_embedding_dim ({}) must equal t5_hidden_size ({})",
                self.text_embedding_dim, self.t5_hidden_size
            )));
        }
        if self.t5_relative_attn_buckets == 0 {
            return Err(Sd3ConfigError::InvalidField(
                "t5_relative_attn_buckets must be > 0".to_string(),
            ));
        }
        Ok(())
    }

    /// Compute the T5 per-head dimension.
    pub fn t5_head_dim(&self) -> usize {
        self.t5_hidden_size / self.t5_num_heads
    }

    /// Compute the CLIP-L per-head dimension.
    pub fn clip_head_dim(&self) -> usize {
        self.clip_hidden_size / self.clip_num_heads
    }

    /// Compute the CLIP-G per-head dimension.
    pub fn clip_g_head_dim(&self) -> usize {
        self.clip_g_hidden_size / self.clip_g_num_heads
    }
}

/// Errors arising from SD3 configuration validation.
#[derive(Debug, thiserror::Error)]
pub enum Sd3ConfigError {
    #[error("Invalid configuration field: {0}")]
    InvalidField(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_t5_fields() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.t5_vocab_size, 32128);
        assert_eq!(cfg.t5_hidden_size, 4096);
        assert_eq!(cfg.t5_num_layers, 24);
        assert_eq!(cfg.t5_num_heads, 64);
        assert_eq!(cfg.t5_relative_attn_buckets, 32);
    }

    #[test]
    fn test_default_clip_l_fields() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.clip_vocab_size, 49408);
        assert_eq!(cfg.clip_hidden_size, 768);
        assert_eq!(cfg.clip_num_layers, 12);
        assert_eq!(cfg.clip_num_heads, 12);
    }

    #[test]
    fn test_default_clip_g_fields() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.clip_g_hidden_size, 1280);
        assert_eq!(cfg.clip_g_num_layers, 32);
        assert_eq!(cfg.clip_g_num_heads, 20);
    }

    #[test]
    fn test_default_combined_fields() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.pooled_embedding_dim, 2048);
        assert_eq!(cfg.max_sequence_length, 77);
        assert_eq!(cfg.max_t5_sequence_length, 256);
    }

    #[test]
    fn test_t5_head_dim() {
        // 4096 / 64 = 64
        assert_eq!(Sd3Config::default().t5_head_dim(), 64);
    }

    #[test]
    fn test_clip_head_dim() {
        // 768 / 12 = 64
        assert_eq!(Sd3Config::default().clip_head_dim(), 64);
    }

    #[test]
    fn test_clip_g_head_dim() {
        // 1280 / 20 = 64
        assert_eq!(Sd3Config::default().clip_g_head_dim(), 64);
    }

    #[test]
    fn test_pooled_dim_equals_clip_l_plus_clip_g() {
        let cfg = Sd3Config::default();
        assert_eq!(
            cfg.pooled_embedding_dim,
            cfg.clip_hidden_size + cfg.clip_g_hidden_size
        );
    }

    #[test]
    fn test_text_embedding_dim_equals_t5_hidden() {
        let cfg = Sd3Config::default();
        assert_eq!(cfg.text_embedding_dim, cfg.t5_hidden_size);
    }

    #[test]
    fn test_validate_default_ok() {
        assert!(Sd3Config::default().validate().is_ok());
    }

    #[test]
    fn test_validate_zero_t5_hidden_size() {
        let cfg = Sd3Config {
            t5_hidden_size: 0,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_t5_heads() {
        let cfg = Sd3Config {
            t5_num_heads: 0,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_t5_hidden_not_divisible_by_heads() {
        let cfg = Sd3Config {
            t5_num_heads: 7,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_clip_hidden_size() {
        let cfg = Sd3Config {
            clip_hidden_size: 0,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_clip_heads() {
        let cfg = Sd3Config {
            clip_num_heads: 0,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_clip_hidden_not_divisible_by_heads() {
        let cfg = Sd3Config {
            clip_num_heads: 7,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_pooled_dim_mismatch() {
        let cfg = Sd3Config {
            pooled_embedding_dim: 2049,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_text_embedding_dim_mismatch() {
        let cfg = Sd3Config {
            text_embedding_dim: 4097,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_validate_zero_relative_attn_buckets() {
        let cfg = Sd3Config {
            t5_relative_attn_buckets: 0,
            ..Sd3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_clone_preserves_fields() {
        let cfg = Sd3Config::default();
        let cloned = cfg.clone();
        assert_eq!(cfg.t5_hidden_size, cloned.t5_hidden_size);
        assert_eq!(cfg.clip_hidden_size, cloned.clip_hidden_size);
        assert_eq!(cfg.pooled_embedding_dim, cloned.pooled_embedding_dim);
    }

    #[test]
    fn test_lcg_varied_t5_heads() {
        let mut s = 109u64;
        let head_options = [32usize, 64];
        for _ in 0..4 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let idx = (s % head_options.len() as u64) as usize;
            let heads = head_options[idx];
            let hidden = heads * 64;
            let cfg = Sd3Config {
                t5_hidden_size: hidden,
                t5_num_heads: heads,
                text_embedding_dim: hidden,
                ..Sd3Config::default()
            };
            assert!(cfg.validate().is_ok(), "t5_heads={heads} failed");
        }
    }
}
