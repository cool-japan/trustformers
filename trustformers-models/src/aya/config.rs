use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Errors for Aya model configuration.
#[derive(Debug, Error)]
pub enum AyaError {
    /// Configuration validation failed.
    #[error("invalid config: {0}")]
    InvalidConfig(String),
    /// Dimension mismatch during computation.
    #[error("dimension mismatch: expected {expected}, got {got}")]
    DimensionMismatch { expected: usize, got: usize },
    /// Empty input provided.
    #[error("empty input provided")]
    EmptyInput,
    /// Requested language is not supported.
    #[error("language '{0}' is not supported by this model")]
    UnsupportedLanguage(String),
}

/// ISO 639-1 language codes supported by Aya-23 (23 languages).
pub const AYA_23_LANGUAGE_CODES: [&str; 23] = [
    "ar", "zh", "cs", "nl", "en", "fr", "de", "el", "he", "hi", "id", "it", "ja", "ko", "pl", "pt",
    "ro", "ru", "es", "tr", "uk", "vi", "fi",
];

/// Configuration for the Aya-23 multilingual model by Cohere.
///
/// Aya-23 is based on the Command-R architecture but with a large multilingual
/// vocabulary (256K tokens) covering 23 languages, 128K context window, and
/// `LayerNorm` (rather than RMSNorm) in the attention blocks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AyaConfig {
    /// Vocabulary size — very large to cover 23 languages (default 256000).
    pub vocab_size: usize,
    /// Hidden dimension size (default 4096).
    pub hidden_size: usize,
    /// Intermediate (FFN) size (default 16384).
    pub intermediate_size: usize,
    /// Number of transformer decoder layers (default 32).
    pub num_hidden_layers: usize,
    /// Number of query attention heads (default 32).
    pub num_attention_heads: usize,
    /// Number of key/value heads for GQA (default 8).
    pub num_key_value_heads: usize,
    /// Dimension of each attention head (default 128).
    pub head_dim: usize,
    /// Maximum sequence length — 128K context (default 131072).
    pub max_position_embeddings: usize,
    /// Epsilon for LayerNorm (default 1e-5).
    pub layer_norm_eps: f64,
    /// RoPE base frequency (default 10000.0).
    pub rope_theta: f64,
    /// Final logit scale factor (default 0.0625 = 1/16).
    pub logit_scale: f32,
    /// Whether to apply QK-normalization inside attention (default false).
    pub use_qk_norm: bool,
    /// Whether to tie input/output embedding weights (default false).
    pub tie_word_embeddings: bool,
    /// Dropout probability for attention weights (default 0.0).
    pub attention_dropout: f32,
    /// ISO 639-1 codes for the 23 supported languages.
    pub supported_languages: Vec<String>,
    /// HuggingFace tokenizer class name.
    pub tokenizer_class: String,
}

impl Default for AyaConfig {
    fn default() -> Self {
        let supported_languages: Vec<String> =
            AYA_23_LANGUAGE_CODES.iter().map(|s| s.to_string()).collect();

        let hidden_size: usize = 4096;
        let num_attention_heads: usize = 32;

        Self {
            vocab_size: 256000,
            hidden_size,
            intermediate_size: 16384,
            num_hidden_layers: 32,
            num_attention_heads,
            num_key_value_heads: 8,
            head_dim: hidden_size / num_attention_heads, // 128
            max_position_embeddings: 131072,
            layer_norm_eps: 1e-5,
            rope_theta: 10000.0,
            logit_scale: 0.0625,
            use_qk_norm: false,
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            supported_languages,
            tokenizer_class: "PreTrainedTokenizer".to_string(),
        }
    }
}

impl AyaConfig {
    /// Validate the configuration, returning an [`AyaError`] when any
    /// constraint is violated.
    pub fn validate(&self) -> Result<(), AyaError> {
        if self.vocab_size == 0 {
            return Err(AyaError::InvalidConfig(
                "vocab_size must be greater than 0".to_string(),
            ));
        }
        if self.hidden_size == 0 {
            return Err(AyaError::InvalidConfig(
                "hidden_size must be greater than 0".to_string(),
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(AyaError::InvalidConfig(
                "num_attention_heads must be greater than 0".to_string(),
            ));
        }
        if self.num_key_value_heads == 0 {
            return Err(AyaError::InvalidConfig(
                "num_key_value_heads must be greater than 0".to_string(),
            ));
        }
        let expected_head_dim = self.hidden_size / self.num_attention_heads;
        if self.head_dim != expected_head_dim {
            return Err(AyaError::InvalidConfig(format!(
                "head_dim ({}) must equal hidden_size ({}) / num_attention_heads ({}), i.e. {}",
                self.head_dim, self.hidden_size, self.num_attention_heads, expected_head_dim,
            )));
        }
        if !self.num_attention_heads.is_multiple_of(self.num_key_value_heads) {
            return Err(AyaError::InvalidConfig(format!(
                "num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                self.num_attention_heads, self.num_key_value_heads,
            )));
        }
        if self.num_hidden_layers == 0 {
            return Err(AyaError::InvalidConfig(
                "num_hidden_layers must be greater than 0".to_string(),
            ));
        }
        if self.intermediate_size == 0 {
            return Err(AyaError::InvalidConfig(
                "intermediate_size must be greater than 0".to_string(),
            ));
        }
        if self.max_position_embeddings == 0 {
            return Err(AyaError::InvalidConfig(
                "max_position_embeddings must be greater than 0".to_string(),
            ));
        }
        if self.supported_languages.is_empty() {
            return Err(AyaError::InvalidConfig(
                "supported_languages must not be empty".to_string(),
            ));
        }
        Ok(())
    }

    /// Return the number of supported languages.
    pub fn supported_language_count(&self) -> usize {
        self.supported_languages.len()
    }

    /// Return `true` when `lang_code` is in the supported language list.
    pub fn supports_language(&self, lang_code: &str) -> bool {
        self.supported_languages.iter().any(|l| l.as_str() == lang_code)
    }

    /// Number of query groups (num_attention_heads / num_key_value_heads).
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Whether grouped-query attention is in use.
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads != self.num_attention_heads
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- LCG ---
    fn lcg_next(state: &mut u64) -> u64 {
        *state = state.wrapping_mul(6364136223846793005u64).wrapping_add(1442695040888963407u64);
        *state
    }

    fn lcg_usize(state: &mut u64, max: usize) -> usize {
        (lcg_next(state) >> 33) as usize % max
    }

    // --- Default (Aya-23-8B) ---

    #[test]
    fn test_aya_default_vocab_size() {
        let cfg = AyaConfig::default();
        assert_eq!(cfg.vocab_size, 256000, "Aya-23 vocab_size must be 256000");
    }

    #[test]
    fn test_aya_default_hidden_size() {
        let cfg = AyaConfig::default();
        assert_eq!(
            cfg.hidden_size, 4096,
            "Aya-23 default hidden_size must be 4096"
        );
    }

    #[test]
    fn test_aya_default_num_attention_heads() {
        let cfg = AyaConfig::default();
        assert_eq!(
            cfg.num_attention_heads, 32,
            "Aya-23 must have 32 attention heads"
        );
    }

    #[test]
    fn test_aya_default_num_key_value_heads() {
        let cfg = AyaConfig::default();
        assert_eq!(cfg.num_key_value_heads, 8, "Aya-23 must have 8 KV heads");
    }

    #[test]
    fn test_aya_default_num_hidden_layers() {
        let cfg = AyaConfig::default();
        assert_eq!(cfg.num_hidden_layers, 32);
    }

    #[test]
    fn test_aya_default_head_dim() {
        let cfg = AyaConfig::default();
        let expected = cfg.hidden_size / cfg.num_attention_heads;
        assert_eq!(
            cfg.head_dim, expected,
            "head_dim must equal hidden_size / num_attention_heads"
        );
        assert_eq!(cfg.head_dim, 128);
    }

    #[test]
    fn test_aya_default_max_position_embeddings() {
        let cfg = AyaConfig::default();
        assert_eq!(
            cfg.max_position_embeddings, 131072,
            "Aya-23 context window is 128K"
        );
    }

    #[test]
    fn test_aya_default_rope_theta() {
        let cfg = AyaConfig::default();
        assert_eq!(cfg.rope_theta, 10000.0);
    }

    #[test]
    fn test_aya_default_supported_languages_count() {
        let cfg = AyaConfig::default();
        assert_eq!(
            cfg.supported_language_count(),
            23,
            "Aya-23 must support exactly 23 languages"
        );
    }

    #[test]
    fn test_aya_default_logit_scale() {
        let cfg = AyaConfig::default();
        assert!((cfg.logit_scale - 0.0625_f32).abs() < 1e-6);
    }

    // --- GQA ---

    #[test]
    fn test_aya_gqa_num_query_groups() {
        let cfg = AyaConfig::default();
        // 32 / 8 = 4 groups
        assert_eq!(
            cfg.num_query_groups(),
            4,
            "GQA groups must be num_heads / num_kv_heads"
        );
    }

    #[test]
    fn test_aya_is_gqa_true() {
        let cfg = AyaConfig::default();
        assert!(cfg.is_gqa(), "Default Aya config uses GQA");
    }

    #[test]
    fn test_aya_is_gqa_false_when_kv_equals_heads() {
        let cfg = AyaConfig {
            num_key_value_heads: 32,
            ..AyaConfig::default()
        };
        assert!(!cfg.is_gqa());
    }

    // --- Language support ---

    #[test]
    fn test_aya_supports_english() {
        let cfg = AyaConfig::default();
        assert!(cfg.supports_language("en"), "Aya-23 must support English");
    }

    #[test]
    fn test_aya_supports_arabic() {
        let cfg = AyaConfig::default();
        assert!(cfg.supports_language("ar"), "Aya-23 must support Arabic");
    }

    #[test]
    fn test_aya_does_not_support_unknown_language() {
        let cfg = AyaConfig::default();
        assert!(!cfg.supports_language("xx"), "Aya-23 must not support 'xx'");
    }

    #[test]
    fn test_aya_all_23_language_codes_are_supported() {
        let cfg = AyaConfig::default();
        for code in &AYA_23_LANGUAGE_CODES {
            assert!(
                cfg.supports_language(code),
                "Aya-23 must support language code '{}'",
                code
            );
        }
    }

    // --- Validation ---

    #[test]
    fn test_aya_validate_passes_for_default() {
        let cfg = AyaConfig::default();
        assert!(
            cfg.validate().is_ok(),
            "Default AyaConfig must pass validation"
        );
    }

    #[test]
    fn test_aya_validate_fails_zero_vocab_size() {
        let cfg = AyaConfig {
            vocab_size: 0,
            ..AyaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_aya_validate_fails_zero_hidden_size() {
        let cfg = AyaConfig {
            hidden_size: 0,
            ..AyaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_aya_validate_fails_when_head_dim_wrong() {
        // Change head_dim without matching hidden_size / num_attention_heads
        let cfg = AyaConfig {
            head_dim: 64,
            ..AyaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_aya_validate_fails_when_kv_heads_not_divisor_of_heads() {
        let cfg = AyaConfig {
            num_key_value_heads: 7,
            ..AyaConfig::default()
        };
        assert!(cfg.validate().is_err());
    }

    // --- LCG sanity ---

    #[test]
    fn test_lcg_produces_values_in_range() {
        let mut state: u64 = 999;
        for _ in 0..50 {
            let v = lcg_usize(&mut state, 100);
            assert!(v < 100, "lcg_usize must produce values in [0, max)");
        }
    }
}
