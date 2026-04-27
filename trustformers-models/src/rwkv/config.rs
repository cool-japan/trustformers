use serde::{Deserialize, Serialize};
use trustformers_core::{
    errors::{invalid_config, Result},
    traits::Config,
};

/// RWKV model configuration
/// Reference: "RWKV: Reinventing RNNs for the Transformer Era" (Peng et al., 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RwkvConfig {
    /// Model dimension (d_model)
    pub n_embd: usize,
    /// Number of layers
    pub n_layer: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Context length
    pub ctx_len: usize,
    /// Model version (e.g., "4", "5", "6")
    pub version: String,
    /// Architecture details
    pub arch_version: String,
    /// Number of attention heads (for compatibility)
    pub n_head: usize,
    /// Head dimension
    pub head_size: usize,
    /// Intermediate dimension in FFN
    pub n_ffn: Option<usize>,
    /// Rescale layer weights
    pub rescale_layer: usize,
    /// Layer normalization epsilon
    pub layer_norm_epsilon: f32,
    /// Bos token ID
    pub bos_token_id: u32,
    /// Eos token ID
    pub eos_token_id: u32,
    /// Pad token ID
    pub pad_token_id: Option<u32>,
    /// Model type identifier
    pub model_type: String,
}

impl Default for RwkvConfig {
    fn default() -> Self {
        Self {
            n_embd: 768,
            n_layer: 12,
            vocab_size: 50277,
            ctx_len: 1024,
            version: "4".to_string(),
            arch_version: "RWKV-4".to_string(),
            n_head: 12,
            head_size: 64,
            n_ffn: None, // Computed as 3.5 * n_embd typically
            rescale_layer: 6,
            layer_norm_epsilon: 1e-5,
            bos_token_id: 0,
            eos_token_id: 0,
            pad_token_id: Some(0),
            model_type: "rwkv".to_string(),
        }
    }
}

impl RwkvConfig {
    /// RWKV-169M configuration (similar to GPT-2 small)
    pub fn rwkv_169m() -> Self {
        Self {
            n_embd: 768,
            n_layer: 12,
            n_head: 12,
            head_size: 64,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// RWKV-430M configuration (similar to GPT-2 medium)
    pub fn rwkv_430m() -> Self {
        Self {
            n_embd: 1024,
            n_layer: 24,
            n_head: 16,
            head_size: 64,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// RWKV-1.5B configuration (similar to GPT-2 large)
    pub fn rwkv_1_5b() -> Self {
        Self {
            n_embd: 1536,
            n_layer: 48,
            n_head: 24,
            head_size: 64,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// RWKV-3B configuration (similar to GPT-2 XL)
    pub fn rwkv_3b() -> Self {
        Self {
            n_embd: 2048,
            n_layer: 32,
            n_head: 32,
            head_size: 64,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// RWKV-7B configuration
    pub fn rwkv_7b() -> Self {
        Self {
            n_embd: 4096,
            n_layer: 32,
            n_head: 32,
            head_size: 128,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// RWKV-14B configuration
    pub fn rwkv_14b() -> Self {
        Self {
            n_embd: 5120,
            n_layer: 40,
            n_head: 40,
            head_size: 128,
            vocab_size: 50277,
            ctx_len: 1024,
            ..Default::default()
        }
    }

    /// Get the FFN intermediate dimension
    pub fn get_n_ffn(&self) -> usize {
        self.n_ffn.unwrap_or({
            // Standard RWKV FFN dimension: 3.5 * n_embd
            (self.n_embd as f32 * 3.5) as usize
        })
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            "RWKV/rwkv-4-169m-pile" | "rwkv-169m" => Some(Self::rwkv_169m()),
            "RWKV/rwkv-4-430m-pile" | "rwkv-430m" => Some(Self::rwkv_430m()),
            "RWKV/rwkv-4-1b5-pile" | "rwkv-1.5b" => Some(Self::rwkv_1_5b()),
            "RWKV/rwkv-4-3b-pile" | "rwkv-3b" => Some(Self::rwkv_3b()),
            "RWKV/rwkv-4-7b-pile" | "rwkv-7b" => Some(Self::rwkv_7b()),
            "RWKV/rwkv-4-14b-pile" | "rwkv-14b" => Some(Self::rwkv_14b()),
            _ => None,
        }
    }
}

impl Config for RwkvConfig {
    fn architecture(&self) -> &'static str {
        "rwkv"
    }

    fn validate(&self) -> Result<()> {
        if self.n_embd == 0 {
            return Err(invalid_config(
                "config_field",
                "n_embd must be greater than 0",
            ));
        }
        if self.n_layer == 0 {
            return Err(invalid_config(
                "config_field",
                "n_layer must be greater than 0",
            ));
        }
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be greater than 0",
            ));
        }
        if self.n_head == 0 {
            return Err(invalid_config(
                "config_field",
                "n_head must be greater than 0",
            ));
        }
        if self.head_size == 0 {
            return Err(invalid_config(
                "config_field",
                "head_size must be greater than 0",
            ));
        }
        if self.n_embd != self.n_head * self.head_size {
            return Err(invalid_config(
                "config_field",
                "n_embd must equal n_head * head_size",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = RwkvConfig::default();
        assert_eq!(config.n_embd, 768);
        assert_eq!(config.n_layer, 12);
        assert_eq!(config.n_head, 12);
        assert_eq!(config.head_size, 64);
        assert_eq!(config.vocab_size, 50277);
    }

    #[test]
    fn test_ffn_computation() {
        let config = RwkvConfig::default();
        assert_eq!(config.get_n_ffn(), 2688); // 768 * 3.5 = 2688

        let config_with_ffn = RwkvConfig {
            n_ffn: Some(3072),
            ..Default::default()
        };
        assert_eq!(config_with_ffn.get_n_ffn(), 3072);
    }

    #[test]
    fn test_predefined_configs() {
        let config_169m = RwkvConfig::rwkv_169m();
        assert_eq!(config_169m.n_embd, 768);
        assert_eq!(config_169m.n_layer, 12);

        let config_14b = RwkvConfig::rwkv_14b();
        assert_eq!(config_14b.n_embd, 5120);
        assert_eq!(config_14b.n_layer, 40);
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = RwkvConfig::from_pretrained_name("RWKV/rwkv-4-169m-pile");
        assert!(config.is_some());
        assert_eq!(config.expect("operation failed").n_embd, 768);

        let config = RwkvConfig::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_config_trait() {
        let config = RwkvConfig::default();
        assert_eq!(config.architecture(), "rwkv");
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_config_validation() {
        let mut config = RwkvConfig::default();
        assert!(config.validate().is_ok());

        // Test invalid n_embd/n_head/head_size relationship
        config.n_embd = 1000;
        assert!(config.validate().is_err());

        // Test zero values
        config = RwkvConfig {
            n_embd: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // ---- RWKV-4-169M specific checks ----

    #[test]
    fn test_rwkv_169m_canonical_dimensions() {
        let config = RwkvConfig::rwkv_169m();
        assert_eq!(config.n_embd, 768, "RWKV-4-169M must have n_embd=768");
        assert_eq!(config.n_layer, 12, "RWKV-4-169M must have 12 layers");
        assert_eq!(config.ctx_len, 1024, "RWKV-4-169M ctx_len must be 1024");
        assert_eq!(config.n_head, 12);
        assert_eq!(config.head_size, 64);
        // Validate consistency
        assert_eq!(config.n_embd, config.n_head * config.head_size);
    }

    #[test]
    fn test_rwkv_169m_vocab_size() {
        let config = RwkvConfig::rwkv_169m();
        assert_eq!(
            config.vocab_size, 50277,
            "RWKV canonical vocab_size is 50277 (NeoX tokenizer)"
        );
    }

    // ---- RWKV-4-1.5B specific checks ----

    #[test]
    fn test_rwkv_1_5b_dimensions() {
        let config = RwkvConfig::rwkv_1_5b();
        // per spec: 2048 embd is NOT 1.5b; actual is 1536 x 48 layers
        assert_eq!(config.n_embd, 1536, "RWKV-1.5B n_embd=1536");
        assert_eq!(config.n_layer, 48, "RWKV-1.5B n_layer=48");
        assert_eq!(config.n_head, 24);
        assert_eq!(config.head_size, 64);
        assert_eq!(config.n_embd, config.n_head * config.head_size);
    }

    // ---- RWKV-4-3B (uses n_embd=2048) ----

    #[test]
    fn test_rwkv_3b_embd_2048() {
        let config = RwkvConfig::rwkv_3b();
        assert_eq!(config.n_embd, 2048, "RWKV-3B n_embd must be 2048");
        assert_eq!(config.n_layer, 32, "RWKV-3B must have 32 layers");
        assert_eq!(config.n_head, 32);
        assert_eq!(config.head_size, 64);
        assert_eq!(config.n_embd, config.n_head * config.head_size);
    }

    // ---- RWKV-4-7B specific checks ----

    #[test]
    fn test_rwkv_7b_dimensions() {
        let config = RwkvConfig::rwkv_7b();
        assert_eq!(config.n_embd, 4096, "RWKV-7B n_embd must be 4096");
        assert_eq!(config.n_layer, 32, "RWKV-7B must have 32 layers");
        assert_eq!(config.n_head, 32);
        // RWKV-5/6 uses head_size=128 for 7B
        assert_eq!(config.head_size, 128, "RWKV-7B head_size must be 128");
        assert_eq!(config.n_embd, config.n_head * config.head_size);
    }

    // ---- head_size for RWKV-5/6 ----

    #[test]
    fn test_rwkv5_head_size_128() {
        // 7B and 14B use head_size=128, suitable for RWKV-5/6 architectures
        let config_7b = RwkvConfig::rwkv_7b();
        let config_14b = RwkvConfig::rwkv_14b();
        assert_eq!(config_7b.head_size, 128);
        assert_eq!(config_14b.head_size, 128);
    }

    #[test]
    fn test_rwkv4_head_size_64() {
        // Small/mid models use head_size=64 (RWKV-4 style)
        let config_169m = RwkvConfig::rwkv_169m();
        let config_430m = RwkvConfig::rwkv_430m();
        assert_eq!(config_169m.head_size, 64);
        assert_eq!(config_430m.head_size, 64);
    }

    // ---- vocab_size canonical value ----

    #[test]
    fn test_all_configs_vocab_50277() {
        for config in [
            RwkvConfig::rwkv_169m(),
            RwkvConfig::rwkv_430m(),
            RwkvConfig::rwkv_1_5b(),
            RwkvConfig::rwkv_3b(),
            RwkvConfig::rwkv_7b(),
            RwkvConfig::rwkv_14b(),
        ] {
            assert_eq!(
                config.vocab_size, 50277,
                "All RWKV configs should use vocab_size=50277"
            );
        }
    }

    // ---- attention_type / model_type ----

    #[test]
    fn test_model_type_string() {
        let config = RwkvConfig::default();
        assert_eq!(config.model_type, "rwkv", "model_type must be 'rwkv'");
        assert_eq!(config.arch_version, "RWKV-4");
    }

    #[test]
    fn test_arch_version_default() {
        let config = RwkvConfig::default();
        // Default is RWKV-4 architecture
        assert_eq!(config.version, "4");
        assert!(config.arch_version.contains("RWKV"));
    }

    // ---- n_ffn computation ----

    #[test]
    fn test_ffn_dim_is_3_5x_n_embd() {
        // Standard RWKV FFN: 3.5 * n_embd (integer truncation)
        for (n_embd, expected_ffn) in [(768usize, 2688usize), (1024, 3584), (4096, 14336)] {
            let config = RwkvConfig {
                n_embd,
                n_head: n_embd / 64,
                head_size: 64,
                ..Default::default()
            };
            let computed = config.get_n_ffn();
            assert_eq!(
                computed, expected_ffn,
                "n_embd={}: expected ffn={}, got {}",
                n_embd, expected_ffn, computed
            );
        }
    }

    #[test]
    fn test_explicit_n_ffn_overrides_default() {
        let config = RwkvConfig {
            n_ffn: Some(4096),
            ..Default::default()
        };
        assert_eq!(
            config.get_n_ffn(),
            4096,
            "Explicit n_ffn must override computed value"
        );
    }

    // ---- Config validation edge cases ----

    #[test]
    fn test_validation_zero_n_layer() {
        let config = RwkvConfig {
            n_layer: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_n_head() {
        let config = RwkvConfig {
            n_head: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_head_size() {
        let config = RwkvConfig {
            head_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_zero_vocab_size() {
        let config = RwkvConfig {
            vocab_size: 0,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_validation_embd_head_mismatch() {
        // n_embd != n_head * head_size
        let config = RwkvConfig {
            n_embd: 769,
            n_head: 12,
            head_size: 64,
            ..Default::default()
        };
        assert!(config.validate().is_err());
    }

    // ---- Serialization round-trip ----

    #[test]
    fn test_serialization_round_trip() {
        let config = RwkvConfig::rwkv_7b();
        let serialized =
            serde_json::to_string(&config).expect("Serialization of RwkvConfig must succeed");
        let deserialized: RwkvConfig =
            serde_json::from_str(&serialized).expect("Deserialization of RwkvConfig must succeed");
        assert_eq!(deserialized.n_embd, config.n_embd);
        assert_eq!(deserialized.n_layer, config.n_layer);
        assert_eq!(deserialized.vocab_size, config.vocab_size);
        assert_eq!(deserialized.head_size, config.head_size);
        assert_eq!(deserialized.model_type, config.model_type);
    }

    #[test]
    fn test_serialization_preserves_n_ffn_none() {
        let config = RwkvConfig::default(); // n_ffn is None
        let serialized = serde_json::to_string(&config).expect("Serialization must succeed");
        let deserialized: RwkvConfig =
            serde_json::from_str(&serialized).expect("Deserialization must succeed");
        assert!(
            deserialized.n_ffn.is_none(),
            "n_ffn=None must round-trip as None"
        );
    }

    #[test]
    fn test_serialization_preserves_n_ffn_some() {
        let config = RwkvConfig {
            n_ffn: Some(8192),
            ..RwkvConfig::rwkv_7b()
        };
        let serialized = serde_json::to_string(&config).expect("Serialization must succeed");
        let deserialized: RwkvConfig =
            serde_json::from_str(&serialized).expect("Deserialization must succeed");
        assert_eq!(deserialized.n_ffn, Some(8192));
    }

    // ---- from_pretrained_name coverage ----

    #[test]
    fn test_from_pretrained_all_known_keys() {
        for key in [
            "RWKV/rwkv-4-169m-pile",
            "rwkv-169m",
            "RWKV/rwkv-4-430m-pile",
            "rwkv-430m",
            "RWKV/rwkv-4-1b5-pile",
            "rwkv-1.5b",
            "RWKV/rwkv-4-3b-pile",
            "rwkv-3b",
            "RWKV/rwkv-4-7b-pile",
            "rwkv-7b",
            "RWKV/rwkv-4-14b-pile",
            "rwkv-14b",
        ] {
            assert!(
                RwkvConfig::from_pretrained_name(key).is_some(),
                "Key '{}' must resolve to a config",
                key
            );
        }
    }

    #[test]
    fn test_from_pretrained_unknown_returns_none() {
        assert!(RwkvConfig::from_pretrained_name("rwkv-999b").is_none());
        assert!(RwkvConfig::from_pretrained_name("").is_none());
    }

    // ---- Clone / Debug ----

    #[test]
    fn test_clone_preserves_all_fields() {
        let original = RwkvConfig::rwkv_14b();
        let cloned = original.clone();
        assert_eq!(cloned.n_embd, original.n_embd);
        assert_eq!(cloned.n_layer, original.n_layer);
        assert_eq!(cloned.n_head, original.n_head);
        assert_eq!(cloned.head_size, original.head_size);
        assert_eq!(cloned.vocab_size, original.vocab_size);
        assert_eq!(cloned.rescale_layer, original.rescale_layer);
    }

    #[test]
    fn test_debug_format_non_empty() {
        let config = RwkvConfig::default();
        let debug_str = format!("{:?}", config);
        assert!(!debug_str.is_empty(), "Debug output must be non-empty");
        assert!(
            debug_str.contains("RwkvConfig"),
            "Debug must contain struct name"
        );
    }
}
