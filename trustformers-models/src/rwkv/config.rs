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
}
