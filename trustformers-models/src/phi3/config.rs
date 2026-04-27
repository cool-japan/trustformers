use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Phi-3 model configuration
/// Reference: "Phi-3 Technical Report: A Highly Capable Language Model Locally On Your Phone" (Microsoft, 2024)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Phi3Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>, // For grouped-query attention
    pub hidden_act: String,
    pub max_position_embeddings: usize,
    pub original_max_position_embeddings: usize,
    pub initializer_range: f32,
    pub rms_norm_eps: f32,
    pub use_cache: bool,
    pub pad_token_id: Option<u32>,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub rope_theta: f32, // Base frequency for RoPE
    pub rope_scaling: Option<RopeScaling>,
    pub attention_bias: bool,
    pub mlp_bias: bool,
    pub model_type: String,
    pub sliding_window: Option<usize>, // For sliding window attention
    pub attention_dropout: f32,
    pub resid_pdrop: f32, // Residual dropout
    pub embd_pdrop: f32,  // Embedding dropout
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String, // "linear", "dynamic", or "longrope"
    pub scaling_factor: f32,
    pub long_factor: Option<Vec<f32>>,  // For longrope scaling
    pub short_factor: Option<Vec<f32>>, // For longrope scaling
}

impl Default for Phi3Config {
    fn default() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None, // Multi-head attention by default
            hidden_act: "silu".to_string(),
            max_position_embeddings: 4096,
            original_max_position_embeddings: 4096,
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: None,
            bos_token_id: 1,
            eos_token_id: 32000,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            model_type: "phi3".to_string(),
            sliding_window: None,
            attention_dropout: 0.0,
            resid_pdrop: 0.0,
            embd_pdrop: 0.0,
        }
    }
}

impl Config for Phi3Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be divisible by num_attention_heads".to_string(),
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if !self.num_attention_heads.is_multiple_of(num_kv_heads) {
                return Err(invalid_config(
                    "config_field",
                    "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                ));
            }
        }

        if self.vocab_size == 0 {
            return Err(invalid_config(
                "config_field",
                "vocab_size must be greater than 0".to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Phi-3"
    }
}

impl Phi3Config {
    /// Phi-3 Mini 3.8B configuration
    /// Small but highly capable model for mobile and edge devices
    pub fn phi3_mini_3_8b() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 3072,
            intermediate_size: 8192,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None, // Standard multi-head attention
            max_position_embeddings: 4096,
            original_max_position_embeddings: 4096,
            rope_theta: 10000.0,
            rms_norm_eps: 1e-5,
            model_type: "phi3-mini".to_string(),
            ..Self::default()
        }
    }

    /// Phi-3 Mini 4K Instruct configuration
    /// Instruction-tuned version of Phi-3 Mini with 4K context
    pub fn phi3_mini_4k_instruct() -> Self {
        Self {
            model_type: "phi3-mini-instruct".to_string(),
            ..Self::phi3_mini_3_8b()
        }
    }

    /// Phi-3 Mini 128K Instruct configuration
    /// Extended context version with 128K token support
    pub fn phi3_mini_128k_instruct() -> Self {
        Self {
            max_position_embeddings: 131072, // 128K context
            rope_scaling: Some(RopeScaling {
                scaling_type: "longrope".to_string(),
                scaling_factor: 32.0,
                long_factor: Some(vec![
                    1.08,
                    1.11,
                    1.14,
                    1.34,
                    1.589_999_9,
                    1.6,
                    1.62,
                    1.65,
                    1.9,
                    2.86,
                    7.4,
                    7.700_000_3,
                    9.099_999,
                    12.2,
                    17.67,
                    24.460_001,
                    28.570_002,
                    30.420_002,
                    30.840_002,
                    32.590_004,
                    32.93,
                    42.320_004,
                    44.960_003,
                    50.34,
                    57.95,
                    60.140_003,
                    62.500_004,
                    63.370_003,
                    63.480_003,
                    63.5,
                    63.52,
                    63.54,
                ]),
                short_factor: Some(vec![
                    1.05,
                    1.05,
                    1.05,
                    1.1,
                    1.1,
                    1.1,
                    1.250_000_1,
                    1.250_000_1,
                    1.400_000_1,
                    1.449_999_9,
                    1.449_999_9,
                    1.449_999_9,
                    1.55,
                    1.9,
                    1.9,
                    1.949_999_9,
                    1.949_999_9,
                    1.949_999_9,
                    1.949_999_9,
                    1.949_999_9,
                    1.95,
                    2.02,
                    2.02,
                    2.02,
                    2.03,
                    2.03,
                    2.03,
                    2.03,
                    2.03,
                    2.03,
                    2.03,
                    2.03,
                ]),
            }),
            model_type: "phi3-mini-128k-instruct".to_string(),
            ..Self::phi3_mini_3_8b()
        }
    }

    /// Phi-3 Small 7B configuration
    /// Larger model with improved performance while maintaining efficiency
    pub fn phi3_small_7b() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 4096,
            intermediate_size: 14336,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            max_position_embeddings: 8192, // 8K context by default
            original_max_position_embeddings: 8192,
            rope_theta: 10000.0,
            model_type: "phi3-small".to_string(),
            ..Self::default()
        }
    }

    /// Phi-3 Small 8K Instruct configuration
    /// Instruction-tuned version with 8K context
    pub fn phi3_small_8k_instruct() -> Self {
        Self {
            model_type: "phi3-small-8k-instruct".to_string(),
            ..Self::phi3_small_7b()
        }
    }

    /// Phi-3 Small 128K Instruct configuration
    /// Extended context version with 128K tokens
    pub fn phi3_small_128k_instruct() -> Self {
        Self {
            max_position_embeddings: 131072, // 128K context
            rope_scaling: Some(RopeScaling {
                scaling_type: "longrope".to_string(),
                scaling_factor: 16.0,
                long_factor: None,  // Use default long factors for small model
                short_factor: None, // Use default short factors for small model
            }),
            model_type: "phi3-small-128k-instruct".to_string(),
            ..Self::phi3_small_7b()
        }
    }

    /// Phi-3 Medium 14B configuration
    /// Largest Phi-3 model with highest capability
    pub fn phi3_medium_14b() -> Self {
        Self {
            vocab_size: 32064,
            hidden_size: 5120,
            intermediate_size: 17920,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(10), // Grouped-query attention for efficiency
            max_position_embeddings: 4096,
            original_max_position_embeddings: 4096,
            rope_theta: 10000.0,
            model_type: "phi3-medium".to_string(),
            ..Self::default()
        }
    }

    /// Phi-3 Medium 4K Instruct configuration
    /// Instruction-tuned version of the medium model
    pub fn phi3_medium_4k_instruct() -> Self {
        Self {
            model_type: "phi3-medium-4k-instruct".to_string(),
            ..Self::phi3_medium_14b()
        }
    }

    /// Phi-3 Medium 128K Instruct configuration
    /// Extended context version of the medium model
    pub fn phi3_medium_128k_instruct() -> Self {
        Self {
            max_position_embeddings: 131072, // 128K context
            rope_scaling: Some(RopeScaling {
                scaling_type: "longrope".to_string(),
                scaling_factor: 32.0,
                long_factor: None, // Use computed factors for medium model
                short_factor: None,
            }),
            model_type: "phi3-medium-128k-instruct".to_string(),
            ..Self::phi3_medium_14b()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads (for grouped-query attention)
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Create configuration from pretrained model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            // Phi-3 Mini models
            "microsoft/Phi-3-mini-4k-instruct" | "phi3-mini-4k-instruct" => {
                Some(Self::phi3_mini_4k_instruct())
            },
            "microsoft/Phi-3-mini-128k-instruct" | "phi3-mini-128k-instruct" => {
                Some(Self::phi3_mini_128k_instruct())
            },
            "phi3-mini" | "phi3-mini-3.8b" => Some(Self::phi3_mini_3_8b()),

            // Phi-3 Small models
            "microsoft/Phi-3-small-8k-instruct" | "phi3-small-8k-instruct" => {
                Some(Self::phi3_small_8k_instruct())
            },
            "microsoft/Phi-3-small-128k-instruct" | "phi3-small-128k-instruct" => {
                Some(Self::phi3_small_128k_instruct())
            },
            "phi3-small" | "phi3-small-7b" => Some(Self::phi3_small_7b()),

            // Phi-3 Medium models
            "microsoft/Phi-3-medium-4k-instruct" | "phi3-medium-4k-instruct" => {
                Some(Self::phi3_medium_4k_instruct())
            },
            "microsoft/Phi-3-medium-128k-instruct" | "phi3-medium-128k-instruct" => {
                Some(Self::phi3_medium_128k_instruct())
            },
            "phi3-medium" | "phi3-medium-14b" => Some(Self::phi3_medium_14b()),

            _ => None,
        }
    }

    /// Check if this is an instruct model
    pub fn is_instruct_model(&self) -> bool {
        self.model_type.contains("instruct")
    }

    /// Check if this model supports extended context (128K)
    pub fn is_long_context(&self) -> bool {
        self.max_position_embeddings > 32768
    }

    /// Get the effective context length considering rope scaling
    pub fn effective_context_length(&self) -> usize {
        if let Some(scaling) = &self.rope_scaling {
            if scaling.scaling_type == "longrope" {
                return self.max_position_embeddings;
            }
            (self.original_max_position_embeddings as f32 * scaling.scaling_factor) as usize
        } else {
            self.max_position_embeddings
        }
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
    }

    #[test]
    fn test_default_config_fields() {
        let cfg = Phi3Config::default();
        assert_eq!(cfg.vocab_size, 32064);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.model_type, "phi3");
        assert!(cfg.use_cache);
        assert!(!cfg.attention_bias);
        assert!(!cfg.mlp_bias);
    }

    #[test]
    fn test_default_validate_passes() {
        let cfg = Phi3Config::default();
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_architecture_name() {
        let cfg = Phi3Config::default();
        assert_eq!(cfg.architecture(), "Phi-3");
    }

    #[test]
    fn test_hidden_not_divisible_by_heads_fails() {
        let cfg = Phi3Config {
            hidden_size: 100,
            num_attention_heads: 32,
            ..Phi3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_kv_heads_not_divisible_fails() {
        let cfg = Phi3Config {
            num_attention_heads: 32,
            num_key_value_heads: Some(7),
            ..Phi3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_zero_vocab_size_fails() {
        let cfg = Phi3Config {
            vocab_size: 0,
            ..Phi3Config::default()
        };
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn test_head_dim_computation() {
        let cfg = Phi3Config::default();
        assert_eq!(cfg.head_dim(), 3072 / 32);
    }

    #[test]
    fn test_num_kv_heads_default() {
        let cfg = Phi3Config::default();
        assert_eq!(cfg.num_kv_heads(), cfg.num_attention_heads);
    }

    #[test]
    fn test_num_kv_heads_gqa() {
        let cfg = Phi3Config {
            num_key_value_heads: Some(8),
            ..Phi3Config::default()
        };
        assert_eq!(cfg.num_kv_heads(), 8);
    }

    #[test]
    fn test_num_query_groups_mha() {
        let cfg = Phi3Config::default();
        assert_eq!(cfg.num_query_groups(), 1);
    }

    #[test]
    fn test_num_query_groups_gqa() {
        let cfg = Phi3Config {
            num_attention_heads: 32,
            num_key_value_heads: Some(8),
            ..Phi3Config::default()
        };
        assert_eq!(cfg.num_query_groups(), 4);
    }

    #[test]
    fn test_phi3_mini_3_8b_config() {
        let cfg = Phi3Config::phi3_mini_3_8b();
        assert_eq!(cfg.vocab_size, 32064);
        assert_eq!(cfg.hidden_size, 3072);
        assert_eq!(cfg.num_hidden_layers, 32);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_phi3_mini_4k_instruct_is_instruct() {
        let cfg = Phi3Config::phi3_mini_4k_instruct();
        assert!(cfg.is_instruct_model());
    }

    #[test]
    fn test_phi3_mini_128k_is_long_context() {
        let cfg = Phi3Config::phi3_mini_128k_instruct();
        assert!(cfg.is_long_context());
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn test_phi3_mini_128k_has_longrope_scaling() {
        let cfg = Phi3Config::phi3_mini_128k_instruct();
        if let Some(scaling) = &cfg.rope_scaling {
            assert_eq!(scaling.scaling_type, "longrope");
            assert!(scaling.long_factor.is_some());
            assert!(scaling.short_factor.is_some());
        } else {
            panic!("expected rope_scaling");
        }
    }

    #[test]
    fn test_phi3_small_7b_config() {
        let cfg = Phi3Config::phi3_small_7b();
        assert_eq!(cfg.vocab_size, 100352);
        assert_eq!(cfg.hidden_size, 4096);
        assert_eq!(cfg.max_position_embeddings, 8192);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_phi3_medium_14b_has_gqa() {
        let cfg = Phi3Config::phi3_medium_14b();
        assert_eq!(cfg.num_key_value_heads, Some(10));
        assert_eq!(cfg.hidden_size, 5120);
        assert!(cfg.validate().is_ok());
    }

    #[test]
    fn test_effective_context_longrope() {
        let cfg = Phi3Config::phi3_mini_128k_instruct();
        // longrope: returns max_position_embeddings directly
        let eff = cfg.effective_context_length();
        assert_eq!(eff, cfg.max_position_embeddings);
    }

    #[test]
    fn test_effective_context_no_scaling() {
        let cfg = Phi3Config::phi3_mini_3_8b();
        assert_eq!(cfg.effective_context_length(), cfg.max_position_embeddings);
    }

    #[test]
    fn test_from_pretrained_name_mini() {
        let result = Phi3Config::from_pretrained_name("phi3-mini");
        assert!(result.is_some());
    }

    #[test]
    fn test_from_pretrained_name_unknown() {
        let result = Phi3Config::from_pretrained_name("unknown-phi-model");
        assert!(result.is_none());
    }

    #[test]
    fn test_phi3_small_128k_instruct_config() {
        let cfg = Phi3Config::phi3_small_128k_instruct();
        assert!(cfg.is_long_context());
        assert!(cfg.rope_scaling.is_some());
    }

    #[test]
    fn test_phi3_medium_128k_instruct_config() {
        let cfg = Phi3Config::phi3_medium_128k_instruct();
        assert!(cfg.is_long_context());
        assert_eq!(cfg.max_position_embeddings, 131072);
    }

    #[test]
    fn test_lcg_deterministic() {
        let mut rng1 = Lcg::new(42);
        let mut rng2 = Lcg::new(42);
        for _ in 0..50 {
            assert_eq!(rng1.next(), rng2.next());
        }
    }

    #[test]
    fn test_medium_4k_instruct_is_instruct() {
        let cfg = Phi3Config::phi3_medium_4k_instruct();
        assert!(cfg.is_instruct_model());
        assert!(!cfg.is_long_context());
    }
}
