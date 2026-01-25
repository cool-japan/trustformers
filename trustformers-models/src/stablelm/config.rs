use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// StableLM model configuration
/// StableLM models by Stability AI, based on LLaMA architecture with specific optimizations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StableLMConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: Option<usize>, // For grouped-query attention
    pub hidden_act: String,
    pub max_position_embeddings: usize,
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
    pub partial_rotary_factor: f32, // StableLM specific: partial rotary embeddings
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String, // "linear" or "dynamic"
    pub scaling_factor: f32,
}

impl Default for StableLMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50432,
            hidden_size: 2560,
            intermediate_size: 6912,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None, // Multi-head attention by default
            hidden_act: "silu".to_string(),
            max_position_embeddings: 4096,
            initializer_range: 0.02,
            rms_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: Some(0),
            bos_token_id: 100257,
            eos_token_id: 100257,
            rope_theta: 10000.0,
            rope_scaling: None,
            attention_bias: false,
            mlp_bias: false,
            partial_rotary_factor: 0.25, // StableLM uses partial rotary embeddings
            model_type: "stablelm".to_string(),
        }
    }
}

impl StableLMConfig {
    /// Create StableLM-3B configuration
    pub fn stablelm_3b() -> Self {
        Self {
            vocab_size: 50432,
            hidden_size: 2560,
            intermediate_size: 6912,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096,
            partial_rotary_factor: 0.25,
            ..Default::default()
        }
    }

    /// Create StableLM-7B configuration
    pub fn stablelm_7b() -> Self {
        Self {
            vocab_size: 50432,
            hidden_size: 4096,
            intermediate_size: 11008,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            max_position_embeddings: 4096,
            partial_rotary_factor: 0.25,
            ..Default::default()
        }
    }

    /// Create StableLM-Zephyr-3B configuration (instruction-tuned variant)
    pub fn stablelm_zephyr_3b() -> Self {
        let mut config = Self::stablelm_3b();
        config.model_type = "stablelm-zephyr".to_string();
        config
    }

    /// Create StableLM-Code-3B configuration (code-specialized variant)
    pub fn stablelm_code_3b() -> Self {
        let mut config = Self::stablelm_3b();
        config.model_type = "stablelm-code".to_string();
        config.vocab_size = 49152; // Different vocab for code models
        config
    }

    /// Create StableLM-2-1.6B configuration (StableLM 2.0 series)
    pub fn stablelm_2_1_6b() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 2048,
            intermediate_size: 5504,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: Some(4), // Grouped-query attention
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            model_type: "stablelm-2".to_string(),
            ..Default::default()
        }
    }

    /// Create StableLM-2-12B configuration
    pub fn stablelm_2_12b() -> Self {
        Self {
            vocab_size: 100352,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 40,
            num_attention_heads: 40,
            num_key_value_heads: Some(8), // Grouped-query attention
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.25,
            model_type: "stablelm-2".to_string(),
            ..Default::default()
        }
    }

    /// Get configuration by model name
    pub fn from_pretrained_name(model_name: &str) -> Option<Self> {
        match model_name {
            "stabilityai/stablelm-3b-4e1t" => Some(Self::stablelm_3b()),
            "stabilityai/stablelm-base-alpha-3b" => Some(Self::stablelm_3b()),
            "stabilityai/stablelm-base-alpha-7b" => Some(Self::stablelm_7b()),
            "stabilityai/stablelm-zephyr-3b" => Some(Self::stablelm_zephyr_3b()),
            "stabilityai/stable-code-3b" => Some(Self::stablelm_code_3b()),
            "stabilityai/stablelm-2-1_6b" => Some(Self::stablelm_2_1_6b()),
            "stabilityai/stablelm-2-12b" => Some(Self::stablelm_2_12b()),
            _ => None,
        }
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".to_string());
        }

        if self.hidden_size == 0 {
            return Err("hidden_size must be > 0".to_string());
        }

        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be > 0".to_string());
        }

        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if num_kv_heads == 0 {
                return Err("num_key_value_heads must be > 0 when specified".to_string());
            }

            if self.num_attention_heads % num_kv_heads != 0 {
                return Err(
                    "num_attention_heads must be divisible by num_key_value_heads".to_string(),
                );
            }
        }

        if self.partial_rotary_factor < 0.0 || self.partial_rotary_factor > 1.0 {
            return Err("partial_rotary_factor must be between 0.0 and 1.0".to_string());
        }

        if !["silu", "gelu", "relu", "swiglu"].contains(&self.hidden_act.as_str()) {
            return Err("Unsupported activation function".to_string());
        }

        Ok(())
    }

    /// Calculate total number of parameters (approximate)
    pub fn estimate_parameters(&self) -> usize {
        let embedding_params = self.vocab_size * self.hidden_size;
        let layer_params = self.num_hidden_layers
            * (
                // Self-attention
                4 * self.hidden_size * self.hidden_size +
            // MLP
            2 * self.hidden_size * self.intermediate_size +
            // Layer norms (2 per layer)
            2 * self.hidden_size
            );
        let final_norm_params = self.hidden_size;
        let lm_head_params = self.vocab_size * self.hidden_size;

        embedding_params + layer_params + final_norm_params + lm_head_params
    }
}

impl Config for StableLMConfig {
    fn architecture(&self) -> &'static str {
        "stablelm"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = StableLMConfig::default();
        assert_eq!(config.model_type, "stablelm");
        assert_eq!(config.hidden_act, "silu");
        assert_eq!(config.partial_rotary_factor, 0.25);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_predefined_configs() {
        let config_3b = StableLMConfig::stablelm_3b();
        assert_eq!(config_3b.hidden_size, 2560);
        assert_eq!(config_3b.num_hidden_layers, 32);
        assert!(config_3b.validate().is_ok());

        let config_7b = StableLMConfig::stablelm_7b();
        assert_eq!(config_7b.hidden_size, 4096);
        assert_eq!(config_7b.num_hidden_layers, 32);
        assert!(config_7b.validate().is_ok());

        let config_2_12b = StableLMConfig::stablelm_2_12b();
        assert_eq!(config_2_12b.hidden_size, 5120);
        assert_eq!(config_2_12b.num_key_value_heads, Some(8));
        assert!(config_2_12b.validate().is_ok());
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = StableLMConfig::from_pretrained_name("stabilityai/stablelm-3b-4e1t");
        assert!(config.is_some());
        assert_eq!(config.expect("operation failed").hidden_size, 2560);

        let config = StableLMConfig::from_pretrained_name("nonexistent/model");
        assert!(config.is_none());
    }

    #[test]
    fn test_config_validation() {
        let mut config = StableLMConfig::default();
        assert!(config.validate().is_ok());

        config.vocab_size = 0;
        assert!(config.validate().is_err());

        config = StableLMConfig::default();
        config.hidden_size = 100;
        config.num_attention_heads = 3; // Not divisible
        assert!(config.validate().is_err());

        config = StableLMConfig::default();
        config.partial_rotary_factor = 1.5; // Out of range
        assert!(config.validate().is_err());
    }

    #[test]
    fn test_parameter_estimation() {
        let config = StableLMConfig::stablelm_3b();
        let params = config.estimate_parameters();

        // StableLM-3B parameter estimation should be reasonable order of magnitude
        assert!(
            params > 1_000_000_000 && params < 10_000_000_000,
            "Expected ~3B params, got {}",
            params
        );
    }

    #[test]
    fn test_config_serialization() {
        let config = StableLMConfig::stablelm_3b();
        let json = serde_json::to_string(&config).expect("operation failed");
        let deserialized: StableLMConfig = serde_json::from_str(&json).expect("operation failed");

        assert_eq!(config.hidden_size, deserialized.hidden_size);
        assert_eq!(
            config.partial_rotary_factor,
            deserialized.partial_rotary_factor
        );
        assert_eq!(config.model_type, deserialized.model_type);
    }
}
