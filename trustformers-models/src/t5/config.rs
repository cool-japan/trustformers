use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct T5Config {
    pub vocab_size: usize,
    pub d_model: usize,
    pub d_kv: usize,
    pub d_ff: usize,
    pub num_layers: usize,
    pub num_decoder_layers: Option<usize>,
    pub num_heads: usize,
    pub relative_attention_num_buckets: usize,
    pub relative_attention_max_distance: usize,
    pub dropout_rate: f32,
    pub layer_norm_epsilon: f32,
    pub initializer_factor: f32,
    pub feed_forward_proj: String,
    pub is_encoder_decoder: bool,
    pub use_cache: bool,
    pub pad_token_id: u32,
    pub eos_token_id: u32,
    pub model_type: String,
}

impl Default for T5Config {
    fn default() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }
}

impl Config for T5Config {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.d_model % self.num_heads != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "d_model must be divisible by num_heads".to_string(),
                ),
            );
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "T5"
    }
}

impl T5Config {
    /// T5-Small configuration (60M parameters)
    pub fn small() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 512,
            d_kv: 64,
            d_ff: 2048,
            num_layers: 6,
            num_decoder_layers: None,
            num_heads: 8,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    /// T5-Base configuration (220M parameters)
    pub fn base() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 768,
            d_kv: 64,
            d_ff: 3072,
            num_layers: 12,
            num_decoder_layers: None,
            num_heads: 12,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    /// T5-Large configuration (770M parameters)
    pub fn large() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 1024,
            d_kv: 64,
            d_ff: 4096,
            num_layers: 24,
            num_decoder_layers: None,
            num_heads: 16,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    /// T5-XL configuration (3B parameters)
    pub fn xl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 2048,
            d_kv: 128,
            d_ff: 8192,
            num_layers: 24,
            num_decoder_layers: None,
            num_heads: 32,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    /// T5-XXL configuration (11B parameters)
    pub fn xxl() -> Self {
        Self {
            vocab_size: 32128,
            d_model: 4096,
            d_kv: 128,
            d_ff: 16384,
            num_layers: 24,
            num_decoder_layers: None,
            num_heads: 64,
            relative_attention_num_buckets: 32,
            relative_attention_max_distance: 128,
            dropout_rate: 0.1,
            layer_norm_epsilon: 1e-6,
            initializer_factor: 1.0,
            feed_forward_proj: "relu".to_string(),
            is_encoder_decoder: true,
            use_cache: true,
            pad_token_id: 0,
            eos_token_id: 1,
            model_type: "t5".to_string(),
        }
    }

    /// Create T5 configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("small") {
            Self::small()
        } else if name_lower.contains("base") {
            Self::base()
        } else if name_lower.contains("large") {
            Self::large()
        } else if name_lower.contains("xl") {
            if name_lower.contains("xxl") {
                Self::xxl()
            } else {
                Self::xl()
            }
        } else {
            // Default to base for unknown variants
            Self::base()
        }
    }
}
