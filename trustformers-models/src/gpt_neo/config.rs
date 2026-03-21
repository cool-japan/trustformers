use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GptNeoConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub attention_types: Vec<String>, // ["global", "local"] pattern
    pub num_heads: usize,
    pub intermediate_size: usize,
    pub window_size: usize, // For local attention
    pub activation_function: String,
    pub resid_dropout: f32,
    pub embed_dropout: f32,
    pub attention_dropout: f32,
    pub max_position_embeddings: usize,
    pub layer_norm_epsilon: f32,
    pub initializer_range: f32,
    pub use_cache: bool,
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub model_type: String,
}

impl Default for GptNeoConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            attention_types: vec!["global".to_string(), "local".to_string()], // Alternating pattern
            num_heads: 12,
            intermediate_size: 3072,
            window_size: 256,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.1,
            embed_dropout: 0.1,
            attention_dropout: 0.1,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }
}

impl Config for GptNeoConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_heads != 0 {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "hidden_size must be divisible by num_heads".to_string(),
                ),
            );
        }
        if self.attention_types.is_empty() {
            return Err(
                trustformers_core::errors::TrustformersError::invalid_config(
                    "attention_types cannot be empty".to_string(),
                ),
            );
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "GPT-Neo"
    }
}

impl GptNeoConfig {
    /// GPT-Neo 125M configuration
    pub fn gpt_neo_125m() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            attention_types: vec![
                "global".to_string(),
                "local".to_string(),
                "global".to_string(),
                "local".to_string(),
                "global".to_string(),
                "local".to_string(),
                "global".to_string(),
                "local".to_string(),
                "global".to_string(),
                "local".to_string(),
                "global".to_string(),
                "local".to_string(),
            ],
            num_heads: 12,
            intermediate_size: 3072,
            window_size: 256,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            embed_dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }

    /// GPT-Neo 1.3B configuration
    pub fn gpt_neo_1_3b() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 2048,
            num_layers: 24,
            attention_types: {
                let mut types = Vec::new();
                for i in 0..24 {
                    if i % 2 == 0 {
                        types.push("global".to_string());
                    } else {
                        types.push("local".to_string());
                    }
                }
                types
            },
            num_heads: 16,
            intermediate_size: 8192,
            window_size: 256,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            embed_dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }

    /// GPT-Neo 2.7B configuration
    pub fn gpt_neo_2_7b() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 2560,
            num_layers: 32,
            attention_types: {
                let mut types = Vec::new();
                for i in 0..32 {
                    if i % 2 == 0 {
                        types.push("global".to_string());
                    } else {
                        types.push("local".to_string());
                    }
                }
                types
            },
            num_heads: 20,
            intermediate_size: 10240,
            window_size: 256,
            activation_function: "gelu_new".to_string(),
            resid_dropout: 0.0,
            embed_dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            layer_norm_epsilon: 1e-5,
            initializer_range: 0.02,
            use_cache: true,
            bos_token_id: 50256,
            eos_token_id: 50256,
            model_type: "gpt_neo".to_string(),
        }
    }

    /// Create GPT-Neo configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("125m") {
            Self::gpt_neo_125m()
        } else if name_lower.contains("1.3b") || name_lower.contains("1_3b") {
            Self::gpt_neo_1_3b()
        } else if name_lower.contains("2.7b") || name_lower.contains("2_7b") {
            Self::gpt_neo_2_7b()
        } else {
            // Default to 125M for unknown variants
            Self::gpt_neo_125m()
        }
    }
}
