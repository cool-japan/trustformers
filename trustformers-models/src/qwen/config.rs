use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Qwen model configuration
/// Reference: "Qwen Technical Report" (Alibaba Cloud, 2023)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QwenConfig {
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
    pub rope_theta: f32,
    pub rope_scaling: Option<RopeScaling>,
    pub attention_dropout: f32,
    pub use_sliding_window: bool,
    pub sliding_window: Option<usize>,
    pub max_window_layers: Option<usize>,
    pub use_logn_attn: bool, // Qwen's LogN attention scaling
    pub logn_list: Option<Vec<f32>>,
    pub model_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RopeScaling {
    pub scaling_type: String,
    pub scaling_factor: f32,
}

impl Default for QwenConfig {
    fn default() -> Self {
        Self {
            vocab_size: 151936, // Qwen's vocabulary size
            hidden_size: 4096,
            intermediate_size: 22016,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: Some(32), // Full multi-head attention
            hidden_act: "silu".to_string(),
            max_position_embeddings: 32768, // Long context support
            initializer_range: 0.02,
            rms_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: Some(151643),
            bos_token_id: 151643,
            eos_token_id: 151645,
            rope_theta: 1000000.0, // Higher theta for better long context
            rope_scaling: None,
            attention_dropout: 0.0,
            use_sliding_window: false,
            sliding_window: None,
            max_window_layers: None,
            use_logn_attn: false,
            logn_list: None,
            model_type: "qwen2".to_string(),
        }
    }
}

impl Config for QwenConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(invalid_config(
                "config_field",
                "hidden_size must be divisible by num_attention_heads".to_string(),
            ));
        }

        if let Some(num_kv_heads) = self.num_key_value_heads {
            if self.num_attention_heads % num_kv_heads != 0 {
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
        "Qwen"
    }
}

impl QwenConfig {
    /// Qwen2 0.5B configuration
    pub fn qwen2_0_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 896,
            intermediate_size: 4864,
            num_hidden_layers: 24,
            num_attention_heads: 14,
            num_key_value_heads: Some(2), // Grouped-query attention
            max_position_embeddings: 32768,
            ..Self::default()
        }
    }

    /// Qwen2 1.5B configuration
    pub fn qwen2_1_5b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 1536,
            intermediate_size: 8960,
            num_hidden_layers: 28,
            num_attention_heads: 12,
            num_key_value_heads: Some(2),
            max_position_embeddings: 32768,
            ..Self::default()
        }
    }

    /// Qwen2 7B configuration
    pub fn qwen2_7b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: Some(4), // Grouped-query attention
            max_position_embeddings: 32768,
            ..Self::default()
        }
    }

    /// Qwen2 72B configuration
    pub fn qwen2_72b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 8192,
            intermediate_size: 29568,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8), // Grouped-query attention
            max_position_embeddings: 32768,
            ..Self::default()
        }
    }

    /// Qwen2.5 7B configuration (enhanced version)
    pub fn qwen2_5_7b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: Some(4),
            max_position_embeddings: 131072, // Extended context
            model_type: "qwen2.5".to_string(),
            ..Self::default()
        }
    }

    /// Qwen2.5 14B configuration
    pub fn qwen2_5_14b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 5120,
            intermediate_size: 13824,
            num_hidden_layers: 48,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072,
            model_type: "qwen2.5".to_string(),
            ..Self::default()
        }
    }

    /// Qwen2.5 32B configuration
    pub fn qwen2_5_32b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 5120,
            intermediate_size: 27392,
            num_hidden_layers: 64,
            num_attention_heads: 40,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072,
            model_type: "qwen2.5".to_string(),
            ..Self::default()
        }
    }

    /// Qwen2.5 72B configuration
    pub fn qwen2_5_72b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 8192,
            intermediate_size: 29568,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8),
            max_position_embeddings: 131072,
            model_type: "qwen2.5".to_string(),
            ..Self::default()
        }
    }

    /// Qwen2.5 Coder configuration (specialized for code)
    pub fn qwen2_5_coder_7b() -> Self {
        Self {
            vocab_size: 151936,
            hidden_size: 3584,
            intermediate_size: 18944,
            num_hidden_layers: 28,
            num_attention_heads: 28,
            num_key_value_heads: Some(4),
            max_position_embeddings: 131072,
            model_type: "qwen2.5-coder".to_string(),
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the number of key-value heads
    pub fn num_kv_heads(&self) -> usize {
        self.num_key_value_heads.unwrap_or(self.num_attention_heads)
    }

    /// Get the number of query groups per key-value head
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_kv_heads()
    }

    /// Check if using grouped-query attention
    pub fn uses_grouped_query_attention(&self) -> bool {
        if let Some(num_kv_heads) = self.num_key_value_heads {
            num_kv_heads < self.num_attention_heads
        } else {
            false
        }
    }

    /// Check if using sliding window attention
    pub fn uses_sliding_window(&self) -> bool {
        self.use_sliding_window && self.sliding_window.is_some()
    }

    /// Get the sliding window size
    pub fn sliding_window_size(&self) -> usize {
        self.sliding_window.unwrap_or(self.max_position_embeddings)
    }

    /// Check if this is Qwen2.5 variant
    pub fn is_qwen2_5(&self) -> bool {
        self.model_type.starts_with("qwen2.5")
    }
}
