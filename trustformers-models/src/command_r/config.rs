use serde::{Deserialize, Serialize};

/// Configuration for Command R models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommandRConfig {
    /// Model name
    pub model_name: String,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Intermediate size in FFN
    pub intermediate_size: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// RMS normalization epsilon
    pub rms_norm_eps: f32,
    /// Rope theta
    pub rope_theta: f32,
    /// Rope scaling factor
    pub rope_scaling_factor: f32,
    /// Attention dropout
    pub attention_dropout: f32,
    /// Hidden dropout
    pub hidden_dropout: f32,
    /// Use bias in linear layers
    pub use_bias: bool,
    /// Tie word embeddings
    pub tie_word_embeddings: bool,
    /// Activation function
    pub activation_function: String,
    /// Layer norm epsilon
    pub layer_norm_eps: f32,
    /// Use logit bias
    pub use_logit_bias: bool,
    /// Logit scale
    pub logit_scale: f32,
    /// Use sliding window attention
    pub use_sliding_window: bool,
    /// Sliding window size
    pub sliding_window_size: usize,
    /// Use flash attention
    pub use_flash_attention: bool,
    /// Pad token id
    pub pad_token_id: Option<usize>,
    /// BOS token id
    pub bos_token_id: Option<usize>,
    /// EOS token id
    pub eos_token_id: Option<usize>,
    /// Model type
    pub model_type: String,
    /// Torch dtype
    pub torch_dtype: String,
    /// Transformers version
    pub transformers_version: String,
}

impl Default for CommandRConfig {
    fn default() -> Self {
        // Default configuration for Command R base model
        Self {
            model_name: "command-r".to_string(),
            vocab_size: 256000,
            hidden_size: 8192,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            num_hidden_layers: 40,
            intermediate_size: 22528,
            max_sequence_length: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
            tie_word_embeddings: false,
            activation_function: "silu".to_string(),
            layer_norm_eps: 1e-5,
            use_logit_bias: false,
            logit_scale: 1.0,
            use_sliding_window: false,
            sliding_window_size: 4096,
            use_flash_attention: true,
            pad_token_id: Some(0),
            bos_token_id: Some(5),
            eos_token_id: Some(255001),
            model_type: "command-r".to_string(),
            torch_dtype: "bfloat16".to_string(),
            transformers_version: "4.39.0".to_string(),
        }
    }
}

impl CommandRConfig {
    /// Create a tiny configuration for testing purposes
    /// Uses very small dimensions to allow fast test execution
    pub fn tiny() -> Self {
        Self {
            model_name: "command-r-tiny".to_string(),
            vocab_size: 1000,
            hidden_size: 64,
            num_attention_heads: 4,
            num_key_value_heads: 4,
            num_hidden_layers: 2,
            intermediate_size: 128,
            max_sequence_length: 128,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
            tie_word_embeddings: false,
            activation_function: "silu".to_string(),
            layer_norm_eps: 1e-5,
            use_logit_bias: false,
            logit_scale: 1.0,
            use_sliding_window: false,
            sliding_window_size: 64,
            use_flash_attention: false,
            pad_token_id: Some(0),
            bos_token_id: Some(1),
            eos_token_id: Some(2),
            model_type: "command-r".to_string(),
            torch_dtype: "float32".to_string(),
            transformers_version: "4.39.0".to_string(),
        }
    }

    /// Create Command R base model configuration
    pub fn command_r() -> Self {
        Self::default()
    }

    /// Create Command R+ model configuration
    pub fn command_r_plus() -> Self {
        Self {
            model_name: "command-r-plus".to_string(),
            vocab_size: 256000,
            hidden_size: 12288,
            num_attention_heads: 96,
            num_key_value_heads: 96,
            num_hidden_layers: 64,
            intermediate_size: 33792,
            max_sequence_length: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
            tie_word_embeddings: false,
            activation_function: "silu".to_string(),
            layer_norm_eps: 1e-5,
            use_logit_bias: false,
            logit_scale: 1.0,
            use_sliding_window: false,
            sliding_window_size: 4096,
            use_flash_attention: true,
            pad_token_id: Some(0),
            bos_token_id: Some(5),
            eos_token_id: Some(255001),
            model_type: "command-r-plus".to_string(),
            torch_dtype: "bfloat16".to_string(),
            transformers_version: "4.39.0".to_string(),
        }
    }

    /// Create Command R 08-2024 model configuration
    pub fn command_r_08_2024() -> Self {
        Self {
            model_name: "command-r-08-2024".to_string(),
            vocab_size: 256000,
            hidden_size: 8192,
            num_attention_heads: 64,
            num_key_value_heads: 64,
            num_hidden_layers: 40,
            intermediate_size: 22528,
            max_sequence_length: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
            tie_word_embeddings: false,
            activation_function: "silu".to_string(),
            layer_norm_eps: 1e-5,
            use_logit_bias: false,
            logit_scale: 1.0,
            use_sliding_window: false,
            sliding_window_size: 4096,
            use_flash_attention: true,
            pad_token_id: Some(0),
            bos_token_id: Some(5),
            eos_token_id: Some(255001),
            model_type: "command-r-08-2024".to_string(),
            torch_dtype: "bfloat16".to_string(),
            transformers_version: "4.39.0".to_string(),
        }
    }

    /// Create Command R+ 08-2024 model configuration
    pub fn command_r_plus_08_2024() -> Self {
        Self {
            model_name: "command-r-plus-08-2024".to_string(),
            vocab_size: 256000,
            hidden_size: 12288,
            num_attention_heads: 96,
            num_key_value_heads: 96,
            num_hidden_layers: 64,
            intermediate_size: 33792,
            max_sequence_length: 131072,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            rope_scaling_factor: 1.0,
            attention_dropout: 0.0,
            hidden_dropout: 0.0,
            use_bias: false,
            tie_word_embeddings: false,
            activation_function: "silu".to_string(),
            layer_norm_eps: 1e-5,
            use_logit_bias: false,
            logit_scale: 1.0,
            use_sliding_window: false,
            sliding_window_size: 4096,
            use_flash_attention: true,
            pad_token_id: Some(0),
            bos_token_id: Some(5),
            eos_token_id: Some(255001),
            model_type: "command-r-plus-08-2024".to_string(),
            torch_dtype: "bfloat16".to_string(),
            transformers_version: "4.39.0".to_string(),
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the key-value head dimension
    pub fn kv_head_dim(&self) -> usize {
        self.hidden_size / self.num_key_value_heads
    }

    /// Check if grouped query attention is used
    pub fn is_gqa(&self) -> bool {
        self.num_key_value_heads != self.num_attention_heads
    }

    /// Get the number of query groups
    pub fn num_query_groups(&self) -> usize {
        self.num_attention_heads / self.num_key_value_heads
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be greater than 0".to_string());
        }
        if self.hidden_size == 0 {
            return Err("hidden_size must be greater than 0".to_string());
        }
        if self.num_attention_heads == 0 {
            return Err("num_attention_heads must be greater than 0".to_string());
        }
        if self.num_key_value_heads == 0 {
            return Err("num_key_value_heads must be greater than 0".to_string());
        }
        if self.num_hidden_layers == 0 {
            return Err("num_hidden_layers must be greater than 0".to_string());
        }
        if self.intermediate_size == 0 {
            return Err("intermediate_size must be greater than 0".to_string());
        }
        if self.max_sequence_length == 0 {
            return Err("max_sequence_length must be greater than 0".to_string());
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err("hidden_size must be divisible by num_attention_heads".to_string());
        }
        if self.num_attention_heads % self.num_key_value_heads != 0 {
            return Err("num_attention_heads must be divisible by num_key_value_heads".to_string());
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_command_r_config() {
        let config = CommandRConfig::command_r();
        assert_eq!(config.model_name, "command-r");
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 8192);
        assert_eq!(config.num_attention_heads, 64);
        assert_eq!(config.num_hidden_layers, 40);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_command_r_plus_config() {
        let config = CommandRConfig::command_r_plus();
        assert_eq!(config.model_name, "command-r-plus");
        assert_eq!(config.vocab_size, 256000);
        assert_eq!(config.hidden_size, 12288);
        assert_eq!(config.num_attention_heads, 96);
        assert_eq!(config.num_hidden_layers, 64);
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_head_dim_calculation() {
        let config = CommandRConfig::command_r();
        assert_eq!(config.head_dim(), 128); // 8192 / 64

        let config_plus = CommandRConfig::command_r_plus();
        assert_eq!(config_plus.head_dim(), 128); // 12288 / 96
    }

    #[test]
    fn test_gqa_detection() {
        let config = CommandRConfig::command_r();
        assert!(!config.is_gqa()); // Same number of heads

        let mut config_gqa = config.clone();
        config_gqa.num_key_value_heads = 32;
        assert!(config_gqa.is_gqa());
        assert_eq!(config_gqa.num_query_groups(), 2); // 64 / 32
    }

    #[test]
    fn test_config_validation() {
        let mut config = CommandRConfig::default();
        assert!(config.validate().is_ok());

        config.vocab_size = 0;
        assert!(config.validate().is_err());

        config.vocab_size = 256000;
        config.hidden_size = 100;
        config.num_attention_heads = 64;
        assert!(config.validate().is_err()); // 100 not divisible by 64
    }
}
