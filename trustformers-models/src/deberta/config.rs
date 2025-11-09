use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebertaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub position_embedding_type: String,
    pub use_cache: bool,
    pub classifier_dropout: Option<f32>,

    // DeBERTa-specific parameters
    pub relative_attention: bool,
    pub max_relative_positions: i32,
    pub pos_att_type: Vec<String>, // ["p2c", "c2p"] for positional attention types
    pub norm_rel_ebd: String,      // "layer_norm" or "none"
    pub share_att_key: bool,
    pub model_type: String,
}

impl Default for DebertaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            type_vocab_size: 0, // DeBERTa doesn't use token type embeddings by default
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,

            // DeBERTa-specific defaults
            relative_attention: true,
            max_relative_positions: -1, // -1 means no limit
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta".to_string(),
        }
    }
}

impl Config for DebertaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.max_relative_positions == 0 {
            return Err(trustformers_core::errors::invalid_config(
                "max_relative_positions",
                "max_relative_positions cannot be 0",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "DeBERTa"
    }
}

impl DebertaConfig {
    /// DeBERTa-Base configuration
    pub fn base() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,

            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta".to_string(),
        }
    }

    /// DeBERTa-Large configuration
    pub fn large() -> Self {
        Self {
            vocab_size: 50265,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,

            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta".to_string(),
        }
    }

    /// DeBERTa-v2-XLarge configuration
    pub fn xlarge() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 1536,
            num_hidden_layers: 24,
            num_attention_heads: 24,
            intermediate_size: 6144,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,

            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta".to_string(),
        }
    }

    /// DeBERTa-v3-Large configuration
    pub fn v3_large() -> Self {
        Self {
            vocab_size: 128100,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 2048,
            type_vocab_size: 0,
            initializer_range: 0.02,
            layer_norm_eps: 1e-7,
            pad_token_id: 0,
            position_embedding_type: "relative_key_query".to_string(),
            use_cache: true,
            classifier_dropout: None,

            relative_attention: true,
            max_relative_positions: -1,
            pos_att_type: vec!["p2c".to_string(), "c2p".to_string()],
            norm_rel_ebd: "layer_norm".to_string(),
            share_att_key: true,
            model_type: "deberta-v2".to_string(),
        }
    }

    /// Create DeBERTa configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("xlarge") || name_lower.contains("xxl") {
            Self::xlarge()
        } else if name_lower.contains("large") {
            if name_lower.contains("v3") {
                Self::v3_large()
            } else {
                Self::large()
            }
        } else {
            // Default to base for unknown variants
            Self::base()
        }
    }
}
