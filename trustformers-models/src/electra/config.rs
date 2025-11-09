use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ElectraConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
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

    // ELECTRA-specific parameters
    pub generator_hidden_size: usize,
    pub generator_num_hidden_layers: usize,
    pub generator_num_attention_heads: usize,
    pub generator_intermediate_size: usize,
    pub discriminator_hidden_size: usize,
    pub discriminator_num_hidden_layers: usize,
    pub discriminator_num_attention_heads: usize,
    pub discriminator_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub model_type: String,
}

impl Default for ElectraConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            embedding_size: 128,
            hidden_size: 256,
            num_hidden_layers: 12,
            num_attention_heads: 4,
            intermediate_size: 1024,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,

            // ELECTRA-specific defaults
            generator_hidden_size: 64,
            generator_num_hidden_layers: 1,
            generator_num_attention_heads: 1,
            generator_intermediate_size: 256,
            discriminator_hidden_size: 256,
            discriminator_num_hidden_layers: 12,
            discriminator_num_attention_heads: 4,
            discriminator_intermediate_size: 1024,
            tie_word_embeddings: true,
            model_type: "electra".to_string(),
        }
    }
}

impl Config for ElectraConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }

        if self.discriminator_hidden_size % self.discriminator_num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "discriminator_hidden_size",
                "discriminator_hidden_size must be divisible by discriminator_num_attention_heads",
            ));
        }

        if self.generator_hidden_size % self.generator_num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "generator_hidden_size",
                "generator_hidden_size must be divisible by generator_num_attention_heads",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "ELECTRA"
    }
}

impl ElectraConfig {
    /// ELECTRA-Small configuration
    pub fn small() -> Self {
        Self {
            vocab_size: 30522,
            embedding_size: 128,
            hidden_size: 256,
            num_hidden_layers: 12,
            num_attention_heads: 4,
            intermediate_size: 1024,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,

            generator_hidden_size: 64,
            generator_num_hidden_layers: 1,
            generator_num_attention_heads: 1,
            generator_intermediate_size: 256,
            discriminator_hidden_size: 256,
            discriminator_num_hidden_layers: 12,
            discriminator_num_attention_heads: 4,
            discriminator_intermediate_size: 1024,
            tie_word_embeddings: true,
            model_type: "electra".to_string(),
        }
    }

    /// ELECTRA-Base configuration
    pub fn base() -> Self {
        Self {
            vocab_size: 30522,
            embedding_size: 768,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,

            generator_hidden_size: 256,
            generator_num_hidden_layers: 12,
            generator_num_attention_heads: 4,
            generator_intermediate_size: 1024,
            discriminator_hidden_size: 768,
            discriminator_num_hidden_layers: 12,
            discriminator_num_attention_heads: 12,
            discriminator_intermediate_size: 3072,
            tie_word_embeddings: true,
            model_type: "electra".to_string(),
        }
    }

    /// ELECTRA-Large configuration
    pub fn large() -> Self {
        Self {
            vocab_size: 30522,
            embedding_size: 1024,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),
            use_cache: true,
            classifier_dropout: None,

            generator_hidden_size: 256,
            generator_num_hidden_layers: 24,
            generator_num_attention_heads: 4,
            generator_intermediate_size: 1024,
            discriminator_hidden_size: 1024,
            discriminator_num_hidden_layers: 24,
            discriminator_num_attention_heads: 16,
            discriminator_intermediate_size: 4096,
            tie_word_embeddings: true,
            model_type: "electra".to_string(),
        }
    }

    /// Create ELECTRA configuration from model name
    pub fn from_pretrained_name(model_name: &str) -> Self {
        let name_lower = model_name.to_lowercase();

        if name_lower.contains("small") {
            Self::small()
        } else if name_lower.contains("large") {
            Self::large()
        } else if name_lower.contains("base") {
            Self::base()
        } else {
            // Default to base for unknown variants
            Self::base()
        }
    }
}
