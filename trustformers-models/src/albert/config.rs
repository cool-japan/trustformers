use serde::{Deserialize, Serialize};
use trustformers_core::errors::{invalid_config, Result};
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlbertConfig {
    pub vocab_size: usize,
    pub embedding_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_hidden_groups: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub inner_group_num: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub classifier_dropout_prob: Option<f32>,
    pub position_embedding_type: String,
    pub pad_token_id: i32,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
}

impl Default for AlbertConfig {
    fn default() -> Self {
        Self::albert_base_v2()
    }
}

impl AlbertConfig {
    pub fn albert_base_v1() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_hidden_groups: 1,
            num_attention_heads: 12,
            intermediate_size: 3072,
            inner_group_num: 1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_base_v2() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_hidden_groups: 1,
            num_attention_heads: 12,
            intermediate_size: 3072,
            inner_group_num: 1,
            hidden_act: "gelu_new".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_large_v1() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_hidden_groups: 1,
            num_attention_heads: 16,
            intermediate_size: 4096,
            inner_group_num: 1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_large_v2() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_hidden_groups: 1,
            num_attention_heads: 16,
            intermediate_size: 4096,
            inner_group_num: 1,
            hidden_act: "gelu_new".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_xlarge_v1() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_hidden_groups: 1,
            num_attention_heads: 32,
            intermediate_size: 8192,
            inner_group_num: 1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_xlarge_v2() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_hidden_groups: 1,
            num_attention_heads: 32,
            intermediate_size: 8192,
            inner_group_num: 1,
            hidden_act: "gelu_new".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_xxlarge_v1() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 4096,
            num_hidden_layers: 12,
            num_hidden_groups: 1,
            num_attention_heads: 64,
            intermediate_size: 16384,
            inner_group_num: 1,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn albert_xxlarge_v2() -> Self {
        Self {
            vocab_size: 30000,
            embedding_size: 128,
            hidden_size: 4096,
            num_hidden_layers: 12,
            num_hidden_groups: 1,
            num_attention_heads: 64,
            intermediate_size: 16384,
            inner_group_num: 1,
            hidden_act: "gelu_new".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            classifier_dropout_prob: None,
            position_embedding_type: "absolute".to_string(),
            pad_token_id: 0,
            bos_token_id: 2,
            eos_token_id: 3,
        }
    }

    pub fn from_pretrained_name(model_name: &str) -> Self {
        match model_name.to_lowercase().as_str() {
            name if name.contains("albert-base-v1") => Self::albert_base_v1(),
            name if name.contains("albert-base-v2") => Self::albert_base_v2(),
            name if name.contains("albert-large-v1") => Self::albert_large_v1(),
            name if name.contains("albert-large-v2") => Self::albert_large_v2(),
            name if name.contains("albert-xlarge-v1") => Self::albert_xlarge_v1(),
            name if name.contains("albert-xlarge-v2") => Self::albert_xlarge_v2(),
            name if name.contains("albert-xxlarge-v1") => Self::albert_xxlarge_v1(),
            name if name.contains("albert-xxlarge-v2") => Self::albert_xxlarge_v2(),
            _ => Self::albert_base_v2(),
        }
    }
}

impl Config for AlbertConfig {
    fn validate(&self) -> Result<()> {
        if self.vocab_size == 0 {
            return Err(invalid_config(
                "vocab_size",
                "vocab_size must be greater than 0",
            ));
        }
        if self.hidden_size == 0 {
            return Err(invalid_config(
                "hidden_size",
                "hidden_size must be greater than 0",
            ));
        }
        if self.embedding_size == 0 {
            return Err(invalid_config(
                "embedding_size",
                "embedding_size must be greater than 0",
            ));
        }
        if self.num_hidden_layers == 0 {
            return Err(invalid_config(
                "num_hidden_layers",
                "num_hidden_layers must be greater than 0",
            ));
        }
        if self.num_hidden_groups == 0 {
            return Err(invalid_config(
                "num_hidden_groups",
                "num_hidden_groups must be greater than 0",
            ));
        }
        if self.num_attention_heads == 0 {
            return Err(invalid_config(
                "num_attention_heads",
                "num_attention_heads must be greater than 0",
            ));
        }
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(invalid_config(
                "hidden_size",
                "hidden_size must be divisible by num_attention_heads",
            ));
        }
        if self.num_hidden_layers % self.num_hidden_groups != 0 {
            return Err(invalid_config(
                "num_hidden_layers",
                "num_hidden_layers must be divisible by num_hidden_groups",
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "albert"
    }
}
