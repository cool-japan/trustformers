use serde::{Deserialize, Serialize};
use trustformers_core::errors::Result;
use trustformers_core::traits::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DistilBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub attention_probs_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub position_embedding_type: Option<String>,
    pub use_cache: Option<bool>,
    pub classifier_dropout: Option<f32>,
    pub sinusoidal_pos_embds: bool,
    pub tie_weights: Option<bool>,
}

impl Default for DistilBertConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 6, // DistilBERT has 6 layers vs BERT's 12
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            attention_probs_dropout_prob: 0.1,
            max_position_embeddings: 512,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: Some("absolute".to_string()),
            use_cache: Some(true),
            classifier_dropout: None,
            sinusoidal_pos_embds: false,
            tie_weights: Some(true),
        }
    }
}

impl DistilBertConfig {
    pub fn distilbert_base() -> Self {
        Self::default()
    }

    pub fn distilbert_base_cased() -> Self {
        Self::default()
    }
}

impl Config for DistilBertConfig {
    fn validate(&self) -> Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::invalid_config(
                "hidden_size",
                format!(
                    "hidden_size {} must be divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            ));
        }
        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "DistilBERT"
    }
}
