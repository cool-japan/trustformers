use serde::{Deserialize, Serialize};

/// Configuration for the BLIP-2 model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blip2Config {
    /// Vision encoder configuration
    pub vision_config: Blip2VisionConfig,
    /// Q-Former configuration
    pub qformer_config: Blip2QFormerConfig,
    /// Text decoder configuration
    pub text_config: Blip2TextConfig,
    /// Number of query tokens for Q-Former
    pub num_query_tokens: usize,
    /// Whether to use ITC (Image-Text Contrastive) loss
    pub use_itc_loss: bool,
    /// Whether to use ITM (Image-Text Matching) loss
    pub use_itm_loss: bool,
    /// Whether to use LM (Language Modeling) loss
    pub use_lm_loss: bool,
    /// Whether to use decoder start token
    pub use_decoder_only_language_model: bool,
}

impl Default for Blip2Config {
    fn default() -> Self {
        Self {
            vision_config: Blip2VisionConfig::default(),
            qformer_config: Blip2QFormerConfig::default(),
            text_config: Blip2TextConfig::default(),
            num_query_tokens: 32,
            use_itc_loss: true,
            use_itm_loss: true,
            use_lm_loss: true,
            use_decoder_only_language_model: true,
        }
    }
}

impl Blip2Config {
    /// Create configuration for BLIP2-OPT-2.7B
    pub fn opt_2_7b() -> Self {
        Self {
            vision_config: Blip2VisionConfig::eva_vit_g(),
            qformer_config: Blip2QFormerConfig::bert_base(),
            text_config: Blip2TextConfig::opt_2_7b(),
            num_query_tokens: 32,
            use_itc_loss: true,
            use_itm_loss: true,
            use_lm_loss: true,
            use_decoder_only_language_model: true,
        }
    }

    /// Create configuration for BLIP2-OPT-6.7B
    pub fn opt_6_7b() -> Self {
        Self {
            vision_config: Blip2VisionConfig::eva_vit_g(),
            qformer_config: Blip2QFormerConfig::bert_base(),
            text_config: Blip2TextConfig::opt_6_7b(),
            num_query_tokens: 32,
            use_itc_loss: true,
            use_itm_loss: true,
            use_lm_loss: true,
            use_decoder_only_language_model: true,
        }
    }

    /// Create configuration for BLIP2-FlanT5-XL
    pub fn flan_t5_xl() -> Self {
        Self {
            vision_config: Blip2VisionConfig::eva_vit_g(),
            qformer_config: Blip2QFormerConfig::bert_base(),
            text_config: Blip2TextConfig::flan_t5_xl(),
            num_query_tokens: 32,
            use_itc_loss: true,
            use_itm_loss: true,
            use_lm_loss: true,
            use_decoder_only_language_model: false,
        }
    }

    /// Create configuration for BLIP2-FlanT5-XXL
    pub fn flan_t5_xxl() -> Self {
        Self {
            vision_config: Blip2VisionConfig::eva_vit_g(),
            qformer_config: Blip2QFormerConfig::bert_base(),
            text_config: Blip2TextConfig::flan_t5_xxl(),
            num_query_tokens: 32,
            use_itc_loss: true,
            use_itm_loss: true,
            use_lm_loss: true,
            use_decoder_only_language_model: false,
        }
    }
}

/// Vision encoder configuration for BLIP-2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blip2VisionConfig {
    /// Hidden size
    pub hidden_size: usize,
    /// Intermediate size in feed-forward layers
    pub intermediate_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Image size
    pub image_size: usize,
    /// Patch size
    pub patch_size: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Dropout
    pub dropout: f64,
    /// Initializer range
    pub initializer_range: f64,
    /// Initializer factor
    pub initializer_factor: f64,
}

impl Default for Blip2VisionConfig {
    fn default() -> Self {
        Self::eva_vit_g()
    }
}

impl Blip2VisionConfig {
    /// EVA ViT-G configuration (used in BLIP-2)
    pub fn eva_vit_g() -> Self {
        Self {
            hidden_size: 1408,
            intermediate_size: 6144,
            num_hidden_layers: 39,
            num_attention_heads: 16,
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_act: "gelu".to_string(),
            layer_norm_eps: 1e-6,
            attention_dropout: 0.0,
            dropout: 0.0,
            initializer_range: 0.02,
            initializer_factor: 1.0,
        }
    }

    /// Number of patches
    pub fn num_patches(&self) -> usize {
        (self.image_size / self.patch_size).pow(2)
    }

    /// Sequence length (patches + CLS token)
    pub fn seq_len(&self) -> usize {
        self.num_patches() + 1
    }
}

/// Q-Former configuration for BLIP-2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blip2QFormerConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Hidden dropout probability
    pub hidden_dropout_prob: f64,
    /// Attention probability dropout
    pub attention_probs_dropout_prob: f64,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Type vocabulary size
    pub type_vocab_size: usize,
    /// Initializer range
    pub initializer_range: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Position embedding type
    pub position_embedding_type: String,
    /// Cross attention frequency
    pub cross_attention_frequency: usize,
    /// Encoder width
    pub encoder_width: usize,
}

impl Default for Blip2QFormerConfig {
    fn default() -> Self {
        Self::bert_base()
    }
}

impl Blip2QFormerConfig {
    /// BERT-base configuration for Q-Former
    pub fn bert_base() -> Self {
        Self {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.0,
            attention_probs_dropout_prob: 0.0,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            position_embedding_type: "absolute".to_string(),
            cross_attention_frequency: 2,
            encoder_width: 1408, // EVA ViT-G hidden size
        }
    }
}

/// Text decoder configuration for BLIP-2
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blip2TextConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// Number of key-value heads (for GQA)
    pub num_key_value_heads: Option<usize>,
    /// Intermediate size
    pub intermediate_size: usize,
    /// Hidden activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Initializer range
    pub initializer_range: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Use cache
    pub use_cache: bool,
    /// Pad token id
    pub pad_token_id: i32,
    /// BOS token id
    pub bos_token_id: i32,
    /// EOS token id
    pub eos_token_id: i32,
    /// Tie word embeddings
    pub tie_word_embeddings: bool,
    /// Is decoder only
    pub is_decoder_only: bool,
    /// Model type
    pub model_type: String,
}

impl Default for Blip2TextConfig {
    fn default() -> Self {
        Self::opt_2_7b()
    }
}

impl Blip2TextConfig {
    /// OPT-2.7B configuration
    pub fn opt_2_7b() -> Self {
        Self {
            vocab_size: 50272,
            hidden_size: 2560,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 10240,
            hidden_act: "relu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 2,
            tie_word_embeddings: true,
            is_decoder_only: true,
            model_type: "opt".to_string(),
        }
    }

    /// OPT-6.7B configuration
    pub fn opt_6_7b() -> Self {
        Self {
            vocab_size: 50272,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 16384,
            hidden_act: "relu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            use_cache: true,
            pad_token_id: 1,
            bos_token_id: 2,
            eos_token_id: 2,
            tie_word_embeddings: true,
            is_decoder_only: true,
            model_type: "opt".to_string(),
        }
    }

    /// Flan-T5-XL configuration
    pub fn flan_t5_xl() -> Self {
        Self {
            vocab_size: 32128,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 5120,
            hidden_act: "relu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.0,
            max_position_embeddings: 512,
            initializer_range: 1.0,
            layer_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 0,
            eos_token_id: 1,
            tie_word_embeddings: false,
            is_decoder_only: false,
            model_type: "t5".to_string(),
        }
    }

    /// Flan-T5-XXL configuration
    pub fn flan_t5_xxl() -> Self {
        Self {
            vocab_size: 32128,
            hidden_size: 4096,
            num_hidden_layers: 24,
            num_attention_heads: 64,
            num_key_value_heads: None,
            intermediate_size: 10240,
            hidden_act: "relu".to_string(),
            dropout: 0.1,
            attention_dropout: 0.0,
            max_position_embeddings: 512,
            initializer_range: 1.0,
            layer_norm_eps: 1e-6,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 0,
            eos_token_id: 1,
            tie_word_embeddings: false,
            is_decoder_only: false,
            model_type: "t5".to_string(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blip2_config_default() {
        let config = Blip2Config::default();
        assert_eq!(config.num_query_tokens, 32);
        assert!(config.use_itc_loss);
        assert!(config.use_itm_loss);
        assert!(config.use_lm_loss);
    }

    #[test]
    fn test_blip2_vision_config() {
        let config = Blip2VisionConfig::eva_vit_g();
        assert_eq!(config.hidden_size, 1408);
        assert_eq!(config.num_hidden_layers, 39);
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 14);
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
        assert_eq!(config.seq_len(), 257); // 256 patches + 1 CLS token
    }

    #[test]
    fn test_blip2_qformer_config() {
        let config = Blip2QFormerConfig::bert_base();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_hidden_layers, 12);
        assert_eq!(config.cross_attention_frequency, 2);
    }

    #[test]
    fn test_blip2_text_config_opt() {
        let config = Blip2TextConfig::opt_2_7b();
        assert_eq!(config.vocab_size, 50272);
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(config.num_hidden_layers, 32);
        assert!(config.is_decoder_only);
        assert_eq!(config.model_type, "opt");
    }

    #[test]
    fn test_blip2_text_config_t5() {
        let config = Blip2TextConfig::flan_t5_xl();
        assert_eq!(config.vocab_size, 32128);
        assert_eq!(config.hidden_size, 2048);
        assert_eq!(config.num_hidden_layers, 24);
        assert!(!config.is_decoder_only);
        assert_eq!(config.model_type, "t5");
    }

    #[test]
    fn test_blip2_model_variants() {
        let opt_2_7b = Blip2Config::opt_2_7b();
        let opt_6_7b = Blip2Config::opt_6_7b();
        let flan_t5_xl = Blip2Config::flan_t5_xl();
        let flan_t5_xxl = Blip2Config::flan_t5_xxl();

        assert_eq!(opt_2_7b.text_config.hidden_size, 2560);
        assert_eq!(opt_6_7b.text_config.hidden_size, 4096);
        assert_eq!(flan_t5_xl.text_config.hidden_size, 2048);
        assert_eq!(flan_t5_xxl.text_config.hidden_size, 4096);

        assert!(opt_2_7b.use_decoder_only_language_model);
        assert!(opt_6_7b.use_decoder_only_language_model);
        assert!(!flan_t5_xl.use_decoder_only_language_model);
        assert!(!flan_t5_xxl.use_decoder_only_language_model);
    }

    #[test]
    fn test_blip2_config_serialization() {
        let config = Blip2Config::opt_2_7b();
        let json = serde_json::to_string(&config).expect("operation failed");
        let deserialized: Blip2Config = serde_json::from_str(&json).expect("operation failed");

        assert_eq!(config.num_query_tokens, deserialized.num_query_tokens);
        assert_eq!(
            config.vision_config.hidden_size,
            deserialized.vision_config.hidden_size
        );
        assert_eq!(
            config.qformer_config.vocab_size,
            deserialized.qformer_config.vocab_size
        );
        assert_eq!(
            config.text_config.model_type,
            deserialized.text_config.model_type
        );
    }
}
