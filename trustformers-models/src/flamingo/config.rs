use serde::{Deserialize, Serialize};

/// Configuration for the Flamingo model
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamingoConfig {
    /// Vision encoder configuration
    pub vision_config: FlamingoVisionConfig,
    /// Language model configuration
    pub language_config: FlamingoLanguageConfig,
    /// Perceiver resampler configuration
    pub perceiver_config: FlamingoPerceiverConfig,
    /// Cross-attention configuration
    pub cross_attention_config: FlamingoXAttentionConfig,
    /// Number of media tokens per image
    pub media_token_length: usize,
    /// Whether to use gated cross-attention layers
    pub use_gated_cross_attention: bool,
    /// Vision-language alignment dimension
    pub vision_language_dim: usize,
    /// Number of in-context examples
    pub num_shots: usize,
    /// Maximum sequence length
    pub max_seq_length: usize,
    /// Whether to freeze vision encoder
    pub freeze_vision_encoder: bool,
    /// Whether to freeze language model
    pub freeze_language_model: bool,
    /// Layer indices where to insert cross-attention
    pub cross_attention_layers: Vec<usize>,
}

impl Default for FlamingoConfig {
    fn default() -> Self {
        Self {
            vision_config: FlamingoVisionConfig::default(),
            language_config: FlamingoLanguageConfig::default(),
            perceiver_config: FlamingoPerceiverConfig::default(),
            cross_attention_config: FlamingoXAttentionConfig::default(),
            media_token_length: 64,
            use_gated_cross_attention: true,
            vision_language_dim: 2048,
            num_shots: 4,
            max_seq_length: 2048,
            freeze_vision_encoder: true,
            freeze_language_model: false,
            cross_attention_layers: vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        }
    }
}

impl FlamingoConfig {
    /// Flamingo-3B configuration
    pub fn flamingo_3b() -> Self {
        Self {
            vision_config: FlamingoVisionConfig::clip_vit_l(),
            language_config: FlamingoLanguageConfig::chinchilla_1b(),
            perceiver_config: FlamingoPerceiverConfig::default(),
            cross_attention_config: FlamingoXAttentionConfig::default(),
            media_token_length: 64,
            use_gated_cross_attention: true,
            vision_language_dim: 2048,
            num_shots: 4,
            max_seq_length: 2048,
            freeze_vision_encoder: true,
            freeze_language_model: false,
            cross_attention_layers: vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23],
        }
    }

    /// Flamingo-9B configuration
    pub fn flamingo_9b() -> Self {
        Self {
            vision_config: FlamingoVisionConfig::clip_vit_l(),
            language_config: FlamingoLanguageConfig::chinchilla_7b(),
            perceiver_config: FlamingoPerceiverConfig::large(),
            cross_attention_config: FlamingoXAttentionConfig::large(),
            media_token_length: 64,
            use_gated_cross_attention: true,
            vision_language_dim: 4096,
            num_shots: 8,
            max_seq_length: 4096,
            freeze_vision_encoder: true,
            freeze_language_model: false,
            cross_attention_layers: vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        }
    }

    /// Flamingo-80B configuration
    pub fn flamingo_80b() -> Self {
        Self {
            vision_config: FlamingoVisionConfig::clip_vit_l(),
            language_config: FlamingoLanguageConfig::chinchilla_70b(),
            perceiver_config: FlamingoPerceiverConfig::large(),
            cross_attention_config: FlamingoXAttentionConfig::large(),
            media_token_length: 64,
            use_gated_cross_attention: true,
            vision_language_dim: 8192,
            num_shots: 16,
            max_seq_length: 8192,
            freeze_vision_encoder: true,
            freeze_language_model: false,
            cross_attention_layers: (1..80).step_by(2).collect(), // Every other layer
        }
    }

    /// OpenFlamingo configuration (open-source version)
    pub fn open_flamingo() -> Self {
        Self {
            vision_config: FlamingoVisionConfig::clip_vit_l(),
            language_config: FlamingoLanguageConfig::mpt_7b(),
            perceiver_config: FlamingoPerceiverConfig::default(),
            cross_attention_config: FlamingoXAttentionConfig::default(),
            media_token_length: 64,
            use_gated_cross_attention: true,
            vision_language_dim: 4096,
            num_shots: 8,
            max_seq_length: 2048,
            freeze_vision_encoder: true,
            freeze_language_model: false,
            cross_attention_layers: vec![1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31],
        }
    }
}

/// Vision encoder configuration for Flamingo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamingoVisionConfig {
    /// Image size
    pub image_size: usize,
    /// Patch size
    pub patch_size: usize,
    /// Number of channels
    pub num_channels: usize,
    /// Hidden size
    pub hidden_size: usize,
    /// Number of hidden layers
    pub num_hidden_layers: usize,
    /// Number of attention heads
    pub num_attention_heads: usize,
    /// MLP hidden size
    pub intermediate_size: usize,
    /// Activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
}

impl Default for FlamingoVisionConfig {
    fn default() -> Self {
        Self::clip_vit_l()
    }
}

impl FlamingoVisionConfig {
    /// CLIP ViT-L/14 configuration
    pub fn clip_vit_l() -> Self {
        Self {
            image_size: 224,
            patch_size: 14,
            num_channels: 3,
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
        }
    }

    /// CLIP ViT-B/16 configuration
    pub fn clip_vit_b() -> Self {
        Self {
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_size: 768,
            num_hidden_layers: 12,
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: "quick_gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
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

/// Language model configuration for Flamingo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamingoLanguageConfig {
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
    /// MLP hidden size
    pub intermediate_size: usize,
    /// Activation function
    pub hidden_act: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Maximum position embeddings
    pub max_position_embeddings: usize,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Initializer range
    pub initializer_range: f64,
    /// RoPE theta
    pub rope_theta: f64,
    /// Use cache
    pub use_cache: bool,
    /// Pad token id
    pub pad_token_id: i32,
    /// BOS token id
    pub bos_token_id: i32,
    /// EOS token id
    pub eos_token_id: i32,
}

impl Default for FlamingoLanguageConfig {
    fn default() -> Self {
        Self::chinchilla_1b()
    }
}

impl FlamingoLanguageConfig {
    /// Chinchilla 1B configuration
    pub fn chinchilla_1b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 2048,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            num_key_value_heads: None,
            intermediate_size: 8192,
            hidden_act: "swiglu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            rope_theta: 10000.0,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Chinchilla 7B configuration
    pub fn chinchilla_7b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 16384,
            hidden_act: "swiglu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 4096,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            rope_theta: 10000.0,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// Chinchilla 70B configuration
    pub fn chinchilla_70b() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 8192,
            num_hidden_layers: 80,
            num_attention_heads: 64,
            num_key_value_heads: Some(8), // GQA
            intermediate_size: 32768,
            hidden_act: "swiglu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 8192,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            rope_theta: 10000.0,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }

    /// MPT-7B configuration (for OpenFlamingo)
    pub fn mpt_7b() -> Self {
        Self {
            vocab_size: 50432,
            hidden_size: 4096,
            num_hidden_layers: 32,
            num_attention_heads: 32,
            num_key_value_heads: None,
            intermediate_size: 16384,
            hidden_act: "gelu".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            max_position_embeddings: 2048,
            layer_norm_eps: 1e-5,
            initializer_range: 0.02,
            rope_theta: 10000.0,
            use_cache: true,
            pad_token_id: 0,
            bos_token_id: 1,
            eos_token_id: 2,
        }
    }
}

/// Perceiver resampler configuration for Flamingo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamingoPerceiverConfig {
    /// Number of latent queries
    pub num_latents: usize,
    /// Latent dimension
    pub latent_dim: usize,
    /// Number of perceiver layers
    pub num_layers: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// MLP hidden size
    pub mlp_hidden_size: usize,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
}

impl Default for FlamingoPerceiverConfig {
    fn default() -> Self {
        Self {
            num_latents: 64,
            latent_dim: 2048,
            num_layers: 6,
            num_heads: 16,
            head_dim: 128,
            mlp_hidden_size: 8192,
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }
}

impl FlamingoPerceiverConfig {
    /// Large perceiver configuration
    pub fn large() -> Self {
        Self {
            num_latents: 64,
            latent_dim: 4096,
            num_layers: 8,
            num_heads: 32,
            head_dim: 128,
            mlp_hidden_size: 16384,
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
        }
    }
}

/// Cross-attention configuration for Flamingo
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlamingoXAttentionConfig {
    /// Cross-attention dimension
    pub cross_attention_dim: usize,
    /// Number of heads for cross-attention
    pub num_heads: usize,
    /// Head dimension
    pub head_dim: usize,
    /// Whether to use gating mechanism
    pub use_gating: bool,
    /// Gating mechanism type
    pub gating_type: String,
    /// Dropout probability
    pub dropout: f64,
    /// Attention dropout
    pub attention_dropout: f64,
    /// Layer norm epsilon
    pub layer_norm_eps: f64,
    /// Whether to use attention bias
    pub use_bias: bool,
}

impl Default for FlamingoXAttentionConfig {
    fn default() -> Self {
        Self {
            cross_attention_dim: 2048,
            num_heads: 16,
            head_dim: 128,
            use_gating: true,
            gating_type: "tanh".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            use_bias: false,
        }
    }
}

impl FlamingoXAttentionConfig {
    /// Large cross-attention configuration
    pub fn large() -> Self {
        Self {
            cross_attention_dim: 4096,
            num_heads: 32,
            head_dim: 128,
            use_gating: true,
            gating_type: "tanh".to_string(),
            dropout: 0.0,
            attention_dropout: 0.0,
            layer_norm_eps: 1e-5,
            use_bias: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_flamingo_config_default() {
        let config = FlamingoConfig::default();
        assert_eq!(config.media_token_length, 64);
        assert!(config.use_gated_cross_attention);
        assert_eq!(config.vision_language_dim, 2048);
        assert_eq!(config.num_shots, 4);
        assert!(config.freeze_vision_encoder);
        assert!(!config.freeze_language_model);
    }

    #[test]
    fn test_flamingo_3b_config() {
        let config = FlamingoConfig::flamingo_3b();
        assert_eq!(config.language_config.hidden_size, 2048);
        assert_eq!(config.language_config.num_hidden_layers, 24);
        assert_eq!(config.vision_config.hidden_size, 1024);
        assert_eq!(config.perceiver_config.latent_dim, 2048);
        assert_eq!(config.cross_attention_config.cross_attention_dim, 2048);
    }

    #[test]
    fn test_flamingo_9b_config() {
        let config = FlamingoConfig::flamingo_9b();
        assert_eq!(config.language_config.hidden_size, 4096);
        assert_eq!(config.language_config.num_hidden_layers, 32);
        assert_eq!(config.vision_language_dim, 4096);
        assert_eq!(config.perceiver_config.latent_dim, 4096);
        assert_eq!(config.cross_attention_config.cross_attention_dim, 4096);
        assert_eq!(config.num_shots, 8);
    }

    #[test]
    fn test_flamingo_80b_config() {
        let config = FlamingoConfig::flamingo_80b();
        assert_eq!(config.language_config.hidden_size, 8192);
        assert_eq!(config.language_config.num_hidden_layers, 80);
        assert_eq!(config.language_config.num_key_value_heads, Some(8)); // GQA
        assert_eq!(config.vision_language_dim, 8192);
        assert_eq!(config.num_shots, 16);
        assert!(config.cross_attention_layers.len() > 30); // Many layers
    }

    #[test]
    fn test_open_flamingo_config() {
        let config = FlamingoConfig::open_flamingo();
        assert_eq!(config.language_config.vocab_size, 50432); // MPT vocab size
        assert_eq!(config.language_config.hidden_size, 4096);
        assert_eq!(config.vision_language_dim, 4096);
        assert_eq!(config.num_shots, 8);
    }

    #[test]
    fn test_vision_config_calculations() {
        let config = FlamingoVisionConfig::clip_vit_l();
        assert_eq!(config.num_patches(), 256); // (224/14)^2 = 16^2 = 256
        assert_eq!(config.seq_len(), 257); // 256 patches + 1 CLS token

        let config_b = FlamingoVisionConfig::clip_vit_b();
        assert_eq!(config_b.num_patches(), 196); // (224/16)^2 = 14^2 = 196
        assert_eq!(config_b.seq_len(), 197); // 196 patches + 1 CLS token
    }

    #[test]
    fn test_language_config_variants() {
        let chinchilla_1b = FlamingoLanguageConfig::chinchilla_1b();
        let chinchilla_7b = FlamingoLanguageConfig::chinchilla_7b();
        let chinchilla_70b = FlamingoLanguageConfig::chinchilla_70b();
        let mpt_7b = FlamingoLanguageConfig::mpt_7b();

        assert_eq!(chinchilla_1b.hidden_size, 2048);
        assert_eq!(chinchilla_7b.hidden_size, 4096);
        assert_eq!(chinchilla_70b.hidden_size, 8192);
        assert_eq!(mpt_7b.hidden_size, 4096);

        assert_eq!(chinchilla_1b.vocab_size, 32000);
        assert_eq!(chinchilla_7b.vocab_size, 32000);
        assert_eq!(chinchilla_70b.vocab_size, 32000);
        assert_eq!(mpt_7b.vocab_size, 50432);

        // Check GQA for 70B model
        assert_eq!(chinchilla_1b.num_key_value_heads, None);
        assert_eq!(chinchilla_7b.num_key_value_heads, None);
        assert_eq!(chinchilla_70b.num_key_value_heads, Some(8));
        assert_eq!(mpt_7b.num_key_value_heads, None);
    }

    #[test]
    fn test_perceiver_config_variants() {
        let default_config = FlamingoPerceiverConfig::default();
        let large_config = FlamingoPerceiverConfig::large();

        assert_eq!(default_config.latent_dim, 2048);
        assert_eq!(large_config.latent_dim, 4096);

        assert_eq!(default_config.num_heads, 16);
        assert_eq!(large_config.num_heads, 32);

        assert_eq!(default_config.num_layers, 6);
        assert_eq!(large_config.num_layers, 8);
    }

    #[test]
    fn test_cross_attention_config_variants() {
        let default_config = FlamingoXAttentionConfig::default();
        let large_config = FlamingoXAttentionConfig::large();

        assert_eq!(default_config.cross_attention_dim, 2048);
        assert_eq!(large_config.cross_attention_dim, 4096);

        assert_eq!(default_config.num_heads, 16);
        assert_eq!(large_config.num_heads, 32);

        assert!(default_config.use_gating);
        assert!(large_config.use_gating);

        assert_eq!(default_config.gating_type, "tanh");
        assert_eq!(large_config.gating_type, "tanh");
    }

    #[test]
    fn test_config_serialization() {
        let config = FlamingoConfig::flamingo_9b();
        let json = serde_json::to_string(&config).unwrap();
        let deserialized: FlamingoConfig = serde_json::from_str(&json).unwrap();

        assert_eq!(config.media_token_length, deserialized.media_token_length);
        assert_eq!(config.vision_language_dim, deserialized.vision_language_dim);
        assert_eq!(
            config.use_gated_cross_attention,
            deserialized.use_gated_cross_attention
        );
        assert_eq!(
            config.language_config.hidden_size,
            deserialized.language_config.hidden_size
        );
        assert_eq!(
            config.vision_config.num_patches(),
            deserialized.vision_config.num_patches()
        );
        assert_eq!(
            config.perceiver_config.num_latents,
            deserialized.perceiver_config.num_latents
        );
        assert_eq!(
            config.cross_attention_config.cross_attention_dim,
            deserialized.cross_attention_config.cross_attention_dim
        );
    }

    #[test]
    fn test_cross_attention_layer_distribution() {
        let config_3b = FlamingoConfig::flamingo_3b();
        let config_9b = FlamingoConfig::flamingo_9b();
        let config_80b = FlamingoConfig::flamingo_80b();

        // Check that cross-attention layers are reasonable
        assert!(!config_3b.cross_attention_layers.is_empty());
        assert!(!config_9b.cross_attention_layers.is_empty());
        assert!(!config_80b.cross_attention_layers.is_empty());

        // Larger models should have more cross-attention layers
        assert!(config_80b.cross_attention_layers.len() > config_9b.cross_attention_layers.len());
        assert!(config_9b.cross_attention_layers.len() >= config_3b.cross_attention_layers.len());

        // All layer indices should be valid
        for &layer_idx in &config_3b.cross_attention_layers {
            assert!(layer_idx < config_3b.language_config.num_hidden_layers);
        }

        for &layer_idx in &config_9b.cross_attention_layers {
            assert!(layer_idx < config_9b.language_config.num_hidden_layers);
        }

        for &layer_idx in &config_80b.cross_attention_layers {
            assert!(layer_idx < config_80b.language_config.num_hidden_layers);
        }
    }
}
