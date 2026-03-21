use serde::{Deserialize, Serialize};
use trustformers_core::traits::Config;

/// Performer model configuration
/// Reference: "Rethinking Attention with Performers" (Choromanski et al., 2020)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformerConfig {
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

    // Performer-specific parameters
    pub num_random_features: usize, // Number of random features for FAVOR+
    pub redraw_features: bool,      // Redraw random features during training
    pub feature_redraw_interval: usize, // Steps between feature redraws
    pub use_favor_plus: bool,       // Use FAVOR+ instead of regular FAVOR
    pub normalize_features: bool,   // Normalize random features
    pub causal_attention: bool,     // Use causal masking
    pub kernel_type: String,        // Kernel type: "relu", "exp", "softmax+"
    pub ortho_features: bool,       // Use orthogonal random features
    pub numerical_stabilizer: f32,  // Small value for numerical stability
}

impl Default for PerformerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 30522,
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

            // Performer defaults
            num_random_features: 256, // Typically much smaller than head_dim
            redraw_features: true,
            feature_redraw_interval: 1000,
            use_favor_plus: true,
            normalize_features: true,
            causal_attention: false, // Set to true for autoregressive models
            kernel_type: "relu".to_string(),
            ortho_features: true,
            numerical_stabilizer: 1e-6,
        }
    }
}

impl Config for PerformerConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.hidden_size % self.num_attention_heads != 0 {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "hidden_size must be divisible by num_attention_heads",
                "PerformerConfig::validate",
            ));
        }

        if !["relu", "exp", "softmax+"].contains(&self.kernel_type.as_str()) {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "kernel_type must be one of: relu, exp, softmax+",
                "PerformerConfig::validate",
            ));
        }

        let head_dim = self.head_dim();
        if self.num_random_features > 2 * head_dim {
            return Err(trustformers_core::errors::TrustformersError::config_error(
                "num_random_features should typically be <= 2 * head_dim for efficiency",
                "PerformerConfig::validate",
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Performer"
    }
}

impl PerformerConfig {
    /// Performer-Base configuration (similar to BERT-Base but with FAVOR+ attention)
    pub fn performer_base() -> Self {
        Self::default()
    }

    /// Performer-Large configuration
    pub fn performer_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            num_attention_heads: 16,
            intermediate_size: 4096,
            num_random_features: 512, // Scale with head dimension
            ..Self::default()
        }
    }

    /// Causal Performer for autoregressive generation
    pub fn performer_causal() -> Self {
        Self {
            causal_attention: true,
            kernel_type: "relu".to_string(), // ReLU kernel works well for causal
            ..Self::default()
        }
    }

    /// Long-sequence Performer (for very long sequences)
    pub fn performer_long() -> Self {
        Self {
            max_position_embeddings: 16384,
            num_random_features: 512, // Fixed size regardless of sequence length
            redraw_features: true,
            feature_redraw_interval: 500, // More frequent redraws for stability
            ..Self::default()
        }
    }

    /// Get the head dimension
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }

    /// Get the approximation ratio (how well FAVOR+ approximates full attention)
    pub fn approximation_quality(&self) -> f32 {
        // Higher num_random_features / head_dim gives better approximation
        self.num_random_features as f32 / self.head_dim() as f32
    }

    /// Check if using efficient attention
    pub fn is_efficient(&self) -> bool {
        self.num_random_features < self.max_position_embeddings
    }
}
