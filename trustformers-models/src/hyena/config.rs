use serde::{Deserialize, Serialize};
use trustformers_core::errors::invalid_config;
use trustformers_core::traits::Config;

/// Hyena model configuration
/// Reference: "HyenaDNA: Long-Range Genomic Sequence Modeling at Single Nucleotide Resolution"
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HyenaConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,

    // Hyena-specific parameters
    pub order: usize,                    // Order of the Hyena operator (typically 2)
    pub filter_order: usize,             // Order of the implicit filter
    pub local_order: usize,              // Local convolution order
    pub outer_mixing: bool,              // Use outer mixing in Hyena blocks
    pub conv_kernel_size: usize,         // Kernel size for local convolutions
    pub use_positional_embeddings: bool, // Use positional embeddings
    pub short_filter_order: usize,       // Short filter for efficiency
    pub modulate: bool,                  // Use modulation in filters
    pub w: f32,                          // Width parameter for filter initialization
    pub wd: f32,                         // Decay parameter for filter initialization
    pub bias: bool,                      // Use bias in linear layers
    pub num_inner_mlps: usize,           // Number of inner MLPs
    pub normalized: bool,                // Use normalization
    pub use_flashfft: bool,              // Use FlashFFT for convolutions
}

impl Default for HyenaConfig {
    fn default() -> Self {
        Self {
            vocab_size: 50257,
            hidden_size: 768,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 32768, // Support long sequences
            initializer_range: 0.02,
            layer_norm_eps: 1e-5,
            pad_token_id: 0,

            // Hyena defaults
            order: 2,
            filter_order: 64,
            local_order: 3,
            outer_mixing: true,
            conv_kernel_size: 3,
            use_positional_embeddings: false,
            short_filter_order: 3,
            modulate: true,
            w: 1.0,
            wd: 0.1,
            bias: true,
            num_inner_mlps: 2,
            normalized: false,
            use_flashfft: true,
        }
    }
}

impl Config for HyenaConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        if self.order < 2 {
            return Err(invalid_config(
                "config_field",
                "Hyena order must be at least 2".to_string(),
            ));
        }

        if self.filter_order == 0 {
            return Err(invalid_config(
                "config_field",
                "filter_order must be greater than 0".to_string(),
            ));
        }

        if self.conv_kernel_size % 2 == 0 {
            return Err(invalid_config(
                "config_field",
                "conv_kernel_size should be odd for symmetric padding".to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "Hyena"
    }
}

impl HyenaConfig {
    /// Hyena-Small configuration (similar to GPT-2 Small)
    pub fn hyena_small() -> Self {
        Self {
            hidden_size: 768,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            max_position_embeddings: 32768,
            ..Self::default()
        }
    }

    /// Hyena-Medium configuration
    pub fn hyena_medium() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            max_position_embeddings: 65536,
            filter_order: 128,
            ..Self::default()
        }
    }

    /// Hyena-Large configuration
    pub fn hyena_large() -> Self {
        Self {
            hidden_size: 1280,
            num_hidden_layers: 36,
            intermediate_size: 5120,
            max_position_embeddings: 131072,
            filter_order: 256,
            ..Self::default()
        }
    }

    /// HyenaDNA configuration for genomic sequences
    pub fn hyena_dna() -> Self {
        Self {
            vocab_size: 12, // DNA nucleotides + special tokens
            hidden_size: 256,
            num_hidden_layers: 8,
            intermediate_size: 1024,
            max_position_embeddings: 1048576, // 1M sequence length
            filter_order: 64,
            order: 2,
            use_positional_embeddings: false,
            ..Self::default()
        }
    }

    /// Long-context Hyena for very long sequences
    pub fn hyena_long() -> Self {
        Self {
            max_position_embeddings: 262144, // 256K context
            filter_order: 512,
            use_flashfft: true,
            ..Self::default()
        }
    }

    /// Get the effective receptive field
    pub fn receptive_field(&self) -> usize {
        // Hyena's receptive field grows with filter order and sequence length
        self.filter_order * self.num_hidden_layers
    }

    /// Get memory complexity relative to attention
    pub fn memory_advantage(&self) -> f32 {
        let seq_len = self.max_position_embeddings as f32;
        let attention_memory = seq_len * seq_len;
        let hyena_memory = seq_len * self.filter_order as f32;
        attention_memory / hyena_memory
    }

    /// Check if configuration supports very long sequences efficiently
    pub fn is_long_context_optimized(&self) -> bool {
        self.max_position_embeddings >= 32768 && self.use_flashfft
    }
}
