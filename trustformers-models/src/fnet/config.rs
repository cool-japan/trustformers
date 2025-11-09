use serde::{Deserialize, Serialize};
use trustformers_core::{errors::invalid_config, traits::Config};

/// FNet model configuration
/// Reference: "FNet: Mixing Tokens with Fourier Transforms" (Lee-Thorp et al., 2021)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FNetConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub intermediate_size: usize,
    pub hidden_act: String,
    pub hidden_dropout_prob: f32,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    pub initializer_range: f32,
    pub layer_norm_eps: f32,
    pub pad_token_id: u32,
    pub position_embedding_type: String,

    // FNet-specific parameters
    pub use_fourier_transform: bool, // Use DFT instead of attention
    pub use_tpu_optimized_fft: bool, // Use TPU-optimized FFT variants
    pub fourier_transform_type: String, // "dft", "real_dft", "dct"
    pub use_bias_in_fourier: bool,   // Add bias after Fourier transform
    pub fourier_dropout_prob: f32,   // Dropout after Fourier layer
}

impl Default for FNetConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000, // Use larger vocab like T5
            hidden_size: 768,
            num_hidden_layers: 12,
            intermediate_size: 3072,
            hidden_act: "gelu".to_string(),
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 4, // FNet often uses more token types
            initializer_range: 0.02,
            layer_norm_eps: 1e-12,
            pad_token_id: 0,
            position_embedding_type: "absolute".to_string(),

            // FNet defaults
            use_fourier_transform: true,
            use_tpu_optimized_fft: false,
            fourier_transform_type: "dft".to_string(),
            use_bias_in_fourier: true,
            fourier_dropout_prob: 0.0, // Usually no dropout on Fourier layer
        }
    }
}

impl Config for FNetConfig {
    fn validate(&self) -> trustformers_core::errors::Result<()> {
        // No head constraints since FNet doesn't use attention heads

        if !["dft", "real_dft", "dct"].contains(&self.fourier_transform_type.as_str()) {
            return Err(trustformers_core::errors::invalid_config(
                "fourier_transform_type",
                "fourier_transform_type must be one of: dft, real_dft, dct",
            ));
        }

        // Check if sequence length is reasonable for FFT
        if self.max_position_embeddings > 8192 {
            return Err(invalid_config(
                "config_field",
                "max_position_embeddings > 8192 may be inefficient for FFT. Consider chunking."
                    .to_string(),
            ));
        }

        Ok(())
    }

    fn architecture(&self) -> &'static str {
        "FNet"
    }
}

impl FNetConfig {
    /// FNet-Base configuration
    pub fn fnet_base() -> Self {
        Self::default()
    }

    /// FNet-Large configuration
    pub fn fnet_large() -> Self {
        Self {
            hidden_size: 1024,
            num_hidden_layers: 24,
            intermediate_size: 4096,
            ..Self::default()
        }
    }

    /// FNet optimized for TPU training
    pub fn fnet_tpu() -> Self {
        Self {
            use_tpu_optimized_fft: true,
            fourier_transform_type: "real_dft".to_string(), // More efficient on TPU
            max_position_embeddings: 1024,                  // Power of 2 for efficiency
            ..Self::default()
        }
    }

    /// FNet with DCT (Discrete Cosine Transform) instead of DFT
    pub fn fnet_dct() -> Self {
        Self {
            fourier_transform_type: "dct".to_string(),
            max_position_embeddings: 1024,
            ..Self::default()
        }
    }

    /// Long-sequence FNet (up to 4K tokens)
    pub fn fnet_long() -> Self {
        Self {
            max_position_embeddings: 4096,
            fourier_transform_type: "real_dft".to_string(), // More efficient for long sequences
            ..Self::default()
        }
    }

    /// Get theoretical complexity advantage over attention
    pub fn complexity_advantage(&self) -> f32 {
        let n = self.max_position_embeddings as f32;
        let attention_complexity = n * n; // O(n²)
        let fourier_complexity = n * n.log2(); // O(n log n)
        attention_complexity / fourier_complexity
    }

    /// Check if configuration is optimized for efficiency
    pub fn is_efficient_config(&self) -> bool {
        // Check if sequence length is power of 2 (optimal for FFT)
        let n = self.max_position_embeddings;
        n > 0 && (n & (n - 1)) == 0
    }

    /// Get recommended batch size for efficiency
    pub fn recommended_batch_size(&self) -> usize {
        // Fourier transforms are more batch-friendly than attention
        match self.hidden_size {
            768 => 64,  // Base model
            1024 => 32, // Large model
            _ => 16,    // Conservative default
        }
    }
}
