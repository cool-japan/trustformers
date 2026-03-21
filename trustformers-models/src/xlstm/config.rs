//! xLSTM Configuration
//!
//! Configuration structures for Extended LSTM (xLSTM) models,
//! including both scalar memory (sLSTM) and matrix memory (mLSTM) variants.
//!
//! Reference: "xLSTM: Extended Long Short-Term Memory" (Hochreiter et al., 2024)

use serde::{Deserialize, Serialize};

/// Configuration for xLSTM models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XLSTMConfig {
    /// Vocabulary size
    pub vocab_size: usize,
    /// Hidden dimension
    pub hidden_size: usize,
    /// Intermediate dimension (feedforward)
    pub intermediate_size: usize,
    /// Number of layers
    pub num_layers: usize,
    /// Number of attention heads for mLSTM
    pub num_heads: usize,
    /// Maximum sequence length
    pub max_sequence_length: usize,
    /// Dropout probability
    pub dropout: f32,
    /// Layer normalization epsilon
    pub layer_norm_epsilon: f64,
    /// xLSTM block configuration
    pub block_config: XLSTMBlockConfig,
    /// Initial forget gate bias (for stability)
    pub initial_forget_gate_bias: f32,
    /// Whether to use pre-layer normalization
    pub use_pre_ln: bool,
    /// Whether to use post-layer normalization
    pub use_post_ln: bool,
    /// Exponential gating configuration
    pub exponential_gating: ExponentialGatingConfig,
}

impl Default for XLSTMConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 768,
            intermediate_size: 3072,
            num_layers: 12,
            num_heads: 12,
            max_sequence_length: 2048,
            dropout: 0.1,
            layer_norm_epsilon: 1e-5,
            block_config: XLSTMBlockConfig::default(),
            initial_forget_gate_bias: 3.0, // High bias for initial forgetting
            use_pre_ln: true,
            use_post_ln: false,
            exponential_gating: ExponentialGatingConfig::default(),
        }
    }
}

/// Configuration for xLSTM block structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct XLSTMBlockConfig {
    /// Block type (sLSTM or mLSTM)
    pub block_type: XLSTMBlockType,
    /// Number of sLSTM blocks before mLSTM
    pub slstm_blocks: usize,
    /// Number of mLSTM blocks
    pub mlstm_blocks: usize,
    /// Pattern of block arrangement
    pub block_pattern: Vec<XLSTMBlockType>,
}

impl Default for XLSTMBlockConfig {
    fn default() -> Self {
        Self {
            block_type: XLSTMBlockType::Mixed,
            slstm_blocks: 4,
            mlstm_blocks: 8,
            block_pattern: vec![
                XLSTMBlockType::SLstm,
                XLSTMBlockType::SLstm,
                XLSTMBlockType::MLstm,
                XLSTMBlockType::MLstm,
                XLSTMBlockType::SLstm,
                XLSTMBlockType::MLstm,
            ],
        }
    }
}

/// xLSTM block types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum XLSTMBlockType {
    /// Scalar memory LSTM
    SLstm,
    /// Matrix memory LSTM
    MLstm,
    /// Mixed blocks (sLSTM + mLSTM)
    Mixed,
}

/// Configuration for exponential gating mechanism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExponentialGatingConfig {
    /// Whether to enable exponential gating
    pub enabled: bool,
    /// Minimum gate value (for numerical stability)
    pub min_gate_value: f32,
    /// Maximum gate value (to prevent overflow)
    pub max_gate_value: f32,
    /// Temperature for exponential function
    pub temperature: f32,
}

impl Default for ExponentialGatingConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_gate_value: 1e-6,
            max_gate_value: 10.0,
            temperature: 1.0,
        }
    }
}

/// Configuration for scalar memory LSTM (sLSTM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SLstmConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Whether to use exponential gating
    pub use_exponential_gating: bool,
    /// Whether to use memory mixing
    pub use_memory_mixing: bool,
    /// Dropout probability
    pub dropout: f32,
}

impl Default for SLstmConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            use_exponential_gating: true,
            use_memory_mixing: true,
            dropout: 0.1,
        }
    }
}

/// Configuration for matrix memory LSTM (mLSTM)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MLstmConfig {
    /// Hidden dimension
    pub hidden_size: usize,
    /// Number of attention heads
    pub num_heads: usize,
    /// Head dimension (derived from hidden_size / num_heads)
    pub head_dim: usize,
    /// Whether to use causal masking
    pub use_causal_mask: bool,
    /// Whether to use exponential gating
    pub use_exponential_gating: bool,
    /// Dropout probability
    pub dropout: f32,
    /// Matrix memory dimension
    pub memory_dimension: usize,
}

impl Default for MLstmConfig {
    fn default() -> Self {
        Self {
            hidden_size: 768,
            num_heads: 12,
            head_dim: 64, // 768 / 12
            use_causal_mask: true,
            use_exponential_gating: true,
            dropout: 0.1,
            memory_dimension: 64,
        }
    }
}

impl MLstmConfig {
    /// Create a new mLSTM configuration with proper head dimension calculation
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        assert!(
            hidden_size % num_heads == 0,
            "Hidden size must be divisible by number of heads"
        );

        Self {
            hidden_size,
            num_heads,
            head_dim: hidden_size / num_heads,
            ..Default::default()
        }
    }
}

/// Predefined xLSTM model variants
impl XLSTMConfig {
    /// Small xLSTM model (similar to BERT-base)
    pub fn small() -> Self {
        Self {
            vocab_size: 32000,
            hidden_size: 512,
            intermediate_size: 2048,
            num_layers: 8,
            num_heads: 8,
            max_sequence_length: 1024,
            ..Default::default()
        }
    }

    /// Base xLSTM model
    pub fn base() -> Self {
        Self::default()
    }

    /// Large xLSTM model
    pub fn large() -> Self {
        Self {
            vocab_size: 50000,
            hidden_size: 1024,
            intermediate_size: 4096,
            num_layers: 24,
            num_heads: 16,
            max_sequence_length: 4096,
            ..Default::default()
        }
    }

    /// xLSTM 7B model (similar to paper)
    pub fn xlstm_7b() -> Self {
        Self {
            vocab_size: 50000,
            hidden_size: 4096,
            intermediate_size: 16384,
            num_layers: 32,
            num_heads: 32,
            max_sequence_length: 8192,
            block_config: XLSTMBlockConfig {
                block_type: XLSTMBlockType::Mixed,
                slstm_blocks: 12,
                mlstm_blocks: 20,
                block_pattern: (0..32)
                    .map(
                        |i| {
                            if i % 3 == 0 {
                                XLSTMBlockType::SLstm
                            } else {
                                XLSTMBlockType::MLstm
                            }
                        },
                    )
                    .collect(),
            },
            ..Default::default()
        }
    }
}
