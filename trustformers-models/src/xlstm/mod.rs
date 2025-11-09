//! Extended LSTM (xLSTM) Implementation
//!
//! This module implements the Extended LSTM architecture as introduced in
//! "xLSTM: Extended Long Short-Term Memory" by Sepp Hochreiter et al. (2024).
//!
//! xLSTM extends traditional LSTM with two key innovations:
//! 1. **Exponential gating**: Enhances memory capacity by allowing gates to take
//!    exponential values rather than being constrained to (0,1)
//! 2. **Enhanced memory structures**: Two variants:
//!    - sLSTM (scalar memory): Enhanced scalar memory with exponential gating
//!    - mLSTM (matrix memory): Matrix-based memory that is fully parallelizable
//!
//! ## Key Features
//!
//! - **Scalability**: Supports models from small (512d) to large (7B parameters)
//! - **Parallelization**: mLSTM blocks are fully parallelizable like transformers
//! - **Long sequences**: Improved handling of long-range dependencies
//! - **Memory efficiency**: Enhanced memory structures with better retention
//! - **Flexible architecture**: Mix and match sLSTM and mLSTM blocks
//!
//! ## Usage
//!
//! ```rust
//! use trustformers_models::xlstm::{XLSTMConfig, XLSTMModel};
//!
//! // Create a base xLSTM model
//! let config = XLSTMConfig::base();
//! let model = XLSTMModel::new(config)?;
//!
//! // Create a large 7B parameter model
//! let large_config = XLSTMConfig::xlstm_7b();
//! let large_model = XLSTMModel::new(large_config)?;
//! ```
//!
//! ## Architecture Variants
//!
//! - **sLSTM blocks**: Scalar memory with exponential gating
//! - **mLSTM blocks**: Matrix memory with multi-head attention-like mechanism
//! - **Mixed models**: Combination of both block types for optimal performance
//!
//! ## Performance Characteristics
//!
//! Based on the original paper, xLSTM achieves:
//! - Competitive performance with transformers on language modeling
//! - Better long-range dependency modeling than standard LSTM
//! - Improved parallelization compared to traditional RNNs
//! - Efficient memory usage for very long sequences

pub mod config;
pub mod model;

pub use config::{
    ExponentialGatingConfig, MLstmConfig, SLstmConfig, XLSTMBlockConfig, XLSTMBlockType,
    XLSTMConfig,
};
pub use model::{
    FeedForward, MLstmBlock, MLstmState, SLstmBlock, SLstmState, XLSTMForCausalLM,
    XLSTMForSequenceClassification, XLSTMLayer, XLSTMModel, XLSTMState,
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_xlstm_config_creation() {
        let config = XLSTMConfig::default();
        assert_eq!(config.vocab_size, 32000);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
        assert_eq!(config.num_heads, 12);
    }

    #[test]
    fn test_xlstm_config_variants() {
        let small = XLSTMConfig::small();
        assert_eq!(small.hidden_size, 512);
        assert_eq!(small.num_layers, 8);

        let large = XLSTMConfig::large();
        assert_eq!(large.hidden_size, 1024);
        assert_eq!(large.num_layers, 24);

        let xlstm_7b = XLSTMConfig::xlstm_7b();
        assert_eq!(xlstm_7b.hidden_size, 4096);
        assert_eq!(xlstm_7b.num_layers, 32);
    }

    #[test]
    fn test_mlstm_config_head_dimension() {
        let config = MLstmConfig::new(768, 12);
        assert_eq!(config.head_dim, 64);
        assert_eq!(config.num_heads, 12);
        assert_eq!(config.hidden_size, 768);
    }

    #[test]
    #[should_panic(expected = "Hidden size must be divisible by number of heads")]
    fn test_mlstm_config_invalid_heads() {
        MLstmConfig::new(768, 11); // Should panic as 768 % 11 != 0
    }

    #[test]
    fn test_xlstm_block_pattern() {
        let config = XLSTMConfig::xlstm_7b();
        assert_eq!(config.block_config.block_pattern.len(), 32);

        // Check that pattern alternates correctly
        for (i, block_type) in config.block_config.block_pattern.iter().enumerate() {
            let expected = if i % 3 == 0 { &XLSTMBlockType::SLstm } else { &XLSTMBlockType::MLstm };
            assert_eq!(block_type, expected);
        }
    }

    #[test]
    fn test_exponential_gating_config() {
        let config = ExponentialGatingConfig::default();
        assert!(config.enabled);
        assert_eq!(config.min_gate_value, 1e-6);
        assert_eq!(config.max_gate_value, 10.0);
        assert_eq!(config.temperature, 1.0);
    }

    #[test]
    fn test_xlstm_state_creation() {
        let config = XLSTMConfig::small();
        let batch_size = 4;

        let state = XLSTMState::new(batch_size, config.hidden_size);

        assert_eq!(state.batch_size, batch_size);
        assert_eq!(state.hidden_size, config.hidden_size);
    }

    #[test]
    fn test_xlstm_model_creation() -> trustformers_core::errors::Result<()> {
        let config = XLSTMConfig::small();
        let model = XLSTMModel::new(config.clone())?;

        assert_eq!(model.config().vocab_size, config.vocab_size);
        assert_eq!(model.config().hidden_size, config.hidden_size);

        Ok(())
    }

    #[test]
    fn test_xlstm_parameter_counting() -> trustformers_core::errors::Result<()> {
        let config = XLSTMConfig::small();
        let model = XLSTMModel::new(config)?;

        let param_count = model.parameter_count();
        assert!(param_count > 0);

        // Basic sanity check - should be reasonable for a 512-hidden model
        assert!(param_count > 1_000_000); // At least 1M parameters
        assert!(param_count < 100_000_000); // But not more than 100M for small model

        Ok(())
    }

    #[test]
    fn test_xlstm_forward_pass() -> trustformers_core::errors::Result<()> {
        let config = XLSTMConfig::small();
        let model = XLSTMModel::new(config.clone())?;

        let input_ids = vec![1u32, 2u32, 3u32, 4u32, 5u32];
        let output = model.forward(input_ids.clone())?;

        // Check that output has reasonable structure
        match output.logits {
            trustformers_core::tensor::Tensor::F32(ref arr) => {
                let shape = arr.shape();
                assert_eq!(shape[0], 1); // batch size
                assert_eq!(shape[1], input_ids.len()); // sequence length
                assert_eq!(shape[2], config.vocab_size); // vocab size
            },
            _ => panic!("Expected F32 tensor"),
        }

        Ok(())
    }

    #[test]
    fn test_xlstm_classification_model() -> trustformers_core::errors::Result<()> {
        let config = XLSTMConfig::small();
        let num_labels = 5;
        let model = XLSTMForSequenceClassification::new(config, num_labels)?;

        let input_ids = vec![1u32, 2u32, 3u32, 4u32, 5u32];
        let output = model.forward(input_ids)?;

        match output {
            trustformers_core::tensor::Tensor::F32(ref arr) => {
                let shape = arr.shape();
                assert_eq!(shape[0], 1); // batch size
                assert_eq!(shape[1], num_labels); // number of labels
            },
            _ => panic!("Expected F32 tensor"),
        }

        Ok(())
    }

    #[test]
    fn test_slstm_block_creation() {
        let block = SLstmBlock::new(512);
        let param_count = block.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_mlstm_block_creation() {
        let block = MLstmBlock::new(768, 12);
        let param_count = block.parameter_count();
        assert!(param_count > 0);
    }

    #[test]
    fn test_feedforward_network() {
        let hidden_size = 512;
        let intermediate_size = 2048;

        let ff = FeedForward::new(hidden_size, intermediate_size);
        let param_count = ff.parameter_count();
        assert!(param_count > 0);

        // Should be 2 linear layers
        let expected = hidden_size * intermediate_size
            + intermediate_size
            + intermediate_size * hidden_size
            + hidden_size;
        assert_eq!(param_count, expected);
    }
}
