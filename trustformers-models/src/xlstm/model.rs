//! Extended LSTM (xLSTM) Model Implementation (Simplified)
//!
//! This module implements a simplified version of the Extended LSTM architecture
//! following the patterns used in other models in this codebase.

use crate::xlstm::config::XLSTMConfig;
use scirs2_core::ndarray::{ArrayD, IxDyn}; // SciRS2 Integration Policy
use trustformers_core::device::Device;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;

/// Extended LSTM model structure
#[derive(Debug, Clone)]
pub struct XLSTMModel {
    config: XLSTMConfig,
    device: Device,
}

impl XLSTMModel {
    /// Create a new xLSTM model on CPU (backward compatibility)
    pub fn new(config: XLSTMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new xLSTM model with specified device
    pub fn new_with_device(config: XLSTMConfig, device: Device) -> Result<Self> {
        Ok(Self { config, device })
    }

    /// Get model configuration
    pub fn config(&self) -> &XLSTMConfig {
        &self.config
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Forward pass for the model
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<XLSTMOutput> {
        let batch_size = 1; // Simplified for single batch
        let seq_len = input_ids.len();

        // Create output tensor placeholder
        let output_shape = vec![batch_size, seq_len, self.config.vocab_size];
        let output_data = vec![0.0f32; batch_size * seq_len * self.config.vocab_size];

        let logits = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        );

        Ok(XLSTMOutput {
            logits,
            hidden_states: None,
            attentions: None,
        })
    }

    /// Count model parameters
    pub fn parameter_count(&self) -> usize {
        // Estimated parameter count for xLSTM
        let embedding_params = self.config.vocab_size * self.config.hidden_size;
        let lstm_params =
            4 * (self.config.hidden_size + self.config.hidden_size) * self.config.hidden_size; // 4 gates
        let layer_params = lstm_params * self.config.num_layers;
        let output_params = self.config.hidden_size * self.config.vocab_size;

        embedding_params + layer_params + output_params
    }
}

/// Output structure for xLSTM model
#[derive(Debug, Clone)]
pub struct XLSTMOutput {
    /// Logits tensor [batch_size, sequence_length, vocab_size]
    pub logits: Tensor,
    /// Optional hidden states
    pub hidden_states: Option<Vec<Tensor>>,
    /// Optional attention weights
    pub attentions: Option<Vec<Tensor>>,
}

/// xLSTM model for causal language modeling
#[derive(Debug, Clone)]
pub struct XLSTMForCausalLM {
    xlstm: XLSTMModel,
    device: Device,
}

impl XLSTMForCausalLM {
    /// Create a new xLSTM for causal LM on CPU (backward compatibility)
    pub fn new(config: XLSTMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new xLSTM for causal LM with specified device
    pub fn new_with_device(config: XLSTMConfig, device: Device) -> Result<Self> {
        let xlstm = XLSTMModel::new_with_device(config, device)?;
        Ok(Self { xlstm, device })
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<XLSTMOutput> {
        self.xlstm.forward(input_ids)
    }

    pub fn parameter_count(&self) -> usize {
        self.xlstm.parameter_count()
    }
}

/// xLSTM model for sequence classification
#[derive(Debug, Clone)]
pub struct XLSTMForSequenceClassification {
    xlstm: XLSTMModel,
    num_labels: usize,
    device: Device,
}

impl XLSTMForSequenceClassification {
    /// Create a new xLSTM for sequence classification on CPU (backward compatibility)
    pub fn new(config: XLSTMConfig, num_labels: usize) -> Result<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    /// Create a new xLSTM for sequence classification with specified device
    pub fn new_with_device(config: XLSTMConfig, num_labels: usize, device: Device) -> Result<Self> {
        let xlstm = XLSTMModel::new_with_device(config, device)?;
        Ok(Self {
            xlstm,
            num_labels,
            device,
        })
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let _xlstm_output = self.xlstm.forward(input_ids)?;

        // Extract logits and pool to classification output
        let output_shape = vec![1, self.num_labels];
        let output_data = vec![0.0f32; self.num_labels];

        let classification_logits = Tensor::F32(
            ArrayD::from_shape_vec(IxDyn(&output_shape), output_data)
                .map_err(|e| TrustformersError::shape_error(e.to_string()))?,
        );

        Ok(classification_logits)
    }

    pub fn parameter_count(&self) -> usize {
        let base_params = self.xlstm.parameter_count();
        let classification_params =
            self.xlstm.config.hidden_size * self.num_labels + self.num_labels;
        base_params + classification_params
    }
}

/// Simplified xLSTM layer implementation
#[derive(Debug, Clone)]
pub struct XLSTMLayer {
    hidden_size: usize,
    block_type: crate::xlstm::config::XLSTMBlockType,
    device: Device,
}

impl XLSTMLayer {
    /// Create a new xLSTM layer on CPU (backward compatibility)
    pub fn new(hidden_size: usize, block_type: crate::xlstm::config::XLSTMBlockType) -> Self {
        Self::new_with_device(hidden_size, block_type, Device::CPU)
    }

    /// Create a new xLSTM layer with specified device
    pub fn new_with_device(
        hidden_size: usize,
        block_type: crate::xlstm::config::XLSTMBlockType,
        device: Device,
    ) -> Self {
        Self {
            hidden_size,
            block_type,
            device,
        }
    }

    /// Get the device this layer is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        match self.block_type {
            crate::xlstm::config::XLSTMBlockType::SLstm => {
                4 * (self.hidden_size + self.hidden_size) * self.hidden_size // 4 LSTM gates
            },
            crate::xlstm::config::XLSTMBlockType::MLstm => {
                3 * self.hidden_size * self.hidden_size + // Q, K, V projections
                3 * self.hidden_size * self.hidden_size // Gate projections
            },
            crate::xlstm::config::XLSTMBlockType::Mixed => {
                // Combined sLSTM and mLSTM parameters
                4 * (self.hidden_size + self.hidden_size) * self.hidden_size
                    + 6 * self.hidden_size * self.hidden_size
            },
        }
    }
}

/// xLSTM state container
#[derive(Debug, Clone)]
pub struct XLSTMState {
    pub batch_size: usize,
    pub hidden_size: usize,
}

impl XLSTMState {
    pub fn new(batch_size: usize, hidden_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
        }
    }
}

/// Simplified sLSTM block (Scalar LSTM with exponential gating)
#[derive(Debug, Clone)]
pub struct SLstmBlock {
    hidden_size: usize,
    device: Device,
}

impl SLstmBlock {
    /// Create a new sLSTM block on CPU (backward compatibility)
    pub fn new(hidden_size: usize) -> Self {
        Self::new_with_device(hidden_size, Device::CPU)
    }

    /// Create a new sLSTM block with specified device
    pub fn new_with_device(hidden_size: usize, device: Device) -> Self {
        Self {
            hidden_size,
            device,
        }
    }

    /// Get the device this block is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        4 * (self.hidden_size + self.hidden_size) * self.hidden_size
    }
}

/// sLSTM state
#[derive(Debug, Clone)]
pub struct SLstmState {
    pub hidden_size: usize,
}

impl SLstmState {
    pub fn new(hidden_size: usize) -> Self {
        Self { hidden_size }
    }
}

/// Simplified mLSTM block (Matrix LSTM with matrix memory)
#[derive(Debug, Clone)]
pub struct MLstmBlock {
    hidden_size: usize,
    #[allow(dead_code)]
    num_heads: usize,
    device: Device,
}

impl MLstmBlock {
    /// Create a new mLSTM block on CPU (backward compatibility)
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self::new_with_device(hidden_size, num_heads, Device::CPU)
    }

    /// Create a new mLSTM block with specified device
    pub fn new_with_device(hidden_size: usize, num_heads: usize, device: Device) -> Self {
        Self {
            hidden_size,
            num_heads,
            device,
        }
    }

    /// Get the device this block is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        6 * self.hidden_size * self.hidden_size // Q, K, V + 3 gate projections
    }
}

/// mLSTM state
#[derive(Debug, Clone)]
pub struct MLstmState {
    pub hidden_size: usize,
    pub num_heads: usize,
}

impl MLstmState {
    pub fn new(hidden_size: usize, num_heads: usize) -> Self {
        Self {
            hidden_size,
            num_heads,
        }
    }
}

/// Simple feedforward network representation
#[derive(Debug, Clone)]
pub struct FeedForward {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    device: Device,
}

impl FeedForward {
    /// Create a new feedforward network on CPU (backward compatibility)
    pub fn new(hidden_size: usize, intermediate_size: usize) -> Self {
        Self::new_with_device(hidden_size, intermediate_size, Device::CPU)
    }

    /// Create a new feedforward network with specified device
    pub fn new_with_device(hidden_size: usize, intermediate_size: usize, device: Device) -> Self {
        Self {
            hidden_size,
            intermediate_size,
            device,
        }
    }

    /// Get the device this network is on
    pub fn device(&self) -> Device {
        self.device
    }

    pub fn parameter_count(&self) -> usize {
        // Two linear layers with bias
        let linear1_params = self.hidden_size * self.intermediate_size + self.intermediate_size;
        let linear2_params = self.intermediate_size * self.hidden_size + self.hidden_size;
        linear1_params + linear2_params
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::xlstm::config::XLSTMBlockType;

    #[test]
    fn test_xlstm_model_device_support() -> Result<()> {
        let config = XLSTMConfig::default();

        // Test CPU creation (default)
        let model_cpu = XLSTMModel::new(config.clone())?;
        assert_eq!(model_cpu.device(), Device::CPU);

        // Test explicit CPU creation
        let model_cpu_explicit = XLSTMModel::new_with_device(config.clone(), Device::CPU)?;
        assert_eq!(model_cpu_explicit.device(), Device::CPU);

        // Test Metal device creation
        let model_metal = XLSTMModel::new_with_device(config.clone(), Device::Metal(0))?;
        assert_eq!(model_metal.device(), Device::Metal(0));

        // Test CUDA device creation
        let model_cuda = XLSTMModel::new_with_device(config.clone(), Device::CUDA(0))?;
        assert_eq!(model_cuda.device(), Device::CUDA(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_for_causal_lm_device_support() -> Result<()> {
        let config = XLSTMConfig::default();

        // Test CPU creation (default)
        let model_cpu = XLSTMForCausalLM::new(config.clone())?;
        assert_eq!(model_cpu.device(), Device::CPU);

        // Test explicit device creation
        let model_metal = XLSTMForCausalLM::new_with_device(config.clone(), Device::Metal(0))?;
        assert_eq!(model_metal.device(), Device::Metal(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_for_sequence_classification_device_support() -> Result<()> {
        let config = XLSTMConfig::default();
        let num_labels = 2;

        // Test CPU creation (default)
        let model_cpu = XLSTMForSequenceClassification::new(config.clone(), num_labels)?;
        assert_eq!(model_cpu.device(), Device::CPU);

        // Test explicit device creation
        let model_cuda = XLSTMForSequenceClassification::new_with_device(
            config.clone(),
            num_labels,
            Device::CUDA(0),
        )?;
        assert_eq!(model_cuda.device(), Device::CUDA(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_layer_device_support() {
        let hidden_size = 768;
        let block_type = XLSTMBlockType::Mixed;

        // Test CPU creation (default)
        let layer_cpu = XLSTMLayer::new(hidden_size, block_type.clone());
        assert_eq!(layer_cpu.device(), Device::CPU);

        // Test explicit device creation
        let layer_metal =
            XLSTMLayer::new_with_device(hidden_size, block_type.clone(), Device::Metal(0));
        assert_eq!(layer_metal.device(), Device::Metal(0));
    }

    #[test]
    fn test_slstm_block_device_support() {
        let hidden_size = 768;

        // Test CPU creation (default)
        let block_cpu = SLstmBlock::new(hidden_size);
        assert_eq!(block_cpu.device(), Device::CPU);

        // Test explicit device creation
        let block_cuda = SLstmBlock::new_with_device(hidden_size, Device::CUDA(0));
        assert_eq!(block_cuda.device(), Device::CUDA(0));
    }

    #[test]
    fn test_mlstm_block_device_support() {
        let hidden_size = 768;
        let num_heads = 12;

        // Test CPU creation (default)
        let block_cpu = MLstmBlock::new(hidden_size, num_heads);
        assert_eq!(block_cpu.device(), Device::CPU);

        // Test explicit device creation
        let block_metal = MLstmBlock::new_with_device(hidden_size, num_heads, Device::Metal(0));
        assert_eq!(block_metal.device(), Device::Metal(0));
    }

    #[test]
    fn test_feedforward_device_support() {
        let hidden_size = 768;
        let intermediate_size = 3072;

        // Test CPU creation (default)
        let ff_cpu = FeedForward::new(hidden_size, intermediate_size);
        assert_eq!(ff_cpu.device(), Device::CPU);

        // Test explicit device creation
        let ff_cuda = FeedForward::new_with_device(hidden_size, intermediate_size, Device::CUDA(0));
        assert_eq!(ff_cuda.device(), Device::CUDA(0));
    }

    #[test]
    fn test_device_propagation() -> Result<()> {
        let config = XLSTMConfig::default();

        // Create model on Metal
        let model = XLSTMModel::new_with_device(config.clone(), Device::Metal(0))?;

        // Verify device is Metal
        assert_eq!(model.device(), Device::Metal(0));

        // Create causal LM model on CUDA
        let causal_lm = XLSTMForCausalLM::new_with_device(config.clone(), Device::CUDA(1))?;
        assert_eq!(causal_lm.device(), Device::CUDA(1));
        assert_eq!(causal_lm.xlstm.device(), Device::CUDA(1));

        Ok(())
    }

    #[test]
    fn test_backward_compatibility() -> Result<()> {
        let config = XLSTMConfig::default();

        // Old API should still work and default to CPU
        let model = XLSTMModel::new(config.clone())?;
        assert_eq!(model.device(), Device::CPU);

        let causal_lm = XLSTMForCausalLM::new(config.clone())?;
        assert_eq!(causal_lm.device(), Device::CPU);

        let seq_class = XLSTMForSequenceClassification::new(config.clone(), 2)?;
        assert_eq!(seq_class.device(), Device::CPU);

        Ok(())
    }
}
