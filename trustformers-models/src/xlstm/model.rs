//! Extended LSTM (xLSTM) model.
//!
//! Puts the [`SLstmBlock`](crate::xlstm::SLstmBlock) and
//! [`MLstmBlock`](crate::xlstm::MLstmBlock) recurrent cells into a pre-LayerNorm
//! residual stack with a gated feed-forward network, an embedding table and a
//! language-modelling head — every one of which holds real learned weights.
//!
//! Reference: Beck et al., "xLSTM: Extended Long Short-Term Memory" (2024).

use crate::xlstm::config::{XLSTMBlockType, XLSTMConfig};
use crate::xlstm::mlstm::{MLstmBlock, MLstmState};
use crate::xlstm::slstm::{SLstmBlock, SLstmState};
use trustformers_core::device::Device;
use trustformers_core::errors::{invalid_input, Result, TrustformersError};
use trustformers_core::layers::{Embedding, LayerNorm, Linear};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Layer;

/// Pick a head count for a layer that only knows its width.
///
/// Returns the largest head count in `1..=8` that divides `hidden_size`, so the
/// matrix memory of an mLSTM block is always well formed.
fn default_head_count(hidden_size: usize) -> usize {
    (1..=8).rev().find(|heads| hidden_size.is_multiple_of(*heads)).unwrap_or(1)
}

/// Extended LSTM model
#[derive(Debug, Clone)]
pub struct XLSTMModel {
    config: XLSTMConfig,
    embeddings: Embedding,
    layers: Vec<XLSTMLayer>,
    block_norms: Vec<LayerNorm>,
    feed_forward_norms: Vec<LayerNorm>,
    feed_forwards: Vec<FeedForward>,
    final_norm: LayerNorm,
    lm_head: Linear,
    device: Device,
}

impl XLSTMModel {
    /// Create a new xLSTM model on CPU (backward compatibility)
    pub fn new(config: XLSTMConfig) -> Result<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new xLSTM model with specified device
    pub fn new_with_device(config: XLSTMConfig, device: Device) -> Result<Self> {
        if config.hidden_size == 0 || config.vocab_size == 0 || config.num_layers == 0 {
            return Err(invalid_input(
                "xLSTM requires vocab_size, hidden_size and num_layers to be greater than zero",
            ));
        }

        let embeddings =
            Embedding::new_with_device(config.vocab_size, config.hidden_size, None, device)?;

        let eps = config.layer_norm_epsilon as f32;
        let mut layers = Vec::with_capacity(config.num_layers);
        let mut block_norms = Vec::with_capacity(config.num_layers);
        let mut feed_forward_norms = Vec::with_capacity(config.num_layers);
        let mut feed_forwards = Vec::with_capacity(config.num_layers);

        for index in 0..config.num_layers {
            let block_type = config.block_type_for_layer(index);
            let layer = XLSTMLayer::new_with_heads(
                config.hidden_size,
                block_type,
                config.num_heads,
                device,
            )?
            .with_forget_gate_bias(config.initial_forget_gate_bias)?;
            layers.push(layer);

            block_norms.push(LayerNorm::new_with_device(
                vec![config.hidden_size],
                eps,
                device,
            )?);
            feed_forward_norms.push(LayerNorm::new_with_device(
                vec![config.hidden_size],
                eps,
                device,
            )?);
            feed_forwards.push(FeedForward::new_with_device(
                config.hidden_size,
                config.intermediate_size,
                device,
            ));
        }

        let final_norm = LayerNorm::new_with_device(vec![config.hidden_size], eps, device)?;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);

        Ok(Self {
            config,
            embeddings,
            layers,
            block_norms,
            feed_forward_norms,
            feed_forwards,
            final_norm,
            lm_head,
            device,
        })
    }

    /// Get model configuration
    pub fn config(&self) -> &XLSTMConfig {
        &self.config
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Run the recurrent stack and return the final `[seq_len, hidden_size]`
    /// hidden states together with the per-layer hidden states.
    pub fn encode(&self, input_ids: &[u32]) -> Result<(Tensor, Vec<Tensor>)> {
        if input_ids.is_empty() {
            return Err(invalid_input("xLSTM forward requires at least one token"));
        }
        if let Some(&max_id) = input_ids.iter().max() {
            if max_id as usize >= self.config.vocab_size {
                return Err(invalid_input(format!(
                    "token id {} is outside the vocabulary of size {}",
                    max_id, self.config.vocab_size
                )));
            }
        }
        if input_ids.len() > self.config.max_sequence_length {
            return Err(invalid_input(format!(
                "sequence length {} exceeds max_sequence_length {}",
                input_ids.len(),
                self.config.max_sequence_length
            )));
        }

        let mut hidden = self.embeddings.forward(input_ids.to_vec())?;
        let mut per_layer = Vec::with_capacity(self.layers.len());

        for index in 0..self.layers.len() {
            // Pre-norm recurrent block with a residual connection.
            let normed = self.block_norms[index].forward(hidden.clone())?;
            let block_out = self.layers[index].forward_sequence(&normed)?;
            hidden = hidden.add(&block_out)?;

            // Pre-norm feed-forward with a residual connection.
            let normed = self.feed_forward_norms[index].forward(hidden.clone())?;
            let ff_out = self.feed_forwards[index].forward_tensor(&normed)?;
            hidden = hidden.add(&ff_out)?;

            per_layer.push(hidden.clone());
        }

        Ok((self.final_norm.forward(hidden)?, per_layer))
    }

    /// Forward pass producing language-modelling logits.
    ///
    /// The `Vec<u32>` input carries a single sequence, so the returned logits have
    /// shape `[1, seq_len, vocab_size]`.
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<XLSTMOutput> {
        let (hidden, per_layer) = self.encode(&input_ids)?;
        let seq_len = input_ids.len();

        let logits = self.lm_head.forward(hidden)?;
        let logits = logits.reshape(&[1, seq_len, self.config.vocab_size])?;

        Ok(XLSTMOutput {
            logits,
            hidden_states: Some(per_layer),
            // xLSTM has no attention matrices: sLSTM is a recurrence and mLSTM
            // keeps a matrix memory instead of pairwise scores.
            attentions: None,
        })
    }

    /// Count model parameters from the real weight tensors.
    pub fn parameter_count(&self) -> usize {
        let mut total = self.embeddings.parameter_count();
        for layer in &self.layers {
            total += layer.parameter_count();
        }
        for norm in &self.block_norms {
            total += norm.parameter_count();
        }
        for norm in &self.feed_forward_norms {
            total += norm.parameter_count();
        }
        for feed_forward in &self.feed_forwards {
            total += feed_forward.parameter_count();
        }
        total += self.final_norm.parameter_count();
        total += self.lm_head.parameter_count();
        total
    }
}

/// Output structure for xLSTM model
#[derive(Debug, Clone)]
pub struct XLSTMOutput {
    /// Logits tensor [batch_size, sequence_length, vocab_size]
    pub logits: Tensor,
    /// Hidden states after each xLSTM layer
    pub hidden_states: Option<Vec<Tensor>>,
    /// Always `None`: xLSTM produces no attention matrices.
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

    /// Access the underlying backbone.
    pub fn backbone(&self) -> &XLSTMModel {
        &self.xlstm
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
    classifier: Linear,
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
        if num_labels == 0 {
            return Err(invalid_input("num_labels must be greater than zero"));
        }
        let hidden_size = config.hidden_size;
        let xlstm = XLSTMModel::new_with_device(config, device)?;
        let classifier = Linear::new_with_device(hidden_size, num_labels, true, device);
        Ok(Self {
            xlstm,
            classifier,
            num_labels,
            device,
        })
    }

    /// Get the device this model is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Classify a sequence.
    ///
    /// The final hidden states are mean-pooled over the sequence and passed
    /// through a learned classification head, giving `[1, num_labels]` logits.
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Tensor> {
        let (hidden, _) = self.xlstm.encode(&input_ids)?;
        let hidden_size = self.xlstm.config.hidden_size;
        let data = hidden.data()?;
        let seq_len = input_ids.len();

        let mut pooled = vec![0.0f32; hidden_size];
        for t in 0..seq_len {
            for (h, slot) in pooled.iter_mut().enumerate() {
                *slot += data[t * hidden_size + h];
            }
        }
        for slot in pooled.iter_mut() {
            *slot /= seq_len as f32;
        }

        let pooled = Tensor::from_vec(pooled, &[1, hidden_size])?;
        self.classifier.forward(pooled)
    }

    pub fn parameter_count(&self) -> usize {
        self.xlstm.parameter_count() + self.classifier.parameter_count()
    }
}

/// One xLSTM recurrent layer.
///
/// Holds an sLSTM block, an mLSTM block, or (for
/// [`XLSTMBlockType::Mixed`](crate::xlstm::XLSTMBlockType::Mixed)) both applied in
/// sequence. Layer normalisation and the feed-forward network live in
/// [`XLSTMModel`], so this type's parameter count is exactly the recurrent cell.
#[derive(Debug, Clone)]
pub struct XLSTMLayer {
    hidden_size: usize,
    block_type: XLSTMBlockType,
    slstm: Option<SLstmBlock>,
    mlstm: Option<MLstmBlock>,
    device: Device,
}

impl XLSTMLayer {
    /// Create a new xLSTM layer on CPU (backward compatibility).
    ///
    /// The head count for an mLSTM block is chosen as the largest value in `1..=8`
    /// that divides `hidden_size`; use [`XLSTMLayer::new_with_heads`] to pin it.
    pub fn new(hidden_size: usize, block_type: XLSTMBlockType) -> Self {
        Self::new_with_device(hidden_size, block_type, Device::CPU)
    }

    /// Create a new xLSTM layer with specified device
    pub fn new_with_device(hidden_size: usize, block_type: XLSTMBlockType, device: Device) -> Self {
        let heads = default_head_count(hidden_size);
        // reason: `new_with_heads` only fails when `hidden_size % heads != 0`, and
        // `default_head_count` returns a divisor of `hidden_size` by construction.
        #[allow(clippy::expect_used)]
        Self::new_with_heads(hidden_size, block_type, heads, device)
            .expect("default head count always divides hidden_size")
    }

    /// Create a layer with an explicit head count for its mLSTM block.
    pub fn new_with_heads(
        hidden_size: usize,
        block_type: XLSTMBlockType,
        num_heads: usize,
        device: Device,
    ) -> Result<Self> {
        let heads = num_heads.max(1);
        let needs_mlstm = matches!(block_type, XLSTMBlockType::MLstm | XLSTMBlockType::Mixed);
        if needs_mlstm && (hidden_size == 0 || !hidden_size.is_multiple_of(heads)) {
            return Err(invalid_input(format!(
                "mLSTM hidden_size {hidden_size} must be a non-zero multiple of num_heads {heads}"
            )));
        }

        let slstm = matches!(block_type, XLSTMBlockType::SLstm | XLSTMBlockType::Mixed)
            .then(|| SLstmBlock::new_with_device(hidden_size, device));
        let mlstm = needs_mlstm.then(|| MLstmBlock::new_with_device(hidden_size, heads, device));

        Ok(Self {
            hidden_size,
            block_type,
            slstm,
            mlstm,
            device,
        })
    }

    /// Bias both blocks' forget gates towards remembering.
    pub fn with_forget_gate_bias(mut self, bias: f32) -> Result<Self> {
        if let Some(block) = self.slstm.take() {
            self.slstm = Some(block.with_forget_gate_bias(bias)?);
        }
        if let Some(block) = self.mlstm.take() {
            self.mlstm = Some(block.with_forget_gate_bias(bias)?);
        }
        Ok(self)
    }

    /// Get the device this layer is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Width of the layer.
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }

    /// Which recurrent cell(s) this layer holds.
    pub fn block_type(&self) -> &XLSTMBlockType {
        &self.block_type
    }

    /// Run the layer's recurrent cell(s) over a sequence.
    pub fn forward_sequence(&self, input: &Tensor) -> Result<Tensor> {
        let mut hidden = match &self.slstm {
            Some(block) => block.forward_sequence(input)?,
            None => input.clone(),
        };
        if let Some(block) = &self.mlstm {
            hidden = block.forward_sequence(&hidden)?;
        }
        if self.slstm.is_none() && self.mlstm.is_none() {
            return Err(TrustformersError::tensor_op_error(
                "xLSTM layer holds no recurrent block",
                "xlstm_layer_forward",
            ));
        }
        Ok(hidden)
    }

    /// Number of learned parameters in this layer's recurrent cell(s).
    pub fn parameter_count(&self) -> usize {
        self.slstm.as_ref().map_or(0, SLstmBlock::parameter_count)
            + self.mlstm.as_ref().map_or(0, MLstmBlock::parameter_count)
    }
}

impl Layer for XLSTMLayer {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_sequence(&input)
    }
}

/// Recurrent state for a whole xLSTM stack.
#[derive(Debug, Clone)]
pub struct XLSTMState {
    /// Number of sequences the state describes.
    pub batch_size: usize,
    /// Width of every layer's state.
    pub hidden_size: usize,
    /// Per-layer sLSTM states, in stack order.
    pub slstm_states: Vec<SLstmState>,
    /// Per-layer mLSTM states, in stack order.
    pub mlstm_states: Vec<MLstmState>,
}

impl XLSTMState {
    /// Create an empty state descriptor.
    pub fn new(batch_size: usize, hidden_size: usize) -> Self {
        Self {
            batch_size,
            hidden_size,
            slstm_states: Vec::new(),
            mlstm_states: Vec::new(),
        }
    }

    /// Create a zero-initialised state for a stack of `layers` xLSTM layers.
    pub fn for_layers(
        batch_size: usize,
        hidden_size: usize,
        layers: &[XLSTMLayer],
        num_heads: usize,
    ) -> Self {
        let mut slstm_states = Vec::new();
        let mut mlstm_states = Vec::new();
        for layer in layers {
            match layer.block_type() {
                XLSTMBlockType::SLstm => slstm_states.push(SLstmState::new(hidden_size)),
                XLSTMBlockType::MLstm => mlstm_states.push(MLstmState::new(hidden_size, num_heads)),
                XLSTMBlockType::Mixed => {
                    slstm_states.push(SLstmState::new(hidden_size));
                    mlstm_states.push(MLstmState::new(hidden_size, num_heads));
                },
            }
        }
        Self {
            batch_size,
            hidden_size,
            slstm_states,
            mlstm_states,
        }
    }

    /// Reset every layer state back to zero.
    pub fn reset(&mut self) {
        self.slstm_states.iter_mut().for_each(SLstmState::reset);
        self.mlstm_states.iter_mut().for_each(MLstmState::reset);
    }
}

/// Gated feed-forward network used between xLSTM layers.
///
/// `down(gelu(up(x)))` with learned weights and biases.
#[derive(Debug, Clone)]
pub struct FeedForward {
    /// Model width.
    pub hidden_size: usize,
    /// Width of the intermediate activation.
    pub intermediate_size: usize,
    up: Linear,
    down: Linear,
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
            up: Linear::new_with_device(hidden_size, intermediate_size, true, device),
            down: Linear::new_with_device(intermediate_size, hidden_size, true, device),
            device,
        }
    }

    /// Get the device this network is on
    pub fn device(&self) -> Device {
        self.device
    }

    /// Apply the network to an activation tensor.
    pub fn forward_tensor(&self, input: &Tensor) -> Result<Tensor> {
        let intermediate = self.up.forward(input.clone())?;
        let activated = intermediate.gelu()?;
        self.down.forward(activated)
    }

    pub fn parameter_count(&self) -> usize {
        self.up.parameter_count() + self.down.parameter_count()
    }
}

impl Layer for FeedForward {
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.forward_tensor(&input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A model small enough to build many times in a test run while still
    /// exercising every code path.
    fn tiny_config() -> XLSTMConfig {
        XLSTMConfig {
            vocab_size: 24,
            hidden_size: 8,
            intermediate_size: 16,
            num_layers: 2,
            num_heads: 2,
            max_sequence_length: 32,
            ..XLSTMConfig::default()
        }
    }

    #[test]
    fn test_xlstm_model_device_support() -> Result<()> {
        let config = tiny_config();

        let model_cpu = XLSTMModel::new(config.clone())?;
        assert_eq!(model_cpu.device(), Device::CPU);

        let model_cpu_explicit = XLSTMModel::new_with_device(config.clone(), Device::CPU)?;
        assert_eq!(model_cpu_explicit.device(), Device::CPU);

        let model_metal = XLSTMModel::new_with_device(config.clone(), Device::Metal(0))?;
        assert_eq!(model_metal.device(), Device::Metal(0));

        let model_cuda = XLSTMModel::new_with_device(config, Device::CUDA(0))?;
        assert_eq!(model_cuda.device(), Device::CUDA(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_for_causal_lm_device_support() -> Result<()> {
        let config = tiny_config();

        let model_cpu = XLSTMForCausalLM::new(config.clone())?;
        assert_eq!(model_cpu.device(), Device::CPU);

        let model_metal = XLSTMForCausalLM::new_with_device(config, Device::Metal(0))?;
        assert_eq!(model_metal.device(), Device::Metal(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_for_sequence_classification_device_support() -> Result<()> {
        let config = tiny_config();
        let num_labels = 2;

        let model_cpu = XLSTMForSequenceClassification::new(config.clone(), num_labels)?;
        assert_eq!(model_cpu.device(), Device::CPU);

        let model_cuda =
            XLSTMForSequenceClassification::new_with_device(config, num_labels, Device::CUDA(0))?;
        assert_eq!(model_cuda.device(), Device::CUDA(0));

        Ok(())
    }

    #[test]
    fn test_xlstm_layer_device_support() {
        let hidden_size = 16;
        let block_type = XLSTMBlockType::Mixed;

        let layer_cpu = XLSTMLayer::new(hidden_size, block_type.clone());
        assert_eq!(layer_cpu.device(), Device::CPU);

        let layer_metal = XLSTMLayer::new_with_device(hidden_size, block_type, Device::Metal(0));
        assert_eq!(layer_metal.device(), Device::Metal(0));
    }

    #[test]
    fn test_feedforward_device_support() {
        let ff_cpu = FeedForward::new(16, 32);
        assert_eq!(ff_cpu.device(), Device::CPU);

        let ff_cuda = FeedForward::new_with_device(16, 32, Device::CUDA(0));
        assert_eq!(ff_cuda.device(), Device::CUDA(0));
    }

    #[test]
    fn test_device_propagation() -> Result<()> {
        let config = tiny_config();

        let model = XLSTMModel::new_with_device(config.clone(), Device::Metal(0))?;
        assert_eq!(model.device(), Device::Metal(0));

        let causal_lm = XLSTMForCausalLM::new_with_device(config, Device::CUDA(1))?;
        assert_eq!(causal_lm.device(), Device::CUDA(1));
        assert_eq!(causal_lm.backbone().device(), Device::CUDA(1));

        Ok(())
    }

    #[test]
    fn test_backward_compatibility() -> Result<()> {
        let config = tiny_config();

        let model = XLSTMModel::new(config.clone())?;
        assert_eq!(model.device(), Device::CPU);

        let causal_lm = XLSTMForCausalLM::new(config.clone())?;
        assert_eq!(causal_lm.device(), Device::CPU);

        let seq_class = XLSTMForSequenceClassification::new(config, 2)?;
        assert_eq!(seq_class.device(), Device::CPU);

        Ok(())
    }

    // ---- Real-weight / real-forward regression tests ----

    /// The model must produce non-zero, input-dependent logits.
    ///
    /// The previous implementation held no parameters at all and returned
    /// `vec![0.0; batch · seq · vocab]`.
    #[test]
    fn test_forward_produces_nonzero_input_dependent_logits() -> Result<()> {
        let config = tiny_config();
        let model = XLSTMModel::new(config.clone())?;

        let out_a = model.forward(vec![1, 2, 3, 4])?;
        let out_b = model.forward(vec![4, 3, 2, 1])?;

        assert_eq!(out_a.logits.shape(), vec![1, 4, config.vocab_size]);
        let a = out_a.logits.data()?;
        let b = out_b.logits.data()?;

        assert!(a.iter().all(|v| v.is_finite()));
        assert!(
            a.iter().any(|v| v.abs() > 1e-6),
            "xLSTM returned an all-zero logit tensor"
        );
        assert!(
            a.iter().zip(b.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "reversing the token order left the logits unchanged"
        );

        Ok(())
    }

    /// The recurrence must actually carry state across the sequence: the logits at
    /// the last position must react to a change at the first position.
    #[test]
    fn test_logits_depend_on_earlier_tokens() -> Result<()> {
        let config = tiny_config();
        let model = XLSTMModel::new(config.clone())?;

        let a = model.forward(vec![1, 5, 5, 5])?.logits.data()?;
        let b = model.forward(vec![7, 5, 5, 5])?.logits.data()?;

        let last = 3 * config.vocab_size;
        assert!(
            a[last..].iter().zip(b[last..].iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "the final position ignores the first token — no recurrent state"
        );

        Ok(())
    }

    /// Per-layer hidden states are reported, one per layer, with real values.
    #[test]
    fn test_hidden_states_are_reported_per_layer() -> Result<()> {
        let config = tiny_config();
        let model = XLSTMModel::new(config.clone())?;
        let out = model.forward(vec![0, 1, 2])?;

        let hidden_states = out.hidden_states.expect("hidden states must be reported");
        assert_eq!(hidden_states.len(), config.num_layers);
        for (index, state) in hidden_states.iter().enumerate() {
            assert_eq!(state.shape(), vec![3, config.hidden_size]);
            assert!(
                state.data()?.iter().any(|v| v.abs() > 1e-6),
                "layer {index} hidden state is all zeros"
            );
        }
        assert!(out.attentions.is_none());

        Ok(())
    }

    /// The parameter count must be the sum of the real tensors, not an estimate.
    #[test]
    fn test_parameter_count_matches_the_real_tensors() -> Result<()> {
        let config = XLSTMConfig {
            vocab_size: 10,
            hidden_size: 4,
            intermediate_size: 8,
            num_layers: 1,
            num_heads: 2,
            max_sequence_length: 16,
            block_config: crate::xlstm::config::XLSTMBlockConfig {
                block_type: XLSTMBlockType::SLstm,
                slstm_blocks: 1,
                mlstm_blocks: 0,
                block_pattern: vec![XLSTMBlockType::SLstm],
            },
            ..XLSTMConfig::default()
        };
        let model = XLSTMModel::new(config.clone())?;

        let hidden = config.hidden_size;
        let embedding = config.vocab_size * hidden;
        let slstm = 4 * (hidden * hidden + hidden) + 4 * (hidden * hidden);
        let feed_forward = (hidden * config.intermediate_size + config.intermediate_size)
            + (config.intermediate_size * hidden + hidden);
        let norms = 3 * (2 * hidden); // two per layer plus the final norm
        let lm_head = hidden * config.vocab_size;

        assert_eq!(
            model.parameter_count(),
            embedding + slstm + feed_forward + norms + lm_head
        );

        Ok(())
    }

    /// Classification must run the real backbone and a real head.
    #[test]
    fn test_classification_head_is_real() -> Result<()> {
        let config = tiny_config();
        let num_labels = 3;
        let model = XLSTMForSequenceClassification::new(config, num_labels)?;

        let a = model.forward(vec![1, 2, 3])?;
        let b = model.forward(vec![5, 6, 7])?;

        assert_eq!(a.shape(), vec![1, num_labels]);
        let a_data = a.data()?;
        let b_data = b.data()?;
        assert!(a_data.iter().all(|v| v.is_finite()));
        assert!(
            a_data.iter().any(|v| v.abs() > 1e-6),
            "classification head returned zeros"
        );
        assert!(
            a_data.iter().zip(b_data.iter()).any(|(x, y)| (x - y).abs() > 1e-6),
            "classification logits ignore the input"
        );

        Ok(())
    }

    /// Out-of-vocabulary tokens and empty inputs are reported, not silently padded.
    #[test]
    fn test_invalid_inputs_are_rejected() -> Result<()> {
        let config = tiny_config();
        let model = XLSTMModel::new(config.clone())?;

        assert!(model.forward(vec![]).is_err());
        assert!(model.forward(vec![config.vocab_size as u32]).is_err());
        let too_long: Vec<u32> = vec![0; config.max_sequence_length + 1];
        assert!(model.forward(too_long).is_err());

        Ok(())
    }

    /// Every block type produces a working layer with real parameters.
    #[test]
    fn test_layer_block_types_all_run() -> Result<()> {
        let hidden = 8usize;
        let input = Tensor::from_vec(
            (0..3 * hidden).map(|i| (i as f32 * 0.19).sin()).collect(),
            &[3, hidden],
        )?;

        for block_type in [
            XLSTMBlockType::SLstm,
            XLSTMBlockType::MLstm,
            XLSTMBlockType::Mixed,
        ] {
            let layer = XLSTMLayer::new_with_heads(hidden, block_type.clone(), 2, Device::CPU)?;
            assert!(
                layer.parameter_count() > 0,
                "{block_type:?} has no parameters"
            );
            let out = layer.forward_sequence(&input)?;
            assert_eq!(out.shape(), vec![3, hidden]);
            let data = out.data()?;
            assert!(data.iter().all(|v| v.is_finite()));
            assert!(
                data.iter().any(|v| v.abs() > 1e-6),
                "{block_type:?} returned zeros"
            );
        }

        // Mixed holds both cells, so it is the sum of the two.
        let slstm = XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::SLstm, 2, Device::CPU)?;
        let mlstm = XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::MLstm, 2, Device::CPU)?;
        let mixed = XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::Mixed, 2, Device::CPU)?;
        assert_eq!(
            mixed.parameter_count(),
            slstm.parameter_count() + mlstm.parameter_count()
        );

        Ok(())
    }

    /// A head count that does not divide the width is rejected.
    #[test]
    fn test_layer_rejects_indivisible_head_count() {
        assert!(XLSTMLayer::new_with_heads(9, XLSTMBlockType::MLstm, 2, Device::CPU).is_err());
        assert!(XLSTMLayer::new_with_heads(9, XLSTMBlockType::SLstm, 2, Device::CPU).is_ok());
    }

    /// `default_head_count` always returns a divisor.
    #[test]
    fn test_default_head_count_divides_the_width() {
        for hidden in 1..=64usize {
            let heads = default_head_count(hidden);
            assert!(
                heads >= 1 && hidden % heads == 0,
                "hidden={hidden} heads={heads}"
            );
        }
    }

    /// The feed-forward network is a real two-layer MLP.
    #[test]
    fn test_feed_forward_is_real() -> Result<()> {
        let ff = FeedForward::new(4, 8);
        assert_eq!(ff.parameter_count(), (4 * 8 + 8) + (8 * 4 + 4));

        let input = Tensor::from_vec(vec![0.5, -0.25, 1.0, -1.5], &[1, 4])?;
        let out = ff.forward_tensor(&input)?;
        assert_eq!(out.shape(), vec![1, 4]);
        let data = out.data()?;
        assert!(data.iter().all(|v| v.is_finite()));
        assert!(data.iter().any(|v| v.abs() > 1e-6));

        Ok(())
    }

    /// The stack state container tracks one state per recurrent cell.
    #[test]
    fn test_state_for_layers_tracks_every_cell() -> Result<()> {
        let hidden = 8usize;
        let layers = vec![
            XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::SLstm, 2, Device::CPU)?,
            XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::MLstm, 2, Device::CPU)?,
            XLSTMLayer::new_with_heads(hidden, XLSTMBlockType::Mixed, 2, Device::CPU)?,
        ];
        let mut state = XLSTMState::for_layers(1, hidden, &layers, 2);
        assert_eq!(state.slstm_states.len(), 2);
        assert_eq!(state.mlstm_states.len(), 2);
        assert_eq!(state.batch_size, 1);
        assert_eq!(state.hidden_size, hidden);

        state.slstm_states[0].cell[0] = 1.0;
        state.reset();
        assert!(state.slstm_states[0].cell.iter().all(|v| *v == 0.0));

        Ok(())
    }
}
