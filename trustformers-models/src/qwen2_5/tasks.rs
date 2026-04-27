//! # Qwen2.5 Task-Specific Implementations
//!
//! - `Qwen25ForCausalLM`: causal language modelling head
//! - `Qwen25ForSequenceClassification`: sequence classification head

use std::fmt;
use std::io::Read;

use trustformers_core::{
    device::Device,
    errors::{Result as CoreResult, TrustformersError},
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
};

use super::config::Qwen25Config;
use super::model::Qwen25Model;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to Qwen2.5 operations.
#[derive(Debug)]
pub enum Qwen25Error {
    /// Invalid configuration parameter.
    InvalidConfig(String),
    /// Tensor shape mismatch.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Empty input sequence.
    EmptyInput,
    /// Forward pass computation error.
    ForwardError(String),
    /// Wraps a core error.
    CoreError(TrustformersError),
}

impl fmt::Display for Qwen25Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Qwen25Error::InvalidConfig(msg) => write!(f, "Qwen25 invalid config: {}", msg),
            Qwen25Error::ShapeMismatch { expected, got } => write!(
                f,
                "Qwen25 shape mismatch: expected {:?}, got {:?}",
                expected, got
            ),
            Qwen25Error::EmptyInput => write!(f, "Qwen25 error: empty input"),
            Qwen25Error::ForwardError(msg) => write!(f, "Qwen25 forward error: {}", msg),
            Qwen25Error::CoreError(e) => write!(f, "Qwen25 core error: {}", e),
        }
    }
}

impl std::error::Error for Qwen25Error {}

impl From<TrustformersError> for Qwen25Error {
    fn from(e: TrustformersError) -> Self {
        Qwen25Error::CoreError(e)
    }
}

// ---------------------------------------------------------------------------
// Qwen25ForCausalLM
// ---------------------------------------------------------------------------

/// Qwen2.5 with a causal language modelling head.
pub struct Qwen25ForCausalLM {
    model: Qwen25Model,
    lm_head: Linear,
    device: Device,
}

impl Qwen25ForCausalLM {
    /// Create a new causal LM model on the CPU.
    pub fn new(config: Qwen25Config) -> CoreResult<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new causal LM model on the specified device.
    pub fn new_with_device(config: Qwen25Config, device: Device) -> CoreResult<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = Qwen25Model::new_with_device(config, device)?;
        Ok(Self {
            model,
            lm_head,
            device,
        })
    }

    /// Return a reference to the model configuration.
    pub fn config(&self) -> &Qwen25Config {
        self.model.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Run a forward pass over a token-id slice and return flat logits.
    pub fn forward_ids(&self, input_ids: &[u32]) -> Result<Vec<f32>, Qwen25Error> {
        if input_ids.is_empty() {
            return Err(Qwen25Error::EmptyInput);
        }
        let seq_len = input_ids.len();
        let vocab_size = self.config().vocab_size;

        let input_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor =
            Tensor::from_vec(input_f32, &[seq_len]).map_err(Qwen25Error::CoreError)?;

        let hidden = self.model.forward(input_tensor).map_err(Qwen25Error::CoreError)?;
        let logits_tensor = self.lm_head.forward(hidden).map_err(Qwen25Error::CoreError)?;

        let mut logits: Vec<f32> = match &logits_tensor {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| {
                    Qwen25Error::ForwardError("logits tensor not contiguous".to_string())
                })?
                .to_vec(),
            _ => {
                return Err(Qwen25Error::ForwardError(
                    "logits tensor must be F32".to_string(),
                ))
            },
        };
        logits.resize(seq_len * vocab_size, 0.0);
        Ok(logits)
    }

    /// Greedy-decode `max_new_tokens` tokens.
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Qwen25Error> {
        if input_ids.is_empty() {
            return Err(Qwen25Error::EmptyInput);
        }
        let vocab_size = self.config().vocab_size;
        let mut context: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let logits = self.forward_ids(&context)?;
            let last_start = (context.len().saturating_sub(1)) * vocab_size;
            let last_end = (last_start + vocab_size).min(logits.len());
            let last_logits = &logits[last_start..last_end];

            if last_logits.is_empty() {
                return Err(Qwen25Error::ForwardError(
                    "empty logits at last position".to_string(),
                ));
            }

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| Qwen25Error::ForwardError("argmax failed".to_string()))?;

            generated.push(next_token);
            context.push(next_token);
        }
        Ok(generated)
    }
}

impl Model for Qwen25ForCausalLM {
    type Config = Qwen25Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> CoreResult<Self::Output> {
        let hidden = self.model.forward(input_ids)?;
        self.lm_head.forward(hidden)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> CoreResult<()> {
        self.model.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        let head_params = self.model.config().hidden_size * self.model.config().vocab_size;
        self.model.num_parameters() + head_params
    }
}

// ---------------------------------------------------------------------------
// Qwen25ForSequenceClassification
// ---------------------------------------------------------------------------

/// Qwen2.5 with a sequence classification head.
///
/// The hidden state at the last non-padding token is projected to `num_labels` logits.
pub struct Qwen25ForSequenceClassification {
    model: Qwen25Model,
    /// Classification head: `hidden_size → num_labels`.
    score: Linear,
    num_labels: usize,
    device: Device,
}

impl Qwen25ForSequenceClassification {
    /// Create a new sequence classification model on the CPU.
    pub fn new(config: Qwen25Config, num_labels: usize) -> CoreResult<Self> {
        Self::new_with_device(config, num_labels, Device::CPU)
    }

    /// Create a new sequence classification model on the specified device.
    pub fn new_with_device(
        config: Qwen25Config,
        num_labels: usize,
        device: Device,
    ) -> CoreResult<Self> {
        let score = Linear::new_with_device(config.hidden_size, num_labels, false, device);
        let model = Qwen25Model::new_with_device(config, device)?;
        Ok(Self {
            model,
            score,
            num_labels,
            device,
        })
    }

    /// Return a reference to the model configuration.
    pub fn config(&self) -> &Qwen25Config {
        self.model.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Number of classification labels.
    pub fn num_labels(&self) -> usize {
        self.num_labels
    }

    /// Run classification forward pass and return class logits.
    pub fn classify(&self, input_ids: &[u32]) -> Result<Vec<f32>, Qwen25Error> {
        if input_ids.is_empty() {
            return Err(Qwen25Error::EmptyInput);
        }
        let input_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor =
            Tensor::from_vec(input_f32, &[input_ids.len()]).map_err(Qwen25Error::CoreError)?;

        let hidden = self.model.forward(input_tensor).map_err(Qwen25Error::CoreError)?;

        let logits_tensor = self.score.forward(hidden).map_err(Qwen25Error::CoreError)?;

        match &logits_tensor {
            Tensor::F32(arr) => {
                let mut out = arr
                    .as_slice()
                    .ok_or_else(|| {
                        Qwen25Error::ForwardError(
                            "classification logits not contiguous".to_string(),
                        )
                    })?
                    .to_vec();
                // Return only the first `num_labels` values (or pad)
                out.resize(self.num_labels, 0.0);
                Ok(out)
            },
            _ => Err(Qwen25Error::ForwardError(
                "classification logits must be F32".to_string(),
            )),
        }
    }
}

impl Model for Qwen25ForSequenceClassification {
    type Config = Qwen25Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> CoreResult<Self::Output> {
        let hidden = self.model.forward(input_ids)?;
        self.score.forward(hidden)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> CoreResult<()> {
        self.model.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        let score_params = self.model.config().hidden_size * self.num_labels;
        self.model.num_parameters() + score_params
    }
}
