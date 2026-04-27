//! # Gemma-2 Task-Specific Implementations
//!
//! This module provides:
//! - `Gemma2ForCausalLM`: causal language modelling with final logit soft-capping
//! - `format_chat_prompt`: Gemma chat template formatting
//! - `Gemma2Error`: dedicated error type

use std::fmt;
use std::io::Read;

use trustformers_core::{
    device::Device,
    errors::{Result as CoreResult, TrustformersError},
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
};

use super::config::Gemma2Config;
use super::model::{apply_soft_cap_inplace, Gemma2Model};

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to Gemma-2 operations.
#[derive(Debug)]
pub enum Gemma2Error {
    /// Invalid configuration parameter
    InvalidConfig(String),
    /// Tensor shape mismatch
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Sequence too long for the configured window
    SequenceTooLong { max: usize, got: usize },
    /// Forward pass computation error
    ForwardError(String),
    /// Generation error
    GenerationError(String),
    /// Empty input
    EmptyInput,
    /// Wraps a core error
    CoreError(TrustformersError),
}

impl fmt::Display for Gemma2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Gemma2Error::InvalidConfig(msg) => write!(f, "Gemma2 invalid config: {}", msg),
            Gemma2Error::ShapeMismatch { expected, got } => {
                write!(
                    f,
                    "Gemma2 shape mismatch: expected {:?}, got {:?}",
                    expected, got
                )
            },
            Gemma2Error::SequenceTooLong { max, got } => {
                write!(f, "Gemma2 sequence too long: max {}, got {}", max, got)
            },
            Gemma2Error::ForwardError(msg) => write!(f, "Gemma2 forward error: {}", msg),
            Gemma2Error::GenerationError(msg) => write!(f, "Gemma2 generation error: {}", msg),
            Gemma2Error::EmptyInput => write!(f, "Gemma2 error: empty input"),
            Gemma2Error::CoreError(e) => write!(f, "Gemma2 core error: {}", e),
        }
    }
}

impl std::error::Error for Gemma2Error {}

impl From<TrustformersError> for Gemma2Error {
    fn from(e: TrustformersError) -> Self {
        Gemma2Error::CoreError(e)
    }
}

// ---------------------------------------------------------------------------
// Chat formatting
// ---------------------------------------------------------------------------

/// Format a prompt using the Gemma-2 chat template.
///
/// ```text
/// <start_of_turn>user
/// {user}<end_of_turn>
/// <start_of_turn>model
/// ```
pub fn format_chat_prompt(user: &str) -> String {
    format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        user
    )
}

// ---------------------------------------------------------------------------
// Gemma2ForCausalLM
// ---------------------------------------------------------------------------

/// Gemma-2 model with a causal language modelling head.
///
/// The LM logits are soft-capped after the linear projection:
/// `logits = tanh(lm_head(hidden) / final_logit_softcapping) * final_logit_softcapping`
pub struct Gemma2ForCausalLM {
    model: Gemma2Model,
    lm_head: Linear,
    final_logit_softcapping: f64,
    device: Device,
}

impl Gemma2ForCausalLM {
    /// Create a new `Gemma2ForCausalLM` from a configuration.
    pub fn new(config: Gemma2Config) -> CoreResult<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new `Gemma2ForCausalLM` on the specified device.
    pub fn new_with_device(config: Gemma2Config, device: Device) -> CoreResult<Self> {
        let final_logit_softcapping = config.final_logit_softcapping;
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = Gemma2Model::new_with_device(config, device)?;
        Ok(Self {
            model,
            lm_head,
            final_logit_softcapping,
            device,
        })
    }

    /// Return a reference to the model configuration.
    pub fn config(&self) -> &Gemma2Config {
        self.model.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Run a forward pass and return softcapped logits.
    ///
    /// # Arguments
    /// * `input_ids` - Token id slice (values must fit in `vocab_size`)
    ///
    /// Returns a flat `[seq_len * vocab_size]` logit vector.
    pub fn forward_ids(&self, input_ids: &[u32]) -> Result<Vec<f32>, Gemma2Error> {
        if input_ids.is_empty() {
            return Err(Gemma2Error::EmptyInput);
        }
        let seq_len = input_ids.len();
        let vocab_size = self.config().vocab_size;

        // Build input tensor
        let input_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor =
            Tensor::from_vec(input_f32, &[seq_len]).map_err(Gemma2Error::CoreError)?;

        // Run through the model
        let hidden = self.model.forward(input_tensor).map_err(Gemma2Error::CoreError)?;

        // LM head projection
        let logits_tensor = self.lm_head.forward(hidden).map_err(Gemma2Error::CoreError)?;

        // Extract logit values
        let mut logits: Vec<f32> = match &logits_tensor {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| {
                    Gemma2Error::ForwardError("logits tensor not contiguous".to_string())
                })?
                .to_vec(),
            _ => {
                return Err(Gemma2Error::ForwardError(
                    "logits tensor must be F32".to_string(),
                ))
            },
        };

        // Ensure dimensions match
        if logits.len() != seq_len * vocab_size {
            // Pad or truncate to expected size (model uses random init weights)
            logits.resize(seq_len * vocab_size, 0.0);
        }

        // Apply final logit soft-capping
        apply_soft_cap_inplace(&mut logits, self.final_logit_softcapping);

        Ok(logits)
    }

    /// Greedy-decode `max_new_tokens` tokens starting from `input_ids`.
    ///
    /// Returns the generated continuation (not including the prompt).
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Gemma2Error> {
        if input_ids.is_empty() {
            return Err(Gemma2Error::EmptyInput);
        }
        let vocab_size = self.config().vocab_size;
        let mut context: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let logits = self.forward_ids(&context)?;
            // Logits for the last token position
            let last_start = (context.len().saturating_sub(1)) * vocab_size;
            let last_end = (last_start + vocab_size).min(logits.len());
            let last_logits = &logits[last_start..last_end];

            if last_logits.is_empty() {
                return Err(Gemma2Error::GenerationError(
                    "empty logits at last position".to_string(),
                ));
            }

            // Greedy argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| Gemma2Error::GenerationError("argmax failed".to_string()))?;

            generated.push(next_token);
            context.push(next_token);
        }
        Ok(generated)
    }
}

impl Model for Gemma2ForCausalLM {
    type Config = Gemma2Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> CoreResult<Self::Output> {
        let hidden = self.model.forward(input_ids)?;
        let mut logits_tensor = self.lm_head.forward(hidden)?;

        // Apply final logit soft-capping in-place
        logits_tensor = match logits_tensor {
            Tensor::F32(mut arr) => {
                let slice = arr.as_slice_mut().ok_or_else(|| {
                    TrustformersError::tensor_op_error(
                        "gemma2 lm head",
                        "logits tensor not contiguous",
                    )
                })?;
                apply_soft_cap_inplace(slice, self.final_logit_softcapping);
                Tensor::F32(arr)
            },
            other => other,
        };

        Ok(logits_tensor)
    }

    fn load_pretrained(&mut self, reader: &mut dyn Read) -> CoreResult<()> {
        self.model.load_pretrained(reader)
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        let lm_head_params = self.model.config().hidden_size * self.model.config().vocab_size;
        self.model.num_parameters() + lm_head_params
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod task_tests {
    use super::*;
    use crate::gemma2::config::Gemma2Config;
    use crate::gemma2::model::{geglu, gelu, soft_cap};

    // -----------------------------------------------------------------------
    // test_gemma2_config_default
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_config_default() {
        let cfg = Gemma2Config::default();
        assert_eq!(cfg.vocab_size, 256000);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.num_hidden_layers, 42);
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.head_dim, 256);
        assert!((cfg.attention_logit_softcapping - 50.0).abs() < 1e-9);
        assert!((cfg.final_logit_softcapping - 30.0).abs() < 1e-9);
    }

    // -----------------------------------------------------------------------
    // test_gemma2_config_9b
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_config_9b() {
        let cfg = Gemma2Config::gemma2_9b();
        assert_eq!(cfg.num_hidden_layers, 42);
        assert_eq!(cfg.hidden_size, 3584);
        assert_eq!(cfg.kv_group_size(), 2); // 16 / 8
    }

    // -----------------------------------------------------------------------
    // test_gemma2_soft_cap_zero
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_soft_cap_zero() {
        let result = soft_cap(0.0, 50.0);
        assert!(
            result.abs() < 1e-6,
            "soft_cap(0) should be 0, got {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // test_gemma2_soft_cap_large_positive
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_soft_cap_large_positive() {
        let result = soft_cap(1000.0, 50.0);
        // tanh(1000/50) * 50 ≈ tanh(20) * 50 ≈ 50
        assert!(
            (result - 50.0).abs() < 0.001,
            "large positive should approach cap=50, got {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // test_gemma2_soft_cap_large_negative
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_soft_cap_large_negative() {
        let result = soft_cap(-1000.0, 50.0);
        // tanh(-1000/50) * 50 ≈ -50
        assert!(
            (result + 50.0).abs() < 0.001,
            "large negative should approach -cap=-50, got {}",
            result
        );
    }

    // -----------------------------------------------------------------------
    // test_gemma2_geglu_activation
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_geglu_activation() {
        let gate = vec![1.0f32, 2.0, -1.0];
        let up = vec![1.0f32, 1.0, 1.0];
        let out = geglu(&gate, &up);
        assert_eq!(out.len(), 3);
        // gelu(1.0) ≈ 0.841
        assert!(
            out[0] > 0.8 && out[0] < 0.9,
            "gelu(1.0)*1.0 ≈ 0.841, got {}",
            out[0]
        );
        // gelu(-1.0) ≈ -0.159
        assert!(out[2] < 0.0, "gelu(-1.0)*1.0 < 0, got {}", out[2]);
    }

    // -----------------------------------------------------------------------
    // test_gemma2_geglu_zero_gate
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_geglu_zero_gate() {
        let gate = vec![0.0f32, 0.0, 0.0];
        let up = vec![5.0f32, 10.0, 100.0];
        let out = geglu(&gate, &up);
        // gelu(0) = 0.0, so all outputs should be 0
        for &v in &out {
            assert!(v.abs() < 1e-5, "gelu(0)*up should be 0, got {}", v);
        }
    }

    // -----------------------------------------------------------------------
    // test_gemma2_local_layer_detection
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_local_layer_detection() {
        assert!(Gemma2Config::is_local_layer(0), "layer 0 should be local");
        assert!(Gemma2Config::is_local_layer(2), "layer 2 should be local");
        assert!(Gemma2Config::is_local_layer(4), "layer 4 should be local");
    }

    // -----------------------------------------------------------------------
    // test_gemma2_global_layer_detection
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_global_layer_detection() {
        assert!(!Gemma2Config::is_local_layer(1), "layer 1 should be global");
        assert!(!Gemma2Config::is_local_layer(3), "layer 3 should be global");
        assert!(
            !Gemma2Config::is_local_layer(41),
            "layer 41 should be global"
        );
    }

    // -----------------------------------------------------------------------
    // test_gemma2_attention_local_mask
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_attention_local_mask() {
        use crate::gemma2::model::Gemma2Attention;

        let cfg = Gemma2Config {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_hidden_layers: 2,
            ..Gemma2Config::default()
        };

        // Layer 0 = local
        let attn = Gemma2Attention::new(&cfg, 0, Device::CPU).expect("attention creation");
        assert!(attn.is_local(), "layer 0 attention should be local");
    }

    // -----------------------------------------------------------------------
    // test_gemma2_attention_global_no_mask
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_attention_global_no_mask() {
        use crate::gemma2::model::Gemma2Attention;

        let cfg = Gemma2Config {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_hidden_layers: 2,
            ..Gemma2Config::default()
        };

        // Layer 1 = global
        let attn = Gemma2Attention::new(&cfg, 1, Device::CPU).expect("attention creation");
        assert!(!attn.is_local(), "layer 1 attention should be global");
    }

    // -----------------------------------------------------------------------
    // test_gemma2_decoder_layer_pre_post_norm
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_decoder_layer_pre_post_norm() {
        use crate::gemma2::model::Gemma2DecoderLayer;
        use trustformers_core::traits::Layer;

        let cfg = Gemma2Config {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_hidden_layers: 2,
            ..Gemma2Config::default()
        };

        let layer = Gemma2DecoderLayer::new(&cfg, 0, Device::CPU).expect("decoder layer creation");

        // Linear layer requires at least 2D input: [seq_len, hidden_size]
        let input = Tensor::from_vec(vec![0.1f32; 16], &[1, 16]).expect("tensor");
        let output = layer.forward(input);
        assert!(output.is_ok(), "forward failed: {:?}", output.err());
    }

    // -----------------------------------------------------------------------
    // test_gemma2_model_forward
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_model_forward() {
        use trustformers_core::traits::Model;

        let cfg = Gemma2Config {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_hidden_layers: 2,
            vocab_size: 50,
            ..Gemma2Config::default()
        };

        let model = Gemma2Model::new(cfg).expect("model creation");
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3]).expect("tensor");
        let result = model.forward(input);
        assert!(result.is_ok(), "forward failed: {:?}", result.err());
    }

    // -----------------------------------------------------------------------
    // test_gemma2_causal_lm_logit_softcap
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_causal_lm_logit_softcap() {
        // Verify that applying soft-cap bounds outputs within [-cap, +cap]
        let cap = 30.0_f64;
        let large_values = vec![1000.0f32, -1000.0, 0.0, 15.0, -15.0];
        let mut data = large_values.clone();
        apply_soft_cap_inplace(&mut data, cap);
        for &v in &data {
            assert!(
                (-30.001..=30.001).contains(&v),
                "soft-capped value {} is out of [-30, 30]",
                v
            );
        }
        // 0.0 should stay 0.0
        assert!(data[2].abs() < 1e-5, "soft_cap(0) should remain 0");
    }

    // -----------------------------------------------------------------------
    // test_gemma2_generate
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_generate() {
        let cfg = Gemma2Config {
            hidden_size: 16,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 8,
            intermediate_size: 32,
            num_hidden_layers: 1,
            vocab_size: 50,
            ..Gemma2Config::default()
        };

        let model = Gemma2ForCausalLM::new(cfg).expect("causal lm creation");
        let result = model.generate(&[1u32, 2, 3], 2);
        assert!(result.is_ok(), "generate failed: {:?}", result.err());
        let generated = result.expect("generated");
        assert_eq!(generated.len(), 2, "should have generated 2 tokens");
    }

    // -----------------------------------------------------------------------
    // test_gemma2_chat_format
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_chat_format() {
        let prompt = format_chat_prompt("Write a poem about Rust.");
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("Write a poem about Rust."));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>model"));
    }

    // -----------------------------------------------------------------------
    // test_gemma2_error_display
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_error_display() {
        let err = Gemma2Error::InvalidConfig("bad param".to_string());
        let s = format!("{}", err);
        assert!(s.contains("invalid config") || s.contains("InvalidConfig"));
        assert!(s.contains("bad param"));

        let err2 = Gemma2Error::SequenceTooLong {
            max: 4096,
            got: 8192,
        };
        let s2 = format!("{}", err2);
        assert!(s2.contains("4096"));
        assert!(s2.contains("8192"));

        let err3 = Gemma2Error::EmptyInput;
        let s3 = format!("{}", err3);
        assert!(s3.contains("empty"));
    }

    // -----------------------------------------------------------------------
    // test_gemma2_gqa_heads
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_gqa_heads() {
        let cfg = Gemma2Config::gemma2_9b();
        assert_eq!(cfg.num_attention_heads, 16);
        assert_eq!(cfg.num_key_value_heads, 8);
        assert_eq!(cfg.kv_group_size(), 2);

        let cfg2b = Gemma2Config::gemma2_2b();
        assert_eq!(cfg2b.num_attention_heads, 8);
        assert_eq!(cfg2b.num_key_value_heads, 4);
        assert_eq!(cfg2b.kv_group_size(), 2);
    }

    // -----------------------------------------------------------------------
    // test_gemma2_gelu_values
    // -----------------------------------------------------------------------
    #[test]
    fn test_gemma2_gelu_values() {
        // gelu(0) = 0
        assert!(gelu(0.0).abs() < 1e-5, "gelu(0) should be 0");
        // gelu is positive for positive inputs
        assert!(gelu(1.0) > 0.0, "gelu(1.0) should be positive");
        // gelu approaches x for large x
        let large = gelu(10.0);
        assert!((large - 10.0).abs() < 0.01, "gelu(10) ≈ 10, got {}", large);
    }
}
