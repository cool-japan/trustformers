//! # DeepSeek-V2 Task-Specific Implementations
//!
//! - `DeepSeekV2ForCausalLM`: causal language modelling head on top of the base model

use std::fmt;
use std::io::Read;

use trustformers_core::{
    device::Device,
    errors::{Result as CoreResult, TrustformersError},
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
};

use super::config::DeepSeekV2Config;
use super::model::DeepSeekV2Model;

// ---------------------------------------------------------------------------
// Error type
// ---------------------------------------------------------------------------

/// Errors specific to DeepSeek-V2 operations.
#[derive(Debug)]
pub enum DeepSeekV2Error {
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

impl fmt::Display for DeepSeekV2Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DeepSeekV2Error::InvalidConfig(msg) => {
                write!(f, "DeepSeekV2 invalid config: {}", msg)
            },
            DeepSeekV2Error::ShapeMismatch { expected, got } => write!(
                f,
                "DeepSeekV2 shape mismatch: expected {:?}, got {:?}",
                expected, got
            ),
            DeepSeekV2Error::EmptyInput => write!(f, "DeepSeekV2 error: empty input"),
            DeepSeekV2Error::ForwardError(msg) => {
                write!(f, "DeepSeekV2 forward error: {}", msg)
            },
            DeepSeekV2Error::CoreError(e) => write!(f, "DeepSeekV2 core error: {}", e),
        }
    }
}

impl std::error::Error for DeepSeekV2Error {}

impl From<TrustformersError> for DeepSeekV2Error {
    fn from(e: TrustformersError) -> Self {
        DeepSeekV2Error::CoreError(e)
    }
}

// ---------------------------------------------------------------------------
// DeepSeekV2ForCausalLM
// ---------------------------------------------------------------------------

/// DeepSeek-V2 with a causal language modelling head.
///
/// The LM head is a simple linear projection `hidden_size → vocab_size`.
/// No weight tying is applied by default (DeepSeek-V2 uses untied embeddings).
pub struct DeepSeekV2ForCausalLM {
    model: DeepSeekV2Model,
    lm_head: Linear,
    device: Device,
}

impl DeepSeekV2ForCausalLM {
    /// Create a new causal LM model on the CPU.
    pub fn new(config: DeepSeekV2Config) -> CoreResult<Self> {
        Self::new_with_device(config, Device::CPU)
    }

    /// Create a new causal LM model on the specified device.
    pub fn new_with_device(config: DeepSeekV2Config, device: Device) -> CoreResult<Self> {
        let lm_head = Linear::new_with_device(config.hidden_size, config.vocab_size, false, device);
        let model = DeepSeekV2Model::new_with_device(config, device)?;
        Ok(Self {
            model,
            lm_head,
            device,
        })
    }

    /// Return a reference to the model configuration.
    pub fn config(&self) -> &DeepSeekV2Config {
        self.model.config()
    }

    pub fn device(&self) -> Device {
        self.device
    }

    /// Run a forward pass over a token-id slice and return flat logits.
    ///
    /// Returns a `Vec<f32>` of length `seq_len * vocab_size`.
    pub fn forward_ids(&self, input_ids: &[u32]) -> Result<Vec<f32>, DeepSeekV2Error> {
        if input_ids.is_empty() {
            return Err(DeepSeekV2Error::EmptyInput);
        }
        let seq_len = input_ids.len();
        let vocab_size = self.config().vocab_size;

        let input_f32: Vec<f32> = input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor =
            Tensor::from_vec(input_f32, &[seq_len]).map_err(DeepSeekV2Error::CoreError)?;

        let hidden = self.model.forward(input_tensor).map_err(DeepSeekV2Error::CoreError)?;
        let logits_tensor = self.lm_head.forward(hidden).map_err(DeepSeekV2Error::CoreError)?;

        let mut logits: Vec<f32> = match &logits_tensor {
            Tensor::F32(arr) => arr
                .as_slice()
                .ok_or_else(|| {
                    DeepSeekV2Error::ForwardError("logits tensor not contiguous".to_string())
                })?
                .to_vec(),
            _ => {
                return Err(DeepSeekV2Error::ForwardError(
                    "logits tensor must be F32".to_string(),
                ))
            },
        };
        logits.resize(seq_len * vocab_size, 0.0);
        Ok(logits)
    }

    /// Greedy-decode `max_new_tokens` tokens starting from `input_ids`.
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, DeepSeekV2Error> {
        if input_ids.is_empty() {
            return Err(DeepSeekV2Error::EmptyInput);
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
                return Err(DeepSeekV2Error::ForwardError(
                    "empty logits at last position".to_string(),
                ));
            }

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| DeepSeekV2Error::ForwardError("argmax failed".to_string()))?;

            generated.push(next_token);
            context.push(next_token);
        }
        Ok(generated)
    }
}

impl Model for DeepSeekV2ForCausalLM {
    type Config = DeepSeekV2Config;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::deepseek_v2::config::{ActivationType, DeepSeekV2Config, TopKMethod};
    use trustformers_core::device::Device;

    fn small_config() -> DeepSeekV2Config {
        DeepSeekV2Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            kv_lora_rank: 8,
            q_lora_rank: 0,
            qk_rope_head_dim: 8,
            qk_nope_head_dim: 8,
            v_head_dim: 8,
            num_experts_per_tok: 2,
            n_routed_experts: 4,
            n_shared_experts: 1,
            routed_scaling_factor: 1.0,
            topk_method: TopKMethod::GroupLimitedGreedy,
            n_group: 2,
            topk_group: 1,
            aux_loss_alpha: 0.001,
            first_k_dense_replace: 1,
            moe_layer_freq: 1,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: ActivationType::SiLU,
            initializer_range: 0.02,
        }
    }

    // ── 1. DeepSeekV2ForCausalLM constructs on CPU ────────────────────────────

    #[test]
    fn test_construction_cpu() {
        let result = DeepSeekV2ForCausalLM::new(small_config());
        assert!(result.is_ok(), "must construct on CPU");
    }

    // ── 2. device() returns CPU ───────────────────────────────────────────────

    #[test]
    fn test_device_is_cpu() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.device(), Device::CPU);
    }

    // ── 3. config accessor returns vocab size ─────────────────────────────────

    #[test]
    fn test_config_accessor_vocab_size() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = DeepSeekV2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 4. forward_ids on valid input succeeds ────────────────────────────────

    #[test]
    fn test_forward_ids_valid_input() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward_ids(&[1u32, 2, 3]);
        assert!(result.is_ok(), "forward_ids must succeed");
    }

    // ── 5. forward_ids empty input returns EmptyInput ─────────────────────────

    #[test]
    fn test_forward_ids_empty_error() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let err = model.forward_ids(&[]);
        assert!(
            matches!(err, Err(DeepSeekV2Error::EmptyInput)),
            "empty input must return EmptyInput"
        );
    }

    // ── 6. forward_ids output length = seq_len * vocab_size ──────────────────

    #[test]
    fn test_forward_ids_output_length() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = DeepSeekV2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        let ids = &[1u32, 2, 3];
        if let Ok(logits) = model.forward_ids(ids) {
            assert_eq!(logits.len(), ids.len() * vocab, "logits length mismatch");
        }
    }

    // ── 7. generate returns correct token count ───────────────────────────────

    #[test]
    fn test_generate_token_count() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.generate(&[1u32, 2], 3);
        assert!(result.is_ok(), "generate must succeed");
        let tokens = result.unwrap_or_else(|_| panic!("generate failed"));
        assert_eq!(tokens.len(), 3, "must return exactly 3 new tokens");
    }

    // ── 8. generate on empty input returns EmptyInput ─────────────────────────

    #[test]
    fn test_generate_empty_input_error() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let err = model.generate(&[], 1);
        assert!(
            matches!(err, Err(DeepSeekV2Error::EmptyInput)),
            "empty must return EmptyInput"
        );
    }

    // ── 9. generated tokens within vocab bounds ───────────────────────────────

    #[test]
    fn test_generated_tokens_within_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = DeepSeekV2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tokens) = model.generate(&[0u32, 1], 4) {
            for &t in &tokens {
                assert!((t as usize) < vocab, "token {t} must be < vocab {vocab}");
            }
        }
    }

    // ── 10. DeepSeekV2Error display EmptyInput ────────────────────────────────

    #[test]
    fn test_error_empty_input_display() {
        let msg = format!("{}", DeepSeekV2Error::EmptyInput);
        assert!(msg.contains("empty"), "EmptyInput must mention 'empty'");
    }

    // ── 11. DeepSeekV2Error display InvalidConfig ─────────────────────────────

    #[test]
    fn test_error_invalid_config_display() {
        let msg = format!("{}", DeepSeekV2Error::InvalidConfig("bad".to_string()));
        assert!(msg.contains("bad"), "InvalidConfig must include message");
    }

    // ── 12. TopKMethod display ────────────────────────────────────────────────

    #[test]
    fn test_top_k_method_display() {
        assert_eq!(
            format!("{}", TopKMethod::GroupLimitedGreedy),
            "GroupLimitedGreedy"
        );
        assert_eq!(format!("{}", TopKMethod::Noaux), "Noaux");
    }

    // ── 13. ActivationType display ────────────────────────────────────────────

    #[test]
    fn test_activation_type_display() {
        assert_eq!(format!("{}", ActivationType::SiLU), "silu");
        assert_eq!(format!("{}", ActivationType::GeLU), "gelu");
    }

    // ── 14. TopKMethod equality ───────────────────────────────────────────────

    #[test]
    fn test_top_k_method_equality() {
        assert_eq!(
            TopKMethod::GroupLimitedGreedy,
            TopKMethod::GroupLimitedGreedy
        );
        assert_ne!(TopKMethod::GroupLimitedGreedy, TopKMethod::Noaux);
    }

    // ── 15. generate is deterministic ────────────────────────────────────────

    #[test]
    fn test_generate_deterministic() {
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let prompt = &[1u32, 2];
        let r1 = model.generate(prompt, 3).unwrap_or_default();
        let r2 = model.generate(prompt, 3).unwrap_or_default();
        assert_eq!(r1, r2, "generate must be deterministic");
    }

    // ── 16. num_parameters via Model trait is nonzero ─────────────────────────

    #[test]
    fn test_model_num_parameters_nonzero() {
        use trustformers_core::traits::Model;
        let model =
            DeepSeekV2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(model.num_parameters() > 0, "num_parameters must be > 0");
    }

    // ── 17. is_dense_layer utility ────────────────────────────────────────────

    #[test]
    fn test_is_dense_layer() {
        let cfg = small_config();
        // Layer 0 is before first_k_dense_replace=1, so layer 0 is dense
        assert!(
            cfg.is_dense_layer(0),
            "layer 0 must be dense (first_k_dense_replace=1)"
        );
        // Layer 1 is not dense
        assert!(!cfg.is_dense_layer(1), "layer 1 must be MoE");
    }

    // ── 18. qk_head_dim utility ───────────────────────────────────────────────

    #[test]
    fn test_qk_head_dim() {
        let cfg = small_config();
        let expected = cfg.qk_rope_head_dim + cfg.qk_nope_head_dim;
        assert_eq!(cfg.qk_head_dim(), expected);
    }
}
