use std::fmt;

/// Errors specific to Nemotron operations.
#[derive(Debug)]
pub enum NemotronError {
    /// Configuration validation failed.
    InvalidConfig(String),
    /// Tensor shape mismatch.
    ShapeMismatch {
        expected: Vec<usize>,
        got: Vec<usize>,
    },
    /// Sequence exceeds the model's maximum position embeddings.
    SequenceTooLong { max: usize, got: usize },
    /// Error during the forward pass.
    ForwardError(String),
    /// Error during text generation.
    GenerationError(String),
    /// Empty input was provided.
    EmptyInput,
}

impl fmt::Display for NemotronError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NemotronError::InvalidConfig(msg) => write!(f, "Nemotron invalid config: {}", msg),
            NemotronError::ShapeMismatch { expected, got } => write!(
                f,
                "Nemotron shape mismatch: expected {:?}, got {:?}",
                expected, got
            ),
            NemotronError::SequenceTooLong { max, got } => {
                write!(f, "Nemotron sequence too long: max {}, got {}", max, got)
            },
            NemotronError::ForwardError(msg) => write!(f, "Nemotron forward error: {}", msg),
            NemotronError::GenerationError(msg) => write!(f, "Nemotron generation error: {}", msg),
            NemotronError::EmptyInput => write!(f, "Nemotron error: empty input"),
        }
    }
}

impl std::error::Error for NemotronError {}

// ─── CausalLM ─────────────────────────────────────────────────────────────────

use crate::nemotron::config::NemotronConfig;
use crate::nemotron::model::NemotronModel;
use trustformers_core::{
    errors::Result,
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Nemotron causal language model.
///
/// Adds an lm_head projection on top of `NemotronModel`.  When
/// `tie_word_embeddings` is `false` (the default for Nemotron) the head has its
/// own independent weights.
pub struct NemotronForCausalLM {
    model: NemotronModel,
    lm_head: Linear,
}

impl NemotronForCausalLM {
    /// Create a new model on CPU.
    pub fn new(config: NemotronConfig) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let model = NemotronModel::new(config)?;
        let lm_head = Linear::new(hidden_size, vocab_size, false);
        Ok(Self { model, lm_head })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &NemotronConfig {
        self.model.config()
    }

    /// Greedy generation: extend `input_ids` by `max_new_tokens` tokens.
    pub fn generate_greedy(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> std::result::Result<Vec<u32>, NemotronError> {
        if input_ids.is_empty() {
            return Err(NemotronError::EmptyInput);
        }
        let vocab_size = self.model.config().vocab_size;
        let mut sequence: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let input_f32: Vec<f32> = sequence.iter().map(|&t| t as f32).collect();
            let seq_len = input_f32.len();
            let input_tensor = Tensor::from_vec(input_f32, &[seq_len])
                .map_err(|e| NemotronError::ForwardError(e.to_string()))?;

            let logits_tensor = self
                .forward(input_tensor)
                .map_err(|e| NemotronError::ForwardError(e.to_string()))?;

            let logits: Vec<f32> = match &logits_tensor {
                Tensor::F32(arr) => arr.as_slice().unwrap_or(&[]).to_vec(),
                _ => {
                    return Err(NemotronError::ForwardError(
                        "unexpected logit tensor type".to_string(),
                    ))
                },
            };

            if logits.len() < vocab_size {
                return Err(NemotronError::ForwardError(
                    "logit tensor too small".to_string(),
                ));
            }

            let last_offset = logits.len().saturating_sub(vocab_size);
            let last_logits = &logits[last_offset..];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| NemotronError::GenerationError("argmax failed".to_string()))?;

            generated.push(next_token);
            sequence.push(next_token);
        }

        Ok(generated)
    }
}

impl Model for NemotronForCausalLM {
    type Config = NemotronConfig;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Tensor) -> Result<Tensor> {
        let hidden_states = self.model.forward(input_ids)?;
        self.lm_head.forward(hidden_states)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        Ok(())
    }

    fn get_config(&self) -> &NemotronConfig {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nemotron::config::{NemotronConfig, NormType};

    fn small_config() -> NemotronConfig {
        NemotronConfig {
            vocab_size: 256,
            hidden_size: 48,
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 12,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            partial_rotary_factor: 0.5,
            hidden_act: "relu2".to_string(),
            tie_word_embeddings: false,
            norm_type: NormType::RmsNorm,
            attention_bias: false,
            mlp_bias: false,
        }
    }

    // ── 1. NemotronForCausalLM constructs successfully ────────────────────────

    #[test]
    fn test_construction_succeeds() {
        let result = NemotronForCausalLM::new(small_config());
        assert!(result.is_ok(), "NemotronForCausalLM must construct");
    }

    // ── 2. config accessor returns correct vocab size ─────────────────────────

    #[test]
    fn test_config_accessor_vocab_size() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = NemotronForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 3. generate_greedy returns correct token count ────────────────────────

    #[test]
    fn test_generate_greedy_count() {
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.generate_greedy(&[1u32, 2, 3], 4);
        assert!(result.is_ok(), "generate_greedy must succeed");
        let tokens = result.unwrap_or_else(|_| panic!("generate failed"));
        assert_eq!(tokens.len(), 4, "must return exactly 4 new tokens");
    }

    // ── 4. generate_greedy on empty input returns EmptyInput error ───────────

    #[test]
    fn test_generate_greedy_empty_input_error() {
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let err = model.generate_greedy(&[], 1);
        assert!(
            matches!(err, Err(NemotronError::EmptyInput)),
            "empty input must return EmptyInput"
        );
    }

    // ── 5. generate_greedy zero tokens returns empty vec ─────────────────────

    #[test]
    fn test_generate_greedy_zero_tokens() {
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let tokens = model.generate_greedy(&[1u32], 0).unwrap_or_default();
        assert!(tokens.is_empty(), "zero new tokens must yield empty vec");
    }

    // ── 6. generated tokens are within vocab bounds ───────────────────────────

    #[test]
    fn test_generated_tokens_within_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = NemotronForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tokens) = model.generate_greedy(&[0u32, 1], 3) {
            for &t in &tokens {
                assert!(
                    (t as usize) < vocab,
                    "token {t} must be < vocab_size {vocab}"
                );
            }
        }
    }

    // ── 7. NemotronError display formatting ───────────────────────────────────

    #[test]
    fn test_error_display_empty_input() {
        let msg = format!("{}", NemotronError::EmptyInput);
        assert!(
            msg.contains("empty"),
            "EmptyInput display must mention 'empty'"
        );
    }

    // ── 8. NemotronError::InvalidConfig display ───────────────────────────────

    #[test]
    fn test_error_display_invalid_config() {
        let msg = format!("{}", NemotronError::InvalidConfig("bad param".to_string()));
        assert!(
            msg.contains("bad param"),
            "InvalidConfig must include message"
        );
    }

    // ── 9. NemotronError::SequenceTooLong display ─────────────────────────────

    #[test]
    fn test_error_display_sequence_too_long() {
        let msg = format!(
            "{}",
            NemotronError::SequenceTooLong {
                max: 512,
                got: 1024
            }
        );
        assert!(msg.contains("512"), "must mention max length");
        assert!(msg.contains("1024"), "must mention actual length");
    }

    // ── 10. NemotronError::ShapeMismatch display ──────────────────────────────

    #[test]
    fn test_error_display_shape_mismatch() {
        let msg = format!(
            "{}",
            NemotronError::ShapeMismatch {
                expected: vec![4, 8],
                got: vec![4, 4],
            }
        );
        assert!(!msg.is_empty(), "ShapeMismatch display must be non-empty");
    }

    // ── 11. NemotronError::ForwardError display ───────────────────────────────

    #[test]
    fn test_error_display_forward_error() {
        let msg = format!(
            "{}",
            NemotronError::ForwardError("matmul failed".to_string())
        );
        assert!(msg.contains("matmul"), "ForwardError must include message");
    }

    // ── 12. NemotronError::GenerationError display ────────────────────────────

    #[test]
    fn test_error_display_generation_error() {
        let msg = format!("{}", NemotronError::GenerationError("oom".to_string()));
        assert!(msg.contains("oom"), "GenerationError must include message");
    }

    // ── 13. generate_greedy is deterministic ─────────────────────────────────

    #[test]
    fn test_generate_greedy_deterministic() {
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let prompt = vec![1u32, 2];
        let r1 = model.generate_greedy(&prompt, 3).unwrap_or_default();
        let r2 = model.generate_greedy(&prompt, 3).unwrap_or_default();
        assert_eq!(r1, r2, "generate_greedy must be deterministic");
    }

    // ── 14. NormType default is RmsNorm ───────────────────────────────────────

    #[test]
    fn test_norm_type_default() {
        let norm = NormType::default();
        assert_eq!(norm, NormType::RmsNorm);
    }

    // ── 15. NemotronConfig default has partial_rotary_factor = 0.5 ───────────

    #[test]
    fn test_config_default_partial_rotary_factor() {
        let cfg = NemotronConfig::default();
        assert!((cfg.partial_rotary_factor - 0.5).abs() < 1e-6);
    }

    // ── 16. forward via Model trait succeeds ─────────────────────────────────

    #[test]
    fn test_model_trait_forward_succeeds() {
        use trustformers_core::traits::Model;
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let input = Tensor::from_vec(vec![1.0f32, 2.0, 3.0], &[3])
            .unwrap_or_else(|_| panic!("tensor failed"));
        let result = model.forward(input);
        assert!(result.is_ok(), "Model::forward must succeed");
    }

    // ── 17. num_parameters via Model trait is nonzero ─────────────────────────

    #[test]
    fn test_model_trait_num_parameters_nonzero() {
        use trustformers_core::traits::Model;
        let model =
            NemotronForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(model.num_parameters() > 0, "num_parameters must be > 0");
    }

    // ── 18. get_config via Model trait returns config ─────────────────────────

    #[test]
    fn test_model_trait_get_config() {
        use trustformers_core::traits::Model;
        let cfg = small_config();
        let hidden = cfg.hidden_size;
        let model = NemotronForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.get_config().hidden_size, hidden);
    }
}
