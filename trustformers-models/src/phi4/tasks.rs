use std::fmt;

/// Errors specific to Phi-4 operations.
#[derive(Debug)]
pub enum Phi4Error {
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

impl fmt::Display for Phi4Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Phi4Error::InvalidConfig(msg) => write!(f, "Phi4 invalid config: {}", msg),
            Phi4Error::ShapeMismatch { expected, got } => write!(
                f,
                "Phi4 shape mismatch: expected {:?}, got {:?}",
                expected, got
            ),
            Phi4Error::SequenceTooLong { max, got } => {
                write!(f, "Phi4 sequence too long: max {}, got {}", max, got)
            },
            Phi4Error::ForwardError(msg) => write!(f, "Phi4 forward error: {}", msg),
            Phi4Error::GenerationError(msg) => write!(f, "Phi4 generation error: {}", msg),
            Phi4Error::EmptyInput => write!(f, "Phi4 error: empty input"),
        }
    }
}

impl std::error::Error for Phi4Error {}

// ─── Causal LM wrapper ────────────────────────────────────────────────────────

use crate::phi4::config::Phi4Config;
use crate::phi4::model::Phi4Model;
use trustformers_core::{
    errors::Result,
    layers::Linear,
    tensor::Tensor,
    traits::{Layer, Model},
};

/// Phi-4 causal language model (adds an lm_head on top of `Phi4Model`).
///
/// When `tie_word_embeddings` is `true` the lm_head is a separate `Linear`
/// layer whose weights are initialised to ones (in a real deployment they
/// would be tied to the embed_tokens matrix at load time).
pub struct Phi4ForCausalLM {
    model: Phi4Model,
    lm_head: Linear,
}

impl Phi4ForCausalLM {
    /// Create a new model on CPU.
    pub fn new(config: Phi4Config) -> Result<Self> {
        let hidden_size = config.hidden_size;
        let vocab_size = config.vocab_size;
        let model = Phi4Model::new(config)?;
        // In a full implementation with tied embeddings, lm_head weight would
        // point to the same underlying storage as embed_tokens; here we
        // allocate an independent layer (no bias).
        let lm_head = Linear::new(hidden_size, vocab_size, false);
        Ok(Self { model, lm_head })
    }

    /// Return the model configuration.
    pub fn config(&self) -> &Phi4Config {
        self.model.config()
    }

    /// Greedy generation: extend `input_ids` by `max_new_tokens` tokens.
    ///
    /// Each step runs a full forward pass and picks the argmax of the final
    /// position's logits.  This is intentionally simple — for production use
    /// a KV-cache and beam search should be applied.
    pub fn generate_greedy(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> std::result::Result<Vec<u32>, Phi4Error> {
        if input_ids.is_empty() {
            return Err(Phi4Error::EmptyInput);
        }
        let vocab_size = self.model.config().vocab_size;
        let mut sequence: Vec<u32> = input_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let input_f32: Vec<f32> = sequence.iter().map(|&t| t as f32).collect();
            let seq_len = input_f32.len();
            let input_tensor = Tensor::from_vec(input_f32, &[seq_len])
                .map_err(|e| Phi4Error::ForwardError(e.to_string()))?;

            let logits_tensor =
                self.forward(input_tensor).map_err(|e| Phi4Error::ForwardError(e.to_string()))?;

            // The logit tensor has shape [seq_len * vocab_size]; take the last
            // position's logits.
            let logits: Vec<f32> = match &logits_tensor {
                Tensor::F32(arr) => arr.as_slice().unwrap_or(&[]).to_vec(),
                _ => {
                    return Err(Phi4Error::ForwardError(
                        "unexpected logit tensor type".to_string(),
                    ))
                },
            };

            if logits.len() < vocab_size {
                return Err(Phi4Error::ForwardError(
                    "logit tensor too small".to_string(),
                ));
            }

            // Take logits for the last token position.
            let last_offset = logits.len().saturating_sub(vocab_size);
            let last_logits = &logits[last_offset..];

            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| Phi4Error::GenerationError("argmax failed".to_string()))?;

            generated.push(next_token);
            sequence.push(next_token);
        }

        Ok(generated)
    }
}

impl Model for Phi4ForCausalLM {
    type Config = Phi4Config;
    type Input = Tensor;
    type Output = Tensor;

    fn forward(&self, input_ids: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input_ids)?;
        self.lm_head.forward(hidden_states)
    }

    fn load_pretrained(&mut self, _reader: &mut dyn std::io::Read) -> Result<()> {
        // Stub: weight loading not implemented in this scaffold.
        Ok(())
    }

    fn get_config(&self) -> &Self::Config {
        self.model.config()
    }

    fn num_parameters(&self) -> usize {
        self.model.num_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::phi4::config::Phi4Config;

    fn small_config() -> Phi4Config {
        Phi4Config {
            vocab_size: 256,
            hidden_size: 48,
            intermediate_size: 96,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 12,
            max_position_embeddings: 64,
            original_max_position_embeddings: 32,
            rms_norm_eps: 1e-5,
            rope_theta: 250000.0,
            hidden_act: "silu".to_string(),
            tie_word_embeddings: false,
            attention_dropout: 0.0,
            embd_dropout: 0.0,
            rope_scaling: None,
        }
    }

    // ── 1. Phi4ForCausalLM constructs successfully ─────────────────────────────

    #[test]
    fn test_construction_succeeds() {
        let result = Phi4ForCausalLM::new(small_config());
        assert!(result.is_ok(), "Phi4ForCausalLM must construct");
    }

    // ── 2. config accessor returns correct vocab size ─────────────────────────

    #[test]
    fn test_config_accessor_vocab_size() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = Phi4ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        assert_eq!(model.config().vocab_size, vocab);
    }

    // ── 3. generate_greedy returns correct token count ────────────────────────

    #[test]
    fn test_generate_greedy_token_count() {
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.generate_greedy(&[1u32, 2, 3], 3);
        assert!(result.is_ok(), "generate_greedy must succeed");
        let tokens = result.unwrap_or_else(|_| panic!("generate failed"));
        assert_eq!(tokens.len(), 3, "must return exactly 3 tokens");
    }

    // ── 4. generate_greedy on empty input returns EmptyInput ─────────────────

    #[test]
    fn test_generate_greedy_empty_input_error() {
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let err = model.generate_greedy(&[], 1);
        assert!(
            matches!(err, Err(Phi4Error::EmptyInput)),
            "empty input must return EmptyInput"
        );
    }

    // ── 5. generate_greedy zero new tokens returns empty vec ─────────────────

    #[test]
    fn test_generate_greedy_zero_tokens() {
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let tokens = model.generate_greedy(&[1u32], 0).unwrap_or_default();
        assert!(tokens.is_empty(), "zero new tokens must return empty vec");
    }

    // ── 6. generated tokens within vocab bounds ───────────────────────────────

    #[test]
    fn test_generated_tokens_within_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = Phi4ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tokens) = model.generate_greedy(&[0u32, 1], 4) {
            for &t in &tokens {
                assert!((t as usize) < vocab, "token {t} must be < vocab {vocab}");
            }
        }
    }

    // ── 7. Phi4Error display EmptyInput ───────────────────────────────────────

    #[test]
    fn test_error_empty_input_display() {
        let msg = format!("{}", Phi4Error::EmptyInput);
        assert!(msg.contains("empty"), "EmptyInput must mention 'empty'");
    }

    // ── 8. Phi4Error display InvalidConfig ───────────────────────────────────

    #[test]
    fn test_error_invalid_config_display() {
        let msg = format!("{}", Phi4Error::InvalidConfig("bad".to_string()));
        assert!(msg.contains("bad"), "InvalidConfig must include message");
    }

    // ── 9. Phi4Error display SequenceTooLong ─────────────────────────────────

    #[test]
    fn test_error_sequence_too_long_display() {
        let msg = format!(
            "{}",
            Phi4Error::SequenceTooLong {
                max: 1024,
                got: 2048
            }
        );
        assert!(msg.contains("1024"), "must mention max");
        assert!(msg.contains("2048"), "must mention actual");
    }

    // ── 10. Phi4Error display ForwardError ────────────────────────────────────

    #[test]
    fn test_error_forward_error_display() {
        let msg = format!("{}", Phi4Error::ForwardError("nan output".to_string()));
        assert!(
            msg.contains("nan output"),
            "ForwardError must include message"
        );
    }

    // ── 11. Phi4Error display GenerationError ────────────────────────────────

    #[test]
    fn test_error_generation_error_display() {
        let msg = format!(
            "{}",
            Phi4Error::GenerationError("argmax failed".to_string())
        );
        assert!(
            msg.contains("argmax"),
            "GenerationError must include message"
        );
    }

    // ── 12. Phi4Error display ShapeMismatch ──────────────────────────────────

    #[test]
    fn test_error_shape_mismatch_display() {
        let msg = format!(
            "{}",
            Phi4Error::ShapeMismatch {
                expected: vec![4, 8],
                got: vec![4, 4]
            }
        );
        assert!(!msg.is_empty(), "ShapeMismatch display must be non-empty");
    }

    // ── 13. Phi4Config::phi4_mini preset fields ───────────────────────────────

    #[test]
    fn test_phi4_mini_config_fields() {
        let cfg = Phi4Config::phi4_mini();
        assert_eq!(cfg.num_hidden_layers, 32);
        assert_eq!(cfg.num_attention_heads, 32);
        assert_eq!(cfg.num_key_value_heads, 8);
    }

    // ── 14. Phi4Config gqa_ratio ──────────────────────────────────────────────

    #[test]
    fn test_gqa_ratio() {
        let cfg = Phi4Config::phi4_mini();
        assert_eq!(cfg.gqa_ratio(), 4, "32 Q / 8 KV = 4");
    }

    // ── 15. generate_greedy is deterministic ─────────────────────────────────

    #[test]
    fn test_generate_greedy_deterministic() {
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let prompt = &[1u32, 2];
        let r1 = model.generate_greedy(prompt, 3).unwrap_or_default();
        let r2 = model.generate_greedy(prompt, 3).unwrap_or_default();
        assert_eq!(r1, r2, "generate_greedy must be deterministic");
    }

    // ── 16. Phi4Config validate rejects zero vocab ────────────────────────────

    #[test]
    fn test_validate_rejects_zero_vocab() {
        let mut cfg = small_config();
        cfg.vocab_size = 0;
        let result = cfg.validate();
        assert!(result.is_err(), "zero vocab_size must fail validation");
    }

    // ── 17. Phi4Config LongRoPE preset has rope_scaling ──────────────────────

    #[test]
    fn test_phi4_14b_longrope_has_rope_scaling() {
        let cfg = Phi4Config::phi4_14b_longrope();
        assert!(
            cfg.rope_scaling.is_some(),
            "LongRoPE config must have rope_scaling"
        );
    }

    // ── 18. Model trait forward works ────────────────────────────────────────

    #[test]
    fn test_model_trait_forward() {
        use trustformers_core::traits::Model;
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let input =
            Tensor::from_vec(vec![1.0f32, 2.0], &[2]).unwrap_or_else(|_| panic!("tensor failed"));
        let result = model.forward(input);
        assert!(result.is_ok(), "Model::forward must succeed");
    }

    // ── 19. num_parameters > 0 ───────────────────────────────────────────────

    #[test]
    fn test_num_parameters_nonzero() {
        use trustformers_core::traits::Model;
        let model = Phi4ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        assert!(model.num_parameters() > 0, "num_parameters must be > 0");
    }
}
