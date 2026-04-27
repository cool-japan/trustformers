//! Task-specific wrappers for Jamba-2.

use crate::jamba2::{
    config::{Jamba2Config, Jamba2ConfigError},
    model::{Jamba2Error, Jamba2Model},
};

/// Top-level error type for Jamba-2 operations.
#[derive(Debug, thiserror::Error)]
pub enum Jamba2TaskError {
    #[error("Configuration error: {0}")]
    Config(#[from] Jamba2ConfigError),
    #[error("Model error: {0}")]
    Model(#[from] Jamba2Error),
    #[error("Logits shape mismatch: expected [seq_len={expected_seq}, vocab={expected_vocab}], got [{got}]")]
    LogitsMismatch {
        expected_seq: usize,
        expected_vocab: usize,
        got: usize,
    },
}

/// Language modelling head output.
pub struct CausalLmOutput {
    /// Logits over the vocabulary: [seq_len, vocab_size]
    pub logits: Vec<Vec<f64>>,
    /// Hidden states from the model: [seq_len, hidden_size]
    pub hidden_states: Vec<Vec<f64>>,
}

/// Jamba-2 causal language model.
///
/// Wraps [`Jamba2Model`] with a tied or untied linear projection from
/// hidden states to vocabulary logits.
pub struct Jamba2ForCausalLM {
    model: Jamba2Model,
    /// LM head weight: [vocab_size × hidden_size]
    lm_head: Vec<Vec<f64>>,
}

impl Jamba2ForCausalLM {
    /// Create a new causal LM from configuration.
    ///
    /// Validates the configuration before building the model.
    pub fn new(config: Jamba2Config) -> Result<Self, Jamba2TaskError> {
        config.validate()?;
        let vocab_size = config.vocab_size;
        let hidden_size = config.hidden_size;

        // Small diagonal LM head initialization
        let lm_head: Vec<Vec<f64>> = (0..vocab_size)
            .map(|i| {
                let mut row = vec![0.0f64; hidden_size];
                row[i % hidden_size] = 0.01;
                row
            })
            .collect();

        let model = Jamba2Model::new(config);
        Ok(Self { model, lm_head })
    }

    /// Forward pass: input token IDs → logits over vocabulary.
    ///
    /// Returns [`CausalLmOutput`] with logits and hidden states.
    pub fn forward(&self, input_ids: &[u32]) -> Result<CausalLmOutput, Jamba2TaskError> {
        let hidden_states = self.model.forward(input_ids)?;

        // Project hidden states to vocabulary logits
        let logits: Vec<Vec<f64>> = hidden_states
            .iter()
            .map(|h| {
                self.lm_head
                    .iter()
                    .map(|row| row.iter().zip(h.iter()).map(|(w, v)| w * v).sum())
                    .collect()
            })
            .collect();

        Ok(CausalLmOutput {
            logits,
            hidden_states,
        })
    }

    /// Greedy token generation given a prompt (input_ids).
    ///
    /// Generates up to `max_new_tokens` new tokens by repeatedly picking the argmax
    /// of the last-position logits.
    pub fn generate(
        &self,
        input_ids: &[u32],
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Jamba2TaskError> {
        if input_ids.is_empty() {
            return Err(Jamba2TaskError::Model(Jamba2Error::EmptyInput));
        }

        let vocab_size = self.model.config().vocab_size;
        let mut tokens: Vec<u32> = input_ids.to_vec();

        for _ in 0..max_new_tokens {
            let output = self.forward(&tokens)?;
            let last_logits = &output.logits[output.logits.len() - 1];

            // Argmax over vocabulary
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i as u32)
                .unwrap_or(0);

            // Clamp to vocab size for safety
            tokens.push(next_token % vocab_size as u32);
        }

        Ok(tokens[input_ids.len()..].to_vec())
    }

    /// Reference to the underlying model.
    pub fn model(&self) -> &Jamba2Model {
        &self.model
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jamba2::config::Jamba2Config;

    fn small_config() -> Jamba2Config {
        Jamba2Config {
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 4,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            head_dim: 8,
            mamba_d_state: 4,
            mamba_d_conv: 2,
            mamba_expand: 2,
            mamba_dt_rank: 8,
            attn_layer_offset: 1,
            attn_layer_period: 2,
            expert_layer_offset: 1,
            expert_layer_period: 2,
            num_experts: 4,
            num_experts_per_tok: 2,
            max_position_embeddings: 64,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            hidden_act: "silu".to_string(),
            attention_dropout: 0.0,
            tie_word_embeddings: false,
        }
    }

    // ── 1. Jamba2ForCausalLM constructs successfully ──────────────────────────

    #[test]
    fn test_construction_succeeds() {
        let result = Jamba2ForCausalLM::new(small_config());
        assert!(
            result.is_ok(),
            "Jamba2ForCausalLM must construct successfully"
        );
    }

    // ── 2. forward pass returns output with logits ────────────────────────────

    #[test]
    fn test_forward_produces_output() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.forward(&[1u32, 2, 3]);
        assert!(result.is_ok(), "forward must succeed");
    }

    // ── 3. logits shape: seq_len rows ─────────────────────────────────────────

    #[test]
    fn test_forward_logits_row_count() {
        let cfg = small_config();
        let model = Jamba2ForCausalLM::new(cfg.clone()).unwrap_or_else(|_| panic!("init failed"));
        let out = model.forward(&[1u32, 2, 3]).unwrap_or_else(|_| panic!("forward failed"));
        assert_eq!(out.logits.len(), 3, "logits must have seq_len rows");
    }

    // ── 4. logits vocab dim matches config ────────────────────────────────────

    #[test]
    fn test_forward_logits_vocab_dim() {
        let cfg = small_config();
        let vocab = cfg.vocab_size;
        let model = Jamba2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        let out = model.forward(&[0u32]).unwrap_or_else(|_| panic!("forward failed"));
        assert_eq!(
            out.logits[0].len(),
            vocab,
            "logit row must have vocab_size columns"
        );
    }

    // ── 5. hidden_states length matches seq_len ───────────────────────────────

    #[test]
    fn test_forward_hidden_states_length() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let out = model.forward(&[1u32, 2]).unwrap_or_else(|_| panic!("forward failed"));
        assert_eq!(
            out.hidden_states.len(),
            2,
            "hidden_states must have seq_len rows"
        );
    }

    // ── 6. generate returns correct token count ───────────────────────────────

    #[test]
    fn test_generate_token_count() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let result = model.generate(&[1u32, 2], 3);
        assert!(result.is_ok(), "generate must succeed");
        let tokens = result.unwrap_or_else(|_| panic!("generate failed"));
        assert_eq!(tokens.len(), 3, "must return exactly 3 new tokens");
    }

    // ── 7. generate on empty input returns error ──────────────────────────────

    #[test]
    fn test_generate_empty_input_error() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let err = model.generate(&[], 1);
        assert!(err.is_err(), "empty input must return error");
    }

    // ── 8. generate zero tokens returns empty vec ─────────────────────────────

    #[test]
    fn test_generate_zero_tokens() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let tokens = model.generate(&[1u32], 0).unwrap_or_default();
        assert!(tokens.is_empty(), "zero new tokens must return empty vec");
    }

    // ── 9. generated tokens within vocab bounds ───────────────────────────────

    #[test]
    fn test_generated_tokens_within_vocab() {
        let cfg = small_config();
        let vocab = cfg.vocab_size as u32;
        let model = Jamba2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(tokens) = model.generate(&[1u32, 2], 3) {
            for &t in &tokens {
                assert!(t < vocab, "token {t} must be < vocab {vocab}");
            }
        }
    }

    // ── 10. model() accessor returns model reference ──────────────────────────

    #[test]
    fn test_model_accessor_works() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let _m = model.model();
    }

    // ── 11. Jamba2TaskError Config variant display ────────────────────────────

    #[test]
    fn test_task_error_config_display() {
        use crate::jamba2::config::Jamba2ConfigError;
        let err = Jamba2TaskError::Config(Jamba2ConfigError::InvalidField("test".to_string()));
        let msg = format!("{err}");
        assert!(!msg.is_empty(), "error display must be non-empty");
    }

    // ── 12. CausalLmOutput logits field accessible ────────────────────────────

    #[test]
    fn test_causal_lm_output_accessible() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(&[0u32]) {
            let _ = &out.logits;
            let _ = &out.hidden_states;
        }
    }

    // ── 13. forward logits are all finite ────────────────────────────────────

    #[test]
    fn test_forward_logits_finite() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(&[1u32]) {
            for row in &out.logits {
                for &v in row {
                    assert!(v.is_finite(), "logit {v} must be finite");
                }
            }
        }
    }

    // ── 14. forward is deterministic ─────────────────────────────────────────

    #[test]
    fn test_forward_deterministic() {
        let model =
            Jamba2ForCausalLM::new(small_config()).unwrap_or_else(|_| panic!("init failed"));
        let ids = &[1u32, 2, 3];
        let r1 = model.forward(ids);
        let r2 = model.forward(ids);
        if let (Ok(a), Ok(b)) = (r1, r2) {
            for (row_a, row_b) in a.logits.iter().zip(b.logits.iter()) {
                assert_eq!(row_a, row_b, "forward must be deterministic");
            }
        }
    }

    // ── 15. validate rejects zero vocab size ──────────────────────────────────

    #[test]
    fn test_validate_rejects_zero_vocab() {
        let mut cfg = small_config();
        cfg.vocab_size = 0;
        let result = Jamba2ForCausalLM::new(cfg);
        assert!(result.is_err(), "zero vocab_size must be rejected");
    }

    // ── 16. hidden_states dim matches hidden_size ─────────────────────────────

    #[test]
    fn test_hidden_states_dim_matches_config() {
        let cfg = small_config();
        let hidden_size = cfg.hidden_size;
        let model = Jamba2ForCausalLM::new(cfg).unwrap_or_else(|_| panic!("init failed"));
        if let Ok(out) = model.forward(&[0u32]) {
            for row in &out.hidden_states {
                assert_eq!(row.len(), hidden_size, "hidden dim must match config");
            }
        }
    }
}
