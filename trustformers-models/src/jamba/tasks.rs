//! Jamba task-specific heads and error types.
//!
//! Provides `JambaForCausalLM` (re-exported from model.rs) plus
//! output structs, a structured `JambaError` enum, and greedy generation
//! helpers that work with the Jamba hybrid SSM+Attention backbone.

use crate::jamba::config::JambaConfig;
use crate::jamba::model::{JambaError, JambaForCausalLM, JambaModel};

// ─────────────────────────────────────────────────────────────────────────────
// Re-export error type from model so callers can import from tasks
// ─────────────────────────────────────────────────────────────────────────────

pub use crate::jamba::model::JambaError as JambaTaskError;

// ─────────────────────────────────────────────────────────────────────────────
// Output structs
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a Jamba causal LM forward pass.
///
/// The `logits` field contains the full vocabulary distribution for each
/// token position.  Hidden states are optionally captured for analysis.
pub struct JambaCausalLMOutput {
    /// Logits over vocabulary: `[seq_len][vocab_size]`.
    pub logits: Vec<Vec<f64>>,
    /// SSM (Mamba) hidden states, one entry per Mamba layer, if requested.
    pub ssm_hidden_states: Option<Vec<Vec<Vec<f64>>>>,
    /// Attention hidden states, one entry per attention layer, if requested.
    pub attention_hidden_states: Option<Vec<Vec<Vec<f64>>>>,
}

// ─────────────────────────────────────────────────────────────────────────────
// JambaCausalLMHead — task wrapper
// ─────────────────────────────────────────────────────────────────────────────

/// Jamba causal language model: backbone + LM head.
///
/// This wraps `JambaForCausalLM` from `model.rs` and adds:
/// - a structured `JambaCausalLMOutput` (including optional hidden states)
/// - greedy generation
/// - convenience constructors for named presets
pub struct JambaCausalLMHead {
    inner: JambaForCausalLM,
    config: JambaConfig,
}

impl JambaCausalLMHead {
    /// Create a new JambaCausalLMHead from an explicit config.
    pub fn new(config: JambaConfig) -> Result<Self, JambaError> {
        let inner = JambaForCausalLM::new(&config);
        Ok(Self { inner, config })
    }

    /// Create from the Jamba-1.5B preset configuration.
    pub fn jamba_1_5b() -> Result<Self, JambaError> {
        Self::new(JambaConfig::jamba_1_5b())
    }

    /// Create from the small test configuration.
    pub fn small_test() -> Result<Self, JambaError> {
        Self::new(JambaConfig::small_test())
    }

    /// Run a forward pass returning the full `JambaCausalLMOutput`.
    ///
    /// `input_ids` must be non-empty and all tokens must be `< vocab_size`.
    pub fn forward(&self, input_ids: &[usize]) -> Result<JambaCausalLMOutput, JambaError> {
        if input_ids.is_empty() {
            return Err(JambaError::EmptyInput);
        }
        for &id in input_ids {
            if id >= self.config.vocab_size {
                return Err(JambaError::LayerError {
                    layer: 0,
                    msg: format!("token id {} >= vocab_size {}", id, self.config.vocab_size),
                });
            }
        }

        let logits = self.inner.forward(input_ids)?;

        Ok(JambaCausalLMOutput {
            logits,
            ssm_hidden_states: None,
            attention_hidden_states: None,
        })
    }

    /// Greedy decoding: append up to `max_new_tokens` tokens.
    ///
    /// At each step the token with the highest logit at the last position is
    /// selected and appended to the context.  Returns only the newly generated
    /// token ids (not the prompt).
    pub fn generate_greedy(
        &self,
        input_ids: &[usize],
        max_new_tokens: usize,
    ) -> Result<Vec<usize>, JambaError> {
        if input_ids.is_empty() {
            return Err(JambaError::EmptyInput);
        }

        let vocab_size = self.config.vocab_size;
        let mut context: Vec<usize> = input_ids.to_vec();
        let mut generated = Vec::with_capacity(max_new_tokens);

        for _ in 0..max_new_tokens {
            let output = self.forward(&context)?;
            let last_logits = &output.logits[context.len() - 1];
            if last_logits.len() != vocab_size {
                return Err(JambaError::LayerError {
                    layer: 0,
                    msg: format!(
                        "logits dim {} != vocab_size {}",
                        last_logits.len(),
                        vocab_size
                    ),
                });
            }
            // Select argmax
            let next_token = last_logits
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, _)| i)
                .ok_or_else(|| JambaError::LayerError {
                    layer: 0,
                    msg: "empty logits vector".to_string(),
                })?;
            context.push(next_token);
            generated.push(next_token);
        }

        Ok(generated)
    }

    /// Reference to the underlying config.
    pub fn config(&self) -> &JambaConfig {
        &self.config
    }

    /// Reference to the underlying model for layer inspection.
    pub fn model(&self) -> &JambaModel {
        self.inner.model()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Format a simple Jamba Instruct-style prompt (AI21 Labs format).
///
/// Returns the string wrapped with the AI21 conversation markers.
pub fn format_jamba_prompt(system: Option<&str>, user: &str) -> String {
    let mut out = String::new();
    if let Some(sys) = system {
        out.push_str("<|system|>\n");
        out.push_str(sys);
        out.push('\n');
    }
    out.push_str("<|user|>\n");
    out.push_str(user);
    out.push_str("\n<|assistant|>\n");
    out
}

/// Compute the number of Mamba (SSM) layers in a config.
pub fn count_ssm_layers(config: &JambaConfig) -> usize {
    (0..config.num_hidden_layers).filter(|&i| !config.is_attention_layer(i)).count()
}

/// Compute the number of attention layers in a config.
pub fn count_attention_layers(config: &JambaConfig) -> usize {
    (0..config.num_hidden_layers).filter(|&i| config.is_attention_layer(i)).count()
}

/// Compute the number of MoE layers in a config.
pub fn count_moe_layers(config: &JambaConfig) -> usize {
    (0..config.num_hidden_layers).filter(|&i| config.is_moe_layer(i)).count()
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jamba::config::JambaConfig;

    fn small_cfg() -> JambaConfig {
        JambaConfig::small_test()
    }

    #[test]
    fn test_tasks_forward_output_shape() {
        let head = JambaCausalLMHead::small_test().expect("should construct");
        let input = vec![0usize, 1, 2];
        let out = head.forward(&input).expect("forward should succeed");
        assert_eq!(
            out.logits.len(),
            3,
            "logits should have one entry per token"
        );
        let cfg = small_cfg();
        assert_eq!(
            out.logits[0].len(),
            cfg.vocab_size,
            "each logit row must match vocab_size"
        );
    }

    #[test]
    fn test_tasks_forward_empty_input_error() {
        let head = JambaCausalLMHead::small_test().expect("construct");
        assert!(
            head.forward(&[]).is_err(),
            "empty input should return an error"
        );
    }

    #[test]
    fn test_tasks_forward_oov_token_error() {
        let head = JambaCausalLMHead::small_test().expect("construct");
        let oov = vec![9999usize]; // vocab_size=256
        assert!(
            head.forward(&oov).is_err(),
            "out-of-vocab token should return error"
        );
    }

    #[test]
    fn test_tasks_generate_greedy_length() {
        let head = JambaCausalLMHead::small_test().expect("construct");
        let input = vec![0usize, 1];
        let generated = head.generate_greedy(&input, 3).expect("generation should succeed");
        assert_eq!(generated.len(), 3, "should generate exactly 3 tokens");
    }

    #[test]
    fn test_tasks_generate_greedy_tokens_in_vocab() {
        let head = JambaCausalLMHead::small_test().expect("construct");
        let input = vec![0usize, 1];
        let cfg = small_cfg();
        let generated = head.generate_greedy(&input, 5).expect("generation should succeed");
        for &tok in &generated {
            assert!(
                tok < cfg.vocab_size,
                "generated token {} must be within vocab_size {}",
                tok,
                cfg.vocab_size
            );
        }
    }

    #[test]
    fn test_tasks_generate_greedy_empty_input_error() {
        let head = JambaCausalLMHead::small_test().expect("construct");
        assert!(
            head.generate_greedy(&[], 3).is_err(),
            "empty input for generation should return error"
        );
    }

    #[test]
    fn test_tasks_format_jamba_prompt_with_system() {
        let prompt = format_jamba_prompt(Some("Be helpful."), "Hello!");
        assert!(
            prompt.contains("<|system|>"),
            "should contain system marker"
        );
        assert!(prompt.contains("Be helpful."), "should contain system text");
        assert!(prompt.contains("<|user|>"), "should contain user marker");
        assert!(prompt.contains("Hello!"), "should contain user text");
        assert!(
            prompt.contains("<|assistant|>"),
            "should contain assistant marker"
        );
    }

    #[test]
    fn test_tasks_format_jamba_prompt_without_system() {
        let prompt = format_jamba_prompt(None, "How are you?");
        assert!(
            !prompt.contains("<|system|>"),
            "should not contain system marker"
        );
        assert!(prompt.contains("<|user|>"), "should contain user marker");
        assert!(prompt.contains("How are you?"), "should contain user text");
    }

    #[test]
    fn test_tasks_count_ssm_layers_small() {
        let cfg = small_cfg();
        // 8 total layers, attention at offset=3, period=8 → only layer 3 is attention
        // So SSM count = 7
        let ssm_count = count_ssm_layers(&cfg);
        let attn_count = count_attention_layers(&cfg);
        assert_eq!(ssm_count + attn_count, cfg.num_hidden_layers);
        assert!(ssm_count > 0);
        assert!(attn_count > 0);
    }

    #[test]
    fn test_tasks_count_moe_layers_small() {
        let cfg = small_cfg();
        let moe_count = count_moe_layers(&cfg);
        let attn_count = count_attention_layers(&cfg);
        // MoE layers are a subset of attention layers
        assert!(moe_count <= attn_count);
    }

    #[test]
    fn test_tasks_config_accessor() {
        let cfg = small_cfg();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        assert_eq!(head.config().vocab_size, cfg.vocab_size);
        assert_eq!(head.config().hidden_size, cfg.hidden_size);
    }

    #[test]
    fn test_tasks_model_layer_count() {
        let cfg = small_cfg();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        let layers = head.model().layers();
        assert_eq!(layers.len(), cfg.num_hidden_layers);
    }

    #[test]
    fn test_tasks_layer_type_classification() {
        let cfg = small_cfg();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        let layers = head.model().layers();
        for (i, layer) in layers.iter().enumerate() {
            if cfg.is_attention_layer(i) {
                assert!(layer.is_attention(), "layer {} should be attention", i);
                assert!(!layer.is_mamba(), "layer {} should not be mamba", i);
            } else {
                assert!(layer.is_mamba(), "layer {} should be mamba", i);
                assert!(!layer.is_attention(), "layer {} should not be attention", i);
            }
        }
    }

    #[test]
    fn test_tasks_moe_layer_subset_of_attention() {
        let cfg = small_cfg();
        let head = JambaCausalLMHead::new(cfg.clone()).expect("construct");
        let layers = head.model().layers();
        for (i, layer) in layers.iter().enumerate() {
            if layer.is_moe() {
                assert!(
                    layer.is_attention(),
                    "MoE layer {} must also be an attention layer",
                    i
                );
            }
        }
    }
}
