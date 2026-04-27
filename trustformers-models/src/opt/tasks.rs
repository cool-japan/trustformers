//! Task-specific heads and utilities for OPT models.
//!
//! Provides:
//! - `OptForCausalLM` — adds an LM head on top of `OptModel`.
//! - `format_completion_prompt` — identity passthrough (OPT uses raw completion).

use crate::opt::config::OptConfig;
use crate::opt::model::{OptError, OptLinear, OptModel};

// ─── OPT Causal LM output ──────────────────────────────────────────────────

/// Output of a single `OptForCausalLM::forward` call.
#[derive(Debug, Clone)]
pub struct OptCausalLMOutput {
    /// Raw logits: flat `[seq_len * vocab_size]` vector.
    pub logits: Vec<f32>,
}

// ─── OptForCausalLM ────────────────────────────────────────────────────────

/// OPT model with an LM head for causal language modelling.
///
/// The LM head is a linear projection from `hidden_size` (or `word_embed_proj_dim`
/// if they differ) to `vocab_size`, without bias.
#[derive(Debug, Clone)]
pub struct OptForCausalLM {
    pub config: OptConfig,
    model: OptModel,
    /// LM head: maps from the decoder's output dimension to `vocab_size`.
    lm_head: OptLinear,
}

impl OptForCausalLM {
    /// Construct a new (untrained) `OptForCausalLM`.
    pub fn new(config: OptConfig) -> Result<Self, OptError> {
        let model = OptModel::new(&config)?;
        // OPT ties the LM head to the token embedding when word_embed_proj_dim == hidden_size.
        // Either way we create an independent (zero-init) projection here as a stand-in.
        let lm_head = OptLinear::new(config.word_embed_proj_dim, config.vocab_size);
        Ok(Self {
            config,
            model,
            lm_head,
        })
    }

    /// Run a full forward pass.
    ///
    /// # Arguments
    ///
    /// * `input_ids` — token indices.
    ///
    /// # Returns
    ///
    /// `OptCausalLMOutput` containing flat logits of shape `[seq_len * vocab_size]`.
    pub fn forward(&self, input_ids: &[u32]) -> Result<OptCausalLMOutput, OptError> {
        if input_ids.is_empty() {
            return Err(OptError::Forward("input_ids must not be empty".to_string()));
        }
        let seq_len = input_ids.len();
        let hidden = self.model.forward(input_ids)?;
        let logits = self.lm_head.forward(&hidden, seq_len)?;
        Ok(OptCausalLMOutput { logits })
    }

    /// Greedy autoregressive generation.
    ///
    /// Appends `max_new_tokens` tokens to `input_ids` by selecting the argmax
    /// of the last-token logits at each step.  Generation stops early if the
    /// EOS token is produced.
    ///
    /// # Errors
    ///
    /// Returns `OptError::Generation` if `max_new_tokens` is zero or the
    /// forward pass fails.
    pub fn generate(&self, input_ids: &[u32], max_new_tokens: usize) -> Result<Vec<u32>, OptError> {
        if max_new_tokens == 0 {
            return Err(OptError::Generation(
                "max_new_tokens must be > 0".to_string(),
            ));
        }
        if input_ids.is_empty() {
            return Err(OptError::Generation(
                "input_ids must not be empty".to_string(),
            ));
        }

        let mut ids: Vec<u32> = input_ids.to_vec();
        let vocab_size = self.config.vocab_size;
        let eos = self.config.eos_token_id;

        for _ in 0..max_new_tokens {
            let out = self.forward(&ids)?;
            let logits = &out.logits;
            let seq_len = ids.len();
            // Logits for the last token: [vocab_size]
            let last_start = (seq_len - 1) * vocab_size;
            let last_logits = &logits[last_start..last_start + vocab_size];
            let next_tok = greedy_argmax(last_logits)
                .ok_or_else(|| OptError::Generation("empty logit slice".to_string()))?;
            ids.push(next_tok);
            if next_tok == eos {
                break;
            }
        }

        Ok(ids)
    }
}

// ─── Prompt formatting ─────────────────────────────────────────────────────

/// Format a completion prompt for OPT.
///
/// OPT does not have a special chat template; the prompt is passed through as-is.
/// This function is provided for API symmetry with other models.
pub fn format_completion_prompt(context: &str) -> String {
    context.to_string()
}

// ─── Internal helpers ──────────────────────────────────────────────────────

/// Return the index of the maximum value in `logits`, or `None` if empty.
fn greedy_argmax(logits: &[f32]) -> Option<u32> {
    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(i, _)| i as u32)
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_config() -> OptConfig {
        OptConfig {
            vocab_size: 100,
            hidden_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 4,
            ffn_dim: 64,
            max_position_embeddings: 16,
            word_embed_proj_dim: 32,
            layer_norm_eps: 1e-5,
            dropout: 0.0,
            do_layer_norm_before: true,
            activation_function: "relu".to_string(),
            use_cache: true,
            bos_token_id: 2,
            eos_token_id: 2,
            pad_token_id: Some(1),
        }
    }

    #[test]
    fn test_opt_causal_lm_forward() {
        let cfg = tiny_config();
        let model = OptForCausalLM::new(cfg.clone()).expect("model creation should succeed");
        let input_ids = vec![0u32, 5, 10];
        let out = model.forward(&input_ids).expect("forward should succeed");
        // logits shape: [seq_len * vocab_size]
        assert_eq!(out.logits.len(), input_ids.len() * cfg.vocab_size);
    }

    #[test]
    fn test_opt_causal_lm_empty_input_error() {
        let cfg = tiny_config();
        let model = OptForCausalLM::new(cfg).expect("model creation should succeed");
        let err = model.forward(&[]);
        assert!(err.is_err());
    }

    #[test]
    fn test_opt_generate() {
        let cfg = tiny_config();
        let model = OptForCausalLM::new(cfg.clone()).expect("model creation should succeed");
        let input_ids = vec![0u32, 5];
        let generated = model.generate(&input_ids, 3).expect("generate should succeed");
        // Generated sequence must be at least as long as the input (and at most +3)
        assert!(generated.len() >= input_ids.len());
        assert!(generated.len() <= input_ids.len() + 3);
        // Prefix must match
        assert_eq!(&generated[..input_ids.len()], input_ids.as_slice());
    }

    #[test]
    fn test_opt_generate_zero_tokens_error() {
        let cfg = tiny_config();
        let model = OptForCausalLM::new(cfg).expect("model creation should succeed");
        let err = model.generate(&[0u32], 0);
        assert!(err.is_err());
    }

    #[test]
    fn test_opt_completion_format() {
        let prompt = "The capital of France is";
        let formatted = format_completion_prompt(prompt);
        assert_eq!(formatted, prompt);
    }

    #[test]
    fn test_opt_completion_format_empty() {
        let formatted = format_completion_prompt("");
        assert_eq!(formatted, "");
    }

    #[test]
    fn test_opt_error_display_causal() {
        let e = OptError::Forward("bad input".to_string());
        assert!(e.to_string().contains("bad input"));
    }
}
