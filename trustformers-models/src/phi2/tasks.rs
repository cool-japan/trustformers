use crate::phi2::config::Phi2Config;
use crate::phi2::model::Phi2ForCausalLM;
use trustformers_core::errors::{Result, TrustformersError};
use trustformers_core::tensor::Tensor;
use trustformers_core::traits::Tokenizer;

// ─────────────────────────────────────────────────────────────────────────────
// Task-specific wrappers for Phi-2
// ─────────────────────────────────────────────────────────────────────────────

/// Output of a Phi-2 causal LM forward pass
pub struct Phi2CausalLMOutput {
    /// Token logits, shape `[seq_len, vocab_size]`
    pub logits: Tensor,
}

/// Code-generation task wrapper around Phi-2
///
/// Phi-2 was trained on Python and natural-language code from the web, making
/// it well-suited for short code-generation tasks despite its small size.
pub struct Phi2ForCodeGeneration {
    inner: Phi2ForCausalLM,
}

impl Phi2ForCodeGeneration {
    /// Construct with random initialisation from a config
    pub fn new(config: Phi2Config) -> Result<Self> {
        let inner = Phi2ForCausalLM::new(config)?;
        Ok(Self { inner })
    }

    pub fn config(&self) -> &Phi2Config {
        self.inner.config()
    }

    pub fn parameter_count(&self) -> usize {
        self.inner.parameter_count()
    }

    /// Run a forward pass and return token logits.
    pub fn forward(&self, input_ids: Vec<u32>) -> Result<Phi2CausalLMOutput> {
        let logits = self.inner.forward(input_ids)?;
        Ok(Phi2CausalLMOutput { logits })
    }

    /// Greedily select the most-probable next token from the **last position**
    /// of a `[seq_len, vocab_size]` logit tensor.
    ///
    /// # Errors
    ///
    /// Fails when the tensor is not `F32`, is not 2-D, or carries no positions
    /// or no vocabulary — each of which means the caller has a bug that a
    /// silently-returned token id 0 would hide.
    pub fn greedy_next_token(&self, logits: &Tensor) -> Result<u32> {
        let Tensor::F32(arr) = logits else {
            return Err(TrustformersError::tensor_op_error(
                "Phi2ForCodeGeneration::greedy_next_token",
                "logits must be an F32 tensor",
            ));
        };
        let shape = arr.shape();
        let (seq_len, vocab_size) = match shape {
            [seq_len, vocab_size] => (*seq_len, *vocab_size),
            [vocab_size] => (1usize, *vocab_size),
            other => {
                return Err(TrustformersError::shape_error(format!(
                    "logits must be [seq_len, vocab_size], got {other:?}"
                )))
            },
        };
        if seq_len == 0 || vocab_size == 0 {
            return Err(TrustformersError::shape_error(format!(
                "logits with shape {shape:?} carry no token scores"
            )));
        }

        // Only the final position predicts the next token; taking the global
        // argmax over the whole tensor would return a position from the middle
        // of the prompt.
        let flat: Vec<f32> = arr.iter().copied().collect();
        let last_row = &flat[(seq_len - 1) * vocab_size..seq_len * vocab_size];
        let best = last_row
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(idx, _)| idx as u32)
            .ok_or_else(|| {
                TrustformersError::shape_error("the final logit row is empty".to_string())
            })?;
        Ok(best)
    }

    /// Autoregressively generate `max_new_tokens` continuation tokens.
    ///
    /// Each step appends the greedy argmax of the last position and feeds the
    /// extended sequence back in. Generation stops early at `eos_token_id`.
    /// Returns the **generated** ids only, not the prompt.
    ///
    /// # Errors
    ///
    /// Fails when the prompt is empty, when `max_new_tokens` is 0, or when a
    /// forward pass fails.
    pub fn generate_tokens(
        &self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
    ) -> Result<Vec<u32>> {
        if prompt_ids.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "cannot generate from an empty prompt".to_string(),
            ));
        }
        if max_new_tokens == 0 {
            return Err(TrustformersError::invalid_input_simple(
                "max_new_tokens must be greater than 0".to_string(),
            ));
        }

        let mut sequence = prompt_ids;
        let mut generated = Vec::with_capacity(max_new_tokens);
        for _ in 0..max_new_tokens {
            let output = self.forward(sequence.clone())?;
            let next = self.greedy_next_token(&output.logits)?;
            generated.push(next);
            if Some(next) == eos_token_id {
                break;
            }
            sequence.push(next);
        }
        Ok(generated)
    }

    /// Generate code as text, decoding through a real tokenizer.
    ///
    /// # Errors
    ///
    /// Fails when generation fails or when the tokenizer cannot decode the
    /// produced ids.
    ///
    /// # What this replaces
    ///
    /// A previous revision ran a *single* forward pass, took one argmax and
    /// returned the formatted string `"# generated code (next_token=1234)"` —
    /// a debug rendering of a token id, presented as generated source code. It
    /// never looped, never consulted a tokenizer, and produced the same shape of
    /// output for every prompt. Callers now have to supply the tokenizer that
    /// turns ids into text, because that is the only way text can actually be
    /// produced.
    pub fn generate_code<T: Tokenizer + ?Sized>(
        &self,
        prompt_ids: Vec<u32>,
        max_new_tokens: usize,
        eos_token_id: Option<u32>,
        tokenizer: &T,
    ) -> Result<String> {
        let generated = self.generate_tokens(prompt_ids, max_new_tokens, eos_token_id)?;
        tokenizer.decode(&generated)
    }
}
