use crate::phi2::config::Phi2Config;
use crate::phi2::model::Phi2ForCausalLM;
use trustformers_core::errors::Result;
use trustformers_core::tensor::Tensor;

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

    /// Greedily select the most-probable next token from the last position
    /// of the logit tensor.
    ///
    /// This is a simplified placeholder — a production implementation would use
    /// a proper sampler with temperature, top-p, and repetition penalties.
    pub fn greedy_next_token(&self, logits: &Tensor) -> Result<u32> {
        match logits {
            Tensor::F32(arr) => {
                let flat: Vec<f32> = arr.iter().copied().collect();
                let best = flat
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                    .map(|(idx, _)| idx as u32)
                    .unwrap_or(0);
                Ok(best)
            },
            _ => Ok(0),
        }
    }

    /// Stub code-generation entry point.
    ///
    /// In a fully integrated system this would run an autoregressive loop,
    /// decode token IDs through a tokenizer and return the generated string.
    /// Here we perform one forward pass and return a placeholder description
    /// so that the test surface covers the full call stack.
    pub fn generate_code(&self, prompt_ids: Vec<u32>) -> Result<String> {
        let output = self.forward(prompt_ids)?;
        let next_token = self.greedy_next_token(&output.logits)?;
        // Real implementation: decode token IDs → UTF-8 string
        Ok(format!("# generated code (next_token={next_token})"))
    }
}
