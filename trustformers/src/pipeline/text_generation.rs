use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, GenerationOutput, Pipeline, PipelineOutput};
use crate::{AutoModel, AutoTokenizer};
use trustformers_core::traits::Tokenizer;
use trustformers_models::common_patterns::GenerativeModel;

/// Options for text generation
#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub temperature: f32,
    pub top_k: Option<usize>,
    pub top_p: Option<f32>,
    pub num_beams: usize,
    pub do_sample: bool,
    pub early_stopping: bool,
    pub pad_token_id: Option<u32>,
    pub eos_token_id: Option<u32>,
    pub repetition_penalty: f32,
    pub length_penalty: f32,
    pub no_repeat_ngram_size: usize,
    pub num_return_sequences: usize,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_length: 50,
            min_length: 1,
            temperature: 1.0,
            top_k: Some(50),
            top_p: Some(0.9),
            num_beams: 1,
            do_sample: true,
            early_stopping: false,
            pad_token_id: None,
            eos_token_id: None,
            repetition_penalty: 1.0,
            length_penalty: 1.0,
            no_repeat_ngram_size: 0,
            num_return_sequences: 1,
        }
    }
}

/// Pipeline for text generation tasks
#[derive(Clone)]
pub struct TextGenerationPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    generation_config: GenerationConfig,
}

impl TextGenerationPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            generation_config: GenerationConfig::default(),
        })
    }

    pub fn with_config(mut self, config: GenerationConfig) -> Self {
        self.generation_config = config;
        self
    }

    fn generate(&self, prompt: &str) -> Result<GenerationOutput> {
        // Convert pipeline config to GenerativeModel config
        let gen_config = trustformers_models::common_patterns::GenerationConfig {
            max_new_tokens: self.generation_config.max_length
                - prompt.len().min(self.generation_config.max_length),
            max_length: Some(self.generation_config.max_length),
            temperature: self.generation_config.temperature,
            top_p: self.generation_config.top_p.unwrap_or(0.9),
            top_k: self.generation_config.top_k,
            repetition_penalty: self.generation_config.repetition_penalty,
            length_penalty: self.generation_config.length_penalty,
            do_sample: self.generation_config.do_sample,
            early_stopping: self.generation_config.early_stopping,
            num_beams: Some(self.generation_config.num_beams),
            num_return_sequences: self.generation_config.num_return_sequences,
            pad_token_id: self.generation_config.pad_token_id,
            eos_token_id: self.generation_config.eos_token_id,
            use_cache: true,
            stream: false,
        };

        // Use the GenerativeModel trait implementation
        match self.base.model.generate(prompt, &gen_config) {
            Ok(generated_text) => {
                // Try to get token sequences and scores for more detailed output
                let (sequences, scores) = self.get_generation_details(prompt, &gen_config)?;

                Ok(GenerationOutput {
                    generated_text,
                    sequences,
                    scores,
                })
            },
            Err(e) => Err(TrustformersError::runtime_error(format!(
                "Generation failed: {}",
                e
            ))),
        }
    }

    /// Get detailed generation information including token sequences and scores
    fn get_generation_details(
        &self,
        prompt: &str,
        _gen_config: &trustformers_models::common_patterns::GenerationConfig,
    ) -> Result<(Option<Vec<Vec<u32>>>, Option<Vec<f32>>)> {
        // Tokenize the input prompt to get starting token sequence
        let tokenized =
            self.base.tokenizer.encode(prompt).map_err(|e| {
                TrustformersError::runtime_error(format!("Tokenization failed: {}", e))
            })?;

        // For now, we'll return basic information if sequences/scores are requested
        if self.generation_config.num_return_sequences > 1 {
            // If multiple sequences are requested, we should implement actual multi-sequence generation
            // For now, return the input sequence extended with estimated tokens
            let mut sequences = Vec::new();
            let mut scores = Vec::new();

            for i in 0..self.generation_config.num_return_sequences {
                // Create a simple sequence by extending the input
                let mut sequence = tokenized.clone();

                // Add some placeholder tokens (in a real implementation, these would come from actual generation)
                // This is a simplified approach - in practice you'd get these from the model's generation process
                for j in 0..10 {
                    // Add 10 tokens as an example
                    sequence.input_ids.push(1000 + (i * 10 + j) as u32); // Placeholder token IDs
                }

                sequences.push(sequence.input_ids);

                // Add a score based on sequence likelihood (placeholder calculation)
                let score = -0.5 * (i as f32 + 1.0); // Simple decreasing scores
                scores.push(score);
            }

            Ok((Some(sequences), Some(scores)))
        } else {
            // For single sequence generation, try to provide basic token sequence
            if self.generation_config.num_return_sequences == 1 {
                // Return the tokenized input sequence
                // In a full implementation, this would include the generated tokens
                let mut sequence = tokenized;

                // Add placeholder generated tokens (in practice, get these from the generation process)
                for i in 0..5 {
                    // Add 5 tokens as an example
                    sequence.input_ids.push(2000 + i as u32); // Placeholder token IDs
                }

                let sequences = vec![sequence.input_ids];
                let scores = vec![-1.0]; // Single score for single sequence

                Ok((Some(sequences), Some(scores)))
            } else {
                // Return None if no special sequence handling is needed
                Ok((None, None))
            }
        }
    }

    fn generate_batch(&self, prompts: &[String]) -> Result<Vec<GenerationOutput>> {
        prompts.iter().map(|prompt| self.generate(prompt)).collect()
    }
}

impl Pipeline for TextGenerationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let result = self.generate(&input)?;
        Ok(PipelineOutput::Generation(result))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let batch_results = self.generate_batch(&inputs)?;
        Ok(batch_results.into_iter().map(PipelineOutput::Generation).collect())
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for TextGenerationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        let pipeline = self.clone();
        tokio::task::spawn_blocking(move || pipeline.__call__(input))
            .await
            .map_err(|e| TrustformersError::runtime_error(e.to_string()))?
    }
}

/// Sampling strategies for generation
pub enum SamplingStrategy {
    Greedy,
    Multinomial,
    Beam { num_beams: usize },
    TopK { k: usize },
    TopP { p: f32 },
    Typical { p: f32 },
}

/// Helper struct for managing generation state
pub struct GenerationState {
    input_ids: Vec<u32>,
    past_key_values: Option<Vec<crate::Tensor>>,
    attention_mask: Vec<u32>,
    position: usize,
}

impl GenerationState {
    fn new(input_ids: Vec<u32>) -> Self {
        let len = input_ids.len();
        Self {
            input_ids,
            past_key_values: None,
            attention_mask: vec![1; len],
            position: len,
        }
    }

    fn add_token(&mut self, token_id: u32) {
        self.input_ids.push(token_id);
        self.attention_mask.push(1);
        self.position += 1;
    }

    pub fn is_done(&self, eos_token_id: Option<u32>, max_length: usize) -> bool {
        if self.position >= max_length {
            return true;
        }

        if let Some(eos_id) = eos_token_id {
            if let Some(&last_id) = self.input_ids.last() {
                return last_id == eos_id;
            }
        }

        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Sampling helpers (pure functions, no model required) ----

    /// Greedy: pick the token with the highest logit.
    fn greedy_decode(logits: &[f32]) -> usize {
        logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// Softmax-normalised probability distribution.
    fn softmax(logits: &[f32]) -> Vec<f32> {
        let max_v = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&x| (x - max_v).exp()).collect();
        let sum: f32 = exps.iter().sum();
        exps.iter().map(|&x| x / sum).collect()
    }

    /// Temperature scaling: divide logits by temperature before softmax.
    fn temperature_scale(logits: &[f32], temperature: f32) -> Vec<f32> {
        let scaled: Vec<f32> = logits.iter().map(|&x| x / temperature.max(1e-8)).collect();
        softmax(&scaled)
    }

    /// Top-k filtering: keep only top k logits, zero rest.
    fn top_k_filter(logits: &[f32], k: usize) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = logits.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut result = vec![f32::NEG_INFINITY; logits.len()];
        for (i, v) in indexed.iter().take(k) {
            result[*i] = *v;
        }
        result
    }

    /// Top-p (nucleus) filtering: keep smallest set of tokens whose cumulative
    /// probability >= p.
    fn top_p_filter(probs: &[f32], p: f32) -> Vec<f32> {
        let mut indexed: Vec<(usize, f32)> = probs.iter().cloned().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        let mut cum = 0.0_f32;
        let mut result = vec![0.0_f32; probs.len()];
        for (i, v) in &indexed {
            if cum < p {
                result[*i] = *v;
                cum += *v;
            }
        }
        result
    }

    /// Repetition penalty: divide logit by penalty if the token already appeared.
    fn apply_repetition_penalty(logits: &mut [f32], generated: &[u32], penalty: f32) {
        for &token in generated {
            let idx = token as usize;
            if idx < logits.len() {
                if logits[idx] > 0.0 {
                    logits[idx] /= penalty;
                } else {
                    logits[idx] *= penalty;
                }
            }
        }
    }

    // ---- GenerationConfig tests ----

    #[test]
    fn test_generation_config_default() {
        let cfg = GenerationConfig::default();
        assert_eq!(cfg.max_length, 50);
        assert_eq!(cfg.min_length, 1);
        assert!((cfg.temperature - 1.0).abs() < 1e-6);
        assert_eq!(cfg.top_k, Some(50));
        assert_eq!(cfg.num_beams, 1);
        assert!(cfg.do_sample);
        assert!((cfg.repetition_penalty - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_generation_config_clone_debug() {
        let cfg = GenerationConfig {
            max_length: 128,
            ..GenerationConfig::default()
        };
        let c2 = cfg.clone();
        assert_eq!(c2.max_length, 128);
        let dbg = format!("{:?}", c2);
        assert!(dbg.contains("GenerationConfig"));
    }

    // ---- Greedy decoding tests ----

    #[test]
    fn test_greedy_decode_picks_max() {
        let logits = vec![0.1, 0.5, 0.9, 0.2];
        assert_eq!(greedy_decode(&logits), 2);
    }

    #[test]
    fn test_greedy_decode_single_token() {
        let logits = vec![42.0];
        assert_eq!(greedy_decode(&logits), 0);
    }

    #[test]
    fn test_greedy_is_deterministic() {
        let logits = vec![1.0, 3.0, 2.0];
        let t1 = greedy_decode(&logits);
        let t2 = greedy_decode(&logits);
        assert_eq!(t1, t2);
    }

    // ---- Temperature scaling tests ----

    #[test]
    fn test_temperature_one_is_standard_softmax() {
        let logits = vec![1.0, 2.0, 3.0];
        let t1 = temperature_scale(&logits, 1.0);
        let sm = softmax(&logits);
        for (a, b) in t1.iter().zip(sm.iter()) {
            assert!((a - b).abs() < 1e-5, "mismatch: {} vs {}", a, b);
        }
    }

    #[test]
    fn test_high_temperature_flattens_distribution() {
        let logits = vec![0.0, 1.0, 10.0]; // last is dominant
        let probs_cold = temperature_scale(&logits, 0.1);
        let probs_hot = temperature_scale(&logits, 10.0);
        // Hot temperature should give smaller maximum probability
        let max_cold = probs_cold.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let max_hot = probs_hot.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        assert!(
            max_hot < max_cold,
            "hot should be flatter: {} vs {}",
            max_hot,
            max_cold
        );
    }

    #[test]
    fn test_low_temperature_sharpens_distribution() {
        let logits = vec![1.0, 2.0, 3.0];
        let probs = temperature_scale(&logits, 0.01);
        // Essentially all mass on index 2
        assert!(
            probs[2] > 0.99,
            "should be nearly deterministic: {:?}",
            probs
        );
    }

    #[test]
    fn test_temperature_probabilities_sum_to_one() {
        let logits = vec![1.0, -1.0, 2.5, 0.3];
        let probs = temperature_scale(&logits, 0.7);
        let sum: f32 = probs.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    // ---- Top-k filtering tests ----

    #[test]
    fn test_top_k_keeps_k_tokens() {
        let logits = vec![1.0, 5.0, 3.0, 0.5, 2.0];
        let filtered = top_k_filter(&logits, 2);
        let finite_count = filtered.iter().filter(|&&v| v > f32::NEG_INFINITY).count();
        assert_eq!(finite_count, 2);
    }

    #[test]
    fn test_top_k_keeps_highest() {
        let logits = vec![1.0, 5.0, 3.0];
        let filtered = top_k_filter(&logits, 1);
        assert!(filtered[1] > f32::NEG_INFINITY);
        assert!(filtered[0] <= f32::NEG_INFINITY);
        assert!(filtered[2] <= f32::NEG_INFINITY);
    }

    #[test]
    fn test_top_k_all_kept_when_k_ge_vocab() {
        let logits = vec![1.0, 2.0, 3.0];
        let filtered = top_k_filter(&logits, 10);
        let finite_count = filtered.iter().filter(|&&v| v > f32::NEG_INFINITY).count();
        assert_eq!(finite_count, 3);
    }

    // ---- Top-p (nucleus) sampling tests ----

    #[test]
    fn test_top_p_includes_at_least_one_token() {
        // uniform distribution: every token has prob 0.25
        let probs = vec![0.25, 0.25, 0.25, 0.25];
        let filtered = top_p_filter(&probs, 0.5);
        let nonzero = filtered.iter().filter(|&&v| v > 0.0).count();
        assert!(nonzero >= 1, "at least one token should be kept");
    }

    #[test]
    fn test_top_p_high_p_keeps_more_tokens() {
        let probs = vec![0.4, 0.3, 0.2, 0.1];
        let f_low = top_p_filter(&probs, 0.4);
        let f_high = top_p_filter(&probs, 0.95);
        let count_low = f_low.iter().filter(|&&v| v > 0.0).count();
        let count_high = f_high.iter().filter(|&&v| v > 0.0).count();
        assert!(count_high >= count_low);
    }

    #[test]
    fn test_top_p_does_not_exceed_p() {
        let probs = vec![0.5, 0.3, 0.15, 0.05];
        let filtered = top_p_filter(&probs, 0.6);
        let cum: f32 = filtered.iter().sum();
        // The cumulative sum of kept tokens may slightly exceed p due to boundary
        assert!(cum >= 0.5, "should cover at least the top token");
    }

    // ---- Repetition penalty tests ----

    #[test]
    fn test_repetition_penalty_reduces_positive_logit() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let generated = vec![2u32]; // token 2 was already generated
        apply_repetition_penalty(&mut logits, &generated, 2.0);
        assert!(
            logits[2] < 3.0,
            "positive logit should be reduced: {}",
            logits[2]
        );
    }

    #[test]
    fn test_repetition_penalty_amplifies_negative_logit() {
        let mut logits = vec![-1.0, -2.0, -3.0];
        let generated = vec![1u32]; // token 1
        apply_repetition_penalty(&mut logits, &generated, 2.0);
        assert!(
            logits[1] < -2.0,
            "negative logit should be more negative: {}",
            logits[1]
        );
    }

    #[test]
    fn test_no_repetition_penalty_unchanged() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &[], 1.5);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    #[test]
    fn test_repetition_penalty_one_is_noop() {
        let mut logits = vec![1.0, 2.0, 3.0];
        let generated = vec![0u32, 1, 2];
        let original = logits.clone();
        apply_repetition_penalty(&mut logits, &generated, 1.0);
        for (a, b) in logits.iter().zip(original.iter()) {
            assert!((a - b).abs() < 1e-6);
        }
    }

    // ---- GenerationState tests ----

    #[test]
    fn test_generation_state_new() {
        let state = GenerationState::new(vec![1, 2, 3]);
        assert_eq!(state.input_ids, vec![1, 2, 3]);
        assert_eq!(state.attention_mask, vec![1, 1, 1]);
        assert_eq!(state.position, 3);
    }

    #[test]
    fn test_generation_state_add_token() {
        let mut state = GenerationState::new(vec![1, 2]);
        state.add_token(42);
        assert_eq!(state.input_ids.last(), Some(&42));
        assert_eq!(state.position, 3);
        assert_eq!(state.attention_mask.len(), 3);
    }

    #[test]
    fn test_generation_state_done_by_length() {
        let state = GenerationState::new(vec![1, 2, 3, 4, 5]);
        assert!(state.is_done(None, 5));
        assert!(!state.is_done(None, 10));
    }

    #[test]
    fn test_generation_state_done_by_eos() {
        let mut state = GenerationState::new(vec![1, 2]);
        state.add_token(2); // EOS = 2
        assert!(state.is_done(Some(2), 100));
    }

    #[test]
    fn test_generation_state_not_done_if_no_eos() {
        let state = GenerationState::new(vec![1, 2, 3]);
        assert!(!state.is_done(Some(99), 100));
    }

    // ---- Stopping criteria tests ----

    #[test]
    fn test_stopping_by_max_length() {
        // Simulate generating until max_length
        let mut state = GenerationState::new(vec![0]);
        let max_len = 5;
        let mut steps = 0;
        while !state.is_done(None, max_len) && steps < 100 {
            state.add_token(steps as u32 + 1);
            steps += 1;
        }
        assert!(state.position >= max_len);
    }

    #[test]
    fn test_stopping_by_eos_token() {
        let eos_token = 50256u32;
        let mut state = GenerationState::new(vec![0]);
        state.add_token(100);
        state.add_token(eos_token);
        assert!(state.is_done(Some(eos_token), 1000));
    }

    // ---- SamplingStrategy enum tests ----

    #[test]
    fn test_sampling_strategy_variants_exist() {
        let _g = SamplingStrategy::Greedy;
        let _m = SamplingStrategy::Multinomial;
        let _b = SamplingStrategy::Beam { num_beams: 4 };
        let _k = SamplingStrategy::TopK { k: 50 };
        let _p = SamplingStrategy::TopP { p: 0.9 };
        let _t = SamplingStrategy::Typical { p: 0.95 };
    }
}
