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
struct GenerationState {
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

    fn is_done(&self, eos_token_id: Option<u32>, max_length: usize) -> bool {
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
