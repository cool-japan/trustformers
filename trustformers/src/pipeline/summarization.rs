use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Pipeline, PipelineOutput};
use crate::{AutoModel, AutoTokenizer};
use trustformers_models::common_patterns::GenerativeModel;

/// Options for summarization
#[derive(Clone, Debug)]
pub struct SummarizationConfig {
    pub max_length: usize,
    pub min_length: usize,
    pub length_penalty: f32,
    pub num_beams: usize,
    pub early_stopping: bool,
}

impl Default for SummarizationConfig {
    fn default() -> Self {
        Self {
            max_length: 142,
            min_length: 56,
            length_penalty: 2.0,
            num_beams: 4,
            early_stopping: true,
        }
    }
}

/// Pipeline for text summarization tasks
#[derive(Clone)]
pub struct SummarizationPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    config: SummarizationConfig,
}

impl SummarizationPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            config: SummarizationConfig::default(),
        })
    }

    pub fn with_config(mut self, config: SummarizationConfig) -> Self {
        self.config = config;
        self
    }

    fn summarize(&self, text: &str) -> Result<String> {
        // Add summarization prefix for T5-style models
        let input_text = if self.is_t5_model() {
            format!("summarize: {}", text)
        } else {
            text.to_string()
        };

        // Create generation config optimized for summarization
        let gen_config = trustformers_models::common_patterns::GenerationConfig {
            max_new_tokens: self.config.max_length.min(150), // Summaries should be concise
            max_length: Some(self.config.max_length),
            temperature: 0.7, // Slightly more deterministic for summaries
            top_p: 0.9,
            top_k: Some(50),
            repetition_penalty: 1.2, // Discourage repetition in summaries
            length_penalty: 1.0,
            do_sample: true,
            early_stopping: true,
            num_beams: Some(4), // Use beam search for better quality
            num_return_sequences: 1,
            pad_token_id: None,
            eos_token_id: None,
            use_cache: true,
            stream: false,
        };

        // Use the GenerativeModel trait
        match self.base.model.generate(&input_text, &gen_config) {
            Ok(summary) => {
                // Post-process the summary
                let processed_summary = self.post_process_summary(&summary, text);
                Ok(processed_summary)
            },
            Err(e) => Err(TrustformersError::pipeline(
                format!("Summarization failed: {}", e),
                "summarization",
            )),
        }
    }

    fn summarize_batch(&self, texts: &[String]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.summarize(text)).collect()
    }

    fn is_t5_model(&self) -> bool {
        match &self.base.model.model_type {
            #[cfg(feature = "t5")]
            crate::automodel::AutoModelType::T5(_)
            | crate::automodel::AutoModelType::T5ForConditionalGeneration(_) => true,
            _ => false,
        }
    }

    fn post_process_summary(&self, summary: &str, original_text: &str) -> String {
        let mut processed = summary.to_string();

        // Remove the original prompt prefix if it exists
        if let Some(summary_part) = processed.strip_prefix("summarize:") {
            processed = summary_part.trim().to_string();
        }

        // If the summary is too short or seems incomplete, provide a basic extractive summary
        if processed.len() < 10 || processed == original_text {
            processed = self.create_extractive_summary(original_text);
        }

        // Clean up common generation artifacts
        processed = processed
            .trim()
            .trim_start_matches("Summary:")
            .trim_start_matches("summary:")
            .trim()
            .to_string();

        // Ensure the summary ends with proper punctuation
        if !processed.is_empty() && !processed.ends_with(['.', '!', '?']) {
            processed.push('.');
        }

        processed
    }

    fn create_extractive_summary(&self, text: &str) -> String {
        // Simple extractive summarization: take the first few sentences
        let sentences: Vec<&str> = text
            .split(&['.', '!', '?'])
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .collect();

        let max_sentences = (sentences.len() / 3).max(1).min(3);
        let summary_sentences: Vec<&str> = sentences.into_iter().take(max_sentences).collect();

        if summary_sentences.is_empty() {
            format!("Summary of text with {} characters.", text.len())
        } else {
            format!("{}.", summary_sentences.join(". "))
        }
    }
}

impl Pipeline for SummarizationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let summary = self.summarize(&input)?;
        Ok(PipelineOutput::Summarization(summary))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let summaries = self.summarize_batch(&inputs)?;
        Ok(summaries.into_iter().map(PipelineOutput::Summarization).collect())
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for SummarizationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        let pipeline = self.clone();
        tokio::task::spawn_blocking(move || pipeline.__call__(input))
            .await
            .map_err(|e| TrustformersError::pipeline(e.to_string(), "summarization"))?
    }
}
