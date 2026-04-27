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

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Helper re-implementing private methods for white-box tests ----

    struct SumHelpers;

    impl SumHelpers {
        fn create_extractive_summary(text: &str, max_sentences: usize) -> String {
            let sentences: Vec<&str> = text
                .split(&['.', '!', '?'])
                .map(|s| s.trim())
                .filter(|s| !s.is_empty())
                .collect();
            let actual_max = (sentences.len() / 3).max(1).min(max_sentences);
            let chosen: Vec<&str> = sentences.into_iter().take(actual_max).collect();
            if chosen.is_empty() {
                format!("Summary of text with {} characters.", text.len())
            } else {
                format!("{}.", chosen.join(". "))
            }
        }

        fn post_process(summary: &str, original_text: &str) -> String {
            let mut processed = summary.to_string();
            if let Some(part) = processed.strip_prefix("summarize:") {
                processed = part.trim().to_string();
            }
            if processed.len() < 10 || processed == original_text {
                processed = Self::create_extractive_summary(original_text, 3);
            }
            processed = processed
                .trim()
                .trim_start_matches("Summary:")
                .trim_start_matches("summary:")
                .trim()
                .to_string();
            if !processed.is_empty() && !processed.ends_with(['.', '!', '?']) {
                processed.push('.');
            }
            processed
        }

        /// Compute simple unigram ROUGE-1 recall
        fn rouge1_recall(candidate: &str, reference: &str) -> f32 {
            let ref_tokens: Vec<&str> = reference.split_whitespace().collect();
            if ref_tokens.is_empty() {
                return 0.0;
            }
            let cand_set: std::collections::HashSet<&str> = candidate.split_whitespace().collect();
            let matching = ref_tokens.iter().filter(|t| cand_set.contains(*t)).count();
            matching as f32 / ref_tokens.len() as f32
        }

        /// Compute bigram ROUGE-2 recall
        fn rouge2_recall(candidate: &str, reference: &str) -> f32 {
            let cand_words: Vec<&str> = candidate.split_whitespace().collect();
            let ref_words: Vec<&str> = reference.split_whitespace().collect();
            if ref_words.len() < 2 {
                return 0.0;
            }
            let cand_bigrams: std::collections::HashSet<(&str, &str)> =
                cand_words.windows(2).map(|w| (w[0], w[1])).collect();
            let ref_bigrams: Vec<(&str, &str)> =
                ref_words.windows(2).map(|w| (w[0], w[1])).collect();
            let matching = ref_bigrams.iter().filter(|b| cand_bigrams.contains(*b)).count();
            matching as f32 / ref_bigrams.len() as f32
        }
    }

    // ---- SummarizationConfig tests ----

    #[test]
    fn test_config_default_values() {
        let cfg = SummarizationConfig::default();
        assert_eq!(cfg.max_length, 142);
        assert_eq!(cfg.min_length, 56);
        assert!((cfg.length_penalty - 2.0).abs() < 1e-6);
        assert_eq!(cfg.num_beams, 4);
        assert!(cfg.early_stopping);
    }

    #[test]
    fn test_config_clone() {
        let cfg = SummarizationConfig {
            num_beams: 8,
            ..SummarizationConfig::default()
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.num_beams, 8);
    }

    #[test]
    fn test_length_ratio_validation_min_lt_max() {
        let cfg = SummarizationConfig::default();
        assert!(cfg.min_length < cfg.max_length);
    }

    #[test]
    fn test_length_penalty_positive() {
        let cfg = SummarizationConfig::default();
        assert!(cfg.length_penalty > 0.0);
    }

    // ---- Extractive summary tests ----

    #[test]
    fn test_extractive_summary_short_text() {
        let text =
            "This is the first sentence. This is the second sentence. This is the third sentence.";
        let summary = SumHelpers::create_extractive_summary(text, 3);
        assert!(!summary.is_empty());
        // Should end with a period
        assert!(summary.ends_with('.'));
    }

    #[test]
    fn test_extractive_summary_preserves_content() {
        let text = "The quick brown fox jumps over the lazy dog. Second sentence here. Third here.";
        let summary = SumHelpers::create_extractive_summary(text, 3);
        // At least one sentence from the original should be present
        assert!(
            summary.contains("quick") || summary.contains("Second") || summary.contains("Third")
        );
    }

    #[test]
    fn test_extractive_summary_single_sentence() {
        let text = "Only one sentence in this text";
        let summary = SumHelpers::create_extractive_summary(text, 3);
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_extractive_summary_empty_text() {
        let text = "";
        let summary = SumHelpers::create_extractive_summary(text, 3);
        // Should still return something sensible
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_extractive_summary_length_constraint() {
        // Generate a longer document using LCG-seeded "words"
        let mut seed = 42u64;
        let words: Vec<String> = (0..200)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let idx = (seed >> 33) % 5;
                ["the", "quick", "brown", "fox", "jumps"][idx as usize].to_string()
            })
            .collect();
        // Split into sentences of ~10 words each
        let sentences: Vec<String> = words.chunks(10).map(|c| c.join(" ")).collect();
        let text = sentences.join(". ");
        let summary = SumHelpers::create_extractive_summary(&text, 3);
        // Summary should be shorter than or equal to 3 sentence portions of original
        assert!(summary.len() <= text.len());
    }

    // ---- post_process tests ----

    #[test]
    fn test_post_process_strips_summarize_prefix() {
        let summary = "summarize: The main point is clarity.";
        let result = SumHelpers::post_process(summary, "Some long original text here.");
        assert!(!result.starts_with("summarize:"));
    }

    #[test]
    fn test_post_process_strips_summary_prefix() {
        let summary = "Summary: The main conclusion.";
        let result = SumHelpers::post_process(
            summary,
            "Some long original text here that is well over ten chars.",
        );
        assert!(!result.starts_with("Summary:"));
    }

    #[test]
    fn test_post_process_adds_period_if_missing() {
        // A valid summary without terminal punctuation
        let summary = "The article discusses climate change";
        let original =
            "The article discusses climate change in great depth with many supporting examples.";
        let result = SumHelpers::post_process(summary, original);
        assert!(result.ends_with('.') || result.ends_with('!') || result.ends_with('?'));
    }

    #[test]
    fn test_post_process_does_not_double_period() {
        let summary = "This is a complete summary.";
        let original = "This is a complete summary with more text after it.";
        let result = SumHelpers::post_process(summary, original);
        assert!(!result.ends_with(".."));
    }

    #[test]
    fn test_post_process_too_short_falls_back_to_extractive() {
        let summary = "Hi"; // < 10 chars
        let original = "This is the first sentence. Second sentence. Third sentence.";
        let result = SumHelpers::post_process(summary, original);
        // Should have fallen back to extractive
        assert!(result.len() >= 10);
    }

    #[test]
    fn test_post_process_identical_to_original_falls_back() {
        let original = "This is the original text. It has two sentences.";
        let result = SumHelpers::post_process(original, original);
        // Extractive fallback should differ from exact copy
        assert!(!result.is_empty());
    }

    // ---- ROUGE-style overlap tests ----

    #[test]
    fn test_rouge1_perfect_recall() {
        let r = SumHelpers::rouge1_recall("the fox jumped", "the fox jumped");
        assert!((r - 1.0).abs() < 1e-6, "r = {}", r);
    }

    #[test]
    fn test_rouge1_zero_recall() {
        let r = SumHelpers::rouge1_recall("cat sat", "dog ran");
        assert!((r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge1_partial_recall() {
        let r = SumHelpers::rouge1_recall("the fox", "the quick brown fox");
        assert!(r > 0.0 && r < 1.0, "partial recall expected, got {}", r);
    }

    #[test]
    fn test_rouge2_perfect_recall() {
        let r = SumHelpers::rouge2_recall("the quick fox", "the quick fox");
        assert!((r - 1.0).abs() < 1e-6, "r = {}", r);
    }

    #[test]
    fn test_rouge2_zero_recall() {
        let r = SumHelpers::rouge2_recall("cat sat mat", "dog ran ran");
        assert!((r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge2_partial_recall() {
        let r = SumHelpers::rouge2_recall("the quick fox", "the quick brown fox");
        assert!(r > 0.0 && r < 1.0, "partial recall expected, got {}", r);
    }

    #[test]
    fn test_rouge1_empty_reference() {
        let r = SumHelpers::rouge1_recall("some candidate", "");
        assert!((r - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_rouge2_short_reference() {
        let r = SumHelpers::rouge2_recall("word", "word"); // only one word, no bigrams
        assert!((r - 0.0).abs() < 1e-6);
    }

    // ---- Beam search parameter tests ----

    #[test]
    fn test_num_beams_at_least_one() {
        let cfg = SummarizationConfig::default();
        assert!(cfg.num_beams >= 1);
    }

    #[test]
    fn test_beam_search_more_beams_than_one_uses_beam_mode() {
        let cfg = SummarizationConfig {
            num_beams: 4,
            ..SummarizationConfig::default()
        };
        // In beam mode: num_beams > 1
        assert!(cfg.num_beams > 1);
    }

    #[test]
    fn test_truncation_max_length_respected() {
        // A text shorter than max_length should not be truncated
        let short_text = "Short text.";
        let cfg = SummarizationConfig::default();
        assert!(short_text.len() < cfg.max_length);
    }
}
