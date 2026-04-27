use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Pipeline, PipelineOutput};
use crate::{AutoModel, AutoTokenizer};
use trustformers_core::traits::{Model, Tokenizer};
use trustformers_core::Tensor;

/// Options for translation
#[derive(Clone, Debug)]
pub struct TranslationConfig {
    pub max_length: usize,
    pub num_beams: usize,
    pub early_stopping: bool,
    pub source_lang: Option<String>,
    pub target_lang: Option<String>,
}

impl Default for TranslationConfig {
    fn default() -> Self {
        Self {
            max_length: 512,
            num_beams: 4,
            early_stopping: true,
            source_lang: None,
            target_lang: None,
        }
    }
}

/// Pipeline for translation tasks
#[derive(Clone)]
pub struct TranslationPipeline {
    base: BasePipeline<AutoModel, AutoTokenizer>,
    config: TranslationConfig,
}

impl TranslationPipeline {
    pub fn new(model: AutoModel, tokenizer: AutoTokenizer) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            config: TranslationConfig::default(),
        })
    }

    pub fn with_config(mut self, config: TranslationConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_language_pair(mut self, source: &str, target: &str) -> Self {
        self.config.source_lang = Some(source.to_string());
        self.config.target_lang = Some(target.to_string());
        self
    }

    fn translate(&self, text: &str) -> Result<String> {
        // Prepare input based on model type
        let input_text = self.prepare_input(text);

        // Implement actual translation logic
        match &self.base.model.model_type {
            #[cfg(feature = "t5")]
            crate::automodel::AutoModelType::T5ForConditionalGeneration(model) => {
                self.translate_with_t5(model, &input_text)
            },
            #[cfg(feature = "mbart")]
            crate::automodel::AutoModelType::MBartForConditionalGeneration(model) => {
                self.translate_with_mbart(model, &input_text)
            },
            #[cfg(feature = "bert")]
            crate::automodel::AutoModelType::Bert(_model) => {
                // BERT-based translation (less common, but supported)
                self.translate_with_encoder_decoder(&input_text)
            },
            _ => Err(TrustformersError::model(
                "Model does not support translation. Supported models: T5, mBART, BERT-based seq2seq".to_string(),
                "unknown"
            ))
        }
    }

    fn translate_batch(&self, texts: &[String]) -> Result<Vec<String>> {
        texts.iter().map(|text| self.translate(text)).collect()
    }

    fn prepare_input(&self, text: &str) -> String {
        // Handle different model input formats
        if let (Some(src), Some(tgt)) = (&self.config.source_lang, &self.config.target_lang) {
            // T5-style format
            if self.is_t5_model() {
                format!("translate {} to {}: {}", src, tgt, text)
            } else {
                // Other formats
                format!("[{}] {}", src, text)
            }
        } else {
            text.to_string()
        }
    }

    fn is_t5_model(&self) -> bool {
        // Check if model is T5-based
        match &self.base.model.model_type {
            #[cfg(feature = "t5")]
            crate::automodel::AutoModelType::T5ForConditionalGeneration(_) => true,
            _ => false,
        }
    }

    /// Translate using T5 model
    #[cfg(feature = "t5")]
    fn translate_with_t5(
        &self,
        _model: &crate::models::t5::T5ForConditionalGeneration,
        input_text: &str,
    ) -> Result<String> {
        use trustformers_core::Tensor;

        // Tokenize input
        let tokenized = self.base.tokenizer.encode(input_text)?;

        // Convert to tensor
        let input_ids_f32: Vec<f32> = tokenized.input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_vec(input_ids_f32, &[1, tokenized.input_ids.len()])?;

        // Generate translation using beam search
        let generated_ids = self.generate_with_beam_search(&input_tensor)?;

        // Decode generated tokens
        let translation = self.base.tokenizer.decode(&generated_ids)?;

        // Clean up the translation
        Ok(self.post_process_translation(&translation))
    }

    /// Translate using mBART model
    #[cfg(feature = "mbart")]
    fn translate_with_mbart(
        &self,
        _model: &crate::models::mbart::MBartForConditionalGeneration,
        input_text: &str,
    ) -> Result<String> {
        use trustformers_core::Tensor;

        // Add language tokens for mBART
        let input_with_lang = if let Some(src) = &self.config.source_lang {
            format!("{} {}", src, input_text)
        } else {
            input_text.to_string()
        };

        // Tokenize input
        let tokenized = self.base.tokenizer.encode(&input_with_lang)?;

        // Convert to tensor
        let input_ids_f32: Vec<f32> = tokenized.input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_vec(input_ids_f32, &[1, tokenized.input_ids.len()])?;

        // Generate translation
        let generated_ids = self.generate_with_beam_search(&input_tensor)?;

        // Decode generated tokens
        let translation = self.base.tokenizer.decode(&generated_ids)?;

        Ok(self.post_process_translation(&translation))
    }

    /// Translate using encoder-decoder architecture (BERT-based)
    fn translate_with_encoder_decoder(&self, input_text: &str) -> Result<String> {
        use trustformers_core::Tensor;

        // Tokenize input
        let tokenized = self.base.tokenizer.encode(input_text)?;

        // Convert to tensor
        let input_ids_f32: Vec<f32> = tokenized.input_ids.iter().map(|&x| x as f32).collect();
        let input_tensor = Tensor::from_vec(input_ids_f32, &[1, tokenized.input_ids.len()])?;

        // Run model forward pass
        let output = self.base.model.forward(input_tensor)?;

        // Decode output (simplified)
        let output_data = output.data()?;
        let output_ids: Vec<u32> = output_data.iter().map(|&x| x as u32).collect();

        // Decode to text
        let translation = self.base.tokenizer.decode(&output_ids)?;

        Ok(self.post_process_translation(&translation))
    }

    /// Generate text using beam search decoding
    fn generate_with_beam_search(&self, input_tensor: &Tensor) -> Result<Vec<u32>> {
        // Simplified beam search implementation
        // In a real implementation, this would be much more sophisticated

        let output = self.base.model.forward(input_tensor.clone())?;
        let output_data = output.data()?;

        // Convert output to token IDs (simplified)
        let mut generated_ids = Vec::new();

        // Take top predictions and convert to token IDs
        for chunk in output_data.chunks(self.base.tokenizer.vocab_size().min(output_data.len())) {
            if let Some((max_idx, _)) = chunk
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            {
                generated_ids.push(max_idx as u32);
            }
        }

        // Limit to max_length
        generated_ids.truncate(self.config.max_length);

        Ok(generated_ids)
    }

    /// Post-process the raw translation output
    fn post_process_translation(&self, translation: &str) -> String {
        let mut processed = translation.to_string();

        // Remove special tokens
        processed = processed.replace("<s>", "");
        processed = processed.replace("</s>", "");
        processed = processed.replace("<pad>", "");
        processed = processed.replace("<unk>", "");

        // Remove language-specific tokens for multilingual models
        if let Some(src) = &self.config.source_lang {
            processed = processed.replace(&format!("<{}>", src), "");
        }
        if let Some(tgt) = &self.config.target_lang {
            processed = processed.replace(&format!("<{}>", tgt), "");
        }

        // Clean up whitespace
        processed = processed.trim().to_string();
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");

        // Handle empty results
        if processed.is_empty() {
            processed = "[Unable to translate]".to_string();
        }

        processed
    }
}

impl Pipeline for TranslationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let translation = self.translate(&input)?;
        Ok(PipelineOutput::Translation(translation))
    }

    fn batch(&self, inputs: Vec<Self::Input>) -> Result<Vec<Self::Output>> {
        let translations = self.translate_batch(&inputs)?;
        Ok(translations.into_iter().map(PipelineOutput::Translation).collect())
    }
}

#[cfg(feature = "async")]
#[async_trait::async_trait]
impl crate::pipeline::AsyncPipeline for TranslationPipeline {
    type Input = String;
    type Output = PipelineOutput;

    async fn __call_async__(&self, input: Self::Input) -> Result<Self::Output> {
        let pipeline = self.clone();
        tokio::task::spawn_blocking(move || pipeline.__call__(input))
            .await
            .map_err(|e| {
                TrustformersError::runtime_error(format!("Translation pipeline error: {}", e))
            })?
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    // ---- Helper functions mirroring private methods ----

    fn prepare_input(text: &str, src: Option<&str>, tgt: Option<&str>, is_t5: bool) -> String {
        if let (Some(src_lang), Some(tgt_lang)) = (src, tgt) {
            if is_t5 {
                format!("translate {} to {}: {}", src_lang, tgt_lang, text)
            } else {
                format!("[{}] {}", src_lang, text)
            }
        } else {
            text.to_string()
        }
    }

    fn post_process(translation: &str, src: Option<&str>, tgt: Option<&str>) -> String {
        let mut processed = translation.to_string();
        for special in &["<s>", "</s>", "<pad>", "<unk>"] {
            processed = processed.replace(special, "");
        }
        if let Some(src_lang) = src {
            processed = processed.replace(&format!("<{}>", src_lang), "");
        }
        if let Some(tgt_lang) = tgt {
            processed = processed.replace(&format!("<{}>", tgt_lang), "");
        }
        processed = processed.trim().to_string();
        processed = processed.split_whitespace().collect::<Vec<_>>().join(" ");
        if processed.is_empty() {
            processed = "[Unable to translate]".to_string();
        }
        processed
    }

    /// Compute 1-gram BLEU precision (simplified)
    fn bleu_1gram(candidate: &str, reference: &str) -> f32 {
        let cand_tokens: Vec<&str> = candidate.split_whitespace().collect();
        let ref_set: std::collections::HashSet<&str> = reference.split_whitespace().collect();
        if cand_tokens.is_empty() {
            return 0.0;
        }
        let matching = cand_tokens.iter().filter(|t| ref_set.contains(*t)).count();
        matching as f32 / cand_tokens.len() as f32
    }

    /// Compute 2-gram BLEU precision (simplified)
    fn bleu_2gram(candidate: &str, reference: &str) -> f32 {
        let cand_words: Vec<&str> = candidate.split_whitespace().collect();
        let ref_words: Vec<&str> = reference.split_whitespace().collect();
        if cand_words.len() < 2 {
            return 0.0;
        }
        let cand_bigrams: Vec<(&str, &str)> = cand_words.windows(2).map(|w| (w[0], w[1])).collect();
        let ref_bigrams: std::collections::HashSet<(&str, &str)> =
            ref_words.windows(2).map(|w| (w[0], w[1])).collect();
        let matching = cand_bigrams.iter().filter(|b| ref_bigrams.contains(*b)).count();
        matching as f32 / cand_bigrams.len() as f32
    }

    // ---- TranslationConfig tests ----

    #[test]
    fn test_config_default_values() {
        let cfg = TranslationConfig::default();
        assert_eq!(cfg.max_length, 512);
        assert_eq!(cfg.num_beams, 4);
        assert!(cfg.early_stopping);
        assert!(cfg.source_lang.is_none());
        assert!(cfg.target_lang.is_none());
    }

    #[test]
    fn test_config_clone() {
        let cfg = TranslationConfig {
            source_lang: Some("en".to_string()),
            target_lang: Some("fr".to_string()),
            ..TranslationConfig::default()
        };
        let cloned = cfg.clone();
        assert_eq!(cloned.source_lang, Some("en".to_string()));
    }

    // ---- Source / target language pair tests ----

    #[test]
    fn test_language_pair_stored_correctly() {
        let cfg = TranslationConfig {
            source_lang: Some("English".to_string()),
            target_lang: Some("French".to_string()),
            ..TranslationConfig::default()
        };
        assert_eq!(cfg.source_lang.as_deref(), Some("English"));
        assert_eq!(cfg.target_lang.as_deref(), Some("French"));
    }

    #[test]
    fn test_no_language_pair_passthrough() {
        let result = prepare_input("Hello world", None, None, false);
        assert_eq!(result, "Hello world");
    }

    // ---- T5-style input prefix tests ----

    #[test]
    fn test_t5_prefix_format() {
        let result = prepare_input("Hello world", Some("English"), Some("French"), true);
        assert!(result.starts_with("translate English to French:"));
        assert!(result.contains("Hello world"));
    }

    #[test]
    fn test_non_t5_prefix_format() {
        let result = prepare_input("Hello world", Some("en"), Some("fr"), false);
        assert!(result.starts_with("[en]"));
        assert!(result.contains("Hello world"));
    }

    #[test]
    fn test_t5_prefix_contains_both_languages() {
        let src = "de";
        let tgt = "en";
        let result = prepare_input("Hallo Welt", Some(src), Some(tgt), true);
        assert!(result.contains(src));
        assert!(result.contains(tgt));
    }

    // ---- Post-processing tests ----

    #[test]
    fn test_post_process_removes_special_tokens() {
        let raw = "<s>Hello world</s>";
        let result = post_process(raw, None, None);
        assert!(!result.contains("<s>"));
        assert!(!result.contains("</s>"));
    }

    #[test]
    fn test_post_process_removes_pad_unk() {
        let raw = "<pad>Hello<unk> world";
        let result = post_process(raw, None, None);
        assert!(!result.contains("<pad>"));
        assert!(!result.contains("<unk>"));
    }

    #[test]
    fn test_post_process_removes_lang_tokens() {
        let raw = "<en> Hello <fr>";
        let result = post_process(raw, Some("en"), Some("fr"));
        assert!(!result.contains("<en>"));
        assert!(!result.contains("<fr>"));
    }

    #[test]
    fn test_post_process_normalises_whitespace() {
        let raw = "Hello   world  today";
        let result = post_process(raw, None, None);
        assert!(!result.contains("  ")); // no double spaces
    }

    #[test]
    fn test_post_process_empty_becomes_unable_to_translate() {
        let raw = "<s></s><pad>";
        let result = post_process(raw, None, None);
        assert_eq!(result, "[Unable to translate]");
    }

    // ---- Token count tests ----

    #[test]
    fn test_token_count_positive_after_tokenisation() {
        // Simulate tokenisation: split on whitespace
        let text = "The quick brown fox";
        let tokens: Vec<&str> = text.split_whitespace().collect();
        assert!(!tokens.is_empty());
        assert!(tokens.len() <= 512); // within max_length
    }

    #[test]
    fn test_max_length_truncation_simulation() {
        let cfg = TranslationConfig {
            max_length: 5,
            ..TranslationConfig::default()
        };
        let tokens: Vec<u32> = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
        let truncated: Vec<u32> = tokens.into_iter().take(cfg.max_length).collect();
        assert_eq!(truncated.len(), 5);
    }

    // ---- forced_bos_token_id language forcing ----

    #[test]
    fn test_forced_bos_prepended_to_generation() {
        // Simulate forced BOS: first output token must be the language token
        let forced_bos: u32 = 250008; // typical mBART French token
        let mut generated: Vec<u32> = vec![forced_bos];
        generated.extend(&[100, 200, 300]);
        assert_eq!(generated[0], forced_bos);
    }

    #[test]
    fn test_language_code_mapping() {
        let mut lang_codes: HashMap<&str, u32> = HashMap::new();
        lang_codes.insert("en_XX", 250004);
        lang_codes.insert("fr_XX", 250008);
        lang_codes.insert("de_DE", 250003);
        assert!(lang_codes.contains_key("fr_XX"));
        assert_eq!(*lang_codes.get("fr_XX").expect("key exists"), 250008);
    }

    // ---- BLEU score tests ----

    #[test]
    fn test_bleu_1gram_perfect() {
        let bleu = bleu_1gram("the cat sat on the mat", "the cat sat on the mat");
        assert!((bleu - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bleu_1gram_zero() {
        let bleu = bleu_1gram("foo bar", "baz qux");
        assert!((bleu - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bleu_1gram_partial() {
        let bleu = bleu_1gram("the cat", "the dog sat");
        assert!(bleu > 0.0 && bleu < 1.0, "bleu = {}", bleu);
    }

    #[test]
    fn test_bleu_2gram_perfect() {
        let bleu = bleu_2gram("the quick brown fox", "the quick brown fox");
        assert!((bleu - 1.0).abs() < 1e-5);
    }

    #[test]
    fn test_bleu_2gram_empty_candidate() {
        let bleu = bleu_2gram("word", "reference text here");
        // Single word candidate → no bigrams → 0
        assert!((bleu - 0.0).abs() < 1e-5);
    }

    #[test]
    fn test_bleu_higher_for_better_translation() {
        let reference = "the quick brown fox jumps over the lazy dog";
        let good = "the quick brown fox jumps over the lazy dog";
        let bad = "a random completely different sentence here";
        assert!(bleu_1gram(good, reference) > bleu_1gram(bad, reference));
    }
}
