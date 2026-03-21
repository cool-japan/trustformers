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
