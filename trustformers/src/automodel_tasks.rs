use crate::core::traits::Model;
use crate::error::{Result, TrustformersError};
use crate::pipeline::audio_classification::{
    AudioClassificationConfig, AudioClassificationInput, AudioClassificationPipeline,
    AudioClassificationResult,
};
use crate::pipeline::image_classification::{
    ImageClassificationConfig, ImageClassificationInput, ImageClassificationPipeline,
    ImageClassificationResult,
};
use std::path::Path;
use trustformers_core::errors::TrustformersError as CoreTrustformersError;

#[derive(Clone)]
pub enum AutoModelForSequenceClassification {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForSequenceClassification),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForSequenceClassification),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForSequenceClassification),
}

impl AutoModelForSequenceClassification {
    pub fn from_config(config: crate::automodel::AutoConfig, num_labels: usize) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForSequenceClassification::Bert(
                    crate::models::bert::BertForSequenceClassification::new(
                        bert_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForSequenceClassification::Roberta(
                    crate::models::roberta::RobertaForSequenceClassification::new(
                        roberta_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForSequenceClassification::Albert(
                    crate::models::albert::AlbertForSequenceClassification::new(
                        albert_config,
                        num_labels,
                    )?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support sequence classification".into(),
                ),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str, num_labels: usize) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config, num_labels)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "bert")]
                AutoModelForSequenceClassification::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForSequenceClassification::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForSequenceClassification::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForTokenClassification {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForTokenClassification),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForTokenClassification),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForTokenClassification),
}

impl AutoModelForTokenClassification {
    pub fn from_config(config: crate::automodel::AutoConfig, num_labels: usize) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForTokenClassification::Bert(
                    crate::models::bert::BertForTokenClassification::new(bert_config, num_labels)?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForTokenClassification::Roberta(
                    crate::models::roberta::RobertaForTokenClassification::new(
                        roberta_config,
                        num_labels,
                    )?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForTokenClassification::Albert(
                    crate::models::albert::AlbertForTokenClassification::new(
                        albert_config,
                        num_labels,
                    )?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support token classification".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained(model_name_or_path: &str, num_labels: usize) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, num_labels, None)
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        num_labels: usize,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained_with_revision(
            model_name_or_path,
            revision,
        )?;
        let mut model = Self::from_config(config, num_labels)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForTokenClassification::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForTokenClassification::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForTokenClassification::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForQuestionAnswering {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForQuestionAnswering),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForQuestionAnswering),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForQuestionAnswering),
}

impl AutoModelForQuestionAnswering {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => {
                Ok(AutoModelForQuestionAnswering::Bert(
                    crate::models::bert::BertForQuestionAnswering::new(bert_config)?,
                ))
            },
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForQuestionAnswering::Roberta(
                    crate::models::roberta::RobertaForQuestionAnswering::new(roberta_config)?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForQuestionAnswering::Albert(
                    crate::models::albert::AlbertForQuestionAnswering::new(albert_config)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support question answering".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        Self::from_pretrained_with_revision(model_name_or_path, None)
    }

    #[cfg(feature = "bert")]
    pub fn from_pretrained_with_revision(
        model_name_or_path: &str,
        revision: Option<&str>,
    ) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained_with_revision(
            model_name_or_path,
            revision,
        )?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForQuestionAnswering::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForQuestionAnswering::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForQuestionAnswering::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForCausalLM {
    #[cfg(feature = "gpt2")]
    Gpt2(crate::models::gpt2::Gpt2LMHeadModel),
    #[cfg(feature = "gpt_neo")]
    GptNeo(crate::models::gpt_neo::GptNeoLMHeadModel),
    #[cfg(feature = "gpt_j")]
    GptJ(crate::models::gpt_j::GptJLMHeadModel),
}

impl AutoModelForCausalLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "gpt2")]
            crate::automodel::AutoConfig::Gpt2(gpt2_config) => Ok(AutoModelForCausalLM::Gpt2(
                crate::models::gpt2::Gpt2LMHeadModel::new(gpt2_config)?,
            )),
            #[cfg(feature = "gpt_neo")]
            crate::automodel::AutoConfig::GptNeo(gpt_neo_config) => {
                Ok(AutoModelForCausalLM::GptNeo(
                    crate::models::gpt_neo::GptNeoLMHeadModel::new(gpt_neo_config)?,
                ))
            },
            #[cfg(feature = "gpt_j")]
            crate::automodel::AutoConfig::GptJ(gpt_j_config) => Ok(AutoModelForCausalLM::GptJ(
                crate::models::gpt_j::GptJLMHeadModel::new(gpt_j_config)?,
            )),
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support causal language modeling".into(),
                ),
            )),
        }
    }

    pub fn generate(
        &mut self,
        inputs: crate::core::traits::TokenizedInput,
        generation_config: crate::pipeline::text_generation::GenerationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            #[cfg(feature = "gpt2")]
            AutoModelForCausalLM::Gpt2(gpt2) => gpt2
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(feature = "gpt_neo")]
            AutoModelForCausalLM::GptNeo(gpt_neo) => gpt_neo
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(feature = "gpt_j")]
            AutoModelForCausalLM::GptJ(gpt_j) => gpt_j
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.temperature,
                    generation_config.top_k,
                    generation_config.top_p,
                )
                .map_err(Into::into),
            #[cfg(not(any(feature = "gpt2", feature = "gpt_neo", feature = "gpt_j")))]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error("No causal LM models available".into()),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "gpt2")]
                AutoModelForCausalLM::Gpt2(gpt2) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt2.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_neo")]
                AutoModelForCausalLM::GptNeo(gpt_neo) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_neo.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "gpt_j")]
                AutoModelForCausalLM::GptJ(gpt_j) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    gpt_j.load_pretrained(&mut reader)?;
                },
                #[cfg(not(any(feature = "gpt2", feature = "gpt_neo", feature = "gpt_j")))]
                _ => {},
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForSeq2SeqLM {
    #[cfg(feature = "t5")]
    T5(crate::models::t5::T5ForConditionalGeneration),
}

impl AutoModelForSeq2SeqLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "t5")]
            crate::automodel::AutoConfig::T5(t5_config) => Ok(AutoModelForSeq2SeqLM::T5(
                crate::models::t5::T5ForConditionalGeneration::new(t5_config)?,
            )),
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support seq2seq language modeling".into(),
                ),
            )),
        }
    }

    #[cfg(feature = "t5")]
    pub fn generate(
        &mut self,
        inputs: crate::core::traits::TokenizedInput,
        generation_config: crate::pipeline::text_generation::GenerationConfig,
    ) -> Result<Vec<u32>> {
        match self {
            AutoModelForSeq2SeqLM::T5(t5) => t5
                .generate(
                    inputs.input_ids,
                    generation_config.max_length,
                    generation_config.num_beams,
                )
                .map_err(Into::into),
        }
    }

    #[cfg(feature = "t5")]
    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                AutoModelForSeq2SeqLM::T5(t5) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    t5.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

#[derive(Clone)]
pub enum AutoModelForMaskedLM {
    #[cfg(feature = "bert")]
    Bert(crate::models::bert::BertForMaskedLM),
    #[cfg(feature = "roberta")]
    Roberta(crate::models::roberta::RobertaForMaskedLM),
    #[cfg(feature = "albert")]
    Albert(crate::models::albert::AlbertForMaskedLM),
}

impl AutoModelForMaskedLM {
    pub fn from_config(config: crate::automodel::AutoConfig) -> Result<Self> {
        match config {
            #[cfg(feature = "bert")]
            crate::automodel::AutoConfig::Bert(bert_config) => Ok(AutoModelForMaskedLM::Bert(
                crate::models::bert::BertForMaskedLM::new(bert_config)?,
            )),
            #[cfg(feature = "roberta")]
            crate::automodel::AutoConfig::Roberta(roberta_config) => {
                Ok(AutoModelForMaskedLM::Roberta(
                    crate::models::roberta::RobertaForMaskedLM::new(roberta_config)?,
                ))
            },
            #[cfg(feature = "albert")]
            crate::automodel::AutoConfig::Albert(albert_config) => {
                Ok(AutoModelForMaskedLM::Albert(
                    crate::models::albert::AlbertForMaskedLM::new(albert_config)?,
                ))
            },
            #[allow(unreachable_patterns)]
            _ => Err(TrustformersError::Core(
                CoreTrustformersError::runtime_error(
                    "Model type does not support masked language modeling".into(),
                ),
            )),
        }
    }

    pub fn from_pretrained(model_name_or_path: &str) -> Result<Self> {
        let config = crate::automodel::AutoConfig::from_pretrained(model_name_or_path)?;
        let mut model = Self::from_config(config)?;

        let weights_path = Path::new(model_name_or_path).join("model.safetensors");
        if weights_path.exists() {
            match &mut model {
                #[cfg(feature = "bert")]
                AutoModelForMaskedLM::Bert(bert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    bert.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "roberta")]
                AutoModelForMaskedLM::Roberta(roberta) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    roberta.load_pretrained(&mut reader)?;
                },
                #[cfg(feature = "albert")]
                AutoModelForMaskedLM::Albert(albert) => {
                    let weights_data = std::fs::read(&weights_path)?;
                    let mut reader = std::io::Cursor::new(weights_data);
                    albert.load_pretrained(&mut reader)?;
                },
            }
        }

        Ok(model)
    }
}

// ---------------------------------------------------------------------------
// AutoModelForAudioClassification
// ---------------------------------------------------------------------------

/// Automatic class for loading audio classification models.
///
/// Provides a high-level interface for audio classification tasks such as
/// keyword spotting, sound event detection, and music genre classification.
/// Compatible with wav2vec2, Whisper, and Audio Spectrogram Transformer (AST)
/// style models.
///
/// # Example
///
/// ```rust,ignore
/// use trustformers::AutoModelForAudioClassification;
/// use trustformers::pipeline::audio_classification::AudioClassificationInput;
///
/// let model = AutoModelForAudioClassification::from_pretrained(
///     "facebook/wav2vec2-base-superb-ks",
/// )?;
///
/// let input = AudioClassificationInput::RawAudio {
///     samples: vec![0.0f32; 16_000],
///     sample_rate: 16_000,
/// };
/// let results = model.classify(&input)?;
/// # Ok::<(), trustformers::TrustformersError>(())
/// ```
pub struct AutoModelForAudioClassification {
    pipeline: AudioClassificationPipeline,
    model_name: String,
}

impl AutoModelForAudioClassification {
    /// Load an audio classification model from a pretrained checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if the pipeline configuration is invalid.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = AudioClassificationConfig {
            model_name: model_name.to_string(),
            ..Default::default()
        };
        let pipeline = AudioClassificationPipeline::new(config)?;
        Ok(Self {
            pipeline,
            model_name: model_name.to_string(),
        })
    }

    /// Load from a local directory containing `config.json` and weights.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Io`] if the directory does not exist.
    pub fn from_local(path: &str) -> Result<Self> {
        let dir = std::path::Path::new(path);
        if !dir.is_dir() {
            return Err(TrustformersError::Io {
                message: format!("Model directory not found: {path}"),
                path: Some(path.to_string()),
                suggestion: Some(
                    "Ensure the path points to a directory with config.json.".to_string(),
                ),
            });
        }
        Self::from_pretrained(path)
    }

    /// Classify a single audio input.
    pub fn classify(
        &self,
        input: &AudioClassificationInput,
    ) -> Result<Vec<AudioClassificationResult>> {
        self.pipeline.classify(input)
    }

    /// Classify a batch of audio inputs.
    pub fn classify_batch(
        &self,
        inputs: &[AudioClassificationInput],
    ) -> Result<Vec<Vec<AudioClassificationResult>>> {
        self.pipeline.classify_batch(inputs)
    }

    /// Access the underlying [`AudioClassificationPipeline`].
    pub fn pipeline(&self) -> &AudioClassificationPipeline {
        &self.pipeline
    }

    /// Return the model name or path used to initialise this instance.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

// ---------------------------------------------------------------------------
// AutoModelForImageClassification
// ---------------------------------------------------------------------------

/// Automatic class for loading image classification models.
///
/// Supports ViT, CLIP, and convolutional backbone architectures.
///
/// # Example
///
/// ```rust,ignore
/// use trustformers::AutoModelForImageClassification;
/// use trustformers::pipeline::image_classification::ImageClassificationInput;
///
/// let model = AutoModelForImageClassification::from_pretrained(
///     "google/vit-base-patch16-224",
/// )?;
///
/// let input = ImageClassificationInput::RgbImage {
///     data: vec![128u8; 224 * 224 * 3],
///     width: 224,
///     height: 224,
/// };
/// let results = model.classify(&input)?;
/// # Ok::<(), trustformers::TrustformersError>(())
/// ```
pub struct AutoModelForImageClassification {
    pipeline: ImageClassificationPipeline,
    model_name: String,
}

impl AutoModelForImageClassification {
    /// Load an image classification model from a pretrained checkpoint.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError`] if the pipeline configuration is invalid.
    pub fn from_pretrained(model_name: &str) -> Result<Self> {
        let config = ImageClassificationConfig {
            model_name: model_name.to_string(),
            ..Default::default()
        };
        let pipeline = ImageClassificationPipeline::new(config)?;
        Ok(Self {
            pipeline,
            model_name: model_name.to_string(),
        })
    }

    /// Load from a local directory.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Io`] if the directory does not exist.
    pub fn from_local(path: &str) -> Result<Self> {
        let dir = std::path::Path::new(path);
        if !dir.is_dir() {
            return Err(TrustformersError::Io {
                message: format!("Model directory not found: {path}"),
                path: Some(path.to_string()),
                suggestion: Some(
                    "Ensure the path points to a directory with config.json.".to_string(),
                ),
            });
        }
        Self::from_pretrained(path)
    }

    /// Classify a single image.
    pub fn classify(
        &self,
        input: &ImageClassificationInput,
    ) -> Result<Vec<ImageClassificationResult>> {
        self.pipeline.classify(input)
    }

    /// Classify a batch of images.
    pub fn classify_batch(
        &self,
        inputs: &[ImageClassificationInput],
    ) -> Result<Vec<Vec<ImageClassificationResult>>> {
        self.pipeline.classify_batch(inputs)
    }

    /// Access the underlying [`ImageClassificationPipeline`].
    pub fn pipeline(&self) -> &ImageClassificationPipeline {
        &self.pipeline
    }

    /// Return the model name or path.
    pub fn model_name(&self) -> &str {
        &self.model_name
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- AutoModelForAudioClassification tests ----

    #[test]
    fn test_audio_classification_from_pretrained_returns_model() {
        let model = AutoModelForAudioClassification::from_pretrained("test-audio-model");
        assert!(model.is_ok(), "Expected Ok result from from_pretrained");
    }

    #[test]
    fn test_audio_classification_model_name_stored_correctly() {
        let model_name = "facebook/wav2vec2-base";
        let model = AutoModelForAudioClassification::from_pretrained(model_name)
            .expect("from_pretrained should succeed");
        assert_eq!(
            model.model_name(),
            model_name,
            "model_name should match input"
        );
    }

    #[test]
    fn test_audio_classification_from_local_missing_dir_returns_error() {
        let result = AutoModelForAudioClassification::from_local("/nonexistent/path/xyz123");
        assert!(result.is_err(), "Expected error for missing directory");
    }

    #[test]
    fn test_audio_classification_from_local_error_contains_path() {
        let path = "/nonexistent/automodel_test_path";
        let result = AutoModelForAudioClassification::from_local(path);
        match result {
            Err(TrustformersError::Io { message, .. }) => {
                assert!(
                    message.contains(path),
                    "Error message should mention the missing path"
                );
            },
            _ => panic!("Expected Io error for missing directory"),
        }
    }

    #[test]
    fn test_audio_classification_pipeline_access() {
        let model = AutoModelForAudioClassification::from_pretrained("test-model")
            .expect("from_pretrained should succeed");
        // Simply accessing pipeline must not panic
        let _pipeline = model.pipeline();
    }

    #[test]
    fn test_audio_classify_empty_audio_returns_result() {
        let model = AutoModelForAudioClassification::from_pretrained("test-model")
            .expect("from_pretrained should succeed");
        let input = AudioClassificationInput::RawAudio {
            samples: vec![0.0f32; 100],
            sample_rate: 16_000,
        };
        // Pipeline may succeed or fail; we just verify it returns a Result without panicking.
        let _result = model.classify(&input);
    }

    #[test]
    fn test_audio_classify_batch_empty_slice_returns_ok() {
        let model = AutoModelForAudioClassification::from_pretrained("test-model")
            .expect("from_pretrained should succeed");
        let result = model.classify_batch(&[]);
        assert!(
            result.is_ok(),
            "classify_batch with empty slice should be Ok"
        );
        assert_eq!(
            result.expect("expected Ok").len(),
            0,
            "Empty batch should yield empty results"
        );
    }

    // ---- AutoModelForImageClassification tests ----

    #[test]
    fn test_image_classification_from_pretrained_returns_model() {
        let model = AutoModelForImageClassification::from_pretrained("google/vit-base-patch16-224");
        assert!(model.is_ok(), "Expected Ok result from from_pretrained");
    }

    #[test]
    fn test_image_classification_model_name_stored_correctly() {
        let model_name = "openai/clip-vit-base-patch32";
        let model = AutoModelForImageClassification::from_pretrained(model_name)
            .expect("from_pretrained should succeed");
        assert_eq!(
            model.model_name(),
            model_name,
            "model_name should match input"
        );
    }

    #[test]
    fn test_image_classification_from_local_missing_dir_returns_error() {
        let result = AutoModelForImageClassification::from_local("/nonexistent/img_path_xyz");
        assert!(result.is_err(), "Expected error for missing directory");
    }

    #[test]
    fn test_image_classification_from_local_error_is_io_variant() {
        let path = "/nonexistent/image_automodel_test";
        let result = AutoModelForImageClassification::from_local(path);
        assert!(
            matches!(result, Err(TrustformersError::Io { .. })),
            "Expected Io variant for missing directory"
        );
    }

    #[test]
    fn test_image_classification_pipeline_access() {
        let model = AutoModelForImageClassification::from_pretrained("test-img-model")
            .expect("from_pretrained should succeed");
        let _pipeline = model.pipeline();
    }

    #[test]
    fn test_image_classify_batch_empty_returns_empty_ok() {
        let model = AutoModelForImageClassification::from_pretrained("test-img-model")
            .expect("from_pretrained should succeed");
        let result = model.classify_batch(&[]);
        assert!(result.is_ok(), "Empty batch classify should return Ok");
        assert_eq!(
            result.expect("Ok").len(),
            0,
            "Empty batch should yield empty results"
        );
    }

    #[test]
    fn test_image_classify_single_input_no_panic() {
        let model = AutoModelForImageClassification::from_pretrained("test-img-model")
            .expect("from_pretrained should succeed");
        let input = ImageClassificationInput::RgbImage {
            data: vec![128u8; 224 * 224 * 3],
            width: 224,
            height: 224,
        };
        // Should not panic regardless of result
        let _result = model.classify(&input);
    }

    // ---- Task routing / unsupported task tests ----

    #[test]
    fn test_audio_model_name_is_nonempty() {
        let model = AutoModelForAudioClassification::from_pretrained("wav2vec2-base")
            .expect("from_pretrained should succeed");
        assert!(
            !model.model_name().is_empty(),
            "model_name must not be empty"
        );
    }

    #[test]
    fn test_image_model_name_is_nonempty() {
        let model = AutoModelForImageClassification::from_pretrained("vit-base")
            .expect("from_pretrained should succeed");
        assert!(
            !model.model_name().is_empty(),
            "model_name must not be empty"
        );
    }

    #[test]
    fn test_audio_from_local_with_existing_dir_does_not_return_io_error() {
        // Use a directory that does exist (e.g. /tmp) to verify from_local proceeds past
        // the directory check (the pipeline creation may still fail for other reasons).
        let result = AutoModelForAudioClassification::from_local("/tmp");
        // Either Ok or a non-Io error is acceptable – just must not be the Io "not found" variant.
        if let Err(TrustformersError::Io { message, .. }) = result {
            assert!(
                !message.contains("not found"),
                "Should not return 'not found' Io error for existing dir"
            );
        }
    }

    #[test]
    fn test_image_from_local_with_existing_dir_does_not_return_io_error() {
        let result = AutoModelForImageClassification::from_local("/tmp");
        if let Err(TrustformersError::Io { message, .. }) = result {
            assert!(
                !message.contains("not found"),
                "Should not return 'not found' Io error for existing dir"
            );
        }
    }

    #[test]
    fn test_multiple_audio_models_have_independent_names() {
        let m1 = AutoModelForAudioClassification::from_pretrained("model-a")
            .expect("from_pretrained for model-a should succeed");
        let m2 = AutoModelForAudioClassification::from_pretrained("model-b")
            .expect("from_pretrained for model-b should succeed");
        assert_ne!(
            m1.model_name(),
            m2.model_name(),
            "Different models must have different names"
        );
    }

    #[test]
    fn test_multiple_image_models_have_independent_names() {
        let m1 = AutoModelForImageClassification::from_pretrained("img-model-a")
            .expect("from_pretrained for img-model-a should succeed");
        let m2 = AutoModelForImageClassification::from_pretrained("img-model-b")
            .expect("from_pretrained for img-model-b should succeed");
        assert_ne!(
            m1.model_name(),
            m2.model_name(),
            "Different models must have different names"
        );
    }

    #[test]
    fn test_audio_classification_long_model_name() {
        let long_name = "a".repeat(256);
        let model = AutoModelForAudioClassification::from_pretrained(&long_name)
            .expect("from_pretrained should succeed even with long name");
        assert_eq!(
            model.model_name(),
            long_name.as_str(),
            "Long model name must be preserved"
        );
    }

    #[test]
    fn test_image_classification_special_chars_in_model_name() {
        let name = "org/model-name_v2.0";
        let model = AutoModelForImageClassification::from_pretrained(name)
            .expect("from_pretrained should succeed");
        assert_eq!(
            model.model_name(),
            name,
            "Special-char model name must be preserved exactly"
        );
    }
}
