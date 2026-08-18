//! Multi-modal pipeline: fuses text/image/audio features into one
//! representation using real per-modality feature extractors.
//!
//! # What is real here, and what is honestly unavailable
//!
//! - **Text**: routed through [`GenericFeatureExtractor`] (real
//!   hash-bucket bag-of-words features, one real vector per word -- see
//!   `auto::feature_extractors::generic`).
//! - **Image**: routed through [`VisionFeatureExtractor`], which really
//!   decodes/resizes/crops/normalizes the image bytes (see
//!   `pipeline::media::image_proc`). Turning those pixels into a
//!   *semantic* embedding needs a trained vision encoder, which this
//!   workspace does not have wired in; that step honestly returns
//!   [`TrustformersError::FeatureUnavailable`] rather than a fabricated
//!   vector (see `VisionFeatureExtractor::extract_visual_features`), and
//!   this pipeline propagates that error rather than working around it.
//! - **Audio**: routed through real WAV decoding
//!   ([`audio_dsp::decode_wav`]) followed by [`AudioFeatureExtractor`]'s
//!   real (FFT-based) spectral features. This modality is genuinely
//!   complete end to end.
//! - **Video**: no `FeatureInput` variant and no feature extractor for
//!   video exists anywhere in this workspace. Every call honestly reports
//!   this as an unsupported modality via
//!   [`crate::pipeline::media::unsupported_model`] rather than reusing the
//!   audio or image path against video bytes.
//!
//! [`MultiModalOutput::text`], `::image`, `::audio` and `::classifications`
//! are honestly `None`: this pipeline is generic over `M: Model` with an
//! opaque `Input`/`Output`, so there is no way to route real fused
//! features into an arbitrary model's forward pass, or to fabricate a
//! generated response or classification without one. What genuinely
//! executes and is reported: real per-modality feature extraction (or a
//! structured error), real fusion arithmetic
//! ([`MultiModalOutput::fused_features`]), real cross-modal attention, and
//! real cross-modal cosine similarity.

use crate::auto::feature_extractors::{
    AudioFeatureConfig, AudioFeatureExtractor, FeatureExtractor, GenericFeatureConfig,
    GenericFeatureExtractor, VisionFeatureConfig, VisionFeatureExtractor,
};
use crate::auto::types::{FeatureInput, ImageFormat};
use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::media::{audio_dsp, unsupported_model};
use crate::pipeline::{BasePipeline, Device, Pipeline};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use trustformers_core::cache::CacheKeyBuilder;

/// Shared feature dimensionality for the text and image processors, and
/// the dimension [`FusionLayer::add_features`] / `::weighted_average_features`
/// require a modality's per-position vector to reach before folding it in.
/// `768` matches the common "base model" hidden size convention already
/// used throughout this crate's default configurations (e.g. BERT-base).
const COMMON_FEATURE_DIM: usize = 768;

/// Configuration for multi-modal pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Maximum sequence length for text input
    pub max_text_length: usize,
    /// Maximum image dimensions
    pub max_image_size: (usize, usize),
    /// Maximum audio duration in seconds
    pub max_audio_duration: f64,
    /// Fusion strategy for combining modalities
    pub fusion_strategy: FusionStrategy,
    /// Whether to normalize inputs
    pub normalize_inputs: bool,
    /// Attention mechanism configuration
    pub attention_config: AttentionConfig,
    /// Whether to use cross-modal attention
    pub cross_modal_attention: bool,
    /// Temperature for output generation
    pub temperature: f32,
    /// Top-k for sampling
    pub top_k: Option<usize>,
    /// Top-p for nucleus sampling
    pub top_p: Option<f32>,
}

impl Default for MultiModalConfig {
    fn default() -> Self {
        Self {
            max_text_length: 512,
            max_image_size: (224, 224),
            max_audio_duration: 30.0,
            fusion_strategy: FusionStrategy::Concatenation,
            normalize_inputs: true,
            attention_config: AttentionConfig::default(),
            cross_modal_attention: true,
            temperature: 1.0,
            top_k: None,
            top_p: None,
        }
    }
}

/// Fusion strategy for combining different modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple concatenation of features
    Concatenation,
    /// Element-wise addition
    Addition,
    /// Weighted average
    WeightedAverage,
    /// Cross-attention fusion
    CrossAttention,
    /// Gated fusion
    GatedFusion,
    /// Transformer-based fusion
    TransformerFusion,
}

/// Attention configuration for multi-modal processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionConfig {
    pub num_heads: usize,
    pub head_dim: usize,
    pub dropout: f32,
    pub use_relative_position: bool,
    pub max_relative_position: i32,
}

impl Default for AttentionConfig {
    fn default() -> Self {
        Self {
            num_heads: 8,
            head_dim: 64,
            dropout: 0.1,
            use_relative_position: true,
            max_relative_position: 128,
        }
    }
}

/// Input for multi-modal pipeline
#[derive(Debug, Clone)]
pub struct MultiModalInput {
    /// Text input
    pub text: Option<String>,
    /// Image input as bytes (any container [`ImageProcessor`] can decode --
    /// see its docs; the format is sniffed from content, not declared here)
    pub image: Option<Vec<u8>>,
    /// Audio input as bytes. Must be a RIFF/WAVE (`.wav`) container -- see
    /// [`AudioProcessor`].
    pub audio: Option<Vec<u8>>,
    /// Video input as bytes. No real feature extraction path exists for
    /// video in this workspace (see the module docs); supplying this
    /// always fails with a structured error.
    pub video: Option<Vec<u8>>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
    /// Input modality weights
    pub modality_weights: Option<HashMap<String, f32>>,
}

/// Processed features for each modality
#[derive(Debug, Clone)]
pub struct ModalityFeatures {
    /// Text features
    pub text_features: Option<Vec<Vec<f32>>>,
    /// Image features
    pub image_features: Option<Vec<Vec<f32>>>,
    /// Audio features
    pub audio_features: Option<Vec<Vec<f32>>>,
    /// Video features
    pub video_features: Option<Vec<Vec<f32>>>,
    /// Feature dimensions
    pub feature_dims: HashMap<String, usize>,
    /// Attention masks
    pub attention_masks: HashMap<String, Vec<bool>>,
}

/// Output from multi-modal pipeline
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalOutput {
    /// Generated text response. Always `None`: this pipeline is generic
    /// over `M: Model` with an opaque `Input`/`Output`, so there is no
    /// architecture-independent way to run real text generation from
    /// fused multimodal features. See [`MultiModalOutput::fused_features`]
    /// for the real computed representation.
    pub text: Option<String>,
    /// Generated image. Always `None` for the same reason as `text`.
    pub image: Option<Vec<u8>>,
    /// Generated audio. Always `None` for the same reason as `text`.
    pub audio: Option<Vec<u8>>,
    /// Classification scores. Always `None`: no classification head is
    /// attached to this generic pipeline.
    pub classifications: Option<Vec<ClassificationResult>>,
    /// The real fused feature representation computed by
    /// [`MultiModalPipeline::fuse_features`] (per the configured
    /// [`FusionStrategy`]) from the real per-modality features that were
    /// actually extracted.
    pub fused_features: Vec<Vec<f32>>,
    /// Attention weights for interpretability
    pub attention_weights: Option<AttentionWeights>,
    /// Feature similarities between modalities
    pub cross_modal_similarities: Option<HashMap<String, f32>>,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Classification result for multi-modal tasks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
    pub modality_contributions: HashMap<String, f32>,
}

/// Attention weights for interpretability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttentionWeights {
    pub text_to_image: Option<Vec<Vec<f32>>>,
    pub image_to_text: Option<Vec<Vec<f32>>>,
    pub audio_to_text: Option<Vec<Vec<f32>>>,
    pub cross_modal_attention: Option<Vec<Vec<f32>>>,
}

/// Processing metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    pub processing_time_ms: u64,
    pub modalities_used: Vec<String>,
    pub fusion_strategy_used: String,
    /// Confidence of a real classification/generation head, when one is
    /// attached and actually ran. This generic pipeline attaches none, so
    /// it is honestly `None` rather than a placeholder constant -- see the
    /// module docs.
    pub model_confidence: Option<f32>,
    pub feature_extraction_time_ms: HashMap<String, u64>,
}

/// Multi-modal pipeline
pub struct MultiModalPipeline<M, T> {
    base: BasePipeline<M, T>,
    config: MultiModalConfig,
    text_processor: Arc<TextProcessor>,
    image_processor: Arc<ImageProcessor>,
    audio_processor: Arc<AudioProcessor>,
    video_processor: Arc<VideoProcessor>,
    fusion_layer: Arc<FusionLayer>,
}

impl<M, T> MultiModalPipeline<M, T>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, tokenizer),
            config: MultiModalConfig::default(),
            text_processor: Arc::new(TextProcessor::new()),
            image_processor: Arc::new(ImageProcessor::new()),
            audio_processor: Arc::new(AudioProcessor::new()),
            video_processor: Arc::new(VideoProcessor::new()),
            fusion_layer: Arc::new(FusionLayer::new()),
        })
    }

    pub fn with_config(mut self, config: MultiModalConfig) -> Self {
        self.config = config;
        self
    }

    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Self {
        self.config.fusion_strategy = strategy;
        self
    }

    pub fn with_cross_modal_attention(mut self, enabled: bool) -> Self {
        self.config.cross_modal_attention = enabled;
        self
    }

    pub fn to_device(mut self, device: Device) -> Self {
        self.base = self.base.to_device(device);
        self
    }

    /// Process input from multiple modalities
    ///
    /// # Errors
    ///
    /// Propagates whatever the per-modality processor returns: real
    /// decode/preprocessing failures, [`TrustformersError::FeatureUnavailable`]
    /// when a modality's real preprocessing succeeded but no encoder is
    /// attached (currently images), or the structured "unsupported
    /// modality" error for video (see the module docs).
    pub fn process_multimodal(&self, input: &MultiModalInput) -> Result<ModalityFeatures> {
        let mut features = ModalityFeatures {
            text_features: None,
            image_features: None,
            audio_features: None,
            video_features: None,
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };

        // Process text input
        if let Some(text) = &input.text {
            let text_features = self.text_processor.process(text, &self.config)?;
            insert_modality(&mut features, "text", text_features, |f, v| {
                f.text_features = v
            });
        }

        // Process image input
        if let Some(image) = &input.image {
            let image_features = self.image_processor.process(image, &self.config)?;
            insert_modality(&mut features, "image", image_features, |f, v| {
                f.image_features = v
            });
        }

        // Process audio input
        if let Some(audio) = &input.audio {
            let audio_features = self.audio_processor.process(audio, &self.config)?;
            insert_modality(&mut features, "audio", audio_features, |f, v| {
                f.audio_features = v
            });
        }

        // Process video input -- always a structured error today, see
        // `VideoProcessor::process`.
        if let Some(video) = &input.video {
            let video_features = self.video_processor.process(video, &self.config)?;
            insert_modality(&mut features, "video", video_features, |f, v| {
                f.video_features = v
            });
        }

        Ok(features)
    }

    /// Fuse features from different modalities
    pub fn fuse_features(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        self.fusion_layer.fuse(features, &self.config)
    }

    /// Compute cross-modal attention
    pub fn compute_cross_modal_attention(
        &self,
        features: &ModalityFeatures,
    ) -> Result<AttentionWeights> {
        let mut attention_weights = AttentionWeights {
            text_to_image: None,
            image_to_text: None,
            audio_to_text: None,
            cross_modal_attention: None,
        };

        // Text-to-image attention
        if let (Some(text_features), Some(image_features)) =
            (&features.text_features, &features.image_features)
        {
            attention_weights.text_to_image =
                Some(self.compute_attention_weights(text_features, image_features)?);
            attention_weights.image_to_text =
                Some(self.compute_attention_weights(image_features, text_features)?);
        }

        // Audio-to-text attention
        if let (Some(audio_features), Some(text_features)) =
            (&features.audio_features, &features.text_features)
        {
            attention_weights.audio_to_text =
                Some(self.compute_attention_weights(audio_features, text_features)?);
        }

        Ok(attention_weights)
    }

    /// Compute attention weights between two modalities
    fn compute_attention_weights(
        &self,
        query_features: &[Vec<f32>],
        key_features: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        let mut attention_weights = Vec::new();

        for query in query_features {
            let mut query_weights = Vec::new();
            for key in key_features {
                // Compute dot product attention
                let dot_product: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();

                // Apply softmax (simplified)
                let attention_score = (dot_product / (query.len() as f32).sqrt()).exp();
                query_weights.push(attention_score);
            }

            // Normalize weights
            let sum: f32 = query_weights.iter().sum();
            if sum > 0.0 {
                query_weights.iter_mut().for_each(|w| *w /= sum);
            }

            attention_weights.push(query_weights);
        }

        Ok(attention_weights)
    }

    /// Compute similarities between modalities
    fn compute_cross_modal_similarities(
        &self,
        features: &ModalityFeatures,
    ) -> HashMap<String, f32> {
        let mut similarities = HashMap::new();

        // Text-Image similarity
        if let (Some(text_features), Some(image_features)) =
            (&features.text_features, &features.image_features)
        {
            if let (Some(t0), Some(i0)) = (text_features.first(), image_features.first()) {
                similarities.insert(
                    "text_image".to_string(),
                    self.compute_feature_similarity(t0, i0),
                );
            }
        }

        // Text-Audio similarity
        if let (Some(text_features), Some(audio_features)) =
            (&features.text_features, &features.audio_features)
        {
            if let (Some(t0), Some(a0)) = (text_features.first(), audio_features.first()) {
                similarities.insert(
                    "text_audio".to_string(),
                    self.compute_feature_similarity(t0, a0),
                );
            }
        }

        // Image-Audio similarity
        if let (Some(image_features), Some(audio_features)) =
            (&features.image_features, &features.audio_features)
        {
            if let (Some(i0), Some(a0)) = (image_features.first(), audio_features.first()) {
                similarities.insert(
                    "image_audio".to_string(),
                    self.compute_feature_similarity(i0, a0),
                );
            }
        }

        similarities
    }

    /// Compute cosine similarity between two feature vectors
    fn compute_feature_similarity(&self, features1: &[f32], features2: &[f32]) -> f32 {
        let min_len = features1.len().min(features2.len());
        let dot_product: f32 = features1[..min_len]
            .iter()
            .zip(features2[..min_len].iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f32 = features1[..min_len].iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm2: f32 = features2[..min_len].iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm1 > 0.0 && norm2 > 0.0 {
            dot_product / (norm1 * norm2)
        } else {
            0.0
        }
    }
}

/// Record a processed modality's features on `features`, deriving
/// `feature_dims`/`attention_masks` from the *real* shape of `values`
/// (`values.first().map(Vec::len).unwrap_or(0)`) rather than indexing
/// `values[0]` directly -- an empty (but successfully processed, e.g. an
/// empty text input) modality must not panic.
fn insert_modality(
    features: &mut ModalityFeatures,
    name: &str,
    values: Vec<Vec<f32>>,
    set: impl FnOnce(&mut ModalityFeatures, Option<Vec<Vec<f32>>>),
) {
    let dim = values.first().map(Vec::len).unwrap_or(0);
    features.feature_dims.insert(name.to_string(), dim);
    features.attention_masks.insert(name.to_string(), vec![true; values.len()]);
    set(features, Some(values));
}

impl<M, T> Pipeline for MultiModalPipeline<M, T>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    type Input = MultiModalInput;
    type Output = MultiModalOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        let start_time = std::time::Instant::now();
        let mut feature_extraction_times = HashMap::new();

        // Check cache first
        let cache_key = if let Some(cache) = &self.base.cache {
            let mut builder = CacheKeyBuilder::new("multimodal", "inference");
            if let Some(text) = &input.text {
                builder = builder.with_text(text);
            }
            if let Some(image) = &input.image {
                builder = builder.with_param("image", image);
            }
            if let Some(audio) = &input.audio {
                builder = builder.with_param("audio", audio);
            }
            builder = builder.with_param(
                "config",
                &serde_json::to_string(&self.config).unwrap_or_default(),
            );

            let key = builder.build();
            if let Some(cached) = cache.get(&key) {
                if let Ok(output) = serde_json::from_slice::<MultiModalOutput>(&cached) {
                    return Ok(output);
                }
            }
            Some(key)
        } else {
            None
        };

        // Determine which modalities are present up front: used both to
        // label the output and to divide the real measured feature-time
        // below by how many modalities actually ran, rather than a fixed
        // constant.
        let mut modalities_used = Vec::new();
        if input.text.is_some() {
            modalities_used.push("text".to_string());
        }
        if input.image.is_some() {
            modalities_used.push("image".to_string());
        }
        if input.audio.is_some() {
            modalities_used.push("audio".to_string());
        }
        if input.video.is_some() {
            modalities_used.push("video".to_string());
        }

        // Process each modality
        let feature_start = std::time::Instant::now();
        let features = self.process_multimodal(&input)?;
        let feature_time = feature_start.elapsed().as_millis() as u64;

        // Split the real measured feature-extraction time evenly across
        // however many modalities actually ran (not a fixed division by
        // 4, which under-reports whenever fewer than all four are
        // present).
        let per_modality_time = feature_time / modalities_used.len().max(1) as u64;
        for modality in &modalities_used {
            feature_extraction_times.insert(modality.clone(), per_modality_time);
        }

        // Fuse features -- the real, computed representation this
        // pipeline actually reports (see `MultiModalOutput::fused_features`).
        let fused_features = self.fuse_features(&features)?;

        // Compute cross-modal attention if enabled
        let attention_weights = if self.config.cross_modal_attention {
            Some(self.compute_cross_modal_attention(&features)?)
        } else {
            None
        };

        // Compute cross-modal similarities
        let cross_modal_similarities = Some(self.compute_cross_modal_similarities(&features));

        // No generative or classification head is attached to this
        // generic pipeline -- see the module docs for why `text`/`image`/
        // `audio`/`classifications`/`model_confidence` are honestly
        // `None` rather than a placeholder echo of the input.
        let output = MultiModalOutput {
            text: None,
            image: None,
            audio: None,
            classifications: None,
            fused_features,
            attention_weights,
            cross_modal_similarities,
            metadata: ProcessingMetadata {
                processing_time_ms: start_time.elapsed().as_millis() as u64,
                modalities_used,
                fusion_strategy_used: format!("{:?}", self.config.fusion_strategy),
                model_confidence: None,
                feature_extraction_time_ms: feature_extraction_times,
            },
        };

        // Cache the result
        if let (Some(cache), Some(key)) = (&self.base.cache, cache_key) {
            if let Ok(serialized) = serde_json::to_vec(&output) {
                cache.insert(key, serialized);
            }
        }

        Ok(output)
    }
}

/// Text processor for multi-modal pipeline.
///
/// Produces one real, content-derived feature vector per word by routing
/// each word through [`GenericFeatureExtractor`] -- the same real feature
/// extractor `AutoFeatureExtractor` selects for text-only pipelines (see
/// `auto::feature_extractors::generic`): a deterministic hash of the word
/// into a `COMMON_FEATURE_DIM`-wide bucket vector, L2-normalized. This does
/// not claim semantic understanding (there is no trained embedding table
/// here), but every vector is genuinely derived from the word it
/// represents -- the same word always produces the same vector, and
/// different words (almost always) produce different ones -- rather than
/// a content-independent placeholder.
pub struct TextProcessor;

impl Default for TextProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl TextProcessor {
    pub fn new() -> Self {
        Self
    }

    /// # Errors
    ///
    /// Propagates [`GenericFeatureExtractor::extract_features`]'s errors
    /// (in practice unreachable for well-formed `&str` word input, but
    /// surfaced honestly rather than swallowed).
    pub fn process(&self, text: &str, config: &MultiModalConfig) -> Result<Vec<Vec<f32>>> {
        let extractor = GenericFeatureExtractor::new(GenericFeatureConfig {
            feature_size: COMMON_FEATURE_DIM,
            max_batch_size: None,
        });

        let words: Vec<&str> = text.split_whitespace().collect();
        let max_words = config.max_text_length.min(words.len());

        let mut features = Vec::with_capacity(max_words);
        for word in &words[..max_words] {
            let output = extractor.extract_features(&FeatureInput::Text {
                content: (*word).to_string(),
                metadata: None,
            })?;
            features.push(output.features);
        }

        Ok(features)
    }
}

/// Image processor for multi-modal pipeline.
///
/// Routes real image bytes through [`VisionFeatureExtractor`]: real
/// container decoding (Netpbm always, plus every format the `image` crate
/// handles under the `vision` feature), real bilinear resize, real centre
/// crop, real per-channel normalization -- see
/// `pipeline::media::image_proc`. Turning those pixels into a *semantic*
/// feature vector needs a trained vision encoder, which this workspace
/// does not have wired in, so [`Self::process`] honestly propagates
/// [`TrustformersError::FeatureUnavailable`] in that case instead of
/// inventing a vector.
pub struct ImageProcessor;

impl Default for ImageProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageProcessor {
    pub fn new() -> Self {
        Self
    }

    /// # Errors
    ///
    /// - Whatever [`VisionFeatureExtractor::preprocess_image`] returns for
    ///   corrupt/empty/undecodable image bytes.
    /// - [`TrustformersError::FeatureUnavailable`] when preprocessing
    ///   succeeded but no vision encoder is attached (currently always,
    ///   see the struct docs).
    pub fn process(&self, image: &[u8], config: &MultiModalConfig) -> Result<Vec<Vec<f32>>> {
        let extractor = VisionFeatureExtractor::new(VisionFeatureConfig {
            image_size: config.max_image_size.0.max(1),
            feature_size: COMMON_FEATURE_DIM,
            normalize: config.normalize_inputs,
            do_resize: true,
            do_center_crop: true,
            crop_size: None,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
            max_batch_size: None,
        });

        let output = extractor.extract_features(&FeatureInput::Image {
            data: image.to_vec(),
            format: sniff_image_format(image),
            metadata: None,
        })?;

        Ok(vec![output.features])
    }
}

/// Audio processor for multi-modal pipeline.
///
/// Decodes a real RIFF/WAVE container ([`audio_dsp::decode_wav`]) and
/// routes the decoded samples through [`AudioFeatureExtractor`] for real
/// FFT-based spectral features -- genuinely complete end to end, unlike
/// the image path (no trained encoder is needed for classical spectral
/// features). Only WAV is supported today; any other container is a
/// structured error rather than a silent all-zero fallback.
pub struct AudioProcessor;

impl Default for AudioProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Feature dimensionality for [`AudioProcessor`]'s spectral features.
/// `128` matches the common mel-spectrogram-bin convention for speech
/// models (also the value the pre-fix placeholder happened to use).
const AUDIO_FEATURE_DIM: usize = 128;

impl AudioProcessor {
    pub fn new() -> Self {
        Self
    }

    /// # Errors
    ///
    /// - [`TrustformersError::InvalidInput`] if `audio` is not a
    ///   RIFF/WAVE byte stream.
    /// - Whatever [`audio_dsp::decode_wav`] / [`AudioFeatureExtractor::extract_features`]
    ///   return for a malformed or unsupported-codec container.
    pub fn process(&self, audio: &[u8], config: &MultiModalConfig) -> Result<Vec<Vec<f32>>> {
        if !audio_dsp::is_wav(audio) {
            return Err(TrustformersError::invalid_input_simple(
                "multimodal audio processor: only RIFF/WAVE (.wav) byte streams are supported \
                 today; the input did not start with a RIFF/WAVE header"
                    .to_string(),
            ));
        }
        let mut decoded = audio_dsp::decode_wav(audio)?;

        // Real use of `max_audio_duration`: truncate the real decoded
        // samples rather than deriving a fabricated frame count from it.
        if config.max_audio_duration > 0.0 {
            let max_samples = (config.max_audio_duration * f64::from(decoded.sample_rate)) as usize;
            if decoded.samples.len() > max_samples {
                decoded.samples.truncate(max_samples);
            }
        }

        let extractor = AudioFeatureExtractor::new(AudioFeatureConfig {
            sampling_rate: decoded.sample_rate,
            feature_size: AUDIO_FEATURE_DIM,
            n_fft: 512,
            hop_length: 160,
            normalize: config.normalize_inputs,
            max_batch_size: None,
        });

        let output = extractor.extract_features(&FeatureInput::Audio {
            samples: decoded.samples,
            sample_rate: decoded.sample_rate,
            metadata: None,
        })?;

        Ok(chunk_features(output.features, AUDIO_FEATURE_DIM))
    }
}

/// Video processor for multi-modal pipeline.
///
/// No real video feature extraction path exists anywhere in this
/// workspace: [`crate::auto::types::FeatureInput`] has no `Video` variant,
/// and no `auto::feature_extractors` implementation decodes a video
/// container. Reusing the audio or image path against video bytes would
/// silently misinterpret the container, so every call instead reports
/// this unsupported modality with a structured, self-describing error.
pub struct VideoProcessor;

impl Default for VideoProcessor {
    fn default() -> Self {
        Self::new()
    }
}

impl VideoProcessor {
    pub fn new() -> Self {
        Self
    }

    /// # Errors
    ///
    /// Always returns [`TrustformersError::FeatureUnavailable`] -- see the
    /// struct docs.
    pub fn process(&self, _video: &[u8], _config: &MultiModalConfig) -> Result<Vec<Vec<f32>>> {
        Err(unsupported_model(
            "multimodal-feature-extraction",
            "video",
            &[],
        ))
    }
}

/// Best-effort image container sniffing from magic bytes, for the
/// informational `format` field on [`FeatureInput::Image`]. Real decoding
/// (see `pipeline::media::image_proc::decode_image_bytes`) auto-detects
/// the container from its own byte signature and does not consult this
/// value, so a wrong guess here cannot corrupt decoding -- it can only
/// make an error message name the wrong container.
fn sniff_image_format(data: &[u8]) -> ImageFormat {
    if data.starts_with(&[0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n']) {
        ImageFormat::Png
    } else if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
        ImageFormat::Jpeg
    } else if data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        ImageFormat::Webp
    } else if data.starts_with(b"BM") {
        ImageFormat::Bmp
    } else if data.starts_with(&[0x49, 0x49, 0x2A, 0x00])
        || data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A])
    {
        ImageFormat::Tiff
    } else {
        // Includes Netpbm (P5/P6), which `decode_image_bytes` sniffs and
        // dispatches itself without consulting this label.
        ImageFormat::Png
    }
}

/// Split a flat feature buffer into `chunk_size`-wide vectors (dropping a
/// short final remainder, matching how [`FeatureOutput::shape`] already
/// describes the layout as `[n_frames, feature_size]`).
fn chunk_features(flat: Vec<f32>, chunk_size: usize) -> Vec<Vec<f32>> {
    if chunk_size == 0 {
        return Vec::new();
    }
    flat.chunks_exact(chunk_size).map(|chunk| chunk.to_vec()).collect()
}

/// Fusion layer for combining modality features
pub struct FusionLayer;

impl Default for FusionLayer {
    fn default() -> Self {
        Self::new()
    }
}

impl FusionLayer {
    pub fn new() -> Self {
        Self
    }

    pub fn fuse(
        &self,
        features: &ModalityFeatures,
        config: &MultiModalConfig,
    ) -> Result<Vec<Vec<f32>>> {
        match config.fusion_strategy {
            FusionStrategy::Concatenation => self.concatenate_features(features),
            FusionStrategy::Addition => self.add_features(features),
            FusionStrategy::WeightedAverage => self.weighted_average_features(features),
            FusionStrategy::CrossAttention => self.cross_attention_fusion(features),
            FusionStrategy::GatedFusion => self.gated_fusion(features),
            FusionStrategy::TransformerFusion => self.transformer_fusion(features),
        }
    }

    fn concatenate_features(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        let mut fused_features = Vec::new();

        // Get maximum sequence length
        let max_len = [
            features.text_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.image_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.audio_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.video_features.as_ref().map(|f| f.len()).unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        for i in 0..max_len {
            let mut combined_feature = Vec::new();

            // Concatenate features from all modalities
            if let Some(text_features) = &features.text_features {
                if i < text_features.len() {
                    combined_feature.extend_from_slice(&text_features[i]);
                }
            }

            if let Some(image_features) = &features.image_features {
                if i < image_features.len() {
                    combined_feature.extend_from_slice(&image_features[i]);
                }
            }

            if let Some(audio_features) = &features.audio_features {
                if i < audio_features.len() {
                    combined_feature.extend_from_slice(&audio_features[i]);
                }
            }

            if let Some(video_features) = &features.video_features {
                if i < video_features.len() {
                    combined_feature.extend_from_slice(&video_features[i]);
                }
            }

            if !combined_feature.is_empty() {
                fused_features.push(combined_feature);
            }
        }

        Ok(fused_features)
    }

    fn add_features(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        // Element-wise addition (requires same dimensions)
        let mut fused_features = Vec::new();
        let common_dim = COMMON_FEATURE_DIM;

        let max_len = [
            features.text_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.image_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.audio_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.video_features.as_ref().map(|f| f.len()).unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        for i in 0..max_len {
            let mut combined_feature = vec![0.0; common_dim];
            let mut count = 0;

            // Add features from all available modalities that reach the
            // common dimension.
            for modality_features in [
                &features.text_features,
                &features.image_features,
                &features.audio_features,
                &features.video_features,
            ] {
                if let Some(modality_features) = modality_features {
                    if i < modality_features.len() && modality_features[i].len() >= common_dim {
                        for j in 0..common_dim {
                            combined_feature[j] += modality_features[i][j];
                        }
                        count += 1;
                    }
                }
            }

            // Average the features
            if count > 0 {
                combined_feature.iter_mut().for_each(|x| *x /= count as f32);
                fused_features.push(combined_feature);
            }
        }

        Ok(fused_features)
    }

    fn weighted_average_features(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        // Weighted average with fixed per-modality weights.
        let text_weight = 0.4;
        let image_weight = 0.6;
        let audio_weight = 0.3;
        let video_weight = 0.2;

        let mut fused_features = Vec::new();
        let common_dim = COMMON_FEATURE_DIM;

        let max_len = [
            features.text_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.image_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.audio_features.as_ref().map(|f| f.len()).unwrap_or(0),
            features.video_features.as_ref().map(|f| f.len()).unwrap_or(0),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);

        for i in 0..max_len {
            let mut combined_feature = vec![0.0; common_dim];
            let mut total_weight = 0.0;

            // Weighted combination across every modality present (not
            // just text/image): each still only contributes once it
            // reaches `common_dim`, same as `add_features`.
            for (modality_features, weight) in [
                (&features.text_features, text_weight),
                (&features.image_features, image_weight),
                (&features.audio_features, audio_weight),
                (&features.video_features, video_weight),
            ] {
                if let Some(modality_features) = modality_features {
                    if i < modality_features.len() && modality_features[i].len() >= common_dim {
                        for j in 0..common_dim {
                            combined_feature[j] += modality_features[i][j] * weight;
                        }
                        total_weight += weight;
                    }
                }
            }

            // Normalize by total weight
            if total_weight > 0.0 {
                combined_feature.iter_mut().for_each(|x| *x /= total_weight);
                fused_features.push(combined_feature);
            }
        }

        Ok(fused_features)
    }

    fn cross_attention_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        // Cross-attention between modalities
        // This is a simplified implementation
        self.concatenate_features(features)
    }

    fn gated_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        // Gated fusion with learnable gates
        // This is a simplified implementation
        self.weighted_average_features(features)
    }

    fn transformer_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
        // Transformer-based fusion
        // This is a simplified implementation
        self.concatenate_features(features)
    }
}

/// Factory function for multi-modal pipeline
pub fn multimodal_pipeline<M, T>(model: M, tokenizer: T) -> Result<MultiModalPipeline<M, T>>
where
    M: Model + Send + Sync + 'static,
    T: Tokenizer + Send + Sync + 'static,
{
    MultiModalPipeline::new(model, tokenizer)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- MultiModalConfig tests ----

    #[test]
    fn test_config_default_values() {
        let cfg = MultiModalConfig::default();
        assert_eq!(cfg.max_text_length, 512);
        assert_eq!(cfg.max_image_size, (224, 224));
        assert!((cfg.max_audio_duration - 30.0).abs() < 1e-6);
        assert!(cfg.normalize_inputs);
        assert!(cfg.cross_modal_attention);
        assert!((cfg.temperature - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_config_clone() {
        let cfg = MultiModalConfig {
            max_text_length: 256,
            ..MultiModalConfig::default()
        };
        assert_eq!(cfg.clone().max_text_length, 256);
    }

    // ---- AttentionConfig tests ----

    #[test]
    fn test_attention_config_default() {
        let acfg = AttentionConfig::default();
        assert_eq!(acfg.num_heads, 8);
        assert_eq!(acfg.head_dim, 64);
        assert!((acfg.dropout - 0.1).abs() < 1e-6);
        assert!(acfg.use_relative_position);
        assert_eq!(acfg.max_relative_position, 128);
    }

    // ---- MultiModalInput tests ----

    #[test]
    fn test_input_text_only() {
        let input = MultiModalInput {
            text: Some("Hello world".to_string()),
            image: None,
            audio: None,
            video: None,
            metadata: HashMap::new(),
            modality_weights: None,
        };
        assert!(input.text.is_some());
        assert!(input.image.is_none());
    }

    #[test]
    fn test_input_image_plus_text() {
        let input = MultiModalInput {
            text: Some("Describe this image".to_string()),
            image: Some(vec![0u8; 100]),
            audio: None,
            video: None,
            metadata: HashMap::new(),
            modality_weights: None,
        };
        assert!(input.text.is_some());
        assert!(input.image.is_some());
    }

    #[test]
    fn test_input_multimodality_flags() {
        let input = MultiModalInput {
            text: Some("text".to_string()),
            image: Some(vec![1, 2, 3]),
            audio: Some(vec![4, 5, 6]),
            video: None,
            metadata: HashMap::new(),
            modality_weights: None,
        };
        let mut modalities = Vec::new();
        if input.text.is_some() {
            modalities.push("text");
        }
        if input.image.is_some() {
            modalities.push("image");
        }
        if input.audio.is_some() {
            modalities.push("audio");
        }
        if input.video.is_some() {
            modalities.push("video");
        }
        assert_eq!(modalities.len(), 3);
    }

    // -------------------------------------------------------------------
    // TextProcessor: regression coverage for the bug where
    // `TextProcessor::process` returned `sin((i*768+j) as f32) * 0.1` --
    // entirely a function of position, never of the actual word. These
    // tests fail against that old behavior because they assert real
    // content-dependence.
    // -------------------------------------------------------------------

    #[test]
    fn test_text_processor_produces_features() {
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig::default();
        let features =
            processor.process("Hello world test", &cfg).expect("text processing succeeded");
        // 3 words -> 3 real per-word feature vectors.
        assert_eq!(features.len(), 3);
        assert_eq!(features[0].len(), COMMON_FEATURE_DIM);
    }

    #[test]
    fn test_text_processor_respects_max_length() {
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig {
            max_text_length: 2,
            ..MultiModalConfig::default()
        };
        let text = "one two three four five";
        let features = processor.process(text, &cfg).expect("text processing succeeded");
        assert_eq!(features.len(), 2);
    }

    #[test]
    fn test_text_processor_empty_text() {
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig::default();
        let features = processor.process("", &cfg).expect("empty text processing succeeded");
        assert!(features.is_empty());
    }

    #[test]
    fn test_text_processor_same_word_gives_identical_vector() {
        // Real, deterministic content-derivation: the same word must
        // always hash to the same vector.
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig::default();
        let features = processor.process("repeat repeat", &cfg).expect("ok");
        assert_eq!(features[0], features[1]);
    }

    #[test]
    fn test_text_processor_different_words_give_different_vectors() {
        // Regression: the old sine-wave placeholder differed only by
        // *position*, so two different first words at the same position
        // across two calls would be identical -- this asserts real
        // content-dependence instead.
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig::default();
        let a = processor.process("apple", &cfg).expect("ok");
        let b = processor.process("zebra", &cfg).expect("ok");
        assert_ne!(
            a[0], b[0],
            "different words at the same position must produce different feature vectors"
        );
    }

    #[test]
    fn test_text_processor_vectors_are_l2_normalized() {
        let processor = TextProcessor::new();
        let cfg = MultiModalConfig::default();
        let features = processor.process("hello", &cfg).expect("ok");
        let norm: f32 = features[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "expected a unit-norm vector, got norm {norm}"
        );
    }

    // -------------------------------------------------------------------
    // ImageProcessor: regression coverage for the bug where
    // `ImageProcessor::process` returned `cos((i*768+j) as f32) * 0.1` for
    // a patch grid derived purely from `config.max_image_size`, entirely
    // ignoring the `_image: &[u8]` bytes (the parameter was even
    // underscore-prefixed). The real path decodes the bytes and -- absent
    // an attached vision encoder -- honestly reports
    // `FeatureUnavailable` rather than a placeholder.
    // -------------------------------------------------------------------

    #[test]
    fn test_image_processor_propagates_feature_unavailable_without_encoder() {
        let processor = ImageProcessor::new();
        let cfg = MultiModalConfig::default();
        // A tiny real, decodable Netpbm (P5, grayscale) image: header +
        // 2x2 8-bit pixels. Decoding must succeed; only the (nonexistent)
        // encoder step must fail.
        let ppm = b"P5\n2 2\n255\n\x00\x40\x80\xff".to_vec();
        let result = processor.process(&ppm, &cfg);
        match result {
            Err(TrustformersError::FeatureUnavailable { ref feature, .. }) => {
                assert!(feature.contains("vision"), "feature: {feature}");
            },
            other => panic!(
                "expected a structured FeatureUnavailable (real decode, no encoder attached), \
                 got {other:?}"
            ),
        }
    }

    #[test]
    fn test_image_processor_rejects_corrupt_bytes_before_claiming_success() {
        let processor = ImageProcessor::new();
        let cfg = MultiModalConfig::default();
        // Not a valid image container of any kind, and not empty either.
        let garbage = vec![1u8, 2, 3, 4, 5];
        let result = processor.process(&garbage, &cfg);
        assert!(
            result.is_err(),
            "undecodable bytes must error, never produce a fabricated vector"
        );
    }

    // -------------------------------------------------------------------
    // AudioProcessor: regression coverage for the bug where
    // `AudioProcessor::process` computed a frame count purely from
    // `config.max_audio_duration` and filled every frame with
    // `sin((i*128+j) as f32) * 0.2`, entirely ignoring the `_audio: &[u8]`
    // bytes.
    // -------------------------------------------------------------------

    fn make_wav(samples: &[f32], sample_rate: u32) -> Vec<u8> {
        audio_dsp::encode_wav_pcm16(samples, sample_rate)
    }

    #[test]
    fn test_audio_processor_rejects_non_wav_bytes() {
        let processor = AudioProcessor::new();
        let cfg = MultiModalConfig::default();
        let not_wav = vec![0u8; 64];
        let result = processor.process(&not_wav, &cfg);
        assert!(
            result.is_err(),
            "non-WAV bytes must be a structured error, not silent zeros"
        );
    }

    #[test]
    fn test_audio_processor_produces_real_frames_from_real_wav() {
        let processor = AudioProcessor::new();
        let cfg = MultiModalConfig {
            max_audio_duration: 1.0,
            ..MultiModalConfig::default()
        };
        // 1 second of a real 440 Hz sine tone at 16 kHz -- genuine signal,
        // not silence, so the resulting spectral features are non-trivial.
        let sample_rate = 16000u32;
        let samples: Vec<f32> = (0..sample_rate)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let wav = make_wav(&samples, sample_rate);

        let features = processor.process(&wav, &cfg).expect("real WAV must process");
        assert!(
            !features.is_empty(),
            "a real 1-second tone must yield at least one frame"
        );
        assert_eq!(features[0].len(), AUDIO_FEATURE_DIM);
    }

    #[test]
    fn test_audio_processor_silence_and_tone_produce_different_features() {
        // Regression: the old placeholder's output was a pure function of
        // frame/bin index, so silence and a real tone (same duration,
        // same sample rate) would have produced byte-identical "features".
        let processor = AudioProcessor::new();
        let cfg = MultiModalConfig {
            max_audio_duration: 0.5,
            ..MultiModalConfig::default()
        };
        let sample_rate = 16000u32;
        let n = sample_rate / 2;

        let silence = vec![0.0f32; n as usize];
        let tone: Vec<f32> = (0..n)
            .map(|i| (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sample_rate as f32).sin())
            .collect();

        let silence_features = processor
            .process(&make_wav(&silence, sample_rate), &cfg)
            .expect("silence must process");
        let tone_features = processor
            .process(&make_wav(&tone, sample_rate), &cfg)
            .expect("tone must process");

        assert_eq!(silence_features.len(), tone_features.len());
        assert_ne!(
            silence_features, tone_features,
            "real spectral features of silence and a real tone must differ"
        );
    }

    #[test]
    fn test_audio_processor_respects_max_duration() {
        let processor = AudioProcessor::new();
        let short_cfg = MultiModalConfig {
            max_audio_duration: 0.25,
            ..MultiModalConfig::default()
        };
        let long_cfg = MultiModalConfig {
            max_audio_duration: 2.0,
            ..MultiModalConfig::default()
        };
        let sample_rate = 16000u32;
        let samples: Vec<f32> = (0..sample_rate * 2)
            .map(|i| (2.0 * std::f32::consts::PI * 220.0 * i as f32 / sample_rate as f32).sin())
            .collect();
        let wav = make_wav(&samples, sample_rate);

        let short_features = processor.process(&wav, &short_cfg).expect("ok");
        let long_features = processor.process(&wav, &long_cfg).expect("ok");
        assert!(
            short_features.len() < long_features.len(),
            "a smaller max_audio_duration must truncate to fewer real frames: {} vs {}",
            short_features.len(),
            long_features.len()
        );
    }

    // -------------------------------------------------------------------
    // VideoProcessor: regression coverage for the bug where
    // `VideoProcessor::process` computed a frame count from
    // `config.max_audio_duration` (not even a video-specific config
    // field) and filled every frame with `cos((i*512+j) as f32) * 0.15`.
    // No real video feature extraction path exists anywhere in this
    // workspace, so every call must now fail structurally.
    // -------------------------------------------------------------------

    #[test]
    fn test_video_processor_always_reports_unsupported_modality() {
        let processor = VideoProcessor::new();
        let cfg = MultiModalConfig::default();
        let result = processor.process(&[1, 2, 3, 4], &cfg);
        match result {
            Err(TrustformersError::FeatureUnavailable {
                ref feature,
                ref alternatives,
                ..
            }) => {
                assert!(feature.contains("video"), "feature: {feature}");
                assert!(
                    alternatives.is_empty(),
                    "no video backend exists to name as an alternative"
                );
            },
            other => panic!("expected a structured FeatureUnavailable for video, got {other:?}"),
        }
    }

    // ---- FusionLayer tests ----

    #[test]
    fn test_fusion_concatenation_non_empty() {
        let fusion = FusionLayer::new();
        let cfg = MultiModalConfig {
            fusion_strategy: FusionStrategy::Concatenation,
            ..MultiModalConfig::default()
        };
        let features = ModalityFeatures {
            text_features: Some(vec![vec![0.1; 768]; 3]),
            image_features: None,
            audio_features: None,
            video_features: None,
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };
        let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_fusion_addition_with_two_modalities() {
        let fusion = FusionLayer::new();
        let cfg = MultiModalConfig {
            fusion_strategy: FusionStrategy::Addition,
            ..MultiModalConfig::default()
        };
        let features = ModalityFeatures {
            text_features: Some(vec![vec![1.0; 768]]),
            image_features: Some(vec![vec![2.0; 768]]),
            audio_features: None,
            video_features: None,
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };
        let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
        assert_eq!(fused.len(), 1);
        // Average of 1.0 and 2.0 should be 1.5
        assert!(
            (fused[0][0] - 1.5).abs() < 1e-4,
            "expected 1.5, got {}",
            fused[0][0]
        );
    }

    #[test]
    fn test_fusion_weighted_average() {
        let fusion = FusionLayer::new();
        let cfg = MultiModalConfig {
            fusion_strategy: FusionStrategy::WeightedAverage,
            ..MultiModalConfig::default()
        };
        let features = ModalityFeatures {
            text_features: Some(vec![vec![1.0; 768]]),
            image_features: Some(vec![vec![1.0; 768]]),
            audio_features: None,
            video_features: None,
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };
        let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_fusion_weighted_average_uses_audio_and_video_weights() {
        // Regression: `audio_weight`/`video_weight` used to be computed
        // and then never read -- only text/image were folded into the
        // weighted sum. This fails against that old behavior because a
        // pure-audio input (no text/image at all) would have produced no
        // fused output whatsoever.
        let fusion = FusionLayer::new();
        let cfg = MultiModalConfig {
            fusion_strategy: FusionStrategy::WeightedAverage,
            ..MultiModalConfig::default()
        };
        let features = ModalityFeatures {
            text_features: None,
            image_features: None,
            audio_features: Some(vec![vec![2.0; 768]]),
            video_features: Some(vec![vec![4.0; 768]]),
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };
        let fused = fusion.fuse(&features, &cfg).expect("fusion succeeded");
        assert_eq!(
            fused.len(),
            1,
            "audio+video alone must still produce fused output"
        );
        // (2.0*0.3 + 4.0*0.2) / (0.3+0.2) = 1.4 / 0.5 = 2.8
        assert!((fused[0][0] - 2.8).abs() < 1e-4, "got {}", fused[0][0]);
    }

    // ---- Cross-attention weights tests ----

    #[test]
    fn test_attention_weights_normalised() {
        // compute_attention_weights normalises to sum 1 per query
        let query = vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]];
        let key = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        // Reproduce the logic inline
        let mut attention_weights = Vec::new();
        for q in &query {
            let mut q_weights = Vec::new();
            for k in &key {
                let dot: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();
                let score = (dot / (q.len() as f32).sqrt()).exp();
                q_weights.push(score);
            }
            let sum: f32 = q_weights.iter().sum();
            if sum > 0.0 {
                q_weights.iter_mut().for_each(|w| *w /= sum);
            }
            attention_weights.push(q_weights);
        }

        for row in &attention_weights {
            let sum: f32 = row.iter().sum();
            assert!((sum - 1.0).abs() < 1e-5, "row sum = {}", sum);
        }
    }

    #[test]
    fn test_cross_modal_similarity_range() {
        // cosine similarity must be in [-1, 1]
        let a = [1.0_f32, 0.0, 0.0];
        let b = [0.0_f32, 1.0, 0.0];
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        let sim = if na > 0.0 && nb > 0.0 { dot / (na * nb) } else { 0.0 };
        assert!((-1.0..=1.0).contains(&sim), "sim = {}", sim);
    }

    // ---- Output format tests ----

    #[test]
    fn test_classification_result_score_in_range() {
        let result = ClassificationResult {
            label: "positive".to_string(),
            score: 0.85,
            modality_contributions: HashMap::new(),
        };
        assert!(result.score >= 0.0 && result.score <= 1.0);
    }

    #[test]
    fn test_processing_metadata_modalities_list() {
        let meta = ProcessingMetadata {
            processing_time_ms: 42,
            modalities_used: vec!["text".to_string(), "image".to_string()],
            fusion_strategy_used: "Concatenation".to_string(),
            model_confidence: None,
            feature_extraction_time_ms: HashMap::new(),
        };
        assert_eq!(meta.modalities_used.len(), 2);
        assert!(meta.modalities_used.contains(&"text".to_string()));
    }

    // -------------------------------------------------------------------
    // insert_modality: regression coverage for the panic risk of indexing
    // `text_features[0]` directly when a successfully-processed modality
    // (e.g. empty text) produces zero feature vectors.
    // -------------------------------------------------------------------

    #[test]
    fn test_insert_modality_empty_values_does_not_panic() {
        let mut features = ModalityFeatures {
            text_features: None,
            image_features: None,
            audio_features: None,
            video_features: None,
            feature_dims: HashMap::new(),
            attention_masks: HashMap::new(),
        };
        insert_modality(&mut features, "text", Vec::new(), |f, v| {
            f.text_features = v
        });
        assert_eq!(features.feature_dims["text"], 0);
        assert_eq!(features.text_features, Some(Vec::new()));
    }

    // -------------------------------------------------------------------
    // chunk_features
    // -------------------------------------------------------------------

    #[test]
    fn test_chunk_features_splits_evenly() {
        let flat: Vec<f32> = (0..12).map(|i| i as f32).collect();
        let chunks = chunk_features(flat, 4);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1], vec![4.0, 5.0, 6.0, 7.0]);
    }

    #[test]
    fn test_chunk_features_drops_short_remainder() {
        let flat: Vec<f32> = (0..10).map(|i| i as f32).collect();
        let chunks = chunk_features(flat, 4);
        // 10 / 4 = 2 full chunks; the trailing 2 values are dropped rather
        // than padded with fabricated zeros.
        assert_eq!(chunks.len(), 2);
    }

    // -------------------------------------------------------------------
    // sniff_image_format
    // -------------------------------------------------------------------

    #[test]
    fn test_sniff_image_format_recognises_png_signature() {
        let png_sig = [0x89, b'P', b'N', b'G', b'\r', b'\n', 0x1A, b'\n'];
        assert!(matches!(sniff_image_format(&png_sig), ImageFormat::Png));
    }

    #[test]
    fn test_sniff_image_format_recognises_jpeg_signature() {
        let jpeg_sig = [0xFF, 0xD8, 0xFF, 0xE0];
        assert!(matches!(sniff_image_format(&jpeg_sig), ImageFormat::Jpeg));
    }
}
