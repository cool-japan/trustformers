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

/// Fusion strategy for combining different modalities. See
/// [`FusionLayer::fuse`] for what each variant genuinely computes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Simple concatenation of features
    Concatenation,
    /// Element-wise addition
    Addition,
    /// Weighted average with fixed per-modality weights
    WeightedAverage,
    /// Real dot-product cross-attention (the same formula
    /// [`MultiModalPipeline::compute_attention_weights`] uses): each
    /// modality attends over every other present modality and the
    /// attention-weighted combination is averaged across modality pairs.
    /// Not a *trained* attention head (this workspace has none wired into
    /// this generic pipeline), but genuinely computed from the real
    /// extracted features, not an alias for a different strategy.
    CrossAttention,
    /// Real, content-derived gating: each modality's contribution is
    /// scaled by `sigmoid(mean(that modality's feature vector))` before
    /// being summed. Not a *learned* gate (no trained gating network is
    /// wired into this workspace), but a genuine per-position, per-modality
    /// gate computed from the real feature values -- unlike a fixed
    /// per-modality weight, it varies with the actual content.
    GatedFusion,
    /// Real self-attention over the concatenated modality sequence,
    /// followed by a residual add and mean-pool -- the two structural
    /// pieces ("attention block" + "residual connection") that
    /// characterise a transformer encoder layer. Not a full trained
    /// transformer stack (no learned feed-forward/projection weights exist
    /// in this generic pipeline), but genuinely self-attentive over the
    /// real fused sequence.
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

    /// Compute attention weights between two modalities. Thin wrapper
    /// around the free function [`dot_product_attention_weights`], which
    /// [`FusionLayer::cross_attention_fusion`] also uses -- kept as an
    /// infallible free function (this computation can never fail) shared
    /// by both call sites rather than duplicated.
    fn compute_attention_weights(
        &self,
        query_features: &[Vec<f32>],
        key_features: &[Vec<f32>],
    ) -> Result<Vec<Vec<f32>>> {
        Ok(dot_product_attention_weights(query_features, key_features))
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
/// short final remainder, matching how `auto::types::FeatureOutput::shape`
/// already describes the layout as `[n_frames, feature_size]`).
fn chunk_features(flat: Vec<f32>, chunk_size: usize) -> Vec<Vec<f32>> {
    if chunk_size == 0 {
        return Vec::new();
    }
    flat.chunks_exact(chunk_size).map(|chunk| chunk.to_vec()).collect()
}

/// Scaled dot-product attention, softmax-normalized: for each vector in
/// `query_features`, scores every vector in `key_features` by
/// `exp(dot(query, key) / sqrt(query.len()))`, then normalizes each query's
/// scores to sum to `1.0`. Shared by
/// [`MultiModalPipeline::compute_attention_weights`] (used for the
/// diagnostic [`AttentionWeights`] this pipeline reports) and
/// [`FusionLayer::cross_attention_fusion`] (which actually folds these
/// weights into the fused feature vector, rather than only reporting them).
///
/// Infallible: `query_features`/`key_features` being empty simply yields an
/// empty (or all-zero-length) result rather than an error.
fn dot_product_attention_weights(
    query_features: &[Vec<f32>],
    key_features: &[Vec<f32>],
) -> Vec<Vec<f32>> {
    let mut attention_weights = Vec::with_capacity(query_features.len());

    for query in query_features {
        let mut scores = Vec::with_capacity(key_features.len());
        for key in key_features {
            let dot_product: f32 = query.iter().zip(key.iter()).map(|(q, k)| q * k).sum();
            // Scaled dot-product attention score (pre-softmax), scaled by
            // sqrt(query dimension) as in "Attention Is All You Need" --
            // keeps the softmax input from growing with feature width.
            scores.push(dot_product / (query.len() as f32).sqrt());
        }

        // Numerically-stable softmax: subtract the row max before
        // exponentiating. Without this, a real (not toy-sized) feature
        // vector easily produces a scaled score in the hundreds --
        // `768`-wide vectors of magnitude `3.0` score around `249` here --
        // and `f32::exp` overflows to `Infinity` well before that,
        // collapsing every weight to `Infinity / Infinity = NaN`.
        // Subtracting the max keeps the largest exponent at `exp(0) = 1`
        // and produces the exact same normalized weights (softmax is
        // shift-invariant), just without ever overflowing.
        let max_score = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut query_weights: Vec<f32> = if max_score.is_finite() {
            scores.iter().map(|&s| (s - max_score).exp()).collect()
        } else {
            // `key_features` was empty (no scores at all): nothing to
            // weight over.
            scores.iter().map(|_| 0.0).collect()
        };

        let sum: f32 = query_weights.iter().sum();
        if sum > 0.0 {
            query_weights.iter_mut().for_each(|w| *w /= sum);
        }

        attention_weights.push(query_weights);
    }

    attention_weights
}

/// Standard logistic sigmoid, `1 / (1 + e^-x)`, mapping any real `x` into
/// `(0, 1)`. Used by [`FusionLayer::gated_fusion`] to turn a modality row's
/// raw mean activation into a bounded gate value.
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// Full self-attention over `sequence` (every row attends over every row,
/// itself included, via [`dot_product_attention_weights`]), then mean-pools
/// the per-row attended results into one fused vector. When `residual` is
/// `true`, each row's original (pre-attention) values are added back before
/// pooling (`x + Attention(x)`) -- the residual connection a transformer
/// encoder layer applies around its attention block.
///
/// Returns `None` for an empty `sequence` (nothing to fuse -- the caller
/// skips this position's output entirely, matching how every other fusion
/// strategy in this module omits a position with no contributing modality
/// rather than emitting an all-zero vector).
fn mean_pooled_self_attention(sequence: &[Vec<f32>], residual: bool) -> Option<Vec<f32>> {
    let dim = sequence.first()?.len();
    if dim == 0 {
        return None;
    }

    let attention_weights = dot_product_attention_weights(sequence, sequence);

    let mut pooled = vec![0.0f32; dim];
    for (row_index, weights) in attention_weights.iter().enumerate() {
        let mut attended = vec![0.0f32; dim];
        for (value_row, &weight) in sequence.iter().zip(weights.iter()) {
            for (a, v) in attended.iter_mut().zip(value_row.iter()) {
                *a += v * weight;
            }
        }
        if residual {
            for (a, original) in attended.iter_mut().zip(sequence[row_index].iter()) {
                *a += original;
            }
        }
        for (p, a) in pooled.iter_mut().zip(attended.iter()) {
            *p += a;
        }
    }

    let n = sequence.len() as f32;
    pooled.iter_mut().for_each(|p| *p /= n);
    Some(pooled)
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
            ]
            .into_iter()
            .flatten()
            {
                if i < modality_features.len() && modality_features[i].len() >= common_dim {
                    for j in 0..common_dim {
                        combined_feature[j] += modality_features[i][j];
                    }
                    count += 1;
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

    /// The modality vectors present at sequence position `i` that reach
    /// `common_dim`, in a fixed (text, image, audio, video) order. Shared
    /// gather step for [`Self::cross_attention_fusion`],
    /// [`Self::gated_fusion`], and [`Self::transformer_fusion`], which all
    /// need "every present modality's row at this position" as a small
    /// token sequence to attend/gate over.
    fn present_features_at(
        &self,
        features: &ModalityFeatures,
        i: usize,
        common_dim: usize,
    ) -> Vec<Vec<f32>> {
        [
            &features.text_features,
            &features.image_features,
            &features.audio_features,
            &features.video_features,
        ]
        .into_iter()
        .flatten()
        .filter(|modality_features| {
            i < modality_features.len() && modality_features[i].len() >= common_dim
        })
        .map(|modality_features| modality_features[i][..common_dim].to_vec())
        .collect()
    }

    /// Real dot-product cross-attention across the modalities present at
    /// each position: builds the small token sequence of present-modality
    /// vectors via [`Self::present_features_at`], computes full
    /// self-attention over it with [`dot_product_attention_weights`] (every
    /// modality attends over every modality present, itself included), then
    /// mean-pools the per-query attended rows into this position's fused
    /// vector. With only one modality present at a position, self-attention
    /// over a length-1 sequence has a single softmax weight of exactly
    /// `1.0`, so the output is that modality's own (unattended) vector --
    /// there is nothing else to cross-attend against.
    fn cross_attention_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
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

        let mut fused_features = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let sequence = self.present_features_at(features, i, common_dim);
            if let Some(fused) = mean_pooled_self_attention(&sequence, false) {
                fused_features.push(fused);
            }
        }
        Ok(fused_features)
    }

    /// Real, content-derived gated fusion: at each position, every present
    /// modality's row is scaled by `sigmoid(mean(row))` -- a real gate
    /// computed from that row's own values, not a fixed per-modality
    /// constant the way [`Self::weighted_average_features`] uses -- then
    /// summed and normalized by the sum of gates. Not a *trained* gating
    /// network (none is wired into this generic pipeline), but a genuine
    /// per-position, content-varying gate rather than an alias for a
    /// different strategy.
    fn gated_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
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

        let mut fused_features = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let sequence = self.present_features_at(features, i, common_dim);
            let mut combined = vec![0.0f32; common_dim];
            let mut total_gate = 0.0f32;
            for row in &sequence {
                let mean_activation = row.iter().sum::<f32>() / row.len() as f32;
                let gate = sigmoid(mean_activation);
                for (c, v) in combined.iter_mut().zip(row.iter()) {
                    *c += v * gate;
                }
                total_gate += gate;
            }
            if total_gate > 0.0 {
                combined.iter_mut().for_each(|x| *x /= total_gate);
                fused_features.push(combined);
            }
        }
        Ok(fused_features)
    }

    /// Real self-attention with a residual connection: the same
    /// per-position self-attention as [`Self::cross_attention_fusion`], but
    /// with each attended row added back to its own pre-attention row
    /// (`x + Attention(x)`) before mean-pooling -- the "attention block +
    /// residual connection" structural pattern that defines a transformer
    /// encoder layer, applied to the real fused sequence. Not a full
    /// trained transformer stack (no learned feed-forward/projection
    /// weights exist in this generic pipeline), but genuinely self-
    /// attentive with a real residual, not an alias for plain concatenation.
    fn transformer_fusion(&self, features: &ModalityFeatures) -> Result<Vec<Vec<f32>>> {
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

        let mut fused_features = Vec::with_capacity(max_len);
        for i in 0..max_len {
            let sequence = self.present_features_at(features, i, common_dim);
            if let Some(fused) = mean_pooled_self_attention(&sequence, true) {
                fused_features.push(fused);
            }
        }
        Ok(fused_features)
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
#[path = "multimodal_tests.rs"]
mod tests;
