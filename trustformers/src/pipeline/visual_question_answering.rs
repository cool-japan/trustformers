//! # Visual question answering pipeline
//!
//! ## What is real here
//!
//! * **Image preprocessing** — [`ImageProcessor`] decodes real image bytes
//!   (Netpbm always; every format the pure-Rust `image` crate handles with the
//!   `vision` feature), resizes with bilinear interpolation and normalises with
//!   the configured per-channel mean/std into a `[1, 3, H, W]` tensor.
//! * **Feature pooling** — the global average pooling and patch splitting in
//!   [`VisualQuestionAnsweringPipeline::extract_global_features`] /
//!   `extract_patch_features`.
//! * **Fusion arithmetic** — [`FusionModule`]'s concatenation, element-wise and
//!   attention-style combinations.
//! * **Ranking** — [`VqaProcessor::score_answers`] softmaxes and sorts real
//!   logits.
//!
//! ## What is not available
//!
//! No vision-language model (ViLT, BLIP-2, LLaVA, …) is wired into this
//! pipeline, so every path that would produce an *answer* now returns a
//! structured [`TrustformersError::FeatureUnavailable`]:
//! [`VisualQuestionAnsweringPipeline::answer_question`], all four
//! [`AnswerGenerator`] strategies, [`AttentionVisualizer::visualize_attention`],
//! [`ReasoningEngine::generate_reasoning_chain`] and [`VqaPipeline::answer`].
//!
//! Previously those returned keyword-triggered answers ("what" → "An object or
//! scene element", "how many" → "2", "is"/"are" → "Yes"), two hardcoded
//! detected objects (a person and a red sedan), fixed attention matrices, and a
//! reasoning chain narrating steps that never ran. None of that survives.
//!
//! Use [`ImageProcessor::process_image`] and [`VqaProcessor`] to prepare inputs
//! for a real model, and [`VqaProcessor::score_answers`] to rank its logits.

use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::media::image_proc;
use crate::pipeline::{BasePipeline, Pipeline};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::Tensor;

/// Vision-language architectures wired into this pipeline.
///
/// Deliberately empty — the pipeline says so rather than pretending.
const SUPPORTED_ARCHITECTURES: &[&str] = &[];

/// Build the shared "no vision-language model" error.
fn vqa_unavailable(stage: &str) -> TrustformersError {
    let supported = if SUPPORTED_ARCHITECTURES.is_empty() {
        "none (no vision-language backbone is wired in yet)".to_string()
    } else {
        SUPPORTED_ARCHITECTURES.join(", ")
    };
    TrustformersError::FeatureUnavailable {
        message: format!(
            "{stage} requires a vision-language model; supported: {supported}. This pipeline \
             never returns keyword-triggered answers, hardcoded detections, or narrated \
             reasoning steps."
        ),
        feature: "vision-language-model".to_string(),
        suggestion: Some(
            "Preprocess with `ImageProcessor`/`VqaProcessor`, run your own model, and rank its \
             logits with `VqaProcessor::score_answers`."
                .to_string(),
        ),
        alternatives: Vec::new(),
    }
}

/// Configuration for visual question answering pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualQuestionAnsweringConfig {
    /// Maximum sequence length for question
    pub max_question_length: usize,
    /// Maximum sequence length for answer
    pub max_answer_length: usize,
    /// Image preprocessing configuration
    pub image_config: ImageConfig,
    /// Fusion strategy for combining vision and text
    pub fusion_strategy: FusionStrategy,
    /// Answer generation strategy
    pub answer_generation: AnswerGenerationStrategy,
    /// Confidence threshold for answers
    pub confidence_threshold: f32,
    /// Number of top answers to return
    pub top_k_answers: usize,
    /// Enable attention visualization
    pub enable_attention_viz: bool,
    /// Enable reasoning chain output
    pub enable_reasoning: bool,
}

impl Default for VisualQuestionAnsweringConfig {
    fn default() -> Self {
        Self {
            max_question_length: 512,
            max_answer_length: 256,
            image_config: ImageConfig::default(),
            fusion_strategy: FusionStrategy::CrossAttention,
            answer_generation: AnswerGenerationStrategy::Generative,
            confidence_threshold: 0.1,
            top_k_answers: 5,
            enable_attention_viz: false,
            enable_reasoning: false,
        }
    }
}

/// Image preprocessing configuration
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageConfig {
    /// Target image size
    pub image_size: (u32, u32),
    /// Normalization parameters
    pub normalize_mean: [f32; 3],
    pub normalize_std: [f32; 3],
    /// Enable data augmentation
    pub enable_augmentation: bool,
    /// Patch size for vision transformer
    pub patch_size: Option<u32>,
    /// Number of patches
    pub num_patches: Option<usize>,
}

impl Default for ImageConfig {
    fn default() -> Self {
        Self {
            image_size: (224, 224),
            normalize_mean: [0.485, 0.456, 0.406],
            normalize_std: [0.229, 0.224, 0.225],
            enable_augmentation: false,
            patch_size: Some(16),
            num_patches: Some(196), // 14x14 patches for 224x224 image with 16x16 patches
        }
    }
}

/// Fusion strategy for combining vision and text modalities
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Cross-attention between vision and text
    #[default]
    CrossAttention,
    /// Concatenation of vision and text features
    Concatenation,
    /// Element-wise addition
    Addition,
    /// Bilinear pooling
    BilinearPooling,
    /// Transformer-based fusion
    TransformerFusion,
    /// Graph-based fusion
    GraphFusion,
}

/// Answer generation strategy
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum AnswerGenerationStrategy {
    /// Generative approach (generate answer tokens)
    #[default]
    Generative,
    /// Extractive approach (extract from pre-defined answers)
    Extractive,
    /// Classification approach (classify into answer categories)
    Classification,
    /// Hybrid approach (combine multiple strategies)
    Hybrid,
}

/// Input for visual question answering pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualQuestionAnsweringInput {
    /// Input image (as bytes or tensor)
    pub image: ImageInput,
    /// Question about the image
    pub question: String,
    /// Optional context or constraints
    pub context: Option<String>,
    /// Optional answer candidates (for extractive QA)
    pub answer_candidates: Option<Vec<String>>,
    /// Optional metadata
    pub metadata: Option<HashMap<String, String>>,
}

/// Image input formats
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ImageInput {
    /// Raw image bytes
    Bytes(Vec<u8>),
    /// Image tensor (RGB format)
    Tensor(Vec<f32>),
    /// File path to image
    Path(String),
    /// URL to image
    Url(String),
    /// Base64 encoded image
    Base64(String),
}

/// Output from visual question answering pipeline
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VisualQuestionAnsweringOutput {
    /// Primary answer
    pub answer: String,
    /// Confidence score for the answer
    pub confidence: f32,
    /// Alternative answers with scores
    pub alternative_answers: Vec<AnswerCandidate>,
    /// Attention visualization data
    pub attention_visualization: Option<AttentionVisualization>,
    /// Reasoning chain
    pub reasoning_chain: Option<Vec<ReasoningStep>>,
    /// Image features used
    pub image_features: Option<ImageFeatures>,
    /// Processing metadata
    pub metadata: ProcessingMetadata,
}

/// Answer candidate with confidence score
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AnswerCandidate {
    /// Answer text
    pub answer: String,
    /// Confidence score (0.0 to 1.0)
    pub confidence: f32,
    /// Supporting evidence
    pub evidence: Option<String>,
    /// Bounding box in image (if applicable)
    pub bbox: Option<BoundingBox>,
}

/// Bounding box for visual grounding
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundingBox {
    /// X coordinate (normalized 0-1)
    pub x: f32,
    /// Y coordinate (normalized 0-1)
    pub y: f32,
    /// Width (normalized 0-1)
    pub width: f32,
    /// Height (normalized 0-1)
    pub height: f32,
    /// Confidence score
    pub confidence: f32,
}

/// Attention visualization data
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionVisualization {
    /// Attention weights between question tokens and image patches
    pub cross_attention_weights: Vec<Vec<f32>>,
    /// Self-attention weights in question
    pub question_self_attention: Vec<Vec<f32>>,
    /// Visual attention heatmap
    pub visual_attention_heatmap: Vec<f32>,
    /// Attention head information
    pub attention_heads: Vec<AttentionHead>,
}

/// Information about attention heads
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct AttentionHead {
    /// Head index
    pub head_id: usize,
    /// Layer index
    pub layer_id: usize,
    /// Attention pattern description
    pub pattern_type: String,
    /// Average attention score
    pub avg_attention: f32,
}

/// Reasoning step in the reasoning chain
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ReasoningStep {
    /// Step description
    pub description: String,
    /// Step type
    pub step_type: ReasoningStepType,
    /// Confidence in this step
    pub confidence: f32,
    /// Supporting evidence
    pub evidence: Option<String>,
    /// Visual grounding
    pub grounding: Option<BoundingBox>,
}

/// Types of reasoning steps
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum ReasoningStepType {
    /// Object detection
    ObjectDetection,
    /// Spatial reasoning
    SpatialReasoning,
    /// Counting
    Counting,
    /// Attribute recognition
    AttributeRecognition,
    /// Relationship reasoning
    RelationshipReasoning,
    /// Temporal reasoning
    TemporalReasoning,
    /// Causal reasoning
    CausalReasoning,
    /// Logical inference
    LogicalInference,
}

/// Image features extracted from the image
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ImageFeatures {
    /// Global image features
    pub global_features: Vec<f32>,
    /// Patch-level features
    pub patch_features: Vec<Vec<f32>>,
    /// Detected objects
    pub detected_objects: Vec<DetectedObject>,
    /// Scene description
    pub scene_description: Option<String>,
    /// Image classification
    pub image_classification: Option<Vec<ClassificationResult>>,
}

/// Detected object in the image
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DetectedObject {
    /// Object class
    pub class: String,
    /// Detection confidence
    pub confidence: f32,
    /// Bounding box
    pub bbox: BoundingBox,
    /// Object attributes
    pub attributes: Option<HashMap<String, String>>,
}

/// Classification result
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ClassificationResult {
    /// Class label
    pub label: String,
    /// Confidence score
    pub confidence: f32,
}

/// Processing metadata
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProcessingMetadata {
    /// Processing time in milliseconds
    pub processing_time_ms: u64,
    /// Model used
    pub model_name: String,
    /// Configuration used
    pub config: String,
    /// Number of tokens processed
    pub tokens_processed: usize,
    /// Memory usage
    pub memory_usage_mb: Option<f32>,
}

/// Visual Question Answering pipeline implementation
pub struct VisualQuestionAnsweringPipeline<M, T>
where
    M: Model + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    base: BasePipeline<M, T>,
    config: VisualQuestionAnsweringConfig,
    image_processor: ImageProcessor,
    fusion_module: FusionModule,
    answer_generator: AnswerGenerator,
    attention_visualizer: Option<AttentionVisualizer>,
    reasoning_engine: Option<ReasoningEngine>,
}

impl<M, T> VisualQuestionAnsweringPipeline<M, T>
where
    M: Model<Input = Tensor, Output = Tensor> + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    /// Create a new visual question answering pipeline
    pub fn new(model: M, tokenizer: T) -> Result<Self> {
        let base = BasePipeline::new(model, tokenizer);
        let config = VisualQuestionAnsweringConfig::default();
        let image_processor = ImageProcessor::new(config.image_config.clone())?;
        let fusion_module = FusionModule::new(config.fusion_strategy.clone())?;
        let answer_generator = AnswerGenerator::new(config.answer_generation.clone())?;

        Ok(Self {
            base,
            config,
            image_processor,
            fusion_module,
            answer_generator,
            attention_visualizer: None,
            reasoning_engine: None,
        })
    }

    /// Set configuration
    pub fn with_config(mut self, config: VisualQuestionAnsweringConfig) -> Self {
        self.config = config;
        self
    }

    /// Set fusion strategy
    pub fn with_fusion_strategy(mut self, strategy: FusionStrategy) -> Result<Self> {
        self.config.fusion_strategy = strategy.clone();
        self.fusion_module = FusionModule::new(strategy)?;
        Ok(self)
    }

    /// Set answer generation strategy
    pub fn with_answer_generation(mut self, strategy: AnswerGenerationStrategy) -> Result<Self> {
        self.config.answer_generation = strategy.clone();
        self.answer_generator = AnswerGenerator::new(strategy)?;
        Ok(self)
    }

    /// Enable attention visualization
    pub fn with_attention_visualization(mut self, enable: bool) -> Self {
        self.config.enable_attention_viz = enable;
        if enable && self.attention_visualizer.is_none() {
            self.attention_visualizer = Some(AttentionVisualizer::new());
        }
        self
    }

    /// Enable reasoning chain output
    pub fn with_reasoning(mut self, enable: bool) -> Self {
        self.config.enable_reasoning = enable;
        if enable && self.reasoning_engine.is_none() {
            self.reasoning_engine = Some(ReasoningEngine::new());
        }
        self
    }

    /// Set confidence threshold
    pub fn with_confidence_threshold(mut self, threshold: f32) -> Self {
        self.config.confidence_threshold = threshold.clamp(0.0, 1.0);
        self
    }

    /// Set number of top answers to return
    pub fn with_top_k_answers(mut self, k: usize) -> Self {
        self.config.top_k_answers = k;
        self
    }

    /// Process visual question answering
    pub fn answer_question(
        &self,
        input: VisualQuestionAnsweringInput,
    ) -> Result<VisualQuestionAnsweringOutput> {
        let start_time = std::time::Instant::now();

        // Validate input
        if input.question.trim().is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "Question cannot be empty".to_string(),
            ));
        }

        // Process image
        let image_tensor = self.image_processor.process_image(&input.image)?;
        let image_features = self.extract_image_features(&image_tensor)?;

        // Process question
        let question_tokens = self.base.tokenizer.encode(&input.question)?;
        let question_ids_f32: Vec<f32> =
            question_tokens.input_ids.iter().map(|&x| x as f32).collect();
        let question_tensor =
            Tensor::from_vec(question_ids_f32, &[1, question_tokens.input_ids.len()])?;

        // Fuse vision and text features (real arithmetic over the real tensors).
        let _fused_features = self.fusion_module.fuse(&image_tensor, &question_tensor)?;

        // There is no vision-language model to answer with. Everything above ran
        // for real; this is where the pipeline honestly stops.
        let _ = start_time;
        Err(vqa_unavailable("visual question answering"))
    }

    /// Extract structured features from an image tensor.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::FeatureUnavailable`]: producing
    /// `detected_objects`, a `scene_description` or an `image_classification`
    /// needs a detector/captioner/classifier, none of which is wired in.
    /// Previously this returned a fixed person-and-red-sedan detection pair, the
    /// string "A scene containing various objects", and indoor/outdoor scores of
    /// 0.8/0.2 for every image.
    ///
    /// The real pooling helpers [`Self::extract_global_features`] and
    /// `extract_patch_features` remain available.
    pub fn extract_image_features(&self, _image_tensor: &Tensor) -> Result<ImageFeatures> {
        Err(vqa_unavailable("image feature extraction"))
    }

    /// Average-pool a flat image buffer into a fixed-size global descriptor.
    ///
    /// Real arithmetic over the supplied data, reusable with any backbone.
    pub fn extract_global_features(&self, image_data: &[f32]) -> Vec<f32> {
        // Simplified global feature extraction (average pooling)
        let chunk_size = 64; // Feature dimension
        let mut global_features = vec![0.0; chunk_size];

        for (i, &value) in image_data.iter().enumerate() {
            global_features[i % chunk_size] += value;
        }

        // Normalize
        let count = image_data.len() as f32 / chunk_size as f32;
        for feature in &mut global_features {
            *feature /= count;
        }

        global_features
    }

    /// Extract patch-level features
    /// Split a flat image buffer into fixed-size patch descriptors.
    ///
    /// Real arithmetic over the supplied data, reusable with any backbone.
    pub fn extract_patch_features(&self, image_data: &[f32]) -> Vec<Vec<f32>> {
        let patch_size = 64; // Feature dimension per patch
        let num_patches = self.config.image_config.num_patches.unwrap_or(196);

        let mut patch_features = Vec::new();

        for i in 0..num_patches {
            let start_idx = (i * patch_size) % image_data.len();
            let end_idx = ((i + 1) * patch_size).min(image_data.len());

            let patch = if end_idx > start_idx {
                image_data[start_idx..end_idx].to_vec()
            } else {
                vec![0.0; patch_size]
            };

            patch_features.push(patch);
        }

        patch_features
    }
}

impl<M, T> Pipeline for VisualQuestionAnsweringPipeline<M, T>
where
    M: Model<Input = Tensor, Output = Tensor> + Clone + Send + Sync + 'static,
    T: Tokenizer + Clone + Send + Sync + 'static,
{
    type Input = VisualQuestionAnsweringInput;
    type Output = VisualQuestionAnsweringOutput;

    fn __call__(&self, input: Self::Input) -> Result<Self::Output> {
        self.answer_question(input)
    }
}

/// Image processor for handling different image formats
pub struct ImageProcessor {
    config: ImageConfig,
}

impl ImageProcessor {
    pub fn new(config: ImageConfig) -> Result<Self> {
        Ok(Self { config })
    }

    pub fn process_image(&self, image: &ImageInput) -> Result<Tensor> {
        match image {
            ImageInput::Bytes(bytes) => self.process_image_bytes(bytes),
            ImageInput::Tensor(tensor_data) => self.process_tensor_data(tensor_data),
            ImageInput::Path(path) => self.process_image_path(path),
            ImageInput::Url(url) => self.process_image_url(url),
            ImageInput::Base64(base64) => self.process_base64_image(base64),
        }
    }

    /// Decode and preprocess real image bytes into a `[1, 3, H, W]` tensor.
    ///
    /// Netpbm always; with the `vision` feature every format the pure-Rust
    /// `image` crate handles. Resize is real bilinear interpolation; the
    /// per-channel mean/std come from the configuration.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::InvalidInput`] for undecodable bytes and
    /// [`TrustformersError::FeatureUnavailable`] when the format needs the
    /// `vision` feature. It never substitutes a synthetic ramp.
    pub fn process_image_bytes(&self, bytes: &[u8]) -> Result<Tensor> {
        if bytes.is_empty() {
            return Err(TrustformersError::invalid_input_simple(
                "image buffer is empty".to_string(),
            ));
        }
        let image = image_proc::decode_image_bytes(bytes)?;
        self.preprocess(&image)
    }

    /// Resize and normalise a decoded image into the model's input tensor.
    ///
    /// # Errors
    ///
    /// Propagates resize/normalisation errors, and rejects a configuration
    /// whose mean/std vectors are not three-element.
    pub fn preprocess(&self, image: &image_proc::RgbImage) -> Result<Tensor> {
        let (width, height) = self.config.image_size;
        let (w, h) = (width as usize, height as usize);
        let resized = image_proc::resize_bilinear(image, h, w)?;

        let to_array = |values: &[f32], what: &str| -> Result<[f32; 3]> {
            <[f32; 3]>::try_from(values).map_err(|_| {
                TrustformersError::invalid_input_simple(format!(
                    "image config `{what}` must have exactly 3 entries, got {}",
                    values.len()
                ))
            })
        };
        let mean = to_array(&self.config.normalize_mean, "normalize_mean")?;
        let std = to_array(&self.config.normalize_std, "normalize_std")?;
        let chw = image_proc::normalize_to_chw(&resized, mean, std)?;

        Tensor::from_vec(chw, &[1, 3, h, w]).map_err(Into::into)
    }

    /// Wrap already-normalised CHW float data in the model's input tensor.
    ///
    /// # Errors
    ///
    /// Returns an error when the buffer length does not match `3 * H * W`.
    pub fn process_tensor_data(&self, tensor_data: &[f32]) -> Result<Tensor> {
        let (width, height) = self.config.image_size;
        let (w, h) = (width as usize, height as usize);
        let expected = 3 * h * w;
        if tensor_data.len() != expected {
            return Err(TrustformersError::invalid_input_simple(format!(
                "tensor data has {} values but 3x{h}x{w} = {expected} were expected",
                tensor_data.len()
            )));
        }
        Tensor::from_vec(tensor_data.to_vec(), &[1, 3, h, w]).map_err(Into::into)
    }

    /// Decode and preprocess an image file from disk.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::Io`] when the file cannot be read, and
    /// whatever [`Self::process_image_bytes`] returns otherwise.
    pub fn process_image_path(&self, path: &str) -> Result<Tensor> {
        let image = image_proc::decode_image_file(path)?;
        self.preprocess(&image)
    }

    /// Fetch and preprocess an image from a URL.
    ///
    /// # Errors
    ///
    /// Always returns [`TrustformersError::FeatureUnavailable`]: this pipeline
    /// performs no network I/O. Download the bytes yourself and call
    /// [`Self::process_image_bytes`].
    pub fn process_image_url(&self, url: &str) -> Result<Tensor> {
        Err(TrustformersError::feature_unavailable(
            format!(
                "the VQA image processor does not fetch URLs (`{url}`); download the bytes and \
                 call `process_image_bytes`"
            ),
            "network-image-fetch",
        ))
    }

    /// Decode a base64 payload and preprocess it.
    ///
    /// # Errors
    ///
    /// Returns [`TrustformersError::InvalidInput`] for malformed base64 or
    /// undecodable image bytes. It never substitutes a synthetic tensor.
    pub fn process_base64_image(&self, encoded: &str) -> Result<Tensor> {
        use base64::Engine as _;
        let bytes =
            base64::engine::general_purpose::STANDARD.decode(encoded.trim()).map_err(|e| {
                TrustformersError::invalid_input_simple(format!(
                    "failed to decode base64 image: {e}"
                ))
            })?;
        self.process_image_bytes(&bytes)
    }
}

/// Fusion module for combining vision and text features
pub struct FusionModule {
    strategy: FusionStrategy,
}

impl FusionModule {
    pub fn new(strategy: FusionStrategy) -> Result<Self> {
        Ok(Self { strategy })
    }

    pub fn fuse(&self, image_tensor: &Tensor, question_tensor: &Tensor) -> Result<Tensor> {
        match self.strategy {
            FusionStrategy::CrossAttention => {
                self.cross_attention_fusion(image_tensor, question_tensor)
            },
            FusionStrategy::Concatenation => {
                self.concatenation_fusion(image_tensor, question_tensor)
            },
            FusionStrategy::Addition => self.addition_fusion(image_tensor, question_tensor),
            FusionStrategy::BilinearPooling => {
                self.bilinear_pooling_fusion(image_tensor, question_tensor)
            },
            FusionStrategy::TransformerFusion => {
                self.transformer_fusion(image_tensor, question_tensor)
            },
            FusionStrategy::GraphFusion => self.graph_fusion(image_tensor, question_tensor),
        }
    }

    fn cross_attention_fusion(
        &self,
        image_tensor: &Tensor,
        question_tensor: &Tensor,
    ) -> Result<Tensor> {
        // Simplified cross-attention fusion
        let image_data = image_tensor.data()?;
        let question_data = question_tensor.data()?;

        // Create attention weights (simplified)
        let attention_dim = image_data.len().min(question_data.len());
        let mut fused_data = Vec::with_capacity(attention_dim);

        for i in 0..attention_dim {
            let img_val = image_data[i % image_data.len()];
            let q_val = question_data[i % question_data.len()];
            fused_data.push(img_val * q_val);
        }

        Tensor::from_vec(fused_data, &[1, attention_dim]).map_err(Into::into)
    }

    fn concatenation_fusion(
        &self,
        image_tensor: &Tensor,
        question_tensor: &Tensor,
    ) -> Result<Tensor> {
        let mut fused_data = Vec::new();
        fused_data.extend(image_tensor.data()?);
        fused_data.extend(question_tensor.data()?);

        let fused_len = fused_data.len();
        Tensor::from_vec(fused_data, &[1, fused_len]).map_err(Into::into)
    }

    fn addition_fusion(&self, image_tensor: &Tensor, question_tensor: &Tensor) -> Result<Tensor> {
        let image_data = image_tensor.data()?;
        let question_data = question_tensor.data()?;

        let min_len = image_data.len().min(question_data.len());
        let fused_data: Vec<f32> = (0..min_len).map(|i| image_data[i] + question_data[i]).collect();

        Tensor::from_vec(fused_data, &[1, min_len]).map_err(Into::into)
    }

    fn bilinear_pooling_fusion(
        &self,
        image_tensor: &Tensor,
        question_tensor: &Tensor,
    ) -> Result<Tensor> {
        let image_data = image_tensor.data()?;
        let question_data = question_tensor.data()?;

        let output_dim = 256; // Fixed output dimension
        let mut fused_data = vec![0.0; output_dim];

        for i in 0..output_dim {
            let img_idx = i % image_data.len();
            let q_idx = i % question_data.len();
            fused_data[i] = image_data[img_idx] * question_data[q_idx];
        }

        Tensor::from_vec(fused_data, &[1, output_dim]).map_err(Into::into)
    }

    fn transformer_fusion(
        &self,
        image_tensor: &Tensor,
        question_tensor: &Tensor,
    ) -> Result<Tensor> {
        // Simplified transformer fusion
        self.cross_attention_fusion(image_tensor, question_tensor)
    }

    fn graph_fusion(&self, image_tensor: &Tensor, question_tensor: &Tensor) -> Result<Tensor> {
        // Simplified graph fusion
        self.concatenation_fusion(image_tensor, question_tensor)
    }
}

/// Answer generator for different generation strategies
pub struct AnswerGenerator {
    strategy: AnswerGenerationStrategy,
}

#[derive(Debug, Clone)]
pub struct AnswerOutput {
    pub answer: String,
    pub confidence: f32,
    pub alternatives: Vec<AnswerCandidate>,
}

impl AnswerGenerator {
    pub fn new(strategy: AnswerGenerationStrategy) -> Result<Self> {
        Ok(Self { strategy })
    }

    pub fn generate_answer(
        &self,
        features: &Tensor,
        question: &str,
        candidates: &Option<Vec<String>>,
        config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        match self.strategy {
            AnswerGenerationStrategy::Generative => {
                self.generative_answer(features, question, config)
            },
            AnswerGenerationStrategy::Extractive => {
                self.extractive_answer(features, question, candidates, config)
            },
            AnswerGenerationStrategy::Classification => {
                self.classification_answer(features, question, config)
            },
            AnswerGenerationStrategy::Hybrid => {
                self.hybrid_answer(features, question, candidates, config)
            },
        }
    }

    /// Free-form answer generation.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`]. Previously this
    /// matched keywords in the question — "what" → "An object or scene
    /// element", "how many" → "2", "is"/"are" → "Yes" — and paired them with
    /// "Alternative answer 1"/"2" and a confidence derived from the mean of the
    /// fused tensor.
    fn generative_answer(
        &self,
        _features: &Tensor,
        _question: &str,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        Err(vqa_unavailable("generative answer decoding"))
    }

    /// Span/candidate extraction.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`]. Previously this
    /// returned the first candidate with a fixed 0.8 confidence, and ranked the
    /// rest by their list position.
    fn extractive_answer(
        &self,
        _features: &Tensor,
        _question: &str,
        _candidates: &Option<Vec<String>>,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        Err(vqa_unavailable("extractive answer selection"))
    }

    /// Fixed-vocabulary classification.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`]. Previously this chose
    /// a class list from question keywords and scored it from tensor
    /// statistics.
    fn classification_answer(
        &self,
        _features: &Tensor,
        _question: &str,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        Err(vqa_unavailable("classification answer selection"))
    }

    /// Combined generative + extractive answering.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`].
    fn hybrid_answer(
        &self,
        _features: &Tensor,
        _question: &str,
        _candidates: &Option<Vec<String>>,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        Err(vqa_unavailable("hybrid answer generation"))
    }
}

/// Attention visualizer for generating attention maps
pub struct AttentionVisualizer;

impl Default for AttentionVisualizer {
    fn default() -> Self {
        Self::new()
    }
}

impl AttentionVisualizer {
    pub fn new() -> Self {
        Self
    }

    /// Produce attention maps for a forward pass.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`]: no model runs, so no
    /// attention weights exist to report. Previously this returned three fixed
    /// 4-wide cross-attention rows, an identity-ish self-attention matrix, a
    /// linear ramp "heatmap" and two invented attention heads.
    pub fn visualize_attention(
        &self,
        _features: &Tensor,
        _image_tensor: &Tensor,
        _question_tensor: &Tensor,
    ) -> Result<AttentionVisualization> {
        Err(vqa_unavailable("attention visualization"))
    }
}

/// Reasoning engine for generating reasoning chains
pub struct ReasoningEngine;

impl Default for ReasoningEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl ReasoningEngine {
    pub fn new() -> Self {
        Self
    }

    /// Produce the chain of reasoning that led to an answer.
    ///
    /// # Errors
    ///
    /// Always [`TrustformersError::FeatureUnavailable`]: no model reasons, so
    /// there is no chain to report. Previously this emitted keyword-selected
    /// steps ("Detecting objects in the image", "Counting detected objects")
    /// with fixed confidences and invented grounding boxes, narrating work that
    /// never happened.
    pub fn generate_reasoning_chain(
        &self,
        _question: &str,
        _answer: &str,
        _image_features: &ImageFeatures,
    ) -> Result<Vec<ReasoningStep>> {
        Err(vqa_unavailable("reasoning chain generation"))
    }
}

// ---------------------------------------------------------------------------
// VQA simplified types and processor (standalone, no Model trait required)
// ---------------------------------------------------------------------------

/// Simplified VQA image input
#[derive(Clone, Debug)]
pub struct VqaImageInput {
    /// Raw pixel bytes (RGB, row-major)
    pub pixels: Vec<u8>,
    /// Image width
    pub width: usize,
    /// Image height
    pub height: usize,
}

/// Simplified VQA input pairing an image with a question
#[derive(Clone, Debug)]
pub struct VqaInput {
    pub image: VqaImageInput,
    pub question: String,
}

/// A single VQA answer with its confidence and vocabulary index
#[derive(Clone, Debug)]
pub struct VqaResult {
    pub answer: String,
    pub score: f32,
    pub answer_id: usize,
}

/// Configuration for the simplified VQA processor
#[derive(Clone, Debug)]
pub struct VqaConfig {
    pub model_id: String,
    pub max_answer_length: usize,
    pub top_k: usize,
    pub image_size: usize,
}

impl Default for VqaConfig {
    fn default() -> Self {
        Self {
            model_id: "dandelin/vilt-b32-finetuned-vqa".to_string(),
            max_answer_length: 30,
            top_k: 5,
            image_size: 384,
        }
    }
}

/// Errors emitted by the simplified VQA components
#[derive(Debug, thiserror::Error)]
pub enum PipelineError {
    #[error("Empty question")]
    EmptyQuestion,
    #[error("Empty image")]
    EmptyImage,
    #[error("Empty answer vocabulary")]
    EmptyVocabulary,
    /// No vision-language model is available to answer with.
    #[error(
        "no vision-language model is wired in for `{requested}`; this processor never derives \
         answers from hashes of the question and image — encode the inputs, run your own model, \
         and rank its logits with `score_answers`"
    )]
    NoModel {
        /// The model id the caller configured.
        requested: String,
    },
}

/// Lightweight VQA processor (no model backend required)
pub struct VqaProcessor {
    /// Simple word-to-id vocabulary built on first use
    vocab: std::collections::HashMap<String, u32>,
}

impl VqaProcessor {
    /// Construct a processor with an empty vocabulary
    pub fn new() -> Self {
        Self {
            vocab: std::collections::HashMap::new(),
        }
    }

    /// Word-level tokenization via vocabulary lookup.
    ///
    /// Unknown words receive a deterministic hash-based id.
    pub fn encode_question(&self, question: &str) -> Vec<u32> {
        question
            .split_whitespace()
            .map(|word| {
                let lower = word.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase();
                if let Some(&id) = self.vocab.get(&lower) {
                    id
                } else {
                    // djb2 fallback for unknown words, clamped to u32
                    let mut h: u64 = 5381;
                    for b in lower.bytes() {
                        h = h.wrapping_mul(33).wrapping_add(b as u64);
                    }
                    (h % 30_000) as u32 + 1
                }
            })
            .collect()
    }

    /// Flatten pixel bytes to f32, then channel-wise normalise to [0, 1].
    ///
    /// Returns a flat `Vec<f32>` of length `pixels.len()`.
    pub fn encode_image_features(image: &VqaImageInput) -> Vec<f32> {
        if image.pixels.is_empty() {
            return Vec::new();
        }
        // Channel-wise min/max normalisation (per-image)
        let as_f32: Vec<f32> = image.pixels.iter().map(|&b| b as f32 / 255.0).collect();
        // Compute per-channel mean and std for normalization (ImageNet stats)
        let means = [0.485_f32, 0.456, 0.406];
        let stds = [0.229_f32, 0.224, 0.225];
        let num_channels = 3;
        as_f32
            .iter()
            .enumerate()
            .map(|(i, &v)| {
                let ch = i % num_channels;
                (v - means[ch]) / stds[ch]
            })
            .collect()
    }

    /// Concatenate text and image feature vectors, then L2-normalise the result.
    pub fn combine_modalities(text_features: &[f32], image_features: &[f32]) -> Vec<f32> {
        let mut combined: Vec<f32> =
            text_features.iter().chain(image_features.iter()).copied().collect();
        let norm: f32 = combined.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 1e-8 {
            for v in &mut combined {
                *v /= norm;
            }
        }
        combined
    }
}

impl Default for VqaProcessor {
    fn default() -> Self {
        Self::new()
    }
}

/// Standalone VQA pipeline that works without a model backend.
pub struct VisualQaPipeline {
    pub config: VqaConfig,
    processor: VqaProcessor,
}

impl VisualQaPipeline {
    pub fn new(config: VqaConfig) -> std::result::Result<Self, PipelineError> {
        Ok(Self {
            config,
            processor: VqaProcessor::new(),
        })
    }

    /// Score logits against an answer vocabulary and return ranked `VqaResult`s.
    pub fn score_answers(logits: &[f32], answer_vocab: &[String]) -> Vec<VqaResult> {
        // Softmax over logits
        let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exps: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
        let sum_exp: f32 = exps.iter().sum();
        let probs: Vec<f32> = if sum_exp > 1e-8 {
            exps.iter().map(|e| e / sum_exp).collect()
        } else {
            vec![1.0 / logits.len() as f32; logits.len()]
        };

        let mut results: Vec<VqaResult> = probs
            .iter()
            .enumerate()
            .zip(answer_vocab.iter())
            .map(|((id, &score), answer)| VqaResult {
                answer: answer.clone(),
                score,
                answer_id: id,
            })
            .collect();

        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        results
    }

    /// Answer a single VQA query.
    ///
    /// # Errors
    ///
    /// Validation errors first, then [`PipelineError::NoModel`]: this processor
    /// has no vision-language backbone. Previously it derived "logits" from a
    /// djb2 hash of the question's words combined with the image's pixel mean,
    /// and softmaxed them into confident-looking answer probabilities.
    ///
    /// Use [`VqaProcessor::encode_question`],
    /// [`VqaProcessor::encode_image_features`] and
    /// [`VqaPipeline::score_answers`] around your own model instead.
    pub fn answer(
        &self,
        input: VqaInput,
        answer_vocab: &[String],
    ) -> std::result::Result<Vec<VqaResult>, PipelineError> {
        if input.question.trim().is_empty() {
            return Err(PipelineError::EmptyQuestion);
        }
        if input.image.pixels.is_empty() {
            return Err(PipelineError::EmptyImage);
        }
        if answer_vocab.is_empty() {
            return Err(PipelineError::EmptyVocabulary);
        }
        Err(PipelineError::NoModel {
            requested: self.config.model_id.clone(),
        })
    }

    /// Answer a batch of VQA queries.
    pub fn answer_batch(
        &self,
        inputs: Vec<VqaInput>,
        answer_vocab: &[String],
    ) -> std::result::Result<Vec<Vec<VqaResult>>, PipelineError> {
        inputs.into_iter().map(|inp| self.answer(inp, answer_vocab)).collect()
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::{Model, TokenizedInput, Tokenizer};
    use crate::AutoConfig;
    use trustformers_core::Tensor;

    #[derive(Clone)]
    struct MockModel {
        config: AutoConfig,
    }

    impl MockModel {
        fn new() -> Self {
            MockModel {
                config: {
                    // Try each available config in order
                    #[cfg(feature = "bert")]
                    {
                        AutoConfig::Bert(Default::default())
                    }
                    #[cfg(all(not(feature = "bert"), feature = "roberta"))]
                    {
                        AutoConfig::Roberta(Default::default())
                    }
                    #[cfg(all(not(feature = "bert"), not(feature = "roberta"), feature = "gpt2"))]
                    {
                        AutoConfig::Gpt2(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        feature = "gpt_neo"
                    ))]
                    {
                        AutoConfig::GptNeo(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        feature = "gpt_j"
                    ))]
                    {
                        AutoConfig::GptJ(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        not(feature = "gpt_j"),
                        feature = "t5"
                    ))]
                    {
                        AutoConfig::T5(Default::default())
                    }
                    #[cfg(all(
                        not(feature = "bert"),
                        not(feature = "roberta"),
                        not(feature = "gpt2"),
                        not(feature = "gpt_neo"),
                        not(feature = "gpt_j"),
                        not(feature = "t5"),
                        feature = "albert"
                    ))]
                    {
                        AutoConfig::Albert(Default::default())
                    }
                    #[cfg(not(any(
                        feature = "bert",
                        feature = "roberta",
                        feature = "gpt2",
                        feature = "gpt_neo",
                        feature = "gpt_j",
                        feature = "t5",
                        feature = "albert"
                    )))]
                    {
                        // If no model features are enabled, we need to enable at least one for testing
                        // Since this is a test context, we'll compile-fail rather than panic at runtime
                        compile_error!("At least one model feature must be enabled for tests (bert, roberta, gpt2, gpt_neo, gpt_j, t5, or albert)")
                    }
                },
            }
        }
    }

    impl Model for MockModel {
        type Input = Tensor;
        type Output = Tensor;
        type Config = AutoConfig;

        fn forward(&self, _input: Self::Input) -> trustformers_core::errors::Result<Self::Output> {
            // Return a dummy tensor for testing
            Tensor::zeros(&[1, 10])
        }

        fn num_parameters(&self) -> usize {
            1000 // Mock parameter count
        }

        fn load_pretrained(
            &mut self,
            _reader: &mut dyn std::io::Read,
        ) -> trustformers_core::errors::Result<()> {
            Ok(()) // Mock implementation
        }

        fn get_config(&self) -> &Self::Config {
            &self.config
        }
    }

    #[derive(Clone)]
    struct MockTokenizer;

    impl MockTokenizer {
        fn new() -> Self {
            MockTokenizer
        }
    }

    impl Tokenizer for MockTokenizer {
        fn encode(&self, _text: &str) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput {
                input_ids: vec![1, 2, 3], // Mock token IDs
                attention_mask: vec![1, 1, 1],
                token_type_ids: Some(vec![0, 0, 0]),
                offset_mapping: None,
                special_tokens_mask: None,
                overflowing_tokens: None,
            })
        }

        fn encode_pair(
            &self,
            _text_a: &str,
            _text_b: &str,
        ) -> trustformers_core::errors::Result<TokenizedInput> {
            Ok(TokenizedInput {
                input_ids: vec![1, 2, 3, 4, 5], // Mock token IDs for pair
                attention_mask: vec![1, 1, 1, 1, 1],
                token_type_ids: Some(vec![0, 0, 0, 1, 1]),
                offset_mapping: None,
                special_tokens_mask: None,
                overflowing_tokens: None,
            })
        }

        fn decode(&self, _token_ids: &[u32]) -> trustformers_core::errors::Result<String> {
            Ok("mock decoded text".to_string())
        }

        fn vocab_size(&self) -> usize {
            1000
        }

        fn get_vocab(&self) -> std::collections::HashMap<String, u32> {
            let mut vocab = std::collections::HashMap::new();
            vocab.insert("test".to_string(), 1);
            vocab.insert("mock".to_string(), 2);
            vocab.insert("token".to_string(), 3);
            vocab
        }

        fn token_to_id(&self, token: &str) -> Option<u32> {
            match token {
                "test" => Some(1),
                "mock" => Some(2),
                "token" => Some(3),
                _ => None,
            }
        }

        fn id_to_token(&self, id: u32) -> Option<String> {
            match id {
                1 => Some("test".to_string()),
                2 => Some("mock".to_string()),
                3 => Some("token".to_string()),
                _ => None,
            }
        }
    }

    #[test]
    fn test_vqa_pipeline_creation() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = VisualQuestionAnsweringPipeline::new(model, tokenizer);
        assert!(pipeline.is_ok());
    }

    #[test]
    fn test_vqa_config() {
        let config = VisualQuestionAnsweringConfig::default();
        assert_eq!(config.max_question_length, 512);
        assert_eq!(config.max_answer_length, 256);
        assert_eq!(config.top_k_answers, 5);
    }

    #[test]
    fn test_image_processor() {
        let config = ImageConfig::default();
        let processor = ImageProcessor::new(config).expect("operation failed in test");
        let image = ImageInput::Tensor(vec![0.5; 224 * 224 * 3]);
        let result = processor.process_image(&image);
        assert!(result.is_ok());
    }

    #[test]
    fn test_fusion_strategies() {
        let fusion =
            FusionModule::new(FusionStrategy::Concatenation).expect("operation failed in test");
        let img_tensor =
            Tensor::from_vec(vec![1.0, 2.0, 3.0], &[1, 3]).expect("tensor operation failed");
        let q_tensor = Tensor::from_vec(vec![4.0, 5.0], &[1, 2]).expect("tensor operation failed");
        let result = fusion.fuse(&img_tensor, &q_tensor);
        assert!(result.is_ok());
    }

    #[test]
    fn test_answer_generator_reports_missing_model() {
        // Regression: every strategy used to return a keyword-triggered answer
        // ("What …" -> "An object or scene element") with an invented
        // confidence and two "Alternative answer N" candidates.
        let config = VisualQuestionAnsweringConfig::default();
        let features =
            Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).expect("tensor operation failed");
        for strategy in [
            AnswerGenerationStrategy::Generative,
            AnswerGenerationStrategy::Extractive,
            AnswerGenerationStrategy::Classification,
            AnswerGenerationStrategy::Hybrid,
        ] {
            let generator = AnswerGenerator::new(strategy).expect("generator");
            match generator.generate_answer(&features, "What is in the image?", &None, &config) {
                Err(TrustformersError::FeatureUnavailable { feature, .. }) => {
                    assert_eq!(feature, "vision-language-model");
                },
                other => panic!("expected FeatureUnavailable, got {other:?}"),
            }
        }
    }

    #[test]
    fn test_reasoning_engine_reports_missing_model() {
        // Regression: this narrated "Detecting objects in the image" and
        // "Counting detected objects" with fixed confidences and invented
        // grounding boxes, for work that never ran.
        let engine = ReasoningEngine::new();
        let image_features = ImageFeatures {
            global_features: vec![0.1, 0.2, 0.3],
            patch_features: vec![],
            detected_objects: vec![],
            scene_description: None,
            image_classification: None,
        };
        assert!(matches!(
            engine.generate_reasoning_chain("How many people are there?", "2", &image_features),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    #[test]
    fn test_attention_visualizer_reports_missing_model() {
        // Regression: this returned fixed 4-wide attention rows, a linear-ramp
        // "heatmap" and two invented attention heads without any model running.
        let visualizer = AttentionVisualizer::new();
        let features =
            Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).expect("tensor operation failed");
        let image_tensor =
            Tensor::from_vec(vec![0.5; 100], &[1, 100]).expect("tensor operation failed");
        let question_tensor =
            Tensor::from_vec(vec![0.3; 50], &[1, 50]).expect("tensor operation failed");
        assert!(matches!(
            visualizer.visualize_attention(&features, &image_tensor, &question_tensor),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    #[test]
    fn test_pipeline_configuration() {
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = VisualQuestionAnsweringPipeline::new(model, tokenizer)
            .expect("operation failed in test")
            .with_fusion_strategy(FusionStrategy::CrossAttention)
            .expect("operation failed in test")
            .with_answer_generation(AnswerGenerationStrategy::Classification)
            .expect("operation failed in test")
            .with_confidence_threshold(0.5)
            .with_top_k_answers(3);

        assert!(matches!(
            pipeline.config.fusion_strategy,
            FusionStrategy::CrossAttention
        ));
        assert!(matches!(
            pipeline.config.answer_generation,
            AnswerGenerationStrategy::Classification
        ));
        assert_eq!(pipeline.config.confidence_threshold, 0.5);
        assert_eq!(pipeline.config.top_k_answers, 3);
    }

    // -----------------------------------------------------------------------
    // VqaProcessor / VisualQaPipeline tests (15+)
    // -----------------------------------------------------------------------

    fn dummy_image(width: usize, height: usize) -> VqaImageInput {
        VqaImageInput {
            pixels: (0..(width * height * 3)).map(|i| (i % 256) as u8).collect(),
            width,
            height,
        }
    }

    fn default_vocab() -> Vec<String> {
        vec![
            "yes".to_string(),
            "no".to_string(),
            "dog".to_string(),
            "cat".to_string(),
            "2".to_string(),
            "3".to_string(),
            "red".to_string(),
            "blue".to_string(),
        ]
    }

    fn default_vqa_pipeline() -> VisualQaPipeline {
        VisualQaPipeline::new(VqaConfig::default()).expect("pipeline creation ok")
    }

    // 1. VqaInput construction
    #[test]
    fn test_vqa_input_construction() {
        let img = dummy_image(4, 4);
        let input = VqaInput {
            image: img.clone(),
            question: "What color is the object?".to_string(),
        };
        assert_eq!(input.question, "What color is the object?");
        assert_eq!(input.image.width, 4);
        assert_eq!(input.image.height, 4);
        assert_eq!(input.image.pixels.len(), 4 * 4 * 3);
    }

    // 2. VqaConfig defaults
    #[test]
    fn test_vqa_config_defaults() {
        let cfg = VqaConfig::default();
        assert_eq!(cfg.top_k, 5);
        assert_eq!(cfg.image_size, 384);
        assert!(cfg.max_answer_length > 0);
        assert!(!cfg.model_id.is_empty());
    }

    // 3. encode_question returns non-empty tokens for a non-empty question
    #[test]
    fn test_encode_question_nonempty() {
        let proc = VqaProcessor::new();
        let tokens = proc.encode_question("What is in the image?");
        assert!(!tokens.is_empty());
    }

    // 4. encode_question: token count == word count
    #[test]
    fn test_encode_question_token_count() {
        let proc = VqaProcessor::new();
        let question = "how many cats are there";
        let tokens = proc.encode_question(question);
        // Every whitespace-separated word should yield one token
        assert_eq!(tokens.len(), question.split_whitespace().count());
    }

    // 5. encode_question: empty string yields empty tokens
    #[test]
    fn test_encode_question_empty() {
        let proc = VqaProcessor::new();
        let tokens = proc.encode_question("");
        assert!(tokens.is_empty());
    }

    // 6. encode_image_features: correct output length
    #[test]
    fn test_encode_image_features_length() {
        let img = dummy_image(8, 8);
        let feats = VqaProcessor::encode_image_features(&img);
        assert_eq!(feats.len(), 8 * 8 * 3);
    }

    // 7. encode_image_features: empty image yields empty output
    #[test]
    fn test_encode_image_features_empty() {
        let empty = VqaImageInput {
            pixels: vec![],
            width: 0,
            height: 0,
        };
        let feats = VqaProcessor::encode_image_features(&empty);
        assert!(feats.is_empty());
    }

    // 8. encode_image_features: all-zero image gives normalised values
    #[test]
    fn test_encode_image_features_normalised() {
        let img = VqaImageInput {
            pixels: vec![128u8; 6 * 3],
            width: 6,
            height: 1,
        };
        let feats = VqaProcessor::encode_image_features(&img);
        // None should be NaN or infinite
        for &v in &feats {
            assert!(v.is_finite(), "feature value {v} is not finite");
        }
    }

    // 9. combine_modalities: output length equals sum of input lengths
    #[test]
    fn test_combine_modalities_length() {
        let text = vec![0.1_f32, 0.2, 0.3];
        let image = vec![0.4_f32, 0.5, 0.6, 0.7];
        let combined = VqaProcessor::combine_modalities(&text, &image);
        assert_eq!(combined.len(), text.len() + image.len());
    }

    // 10. combine_modalities: result is L2-normalised
    #[test]
    fn test_combine_modalities_normalised() {
        let text = vec![1.0_f32, 2.0, 3.0];
        let image = vec![4.0_f32, 5.0];
        let combined = VqaProcessor::combine_modalities(&text, &image);
        let norm: f32 = combined.iter().map(|v| v * v).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}, expected ~1.0");
    }

    // 11. combine_modalities: all-zero inputs don't panic
    #[test]
    fn test_combine_modalities_zeros() {
        let combined = VqaProcessor::combine_modalities(&[0.0; 4], &[0.0; 3]);
        assert_eq!(combined.len(), 7);
        // All should remain zero (no division)
        for &v in &combined {
            assert_eq!(v, 0.0);
        }
    }

    // 12. score_answers: softmax sums to ~1
    #[test]
    fn test_score_answers_probability_sum() {
        let logits = vec![1.0_f32, 2.0, 0.5, 3.0];
        let vocab: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let results = VisualQaPipeline::score_answers(&logits, &vocab);
        let total: f32 = results.iter().map(|r| r.score).sum();
        assert!(
            (total - 1.0).abs() < 1e-5,
            "scores sum to {total}, expected 1.0"
        );
    }

    // 13. score_answers: results are sorted descending by score
    #[test]
    fn test_score_answers_sorted() {
        let logits = vec![0.1_f32, 5.0, 2.0, 0.5];
        let vocab: Vec<String> = ["a", "b", "c", "d"].iter().map(|s| s.to_string()).collect();
        let results = VisualQaPipeline::score_answers(&logits, &vocab);
        for i in 1..results.len() {
            assert!(
                results[i - 1].score >= results[i].score,
                "scores not sorted at index {i}: {} < {}",
                results[i - 1].score,
                results[i].score
            );
        }
    }

    // 14. score_answers: answer_id matches vocabulary index
    #[test]
    fn test_score_answers_answer_id() {
        let logits = vec![1.0_f32, 2.0, 3.0];
        let vocab: Vec<String> = ["cat", "dog", "bird"].iter().map(|s| s.to_string()).collect();
        let results = VisualQaPipeline::score_answers(&logits, &vocab);
        // The highest logit is at index 2 ("bird") — it should be first
        assert_eq!(results[0].answer, "bird");
        assert_eq!(results[0].answer_id, 2);
    }

    // 15. answer: returns Ok with correct number of results
    #[test]
    fn test_vqa_pipeline_answer_reports_missing_model() {
        // Regression: `answer` used to build "logits" from a djb2 hash of the
        // question's words mixed with the image's pixel mean, and softmax them
        // into confident answer probabilities.
        let pipeline = default_vqa_pipeline();
        let input = VqaInput {
            image: dummy_image(16, 16),
            question: "What is this?".to_string(),
        };
        match pipeline.answer(input, &default_vocab()) {
            Err(PipelineError::NoModel { requested }) => {
                assert_eq!(requested, pipeline.config.model_id);
            },
            other => panic!("expected NoModel, got {other:?}"),
        }
    }

    // 16. answer: empty question returns error
    #[test]
    fn test_vqa_pipeline_empty_question_error() {
        let pipeline = default_vqa_pipeline();
        let input = VqaInput {
            image: dummy_image(8, 8),
            question: "   ".to_string(),
        };
        let err = pipeline.answer(input, &default_vocab()).unwrap_err();
        assert!(matches!(err, PipelineError::EmptyQuestion));
    }

    // 17. answer: empty image returns error
    #[test]
    fn test_vqa_pipeline_empty_image_error() {
        let pipeline = default_vqa_pipeline();
        let input = VqaInput {
            image: VqaImageInput {
                pixels: vec![],
                width: 0,
                height: 0,
            },
            question: "Is there anything?".to_string(),
        };
        let err = pipeline.answer(input, &default_vocab()).unwrap_err();
        assert!(matches!(err, PipelineError::EmptyImage));
    }

    // 18. answer: empty vocabulary returns error
    #[test]
    fn test_vqa_pipeline_empty_vocab_error() {
        let pipeline = default_vqa_pipeline();
        let input = VqaInput {
            image: dummy_image(4, 4),
            question: "What is this?".to_string(),
        };
        let err = pipeline.answer(input, &[]).unwrap_err();
        assert!(matches!(err, PipelineError::EmptyVocabulary));
    }

    // 19. answer_batch: returns one Vec<VqaResult> per input
    #[test]
    fn test_vqa_pipeline_answer_batch_reports_missing_model() {
        let pipeline = default_vqa_pipeline();
        let vocab = default_vocab();
        let inputs: Vec<VqaInput> = (0..3)
            .map(|i| VqaInput {
                image: dummy_image(4 + i, 4 + i),
                question: format!("Question {i}?"),
            })
            .collect();
        assert!(matches!(
            pipeline.answer_batch(inputs, &vocab),
            Err(PipelineError::NoModel { .. })
        ));
    }

    // 20. score_answers: single-answer vocabulary scores to 1.0
    #[test]
    fn test_score_answers_single_vocab() {
        let logits = vec![0.5_f32];
        let vocab = vec!["yes".to_string()];
        let results = VisualQaPipeline::score_answers(&logits, &vocab);
        assert_eq!(results.len(), 1);
        assert!((results[0].score - 1.0).abs() < 1e-5);
    }

    // 21. top_k config is respected
    #[test]
    fn test_score_answers_ranks_real_logits() {
        // The ranking half stays real and usable with your own model's logits.
        let vocab = default_vocab(); // 8 entries
        let mut logits = vec![0.0f32; vocab.len()];
        logits[3] = 4.0;
        let results = VisualQaPipeline::score_answers(&logits, &vocab);
        assert_eq!(results.len(), vocab.len());
        assert_eq!(results[0].answer_id, 3);
        for w in results.windows(2) {
            assert!(w[0].score >= w[1].score, "scores must be sorted descending");
        }
        let sum: f32 = results.iter().map(|r| r.score).sum();
        assert!((sum - 1.0).abs() < 1e-5, "probabilities sum to {sum}");
    }

    // ---- Real image preprocessing regressions ----------------------------

    /// Build a binary P6 PPM fixture with a distinctive gradient.
    fn ppm_fixture(h: usize, w: usize, flip: bool) -> Vec<u8> {
        let mut bytes = format!("P6\n{w} {h}\n255\n").into_bytes();
        for y in 0..h {
            for x in 0..w {
                let (r, g) = if flip {
                    ((y * 255 / h.max(1)) as u8, (x * 255 / w.max(1)) as u8)
                } else {
                    ((x * 255 / w.max(1)) as u8, (y * 255 / h.max(1)) as u8)
                };
                bytes.extend_from_slice(&[r, g, 32u8]);
            }
        }
        bytes
    }

    fn small_processor() -> ImageProcessor {
        ImageProcessor::new(ImageConfig {
            image_size: (8, 8),
            ..Default::default()
        })
        .expect("processor")
    }

    #[test]
    fn test_process_image_bytes_decodes_real_pixels() {
        // Regression: this used to ignore `bytes` entirely and build a linear
        // ramp `(i / size - mean) / std` as the "preprocessed image".
        let processor = small_processor();
        let tensor = processor
            .process_image_bytes(&ppm_fixture(16, 16, false))
            .expect("decode + preprocess");
        assert_eq!(tensor.shape(), vec![1, 3, 8, 8]);
        let values = tensor.to_vec_f32().expect("values");
        assert!(values.iter().any(|&v| v != 0.0));

        let flipped = processor
            .process_image_bytes(&ppm_fixture(16, 16, true))
            .expect("decode + preprocess")
            .to_vec_f32()
            .expect("values");
        assert_ne!(
            values, flipped,
            "different images must preprocess differently"
        );
    }

    #[test]
    fn test_process_image_bytes_rejects_garbage() {
        let processor = small_processor();
        assert!(processor.process_image_bytes(&[]).is_err());
        assert!(processor.process_image_bytes(&[0u8; 64]).is_err());
    }

    #[test]
    fn test_process_image_path_reads_a_real_file() {
        let processor = small_processor();
        let path = std::env::temp_dir().join("trustformers-vqa-fixture.ppm");
        std::fs::write(&path, ppm_fixture(12, 12, false)).expect("write fixture");
        let tensor = processor.process_image_path(&path.to_string_lossy()).expect("decode");
        let _ = std::fs::remove_file(&path);
        assert_eq!(tensor.shape(), vec![1, 3, 8, 8]);
    }

    #[test]
    fn test_process_image_path_reports_missing_file() {
        let processor = small_processor();
        let path = std::env::temp_dir().join("trustformers-vqa-missing.ppm");
        let _ = std::fs::remove_file(&path);
        assert!(processor.process_image_path(&path.to_string_lossy()).is_err());
    }

    #[test]
    fn test_process_base64_image_round_trips() {
        use base64::Engine as _;
        let processor = small_processor();
        let encoded = base64::engine::general_purpose::STANDARD.encode(ppm_fixture(10, 10, false));
        let tensor = processor.process_base64_image(&encoded).expect("decode");
        assert_eq!(tensor.shape(), vec![1, 3, 8, 8]);
        assert!(processor.process_base64_image("!!!not base64!!!").is_err());
    }

    #[test]
    fn test_process_image_url_reports_no_network() {
        let processor = small_processor();
        assert!(matches!(
            processor.process_image_url("https://example.invalid/cat.png"),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    #[test]
    fn test_process_tensor_data_validates_length() {
        let processor = small_processor();
        assert!(processor.process_tensor_data(&vec![0.5f32; 3 * 8 * 8]).is_ok());
        assert!(processor.process_tensor_data(&[0.5f32; 10]).is_err());
    }

    #[test]
    fn test_extract_image_features_reports_missing_model() {
        // Regression: this returned a hardcoded person + red sedan detection,
        // the caption "A scene containing various objects", and indoor/outdoor
        // scores of 0.8/0.2 for every image.
        let model = MockModel::new();
        let tokenizer = MockTokenizer::new();
        let pipeline = VisualQuestionAnsweringPipeline::new(model, tokenizer).expect("pipeline");
        let tensor = Tensor::from_vec(vec![0.5f32; 12], &[1, 3, 2, 2]).expect("tensor");
        assert!(matches!(
            pipeline.extract_image_features(&tensor),
            Err(TrustformersError::FeatureUnavailable { .. })
        ));
    }

    // 22. encode_image_features: larger images give proportionally larger feature vectors
    #[test]
    fn test_encode_image_features_scale_with_size() {
        let small = dummy_image(4, 4);
        let large = dummy_image(8, 8);
        let f_small = VqaProcessor::encode_image_features(&small);
        let f_large = VqaProcessor::encode_image_features(&large);
        assert_eq!(f_large.len(), 4 * f_small.len());
    }
}
