use crate::core::traits::{Model, Tokenizer};
use crate::error::{Result, TrustformersError};
use crate::pipeline::{BasePipeline, Pipeline};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trustformers_core::Tensor;

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

        // Fuse vision and text features
        let fused_features = self.fusion_module.fuse(&image_tensor, &question_tensor)?;

        // Generate answer
        let answer_output = self.answer_generator.generate_answer(
            &fused_features,
            &input.question,
            &input.answer_candidates,
            &self.config,
        )?;

        // Extract attention visualization if enabled
        let attention_visualization = if self.config.enable_attention_viz {
            self.attention_visualizer
                .as_ref()
                .map(|viz| {
                    viz.visualize_attention(&fused_features, &image_tensor, &question_tensor)
                })
                .transpose()?
        } else {
            None
        };

        // Generate reasoning chain if enabled
        let reasoning_chain = if self.config.enable_reasoning {
            self.reasoning_engine
                .as_ref()
                .map(|engine| {
                    engine.generate_reasoning_chain(
                        &input.question,
                        &answer_output.answer,
                        &image_features,
                    )
                })
                .transpose()?
        } else {
            None
        };

        let processing_time = start_time.elapsed().as_millis() as u64;

        Ok(VisualQuestionAnsweringOutput {
            answer: answer_output.answer,
            confidence: answer_output.confidence,
            alternative_answers: answer_output.alternatives,
            attention_visualization,
            reasoning_chain,
            image_features: Some(image_features),
            metadata: ProcessingMetadata {
                processing_time_ms: processing_time,
                model_name: "vqa-model".to_string(),
                config: serde_json::to_string(&self.config).unwrap_or_default(),
                tokens_processed: question_tokens.input_ids.len(),
                memory_usage_mb: None,
            },
        })
    }

    /// Extract features from image tensor
    fn extract_image_features(&self, image_tensor: &Tensor) -> Result<ImageFeatures> {
        // Run image through model to get features
        let image_output = self.base.model.forward(image_tensor.clone())?;
        let image_data = image_output.data()?;

        // Extract global features (average pooling)
        let global_features = self.extract_global_features(&image_data);

        // Extract patch features
        let patch_features = self.extract_patch_features(&image_data);

        // Simulate object detection
        let detected_objects = self.simulate_object_detection();

        // Generate scene description
        let scene_description = Some("A scene containing various objects".to_string());

        // Generate image classification
        let image_classification = Some(vec![
            ClassificationResult {
                label: "indoor".to_string(),
                confidence: 0.8,
            },
            ClassificationResult {
                label: "outdoor".to_string(),
                confidence: 0.2,
            },
        ]);

        Ok(ImageFeatures {
            global_features,
            patch_features,
            detected_objects,
            scene_description,
            image_classification,
        })
    }

    /// Extract global image features
    fn extract_global_features(&self, image_data: &[f32]) -> Vec<f32> {
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
    fn extract_patch_features(&self, image_data: &[f32]) -> Vec<Vec<f32>> {
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

    /// Simulate object detection (placeholder)
    fn simulate_object_detection(&self) -> Vec<DetectedObject> {
        vec![
            DetectedObject {
                class: "person".to_string(),
                confidence: 0.9,
                bbox: BoundingBox {
                    x: 0.2,
                    y: 0.3,
                    width: 0.3,
                    height: 0.6,
                    confidence: 0.9,
                },
                attributes: Some(
                    [
                        ("age".to_string(), "adult".to_string()),
                        ("gender".to_string(), "unknown".to_string()),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                ),
            },
            DetectedObject {
                class: "car".to_string(),
                confidence: 0.8,
                bbox: BoundingBox {
                    x: 0.6,
                    y: 0.4,
                    width: 0.3,
                    height: 0.3,
                    confidence: 0.8,
                },
                attributes: Some(
                    [
                        ("color".to_string(), "red".to_string()),
                        ("type".to_string(), "sedan".to_string()),
                    ]
                    .iter()
                    .cloned()
                    .collect(),
                ),
            },
        ]
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

    fn process_image_bytes(&self, _bytes: &[u8]) -> Result<Tensor> {
        // Placeholder: decode image and preprocess
        let (width, height) = self.config.image_size;
        let channels = 3;
        let size = (width * height * channels) as usize;

        // Create normalized tensor
        let data: Vec<f32> = (0..size)
            .map(|i| {
                (i as f32 / size as f32 - self.config.normalize_mean[i % 3])
                    / self.config.normalize_std[i % 3]
            })
            .collect();

        Tensor::from_vec(
            data,
            &[1, channels as usize, height as usize, width as usize],
        )
        .map_err(Into::into)
    }

    fn process_tensor_data(&self, tensor_data: &[f32]) -> Result<Tensor> {
        let (width, height) = self.config.image_size;
        let channels = 3;

        // Normalize the tensor data
        let normalized_data: Vec<f32> = tensor_data
            .iter()
            .enumerate()
            .map(|(i, &val)| {
                (val - self.config.normalize_mean[i % 3]) / self.config.normalize_std[i % 3]
            })
            .collect();

        Tensor::from_vec(
            normalized_data,
            &[1, channels, height as usize, width as usize],
        )
        .map_err(Into::into)
    }

    fn process_image_path(&self, _path: &str) -> Result<Tensor> {
        // Placeholder: load image from path
        self.process_image_bytes(&[])
    }

    fn process_image_url(&self, _url: &str) -> Result<Tensor> {
        // Placeholder: download and process image
        self.process_image_bytes(&[])
    }

    fn process_base64_image(&self, _base64: &str) -> Result<Tensor> {
        // Placeholder: decode base64 and process
        self.process_image_bytes(&[])
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

    fn generative_answer(
        &self,
        features: &Tensor,
        question: &str,
        config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        // Simplified generative answer
        let answer = if question.to_lowercase().contains("what") {
            "An object or scene element"
        } else if question.to_lowercase().contains("how many") {
            "2"
        } else if question.to_lowercase().contains("where") {
            "In the center of the image"
        } else if question.to_lowercase().contains("who") {
            "A person"
        } else if question.to_lowercase().contains("when") {
            "During the day"
        } else if question.to_lowercase().contains("why") {
            "Due to the context of the scene"
        } else if question.to_lowercase().contains("is") || question.to_lowercase().contains("are")
        {
            "Yes"
        } else {
            "I cannot determine the answer from the image"
        };

        let features_data = features.data()?;
        let confidence =
            0.7 + (features_data.iter().sum::<f32>() / features_data.len() as f32).abs() * 0.3;
        let confidence = confidence.clamp(0.0, 1.0);

        let alternatives = vec![
            AnswerCandidate {
                answer: "Alternative answer 1".to_string(),
                confidence: confidence * 0.8,
                evidence: Some("Based on visual features".to_string()),
                bbox: None,
            },
            AnswerCandidate {
                answer: "Alternative answer 2".to_string(),
                confidence: confidence * 0.6,
                evidence: Some("Based on question context".to_string()),
                bbox: None,
            },
        ];

        Ok(AnswerOutput {
            answer: answer.to_string(),
            confidence,
            alternatives,
        })
    }

    fn extractive_answer(
        &self,
        _features: &Tensor,
        _question: &str,
        candidates: &Option<Vec<String>>,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        let default_candidates = vec![
            "yes".to_string(),
            "no".to_string(),
            "person".to_string(),
            "car".to_string(),
            "building".to_string(),
        ];
        let candidates = candidates.as_ref().unwrap_or(&default_candidates);

        let answer = candidates.first().unwrap_or(&"unknown".to_string()).clone();
        let confidence = 0.8;

        let alternatives = candidates
            .iter()
            .enumerate()
            .map(|(i, candidate)| AnswerCandidate {
                answer: candidate.clone(),
                confidence: 0.9 - (i as f32 * 0.1),
                evidence: Some("Extracted from candidates".to_string()),
                bbox: None,
            })
            .collect();

        Ok(AnswerOutput {
            answer,
            confidence,
            alternatives,
        })
    }

    fn classification_answer(
        &self,
        features: &Tensor,
        question: &str,
        _config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        let classes = if question.to_lowercase().contains("color") {
            vec!["red", "blue", "green", "yellow", "black", "white"]
        } else if question.to_lowercase().contains("animal") {
            vec!["cat", "dog", "bird", "horse", "cow", "sheep"]
        } else {
            vec!["yes", "no", "maybe"]
        };

        let feature_sum = features.data()?.iter().sum::<f32>();
        let class_idx = (feature_sum.abs() as usize) % classes.len();
        let answer = classes[class_idx].to_string();
        let confidence = 0.8;

        let alternatives = classes
            .iter()
            .enumerate()
            .map(|(i, &class)| AnswerCandidate {
                answer: class.to_string(),
                confidence: if i == class_idx { confidence } else { confidence * 0.5 },
                evidence: Some("Classification result".to_string()),
                bbox: None,
            })
            .collect();

        Ok(AnswerOutput {
            answer,
            confidence,
            alternatives,
        })
    }

    fn hybrid_answer(
        &self,
        features: &Tensor,
        question: &str,
        candidates: &Option<Vec<String>>,
        config: &VisualQuestionAnsweringConfig,
    ) -> Result<AnswerOutput> {
        // Combine multiple strategies
        let generative_result = self.generative_answer(features, question, config)?;
        let classification_result = self.classification_answer(features, question, config)?;

        let answer = if generative_result.confidence > classification_result.confidence {
            generative_result.answer
        } else {
            classification_result.answer
        };

        let confidence = (generative_result.confidence + classification_result.confidence) / 2.0;

        let mut alternatives = generative_result.alternatives;
        alternatives.extend(classification_result.alternatives);
        alternatives.sort_by(|a, b| {
            b.confidence.partial_cmp(&a.confidence).unwrap_or(std::cmp::Ordering::Equal)
        });
        alternatives.truncate(config.top_k_answers);

        Ok(AnswerOutput {
            answer,
            confidence,
            alternatives,
        })
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

    pub fn visualize_attention(
        &self,
        _features: &Tensor,
        _image_tensor: &Tensor,
        _question_tensor: &Tensor,
    ) -> Result<AttentionVisualization> {
        // Placeholder attention visualization
        let cross_attention_weights = vec![
            vec![0.1, 0.2, 0.3, 0.4],
            vec![0.2, 0.3, 0.4, 0.1],
            vec![0.3, 0.4, 0.1, 0.2],
        ];

        let question_self_attention = vec![
            vec![0.8, 0.1, 0.1],
            vec![0.1, 0.8, 0.1],
            vec![0.1, 0.1, 0.8],
        ];

        let visual_attention_heatmap = (0..196).map(|i| (i as f32 / 196.0) * 0.5 + 0.5).collect();

        let attention_heads = vec![
            AttentionHead {
                head_id: 0,
                layer_id: 0,
                pattern_type: "object-focused".to_string(),
                avg_attention: 0.7,
            },
            AttentionHead {
                head_id: 1,
                layer_id: 0,
                pattern_type: "spatial-reasoning".to_string(),
                avg_attention: 0.6,
            },
        ];

        Ok(AttentionVisualization {
            cross_attention_weights,
            question_self_attention,
            visual_attention_heatmap,
            attention_heads,
        })
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

    pub fn generate_reasoning_chain(
        &self,
        question: &str,
        answer: &str,
        _image_features: &ImageFeatures,
    ) -> Result<Vec<ReasoningStep>> {
        let mut reasoning_steps = Vec::new();

        // Analyze question type and generate appropriate reasoning steps
        if question.to_lowercase().contains("how many") {
            reasoning_steps.push(ReasoningStep {
                description: "Detecting objects in the image".to_string(),
                step_type: ReasoningStepType::ObjectDetection,
                confidence: 0.9,
                evidence: Some("Multiple objects detected".to_string()),
                grounding: Some(BoundingBox {
                    x: 0.1,
                    y: 0.1,
                    width: 0.8,
                    height: 0.8,
                    confidence: 0.8,
                }),
            });

            reasoning_steps.push(ReasoningStep {
                description: "Counting detected objects".to_string(),
                step_type: ReasoningStepType::Counting,
                confidence: 0.8,
                evidence: Some(format!("Counted objects to determine answer: {}", answer)),
                grounding: None,
            });
        } else if question.to_lowercase().contains("where") {
            reasoning_steps.push(ReasoningStep {
                description: "Analyzing spatial relationships".to_string(),
                step_type: ReasoningStepType::SpatialReasoning,
                confidence: 0.8,
                evidence: Some("Located object position in image".to_string()),
                grounding: Some(BoundingBox {
                    x: 0.3,
                    y: 0.3,
                    width: 0.4,
                    height: 0.4,
                    confidence: 0.7,
                }),
            });
        } else if question.to_lowercase().contains("what") {
            reasoning_steps.push(ReasoningStep {
                description: "Identifying objects and attributes".to_string(),
                step_type: ReasoningStepType::AttributeRecognition,
                confidence: 0.9,
                evidence: Some("Recognized object attributes".to_string()),
                grounding: None,
            });
        }

        // Add final inference step
        reasoning_steps.push(ReasoningStep {
            description: format!("Concluding that the answer is: {}", answer),
            step_type: ReasoningStepType::LogicalInference,
            confidence: 0.7,
            evidence: Some("Based on visual analysis and reasoning".to_string()),
            grounding: None,
        });

        Ok(reasoning_steps)
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

    /// Answer a single VQA query (stub: uses feature hashing for deterministic answers).
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

        let q_tokens = self.processor.encode_question(&input.question);
        let img_feats = VqaProcessor::encode_image_features(&input.image);

        // Derive pseudo-logits from feature hashes to produce deterministic scores
        let q_text_feats: Vec<f32> = q_tokens.iter().map(|&t| t as f32 / 30_000.0).collect();
        let combined = VqaProcessor::combine_modalities(&q_text_feats, &img_feats);

        let logits: Vec<f32> = answer_vocab
            .iter()
            .enumerate()
            .map(|(i, _)| {
                let seed = combined.get(i % combined.len().max(1)).copied().unwrap_or(0.0);
                seed + (i as f32 * 0.01)
            })
            .collect();

        let mut results = Self::score_answers(&logits, answer_vocab);
        results.truncate(self.config.top_k);
        Ok(results)
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
    fn test_answer_generator() {
        let generator = AnswerGenerator::new(AnswerGenerationStrategy::Generative)
            .expect("operation failed in test");
        let features =
            Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).expect("tensor operation failed");
        let config = VisualQuestionAnsweringConfig::default();
        let result = generator.generate_answer(&features, "What is in the image?", &None, &config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_reasoning_engine() {
        let engine = ReasoningEngine::new();
        let image_features = ImageFeatures {
            global_features: vec![0.1, 0.2, 0.3],
            patch_features: vec![],
            detected_objects: vec![],
            scene_description: None,
            image_classification: None,
        };
        let result =
            engine.generate_reasoning_chain("How many people are there?", "2", &image_features);
        assert!(result.is_ok());
        assert!(!result.expect("operation failed in test").is_empty());
    }

    #[test]
    fn test_attention_visualizer() {
        let visualizer = AttentionVisualizer::new();
        let features =
            Tensor::from_vec(vec![0.1, 0.2, 0.3], &[1, 3]).expect("tensor operation failed");
        let image_tensor =
            Tensor::from_vec(vec![0.5; 100], &[1, 100]).expect("tensor operation failed");
        let question_tensor =
            Tensor::from_vec(vec![0.3; 50], &[1, 50]).expect("tensor operation failed");
        let result = visualizer.visualize_attention(&features, &image_tensor, &question_tensor);
        assert!(result.is_ok());
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
    fn test_vqa_pipeline_answer_ok() {
        let pipeline = default_vqa_pipeline();
        let input = VqaInput {
            image: dummy_image(16, 16),
            question: "What is this?".to_string(),
        };
        let vocab = default_vocab();
        let results = pipeline.answer(input, &vocab).expect("answer failed");
        assert!(!results.is_empty());
        assert!(results.len() <= pipeline.config.top_k);
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
    fn test_vqa_pipeline_answer_batch_count() {
        let pipeline = default_vqa_pipeline();
        let vocab = default_vocab();
        let inputs: Vec<VqaInput> = (0..3)
            .map(|i| VqaInput {
                image: dummy_image(4 + i, 4 + i),
                question: format!("Question {i}?"),
            })
            .collect();
        let batch_results = pipeline.answer_batch(inputs, &vocab).expect("batch failed");
        assert_eq!(batch_results.len(), 3);
        for results in &batch_results {
            assert!(!results.is_empty());
        }
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
    fn test_vqa_pipeline_top_k_respected() {
        let pipeline = VisualQaPipeline::new(VqaConfig {
            top_k: 2,
            ..Default::default()
        })
        .expect("ok");
        let vocab = default_vocab(); // 8 entries
        let input = VqaInput {
            image: dummy_image(4, 4),
            question: "What animal?".to_string(),
        };
        let results = pipeline.answer(input, &vocab).expect("answer ok");
        assert!(results.len() <= 2);
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
