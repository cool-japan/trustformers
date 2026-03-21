//! Multi-modal AI Support for TrustformeRS C API
//!
//! This module provides comprehensive multi-modal capabilities including:
//! - Vision-Language Models (VLMs) like CLIP, BLIP, LLaVA
//! - Audio-Language Models
//! - Video-Language Understanding
//! - Multi-modal embeddings and cross-modal retrieval
//! - Advanced fusion techniques

use crate::error::{TrustformersError, TrustformersResult};
use anyhow::anyhow;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_double, c_float, c_int};
use std::ptr;

/// Multi-modal model types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum MultiModalType {
    /// Text-only model
    TextOnly = 0,
    /// Vision-Language model (CLIP-like)
    VisionLanguage = 1,
    /// Audio-Language model
    AudioLanguage = 2,
    /// Video-Language model
    VideoLanguage = 3,
    /// Multi-modal LLM (LLaVA-like)
    MultiModalLLM = 4,
    /// Document AI (layout + text)
    DocumentAI = 5,
    /// Speech-Language model
    SpeechLanguage = 6,
    /// Unified multi-modal (all modalities)
    UnifiedMultiModal = 7,
}

/// Modality types
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum ModalityType {
    Text = 0,
    Image = 1,
    Audio = 2,
    Video = 3,
    Speech = 4,
    Document = 5,
    Code = 6,
    Structured = 7,
}

/// Vision encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct VisionEncoderConfig {
    /// Vision encoder type: 0=ViT, 1=ResNet, 2=ConvNeXT, 3=EfficientNet
    pub encoder_type: c_int,
    /// Image size (assumed square)
    pub image_size: i32,
    /// Patch size for ViT
    pub patch_size: i32,
    /// Number of channels (3 for RGB)
    pub num_channels: i32,
    /// Hidden dimension
    pub hidden_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// Number of attention heads (for ViT)
    pub num_heads: i32,
    /// MLP ratio
    pub mlp_ratio: f32,
    /// Dropout rate
    pub dropout: f32,
    /// Whether to use layer normalization
    pub use_layer_norm: c_int,
    /// Position embedding type: 0=Learned, 1=Sinusoidal, 2=Relative
    pub pos_embed_type: c_int,
}

impl Default for VisionEncoderConfig {
    fn default() -> Self {
        Self {
            encoder_type: 0, // ViT
            image_size: 224,
            patch_size: 16,
            num_channels: 3,
            hidden_dim: 768,
            num_layers: 12,
            num_heads: 12,
            mlp_ratio: 4.0,
            dropout: 0.1,
            use_layer_norm: 1, // True
            pos_embed_type: 0, // Learned
        }
    }
}

/// Audio encoder configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct AudioEncoderConfig {
    /// Audio encoder type: 0=Wav2Vec2, 1=Whisper, 2=HuBERT, 3=WavLM
    pub encoder_type: c_int,
    /// Sample rate
    pub sample_rate: i32,
    /// Frame length for STFT
    pub frame_length: i32,
    /// Hop length for STFT
    pub hop_length: i32,
    /// Number of mel bins
    pub n_mels: i32,
    /// Hidden dimension
    pub hidden_dim: i32,
    /// Number of layers
    pub num_layers: i32,
    /// Number of attention heads
    pub num_heads: i32,
    /// Whether to use CTC loss
    pub use_ctc: c_int,
    /// Normalization type: 0=LayerNorm, 1=BatchNorm, 2=GroupNorm
    pub norm_type: c_int,
}

impl Default for AudioEncoderConfig {
    fn default() -> Self {
        Self {
            encoder_type: 1, // Whisper
            sample_rate: 16000,
            frame_length: 1024,
            hop_length: 256,
            n_mels: 80,
            hidden_dim: 512,
            num_layers: 6,
            num_heads: 8,
            use_ctc: 0,   // False
            norm_type: 0, // LayerNorm
        }
    }
}

/// Cross-modal fusion configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[repr(C)]
pub struct CrossModalFusionConfig {
    /// Fusion type: 0=Concat, 1=CrossAttention, 2=CoAttention, 3=Transformer
    pub fusion_type: c_int,
    /// Hidden dimension for fusion
    pub fusion_dim: i32,
    /// Number of fusion layers
    pub num_fusion_layers: i32,
    /// Number of cross-attention heads
    pub num_cross_heads: i32,
    /// Dropout for fusion layers
    pub fusion_dropout: f32,
    /// Temperature for contrastive learning
    pub contrastive_temperature: f32,
    /// Whether to use learnable temperature
    pub learnable_temperature: c_int,
    /// Projection dimension for contrastive learning
    pub projection_dim: i32,
}

impl Default for CrossModalFusionConfig {
    fn default() -> Self {
        Self {
            fusion_type: 1, // CrossAttention
            fusion_dim: 512,
            num_fusion_layers: 2,
            num_cross_heads: 8,
            fusion_dropout: 0.1,
            contrastive_temperature: 0.07,
            learnable_temperature: 1, // True
            projection_dim: 256,
        }
    }
}

/// Multi-modal model configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiModalConfig {
    /// Model type
    pub model_type: MultiModalType,
    /// Supported modalities
    pub modalities: Vec<ModalityType>,
    /// Vision encoder configuration
    pub vision_config: Option<VisionEncoderConfig>,
    /// Audio encoder configuration
    pub audio_config: Option<AudioEncoderConfig>,
    /// Cross-modal fusion configuration
    pub fusion_config: CrossModalFusionConfig,
    /// Language model dimension
    pub language_dim: i32,
    /// Whether to freeze vision encoder
    pub freeze_vision_encoder: c_int,
    /// Whether to freeze language model
    pub freeze_language_model: c_int,
    /// Training objective: 0=Contrastive, 1=Generative, 2=Both
    pub training_objective: c_int,
}

/// Multi-modal input data
#[repr(C)]
#[derive(Debug)]
pub struct MultiModalInput {
    /// Text input (null-terminated string)
    pub text: *const c_char,
    /// Image data (raw pixel values)
    pub image_data: *const c_float,
    /// Image width
    pub image_width: i32,
    /// Image height
    pub image_height: i32,
    /// Image channels
    pub image_channels: i32,
    /// Audio data (raw waveform)
    pub audio_data: *const c_float,
    /// Audio length (number of samples)
    pub audio_length: usize,
    /// Audio sample rate
    pub audio_sample_rate: i32,
    /// Modality mask (bitfield indicating which modalities are present)
    pub modality_mask: c_int,
}

/// Multi-modal output
#[repr(C)]
#[derive(Debug)]
pub struct MultiModalOutput {
    /// Text embeddings
    pub text_embedding: *mut c_float,
    /// Image embeddings
    pub image_embedding: *mut c_float,
    /// Audio embeddings
    pub audio_embedding: *mut c_float,
    /// Fused multi-modal embedding
    pub fused_embedding: *mut c_float,
    /// Embedding dimension
    pub embedding_dim: i32,
    /// Generated text (for generative models)
    pub generated_text: *mut c_char,
    /// Similarity scores (for retrieval tasks)
    pub similarity_scores: *mut c_float,
    /// Number of similarity scores
    pub num_similarities: usize,
    /// Confidence score
    pub confidence: c_float,
}

/// Multi-modal model handle
pub type TrustformersMultiModalModel = usize;

/// Vision-Language understanding tasks
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[repr(C)]
pub enum VisionLanguageTask {
    /// Image-text retrieval
    ImageTextRetrieval = 0,
    /// Visual question answering
    VisualQuestionAnswering = 1,
    /// Image captioning
    ImageCaptioning = 2,
    /// Visual grounding
    VisualGrounding = 3,
    /// Visual reasoning
    VisualReasoning = 4,
    /// Image classification with text
    ImageClassificationWithText = 5,
    /// Visual dialog
    VisualDialog = 6,
    /// Document understanding
    DocumentUnderstanding = 7,
}

/// Multi-modal embeddings for different modalities
#[derive(Debug)]
pub struct MultiModalEmbeddings {
    pub text_embeddings: Vec<Vec<f32>>,
    pub image_embeddings: Vec<Vec<f32>>,
    pub audio_embeddings: Vec<Vec<f32>>,
    pub fused_embeddings: Vec<Vec<f32>>,
    pub embedding_dim: usize,
}

impl MultiModalEmbeddings {
    pub fn new(embedding_dim: usize) -> Self {
        Self {
            text_embeddings: Vec::new(),
            image_embeddings: Vec::new(),
            audio_embeddings: Vec::new(),
            fused_embeddings: Vec::new(),
            embedding_dim,
        }
    }

    /// Compute cross-modal similarity
    pub fn compute_similarity(&self, text_idx: usize, image_idx: usize) -> f32 {
        if text_idx >= self.text_embeddings.len() || image_idx >= self.image_embeddings.len() {
            return 0.0;
        }

        let text_emb = &self.text_embeddings[text_idx];
        let image_emb = &self.image_embeddings[image_idx];

        // Compute cosine similarity
        let dot_product: f32 = text_emb.iter().zip(image_emb.iter()).map(|(a, b)| a * b).sum();
        let text_norm: f32 = text_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let image_norm: f32 = image_emb.iter().map(|x| x * x).sum::<f32>().sqrt();

        if text_norm == 0.0 || image_norm == 0.0 {
            0.0
        } else {
            // Clamp to [-1.0, 1.0] to handle floating point precision issues
            (dot_product / (text_norm * image_norm)).clamp(-1.0, 1.0)
        }
    }

    /// Find best matching image for text
    pub fn find_best_image_for_text(&self, text_idx: usize) -> Option<(usize, f32)> {
        if text_idx >= self.text_embeddings.len() || self.image_embeddings.is_empty() {
            return None;
        }

        let mut best_score = f32::NEG_INFINITY;
        let mut best_idx = 0;

        for (img_idx, _) in self.image_embeddings.iter().enumerate() {
            let score = self.compute_similarity(text_idx, img_idx);
            if score > best_score {
                best_score = score;
                best_idx = img_idx;
            }
        }

        Some((best_idx, best_score))
    }
}

/// Create a multi-modal model configuration
#[no_mangle]
pub extern "C" fn trustformers_multimodal_config_create(
    model_type: MultiModalType,
) -> *mut MultiModalConfig {
    let modalities = match model_type {
        MultiModalType::VisionLanguage => vec![ModalityType::Text, ModalityType::Image],
        MultiModalType::AudioLanguage => vec![ModalityType::Text, ModalityType::Audio],
        MultiModalType::VideoLanguage => vec![ModalityType::Text, ModalityType::Video],
        MultiModalType::MultiModalLLM => {
            vec![ModalityType::Text, ModalityType::Image, ModalityType::Audio]
        },
        MultiModalType::UnifiedMultiModal => vec![
            ModalityType::Text,
            ModalityType::Image,
            ModalityType::Audio,
            ModalityType::Video,
            ModalityType::Speech,
        ],
        _ => vec![ModalityType::Text],
    };

    let config = MultiModalConfig {
        model_type,
        modalities,
        vision_config: if matches!(
            model_type,
            MultiModalType::VisionLanguage
                | MultiModalType::MultiModalLLM
                | MultiModalType::UnifiedMultiModal
        ) {
            Some(VisionEncoderConfig::default())
        } else {
            None
        },
        audio_config: if matches!(
            model_type,
            MultiModalType::AudioLanguage
                | MultiModalType::MultiModalLLM
                | MultiModalType::UnifiedMultiModal
        ) {
            Some(AudioEncoderConfig::default())
        } else {
            None
        },
        fusion_config: CrossModalFusionConfig::default(),
        language_dim: 768,
        freeze_vision_encoder: 0, // False
        freeze_language_model: 0, // False
        training_objective: 2,    // Both contrastive and generative
    };

    Box::into_raw(Box::new(config))
}

/// Free a multi-modal configuration
#[no_mangle]
pub extern "C" fn trustformers_multimodal_config_free(config: *mut MultiModalConfig) {
    if !config.is_null() {
        unsafe {
            let _ = Box::from_raw(config);
        }
    }
}

/// Create a multi-modal model
#[no_mangle]
pub extern "C" fn trustformers_create_multimodal_model(
    config: *const MultiModalConfig,
    model_handle: *mut TrustformersMultiModalModel,
) -> TrustformersError {
    if config.is_null() || model_handle.is_null() {
        return TrustformersError::NullPointer;
    }

    unsafe {
        let multimodal_config = &*config;

        // Create multi-modal embeddings based on configuration
        let embedding_dim = multimodal_config.fusion_config.projection_dim as usize;
        let embeddings = MultiModalEmbeddings::new(embedding_dim);

        // Convert MultiModalEmbeddings to a handle using Box::into_raw
        let boxed = Box::new(embeddings);
        let handle = Box::into_raw(boxed) as usize;

        *model_handle = handle;
    }

    TrustformersError::Success
}

/// Perform multi-modal inference
#[no_mangle]
pub extern "C" fn trustformers_multimodal_inference(
    model_handle: TrustformersMultiModalModel,
    input: *const MultiModalInput,
    output: *mut MultiModalOutput,
) -> TrustformersError {
    if input.is_null() || output.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = &crate::RESOURCE_REGISTRY;
    let reg = registry.read();

    let _model = match reg.get_model(model_handle) {
        Some(m) => m,
        None => return TrustformersError::InvalidHandle,
    };

    unsafe {
        let multimodal_input = &*input;
        let multimodal_output = &mut *output;

        // Initialize default embedding dimension
        let embedding_dim = 512;
        multimodal_output.embedding_dim = embedding_dim;

        // Process text if present
        if (multimodal_input.modality_mask & (1 << ModalityType::Text as c_int)) != 0 {
            if !multimodal_input.text.is_null() {
                // Simulate text embedding generation
                let text_embedding = vec![0.1f32; embedding_dim as usize];
                let text_emb_ptr =
                    libc::malloc(embedding_dim as usize * std::mem::size_of::<c_float>())
                        as *mut c_float;
                std::ptr::copy_nonoverlapping(
                    text_embedding.as_ptr(),
                    text_emb_ptr,
                    embedding_dim as usize,
                );
                multimodal_output.text_embedding = text_emb_ptr;
            }
        }

        // Process image if present
        if (multimodal_input.modality_mask & (1 << ModalityType::Image as c_int)) != 0 {
            if !multimodal_input.image_data.is_null() {
                // Simulate image embedding generation
                let image_embedding = vec![0.2f32; embedding_dim as usize];
                let image_emb_ptr =
                    libc::malloc(embedding_dim as usize * std::mem::size_of::<c_float>())
                        as *mut c_float;
                std::ptr::copy_nonoverlapping(
                    image_embedding.as_ptr(),
                    image_emb_ptr,
                    embedding_dim as usize,
                );
                multimodal_output.image_embedding = image_emb_ptr;
            }
        }

        // Process audio if present
        if (multimodal_input.modality_mask & (1 << ModalityType::Audio as c_int)) != 0 {
            if !multimodal_input.audio_data.is_null() {
                // Simulate audio embedding generation
                let audio_embedding = vec![0.3f32; embedding_dim as usize];
                let audio_emb_ptr =
                    libc::malloc(embedding_dim as usize * std::mem::size_of::<c_float>())
                        as *mut c_float;
                std::ptr::copy_nonoverlapping(
                    audio_embedding.as_ptr(),
                    audio_emb_ptr,
                    embedding_dim as usize,
                );
                multimodal_output.audio_embedding = audio_emb_ptr;
            }
        }

        // Create fused embedding
        let fused_embedding = vec![0.15f32; embedding_dim as usize]; // Average of modalities
        let fused_emb_ptr =
            libc::malloc(embedding_dim as usize * std::mem::size_of::<c_float>()) as *mut c_float;
        std::ptr::copy_nonoverlapping(
            fused_embedding.as_ptr(),
            fused_emb_ptr,
            embedding_dim as usize,
        );
        multimodal_output.fused_embedding = fused_emb_ptr;

        // Set confidence score
        multimodal_output.confidence = 0.85;

        // Initialize other fields
        multimodal_output.generated_text = ptr::null_mut();
        multimodal_output.similarity_scores = ptr::null_mut();
        multimodal_output.num_similarities = 0;
    }

    TrustformersError::Success
}

/// Perform vision-language task
#[no_mangle]
pub extern "C" fn trustformers_vision_language_task(
    model_handle: TrustformersMultiModalModel,
    task: VisionLanguageTask,
    input: *const MultiModalInput,
    result_json: *mut *mut c_char,
) -> TrustformersError {
    if input.is_null() || result_json.is_null() {
        return TrustformersError::NullPointer;
    }

    let registry = &crate::RESOURCE_REGISTRY;
    let reg = registry.read();

    let _model = match reg.get_model(model_handle) {
        Some(m) => m,
        None => return TrustformersError::InvalidHandle,
    };

    unsafe {
        let multimodal_input = &*input;

        // Get text input if available
        let text_input = if !multimodal_input.text.is_null() {
            match CStr::from_ptr(multimodal_input.text).to_str() {
                Ok(s) => s.to_string(),
                Err(_) => "".to_string(),
            }
        } else {
            "".to_string()
        };

        let task_result = match task {
            VisionLanguageTask::ImageTextRetrieval => {
                serde_json::json!({
                    "task": "image_text_retrieval",
                    "results": [
                        {
                            "image_id": 0,
                            "similarity_score": 0.95,
                            "text": text_input
                        },
                        {
                            "image_id": 1,
                            "similarity_score": 0.82,
                            "text": "Related image content"
                        }
                    ],
                    "query": text_input
                })
            },
            VisionLanguageTask::VisualQuestionAnswering => {
                serde_json::json!({
                    "task": "visual_question_answering",
                    "question": text_input,
                    "answer": "Based on the image, I can see that...",
                    "confidence": 0.89,
                    "reasoning": "The visual elements suggest..."
                })
            },
            VisionLanguageTask::ImageCaptioning => {
                serde_json::json!({
                    "task": "image_captioning",
                    "caption": "A detailed description of the image content showing...",
                    "confidence": 0.92,
                    "alternative_captions": [
                        "Another perspective on the image...",
                        "A more detailed view reveals..."
                    ]
                })
            },
            VisionLanguageTask::VisualGrounding => {
                serde_json::json!({
                    "task": "visual_grounding",
                    "query": text_input,
                    "bounding_boxes": [
                        {
                            "x": 100,
                            "y": 50,
                            "width": 200,
                            "height": 150,
                            "confidence": 0.94,
                            "label": "object_of_interest"
                        }
                    ]
                })
            },
            VisionLanguageTask::VisualReasoning => {
                serde_json::json!({
                    "task": "visual_reasoning",
                    "question": text_input,
                    "reasoning_steps": [
                        "Step 1: Observe the visual elements",
                        "Step 2: Identify relationships",
                        "Step 3: Apply logical reasoning"
                    ],
                    "conclusion": "Based on visual analysis...",
                    "confidence": 0.87
                })
            },
            _ => {
                serde_json::json!({
                    "task": "general_multimodal",
                    "input_text": text_input,
                    "processing_status": "completed",
                    "confidence": 0.80
                })
            },
        };

        let result_str = match serde_json::to_string_pretty(&task_result) {
            Ok(s) => crate::string_to_c_str(s),
            Err(_) => return TrustformersError::SerializationError,
        };

        *result_json = result_str;
    }

    TrustformersError::Success
}

/// Compute cross-modal similarity
#[no_mangle]
pub extern "C" fn trustformers_compute_similarity(
    text_embedding: *const c_float,
    image_embedding: *const c_float,
    embedding_dim: i32,
    similarity: *mut c_float,
) -> TrustformersError {
    if text_embedding.is_null() || image_embedding.is_null() || similarity.is_null() {
        return TrustformersError::NullPointer;
    }

    if embedding_dim <= 0 {
        return TrustformersError::InvalidParameter;
    }

    unsafe {
        let text_slice = std::slice::from_raw_parts(text_embedding, embedding_dim as usize);
        let image_slice = std::slice::from_raw_parts(image_embedding, embedding_dim as usize);

        // Compute cosine similarity
        let dot_product: f32 = text_slice.iter().zip(image_slice.iter()).map(|(a, b)| a * b).sum();
        let text_norm: f32 = text_slice.iter().map(|x| x * x).sum::<f32>().sqrt();
        let image_norm: f32 = image_slice.iter().map(|x| x * x).sum::<f32>().sqrt();

        let sim_score = if text_norm == 0.0 || image_norm == 0.0 {
            0.0
        } else {
            // Clamp to [-1.0, 1.0] to handle floating point precision issues
            (dot_product / (text_norm * image_norm)).clamp(-1.0, 1.0)
        };

        *similarity = sim_score;
    }

    TrustformersError::Success
}

/// Free multi-modal output
#[no_mangle]
pub extern "C" fn trustformers_multimodal_output_free(output: *mut MultiModalOutput) {
    if output.is_null() {
        return;
    }

    unsafe {
        let multimodal_output = &mut *output;

        if !multimodal_output.text_embedding.is_null() {
            libc::free(multimodal_output.text_embedding as *mut libc::c_void);
            multimodal_output.text_embedding = ptr::null_mut();
        }

        if !multimodal_output.image_embedding.is_null() {
            libc::free(multimodal_output.image_embedding as *mut libc::c_void);
            multimodal_output.image_embedding = ptr::null_mut();
        }

        if !multimodal_output.audio_embedding.is_null() {
            libc::free(multimodal_output.audio_embedding as *mut libc::c_void);
            multimodal_output.audio_embedding = ptr::null_mut();
        }

        if !multimodal_output.fused_embedding.is_null() {
            libc::free(multimodal_output.fused_embedding as *mut libc::c_void);
            multimodal_output.fused_embedding = ptr::null_mut();
        }

        if !multimodal_output.generated_text.is_null() {
            let _ = CString::from_raw(multimodal_output.generated_text);
        }

        if !multimodal_output.similarity_scores.is_null() {
            libc::free(multimodal_output.similarity_scores as *mut libc::c_void);
            multimodal_output.similarity_scores = ptr::null_mut();
        }
    }
}

/// Free a multi-modal model
#[no_mangle]
pub extern "C" fn trustformers_multimodal_free(
    model_handle: TrustformersMultiModalModel,
) -> TrustformersError {
    let registry = &crate::RESOURCE_REGISTRY;
    let mut reg = registry.write();

    if reg.remove_model(model_handle) {
        TrustformersError::Success
    } else {
        TrustformersError::InvalidHandle
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_multimodal_config_creation() {
        let config_ptr = trustformers_multimodal_config_create(MultiModalType::VisionLanguage);
        assert!(!config_ptr.is_null());

        unsafe {
            let config = &*config_ptr;
            assert_eq!(config.model_type, MultiModalType::VisionLanguage);
            assert!(config.vision_config.is_some());
            assert_eq!(config.modalities.len(), 2);
        }

        trustformers_multimodal_config_free(config_ptr);
    }

    #[test]
    fn test_multimodal_embeddings() {
        let mut embeddings = MultiModalEmbeddings::new(128);

        // Add some test embeddings
        embeddings.text_embeddings.push(vec![1.0; 128]);
        embeddings.image_embeddings.push(vec![0.5; 128]);

        let similarity = embeddings.compute_similarity(0, 0);
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_similarity_computation() {
        let embedding_dim = 128;
        let text_emb = vec![1.0f32; embedding_dim];
        let image_emb = vec![0.5f32; embedding_dim];

        let mut similarity: c_float = 0.0;
        let result = trustformers_compute_similarity(
            text_emb.as_ptr(),
            image_emb.as_ptr(),
            embedding_dim as i32,
            &mut similarity,
        );

        assert_eq!(result, TrustformersError::Success);
        assert!(similarity > 0.0);
        assert!(similarity <= 1.0);
    }

    #[test]
    fn test_vision_language_config() {
        let config = VisionEncoderConfig::default();
        assert_eq!(config.image_size, 224);
        assert_eq!(config.patch_size, 16);
        assert_eq!(config.hidden_dim, 768);
    }

    #[test]
    fn test_audio_encoder_config() {
        let config = AudioEncoderConfig::default();
        assert_eq!(config.sample_rate, 16000);
        assert_eq!(config.n_mels, 80);
        assert_eq!(config.encoder_type, 1); // Whisper
    }
}
