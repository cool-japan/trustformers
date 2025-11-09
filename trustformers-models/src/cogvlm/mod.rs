//! CogVLM: Visual Expert for Pretrained Language Models
//!
//! This module implements the CogVLM architecture, which introduces visual experts
//! to enhance language models with vision capabilities. CogVLM can handle both
//! images and videos through its CogVideo variant.
//!
//! # Architecture Overview
//!
//! CogVLM consists of:
//! - A vision transformer (EVA-CLIP G) for encoding visual information
//! - A language model with visual expert modules injected at specific layers
//! - Cross-modal attention mechanisms for vision-language interaction
//! - Support for both image and video understanding (CogVideo)
//!
//! # Key Features
//!
//! - **Visual Expert Architecture**: Specialized modules for vision-language fusion
//! - **EVA-CLIP G Backbone**: High-quality vision encoder
//! - **Multi-stage Training**: Support for different training stages
//! - **Video Understanding**: CogVideo for temporal modeling
//! - **Flexible Configuration**: Support for various model sizes and configurations
//!
//! # References
//!
//! - "CogVLM: Visual Expert for Pretrained Language Models" (Wang et al., 2023)
//! - "CogVideo: Large-scale Pretraining for Text-to-Video Generation via Transformers"

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{CogVideoConfig, CogVlmConfig, CogVlmVisionConfig};
pub use model::{
    CogVideoInput, CogVideoModel, CogVlmInput, CogVlmModel, CogVlmOutput, CogVlmVisionTransformer,
    TemporalEncoder, VisualExpert,
};

use trustformers_core::errors::Result;

/// Create a CogVLM-Chat-17B model with default configuration
pub fn cogvlm_chat_17b() -> Result<CogVlmModel> {
    let config = CogVlmConfig::cogvlm_chat_17b();
    CogVlmModel::new(config)
}

/// Create a CogVLM-Base-17B model with default configuration
pub fn cogvlm_base_17b() -> Result<CogVlmModel> {
    let config = CogVlmConfig::cogvlm_base_17b();
    CogVlmModel::new(config)
}

/// Create a CogVLM-Grounding-17B model for visual grounding tasks
pub fn cogvlm_grounding_17b() -> Result<CogVlmModel> {
    let config = CogVlmConfig::cogvlm_grounding_17b();
    CogVlmModel::new(config)
}

/// Create a CogVideo model for video understanding
pub fn cogvideo() -> Result<CogVideoModel> {
    let config = CogVideoConfig::default();
    CogVideoModel::new(config)
}

/// Create a CogVLM model from a pretrained model name
///
/// # Supported Models
///
/// - "cogvlm-chat-17b" or "THUDM/cogvlm-chat-hf"
/// - "cogvlm-base-17b" or "THUDM/cogvlm-base-hf"
/// - "cogvlm-grounding-17b" or "THUDM/cogvlm-grounding-generalist-hf"
/// - "cogvideo" or "THUDM/cogvideo-chat"
pub fn from_pretrained(model_name: &str) -> Result<CogVlmModel> {
    let config = CogVlmConfig::from_pretrained_name(model_name).ok_or_else(|| {
        trustformers_core::errors::TrustformersError::invalid_config(format!(
            "Unknown model name: {}",
            model_name
        ))
    })?;

    CogVlmModel::new(config)
}

/// Create a CogVideo model from a pretrained model name
pub fn cogvideo_from_pretrained(model_name: &str) -> Result<CogVideoModel> {
    if model_name.contains("cogvideo") {
        cogvideo()
    } else {
        Err(
            trustformers_core::errors::TrustformersError::invalid_config(format!(
                "Unknown CogVideo model name: {}",
                model_name
            )),
        )
    }
}

/// Utility function to create a vision-only CogVLM for feature extraction
pub fn vision_encoder(config: CogVlmVisionConfig) -> Result<CogVlmVisionTransformer> {
    CogVlmVisionTransformer::new(config)
}

/// Create a visual expert module standalone
pub fn visual_expert(config: CogVlmConfig) -> Result<VisualExpert> {
    VisualExpert::new(config)
}

/// Available model configurations
pub fn available_models() -> Vec<&'static str> {
    vec![
        "cogvlm-chat-17b",
        "cogvlm-base-17b",
        "cogvlm-grounding-17b",
        "cogvideo",
        "THUDM/cogvlm-chat-hf",
        "THUDM/cogvlm-base-hf",
        "THUDM/cogvlm-grounding-generalist-hf",
        "THUDM/cogvideo-chat",
    ]
}

/// Model capabilities and recommended use cases
pub fn model_info(model_name: &str) -> Option<ModelInfo> {
    match model_name {
        "cogvlm-chat-17b" | "THUDM/cogvlm-chat-hf" => Some(ModelInfo {
            name: "CogVLM-Chat-17B",
            description: "Conversational multimodal model for visual Q&A and chat",
            use_cases: vec!["Visual Q&A", "Image captioning", "Visual chat"],
            parameters: "17B",
            context_length: 2048,
            supports_video: false,
        }),
        "cogvlm-base-17b" | "THUDM/cogvlm-base-hf" => Some(ModelInfo {
            name: "CogVLM-Base-17B",
            description: "Base multimodal model for fine-tuning",
            use_cases: vec!["Fine-tuning", "Research", "Custom applications"],
            parameters: "17B",
            context_length: 2048,
            supports_video: false,
        }),
        "cogvlm-grounding-17b" | "THUDM/cogvlm-grounding-generalist-hf" => Some(ModelInfo {
            name: "CogVLM-Grounding-17B",
            description: "Specialized for visual grounding and object localization",
            use_cases: vec!["Visual grounding", "Object detection", "Spatial reasoning"],
            parameters: "17B",
            context_length: 2048,
            supports_video: false,
        }),
        "cogvideo" | "THUDM/cogvideo-chat" => Some(ModelInfo {
            name: "CogVideo",
            description: "Video understanding and generation model",
            use_cases: vec!["Video Q&A", "Video captioning", "Temporal reasoning"],
            parameters: "17B",
            context_length: 4096,
            supports_video: true,
        }),
        _ => None,
    }
}

/// Information about a specific model
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub name: &'static str,
    pub description: &'static str,
    pub use_cases: Vec<&'static str>,
    pub parameters: &'static str,
    pub context_length: usize,
    pub supports_video: bool,
}
