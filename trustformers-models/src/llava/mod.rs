//! # LLaVA (Large Language and Vision Assistant)
//!
//! LLaVA is a multimodal conversational AI that combines a vision encoder with a large language model
//! to enable visual instruction tuning and multimodal understanding.
//!
//! ## Key Innovations
//!
//! LLaVA introduces several important innovations:
//! - **Visual Instruction Tuning**: Training on visual instruction-following data
//! - **Multimodal Architecture**: Combines CLIP vision encoder with language models
//! - **Vision-Language Alignment**: Projects visual features to language space
//! - **Conversational Interface**: Enables natural conversations about images
//!
//! ## Architecture Overview
//!
//! LLaVA consists of three main components:
//! - **Vision Encoder**: CLIP ViT for processing images into patch embeddings
//! - **Multimodal Projector**: MLP to project vision features to language space
//! - **Language Model**: LLaMA/Vicuna for text generation and reasoning
//!
//! ## Model Variants
//!
//! Available LLaVA model configurations:
//! - **LLaVA-7B**: Original 7B model based on Vicuna-7B
//! - **LLaVA-13B**: Larger 13B model for improved performance
//! - **LLaVA-v1.5-7B/13B**: Enhanced versions with better vision processing
//! - **LLaVA-v1.6 (NeXT)**: Latest generation with high-resolution support
//! - **LLaVA-Phi3**: Efficient variant using Phi-3 as the language backbone
//!
//! ## Usage Examples
//!
//! ### Basic Image Understanding
//! ```rust,no_run
//! use trustformers_models::llava::{LlavaForConditionalGeneration, LlavaConfig};
//! use trustformers_core::tensor::Tensor;
//!
//! let config = LlavaConfig::llava_v1_5_7b();
//! let model = LlavaForConditionalGeneration::new(config)?;
//!
//! // Process image and text
//! let pixel_values = Tensor::randn(&[1, 3, 336, 336])?; // Batch of images
//! let input_ids = Tensor::from_vec(vec![1, 2, 3, 4], &[1, 4])?; // Text tokens
//!
//! let output = model.forward_multimodal(input_ids, Some(pixel_values), None)?;
//! println!("Generated logits shape: {:?}", output.logits.shape());
//! ```
//!
//! ### Visual Question Answering
//! ```rust,no_run
//! use trustformers_models::llava::{LlavaForConditionalGeneration, LlavaConfig};
//!
//! let config = LlavaConfig::llava_v1_6_7b()
//!     .with_high_resolution(true);  // Enable high-res processing
//!
//! let model = LlavaForConditionalGeneration::new(config)?;
//!
//! // Question: "What do you see in this image?"
//! // The model would process both the image and question together
//! ```
//!
//! ### High-Resolution Image Processing
//! ```rust,no_run
//! use trustformers_models::llava::{LlavaForConditionalGeneration, LlavaConfig};
//!
//! let mut config = LlavaConfig::llava_v1_6_34b();
//! config.with_high_resolution(true)
//!       .with_vision_tower("openai/clip-vit-large-patch14-336");
//!
//! let model = LlavaForConditionalGeneration::new(config)?;
//!
//! // Process high-resolution images with multiple grid configurations
//! ```
//!
//! ## Vision Processing Features
//!
//! ### Patch Embedding
//! - Divides images into fixed-size patches
//! - Each patch is linearly projected to hidden dimension
//! - Position embeddings added for spatial awareness
//!
//! ### Multimodal Projection
//! - Projects vision features to language model space
//! - Supports multiple projector architectures (linear, MLP)
//! - Enables seamless vision-language integration
//!
//! ### High-Resolution Support
//! LLaVA-v1.6 supports flexible image aspect ratios:
//! - Dynamic grid configurations
//! - Multiple resolution processing
//! - Adaptive patch merging strategies
//!
//! ## Training Methodology
//!
//! LLaVA uses a two-stage training process:
//! 1. **Pre-training**: Vision-language alignment on image-caption pairs
//! 2. **Fine-tuning**: Visual instruction tuning on conversation data
//!
//! ### Visual Instruction Tuning
//! - Multi-turn conversations about images
//! - Complex reasoning and analysis tasks
//! - Integration of vision and language understanding
//!
//! ## Performance Optimization
//!
//! - **Efficient Vision Processing**: Optimized patch extraction and encoding
//! - **Memory Management**: Careful handling of large image tensors
//! - **Attention Optimization**: Efficient cross-modal attention patterns
//! - **Caching**: Smart caching of vision features for multi-turn conversations
//!
//! ## Applications
//!
//! LLaVA is suitable for:
//! - **Visual Question Answering**: Answer questions about image content
//! - **Image Captioning**: Generate detailed descriptions of images
//! - **Visual Reasoning**: Complex reasoning about visual scenes
//! - **Multimodal Conversations**: Interactive discussions about images
//! - **Educational Applications**: Visual learning and explanation
//! - **Accessibility**: Describing images for visually impaired users
//!
//! ## Model Selection Guide
//!
//! Choose the right LLaVA variant based on your needs:
//! - **LLaVA-7B**: Good balance of performance and efficiency
//! - **LLaVA-13B**: Better performance, higher resource requirements
//! - **LLaVA-v1.5**: Improved vision processing and instruction following
//! - **LLaVA-v1.6**: Best performance with high-resolution support
//! - **LLaVA-Phi3**: Most efficient option for resource-constrained environments
//!
//! ## Research and Development
//!
//! LLaVA enables research in:
//! - Multimodal understanding and reasoning
//! - Vision-language alignment techniques
//! - Visual instruction tuning methodologies
//! - Cross-modal attention mechanisms
//! - Efficient multimodal architectures

pub mod config;
pub mod model;

#[cfg(test)]
mod tests;

pub use config::{LlavaConfig, LlavaVisionConfig};
pub use model::{
    LlavaAttention, LlavaDecoderLayer, LlavaForConditionalGeneration, LlavaLanguageModel,
    LlavaLanguageOutput, LlavaMLP, LlavaMultiModalProjector, LlavaOutput, LlavaVisionAttention,
    LlavaVisionEmbeddings, LlavaVisionEncoder, LlavaVisionEncoderLayer, LlavaVisionMLP,
    LlavaVisionTransformer,
};
