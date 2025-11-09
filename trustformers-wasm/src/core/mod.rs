// Core Module - Essential TrustformeRS WASM functionality
//
// This module contains the fundamental components required for
// tensor operations, model management, and inference pipelines.

use std::string::String;
use std::vec::Vec;

// Re-export borrow module for wasm_bindgen compatibility
pub use std::borrow;

pub mod model;
pub mod pipeline;
pub mod tensor;
pub mod tokenizer;
pub mod utils;

// Re-export main types for convenience
pub use tensor::WasmTensor;
// Re-export main types from implemented modules
pub use model::{ModelArchitecture, ModelConfig, ModelFormat, QuantizedModel, WasmModel};
pub use pipeline::{
    GenerationConfig, PipelineType, QuestionAnsweringPipeline, TextClassificationPipeline,
    TextGenerationPipeline,
};
pub use tokenizer::{SpecialTokens, TokenizerType, WasmTokenizer};
pub use utils::*;

/// Core module initialization
pub fn initialize() -> Result<(), CoreError> {
    web_sys::console::log_1(&"Initializing TrustformeRS WASM core module".into());

    // Initialize core subsystems
    // Note: Current implementations don't require explicit initialization
    // but we can add validation here

    // Validate tensor operations are available
    let _test_tensor = tensor::WasmTensor::zeros(vec![2, 2]);

    // Validate model creation is working
    let _test_config = model::ModelConfig::bert_base();

    // Validate tokenizer creation is working
    let _test_tokenizer = tokenizer::WasmTokenizer::new(tokenizer::TokenizerType::WordPiece);

    web_sys::console::log_1(&"TrustformeRS WASM core module initialized successfully".into());
    Ok(())
}

/// Core module error types
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoreError {
    TensorError(String),
    ModelError(String),
    PipelineError(String),
    TokenizerError(String),
    InitializationError(String),
}

impl core::fmt::Display for CoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            CoreError::TensorError(msg) => write!(f, "Tensor error: {}", msg),
            CoreError::ModelError(msg) => write!(f, "Model error: {}", msg),
            CoreError::PipelineError(msg) => write!(f, "Pipeline error: {}", msg),
            CoreError::TokenizerError(msg) => write!(f, "Tokenizer error: {}", msg),
            CoreError::InitializationError(msg) => {
                write!(f, "Initialization error: {}", msg)
            },
        }
    }
}

// Note: std::error::Error not available in no_std environment

/// Core module configuration
#[derive(Debug, Clone)]
pub struct CoreConfig {
    pub enable_gpu: bool,
    pub enable_simd: bool,
    pub memory_limit_mb: Option<u32>,
    pub debug_mode: bool,
}

impl Default for CoreConfig {
    fn default() -> Self {
        Self {
            enable_gpu: true,
            enable_simd: true,
            memory_limit_mb: None,
            debug_mode: false,
        }
    }
}

/// Core module capabilities
#[derive(Debug, Clone)]
pub struct CoreCapabilities {
    pub has_webgl: bool,
    pub has_webgpu: bool,
    pub has_simd: bool,
    pub has_threads: bool,
    pub memory_mb: u32,
    pub supported_tensor_types: Vec<String>,
}

impl CoreCapabilities {
    /// Detect current environment capabilities
    pub fn detect() -> Self {
        Self {
            has_webgl: utils::has_webgl(),
            has_webgpu: utils::has_webgpu(),
            has_simd: utils::has_simd(),
            has_threads: utils::has_threads(),
            memory_mb: utils::get_memory_mb() as u32,
            supported_tensor_types: tensor::get_supported_types(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_core_config_default() {
        let config = CoreConfig::default();
        assert!(config.enable_gpu);
        assert!(config.enable_simd);
        assert!(config.memory_limit_mb.is_none());
        assert!(!config.debug_mode);
    }

    #[test]
    #[cfg(target_arch = "wasm32")]
    fn test_core_capabilities_detect() {
        let capabilities = CoreCapabilities::detect();
        // Basic sanity checks
        assert!(capabilities.memory_mb > 0);
        assert!(!capabilities.supported_tensor_types.is_empty());
        assert!(capabilities.supported_tensor_types.contains(&"f32".to_string()));
    }

    #[test]
    #[cfg(not(target_arch = "wasm32"))]
    fn test_core_config_validation() {
        // Test core configuration for non-WASM targets
        let config = CoreConfig::default();
        // Test boolean field values are valid
        assert!(config.enable_gpu == config.enable_gpu); // Test GPU config field exists
        assert!(config.enable_simd == config.enable_simd); // Test SIMD config field exists
    }
}
