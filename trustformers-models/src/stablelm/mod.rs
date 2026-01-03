//! StableLM Model Implementation
//!
//! StableLM is a family of open-source language models developed by Stability AI.
//! Based on the LLaMA architecture with some optimizations and different configurations.
//!
//! Key features:
//! - RMSNorm for layer normalization
//! - SwiGLU/SiLU activation functions
//! - Rotary Position Embeddings (RoPE) with partial rotary factor
//! - Grouped-query attention in newer versions
//! - Various model sizes: 1.6B, 3B, 7B, 12B parameters
//!
//! References:
//! - StableLM models: https://github.com/Stability-AI/StableLM
//! - Based on LLaMA architecture innovations

pub mod config;
pub mod model;

pub use config::{RopeScaling, StableLMConfig};
pub use model::{
    StableLMAttention, StableLMCausalLMOutputs, StableLMDecoderLayer, StableLMEmbeddings,
    StableLMForCausalLM, StableLMMLP, StableLMModel, StableLMOutputs,
};

use trustformers_core::errors::TrustformersError;

/// Re-export common types for convenience
pub type StableLM3B = StableLMForCausalLM;
pub type StableLM7B = StableLMForCausalLM;
pub type StableLMZephyr = StableLMForCausalLM;
pub type StableLMCode = StableLMForCausalLM;

/// Model variant identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StableLMVariant {
    /// StableLM-3B base model
    Base3B,
    /// StableLM-7B base model
    Base7B,
    /// StableLM-Zephyr (instruction-tuned)
    Zephyr3B,
    /// StableLM-Code (code-specialized)
    Code3B,
    /// StableLM-2-1.6B (second generation)
    V2_1_6B,
    /// StableLM-2-12B (second generation)
    V2_12B,
}

impl StableLMVariant {
    /// Get the default configuration for this variant
    pub fn config(self) -> StableLMConfig {
        match self {
            StableLMVariant::Base3B => StableLMConfig::stablelm_3b(),
            StableLMVariant::Base7B => StableLMConfig::stablelm_7b(),
            StableLMVariant::Zephyr3B => StableLMConfig::stablelm_zephyr_3b(),
            StableLMVariant::Code3B => StableLMConfig::stablelm_code_3b(),
            StableLMVariant::V2_1_6B => StableLMConfig::stablelm_2_1_6b(),
            StableLMVariant::V2_12B => StableLMConfig::stablelm_2_12b(),
        }
    }

    /// Get the HuggingFace model name for this variant
    pub fn model_name(self) -> &'static str {
        match self {
            StableLMVariant::Base3B => "stabilityai/stablelm-3b-4e1t",
            StableLMVariant::Base7B => "stabilityai/stablelm-base-alpha-7b",
            StableLMVariant::Zephyr3B => "stabilityai/stablelm-zephyr-3b",
            StableLMVariant::Code3B => "stabilityai/stable-code-3b",
            StableLMVariant::V2_1_6B => "stabilityai/stablelm-2-1_6b",
            StableLMVariant::V2_12B => "stabilityai/stablelm-2-12b",
        }
    }

    /// Get approximate parameter count
    pub fn parameter_count(self) -> usize {
        match self {
            StableLMVariant::V2_1_6B => 1_600_000_000,
            StableLMVariant::Base3B | StableLMVariant::Zephyr3B | StableLMVariant::Code3B => {
                3_000_000_000
            },
            StableLMVariant::Base7B => 7_000_000_000,
            StableLMVariant::V2_12B => 12_000_000_000,
        }
    }

    /// Check if this variant supports grouped-query attention
    pub fn has_grouped_query_attention(self) -> bool {
        matches!(self, StableLMVariant::V2_1_6B | StableLMVariant::V2_12B)
    }
}

use trustformers_core::device::Device;

/// Helper function to create a StableLM model from a variant
pub fn create_model(variant: StableLMVariant) -> Result<StableLMForCausalLM, TrustformersError> {
    let config = variant.config();
    StableLMForCausalLM::new(config)
}

/// Helper function to create a StableLM model from a variant with device support
pub fn create_model_with_device(
    variant: StableLMVariant,
    device: Device,
) -> Result<StableLMForCausalLM, TrustformersError> {
    let config = variant.config();
    StableLMForCausalLM::new_with_device(config, device)
}

/// Helper function to create a StableLM model from a HuggingFace model name
pub fn from_pretrained_name(
    model_name: &str,
) -> Option<Result<StableLMForCausalLM, TrustformersError>> {
    StableLMConfig::from_pretrained_name(model_name).map(StableLMForCausalLM::new)
}

/// Helper function to create a StableLM model from a HuggingFace model name with device support
pub fn from_pretrained_name_with_device(
    model_name: &str,
    device: Device,
) -> Option<Result<StableLMForCausalLM, TrustformersError>> {
    StableLMConfig::from_pretrained_name(model_name)
        .map(|config| StableLMForCausalLM::new_with_device(config, device))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_variant_configs() {
        let variant = StableLMVariant::Base3B;
        let config = variant.config();
        assert_eq!(config.hidden_size, 2560);
        assert_eq!(variant.model_name(), "stabilityai/stablelm-3b-4e1t");
        assert!(!variant.has_grouped_query_attention());

        let variant = StableLMVariant::V2_12B;
        let config = variant.config();
        assert_eq!(config.hidden_size, 5120);
        assert!(variant.has_grouped_query_attention());
    }

    #[test]
    #[ignore] // Heavy test - creates StableLM 3B model, run with --ignored
    fn test_create_model() {
        let model = create_model(StableLMVariant::Base3B);
        assert_eq!(model.unwrap().model.config.hidden_size, 2560);
    }

    #[test]
    #[ignore] // Heavy test - creates StableLM 3B model, run with --ignored
    fn test_from_pretrained_name() {
        let model = from_pretrained_name("stabilityai/stablelm-3b-4e1t");
        assert!(model.is_some());

        let model = from_pretrained_name("nonexistent/model");
        assert!(model.is_none());
    }
}
