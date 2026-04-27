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
//! - StableLM models: <https://github.com/Stability-AI/StableLM>
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
    fn test_create_model() -> Result<(), TrustformersError> {
        let model = create_model(StableLMVariant::Base3B)?;
        assert_eq!(model.model.config.hidden_size, 2560);
        Ok(())
    }

    #[test]
    #[ignore] // Heavy test - creates StableLM 3B model, run with --ignored
    fn test_from_pretrained_name() {
        let model = from_pretrained_name("stabilityai/stablelm-3b-4e1t");
        assert!(model.is_some());

        let model = from_pretrained_name("nonexistent/model");
        assert!(model.is_none());
    }

    // ---- StableLM variant identity ----

    #[test]
    fn test_base3b_hidden_size() {
        let config = StableLMVariant::Base3B.config();
        assert_eq!(
            config.hidden_size, 2560,
            "StableLM-3B hidden_size must be 2560"
        );
    }

    #[test]
    fn test_base3b_num_layers() {
        let config = StableLMVariant::Base3B.config();
        assert_eq!(
            config.num_hidden_layers, 32,
            "StableLM-3B must have 32 layers"
        );
    }

    #[test]
    fn test_base7b_hidden_size() {
        let config = StableLMVariant::Base7B.config();
        assert_eq!(
            config.hidden_size, 4096,
            "StableLM-7B hidden_size must be 4096"
        );
    }

    #[test]
    fn test_zephyr3b_model_type() {
        let config = StableLMVariant::Zephyr3B.config();
        assert!(
            config.model_type.contains("zephyr"),
            "Zephyr model_type must contain 'zephyr'"
        );
    }

    #[test]
    fn test_code3b_vocab_size_differs() {
        let base_config = StableLMVariant::Base3B.config();
        let code_config = StableLMVariant::Code3B.config();
        // Code model uses different (smaller) vocab for code tokens
        assert_ne!(
            base_config.vocab_size, code_config.vocab_size,
            "Code variant must have a different vocab size"
        );
    }

    #[test]
    fn test_v2_1_6b_gqa() {
        let variant = StableLMVariant::V2_1_6B;
        assert!(
            variant.has_grouped_query_attention(),
            "V2_1_6B must support GQA"
        );
        let config = variant.config();
        assert!(
            config.num_key_value_heads.is_some(),
            "V2_1_6B must set num_key_value_heads"
        );
    }

    #[test]
    fn test_v2_12b_hidden_size() {
        let config = StableLMVariant::V2_12B.config();
        assert_eq!(
            config.hidden_size, 5120,
            "StableLM-2-12B hidden_size must be 5120"
        );
    }

    #[test]
    fn test_v2_12b_layers() {
        let config = StableLMVariant::V2_12B.config();
        assert_eq!(
            config.num_hidden_layers, 40,
            "StableLM-2-12B must have 40 layers"
        );
    }

    // ---- Rotary embedding (partial_rotary_factor) ----

    #[test]
    fn test_partial_rotary_factor_3b() {
        let config = StableLMVariant::Base3B.config();
        assert!(
            (config.partial_rotary_factor - 0.25).abs() < 1e-6,
            "StableLM-3B partial_rotary_factor must be 0.25, got {}",
            config.partial_rotary_factor
        );
    }

    #[test]
    fn test_partial_rotary_factor_in_range() {
        for variant in [
            StableLMVariant::Base3B,
            StableLMVariant::Base7B,
            StableLMVariant::Zephyr3B,
            StableLMVariant::Code3B,
            StableLMVariant::V2_1_6B,
            StableLMVariant::V2_12B,
        ] {
            let config = variant.config();
            assert!(
                config.partial_rotary_factor > 0.0 && config.partial_rotary_factor <= 1.0,
                "{:?} has out-of-range partial_rotary_factor: {}",
                variant,
                config.partial_rotary_factor
            );
        }
    }

    #[test]
    fn test_rotary_dim_derived_from_factor() {
        // rotary_dim = head_dim * partial_rotary_factor  (must be integer)
        let config = StableLMVariant::Base3B.config();
        let head_dim = config.hidden_size / config.num_attention_heads;
        let rotary_dim = (head_dim as f32 * config.partial_rotary_factor).round() as usize;
        assert!(rotary_dim > 0, "Rotary dimension must be positive");
        assert!(
            rotary_dim <= head_dim,
            "Rotary dimension cannot exceed head_dim"
        );
    }

    // ---- Validation ----

    #[test]
    fn test_all_variants_validate() {
        for variant in [
            StableLMVariant::Base3B,
            StableLMVariant::Base7B,
            StableLMVariant::Zephyr3B,
            StableLMVariant::Code3B,
            StableLMVariant::V2_1_6B,
            StableLMVariant::V2_12B,
        ] {
            let config = variant.config();
            assert!(
                config.validate().is_ok(),
                "{:?} failed validation: {:?}",
                variant,
                config.validate()
            );
        }
    }

    #[test]
    fn test_variant_parameter_counts_ordered() {
        let p_1_6b = StableLMVariant::V2_1_6B.parameter_count();
        let p_3b = StableLMVariant::Base3B.parameter_count();
        let p_7b = StableLMVariant::Base7B.parameter_count();
        let p_12b = StableLMVariant::V2_12B.parameter_count();
        assert!(p_1_6b < p_3b, "1.6B must have fewer params than 3B");
        assert!(p_3b < p_7b, "3B must have fewer params than 7B");
        assert!(p_7b < p_12b, "7B must have fewer params than 12B");
    }

    // ---- Context length ----

    #[test]
    fn test_context_length_at_least_4096() {
        for variant in [
            StableLMVariant::Base3B,
            StableLMVariant::Base7B,
            StableLMVariant::V2_1_6B,
        ] {
            let config = variant.config();
            assert!(
                config.max_position_embeddings >= 4096,
                "{:?} context length {} < 4096",
                variant,
                config.max_position_embeddings
            );
        }
    }

    // ---- Vocab size ----

    #[test]
    fn test_vocab_size_nonzero() {
        for variant in [
            StableLMVariant::Base3B,
            StableLMVariant::V2_1_6B,
            StableLMVariant::V2_12B,
        ] {
            let config = variant.config();
            assert!(config.vocab_size > 0, "{:?} has zero vocab size", variant);
        }
    }

    #[test]
    fn test_v2_vocab_larger_than_v1() {
        // V2 models use a larger vocabulary (100352 vs 50432)
        let v1_config = StableLMVariant::Base3B.config();
        let v2_config = StableLMVariant::V2_1_6B.config();
        assert!(
            v2_config.vocab_size > v1_config.vocab_size,
            "V2 vocab ({}) must exceed V1 vocab ({})",
            v2_config.vocab_size,
            v1_config.vocab_size
        );
    }

    // ---- Model name lookup ----

    #[test]
    fn test_from_pretrained_name_returns_none_for_unknown() {
        assert!(from_pretrained_name("this-model-does-not-exist").is_none());
    }

    #[test]
    fn test_model_names_are_unique() {
        let names = [
            StableLMVariant::Base3B.model_name(),
            StableLMVariant::Base7B.model_name(),
            StableLMVariant::Zephyr3B.model_name(),
            StableLMVariant::Code3B.model_name(),
            StableLMVariant::V2_1_6B.model_name(),
            StableLMVariant::V2_12B.model_name(),
        ];
        let mut deduped = names.to_vec();
        deduped.sort_unstable();
        deduped.dedup();
        assert_eq!(deduped.len(), names.len(), "All model names must be unique");
    }

    // ---- Equality & Clone ----

    #[test]
    fn test_variant_eq() {
        assert_eq!(StableLMVariant::Base3B, StableLMVariant::Base3B);
        assert_ne!(StableLMVariant::Base3B, StableLMVariant::Base7B);
    }

    #[test]
    fn test_config_clone() {
        let config = StableLMVariant::Base3B.config();
        let cloned = config.clone();
        assert_eq!(config.hidden_size, cloned.hidden_size);
        assert_eq!(config.partial_rotary_factor, cloned.partial_rotary_factor);
    }
}
