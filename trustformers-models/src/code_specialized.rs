//! # Code-Specialized Models
//!
//! This module provides specialized model configurations and implementations
//! optimized for code generation, understanding, and analysis tasks.
//!
//! ## Features
//!
//! - **Extended Context**: Most models support 16K-128K token contexts for long code files
//! - **Specialized Vocabularies**: Enhanced tokenization for programming languages
//! - **Fill-in-the-Middle**: Support for code completion and infilling tasks
//! - **Multi-language Support**: Optimized for Python, JavaScript, Java, C++, and more
//! - **Code-specific Attention**: Enhanced patterns for hierarchical code structure
//!
//! ## Supported Model Families
//!
//! ### CodeLlama Family
//! - CodeLlama 7B, 13B, 34B variants
//! - CodeLlama-Instruct for instruction following
//! - CodeLlama-Python for Python specialization
//!
//! ### StarCoder Family
//! - StarCoder 15B base model
//! - StarCoderBase for training
//! - StarCoder2 variants
//!
//! ### DeepSeek Coder Family
//! - DeepSeek-Coder 1B, 7B, 33B variants
//! - Instruct and base versions
//!
//! ### Qwen Coder Family
//! - Qwen2.5-Coder 1.5B, 7B, 32B variants
//! - Long context support up to 128K tokens
//!
//! ## Example Usage
//!
//! ```rust
//! use trustformers_models::code_specialized::{CodeLlamaConfig, CodeLlamaForCausalLM};
//!
//! // Create a CodeLlama 7B model
//! let config = CodeLlamaConfig::code_llama_7b();
//! let model = CodeLlamaForCausalLM::new(config)?;
//!
//! // For code completion
//! let input = "def fibonacci(n):\n    if n <= 1:\n        return n\n    return";
//! let completion = model.generate(input, 50)?;
//! ```

use anyhow::Error;
use serde::{Deserialize, Serialize};
use trustformers_core::errors::{invalid_config, Result};
use trustformers_core::tensor::Tensor;
use trustformers_core::{Config, Layer, Model};

#[cfg(feature = "llama")]
use crate::llama::{LlamaConfig, LlamaModel};

/// Configuration for code-specialized models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSpecializedConfig {
    /// Base model configuration
    pub base_config: LlamaConfig,
    /// Code-specific vocabulary size
    pub code_vocab_size: Option<usize>,
    /// Fill-in-the-middle support
    pub fill_in_middle: bool,
    /// Supported programming languages
    pub supported_languages: Vec<String>,
    /// Code-specific special tokens
    pub special_tokens: CodeSpecialTokens,
    /// Context length optimized for code
    pub code_context_length: usize,
    /// Whether to use hierarchical attention for code structure
    pub hierarchical_attention: bool,
    /// Model variant type
    pub model_variant: CodeModelVariant,
}

/// Code-specific special tokens
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeSpecialTokens {
    /// Fill-in-the-middle prefix token
    pub fim_prefix: String,
    /// Fill-in-the-middle middle token
    pub fim_middle: String,
    /// Fill-in-the-middle suffix token
    pub fim_suffix: String,
    /// End of text token
    pub eot_token: String,
    /// Repository boundary token
    pub repo_token: String,
    /// File boundary token
    pub file_token: String,
}

/// Code model variant types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum CodeModelVariant {
    /// Standard CodeLlama variant
    CodeLlama,
    /// CodeLlama with instruction tuning
    CodeLlamaInstruct,
    /// CodeLlama specialized for Python
    CodeLlamaPython,
    /// StarCoder variant
    StarCoder,
    /// StarCoder base (for training)
    StarCoderBase,
    /// StarCoder2 variant
    StarCoder2,
    /// DeepSeek Coder variant
    DeepSeekCoder,
    /// DeepSeek Coder with instructions
    DeepSeekCoderInstruct,
    /// Qwen Coder variant
    QwenCoder,
}

impl Default for CodeSpecialTokens {
    fn default() -> Self {
        Self {
            fim_prefix: "<PRE>".to_string(),
            fim_middle: "<MID>".to_string(),
            fim_suffix: "<SUF>".to_string(),
            eot_token: "<|endoftext|>".to_string(),
            repo_token: "<|repo_token|>".to_string(),
            file_token: "<|file_token|>".to_string(),
        }
    }
}

impl Default for CodeSpecializedConfig {
    fn default() -> Self {
        Self {
            base_config: LlamaConfig::default(),
            code_vocab_size: None,
            fill_in_middle: true,
            supported_languages: vec![
                "python".to_string(),
                "javascript".to_string(),
                "typescript".to_string(),
                "java".to_string(),
                "cpp".to_string(),
                "c".to_string(),
                "rust".to_string(),
                "go".to_string(),
                "html".to_string(),
                "css".to_string(),
                "sql".to_string(),
                "bash".to_string(),
            ],
            special_tokens: CodeSpecialTokens::default(),
            code_context_length: 16384,
            hierarchical_attention: true,
            model_variant: CodeModelVariant::CodeLlama,
        }
    }
}

impl CodeSpecializedConfig {
    /// CodeLlama 7B configuration
    pub fn code_llama_7b() -> Self {
        Self {
            base_config: LlamaConfig::code_llama_7b(),
            code_vocab_size: Some(32016),
            code_context_length: 16384,
            model_variant: CodeModelVariant::CodeLlama,
            ..Self::default()
        }
    }

    /// CodeLlama 13B configuration
    pub fn code_llama_13b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32016,
                hidden_size: 5120,
                intermediate_size: 13824,
                num_hidden_layers: 40,
                num_attention_heads: 40,
                max_position_embeddings: 16384,
                ..LlamaConfig::llama2_13b()
            },
            code_vocab_size: Some(32016),
            code_context_length: 16384,
            model_variant: CodeModelVariant::CodeLlama,
            ..Self::default()
        }
    }

    /// CodeLlama 34B configuration
    pub fn code_llama_34b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32016,
                hidden_size: 8192,
                intermediate_size: 22016,
                num_hidden_layers: 48,
                num_attention_heads: 64,
                num_key_value_heads: Some(8), // Grouped-query attention
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(32016),
            code_context_length: 16384,
            model_variant: CodeModelVariant::CodeLlama,
            ..Self::default()
        }
    }

    /// CodeLlama 7B Instruct configuration
    pub fn code_llama_7b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::CodeLlamaInstruct,
            ..Self::code_llama_7b()
        }
    }

    /// CodeLlama 13B Instruct configuration
    pub fn code_llama_13b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::CodeLlamaInstruct,
            ..Self::code_llama_13b()
        }
    }

    /// CodeLlama 34B Instruct configuration
    pub fn code_llama_34b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::CodeLlamaInstruct,
            ..Self::code_llama_34b()
        }
    }

    /// CodeLlama 7B Python configuration
    pub fn code_llama_7b_python() -> Self {
        Self {
            supported_languages: vec!["python".to_string()],
            model_variant: CodeModelVariant::CodeLlamaPython,
            ..Self::code_llama_7b()
        }
    }

    /// CodeLlama 13B Python configuration
    pub fn code_llama_13b_python() -> Self {
        Self {
            supported_languages: vec!["python".to_string()],
            model_variant: CodeModelVariant::CodeLlamaPython,
            ..Self::code_llama_13b()
        }
    }

    /// CodeLlama 34B Python configuration
    pub fn code_llama_34b_python() -> Self {
        Self {
            supported_languages: vec!["python".to_string()],
            model_variant: CodeModelVariant::CodeLlamaPython,
            ..Self::code_llama_34b()
        }
    }

    /// StarCoder 15B configuration
    pub fn starcoder_15b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 49152,
                hidden_size: 6144,
                intermediate_size: 24576,
                num_hidden_layers: 40,
                num_attention_heads: 48,
                max_position_embeddings: 8192,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(49152),
            code_context_length: 8192,
            model_variant: CodeModelVariant::StarCoder,
            special_tokens: CodeSpecialTokens {
                fim_prefix: "<fim_prefix>".to_string(),
                fim_middle: "<fim_middle>".to_string(),
                fim_suffix: "<fim_suffix>".to_string(),
                eot_token: "<|endoftext|>".to_string(),
                repo_token: "<reponame>".to_string(),
                file_token: "<filename>".to_string(),
            },
            ..Self::default()
        }
    }

    /// StarCoderBase 15B configuration (for training)
    pub fn starcoder_base_15b() -> Self {
        Self {
            model_variant: CodeModelVariant::StarCoderBase,
            ..Self::starcoder_15b()
        }
    }

    /// StarCoder2 7B configuration
    pub fn starcoder2_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 49152,
                hidden_size: 4096,
                intermediate_size: 16384,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                num_key_value_heads: Some(4), // Grouped-query attention
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(49152),
            code_context_length: 16384,
            model_variant: CodeModelVariant::StarCoder2,
            ..Self::default()
        }
    }

    /// StarCoder2 15B configuration
    pub fn starcoder2_15b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 49152,
                hidden_size: 6144,
                intermediate_size: 24576,
                num_hidden_layers: 40,
                num_attention_heads: 48,
                num_key_value_heads: Some(6), // Grouped-query attention
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(49152),
            code_context_length: 16384,
            model_variant: CodeModelVariant::StarCoder2,
            ..Self::default()
        }
    }

    /// DeepSeek Coder 1B configuration
    pub fn deepseek_coder_1b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 2048,
                intermediate_size: 5504,
                num_hidden_layers: 24,
                num_attention_heads: 16,
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(32000),
            code_context_length: 16384,
            model_variant: CodeModelVariant::DeepSeekCoder,
            ..Self::default()
        }
    }

    /// DeepSeek Coder 7B configuration
    pub fn deepseek_coder_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 4096,
                intermediate_size: 11008,
                num_hidden_layers: 32,
                num_attention_heads: 32,
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(32000),
            code_context_length: 16384,
            model_variant: CodeModelVariant::DeepSeekCoder,
            ..Self::default()
        }
    }

    /// DeepSeek Coder 33B configuration
    pub fn deepseek_coder_33b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 32000,
                hidden_size: 7168,
                intermediate_size: 20480,
                num_hidden_layers: 62,
                num_attention_heads: 56,
                num_key_value_heads: Some(8), // Grouped-query attention
                max_position_embeddings: 16384,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(32000),
            code_context_length: 16384,
            model_variant: CodeModelVariant::DeepSeekCoder,
            ..Self::default()
        }
    }

    /// DeepSeek Coder 1B Instruct configuration
    pub fn deepseek_coder_1b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::DeepSeekCoderInstruct,
            ..Self::deepseek_coder_1b()
        }
    }

    /// DeepSeek Coder 7B Instruct configuration
    pub fn deepseek_coder_7b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::DeepSeekCoderInstruct,
            ..Self::deepseek_coder_7b()
        }
    }

    /// DeepSeek Coder 33B Instruct configuration
    pub fn deepseek_coder_33b_instruct() -> Self {
        Self {
            model_variant: CodeModelVariant::DeepSeekCoderInstruct,
            ..Self::deepseek_coder_33b()
        }
    }

    /// Qwen2.5 Coder 1.5B configuration
    pub fn qwen_coder_1_5b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 151936,
                hidden_size: 1536,
                intermediate_size: 8960,
                num_hidden_layers: 28,
                num_attention_heads: 12,
                num_key_value_heads: Some(2),
                max_position_embeddings: 131072,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(151936),
            code_context_length: 131072,
            model_variant: CodeModelVariant::QwenCoder,
            ..Self::default()
        }
    }

    /// Qwen2.5 Coder 7B configuration
    pub fn qwen_coder_7b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 151936,
                hidden_size: 3584,
                intermediate_size: 18944,
                num_hidden_layers: 28,
                num_attention_heads: 28,
                num_key_value_heads: Some(4),
                max_position_embeddings: 131072,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(151936),
            code_context_length: 131072,
            model_variant: CodeModelVariant::QwenCoder,
            ..Self::default()
        }
    }

    /// Qwen2.5 Coder 32B configuration
    pub fn qwen_coder_32b() -> Self {
        Self {
            base_config: LlamaConfig {
                vocab_size: 151936,
                hidden_size: 5120,
                intermediate_size: 27392,
                num_hidden_layers: 64,
                num_attention_heads: 40,
                num_key_value_heads: Some(8),
                max_position_embeddings: 131072,
                ..LlamaConfig::default()
            },
            code_vocab_size: Some(151936),
            code_context_length: 131072,
            model_variant: CodeModelVariant::QwenCoder,
            ..Self::default()
        }
    }

    /// Create configuration from model name
    pub fn from_pretrained_name(name: &str) -> Option<Self> {
        match name {
            // CodeLlama variants
            "codellama/CodeLlama-7b-hf" | "code-llama-7b" => Some(Self::code_llama_7b()),
            "codellama/CodeLlama-13b-hf" | "code-llama-13b" => Some(Self::code_llama_13b()),
            "codellama/CodeLlama-34b-hf" | "code-llama-34b" => Some(Self::code_llama_34b()),
            "codellama/CodeLlama-7b-Instruct-hf" | "code-llama-7b-instruct" => {
                Some(Self::code_llama_7b_instruct())
            },
            "codellama/CodeLlama-13b-Instruct-hf" | "code-llama-13b-instruct" => {
                Some(Self::code_llama_13b_instruct())
            },
            "codellama/CodeLlama-34b-Instruct-hf" | "code-llama-34b-instruct" => {
                Some(Self::code_llama_34b_instruct())
            },
            "codellama/CodeLlama-7b-Python-hf" | "code-llama-7b-python" => {
                Some(Self::code_llama_7b_python())
            },
            "codellama/CodeLlama-13b-Python-hf" | "code-llama-13b-python" => {
                Some(Self::code_llama_13b_python())
            },
            "codellama/CodeLlama-34b-Python-hf" | "code-llama-34b-python" => {
                Some(Self::code_llama_34b_python())
            },

            // StarCoder variants
            "bigcode/starcoder" | "starcoder-15b" => Some(Self::starcoder_15b()),
            "bigcode/starcoderbase" | "starcoder-base-15b" => Some(Self::starcoder_base_15b()),
            "bigcode/starcoder2-7b" | "starcoder2-7b" => Some(Self::starcoder2_7b()),
            "bigcode/starcoder2-15b" | "starcoder2-15b" => Some(Self::starcoder2_15b()),

            // DeepSeek Coder variants
            "deepseek-ai/deepseek-coder-1.3b-base" | "deepseek-coder-1b" => {
                Some(Self::deepseek_coder_1b())
            },
            "deepseek-ai/deepseek-coder-6.7b-base" | "deepseek-coder-7b" => {
                Some(Self::deepseek_coder_7b())
            },
            "deepseek-ai/deepseek-coder-33b-base" | "deepseek-coder-33b" => {
                Some(Self::deepseek_coder_33b())
            },
            "deepseek-ai/deepseek-coder-1.3b-instruct" | "deepseek-coder-1b-instruct" => {
                Some(Self::deepseek_coder_1b_instruct())
            },
            "deepseek-ai/deepseek-coder-6.7b-instruct" | "deepseek-coder-7b-instruct" => {
                Some(Self::deepseek_coder_7b_instruct())
            },
            "deepseek-ai/deepseek-coder-33b-instruct" | "deepseek-coder-33b-instruct" => {
                Some(Self::deepseek_coder_33b_instruct())
            },

            // Qwen Coder variants
            "Qwen/Qwen2.5-Coder-1.5B" | "qwen-coder-1.5b" => Some(Self::qwen_coder_1_5b()),
            "Qwen/Qwen2.5-Coder-7B" | "qwen-coder-7b" => Some(Self::qwen_coder_7b()),
            "Qwen/Qwen2.5-Coder-32B" | "qwen-coder-32b" => Some(Self::qwen_coder_32b()),

            _ => None,
        }
    }

    /// Get all available model names
    pub fn available_models() -> Vec<&'static str> {
        vec![
            // CodeLlama
            "code-llama-7b",
            "code-llama-13b",
            "code-llama-34b",
            "code-llama-7b-instruct",
            "code-llama-13b-instruct",
            "code-llama-34b-instruct",
            "code-llama-7b-python",
            "code-llama-13b-python",
            "code-llama-34b-python",
            // StarCoder
            "starcoder-15b",
            "starcoder-base-15b",
            "starcoder2-7b",
            "starcoder2-15b",
            // DeepSeek Coder
            "deepseek-coder-1b",
            "deepseek-coder-7b",
            "deepseek-coder-33b",
            "deepseek-coder-1b-instruct",
            "deepseek-coder-7b-instruct",
            "deepseek-coder-33b-instruct",
            // Qwen Coder
            "qwen-coder-1.5b",
            "qwen-coder-7b",
            "qwen-coder-32b",
        ]
    }

    /// Check if configuration is valid
    pub fn validate(&self) -> Result<()> {
        self.base_config.validate()?;

        if self.code_context_length == 0 {
            return Err(invalid_config(
                "code_context_length",
                "Code context length must be greater than 0",
            ));
        }

        if self.supported_languages.is_empty() {
            return Err(invalid_config(
                "supported_languages",
                "At least one programming language must be supported",
            ));
        }

        Ok(())
    }

    /// Get the effective vocabulary size
    pub fn effective_vocab_size(&self) -> usize {
        self.code_vocab_size.unwrap_or(self.base_config.vocab_size)
    }

    /// Check if model supports fill-in-the-middle
    pub fn supports_fim(&self) -> bool {
        self.fill_in_middle
    }

    /// Check if model supports a specific programming language
    pub fn supports_language(&self, language: &str) -> bool {
        self.supported_languages.iter().any(|lang| lang.eq_ignore_ascii_case(language))
    }

    /// Get model architecture name
    pub fn architecture(&self) -> &'static str {
        match self.model_variant {
            CodeModelVariant::CodeLlama => "CodeLlama",
            CodeModelVariant::CodeLlamaInstruct => "CodeLlama-Instruct",
            CodeModelVariant::CodeLlamaPython => "CodeLlama-Python",
            CodeModelVariant::StarCoder => "StarCoder",
            CodeModelVariant::StarCoderBase => "StarCoderBase",
            CodeModelVariant::StarCoder2 => "StarCoder2",
            CodeModelVariant::DeepSeekCoder => "DeepSeekCoder",
            CodeModelVariant::DeepSeekCoderInstruct => "DeepSeekCoder-Instruct",
            CodeModelVariant::QwenCoder => "QwenCoder",
        }
    }
}

/// Code-specialized model implementation
pub struct CodeSpecializedModel {
    base_model: LlamaModel,
    config: CodeSpecializedConfig,
}

impl CodeSpecializedModel {
    /// Create a new code-specialized model
    pub fn new(config: CodeSpecializedConfig) -> Result<Self> {
        config.validate()?;
        let base_model = LlamaModel::new(config.base_config.clone())?;

        Ok(Self { base_model, config })
    }

    /// Get the configuration
    pub fn config(&self) -> &CodeSpecializedConfig {
        &self.config
    }

    /// Check if model supports fill-in-the-middle
    pub fn supports_fim(&self) -> bool {
        self.config.supports_fim()
    }

    /// Check if model supports a specific programming language
    pub fn supports_language(&self, language: &str) -> bool {
        self.config.supports_language(language)
    }

    /// Get supported programming languages
    pub fn supported_languages(&self) -> &[String] {
        &self.config.supported_languages
    }

    /// Create model from pretrained name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let config = CodeSpecializedConfig::from_pretrained_name(name)
            .ok_or_else(|| Error::msg(format!("Unknown code model: {}", name)))?;
        Self::new(config)
    }
}

impl Layer for CodeSpecializedModel {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        self.base_model.forward(input)
    }
}

/// Code-specialized model with language modeling head
pub struct CodeSpecializedForCausalLM {
    model: CodeSpecializedModel,
    lm_head: trustformers_core::layers::Linear,
}

impl CodeSpecializedForCausalLM {
    /// Create a new code-specialized model for causal language modeling
    pub fn new(config: CodeSpecializedConfig) -> Result<Self> {
        let vocab_size = config.effective_vocab_size();
        let hidden_size = config.base_config.hidden_size;

        let model = CodeSpecializedModel::new(config)?;
        let lm_head = trustformers_core::layers::Linear::new(hidden_size, vocab_size, false);

        Ok(Self { model, lm_head })
    }

    /// Get the configuration
    pub fn config(&self) -> &CodeSpecializedConfig {
        self.model.config()
    }

    /// Create model from pretrained name
    pub fn from_pretrained_name(name: &str) -> Result<Self> {
        let config = CodeSpecializedConfig::from_pretrained_name(name)
            .ok_or_else(|| Error::msg(format!("Unknown code model: {}", name)))?;
        Self::new(config)
    }
}

impl Layer for CodeSpecializedForCausalLM {
    type Input = Vec<u32>; // Token IDs
    type Output = Tensor;

    fn forward(&self, input: Self::Input) -> Result<Self::Output> {
        let hidden_states = self.model.forward(input)?;
        self.lm_head.forward(hidden_states)
    }
}

// Convenience type aliases for common code models
pub type CodeLlamaConfig = CodeSpecializedConfig;
pub type CodeLlamaModel = CodeSpecializedModel;
pub type CodeLlamaForCausalLM = CodeSpecializedForCausalLM;

pub type StarCoderConfig = CodeSpecializedConfig;
pub type StarCoderModel = CodeSpecializedModel;
pub type StarCoderForCausalLM = CodeSpecializedForCausalLM;

pub type DeepSeekCoderConfig = CodeSpecializedConfig;
pub type DeepSeekCoderModel = CodeSpecializedModel;
pub type DeepSeekCoderForCausalLM = CodeSpecializedForCausalLM;

pub type QwenCoderConfig = CodeSpecializedConfig;
pub type QwenCoderModel = CodeSpecializedModel;
pub type QwenCoderForCausalLM = CodeSpecializedForCausalLM;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_specialized_config_creation() {
        let config = CodeSpecializedConfig::code_llama_7b();
        assert_eq!(config.base_config.vocab_size, 32016);
        assert_eq!(config.code_context_length, 16384);
        assert_eq!(config.model_variant, CodeModelVariant::CodeLlama);
        assert!(config.supports_fim());
    }

    #[test]
    fn test_starcoder_config() {
        let config = CodeSpecializedConfig::starcoder_15b();
        assert_eq!(config.base_config.vocab_size, 49152);
        assert_eq!(config.code_context_length, 8192);
        assert_eq!(config.model_variant, CodeModelVariant::StarCoder);
        assert_eq!(config.special_tokens.fim_prefix, "<fim_prefix>");
    }

    #[test]
    fn test_deepseek_coder_config() {
        let config = CodeSpecializedConfig::deepseek_coder_7b();
        assert_eq!(config.base_config.vocab_size, 32000);
        assert_eq!(config.code_context_length, 16384);
        assert_eq!(config.model_variant, CodeModelVariant::DeepSeekCoder);
    }

    #[test]
    fn test_qwen_coder_config() {
        let config = CodeSpecializedConfig::qwen_coder_7b();
        assert_eq!(config.base_config.vocab_size, 151936);
        assert_eq!(config.code_context_length, 131072);
        assert_eq!(config.model_variant, CodeModelVariant::QwenCoder);
    }

    #[test]
    fn test_from_pretrained_name() {
        let config = CodeSpecializedConfig::from_pretrained_name("code-llama-7b");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.model_variant, CodeModelVariant::CodeLlama);

        let config = CodeSpecializedConfig::from_pretrained_name("starcoder-15b");
        assert!(config.is_some());
        let config = config.unwrap();
        assert_eq!(config.model_variant, CodeModelVariant::StarCoder);

        let config = CodeSpecializedConfig::from_pretrained_name("unknown-model");
        assert!(config.is_none());
    }

    #[test]
    fn test_available_models() {
        let models = CodeSpecializedConfig::available_models();
        assert!(models.contains(&"code-llama-7b"));
        assert!(models.contains(&"starcoder-15b"));
        assert!(models.contains(&"deepseek-coder-7b"));
        assert!(models.contains(&"qwen-coder-7b"));
        assert!(models.len() >= 20); // Should have at least 20 models
    }

    #[test]
    fn test_language_support() {
        let config = CodeSpecializedConfig::default();
        assert!(config.supports_language("python"));
        assert!(config.supports_language("Python"));
        assert!(config.supports_language("PYTHON"));
        assert!(config.supports_language("rust"));
        assert!(!config.supports_language("cobol"));
    }

    #[test]
    fn test_python_specialized_config() {
        let config = CodeSpecializedConfig::code_llama_7b_python();
        assert_eq!(config.supported_languages.len(), 1);
        assert!(config.supports_language("python"));
        assert!(!config.supports_language("java"));
        assert_eq!(config.model_variant, CodeModelVariant::CodeLlamaPython);
    }

    #[test]
    fn test_instruct_variants() {
        let config = CodeSpecializedConfig::code_llama_7b_instruct();
        assert_eq!(config.model_variant, CodeModelVariant::CodeLlamaInstruct);

        let config = CodeSpecializedConfig::deepseek_coder_7b_instruct();
        assert_eq!(
            config.model_variant,
            CodeModelVariant::DeepSeekCoderInstruct
        );
    }

    #[test]
    fn test_config_validation() {
        let config = CodeSpecializedConfig::default();
        assert!(config.validate().is_ok());

        let mut invalid_config = CodeSpecializedConfig::default();
        invalid_config.code_context_length = 0;
        assert!(invalid_config.validate().is_err());

        let mut invalid_config = CodeSpecializedConfig::default();
        invalid_config.supported_languages.clear();
        assert!(invalid_config.validate().is_err());
    }

    #[test]
    fn test_architecture_names() {
        let config = CodeSpecializedConfig::code_llama_7b();
        assert_eq!(config.architecture(), "CodeLlama");

        let config = CodeSpecializedConfig::starcoder_15b();
        assert_eq!(config.architecture(), "StarCoder");

        let config = CodeSpecializedConfig::deepseek_coder_7b_instruct();
        assert_eq!(config.architecture(), "DeepSeekCoder-Instruct");
    }

    #[test]
    fn test_effective_vocab_size() {
        let config = CodeSpecializedConfig::code_llama_7b();
        assert_eq!(config.effective_vocab_size(), 32016);

        let mut config = CodeSpecializedConfig::default();
        config.code_vocab_size = None;
        config.base_config.vocab_size = 50000;
        assert_eq!(config.effective_vocab_size(), 50000);
    }

    #[test]
    fn test_model_creation() {
        let config = CodeSpecializedConfig {
            base_config: LlamaConfig {
                vocab_size: 1000,
                hidden_size: 64,
                intermediate_size: 256,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                max_position_embeddings: 512,
                ..LlamaConfig::default()
            },
            code_context_length: 512,
            ..CodeSpecializedConfig::default()
        };

        let model = CodeSpecializedModel::new(config.clone());
        assert!(model.is_ok());
        let model = model.unwrap();
        assert!(model.supports_fim());
        assert!(model.supports_language("python"));

        let causal_lm = CodeSpecializedForCausalLM::new(config);
        assert!(causal_lm.is_ok());
    }

    #[test]
    fn test_grouped_query_attention_configs() {
        let config = CodeSpecializedConfig::code_llama_34b();
        assert_eq!(config.base_config.num_key_value_heads, Some(8));

        let config = CodeSpecializedConfig::starcoder2_7b();
        assert_eq!(config.base_config.num_key_value_heads, Some(4));

        let config = CodeSpecializedConfig::deepseek_coder_33b();
        assert_eq!(config.base_config.num_key_value_heads, Some(8));
    }

    #[test]
    fn test_context_lengths() {
        let config = CodeSpecializedConfig::code_llama_7b();
        assert_eq!(config.code_context_length, 16384);

        let config = CodeSpecializedConfig::starcoder_15b();
        assert_eq!(config.code_context_length, 8192);

        let config = CodeSpecializedConfig::qwen_coder_7b();
        assert_eq!(config.code_context_length, 131072);
    }
}
