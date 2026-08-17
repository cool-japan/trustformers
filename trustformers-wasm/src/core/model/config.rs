//! Model architecture/format enums and per-architecture default configs.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

/// Supported model architectures.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelArchitecture {
    Bert,
    GPT2,
    T5,
    Llama,
    Mistral,
}

/// Supported model formats for loading and inference.
#[wasm_bindgen]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelFormat {
    /// ONNX (Open Neural Network Exchange) format
    Onnx,
    /// GGUF (GPT-Generated Unified Format) for quantized models
    Gguf,
    /// SafeTensors format (Hugging Face) — the only format this crate can
    /// actually load weights from today.
    SafeTensors,
    /// TensorRT engine format (NVIDIA) — detection/metadata only, weight
    /// loading is unsupported (pre-compiled GPU binary).
    TensorRT,
    /// Core ML model format (Apple) — detection/metadata only, weight
    /// loading is unsupported (no protobuf parser).
    CoreML,
    /// TensorFlow Lite format — detection/metadata only, weight loading is
    /// unsupported (no FlatBuffers parser).
    TensorFlowLite,
    /// PyTorch JIT traced models
    TorchScript,
    /// Custom binary format
    CustomBinary,
    /// JSON format
    Json,
}

/// Model configuration.
#[wasm_bindgen]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub architecture: ModelArchitecture,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub max_position_embeddings: usize,
    pub intermediate_size: usize,
    pub hidden_dropout_prob: f32,
    pub attention_dropout_prob: f32,
}

#[wasm_bindgen]
impl ModelConfig {
    /// Create a new model configuration using this crate's default sizing
    /// for the given architecture.
    #[wasm_bindgen(constructor)]
    pub fn new(architecture: ModelArchitecture) -> Self {
        match architecture {
            ModelArchitecture::Bert => Self::bert_base(),
            ModelArchitecture::GPT2 => Self::gpt2_base(),
            ModelArchitecture::T5 => Self::t5_small(),
            ModelArchitecture::Llama => Self::llama_7b(),
            ModelArchitecture::Mistral => Self::mistral_7b(),
        }
    }

    /// BERT base configuration.
    pub fn bert_base() -> Self {
        Self {
            architecture: ModelArchitecture::Bert,
            vocab_size: 30522,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 512,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_dropout_prob: 0.1,
        }
    }

    /// GPT-2 base configuration.
    pub fn gpt2_base() -> Self {
        Self {
            architecture: ModelArchitecture::GPT2,
            vocab_size: 50257,
            hidden_size: 768,
            num_layers: 12,
            num_heads: 12,
            max_position_embeddings: 1024,
            intermediate_size: 3072,
            hidden_dropout_prob: 0.1,
            attention_dropout_prob: 0.1,
        }
    }

    /// T5 small configuration.
    pub fn t5_small() -> Self {
        Self {
            architecture: ModelArchitecture::T5,
            vocab_size: 32128,
            hidden_size: 512,
            num_layers: 6,
            num_heads: 8,
            max_position_embeddings: 512,
            intermediate_size: 2048,
            hidden_dropout_prob: 0.1,
            attention_dropout_prob: 0.1,
        }
    }

    /// LLaMA 7B configuration.
    pub fn llama_7b() -> Self {
        Self {
            architecture: ModelArchitecture::Llama,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_position_embeddings: 2048,
            intermediate_size: 11008,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
        }
    }

    /// Mistral 7B configuration.
    pub fn mistral_7b() -> Self {
        Self {
            architecture: ModelArchitecture::Mistral,
            vocab_size: 32000,
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            max_position_embeddings: 8192,
            intermediate_size: 14336,
            hidden_dropout_prob: 0.0,
            attention_dropout_prob: 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_config() {
        let config = ModelConfig::bert_base();
        assert_eq!(config.vocab_size, 30522);
        assert_eq!(config.hidden_size, 768);
        assert_eq!(config.num_layers, 12);
    }

    #[test]
    fn test_all_architectures_construct() {
        for arch in [
            ModelArchitecture::Bert,
            ModelArchitecture::GPT2,
            ModelArchitecture::T5,
            ModelArchitecture::Llama,
            ModelArchitecture::Mistral,
        ] {
            let cfg = ModelConfig::new(arch);
            assert_eq!(cfg.architecture, arch);
            assert!(
                cfg.hidden_size.is_multiple_of(cfg.num_heads),
                "hidden_size must divide evenly by num_heads"
            );
        }
    }
}
