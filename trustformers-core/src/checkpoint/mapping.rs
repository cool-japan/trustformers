//! Weight mapping rules for converting between different framework conventions

use anyhow::Result;
use regex::Regex;
use serde::{Deserialize, Serialize};

/// Mapping rules for converting weight names between frameworks
#[derive(Debug, Clone)]
pub struct WeightMapping {
    rules: Vec<WeightMappingRule>,
    #[allow(dead_code)]
    model_type: ModelType,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModelType {
    BERT,
    GPT2,
    T5,
    LLaMA,
    Generic,
}

/// Individual mapping rule
#[derive(Debug, Clone)]
pub struct WeightMappingRule {
    pub pattern: Regex,
    pub replacement: String,
    pub transform: Option<WeightTransform>,
}

/// Transformations that may be needed when converting weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WeightTransform {
    /// No transformation
    Identity,
    /// Transpose specific dimensions
    Transpose(Vec<usize>),
    /// Reshape to new dimensions
    Reshape(Vec<isize>), // -1 for inferred dimension
    /// Split into multiple tensors
    Split { axis: usize, sizes: Vec<usize> },
    /// Merge multiple tensors
    Merge { axis: usize },
    /// Convert convolution weights format
    ConvFormat { from: ConvFormat, to: ConvFormat },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq)]
pub enum ConvFormat {
    NCHW, // PyTorch default
    NHWC, // TensorFlow default
}

impl WeightMapping {
    pub fn new(model_type: ModelType) -> Self {
        let rules = match model_type {
            ModelType::BERT => Self::bert_rules(),
            ModelType::GPT2 => Self::gpt2_rules(),
            ModelType::T5 => Self::t5_rules(),
            ModelType::LLaMA => Self::llama_rules(),
            ModelType::Generic => Vec::new(),
        };

        Self { rules, model_type }
    }

    /// Map PyTorch weight name to TensorFlow format
    pub fn pytorch_to_tensorflow(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        for rule in &self.rules {
            if rule.pattern.is_match(name) {
                let new_name = rule.pattern.replace(name, &rule.replacement).to_string();
                return Ok((new_name, rule.transform.clone()));
            }
        }

        // Default mapping if no rule matches
        Ok((self.default_pytorch_to_tf(name), None))
    }

    /// Map TensorFlow weight name to PyTorch format
    pub fn tensorflow_to_pytorch(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        // Reverse mapping - this is simplified, in practice we'd need reverse rules
        Ok((self.default_tf_to_pytorch(name), None))
    }

    /// Map JAX weight name to PyTorch format
    pub fn jax_to_pytorch(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        // JAX uses hierarchical names with dots
        let pytorch_name = name.replace("params.", "").replace(".", "_");
        Ok((pytorch_name, None))
    }

    /// Map PyTorch weight name to JAX format
    pub fn pytorch_to_jax(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        // Convert underscores to dots for JAX hierarchical structure
        let parts: Vec<&str> = name.split('_').collect();
        let jax_name = format!("params.{}", parts.join("."));
        Ok((jax_name, None))
    }

    fn bert_rules() -> Vec<WeightMappingRule> {
        vec![
            // Embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.word_embeddings\.weight$").expect("valid regex"),
                replacement: "bert/embeddings/word_embeddings".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.position_embeddings\.weight$")
                    .expect("valid regex"),
                replacement: "bert/embeddings/position_embeddings".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.token_type_embeddings\.weight$")
                    .expect("valid regex"),
                replacement: "bert/embeddings/token_type_embeddings".to_string(),
                transform: None,
            },
            // Layer normalization
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.LayerNorm\.weight$").expect("valid regex"),
                replacement: "bert/embeddings/LayerNorm/gamma".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.LayerNorm\.bias$").expect("valid regex"),
                replacement: "bert/embeddings/LayerNorm/beta".to_string(),
                transform: None,
            },
            // Encoder layers
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.query\.weight$")
                    .expect("valid regex"),
                replacement: "bert/encoder/layer_$1/attention/self/query/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.key\.weight$")
                    .expect("valid regex"),
                replacement: "bert/encoder/layer_$1/attention/self/key/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.value\.weight$")
                    .expect("valid regex"),
                replacement: "bert/encoder/layer_$1/attention/self/value/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Output projection
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.weight$")
                    .expect("valid regex"),
                replacement: "bert/encoder/layer_$1/attention/output/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // FFN layers
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.intermediate\.dense\.weight$")
                    .expect("valid regex"),
                replacement: "bert/encoder/layer_$1/intermediate/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.output\.dense\.weight$").unwrap(),
                replacement: "bert/encoder/layer_$1/output/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
        ]
    }

    fn gpt2_rules() -> Vec<WeightMappingRule> {
        vec![
            // Token embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^wte\.weight$").expect("valid regex"),
                replacement: "model/wte".to_string(),
                transform: None,
            },
            // Position embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^wpe\.weight$").expect("valid regex"),
                replacement: "model/wpe".to_string(),
                transform: None,
            },
            // Transformer blocks
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.attn\.c_attn\.weight$").expect("valid regex"),
                replacement: "model/h$1/attn/c_attn/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.attn\.c_proj\.weight$").expect("valid regex"),
                replacement: "model/h$1/attn/c_proj/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.mlp\.c_fc\.weight$").expect("valid regex"),
                replacement: "model/h$1/mlp/c_fc/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.mlp\.c_proj\.weight$").expect("valid regex"),
                replacement: "model/h$1/mlp/c_proj/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Layer norms
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.ln_1\.weight$").expect("valid regex"),
                replacement: "model/h$1/ln_1/g".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.ln_2\.weight$").expect("valid regex"),
                replacement: "model/h$1/ln_2/g".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^ln_f\.weight$").expect("valid regex"),
                replacement: "model/ln_f/g".to_string(),
                transform: None,
            },
        ]
    }

    fn t5_rules() -> Vec<WeightMappingRule> {
        vec![
            // Shared embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^shared\.weight$").expect("valid regex"),
                replacement: "shared/embedding".to_string(),
                transform: None,
            },
            // Encoder blocks
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.q\.weight$")
                    .expect("valid regex"),
                replacement: "encoder/block_$1/layer_0/SelfAttention/q".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.k\.weight$")
                    .expect("valid regex"),
                replacement: "encoder/block_$1/layer_0/SelfAttention/k".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.v\.weight$")
                    .expect("valid regex"),
                replacement: "encoder/block_$1/layer_0/SelfAttention/v".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.o\.weight$")
                    .expect("valid regex"),
                replacement: "encoder/block_$1/layer_0/SelfAttention/o".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Decoder blocks
            WeightMappingRule {
                pattern: Regex::new(r"^decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.q\.weight$")
                    .expect("valid regex"),
                replacement: "decoder/block_$1/layer_0/SelfAttention/q".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Add more T5 specific rules...
        ]
    }

    fn llama_rules() -> Vec<WeightMappingRule> {
        vec![
            // Token embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^model\.embed_tokens\.weight$").expect("valid regex"),
                replacement: "model.embed_tokens.weight".to_string(),
                transform: None,
            },
            // Layers
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.self_attn.q_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.self_attn.k_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.self_attn.v_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.self_attn.o_proj.weight".to_string(),
                transform: None,
            },
            // MLP
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.mlp.gate_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.mlp.up_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$").unwrap(),
                replacement: "model.layers.$1.mlp.down_proj.weight".to_string(),
                transform: None,
            },
            // RMS Norm
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.input_layernorm\.weight$").unwrap(),
                replacement: "model.layers.$1.input_layernorm.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$")
                    .expect("valid regex"),
                replacement: "model.layers.$1.post_attention_layernorm.weight".to_string(),
                transform: None,
            },
        ]
    }

    fn default_pytorch_to_tf(&self, name: &str) -> String {
        // Default conversion: replace . with / and weight with kernel
        name.replace('.', "/")
            .replace("weight", "kernel")
            .replace("LayerNorm", "layer_norm")
    }

    fn default_tf_to_pytorch(&self, name: &str) -> String {
        // Default reverse conversion
        name.replace('/', ".")
            .replace("kernel", "weight")
            .replace("layer_norm", "LayerNorm")
    }
}

/// Layer-level mapping for structural differences
#[derive(Debug, Clone)]
pub struct LayerMapping {
    pub source_layers: Vec<String>,
    pub target_layers: Vec<String>,
    pub merge_strategy: Option<MergeStrategy>,
}

#[derive(Debug, Clone)]
pub enum MergeStrategy {
    /// Concatenate along a specific axis
    Concatenate { axis: usize },
    /// Add tensors element-wise
    Add,
    /// Average tensors
    Average,
    /// Custom function
    Custom(String),
}

impl LayerMapping {
    pub fn new(source: Vec<String>, target: Vec<String>) -> Self {
        Self {
            source_layers: source,
            target_layers: target,
            merge_strategy: None,
        }
    }

    pub fn with_merge_strategy(mut self, strategy: MergeStrategy) -> Self {
        self.merge_strategy = Some(strategy);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bert_mapping() {
        let mapping = WeightMapping::new(ModelType::BERT);

        let (tf_name, transform) = mapping
            .pytorch_to_tensorflow("encoder.layer.0.attention.self.query.weight")
            .unwrap();

        assert_eq!(tf_name, "bert/encoder/layer_0/attention/self/query/kernel");
        assert!(matches!(transform, Some(WeightTransform::Transpose(_))));
    }

    #[test]
    fn test_gpt2_mapping() {
        let mapping = WeightMapping::new(ModelType::GPT2);

        let (tf_name, _) = mapping.pytorch_to_tensorflow("wte.weight").unwrap();
        assert_eq!(tf_name, "model/wte");

        let (tf_name, transform) = mapping.pytorch_to_tensorflow("h.0.attn.c_attn.weight").unwrap();
        assert_eq!(tf_name, "model/h0/attn/c_attn/kernel");
        assert!(matches!(transform, Some(WeightTransform::Transpose(_))));
    }

    #[test]
    fn test_jax_mapping() {
        let mapping = WeightMapping::new(ModelType::Generic);

        let (jax_name, _) =
            mapping.pytorch_to_jax("encoder_layer_0_attention_query_weight").unwrap();
        assert_eq!(jax_name, "params.encoder.layer.0.attention.query.weight");

        let (pytorch_name, _) =
            mapping.jax_to_pytorch("params.encoder.layer.0.attention.query.weight").unwrap();
        assert_eq!(pytorch_name, "encoder_layer_0_attention_query_weight");
    }
}
