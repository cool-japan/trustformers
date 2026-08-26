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
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
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

impl WeightTransform {
    /// The transform that undoes this one.
    ///
    /// `None` for transforms that cannot be inverted from the transform alone:
    /// `Reshape` (the original shape is not recorded) and `Split`/`Merge`
    /// (they change the number of tensors).
    pub fn inverse(&self) -> Option<WeightTransform> {
        match self {
            WeightTransform::Identity => Some(WeightTransform::Identity),
            // A permutation is its own kind of inverse: invert the permutation.
            WeightTransform::Transpose(permutation) => {
                let mut inverse = vec![0usize; permutation.len()];
                for (position, axis) in permutation.iter().enumerate() {
                    if *axis >= permutation.len() {
                        return None;
                    }
                    inverse[*axis] = position;
                }
                Some(WeightTransform::Transpose(inverse))
            },
            WeightTransform::ConvFormat { from, to } => Some(WeightTransform::ConvFormat {
                from: *to,
                to: *from,
            }),
            // The pre-reshape shape is not recorded, so this cannot be undone.
            WeightTransform::Reshape(_) => None,
            // These change the tensor count; the inverse is the other one, but
            // the conversion pipeline has no multi-tensor path to apply it.
            WeightTransform::Split { .. } | WeightTransform::Merge { .. } => None,
        }
    }
}

impl WeightMapping {
    pub fn new(model_type: ModelType) -> Self {
        let rules = match model_type {
            ModelType::BERT => Self::bert_rules().unwrap_or_default(),
            ModelType::GPT2 => Self::gpt2_rules().unwrap_or_default(),
            ModelType::T5 => Self::t5_rules().unwrap_or_default(),
            ModelType::LLaMA => Self::llama_rules().unwrap_or_default(),
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

    /// Map a TensorFlow weight name back to PyTorch format.
    ///
    /// The forward rules are applied in reverse: the rule whose *replacement*
    /// produced this name is found and its transform is inverted. Dropping the
    /// transform (as this used to) silently produced transposed weights,
    /// because TensorFlow stores dense kernels as `[in, out]` and PyTorch as
    /// `[out, in]`.
    ///
    /// Rules whose transform has no inverse (`Split` / `Merge`) are reported as
    /// an error rather than mapped without their transform.
    pub fn tensorflow_to_pytorch(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        for rule in &self.rules {
            // The forward direction rewrote `pattern` into `replacement`;
            // recognise the rewritten form to walk back.
            let Some(reverse) = Self::reverse_pattern(rule) else {
                continue;
            };
            if !reverse.is_match(name) {
                continue;
            }

            let original = reverse.replace(name, Self::pattern_template(rule)).to_string();
            let transform = match &rule.transform {
                None => None,
                Some(transform) => Some(transform.inverse().ok_or_else(|| {
                    anyhow::anyhow!(
                        "weight '{}' maps back through a {:?} transform, which has no inverse",
                        name,
                        transform
                    )
                })?),
            };
            return Ok((original, transform));
        }

        Ok((self.default_tf_to_pytorch(name), None))
    }

    /// Build the regex that recognises a rule's *output* names.
    ///
    /// Returns `None` for replacements that use capture groups, which cannot be
    /// mechanically reversed; such a rule is skipped rather than mis-applied.
    fn reverse_pattern(rule: &WeightMappingRule) -> Option<Regex> {
        if !rule.replacement.contains('$') {
            // A literal replacement: match it exactly.
            return Regex::new(&format!("^{}$", regex::escape(&rule.replacement))).ok();
        }

        // `foo/$1/bar` -> `^foo/(.+)/bar$`, so the captured text can be put back.
        let mut pattern = String::from("^");
        let mut characters = rule.replacement.chars().peekable();
        while let Some(character) = characters.next() {
            if character == '$' {
                // Skip the group number.
                while characters.peek().is_some_and(|next| next.is_ascii_digit()) {
                    characters.next();
                }
                pattern.push_str("(.+)");
            } else {
                pattern.push_str(&regex::escape(&character.to_string()));
            }
        }
        pattern.push('$');
        Regex::new(&pattern).ok()
    }

    /// The template that rebuilds a PyTorch name from a reversed match.
    ///
    /// The forward `pattern` is a regex; its literal parts plus `$n` for each
    /// capture group reconstruct the original name.
    fn pattern_template(rule: &WeightMappingRule) -> String {
        let source = rule.pattern.as_str();
        let mut template = String::with_capacity(source.len());
        let mut group = 0usize;
        let mut characters = source.chars().peekable();

        while let Some(character) = characters.next() {
            match character {
                '^' | '$' => {},
                '\\' => {
                    // Escaped literal: keep the escaped character.
                    if let Some(next) = characters.next() {
                        template.push(next);
                    }
                },
                '(' => {
                    group += 1;
                    template.push_str(&format!("${}", group));
                    // Skip to the matching ')'.
                    let mut depth = 1;
                    for inner in characters.by_ref() {
                        match inner {
                            '(' => depth += 1,
                            ')' => {
                                depth -= 1;
                                if depth == 0 {
                                    break;
                                }
                            },
                            _ => {},
                        }
                    }
                },
                other => template.push(other),
            }
        }

        template
    }

    /// Map JAX weight name to PyTorch format.
    ///
    /// JAX/Flax parameter trees use `.`-separated paths under `params`; the
    /// PyTorch equivalent is the same path with `.` separators, so only the
    /// `params.` prefix is stripped. Replacing every `.` with `_` (as this used
    /// to) destroyed the module hierarchy and produced names no PyTorch
    /// state dict contains.
    pub fn jax_to_pytorch(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        let pytorch_name = name.strip_prefix("params.").unwrap_or(name).to_string();
        Ok((pytorch_name, None))
    }

    /// Map PyTorch weight name to JAX format.
    ///
    /// Inverse of [`Self::jax_to_pytorch`]: the `.`-separated PyTorch path is
    /// kept as-is under a `params.` prefix. Splitting on `_` (as this used to)
    /// mangled every name containing an underscore, such as
    /// `layer_norm.weight`.
    pub fn pytorch_to_jax(&self, name: &str) -> Result<(String, Option<WeightTransform>)> {
        Ok((format!("params.{}", name), None))
    }

    fn bert_rules() -> Result<Vec<WeightMappingRule>> {
        Ok(vec![
            // Embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.word_embeddings\.weight$")?,
                replacement: "bert/embeddings/word_embeddings".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.position_embeddings\.weight$")?,
                replacement: "bert/embeddings/position_embeddings".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.token_type_embeddings\.weight$")?,
                replacement: "bert/embeddings/token_type_embeddings".to_string(),
                transform: None,
            },
            // Layer normalization
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.LayerNorm\.weight$")?,
                replacement: "bert/embeddings/LayerNorm/gamma".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^embeddings\.LayerNorm\.bias$")?,
                replacement: "bert/embeddings/LayerNorm/beta".to_string(),
                transform: None,
            },
            // Encoder layers
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.query\.weight$")?,
                replacement: "bert/encoder/layer_$1/attention/self/query/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.key\.weight$")?,
                replacement: "bert/encoder/layer_$1/attention/self/key/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.self\.value\.weight$")?,
                replacement: "bert/encoder/layer_$1/attention/self/value/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Output projection
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.attention\.output\.dense\.weight$")?,
                replacement: "bert/encoder/layer_$1/attention/output/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // FFN layers
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.intermediate\.dense\.weight$")?,
                replacement: "bert/encoder/layer_$1/intermediate/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^encoder\.layer\.(\d+)\.output\.dense\.weight$")?,
                replacement: "bert/encoder/layer_$1/output/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
        ])
    }

    fn gpt2_rules() -> Result<Vec<WeightMappingRule>> {
        Ok(vec![
            // Token embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^wte\.weight$")?,
                replacement: "model/wte".to_string(),
                transform: None,
            },
            // Position embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^wpe\.weight$")?,
                replacement: "model/wpe".to_string(),
                transform: None,
            },
            // Transformer blocks
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.attn\.c_attn\.weight$")?,
                replacement: "model/h$1/attn/c_attn/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.attn\.c_proj\.weight$")?,
                replacement: "model/h$1/attn/c_proj/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.mlp\.c_fc\.weight$")?,
                replacement: "model/h$1/mlp/c_fc/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.mlp\.c_proj\.weight$")?,
                replacement: "model/h$1/mlp/c_proj/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Layer norms
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.ln_1\.weight$")?,
                replacement: "model/h$1/ln_1/g".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^h\.(\d+)\.ln_2\.weight$")?,
                replacement: "model/h$1/ln_2/g".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^ln_f\.weight$")?,
                replacement: "model/ln_f/g".to_string(),
                transform: None,
            },
        ])
    }

    fn t5_rules() -> Result<Vec<WeightMappingRule>> {
        Ok(vec![
            // Shared embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^shared\.weight$")?,
                replacement: "shared/embedding".to_string(),
                transform: None,
            },
            // Encoder blocks
            WeightMappingRule {
                pattern: Regex::new(
                    r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.q\.weight$",
                )?,
                replacement: "encoder/block_$1/layer_0/SelfAttention/q".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(
                    r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.k\.weight$",
                )?,
                replacement: "encoder/block_$1/layer_0/SelfAttention/k".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(
                    r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.v\.weight$",
                )?,
                replacement: "encoder/block_$1/layer_0/SelfAttention/v".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            WeightMappingRule {
                pattern: Regex::new(
                    r"^encoder\.block\.(\d+)\.layer\.0\.SelfAttention\.o\.weight$",
                )?,
                replacement: "encoder/block_$1/layer_0/SelfAttention/o".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Decoder blocks
            WeightMappingRule {
                pattern: Regex::new(
                    r"^decoder\.block\.(\d+)\.layer\.0\.SelfAttention\.q\.weight$",
                )?,
                replacement: "decoder/block_$1/layer_0/SelfAttention/q".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            },
            // Add more T5 specific rules...
        ])
    }

    fn llama_rules() -> Result<Vec<WeightMappingRule>> {
        Ok(vec![
            // Token embeddings
            WeightMappingRule {
                pattern: Regex::new(r"^model\.embed_tokens\.weight$")?,
                replacement: "model.embed_tokens.weight".to_string(),
                transform: None,
            },
            // Layers
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.q_proj\.weight$")?,
                replacement: "model.layers.$1.self_attn.q_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.k_proj\.weight$")?,
                replacement: "model.layers.$1.self_attn.k_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.v_proj\.weight$")?,
                replacement: "model.layers.$1.self_attn.v_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.self_attn\.o_proj\.weight$")?,
                replacement: "model.layers.$1.self_attn.o_proj.weight".to_string(),
                transform: None,
            },
            // MLP
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.gate_proj\.weight$")?,
                replacement: "model.layers.$1.mlp.gate_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.up_proj\.weight$")?,
                replacement: "model.layers.$1.mlp.up_proj.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.mlp\.down_proj\.weight$")?,
                replacement: "model.layers.$1.mlp.down_proj.weight".to_string(),
                transform: None,
            },
            // RMS Norm
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.input_layernorm\.weight$")?,
                replacement: "model.layers.$1.input_layernorm.weight".to_string(),
                transform: None,
            },
            WeightMappingRule {
                pattern: Regex::new(r"^model\.layers\.(\d+)\.post_attention_layernorm\.weight$")?,
                replacement: "model.layers.$1.post_attention_layernorm.weight".to_string(),
                transform: None,
            },
        ])
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

    /// Regression test: `WeightTransform` had no inverse, so
    /// `tensorflow_to_pytorch` dropped every transform and silently produced
    /// transposed weights.
    #[test]
    fn test_transform_inverses() {
        assert_eq!(
            WeightTransform::Identity.inverse(),
            Some(WeightTransform::Identity)
        );

        // Inverting a permutation twice is the identity permutation.
        let permutation = WeightTransform::Transpose(vec![2, 0, 1]);
        let inverse = permutation.inverse().expect("a permutation is invertible");
        assert_eq!(inverse, WeightTransform::Transpose(vec![1, 2, 0]));
        assert_eq!(inverse.inverse(), Some(permutation));

        // A simple 2-D transpose is its own inverse.
        let swap = WeightTransform::Transpose(vec![1, 0]);
        assert_eq!(swap.inverse(), Some(swap.clone()));

        assert_eq!(
            WeightTransform::ConvFormat {
                from: ConvFormat::NCHW,
                to: ConvFormat::NHWC
            }
            .inverse(),
            Some(WeightTransform::ConvFormat {
                from: ConvFormat::NHWC,
                to: ConvFormat::NCHW
            })
        );

        // These genuinely have no in-place inverse and must say so.
        assert!(WeightTransform::Reshape(vec![-1, 8]).inverse().is_none());
        assert!(WeightTransform::Split {
            axis: 0,
            sizes: vec![1, 1]
        }
        .inverse()
        .is_none());
        assert!(WeightTransform::Merge { axis: 0 }.inverse().is_none());
    }

    /// The reverse mapping must carry the inverted transform, not `None`.
    #[test]
    fn test_reverse_mapping_carries_the_inverse_transform() -> Result<()> {
        let mapping = WeightMapping {
            rules: vec![WeightMappingRule {
                pattern: Regex::new(r"^encoder\.dense\.weight$")?,
                replacement: "encoder/dense/kernel".to_string(),
                transform: Some(WeightTransform::Transpose(vec![1, 0])),
            }],
            model_type: ModelType::Generic,
        };

        let (tf_name, forward_transform) = mapping.pytorch_to_tensorflow("encoder.dense.weight")?;
        assert_eq!(tf_name, "encoder/dense/kernel");
        assert_eq!(
            forward_transform,
            Some(WeightTransform::Transpose(vec![1, 0]))
        );

        let (pt_name, reverse_transform) = mapping.tensorflow_to_pytorch("encoder/dense/kernel")?;
        assert_eq!(pt_name, "encoder.dense.weight", "the name must round-trip");
        assert_eq!(
            reverse_transform,
            Some(WeightTransform::Transpose(vec![1, 0])),
            "the reverse direction must transpose back, not silently skip it"
        );
        assert!(
            reverse_transform.is_some(),
            "dropping the transform is what produced transposed weights"
        );

        Ok(())
    }

    /// JAX names must keep their module hierarchy in both directions.
    #[test]
    fn test_jax_name_mapping_round_trips() -> Result<()> {
        let mapping = WeightMapping::new(ModelType::Generic);

        let (jax, _) = mapping.pytorch_to_jax("encoder.layer_norm.weight")?;
        assert_eq!(jax, "params.encoder.layer_norm.weight");

        let (pytorch, _) = mapping.jax_to_pytorch(&jax)?;
        assert_eq!(
            pytorch, "encoder.layer_norm.weight",
            "the underscore in layer_norm must survive the round trip"
        );

        Ok(())
    }

    #[test]
    fn test_bert_mapping() {
        let mapping = WeightMapping::new(ModelType::BERT);

        let (tf_name, transform) = mapping
            .pytorch_to_tensorflow("encoder.layer.0.attention.self.query.weight")
            .expect("operation failed in test");

        assert_eq!(tf_name, "bert/encoder/layer_0/attention/self/query/kernel");
        assert!(matches!(transform, Some(WeightTransform::Transpose(_))));
    }

    #[test]
    fn test_gpt2_mapping() {
        let mapping = WeightMapping::new(ModelType::GPT2);

        let (tf_name, _) =
            mapping.pytorch_to_tensorflow("wte.weight").expect("tensor operation failed");
        assert_eq!(tf_name, "model/wte");

        let (tf_name, transform) = mapping
            .pytorch_to_tensorflow("h.0.attn.c_attn.weight")
            .expect("tensor operation failed");
        assert_eq!(tf_name, "model/h0/attn/c_attn/kernel");
        assert!(matches!(transform, Some(WeightTransform::Transpose(_))));
    }

    /// Regression test: this used to assert the `_`-splitting behaviour, which
    /// mangled any PyTorch name containing an underscore. PyTorch state dict
    /// keys are `.`-separated; the mapping must preserve them.
    #[test]
    fn test_jax_mapping() {
        let mapping = WeightMapping::new(ModelType::Generic);

        let (jax_name, _) = mapping
            .pytorch_to_jax("encoder.layer.0.attention.query.weight")
            .expect("operation failed in test");
        assert_eq!(jax_name, "params.encoder.layer.0.attention.query.weight");

        let (pytorch_name, _) = mapping
            .jax_to_pytorch("params.encoder.layer.0.attention.query.weight")
            .expect("operation failed in test");
        assert_eq!(pytorch_name, "encoder.layer.0.attention.query.weight");
    }
}
